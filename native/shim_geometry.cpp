/*
 * Copyright (c) 2026 Enaium
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * cvk_ shim for the OpenCV "geometry" module: multi-view 3D vision
 * (3d.hpp) and the planar-subdivision Subdiv2D (2d.hpp).
 *
 * Every export is noexcept: bodies run inside guarded(), failures are
 * reported through cvk_last_error() and Mat-returning functions return
 * NULL. Multi-result functions write extra handles through `cvk_mat_t **`
 * out-params; the caller owns them (one cvk_mat_release each). Lists of
 * Mats travel as CV_32SC2 packed-pointer Mats (two int32s per 64-bit Mat
 * address — the SDK's Converters.vector_Mat_to_Mat wire format).
 */
#include "opencv_kmp.h"
#include "opencv_kmp_geometry.h"

#include <opencv2/geometry/2d.hpp>
#include <opencv2/geometry/3d.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void record_error(const char *message) {
    g_last_error = message != nullptr ? message : "unknown error";
}

template <typename F>
auto guarded(F &&body) -> decltype(body()) {
    try {
        return body();
    } catch (const cv::Exception &e) {
        record_error(e.what());
    } catch (const std::exception &e) {
        record_error(e.what());
    } catch (...) {
        record_error("unknown native error");
    }
    return decltype(body())();
}

cv::Mat *require(cvk_mat_t *mat) {
    if (mat == nullptr) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<cv::Mat *>(mat);
}

const cv::Mat *require_const(const cvk_mat_t *mat) {
    return require(const_cast<cvk_mat_t *>(mat));
}

/** Heap-allocates each Mat and packs the addresses as a CV_32SC2 Mat. */
cv::Mat pack_mats(const std::vector<cv::Mat> &mats) {
    if (mats.empty()) return cv::Mat();
    cv::Mat out(static_cast<int>(mats.size()), 1, CV_32SC2);
    for (size_t i = 0; i < mats.size(); ++i) {
        const uint64_t addr = reinterpret_cast<uint64_t>(new cv::Mat(mats[i]));
        out.at<cv::Vec2i>(static_cast<int>(i), 0) = cv::Vec2i(
            static_cast<int>(addr >> 32), static_cast<int>(addr & 0xffffffffu));
    }
    return out;
}

/** Decodes a CV_32SC2 packed-pointer Mat back into Mat handles (by copy). */
std::vector<cv::Mat> unpack_mats(const cv::Mat &packed) {
    std::vector<cv::Mat> out;
    if (packed.empty()) return out;
    const int count = packed.rows;
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const cv::Vec2i v = packed.at<cv::Vec2i>(i, 0);
        const uint64_t addr = (static_cast<uint64_t>(v[0]) << 32) |
                              (static_cast<uint64_t>(v[1]) & 0xffffffffu);
        out.emplace_back(*reinterpret_cast<cv::Mat *>(addr));
    }
    return out;
}

/** Builds a cv::UsacParams from the flat scalar ABI arguments. */
cv::UsacParams make_usac(double confidence, int is_parallel, int lo_iterations, int lo_method,
                         int lo_sample_size, int max_iterations, int neighbors_search,
                         int random_generator_state, int sampler, int score, double threshold,
                         int final_polisher, int final_polisher_iterations) {
    cv::UsacParams p;
    p.confidence = confidence;
    p.isParallel = is_parallel != 0;
    p.loIterations = lo_iterations;
    p.loMethod = static_cast<cv::LocalOptimMethod>(lo_method);
    p.loSampleSize = lo_sample_size;
    p.maxIterations = max_iterations;
    p.neighborsSearch = static_cast<cv::NeighborSearchMethod>(neighbors_search);
    p.randomGeneratorState = random_generator_state;
    p.sampler = static_cast<cv::SamplingMethod>(sampler);
    p.score = static_cast<cv::ScoreMethod>(score);
    p.threshold = threshold;
    p.final_polisher = static_cast<cv::PolishingMethod>(final_polisher);
    p.final_polisher_iterations = final_polisher_iterations;
    return p;
}

} // namespace

extern "C" {

/* =========================================================================
 * Multi-view 3D vision
 * ========================================================================= */

cvk_mat_t *cvk_rodrigues(const cvk_mat_t *src) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat dst;
        cv::Rodrigues(*s, dst);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(dst));
    });
}

cvk_mat_t *cvk_rodrigues_jacobian(const cvk_mat_t *src, cvk_mat_t **out_jacobian) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat dst, jacobian;
        cv::Rodrigues(*s, dst, jacobian);
        *out_jacobian = reinterpret_cast<cvk_mat_t *>(new cv::Mat(jacobian));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(dst));
    });
}

cvk_mat_t *cvk_find_homography(const cvk_mat_t *src_points, const cvk_mat_t *dst_points,
                               int method, double ransac_reproj_threshold, int max_iters,
                               double confidence) {
    const cv::Mat *src = require_const(src_points);
    if (src == nullptr) return nullptr;
    const cv::Mat *dst = require_const(dst_points);
    if (dst == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat h = cv::findHomography(*src, *dst, method, ransac_reproj_threshold, cv::noArray(),
                                       max_iters, confidence);
        if (h.empty()) {
            record_error("findHomography: no homography could be estimated");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(h));
    });
}

cvk_mat_t *cvk_find_homography_masked(const cvk_mat_t *src_points, const cvk_mat_t *dst_points,
                                      int method, double ransac_reproj_threshold, int max_iters,
                                      double confidence, cvk_mat_t **out_mask) {
    const cv::Mat *src = require_const(src_points);
    if (src == nullptr) return nullptr;
    const cv::Mat *dst = require_const(dst_points);
    if (dst == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat h = cv::findHomography(*src, *dst, method, ransac_reproj_threshold, mask, max_iters,
                                       confidence);
        if (h.empty()) {
            record_error("findHomography: no homography could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(h));
    });
}

cvk_mat_t *cvk_find_homography_usac(const cvk_mat_t *src_points, const cvk_mat_t *dst_points,
                                    cvk_mat_t **out_mask,
                                    double confidence, int is_parallel, int lo_iterations,
                                    int lo_method, int lo_sample_size, int max_iterations,
                                    int neighbors_search, int random_generator_state, int sampler,
                                    int score, double threshold, int final_polisher,
                                    int final_polisher_iterations) {
    const cv::Mat *src = require_const(src_points);
    if (src == nullptr) return nullptr;
    const cv::Mat *dst = require_const(dst_points);
    if (dst == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat h = cv::findHomography(*src, *dst, mask,
                                       make_usac(confidence, is_parallel, lo_iterations, lo_method,
                                                 lo_sample_size, max_iterations, neighbors_search,
                                                 random_generator_state, sampler, score, threshold,
                                                 final_polisher, final_polisher_iterations));
        if (h.empty()) {
            record_error("findHomography: no homography could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(h));
    });
}

void cvk_rq_decomp_3x3(const cvk_mat_t *src, int want_axes,
                       cvk_mat_t **out_r, cvk_mat_t **out_q,
                       cvk_mat_t **out_qx, cvk_mat_t **out_qy, cvk_mat_t **out_qz,
                       double *out_euler3) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr) return;
    guarded([&]() {
        cv::Mat r, q, qx, qy, qz;
        cv::Vec3d euler;
        if (want_axes != 0) {
            euler = cv::RQDecomp3x3(*s, r, q, qx, qy, qz);
        } else {
            euler = cv::RQDecomp3x3(*s, r, q);
        }
        *out_r = reinterpret_cast<cvk_mat_t *>(new cv::Mat(r));
        *out_q = reinterpret_cast<cvk_mat_t *>(new cv::Mat(q));
        if (want_axes != 0) {
            *out_qx = reinterpret_cast<cvk_mat_t *>(new cv::Mat(qx));
            *out_qy = reinterpret_cast<cvk_mat_t *>(new cv::Mat(qy));
            *out_qz = reinterpret_cast<cvk_mat_t *>(new cv::Mat(qz));
        }
        out_euler3[0] = euler[0];
        out_euler3[1] = euler[1];
        out_euler3[2] = euler[2];
    });
}

void cvk_mat_mul_deriv(const cvk_mat_t *a, const cvk_mat_t *b,
                       cvk_mat_t **out_dabda, cvk_mat_t **out_dabdb) {
    const cv::Mat *ma = require_const(a);
    if (ma == nullptr) return;
    const cv::Mat *mb = require_const(b);
    if (mb == nullptr) return;
    guarded([&]() {
        cv::Mat dabda, dabdb;
        cv::matMulDeriv(*ma, *mb, dabda, dabdb);
        *out_dabda = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dabda));
        *out_dabdb = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dabdb));
    });
}

void cvk_compose_rt(const cvk_mat_t *rvec1, const cvk_mat_t *tvec1,
                    const cvk_mat_t *rvec2, const cvk_mat_t *tvec2,
                    int want_derivatives,
                    cvk_mat_t **out_rvec3, cvk_mat_t **out_tvec3,
                    cvk_mat_t **out_dr3dr1, cvk_mat_t **out_dr3dt1,
                    cvk_mat_t **out_dr3dr2, cvk_mat_t **out_dr3dt2,
                    cvk_mat_t **out_dt3dr1, cvk_mat_t **out_dt3dt1,
                    cvk_mat_t **out_dt3dr2, cvk_mat_t **out_dt3dt2) {
    const cv::Mat *r1 = require_const(rvec1);
    if (r1 == nullptr) return;
    const cv::Mat *t1 = require_const(tvec1);
    if (t1 == nullptr) return;
    const cv::Mat *r2 = require_const(rvec2);
    if (r2 == nullptr) return;
    const cv::Mat *t2 = require_const(tvec2);
    if (t2 == nullptr) return;
    guarded([&]() {
        cv::Mat rvec3, tvec3;
        if (want_derivatives != 0) {
            cv::Mat dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2;
            cv::composeRT(*r1, *t1, *r2, *t2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2,
                          dt3dr1, dt3dt1, dt3dr2, dt3dt2);
            *out_dr3dr1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dr3dr1));
            *out_dr3dt1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dr3dt1));
            *out_dr3dr2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dr3dr2));
            *out_dr3dt2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dr3dt2));
            *out_dt3dr1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dt3dr1));
            *out_dt3dt1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dt3dt1));
            *out_dt3dr2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dt3dr2));
            *out_dt3dt2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dt3dt2));
        } else {
            cv::composeRT(*r1, *t1, *r2, *t2, rvec3, tvec3);
        }
        *out_rvec3 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(rvec3));
        *out_tvec3 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(tvec3));
    });
}

cvk_mat_t *cvk_project_points(const cvk_mat_t *object_points, const cvk_mat_t *rvec,
                              const cvk_mat_t *tvec, const cvk_mat_t *camera_matrix,
                              const cvk_mat_t *dist_coeffs) {
    const cv::Mat *obj = require_const(object_points);
    if (obj == nullptr) return nullptr;
    const cv::Mat *r = require_const(rvec);
    if (r == nullptr) return nullptr;
    const cv::Mat *t = require_const(tvec);
    if (t == nullptr) return nullptr;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return nullptr;
    const cv::Mat *d = require_const(dist_coeffs);
    if (d == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat image_points;
        cv::projectPoints(*obj, *r, *t, *k, *d, image_points);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(image_points));
    });
}

cvk_mat_t *cvk_project_points_jacobian(const cvk_mat_t *object_points, const cvk_mat_t *rvec,
                                       const cvk_mat_t *tvec, const cvk_mat_t *camera_matrix,
                                       const cvk_mat_t *dist_coeffs, double aspect_ratio,
                                       cvk_mat_t **out_jacobian) {
    const cv::Mat *obj = require_const(object_points);
    if (obj == nullptr) return nullptr;
    const cv::Mat *r = require_const(rvec);
    if (r == nullptr) return nullptr;
    const cv::Mat *t = require_const(tvec);
    if (t == nullptr) return nullptr;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return nullptr;
    const cv::Mat *d = require_const(dist_coeffs);
    if (d == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat image_points, jacobian;
        cv::projectPoints(*obj, *r, *t, *k, *d, image_points, jacobian, aspect_ratio);
        *out_jacobian = reinterpret_cast<cvk_mat_t *>(new cv::Mat(jacobian));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(image_points));
    });
}

int cvk_solve_pnp(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                  const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                  int use_extrinsic_guess, int flags,
                  cvk_mat_t **out_rvec, cvk_mat_t **out_tvec) {
    const cv::Mat *obj = require_const(object_points);
    if (obj == nullptr) return 0;
    const cv::Mat *img = require_const(image_points);
    if (img == nullptr) return 0;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return 0;
    const cv::Mat *d = require_const(dist_coeffs);
    if (d == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat rvec, tvec;
        const bool ok = cv::solvePnP(*obj, *img, *k, *d, rvec, tvec,
                                     use_extrinsic_guess != 0, flags);
        if (!ok) {
            record_error("solvePnP: no pose could be estimated");
            return 0;
        }
        *out_rvec = reinterpret_cast<cvk_mat_t *>(new cv::Mat(rvec));
        *out_tvec = reinterpret_cast<cvk_mat_t *>(new cv::Mat(tvec));
        return 1;
    });
}

int cvk_solve_pnp_ransac(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                         const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                         int use_extrinsic_guess, int iterations_count, float reprojection_error,
                         double confidence, int flags,
                         cvk_mat_t **out_rvec, cvk_mat_t **out_tvec, cvk_mat_t **out_inliers) {
    const cv::Mat *obj = require_const(object_points);
    if (obj == nullptr) return 0;
    const cv::Mat *img = require_const(image_points);
    if (img == nullptr) return 0;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return 0;
    const cv::Mat *d = require_const(dist_coeffs);
    if (d == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat rvec, tvec, inliers;
        const bool ok = cv::solvePnPRansac(*obj, *img, *k, *d, rvec, tvec,
                                           use_extrinsic_guess != 0, iterations_count,
                                           reprojection_error, confidence, inliers, flags);
        if (!ok) {
            record_error("solvePnPRansac: no pose could be estimated");
            return 0;
        }
        *out_rvec = reinterpret_cast<cvk_mat_t *>(new cv::Mat(rvec));
        *out_tvec = reinterpret_cast<cvk_mat_t *>(new cv::Mat(tvec));
        *out_inliers = reinterpret_cast<cvk_mat_t *>(new cv::Mat(inliers));
        return 1;
    });
}

int cvk_solve_pnp_ransac_usac(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                              const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                              cvk_mat_t **out_rvec, cvk_mat_t **out_tvec, cvk_mat_t **out_inliers,
                              double confidence, int is_parallel, int lo_iterations,
                              int lo_method, int lo_sample_size, int max_iterations,
                              int neighbors_search, int random_generator_state, int sampler,
                              int score, double threshold, int final_polisher,
                              int final_polisher_iterations) {
    const cv::Mat *obj = require_const(object_points);
    if (obj == nullptr) return 0;
    const cv::Mat *img = require_const(image_points);
    if (img == nullptr) return 0;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return 0;
    const cv::Mat *d = require_const(dist_coeffs);
    if (d == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat camera = *k; // InputOutputArray in the USAC overload: work on a copy
        cv::Mat rvec, tvec, inliers;
        const bool ok = cv::solvePnPRansac(
            *obj, *img, camera, *d, rvec, tvec, inliers,
            make_usac(confidence, is_parallel, lo_iterations, lo_method, lo_sample_size,
                      max_iterations, neighbors_search, random_generator_state, sampler, score,
                      threshold, final_polisher, final_polisher_iterations));
        if (!ok) {
            record_error("solvePnPRansac: no pose could be estimated");
            return 0;
        }
        *out_rvec = reinterpret_cast<cvk_mat_t *>(new cv::Mat(rvec));
        *out_tvec = reinterpret_cast<cvk_mat_t *>(new cv::Mat(tvec));
        *out_inliers = reinterpret_cast<cvk_mat_t *>(new cv::Mat(inliers));
        return 1;
    });
}

int cvk_solve_pnp_generic(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                          const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                          int use_extrinsic_guess, int flags,
                          const cvk_mat_t *rvec, const cvk_mat_t *tvec,
                          cvk_mat_t **out_rvecs, cvk_mat_t **out_tvecs,
                          cvk_mat_t **out_reprojection_error) {
    const cv::Mat *obj = require_const(object_points);
    if (obj == nullptr) return 0;
    const cv::Mat *img = require_const(image_points);
    if (img == nullptr) return 0;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return 0;
    const cv::Mat *d = require_const(dist_coeffs);
    if (d == nullptr) return 0;
    return guarded([&]() -> int {
        std::vector<cv::Mat> rvecs, tvecs;
        cv::Mat reprojection_error;
        cv::_InputArray rvec_in = cv::noArray();
        if (rvec != nullptr) rvec_in = cv::_InputArray(*require_const(rvec));
        cv::_InputArray tvec_in = cv::noArray();
        if (tvec != nullptr) tvec_in = cv::_InputArray(*require_const(tvec));
        if (out_reprojection_error != nullptr) {
            const int count = cv::solvePnPGeneric(*obj, *img, *k, *d, rvecs, tvecs,
                                                  use_extrinsic_guess != 0, flags, rvec_in, tvec_in,
                                                  reprojection_error);
            *out_reprojection_error = reinterpret_cast<cvk_mat_t *>(new cv::Mat(reprojection_error));
            *out_rvecs = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(rvecs)));
            *out_tvecs = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(tvecs)));
            return count;
        }
        const int count = cv::solvePnPGeneric(*obj, *img, *k, *d, rvecs, tvecs,
                                              use_extrinsic_guess != 0, flags, rvec_in, tvec_in);
        *out_rvecs = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(rvecs)));
        *out_tvecs = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(tvecs)));
        return count;
    });
}

cvk_mat_t *cvk_convert_points_to_homogeneous(const cvk_mat_t *src, int dtype) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat dst;
        cv::convertPointsToHomogeneous(*s, dst, dtype);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(dst));
    });
}

cvk_mat_t *cvk_convert_points_from_homogeneous(const cvk_mat_t *src, int dtype) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat dst;
        cv::convertPointsFromHomogeneous(*s, dst, dtype);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(dst));
    });
}

cvk_mat_t *cvk_find_fundamental_mat(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                    int method, double ransac_reproj_threshold, double confidence,
                                    int max_iters) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat f = cv::findFundamentalMat(*p1, *p2, method, ransac_reproj_threshold, confidence,
                                           max_iters, cv::noArray());
        if (f.empty()) {
            record_error("findFundamentalMat: no fundamental matrix could be estimated");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(f));
    });
}

cvk_mat_t *cvk_find_fundamental_mat_masked(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                           int method, double ransac_reproj_threshold,
                                           double confidence, int max_iters,
                                           cvk_mat_t **out_mask) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat f = cv::findFundamentalMat(*p1, *p2, method, ransac_reproj_threshold, confidence,
                                           max_iters, mask);
        if (f.empty()) {
            record_error("findFundamentalMat: no fundamental matrix could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(f));
    });
}

cvk_mat_t *cvk_find_fundamental_mat_usac(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                         cvk_mat_t **out_mask,
                                         double confidence, int is_parallel, int lo_iterations,
                                         int lo_method, int lo_sample_size, int max_iterations,
                                         int neighbors_search, int random_generator_state,
                                         int sampler, int score, double threshold,
                                         int final_polisher, int final_polisher_iterations) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat f = cv::findFundamentalMat(
            *p1, *p2, mask,
            make_usac(confidence, is_parallel, lo_iterations, lo_method, lo_sample_size,
                      max_iterations, neighbors_search, random_generator_state, sampler, score,
                      threshold, final_polisher, final_polisher_iterations));
        if (f.empty()) {
            record_error("findFundamentalMat: no fundamental matrix could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(f));
    });
}

cvk_mat_t *cvk_find_essential_mat(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                  const cvk_mat_t *camera_matrix, int method, double prob,
                                  double threshold, int max_iters) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat e = cv::findEssentialMat(*p1, *p2, *k, method, prob, threshold, max_iters,
                                         cv::noArray());
        if (e.empty()) {
            record_error("findEssentialMat: no essential matrix could be estimated");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
    });
}

cvk_mat_t *cvk_find_essential_mat_masked(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                         const cvk_mat_t *camera_matrix, int method, double prob,
                                         double threshold, int max_iters, cvk_mat_t **out_mask) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat e = cv::findEssentialMat(*p1, *p2, *k, method, prob, threshold, max_iters, mask);
        if (e.empty()) {
            record_error("findEssentialMat: no essential matrix could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
    });
}

cvk_mat_t *cvk_find_essential_mat_focal(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                        double focal, double pp_x, double pp_y, int method,
                                        double prob, double threshold, int max_iters) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat e = cv::findEssentialMat(*p1, *p2, focal, cv::Point2d(pp_x, pp_y), method, prob,
                                         threshold, max_iters, cv::noArray());
        if (e.empty()) {
            record_error("findEssentialMat: no essential matrix could be estimated");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
    });
}

cvk_mat_t *cvk_find_essential_mat_focal_masked(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                               double focal, double pp_x, double pp_y, int method,
                                               double prob, double threshold, int max_iters,
                                               cvk_mat_t **out_mask) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat e = cv::findEssentialMat(*p1, *p2, focal, cv::Point2d(pp_x, pp_y), method, prob,
                                         threshold, max_iters, mask);
        if (e.empty()) {
            record_error("findEssentialMat: no essential matrix could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
    });
}

cvk_mat_t *cvk_find_essential_mat_stereo(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                         const cvk_mat_t *camera_matrix1,
                                         const cvk_mat_t *dist_coeffs1,
                                         const cvk_mat_t *camera_matrix2,
                                         const cvk_mat_t *dist_coeffs2, int method, double prob,
                                         double threshold) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    const cv::Mat *k1 = require_const(camera_matrix1);
    if (k1 == nullptr) return nullptr;
    const cv::Mat *d1 = require_const(dist_coeffs1);
    if (d1 == nullptr) return nullptr;
    const cv::Mat *k2 = require_const(camera_matrix2);
    if (k2 == nullptr) return nullptr;
    const cv::Mat *d2 = require_const(dist_coeffs2);
    if (d2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat e = cv::findEssentialMat(*p1, *p2, *k1, *d1, *k2, *d2, method, prob, threshold,
                                         cv::noArray());
        if (e.empty()) {
            record_error("findEssentialMat: no essential matrix could be estimated");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
    });
}

cvk_mat_t *cvk_find_essential_mat_stereo_masked(const cvk_mat_t *points1,
                                                const cvk_mat_t *points2,
                                                const cvk_mat_t *camera_matrix1,
                                                const cvk_mat_t *dist_coeffs1,
                                                const cvk_mat_t *camera_matrix2,
                                                const cvk_mat_t *dist_coeffs2, int method,
                                                double prob, double threshold,
                                                cvk_mat_t **out_mask) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    const cv::Mat *k1 = require_const(camera_matrix1);
    if (k1 == nullptr) return nullptr;
    const cv::Mat *d1 = require_const(dist_coeffs1);
    if (d1 == nullptr) return nullptr;
    const cv::Mat *k2 = require_const(camera_matrix2);
    if (k2 == nullptr) return nullptr;
    const cv::Mat *d2 = require_const(dist_coeffs2);
    if (d2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat e = cv::findEssentialMat(*p1, *p2, *k1, *d1, *k2, *d2, method, prob, threshold,
                                         mask);
        if (e.empty()) {
            record_error("findEssentialMat: no essential matrix could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
    });
}

cvk_mat_t *cvk_find_essential_mat_stereo_usac(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                              const cvk_mat_t *camera_matrix1,
                                              const cvk_mat_t *camera_matrix2,
                                              const cvk_mat_t *dist_coeffs1,
                                              const cvk_mat_t *dist_coeffs2,
                                              cvk_mat_t **out_mask,
                                              double confidence, int is_parallel,
                                              int lo_iterations, int lo_method,
                                              int lo_sample_size, int max_iterations,
                                              int neighbors_search, int random_generator_state,
                                              int sampler, int score, double threshold,
                                              int final_polisher, int final_polisher_iterations) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return nullptr;
    const cv::Mat *k1 = require_const(camera_matrix1);
    if (k1 == nullptr) return nullptr;
    const cv::Mat *k2 = require_const(camera_matrix2);
    if (k2 == nullptr) return nullptr;
    const cv::Mat *d1 = require_const(dist_coeffs1);
    if (d1 == nullptr) return nullptr;
    const cv::Mat *d2 = require_const(dist_coeffs2);
    if (d2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat mask;
        cv::Mat e = cv::findEssentialMat(
            *p1, *p2, *k1, *k2, *d1, *d2, mask,
            make_usac(confidence, is_parallel, lo_iterations, lo_method, lo_sample_size,
                      max_iterations, neighbors_search, random_generator_state, sampler, score,
                      threshold, final_polisher, final_polisher_iterations));
        if (e.empty()) {
            record_error("findEssentialMat: no essential matrix could be estimated");
            return nullptr;
        }
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
    });
}

void cvk_decompose_essential_mat(const cvk_mat_t *e,
                                 cvk_mat_t **out_r1, cvk_mat_t **out_r2, cvk_mat_t **out_t) {
    const cv::Mat *em = require_const(e);
    if (em == nullptr) return;
    guarded([&]() {
        cv::Mat r1, r2, t;
        cv::decomposeEssentialMat(*em, r1, r2, t);
        *out_r1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(r1));
        *out_r2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(r2));
        *out_t = reinterpret_cast<cvk_mat_t *>(new cv::Mat(t));
    });
}

int cvk_recover_pose(const cvk_mat_t *points1, const cvk_mat_t *points2,
                     const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                     const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                     int method, double prob, double threshold, const cvk_mat_t *mask_in,
                     cvk_mat_t **out_e, cvk_mat_t **out_r, cvk_mat_t **out_t,
                     cvk_mat_t **out_mask) {
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return 0;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return 0;
    const cv::Mat *k1 = require_const(camera_matrix1);
    if (k1 == nullptr) return 0;
    const cv::Mat *d1 = require_const(dist_coeffs1);
    if (d1 == nullptr) return 0;
    const cv::Mat *k2 = require_const(camera_matrix2);
    if (k2 == nullptr) return 0;
    const cv::Mat *d2 = require_const(dist_coeffs2);
    if (d2 == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat mask;
        if (mask_in != nullptr) mask = *require_const(mask_in);
        cv::Mat e, r, t;
        const int count = cv::recoverPose(*p1, *p2, *k1, *d1, *k2, *d2, e, r, t, method, prob,
                                          threshold, mask);
        *out_e = reinterpret_cast<cvk_mat_t *>(new cv::Mat(e));
        *out_r = reinterpret_cast<cvk_mat_t *>(new cv::Mat(r));
        *out_t = reinterpret_cast<cvk_mat_t *>(new cv::Mat(t));
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return count;
    });
}

int cvk_recover_pose_e(const cvk_mat_t *e, const cvk_mat_t *points1, const cvk_mat_t *points2,
                       const cvk_mat_t *camera_matrix, const cvk_mat_t *mask_in,
                       cvk_mat_t **out_r, cvk_mat_t **out_t, cvk_mat_t **out_mask) {
    const cv::Mat *em = require_const(e);
    if (em == nullptr) return 0;
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return 0;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return 0;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat mask;
        if (mask_in != nullptr) mask = *require_const(mask_in);
        cv::Mat r, t;
        const int count = cv::recoverPose(*em, *p1, *p2, *k, r, t, mask);
        *out_r = reinterpret_cast<cvk_mat_t *>(new cv::Mat(r));
        *out_t = reinterpret_cast<cvk_mat_t *>(new cv::Mat(t));
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return count;
    });
}

int cvk_recover_pose_e_focal(const cvk_mat_t *e, const cvk_mat_t *points1,
                             const cvk_mat_t *points2, double focal, double pp_x, double pp_y,
                             const cvk_mat_t *mask_in,
                             cvk_mat_t **out_r, cvk_mat_t **out_t, cvk_mat_t **out_mask) {
    const cv::Mat *em = require_const(e);
    if (em == nullptr) return 0;
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return 0;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat mask;
        if (mask_in != nullptr) mask = *require_const(mask_in);
        cv::Mat r, t;
        const int count =
            cv::recoverPose(*em, *p1, *p2, r, t, focal, cv::Point2d(pp_x, pp_y), mask);
        *out_r = reinterpret_cast<cvk_mat_t *>(new cv::Mat(r));
        *out_t = reinterpret_cast<cvk_mat_t *>(new cv::Mat(t));
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        return count;
    });
}

int cvk_recover_pose_e_distance(const cvk_mat_t *e, const cvk_mat_t *points1,
                                const cvk_mat_t *points2, const cvk_mat_t *camera_matrix,
                                double distance_thresh, const cvk_mat_t *mask_in,
                                cvk_mat_t **out_r, cvk_mat_t **out_t, cvk_mat_t **out_mask,
                                cvk_mat_t **out_triangulated) {
    const cv::Mat *em = require_const(e);
    if (em == nullptr) return 0;
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return 0;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return 0;
    const cv::Mat *k = require_const(camera_matrix);
    if (k == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat mask;
        if (mask_in != nullptr) mask = *require_const(mask_in);
        cv::Mat r, t, triangulated;
        const int count = cv::recoverPose(*em, *p1, *p2, *k, r, t, distance_thresh, mask,
                                          triangulated);
        *out_r = reinterpret_cast<cvk_mat_t *>(new cv::Mat(r));
        *out_t = reinterpret_cast<cvk_mat_t *>(new cv::Mat(t));
        *out_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mask));
        *out_triangulated = reinterpret_cast<cvk_mat_t *>(new cv::Mat(triangulated));
        return count;
    });
}

cvk_mat_t *cvk_triangulate_points(const cvk_mat_t *proj_matr1, const cvk_mat_t *proj_matr2,
                                  const cvk_mat_t *proj_points1, const cvk_mat_t *proj_points2) {
    const cv::Mat *p1 = require_const(proj_matr1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(proj_matr2);
    if (p2 == nullptr) return nullptr;
    const cv::Mat *q1 = require_const(proj_points1);
    if (q1 == nullptr) return nullptr;
    const cv::Mat *q2 = require_const(proj_points2);
    if (q2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat points4d;
        cv::triangulatePoints(*p1, *p2, *q1, *q2, points4d);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(points4d));
    });
}

void cvk_correct_matches(const cvk_mat_t *f, const cvk_mat_t *points1, const cvk_mat_t *points2,
                         cvk_mat_t **out_new_points1, cvk_mat_t **out_new_points2) {
    const cv::Mat *fm = require_const(f);
    if (fm == nullptr) return;
    const cv::Mat *p1 = require_const(points1);
    if (p1 == nullptr) return;
    const cv::Mat *p2 = require_const(points2);
    if (p2 == nullptr) return;
    guarded([&]() {
        cv::Mat np1, np2;
        cv::correctMatches(*fm, *p1, *p2, np1, np2);
        *out_new_points1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(np1));
        *out_new_points2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(np2));
    });
}

int cvk_estimate_affine_3d(const cvk_mat_t *src, const cvk_mat_t *dst, double ransac_threshold,
                           double confidence, cvk_mat_t **out_transform,
                           cvk_mat_t **out_inliers) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr) return 0;
    const cv::Mat *d = require_const(dst);
    if (d == nullptr) return 0;
    return guarded([&]() -> int {
        cv::Mat transform, inliers;
        const bool ok = cv::estimateAffine3D(*s, *d, transform, inliers, ransac_threshold,
                                             confidence);
        if (!ok) {
            record_error("estimateAffine3D: no transformation could be estimated");
            return 0;
        }
        *out_transform = reinterpret_cast<cvk_mat_t *>(new cv::Mat(transform));
        *out_inliers = reinterpret_cast<cvk_mat_t *>(new cv::Mat(inliers));
        return 1;
    });
}

cvk_mat_t *cvk_estimate_affine_3d_umeyama(const cvk_mat_t *src, const cvk_mat_t *dst,
                                          int force_rotation, double *out_scale) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr) return nullptr;
    const cv::Mat *d = require_const(dst);
    if (d == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        double scale = 1.0;
        cv::Mat transform = cv::estimateAffine3D(*s, *d, &scale, force_rotation != 0);
        if (transform.empty()) {
            record_error("estimateAffine3D: no transformation could be estimated");
            return nullptr;
        }
        if (out_scale != nullptr) *out_scale = scale;
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(transform));
    });
}

cvk_mat_t *cvk_estimate_affine_2d_usac(const cvk_mat_t *pts1, const cvk_mat_t *pts2,
                                       cvk_mat_t **out_inliers,
                                       double confidence, int is_parallel, int lo_iterations,
                                       int lo_method, int lo_sample_size, int max_iterations,
                                       int neighbors_search, int random_generator_state,
                                       int sampler, int score, double threshold,
                                       int final_polisher, int final_polisher_iterations) {
    const cv::Mat *p1 = require_const(pts1);
    if (p1 == nullptr) return nullptr;
    const cv::Mat *p2 = require_const(pts2);
    if (p2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat inliers;
        cv::Mat transform = cv::estimateAffine2D(
            *p1, *p2, inliers,
            make_usac(confidence, is_parallel, lo_iterations, lo_method, lo_sample_size,
                      max_iterations, neighbors_search, random_generator_state, sampler, score,
                      threshold, final_polisher, final_polisher_iterations));
        *out_inliers = reinterpret_cast<cvk_mat_t *>(new cv::Mat(inliers));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(transform));
    });
}

int cvk_decompose_homography_mat(const cvk_mat_t *h, const cvk_mat_t *k,
                                 cvk_mat_t **out_rotations, cvk_mat_t **out_translations,
                                 cvk_mat_t **out_normals) {
    const cv::Mat *hm = require_const(h);
    if (hm == nullptr) return 0;
    const cv::Mat *km = require_const(k);
    if (km == nullptr) return 0;
    return guarded([&]() -> int {
        std::vector<cv::Mat> rotations, translations, normals;
        const int count = cv::decomposeHomographyMat(*hm, *km, rotations, translations, normals);
        *out_rotations = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(rotations)));
        *out_translations = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(translations)));
        *out_normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(normals)));
        return count;
    });
}

void cvk_filter_homography_decomp(const cvk_mat_t *rotations, const cvk_mat_t *normals,
                                  const cvk_mat_t *before_points, const cvk_mat_t *after_points,
                                  const cvk_mat_t *points_mask,
                                  cvk_mat_t **out_possible_solutions) {
    const cv::Mat *r = require_const(rotations);
    if (r == nullptr) return;
    const cv::Mat *n = require_const(normals);
    if (n == nullptr) return;
    const cv::Mat *bp = require_const(before_points);
    if (bp == nullptr) return;
    const cv::Mat *ap = require_const(after_points);
    if (ap == nullptr) return;
    guarded([&]() {
        const std::vector<cv::Mat> rots = unpack_mats(*r);
        const std::vector<cv::Mat> norms = unpack_mats(*n);
        cv::Mat solutions;
        if (points_mask != nullptr) {
            cv::filterHomographyDecompByVisibleRefpoints(rots, norms, *bp, *ap, solutions,
                                                         *require_const(points_mask));
        } else {
            cv::filterHomographyDecompByVisibleRefpoints(rots, norms, *bp, *ap, solutions);
        }
        *out_possible_solutions = reinterpret_cast<cvk_mat_t *>(new cv::Mat(solutions));
    });
}

/* =========================================================================
 * Subdiv2D
 * ========================================================================= */

struct cvk_subdiv2d {
    cv::Subdiv2D obj;
};

cvk_subdiv2d_t *cvk_subdiv2d_create(void) {
    return guarded([&]() -> cvk_subdiv2d_t * {
        return reinterpret_cast<cvk_subdiv2d_t *>(new cvk_subdiv2d);
    });
}

cvk_subdiv2d_t *cvk_subdiv2d_create_with_rect(int x, int y, int width, int height) {
    return guarded([&]() -> cvk_subdiv2d_t * {
        auto *h = new cvk_subdiv2d;
        h->obj = cv::Subdiv2D(cv::Rect(x, y, width, height));
        return reinterpret_cast<cvk_subdiv2d_t *>(h);
    });
}

void cvk_subdiv2d_init_delaunay(cvk_subdiv2d_t *h, int x, int y, int width, int height) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return;
    }
    guarded([&]() { reinterpret_cast<cvk_subdiv2d *>(h)->obj.initDelaunay(
                        cv::Rect(x, y, width, height)); });
}

int cvk_subdiv2d_insert_point(cvk_subdiv2d_t *h, double x, double y) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return -1;
    }
    return guarded([&]() -> int {
        return reinterpret_cast<cvk_subdiv2d *>(h)->obj.insert(cv::Point2f(
            static_cast<float>(x), static_cast<float>(y)));
    });
}

void cvk_subdiv2d_insert_points(cvk_subdiv2d_t *h, const cvk_mat_t *points) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return;
    }
    const cv::Mat *p = require_const(points);
    if (p == nullptr) return;
    guarded([&]() {
        std::vector<cv::Point2f> pts;
        pts.reserve(static_cast<size_t>(p->rows));
        for (int i = 0; i < p->rows; ++i) {
            pts.emplace_back(p->at<float>(i, 0), p->at<float>(i, 1));
        }
        reinterpret_cast<cvk_subdiv2d *>(h)->obj.insert(pts);
    });
}

int cvk_subdiv2d_locate(const cvk_subdiv2d_t *h, double x, double y,
                        int *out_edge, int *out_vertex) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return -2;
    }
    return guarded([&]() -> int {
        int edge = 0, vertex = 0;
        cv::Subdiv2D &subdiv = reinterpret_cast<cvk_subdiv2d *>(const_cast<cvk_subdiv2d_t *>(h))->obj;
        const int code = subdiv.locate(cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                                       edge, vertex);
        *out_edge = edge;
        *out_vertex = vertex;
        return code;
    });
}

int cvk_subdiv2d_find_nearest(const cvk_subdiv2d_t *h, double x, double y,
                              double *out_nearest_x, double *out_nearest_y) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return -1;
    }
    return guarded([&]() -> int {
        cv::Point2f nearest;
        cv::Subdiv2D &subdiv = reinterpret_cast<cvk_subdiv2d *>(const_cast<cvk_subdiv2d_t *>(h))->obj;
        const int vertex = subdiv.findNearest(
            cv::Point2f(static_cast<float>(x), static_cast<float>(y)), &nearest);
        *out_nearest_x = nearest.x;
        *out_nearest_y = nearest.y;
        return vertex;
    });
}

cvk_mat_t *cvk_subdiv2d_get_edge_list(const cvk_subdiv2d_t *h) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Vec4f> edges;
        reinterpret_cast<const cvk_subdiv2d *>(h)->obj.getEdgeList(edges);
        if (edges.empty()) return reinterpret_cast<cvk_mat_t *>(new cv::Mat());
        cv::Mat out(static_cast<int>(edges.size()), 1, CV_32FC4);
        for (size_t i = 0; i < edges.size(); ++i) {
            out.at<cv::Vec4f>(static_cast<int>(i), 0) = edges[i];
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

cvk_mat_t *cvk_subdiv2d_get_leading_edge_list(const cvk_subdiv2d_t *h) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        std::vector<int> edges;
        reinterpret_cast<const cvk_subdiv2d *>(h)->obj.getLeadingEdgeList(edges);
        if (edges.empty()) return reinterpret_cast<cvk_mat_t *>(new cv::Mat());
        cv::Mat out(static_cast<int>(edges.size()), 1, CV_32SC1);
        for (size_t i = 0; i < edges.size(); ++i) {
            out.at<int>(static_cast<int>(i), 0) = edges[i];
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

cvk_mat_t *cvk_subdiv2d_get_triangle_list(const cvk_subdiv2d_t *h) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Vec6f> triangles;
        reinterpret_cast<const cvk_subdiv2d *>(h)->obj.getTriangleList(triangles);
        if (triangles.empty()) return reinterpret_cast<cvk_mat_t *>(new cv::Mat());
        cv::Mat out(static_cast<int>(triangles.size()), 1, CV_MAKETYPE(CV_32F, 6));
        for (size_t i = 0; i < triangles.size(); ++i) {
            out.at<cv::Vec6f>(static_cast<int>(i), 0) = triangles[i];
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

cvk_mat_t *cvk_subdiv2d_get_voronoi_facet_list(const cvk_subdiv2d_t *h, const cvk_mat_t *idx,
                                               cvk_mat_t **out_facet_centers) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return nullptr;
    }
    const cv::Mat *idxs = require_const(idx);
    if (idxs == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<int> ids;
        if (!idxs->empty()) {
            ids.reserve(static_cast<size_t>(idxs->rows));
            for (int i = 0; i < idxs->rows; ++i) {
                ids.push_back(idxs->at<int>(i, 0));
            }
        }
        std::vector<std::vector<cv::Point2f>> facets;
        std::vector<cv::Point2f> centers;
        reinterpret_cast<cvk_subdiv2d *>(const_cast<cvk_subdiv2d_t *>(h))
            ->obj.getVoronoiFacetList(ids, facets, centers);
        std::vector<cv::Mat> facet_mats;
        facet_mats.reserve(facets.size());
        for (const auto &facet : facets) {
            cv::Mat fm(static_cast<int>(facet.size()), 1, CV_32FC2);
            for (size_t i = 0; i < facet.size(); ++i) {
                fm.at<cv::Vec2f>(static_cast<int>(i), 0) =
                    cv::Vec2f(facet[i].x, facet[i].y);
            }
            facet_mats.push_back(fm);
        }
        cv::Mat centers_mat(static_cast<int>(centers.size()), 1, CV_32FC2);
        for (size_t i = 0; i < centers.size(); ++i) {
            centers_mat.at<cv::Vec2f>(static_cast<int>(i), 0) =
                cv::Vec2f(centers[i].x, centers[i].y);
        }
        *out_facet_centers = reinterpret_cast<cvk_mat_t *>(new cv::Mat(centers_mat));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(pack_mats(facet_mats)));
    });
}

void cvk_subdiv2d_get_vertex(const cvk_subdiv2d_t *h, int vertex,
                             double *out_x, double *out_y, int *out_first_edge) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return;
    }
    guarded([&]() {
        int first_edge = 0;
        const cv::Point2f pt = reinterpret_cast<const cvk_subdiv2d *>(h)->obj.getVertex(
            vertex, &first_edge);
        *out_x = pt.x;
        *out_y = pt.y;
        *out_first_edge = first_edge;
    });
}

int cvk_subdiv2d_get_edge(const cvk_subdiv2d_t *h, int edge, int next_edge_type) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return 0;
    }
    return guarded([&]() -> int {
        return reinterpret_cast<const cvk_subdiv2d *>(h)->obj.getEdge(edge, next_edge_type);
    });
}

int cvk_subdiv2d_next_edge(const cvk_subdiv2d_t *h, int edge) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return 0;
    }
    return guarded([&]() -> int {
        return reinterpret_cast<const cvk_subdiv2d *>(h)->obj.nextEdge(edge);
    });
}

int cvk_subdiv2d_rotate_edge(const cvk_subdiv2d_t *h, int edge, int rotate) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return 0;
    }
    return guarded([&]() -> int {
        return reinterpret_cast<const cvk_subdiv2d *>(h)->obj.rotateEdge(edge, rotate);
    });
}

int cvk_subdiv2d_sym_edge(const cvk_subdiv2d_t *h, int edge) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return 0;
    }
    return guarded([&]() -> int {
        return reinterpret_cast<const cvk_subdiv2d *>(h)->obj.symEdge(edge);
    });
}

int cvk_subdiv2d_edge_org(const cvk_subdiv2d_t *h, int edge,
                          double *out_x, double *out_y) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return 0;
    }
    return guarded([&]() -> int {
        cv::Point2f orgpt;
        const int vertex =
            reinterpret_cast<const cvk_subdiv2d *>(h)->obj.edgeOrg(edge, &orgpt);
        *out_x = orgpt.x;
        *out_y = orgpt.y;
        return vertex;
    });
}

int cvk_subdiv2d_edge_dst(const cvk_subdiv2d_t *h, int edge,
                          double *out_x, double *out_y) {
    if (h == nullptr) {
        record_error("null Subdiv2D handle");
        return 0;
    }
    return guarded([&]() -> int {
        cv::Point2f dstpt;
        const int vertex =
            reinterpret_cast<const cvk_subdiv2d *>(h)->obj.edgeDst(edge, &dstpt);
        *out_x = dstpt.x;
        *out_y = dstpt.y;
        return vertex;
    });
}

void cvk_subdiv2d_release(cvk_subdiv2d_t *h) {
    delete reinterpret_cast<cvk_subdiv2d *>(h);
}

} /* extern "C" */

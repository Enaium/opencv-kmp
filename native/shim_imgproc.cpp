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
 * cvk_ C ABI implementations for the imgproc statics slice (opencv_kmp_imgproc.h).
 * Every exported function is noexcept; failures return NULL / 0 / NaN and are
 * recorded through the module-local error slot (see cvk_last_error in
 * opencv_shim.cpp for the shared slot the Kotlin layer reads).
 */
#include "opencv_kmp.h"
#include "opencv_kmp_imgproc.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>

#include <algorithm>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string g_imgproc_error;

void record_error(const char *message) {
    g_imgproc_error = message != nullptr ? message : "unknown error";
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

cv::RotatedRect to_rotated_rect(cvk_rotated_rect_t r) {
    return cv::RotatedRect(cv::Point2f(static_cast<float>(r.cx), static_cast<float>(r.cy)),
                           cv::Size2f(static_cast<float>(r.width), static_cast<float>(r.height)),
                           static_cast<float>(r.angle));
}

} // namespace

extern "C" {

/* =========================================================================
 * kernels / filters
 * ========================================================================= */

int cvk_get_deriv_kernels(int dx, int dy, int ksize, int normalize, int ktype,
                          cvk_mat_t **kx, cvk_mat_t **ky) {
    if (kx == nullptr || ky == nullptr) {
        record_error("null output handle in getDerivKernels");
        return 0;
    }
    *kx = nullptr;
    *ky = nullptr;
    return guarded([&]() -> int {
        auto *x = new cv::Mat();
        auto *y = new cv::Mat();
        cv::getDerivKernels(*x, *y, dx, dy, ksize, normalize != 0, ktype);
        *kx = reinterpret_cast<cvk_mat_t *>(x);
        *ky = reinterpret_cast<cvk_mat_t *>(y);
        return 1;
    });
}

cvk_mat_t *cvk_get_gabor_kernel(int ksize_width, int ksize_height, double sigma,
                                double theta, double lambd, double gamma,
                                double psi, int ktype) {
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat kernel = cv::getGaborKernel(cv::Size(ksize_width, ksize_height),
                                            sigma, theta, lambd, gamma, psi, ktype);
        if (kernel.empty()) {
            record_error("getGaborKernel produced an empty kernel");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(kernel));
    });
}

cvk_mat_t *cvk_sep_filter_2d(const cvk_mat_t *src, int ddepth,
                             const cvk_mat_t *kx, const cvk_mat_t *ky,
                             int anchor_x, int anchor_y, double delta,
                             int border_type) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *mkx = require_const(kx);
    const cv::Mat *mky = require_const(ky);
    if (m == nullptr || mkx == nullptr || mky == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::sepFilter2D(*m, *dst, ddepth, *mkx, *mky,
                        cv::Point(anchor_x, anchor_y), delta, border_type);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * corners
 * ========================================================================= */

cvk_mat_t *cvk_pre_corner_detect(const cvk_mat_t *src, int ksize,
                                 int border_type) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::preCornerDetect(*m, *dst, ksize, border_type);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_corner_eigen_vals_and_vecs(const cvk_mat_t *src, int block_size,
                                          int ksize, int border_type) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::cornerEigenValsAndVecs(*m, *dst, block_size, ksize, border_type);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * color conversion
 * ========================================================================= */

cvk_mat_t *cvk_cvt_color_two_plane(const cvk_mat_t *src1, const cvk_mat_t *src2,
                                   int code) {
    const cv::Mat *m1 = require_const(src1);
    const cv::Mat *m2 = require_const(src2);
    if (m1 == nullptr || m2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::cvtColorTwoPlane(*m1, *m2, *dst, code);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * calibration helpers
 * ========================================================================= */

int cvk_init_undistort_rectify_map(const cvk_mat_t *camera_matrix,
                                   const cvk_mat_t *dist_coeffs,
                                   const cvk_mat_t *r,
                                   const cvk_mat_t *new_camera_matrix,
                                   int size_width, int size_height, int m1_type,
                                   cvk_mat_t **map1, cvk_mat_t **map2) {
    const cv::Mat *cam = require_const(camera_matrix);
    const cv::Mat *dist = require_const(dist_coeffs);
    if (cam == nullptr || dist == nullptr || map1 == nullptr || map2 == nullptr) {
        record_error("null argument in initUndistortRectifyMap");
        return 0;
    }
    *map1 = nullptr;
    *map2 = nullptr;
    return guarded([&]() -> int {
        auto *m1 = new cv::Mat();
        auto *m2 = new cv::Mat();
        cv::initUndistortRectifyMap(
            *cam, *dist,
            r != nullptr ? *require_const(r) : cv::noArray(),
            new_camera_matrix != nullptr ? *require_const(new_camera_matrix) : cv::noArray(),
            cv::Size(size_width, size_height), m1_type, *m1, *m2);
        *map1 = reinterpret_cast<cvk_mat_t *>(m1);
        *map2 = reinterpret_cast<cvk_mat_t *>(m2);
        return 1;
    });
}

cvk_mat_t *cvk_undistort_points(const cvk_mat_t *src,
                                const cvk_mat_t *camera_matrix,
                                const cvk_mat_t *dist_coeffs,
                                const cvk_mat_t *r, const cvk_mat_t *p) {
    const cv::Mat *s = require_const(src);
    const cv::Mat *cam = require_const(camera_matrix);
    const cv::Mat *dist = require_const(dist_coeffs);
    if (s == nullptr || cam == nullptr || dist == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::undistortPoints(*s, *dst, *cam, *dist,
                            r != nullptr ? *require_const(r) : cv::noArray(),
                            p != nullptr ? *require_const(p) : cv::noArray());
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_get_default_new_camera_matrix(const cvk_mat_t *camera_matrix,
                                             int size_width, int size_height,
                                             int center_principal_point) {
    const cv::Mat *cam = require_const(camera_matrix);
    if (cam == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = cv::getDefaultNewCameraMatrix(
            *cam, cv::Size(size_width, size_height), center_principal_point != 0);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
    });
}

cvk_mat_t *cvk_estimate_affine_2d(const cvk_mat_t *from, const cvk_mat_t *to,
                                  int method, double ransac_reproj_threshold,
                                  long long max_iters, double confidence,
                                  long long refine_iters) {
    const cv::Mat *f = require_const(from);
    const cv::Mat *t = require_const(to);
    if (f == nullptr || t == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = cv::estimateAffine2D(*f, *t, cv::noArray(), method,
                                         ransac_reproj_threshold,
                                         static_cast<size_t>(max_iters), confidence,
                                         static_cast<size_t>(refine_iters));
        if (m.empty()) {
            record_error("estimateAffine2D could not estimate a transformation");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
    });
}

cvk_mat_t *cvk_estimate_affine_partial_2d(const cvk_mat_t *from,
                                          const cvk_mat_t *to, int method,
                                          double ransac_reproj_threshold,
                                          long long max_iters, double confidence,
                                          long long refine_iters) {
    const cv::Mat *f = require_const(from);
    const cv::Mat *t = require_const(to);
    if (f == nullptr || t == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = cv::estimateAffinePartial2D(*f, *t, cv::noArray(), method,
                                                ransac_reproj_threshold,
                                                static_cast<size_t>(max_iters),
                                                confidence,
                                                static_cast<size_t>(refine_iters));
        if (m.empty()) {
            record_error("estimateAffinePartial2D could not estimate a transformation");
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
    });
}

/* =========================================================================
 * distance transform
 * ========================================================================= */

int cvk_distance_transform_with_labels(const cvk_mat_t *src, int distance_type,
                                       int mask_size, int label_type,
                                       cvk_mat_t **dst, cvk_mat_t **labels) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr || dst == nullptr || labels == nullptr) {
        record_error("null argument in distanceTransformWithLabels");
        return 0;
    }
    *dst = nullptr;
    *labels = nullptr;
    return guarded([&]() -> int {
        auto *d = new cv::Mat();
        auto *l = new cv::Mat();
        cv::distanceTransform(*s, *d, *l, distance_type, mask_size, label_type);
        *dst = reinterpret_cast<cvk_mat_t *>(d);
        *labels = reinterpret_cast<cvk_mat_t *>(l);
        return 1;
    });
}

/* =========================================================================
 * contours / geometry
 * ========================================================================= */

cvk_mat_t *cvk_convex_hull(const cvk_mat_t *points, int clockwise,
                           int return_points) {
    const cv::Mat *p = require_const(points);
    if (p == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *hull = new cv::Mat();
        cv::convexHull(*p, *hull, clockwise != 0, return_points != 0);
        return reinterpret_cast<cvk_mat_t *>(hull);
    });
}

int cvk_is_contour_convex(const cvk_mat_t *points) {
    const cv::Mat *p = require_const(points);
    if (p == nullptr) return 0;
    return guarded([&] { return cv::isContourConvex(*p) ? 1 : 0; });
}

cvk_mat_t *cvk_convexity_defects(const cvk_mat_t *contour,
                                 const cvk_mat_t *hull_idx) {
    const cv::Mat *c = require_const(contour);
    const cv::Mat *h = require_const(hull_idx);
    if (c == nullptr || h == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *defects = new cv::Mat();
        cv::convexityDefects(*c, *h, *defects);
        return reinterpret_cast<cvk_mat_t *>(defects);
    });
}

cvk_mat_t *cvk_fit_line(const cvk_mat_t *points, int dist_type, double param,
                        double reps, double aeps) {
    const cv::Mat *p = require_const(points);
    if (p == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *line = new cv::Mat();
        cv::fitLine(*p, *line, dist_type, param, reps, aeps);
        return reinterpret_cast<cvk_mat_t *>(line);
    });
}

cvk_mat_t *cvk_box_points(cvk_rotated_rect_t box) {
    return guarded([&]() -> cvk_mat_t * {
        auto *points = new cv::Mat();
        cv::boxPoints(to_rotated_rect(box), *points);
        return reinterpret_cast<cvk_mat_t *>(points);
    });
}

int cvk_rotated_rectangle_intersection(cvk_rotated_rect_t rect1,
                                       cvk_rotated_rect_t rect2,
                                       cvk_mat_t **intersection) {
    if (intersection == nullptr) {
        record_error("null output handle in rotatedRectangleIntersection");
        return 0;
    }
    *intersection = nullptr;
    return guarded([&]() -> int {
        auto *poly = new cv::Mat();
        const int type = cv::rotatedRectangleIntersection(to_rotated_rect(rect1),
                                                          to_rotated_rect(rect2), *poly);
        *intersection = reinterpret_cast<cvk_mat_t *>(poly);
        return type;
    });
}

double cvk_point_polygon_test(const cvk_mat_t *contour, double x, double y,
                              int measure_dist) {
    const cv::Mat *c = require_const(contour);
    if (c == nullptr) return 0.0;
    return guarded([&] {
        return cv::pointPolygonTest(
            *c, cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
            measure_dist != 0);
    });
}

float cvk_intersect_convex_convex(const cvk_mat_t *p1, const cvk_mat_t *p2,
                                  int handle_nested, cvk_mat_t **p12) {
    const cv::Mat *a = require_const(p1);
    const cv::Mat *b = require_const(p2);
    if (a == nullptr || b == nullptr || p12 == nullptr) {
        record_error("null argument in intersectConvexConvex");
        return 0.0f;
    }
    *p12 = nullptr;
    return guarded([&]() -> float {
        auto *poly = new cv::Mat();
        const float area = cv::intersectConvexConvex(*a, *b, *poly, handle_nested != 0);
        *p12 = reinterpret_cast<cvk_mat_t *>(poly);
        return area;
    });
}

void cvk_hu_moments(cvk_moments_t moments, double out[7]) {
    if (out == nullptr) return;
    guarded([&]() -> void * {
        // The 10-arg Moments ctor runs completeMomentState, filling the
        // central (mu) and normalized (nu) moments HuMoments needs.
        cv::Moments m(moments.m00, moments.m10, moments.m01, moments.m20,
                      moments.m11, moments.m02, moments.m30, moments.m21,
                      moments.m12, moments.m03);
        cv::HuMoments(m, out);
        return nullptr;
    });
}

/* =========================================================================
 * pyramids
 * ========================================================================= */

int cvk_build_pyramid(const cvk_mat_t *src, int max_level, int border_type,
                      cvk_mat_t **levels,
                      int max_count) {
    const cv::Mat *s = require_const(src);
    if (s == nullptr || levels == nullptr || max_count <= 0) {
        record_error("null argument in buildPyramid");
        return 0;
    }
    for (int i = 0; i < max_count; ++i) levels[i] = nullptr;
    return guarded([&]() -> int {
        std::vector<cv::Mat> pyramid;
        cv::buildPyramid(*s, pyramid, max_level, border_type);
        const int count = std::min(static_cast<int>(pyramid.size()), max_count);
        for (int i = 0; i < count; ++i) {
            levels[i] = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pyramid[i]));
        }
        return count;
    });
}

} /* extern "C" */

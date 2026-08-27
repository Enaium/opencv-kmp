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
 * cvk_ C ABI implementation for the OpenCV "features" module (see
 * opencv_kmp_features.h). Every exported function is noexcept: bodies run
 * inside `guarded` and failures are reported through cvk_last_error().
 *
 * KeyPoint collections marshal as CV_32FC(7) Mats (x, y, size, angle,
 * response, octave, classId); MSER/blob contours use the shared contour
 * flat buffer; float vectors (AffineFeature view params) use CV_32FC1 Mats.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_features.h"

#include <opencv2/features.hpp>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <vector>

/* Opaque handles completing the C forward declarations. Every concrete
 * detector struct holds exactly one cv::Ptr of its algorithm type; the
 * layout (one Ptr) matches cvk_feature2d, so any handle can be viewed
 * through the generic Feature2D prefix (used by cvk_affine_create). */
struct cvk_feature2d { cv::Ptr<cv::Feature2D> ptr; };
struct cvk_sift { cv::Ptr<cv::SIFT> ptr; };
struct cvk_orb { cv::Ptr<cv::ORB> ptr; };
struct cvk_mser { cv::Ptr<cv::MSER> ptr; };
struct cvk_fast_feature_detector { cv::Ptr<cv::FastFeatureDetector> ptr; };
struct cvk_gftt_detector { cv::Ptr<cv::GFTTDetector> ptr; };
struct cvk_simple_blob_detector { cv::Ptr<cv::SimpleBlobDetector> ptr; };
struct cvk_affine { cv::Ptr<cv::AffineFeature> ptr; };

namespace {

thread_local std::string g_features_str;

void record_error(const char *message) {
    try {
        g_features_str = message != nullptr ? message : "unknown error";
    } catch (...) {
        /* no alternative error channel */
    }
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

/* ---- KeyPoint <-> CV_32FC(7) Mat -------------------------------------- */

void keypoints_to_mat(const std::vector<cv::KeyPoint> &kps, cv::Mat &m) {
    m = cv::Mat(static_cast<int>(kps.size()), 1, CV_32FC(7));
    for (int i = 0; i < static_cast<int>(kps.size()); ++i) {
        float *row = m.ptr<float>(i);
        row[0] = kps[i].pt.x;
        row[1] = kps[i].pt.y;
        row[2] = kps[i].size;
        row[3] = kps[i].angle;
        row[4] = kps[i].response;
        row[5] = static_cast<float>(kps[i].octave);
        row[6] = static_cast<float>(kps[i].class_id);
    }
}

void mat_to_keypoints(const cv::Mat &m, std::vector<cv::KeyPoint> &kps) {
    kps.clear();
    const int count = m.rows;
    if (count <= 0) return;
    kps.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const float *row = m.ptr<float>(i);
        kps.emplace_back(cv::Point2f(row[0], row[1]), row[2], row[3], row[4],
                         static_cast<int>(row[5]), static_cast<int>(row[6]));
    }
}

/* ---- float vector <-> CV_32FC1 Mat ------------------------------------ */

std::vector<float> floats_of(const cv::Mat &m) {
    std::vector<float> out;
    out.reserve(static_cast<size_t>(m.rows) * static_cast<size_t>(m.cols));
    for (int r = 0; r < m.rows; ++r) {
        const float *row = m.ptr<float>(r);
        for (int c = 0; c < m.cols; ++c) out.push_back(row[c]);
    }
    return out;
}

cv::Mat mat_of_floats(const std::vector<float> &v) {
    cv::Mat m(static_cast<int>(v.size()), 1, CV_32FC1);
    for (size_t i = 0; i < v.size(); ++i) {
        m.ptr<float>(static_cast<int>(i))[0] = v[i];
    }
    return m;
}

/* ---- contour flat buffer (same wire format as cvk_find_contours) ------ */

void put_u32le(unsigned char *p, unsigned int value) {
    p[0] = static_cast<unsigned char>(value & 0xFFu);
    p[1] = static_cast<unsigned char>((value >> 8) & 0xFFu);
    p[2] = static_cast<unsigned char>((value >> 16) & 0xFFu);
    p[3] = static_cast<unsigned char>((value >> 24) & 0xFFu);
}

unsigned char *encode_contours(const std::vector<std::vector<cv::Point>> &contours,
                               size_t *out_len) {
    size_t total = 4;
    for (const auto &pts : contours) total += 4 + pts.size() * 8;
    auto *buf = static_cast<unsigned char *>(std::malloc(total));
    if (buf == nullptr) throw std::bad_alloc();
    put_u32le(buf, static_cast<unsigned int>(contours.size()));
    size_t off = 4;
    for (const auto &pts : contours) {
        put_u32le(buf + off, static_cast<unsigned int>(pts.size()));
        off += 4;
        for (const cv::Point &pt : pts) {
            put_u32le(buf + off, static_cast<unsigned int>(pt.x));
            put_u32le(buf + off + 4, static_cast<unsigned int>(pt.y));
            off += 8;
        }
    }
    if (out_len != nullptr) *out_len = total;
    return buf;
}

/* ---- shared Feature2D op bodies (per-class via macro below) ----------- */

template <typename PtrT>
cvk_mat_t *feature2d_detect_impl(const PtrT &ptr, const cvk_mat_t *image,
                                 const cvk_mat_t *mask) {
    const cv::Mat *img = require_const(image);
    if (img == nullptr) return nullptr;
    std::vector<cv::KeyPoint> keypoints;
    if (mask != nullptr) {
        const cv::Mat *mk = require_const(mask);
        if (mk == nullptr) return nullptr;
        ptr->detect(*img, keypoints, *mk);
    } else {
        ptr->detect(*img, keypoints);
    }
    cv::Mat out;
    keypoints_to_mat(keypoints, out);
    return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
}

template <typename PtrT>
int feature2d_compute_impl(const PtrT &ptr, const cvk_mat_t *image,
                           const cvk_mat_t *keypoints, cvk_mat_t **keypoints_out,
                           cvk_mat_t **descriptors_out) {
    if (keypoints_out != nullptr) *keypoints_out = nullptr;
    if (descriptors_out != nullptr) *descriptors_out = nullptr;
    const cv::Mat *img = require_const(image);
    const cv::Mat *kps = require_const(keypoints);
    if (img == nullptr || kps == nullptr) return 0;
    std::vector<cv::KeyPoint> kp;
    mat_to_keypoints(*kps, kp);
    cv::Mat descriptors;
    ptr->compute(*img, kp, descriptors);
    cv::Mat kp_mat;
    keypoints_to_mat(kp, kp_mat);
    if (keypoints_out != nullptr) {
        *keypoints_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(kp_mat));
    }
    if (descriptors_out != nullptr) {
        *descriptors_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(descriptors));
    }
    return 1;
}

template <typename PtrT>
cvk_mat_t *feature2d_detect_and_compute_impl(const PtrT &ptr,
                                             const cvk_mat_t *image,
                                             const cvk_mat_t *mask,
                                             cvk_mat_t **descriptors_out) {
    if (descriptors_out != nullptr) *descriptors_out = nullptr;
    const cv::Mat *img = require_const(image);
    if (img == nullptr) return nullptr;
    std::vector<cv::KeyPoint> kp;
    cv::Mat descriptors;
    if (mask != nullptr) {
        const cv::Mat *mk = require_const(mask);
        if (mk == nullptr) return nullptr;
        ptr->detectAndCompute(*img, *mk, kp, descriptors);
    } else {
        ptr->detectAndCompute(*img, cv::noArray(), kp, descriptors);
    }
    cv::Mat kp_mat;
    keypoints_to_mat(kp, kp_mat);
    if (descriptors_out != nullptr) {
        *descriptors_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(descriptors));
    }
    return reinterpret_cast<cvk_mat_t *>(new cv::Mat(kp_mat));
}

/* ---- SimpleBlobDetector::Params expansion ----------------------------- */

cv::SimpleBlobDetector::Params blob_params(float threshold_step, float min_threshold,
                                           float max_threshold, long long min_repeatability,
                                           float min_dist_between_blobs, int filter_by_color,
                                           int blob_color, int filter_by_area, float min_area,
                                           float max_area, int filter_by_circularity,
                                           float min_circularity, float max_circularity,
                                           int filter_by_inertia, float min_inertia_ratio,
                                           float max_inertia_ratio, int filter_by_convexity,
                                           float min_convexity, float max_convexity,
                                           int collect_contours) {
    cv::SimpleBlobDetector::Params params;
    params.thresholdStep = threshold_step;
    params.minThreshold = min_threshold;
    params.maxThreshold = max_threshold;
    params.minRepeatability = static_cast<size_t>(min_repeatability);
    params.minDistBetweenBlobs = min_dist_between_blobs;
    params.filterByColor = filter_by_color != 0;
    params.blobColor = static_cast<unsigned char>(blob_color);
    params.filterByArea = filter_by_area != 0;
    params.minArea = min_area;
    params.maxArea = max_area;
    params.filterByCircularity = filter_by_circularity != 0;
    params.minCircularity = min_circularity;
    params.maxCircularity = max_circularity;
    params.filterByInertia = filter_by_inertia != 0;
    params.minInertiaRatio = min_inertia_ratio;
    params.maxInertiaRatio = max_inertia_ratio;
    params.filterByConvexity = filter_by_convexity != 0;
    params.minConvexity = min_convexity;
    params.maxConvexity = max_convexity;
    params.collectContours = collect_contours != 0;
    return params;
}

void blob_params_to_out(const cv::SimpleBlobDetector::Params &params, double *out) {
    out[0] = params.thresholdStep;
    out[1] = params.minThreshold;
    out[2] = params.maxThreshold;
    out[3] = static_cast<double>(params.minRepeatability);
    out[4] = params.minDistBetweenBlobs;
    out[5] = params.filterByColor ? 1.0 : 0.0;
    out[6] = params.blobColor;
    out[7] = params.filterByArea ? 1.0 : 0.0;
    out[8] = params.minArea;
    out[9] = params.maxArea;
    out[10] = params.filterByCircularity ? 1.0 : 0.0;
    out[11] = params.minCircularity;
    out[12] = params.maxCircularity;
    out[13] = params.filterByInertia ? 1.0 : 0.0;
    out[14] = params.minInertiaRatio;
    out[15] = params.maxInertiaRatio;
    out[16] = params.filterByConvexity ? 1.0 : 0.0;
    out[17] = params.minConvexity;
    out[18] = params.maxConvexity;
    out[19] = params.collectContours ? 1.0 : 0.0;
}

} /* namespace */

/* =========================================================================
 * Shared per-class bodies
 *
 * CVK_FEATURES_ALG_FUNCS: the Algorithm quartet (clear/empty/save/
 * getDefaultName) for a handle struct storing cv::Ptr<cvT>.
 * CVK_FEATURES_FEATURE2D_FUNCS: the Feature2D surface delegating to the
 * impl templates above.
 * ========================================================================= */

#define CVK_FEATURES_ALG_FUNCS(T)                                             \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                    \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                 \
    }                                                                         \
    int cvk_##T##_empty(cvk_##T##_t *h) {                                     \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });     \
    }                                                                         \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {               \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        guarded([&]() -> int { p->ptr->save(filename); return 0; });          \
    }                                                                         \
    const char *cvk_##T##_get_default_name(cvk_##T##_t *h) {                  \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        return guarded([&]() -> const char * {                                \
            g_features_str = p->ptr->getDefaultName();                        \
            return g_features_str.c_str();                                    \
        });                                                                   \
    }

#define CVK_FEATURES_FEATURE2D_FUNCS(T)                                       \
    cvk_mat_t *cvk_##T##_detect(const cvk_##T##_t *h, const cvk_mat_t *image, \
                                const cvk_mat_t *mask) {                      \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        return guarded([&]() -> cvk_mat_t * {                                 \
            return feature2d_detect_impl(p->ptr, image, mask);                \
        });                                                                   \
    }                                                                         \
    int cvk_##T##_compute(const cvk_##T##_t *h, const cvk_mat_t *image,       \
                          const cvk_mat_t *keypoints,                         \
                          cvk_mat_t **keypoints_out,                          \
                          cvk_mat_t **descriptors_out) {                      \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        return guarded([&]() -> int {                                         \
            return feature2d_compute_impl(p->ptr, image, keypoints,           \
                                          keypoints_out, descriptors_out);    \
        });                                                                   \
    }                                                                         \
    cvk_mat_t *cvk_##T##_detect_and_compute(                                  \
        const cvk_##T##_t *h, const cvk_mat_t *image, const cvk_mat_t *mask,  \
        cvk_mat_t **descriptors_out) {                                        \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        return guarded([&]() -> cvk_mat_t * {                                 \
            return feature2d_detect_and_compute_impl(p->ptr, image, mask,     \
                                                     descriptors_out);        \
        });                                                                   \
    }                                                                         \
    int cvk_##T##_descriptor_size(const cvk_##T##_t *h) {                     \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        return guarded([&]() -> int { return p->ptr->descriptorSize(); });    \
    }                                                                         \
    int cvk_##T##_descriptor_type(const cvk_##T##_t *h) {                     \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        return guarded([&]() -> int { return p->ptr->descriptorType(); });    \
    }                                                                         \
    int cvk_##T##_default_norm(const cvk_##T##_t *h) {                        \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        return guarded([&]() -> int { return p->ptr->defaultNorm(); });       \
    }                                                                         \
    void cvk_##T##_write(const cvk_##T##_t *h, const char *filename) {        \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        guarded([&]() -> int { p->ptr->write(filename); return 0; });         \
    }                                                                         \
    void cvk_##T##_read(const cvk_##T##_t *h, const char *filename) {         \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                       \
        guarded([&]() -> int { p->ptr->read(filename); return 0; });          \
    }

/* =========================================================================
 * SIFT
 * ========================================================================= */

extern "C" {

cvk_sift_t *cvk_sift_create(int nfeatures, int n_octave_layers,
                            double contrast_threshold, double edge_threshold,
                            double sigma, int descriptor_type,
                            int enable_precise_upscale) {
    return guarded([&]() -> cvk_sift_t * {
        auto *h = new cvk_sift;
        if (descriptor_type < 0) {
            h->ptr = cv::SIFT::create(nfeatures, n_octave_layers, contrast_threshold,
                                      edge_threshold, sigma, enable_precise_upscale != 0);
        } else {
            h->ptr = cv::SIFT::create(nfeatures, n_octave_layers, contrast_threshold,
                                      edge_threshold, sigma, descriptor_type,
                                      enable_precise_upscale != 0);
        }
        return reinterpret_cast<cvk_sift_t *>(h);
    });
}

CVK_FEATURES_FEATURE2D_FUNCS(sift)
CVK_FEATURES_ALG_FUNCS(sift)

void cvk_sift_set_n_features(cvk_sift_t *h, int max_features) {
    auto *p = reinterpret_cast<cvk_sift *>(h);
    guarded([&]() -> int { p->ptr->setNFeatures(max_features); return 0; });
}

int cvk_sift_get_n_features(const cvk_sift_t *h) {
    auto *p = reinterpret_cast<const cvk_sift *>(h);
    return guarded([&]() -> int { return p->ptr->getNFeatures(); });
}

void cvk_sift_set_n_octave_layers(cvk_sift_t *h, int n_octave_layers) {
    auto *p = reinterpret_cast<cvk_sift *>(h);
    guarded([&]() -> int { p->ptr->setNOctaveLayers(n_octave_layers); return 0; });
}

int cvk_sift_get_n_octave_layers(const cvk_sift_t *h) {
    auto *p = reinterpret_cast<const cvk_sift *>(h);
    return guarded([&]() -> int { return p->ptr->getNOctaveLayers(); });
}

void cvk_sift_set_contrast_threshold(cvk_sift_t *h, double contrast_threshold) {
    auto *p = reinterpret_cast<cvk_sift *>(h);
    guarded([&]() -> int { p->ptr->setContrastThreshold(contrast_threshold); return 0; });
}

double cvk_sift_get_contrast_threshold(const cvk_sift_t *h) {
    auto *p = reinterpret_cast<const cvk_sift *>(h);
    return guarded([&]() -> double { return p->ptr->getContrastThreshold(); });
}

void cvk_sift_set_edge_threshold(cvk_sift_t *h, double edge_threshold) {
    auto *p = reinterpret_cast<cvk_sift *>(h);
    guarded([&]() -> int { p->ptr->setEdgeThreshold(edge_threshold); return 0; });
}

double cvk_sift_get_edge_threshold(const cvk_sift_t *h) {
    auto *p = reinterpret_cast<const cvk_sift *>(h);
    return guarded([&]() -> double { return p->ptr->getEdgeThreshold(); });
}

void cvk_sift_set_sigma(cvk_sift_t *h, double sigma) {
    auto *p = reinterpret_cast<cvk_sift *>(h);
    guarded([&]() -> int { p->ptr->setSigma(sigma); return 0; });
}

double cvk_sift_get_sigma(const cvk_sift_t *h) {
    auto *p = reinterpret_cast<const cvk_sift *>(h);
    return guarded([&]() -> double { return p->ptr->getSigma(); });
}

void cvk_sift_release(cvk_sift_t *h) {
    delete reinterpret_cast<cvk_sift *>(h);
}

/* =========================================================================
 * ORB
 * ========================================================================= */

cvk_orb_t *cvk_orb_create(int nfeatures, float scale_factor, int nlevels,
                          int edge_threshold, int first_level, int wta_k,
                          int score_type, int patch_size, int fast_threshold) {
    return guarded([&]() -> cvk_orb_t * {
        auto *h = new cvk_orb;
        h->ptr = cv::ORB::create(nfeatures, scale_factor, nlevels, edge_threshold,
                                 first_level, wta_k,
                                 static_cast<cv::ORB::ScoreType>(score_type),
                                 patch_size, fast_threshold);
        return reinterpret_cast<cvk_orb_t *>(h);
    });
}

CVK_FEATURES_FEATURE2D_FUNCS(orb)
CVK_FEATURES_ALG_FUNCS(orb)

void cvk_orb_set_max_features(cvk_orb_t *h, int max_features) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setMaxFeatures(max_features); return 0; });
}

int cvk_orb_get_max_features(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return p->ptr->getMaxFeatures(); });
}

void cvk_orb_set_scale_factor(cvk_orb_t *h, double scale_factor) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setScaleFactor(scale_factor); return 0; });
}

double cvk_orb_get_scale_factor(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> double { return p->ptr->getScaleFactor(); });
}

void cvk_orb_set_n_levels(cvk_orb_t *h, int nlevels) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setNLevels(nlevels); return 0; });
}

int cvk_orb_get_n_levels(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return p->ptr->getNLevels(); });
}

void cvk_orb_set_edge_threshold(cvk_orb_t *h, int edge_threshold) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setEdgeThreshold(edge_threshold); return 0; });
}

int cvk_orb_get_edge_threshold(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return p->ptr->getEdgeThreshold(); });
}

void cvk_orb_set_first_level(cvk_orb_t *h, int first_level) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setFirstLevel(first_level); return 0; });
}

int cvk_orb_get_first_level(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return p->ptr->getFirstLevel(); });
}

void cvk_orb_set_wta_k(cvk_orb_t *h, int wta_k) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setWTA_K(wta_k); return 0; });
}

int cvk_orb_get_wta_k(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return p->ptr->getWTA_K(); });
}

void cvk_orb_set_score_type(cvk_orb_t *h, int score_type) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int {
        p->ptr->setScoreType(static_cast<cv::ORB::ScoreType>(score_type));
        return 0;
    });
}

int cvk_orb_get_score_type(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return static_cast<int>(p->ptr->getScoreType()); });
}

void cvk_orb_set_patch_size(cvk_orb_t *h, int patch_size) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setPatchSize(patch_size); return 0; });
}

int cvk_orb_get_patch_size(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return p->ptr->getPatchSize(); });
}

void cvk_orb_set_fast_threshold(cvk_orb_t *h, int fast_threshold) {
    auto *p = reinterpret_cast<cvk_orb *>(h);
    guarded([&]() -> int { p->ptr->setFastThreshold(fast_threshold); return 0; });
}

int cvk_orb_get_fast_threshold(const cvk_orb_t *h) {
    auto *p = reinterpret_cast<const cvk_orb *>(h);
    return guarded([&]() -> int { return p->ptr->getFastThreshold(); });
}

void cvk_orb_release(cvk_orb_t *h) {
    delete reinterpret_cast<cvk_orb *>(h);
}

/* =========================================================================
 * MSER
 * ========================================================================= */

cvk_mser_t *cvk_mser_create(int delta, int min_area, int max_area,
                            double max_variation, double min_diversity,
                            int max_evolution, double area_threshold,
                            double min_margin, int edge_blur_size) {
    return guarded([&]() -> cvk_mser_t * {
        auto *h = new cvk_mser;
        h->ptr = cv::MSER::create(delta, min_area, max_area, max_variation,
                                  min_diversity, max_evolution, area_threshold,
                                  min_margin, edge_blur_size);
        return reinterpret_cast<cvk_mser_t *>(h);
    });
}

CVK_FEATURES_FEATURE2D_FUNCS(mser)
CVK_FEATURES_ALG_FUNCS(mser)

unsigned char *cvk_mser_detect_regions(const cvk_mser_t *h,
                                       const cvk_mat_t *image,
                                       cvk_mat_t **bboxes, size_t *out_len) {
    if (bboxes != nullptr) *bboxes = nullptr;
    if (out_len != nullptr) *out_len = 0;
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> unsigned char * {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return nullptr;
        std::vector<std::vector<cv::Point>> msers;
        std::vector<cv::Rect> boxes;
        p->ptr->detectRegions(*img, msers, boxes);
        cv::Mat bmat(static_cast<int>(boxes.size()), 1, CV_32SC4);
        for (size_t i = 0; i < boxes.size(); ++i) {
            int *row = bmat.ptr<int>(static_cast<int>(i));
            row[0] = boxes[i].x;
            row[1] = boxes[i].y;
            row[2] = boxes[i].width;
            row[3] = boxes[i].height;
        }
        if (bboxes != nullptr) {
            *bboxes = reinterpret_cast<cvk_mat_t *>(new cv::Mat(bmat));
        }
        return encode_contours(msers, out_len);
    });
}

void cvk_mser_set_delta(cvk_mser_t *h, int delta) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setDelta(delta); return 0; });
}

int cvk_mser_get_delta(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> int { return p->ptr->getDelta(); });
}

void cvk_mser_set_min_area(cvk_mser_t *h, int min_area) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setMinArea(min_area); return 0; });
}

int cvk_mser_get_min_area(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> int { return p->ptr->getMinArea(); });
}

void cvk_mser_set_max_area(cvk_mser_t *h, int max_area) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setMaxArea(max_area); return 0; });
}

int cvk_mser_get_max_area(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> int { return p->ptr->getMaxArea(); });
}

void cvk_mser_set_max_variation(cvk_mser_t *h, double max_variation) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setMaxVariation(max_variation); return 0; });
}

double cvk_mser_get_max_variation(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> double { return p->ptr->getMaxVariation(); });
}

void cvk_mser_set_min_diversity(cvk_mser_t *h, double min_diversity) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setMinDiversity(min_diversity); return 0; });
}

double cvk_mser_get_min_diversity(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> double { return p->ptr->getMinDiversity(); });
}

void cvk_mser_set_max_evolution(cvk_mser_t *h, int max_evolution) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setMaxEvolution(max_evolution); return 0; });
}

int cvk_mser_get_max_evolution(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> int { return p->ptr->getMaxEvolution(); });
}

void cvk_mser_set_area_threshold(cvk_mser_t *h, double area_threshold) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setAreaThreshold(area_threshold); return 0; });
}

double cvk_mser_get_area_threshold(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> double { return p->ptr->getAreaThreshold(); });
}

void cvk_mser_set_min_margin(cvk_mser_t *h, double min_margin) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setMinMargin(min_margin); return 0; });
}

double cvk_mser_get_min_margin(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> double { return p->ptr->getMinMargin(); });
}

void cvk_mser_set_edge_blur_size(cvk_mser_t *h, int edge_blur_size) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setEdgeBlurSize(edge_blur_size); return 0; });
}

int cvk_mser_get_edge_blur_size(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> int { return p->ptr->getEdgeBlurSize(); });
}

void cvk_mser_set_pass2_only(cvk_mser_t *h, int f) {
    auto *p = reinterpret_cast<cvk_mser *>(h);
    guarded([&]() -> int { p->ptr->setPass2Only(f != 0); return 0; });
}

int cvk_mser_get_pass2_only(const cvk_mser_t *h) {
    auto *p = reinterpret_cast<const cvk_mser *>(h);
    return guarded([&]() -> int { return p->ptr->getPass2Only() ? 1 : 0; });
}

void cvk_mser_release(cvk_mser_t *h) {
    delete reinterpret_cast<cvk_mser *>(h);
}

/* =========================================================================
 * FastFeatureDetector
 * ========================================================================= */

cvk_fast_feature_detector_t *cvk_fast_feature_detector_create(
    int threshold, int nonmax_suppression, int type) {
    return guarded([&]() -> cvk_fast_feature_detector_t * {
        auto *h = new cvk_fast_feature_detector;
        h->ptr = cv::FastFeatureDetector::create(
            threshold, nonmax_suppression != 0,
            static_cast<cv::FastFeatureDetector::DetectorType>(type));
        return reinterpret_cast<cvk_fast_feature_detector_t *>(h);
    });
}

CVK_FEATURES_FEATURE2D_FUNCS(fast_feature_detector)
CVK_FEATURES_ALG_FUNCS(fast_feature_detector)

void cvk_fast_feature_detector_set_threshold(cvk_fast_feature_detector_t *h,
                                             int threshold) {
    auto *p = reinterpret_cast<cvk_fast_feature_detector *>(h);
    guarded([&]() -> int { p->ptr->setThreshold(threshold); return 0; });
}

int cvk_fast_feature_detector_get_threshold(const cvk_fast_feature_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_fast_feature_detector *>(h);
    return guarded([&]() -> int { return p->ptr->getThreshold(); });
}

void cvk_fast_feature_detector_set_nonmax_suppression(cvk_fast_feature_detector_t *h,
                                                      int f) {
    auto *p = reinterpret_cast<cvk_fast_feature_detector *>(h);
    guarded([&]() -> int { p->ptr->setNonmaxSuppression(f != 0); return 0; });
}

int cvk_fast_feature_detector_get_nonmax_suppression(
    const cvk_fast_feature_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_fast_feature_detector *>(h);
    return guarded([&]() -> int { return p->ptr->getNonmaxSuppression() ? 1 : 0; });
}

void cvk_fast_feature_detector_set_type(cvk_fast_feature_detector_t *h, int type) {
    auto *p = reinterpret_cast<cvk_fast_feature_detector *>(h);
    guarded([&]() -> int {
        p->ptr->setType(static_cast<cv::FastFeatureDetector::DetectorType>(type));
        return 0;
    });
}

int cvk_fast_feature_detector_get_type(const cvk_fast_feature_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_fast_feature_detector *>(h);
    return guarded([&]() -> int {
        return static_cast<int>(p->ptr->getType());
    });
}

void cvk_fast_feature_detector_release(cvk_fast_feature_detector_t *h) {
    delete reinterpret_cast<cvk_fast_feature_detector *>(h);
}

/* =========================================================================
 * GFTTDetector
 * ========================================================================= */

cvk_gftt_detector_t *cvk_gftt_detector_create(
    int max_corners, double quality_level, double min_distance, int block_size,
    int gradient_size, int use_harris_detector, double k) {
    return guarded([&]() -> cvk_gftt_detector_t * {
        auto *h = new cvk_gftt_detector;
        if (gradient_size < 0) {
            h->ptr = cv::GFTTDetector::create(max_corners, quality_level, min_distance,
                                              block_size, use_harris_detector != 0, k);
        } else {
            h->ptr = cv::GFTTDetector::create(max_corners, quality_level, min_distance,
                                              block_size, gradient_size,
                                              use_harris_detector != 0, k);
        }
        return reinterpret_cast<cvk_gftt_detector_t *>(h);
    });
}

CVK_FEATURES_FEATURE2D_FUNCS(gftt_detector)
CVK_FEATURES_ALG_FUNCS(gftt_detector)

void cvk_gftt_detector_set_max_features(cvk_gftt_detector_t *h, int max_features) {
    auto *p = reinterpret_cast<cvk_gftt_detector *>(h);
    guarded([&]() -> int { p->ptr->setMaxFeatures(max_features); return 0; });
}

int cvk_gftt_detector_get_max_features(const cvk_gftt_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_gftt_detector *>(h);
    return guarded([&]() -> int { return p->ptr->getMaxFeatures(); });
}

void cvk_gftt_detector_set_quality_level(cvk_gftt_detector_t *h, double qlevel) {
    auto *p = reinterpret_cast<cvk_gftt_detector *>(h);
    guarded([&]() -> int { p->ptr->setQualityLevel(qlevel); return 0; });
}

double cvk_gftt_detector_get_quality_level(const cvk_gftt_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_gftt_detector *>(h);
    return guarded([&]() -> double { return p->ptr->getQualityLevel(); });
}

void cvk_gftt_detector_set_min_distance(cvk_gftt_detector_t *h, double min_distance) {
    auto *p = reinterpret_cast<cvk_gftt_detector *>(h);
    guarded([&]() -> int { p->ptr->setMinDistance(min_distance); return 0; });
}

double cvk_gftt_detector_get_min_distance(const cvk_gftt_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_gftt_detector *>(h);
    return guarded([&]() -> double { return p->ptr->getMinDistance(); });
}

void cvk_gftt_detector_set_block_size(cvk_gftt_detector_t *h, int block_size) {
    auto *p = reinterpret_cast<cvk_gftt_detector *>(h);
    guarded([&]() -> int { p->ptr->setBlockSize(block_size); return 0; });
}

int cvk_gftt_detector_get_block_size(const cvk_gftt_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_gftt_detector *>(h);
    return guarded([&]() -> int { return p->ptr->getBlockSize(); });
}

void cvk_gftt_detector_set_gradient_size(cvk_gftt_detector_t *h, int gradient_size) {
    auto *p = reinterpret_cast<cvk_gftt_detector *>(h);
    guarded([&]() -> int { p->ptr->setGradientSize(gradient_size); return 0; });
}

int cvk_gftt_detector_get_gradient_size(const cvk_gftt_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_gftt_detector *>(h);
    return guarded([&]() -> int { return p->ptr->getGradientSize(); });
}

void cvk_gftt_detector_set_harris_detector(cvk_gftt_detector_t *h, int val) {
    auto *p = reinterpret_cast<cvk_gftt_detector *>(h);
    guarded([&]() -> int { p->ptr->setHarrisDetector(val != 0); return 0; });
}

int cvk_gftt_detector_get_harris_detector(const cvk_gftt_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_gftt_detector *>(h);
    return guarded([&]() -> int { return p->ptr->getHarrisDetector() ? 1 : 0; });
}

void cvk_gftt_detector_set_k(cvk_gftt_detector_t *h, double k) {
    auto *p = reinterpret_cast<cvk_gftt_detector *>(h);
    guarded([&]() -> int { p->ptr->setK(k); return 0; });
}

double cvk_gftt_detector_get_k(const cvk_gftt_detector_t *h) {
    auto *p = reinterpret_cast<const cvk_gftt_detector *>(h);
    return guarded([&]() -> double { return p->ptr->getK(); });
}

void cvk_gftt_detector_release(cvk_gftt_detector_t *h) {
    delete reinterpret_cast<cvk_gftt_detector *>(h);
}

/* =========================================================================
 * SimpleBlobDetector
 * ========================================================================= */

cvk_simple_blob_detector_t *cvk_simple_blob_detector_create(
    float threshold_step, float min_threshold, float max_threshold,
    long long min_repeatability, float min_dist_between_blobs,
    int filter_by_color, int blob_color,
    int filter_by_area, float min_area, float max_area,
    int filter_by_circularity, float min_circularity, float max_circularity,
    int filter_by_inertia, float min_inertia_ratio, float max_inertia_ratio,
    int filter_by_convexity, float min_convexity, float max_convexity,
    int collect_contours) {
    return guarded([&]() -> cvk_simple_blob_detector_t * {
        auto *h = new cvk_simple_blob_detector;
        h->ptr = cv::SimpleBlobDetector::create(blob_params(
            threshold_step, min_threshold, max_threshold, min_repeatability,
            min_dist_between_blobs, filter_by_color, blob_color, filter_by_area,
            min_area, max_area, filter_by_circularity, min_circularity,
            max_circularity, filter_by_inertia, min_inertia_ratio,
            max_inertia_ratio, filter_by_convexity, min_convexity, max_convexity,
            collect_contours));
        return reinterpret_cast<cvk_simple_blob_detector_t *>(h);
    });
}

CVK_FEATURES_FEATURE2D_FUNCS(simple_blob_detector)
CVK_FEATURES_ALG_FUNCS(simple_blob_detector)

int cvk_simple_blob_detector_set_params(cvk_simple_blob_detector_t *h,
                                        float threshold_step, float min_threshold,
                                        float max_threshold, long long min_repeatability,
                                        float min_dist_between_blobs,
                                        int filter_by_color, int blob_color,
                                        int filter_by_area, float min_area,
                                        float max_area, int filter_by_circularity,
                                        float min_circularity, float max_circularity,
                                        int filter_by_inertia, float min_inertia_ratio,
                                        float max_inertia_ratio, int filter_by_convexity,
                                        float min_convexity, float max_convexity,
                                        int collect_contours) {
    auto *p = reinterpret_cast<cvk_simple_blob_detector *>(h);
    return guarded([&]() -> int {
        p->ptr->setParams(blob_params(
            threshold_step, min_threshold, max_threshold, min_repeatability,
            min_dist_between_blobs, filter_by_color, blob_color, filter_by_area,
            min_area, max_area, filter_by_circularity, min_circularity,
            max_circularity, filter_by_inertia, min_inertia_ratio,
            max_inertia_ratio, filter_by_convexity, min_convexity, max_convexity,
            collect_contours));
        return 1;
    });
}

int cvk_simple_blob_detector_get_params(const cvk_simple_blob_detector_t *h,
                                        double *out20) {
    if (out20 == nullptr) return 0;
    auto *p = reinterpret_cast<const cvk_simple_blob_detector *>(h);
    return guarded([&]() -> int {
        blob_params_to_out(p->ptr->getParams(), out20);
        return 1;
    });
}

unsigned char *cvk_simple_blob_detector_get_blob_contours(
    const cvk_simple_blob_detector_t *h, size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    auto *p = reinterpret_cast<const cvk_simple_blob_detector *>(h);
    return guarded([&]() -> unsigned char * {
        const std::vector<std::vector<cv::Point>> &contours = p->ptr->getBlobContours();
        return encode_contours(contours, out_len);
    });
}

void cvk_simple_blob_detector_release(cvk_simple_blob_detector_t *h) {
    delete reinterpret_cast<cvk_simple_blob_detector *>(h);
}

/* =========================================================================
 * AffineFeature
 * ========================================================================= */

cvk_affine_t *cvk_affine_create(const cvk_feature2d_t *backend, int max_tilt,
                                int min_tilt, float tilt_step,
                                float rotate_step_base) {
    return guarded([&]() -> cvk_affine_t * {
        auto *h = new cvk_affine;
        const auto *view = reinterpret_cast<const cvk_feature2d *>(backend);
        h->ptr = cv::AffineFeature::create(view->ptr, max_tilt, min_tilt, tilt_step,
                                           rotate_step_base);
        return reinterpret_cast<cvk_affine_t *>(h);
    });
}

CVK_FEATURES_FEATURE2D_FUNCS(affine)
CVK_FEATURES_ALG_FUNCS(affine)

int cvk_affine_set_view_params(const cvk_affine_t *h, const cvk_mat_t *tilts,
                               const cvk_mat_t *rolls) {
    auto *p = reinterpret_cast<const cvk_affine *>(h);
    return guarded([&]() -> int {
        const cv::Mat *t = require_const(tilts);
        const cv::Mat *r = require_const(rolls);
        if (t == nullptr || r == nullptr) return 0;
        p->ptr->setViewParams(floats_of(*t), floats_of(*r));
        return 1;
    });
}

int cvk_affine_get_view_params(const cvk_affine_t *h, cvk_mat_t **tilts_out,
                               cvk_mat_t **rolls_out) {
    if (tilts_out != nullptr) *tilts_out = nullptr;
    if (rolls_out != nullptr) *rolls_out = nullptr;
    auto *p = reinterpret_cast<const cvk_affine *>(h);
    return guarded([&]() -> int {
        std::vector<float> tilts;
        std::vector<float> rolls;
        p->ptr->getViewParams(tilts, rolls);
        if (tilts_out != nullptr) {
            *tilts_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mat_of_floats(tilts)));
        }
        if (rolls_out != nullptr) {
            *rolls_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mat_of_floats(rolls)));
        }
        return 1;
    });
}

void cvk_affine_release(cvk_affine_t *h) {
    delete reinterpret_cast<cvk_affine *>(h);
}

} /* extern "C" */

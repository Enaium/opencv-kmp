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
 * cvk_ C ABI implementation for the OpenCV "video" module
 * (opencv_kmp_video.h): background subtraction, optical flow, ECC image
 * alignment and the Kalman filter.
 *
 * Every exported function is noexcept: bodies run inside `guarded`, failures
 * are reported through the thread-local error string and functions return
 * NULL / 0 as documented in the header.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_video.h"

#include <opencv2/video.hpp>

#include <cstdint>
#include <memory>
#include <string>

struct cvk_background_subtractor_knn { cv::Ptr<cv::BackgroundSubtractorKNN> ptr; };
struct cvk_background_subtractor_mog2 { cv::Ptr<cv::BackgroundSubtractorMOG2> ptr; };
struct cvk_farneback_optical_flow { cv::Ptr<cv::FarnebackOpticalFlow> ptr; };
struct cvk_dis_optical_flow { cv::Ptr<cv::DISOpticalFlow> ptr; };
struct cvk_sparse_pyr_lk_optical_flow { cv::Ptr<cv::SparsePyrLKOpticalFlow> ptr; };
struct cvk_variational_refinement { cv::Ptr<cv::VariationalRefinement> ptr; };
struct cvk_kalman_filter { cv::KalmanFilter kf; };

namespace {

thread_local std::string g_video_str;
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

cv::TermCriteria term_criteria(int type, int max_count, double epsilon) {
    return cv::TermCriteria(type, max_count, epsilon);
}

/* `name` is the snake_case C symbol suffix, `member` the camelCase C++
 * method name; e.g. CVK_GET_SET_INT(T, n_samples, NSamples) emits
 * cvk_<T>_get_n_samples() calling getNSamples(). */

#define CVK_GET_SET_INT(T, name, member)                                          \
    int cvk_##T##_get_##name(const cvk_##T##_t *h) {                              \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                           \
        return guarded([&]() -> int { return p->ptr->get##member(); });           \
    }                                                                             \
    void cvk_##T##_set_##name(cvk_##T##_t *h, int value) {                        \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        guarded([&]() -> int { p->ptr->set##member(value); return 0; });          \
    }

#define CVK_GET_SET_DOUBLE(T, name, member)                                       \
    double cvk_##T##_get_##name(const cvk_##T##_t *h) {                           \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                           \
        return guarded([&]() -> double { return p->ptr->get##member(); });        \
    }                                                                             \
    void cvk_##T##_set_##name(cvk_##T##_t *h, double value) {                     \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        guarded([&]() -> int { p->ptr->set##member(value); return 0; });          \
    }

#define CVK_GET_SET_FLOAT(T, name, member)                                        \
    float cvk_##T##_get_##name(const cvk_##T##_t *h) {                            \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                           \
        return guarded([&]() -> float { return p->ptr->get##member(); });         \
    }                                                                             \
    void cvk_##T##_set_##name(cvk_##T##_t *h, float value) {                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        guarded([&]() -> int { p->ptr->set##member(value); return 0; });          \
    }

#define CVK_GET_SET_BOOL(T, name, member)                                         \
    int cvk_##T##_get_##name(const cvk_##T##_t *h) {                              \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                           \
        return guarded([&]() -> int { return p->ptr->get##member() ? 1 : 0; });   \
    }                                                                             \
    void cvk_##T##_set_##name(cvk_##T##_t *h, int value) {                        \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        guarded([&]() -> int { p->ptr->set##member(value != 0); return 0; });     \
    }

/** Generates the four Algorithm interface functions for a Ptr handle. */
#define CVK_ALG_FUNCS(T)                                                          \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                        \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                     \
    }                                                                             \
    int cvk_##T##_empty(cvk_##T##_t *h) {                                         \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });         \
    }                                                                             \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {                   \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        guarded([&]() -> int { p->ptr->save(filename); return 0; });              \
    }                                                                             \
    const char *cvk_##T##_get_default_name(cvk_##T##_t *h) {                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                 \
        return guarded([&]() -> const char * {                                    \
            g_video_str = p->ptr->getDefaultName();                               \
            return g_video_str.c_str();                                           \
        });                                                                       \
    }

}  // namespace

extern "C" {

/* =====================================================================
 * video statics
 * ===================================================================== */

void cvk_calc_optical_flow_farneback(const cvk_mat_t *prev, const cvk_mat_t *next,
                                     cvk_mat_t *flow, double pyr_scale, int levels,
                                     int winsize, int iterations, int poly_n,
                                     double poly_sigma, int flags) {
    guarded([&]() -> int {
        cv::calcOpticalFlowFarneback(*require_const(prev), *require_const(next),
                                     *require(flow), pyr_scale, levels, winsize,
                                     iterations, poly_n, poly_sigma, flags);
        return 0;
    });
}

void cvk_calc_optical_flow_pyr_lk(const cvk_mat_t *prev_img, const cvk_mat_t *next_img,
                                  const cvk_mat_t *prev_pts, cvk_mat_t *next_pts,
                                  cvk_mat_t *status, cvk_mat_t *err,
                                  int win_w, int win_h, int max_level,
                                  int tc_type, int tc_max_count, double tc_epsilon,
                                  int flags, double min_eig_threshold) {
    guarded([&]() -> int {
        cv::calcOpticalFlowPyrLK(
            *require_const(prev_img), *require_const(next_img), *require_const(prev_pts),
            *require(next_pts), *require(status), *require(err),
            cv::Size(win_w, win_h), max_level,
            term_criteria(tc_type, tc_max_count, tc_epsilon), flags, min_eig_threshold);
        return 0;
    });
}

double cvk_compute_ecc(const cvk_mat_t *template_image, const cvk_mat_t *input_image,
                       const cvk_mat_t *input_mask) {
    return guarded([&]() -> double {
        return cv::computeECC(*require_const(template_image), *require_const(input_image),
                              input_mask ? *require_const(input_mask) : cv::noArray());
    });
}

double cvk_find_transform_ecc(const cvk_mat_t *template_image, const cvk_mat_t *input_image,
                              cvk_mat_t *warp_matrix, int motion_type,
                              int tc_type, int tc_max_count, double tc_epsilon,
                              const cvk_mat_t *input_mask, int gauss_filt_size) {
    return guarded([&]() -> double {
        return cv::findTransformECC(
            *require_const(template_image), *require_const(input_image), *require(warp_matrix),
            motion_type, term_criteria(tc_type, tc_max_count, tc_epsilon),
            input_mask ? *require_const(input_mask) : cv::noArray(), gauss_filt_size);
    });
}

double cvk_find_transform_ecc_with_mask(const cvk_mat_t *template_image,
                                        const cvk_mat_t *input_image,
                                        const cvk_mat_t *template_mask,
                                        const cvk_mat_t *input_mask,
                                        cvk_mat_t *warp_matrix, int motion_type,
                                        int tc_type, int tc_max_count, double tc_epsilon,
                                        int gauss_filt_size) {
    return guarded([&]() -> double {
        return cv::findTransformECCWithMask(
            *require_const(template_image), *require_const(input_image),
            *require_const(template_mask), *require_const(input_mask), *require(warp_matrix),
            motion_type, term_criteria(tc_type, tc_max_count, tc_epsilon), gauss_filt_size);
    });
}

double cvk_find_transform_ecc_multi_scale(const cvk_mat_t *reference, const cvk_mat_t *sample,
                                          cvk_mat_t *warp_matrix, int motion_type,
                                          int tc_type, int tc_max_count, double tc_epsilon,
                                          const cvk_mat_t *iters_per_level,
                                          int gauss_filt_size, int nlevels, int interpolation,
                                          const cvk_mat_t *reference_mask,
                                          const cvk_mat_t *sample_mask) {
    return guarded([&]() -> double {
        cv::ECCParameters params;
        params.motionType = motion_type;
        params.criteria = term_criteria(tc_type, tc_max_count, tc_epsilon);
        if (iters_per_level) {
            const cv::Mat &iters = *require_const(iters_per_level);
            if (!iters.empty()) {
                iters.forEach<int>([&](const int &v, const int *) {
                    params.itersPerLevel.push_back(v);
                });
            }
        }
        params.gaussFiltSize = gauss_filt_size;
        params.nlevels = nlevels;
        params.interpolation = interpolation;
        return cv::findTransformECCMultiScale(
            *require_const(reference), *require_const(sample), *require(warp_matrix), params,
            reference_mask ? *require_const(reference_mask) : cv::noArray(),
            sample_mask ? *require_const(sample_mask) : cv::noArray());
    });
}

/* =====================================================================
 * BackgroundSubtractorKNN
 * ===================================================================== */

cvk_background_subtractor_knn_t *
cvk_background_subtractor_knn_create(int history, double dist2_threshold, int detect_shadows) {
    return guarded([&]() -> cvk_background_subtractor_knn_t * {
        auto *h = new cvk_background_subtractor_knn;
        h->ptr = cv::createBackgroundSubtractorKNN(history, dist2_threshold, detect_shadows != 0);
        return reinterpret_cast<cvk_background_subtractor_knn_t *>(h);
    });
}

void cvk_background_subtractor_knn_apply(cvk_background_subtractor_knn_t *h,
                                         const cvk_mat_t *image, cvk_mat_t *fgmask,
                                         double learning_rate) {
    auto *p = reinterpret_cast<cvk_background_subtractor_knn *>(h);
    guarded([&]() -> int {
        p->ptr->apply(*require_const(image), *require(fgmask), learning_rate);
        return 0;
    });
}

void cvk_background_subtractor_knn_apply_mask(cvk_background_subtractor_knn_t *h,
                                              const cvk_mat_t *image,
                                              const cvk_mat_t *known_foreground_mask,
                                              cvk_mat_t *fgmask, double learning_rate) {
    auto *p = reinterpret_cast<cvk_background_subtractor_knn *>(h);
    guarded([&]() -> int {
        p->ptr->apply(*require_const(image), *require_const(known_foreground_mask),
                      *require(fgmask), learning_rate);
        return 0;
    });
}

cvk_mat_t *cvk_background_subtractor_knn_get_background_image(cvk_background_subtractor_knn_t *h) {
    auto *p = reinterpret_cast<cvk_background_subtractor_knn *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat background;
        p->ptr->getBackgroundImage(background);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(background)));
    });
}

CVK_GET_SET_INT(background_subtractor_knn, history, History)
CVK_GET_SET_INT(background_subtractor_knn, n_samples, NSamples)
CVK_GET_SET_DOUBLE(background_subtractor_knn, dist2_threshold, Dist2Threshold)
CVK_GET_SET_INT(background_subtractor_knn, knn_samples, kNNSamples)
CVK_GET_SET_BOOL(background_subtractor_knn, detect_shadows, DetectShadows)
CVK_GET_SET_INT(background_subtractor_knn, shadow_value, ShadowValue)
CVK_GET_SET_DOUBLE(background_subtractor_knn, shadow_threshold, ShadowThreshold)

/* =====================================================================
 * BackgroundSubtractorMOG2
 * ===================================================================== */

cvk_background_subtractor_mog2_t *
cvk_background_subtractor_mog2_create(int history, double var_threshold, int detect_shadows) {
    return guarded([&]() -> cvk_background_subtractor_mog2_t * {
        auto *h = new cvk_background_subtractor_mog2;
        h->ptr = cv::createBackgroundSubtractorMOG2(history, var_threshold, detect_shadows != 0);
        return reinterpret_cast<cvk_background_subtractor_mog2_t *>(h);
    });
}

void cvk_background_subtractor_mog2_apply(cvk_background_subtractor_mog2_t *h,
                                          const cvk_mat_t *image, cvk_mat_t *fgmask,
                                          double learning_rate) {
    auto *p = reinterpret_cast<cvk_background_subtractor_mog2 *>(h);
    guarded([&]() -> int {
        p->ptr->apply(*require_const(image), *require(fgmask), learning_rate);
        return 0;
    });
}

void cvk_background_subtractor_mog2_apply_mask(cvk_background_subtractor_mog2_t *h,
                                               const cvk_mat_t *image,
                                               const cvk_mat_t *known_foreground_mask,
                                               cvk_mat_t *fgmask, double learning_rate) {
    auto *p = reinterpret_cast<cvk_background_subtractor_mog2 *>(h);
    guarded([&]() -> int {
        p->ptr->apply(*require_const(image), *require_const(known_foreground_mask),
                      *require(fgmask), learning_rate);
        return 0;
    });
}

cvk_mat_t *cvk_background_subtractor_mog2_get_background_image(cvk_background_subtractor_mog2_t *h) {
    auto *p = reinterpret_cast<cvk_background_subtractor_mog2 *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat background;
        p->ptr->getBackgroundImage(background);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(background)));
    });
}

CVK_GET_SET_INT(background_subtractor_mog2, history, History)
CVK_GET_SET_INT(background_subtractor_mog2, n_mixtures, NMixtures)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, background_ratio, BackgroundRatio)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, var_threshold, VarThreshold)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, var_threshold_gen, VarThresholdGen)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, var_init, VarInit)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, var_min, VarMin)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, var_max, VarMax)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, complexity_reduction_threshold,
                   ComplexityReductionThreshold)
CVK_GET_SET_BOOL(background_subtractor_mog2, detect_shadows, DetectShadows)
CVK_GET_SET_INT(background_subtractor_mog2, shadow_value, ShadowValue)
CVK_GET_SET_DOUBLE(background_subtractor_mog2, shadow_threshold, ShadowThreshold)

/* =====================================================================
 * FarnebackOpticalFlow
 * ===================================================================== */

cvk_farneback_optical_flow_t *cvk_farneback_optical_flow_create(
    int num_levels, double pyr_scale, int fast_pyramids, int win_size, int num_iters,
    int poly_n, double poly_sigma, int flags) {
    return guarded([&]() -> cvk_farneback_optical_flow_t * {
        auto *h = new cvk_farneback_optical_flow;
        h->ptr = cv::FarnebackOpticalFlow::create(num_levels, pyr_scale, fast_pyramids != 0,
                                                  win_size, num_iters, poly_n, poly_sigma, flags);
        return reinterpret_cast<cvk_farneback_optical_flow_t *>(h);
    });
}

void cvk_farneback_optical_flow_calc(cvk_farneback_optical_flow_t *h,
                                     const cvk_mat_t *i0, const cvk_mat_t *i1,
                                     cvk_mat_t *flow) {
    auto *p = reinterpret_cast<cvk_farneback_optical_flow *>(h);
    guarded([&]() -> int {
        p->ptr->calc(*require_const(i0), *require_const(i1), *require(flow));
        return 0;
    });
}

void cvk_farneback_optical_flow_collect_garbage(cvk_farneback_optical_flow_t *h) {
    auto *p = reinterpret_cast<cvk_farneback_optical_flow *>(h);
    guarded([&]() -> int { p->ptr->collectGarbage(); return 0; });
}

CVK_GET_SET_INT(farneback_optical_flow, num_levels, NumLevels)
CVK_GET_SET_DOUBLE(farneback_optical_flow, pyr_scale, PyrScale)
CVK_GET_SET_BOOL(farneback_optical_flow, fast_pyramids, FastPyramids)
CVK_GET_SET_INT(farneback_optical_flow, win_size, WinSize)
CVK_GET_SET_INT(farneback_optical_flow, num_iters, NumIters)
CVK_GET_SET_INT(farneback_optical_flow, poly_n, PolyN)
CVK_GET_SET_DOUBLE(farneback_optical_flow, poly_sigma, PolySigma)
CVK_GET_SET_INT(farneback_optical_flow, flags, Flags)

/* =====================================================================
 * DISOpticalFlow
 * ===================================================================== */

cvk_dis_optical_flow_t *cvk_dis_optical_flow_create(int preset) {
    return guarded([&]() -> cvk_dis_optical_flow_t * {
        auto *h = new cvk_dis_optical_flow;
        h->ptr = cv::DISOpticalFlow::create(preset);
        return reinterpret_cast<cvk_dis_optical_flow_t *>(h);
    });
}

void cvk_dis_optical_flow_calc(cvk_dis_optical_flow_t *h,
                               const cvk_mat_t *i0, const cvk_mat_t *i1, cvk_mat_t *flow) {
    auto *p = reinterpret_cast<cvk_dis_optical_flow *>(h);
    guarded([&]() -> int {
        p->ptr->calc(*require_const(i0), *require_const(i1), *require(flow));
        return 0;
    });
}

void cvk_dis_optical_flow_collect_garbage(cvk_dis_optical_flow_t *h) {
    auto *p = reinterpret_cast<cvk_dis_optical_flow *>(h);
    guarded([&]() -> int { p->ptr->collectGarbage(); return 0; });
}

CVK_GET_SET_INT(dis_optical_flow, finest_scale, FinestScale)
CVK_GET_SET_INT(dis_optical_flow, coarsest_scale, CoarsestScale)
CVK_GET_SET_INT(dis_optical_flow, patch_size, PatchSize)
CVK_GET_SET_INT(dis_optical_flow, patch_stride, PatchStride)
CVK_GET_SET_INT(dis_optical_flow, gradient_descent_iterations, GradientDescentIterations)
CVK_GET_SET_INT(dis_optical_flow, variational_refinement_iterations,
                VariationalRefinementIterations)
CVK_GET_SET_FLOAT(dis_optical_flow, variational_refinement_alpha,
                  VariationalRefinementAlpha)
CVK_GET_SET_FLOAT(dis_optical_flow, variational_refinement_delta,
                  VariationalRefinementDelta)
CVK_GET_SET_FLOAT(dis_optical_flow, variational_refinement_gamma,
                  VariationalRefinementGamma)
CVK_GET_SET_FLOAT(dis_optical_flow, variational_refinement_epsilon,
                  VariationalRefinementEpsilon)
CVK_GET_SET_BOOL(dis_optical_flow, use_mean_normalization, UseMeanNormalization)
CVK_GET_SET_BOOL(dis_optical_flow, use_spatial_propagation, UseSpatialPropagation)

/* =====================================================================
 * SparsePyrLKOpticalFlow
 * ===================================================================== */

cvk_sparse_pyr_lk_optical_flow_t *cvk_sparse_pyr_lk_optical_flow_create(
    int win_w, int win_h, int max_level, int tc_type, int tc_max_count, double tc_epsilon,
    int flags, double min_eig_threshold) {
    return guarded([&]() -> cvk_sparse_pyr_lk_optical_flow_t * {
        auto *h = new cvk_sparse_pyr_lk_optical_flow;
        h->ptr = cv::SparsePyrLKOpticalFlow::create(
            cv::Size(win_w, win_h), max_level,
            term_criteria(tc_type, tc_max_count, tc_epsilon), flags, min_eig_threshold);
        return reinterpret_cast<cvk_sparse_pyr_lk_optical_flow_t *>(h);
    });
}

void cvk_sparse_pyr_lk_optical_flow_calc(cvk_sparse_pyr_lk_optical_flow_t *h,
                                         const cvk_mat_t *prev_img, const cvk_mat_t *next_img,
                                         const cvk_mat_t *prev_pts, cvk_mat_t *next_pts,
                                         cvk_mat_t *status, cvk_mat_t *err) {
    auto *p = reinterpret_cast<cvk_sparse_pyr_lk_optical_flow *>(h);
    guarded([&]() -> int {
        p->ptr->calc(*require_const(prev_img), *require_const(next_img),
                     *require_const(prev_pts), *require(next_pts), *require(status),
                     err ? *require(err) : cv::noArray());
        return 0;
    });
}

int cvk_sparse_pyr_lk_optical_flow_get_win_w(const cvk_sparse_pyr_lk_optical_flow_t *h) {
    auto *p = reinterpret_cast<const cvk_sparse_pyr_lk_optical_flow *>(h);
    return guarded([&]() -> int { return p->ptr->getWinSize().width; });
}

int cvk_sparse_pyr_lk_optical_flow_get_win_h(const cvk_sparse_pyr_lk_optical_flow_t *h) {
    auto *p = reinterpret_cast<const cvk_sparse_pyr_lk_optical_flow *>(h);
    return guarded([&]() -> int { return p->ptr->getWinSize().height; });
}

void cvk_sparse_pyr_lk_optical_flow_set_win_size(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                 int win_w, int win_h) {
    auto *p = reinterpret_cast<cvk_sparse_pyr_lk_optical_flow *>(h);
    guarded([&]() -> int { p->ptr->setWinSize(cv::Size(win_w, win_h)); return 0; });
}

CVK_GET_SET_INT(sparse_pyr_lk_optical_flow, max_level, MaxLevel)

void cvk_sparse_pyr_lk_optical_flow_get_term_criteria(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                      int *tc_type, int *tc_max_count,
                                                      double *tc_epsilon) {
    auto *p = reinterpret_cast<cvk_sparse_pyr_lk_optical_flow *>(h);
    guarded([&]() -> int {
        cv::TermCriteria tc = p->ptr->getTermCriteria();
        if (tc_type) *tc_type = tc.type;
        if (tc_max_count) *tc_max_count = tc.maxCount;
        if (tc_epsilon) *tc_epsilon = tc.epsilon;
        return 0;
    });
}

void cvk_sparse_pyr_lk_optical_flow_set_term_criteria(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                      int tc_type, int tc_max_count,
                                                      double tc_epsilon) {
    auto *p = reinterpret_cast<cvk_sparse_pyr_lk_optical_flow *>(h);
    guarded([&]() -> int {
        cv::TermCriteria tc = term_criteria(tc_type, tc_max_count, tc_epsilon);
        p->ptr->setTermCriteria(tc);
        return 0;
    });
}

CVK_GET_SET_INT(sparse_pyr_lk_optical_flow, flags, Flags)
CVK_GET_SET_DOUBLE(sparse_pyr_lk_optical_flow, min_eig_threshold, MinEigThreshold)

/* =====================================================================
 * VariationalRefinement
 * ===================================================================== */

cvk_variational_refinement_t *cvk_variational_refinement_create(void) {
    return guarded([&]() -> cvk_variational_refinement_t * {
        auto *h = new cvk_variational_refinement;
        h->ptr = cv::VariationalRefinement::create();
        return reinterpret_cast<cvk_variational_refinement_t *>(h);
    });
}

void cvk_variational_refinement_calc(cvk_variational_refinement_t *h,
                                     const cvk_mat_t *i0, const cvk_mat_t *i1,
                                     cvk_mat_t *flow) {
    auto *p = reinterpret_cast<cvk_variational_refinement *>(h);
    guarded([&]() -> int {
        p->ptr->calc(*require_const(i0), *require_const(i1), *require(flow));
        return 0;
    });
}

void cvk_variational_refinement_calc_uv(cvk_variational_refinement_t *h,
                                        const cvk_mat_t *i0, const cvk_mat_t *i1,
                                        cvk_mat_t *flow_u, cvk_mat_t *flow_v) {
    auto *p = reinterpret_cast<cvk_variational_refinement *>(h);
    guarded([&]() -> int {
        p->ptr->calcUV(*require_const(i0), *require_const(i1), *require(flow_u),
                       *require(flow_v));
        return 0;
    });
}

void cvk_variational_refinement_collect_garbage(cvk_variational_refinement_t *h) {
    auto *p = reinterpret_cast<cvk_variational_refinement *>(h);
    guarded([&]() -> int { p->ptr->collectGarbage(); return 0; });
}

CVK_GET_SET_INT(variational_refinement, fixed_point_iterations, FixedPointIterations)
CVK_GET_SET_INT(variational_refinement, sor_iterations, SorIterations)
CVK_GET_SET_FLOAT(variational_refinement, omega, Omega)
CVK_GET_SET_FLOAT(variational_refinement, alpha, Alpha)
CVK_GET_SET_FLOAT(variational_refinement, delta, Delta)
CVK_GET_SET_FLOAT(variational_refinement, gamma, Gamma)
CVK_GET_SET_FLOAT(variational_refinement, epsilon, Epsilon)

/* =====================================================================
 * Algorithm interface + release
 * ===================================================================== */

CVK_ALG_FUNCS(background_subtractor_knn)
CVK_ALG_FUNCS(background_subtractor_mog2)
CVK_ALG_FUNCS(farneback_optical_flow)
CVK_ALG_FUNCS(dis_optical_flow)
CVK_ALG_FUNCS(sparse_pyr_lk_optical_flow)
CVK_ALG_FUNCS(variational_refinement)

void cvk_background_subtractor_knn_release(cvk_background_subtractor_knn_t *h) {
    delete reinterpret_cast<cvk_background_subtractor_knn *>(h);
}

void cvk_background_subtractor_mog2_release(cvk_background_subtractor_mog2_t *h) {
    delete reinterpret_cast<cvk_background_subtractor_mog2 *>(h);
}

void cvk_farneback_optical_flow_release(cvk_farneback_optical_flow_t *h) {
    delete reinterpret_cast<cvk_farneback_optical_flow *>(h);
}

void cvk_dis_optical_flow_release(cvk_dis_optical_flow_t *h) {
    delete reinterpret_cast<cvk_dis_optical_flow *>(h);
}

void cvk_sparse_pyr_lk_optical_flow_release(cvk_sparse_pyr_lk_optical_flow_t *h) {
    delete reinterpret_cast<cvk_sparse_pyr_lk_optical_flow *>(h);
}

void cvk_variational_refinement_release(cvk_variational_refinement_t *h) {
    delete reinterpret_cast<cvk_variational_refinement *>(h);
}

/* =====================================================================
 * KalmanFilter
 * ===================================================================== */

cvk_kalman_filter_t *cvk_kalman_filter_create(int dynam_params, int measure_params,
                                              int control_params, int type) {
    return guarded([&]() -> cvk_kalman_filter_t * {
        auto h = std::unique_ptr<cvk_kalman_filter>(new cvk_kalman_filter);
        if (dynam_params > 0 && measure_params > 0) {
            h->kf.init(dynam_params, measure_params, control_params, type);
        }
        return reinterpret_cast<cvk_kalman_filter_t *>(h.release());
    });
}

cvk_mat_t *cvk_kalman_filter_predict(cvk_kalman_filter_t *h, const cvk_mat_t *control) {
    auto *p = reinterpret_cast<cvk_kalman_filter *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat &result =
            p->kf.predict(control ? *require_const(control) : cv::Mat());
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(result));
    });
}

cvk_mat_t *cvk_kalman_filter_correct(cvk_kalman_filter_t *h, const cvk_mat_t *measurement) {
    auto *p = reinterpret_cast<cvk_kalman_filter *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat &result = p->kf.correct(*require_const(measurement));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(result));
    });
}

/* `name` is the snake_case C symbol suffix, `member` the cv::KalmanFilter
 * public Mat member. */

#define CVK_KF_GET(name, member)                                                  \
    cvk_mat_t *cvk_kalman_filter_get_##name(const cvk_kalman_filter_t *h) {       \
        auto *p = reinterpret_cast<const cvk_kalman_filter *>(h);                 \
        return guarded([&]() -> cvk_mat_t * {                                     \
            return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->kf.member));       \
        });                                                                       \
    }

#define CVK_KF_SET(name, member)                                                  \
    void cvk_kalman_filter_set_##name(cvk_kalman_filter_t *h, const cvk_mat_t *m) { \
        auto *p = reinterpret_cast<cvk_kalman_filter *>(h);                       \
        guarded([&]() -> int { p->kf.member = *require_const(m); return 0; });    \
    }

CVK_KF_GET(state_pre, statePre)
CVK_KF_SET(state_pre, statePre)
CVK_KF_GET(state_post, statePost)
CVK_KF_SET(state_post, statePost)
CVK_KF_GET(transition_matrix, transitionMatrix)
CVK_KF_SET(transition_matrix, transitionMatrix)
CVK_KF_GET(control_matrix, controlMatrix)
CVK_KF_SET(control_matrix, controlMatrix)
CVK_KF_GET(measurement_matrix, measurementMatrix)
CVK_KF_SET(measurement_matrix, measurementMatrix)
CVK_KF_GET(process_noise_cov, processNoiseCov)
CVK_KF_SET(process_noise_cov, processNoiseCov)
CVK_KF_GET(measurement_noise_cov, measurementNoiseCov)
CVK_KF_SET(measurement_noise_cov, measurementNoiseCov)
CVK_KF_GET(error_cov_pre, errorCovPre)
CVK_KF_SET(error_cov_pre, errorCovPre)
CVK_KF_GET(gain, gain)
CVK_KF_SET(gain, gain)
CVK_KF_GET(error_cov_post, errorCovPost)
CVK_KF_SET(error_cov_post, errorCovPost)

void cvk_kalman_filter_release(cvk_kalman_filter_t *h) {
    delete reinterpret_cast<cvk_kalman_filter *>(h);
}

} /* extern "C" */

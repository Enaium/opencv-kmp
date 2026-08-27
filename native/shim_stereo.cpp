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
 * C shim for the OpenCV stereo module (cvk_stereo_*).
 *
 * One opaque handle type backs the whole StereoMatcher hierarchy; the
 * concrete-class accessors dynamic-cast the stored cv::Ptr.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_stereo.h"

#include <opencv2/stereo.hpp>

#include <string>

struct cvk_stereo_matcher {
    cv::Ptr<cv::StereoMatcher> ptr;
};

namespace {

thread_local std::string g_stereo_str;

void record_error(const char *m) { g_stereo_str = m ? m : "unknown error"; }

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

cv::Mat *require(cvk_mat_t *m) {
    if (!m) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<cv::Mat *>(m);
}

const cv::Mat *require_const(const cvk_mat_t *m) {
    if (!m) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<const cv::Mat *>(m);
}

/* Generates the four Algorithm methods for a handle struct. */
#define CVK_ALG_FUNCS(T)                                                       \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                     \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        if (!p) {                                                              \
            record_error("null handle");                                       \
            return;                                                            \
        }                                                                      \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                  \
    }                                                                          \
    int cvk_##T##_empty(const cvk_##T##_t *h) {                                \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                        \
        if (!p) {                                                              \
            record_error("null handle");                                       \
            return 1;                                                          \
        }                                                                      \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });      \
    }                                                                          \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {                \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        if (!p) {                                                              \
            record_error("null handle");                                       \
            return;                                                            \
        }                                                                      \
        guarded([&]() -> int { p->ptr->save(filename); return 0; });           \
    }                                                                          \
    const char *cvk_##T##_get_default_name(const cvk_##T##_t *h) {             \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                        \
        if (!p) {                                                              \
            record_error("null handle");                                       \
            return nullptr;                                                    \
        }                                                                      \
        return guarded([&]() -> const char * {                                 \
            g_stereo_str = p->ptr->getDefaultName();                           \
            return g_stereo_str.c_str();                                       \
        });                                                                    \
    }

}  // namespace

extern "C" {

/* ---- factories -------------------------------------------------------- */

cvk_stereo_matcher_t *cvk_stereo_matcher_create_bm(int num_disparities, int block_size) {
    return guarded([&]() -> cvk_stereo_matcher_t * {
        auto *h = new cvk_stereo_matcher;
        h->ptr = cv::StereoBM::create(num_disparities, block_size);
        return reinterpret_cast<cvk_stereo_matcher_t *>(h);
    });
}

cvk_stereo_matcher_t *cvk_stereo_matcher_create_sgbm(int min_disparity, int num_disparities,
                                                     int block_size, int p1, int p2,
                                                     int disp12_max_diff, int pre_filter_cap,
                                                     int uniqueness_ratio, int speckle_window_size,
                                                     int speckle_range, int mode) {
    return guarded([&]() -> cvk_stereo_matcher_t * {
        auto *h = new cvk_stereo_matcher;
        h->ptr = cv::StereoSGBM::create(min_disparity, num_disparities, block_size, p1, p2,
                                        disp12_max_diff, pre_filter_cap, uniqueness_ratio,
                                        speckle_window_size, speckle_range, mode);
        return reinterpret_cast<cvk_stereo_matcher_t *>(h);
    });
}

cvk_mat_t *cvk_stereo_matcher_compute(const cvk_stereo_matcher_t *h, const cvk_mat_t *left,
                                      const cvk_mat_t *right) {
    auto *p = reinterpret_cast<const cvk_stereo_matcher *>(h);
    const cv::Mat *l = require_const(left);
    const cv::Mat *r = require_const(right);
    if (!p || !l || !r) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->compute(*l, *r, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* ---- Algorithm surface ------------------------------------------------- */

CVK_ALG_FUNCS(stereo_matcher)

/* ---- StereoMatcher properties ------------------------------------------ */

#define CVK_MATCHER_GET_SET(T, name, cpp_get, cpp_set)                      \
    int cvk_stereo_matcher_get_##name(const cvk_stereo_matcher_t *h) {      \
        auto *p = reinterpret_cast<const cvk_stereo_matcher *>(h);          \
        if (!p) {                                                           \
            record_error("null StereoMatcher handle");                      \
            return 0;                                                       \
        }                                                                   \
        return guarded([&]() -> int { return p->ptr->cpp_get(); });         \
    }                                                                       \
    void cvk_stereo_matcher_set_##name(cvk_stereo_matcher_t *h, int v) {    \
        auto *p = reinterpret_cast<cvk_stereo_matcher *>(h);                \
        if (!p) {                                                           \
            record_error("null StereoMatcher handle");                      \
            return;                                                         \
        }                                                                   \
        guarded([&]() -> int { p->ptr->cpp_set(v); return 0; });            \
    }

CVK_MATCHER_GET_SET(, min_disparity, getMinDisparity, setMinDisparity)
CVK_MATCHER_GET_SET(, num_disparities, getNumDisparities, setNumDisparities)
CVK_MATCHER_GET_SET(, block_size, getBlockSize, setBlockSize)
CVK_MATCHER_GET_SET(, speckle_window_size, getSpeckleWindowSize, setSpeckleWindowSize)
CVK_MATCHER_GET_SET(, speckle_range, getSpeckleRange, setSpeckleRange)
CVK_MATCHER_GET_SET(, disp12_max_diff, getDisp12MaxDiff, setDisp12MaxDiff)

#undef CVK_MATCHER_GET_SET

/* ---- StereoBM properties ----------------------------------------------- */

namespace {

const cv::Ptr<cv::StereoBM> as_bm(const cvk_stereo_matcher_t *h, bool *ok) {
    auto *p = reinterpret_cast<const cvk_stereo_matcher *>(h);
    *ok = false;
    if (!p) {
        record_error("null StereoMatcher handle");
        return cv::Ptr<cv::StereoBM>();
    }
    cv::Ptr<cv::StereoBM> bm = p->ptr.dynamicCast<cv::StereoBM>();
    if (!bm) {
        record_error("handle is not a StereoBM");
        return cv::Ptr<cv::StereoBM>();
    }
    *ok = true;
    return bm;
}

}  // namespace

#define CVK_BM_GET_SET(name, cpp_get, cpp_set)                              \
    int cvk_stereo_bm_get_##name(const cvk_stereo_matcher_t *h) {           \
        bool ok = false;                                                    \
        cv::Ptr<cv::StereoBM> bm = as_bm(h, &ok);                           \
        if (!ok) return 0;                                                  \
        return guarded([&]() -> int { return bm->cpp_get(); });             \
    }                                                                       \
    void cvk_stereo_bm_set_##name(cvk_stereo_matcher_t *h, int v) {         \
        bool ok = false;                                                    \
        cv::Ptr<cv::StereoBM> bm = as_bm(h, &ok);                           \
        if (!ok) return;                                                    \
        guarded([&]() -> int { bm->cpp_set(v); return 0; });                \
    }

CVK_BM_GET_SET(pre_filter_type, getPreFilterType, setPreFilterType)
CVK_BM_GET_SET(pre_filter_size, getPreFilterSize, setPreFilterSize)
CVK_BM_GET_SET(pre_filter_cap, getPreFilterCap, setPreFilterCap)
CVK_BM_GET_SET(texture_threshold, getTextureThreshold, setTextureThreshold)
CVK_BM_GET_SET(uniqueness_ratio, getUniquenessRatio, setUniquenessRatio)
CVK_BM_GET_SET(smaller_block_size, getSmallerBlockSize, setSmallerBlockSize)

#undef CVK_BM_GET_SET

cvk_rect_t cvk_stereo_bm_get_roi1(const cvk_stereo_matcher_t *h) {
    bool ok = false;
    cv::Ptr<cv::StereoBM> bm = as_bm(h, &ok);
    if (!ok) return cvk_rect_t{0, 0, 0, 0};
    return guarded([&]() -> cvk_rect_t {
        cv::Rect r = bm->getROI1();
        return cvk_rect_t{r.x, r.y, r.width, r.height};
    });
}

void cvk_stereo_bm_set_roi1(cvk_stereo_matcher_t *h, cvk_rect_t roi) {
    bool ok = false;
    cv::Ptr<cv::StereoBM> bm = as_bm(h, &ok);
    if (!ok) return;
    guarded([&]() -> int {
        bm->setROI1(cv::Rect(roi.x, roi.y, roi.width, roi.height));
        return 0;
    });
}

cvk_rect_t cvk_stereo_bm_get_roi2(const cvk_stereo_matcher_t *h) {
    bool ok = false;
    cv::Ptr<cv::StereoBM> bm = as_bm(h, &ok);
    if (!ok) return cvk_rect_t{0, 0, 0, 0};
    return guarded([&]() -> cvk_rect_t {
        cv::Rect r = bm->getROI2();
        return cvk_rect_t{r.x, r.y, r.width, r.height};
    });
}

void cvk_stereo_bm_set_roi2(cvk_stereo_matcher_t *h, cvk_rect_t roi) {
    bool ok = false;
    cv::Ptr<cv::StereoBM> bm = as_bm(h, &ok);
    if (!ok) return;
    guarded([&]() -> int {
        bm->setROI2(cv::Rect(roi.x, roi.y, roi.width, roi.height));
        return 0;
    });
}

/* ---- StereoSGBM properties --------------------------------------------- */

namespace {

const cv::Ptr<cv::StereoSGBM> as_sgbm(const cvk_stereo_matcher_t *h, bool *ok) {
    auto *p = reinterpret_cast<const cvk_stereo_matcher *>(h);
    *ok = false;
    if (!p) {
        record_error("null StereoMatcher handle");
        return cv::Ptr<cv::StereoSGBM>();
    }
    cv::Ptr<cv::StereoSGBM> sgbm = p->ptr.dynamicCast<cv::StereoSGBM>();
    if (!sgbm) {
        record_error("handle is not a StereoSGBM");
        return cv::Ptr<cv::StereoSGBM>();
    }
    *ok = true;
    return sgbm;
}

}  // namespace

#define CVK_SGBM_GET_SET(name, cpp_get, cpp_set)                             \
    int cvk_stereo_sgbm_get_##name(const cvk_stereo_matcher_t *h) {          \
        bool ok = false;                                                     \
        cv::Ptr<cv::StereoSGBM> sgbm = as_sgbm(h, &ok);                      \
        if (!ok) return 0;                                                   \
        return guarded([&]() -> int { return sgbm->cpp_get(); });            \
    }                                                                        \
    void cvk_stereo_sgbm_set_##name(cvk_stereo_matcher_t *h, int v) {        \
        bool ok = false;                                                     \
        cv::Ptr<cv::StereoSGBM> sgbm = as_sgbm(h, &ok);                      \
        if (!ok) return;                                                     \
        guarded([&]() -> int { sgbm->cpp_set(v); return 0; });               \
    }

CVK_SGBM_GET_SET(pre_filter_cap, getPreFilterCap, setPreFilterCap)
CVK_SGBM_GET_SET(uniqueness_ratio, getUniquenessRatio, setUniquenessRatio)
CVK_SGBM_GET_SET(p1, getP1, setP1)
CVK_SGBM_GET_SET(p2, getP2, setP2)
CVK_SGBM_GET_SET(mode, getMode, setMode)

#undef CVK_SGBM_GET_SET

void cvk_stereo_matcher_release(cvk_stereo_matcher_t *h) {
    delete reinterpret_cast<cvk_stereo_matcher *>(h);
}

/* ---- Stereo free functions --------------------------------------------- */

int cvk_stereo_rectify(const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                       const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                       int image_width, int image_height, const cvk_mat_t *r,
                       const cvk_mat_t *t, int flags, double alpha, int new_image_width,
                       int new_image_height, cvk_mat_t **r1, cvk_mat_t **r2,
                       cvk_mat_t **p1, cvk_mat_t **p2, cvk_mat_t **q, cvk_rect_t *roi1,
                       cvk_rect_t *roi2) {
    const cv::Mat *cm1 = require_const(camera_matrix1);
    const cv::Mat *dc1 = require_const(dist_coeffs1);
    const cv::Mat *cm2 = require_const(camera_matrix2);
    const cv::Mat *dc2 = require_const(dist_coeffs2);
    const cv::Mat *rr = require_const(r);
    const cv::Mat *tt = require_const(t);
    if (!cm1 || !dc1 || !cm2 || !dc2 || !rr || !tt) return 0;
    return guarded([&]() -> int {
        cv::Mat out_r1, out_r2, out_p1, out_p2, out_q;
        cv::Rect valid1, valid2;
        cv::Rect *p_valid1 = roi1 != nullptr ? &valid1 : nullptr;
        cv::Rect *p_valid2 = roi2 != nullptr ? &valid2 : nullptr;
        cv::stereoRectify(*cm1, *dc1, *cm2, *dc2,
                          cv::Size(image_width, image_height), *rr, *tt, out_r1, out_r2,
                          out_p1, out_p2, out_q, flags, alpha,
                          cv::Size(new_image_width, new_image_height), p_valid1, p_valid2);
        if (r1) *r1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_r1));
        if (r2) *r2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_r2));
        if (p1) *p1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_p1));
        if (p2) *p2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_p2));
        if (q) *q = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_q));
        if (roi1) *roi1 = cvk_rect_t{valid1.x, valid1.y, valid1.width, valid1.height};
        if (roi2) *roi2 = cvk_rect_t{valid2.x, valid2.y, valid2.width, valid2.height};
        return 1;
    });
}

int cvk_stereo_rectify_uncalibrated(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                    const cvk_mat_t *f, int img_width, int img_height,
                                    double threshold, cvk_mat_t **h1, cvk_mat_t **h2) {
    const cv::Mat *p1 = require_const(points1);
    const cv::Mat *p2 = require_const(points2);
    const cv::Mat *ff = require_const(f);
    if (!p1 || !p2 || !ff) return 0;
    return guarded([&]() -> int {
        cv::Mat out_h1, out_h2;
        bool ok = cv::stereoRectifyUncalibrated(*p1, *p2, *ff,
                                                cv::Size(img_width, img_height), out_h1,
                                                out_h2, threshold);
        if (h1) *h1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_h1));
        if (h2) *h2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_h2));
        return ok ? 1 : 0;
    });
}

int cvk_stereo_fisheye_rectify(const cvk_mat_t *k1, const cvk_mat_t *d1,
                               const cvk_mat_t *k2, const cvk_mat_t *d2, int image_width,
                               int image_height, const cvk_mat_t *r, const cvk_mat_t *tvec,
                               int flags, int new_image_width, int new_image_height,
                               double balance, double fov_scale, cvk_mat_t **r1,
                               cvk_mat_t **r2, cvk_mat_t **p1, cvk_mat_t **p2,
                               cvk_mat_t **q) {
    const cv::Mat *kk1 = require_const(k1);
    const cv::Mat *dd1 = require_const(d1);
    const cv::Mat *kk2 = require_const(k2);
    const cv::Mat *dd2 = require_const(d2);
    const cv::Mat *rr = require_const(r);
    const cv::Mat *tt = require_const(tvec);
    if (!kk1 || !dd1 || !kk2 || !dd2 || !rr || !tt) return 0;
    return guarded([&]() -> int {
        cv::Mat out_r1, out_r2, out_p1, out_p2, out_q;
        cv::fisheye::stereoRectify(*kk1, *dd1, *kk2, *dd2,
                                   cv::Size(image_width, image_height), *rr, *tt, out_r1,
                                   out_r2, out_p1, out_p2, out_q, flags,
                                   cv::Size(new_image_width, new_image_height), balance,
                                   fov_scale);
        if (r1) *r1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_r1));
        if (r2) *r2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_r2));
        if (p1) *p1 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_p1));
        if (p2) *p2 = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_p2));
        if (q) *q = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out_q));
        return 1;
    });
}

void cvk_stereo_filter_speckles(cvk_mat_t *img, double new_val, int max_speckle_size,
                                double max_diff) {
    cv::Mat *m = require(img);
    if (!m) return;
    guarded([&]() -> int {
        cv::filterSpeckles(*m, new_val, max_speckle_size, max_diff);
        return 0;
    });
}

void cvk_stereo_filter_speckles_buf(cvk_mat_t *img, double new_val, int max_speckle_size,
                                    double max_diff, cvk_mat_t *buf) {
    cv::Mat *m = require(img);
    cv::Mat *b = require(buf);
    if (!m || !b) return;
    guarded([&]() -> int {
        cv::filterSpeckles(*m, new_val, max_speckle_size, max_diff, *b);
        return 0;
    });
}

cvk_rect_t cvk_stereo_get_valid_disparity_roi(cvk_rect_t roi1, cvk_rect_t roi2,
                                              int min_disparity, int number_of_disparities,
                                              int block_size) {
    return guarded([&]() -> cvk_rect_t {
        cv::Rect r1(roi1.x, roi1.y, roi1.width, roi1.height);
        cv::Rect r2(roi2.x, roi2.y, roi2.width, roi2.height);
        cv::Rect r = cv::getValidDisparityROI(r1, r2, min_disparity, number_of_disparities,
                                              block_size);
        return cvk_rect_t{r.x, r.y, r.width, r.height};
    });
}

void cvk_stereo_validate_disparity(cvk_mat_t *disparity, const cvk_mat_t *cost,
                                   int min_disparity, int number_of_disparities,
                                   int disp12_max_disp) {
    cv::Mat *d = require(disparity);
    const cv::Mat *c = require_const(cost);
    if (!d || !c) return;
    guarded([&]() -> int {
        cv::validateDisparity(*d, *c, min_disparity, number_of_disparities, disp12_max_disp);
        return 0;
    });
}

cvk_mat_t *cvk_stereo_reproject_image_to_3d(const cvk_mat_t *disparity,
                                            const cvk_mat_t *q, int handle_missing_values,
                                            int ddepth) {
    const cv::Mat *d = require_const(disparity);
    const cv::Mat *qq = require_const(q);
    if (!d || !qq) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::reprojectImageTo3D(*d, *dst, *qq, handle_missing_values != 0, ddepth);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

} /* extern "C" */

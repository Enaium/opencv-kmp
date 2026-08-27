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
 * cvk_ C ABI declarations for the OpenCV "stereo" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * A single opaque handle type backs the StereoMatcher hierarchy: StereoBM
 * and StereoSGBM instances are created through the matching factory and the
 * concrete-class accessors dynamic-cast the wrapped cv::Ptr internally.
 */
#ifndef OPENCV_KMP_STEREO_H
#define OPENCV_KMP_STEREO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvk_stereo_matcher cvk_stereo_matcher_t;

/* ---- factories (return a handle to a cv::Ptr<cv::StereoMatcher>) ---- */

/** cv::StereoBM::create(numDisparities, blockSize). */
cvk_stereo_matcher_t *cvk_stereo_matcher_create_bm(int num_disparities, int block_size);

/** cv::StereoSGBM::create(minDisparity, ..., mode). */
cvk_stereo_matcher_t *cvk_stereo_matcher_create_sgbm(int min_disparity, int num_disparities,
                                                     int block_size, int p1, int p2,
                                                     int disp12_max_diff, int pre_filter_cap,
                                                     int uniqueness_ratio, int speckle_window_size,
                                                     int speckle_range, int mode);

/** cv::StereoMatcher::compute; returns a new Mat with the disparity map. */
cvk_mat_t *cvk_stereo_matcher_compute(const cvk_stereo_matcher_t *h, const cvk_mat_t *left,
                                      const cvk_mat_t *right);

/* ---- Algorithm surface (clear/empty/save/getDefaultName) ---- */

void cvk_stereo_matcher_clear(cvk_stereo_matcher_t *h);
int cvk_stereo_matcher_empty(const cvk_stereo_matcher_t *h);
void cvk_stereo_matcher_save(cvk_stereo_matcher_t *h, const char *filename);
const char *cvk_stereo_matcher_get_default_name(const cvk_stereo_matcher_t *h);

/* ---- StereoMatcher properties (valid on any matcher) ---- */

int cvk_stereo_matcher_get_min_disparity(const cvk_stereo_matcher_t *h);
void cvk_stereo_matcher_set_min_disparity(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_matcher_get_num_disparities(const cvk_stereo_matcher_t *h);
void cvk_stereo_matcher_set_num_disparities(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_matcher_get_block_size(const cvk_stereo_matcher_t *h);
void cvk_stereo_matcher_set_block_size(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_matcher_get_speckle_window_size(const cvk_stereo_matcher_t *h);
void cvk_stereo_matcher_set_speckle_window_size(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_matcher_get_speckle_range(const cvk_stereo_matcher_t *h);
void cvk_stereo_matcher_set_speckle_range(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_matcher_get_disp12_max_diff(const cvk_stereo_matcher_t *h);
void cvk_stereo_matcher_set_disp12_max_diff(cvk_stereo_matcher_t *h, int v);

/* ---- StereoBM properties (handle must wrap a StereoBM) ---- */

int cvk_stereo_bm_get_pre_filter_type(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_pre_filter_type(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_bm_get_pre_filter_size(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_pre_filter_size(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_bm_get_pre_filter_cap(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_pre_filter_cap(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_bm_get_texture_threshold(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_texture_threshold(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_bm_get_uniqueness_ratio(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_uniqueness_ratio(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_bm_get_smaller_block_size(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_smaller_block_size(cvk_stereo_matcher_t *h, int v);
cvk_rect_t cvk_stereo_bm_get_roi1(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_roi1(cvk_stereo_matcher_t *h, cvk_rect_t roi);
cvk_rect_t cvk_stereo_bm_get_roi2(const cvk_stereo_matcher_t *h);
void cvk_stereo_bm_set_roi2(cvk_stereo_matcher_t *h, cvk_rect_t roi);

/* ---- StereoSGBM properties (handle must wrap a StereoSGBM) ---- */

int cvk_stereo_sgbm_get_pre_filter_cap(const cvk_stereo_matcher_t *h);
void cvk_stereo_sgbm_set_pre_filter_cap(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_sgbm_get_uniqueness_ratio(const cvk_stereo_matcher_t *h);
void cvk_stereo_sgbm_set_uniqueness_ratio(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_sgbm_get_p1(const cvk_stereo_matcher_t *h);
void cvk_stereo_sgbm_set_p1(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_sgbm_get_p2(const cvk_stereo_matcher_t *h);
void cvk_stereo_sgbm_set_p2(cvk_stereo_matcher_t *h, int v);
int cvk_stereo_sgbm_get_mode(const cvk_stereo_matcher_t *h);
void cvk_stereo_sgbm_set_mode(cvk_stereo_matcher_t *h, int v);

/** Frees the handle (exactly once). */
void cvk_stereo_matcher_release(cvk_stereo_matcher_t *h);

/* =========================================================================
 * Stereo free functions (org.opencv.stereo.Stereo parity)
 * ========================================================================= */

/**
 * cv::stereoRectify. Writes the four rectification/projection matrices into
 * r1/r2/p1/p2/q and, when roi1/roi2 are non-NULL, the valid pixel ROIs.
 * Returns 1 on success, 0 on failure.
 */
int cvk_stereo_rectify(const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                       const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                       int image_width, int image_height, const cvk_mat_t *r,
                       const cvk_mat_t *t, int flags, double alpha, int new_image_width,
                       int new_image_height, cvk_mat_t **r1, cvk_mat_t **r2,
                       cvk_mat_t **p1, cvk_mat_t **p2, cvk_mat_t **q, cvk_rect_t *roi1,
                       cvk_rect_t *roi2);

/**
 * cv::stereoRectifyUncalibrated. Writes H1/H2 handles; returns 1 on success.
 */
int cvk_stereo_rectify_uncalibrated(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                    const cvk_mat_t *f, int img_width, int img_height,
                                    double threshold, cvk_mat_t **h1, cvk_mat_t **h2);

/** cv::fisheye::stereoRectify (4x4 outputs + no ROIs). */
int cvk_stereo_fisheye_rectify(const cvk_mat_t *k1, const cvk_mat_t *d1,
                               const cvk_mat_t *k2, const cvk_mat_t *d2, int image_width,
                               int image_height, const cvk_mat_t *r, const cvk_mat_t *tvec,
                               int flags, int new_image_width, int new_image_height,
                               double balance, double fov_scale, cvk_mat_t **r1,
                               cvk_mat_t **r2, cvk_mat_t **p1, cvk_mat_t **p2,
                               cvk_mat_t **q);

/** cv::filterSpeckles; mutates img in place (buf variant uses a caller buffer). */
void cvk_stereo_filter_speckles(cvk_mat_t *img, double new_val, int max_speckle_size,
                                double max_diff);
void cvk_stereo_filter_speckles_buf(cvk_mat_t *img, double new_val, int max_speckle_size,
                                    double max_diff, cvk_mat_t *buf);

/** cv::getValidDisparityROI; all-zero rect on failure. */
cvk_rect_t cvk_stereo_get_valid_disparity_roi(cvk_rect_t roi1, cvk_rect_t roi2,
                                              int min_disparity, int number_of_disparities,
                                              int block_size);

/** cv::validateDisparity; mutates disparity in place. */
void cvk_stereo_validate_disparity(cvk_mat_t *disparity, const cvk_mat_t *cost,
                                   int min_disparity, int number_of_disparities,
                                   int disp12_max_disp);

/** cv::reprojectImageTo3D; returns a new Mat (the _3dImage). */
cvk_mat_t *cvk_stereo_reproject_image_to_3d(const cvk_mat_t *disparity,
                                            const cvk_mat_t *q, int handle_missing_values,
                                            int ddepth);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_STEREO_H */

/*
 * cvk_ C ABI declarations for the OpenCV "photo" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 */
#ifndef OPENCV_KMP_PHOTO_H
#define OPENCV_KMP_PHOTO_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles; struct bodies live in shim_photo.cpp. */
typedef struct cvk_tonemap cvk_tonemap_t;
typedef struct cvk_tonemap_drago cvk_tonemap_drago_t;
typedef struct cvk_tonemap_mantiuk cvk_tonemap_mantiuk_t;
typedef struct cvk_tonemap_reinhard cvk_tonemap_reinhard_t;
typedef struct cvk_align_mtb cvk_align_mtb_t;
typedef struct cvk_calibrate_debevec cvk_calibrate_debevec_t;
typedef struct cvk_calibrate_robertson cvk_calibrate_robertson_t;
typedef struct cvk_merge_debevec cvk_merge_debevec_t;
typedef struct cvk_merge_mertens cvk_merge_mertens_t;
typedef struct cvk_merge_robertson cvk_merge_robertson_t;
typedef struct cvk_color_correction_model cvk_color_correction_model_t;
typedef struct cvk_intelligent_scissors_mb cvk_intelligent_scissors_mb_t;

/* =========================================================================
 * Photo statics
 * ========================================================================= */

/** cv::inpaint; flags: INPAINT_NS=0 or INPAINT_TELEA=1. */
cvk_mat_t *cvk_inpaint(const cvk_mat_t *src, const cvk_mat_t *inpaint_mask,
                       double inpaint_radius, int flags);

/** cv::fastNlMeansDenoising with a scalar h. */
cvk_mat_t *cvk_fast_nl_means_denoising(const cvk_mat_t *src, float h,
                                       int template_window_size, int search_window_size);

/** cv::fastNlMeansDenoising with a per-channel vector h (CV_32FC1 Mat). */
cvk_mat_t *cvk_fast_nl_means_denoising_h(const cvk_mat_t *src, const cvk_mat_t *h,
                                         int template_window_size, int search_window_size,
                                         int norm_type);

/** cv::fastNlMeansDenoisingMulti (grayscale sequence, scalar h). */
cvk_mat_t *cvk_fast_nl_means_denoising_multi(const cvk_mat_t *const *src, int count,
                                             int img_to_denoise_index,
                                             int temporal_window_size, float h,
                                             int template_window_size, int search_window_size);

/** cv::fastNlMeansDenoisingMulti with a per-channel vector h. */
cvk_mat_t *cvk_fast_nl_means_denoising_multi_h(const cvk_mat_t *const *src, int count,
                                               int img_to_denoise_index,
                                               int temporal_window_size,
                                               const cvk_mat_t *h,
                                               int template_window_size,
                                               int search_window_size, int norm_type);

/** cv::fastNlMeansDenoisingColored. */
cvk_mat_t *cvk_fast_nl_means_denoising_colored(const cvk_mat_t *src, float h, float h_color,
                                               int template_window_size, int search_window_size);

/** cv::fastNlMeansDenoisingColoredMulti (color sequence). */
cvk_mat_t *cvk_fast_nl_means_denoising_colored_multi(const cvk_mat_t *const *src, int count,
                                                     int img_to_denoise_index,
                                                     int temporal_window_size,
                                                     float h, float h_color,
                                                     int template_window_size,
                                                     int search_window_size);

/** cv::denoise_TVL1; result is CV_32FC3. */
cvk_mat_t *cvk_denoise_tvl1(const cvk_mat_t *const *src, int count, double lambda, int niters);

/** cv::decolor; out params receive caller-owned Mat handles. */
void cvk_decolor(const cvk_mat_t *src, cvk_mat_t **grayscale, cvk_mat_t **color_boost);

/** cv::seamlessClone; flags: NORMAL_CLONE=1 etc. */
cvk_mat_t *cvk_seamless_clone(const cvk_mat_t *src, const cvk_mat_t *dst,
                              const cvk_mat_t *mask, int p_x, int p_y, int flags);

/** cv::colorChange. */
cvk_mat_t *cvk_color_change(const cvk_mat_t *src, const cvk_mat_t *mask,
                            float red_mul, float green_mul, float blue_mul);

/** cv::illuminationChange. */
cvk_mat_t *cvk_illumination_change(const cvk_mat_t *src, const cvk_mat_t *mask,
                                   float alpha, float beta);

/** cv::textureFlattening. */
cvk_mat_t *cvk_texture_flattening(const cvk_mat_t *src, const cvk_mat_t *mask,
                                  float low_threshold, float high_threshold,
                                  int kernel_size);

/** cv::edgePreservingFilter; flags: RECURS_FILTER=1 or NORMCONV_FILTER=2. */
cvk_mat_t *cvk_edge_preserving_filter(const cvk_mat_t *src, int flags,
                                      float sigma_s, float sigma_r);

/** cv::detailEnhance. */
cvk_mat_t *cvk_detail_enhance(const cvk_mat_t *src, float sigma_s, float sigma_r);

/** cv::pencilSketch; out params receive caller-owned Mat handles. */
void cvk_pencil_sketch(const cvk_mat_t *src, cvk_mat_t **dst1, cvk_mat_t **dst2,
                       float sigma_s, float sigma_r, float shade_factor);

/** cv::stylization. */
cvk_mat_t *cvk_stylization(const cvk_mat_t *src, float sigma_s, float sigma_r);

/** cv::correctChromaticAberration. */
cvk_mat_t *cvk_correct_chromatic_aberration(const cvk_mat_t *input,
                                            const cvk_mat_t *coefficients,
                                            int image_width, int image_height,
                                            int calib_degree, int bayer_pattern);

/** cv::ccm::gammaCorrection. */
cvk_mat_t *cvk_gamma_correction(const cvk_mat_t *src, double gamma);

/* =========================================================================
 * Tonemap (cv::createTonemap)
 * ========================================================================= */

cvk_tonemap_t *cvk_tonemap_create(float gamma);
void cvk_tonemap_release(cvk_tonemap_t *h);
cvk_mat_t *cvk_tonemap_process(const cvk_tonemap_t *h, const cvk_mat_t *src);
float cvk_tonemap_get_gamma(const cvk_tonemap_t *h);
void cvk_tonemap_set_gamma(cvk_tonemap_t *h, float gamma);
void cvk_tonemap_clear(cvk_tonemap_t *h);
int cvk_tonemap_empty(const cvk_tonemap_t *h);
void cvk_tonemap_save(cvk_tonemap_t *h, const char *filename);
const char *cvk_tonemap_get_default_name(const cvk_tonemap_t *h);

/* =========================================================================
 * TonemapDrago (cv::createTonemapDrago)
 * ========================================================================= */

cvk_tonemap_drago_t *cvk_tonemap_drago_create(float gamma, float saturation, float bias);
void cvk_tonemap_drago_release(cvk_tonemap_drago_t *h);
cvk_mat_t *cvk_tonemap_drago_process(const cvk_tonemap_drago_t *h, const cvk_mat_t *src);
float cvk_tonemap_drago_get_gamma(const cvk_tonemap_drago_t *h);
void cvk_tonemap_drago_set_gamma(cvk_tonemap_drago_t *h, float gamma);
float cvk_tonemap_drago_get_saturation(const cvk_tonemap_drago_t *h);
void cvk_tonemap_drago_set_saturation(cvk_tonemap_drago_t *h, float saturation);
float cvk_tonemap_drago_get_bias(const cvk_tonemap_drago_t *h);
void cvk_tonemap_drago_set_bias(cvk_tonemap_drago_t *h, float bias);
void cvk_tonemap_drago_clear(cvk_tonemap_drago_t *h);
int cvk_tonemap_drago_empty(const cvk_tonemap_drago_t *h);
void cvk_tonemap_drago_save(cvk_tonemap_drago_t *h, const char *filename);
const char *cvk_tonemap_drago_get_default_name(const cvk_tonemap_drago_t *h);

/* =========================================================================
 * TonemapMantiuk (cv::createTonemapMantiuk)
 * ========================================================================= */

cvk_tonemap_mantiuk_t *cvk_tonemap_mantiuk_create(float gamma, float scale, float saturation);
void cvk_tonemap_mantiuk_release(cvk_tonemap_mantiuk_t *h);
cvk_mat_t *cvk_tonemap_mantiuk_process(const cvk_tonemap_mantiuk_t *h, const cvk_mat_t *src);
float cvk_tonemap_mantiuk_get_gamma(const cvk_tonemap_mantiuk_t *h);
void cvk_tonemap_mantiuk_set_gamma(cvk_tonemap_mantiuk_t *h, float gamma);
float cvk_tonemap_mantiuk_get_scale(const cvk_tonemap_mantiuk_t *h);
void cvk_tonemap_mantiuk_set_scale(cvk_tonemap_mantiuk_t *h, float scale);
float cvk_tonemap_mantiuk_get_saturation(const cvk_tonemap_mantiuk_t *h);
void cvk_tonemap_mantiuk_set_saturation(cvk_tonemap_mantiuk_t *h, float saturation);
void cvk_tonemap_mantiuk_clear(cvk_tonemap_mantiuk_t *h);
int cvk_tonemap_mantiuk_empty(const cvk_tonemap_mantiuk_t *h);
void cvk_tonemap_mantiuk_save(cvk_tonemap_mantiuk_t *h, const char *filename);
const char *cvk_tonemap_mantiuk_get_default_name(const cvk_tonemap_mantiuk_t *h);

/* =========================================================================
 * TonemapReinhard (cv::createTonemapReinhard)
 * ========================================================================= */

cvk_tonemap_reinhard_t *cvk_tonemap_reinhard_create(float gamma, float intensity,
                                                    float light_adapt, float color_adapt);
void cvk_tonemap_reinhard_release(cvk_tonemap_reinhard_t *h);
cvk_mat_t *cvk_tonemap_reinhard_process(const cvk_tonemap_reinhard_t *h, const cvk_mat_t *src);
float cvk_tonemap_reinhard_get_gamma(const cvk_tonemap_reinhard_t *h);
void cvk_tonemap_reinhard_set_gamma(cvk_tonemap_reinhard_t *h, float gamma);
float cvk_tonemap_reinhard_get_intensity(const cvk_tonemap_reinhard_t *h);
void cvk_tonemap_reinhard_set_intensity(cvk_tonemap_reinhard_t *h, float intensity);
float cvk_tonemap_reinhard_get_light_adaptation(const cvk_tonemap_reinhard_t *h);
void cvk_tonemap_reinhard_set_light_adaptation(cvk_tonemap_reinhard_t *h, float light_adapt);
float cvk_tonemap_reinhard_get_color_adaptation(const cvk_tonemap_reinhard_t *h);
void cvk_tonemap_reinhard_set_color_adaptation(cvk_tonemap_reinhard_t *h, float color_adapt);
void cvk_tonemap_reinhard_clear(cvk_tonemap_reinhard_t *h);
int cvk_tonemap_reinhard_empty(const cvk_tonemap_reinhard_t *h);
void cvk_tonemap_reinhard_save(cvk_tonemap_reinhard_t *h, const char *filename);
const char *cvk_tonemap_reinhard_get_default_name(const cvk_tonemap_reinhard_t *h);

/* =========================================================================
 * AlignMTB (cv::createAlignMTB)
 * ========================================================================= */

cvk_align_mtb_t *cvk_align_mtb_create(int max_bits, int exclude_range, int cut);
void cvk_align_mtb_release(cvk_align_mtb_t *h);

/** Aligns src into caller-owned Mats written to out (capacity max_out); returns count. */
int cvk_align_mtb_process(const cvk_align_mtb_t *h, const cvk_mat_t *const *src, int count,
                          cvk_mat_t **out, int max_out);
int cvk_align_mtb_process_times(const cvk_align_mtb_t *h, const cvk_mat_t *const *src, int count,
                                const cvk_mat_t *times, const cvk_mat_t *response,
                                cvk_mat_t **out, int max_out);

/** Writes shift x,y into out_xy[2]. */
void cvk_align_mtb_calculate_shift(const cvk_align_mtb_t *h, const cvk_mat_t *img0,
                                   const cvk_mat_t *img1, int *out_xy);

/** Shifts src by (shift_x, shift_y), filling new regions with zeros. */
cvk_mat_t *cvk_align_mtb_shift_mat(const cvk_align_mtb_t *h, const cvk_mat_t *src,
                                   int shift_x, int shift_y);

/** Writes threshold bitmap and exclude bitmap handles into out[2]. */
void cvk_align_mtb_compute_bitmaps(const cvk_align_mtb_t *h, const cvk_mat_t *img,
                                   cvk_mat_t **tb, cvk_mat_t **eb);

int cvk_align_mtb_get_max_bits(const cvk_align_mtb_t *h);
void cvk_align_mtb_set_max_bits(cvk_align_mtb_t *h, int max_bits);
int cvk_align_mtb_get_exclude_range(const cvk_align_mtb_t *h);
void cvk_align_mtb_set_exclude_range(cvk_align_mtb_t *h, int exclude_range);
int cvk_align_mtb_get_cut(const cvk_align_mtb_t *h);
void cvk_align_mtb_set_cut(cvk_align_mtb_t *h, int cut);
void cvk_align_mtb_clear(cvk_align_mtb_t *h);
int cvk_align_mtb_empty(const cvk_align_mtb_t *h);
void cvk_align_mtb_save(cvk_align_mtb_t *h, const char *filename);
const char *cvk_align_mtb_get_default_name(const cvk_align_mtb_t *h);

/* =========================================================================
 * CalibrateDebevec (cv::createCalibrateDebevec)
 * ========================================================================= */

cvk_calibrate_debevec_t *cvk_calibrate_debevec_create(int samples, float lambda, int random);
void cvk_calibrate_debevec_release(cvk_calibrate_debevec_t *h);
cvk_mat_t *cvk_calibrate_debevec_process(const cvk_calibrate_debevec_t *h,
                                         const cvk_mat_t *const *src, int count,
                                         const cvk_mat_t *times);
float cvk_calibrate_debevec_get_lambda(const cvk_calibrate_debevec_t *h);
void cvk_calibrate_debevec_set_lambda(cvk_calibrate_debevec_t *h, float lambda);
int cvk_calibrate_debevec_get_samples(const cvk_calibrate_debevec_t *h);
void cvk_calibrate_debevec_set_samples(cvk_calibrate_debevec_t *h, int samples);
int cvk_calibrate_debevec_get_random(const cvk_calibrate_debevec_t *h);
void cvk_calibrate_debevec_set_random(cvk_calibrate_debevec_t *h, int random);
void cvk_calibrate_debevec_clear(cvk_calibrate_debevec_t *h);
int cvk_calibrate_debevec_empty(const cvk_calibrate_debevec_t *h);
void cvk_calibrate_debevec_save(cvk_calibrate_debevec_t *h, const char *filename);
const char *cvk_calibrate_debevec_get_default_name(const cvk_calibrate_debevec_t *h);

/* =========================================================================
 * CalibrateRobertson (cv::createCalibrateRobertson)
 * ========================================================================= */

cvk_calibrate_robertson_t *cvk_calibrate_robertson_create(int max_iter, float threshold);
void cvk_calibrate_robertson_release(cvk_calibrate_robertson_t *h);
cvk_mat_t *cvk_calibrate_robertson_process(const cvk_calibrate_robertson_t *h,
                                           const cvk_mat_t *const *src, int count,
                                           const cvk_mat_t *times);
cvk_mat_t *cvk_calibrate_robertson_get_radiance(const cvk_calibrate_robertson_t *h);
int cvk_calibrate_robertson_get_max_iter(const cvk_calibrate_robertson_t *h);
void cvk_calibrate_robertson_set_max_iter(cvk_calibrate_robertson_t *h, int max_iter);
float cvk_calibrate_robertson_get_threshold(const cvk_calibrate_robertson_t *h);
void cvk_calibrate_robertson_set_threshold(cvk_calibrate_robertson_t *h, float threshold);
void cvk_calibrate_robertson_clear(cvk_calibrate_robertson_t *h);
int cvk_calibrate_robertson_empty(const cvk_calibrate_robertson_t *h);
void cvk_calibrate_robertson_save(cvk_calibrate_robertson_t *h, const char *filename);
const char *cvk_calibrate_robertson_get_default_name(const cvk_calibrate_robertson_t *h);

/* =========================================================================
 * MergeDebevec (cv::createMergeDebevec)
 * ========================================================================= */

cvk_merge_debevec_t *cvk_merge_debevec_create(void);
void cvk_merge_debevec_release(cvk_merge_debevec_t *h);
cvk_mat_t *cvk_merge_debevec_process(const cvk_merge_debevec_t *h,
                                     const cvk_mat_t *const *src, int count,
                                     const cvk_mat_t *times);
cvk_mat_t *cvk_merge_debevec_process_response(const cvk_merge_debevec_t *h,
                                              const cvk_mat_t *const *src, int count,
                                              const cvk_mat_t *times,
                                              const cvk_mat_t *response);
void cvk_merge_debevec_clear(cvk_merge_debevec_t *h);
int cvk_merge_debevec_empty(const cvk_merge_debevec_t *h);
void cvk_merge_debevec_save(cvk_merge_debevec_t *h, const char *filename);
const char *cvk_merge_debevec_get_default_name(const cvk_merge_debevec_t *h);

/* =========================================================================
 * MergeMertens (cv::createMergeMertens)
 * ========================================================================= */

cvk_merge_mertens_t *cvk_merge_mertens_create(float contrast_weight, float saturation_weight,
                                              float exposure_weight);
void cvk_merge_mertens_release(cvk_merge_mertens_t *h);
cvk_mat_t *cvk_merge_mertens_process(const cvk_merge_mertens_t *h,
                                     const cvk_mat_t *const *src, int count);
cvk_mat_t *cvk_merge_mertens_process_response(const cvk_merge_mertens_t *h,
                                              const cvk_mat_t *const *src, int count,
                                              const cvk_mat_t *times,
                                              const cvk_mat_t *response);
float cvk_merge_mertens_get_contrast_weight(const cvk_merge_mertens_t *h);
void cvk_merge_mertens_set_contrast_weight(cvk_merge_mertens_t *h, float contrast_weight);
float cvk_merge_mertens_get_saturation_weight(const cvk_merge_mertens_t *h);
void cvk_merge_mertens_set_saturation_weight(cvk_merge_mertens_t *h, float saturation_weight);
float cvk_merge_mertens_get_exposure_weight(const cvk_merge_mertens_t *h);
void cvk_merge_mertens_set_exposure_weight(cvk_merge_mertens_t *h, float exposure_weight);
void cvk_merge_mertens_clear(cvk_merge_mertens_t *h);
int cvk_merge_mertens_empty(const cvk_merge_mertens_t *h);
void cvk_merge_mertens_save(cvk_merge_mertens_t *h, const char *filename);
const char *cvk_merge_mertens_get_default_name(const cvk_merge_mertens_t *h);

/* =========================================================================
 * MergeRobertson (cv::createMergeRobertson)
 * ========================================================================= */

cvk_merge_robertson_t *cvk_merge_robertson_create(void);
void cvk_merge_robertson_release(cvk_merge_robertson_t *h);
cvk_mat_t *cvk_merge_robertson_process(const cvk_merge_robertson_t *h,
                                       const cvk_mat_t *const *src, int count,
                                       const cvk_mat_t *times);
cvk_mat_t *cvk_merge_robertson_process_response(const cvk_merge_robertson_t *h,
                                                const cvk_mat_t *const *src, int count,
                                                const cvk_mat_t *times,
                                                const cvk_mat_t *response);
void cvk_merge_robertson_clear(cvk_merge_robertson_t *h);
int cvk_merge_robertson_empty(const cvk_merge_robertson_t *h);
void cvk_merge_robertson_save(cvk_merge_robertson_t *h, const char *filename);
const char *cvk_merge_robertson_get_default_name(const cvk_merge_robertson_t *h);

/* =========================================================================
 * ColorCorrectionModel (cv::ccm::ColorCorrectionModel)
 * ========================================================================= */

/** Default-constructed model (no patches loaded). */
cvk_color_correction_model_t *cvk_color_correction_model_create_empty(void);

/** const_color is a ColorCheckerType value (COLORCHECKER_MACBETH=0 etc.). */
cvk_color_correction_model_t *cvk_color_correction_model_create(const cvk_mat_t *src,
                                                                int const_color);
void cvk_color_correction_model_release(cvk_color_correction_model_t *h);
void cvk_color_correction_model_set_linearization_gamma(cvk_color_correction_model_t *h,
                                                        double gamma);
void cvk_color_correction_model_set_linearization_degree(cvk_color_correction_model_t *h,
                                                         int deg);
void cvk_color_correction_model_set_saturated_threshold(cvk_color_correction_model_t *h,
                                                        double lower, double upper);
void cvk_color_correction_model_set_weights_list(cvk_color_correction_model_t *h,
                                                 const cvk_mat_t *weights);
void cvk_color_correction_model_set_weight_coeff(cvk_color_correction_model_t *h,
                                                 double weights_coeff);
void cvk_color_correction_model_set_max_count(cvk_color_correction_model_t *h, int max_count);
void cvk_color_correction_model_set_epsilon(cvk_color_correction_model_t *h, double epsilon);
void cvk_color_correction_model_set_rgb(cvk_color_correction_model_t *h, int rgb);
cvk_mat_t *cvk_color_correction_model_compute(const cvk_color_correction_model_t *h);
cvk_mat_t *cvk_color_correction_model_get_color_correction_matrix(
        const cvk_color_correction_model_t *h);
double cvk_color_correction_model_get_loss(const cvk_color_correction_model_t *h);
cvk_mat_t *cvk_color_correction_model_get_src_linear_rgb(const cvk_color_correction_model_t *h);
cvk_mat_t *cvk_color_correction_model_get_ref_linear_rgb(const cvk_color_correction_model_t *h);
cvk_mat_t *cvk_color_correction_model_get_mask(const cvk_color_correction_model_t *h);
cvk_mat_t *cvk_color_correction_model_get_weights(const cvk_color_correction_model_t *h);
cvk_mat_t *cvk_color_correction_model_correct_image(const cvk_color_correction_model_t *h,
                                                    const cvk_mat_t *src, int islinear);

/* =========================================================================
 * IntelligentScissorsMB (cv::segmentation::IntelligentScissorsMB)
 * ========================================================================= */

cvk_intelligent_scissors_mb_t *cvk_intelligent_scissors_mb_create(void);
void cvk_intelligent_scissors_mb_release(cvk_intelligent_scissors_mb_t *h);
void cvk_intelligent_scissors_mb_set_weights(cvk_intelligent_scissors_mb_t *h,
                                             float weight_non_edge,
                                             float weight_gradient_direction,
                                             float weight_gradient_magnitude);
void cvk_intelligent_scissors_mb_set_gradient_magnitude_max_limit(
        cvk_intelligent_scissors_mb_t *h, float gradient_magnitude_threshold_max);
void cvk_intelligent_scissors_mb_set_edge_feature_zero_crossing_parameters(
        cvk_intelligent_scissors_mb_t *h, float gradient_magnitude_min_value);
void cvk_intelligent_scissors_mb_set_edge_feature_canny_parameters(
        cvk_intelligent_scissors_mb_t *h, double threshold1, double threshold2,
        int aperture_size, int l2gradient);
void cvk_intelligent_scissors_mb_apply_image(cvk_intelligent_scissors_mb_t *h,
                                             const cvk_mat_t *image);
void cvk_intelligent_scissors_mb_apply_image_features(cvk_intelligent_scissors_mb_t *h,
                                                      const cvk_mat_t *non_edge,
                                                      const cvk_mat_t *gradient_direction,
                                                      const cvk_mat_t *gradient_magnitude,
                                                      const cvk_mat_t *image);
void cvk_intelligent_scissors_mb_build_map(const cvk_intelligent_scissors_mb_t *h,
                                           int source_x, int source_y);
cvk_mat_t *cvk_intelligent_scissors_mb_get_contour(const cvk_intelligent_scissors_mb_t *h,
                                                   int target_x, int target_y, int backward);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_PHOTO_H */

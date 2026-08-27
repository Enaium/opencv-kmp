/*
 * cvk_ C ABI declarations for the OpenCV "video" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 */
#ifndef OPENCV_KMP_VIDEO_H
#define OPENCV_KMP_VIDEO_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- opaque handles ---------------------------------------------------- */

typedef struct cvk_background_subtractor_knn cvk_background_subtractor_knn_t;
typedef struct cvk_background_subtractor_mog2 cvk_background_subtractor_mog2_t;
typedef struct cvk_farneback_optical_flow cvk_farneback_optical_flow_t;
typedef struct cvk_dis_optical_flow cvk_dis_optical_flow_t;
typedef struct cvk_sparse_pyr_lk_optical_flow cvk_sparse_pyr_lk_optical_flow_t;
typedef struct cvk_variational_refinement cvk_variational_refinement_t;
typedef struct cvk_kalman_filter cvk_kalman_filter_t;

/* ---- video statics ----------------------------------------------------- */

/** cv::calcOpticalFlowFarneback; writes the CV_32FC2 flow into `flow`. */
void cvk_calc_optical_flow_farneback(const cvk_mat_t *prev, const cvk_mat_t *next,
                                     cvk_mat_t *flow, double pyr_scale, int levels,
                                     int winsize, int iterations, int poly_n,
                                     double poly_sigma, int flags);

/** cv::calcOpticalFlowPyrLK; writes nextPts/status/err in place. */
void cvk_calc_optical_flow_pyr_lk(const cvk_mat_t *prev_img, const cvk_mat_t *next_img,
                                  const cvk_mat_t *prev_pts, cvk_mat_t *next_pts,
                                  cvk_mat_t *status, cvk_mat_t *err,
                                  int win_w, int win_h, int max_level,
                                  int tc_type, int tc_max_count, double tc_epsilon,
                                  int flags, double min_eig_threshold);

/** cv::computeECC; `input_mask` may be NULL for no mask. */
double cvk_compute_ecc(const cvk_mat_t *template_image, const cvk_mat_t *input_image,
                       const cvk_mat_t *input_mask);

/** cv::findTransformECC; mutates warp_matrix, `input_mask` may be NULL. */
double cvk_find_transform_ecc(const cvk_mat_t *template_image, const cvk_mat_t *input_image,
                              cvk_mat_t *warp_matrix, int motion_type,
                              int tc_type, int tc_max_count, double tc_epsilon,
                              const cvk_mat_t *input_mask, int gauss_filt_size);

/** cv::findTransformECCWithMask; mutates warp_matrix. */
double cvk_find_transform_ecc_with_mask(const cvk_mat_t *template_image,
                                        const cvk_mat_t *input_image,
                                        const cvk_mat_t *template_mask,
                                        const cvk_mat_t *input_mask,
                                        cvk_mat_t *warp_matrix, int motion_type,
                                        int tc_type, int tc_max_count, double tc_epsilon,
                                        int gauss_filt_size);

/** cv::findTransformECCMultiScale; ECCParameters is expanded to scalars,
 *  iters_per_level is a CV_32SC1 Mat (NULL = empty); masks may be NULL. */
double cvk_find_transform_ecc_multi_scale(const cvk_mat_t *reference, const cvk_mat_t *sample,
                                          cvk_mat_t *warp_matrix, int motion_type,
                                          int tc_type, int tc_max_count, double tc_epsilon,
                                          const cvk_mat_t *iters_per_level,
                                          int gauss_filt_size, int nlevels, int interpolation,
                                          const cvk_mat_t *reference_mask,
                                          const cvk_mat_t *sample_mask);

/* ---- BackgroundSubtractorKNN (cv::BackgroundSubtractorKNN) -------------- */

cvk_background_subtractor_knn_t *
cvk_background_subtractor_knn_create(int history, double dist2_threshold, int detect_shadows);

void cvk_background_subtractor_knn_apply(cvk_background_subtractor_knn_t *h,
                                         const cvk_mat_t *image, cvk_mat_t *fgmask,
                                         double learning_rate);

void cvk_background_subtractor_knn_apply_mask(cvk_background_subtractor_knn_t *h,
                                              const cvk_mat_t *image,
                                              const cvk_mat_t *known_foreground_mask,
                                              cvk_mat_t *fgmask, double learning_rate);

/** Returns a new Mat with the estimated background image. */
cvk_mat_t *cvk_background_subtractor_knn_get_background_image(cvk_background_subtractor_knn_t *h);

int cvk_background_subtractor_knn_get_history(const cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_set_history(cvk_background_subtractor_knn_t *h, int history);
int cvk_background_subtractor_knn_get_n_samples(const cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_set_n_samples(cvk_background_subtractor_knn_t *h, int n_samples);
double cvk_background_subtractor_knn_get_dist2_threshold(const cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_set_dist2_threshold(cvk_background_subtractor_knn_t *h,
                                                       double dist2_threshold);
int cvk_background_subtractor_knn_get_knn_samples(const cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_set_knn_samples(cvk_background_subtractor_knn_t *h,
                                                   int knn_samples);
int cvk_background_subtractor_knn_get_detect_shadows(const cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_set_detect_shadows(cvk_background_subtractor_knn_t *h,
                                                      int detect_shadows);
int cvk_background_subtractor_knn_get_shadow_value(const cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_set_shadow_value(cvk_background_subtractor_knn_t *h,
                                                    int shadow_value);
double cvk_background_subtractor_knn_get_shadow_threshold(const cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_set_shadow_threshold(cvk_background_subtractor_knn_t *h,
                                                        double shadow_threshold);

/* ---- BackgroundSubtractorMOG2 (cv::BackgroundSubtractorMOG2) ------------ */

cvk_background_subtractor_mog2_t *
cvk_background_subtractor_mog2_create(int history, double var_threshold, int detect_shadows);

void cvk_background_subtractor_mog2_apply(cvk_background_subtractor_mog2_t *h,
                                          const cvk_mat_t *image, cvk_mat_t *fgmask,
                                          double learning_rate);

void cvk_background_subtractor_mog2_apply_mask(cvk_background_subtractor_mog2_t *h,
                                               const cvk_mat_t *image,
                                               const cvk_mat_t *known_foreground_mask,
                                               cvk_mat_t *fgmask, double learning_rate);

/** Returns a new Mat with the estimated background image. */
cvk_mat_t *cvk_background_subtractor_mog2_get_background_image(cvk_background_subtractor_mog2_t *h);

int cvk_background_subtractor_mog2_get_history(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_history(cvk_background_subtractor_mog2_t *h, int history);
int cvk_background_subtractor_mog2_get_n_mixtures(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_n_mixtures(cvk_background_subtractor_mog2_t *h,
                                                   int n_mixtures);
double cvk_background_subtractor_mog2_get_background_ratio(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_background_ratio(cvk_background_subtractor_mog2_t *h,
                                                         double background_ratio);
double cvk_background_subtractor_mog2_get_var_threshold(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_var_threshold(cvk_background_subtractor_mog2_t *h,
                                                      double var_threshold);
double cvk_background_subtractor_mog2_get_var_threshold_gen(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_var_threshold_gen(cvk_background_subtractor_mog2_t *h,
                                                          double var_threshold_gen);
double cvk_background_subtractor_mog2_get_var_init(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_var_init(cvk_background_subtractor_mog2_t *h,
                                                 double var_init);
double cvk_background_subtractor_mog2_get_var_min(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_var_min(cvk_background_subtractor_mog2_t *h,
                                                double var_min);
double cvk_background_subtractor_mog2_get_var_max(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_var_max(cvk_background_subtractor_mog2_t *h,
                                                double var_max);
double cvk_background_subtractor_mog2_get_complexity_reduction_threshold(
    const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_complexity_reduction_threshold(
    cvk_background_subtractor_mog2_t *h, double complexity_reduction_threshold);
int cvk_background_subtractor_mog2_get_detect_shadows(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_detect_shadows(cvk_background_subtractor_mog2_t *h,
                                                       int detect_shadows);
int cvk_background_subtractor_mog2_get_shadow_value(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_shadow_value(cvk_background_subtractor_mog2_t *h,
                                                     int shadow_value);
double cvk_background_subtractor_mog2_get_shadow_threshold(const cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_set_shadow_threshold(cvk_background_subtractor_mog2_t *h,
                                                         double shadow_threshold);

/* ---- FarnebackOpticalFlow (cv::FarnebackOpticalFlow) -------------------- */

cvk_farneback_optical_flow_t *cvk_farneback_optical_flow_create(
    int num_levels, double pyr_scale, int fast_pyramids, int win_size, int num_iters,
    int poly_n, double poly_sigma, int flags);

void cvk_farneback_optical_flow_calc(cvk_farneback_optical_flow_t *h,
                                     const cvk_mat_t *i0, const cvk_mat_t *i1,
                                     cvk_mat_t *flow);
void cvk_farneback_optical_flow_collect_garbage(cvk_farneback_optical_flow_t *h);

int cvk_farneback_optical_flow_get_num_levels(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_num_levels(cvk_farneback_optical_flow_t *h, int num_levels);
double cvk_farneback_optical_flow_get_pyr_scale(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_pyr_scale(cvk_farneback_optical_flow_t *h, double pyr_scale);
int cvk_farneback_optical_flow_get_fast_pyramids(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_fast_pyramids(cvk_farneback_optical_flow_t *h,
                                                  int fast_pyramids);
int cvk_farneback_optical_flow_get_win_size(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_win_size(cvk_farneback_optical_flow_t *h, int win_size);
int cvk_farneback_optical_flow_get_num_iters(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_num_iters(cvk_farneback_optical_flow_t *h, int num_iters);
int cvk_farneback_optical_flow_get_poly_n(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_poly_n(cvk_farneback_optical_flow_t *h, int poly_n);
double cvk_farneback_optical_flow_get_poly_sigma(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_poly_sigma(cvk_farneback_optical_flow_t *h, double poly_sigma);
int cvk_farneback_optical_flow_get_flags(const cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_set_flags(cvk_farneback_optical_flow_t *h, int flags);

/* ---- DISOpticalFlow (cv::DISOpticalFlow) -------------------------------- */

cvk_dis_optical_flow_t *cvk_dis_optical_flow_create(int preset);

void cvk_dis_optical_flow_calc(cvk_dis_optical_flow_t *h,
                               const cvk_mat_t *i0, const cvk_mat_t *i1, cvk_mat_t *flow);
void cvk_dis_optical_flow_collect_garbage(cvk_dis_optical_flow_t *h);

int cvk_dis_optical_flow_get_finest_scale(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_finest_scale(cvk_dis_optical_flow_t *h, int val);
int cvk_dis_optical_flow_get_coarsest_scale(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_coarsest_scale(cvk_dis_optical_flow_t *h, int val);
int cvk_dis_optical_flow_get_patch_size(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_patch_size(cvk_dis_optical_flow_t *h, int val);
int cvk_dis_optical_flow_get_patch_stride(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_patch_stride(cvk_dis_optical_flow_t *h, int val);
int cvk_dis_optical_flow_get_gradient_descent_iterations(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_gradient_descent_iterations(cvk_dis_optical_flow_t *h, int val);
int cvk_dis_optical_flow_get_variational_refinement_iterations(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_variational_refinement_iterations(cvk_dis_optical_flow_t *h, int val);
float cvk_dis_optical_flow_get_variational_refinement_alpha(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_variational_refinement_alpha(cvk_dis_optical_flow_t *h, float val);
float cvk_dis_optical_flow_get_variational_refinement_delta(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_variational_refinement_delta(cvk_dis_optical_flow_t *h, float val);
float cvk_dis_optical_flow_get_variational_refinement_gamma(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_variational_refinement_gamma(cvk_dis_optical_flow_t *h, float val);
float cvk_dis_optical_flow_get_variational_refinement_epsilon(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_variational_refinement_epsilon(cvk_dis_optical_flow_t *h, float val);
int cvk_dis_optical_flow_get_use_mean_normalization(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_use_mean_normalization(cvk_dis_optical_flow_t *h, int val);
int cvk_dis_optical_flow_get_use_spatial_propagation(const cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_set_use_spatial_propagation(cvk_dis_optical_flow_t *h, int val);

/* ---- SparsePyrLKOpticalFlow (cv::SparsePyrLKOpticalFlow) ---------------- */

cvk_sparse_pyr_lk_optical_flow_t *cvk_sparse_pyr_lk_optical_flow_create(
    int win_w, int win_h, int max_level, int tc_type, int tc_max_count, double tc_epsilon,
    int flags, double min_eig_threshold);

/** Writes next_pts/status in place; `err` may be NULL. */
void cvk_sparse_pyr_lk_optical_flow_calc(cvk_sparse_pyr_lk_optical_flow_t *h,
                                         const cvk_mat_t *prev_img, const cvk_mat_t *next_img,
                                         const cvk_mat_t *prev_pts, cvk_mat_t *next_pts,
                                         cvk_mat_t *status, cvk_mat_t *err);

int cvk_sparse_pyr_lk_optical_flow_get_win_w(const cvk_sparse_pyr_lk_optical_flow_t *h);
int cvk_sparse_pyr_lk_optical_flow_get_win_h(const cvk_sparse_pyr_lk_optical_flow_t *h);
void cvk_sparse_pyr_lk_optical_flow_set_win_size(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                 int win_w, int win_h);
int cvk_sparse_pyr_lk_optical_flow_get_max_level(const cvk_sparse_pyr_lk_optical_flow_t *h);
void cvk_sparse_pyr_lk_optical_flow_set_max_level(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                  int max_level);
void cvk_sparse_pyr_lk_optical_flow_get_term_criteria(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                      int *tc_type, int *tc_max_count,
                                                      double *tc_epsilon);
void cvk_sparse_pyr_lk_optical_flow_set_term_criteria(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                      int tc_type, int tc_max_count,
                                                      double tc_epsilon);
int cvk_sparse_pyr_lk_optical_flow_get_flags(const cvk_sparse_pyr_lk_optical_flow_t *h);
void cvk_sparse_pyr_lk_optical_flow_set_flags(cvk_sparse_pyr_lk_optical_flow_t *h, int flags);
double cvk_sparse_pyr_lk_optical_flow_get_min_eig_threshold(
    const cvk_sparse_pyr_lk_optical_flow_t *h);
void cvk_sparse_pyr_lk_optical_flow_set_min_eig_threshold(cvk_sparse_pyr_lk_optical_flow_t *h,
                                                          double min_eig_threshold);

/* ---- VariationalRefinement (cv::VariationalRefinement) ------------------ */

cvk_variational_refinement_t *cvk_variational_refinement_create(void);

void cvk_variational_refinement_calc(cvk_variational_refinement_t *h,
                                     const cvk_mat_t *i0, const cvk_mat_t *i1,
                                     cvk_mat_t *flow);
void cvk_variational_refinement_calc_uv(cvk_variational_refinement_t *h,
                                        const cvk_mat_t *i0, const cvk_mat_t *i1,
                                        cvk_mat_t *flow_u, cvk_mat_t *flow_v);
void cvk_variational_refinement_collect_garbage(cvk_variational_refinement_t *h);

int cvk_variational_refinement_get_fixed_point_iterations(const cvk_variational_refinement_t *h);
void cvk_variational_refinement_set_fixed_point_iterations(cvk_variational_refinement_t *h, int val);
int cvk_variational_refinement_get_sor_iterations(const cvk_variational_refinement_t *h);
void cvk_variational_refinement_set_sor_iterations(cvk_variational_refinement_t *h, int val);
float cvk_variational_refinement_get_omega(const cvk_variational_refinement_t *h);
void cvk_variational_refinement_set_omega(cvk_variational_refinement_t *h, float val);
float cvk_variational_refinement_get_alpha(const cvk_variational_refinement_t *h);
void cvk_variational_refinement_set_alpha(cvk_variational_refinement_t *h, float val);
float cvk_variational_refinement_get_delta(const cvk_variational_refinement_t *h);
void cvk_variational_refinement_set_delta(cvk_variational_refinement_t *h, float val);
float cvk_variational_refinement_get_gamma(const cvk_variational_refinement_t *h);
void cvk_variational_refinement_set_gamma(cvk_variational_refinement_t *h, float val);
float cvk_variational_refinement_get_epsilon(const cvk_variational_refinement_t *h);
void cvk_variational_refinement_set_epsilon(cvk_variational_refinement_t *h, float val);

/* ---- Algorithm interface (clear/empty/save/getDefaultName) -------------- */

void cvk_background_subtractor_knn_clear(cvk_background_subtractor_knn_t *h);
int cvk_background_subtractor_knn_empty(cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_knn_save(cvk_background_subtractor_knn_t *h, const char *filename);
const char *cvk_background_subtractor_knn_get_default_name(cvk_background_subtractor_knn_t *h);

void cvk_background_subtractor_mog2_clear(cvk_background_subtractor_mog2_t *h);
int cvk_background_subtractor_mog2_empty(cvk_background_subtractor_mog2_t *h);
void cvk_background_subtractor_mog2_save(cvk_background_subtractor_mog2_t *h, const char *filename);
const char *cvk_background_subtractor_mog2_get_default_name(cvk_background_subtractor_mog2_t *h);

void cvk_farneback_optical_flow_clear(cvk_farneback_optical_flow_t *h);
int cvk_farneback_optical_flow_empty(cvk_farneback_optical_flow_t *h);
void cvk_farneback_optical_flow_save(cvk_farneback_optical_flow_t *h, const char *filename);
const char *cvk_farneback_optical_flow_get_default_name(cvk_farneback_optical_flow_t *h);

void cvk_dis_optical_flow_clear(cvk_dis_optical_flow_t *h);
int cvk_dis_optical_flow_empty(cvk_dis_optical_flow_t *h);
void cvk_dis_optical_flow_save(cvk_dis_optical_flow_t *h, const char *filename);
const char *cvk_dis_optical_flow_get_default_name(cvk_dis_optical_flow_t *h);

void cvk_sparse_pyr_lk_optical_flow_clear(cvk_sparse_pyr_lk_optical_flow_t *h);
int cvk_sparse_pyr_lk_optical_flow_empty(cvk_sparse_pyr_lk_optical_flow_t *h);
void cvk_sparse_pyr_lk_optical_flow_save(cvk_sparse_pyr_lk_optical_flow_t *h, const char *filename);
const char *cvk_sparse_pyr_lk_optical_flow_get_default_name(cvk_sparse_pyr_lk_optical_flow_t *h);

void cvk_variational_refinement_clear(cvk_variational_refinement_t *h);
int cvk_variational_refinement_empty(cvk_variational_refinement_t *h);
void cvk_variational_refinement_save(cvk_variational_refinement_t *h, const char *filename);
const char *cvk_variational_refinement_get_default_name(cvk_variational_refinement_t *h);

/* ---- release ----------------------------------------------------------- */

void cvk_background_subtractor_knn_release(cvk_background_subtractor_knn_t *h);
void cvk_background_subtractor_mog2_release(cvk_background_subtractor_mog2_t *h);
void cvk_farneback_optical_flow_release(cvk_farneback_optical_flow_t *h);
void cvk_dis_optical_flow_release(cvk_dis_optical_flow_t *h);
void cvk_sparse_pyr_lk_optical_flow_release(cvk_sparse_pyr_lk_optical_flow_t *h);
void cvk_variational_refinement_release(cvk_variational_refinement_t *h);

/* ---- KalmanFilter (cv::KalmanFilter, plain object, not a Ptr) ----------- */

cvk_kalman_filter_t *cvk_kalman_filter_create(int dynam_params, int measure_params,
                                              int control_params, int type);

/** cv::KalmanFilter::predict; `control` may be NULL. Returns a new Mat. */
cvk_mat_t *cvk_kalman_filter_predict(cvk_kalman_filter_t *h, const cvk_mat_t *control);

/** cv::KalmanFilter::correct. Returns a new Mat. */
cvk_mat_t *cvk_kalman_filter_correct(cvk_kalman_filter_t *h, const cvk_mat_t *measurement);

/* state matrix getters return a new Mat sharing the filter's data; setters
 * assign the matrix (refcounted shallow copy, like cv::Mat assignment). */

cvk_mat_t *cvk_kalman_filter_get_state_pre(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_state_pre(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_state_post(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_state_post(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_transition_matrix(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_transition_matrix(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_control_matrix(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_control_matrix(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_measurement_matrix(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_measurement_matrix(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_process_noise_cov(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_process_noise_cov(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_measurement_noise_cov(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_measurement_noise_cov(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_error_cov_pre(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_error_cov_pre(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_gain(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_gain(cvk_kalman_filter_t *h, const cvk_mat_t *mat);
cvk_mat_t *cvk_kalman_filter_get_error_cov_post(const cvk_kalman_filter_t *h);
void cvk_kalman_filter_set_error_cov_post(cvk_kalman_filter_t *h, const cvk_mat_t *mat);

void cvk_kalman_filter_release(cvk_kalman_filter_t *h);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_VIDEO_H */

/*
 * cvk_ C ABI declarations for the OpenCV "core" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 */
#ifndef OPENCV_KMP_CORE_H
#define OPENCV_KMP_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

/** Message of the last failure recorded by this module, or NULL. */
const char *cvk_core_last_error(void);

/* ---- scalar math (Core statics) -------------------------------------- */

/** \f$\sqrt[3]{val}\f$ with correct sign for negative arguments. */
float cvk_cube_root(float val);

/** Fast arctangent of (y, x) in degrees; result in [-180, 180]. */
float cvk_fast_atan2(float y, float x);

/** Computes the source coordinate for a mirrored border access. */
int cvk_border_interpolate(int p, int len, int border_type);

/* ---- RNG (cv::RNG wrapper) ------------------------------------------- */

typedef struct cvk_rng cvk_rng_t;

/** Owned copy of the thread-local default generator (cv::theRNG state). */
cvk_rng_t *cvk_rng_from_global(void);

/** New generator seeded with `seed` (0 selects the default seed). */
cvk_rng_t *cvk_rng_create(unsigned long long seed);

/** Next raw 32-bit random word (MWC algorithm). */
unsigned int cvk_rng_next(cvk_rng_t *rng);

/** Uniform integer from [a, b). */
int cvk_rng_uniform_int(cvk_rng_t *rng, int a, int b);

/** Uniform float from [a, b). */
float cvk_rng_uniform_float(cvk_rng_t *rng, float a, float b);

/** Uniform double from [a, b). */
double cvk_rng_uniform_double(cvk_rng_t *rng, double a, double b);

/** Gaussian N(0, sigma) sample. */
double cvk_rng_gaussian(cvk_rng_t *rng, double sigma);

/** Frees a cvk_rng_create/cvk_rng_from_global handle (NULL tolerated). */
void cvk_rng_release(cvk_rng_t *rng);

/* ---- array operations (Core statics) ---------------------------------- */

/**
 * Copies selected channels between the matrices in `srcs` and `dsts`
 * (cv::mixChannels). `from_to` holds [srcIdx, dstIdx] channel pairs;
 * dst matrices must be pre-allocated with the source size/depth.
 */
void cvk_mix_channels(cvk_mat_t **srcs, int nsrcs, cvk_mat_t **dsts,
                      int ndsts, const int *from_to, size_t from_to_len);

/**
 * Naive nearest-neighbor distance matrix (cv::batchDistance). `dist` is
 * pre-allocated nsrc1 x nsrc2 of `dtype`; `nidx` is nsrc1 x 1 CV_32S.
 */
int cvk_batch_distance(const cvk_mat_t *src1, const cvk_mat_t *src2,
                       cvk_mat_t *dist, int dtype, cvk_mat_t *nidx,
                       int norm_type, int k, const cvk_mat_t *mask,
                       int update, int crosscheck);

/** Covariance matrix and mean of a sample set (cv::calcCovarMatrix). */
void cvk_calc_covar_matrix(const cvk_mat_t *samples, cvk_mat_t *covar,
                           cvk_mat_t *mean, int flags, int ctype);

/** Mirrors one triangle of a square matrix onto the other, in place. */
void cvk_complete_symm(cvk_mat_t *m, int lower_to_upper);

/** Real roots of a cubic; returns the root count, roots as a CV_32F Mat. */
int cvk_solve_cubic(const cvk_mat_t *coeffs, cvk_mat_t **roots);

/** Real/complex roots of a polynomial; returns the solution accuracy. */
double cvk_solve_poly(const cvk_mat_t *coeffs, cvk_mat_t **roots,
                      int max_iters);

/** dst = scale * (src-delta)^T * (src-delta) (aTa) or the reverse. */
cvk_mat_t *cvk_mul_transposed(const cvk_mat_t *src, int a_ta,
                              const cvk_mat_t *delta, double scale,
                              int dtype);

/** Flips an N-dimensional array along one axis (cv::flipND). */
cvk_mat_t *cvk_flip_nd(const cvk_mat_t *src, int axis);

/** Broadcasts src to the shape stored in `shape` (CV_32SC1 row). */
cvk_mat_t *cvk_broadcast(const cvk_mat_t *src, const cvk_mat_t *shape);

/** Permutes N-dimensional axes; `order` is a length-dims permutation. */
cvk_mat_t *cvk_transpose_nd(const cvk_mat_t *src, const int *order,
                            size_t order_len);

/** dst = src where mask is non-zero, 0 elsewhere (fresh dst). */
cvk_mat_t *cvk_copy_to(const cvk_mat_t *src, const cvk_mat_t *mask);

/** dst = a*alpha + b (cv::scaleAdd). */
cvk_mat_t *cvk_scale_add(const cvk_mat_t *a, double alpha, const cvk_mat_t *b);

/** Generalized matrix product with GemmFlags (c may be NULL). */
cvk_mat_t *cvk_gemm_flags(const cvk_mat_t *a, const cvk_mat_t *b,
                          double alpha, const cvk_mat_t *c, double gamma,
                          int flags);

/** Eigen decomposition of a general (non-symmetric) square matrix. */
void cvk_eigen_non_symmetric(const cvk_mat_t *src, cvk_mat_t **eigenvalues,
                             cvk_mat_t **eigenvectors);

/** 255 where all channels are finite, else 0 (CV_8UC1 result). */
cvk_mat_t *cvk_finite_mask(const cvk_mat_t *src);

/* ---- masked statistics (Core statics with mask) ----------------------- */

/** minVal,maxVal,minX,minY,maxX,maxY of the masked region into out6. */
void cvk_mat_min_max_loc_masked(const cvk_mat_t *src, const cvk_mat_t *mask,
                                double *out6);

/** Mean over the masked region (cv::mean with mask). */
cvk_scalar_t cvk_mat_mean_masked(const cvk_mat_t *src, const cvk_mat_t *mask);

/** Norm of src restricted to the masked region. */
double cvk_norm_masked(const cvk_mat_t *src, int norm_type,
                       const cvk_mat_t *mask);

/** Difference norm of a and b restricted to the masked region. */
double cvk_norm_diff_masked(const cvk_mat_t *a, const cvk_mat_t *b,
                            int norm_type, const cvk_mat_t *mask);

/* ---- range check / shuffle -------------------------------------------- */

/** True when every element is finite and within [min_val, max_val]. */
int cvk_check_range(const cvk_mat_t *a, int quiet, double min_val,
                    double max_val);

/** Randomly shuffles the elements of a 1D array, in place. */
void cvk_rand_shuffle(cvk_mat_t *dst, double iter_factor);

/* ---- environment / runtime info --------------------------------------- */

long long cvk_get_tick_count(void);
double cvk_get_tick_frequency(void);
int cvk_get_number_of_cpus(void);
int cvk_check_hardware_support(int feature);
const char *cvk_get_hardware_feature_name(int feature);
const char *cvk_get_version_string(void);
int cvk_get_version_major(void);
int cvk_get_version_minor(void);
int cvk_get_version_revision(void);
long long cvk_get_cpu_tick_count(void);
int cvk_get_thread_num(void);
int cvk_get_default_algorithm_hint(void);
int cvk_use_optimized(void);
void cvk_set_use_optimized(int onoff);
const char *cvk_get_cpu_features_line(void);
int cvk_use_ipp(void);
void cvk_set_use_ipp(int flag);
const char *cvk_get_ipp_version(void);
int cvk_use_ipp_not_exact(void);
void cvk_set_use_ipp_not_exact(int flag);
const char *cvk_find_file(const char *relative_path, int required,
                          int silent_mode);
const char *cvk_find_file_or_keep(const char *relative_path, int silent_mode);
void cvk_add_samples_data_search_path(const char *path);
void cvk_add_samples_data_search_sub_directory(const char *subdir);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_CORE_H */

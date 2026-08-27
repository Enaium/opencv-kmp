/*
 * cvk_ C ABI declarations for the OpenCV "features" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * Every detector owns an opaque handle wrapping a cv::Ptr of the concrete
 * algorithm type. The Feature2D surface (detect/compute/detectAndCompute,
 * descriptorSize/Type/defaultNorm, write/read) plus the Algorithm quartet
 * (clear/empty/save/getDefaultName) is declared once per concrete class;
 * the shim generates the bodies with macros.
 *
 * KeyPoint collections travel as CV_32FC(7) Mats (one row per keypoint:
 * x, y, size, angle, response, octave, classId), the MatOfKeyPoint wire
 * format. MSER contours / blob contours travel as the shared contour flat
 * buffer (see cvk_find_contours); float vectors (AffineFeature view
 * params) travel as CV_32FC1 Mats.
 */
#ifndef OPENCV_KMP_FEATURES_H
#define OPENCV_KMP_FEATURES_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- opaque handles --------------------------------------------------- */

/** Generic Feature2D view; layout-compatible prefix of every concrete handle. */
typedef struct cvk_feature2d cvk_feature2d_t;
typedef struct cvk_sift cvk_sift_t;
typedef struct cvk_orb cvk_orb_t;
typedef struct cvk_mser cvk_mser_t;
typedef struct cvk_fast_feature_detector cvk_fast_feature_detector_t;
typedef struct cvk_gftt_detector cvk_gftt_detector_t;
typedef struct cvk_simple_blob_detector cvk_simple_blob_detector_t;
typedef struct cvk_affine cvk_affine_t;

/* =========================================================================
 * Shared Feature2D surface (declared per concrete class via this macro)
 *
 * detect:        returns a new CV_32FC(7) keypoint Mat (NULL on failure).
 * compute:       takes the input keypoints Mat (CV_IN_OUT); writes new
 *                keypoints/descriptors Mats into the out params; returns 0/1.
 * detectAndCompute: returns a new keypoints Mat, writes the descriptors Mat
 *                into *descriptors_out.
 * ========================================================================= */

#define CVK_FEATURES_FEATURE2D_DECLS(T)                                       \
    cvk_mat_t *cvk_##T##_detect(const cvk_##T##_t *h, const cvk_mat_t *image, \
                                const cvk_mat_t *mask);                       \
    int cvk_##T##_compute(const cvk_##T##_t *h, const cvk_mat_t *image,       \
                          const cvk_mat_t *keypoints,                         \
                          cvk_mat_t **keypoints_out,                          \
                          cvk_mat_t **descriptors_out);                       \
    cvk_mat_t *cvk_##T##_detect_and_compute(const cvk_##T##_t *h,             \
                                            const cvk_mat_t *image,           \
                                            const cvk_mat_t *mask,            \
                                            cvk_mat_t **descriptors_out);     \
    int cvk_##T##_descriptor_size(const cvk_##T##_t *h);                      \
    int cvk_##T##_descriptor_type(const cvk_##T##_t *h);                      \
    int cvk_##T##_default_norm(const cvk_##T##_t *h);                         \
    void cvk_##T##_write(const cvk_##T##_t *h, const char *filename);         \
    void cvk_##T##_read(const cvk_##T##_t *h, const char *filename)

/* =========================================================================
 * SIFT
 * ========================================================================= */

/**
 * cv::SIFT::create. descriptor_type < 0 selects the default descriptor
 * (CV_32F); otherwise it must be CV_32F or CV_8U.
 */
cvk_sift_t *cvk_sift_create(int nfeatures, int n_octave_layers,
                            double contrast_threshold, double edge_threshold,
                            double sigma, int descriptor_type,
                            int enable_precise_upscale);

CVK_FEATURES_FEATURE2D_DECLS(sift);

void cvk_sift_clear(cvk_sift_t *h);
int cvk_sift_empty(cvk_sift_t *h);
void cvk_sift_save(cvk_sift_t *h, const char *filename);
const char *cvk_sift_get_default_name(cvk_sift_t *h);

void cvk_sift_set_n_features(cvk_sift_t *h, int max_features);
int cvk_sift_get_n_features(const cvk_sift_t *h);
void cvk_sift_set_n_octave_layers(cvk_sift_t *h, int n_octave_layers);
int cvk_sift_get_n_octave_layers(const cvk_sift_t *h);
void cvk_sift_set_contrast_threshold(cvk_sift_t *h, double contrast_threshold);
double cvk_sift_get_contrast_threshold(const cvk_sift_t *h);
void cvk_sift_set_edge_threshold(cvk_sift_t *h, double edge_threshold);
double cvk_sift_get_edge_threshold(const cvk_sift_t *h);
void cvk_sift_set_sigma(cvk_sift_t *h, double sigma);
double cvk_sift_get_sigma(const cvk_sift_t *h);

void cvk_sift_release(cvk_sift_t *h);

/* =========================================================================
 * ORB
 * ========================================================================= */

/** cv::ORB::create; score_type is ORB::ScoreType (0=HARRIS, 1=FAST). */
cvk_orb_t *cvk_orb_create(int nfeatures, float scale_factor, int nlevels,
                          int edge_threshold, int first_level, int wta_k,
                          int score_type, int patch_size, int fast_threshold);

CVK_FEATURES_FEATURE2D_DECLS(orb);

void cvk_orb_clear(cvk_orb_t *h);
int cvk_orb_empty(cvk_orb_t *h);
void cvk_orb_save(cvk_orb_t *h, const char *filename);
const char *cvk_orb_get_default_name(cvk_orb_t *h);

void cvk_orb_set_max_features(cvk_orb_t *h, int max_features);
int cvk_orb_get_max_features(const cvk_orb_t *h);
void cvk_orb_set_scale_factor(cvk_orb_t *h, double scale_factor);
double cvk_orb_get_scale_factor(const cvk_orb_t *h);
void cvk_orb_set_n_levels(cvk_orb_t *h, int nlevels);
int cvk_orb_get_n_levels(const cvk_orb_t *h);
void cvk_orb_set_edge_threshold(cvk_orb_t *h, int edge_threshold);
int cvk_orb_get_edge_threshold(const cvk_orb_t *h);
void cvk_orb_set_first_level(cvk_orb_t *h, int first_level);
int cvk_orb_get_first_level(const cvk_orb_t *h);
void cvk_orb_set_wta_k(cvk_orb_t *h, int wta_k);
int cvk_orb_get_wta_k(const cvk_orb_t *h);
void cvk_orb_set_score_type(cvk_orb_t *h, int score_type);
int cvk_orb_get_score_type(const cvk_orb_t *h);
void cvk_orb_set_patch_size(cvk_orb_t *h, int patch_size);
int cvk_orb_get_patch_size(const cvk_orb_t *h);
void cvk_orb_set_fast_threshold(cvk_orb_t *h, int fast_threshold);
int cvk_orb_get_fast_threshold(const cvk_orb_t *h);

void cvk_orb_release(cvk_orb_t *h);

/* =========================================================================
 * MSER
 * ========================================================================= */

/** cv::MSER::create with the full parameter set. */
cvk_mser_t *cvk_mser_create(int delta, int min_area, int max_area,
                            double max_variation, double min_diversity,
                            int max_evolution, double area_threshold,
                            double min_margin, int edge_blur_size);

CVK_FEATURES_FEATURE2D_DECLS(mser);

void cvk_mser_clear(cvk_mser_t *h);
int cvk_mser_empty(cvk_mser_t *h);
void cvk_mser_save(cvk_mser_t *h, const char *filename);
const char *cvk_mser_get_default_name(cvk_mser_t *h);

/**
 * cv::MSER::detectRegions. Writes the region point-sets into a malloc'd
 * contour flat buffer (freed with cvk_free_buffer; length in *out_len) and
 * a new Nx1 CV_32SC4 bounding-box Mat into *bboxes.
 */
unsigned char *cvk_mser_detect_regions(const cvk_mser_t *h,
                                       const cvk_mat_t *image,
                                       cvk_mat_t **bboxes, size_t *out_len);

void cvk_mser_set_delta(cvk_mser_t *h, int delta);
int cvk_mser_get_delta(const cvk_mser_t *h);
void cvk_mser_set_min_area(cvk_mser_t *h, int min_area);
int cvk_mser_get_min_area(const cvk_mser_t *h);
void cvk_mser_set_max_area(cvk_mser_t *h, int max_area);
int cvk_mser_get_max_area(const cvk_mser_t *h);
void cvk_mser_set_max_variation(cvk_mser_t *h, double max_variation);
double cvk_mser_get_max_variation(const cvk_mser_t *h);
void cvk_mser_set_min_diversity(cvk_mser_t *h, double min_diversity);
double cvk_mser_get_min_diversity(const cvk_mser_t *h);
void cvk_mser_set_max_evolution(cvk_mser_t *h, int max_evolution);
int cvk_mser_get_max_evolution(const cvk_mser_t *h);
void cvk_mser_set_area_threshold(cvk_mser_t *h, double area_threshold);
double cvk_mser_get_area_threshold(const cvk_mser_t *h);
void cvk_mser_set_min_margin(cvk_mser_t *h, double min_margin);
double cvk_mser_get_min_margin(const cvk_mser_t *h);
void cvk_mser_set_edge_blur_size(cvk_mser_t *h, int edge_blur_size);
int cvk_mser_get_edge_blur_size(const cvk_mser_t *h);
void cvk_mser_set_pass2_only(cvk_mser_t *h, int f);
int cvk_mser_get_pass2_only(const cvk_mser_t *h);

void cvk_mser_release(cvk_mser_t *h);

/* =========================================================================
 * FastFeatureDetector
 * ========================================================================= */

/** cv::FastFeatureDetector::create; type is a DetectorType value. */
cvk_fast_feature_detector_t *cvk_fast_feature_detector_create(
    int threshold, int nonmax_suppression, int type);

CVK_FEATURES_FEATURE2D_DECLS(fast_feature_detector);

void cvk_fast_feature_detector_clear(cvk_fast_feature_detector_t *h);
int cvk_fast_feature_detector_empty(cvk_fast_feature_detector_t *h);
void cvk_fast_feature_detector_save(cvk_fast_feature_detector_t *h,
                                    const char *filename);
const char *cvk_fast_feature_detector_get_default_name(
    cvk_fast_feature_detector_t *h);

void cvk_fast_feature_detector_set_threshold(cvk_fast_feature_detector_t *h,
                                             int threshold);
int cvk_fast_feature_detector_get_threshold(const cvk_fast_feature_detector_t *h);
void cvk_fast_feature_detector_set_nonmax_suppression(
    cvk_fast_feature_detector_t *h, int f);
int cvk_fast_feature_detector_get_nonmax_suppression(
    const cvk_fast_feature_detector_t *h);
void cvk_fast_feature_detector_set_type(cvk_fast_feature_detector_t *h, int type);
int cvk_fast_feature_detector_get_type(const cvk_fast_feature_detector_t *h);

void cvk_fast_feature_detector_release(cvk_fast_feature_detector_t *h);

/* =========================================================================
 * GFTTDetector
 * ========================================================================= */

/**
 * cv::GFTTDetector::create. gradient_size < 0 selects the classic
 * (blockSize-only) overload; otherwise the Sobel-gradient overload is used.
 */
cvk_gftt_detector_t *cvk_gftt_detector_create(
    int max_corners, double quality_level, double min_distance, int block_size,
    int gradient_size, int use_harris_detector, double k);

CVK_FEATURES_FEATURE2D_DECLS(gftt_detector);

void cvk_gftt_detector_clear(cvk_gftt_detector_t *h);
int cvk_gftt_detector_empty(cvk_gftt_detector_t *h);
void cvk_gftt_detector_save(cvk_gftt_detector_t *h, const char *filename);
const char *cvk_gftt_detector_get_default_name(cvk_gftt_detector_t *h);

void cvk_gftt_detector_set_max_features(cvk_gftt_detector_t *h, int max_features);
int cvk_gftt_detector_get_max_features(const cvk_gftt_detector_t *h);
void cvk_gftt_detector_set_quality_level(cvk_gftt_detector_t *h, double qlevel);
double cvk_gftt_detector_get_quality_level(const cvk_gftt_detector_t *h);
void cvk_gftt_detector_set_min_distance(cvk_gftt_detector_t *h, double min_distance);
double cvk_gftt_detector_get_min_distance(const cvk_gftt_detector_t *h);
void cvk_gftt_detector_set_block_size(cvk_gftt_detector_t *h, int block_size);
int cvk_gftt_detector_get_block_size(const cvk_gftt_detector_t *h);
void cvk_gftt_detector_set_gradient_size(cvk_gftt_detector_t *h, int gradient_size);
int cvk_gftt_detector_get_gradient_size(const cvk_gftt_detector_t *h);
void cvk_gftt_detector_set_harris_detector(cvk_gftt_detector_t *h, int val);
int cvk_gftt_detector_get_harris_detector(const cvk_gftt_detector_t *h);
void cvk_gftt_detector_set_k(cvk_gftt_detector_t *h, double k);
double cvk_gftt_detector_get_k(const cvk_gftt_detector_t *h);

void cvk_gftt_detector_release(cvk_gftt_detector_t *h);

/* =========================================================================
 * SimpleBlobDetector
 * ========================================================================= */

/**
 * cv::SimpleBlobDetector::create with every Params field expanded as a
 * scalar argument (the Kotlin Params data class is pure, no native object).
 * Boolean fields are 0/1 ints, blob_color is a 0..255 int.
 */
cvk_simple_blob_detector_t *cvk_simple_blob_detector_create(
    float threshold_step, float min_threshold, float max_threshold,
    long long min_repeatability, float min_dist_between_blobs,
    int filter_by_color, int blob_color,
    int filter_by_area, float min_area, float max_area,
    int filter_by_circularity, float min_circularity, float max_circularity,
    int filter_by_inertia, float min_inertia_ratio, float max_inertia_ratio,
    int filter_by_convexity, float min_convexity, float max_convexity,
    int collect_contours);

CVK_FEATURES_FEATURE2D_DECLS(simple_blob_detector);

void cvk_simple_blob_detector_clear(cvk_simple_blob_detector_t *h);
int cvk_simple_blob_detector_empty(cvk_simple_blob_detector_t *h);
void cvk_simple_blob_detector_save(cvk_simple_blob_detector_t *h,
                                   const char *filename);
const char *cvk_simple_blob_detector_get_default_name(
    cvk_simple_blob_detector_t *h);

/** Replaces the params (same expanded-scalar layout as the create above). */
int cvk_simple_blob_detector_set_params(cvk_simple_blob_detector_t *h,
                                        float threshold_step,
                                        float min_threshold, float max_threshold,
                                        long long min_repeatability,
                                        float min_dist_between_blobs,
                                        int filter_by_color, int blob_color,
                                        int filter_by_area, float min_area,
                                        float max_area, int filter_by_circularity,
                                        float min_circularity,
                                        float max_circularity,
                                        int filter_by_inertia,
                                        float min_inertia_ratio,
                                        float max_inertia_ratio,
                                        int filter_by_convexity,
                                        float min_convexity,
                                        float max_convexity,
                                        int collect_contours);

/** Writes the 20 Params fields (bools as 0/1) into out[20]. */
int cvk_simple_blob_detector_get_params(const cvk_simple_blob_detector_t *h,
                                        double *out20);

/**
 * Returns the contours of the blobs detected by the last detect() call as a
 * malloc'd contour flat buffer (cvk_free_buffer, length in *out_len).
 */
unsigned char *cvk_simple_blob_detector_get_blob_contours(
    const cvk_simple_blob_detector_t *h, size_t *out_len);

void cvk_simple_blob_detector_release(cvk_simple_blob_detector_t *h);

/* =========================================================================
 * AffineFeature
 * ========================================================================= */

/**
 * cv::AffineFeature::create wrapping a Feature2D backend. `backend` is any
 * concrete detector handle viewed through the cvk_feature2d_t prefix.
 */
cvk_affine_t *cvk_affine_create(const cvk_feature2d_t *backend, int max_tilt,
                                int min_tilt, float tilt_step,
                                float rotate_step_base);

CVK_FEATURES_FEATURE2D_DECLS(affine);

void cvk_affine_clear(cvk_affine_t *h);
int cvk_affine_empty(cvk_affine_t *h);
void cvk_affine_save(cvk_affine_t *h, const char *filename);
const char *cvk_affine_get_default_name(cvk_affine_t *h);

/** tilts/rolls are CV_32FC1 Mats (vector<float>). */
int cvk_affine_set_view_params(const cvk_affine_t *h, const cvk_mat_t *tilts,
                               const cvk_mat_t *rolls);
int cvk_affine_get_view_params(const cvk_affine_t *h, cvk_mat_t **tilts_out,
                               cvk_mat_t **rolls_out);

void cvk_affine_release(cvk_affine_t *h);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_FEATURES_H */

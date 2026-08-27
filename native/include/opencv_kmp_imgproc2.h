/*
 * cvk_ C ABI declarations for opencv-kmp slice "imgproc2".
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here back both Kotlin/Native (cinterop) and the
 * JVM (JNI).
 *
 * Object wrappers ported in this slice:
 *   - cv::LineSegmentDetector (cvk_line_segment_detector_*)
 *   - cv::GeneralizedHoughBallard (cvk_gh_ballard_*)
 *   - cv::GeneralizedHoughGuil   (cvk_gh_guil_*)
 *
 * The two GeneralizedHough variants get their own handle type and function
 * namespace (no discriminator parameter): they are distinct C++ classes
 * with variant-specific setters, and per-variant names keep the C ABI
 * type-checkable. The five canny/minDist/dp/maxBufferSize fields common to
 * the base class are exposed per variant as cvk_gh_ballard_set_* and
 * cvk_gh_guil_set_*.
 *
 * Mat outputs follow the slice convention: the primary result is returned
 * as a new caller-owned cvk_mat_t*, extra outputs come back through
 * out-params (each caller-owned when non-NULL). Every function is noexcept;
 * failures report through cvk_last_error() and return NULL / 0.
 */
#ifndef OPENCV_KMP_IMGPROC2_H
#define OPENCV_KMP_IMGPROC2_H

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to a cv::Ptr<cv::LineSegmentDetector>. */
typedef struct cvk_line_segment_detector cvk_line_segment_detector_t;

/** Opaque handle to a cv::Ptr<cv::GeneralizedHoughBallard>. */
typedef struct cvk_gh_ballard cvk_gh_ballard_t;

/** Opaque handle to a cv::Ptr<cv::GeneralizedHoughGuil>. */
typedef struct cvk_gh_guil cvk_gh_guil_t;

/* =========================================================================
 * LineSegmentDetector
 * ========================================================================= */

/**
 * Creates a line segment detector; `refine` is a cv::LineSegmentDetectorModes
 * value (0=NONE, 1=STD, 2=ADV). NULL on failure.
 */
cvk_line_segment_detector_t *cvk_line_segment_detector_create(
    int refine, double scale, double sigma_scale, double quant, double ang_th,
    double log_eps, double density_th, int n_bins);

/**
 * Finds lines in a CV_8UC1 [image]. Returns the Nx1 CV_32FC4 lines Mat
 * (caller-owned); when non-NULL, *width, *prec and *nfa receive Nx1
 * CV_64FC1 Mats (nfa is only filled for LSD_REFINE_ADV). All outputs are
 * caller-owned and must be released with cvk_mat_release.
 */
cvk_mat_t *cvk_line_segment_detector_detect(
    const cvk_line_segment_detector_t *h, const cvk_mat_t *image,
    cvk_mat_t **width, cvk_mat_t **prec, cvk_mat_t **nfa);

/** Draws the [lines] onto [image] in place. */
void cvk_line_segment_detector_draw_segments(
    const cvk_line_segment_detector_t *h, cvk_mat_t *image, const cvk_mat_t *lines);

/**
 * Draws lines1 (blue) and lines2 (red) onto an optional [image] and returns
 * the count of non-overlapping (mismatching) pixels. Pass NULL for [image]
 * to skip the drawing.
 */
int cvk_line_segment_detector_compare_segments(
    const cvk_line_segment_detector_t *h, int size_width, int size_height,
    const cvk_mat_t *lines1, const cvk_mat_t *lines2, cvk_mat_t *image);

/* cv::Algorithm surface */
void cvk_line_segment_detector_clear(cvk_line_segment_detector_t *h);
int cvk_line_segment_detector_empty(cvk_line_segment_detector_t *h);
void cvk_line_segment_detector_save(cvk_line_segment_detector_t *h, const char *filename);
const char *cvk_line_segment_detector_get_default_name(cvk_line_segment_detector_t *h);

/** Frees the handle (exactly once; NULL tolerated). */
void cvk_line_segment_detector_release(cvk_line_segment_detector_t *h);

/* =========================================================================
 * GeneralizedHoughBallard
 * ========================================================================= */

/** Creates a position-only generalized Hough detector; NULL on failure. */
cvk_gh_ballard_t *cvk_gh_ballard_create(void);

/**
 * Sets the template image; (center_x, center_y) < 0 uses the template
 * center (cv::Point(-1,-1) default).
 */
void cvk_gh_ballard_set_template(const cvk_gh_ballard_t *h, const cvk_mat_t *templ,
                                 double center_x, double center_y);

/** Sets the template from precomputed edge / gradient images. */
void cvk_gh_ballard_set_template_edges(const cvk_gh_ballard_t *h, const cvk_mat_t *edges,
                                       const cvk_mat_t *dx, const cvk_mat_t *dy,
                                       double center_x, double center_y);

/**
 * Finds the template in [image]. Returns the 1xN CV_32FC4 positions Mat
 * (caller-owned); when non-NULL, *votes receives the 1xN CV_32SC3 votes Mat
 * (may be empty when no votes were accumulated).
 */
cvk_mat_t *cvk_gh_ballard_detect(const cvk_gh_ballard_t *h, const cvk_mat_t *image,
                                 cvk_mat_t **votes);

/** detect() over precomputed edge / gradient images. */
cvk_mat_t *cvk_gh_ballard_detect_edges(const cvk_gh_ballard_t *h, const cvk_mat_t *edges,
                                       const cvk_mat_t *dx, const cvk_mat_t *dy,
                                       cvk_mat_t **votes);

/* base-class fields (canny thresholds / minDist / dp / maxBufferSize) */
void cvk_gh_ballard_set_canny_low_thresh(cvk_gh_ballard_t *h, int v);
int cvk_gh_ballard_get_canny_low_thresh(const cvk_gh_ballard_t *h);
void cvk_gh_ballard_set_canny_high_thresh(cvk_gh_ballard_t *h, int v);
int cvk_gh_ballard_get_canny_high_thresh(const cvk_gh_ballard_t *h);
void cvk_gh_ballard_set_min_dist(cvk_gh_ballard_t *h, double v);
double cvk_gh_ballard_get_min_dist(const cvk_gh_ballard_t *h);
void cvk_gh_ballard_set_dp(cvk_gh_ballard_t *h, double v);
double cvk_gh_ballard_get_dp(const cvk_gh_ballard_t *h);
void cvk_gh_ballard_set_max_buffer_size(cvk_gh_ballard_t *h, int v);
int cvk_gh_ballard_get_max_buffer_size(const cvk_gh_ballard_t *h);

/* Ballard-specific fields */
void cvk_gh_ballard_set_levels(cvk_gh_ballard_t *h, int v);
int cvk_gh_ballard_get_levels(const cvk_gh_ballard_t *h);
void cvk_gh_ballard_set_votes_threshold(cvk_gh_ballard_t *h, int v);
int cvk_gh_ballard_get_votes_threshold(const cvk_gh_ballard_t *h);

/* cv::Algorithm surface */
void cvk_gh_ballard_clear(cvk_gh_ballard_t *h);
int cvk_gh_ballard_empty(cvk_gh_ballard_t *h);
void cvk_gh_ballard_save(cvk_gh_ballard_t *h, const char *filename);
const char *cvk_gh_ballard_get_default_name(cvk_gh_ballard_t *h);

/** Frees the handle (exactly once; NULL tolerated). */
void cvk_gh_ballard_release(cvk_gh_ballard_t *h);

/* =========================================================================
 * GeneralizedHoughGuil
 * ========================================================================= */

/** Creates a position/rotation/scale generalized Hough detector; NULL on failure. */
cvk_gh_guil_t *cvk_gh_guil_create(void);

/** Sets the template image; (center_x, center_y) < 0 uses the template center. */
void cvk_gh_guil_set_template(const cvk_gh_guil_t *h, const cvk_mat_t *templ,
                              double center_x, double center_y);

/** Sets the template from precomputed edge / gradient images. */
void cvk_gh_guil_set_template_edges(const cvk_gh_guil_t *h, const cvk_mat_t *edges,
                                    const cvk_mat_t *dx, const cvk_mat_t *dy,
                                    double center_x, double center_y);

/**
 * Finds the template in [image]. Returns the 1xN CV_32FC4 positions Mat
 * (caller-owned); when non-NULL, *votes receives the 1xN CV_32SC3 votes Mat
 * (may be empty when no votes were accumulated).
 */
cvk_mat_t *cvk_gh_guil_detect(const cvk_gh_guil_t *h, const cvk_mat_t *image,
                              cvk_mat_t **votes);

/** detect() over precomputed edge / gradient images. */
cvk_mat_t *cvk_gh_guil_detect_edges(const cvk_gh_guil_t *h, const cvk_mat_t *edges,
                                    const cvk_mat_t *dx, const cvk_mat_t *dy,
                                    cvk_mat_t **votes);

/* base-class fields */
void cvk_gh_guil_set_canny_low_thresh(cvk_gh_guil_t *h, int v);
int cvk_gh_guil_get_canny_low_thresh(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_canny_high_thresh(cvk_gh_guil_t *h, int v);
int cvk_gh_guil_get_canny_high_thresh(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_min_dist(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_min_dist(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_dp(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_dp(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_max_buffer_size(cvk_gh_guil_t *h, int v);
int cvk_gh_guil_get_max_buffer_size(const cvk_gh_guil_t *h);

/* Guil-specific fields */
void cvk_gh_guil_set_xi(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_xi(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_levels(cvk_gh_guil_t *h, int v);
int cvk_gh_guil_get_levels(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_angle_epsilon(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_angle_epsilon(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_min_angle(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_min_angle(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_max_angle(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_max_angle(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_angle_step(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_angle_step(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_angle_thresh(cvk_gh_guil_t *h, int v);
int cvk_gh_guil_get_angle_thresh(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_min_scale(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_min_scale(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_max_scale(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_max_scale(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_scale_step(cvk_gh_guil_t *h, double v);
double cvk_gh_guil_get_scale_step(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_scale_thresh(cvk_gh_guil_t *h, int v);
int cvk_gh_guil_get_scale_thresh(const cvk_gh_guil_t *h);
void cvk_gh_guil_set_pos_thresh(cvk_gh_guil_t *h, int v);
int cvk_gh_guil_get_pos_thresh(const cvk_gh_guil_t *h);

/* cv::Algorithm surface */
void cvk_gh_guil_clear(cvk_gh_guil_t *h);
int cvk_gh_guil_empty(cvk_gh_guil_t *h);
void cvk_gh_guil_save(cvk_gh_guil_t *h, const char *filename);
const char *cvk_gh_guil_get_default_name(cvk_gh_guil_t *h);

/** Frees the handle (exactly once; NULL tolerated). */
void cvk_gh_guil_release(cvk_gh_guil_t *h);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_IMGPROC2_H */

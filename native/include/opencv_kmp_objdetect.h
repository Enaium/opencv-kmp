/*
 * cvk_ C ABI declarations for the OpenCV "objdetect" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * ArUco surface: Dictionary, Board, GridBoard, CharucoBoard, ArucoDetector,
 * CharucoDetector, the DetectorParameters/CharucoParameters/RefineParameters
 * holders (as plain data structs) and the chessboard/circles-grid statics.
 *
 * List<Mat> marshalling: vectors of Mats travel as a CV_32SC2 Nx1 "container"
 * Mat whose row i holds the hi/lo 32-bit halves of the i-th Mat's address
 * (the same wire format org.opencv.utils.Converters.vector_Mat_to_Mat uses).
 * Output containers carry heap-allocated Mat handles owned by the caller;
 * input containers reference Mats that must stay alive during the call.
 */
#ifndef OPENCV_KMP_OBJDETECT_H
#define OPENCV_KMP_OBJDETECT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvk_dictionary cvk_dictionary_t;
typedef struct cvk_board cvk_board_t;
typedef struct cvk_grid_board cvk_grid_board_t;
typedef struct cvk_charuco_board cvk_charuco_board_t;
typedef struct cvk_aruco_detector cvk_aruco_detector_t;
typedef struct cvk_charuco_detector cvk_charuco_detector_t;

/* =========================================================================
 * Params holders (plain data, no native Params objects).
 *
 * Field order below is the wire order used by the JVM DoubleArray transport
 * and MUST match the Kotlin data class declaration order (Objdetect.kt) and
 * the JNI <-> struct conversion (jni_objdetect.cpp).
 * ========================================================================= */

typedef struct cvk_detector_params {
    int adaptive_thresh_win_size_min;             /* 3 */
    int adaptive_thresh_win_size_max;             /* 23 */
    int adaptive_thresh_win_size_step;            /* 10 */
    double adaptive_thresh_constant;              /* 7 */
    double min_marker_perimeter_rate;             /* 0.03 */
    double max_marker_perimeter_rate;             /* 4.0 */
    double polygonal_approx_accuracy_rate;        /* 0.03 */
    double min_corner_distance_rate;              /* 0.05 */
    int min_distance_to_border;                   /* 3 */
    double min_marker_distance_rate;              /* 0.125 */
    float min_group_distance;                     /* 0.21 */
    int corner_refinement_method;                 /* CORNER_REFINE_NONE */
    int corner_refinement_win_size;               /* 5 */
    float relative_corner_refinment_win_size;     /* 0.3 */
    int corner_refinement_max_iterations;         /* 30 */
    double corner_refinement_min_accuracy;        /* 0.1 */
    int marker_border_bits;                       /* 1 */
    int perspective_remove_pixel_per_cell;        /* 4 */
    double perspective_remove_ignored_margin_per_cell; /* 0.13 */
    double max_erroneous_bits_in_border_rate;     /* 0.35 */
    double min_otsu_std_dev;                      /* 5.0 */
    double error_correction_rate;                 /* 0.6 */
    float april_tag_quad_decimate;                /* 0.0 */
    float april_tag_quad_sigma;                   /* 0.0 */
    int april_tag_min_cluster_pixels;             /* 5 */
    int april_tag_max_nmaxima;                    /* 10 */
    float april_tag_critical_rad;                 /* 10*PI/180 */
    float april_tag_max_line_fit_mse;             /* 10.0 */
    int april_tag_min_white_black_diff;           /* 5 */
    int april_tag_deglitch;                       /* 0 */
    int detect_inverted_marker;                   /* false */
    int use_aruco;                                /* useAruco3Detection, false */
    int min_side_length_canonical_img;            /* 32 */
    float min_marker_length_ratio_original_img;   /* 0.0 */
    float valid_bit_id_threshold;                 /* 0.49 */
} cvk_detector_params_t;

typedef struct cvk_refine_params {
    float min_rep_distance;      /* 10.f */
    float error_correction_rate; /* 3.f */
    int check_all_orders;        /* true */
} cvk_refine_params_t;

typedef struct cvk_charuco_params {
    cvk_mat_t *camera_matrix; /* NULL -> empty Mat */
    cvk_mat_t *dist_coeffs;   /* NULL -> empty Mat */
    int min_markers;          /* 2 */
    int try_refine_markers;   /* false */
    int check_markers;        /* true */
} cvk_charuco_params_t;

/* =========================================================================
 * Dictionary
 * ========================================================================= */

/** Builds a dictionary from its byte list; release with cvk_dictionary_release. */
cvk_dictionary_t *cvk_dictionary_create(const cvk_mat_t *bytes_list, int marker_size,
                                        int max_correction_bits);

/** Frees a cvk_dictionary_create / cvk_get_predefined_dictionary handle. */
void cvk_dictionary_release(cvk_dictionary_t *h);

/** Copy of the marker codewords Mat (CV_8UC4, rows x 4*nbytes). */
cvk_mat_t *cvk_dictionary_get_bytes_list(const cvk_dictionary_t *h);

/** Replaces the marker codewords. */
void cvk_dictionary_set_bytes_list(cvk_dictionary_t *h, const cvk_mat_t *bytes_list);

/** Number of bits per marker dimension. */
int cvk_dictionary_get_marker_size(const cvk_dictionary_t *h);

void cvk_dictionary_set_marker_size(cvk_dictionary_t *h, int marker_size);

/** Maximum number of bits that can be corrected. */
int cvk_dictionary_get_max_correction_bits(const cvk_dictionary_t *h);

void cvk_dictionary_set_max_correction_bits(cvk_dictionary_t *h, int max_correction_bits);

/**
 * Identifies a marker bit matrix; writes the found id and rotation into
 * *idx_out/*rotation_out and returns 1 when identified, 0 otherwise.
 */
int cvk_dictionary_identify(const cvk_dictionary_t *h, const cvk_mat_t *only_bits,
                            double max_correction_rate, int *idx_out, int *rotation_out);

/** identify() over a [0;1] per-cell pixel-ratio matrix. */
int cvk_dictionary_identify_pixel_ratio(const cvk_dictionary_t *h,
                                        const cvk_mat_t *only_cell_pixel_ratio,
                                        double max_correction_rate,
                                        float valid_bit_id_threshold,
                                        int *idx_out, int *rotation_out);

/** Hamming distance of the input bits to a specific id (allRotations != 0). */
int cvk_dictionary_get_distance_to_id(const cvk_dictionary_t *h, const cvk_mat_t *bits,
                                      int id, int all_rotations);

/** Renders the canonical marker image (side_pixels x side_pixels). */
cvk_mat_t *cvk_dictionary_generate_image_marker(const cvk_dictionary_t *h, int id,
                                                int side_pixels, int border_bits);

/** Ground-truth bit matrix (marker_size x marker_size, CV_8UC1). */
cvk_mat_t *cvk_dictionary_get_marker_bits(const cvk_dictionary_t *h, int marker_id,
                                          int rotation_id);

/** Static: transforms a bit matrix into the 4-rotation byte list (CV_8UC4). */
cvk_mat_t *cvk_dictionary_get_byte_list_from_bits(const cvk_mat_t *bits);

/** Static: transforms a byte list row back into a bit matrix. */
cvk_mat_t *cvk_dictionary_get_bits_from_byte_list(const cvk_mat_t *byte_list,
                                                  int marker_size, int rotation_id);

/** Returns one of the predefined dictionaries (PredefinedDictionaryType id). */
cvk_dictionary_t *cvk_get_predefined_dictionary(int dict);

/** Extends a base dictionary to n_markers; NULL base -> fresh dictionary. */
cvk_dictionary_t *cvk_extend_dictionary(int n_markers, int marker_size,
                                        const cvk_dictionary_t *base_dictionary,
                                        int random_seed);

/* =========================================================================
 * Board
 * ========================================================================= */

/**
 * Common board constructor. obj_points is a vector-of-Mat container (each
 * Mat CV_32FC3 4x1 marker corners); ids is CV_32SC1.
 */
cvk_board_t *cvk_board_create(const cvk_mat_t *obj_points, const cvk_dictionary_t *dictionary,
                              const cvk_mat_t *ids);

/** Frees a board handle (also frees grid/charuco boards). */
void cvk_board_release(cvk_board_t *h);

/** New handle holding a copy of the board's dictionary. */
cvk_dictionary_t *cvk_board_get_dictionary(const cvk_board_t *h);

/** vector-of-Mat container; each Mat is the CV_32FC3 4x1 corners of a marker. */
cvk_mat_t *cvk_board_get_obj_points(const cvk_board_t *h);

/** Marker ids as a CV_32SC1 Nx1 Mat. */
cvk_mat_t *cvk_board_get_ids(const cvk_board_t *h);

/** Writes the bottom-right board corner into out3 (x, y, z). */
void cvk_board_get_right_bottom_corner(const cvk_board_t *h, double *out3);

/**
 * Matches detected markers against the board; returns objPoints and imgPoints
 * Mats (usable with solvePnP). detected_corners is a vector-of-Mat container,
 * detected_ids a CV_32SC1 Mat. Returns 1 on success.
 */
int cvk_board_match_image_points(const cvk_board_t *h, const cvk_mat_t *detected_corners,
                                 const cvk_mat_t *detected_ids, cvk_mat_t **obj_points_out,
                                 cvk_mat_t **img_points_out);

/** Draws the planar board centered on a new out_size image. */
cvk_mat_t *cvk_board_generate_image(const cvk_board_t *h, double out_size_width,
                                    double out_size_height, int margin_size, int border_bits);

/* =========================================================================
 * GridBoard / CharucoBoard
 * ========================================================================= */

/** GridBoard constructor; ids is CV_32SC1, pass NULL for the default 0..N-1 ids. */
cvk_grid_board_t *cvk_grid_board_create(double size_width, double size_height,
                                        float marker_length, float marker_separation,
                                        const cvk_dictionary_t *dictionary,
                                        const cvk_mat_t *ids);

void cvk_grid_board_release(cvk_grid_board_t *h);

/** Writes the marker grid size (width, height) into out2. */
void cvk_grid_board_get_grid_size(const cvk_grid_board_t *h, int *out2);

float cvk_grid_board_get_marker_length(const cvk_grid_board_t *h);

float cvk_grid_board_get_marker_separation(const cvk_grid_board_t *h);

/** CharucoBoard constructor; ids is CV_32SC1, pass NULL for the default ids. */
cvk_charuco_board_t *cvk_charuco_board_create(double size_width, double size_height,
                                              float square_length, float marker_length,
                                              const cvk_dictionary_t *dictionary,
                                              const cvk_mat_t *ids);

void cvk_charuco_board_release(cvk_charuco_board_t *h);

void cvk_charuco_board_set_legacy_pattern(cvk_charuco_board_t *h, int legacy_pattern);

int cvk_charuco_board_get_legacy_pattern(const cvk_charuco_board_t *h);

/** Writes the chessboard square count (width, height) into out2. */
void cvk_charuco_board_get_chessboard_size(const cvk_charuco_board_t *h, int *out2);

float cvk_charuco_board_get_square_length(const cvk_charuco_board_t *h);

float cvk_charuco_board_get_marker_length(const cvk_charuco_board_t *h);

/** Chessboard corners as a CV_32FC3 Nx1 Mat. */
cvk_mat_t *cvk_charuco_board_get_chessboard_corners(const cvk_charuco_board_t *h);

/** 1 when the given charuco corner ids are collinear. */
int cvk_charuco_board_check_charuco_corners_collinear(const cvk_charuco_board_t *h,
                                                      const cvk_mat_t *charuco_ids);

/* =========================================================================
 * ArucoDetector (Algorithm subclass)
 * ========================================================================= */

/** Basic ArucoDetector constructor. */
cvk_aruco_detector_t *cvk_aruco_detector_create(const cvk_dictionary_t *dictionary,
                                                const cvk_detector_params_t *detector_params,
                                                const cvk_refine_params_t *refine_params);

void cvk_aruco_detector_release(cvk_aruco_detector_t *h);

/**
 * Detects markers; writes vector-of-Mat containers / Mats into the outs
 * (caller-owned) and returns 1 on success.
 */
int cvk_aruco_detector_detect_markers(const cvk_aruco_detector_t *h, const cvk_mat_t *image,
                                      cvk_mat_t **corners_out, cvk_mat_t **ids_out,
                                      cvk_mat_t **rejected_out);

/** detectMarkers + per-marker confidence Mat (CV_32FC1). */
int cvk_aruco_detector_detect_markers_with_confidence(const cvk_aruco_detector_t *h,
                                                      const cvk_mat_t *image,
                                                      cvk_mat_t **corners_out,
                                                      cvk_mat_t **ids_out,
                                                      cvk_mat_t **confidence_out,
                                                      cvk_mat_t **rejected_out);

/** Refines detection against a board layout; camera/distortion Mats may be NULL. */
int cvk_aruco_detector_refine_detected_markers(const cvk_aruco_detector_t *h,
                                               const cvk_mat_t *image,
                                               const cvk_board_t *board,
                                               const cvk_mat_t *detected_corners,
                                               const cvk_mat_t *detected_ids,
                                               const cvk_mat_t *rejected_corners,
                                               const cvk_mat_t *camera_matrix,
                                               const cvk_mat_t *dist_coeffs,
                                               cvk_mat_t **corners_out, cvk_mat_t **ids_out,
                                               cvk_mat_t **rejected_out,
                                               cvk_mat_t **recovered_idxs_out);

/** detectMarkers over the configured dictionary list + dictIndices Mat. */
int cvk_aruco_detector_detect_markers_multi_dict(const cvk_aruco_detector_t *h,
                                                 const cvk_mat_t *image,
                                                 cvk_mat_t **corners_out, cvk_mat_t **ids_out,
                                                 cvk_mat_t **rejected_out,
                                                 cvk_mat_t **dict_indices_out);

/** New handle holding a copy of the first configured dictionary. */
cvk_dictionary_t *cvk_aruco_detector_get_dictionary(const cvk_aruco_detector_t *h);

/** Replaces the first configured dictionary (copied). */
void cvk_aruco_detector_set_dictionary(cvk_aruco_detector_t *h,
                                       const cvk_dictionary_t *dictionary);

/** Fills out with the current detector parameters. */
void cvk_aruco_detector_get_detector_params(const cvk_aruco_detector_t *h,
                                            cvk_detector_params_t *out);

void cvk_aruco_detector_set_detector_params(cvk_aruco_detector_t *h,
                                            const cvk_detector_params_t *params);

void cvk_aruco_detector_get_refine_params(const cvk_aruco_detector_t *h,
                                          cvk_refine_params_t *out);

void cvk_aruco_detector_set_refine_params(cvk_aruco_detector_t *h,
                                          const cvk_refine_params_t *params);

/* Algorithm surface. */
void cvk_aruco_detector_clear(cvk_aruco_detector_t *h);
int cvk_aruco_detector_empty(cvk_aruco_detector_t *h);
void cvk_aruco_detector_save(cvk_aruco_detector_t *h, const char *filename);
const char *cvk_aruco_detector_get_default_name(cvk_aruco_detector_t *h);

/* =========================================================================
 * CharucoDetector (Algorithm subclass)
 * ========================================================================= */

/** Basic CharucoDetector constructor. */
cvk_charuco_detector_t *cvk_charuco_detector_create(const cvk_charuco_board_t *board,
                                                    const cvk_charuco_params_t *charuco_params,
                                                    const cvk_detector_params_t *detector_params,
                                                    const cvk_refine_params_t *refine_params);

void cvk_charuco_detector_release(cvk_charuco_detector_t *h);

/** New handle holding a copy of the detector's board. */
cvk_charuco_board_t *cvk_charuco_detector_get_board(const cvk_charuco_detector_t *h);

/** Replaces the board (copied) and resyncs the internal ArucoDetector. */
void cvk_charuco_detector_set_board(cvk_charuco_detector_t *h,
                                    const cvk_charuco_board_t *board);

/** Fills out with the current charuco parameters (Mats are new handles). */
void cvk_charuco_detector_get_charuco_params(const cvk_charuco_detector_t *h,
                                             cvk_charuco_params_t *out);

void cvk_charuco_detector_set_charuco_params(cvk_charuco_detector_t *h,
                                             const cvk_charuco_params_t *params);

void cvk_charuco_detector_get_detector_params(const cvk_charuco_detector_t *h,
                                              cvk_detector_params_t *out);

void cvk_charuco_detector_set_detector_params(cvk_charuco_detector_t *h,
                                              const cvk_detector_params_t *params);

void cvk_charuco_detector_get_refine_params(const cvk_charuco_detector_t *h,
                                            cvk_refine_params_t *out);

void cvk_charuco_detector_set_refine_params(cvk_charuco_detector_t *h,
                                            const cvk_refine_params_t *params);

/**
 * Interpolates charuco corners. marker_corners (vector-of-Mat container) and
 * marker_ids (CV_32SC1 Mat) may be empty/NULL to let the detector find the
 * markers itself; they are returned updated. Returns 1 on success.
 */
int cvk_charuco_detector_detect_board(const cvk_charuco_detector_t *h, const cvk_mat_t *image,
                                      const cvk_mat_t *marker_corners,
                                      const cvk_mat_t *marker_ids,
                                      cvk_mat_t **charuco_corners_out,
                                      cvk_mat_t **charuco_ids_out,
                                      cvk_mat_t **marker_corners_out,
                                      cvk_mat_t **marker_ids_out);

/** Detects charuco diamonds (board must be 3x3). */
int cvk_charuco_detector_detect_diamonds(const cvk_charuco_detector_t *h, const cvk_mat_t *image,
                                         const cvk_mat_t *marker_corners,
                                         const cvk_mat_t *marker_ids,
                                         cvk_mat_t **diamond_corners_out,
                                         cvk_mat_t **diamond_ids_out,
                                         cvk_mat_t **marker_corners_out,
                                         cvk_mat_t **marker_ids_out);

/* Algorithm surface. */
void cvk_charuco_detector_clear(cvk_charuco_detector_t *h);
int cvk_charuco_detector_empty(cvk_charuco_detector_t *h);
void cvk_charuco_detector_save(cvk_charuco_detector_t *h, const char *filename);
const char *cvk_charuco_detector_get_default_name(cvk_charuco_detector_t *h);

/* =========================================================================
 * Objdetect statics (chessboard / circles-grid / ArUco helpers)
 * ========================================================================= */

/**
 * Finds chessboard corners; writes a new CV_32FC2 Nx1 Mat into *corners_out
 * and returns 1 when the whole pattern was found.
 */
int cvk_find_chessboard_corners(const cvk_mat_t *image, double pattern_width,
                                double pattern_height, int flags, cvk_mat_t **corners_out);

/** 1 when the image contains a chessboard of the given size. */
int cvk_check_chessboard(const cvk_mat_t *image, double size_width, double size_height);

/** Sector-based chessboard detection; also returns the CV_8UC1 meta Mat. */
int cvk_find_chessboard_corners_sb_with_meta(const cvk_mat_t *image, double pattern_width,
                                             double pattern_height, int flags,
                                             cvk_mat_t **corners_out, cvk_mat_t **meta_out);

/** Sector-based chessboard detection without the meta output. */
int cvk_find_chessboard_corners_sb(const cvk_mat_t *image, double pattern_width,
                                   double pattern_height, int flags, cvk_mat_t **corners_out);

/**
 * Estimates chessboard sharpness; writes the average profile into
 * *scalar_out (v0..v3) and a new CV_32FC1 per-profile Mat into
 * *sharpness_out. Returns 1 on success.
 */
int cvk_estimate_chessboard_sharpness(const cvk_mat_t *image, double pattern_width,
                                      double pattern_height, const cvk_mat_t *corners,
                                      float rise_distance, int vertical,
                                      cvk_scalar_t *scalar_out, cvk_mat_t **sharpness_out);

/** Refines chessboard corners in place; 1 when successful. */
int cvk_find4_quad_corner_subpix(const cvk_mat_t *image, cvk_mat_t *corners,
                                 double region_width, double region_height);

/** Draws detected chessboard corners onto image. */
void cvk_draw_chessboard_corners(cvk_mat_t *image, double pattern_width,
                                 double pattern_height, const cvk_mat_t *corners,
                                 int pattern_was_found);

/** Finds a circles grid; writes a new CV_32FC2 centers Mat, returns 1 on success. */
int cvk_find_circles_grid(const cvk_mat_t *image, double pattern_width, double pattern_height,
                          int flags, cvk_mat_t **centers_out);

/** Draws detected ArUco markers (corners is a vector-of-Mat container). */
void cvk_draw_detected_markers(cvk_mat_t *image, const cvk_mat_t *corners,
                               const cvk_mat_t *ids, cvk_scalar_t border_color);

/** Static generateImageMarker; returns the new side_pixels image. */
cvk_mat_t *cvk_generate_image_marker(const cvk_dictionary_t *dictionary, int id,
                                     int side_pixels, int border_bits);

/** Draws detected charuco corners. */
void cvk_draw_detected_corners_charuco(cvk_mat_t *image, const cvk_mat_t *charuco_corners,
                                       const cvk_mat_t *charuco_ids, cvk_scalar_t corner_color);

/** Draws detected charuco diamonds (diamond_corners is a vector-of-Mat container). */
void cvk_draw_detected_diamonds(cvk_mat_t *image, const cvk_mat_t *diamond_corners,
                                const cvk_mat_t *diamond_ids, cvk_scalar_t border_color);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_OBJDETECT_H */

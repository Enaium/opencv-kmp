/*
 * cvk_ C ABI declarations for the OpenCV "calib" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * List-of-Mat wire format: every "vector of vector of points" argument
 * (objectPoints / imagePoints / rvecs / tvecs / ...) travels as a
 * CV_32SC2 Nx1 Mat whose i-th row holds the high and low 32 bits of the
 * i-th cv::Mat address (the same encoding org.opencv.utils.Converters
 * uses). Output Mats (cameraMatrix, distCoeffs, R, T, E, F, K, D, ...) are
 * written in place into the handles the caller allocates; the shim is
 * noexcept and reports failures through cvk_last_error().
 */
#ifndef OPENCV_KMP_CALIB_H
#define OPENCV_KMP_CALIB_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Calibration
 * ========================================================================= */

/** cv::initCameraMatrix2D; returns a new 3x3 Mat or NULL on failure. */
cvk_mat_t *cvk_init_camera_matrix_2d(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points,
                                     int image_width, int image_height,
                                     double aspect_ratio);

/**
 * cv::calibrateCamera. rvecs/tvecs are CV_32SC2 Nx1 output wires filled
 * with per-view 3x1 rotation/translation Mat handles. Returns the RMS
 * re-projection error, or NaN on failure.
 */
double cvk_calibrate_camera(const cvk_mat_t *object_points,
                            const cvk_mat_t *image_points,
                            int image_width, int image_height,
                            cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                            cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                            int flags, int criteria_type, int criteria_max_count,
                            double criteria_epsilon);

/** cv::calibrateCamera with stdDeviation/perViewErrors outputs. */
double cvk_calibrate_camera_extended(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points,
                                     int image_width, int image_height,
                                     cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                                     cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                     cvk_mat_t *std_deviations_intrinsics,
                                     cvk_mat_t *std_deviations_extrinsics,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon);

/** cv::calibrateCameraRO (object-releasing extension). */
double cvk_calibrate_camera_ro(const cvk_mat_t *object_points,
                               const cvk_mat_t *image_points,
                               int image_width, int image_height, int i_fixed_point,
                               cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                               cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                               cvk_mat_t *new_obj_points,
                               int flags, int criteria_type, int criteria_max_count,
                               double criteria_epsilon);

/** cv::calibrateCameraRO with stdDeviation/perViewErrors outputs. */
double cvk_calibrate_camera_ro_extended(const cvk_mat_t *object_points,
                                        const cvk_mat_t *image_points,
                                        int image_width, int image_height, int i_fixed_point,
                                        cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                                        cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                        cvk_mat_t *new_obj_points,
                                        cvk_mat_t *std_deviations_intrinsics,
                                        cvk_mat_t *std_deviations_extrinsics,
                                        cvk_mat_t *std_deviations_obj_points,
                                        cvk_mat_t *per_view_errors,
                                        int flags, int criteria_type, int criteria_max_count,
                                        double criteria_epsilon);

/* =========================================================================
 * Stereo calibration
 * ========================================================================= */

/** cv::stereoCalibrate (no rvecs/tvecs/perViewErrors outputs). */
double cvk_stereo_calibrate(const cvk_mat_t *object_points,
                            const cvk_mat_t *image_points1,
                            const cvk_mat_t *image_points2,
                            int image_width, int image_height,
                            cvk_mat_t *camera_matrix1, cvk_mat_t *dist_coeffs1,
                            cvk_mat_t *camera_matrix2, cvk_mat_t *dist_coeffs2,
                            cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                            int flags, int criteria_type, int criteria_max_count,
                            double criteria_epsilon);

/** cv::stereoCalibrate with a perViewErrors output. */
double cvk_stereo_calibrate_per_view(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points1,
                                     const cvk_mat_t *image_points2,
                                     int image_width, int image_height,
                                     cvk_mat_t *camera_matrix1, cvk_mat_t *dist_coeffs1,
                                     cvk_mat_t *camera_matrix2, cvk_mat_t *dist_coeffs2,
                                     cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon);

/** cv::stereoCalibrate with rvecs/tvecs/perViewErrors outputs. */
double cvk_stereo_calibrate_extended(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points1,
                                     const cvk_mat_t *image_points2,
                                     int image_width, int image_height,
                                     cvk_mat_t *camera_matrix1, cvk_mat_t *dist_coeffs1,
                                     cvk_mat_t *camera_matrix2, cvk_mat_t *dist_coeffs2,
                                     cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                                     cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon);

/**
 * cv::registerCameras: extrinsic-only calibration of a camera pair with
 * known intrinsics (possibly mixed pinhole/fisheye models).
 */
double cvk_register_cameras(const cvk_mat_t *object_points1,
                            const cvk_mat_t *object_points2,
                            const cvk_mat_t *image_points1,
                            const cvk_mat_t *image_points2,
                            const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                            int camera_model1,
                            const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                            int camera_model2,
                            cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                            cvk_mat_t *per_view_errors,
                            int flags, int criteria_type, int criteria_max_count,
                            double criteria_epsilon);

/** cv::registerCameras with rvecs/tvecs/perViewErrors outputs. */
double cvk_register_cameras_extended(const cvk_mat_t *object_points1,
                                     const cvk_mat_t *object_points2,
                                     const cvk_mat_t *image_points1,
                                     const cvk_mat_t *image_points2,
                                     const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                                     int camera_model1,
                                     const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                                     int camera_model2,
                                     cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                                     cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon);

/* =========================================================================
 * Fisheye calibration (cv::fisheye)
 * ========================================================================= */

/** cv::fisheye::calibrate. */
double cvk_fisheye_calibrate(const cvk_mat_t *object_points,
                             const cvk_mat_t *image_points,
                             int image_width, int image_height,
                             cvk_mat_t *k, cvk_mat_t *d,
                             cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                             int flags, int criteria_type, int criteria_max_count,
                             double criteria_epsilon);

/** cv::fisheye::stereoCalibrate with per-view rvecs/tvecs outputs. */
double cvk_fisheye_stereo_calibrate(const cvk_mat_t *object_points,
                                    const cvk_mat_t *image_points1,
                                    const cvk_mat_t *image_points2,
                                    int image_width, int image_height,
                                    cvk_mat_t *k1, cvk_mat_t *d1,
                                    cvk_mat_t *k2, cvk_mat_t *d2,
                                    cvk_mat_t *r, cvk_mat_t *t,
                                    cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                    int flags, int criteria_type, int criteria_max_count,
                                    double criteria_epsilon);

/** cv::fisheye::stereoCalibrate (pose-only overload). */
double cvk_fisheye_stereo_calibrate_pose(const cvk_mat_t *object_points,
                                         const cvk_mat_t *image_points1,
                                         const cvk_mat_t *image_points2,
                                         int image_width, int image_height,
                                         cvk_mat_t *k1, cvk_mat_t *d1,
                                         cvk_mat_t *k2, cvk_mat_t *d2,
                                         cvk_mat_t *r, cvk_mat_t *t,
                                         int flags, int criteria_type, int criteria_max_count,
                                         double criteria_epsilon);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_CALIB_H */

/*
 * cvk_ C ABI declarations for the OpenCV "imgproc" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 */
#ifndef OPENCV_KMP_IMGPROC_H
#define OPENCV_KMP_IMGPROC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Rotated rectangle (cv::RotatedRect) wire struct. */
typedef struct cvk_rotated_rect {
    double cx;
    double cy;
    double width;
    double height;
    double angle;
} cvk_rotated_rect_t;

/* Raw spatial moments (cv::Moments raw part; mu/nu derive from these). */
typedef struct cvk_moments {
    double m00;
    double m10;
    double m01;
    double m20;
    double m11;
    double m02;
    double m30;
    double m21;
    double m12;
    double m03;
} cvk_moments_t;

/* =========================================================================
 * kernels / filters
 * ========================================================================= */

/** cv::getDerivKernels; writes kx/ky handles and returns 1 on success. */
int cvk_get_deriv_kernels(int dx, int dy, int ksize, int normalize, int ktype,
                          cvk_mat_t **kx, cvk_mat_t **ky);

/** cv::getGaborKernel; NULL on failure. */
cvk_mat_t *cvk_get_gabor_kernel(int ksize_width, int ksize_height, double sigma,
                                double theta, double lambd, double gamma,
                                double psi, int ktype);

/** cv::sepFilter2D with a separable kernel pair; new Mat result. */
cvk_mat_t *cvk_sep_filter_2d(const cvk_mat_t *src, int ddepth,
                             const cvk_mat_t *kx, const cvk_mat_t *ky,
                             int anchor_x, int anchor_y, double delta,
                             int border_type);

/* =========================================================================
 * corners
 * ========================================================================= */

/** cv::preCornerDetect; new CV_32F Mat result. */
cvk_mat_t *cvk_pre_corner_detect(const cvk_mat_t *src, int ksize,
                                 int border_type);

/** cv::cornerEigenValsAndVecs; new CV_32FC6 Mat result. */
cvk_mat_t *cvk_corner_eigen_vals_and_vecs(const cvk_mat_t *src, int block_size,
                                          int ksize, int border_type);

/* =========================================================================
 * color conversion
 * ========================================================================= */

/** cv::cvtColorTwoPlane (NV12/NV21-style two-plane sources). */
cvk_mat_t *cvk_cvt_color_two_plane(const cvk_mat_t *src1, const cvk_mat_t *src2,
                                   int code);

/* =========================================================================
 * calibration helpers (calib3d surface mapped to this slice by the port plan)
 * ========================================================================= */

/** cv::initUndistortRectifyMap; writes map1/map2 and returns 1 on success. */
int cvk_init_undistort_rectify_map(const cvk_mat_t *camera_matrix,
                                   const cvk_mat_t *dist_coeffs,
                                   const cvk_mat_t *r,
                                   const cvk_mat_t *new_camera_matrix,
                                   int size_width, int size_height, int m1_type,
                                   cvk_mat_t **map1, cvk_mat_t **map2);

/** cv::undistortPoints; r/p may be NULL for cv::noArray(). */
cvk_mat_t *cvk_undistort_points(const cvk_mat_t *src,
                                const cvk_mat_t *camera_matrix,
                                const cvk_mat_t *dist_coeffs,
                                const cvk_mat_t *r, const cvk_mat_t *p);

/** cv::getDefaultNewCameraMatrix; size 0x0 means "empty Size()". */
cvk_mat_t *cvk_get_default_new_camera_matrix(const cvk_mat_t *camera_matrix,
                                             int size_width, int size_height,
                                             int center_principal_point);

/** cv::estimateAffine2D; NULL when no transformation could be estimated. */
cvk_mat_t *cvk_estimate_affine_2d(const cvk_mat_t *from, const cvk_mat_t *to,
                                  int method, double ransac_reproj_threshold,
                                  long long max_iters, double confidence,
                                  long long refine_iters);

/** cv::estimateAffinePartial2D; NULL when no transformation could be estimated. */
cvk_mat_t *cvk_estimate_affine_partial_2d(const cvk_mat_t *from,
                                          const cvk_mat_t *to, int method,
                                          double ransac_reproj_threshold,
                                          long long max_iters, double confidence,
                                          long long refine_iters);

/* =========================================================================
 * distance transform
 * ========================================================================= */

/** cv::distanceTransform with labels; writes dst/labels and returns 1. */
int cvk_distance_transform_with_labels(const cvk_mat_t *src, int distance_type,
                                       int mask_size, int label_type,
                                       cvk_mat_t **dst, cvk_mat_t **labels);

/* =========================================================================
 * contours / geometry (surface mapped to this slice by the port plan)
 * ========================================================================= */

/** cv::convexHull; returnPoints=0 yields CV_32S indices. */
cvk_mat_t *cvk_convex_hull(const cvk_mat_t *points, int clockwise,
                           int return_points);

/** cv::isContourConvex; 1 when convex, 0 otherwise. */
int cvk_is_contour_convex(const cvk_mat_t *points);

/** cv::convexityDefects; new Nx4 CV_32S Mat result. */
cvk_mat_t *cvk_convexity_defects(const cvk_mat_t *contour,
                                 const cvk_mat_t *hull_idx);

/** cv::fitLine; new 4x1 CV_32F line [vx, vy, x0, y0]. */
cvk_mat_t *cvk_fit_line(const cvk_mat_t *points, int dist_type, double param,
                        double reps, double aeps);

/** cv::boxPoints; new 4x2 CV_32F Mat result. */
cvk_mat_t *cvk_box_points(cvk_rotated_rect_t box);

/** cv::rotatedRectangleIntersection; returns the intersect type (0/1/2). */
int cvk_rotated_rectangle_intersection(cvk_rotated_rect_t rect1,
                                       cvk_rotated_rect_t rect2,
                                       cvk_mat_t **intersection);

/** cv::pointPolygonTest; signed distance or +1/-1/0 by measure_dist. */
double cvk_point_polygon_test(const cvk_mat_t *contour, double x, double y,
                              int measure_dist);

/** cv::intersectConvexConvex; returns the intersection area (float). */
float cvk_intersect_convex_convex(const cvk_mat_t *p1, const cvk_mat_t *p2,
                                  int handle_nested, cvk_mat_t **p12);

/** cv::HuMoments; writes 7 invariants into out[]. */
void cvk_hu_moments(cvk_moments_t moments, double out[7]);

/* =========================================================================
 * pyramids
 * ========================================================================= */

/** cv::buildPyramid; writes max_level+1 level handles, returns the count. */
int cvk_build_pyramid(const cvk_mat_t *src, int max_level, int border_type,
                      cvk_mat_t **levels,
                      int max_count);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_IMGPROC_H */

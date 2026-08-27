/*
 * cvk_ C ABI declarations for the OpenCV "geometry" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * Conventions (see opencv_kmp.h): every function is noexcept and reports
 * failures through cvk_last_error(); Mat-returning functions return NULL on
 * failure. Functions with several Mat results write the extra handles into
 * caller-provided `cvk_mat_t **` out-params — those handles are owned by the
 * caller and must be freed with exactly one cvk_mat_release() each. Lists of
 * Mats travel as CV_32SC2 "packed pointer" Mats (two int32s per 64-bit Mat
 * address, matching the SDK's Converters.vector_Mat_to_Mat wire format);
 * every element points at a heap-allocated cv::Mat the receiver owns.
 *
 * Subdiv2D lives here too (opencv2/geometry/2d.hpp). Points travel as plain
 * doubles; edge/vertex lists and triangle lists come back as Mats in the
 * same wire types the Java SDK uses (CV_32FC4 / CV_32SC1 / CV_32FC6).
 */
#ifndef OPENCV_KMP_GEOMETRY_H
#define OPENCV_KMP_GEOMETRY_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Multi-view 3D vision (opencv2/geometry/3d.hpp)
 * ========================================================================= */

/** Rotation vector <-> rotation matrix conversion (cv::Rodrigues). */
cvk_mat_t *cvk_rodrigues(const cvk_mat_t *src);

/** cv::Rodrigues plus the 3x9/9x3 Jacobian written into *out_jacobian. */
cvk_mat_t *cvk_rodrigues_jacobian(const cvk_mat_t *src, cvk_mat_t **out_jacobian);

/** cv::findHomography; NULL on failure (e.g. degenerate point sets). */
cvk_mat_t *cvk_find_homography(const cvk_mat_t *src_points, const cvk_mat_t *dst_points,
                               int method, double ransac_reproj_threshold, int max_iters,
                               double confidence);

/** cv::findHomography with the inlier mask written into *out_mask. */
cvk_mat_t *cvk_find_homography_masked(const cvk_mat_t *src_points, const cvk_mat_t *dst_points,
                                      int method, double ransac_reproj_threshold, int max_iters,
                                      double confidence, cvk_mat_t **out_mask);

/** cv::findHomography(src, dst, mask, params) — USAC variant with a mask. */
cvk_mat_t *cvk_find_homography_usac(const cvk_mat_t *src_points, const cvk_mat_t *dst_points,
                                    cvk_mat_t **out_mask,
                                    double confidence, int is_parallel, int lo_iterations,
                                    int lo_method, int lo_sample_size, int max_iterations,
                                    int neighbors_search, int random_generator_state, int sampler,
                                    int score, double threshold, int final_polisher,
                                    int final_polisher_iterations);

/**
 * cv::RQDecomp3x3. Writes mtxR and mtxQ; when want_axes != 0 also Qx/Qy/Qz.
 * The three Euler angles (degrees) are written into out_euler3[0..2].
 */
void cvk_rq_decomp_3x3(const cvk_mat_t *src, int want_axes,
                       cvk_mat_t **out_r, cvk_mat_t **out_q,
                       cvk_mat_t **out_qx, cvk_mat_t **out_qy, cvk_mat_t **out_qz,
                       double *out_euler3);

/** cv::matMulDeriv: d(A*B)/dA and d(A*B)/dB. */
void cvk_mat_mul_deriv(const cvk_mat_t *a, const cvk_mat_t *b,
                       cvk_mat_t **out_dabda, cvk_mat_t **out_dabdb);

/**
 * cv::composeRT. Always writes rvec3/tvec3; when want_derivatives != 0 also
 * all eight derivative Mats (they may be passed as NULL pointers otherwise).
 */
void cvk_compose_rt(const cvk_mat_t *rvec1, const cvk_mat_t *tvec1,
                    const cvk_mat_t *rvec2, const cvk_mat_t *tvec2,
                    int want_derivatives,
                    cvk_mat_t **out_rvec3, cvk_mat_t **out_tvec3,
                    cvk_mat_t **out_dr3dr1, cvk_mat_t **out_dr3dt1,
                    cvk_mat_t **out_dr3dr2, cvk_mat_t **out_dr3dt2,
                    cvk_mat_t **out_dt3dr1, cvk_mat_t **out_dt3dt1,
                    cvk_mat_t **out_dt3dr2, cvk_mat_t **out_dt3dt2);

/** cv::projectPoints: 3D -> 2D with intrinsics/distortion. */
cvk_mat_t *cvk_project_points(const cvk_mat_t *object_points, const cvk_mat_t *rvec,
                              const cvk_mat_t *tvec, const cvk_mat_t *camera_matrix,
                              const cvk_mat_t *dist_coeffs);

/** cv::projectPoints plus the 2Nx(10+k) Jacobian into *out_jacobian. */
cvk_mat_t *cvk_project_points_jacobian(const cvk_mat_t *object_points, const cvk_mat_t *rvec,
                                       const cvk_mat_t *tvec, const cvk_mat_t *camera_matrix,
                                       const cvk_mat_t *dist_coeffs, double aspect_ratio,
                                       cvk_mat_t **out_jacobian);

/** cv::solvePnP; returns 1 on success, 0 on failure (rvec/tvec then NULL). */
int cvk_solve_pnp(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                  const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                  int use_extrinsic_guess, int flags,
                  cvk_mat_t **out_rvec, cvk_mat_t **out_tvec);

/** cv::solvePnPRansac; returns 1 on success; inliers written when non-NULL. */
int cvk_solve_pnp_ransac(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                         const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                         int use_extrinsic_guess, int iterations_count, float reprojection_error,
                         double confidence, int flags,
                         cvk_mat_t **out_rvec, cvk_mat_t **out_tvec, cvk_mat_t **out_inliers);

/** cv::solvePnPRansac(object, image, cameraMatrix, dist, rvec, tvec, inliers, params). */
int cvk_solve_pnp_ransac_usac(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                              const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                              cvk_mat_t **out_rvec, cvk_mat_t **out_tvec, cvk_mat_t **out_inliers,
                              double confidence, int is_parallel, int lo_iterations,
                              int lo_method, int lo_sample_size, int max_iterations,
                              int neighbors_search, int random_generator_state, int sampler,
                              int score, double threshold, int final_polisher,
                              int final_polisher_iterations);

/**
 * cv::solvePnPGeneric. rvecs/tvecs come back as CV_32SC2 packed-pointer
 * Mats; reprojectionError (Nx1 CV_64F) written only when out_reprojection_error
 * is non-NULL. Returns the number of solutions.
 */
int cvk_solve_pnp_generic(const cvk_mat_t *object_points, const cvk_mat_t *image_points,
                          const cvk_mat_t *camera_matrix, const cvk_mat_t *dist_coeffs,
                          int use_extrinsic_guess, int flags,
                          const cvk_mat_t *rvec, const cvk_mat_t *tvec,
                          cvk_mat_t **out_rvecs, cvk_mat_t **out_tvecs,
                          cvk_mat_t **out_reprojection_error);

/** cv::convertPointsToHomogeneous (dtype -1 keeps the source depth). */
cvk_mat_t *cvk_convert_points_to_homogeneous(const cvk_mat_t *src, int dtype);

/** cv::convertPointsFromHomogeneous (dtype -1 keeps the source depth). */
cvk_mat_t *cvk_convert_points_from_homogeneous(const cvk_mat_t *src, int dtype);

/** cv::findFundamentalMat; NULL on failure (insufficient points). */
cvk_mat_t *cvk_find_fundamental_mat(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                    int method, double ransac_reproj_threshold, double confidence,
                                    int max_iters);

/** cv::findFundamentalMat with the inlier mask written into *out_mask. */
cvk_mat_t *cvk_find_fundamental_mat_masked(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                           int method, double ransac_reproj_threshold,
                                           double confidence, int max_iters,
                                           cvk_mat_t **out_mask);

/** cv::findFundamentalMat(points1, points2, mask, params) — USAC variant. */
cvk_mat_t *cvk_find_fundamental_mat_usac(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                         cvk_mat_t **out_mask,
                                         double confidence, int is_parallel, int lo_iterations,
                                         int lo_method, int lo_sample_size, int max_iterations,
                                         int neighbors_search, int random_generator_state,
                                         int sampler, int score, double threshold,
                                         int final_polisher, int final_polisher_iterations);

/** cv::findEssentialMat with a shared camera matrix. */
cvk_mat_t *cvk_find_essential_mat(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                  const cvk_mat_t *camera_matrix, int method, double prob,
                                  double threshold, int max_iters);

/** cv::findEssentialMat + mask (shared camera matrix). */
cvk_mat_t *cvk_find_essential_mat_masked(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                         const cvk_mat_t *camera_matrix, int method, double prob,
                                         double threshold, int max_iters, cvk_mat_t **out_mask);

/** cv::findEssentialMat with focal length + principal point. */
cvk_mat_t *cvk_find_essential_mat_focal(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                        double focal, double pp_x, double pp_y, int method,
                                        double prob, double threshold, int max_iters);

/** cv::findEssentialMat (focal form) + mask. */
cvk_mat_t *cvk_find_essential_mat_focal_masked(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                               double focal, double pp_x, double pp_y, int method,
                                               double prob, double threshold, int max_iters,
                                               cvk_mat_t **out_mask);

/** cv::findEssentialMat for two calibrated cameras. */
cvk_mat_t *cvk_find_essential_mat_stereo(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                         const cvk_mat_t *camera_matrix1,
                                         const cvk_mat_t *dist_coeffs1,
                                         const cvk_mat_t *camera_matrix2,
                                         const cvk_mat_t *dist_coeffs2, int method, double prob,
                                         double threshold);

/** cv::findEssentialMat (stereo form) + mask. */
cvk_mat_t *cvk_find_essential_mat_stereo_masked(const cvk_mat_t *points1,
                                                const cvk_mat_t *points2,
                                                const cvk_mat_t *camera_matrix1,
                                                const cvk_mat_t *dist_coeffs1,
                                                const cvk_mat_t *camera_matrix2,
                                                const cvk_mat_t *dist_coeffs2, int method,
                                                double prob, double threshold,
                                                cvk_mat_t **out_mask);

/** cv::findEssentialMat(..., mask, params) — USAC variant, two cameras. */
cvk_mat_t *cvk_find_essential_mat_stereo_usac(const cvk_mat_t *points1, const cvk_mat_t *points2,
                                              const cvk_mat_t *camera_matrix1,
                                              const cvk_mat_t *camera_matrix2,
                                              const cvk_mat_t *dist_coeffs1,
                                              const cvk_mat_t *dist_coeffs2,
                                              cvk_mat_t **out_mask,
                                              double confidence, int is_parallel,
                                              int lo_iterations, int lo_method,
                                              int lo_sample_size, int max_iterations,
                                              int neighbors_search, int random_generator_state,
                                              int sampler, int score, double threshold,
                                              int final_polisher, int final_polisher_iterations);

/** cv::decomposeEssentialMat: E -> R1, R2, t (unit length). */
void cvk_decompose_essential_mat(const cvk_mat_t *e,
                                 cvk_mat_t **out_r1, cvk_mat_t **out_r2, cvk_mat_t **out_t);

/**
 * cv::recoverPose (two calibrated cameras); returns inlier count. mask_in is
 * an optional input inlier mask (may be NULL); the chirality-checked output
 * mask is written into *out_mask.
 */
int cvk_recover_pose(const cvk_mat_t *points1, const cvk_mat_t *points2,
                     const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                     const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                     int method, double prob, double threshold, const cvk_mat_t *mask_in,
                     cvk_mat_t **out_e, cvk_mat_t **out_r, cvk_mat_t **out_t,
                     cvk_mat_t **out_mask);

/** cv::recoverPose(E, points, cameraMatrix, R, t, mask); mask_in may be NULL. */
int cvk_recover_pose_e(const cvk_mat_t *e, const cvk_mat_t *points1, const cvk_mat_t *points2,
                       const cvk_mat_t *camera_matrix, const cvk_mat_t *mask_in,
                       cvk_mat_t **out_r, cvk_mat_t **out_t, cvk_mat_t **out_mask);

/** cv::recoverPose(E, points, R, t, focal, pp, mask); mask_in may be NULL. */
int cvk_recover_pose_e_focal(const cvk_mat_t *e, const cvk_mat_t *points1,
                             const cvk_mat_t *points2, double focal, double pp_x, double pp_y,
                             const cvk_mat_t *mask_in,
                             cvk_mat_t **out_r, cvk_mat_t **out_t, cvk_mat_t **out_mask);

/** cv::recoverPose(..., distanceThresh, mask, triangulatedPoints); mask_in may be NULL. */
int cvk_recover_pose_e_distance(const cvk_mat_t *e, const cvk_mat_t *points1,
                                const cvk_mat_t *points2, const cvk_mat_t *camera_matrix,
                                double distance_thresh, const cvk_mat_t *mask_in,
                                cvk_mat_t **out_r, cvk_mat_t **out_t, cvk_mat_t **out_mask,
                                cvk_mat_t **out_triangulated);

/** cv::triangulatePoints: 4xN homogeneous points. */
cvk_mat_t *cvk_triangulate_points(const cvk_mat_t *proj_matr1, const cvk_mat_t *proj_matr2,
                                  const cvk_mat_t *proj_points1, const cvk_mat_t *proj_points2);

/** cv::correctMatches: F-consistent refined point pairs. */
void cvk_correct_matches(const cvk_mat_t *f, const cvk_mat_t *points1, const cvk_mat_t *points2,
                         cvk_mat_t **out_new_points1, cvk_mat_t **out_new_points2);

/** cv::estimateAffine3D (RANSAC); returns 1 when a solution was found. */
int cvk_estimate_affine_3d(const cvk_mat_t *src, const cvk_mat_t *dst, double ransac_threshold,
                           double confidence, cvk_mat_t **out_transform,
                           cvk_mat_t **out_inliers);

/** cv::estimateAffine3D (Umeyama); scale written into *out_scale when non-NULL. */
cvk_mat_t *cvk_estimate_affine_3d_umeyama(const cvk_mat_t *src, const cvk_mat_t *dst,
                                          int force_rotation, double *out_scale);

/** cv::estimateAffine2D(pts1, pts2, inliers, params) — USAC variant.
 *  The plain RANSAC/LMEDS estimateAffine2D/estimateAffinePartial2D live in
 *  opencv_kmp_imgproc.h (owned by the imgproc slice). */
cvk_mat_t *cvk_estimate_affine_2d_usac(const cvk_mat_t *pts1, const cvk_mat_t *pts2,
                                       cvk_mat_t **out_inliers,
                                       double confidence, int is_parallel, int lo_iterations,
                                       int lo_method, int lo_sample_size, int max_iterations,
                                       int neighbors_search, int random_generator_state,
                                       int sampler, int score, double threshold,
                                       int final_polisher, int final_polisher_iterations);

/**
 * cv::decomposeHomographyMat. The three output lists arrive as CV_32SC2
 * packed-pointer Mats; returns the number of solutions (up to 4).
 */
int cvk_decompose_homography_mat(const cvk_mat_t *h, const cvk_mat_t *k,
                                 cvk_mat_t **out_rotations, cvk_mat_t **out_translations,
                                 cvk_mat_t **out_normals);

/**
 * cv::filterHomographyDecompByVisibleRefpoints. rotations/normals are CV_32SC2
 * packed-pointer Mats of the decomposition output; possibleSolutions is a
 * CV_32S index Mat written into *out_possible_solutions. points_mask may be NULL.
 */
void cvk_filter_homography_decomp(const cvk_mat_t *rotations, const cvk_mat_t *normals,
                                  const cvk_mat_t *before_points, const cvk_mat_t *after_points,
                                  const cvk_mat_t *points_mask,
                                  cvk_mat_t **out_possible_solutions);

/* =========================================================================
 * Subdiv2D (opencv2/geometry/2d.hpp)
 * ========================================================================= */

typedef struct cvk_subdiv2d cvk_subdiv2d_t;

/** cv::Subdiv2D(); call initDelaunay before inserting points. */
cvk_subdiv2d_t *cvk_subdiv2d_create(void);

/** cv::Subdiv2D(Rect). */
cvk_subdiv2d_t *cvk_subdiv2d_create_with_rect(int x, int y, int width, int height);

/** cv::Subdiv2D::initDelaunay(Rect). */
void cvk_subdiv2d_init_delaunay(cvk_subdiv2d_t *h, int x, int y, int width, int height);

/** cv::Subdiv2D::insert(Point2f); returns the vertex id. */
int cvk_subdiv2d_insert_point(cvk_subdiv2d_t *h, double x, double y);

/** cv::Subdiv2D::insert(vector<Point2f>); points is an Nx1 CV_32FC2 Mat. */
void cvk_subdiv2d_insert_points(cvk_subdiv2d_t *h, const cvk_mat_t *points);

/**
 * cv::Subdiv2D::locate(pt, edge, vertex). Returns a PTLOC_* code; edge and
 * vertex are written through the out-params.
 */
int cvk_subdiv2d_locate(const cvk_subdiv2d_t *h, double x, double y,
                        int *out_edge, int *out_vertex);

/** cv::Subdiv2D::findNearest(pt, nearestPt); returns the vertex id. */
int cvk_subdiv2d_find_nearest(const cvk_subdiv2d_t *h, double x, double y,
                              double *out_nearest_x, double *out_nearest_y);

/** cv::Subdiv2D::getEdgeList as an Nx1 CV_32FC4 Mat (org_x,org_y,dst_x,dst_y). */
cvk_mat_t *cvk_subdiv2d_get_edge_list(const cvk_subdiv2d_t *h);

/** cv::Subdiv2D::getLeadingEdgeList as an Nx1 CV_32SC1 Mat. */
cvk_mat_t *cvk_subdiv2d_get_leading_edge_list(const cvk_subdiv2d_t *h);

/** cv::Subdiv2D::getTriangleList as an Nx1 CV_32FC6 Mat (6 floats per row). */
cvk_mat_t *cvk_subdiv2d_get_triangle_list(const cvk_subdiv2d_t *h);

/**
 * cv::Subdiv2D::getVoronoiFacetList(idx, facetList, facetCenters). idx is an
 * Nx1 CV_32SC1 Mat (may be empty for all facets). The facet list arrives as a
 * CV_32SC2 packed-pointer Mat of CV_32FC2 facet Mats; facetCenters is an Nx1
 * CV_32FC2 Mat written into *out_facet_centers.
 */
cvk_mat_t *cvk_subdiv2d_get_voronoi_facet_list(const cvk_subdiv2d_t *h, const cvk_mat_t *idx,
                                               cvk_mat_t **out_facet_centers);

/** cv::Subdiv2D::getVertex(vertex, firstEdge); writes x, y and first edge. */
void cvk_subdiv2d_get_vertex(const cvk_subdiv2d_t *h, int vertex,
                             double *out_x, double *out_y, int *out_first_edge);

/** cv::Subdiv2D::getEdge(edge, nextEdgeType). */
int cvk_subdiv2d_get_edge(const cvk_subdiv2d_t *h, int edge, int next_edge_type);

/** cv::Subdiv2D::nextEdge(edge). */
int cvk_subdiv2d_next_edge(const cvk_subdiv2d_t *h, int edge);

/** cv::Subdiv2D::rotateEdge(edge, rotate). */
int cvk_subdiv2d_rotate_edge(const cvk_subdiv2d_t *h, int edge, int rotate);

/** cv::Subdiv2D::symEdge(edge). */
int cvk_subdiv2d_sym_edge(const cvk_subdiv2d_t *h, int edge);

/** cv::Subdiv2D::edgeOrg(edge, orgpt); returns the vertex id. */
int cvk_subdiv2d_edge_org(const cvk_subdiv2d_t *h, int edge,
                          double *out_x, double *out_y);

/** cv::Subdiv2D::edgeDst(edge, dstpt); returns the vertex id. */
int cvk_subdiv2d_edge_dst(const cvk_subdiv2d_t *h, int edge,
                          double *out_x, double *out_y);

/** Frees a Subdiv2D handle (exactly once). */
void cvk_subdiv2d_release(cvk_subdiv2d_t *h);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_GEOMETRY_H */

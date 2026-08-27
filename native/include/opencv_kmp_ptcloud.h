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
 * cvk_ C ABI declarations for the OpenCV "ptcloud" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * Mat-valued results are always fresh handles owned by the caller; functions
 * with two or more Mat outputs write caller-owned handles through `cvk_mat_t
 * **` out-params (NULL outputs are skipped where documented).
 */
#ifndef OPENCV_KMP_PTCLOUD_H
#define OPENCV_KMP_PTCLOUD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvk_octree cvk_octree_t;
typedef struct cvk_odometry cvk_odometry_t;
typedef struct cvk_odometry_frame cvk_odometry_frame_t;
typedef struct cvk_rgbd_normals cvk_rgbd_normals_t;
typedef struct cvk_volume cvk_volume_t;
typedef struct cvk_pose_graph cvk_pose_graph_t;

/* =========================================================================
 * Octree (cv::Ptr<cv::Octree>)
 * ========================================================================= */

/** cv::Octree::createWithDepth(maxDepth, size, origin, withColors). */
cvk_octree_t *cvk_octree_create_with_depth(int max_depth, double size, double origin_x,
                                           double origin_y, double origin_z, int with_colors);

/** cv::Octree::createWithDepth(maxDepth, pointCloud, colors); colors may be NULL. */
cvk_octree_t *cvk_octree_create_with_depth_cloud(int max_depth, const cvk_mat_t *point_cloud,
                                                 const cvk_mat_t *colors);

/** cv::Octree::createWithResolution(resolution, size, origin, withColors). */
cvk_octree_t *cvk_octree_create_with_resolution(double resolution, double size,
                                                double origin_x, double origin_y,
                                                double origin_z, int with_colors);

/** cv::Octree::createWithResolution(resolution, pointCloud, colors); colors may be NULL. */
cvk_octree_t *cvk_octree_create_with_resolution_cloud(double resolution,
                                                      const cvk_mat_t *point_cloud,
                                                      const cvk_mat_t *colors);

int cvk_octree_insert_point(cvk_octree_t *h, double x, double y, double z);
int cvk_octree_insert_point_color(cvk_octree_t *h, double x, double y, double z, double cx,
                                  double cy, double cz);
int cvk_octree_is_point_in_bound(const cvk_octree_t *h, double x, double y, double z);
int cvk_octree_empty(const cvk_octree_t *h);
void cvk_octree_clear(cvk_octree_t *h);
int cvk_octree_delete_point(cvk_octree_t *h, double x, double y, double z);

/** cv::Octree::getPointCloudByOctree; single-Mat variant. */
cvk_mat_t *cvk_octree_get_point_cloud(cvk_octree_t *h);

/** cv::Octree::getPointCloudByOctree(points, colors); writes both handles. */
void cvk_octree_get_point_cloud_color(cvk_octree_t *h, cvk_mat_t **points, cvk_mat_t **colors);

/** cv::Octree::radiusNNSearch; square_dists may be NULL. Returns point count. */
int cvk_octree_radius_nn_search(const cvk_octree_t *h, double qx, double qy, double qz,
                                float radius, cvk_mat_t **points, cvk_mat_t **square_dists);

/** cv::Octree::radiusNNSearch with colors output. */
int cvk_octree_radius_nn_search_color(const cvk_octree_t *h, double qx, double qy, double qz,
                                      float radius, cvk_mat_t **points, cvk_mat_t **colors,
                                      cvk_mat_t **square_dists);

/** cv::Octree::KNNSearch; square_dists may be NULL. */
void cvk_octree_knn_search(const cvk_octree_t *h, double qx, double qy, double qz, int k,
                           cvk_mat_t **points, cvk_mat_t **square_dists);

/** cv::Octree::KNNSearch with colors output. */
void cvk_octree_knn_search_color(const cvk_octree_t *h, double qx, double qy, double qz, int k,
                                 cvk_mat_t **points, cvk_mat_t **colors,
                                 cvk_mat_t **square_dists);

void cvk_octree_release(cvk_octree_t *h);

/* =========================================================================
 * Odometry / OdometryFrame (raw cv::Odometry / cv::OdometryFrame pointers)
 * ========================================================================= */

/** cv::Odometry() / cv::Odometry(otype). */
cvk_odometry_t *cvk_odometry_create(void);
cvk_odometry_t *cvk_odometry_create_type(int otype);

/**
 * cv::Odometry(otype, settings, algtype) with the OdometrySettings fields
 * expanded. Mat-valued settings (cameraMatrix, iterCounts,
 * minGradientMagnitudes) may be NULL to keep the C++ defaults.
 */
cvk_odometry_t *cvk_odometry_create_settings(
    int otype, const cvk_mat_t *camera_matrix, const cvk_mat_t *iter_counts, float min_depth,
    float max_depth, float max_depth_diff, float max_points_part, int sobel_size,
    double sobel_scale, int normal_win_size, float normal_diff_threshold, int normal_method,
    float angle_threshold, float max_translation, float max_rotation,
    float min_gradient_magnitude, const cvk_mat_t *min_gradient_magnitudes, int algtype);

void cvk_odometry_prepare_frame(const cvk_odometry_t *h, cvk_odometry_frame_t *frame);
void cvk_odometry_prepare_frames(const cvk_odometry_t *h, cvk_odometry_frame_t *src_frame,
                                 cvk_odometry_frame_t *dst_frame);

/** compute(OdometryFrame, OdometryFrame, Rt); writes rt; returns success. */
int cvk_odometry_compute_frames(const cvk_odometry_t *h, const cvk_odometry_frame_t *src_frame,
                                const cvk_odometry_frame_t *dst_frame, cvk_mat_t **rt);

/** compute(Mat srcDepth, Mat dstDepth, Rt); writes rt; returns success. */
int cvk_odometry_compute_depth(const cvk_odometry_t *h, const cvk_mat_t *src_depth,
                               const cvk_mat_t *dst_depth, cvk_mat_t **rt);

/** compute(Mat srcDepth, Mat srcRGB, Mat dstDepth, Mat dstRGB, Rt). */
int cvk_odometry_compute_rgbd(const cvk_odometry_t *h, const cvk_mat_t *src_depth,
                              const cvk_mat_t *src_rgb, const cvk_mat_t *dst_depth,
                              const cvk_mat_t *dst_rgb, cvk_mat_t **rt);

/** cv::Odometry::getNormalsComputer; new refcounted handle (may be NULL). */
cvk_rgbd_normals_t *cvk_odometry_get_normals_computer(const cvk_odometry_t *h);

void cvk_odometry_release(cvk_odometry_t *h);

/** cv::OdometryFrame(depth, image, mask, normals); NULL args mean empty Mats. */
cvk_odometry_frame_t *cvk_odometry_frame_create(const cvk_mat_t *depth, const cvk_mat_t *image,
                                                const cvk_mat_t *mask, const cvk_mat_t *normals);

cvk_mat_t *cvk_odometry_frame_get_image(const cvk_odometry_frame_t *h);
cvk_mat_t *cvk_odometry_frame_get_gray_image(const cvk_odometry_frame_t *h);
cvk_mat_t *cvk_odometry_frame_get_depth(const cvk_odometry_frame_t *h);
cvk_mat_t *cvk_odometry_frame_get_processed_depth(const cvk_odometry_frame_t *h);
cvk_mat_t *cvk_odometry_frame_get_mask(const cvk_odometry_frame_t *h);
cvk_mat_t *cvk_odometry_frame_get_normals(const cvk_odometry_frame_t *h);
int cvk_odometry_frame_get_pyramid_levels(const cvk_odometry_frame_t *h);
cvk_mat_t *cvk_odometry_frame_get_pyramid_at(const cvk_odometry_frame_t *h, int pyr_type,
                                             unsigned long long level);

void cvk_odometry_frame_release(cvk_odometry_frame_t *h);

/* =========================================================================
 * RgbdNormals (cv::Ptr<cv::RgbdNormals>)
 * ========================================================================= */

/** cv::RgbdNormals::create(rows, cols, depth, K, window_size, diff_threshold, method). */
cvk_rgbd_normals_t *cvk_rgbd_normals_create(int rows, int cols, int depth,
                                            const cvk_mat_t *k, int window_size,
                                            float diff_threshold, int method);

/** cv::RgbdNormals::apply; returns a new normals Mat. */
cvk_mat_t *cvk_rgbd_normals_apply(const cvk_rgbd_normals_t *h, const cvk_mat_t *points);

void cvk_rgbd_normals_cache(const cvk_rgbd_normals_t *h);
int cvk_rgbd_normals_get_rows(const cvk_rgbd_normals_t *h);
void cvk_rgbd_normals_set_rows(cvk_rgbd_normals_t *h, int v);
int cvk_rgbd_normals_get_cols(const cvk_rgbd_normals_t *h);
void cvk_rgbd_normals_set_cols(cvk_rgbd_normals_t *h, int v);
int cvk_rgbd_normals_get_window_size(const cvk_rgbd_normals_t *h);
void cvk_rgbd_normals_set_window_size(cvk_rgbd_normals_t *h, int v);
int cvk_rgbd_normals_get_depth(const cvk_rgbd_normals_t *h);
cvk_mat_t *cvk_rgbd_normals_get_k(const cvk_rgbd_normals_t *h);
void cvk_rgbd_normals_set_k(cvk_rgbd_normals_t *h, const cvk_mat_t *k);
int cvk_rgbd_normals_get_method(const cvk_rgbd_normals_t *h);

void cvk_rgbd_normals_release(cvk_rgbd_normals_t *h);

/* =========================================================================
 * Volume (raw cv::Volume pointer) / VolumeSettings (expanded args)
 * ========================================================================= */

/** cv::Volume(vtype) with default settings for that volume type. */
cvk_volume_t *cvk_volume_create(int vtype);

/**
 * cv::Volume(vtype, settings) with the VolumeSettings fields expanded.
 * Mat-valued settings (volumePose, volumeResolution, cameraIntegrateIntrinsics,
 * cameraRaycastIntrinsics) may be NULL to keep the C++ defaults.
 */
cvk_volume_t *cvk_volume_create_settings(
    int vtype, int integrate_width, int integrate_height, int raycast_width,
    int raycast_height, float depth_factor, float voxel_size, float tsdf_truncate_distance,
    float max_depth, int max_weight, float raycast_step_factor, const cvk_mat_t *volume_pose,
    const cvk_mat_t *volume_resolution, const cvk_mat_t *camera_integrate_intrinsics,
    const cvk_mat_t *camera_raycast_intrinsics);

void cvk_volume_integrate_frame(cvk_volume_t *h, const cvk_odometry_frame_t *frame,
                                const cvk_mat_t *pose);
void cvk_volume_integrate(cvk_volume_t *h, const cvk_mat_t *depth, const cvk_mat_t *pose);
void cvk_volume_integrate_color(cvk_volume_t *h, const cvk_mat_t *depth,
                                const cvk_mat_t *image, const cvk_mat_t *pose);

/** cv::Volume::raycast; writes points and normals handles. */
void cvk_volume_raycast(const cvk_volume_t *h, const cvk_mat_t *camera_pose,
                        cvk_mat_t **points, cvk_mat_t **normals);

/** cv::Volume::raycast(points, normals, colors); writes all three handles. */
void cvk_volume_raycast_color(const cvk_volume_t *h, const cvk_mat_t *camera_pose,
                              cvk_mat_t **points, cvk_mat_t **normals, cvk_mat_t **colors);

/** cv::Volume::raycast(cameraPose, height, width, K, points, normals). */
void cvk_volume_raycast_ex(const cvk_volume_t *h, const cvk_mat_t *camera_pose, int height,
                           int width, const cvk_mat_t *k, cvk_mat_t **points,
                           cvk_mat_t **normals);

/** cv::Volume::raycast(cameraPose, height, width, K, points, normals, colors). */
void cvk_volume_raycast_ex_color(const cvk_volume_t *h, const cvk_mat_t *camera_pose,
                                 int height, int width, const cvk_mat_t *k, cvk_mat_t **points,
                                 cvk_mat_t **normals, cvk_mat_t **colors);

/** cv::Volume::fetchNormals(points, normals); returns a new normals Mat. */
cvk_mat_t *cvk_volume_fetch_normals(const cvk_volume_t *h, const cvk_mat_t *points);

void cvk_volume_fetch_points_normals(const cvk_volume_t *h, cvk_mat_t **points,
                                     cvk_mat_t **normals);
void cvk_volume_fetch_points_normals_colors(const cvk_volume_t *h, cvk_mat_t **points,
                                            cvk_mat_t **normals, cvk_mat_t **colors);

void cvk_volume_reset(cvk_volume_t *h);
int cvk_volume_get_visible_blocks(const cvk_volume_t *h);
unsigned long long cvk_volume_get_total_volume_units(const cvk_volume_t *h);
cvk_mat_t *cvk_volume_get_bounding_box(const cvk_volume_t *h, int precision);
void cvk_volume_set_enable_growth(cvk_volume_t *h, int v);
int cvk_volume_get_enable_growth(const cvk_volume_t *h);

void cvk_volume_release(cvk_volume_t *h);

/* =========================================================================
 * PoseGraph (cv::Ptr<cv::detail::PoseGraph>)
 * ========================================================================= */

/** cv::detail::PoseGraph::create(). */
cvk_pose_graph_t *cvk_pose_graph_create(void);

/** addNode(nodeId, 4x4 pose Mat (32F/64F), fixed). */
void cvk_pose_graph_add_node(cvk_pose_graph_t *h, unsigned long long node_id,
                             const cvk_mat_t *pose, int fixed);

/** addEdge(source, target, 4x4 transformation Mat, 6x6 information Mat or NULL). */
void cvk_pose_graph_add_edge(cvk_pose_graph_t *h, unsigned long long source,
                             unsigned long long target, const cvk_mat_t *transformation,
                             const cvk_mat_t *information);

/** cv::detail::PoseGraph::optimize(); returns iterations elapsed, -1 on failure. */
int cvk_pose_graph_optimize(cvk_pose_graph_t *h);

/** getNodePose(nodeId); returns a new 4x4 CV_64F Mat. */
cvk_mat_t *cvk_pose_graph_get_pose(const cvk_pose_graph_t *h, unsigned long long node_id);

void cvk_pose_graph_release(cvk_pose_graph_t *h);

/* =========================================================================
 * Ptcloud free functions (org.opencv.ptcloud.Ptcloud parity)
 * ========================================================================= */

/** cv::loadPointCloud; writes vertices (normals/rgb may be NULL). */
void cvk_ptcloud_load_point_cloud(const char *filename, cvk_mat_t **vertices,
                                  cvk_mat_t **normals, cvk_mat_t **rgb);

/** cv::savePointCloud; normals/rgb may be NULL. */
void cvk_ptcloud_save_point_cloud(const char *filename, const cvk_mat_t *vertices,
                                  const cvk_mat_t *normals, const cvk_mat_t *rgb);

/**
 * cv::loadMesh. `indices` is a dynamic array of Mat handles (one per face)
 * allocated with new[]; its length goes to *indices_count. Free the array
 * with cvk_free_mat_array (the Mats themselves are freed by cvk_mat_release).
 * normals/colors/tex_coords may be NULL.
 */
void cvk_ptcloud_load_mesh(const char *filename, cvk_mat_t **vertices, cvk_mat_t ***indices,
                           int *indices_count, cvk_mat_t **normals, cvk_mat_t **colors,
                           cvk_mat_t **tex_coords);

/** cv::saveMesh; normals/colors/tex_coords may be NULL. */
void cvk_ptcloud_save_mesh(const char *filename, const cvk_mat_t *vertices,
                           const cvk_mat_t *const *indices, int indices_count,
                           const cvk_mat_t *normals, const cvk_mat_t *colors,
                           const cvk_mat_t *tex_coords);

/** Frees a cvk_mat_t* array produced by cvk_ptcloud_load_mesh (not the Mats). */
void cvk_free_mat_array(cvk_mat_t **arr);

/** cv::triangleRasterize; settings expanded (shading/culling/glCompatible). */
void cvk_ptcloud_triangle_rasterize(const cvk_mat_t *vertices, const cvk_mat_t *indices,
                                    const cvk_mat_t *colors, cvk_mat_t *color_buf,
                                    cvk_mat_t *depth_buf, const cvk_mat_t *world2cam,
                                    double fov_y, double z_near, double z_far, int shading_type,
                                    int culling_mode, int gl_compatible_mode);

/** cv::triangleRasterizeDepth; settings expanded. */
void cvk_ptcloud_triangle_rasterize_depth(const cvk_mat_t *vertices, const cvk_mat_t *indices,
                                          cvk_mat_t *depth_buf, const cvk_mat_t *world2cam,
                                          double fov_y, double z_near, double z_far,
                                          int shading_type, int culling_mode,
                                          int gl_compatible_mode);

/** cv::triangleRasterizeColor; settings expanded. */
void cvk_ptcloud_triangle_rasterize_color(const cvk_mat_t *vertices, const cvk_mat_t *indices,
                                          const cvk_mat_t *colors, cvk_mat_t *color_buf,
                                          const cvk_mat_t *world2cam, double fov_y,
                                          double z_near, double z_far, int shading_type,
                                          int culling_mode, int gl_compatible_mode);

/** cv::registerDepth; writes registered_depth; returns success. */
int cvk_ptcloud_register_depth(const cvk_mat_t *unregistered_camera_matrix,
                               const cvk_mat_t *registered_camera_matrix,
                               const cvk_mat_t *registered_dist_coeffs, const cvk_mat_t *rt,
                               const cvk_mat_t *unregistered_depth, int output_width,
                               int output_height, int depth_dilation, cvk_mat_t **registered_depth);

/** cv::depthTo3dSparse; returns a new points3d Mat (4-channel). */
cvk_mat_t *cvk_ptcloud_depth_to_3d_sparse(const cvk_mat_t *depth, const cvk_mat_t *in_k,
                                          const cvk_mat_t *in_points);

/** cv::depthTo3d; mask may be NULL; returns a new points3d Mat. */
cvk_mat_t *cvk_ptcloud_depth_to_3d(const cvk_mat_t *depth, const cvk_mat_t *k,
                                   const cvk_mat_t *mask);

/** cv::rescaleDepth; returns a new out Mat. */
cvk_mat_t *cvk_ptcloud_rescale_depth(const cvk_mat_t *in, int type, double depth_factor);

/** cv::warpFrame; writes warped depth/image/mask handles (any may be NULL). */
void cvk_ptcloud_warp_frame(const cvk_mat_t *depth, const cvk_mat_t *image,
                            const cvk_mat_t *mask, const cvk_mat_t *rt,
                            const cvk_mat_t *camera_matrix, cvk_mat_t **warped_depth,
                            cvk_mat_t **warped_image, cvk_mat_t **warped_mask);

/** cv::findPlanes; writes mask and plane_coefficients handles. */
void cvk_ptcloud_find_planes(const cvk_mat_t *points3d, const cvk_mat_t *normals,
                             cvk_mat_t **mask, cvk_mat_t **plane_coefficients, int block_size,
                             int min_size, double threshold, double sensor_error_a,
                             double sensor_error_b, double sensor_error_c, int method);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_PTCLOUD_H */

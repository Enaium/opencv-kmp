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
 * JNI bridge for the ptcloud module: thin wrappers around the cvk_ C ABI in
 * native/shim_ptcloud.cpp. Every function maps to a member of the Kotlin
 * `internal object JniPtcloud`.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_ptcloud.h"

#include <cstdint>
#include <vector>

static inline cvk_mat_t *as_mat(jlong h) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(h));
}

static inline jlong as_handle(const cvk_mat_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}

static inline cvk_octree_t *as_octree(jlong h) {
    return reinterpret_cast<cvk_octree_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_odometry_t *as_odometry(jlong h) {
    return reinterpret_cast<cvk_odometry_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_odometry_frame_t *as_frame(jlong h) {
    return reinterpret_cast<cvk_odometry_frame_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_rgbd_normals_t *as_normals(jlong h) {
    return reinterpret_cast<cvk_rgbd_normals_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_volume_t *as_volume(jlong h) {
    return reinterpret_cast<cvk_volume_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_pose_graph_t *as_pose_graph(jlong h) {
    return reinterpret_cast<cvk_pose_graph_t *>(static_cast<uintptr_t>(h));
}

static inline jlong as_octree_handle(const cvk_octree_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_odometry_handle(const cvk_odometry_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_frame_handle(const cvk_odometry_frame_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_normals_handle(const cvk_rgbd_normals_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_volume_handle(const cvk_volume_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_pose_graph_handle(const cvk_pose_graph_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static jlongArray handles_array(JNIEnv *env, const jlong *vals, jsize count) {
    jlongArray arr = env->NewLongArray(count);
    if (arr == nullptr) return nullptr;
    env->SetLongArrayRegion(arr, 0, count, vals);
    return arr;
}

extern "C" {

/* ---- Octree ------------------------------------------------------------ */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeCreateWithDepth(JNIEnv *, jobject, jint maxDepth,
                                                        jdouble size, jdouble ox, jdouble oy,
                                                        jdouble oz, jboolean withColors) {
    return as_octree_handle(cvk_octree_create_with_depth(maxDepth, size, ox, oy, oz,
                                                         withColors ? 1 : 0));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeCreateWithDepthCloud(JNIEnv *, jobject, jint maxDepth,
                                                             jlong pointCloud, jlong colors) {
    return as_octree_handle(cvk_octree_create_with_depth_cloud(
        maxDepth, as_mat(pointCloud), colors != 0 ? as_mat(colors) : nullptr));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeCreateWithResolution(JNIEnv *, jobject,
                                                             jdouble resolution, jdouble size,
                                                             jdouble ox, jdouble oy, jdouble oz,
                                                             jboolean withColors) {
    return as_octree_handle(cvk_octree_create_with_resolution(resolution, size, ox, oy, oz,
                                                              withColors ? 1 : 0));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeCreateWithResolutionCloud(JNIEnv *, jobject,
                                                                  jdouble resolution,
                                                                  jlong pointCloud,
                                                                  jlong colors) {
    return as_octree_handle(cvk_octree_create_with_resolution_cloud(
        resolution, as_mat(pointCloud), colors != 0 ? as_mat(colors) : nullptr));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeInsertPoint(JNIEnv *, jobject, jlong h, jdouble x,
                                                    jdouble y, jdouble z) {
    return cvk_octree_insert_point(as_octree(h), x, y, z) != 0;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeInsertPointColor(JNIEnv *, jobject, jlong h, jdouble x,
                                                         jdouble y, jdouble z, jdouble cx,
                                                         jdouble cy, jdouble cz) {
    return cvk_octree_insert_point_color(as_octree(h), x, y, z, cx, cy, cz) != 0;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeIsPointInBound(JNIEnv *, jobject, jlong h, jdouble x,
                                                       jdouble y, jdouble z) {
    return cvk_octree_is_point_in_bound(as_octree(h), x, y, z) != 0;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeEmpty(JNIEnv *, jobject, jlong h) {
    return cvk_octree_empty(as_octree(h)) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeClear(JNIEnv *, jobject, jlong h) {
    cvk_octree_clear(as_octree(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeDeletePoint(JNIEnv *, jobject, jlong h, jdouble x,
                                                    jdouble y, jdouble z) {
    return cvk_octree_delete_point(as_octree(h), x, y, z) != 0;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeGetPointCloud(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_octree_get_point_cloud(as_octree(h)));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeGetPointCloudColor(JNIEnv *env, jobject, jlong h) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *colors = nullptr;
    cvk_octree_get_point_cloud_color(as_octree(h), &points, &colors);
    jlong vals[2] = {as_handle(points), as_handle(colors)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeRadiusNNSearch(JNIEnv *env, jobject, jlong h, jdouble qx,
                                                       jdouble qy, jdouble qz, jfloat radius) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *dists = nullptr;
    const int count = cvk_octree_radius_nn_search(as_octree(h), qx, qy, qz, radius, &points,
                                                  &dists);
    jlong vals[3] = {count, as_handle(points), as_handle(dists)};
    return handles_array(env, vals, 3);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeRadiusNNSearchColor(JNIEnv *env, jobject, jlong h,
                                                            jdouble qx, jdouble qy, jdouble qz,
                                                            jfloat radius) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *colors = nullptr;
    cvk_mat_t *dists = nullptr;
    const int count = cvk_octree_radius_nn_search_color(as_octree(h), qx, qy, qz, radius,
                                                        &points, &colors, &dists);
    jlong vals[4] = {count, as_handle(points), as_handle(colors), as_handle(dists)};
    return handles_array(env, vals, 4);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeKNNSearch(JNIEnv *env, jobject, jlong h, jdouble qx,
                                                  jdouble qy, jdouble qz, jint k) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *dists = nullptr;
    cvk_octree_knn_search(as_octree(h), qx, qy, qz, k, &points, &dists);
    jlong vals[2] = {as_handle(points), as_handle(dists)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeKNNSearchColor(JNIEnv *env, jobject, jlong h, jdouble qx,
                                                       jdouble qy, jdouble qz, jint k) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *colors = nullptr;
    cvk_mat_t *dists = nullptr;
    cvk_octree_knn_search_color(as_octree(h), qx, qy, qz, k, &points, &colors, &dists);
    jlong vals[3] = {as_handle(points), as_handle(colors), as_handle(dists)};
    return handles_array(env, vals, 3);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_octreeRelease(JNIEnv *, jobject, jlong h) {
    cvk_octree_release(as_octree(h));
}

/* ---- Odometry ---------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryCreate(JNIEnv *, jobject) {
    return as_odometry_handle(cvk_odometry_create());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryCreateType(JNIEnv *, jobject, jint otype) {
    return as_odometry_handle(cvk_odometry_create_type(otype));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryCreateSettings(
    JNIEnv *, jobject, jint otype, jlong cameraMatrix, jlong iterCounts, jfloat minDepth,
    jfloat maxDepth, jfloat maxDepthDiff, jfloat maxPointsPart, jint sobelSize,
    jdouble sobelScale, jint normalWinSize, jfloat normalDiffThreshold, jint normalMethod,
    jfloat angleThreshold, jfloat maxTranslation, jfloat maxRotation,
    jfloat minGradientMagnitude, jlong minGradientMagnitudes, jint algtype) {
    return as_odometry_handle(cvk_odometry_create_settings(
        otype, cameraMatrix != 0 ? as_mat(cameraMatrix) : nullptr,
        iterCounts != 0 ? as_mat(iterCounts) : nullptr, minDepth, maxDepth, maxDepthDiff,
        maxPointsPart, sobelSize, sobelScale, normalWinSize, normalDiffThreshold,
        normalMethod, angleThreshold, maxTranslation, maxRotation, minGradientMagnitude,
        minGradientMagnitudes != 0 ? as_mat(minGradientMagnitudes) : nullptr, algtype));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryPrepareFrame(JNIEnv *, jobject, jlong h, jlong frame) {
    cvk_odometry_prepare_frame(as_odometry(h), as_frame(frame));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryPrepareFrames(JNIEnv *, jobject, jlong h, jlong src,
                                                        jlong dst) {
    cvk_odometry_prepare_frames(as_odometry(h), as_frame(src), as_frame(dst));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryComputeFrames(JNIEnv *env, jobject, jlong h, jlong src,
                                                        jlong dst) {
    cvk_mat_t *rt = nullptr;
    const int ok = cvk_odometry_compute_frames(as_odometry(h), as_frame(src), as_frame(dst),
                                               &rt);
    jlong vals[2] = {ok != 0 ? 1 : 0, as_handle(rt)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryComputeDepth(JNIEnv *env, jobject, jlong h,
                                                       jlong srcDepth, jlong dstDepth) {
    cvk_mat_t *rt = nullptr;
    const int ok = cvk_odometry_compute_depth(as_odometry(h), as_mat(srcDepth),
                                              as_mat(dstDepth), &rt);
    jlong vals[2] = {ok != 0 ? 1 : 0, as_handle(rt)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryComputeRgbd(JNIEnv *env, jobject, jlong h,
                                                      jlong srcDepth, jlong srcRgb,
                                                      jlong dstDepth, jlong dstRgb) {
    cvk_mat_t *rt = nullptr;
    const int ok = cvk_odometry_compute_rgbd(as_odometry(h), as_mat(srcDepth), as_mat(srcRgb),
                                             as_mat(dstDepth), as_mat(dstRgb), &rt);
    jlong vals[2] = {ok != 0 ? 1 : 0, as_handle(rt)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryGetNormalsComputer(JNIEnv *, jobject, jlong h) {
    return as_normals_handle(cvk_odometry_get_normals_computer(as_odometry(h)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryRelease(JNIEnv *, jobject, jlong h) {
    cvk_odometry_release(as_odometry(h));
}

/* ---- OdometryFrame ----------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryFrameCreate(JNIEnv *, jobject, jlong depth,
                                                      jlong image, jlong mask, jlong normals) {
    return as_frame_handle(cvk_odometry_frame_create(
        depth != 0 ? as_mat(depth) : nullptr, image != 0 ? as_mat(image) : nullptr,
        mask != 0 ? as_mat(mask) : nullptr, normals != 0 ? as_mat(normals) : nullptr));
}

#define FRAME_GETTER(java_suffix, c_suffix)                                       \
    JNIEXPORT jlong JNICALL                                                       \
    Java_cn_enaium_opencv_JniPtcloud_odometryFrameGet##java_suffix(JNIEnv *,      \
                                                                    jobject,       \
                                                                    jlong h) {     \
        return as_handle(cvk_odometry_frame_get_##c_suffix(as_frame(h)));         \
    }

FRAME_GETTER(Image, image)
FRAME_GETTER(GrayImage, gray_image)
FRAME_GETTER(Depth, depth)
FRAME_GETTER(ProcessedDepth, processed_depth)
FRAME_GETTER(Mask, mask)
FRAME_GETTER(Normals, normals)

#undef FRAME_GETTER

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryFrameGetPyramidLevels(JNIEnv *, jobject, jlong h) {
    return cvk_odometry_frame_get_pyramid_levels(as_frame(h));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryFrameGetPyramidAt(JNIEnv *, jobject, jlong h,
                                                            jint pyrType, jlong level) {
    return as_handle(cvk_odometry_frame_get_pyramid_at(as_frame(h), pyrType, level));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_odometryFrameRelease(JNIEnv *, jobject, jlong h) {
    cvk_odometry_frame_release(as_frame(h));
}

/* ---- RgbdNormals ------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsCreate(JNIEnv *, jobject, jint rows, jint cols,
                                                    jint depth, jlong k, jint windowSize,
                                                    jfloat diffThreshold, jint method) {
    return as_normals_handle(cvk_rgbd_normals_create(rows, cols, depth,
                                                     k != 0 ? as_mat(k) : nullptr, windowSize,
                                                     diffThreshold, method));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsApply(JNIEnv *, jobject, jlong h, jlong points) {
    return as_handle(cvk_rgbd_normals_apply(as_normals(h), as_mat(points)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsCache(JNIEnv *, jobject, jlong h) {
    cvk_rgbd_normals_cache(as_normals(h));
}

#define NORMALS_GET_SET(java_suffix, c_suffix)                                     \
    JNIEXPORT jint JNICALL                                                         \
    Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsGet##java_suffix(JNIEnv *, jobject, \
                                                                 jlong h) {         \
        return cvk_rgbd_normals_get_##c_suffix(as_normals(h));                     \
    }                                                                               \
    JNIEXPORT void JNICALL                                                          \
    Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsSet##java_suffix(JNIEnv *, jobject, \
                                                                 jlong h, jint v) {  \
        cvk_rgbd_normals_set_##c_suffix(as_normals(h), v);                         \
    }

NORMALS_GET_SET(Rows, rows)
NORMALS_GET_SET(Cols, cols)
NORMALS_GET_SET(WindowSize, window_size)

#undef NORMALS_GET_SET

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsGetDepth(JNIEnv *, jobject, jlong h) {
    return cvk_rgbd_normals_get_depth(as_normals(h));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsGetK(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_rgbd_normals_get_k(as_normals(h)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsSetK(JNIEnv *, jobject, jlong h, jlong k) {
    cvk_rgbd_normals_set_k(as_normals(h), as_mat(k));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsGetMethod(JNIEnv *, jobject, jlong h) {
    return cvk_rgbd_normals_get_method(as_normals(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_rgbdNormalsRelease(JNIEnv *, jobject, jlong h) {
    cvk_rgbd_normals_release(as_normals(h));
}

/* ---- Volume ------------------------------------------------------------ */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeCreate(JNIEnv *, jobject, jint vtype) {
    return as_volume_handle(cvk_volume_create(vtype));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeCreateSettings(
    JNIEnv *, jobject, jint vtype, jint integrateWidth, jint integrateHeight, jint raycastWidth,
    jint raycastHeight, jfloat depthFactor, jfloat voxelSize, jfloat tsdfTruncateDistance,
    jfloat maxDepth, jint maxWeight, jfloat raycastStepFactor, jlong volumePose,
    jlong volumeResolution, jlong cameraIntegrateIntrinsics, jlong cameraRaycastIntrinsics) {
    return as_volume_handle(cvk_volume_create_settings(
        vtype, integrateWidth, integrateHeight, raycastWidth, raycastHeight, depthFactor,
        voxelSize, tsdfTruncateDistance, maxDepth, maxWeight, raycastStepFactor,
        volumePose != 0 ? as_mat(volumePose) : nullptr,
        volumeResolution != 0 ? as_mat(volumeResolution) : nullptr,
        cameraIntegrateIntrinsics != 0 ? as_mat(cameraIntegrateIntrinsics) : nullptr,
        cameraRaycastIntrinsics != 0 ? as_mat(cameraRaycastIntrinsics) : nullptr));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeIntegrateFrame(JNIEnv *, jobject, jlong h, jlong frame,
                                                       jlong pose) {
    cvk_volume_integrate_frame(as_volume(h), as_frame(frame), as_mat(pose));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeIntegrate(JNIEnv *, jobject, jlong h, jlong depth,
                                                  jlong pose) {
    cvk_volume_integrate(as_volume(h), as_mat(depth), as_mat(pose));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeIntegrateColor(JNIEnv *, jobject, jlong h, jlong depth,
                                                       jlong image, jlong pose) {
    cvk_volume_integrate_color(as_volume(h), as_mat(depth), as_mat(image), as_mat(pose));
}

#define VOLUME_2OUT(java_suffix, c_expr)                                             \
    JNIEXPORT jlongArray JNICALL                                                     \
    Java_cn_enaium_opencv_JniPtcloud_##java_suffix(JNIEnv *env, jobject, jlong h,    \
                                                   jlong cameraPose) {               \
        cvk_mat_t *points = nullptr;                                                 \
        cvk_mat_t *normals = nullptr;                                                \
        c_expr;                                                                      \
        jlong vals[2] = {as_handle(points), as_handle(normals)};                     \
        return handles_array(env, vals, 2);                                          \
    }

VOLUME_2OUT(volumeRaycast, cvk_volume_raycast(as_volume(h), as_mat(cameraPose), &points, &normals))

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeRaycastColor(JNIEnv *env, jobject, jlong h,
                                                     jlong cameraPose) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *normals = nullptr;
    cvk_mat_t *colors = nullptr;
    cvk_volume_raycast_color(as_volume(h), as_mat(cameraPose), &points, &normals, &colors);
    jlong vals[3] = {as_handle(points), as_handle(normals), as_handle(colors)};
    return handles_array(env, vals, 3);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeRaycastEx(JNIEnv *env, jobject, jlong h,
                                                  jlong cameraPose, jint height, jint width,
                                                  jlong k) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *normals = nullptr;
    cvk_volume_raycast_ex(as_volume(h), as_mat(cameraPose), height, width, as_mat(k), &points,
                          &normals);
    jlong vals[2] = {as_handle(points), as_handle(normals)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeRaycastExColor(JNIEnv *env, jobject, jlong h,
                                                       jlong cameraPose, jint height,
                                                       jint width, jlong k) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *normals = nullptr;
    cvk_mat_t *colors = nullptr;
    cvk_volume_raycast_ex_color(as_volume(h), as_mat(cameraPose), height, width, as_mat(k),
                                &points, &normals, &colors);
    jlong vals[3] = {as_handle(points), as_handle(normals), as_handle(colors)};
    return handles_array(env, vals, 3);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeFetchNormals(JNIEnv *, jobject, jlong h, jlong points) {
    return as_handle(cvk_volume_fetch_normals(as_volume(h), as_mat(points)));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeFetchPointsNormals(JNIEnv *env, jobject, jlong h) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *normals = nullptr;
    cvk_volume_fetch_points_normals(as_volume(h), &points, &normals);
    jlong vals[2] = {as_handle(points), as_handle(normals)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeFetchPointsNormalsColors(JNIEnv *env, jobject,
                                                                 jlong h) {
    cvk_mat_t *points = nullptr;
    cvk_mat_t *normals = nullptr;
    cvk_mat_t *colors = nullptr;
    cvk_volume_fetch_points_normals_colors(as_volume(h), &points, &normals, &colors);
    jlong vals[3] = {as_handle(points), as_handle(normals), as_handle(colors)};
    return handles_array(env, vals, 3);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeReset(JNIEnv *, jobject, jlong h) {
    cvk_volume_reset(as_volume(h));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeGetVisibleBlocks(JNIEnv *, jobject, jlong h) {
    return cvk_volume_get_visible_blocks(as_volume(h));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeGetTotalVolumeUnits(JNIEnv *, jobject, jlong h) {
    return static_cast<jlong>(cvk_volume_get_total_volume_units(as_volume(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeGetBoundingBox(JNIEnv *, jobject, jlong h,
                                                       jint precision) {
    return as_handle(cvk_volume_get_bounding_box(as_volume(h), precision));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeSetEnableGrowth(JNIEnv *, jobject, jlong h,
                                                        jboolean v) {
    cvk_volume_set_enable_growth(as_volume(h), v ? 1 : 0);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeGetEnableGrowth(JNIEnv *, jobject, jlong h) {
    return cvk_volume_get_enable_growth(as_volume(h)) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_volumeRelease(JNIEnv *, jobject, jlong h) {
    cvk_volume_release(as_volume(h));
}

/* ---- PoseGraph --------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_poseGraphCreate(JNIEnv *, jobject) {
    return as_pose_graph_handle(cvk_pose_graph_create());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_poseGraphAddNode(JNIEnv *, jobject, jlong h, jlong nodeId,
                                                   jlong pose, jboolean fixed) {
    cvk_pose_graph_add_node(as_pose_graph(h), static_cast<unsigned long long>(nodeId),
                            as_mat(pose), fixed ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_poseGraphAddEdge(JNIEnv *, jobject, jlong h, jlong source,
                                                   jlong target, jlong transformation,
                                                   jlong information) {
    cvk_pose_graph_add_edge(as_pose_graph(h), static_cast<unsigned long long>(source),
                            static_cast<unsigned long long>(target), as_mat(transformation),
                            information != 0 ? as_mat(information) : nullptr);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPtcloud_poseGraphOptimize(JNIEnv *, jobject, jlong h) {
    return cvk_pose_graph_optimize(as_pose_graph(h));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_poseGraphGetPose(JNIEnv *, jobject, jlong h, jlong nodeId) {
    return as_handle(cvk_pose_graph_get_pose(as_pose_graph(h),
                                             static_cast<unsigned long long>(nodeId)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_poseGraphRelease(JNIEnv *, jobject, jlong h) {
    cvk_pose_graph_release(as_pose_graph(h));
}

/* ---- Ptcloud free functions -------------------------------------------- */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudLoadPointCloud(JNIEnv *env, jobject, jstring filename) {
    const char *name = env->GetStringUTFChars(filename, nullptr);
    if (name == nullptr) return nullptr;
    cvk_mat_t *vertices = nullptr;
    cvk_mat_t *normals = nullptr;
    cvk_mat_t *rgb = nullptr;
    cvk_ptcloud_load_point_cloud(name, &vertices, &normals, &rgb);
    env->ReleaseStringUTFChars(filename, name);
    jlong vals[3] = {as_handle(vertices), as_handle(normals), as_handle(rgb)};
    return handles_array(env, vals, 3);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudSavePointCloud(JNIEnv *env, jobject, jstring filename,
                                                        jlong vertices, jlong normals,
                                                        jlong rgb) {
    const char *name = env->GetStringUTFChars(filename, nullptr);
    if (name == nullptr) return;
    cvk_ptcloud_save_point_cloud(name, as_mat(vertices),
                                 normals != 0 ? as_mat(normals) : nullptr,
                                 rgb != 0 ? as_mat(rgb) : nullptr);
    env->ReleaseStringUTFChars(filename, name);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudLoadMesh(JNIEnv *env, jobject, jstring filename) {
    const char *name = env->GetStringUTFChars(filename, nullptr);
    if (name == nullptr) return nullptr;
    cvk_mat_t *vertices = nullptr;
    cvk_mat_t *normals = nullptr;
    cvk_mat_t *colors = nullptr;
    cvk_mat_t *tex = nullptr;
    cvk_mat_t **indices = nullptr;
    int count = 0;
    cvk_ptcloud_load_mesh(name, &vertices, &indices, &count, &normals, &colors, &tex);
    env->ReleaseStringUTFChars(filename, name);
    jlongArray arr = env->NewLongArray(4 + count);
    if (arr == nullptr) {
        cvk_free_mat_array(indices);
        return nullptr;
    }
    jlong head[4] = {as_handle(vertices), as_handle(normals), as_handle(colors),
                     as_handle(tex)};
    env->SetLongArrayRegion(arr, 0, 4, head);
    if (count > 0) {
        std::vector<jlong> faces(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            faces[static_cast<size_t>(i)] = as_handle(indices[i]);
        }
        env->SetLongArrayRegion(arr, 4, count, faces.data());
    }
    cvk_free_mat_array(indices);
    return arr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudSaveMesh(JNIEnv *env, jobject, jstring filename,
                                                 jlong vertices, jlongArray indices,
                                                 jlong normals, jlong colors, jlong texCoords) {
    const char *name = env->GetStringUTFChars(filename, nullptr);
    if (name == nullptr) return;
    std::vector<const cvk_mat_t *> faces;
    if (indices != nullptr) {
        const jsize count = env->GetArrayLength(indices);
        jlong *elements = env->GetLongArrayElements(indices, nullptr);
        if (elements != nullptr) {
            faces.reserve(static_cast<size_t>(count));
            for (jsize i = 0; i < count; ++i) {
                faces.push_back(as_mat(elements[i]));
            }
            env->ReleaseLongArrayElements(indices, elements, JNI_ABORT);
        }
    }
    cvk_ptcloud_save_mesh(name, as_mat(vertices),
                          faces.empty() ? nullptr : faces.data(),
                          static_cast<int>(faces.size()),
                          normals != 0 ? as_mat(normals) : nullptr,
                          colors != 0 ? as_mat(colors) : nullptr,
                          texCoords != 0 ? as_mat(texCoords) : nullptr);
    env->ReleaseStringUTFChars(filename, name);
}

#define RASTERIZE_ARGS(env_name, c_args)                                       \
    JNIEXPORT void JNICALL                                                     \
    Java_cn_enaium_opencv_JniPtcloud_##env_name(                               \
        JNIEnv *, jobject, jlong vertices, jlong indices, jlong colors,        \
        jlong colorBuf, jlong depthBuf, jlong world2cam, jdouble fovY,         \
        jdouble zNear, jdouble zFar, jint shadingType, jint cullingMode,       \
        jint glCompatibleMode) {                                               \
        c_args;                                                                \
    }

RASTERIZE_ARGS(ptcloudTriangleRasterize, cvk_ptcloud_triangle_rasterize(
    as_mat(vertices), as_mat(indices), as_mat(colors), as_mat(colorBuf), as_mat(depthBuf),
    as_mat(world2cam), fovY, zNear, zFar, shadingType, cullingMode, glCompatibleMode))

#undef RASTERIZE_ARGS

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudTriangleRasterizeDepth(
    JNIEnv *, jobject, jlong vertices, jlong indices, jlong depthBuf, jlong world2cam,
    jdouble fovY, jdouble zNear, jdouble zFar, jint shadingType, jint cullingMode,
    jint glCompatibleMode) {
    cvk_ptcloud_triangle_rasterize_depth(as_mat(vertices), as_mat(indices), as_mat(depthBuf),
                                         as_mat(world2cam), fovY, zNear, zFar, shadingType,
                                         cullingMode, glCompatibleMode);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudTriangleRasterizeColor(
    JNIEnv *, jobject, jlong vertices, jlong indices, jlong colors, jlong colorBuf,
    jlong world2cam, jdouble fovY, jdouble zNear, jdouble zFar, jint shadingType,
    jint cullingMode, jint glCompatibleMode) {
    cvk_ptcloud_triangle_rasterize_color(as_mat(vertices), as_mat(indices), as_mat(colors),
                                         as_mat(colorBuf), as_mat(world2cam), fovY, zNear,
                                         zFar, shadingType, cullingMode, glCompatibleMode);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudRegisterDepth(JNIEnv *env, jobject,
                                                       jlong unregisteredCameraMatrix,
                                                       jlong registeredCameraMatrix,
                                                       jlong registeredDistCoeffs, jlong rt,
                                                       jlong unregisteredDepth, jint outputWidth,
                                                       jint outputHeight,
                                                       jboolean depthDilation) {
    cvk_mat_t *registered = nullptr;
    const int ok = cvk_ptcloud_register_depth(
        as_mat(unregisteredCameraMatrix), as_mat(registeredCameraMatrix),
        as_mat(registeredDistCoeffs), as_mat(rt), as_mat(unregisteredDepth), outputWidth,
        outputHeight, depthDilation ? 1 : 0, &registered);
    jlong vals[2] = {ok != 0 ? 1 : 0, as_handle(registered)};
    return handles_array(env, vals, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudDepthTo3dSparse(JNIEnv *, jobject, jlong depth,
                                                         jlong inK, jlong inPoints) {
    return as_handle(cvk_ptcloud_depth_to_3d_sparse(as_mat(depth), as_mat(inK),
                                                    as_mat(inPoints)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudDepthTo3d(JNIEnv *, jobject, jlong depth, jlong k,
                                                   jlong mask) {
    return as_handle(cvk_ptcloud_depth_to_3d(as_mat(depth), as_mat(k),
                                             mask != 0 ? as_mat(mask) : nullptr));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudRescaleDepth(JNIEnv *, jobject, jlong in, jint type,
                                                      jdouble depthFactor) {
    return as_handle(cvk_ptcloud_rescale_depth(as_mat(in), type, depthFactor));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudWarpFrame(JNIEnv *env, jobject, jlong depth,
                                                   jlong image, jlong mask, jlong rt,
                                                   jlong cameraMatrix) {
    cvk_mat_t *warpedDepth = nullptr;
    cvk_mat_t *warpedImage = nullptr;
    cvk_mat_t *warpedMask = nullptr;
    cvk_ptcloud_warp_frame(as_mat(depth), image != 0 ? as_mat(image) : nullptr,
                           mask != 0 ? as_mat(mask) : nullptr, as_mat(rt),
                           as_mat(cameraMatrix), &warpedDepth, &warpedImage, &warpedMask);
    jlong vals[3] = {as_handle(warpedDepth), as_handle(warpedImage), as_handle(warpedMask)};
    return handles_array(env, vals, 3);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPtcloud_ptcloudFindPlanes(JNIEnv *env, jobject, jlong points3d,
                                                    jlong normals, jint blockSize, jint minSize,
                                                    jdouble threshold, jdouble sensorErrorA,
                                                    jdouble sensorErrorB, jdouble sensorErrorC,
                                                    jint method) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *coefficients = nullptr;
    cvk_ptcloud_find_planes(as_mat(points3d), normals != 0 ? as_mat(normals) : nullptr, &mask,
                            &coefficients, blockSize, minSize, threshold, sensorErrorA,
                            sensorErrorB, sensorErrorC, method);
    jlong vals[2] = {as_handle(mask), as_handle(coefficients)};
    return handles_array(env, vals, 2);
}

} /* extern "C" */

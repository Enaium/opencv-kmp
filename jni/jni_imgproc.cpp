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
 * JNI wrappers for the imgproc statics slice. Every Java_cn_enaium_opencv_
 * JniImgproc_* function forwards to the cvk_ ABI in shim_imgproc.cpp; Mat
 * handles travel as jlong pointers.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_imgproc.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

static inline cvk_mat_t *as_mat(jlong h) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(h));
}

static inline jlong as_handle(const cvk_mat_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}

/** Wraps freshly allocated Mat handles into a jlongArray (releases on OOM). */
static jlongArray handles_to_array(JNIEnv *env, cvk_mat_t *const *handles, jsize count) {
    jlongArray result = env->NewLongArray(count);
    if (result == nullptr) {
        for (jsize i = 0; i < count; ++i) cvk_mat_release(handles[i]);
        return nullptr;
    }
    std::vector<jlong> values(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = as_handle(handles[i]);
    }
    env->SetLongArrayRegion(result, 0, count, values.data());
    return result;
}

} // namespace

extern "C" {

/* =========================================================================
 * kernels / filters
 * ========================================================================= */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc_getDerivKernels(JNIEnv *env, jobject, jint dx,
                                                 jint dy, jint ksize,
                                                 jboolean normalize, jint ktype) {
    cvk_mat_t *kx = nullptr;
    cvk_mat_t *ky = nullptr;
    if (!cvk_get_deriv_kernels(dx, dy, ksize, normalize != JNI_FALSE, ktype,
                               &kx, &ky)) {
        return env->NewLongArray(0);
    }
    cvk_mat_t *handles[2] = {kx, ky};
    return handles_to_array(env, handles, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_getGaborKernel(JNIEnv *, jobject, jint width,
                                                jint height, jdouble sigma,
                                                jdouble theta, jdouble lambda,
                                                jdouble gamma, jdouble psi,
                                                jint ktype) {
    return as_handle(cvk_get_gabor_kernel(width, height, sigma, theta, lambda,
                                          gamma, psi, ktype));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_sepFilter2D(JNIEnv *, jobject, jlong src,
                                             jint ddepth, jlong kernel_x,
                                             jlong kernel_y, jint anchor_x,
                                             jint anchor_y, jdouble delta,
                                             jint border_type) {
    return as_handle(cvk_sep_filter_2d(as_mat(src), ddepth, as_mat(kernel_x),
                                       as_mat(kernel_y), anchor_x, anchor_y,
                                       delta, border_type));
}

/* =========================================================================
 * corners
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_preCornerDetect(JNIEnv *, jobject, jlong src,
                                                 jint ksize, jint border_type) {
    return as_handle(cvk_pre_corner_detect(as_mat(src), ksize, border_type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_cornerEigenValsAndVecs(JNIEnv *, jobject,
                                                        jlong src,
                                                        jint block_size,
                                                        jint ksize,
                                                        jint border_type) {
    return as_handle(cvk_corner_eigen_vals_and_vecs(as_mat(src), block_size,
                                                    ksize, border_type));
}

/* =========================================================================
 * color conversion
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_cvtColorTwoPlane(JNIEnv *, jobject, jlong src1,
                                                  jlong src2, jint code) {
    return as_handle(cvk_cvt_color_two_plane(as_mat(src1), as_mat(src2), code));
}

/* =========================================================================
 * calibration helpers
 * ========================================================================= */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc_initUndistortRectifyMap(
    JNIEnv *env, jobject, jlong camera_matrix, jlong dist_coeffs, jlong r,
    jlong new_camera_matrix, jint width, jint height, jint m1_type) {
    cvk_mat_t *map1 = nullptr;
    cvk_mat_t *map2 = nullptr;
    if (!cvk_init_undistort_rectify_map(
            as_mat(camera_matrix), as_mat(dist_coeffs),
            r != 0 ? as_mat(r) : nullptr,
            new_camera_matrix != 0 ? as_mat(new_camera_matrix) : nullptr,
            width, height, m1_type, &map1, &map2)) {
        return env->NewLongArray(0);
    }
    cvk_mat_t *handles[2] = {map1, map2};
    return handles_to_array(env, handles, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_undistortPoints(JNIEnv *, jobject, jlong src,
                                                 jlong camera_matrix,
                                                 jlong dist_coeffs, jlong r,
                                                 jlong p) {
    return as_handle(cvk_undistort_points(as_mat(src), as_mat(camera_matrix),
                                          as_mat(dist_coeffs),
                                          r != 0 ? as_mat(r) : nullptr,
                                          p != 0 ? as_mat(p) : nullptr));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_getDefaultNewCameraMatrix(JNIEnv *, jobject,
                                                           jlong camera_matrix,
                                                           jint width,
                                                           jint height,
                                                           jboolean center_pp) {
    return as_handle(cvk_get_default_new_camera_matrix(
        as_mat(camera_matrix), width, height, center_pp != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_estimateAffine2D(JNIEnv *, jobject, jlong from,
                                                  jlong to, jint method,
                                                  jdouble ransac_threshold,
                                                  jlong max_iters,
                                                  jdouble confidence,
                                                  jlong refine_iters) {
    return as_handle(cvk_estimate_affine_2d(
        as_mat(from), as_mat(to), method, ransac_threshold, max_iters,
        confidence, refine_iters));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_estimateAffinePartial2D(JNIEnv *, jobject,
                                                         jlong from, jlong to,
                                                         jint method,
                                                         jdouble ransac_threshold,
                                                         jlong max_iters,
                                                         jdouble confidence,
                                                         jlong refine_iters) {
    return as_handle(cvk_estimate_affine_partial_2d(
        as_mat(from), as_mat(to), method, ransac_threshold, max_iters,
        confidence, refine_iters));
}

/* =========================================================================
 * distance transform
 * ========================================================================= */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc_distanceTransformWithLabels(
    JNIEnv *env, jobject, jlong src, jint distance_type, jint mask_size,
    jint label_type) {
    cvk_mat_t *dst = nullptr;
    cvk_mat_t *labels = nullptr;
    if (!cvk_distance_transform_with_labels(as_mat(src), distance_type,
                                            mask_size, label_type, &dst,
                                            &labels)) {
        return env->NewLongArray(0);
    }
    cvk_mat_t *handles[2] = {dst, labels};
    return handles_to_array(env, handles, 2);
}

/* =========================================================================
 * contours / geometry
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_convexHull(JNIEnv *, jobject, jlong points,
                                            jboolean clockwise,
                                            jboolean return_points) {
    return as_handle(cvk_convex_hull(as_mat(points), clockwise != JNI_FALSE,
                                     return_points != JNI_FALSE));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgproc_isContourConvex(JNIEnv *, jobject, jlong points) {
    return cvk_is_contour_convex(as_mat(points)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_convexityDefects(JNIEnv *, jobject,
                                                  jlong contour, jlong hull_idx) {
    return as_handle(cvk_convexity_defects(as_mat(contour), as_mat(hull_idx)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_fitLine(JNIEnv *, jobject, jlong points,
                                         jint dist_type, jdouble param,
                                         jdouble reps, jdouble aeps) {
    return as_handle(cvk_fit_line(as_mat(points), dist_type, param, reps, aeps));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc_boxPoints(JNIEnv *, jobject, jdouble cx,
                                           jdouble cy, jdouble width,
                                           jdouble height, jdouble angle) {
    cvk_rotated_rect_t box;
    box.cx = cx;
    box.cy = cy;
    box.width = width;
    box.height = height;
    box.angle = angle;
    return as_handle(cvk_box_points(box));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc_rotatedRectangleIntersection(
    JNIEnv *env, jobject, jdouble r1cx, jdouble r1cy, jdouble r1w, jdouble r1h,
    jdouble r1a, jdouble r2cx, jdouble r2cy, jdouble r2w, jdouble r2h,
    jdouble r2a) {
    cvk_rotated_rect_t rect1;
    rect1.cx = r1cx;
    rect1.cy = r1cy;
    rect1.width = r1w;
    rect1.height = r1h;
    rect1.angle = r1a;
    cvk_rotated_rect_t rect2;
    rect2.cx = r2cx;
    rect2.cy = r2cy;
    rect2.width = r2w;
    rect2.height = r2h;
    rect2.angle = r2a;
    cvk_mat_t *intersection = nullptr;
    const jlong type = cvk_rotated_rectangle_intersection(rect1, rect2, &intersection);
    const jlong out[2] = {type, as_handle(intersection)};
    jlongArray result = env->NewLongArray(2);
    if (result == nullptr) {
        cvk_mat_release(intersection);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 2, out);
    return result;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniImgproc_pointPolygonTest(JNIEnv *, jobject, jlong contour,
                                                  jdouble x, jdouble y,
                                                  jboolean measure_dist) {
    return cvk_point_polygon_test(as_mat(contour), x, y, measure_dist != JNI_FALSE);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc_intersectConvexConvex(JNIEnv *env, jobject,
                                                       jlong p1, jlong p2,
                                                       jboolean handle_nested) {
    cvk_mat_t *polygon = nullptr;
    const jfloat area = cvk_intersect_convex_convex(
        as_mat(p1), as_mat(p2), handle_nested != JNI_FALSE, &polygon);
    // [polygon handle, float area bits] — the area travels as its IEEE-754
    // bit pattern so no precision is lost on the jlong hop.
    jint area_bits = 0;
    std::memcpy(&area_bits, &area, sizeof(area_bits));
    const jlong out[2] = {as_handle(polygon), area_bits};
    jlongArray result = env->NewLongArray(2);
    if (result == nullptr) {
        cvk_mat_release(polygon);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 2, out);
    return result;
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniImgproc_huMoments(JNIEnv *env, jobject, jdouble m00,
                                           jdouble m10, jdouble m01,
                                           jdouble m20, jdouble m11,
                                           jdouble m02, jdouble m30,
                                           jdouble m21, jdouble m12,
                                           jdouble m03) {
    cvk_moments_t moments;
    moments.m00 = m00;
    moments.m10 = m10;
    moments.m01 = m01;
    moments.m20 = m20;
    moments.m11 = m11;
    moments.m02 = m02;
    moments.m30 = m30;
    moments.m21 = m21;
    moments.m12 = m12;
    moments.m03 = m03;
    double out[7] = {0, 0, 0, 0, 0, 0, 0};
    cvk_hu_moments(moments, out);
    jdoubleArray result = env->NewDoubleArray(7);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 7, out);
    return result;
}

/* =========================================================================
 * pyramids
 * ========================================================================= */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc_buildPyramid(JNIEnv *env, jobject, jlong src,
                                              jint max_level, jint border_type) {
    std::vector<cvk_mat_t *> levels(static_cast<size_t>(max_level) + 1, nullptr);
    const int count = cvk_build_pyramid(as_mat(src), max_level, border_type, levels.data(),
                                        static_cast<int>(levels.size()));
    if (count <= 0) return env->NewLongArray(0);
    return handles_to_array(env, levels.data(), static_cast<jsize>(count));
}

} /* extern "C" */

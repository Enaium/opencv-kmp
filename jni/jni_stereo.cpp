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
 * JNI bridge for the stereo module: thin wrappers around the cvk_ C ABI in
 * native/shim_stereo.cpp. Every function maps to a member of the Kotlin
 * `internal object JniStereo`.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_stereo.h"

#include <cstdint>
#include <cstring>

static inline cvk_mat_t *as_mat(jlong h) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(h));
}

static inline jlong as_handle(const cvk_mat_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}

static inline cvk_stereo_matcher_t *as_matcher(jlong h) {
    return reinterpret_cast<cvk_stereo_matcher_t *>(static_cast<uintptr_t>(h));
}

static inline jlong as_matcher_handle(const cvk_stereo_matcher_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}

static jintArray rect_to_jints(JNIEnv *env, cvk_rect_t r) {
    jintArray arr = env->NewIntArray(4);
    if (arr == nullptr) return nullptr;
    jint vals[4] = {r.x, r.y, r.width, r.height};
    env->SetIntArrayRegion(arr, 0, 4, vals);
    return arr;
}

static cvk_rect_t jints_to_rect(JNIEnv *env, jintArray arr) {
    cvk_rect_t r = {0, 0, 0, 0};
    if (arr == nullptr) return r;
    jint vals[4];
    env->GetIntArrayRegion(arr, 0, 4, vals);
    r.x = vals[0];
    r.y = vals[1];
    r.width = vals[2];
    r.height = vals[3];
    return r;
}

static jlongArray mat_handles(JNIEnv *env, const cvk_mat_t *const *mats, jsize count) {
    jlongArray arr = env->NewLongArray(count);
    if (arr == nullptr) return nullptr;
    jlong vals[8];
    for (jsize i = 0; i < count; ++i) {
        vals[i] = as_handle(mats[i]);
    }
    env->SetLongArrayRegion(arr, 0, count, vals);
    return arr;
}

extern "C" {

/* ---- factories -------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniStereo_stereoBmCreate(JNIEnv *, jobject, jint numDisparities,
                                                jint blockSize) {
    return as_matcher_handle(cvk_stereo_matcher_create_bm(numDisparities, blockSize));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniStereo_stereoSgbmCreate(JNIEnv *, jobject, jint minDisparity,
                                                  jint numDisparities, jint blockSize, jint p1,
                                                  jint p2, jint disp12MaxDiff, jint preFilterCap,
                                                  jint uniquenessRatio, jint speckleWindowSize,
                                                  jint speckleRange, jint mode) {
    return as_matcher_handle(cvk_stereo_matcher_create_sgbm(
        minDisparity, numDisparities, blockSize, p1, p2, disp12MaxDiff, preFilterCap,
        uniquenessRatio, speckleWindowSize, speckleRange, mode));
}

/* ---- StereoMatcher ----------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniStereo_stereoMatcherCompute(JNIEnv *, jobject, jlong matcher,
                                                      jlong left, jlong right) {
    return as_handle(cvk_stereo_matcher_compute(as_matcher(matcher), as_mat(left), as_mat(right)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoMatcherClear(JNIEnv *, jobject, jlong matcher) {
    cvk_stereo_matcher_clear(as_matcher(matcher));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniStereo_stereoMatcherEmpty(JNIEnv *, jobject, jlong matcher) {
    return cvk_stereo_matcher_empty(as_matcher(matcher)) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoMatcherSave(JNIEnv *env, jobject, jlong matcher,
                                                   jstring filename) {
    const char *name = env->GetStringUTFChars(filename, nullptr);
    if (name == nullptr) return;
    cvk_stereo_matcher_save(as_matcher(matcher), name);
    env->ReleaseStringUTFChars(filename, name);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniStereo_stereoMatcherGetDefaultName(JNIEnv *env, jobject,
                                                             jlong matcher) {
    const char *name = cvk_stereo_matcher_get_default_name(as_matcher(matcher));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

#define MATCHER_GET_SET(java_suffix, c_suffix)                                        \
    JNIEXPORT jint JNICALL                                                            \
    Java_cn_enaium_opencv_JniStereo_stereoMatcherGet##java_suffix(JNIEnv *, jobject,   \
                                                                  jlong matcher) {     \
        return cvk_stereo_matcher_get_##c_suffix(as_matcher(matcher));                \
    }                                                                                  \
    JNIEXPORT void JNICALL                                                             \
    Java_cn_enaium_opencv_JniStereo_stereoMatcherSet##java_suffix(JNIEnv *, jobject,   \
                                                                  jlong matcher,        \
                                                                  jint value) {         \
        cvk_stereo_matcher_set_##c_suffix(as_matcher(matcher), value);                \
    }

MATCHER_GET_SET(MinDisparity, min_disparity)
MATCHER_GET_SET(NumDisparities, num_disparities)
MATCHER_GET_SET(BlockSize, block_size)
MATCHER_GET_SET(SpeckleWindowSize, speckle_window_size)
MATCHER_GET_SET(SpeckleRange, speckle_range)
MATCHER_GET_SET(Disp12MaxDiff, disp12_max_diff)

#undef MATCHER_GET_SET

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoMatcherRelease(JNIEnv *, jobject, jlong matcher) {
    cvk_stereo_matcher_release(as_matcher(matcher));
}

/* ---- StereoBM ---------------------------------------------------------- */

#define BM_GET_SET(java_suffix, c_suffix)                                            \
    JNIEXPORT jint JNICALL                                                           \
    Java_cn_enaium_opencv_JniStereo_stereoBmGet##java_suffix(JNIEnv *, jobject,      \
                                                             jlong matcher) {        \
        return cvk_stereo_bm_get_##c_suffix(as_matcher(matcher));                    \
    }                                                                                 \
    JNIEXPORT void JNICALL                                                            \
    Java_cn_enaium_opencv_JniStereo_stereoBmSet##java_suffix(JNIEnv *, jobject,      \
                                                             jlong matcher,           \
                                                             jint value) {            \
        cvk_stereo_bm_set_##c_suffix(as_matcher(matcher), value);                    \
    }

BM_GET_SET(PreFilterType, pre_filter_type)
BM_GET_SET(PreFilterSize, pre_filter_size)
BM_GET_SET(PreFilterCap, pre_filter_cap)
BM_GET_SET(TextureThreshold, texture_threshold)
BM_GET_SET(UniquenessRatio, uniqueness_ratio)
BM_GET_SET(SmallerBlockSize, smaller_block_size)

#undef BM_GET_SET

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniStereo_stereoBmGetRoi1(JNIEnv *env, jobject, jlong matcher) {
    return rect_to_jints(env, cvk_stereo_bm_get_roi1(as_matcher(matcher)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoBmSetRoi1(JNIEnv *env, jobject, jlong matcher,
                                                jintArray roi) {
    cvk_stereo_bm_set_roi1(as_matcher(matcher), jints_to_rect(env, roi));
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniStereo_stereoBmGetRoi2(JNIEnv *env, jobject, jlong matcher) {
    return rect_to_jints(env, cvk_stereo_bm_get_roi2(as_matcher(matcher)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoBmSetRoi2(JNIEnv *env, jobject, jlong matcher,
                                                jintArray roi) {
    cvk_stereo_bm_set_roi2(as_matcher(matcher), jints_to_rect(env, roi));
}

/* ---- StereoSGBM -------------------------------------------------------- */

#define SGBM_GET_SET(java_suffix, c_suffix)                                           \
    JNIEXPORT jint JNICALL                                                            \
    Java_cn_enaium_opencv_JniStereo_stereoSgbmGet##java_suffix(JNIEnv *, jobject,     \
                                                               jlong matcher) {       \
        return cvk_stereo_sgbm_get_##c_suffix(as_matcher(matcher));                   \
    }                                                                                  \
    JNIEXPORT void JNICALL                                                             \
    Java_cn_enaium_opencv_JniStereo_stereoSgbmSet##java_suffix(JNIEnv *, jobject,     \
                                                               jlong matcher,          \
                                                               jint value) {           \
        cvk_stereo_sgbm_set_##c_suffix(as_matcher(matcher), value);                   \
    }

SGBM_GET_SET(PreFilterCap, pre_filter_cap)
SGBM_GET_SET(UniquenessRatio, uniqueness_ratio)
SGBM_GET_SET(P1, p1)
SGBM_GET_SET(P2, p2)
SGBM_GET_SET(Mode, mode)

#undef SGBM_GET_SET

/* ---- Stereo free functions --------------------------------------------- */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniStereo_stereoRectify(JNIEnv *env, jobject, jlong cm1, jlong dc1,
                                               jlong cm2, jlong dc2, jint imageWidth,
                                               jint imageHeight, jlong r, jlong t, jint flags,
                                               jdouble alpha, jint newWidth, jint newHeight,
                                               jintArray roi1Out, jintArray roi2Out) {
    cvk_rect_t roi1 = {0, 0, 0, 0};
    cvk_rect_t roi2 = {0, 0, 0, 0};
    cvk_mat_t *r1 = nullptr;
    cvk_mat_t *r2 = nullptr;
    cvk_mat_t *p1 = nullptr;
    cvk_mat_t *p2 = nullptr;
    cvk_mat_t *q = nullptr;
    const int ok = cvk_stereo_rectify(as_mat(cm1), as_mat(dc1), as_mat(cm2), as_mat(dc2),
                                      imageWidth, imageHeight, as_mat(r), as_mat(t), flags,
                                      alpha, newWidth, newHeight, &r1, &r2, &p1, &p2, &q,
                                      &roi1, &roi2);
    if (ok == 0) return nullptr;
    const cvk_mat_t *mats[5] = {r1, r2, p1, p2, q};
    jlongArray out = mat_handles(env, mats, 5);
    if (roi1Out != nullptr) {
        jint vals[4] = {roi1.x, roi1.y, roi1.width, roi1.height};
        env->SetIntArrayRegion(roi1Out, 0, 4, vals);
    }
    if (roi2Out != nullptr) {
        jint vals[4] = {roi2.x, roi2.y, roi2.width, roi2.height};
        env->SetIntArrayRegion(roi2Out, 0, 4, vals);
    }
    return out;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniStereo_stereoRectifyUncalibrated(JNIEnv *env, jobject, jlong points1,
                                                           jlong points2, jlong f,
                                                           jint imgWidth, jint imgHeight,
                                                           jdouble threshold) {
    cvk_mat_t *h1 = nullptr;
    cvk_mat_t *h2 = nullptr;
    const int ok = cvk_stereo_rectify_uncalibrated(as_mat(points1), as_mat(points2), as_mat(f),
                                                   imgWidth, imgHeight, threshold, &h1, &h2);
    jlongArray arr = env->NewLongArray(3);
    if (arr == nullptr) return nullptr;
    jlong vals[3] = {ok != 0 ? 1 : 0, as_handle(h1), as_handle(h2)};
    env->SetLongArrayRegion(arr, 0, 3, vals);
    return arr;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniStereo_stereoFisheyeRectify(JNIEnv *env, jobject, jlong k1, jlong d1,
                                                      jlong k2, jlong d2, jint imageWidth,
                                                      jint imageHeight, jlong r, jlong tvec,
                                                      jint flags, jint newWidth, jint newHeight,
                                                      jdouble balance, jdouble fovScale) {
    cvk_mat_t *r1 = nullptr;
    cvk_mat_t *r2 = nullptr;
    cvk_mat_t *p1 = nullptr;
    cvk_mat_t *p2 = nullptr;
    cvk_mat_t *q = nullptr;
    const int ok = cvk_stereo_fisheye_rectify(as_mat(k1), as_mat(d1), as_mat(k2), as_mat(d2),
                                              imageWidth, imageHeight, as_mat(r), as_mat(tvec),
                                              flags, newWidth, newHeight, balance, fovScale,
                                              &r1, &r2, &p1, &p2, &q);
    if (ok == 0) return nullptr;
    const cvk_mat_t *mats[5] = {r1, r2, p1, p2, q};
    return mat_handles(env, mats, 5);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoFilterSpeckles(JNIEnv *, jobject, jlong img,
                                                      jdouble newVal, jint maxSpeckleSize,
                                                      jdouble maxDiff) {
    cvk_stereo_filter_speckles(as_mat(img), newVal, maxSpeckleSize, maxDiff);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoFilterSpecklesBuf(JNIEnv *, jobject, jlong img,
                                                         jdouble newVal, jint maxSpeckleSize,
                                                         jdouble maxDiff, jlong buf) {
    cvk_stereo_filter_speckles_buf(as_mat(img), newVal, maxSpeckleSize, maxDiff, as_mat(buf));
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniStereo_stereoGetValidDisparityROI(JNIEnv *env, jobject, jintArray roi1,
                                                            jintArray roi2, jint minDisparity,
                                                            jint numberOfDisparities,
                                                            jint blockSize) {
    cvk_rect_t r1 = jints_to_rect(env, roi1);
    cvk_rect_t r2 = jints_to_rect(env, roi2);
    cvk_rect_t out = cvk_stereo_get_valid_disparity_roi(r1, r2, minDisparity,
                                                        numberOfDisparities, blockSize);
    return rect_to_jints(env, out);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniStereo_stereoValidateDisparity(JNIEnv *, jobject, jlong disparity,
                                                         jlong cost, jint minDisparity,
                                                         jint numberOfDisparities,
                                                         jint disp12MaxDisp) {
    cvk_stereo_validate_disparity(as_mat(disparity), as_mat(cost), minDisparity,
                                  numberOfDisparities, disp12MaxDisp);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniStereo_stereoReprojectImageTo3D(JNIEnv *, jobject, jlong disparity,
                                                          jlong q, jboolean handleMissingValues,
                                                          jint ddepth) {
    return as_handle(cvk_stereo_reproject_image_to_3d(
        as_mat(disparity), as_mat(q), handleMissingValues ? 1 : 0, ddepth));
}

} /* extern "C" */

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
 * JNI bridge for the imgproc2 slice: Java_cn_enaium_opencv_JniImgproc2_*
 * wrappers around the cvk_ C ABI in native/shim_imgproc2.cpp. Multi-Mat
 * outputs (detect) travel back as a fixed-order jlongArray of handles; the
 * Kotlin side unpacks them into JvmMat instances.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_imgproc2.h"

#include <cstdint>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_line_segment_detector_t *as_line_segment_detector(jlong handle) {
    return reinterpret_cast<cvk_line_segment_detector_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_gh_ballard_t *as_gh_ballard(jlong handle) {
    return reinterpret_cast<cvk_gh_ballard_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_gh_guil_t *as_gh_guil(jlong handle) {
    return reinterpret_cast<cvk_gh_guil_t *>(static_cast<uintptr_t>(handle));
}

/* Packs `count` Mat handles into a caller-owned jlongArray (0 handles are kept). */
static jlongArray as_handle_array(JNIEnv *env, const cvk_mat_t *const *handles, jsize count) {
    jlongArray result = env->NewLongArray(count);
    if (result == nullptr) return nullptr;
    std::vector<jlong> values(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = as_handle(handles[i]);
    }
    env->SetLongArrayRegion(result, 0, count, values.data());
    return result;
}

/* Emits the four Algorithm member wrappers for one handle type. */
#define CVK_ALG_JNI(JPREFIX, CPREFIX)                                          \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##Clear( \
        JNIEnv *, jobject, jlong h) {                                          \
        cvk_##CPREFIX##_clear(as_##CPREFIX(h));                                \
    }                                                                          \
    JNIEXPORT jboolean JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##Empty( \
        JNIEnv *, jobject, jlong h) {                                          \
        return cvk_##CPREFIX##_empty(as_##CPREFIX(h)) != 0 ? JNI_TRUE : JNI_FALSE; \
    }                                                                          \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##Save(  \
        JNIEnv *env, jobject, jlong h, jstring filename) {                     \
        const char *utf = filename != nullptr                                 \
            ? env->GetStringUTFChars(filename, nullptr) : nullptr;             \
        if (filename != nullptr && utf == nullptr) return;                     \
        cvk_##CPREFIX##_save(as_##CPREFIX(h), utf);                            \
        if (utf != nullptr) env->ReleaseStringUTFChars(filename, utf);         \
    }                                                                          \
    JNIEXPORT jstring JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##GetDefaultName( \
        JNIEnv *env, jobject, jlong h) {                                       \
        const char *name = cvk_##CPREFIX##_get_default_name(as_##CPREFIX(h));  \
        return name != nullptr ? env->NewStringUTF(name) : nullptr;            \
    }

/* Emits set_/get_ JNI wrappers for one integer GH field. */
#define CVK_GH_INT_PAIR(JPREFIX, CPREFIX, CAMEL, SNAKE)                        \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##Set##CAMEL( \
        JNIEnv *, jobject, jlong h, jint value) {                              \
        cvk_##CPREFIX##_set_##SNAKE(as_##CPREFIX(h), value);                   \
    }                                                                          \
    JNIEXPORT jint JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##Get##CAMEL( \
        JNIEnv *, jobject, jlong h) {                                          \
        return cvk_##CPREFIX##_get_##SNAKE(as_##CPREFIX(h));                   \
    }

/* Emits set_/get_ JNI wrappers for one double GH field. */
#define CVK_GH_DBL_PAIR(JPREFIX, CPREFIX, CAMEL, SNAKE)                        \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##Set##CAMEL( \
        JNIEnv *, jobject, jlong h, jdouble value) {                           \
        cvk_##CPREFIX##_set_##SNAKE(as_##CPREFIX(h), value);                   \
    }                                                                          \
    JNIEXPORT jdouble JNICALL Java_cn_enaium_opencv_JniImgproc2_##JPREFIX##Get##CAMEL( \
        JNIEnv *, jobject, jlong h) {                                          \
        return cvk_##CPREFIX##_get_##SNAKE(as_##CPREFIX(h));                   \
    }

extern "C" {

/* =============================================================
 * LineSegmentDetector
 * ============================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc2_lsdCreate(JNIEnv *, jobject, jint refine,
                                            jdouble scale, jdouble sigma_scale,
                                            jdouble quant, jdouble ang_th,
                                            jdouble log_eps, jdouble density_th,
                                            jint n_bins) {
    return reinterpret_cast<jlong>(cvk_line_segment_detector_create(
        refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc2_lsdDetect(JNIEnv *env, jobject, jlong h, jlong image) {
    cvk_mat_t *lines = nullptr;
    cvk_mat_t *width = nullptr;
    cvk_mat_t *prec = nullptr;
    cvk_mat_t *nfa = nullptr;
    lines = cvk_line_segment_detector_detect(as_line_segment_detector(h), as_mat(image), &width, &prec, &nfa);
    const cvk_mat_t *handles[4] = {lines, width, prec, nfa};
    return as_handle_array(env, handles, 4);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_lsdDrawSegments(JNIEnv *, jobject, jlong h,
                                                  jlong image, jlong lines) {
    cvk_line_segment_detector_draw_segments(as_line_segment_detector(h), as_mat(image), as_mat(lines));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniImgproc2_lsdCompareSegments(JNIEnv *, jobject, jlong h,
                                                     jint width, jint height,
                                                     jlong lines1, jlong lines2,
                                                     jlong image) {
    return cvk_line_segment_detector_compare_segments(
        as_line_segment_detector(h), width, height, as_mat(lines1), as_mat(lines2), as_mat(image));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_lsdRelease(JNIEnv *, jobject, jlong h) {
    cvk_line_segment_detector_release(as_line_segment_detector(h));
}

CVK_ALG_JNI(lsd, line_segment_detector)

/* =============================================================
 * GeneralizedHoughBallard
 * ============================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghBallardCreate(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(cvk_gh_ballard_create());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghBallardSetTemplate(JNIEnv *, jobject, jlong h,
                                                       jlong templ, jdouble cx, jdouble cy) {
    cvk_gh_ballard_set_template(as_gh_ballard(h), as_mat(templ), cx, cy);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghBallardSetTemplateEdges(JNIEnv *, jobject, jlong h,
                                                            jlong edges, jlong dx, jlong dy,
                                                            jdouble cx, jdouble cy) {
    cvk_gh_ballard_set_template_edges(as_gh_ballard(h), as_mat(edges), as_mat(dx),
                                      as_mat(dy), cx, cy);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghBallardDetect(JNIEnv *env, jobject, jlong h, jlong image) {
    cvk_mat_t *positions = nullptr;
    cvk_mat_t *votes = nullptr;
    positions = cvk_gh_ballard_detect(as_gh_ballard(h), as_mat(image), &votes);
    const cvk_mat_t *handles[2] = {positions, votes};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghBallardDetectEdges(JNIEnv *env, jobject, jlong h,
                                                       jlong edges, jlong dx, jlong dy) {
    cvk_mat_t *positions = nullptr;
    cvk_mat_t *votes = nullptr;
    positions = cvk_gh_ballard_detect_edges(as_gh_ballard(h), as_mat(edges),
                                            as_mat(dx), as_mat(dy), &votes);
    const cvk_mat_t *handles[2] = {positions, votes};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghBallardRelease(JNIEnv *, jobject, jlong h) {
    cvk_gh_ballard_release(as_gh_ballard(h));
}

CVK_ALG_JNI(ghBallard, gh_ballard)

CVK_GH_INT_PAIR(ghBallard, gh_ballard, CannyLowThresh, canny_low_thresh)
CVK_GH_INT_PAIR(ghBallard, gh_ballard, CannyHighThresh, canny_high_thresh)
CVK_GH_DBL_PAIR(ghBallard, gh_ballard, MinDist, min_dist)
CVK_GH_DBL_PAIR(ghBallard, gh_ballard, Dp, dp)
CVK_GH_INT_PAIR(ghBallard, gh_ballard, MaxBufferSize, max_buffer_size)
CVK_GH_INT_PAIR(ghBallard, gh_ballard, Levels, levels)
CVK_GH_INT_PAIR(ghBallard, gh_ballard, VotesThreshold, votes_threshold)

/* =============================================================
 * GeneralizedHoughGuil
 * ============================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghGuilCreate(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(cvk_gh_guil_create());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghGuilSetTemplate(JNIEnv *, jobject, jlong h,
                                                    jlong templ, jdouble cx, jdouble cy) {
    cvk_gh_guil_set_template(as_gh_guil(h), as_mat(templ), cx, cy);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghGuilSetTemplateEdges(JNIEnv *, jobject, jlong h,
                                                         jlong edges, jlong dx, jlong dy,
                                                         jdouble cx, jdouble cy) {
    cvk_gh_guil_set_template_edges(as_gh_guil(h), as_mat(edges), as_mat(dx),
                                   as_mat(dy), cx, cy);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghGuilDetect(JNIEnv *env, jobject, jlong h, jlong image) {
    cvk_mat_t *positions = nullptr;
    cvk_mat_t *votes = nullptr;
    positions = cvk_gh_guil_detect(as_gh_guil(h), as_mat(image), &votes);
    const cvk_mat_t *handles[2] = {positions, votes};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghGuilDetectEdges(JNIEnv *env, jobject, jlong h,
                                                    jlong edges, jlong dx, jlong dy) {
    cvk_mat_t *positions = nullptr;
    cvk_mat_t *votes = nullptr;
    positions = cvk_gh_guil_detect_edges(as_gh_guil(h), as_mat(edges),
                                         as_mat(dx), as_mat(dy), &votes);
    const cvk_mat_t *handles[2] = {positions, votes};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgproc2_ghGuilRelease(JNIEnv *, jobject, jlong h) {
    cvk_gh_guil_release(as_gh_guil(h));
}

CVK_ALG_JNI(ghGuil, gh_guil)

CVK_GH_INT_PAIR(ghGuil, gh_guil, CannyLowThresh, canny_low_thresh)
CVK_GH_INT_PAIR(ghGuil, gh_guil, CannyHighThresh, canny_high_thresh)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, MinDist, min_dist)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, Dp, dp)
CVK_GH_INT_PAIR(ghGuil, gh_guil, MaxBufferSize, max_buffer_size)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, Xi, xi)
CVK_GH_INT_PAIR(ghGuil, gh_guil, Levels, levels)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, AngleEpsilon, angle_epsilon)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, MinAngle, min_angle)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, MaxAngle, max_angle)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, AngleStep, angle_step)
CVK_GH_INT_PAIR(ghGuil, gh_guil, AngleThresh, angle_thresh)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, MinScale, min_scale)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, MaxScale, max_scale)
CVK_GH_DBL_PAIR(ghGuil, gh_guil, ScaleStep, scale_step)
CVK_GH_INT_PAIR(ghGuil, gh_guil, ScaleThresh, scale_thresh)
CVK_GH_INT_PAIR(ghGuil, gh_guil, PosThresh, pos_thresh)

} /* extern "C" */

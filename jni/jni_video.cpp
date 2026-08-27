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
 * JNI bridge for the org.opencv.video surface: thin
 * Java_cn_enaium_opencv_JniVideo_* wrappers around the cvk_ C ABI in
 * native/shim_video.cpp. Handles arrive as jlong pointers; no exceptions
 * may cross the boundary (the shim is noexcept).
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_video.h"

#include <cstdint>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_background_subtractor_knn_t *as_background_subtractor_knn(jlong handle) {
    return reinterpret_cast<cvk_background_subtractor_knn_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_background_subtractor_mog2_t *as_background_subtractor_mog2(jlong handle) {
    return reinterpret_cast<cvk_background_subtractor_mog2_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_farneback_optical_flow_t *as_farneback_optical_flow(jlong handle) {
    return reinterpret_cast<cvk_farneback_optical_flow_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_dis_optical_flow_t *as_dis_optical_flow(jlong handle) {
    return reinterpret_cast<cvk_dis_optical_flow_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_sparse_pyr_lk_optical_flow_t *as_sparse_pyr_lk_optical_flow(jlong handle) {
    return reinterpret_cast<cvk_sparse_pyr_lk_optical_flow_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_variational_refinement_t *as_variational_refinement(jlong handle) {
    return reinterpret_cast<cvk_variational_refinement_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_kalman_filter_t *as_kalman_filter(jlong handle) {
    return reinterpret_cast<cvk_kalman_filter_t *>(static_cast<uintptr_t>(handle));
}
static inline jlong as_knn_handle(const cvk_background_subtractor_knn_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_mog2_handle(const cvk_background_subtractor_mog2_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_farneback_handle(const cvk_farneback_optical_flow_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_dis_handle(const cvk_dis_optical_flow_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_sparse_lk_handle(const cvk_sparse_pyr_lk_optical_flow_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_var_ref_handle(const cvk_variational_refinement_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

static inline jlong as_kalman_handle(const cvk_kalman_filter_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}


static inline const char *as_utf8(JNIEnv *env, jstring value) {
    return env->GetStringUTFChars(value, nullptr);
}

static inline void release_utf8(JNIEnv *env, jstring value, const char *text) {
    if (text != nullptr) env->ReleaseStringUTFChars(value, text);
}

static inline jstring to_jstring(JNIEnv *env, const char *text) {
    return text != nullptr ? env->NewStringUTF(text) : nullptr;
}

/* CLS: Kotlin camelCase class prefix used in the JNI symbol, CAMEL: camelCase
 * member used in the JNI symbol, HANDLE: snake_case cvk handle prefix,
 * SUFFIX: snake_case cvk member. */

#define CVK_JNI_INT_GET_SET(CLS, CAMEL, HANDLE, SUFFIX)                             \
    JNIEXPORT jint JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Get##CAMEL(       \
        JNIEnv *, jobject, jlong handle) {                                         \
        return cvk_##HANDLE##_get_##SUFFIX(as_##HANDLE(handle));                   \
    }                                                                              \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Set##CAMEL(       \
        JNIEnv *, jobject, jlong handle, jint value) {                             \
        cvk_##HANDLE##_set_##SUFFIX(as_##HANDLE(handle), value);                   \
    }

#define CVK_JNI_DOUBLE_GET_SET(CLS, CAMEL, HANDLE, SUFFIX)                         \
    JNIEXPORT jdouble JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Get##CAMEL(   \
        JNIEnv *, jobject, jlong handle) {                                         \
        return cvk_##HANDLE##_get_##SUFFIX(as_##HANDLE(handle));                  \
    }                                                                              \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Set##CAMEL(      \
        JNIEnv *, jobject, jlong handle, jdouble value) {                          \
        cvk_##HANDLE##_set_##SUFFIX(as_##HANDLE(handle), value);                  \
    }

#define CVK_JNI_FLOAT_GET_SET(CLS, CAMEL, HANDLE, SUFFIX)                          \
    JNIEXPORT jfloat JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Get##CAMEL(    \
        JNIEnv *, jobject, jlong handle) {                                         \
        return cvk_##HANDLE##_get_##SUFFIX(as_##HANDLE(handle));                  \
    }                                                                              \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Set##CAMEL(      \
        JNIEnv *, jobject, jlong handle, jfloat value) {                           \
        cvk_##HANDLE##_set_##SUFFIX(as_##HANDLE(handle), value);                  \
    }

#define CVK_JNI_BOOL_GET_SET(CLS, CAMEL, HANDLE, SUFFIX)                           \
    JNIEXPORT jboolean JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Get##CAMEL(  \
        JNIEnv *, jobject, jlong handle) {                                         \
        return cvk_##HANDLE##_get_##SUFFIX(as_##HANDLE(handle)) != 0               \
                   ? JNI_TRUE                                                      \
                   : JNI_FALSE;                                                    \
    }                                                                              \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Set##CAMEL(      \
        JNIEnv *, jobject, jlong handle, jboolean value) {                         \
        cvk_##HANDLE##_set_##SUFFIX(as_##HANDLE(handle), value != JNI_FALSE);     \
    }

#define CVK_JNI_ALG_FUNCS(CLS, HANDLE)                                             \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Clear(            \
        JNIEnv *, jobject, jlong handle) {                                         \
        cvk_##HANDLE##_clear(as_##HANDLE(handle));                                 \
    }                                                                              \
    JNIEXPORT jboolean JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Empty(        \
        JNIEnv *, jobject, jlong handle) {                                         \
        return cvk_##HANDLE##_empty(as_##HANDLE(handle)) != 0 ? JNI_TRUE           \
                                                              : JNI_FALSE;         \
    }                                                                              \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Save(             \
        JNIEnv *env, jobject, jlong handle, jstring filename) {                    \
        const char *path = as_utf8(env, filename);                                 \
        cvk_##HANDLE##_save(as_##HANDLE(handle), path);                            \
        release_utf8(env, filename, path);                                         \
    }                                                                              \
    JNIEXPORT jstring JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##GetDefaultName(\
        JNIEnv *env, jobject, jlong handle) {                                      \
        return to_jstring(env, cvk_##HANDLE##_get_default_name(as_##HANDLE(handle))); \
    }

#define CVK_JNI_RELEASE(CLS, HANDLE)                                               \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_##CLS##Release(          \
        JNIEnv *, jobject, jlong handle) {                                         \
        cvk_##HANDLE##_release(as_##HANDLE(handle));                               \
    }

extern "C" {

// ---------------------------------------------------------- video statics

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_calcOpticalFlowFarneback(
    JNIEnv *, jobject, jlong prev, jlong next, jlong flow, jdouble pyrScale, jint levels,
    jint winSize, jint iterations, jint polyN, jdouble polySigma, jint flags) {
    cvk_calc_optical_flow_farneback(as_mat(prev), as_mat(next), as_mat(flow), pyrScale,
                                    levels, winSize, iterations, polyN, polySigma, flags);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_calcOpticalFlowPyrLK(
    JNIEnv *, jobject, jlong prevImg, jlong nextImg, jlong prevPts, jlong nextPts,
    jlong status, jlong err, jint winW, jint winH, jint maxLevel, jint tcType,
    jint tcMaxCount, jdouble tcEpsilon, jint flags, jdouble minEigThreshold) {
    cvk_calc_optical_flow_pyr_lk(as_mat(prevImg), as_mat(nextImg), as_mat(prevPts),
                                 as_mat(nextPts), as_mat(status), as_mat(err), winW, winH,
                                 maxLevel, tcType, tcMaxCount, tcEpsilon, flags,
                                 minEigThreshold);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniVideo_computeECC(JNIEnv *, jobject, jlong templateImage,
                                          jlong inputImage, jlong inputMask) {
    return cvk_compute_ecc(as_mat(templateImage), as_mat(inputImage), as_mat(inputMask));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniVideo_findTransformECC(
    JNIEnv *, jobject, jlong templateImage, jlong inputImage, jlong warpMatrix,
    jint motionType, jint tcType, jint tcMaxCount, jdouble tcEpsilon, jlong inputMask,
    jint gaussFiltSize) {
    return cvk_find_transform_ecc(as_mat(templateImage), as_mat(inputImage),
                                  as_mat(warpMatrix), motionType, tcType, tcMaxCount,
                                  tcEpsilon, as_mat(inputMask), gaussFiltSize);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniVideo_findTransformECCWithMask(
    JNIEnv *, jobject, jlong templateImage, jlong inputImage, jlong templateMask,
    jlong inputMask, jlong warpMatrix, jint motionType, jint tcType, jint tcMaxCount,
    jdouble tcEpsilon, jint gaussFiltSize) {
    return cvk_find_transform_ecc_with_mask(as_mat(templateImage), as_mat(inputImage),
                                            as_mat(templateMask), as_mat(inputMask),
                                            as_mat(warpMatrix), motionType, tcType,
                                            tcMaxCount, tcEpsilon, gaussFiltSize);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniVideo_findTransformECCMultiScale(
    JNIEnv *, jobject, jlong reference, jlong sample, jlong warpMatrix, jint motionType,
    jint tcType, jint tcMaxCount, jdouble tcEpsilon, jlong itersPerLevel,
    jint gaussFiltSize, jint nlevels, jint interpolation, jlong referenceMask,
    jlong sampleMask) {
    return cvk_find_transform_ecc_multi_scale(
        as_mat(reference), as_mat(sample), as_mat(warpMatrix), motionType, tcType,
        tcMaxCount, tcEpsilon, as_mat(itersPerLevel), gaussFiltSize, nlevels,
        interpolation, as_mat(referenceMask), as_mat(sampleMask));
}

// --------------------------------------------------- BackgroundSubtractorKNN

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_createBackgroundSubtractorKNN(
    JNIEnv *, jobject, jint history, jdouble dist2Threshold, jboolean detectShadows) {
    return as_knn_handle(cvk_background_subtractor_knn_create(
        history, dist2Threshold, detectShadows != JNI_FALSE));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorKNNRelease(JNIEnv *, jobject, jlong handle) {
    cvk_background_subtractor_knn_release(as_background_subtractor_knn(handle));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorKNNApply(
    JNIEnv *, jobject, jlong handle, jlong image, jlong fgmask, jdouble learningRate) {
    cvk_background_subtractor_knn_apply(as_background_subtractor_knn(handle),
                                        as_mat(image), as_mat(fgmask), learningRate);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorKNNApplyMask(
    JNIEnv *, jobject, jlong handle, jlong image, jlong knownForegroundMask, jlong fgmask,
    jdouble learningRate) {
    cvk_background_subtractor_knn_apply_mask(as_background_subtractor_knn(handle),
                                             as_mat(image), as_mat(knownForegroundMask),
                                             as_mat(fgmask), learningRate);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorKNNGetBackgroundImage(
    JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_background_subtractor_knn_get_background_image(
        as_background_subtractor_knn(handle)));
}

CVK_JNI_INT_GET_SET(backgroundSubtractorKNN, History, background_subtractor_knn, history)
CVK_JNI_INT_GET_SET(backgroundSubtractorKNN, NSamples, background_subtractor_knn, n_samples)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorKNN, Dist2Threshold, background_subtractor_knn,
                       dist2_threshold)
CVK_JNI_INT_GET_SET(backgroundSubtractorKNN, kNNSamples, background_subtractor_knn,
                    knn_samples)
CVK_JNI_BOOL_GET_SET(backgroundSubtractorKNN, DetectShadows, background_subtractor_knn,
                     detect_shadows)
CVK_JNI_INT_GET_SET(backgroundSubtractorKNN, ShadowValue, background_subtractor_knn,
                    shadow_value)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorKNN, ShadowThreshold, background_subtractor_knn,
                       shadow_threshold)
CVK_JNI_ALG_FUNCS(backgroundSubtractorKNN, background_subtractor_knn)

// --------------------------------------------------- BackgroundSubtractorMOG2

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_createBackgroundSubtractorMOG2(
    JNIEnv *, jobject, jint history, jdouble varThreshold, jboolean detectShadows) {
    return as_mog2_handle(cvk_background_subtractor_mog2_create(
        history, varThreshold, detectShadows != JNI_FALSE));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorMOG2Release(JNIEnv *, jobject, jlong handle) {
    cvk_background_subtractor_mog2_release(as_background_subtractor_mog2(handle));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorMOG2Apply(
    JNIEnv *, jobject, jlong handle, jlong image, jlong fgmask, jdouble learningRate) {
    cvk_background_subtractor_mog2_apply(as_background_subtractor_mog2(handle),
                                         as_mat(image), as_mat(fgmask), learningRate);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorMOG2ApplyMask(
    JNIEnv *, jobject, jlong handle, jlong image, jlong knownForegroundMask, jlong fgmask,
    jdouble learningRate) {
    cvk_background_subtractor_mog2_apply_mask(as_background_subtractor_mog2(handle),
                                              as_mat(image), as_mat(knownForegroundMask),
                                              as_mat(fgmask), learningRate);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_backgroundSubtractorMOG2GetBackgroundImage(
    JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_background_subtractor_mog2_get_background_image(
        as_background_subtractor_mog2(handle)));
}

CVK_JNI_INT_GET_SET(backgroundSubtractorMOG2, History, background_subtractor_mog2, history)
CVK_JNI_INT_GET_SET(backgroundSubtractorMOG2, NMixtures, background_subtractor_mog2,
                    n_mixtures)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, BackgroundRatio, background_subtractor_mog2,
                       background_ratio)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, VarThreshold, background_subtractor_mog2,
                       var_threshold)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, VarThresholdGen, background_subtractor_mog2,
                       var_threshold_gen)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, VarInit, background_subtractor_mog2,
                       var_init)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, VarMin, background_subtractor_mog2,
                       var_min)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, VarMax, background_subtractor_mog2,
                       var_max)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, ComplexityReductionThreshold,
                       background_subtractor_mog2, complexity_reduction_threshold)
CVK_JNI_BOOL_GET_SET(backgroundSubtractorMOG2, DetectShadows, background_subtractor_mog2,
                     detect_shadows)
CVK_JNI_INT_GET_SET(backgroundSubtractorMOG2, ShadowValue, background_subtractor_mog2,
                    shadow_value)
CVK_JNI_DOUBLE_GET_SET(backgroundSubtractorMOG2, ShadowThreshold, background_subtractor_mog2,
                       shadow_threshold)
CVK_JNI_ALG_FUNCS(backgroundSubtractorMOG2, background_subtractor_mog2)

// ---------------------------------------------------- FarnebackOpticalFlow

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_createFarnebackOpticalFlow(
    JNIEnv *, jobject, jint numLevels, jdouble pyrScale, jboolean fastPyramids,
    jint winSize, jint numIters, jint polyN, jdouble polySigma, jint flags) {
    return as_farneback_handle(cvk_farneback_optical_flow_create(
        numLevels, pyrScale, fastPyramids != JNI_FALSE, winSize, numIters, polyN,
        polySigma, flags));
}

CVK_JNI_RELEASE(farnebackOpticalFlow, farneback_optical_flow)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_farnebackOpticalFlowCalc(
    JNIEnv *, jobject, jlong handle, jlong i0, jlong i1, jlong flow) {
    cvk_farneback_optical_flow_calc(as_farneback_optical_flow(handle), as_mat(i0),
                                    as_mat(i1), as_mat(flow));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_farnebackOpticalFlowCollectGarbage(
    JNIEnv *, jobject, jlong handle) {
    cvk_farneback_optical_flow_collect_garbage(as_farneback_optical_flow(handle));
}

CVK_JNI_INT_GET_SET(farnebackOpticalFlow, NumLevels, farneback_optical_flow, num_levels)
CVK_JNI_DOUBLE_GET_SET(farnebackOpticalFlow, PyrScale, farneback_optical_flow, pyr_scale)
CVK_JNI_BOOL_GET_SET(farnebackOpticalFlow, FastPyramids, farneback_optical_flow,
                     fast_pyramids)
CVK_JNI_INT_GET_SET(farnebackOpticalFlow, WinSize, farneback_optical_flow, win_size)
CVK_JNI_INT_GET_SET(farnebackOpticalFlow, NumIters, farneback_optical_flow, num_iters)
CVK_JNI_INT_GET_SET(farnebackOpticalFlow, PolyN, farneback_optical_flow, poly_n)
CVK_JNI_DOUBLE_GET_SET(farnebackOpticalFlow, PolySigma, farneback_optical_flow, poly_sigma)
CVK_JNI_INT_GET_SET(farnebackOpticalFlow, Flags, farneback_optical_flow, flags)
CVK_JNI_ALG_FUNCS(farnebackOpticalFlow, farneback_optical_flow)

// -------------------------------------------------------- DISOpticalFlow

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_createDisOpticalFlow(JNIEnv *, jobject, jint preset) {
    return as_dis_handle(cvk_dis_optical_flow_create(preset));
}

CVK_JNI_RELEASE(disOpticalFlow, dis_optical_flow)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_disOpticalFlowCalc(
    JNIEnv *, jobject, jlong handle, jlong i0, jlong i1, jlong flow) {
    cvk_dis_optical_flow_calc(as_dis_optical_flow(handle), as_mat(i0), as_mat(i1),
                              as_mat(flow));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_disOpticalFlowCollectGarbage(
    JNIEnv *, jobject, jlong handle) {
    cvk_dis_optical_flow_collect_garbage(as_dis_optical_flow(handle));
}

CVK_JNI_INT_GET_SET(disOpticalFlow, FinestScale, dis_optical_flow, finest_scale)
CVK_JNI_INT_GET_SET(disOpticalFlow, CoarsestScale, dis_optical_flow, coarsest_scale)
CVK_JNI_INT_GET_SET(disOpticalFlow, PatchSize, dis_optical_flow, patch_size)
CVK_JNI_INT_GET_SET(disOpticalFlow, PatchStride, dis_optical_flow, patch_stride)
CVK_JNI_INT_GET_SET(disOpticalFlow, GradientDescentIterations, dis_optical_flow,
                    gradient_descent_iterations)
CVK_JNI_INT_GET_SET(disOpticalFlow, VariationalRefinementIterations, dis_optical_flow,
                    variational_refinement_iterations)
CVK_JNI_FLOAT_GET_SET(disOpticalFlow, VariationalRefinementAlpha, dis_optical_flow,
                      variational_refinement_alpha)
CVK_JNI_FLOAT_GET_SET(disOpticalFlow, VariationalRefinementDelta, dis_optical_flow,
                      variational_refinement_delta)
CVK_JNI_FLOAT_GET_SET(disOpticalFlow, VariationalRefinementGamma, dis_optical_flow,
                      variational_refinement_gamma)
CVK_JNI_FLOAT_GET_SET(disOpticalFlow, VariationalRefinementEpsilon, dis_optical_flow,
                      variational_refinement_epsilon)
CVK_JNI_BOOL_GET_SET(disOpticalFlow, UseMeanNormalization, dis_optical_flow,
                     use_mean_normalization)
CVK_JNI_BOOL_GET_SET(disOpticalFlow, UseSpatialPropagation, dis_optical_flow,
                     use_spatial_propagation)
CVK_JNI_ALG_FUNCS(disOpticalFlow, dis_optical_flow)

// -------------------------------------------------- SparsePyrLKOpticalFlow

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_createSparsePyrLKOpticalFlow(
    JNIEnv *, jobject, jint winW, jint winH, jint maxLevel, jint tcType, jint tcMaxCount,
    jdouble tcEpsilon, jint flags, jdouble minEigThreshold) {
    return as_sparse_lk_handle(cvk_sparse_pyr_lk_optical_flow_create(
        winW, winH, maxLevel, tcType, tcMaxCount, tcEpsilon, flags, minEigThreshold));
}

CVK_JNI_RELEASE(sparsePyrLKOpticalFlow, sparse_pyr_lk_optical_flow)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_sparsePyrLKOpticalFlowCalc(
    JNIEnv *, jobject, jlong handle, jlong prevImg, jlong nextImg, jlong prevPts,
    jlong nextPts, jlong status, jlong err) {
    cvk_sparse_pyr_lk_optical_flow_calc(as_sparse_pyr_lk_optical_flow(handle),
                                        as_mat(prevImg), as_mat(nextImg), as_mat(prevPts),
                                        as_mat(nextPts), as_mat(status), as_mat(err));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniVideo_sparsePyrLKOpticalFlowGetWinW(
    JNIEnv *, jobject, jlong handle) {
    return cvk_sparse_pyr_lk_optical_flow_get_win_w(as_sparse_pyr_lk_optical_flow(handle));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniVideo_sparsePyrLKOpticalFlowGetWinH(
    JNIEnv *, jobject, jlong handle) {
    return cvk_sparse_pyr_lk_optical_flow_get_win_h(as_sparse_pyr_lk_optical_flow(handle));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_sparsePyrLKOpticalFlowSetWinSize(
    JNIEnv *, jobject, jlong handle, jint winW, jint winH) {
    cvk_sparse_pyr_lk_optical_flow_set_win_size(as_sparse_pyr_lk_optical_flow(handle),
                                                winW, winH);
}

CVK_JNI_INT_GET_SET(sparsePyrLKOpticalFlow, MaxLevel, sparse_pyr_lk_optical_flow, max_level)

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniVideo_sparsePyrLKOpticalFlowGetTermCriteria(
    JNIEnv *env, jobject, jlong handle) {
    int tcType = 0;
    int tcMaxCount = 0;
    double tcEpsilon = 0.0;
    cvk_sparse_pyr_lk_optical_flow_get_term_criteria(as_sparse_pyr_lk_optical_flow(handle),
                                                     &tcType, &tcMaxCount, &tcEpsilon);
    jdoubleArray out = env->NewDoubleArray(3);
    if (out != nullptr) {
        jdouble values[3] = {static_cast<jdouble>(tcType),
                             static_cast<jdouble>(tcMaxCount), tcEpsilon};
        env->SetDoubleArrayRegion(out, 0, 3, values);
    }
    return out;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_sparsePyrLKOpticalFlowSetTermCriteria(
    JNIEnv *, jobject, jlong handle, jint tcType, jint tcMaxCount, jdouble tcEpsilon) {
    cvk_sparse_pyr_lk_optical_flow_set_term_criteria(as_sparse_pyr_lk_optical_flow(handle),
                                                     tcType, tcMaxCount, tcEpsilon);
}

CVK_JNI_INT_GET_SET(sparsePyrLKOpticalFlow, Flags, sparse_pyr_lk_optical_flow, flags)
CVK_JNI_DOUBLE_GET_SET(sparsePyrLKOpticalFlow, MinEigThreshold,
                       sparse_pyr_lk_optical_flow, min_eig_threshold)
CVK_JNI_ALG_FUNCS(sparsePyrLKOpticalFlow, sparse_pyr_lk_optical_flow)

// ---------------------------------------------------- VariationalRefinement

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_createVariationalRefinement(JNIEnv *, jobject) {
    return as_var_ref_handle(cvk_variational_refinement_create());
}

CVK_JNI_RELEASE(variationalRefinement, variational_refinement)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_variationalRefinementCalc(
    JNIEnv *, jobject, jlong handle, jlong i0, jlong i1, jlong flow) {
    cvk_variational_refinement_calc(as_variational_refinement(handle), as_mat(i0),
                                    as_mat(i1), as_mat(flow));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_variationalRefinementCalcUV(
    JNIEnv *, jobject, jlong handle, jlong i0, jlong i1, jlong flowU, jlong flowV) {
    cvk_variational_refinement_calc_uv(as_variational_refinement(handle), as_mat(i0),
                                       as_mat(i1), as_mat(flowU), as_mat(flowV));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_variationalRefinementCollectGarbage(
    JNIEnv *, jobject, jlong handle) {
    cvk_variational_refinement_collect_garbage(as_variational_refinement(handle));
}

CVK_JNI_INT_GET_SET(variationalRefinement, FixedPointIterations, variational_refinement,
                    fixed_point_iterations)
CVK_JNI_INT_GET_SET(variationalRefinement, SorIterations, variational_refinement,
                    sor_iterations)
CVK_JNI_FLOAT_GET_SET(variationalRefinement, Omega, variational_refinement, omega)
CVK_JNI_FLOAT_GET_SET(variationalRefinement, Alpha, variational_refinement, alpha)
CVK_JNI_FLOAT_GET_SET(variationalRefinement, Delta, variational_refinement, delta)
CVK_JNI_FLOAT_GET_SET(variationalRefinement, Gamma, variational_refinement, gamma)
CVK_JNI_FLOAT_GET_SET(variationalRefinement, Epsilon, variational_refinement, epsilon)
CVK_JNI_ALG_FUNCS(variationalRefinement, variational_refinement)

// ------------------------------------------------------------ KalmanFilter

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_kalmanFilterCreate(
    JNIEnv *, jobject, jint dynamParams, jint measureParams, jint controlParams, jint type) {
    return as_kalman_handle(cvk_kalman_filter_create(
        dynamParams, measureParams, controlParams, type));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo_kalmanFilterRelease(JNIEnv *, jobject, jlong handle) {
    cvk_kalman_filter_release(as_kalman_filter(handle));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_kalmanFilterPredict(JNIEnv *, jobject, jlong handle,
                                                   jlong control) {
    return as_handle(cvk_kalman_filter_predict(as_kalman_filter(handle), as_mat(control)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo_kalmanFilterCorrect(JNIEnv *, jobject, jlong handle,
                                                   jlong measurement) {
    return as_handle(cvk_kalman_filter_correct(as_kalman_filter(handle),
                                               as_mat(measurement)));
}

#define CVK_JNI_KF_GET_SET(CAMEL, SUFFIX)                                          \
    JNIEXPORT jlong JNICALL Java_cn_enaium_opencv_JniVideo_kalmanFilterGet##CAMEL( \
        JNIEnv *, jobject, jlong handle) {                                         \
        return as_handle(cvk_kalman_filter_get_##SUFFIX(as_kalman_filter(handle))); \
    }                                                                              \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniVideo_kalmanFilterSet##CAMEL(  \
        JNIEnv *, jobject, jlong handle, jlong mat) {                              \
        cvk_kalman_filter_set_##SUFFIX(as_kalman_filter(handle), as_mat(mat));     \
    }

CVK_JNI_KF_GET_SET(StatePre, state_pre)
CVK_JNI_KF_GET_SET(StatePost, state_post)
CVK_JNI_KF_GET_SET(TransitionMatrix, transition_matrix)
CVK_JNI_KF_GET_SET(ControlMatrix, control_matrix)
CVK_JNI_KF_GET_SET(MeasurementMatrix, measurement_matrix)
CVK_JNI_KF_GET_SET(ProcessNoiseCov, process_noise_cov)
CVK_JNI_KF_GET_SET(MeasurementNoiseCov, measurement_noise_cov)
CVK_JNI_KF_GET_SET(ErrorCovPre, error_cov_pre)
CVK_JNI_KF_GET_SET(Gain, gain)
CVK_JNI_KF_GET_SET(ErrorCovPost, error_cov_post)

} /* extern "C" */

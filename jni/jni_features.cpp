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
 * JNI bridge for the features module: thin Java_cn_enaium_opencv_JniFeatures_*
 * wrappers around the cvk_ C ABI in native/shim_features.cpp. The Kotlin
 * object is `internal object JniFeatures`. Mat handles travel as jlong
 * pointers; two-Mat results (compute, detectAndCompute, affine view params)
 * come back as jlong[2]; the MSER region flat buffer travels as jbyteArray
 * with the bounding-box Mat written into a caller-provided jlongArray.
 */
#include <jni.h>
#include <cstdint>
#include <cstdlib>

#include "opencv_kmp.h"
#include "opencv_kmp_features.h"

static inline cvk_mat_t *as_mat(jlong h) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(h));
}

static inline jlong as_handle(const void *p) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

static inline cvk_feature2d_t *as_feature2d(jlong h) {
    return reinterpret_cast<cvk_feature2d_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_sift_t *as_sift(jlong h) {
    return reinterpret_cast<cvk_sift_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_orb_t *as_orb(jlong h) {
    return reinterpret_cast<cvk_orb_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_mser_t *as_mser(jlong h) {
    return reinterpret_cast<cvk_mser_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_fast_feature_detector_t *as_fast(jlong h) {
    return reinterpret_cast<cvk_fast_feature_detector_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_gftt_detector_t *as_gftt(jlong h) {
    return reinterpret_cast<cvk_gftt_detector_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_simple_blob_detector_t *as_blob(jlong h) {
    return reinterpret_cast<cvk_simple_blob_detector_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_affine_t *as_affine(jlong h) {
    return reinterpret_cast<cvk_affine_t *>(static_cast<uintptr_t>(h));
}

static inline jlongArray new_long_array_2(JNIEnv *env, jlong a, jlong b) {
    jlong out[2] = {a, b};
    jlongArray result = env->NewLongArray(2);
    if (result != nullptr) env->SetLongArrayRegion(result, 0, 2, out);
    return result;
}

template <typename F>
static inline void with_utf(JNIEnv *env, jstring s, F &&body) {
    if (s == nullptr) return;
    const char *utf = env->GetStringUTFChars(s, nullptr);
    if (utf == nullptr) return;
    body(utf);
    env->ReleaseStringUTFChars(s, utf);
}

/* -------------------------------------------------------------------------
 * Shared Feature2D + Algorithm surface, generated per class.
 * T: cvk_ prefix of the C functions; J: class suffix used in the JNI symbol
 * (matches the per-class JniFeatures external funs); AS: handle converter.
 * ------------------------------------------------------------------------- */

#define CVK_FEATURES_JNI_COMMON(T, J, AS)                                      \
    JNIEXPORT jlong JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Detect(    \
        JNIEnv *, jobject, jlong h, jlong image, jlong mask) {                 \
        return as_handle(cvk_##T##_detect(AS(h), as_mat(image), as_mat(mask))); \
    }                                                                          \
    JNIEXPORT jlongArray JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Compute( \
        JNIEnv *env, jobject, jlong h, jlong image, jlong keypoints) {         \
        cvk_mat_t *kp = nullptr;                                               \
        cvk_mat_t *desc = nullptr;                                             \
        cvk_##T##_compute(AS(h), as_mat(image), as_mat(keypoints), &kp, &desc); \
        return new_long_array_2(env, as_handle(kp), as_handle(desc));          \
    }                                                                          \
    JNIEXPORT jlongArray JNICALL Java_cn_enaium_opencv_JniFeatures_##J##DetectAndCompute( \
        JNIEnv *env, jobject, jlong h, jlong image, jlong mask) {              \
        cvk_mat_t *desc = nullptr;                                             \
        cvk_mat_t *kp = cvk_##T##_detect_and_compute(AS(h), as_mat(image),     \
                                                     as_mat(mask), &desc);     \
        return new_long_array_2(env, as_handle(kp), as_handle(desc));          \
    }                                                                          \
    JNIEXPORT jint JNICALL Java_cn_enaium_opencv_JniFeatures_##J##DescriptorSize( \
        JNIEnv *, jobject, jlong h) {                                          \
        return cvk_##T##_descriptor_size(AS(h));                               \
    }                                                                          \
    JNIEXPORT jint JNICALL Java_cn_enaium_opencv_JniFeatures_##J##DescriptorType( \
        JNIEnv *, jobject, jlong h) {                                          \
        return cvk_##T##_descriptor_type(AS(h));                               \
    }                                                                          \
    JNIEXPORT jint JNICALL Java_cn_enaium_opencv_JniFeatures_##J##DefaultNorm( \
        JNIEnv *, jobject, jlong h) {                                          \
        return cvk_##T##_default_norm(AS(h));                                  \
    }                                                                          \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Write(      \
        JNIEnv *env, jobject, jlong h, jstring filename) {                     \
        with_utf(env, filename, [&](const char *utf) { cvk_##T##_write(AS(h), utf); }); \
    }                                                                          \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Read(       \
        JNIEnv *env, jobject, jlong h, jstring filename) {                     \
        with_utf(env, filename, [&](const char *utf) { cvk_##T##_read(AS(h), utf); }); \
    }                                                                          \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Clear(      \
        JNIEnv *, jobject, jlong h) {                                          \
        cvk_##T##_clear(AS(h));                                                \
    }                                                                          \
    JNIEXPORT jboolean JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Empty(  \
        JNIEnv *, jobject, jlong h) {                                          \
        return cvk_##T##_empty(AS(h)) != 0 ? JNI_TRUE : JNI_FALSE;             \
    }                                                                          \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Save(       \
        JNIEnv *env, jobject, jlong h, jstring filename) {                     \
        with_utf(env, filename, [&](const char *utf) { cvk_##T##_save(AS(h), utf); }); \
    }                                                                          \
    JNIEXPORT jstring JNICALL Java_cn_enaium_opencv_JniFeatures_##J##GetDefaultName( \
        JNIEnv *env, jobject, jlong h) {                                       \
        const char *name = cvk_##T##_get_default_name(AS(h));                  \
        return name != nullptr ? env->NewStringUTF(name) : nullptr;            \
    }                                                                          \
    JNIEXPORT void JNICALL Java_cn_enaium_opencv_JniFeatures_##J##Release(    \
        JNIEnv *, jobject, jlong h) {                                          \
        cvk_##T##_release(AS(h));                                              \
    }

extern "C" {

/* =========================================================================
 * SIFT
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures_siftCreate(JNIEnv *, jobject, jint nfeatures,
                                             jint n_octave_layers,
                                             jdouble contrast_threshold,
                                             jdouble edge_threshold, jdouble sigma,
                                             jint descriptor_type,
                                             jboolean enable_precise_upscale) {
    return as_handle(cvk_sift_create(nfeatures, n_octave_layers, contrast_threshold,
                                     edge_threshold, sigma, descriptor_type,
                                     enable_precise_upscale != JNI_FALSE));
}

CVK_FEATURES_JNI_COMMON(sift, sift, as_sift)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_siftSetNFeatures(JNIEnv *, jobject, jlong h,
                                                   jint max_features) {
    cvk_sift_set_n_features(as_sift(h), max_features);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_siftGetNFeatures(JNIEnv *, jobject, jlong h) {
    return cvk_sift_get_n_features(as_sift(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_siftSetNOctaveLayers(JNIEnv *, jobject, jlong h,
                                                       jint n_octave_layers) {
    cvk_sift_set_n_octave_layers(as_sift(h), n_octave_layers);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_siftGetNOctaveLayers(JNIEnv *, jobject, jlong h) {
    return cvk_sift_get_n_octave_layers(as_sift(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_siftSetContrastThreshold(JNIEnv *, jobject,
                                                           jlong h,
                                                           jdouble threshold) {
    cvk_sift_set_contrast_threshold(as_sift(h), threshold);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_siftGetContrastThreshold(JNIEnv *, jobject,
                                                           jlong h) {
    return cvk_sift_get_contrast_threshold(as_sift(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_siftSetEdgeThreshold(JNIEnv *, jobject, jlong h,
                                                       jdouble threshold) {
    cvk_sift_set_edge_threshold(as_sift(h), threshold);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_siftGetEdgeThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_sift_get_edge_threshold(as_sift(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_siftSetSigma(JNIEnv *, jobject, jlong h,
                                               jdouble sigma) {
    cvk_sift_set_sigma(as_sift(h), sigma);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_siftGetSigma(JNIEnv *, jobject, jlong h) {
    return cvk_sift_get_sigma(as_sift(h));
}

/* =========================================================================
 * ORB
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures_orbCreate(JNIEnv *, jobject, jint nfeatures,
                                            jfloat scale_factor, jint nlevels,
                                            jint edge_threshold, jint first_level,
                                            jint wta_k, jint score_type,
                                            jint patch_size, jint fast_threshold) {
    return as_handle(cvk_orb_create(nfeatures, scale_factor, nlevels, edge_threshold,
                                    first_level, wta_k, score_type, patch_size,
                                    fast_threshold));
}

CVK_FEATURES_JNI_COMMON(orb, orb, as_orb)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetMaxFeatures(JNIEnv *, jobject, jlong h,
                                                    jint max_features) {
    cvk_orb_set_max_features(as_orb(h), max_features);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetMaxFeatures(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_max_features(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetScaleFactor(JNIEnv *, jobject, jlong h,
                                                    jdouble scale_factor) {
    cvk_orb_set_scale_factor(as_orb(h), scale_factor);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetScaleFactor(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_scale_factor(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetNLevels(JNIEnv *, jobject, jlong h,
                                                jint nlevels) {
    cvk_orb_set_n_levels(as_orb(h), nlevels);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetNLevels(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_n_levels(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetEdgeThreshold(JNIEnv *, jobject, jlong h,
                                                      jint edge_threshold) {
    cvk_orb_set_edge_threshold(as_orb(h), edge_threshold);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetEdgeThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_edge_threshold(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetFirstLevel(JNIEnv *, jobject, jlong h,
                                                   jint first_level) {
    cvk_orb_set_first_level(as_orb(h), first_level);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetFirstLevel(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_first_level(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetWtaK(JNIEnv *, jobject, jlong h,
                                             jint wta_k) {
    cvk_orb_set_wta_k(as_orb(h), wta_k);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetWtaK(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_wta_k(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetScoreType(JNIEnv *, jobject, jlong h,
                                                  jint score_type) {
    cvk_orb_set_score_type(as_orb(h), score_type);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetScoreType(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_score_type(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetPatchSize(JNIEnv *, jobject, jlong h,
                                                  jint patch_size) {
    cvk_orb_set_patch_size(as_orb(h), patch_size);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetPatchSize(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_patch_size(as_orb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_orbSetFastThreshold(JNIEnv *, jobject, jlong h,
                                                      jint fast_threshold) {
    cvk_orb_set_fast_threshold(as_orb(h), fast_threshold);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_orbGetFastThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_orb_get_fast_threshold(as_orb(h));
}

/* =========================================================================
 * MSER
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures_mserCreate(JNIEnv *, jobject, jint delta,
                                             jint min_area, jint max_area,
                                             jdouble max_variation,
                                             jdouble min_diversity,
                                             jint max_evolution,
                                             jdouble area_threshold,
                                             jdouble min_margin,
                                             jint edge_blur_size) {
    return as_handle(cvk_mser_create(delta, min_area, max_area, max_variation,
                                     min_diversity, max_evolution, area_threshold,
                                     min_margin, edge_blur_size));
}

CVK_FEATURES_JNI_COMMON(mser, mser, as_mser)

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniFeatures_mserDetectRegions(JNIEnv *env, jobject, jlong h,
                                                    jlong image, jlongArray out) {
    cvk_mat_t *bboxes = nullptr;
    size_t length = 0;
    unsigned char *buffer = cvk_mser_detect_regions(as_mser(h), as_mat(image),
                                                    &bboxes, &length);
    if (out != nullptr && env->GetArrayLength(out) >= 1) {
        jlong handle = as_handle(bboxes);
        env->SetLongArrayRegion(out, 0, 1, &handle);
    }
    if (buffer == nullptr) return nullptr;
    jbyteArray result = env->NewByteArray(static_cast<jsize>(length));
    if (result != nullptr) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(length),
                                reinterpret_cast<const jbyte *>(buffer));
    }
    std::free(buffer);
    return result;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetDelta(JNIEnv *, jobject, jlong h, jint delta) {
    cvk_mser_set_delta(as_mser(h), delta);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetDelta(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_delta(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetMinArea(JNIEnv *, jobject, jlong h,
                                                 jint min_area) {
    cvk_mser_set_min_area(as_mser(h), min_area);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetMinArea(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_min_area(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetMaxArea(JNIEnv *, jobject, jlong h,
                                                 jint max_area) {
    cvk_mser_set_max_area(as_mser(h), max_area);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetMaxArea(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_max_area(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetMaxVariation(JNIEnv *, jobject, jlong h,
                                                      jdouble max_variation) {
    cvk_mser_set_max_variation(as_mser(h), max_variation);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetMaxVariation(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_max_variation(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetMinDiversity(JNIEnv *, jobject, jlong h,
                                                      jdouble min_diversity) {
    cvk_mser_set_min_diversity(as_mser(h), min_diversity);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetMinDiversity(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_min_diversity(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetMaxEvolution(JNIEnv *, jobject, jlong h,
                                                      jint max_evolution) {
    cvk_mser_set_max_evolution(as_mser(h), max_evolution);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetMaxEvolution(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_max_evolution(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetAreaThreshold(JNIEnv *, jobject, jlong h,
                                                       jdouble area_threshold) {
    cvk_mser_set_area_threshold(as_mser(h), area_threshold);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetAreaThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_area_threshold(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetMinMargin(JNIEnv *, jobject, jlong h,
                                                   jdouble min_margin) {
    cvk_mser_set_min_margin(as_mser(h), min_margin);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetMinMargin(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_min_margin(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetEdgeBlurSize(JNIEnv *, jobject, jlong h,
                                                      jint edge_blur_size) {
    cvk_mser_set_edge_blur_size(as_mser(h), edge_blur_size);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetEdgeBlurSize(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_edge_blur_size(as_mser(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_mserSetPass2Only(JNIEnv *, jobject, jlong h,
                                                   jboolean f) {
    cvk_mser_set_pass2_only(as_mser(h), f != JNI_FALSE);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures_mserGetPass2Only(JNIEnv *, jobject, jlong h) {
    return cvk_mser_get_pass2_only(as_mser(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

/* =========================================================================
 * FastFeatureDetector
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures_fastCreate(JNIEnv *, jobject, jint threshold,
                                             jboolean nonmax_suppression, jint type) {
    return as_handle(cvk_fast_feature_detector_create(threshold,
                                                      nonmax_suppression != JNI_FALSE,
                                                      type));
}

CVK_FEATURES_JNI_COMMON(fast_feature_detector, fast, as_fast)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_fastSetThreshold(JNIEnv *, jobject, jlong h,
                                                   jint threshold) {
    cvk_fast_feature_detector_set_threshold(as_fast(h), threshold);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_fastGetThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_fast_feature_detector_get_threshold(as_fast(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_fastSetNonmaxSuppression(JNIEnv *, jobject,
                                                           jlong h, jboolean f) {
    cvk_fast_feature_detector_set_nonmax_suppression(as_fast(h), f != JNI_FALSE);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures_fastGetNonmaxSuppression(JNIEnv *, jobject,
                                                           jlong h) {
    return cvk_fast_feature_detector_get_nonmax_suppression(as_fast(h)) != 0
               ? JNI_TRUE
               : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_fastSetType(JNIEnv *, jobject, jlong h, jint type) {
    cvk_fast_feature_detector_set_type(as_fast(h), type);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_fastGetType(JNIEnv *, jobject, jlong h) {
    return cvk_fast_feature_detector_get_type(as_fast(h));
}

/* =========================================================================
 * GFTTDetector
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttCreate(JNIEnv *, jobject, jint max_corners,
                                             jdouble quality_level,
                                             jdouble min_distance, jint block_size,
                                             jint gradient_size,
                                             jboolean use_harris_detector,
                                             jdouble k) {
    return as_handle(cvk_gftt_detector_create(max_corners, quality_level, min_distance,
                                              block_size, gradient_size,
                                              use_harris_detector != JNI_FALSE, k));
}

CVK_FEATURES_JNI_COMMON(gftt_detector, gftt, as_gftt)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttSetMaxFeatures(JNIEnv *, jobject, jlong h,
                                                     jint max_features) {
    cvk_gftt_detector_set_max_features(as_gftt(h), max_features);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttGetMaxFeatures(JNIEnv *, jobject, jlong h) {
    return cvk_gftt_detector_get_max_features(as_gftt(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttSetQualityLevel(JNIEnv *, jobject, jlong h,
                                                      jdouble qlevel) {
    cvk_gftt_detector_set_quality_level(as_gftt(h), qlevel);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttGetQualityLevel(JNIEnv *, jobject, jlong h) {
    return cvk_gftt_detector_get_quality_level(as_gftt(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttSetMinDistance(JNIEnv *, jobject, jlong h,
                                                     jdouble min_distance) {
    cvk_gftt_detector_set_min_distance(as_gftt(h), min_distance);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttGetMinDistance(JNIEnv *, jobject, jlong h) {
    return cvk_gftt_detector_get_min_distance(as_gftt(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttSetBlockSize(JNIEnv *, jobject, jlong h,
                                                   jint block_size) {
    cvk_gftt_detector_set_block_size(as_gftt(h), block_size);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttGetBlockSize(JNIEnv *, jobject, jlong h) {
    return cvk_gftt_detector_get_block_size(as_gftt(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttSetGradientSize(JNIEnv *, jobject, jlong h,
                                                      jint gradient_size) {
    cvk_gftt_detector_set_gradient_size(as_gftt(h), gradient_size);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttGetGradientSize(JNIEnv *, jobject, jlong h) {
    return cvk_gftt_detector_get_gradient_size(as_gftt(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttSetHarrisDetector(JNIEnv *, jobject, jlong h,
                                                        jboolean val) {
    cvk_gftt_detector_set_harris_detector(as_gftt(h), val != JNI_FALSE);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttGetHarrisDetector(JNIEnv *, jobject, jlong h) {
    return cvk_gftt_detector_get_harris_detector(as_gftt(h)) != 0 ? JNI_TRUE
                                                                  : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttSetK(JNIEnv *, jobject, jlong h, jdouble k) {
    cvk_gftt_detector_set_k(as_gftt(h), k);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniFeatures_gfttGetK(JNIEnv *, jobject, jlong h) {
    return cvk_gftt_detector_get_k(as_gftt(h));
}

/* =========================================================================
 * SimpleBlobDetector
 * ========================================================================= */

#define CVK_BLOB_ARGS                                                          \
    jfloat threshold_step, jfloat min_threshold, jfloat max_threshold,         \
        jlong min_repeatability, jfloat min_dist_between_blobs,                \
        jint filter_by_color, jint blob_color, jint filter_by_area,            \
        jfloat min_area, jfloat max_area, jint filter_by_circularity,          \
        jfloat min_circularity, jfloat max_circularity, jint filter_by_inertia, \
        jfloat min_inertia_ratio, jfloat max_inertia_ratio,                    \
        jint filter_by_convexity, jfloat min_convexity, jfloat max_convexity,  \
        jint collect_contours

#define CVK_BLOB_CALL(expr)                                                    \
    expr(threshold_step, min_threshold, max_threshold, min_repeatability,      \
         min_dist_between_blobs, filter_by_color, blob_color, filter_by_area,  \
         min_area, max_area, filter_by_circularity, min_circularity,           \
         max_circularity, filter_by_inertia, min_inertia_ratio,                \
         max_inertia_ratio, filter_by_convexity, min_convexity, max_convexity, \
         collect_contours)

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures_simpleBlobDetectorCreate(JNIEnv *, jobject,
                                                           CVK_BLOB_ARGS) {
    return as_handle(CVK_BLOB_CALL(cvk_simple_blob_detector_create));
}

CVK_FEATURES_JNI_COMMON(simple_blob_detector, simpleBlobDetector, as_blob)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_simpleBlobDetectorSetParams(JNIEnv *, jobject,
                                                              jlong h,
                                                              CVK_BLOB_ARGS) {
    CVK_BLOB_CALL([&](auto... args) {
        cvk_simple_blob_detector_set_params(as_blob(h), args...);
    });
}

#undef CVK_BLOB_ARGS
#undef CVK_BLOB_CALL

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniFeatures_simpleBlobDetectorGetParams(JNIEnv *env, jobject,
                                                              jlong h) {
    double out[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    cvk_simple_blob_detector_get_params(as_blob(h), out);
    jdoubleArray result = env->NewDoubleArray(20);
    if (result != nullptr) env->SetDoubleArrayRegion(result, 0, 20, out);
    return result;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniFeatures_simpleBlobDetectorGetBlobContours(JNIEnv *env,
                                                                    jobject,
                                                                    jlong h) {
    size_t length = 0;
    unsigned char *buffer = cvk_simple_blob_detector_get_blob_contours(as_blob(h),
                                                                       &length);
    if (buffer == nullptr) return nullptr;
    jbyteArray result = env->NewByteArray(static_cast<jsize>(length));
    if (result != nullptr) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(length),
                                reinterpret_cast<const jbyte *>(buffer));
    }
    std::free(buffer);
    return result;
}

/* =========================================================================
 * AffineFeature
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures_affineCreate(JNIEnv *, jobject, jlong backend,
                                               jint max_tilt, jint min_tilt,
                                               jfloat tilt_step,
                                               jfloat rotate_step_base) {
    return as_handle(cvk_affine_create(as_feature2d(backend), max_tilt, min_tilt,
                                       tilt_step, rotate_step_base));
}

CVK_FEATURES_JNI_COMMON(affine, affine, as_affine)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures_affineSetViewParams(JNIEnv *, jobject, jlong h,
                                                      jlong tilts, jlong rolls) {
    cvk_affine_set_view_params(as_affine(h), as_mat(tilts), as_mat(rolls));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniFeatures_affineGetViewParams(JNIEnv *env, jobject,
                                                      jlong h) {
    cvk_mat_t *tilts = nullptr;
    cvk_mat_t *rolls = nullptr;
    cvk_affine_get_view_params(as_affine(h), &tilts, &rolls);
    return new_long_array_2(env, as_handle(tilts), as_handle(rolls));
}

} /* extern "C" */

/*
 * JNI bridge for the OpenCV "photo" module (Java_cn_enaium_opencv_JniPhoto_*).
 * Forwards every call to the cvk_ shim layer; Mat handles travel as jlong
 * pointers, Mat lists as jlongArray, two-output filters as jlongArray pairs.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_photo.h"

#include <cstdint>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

#define PHOTO_HANDLE(T)                                                             \
    static inline cvk_##T##_t *as_##T(jlong handle) {                               \
        return reinterpret_cast<cvk_##T##_t *>(static_cast<uintptr_t>(handle));     \
    }

PHOTO_HANDLE(tonemap)
PHOTO_HANDLE(tonemap_drago)
PHOTO_HANDLE(tonemap_mantiuk)
PHOTO_HANDLE(tonemap_reinhard)
PHOTO_HANDLE(align_mtb)
PHOTO_HANDLE(calibrate_debevec)
PHOTO_HANDLE(calibrate_robertson)
PHOTO_HANDLE(merge_debevec)
PHOTO_HANDLE(merge_mertens)
PHOTO_HANDLE(merge_robertson)
PHOTO_HANDLE(color_correction_model)
PHOTO_HANDLE(intelligent_scissors_mb)

#undef PHOTO_HANDLE

/** Unpacks a jlongArray of Mat handles into a std::vector. */
static std::vector<const cvk_mat_t *> mats_from(JNIEnv *env, jlongArray array) {
    std::vector<const cvk_mat_t *> mats;
    if (array == nullptr) return mats;
    const jsize count = env->GetArrayLength(array);
    if (count <= 0) return mats;
    jlong *elements = env->GetLongArrayElements(array, nullptr);
    if (elements == nullptr) return mats;
    mats.reserve(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        mats.push_back(as_mat(elements[i]));
    }
    env->ReleaseLongArrayElements(array, elements, JNI_ABORT);
    return mats;
}

/** Builds a jlongArray from Mat handles produced by the C shim. */
static jlongArray handles_to_array(JNIEnv *env, cvk_mat_t *const *handles, int count) {
    if (handles == nullptr || count <= 0) return nullptr;
    jlongArray result = env->NewLongArray(count);
    if (result == nullptr) return nullptr;
    jlong *dst = env->GetLongArrayElements(result, nullptr);
    if (dst == nullptr) return result;
    for (int i = 0; i < count; ++i) {
        dst[i] = as_handle(handles[i]);
    }
    env->ReleaseLongArrayElements(result, dst, 0);
    return result;
}

extern "C" {

/* =========================================================================
 * Photo statics
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_inpaint(JNIEnv *, jobject, jlong src, jlong mask,
                                       jdouble radius, jint flags) {
    return as_handle(cvk_inpaint(as_mat(src), as_mat(mask), radius, flags));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_fastNlMeansDenoising(JNIEnv *, jobject, jlong src, jfloat h,
                                                    jint template_window_size,
                                                    jint search_window_size) {
    return as_handle(cvk_fast_nl_means_denoising(as_mat(src), h, template_window_size,
                                                 search_window_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_fastNlMeansDenoisingH(JNIEnv *, jobject, jlong src, jlong h,
                                                     jint template_window_size,
                                                     jint search_window_size,
                                                     jint norm_type) {
    return as_handle(cvk_fast_nl_means_denoising_h(as_mat(src), as_mat(h),
                                                   template_window_size, search_window_size,
                                                   norm_type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_fastNlMeansDenoisingMulti(JNIEnv *env, jobject,
                                                         jlongArray src,
                                                         jint img_to_denoise_index,
                                                         jint temporal_window_size, jfloat h,
                                                         jint template_window_size,
                                                         jint search_window_size) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_fast_nl_means_denoising_multi(
            mats.data(), static_cast<int>(mats.size()), img_to_denoise_index,
            temporal_window_size, h, template_window_size, search_window_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_fastNlMeansDenoisingMultiH(JNIEnv *env, jobject,
                                                          jlongArray src,
                                                          jint img_to_denoise_index,
                                                          jint temporal_window_size,
                                                          jlong h,
                                                          jint template_window_size,
                                                          jint search_window_size,
                                                          jint norm_type) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_fast_nl_means_denoising_multi_h(
            mats.data(), static_cast<int>(mats.size()), img_to_denoise_index,
            temporal_window_size, as_mat(h), template_window_size, search_window_size,
            norm_type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_fastNlMeansDenoisingColored(JNIEnv *, jobject, jlong src,
                                                           jfloat h, jfloat h_color,
                                                           jint template_window_size,
                                                           jint search_window_size) {
    return as_handle(cvk_fast_nl_means_denoising_colored(as_mat(src), h, h_color,
                                                         template_window_size,
                                                         search_window_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_fastNlMeansDenoisingColoredMulti(JNIEnv *env, jobject,
                                                                jlongArray src,
                                                                jint img_to_denoise_index,
                                                                jint temporal_window_size,
                                                                jfloat h, jfloat h_color,
                                                                jint template_window_size,
                                                                jint search_window_size) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_fast_nl_means_denoising_colored_multi(
            mats.data(), static_cast<int>(mats.size()), img_to_denoise_index,
            temporal_window_size, h, h_color, template_window_size, search_window_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_denoiseTvl1(JNIEnv *env, jobject, jlongArray src,
                                           jdouble lambda, jint niters) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_denoise_tvl1(mats.data(), static_cast<int>(mats.size()), lambda,
                                      niters));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPhoto_decolor(JNIEnv *env, jobject, jlong src) {
    cvk_mat_t *grayscale = nullptr;
    cvk_mat_t *color_boost = nullptr;
    cvk_decolor(as_mat(src), &grayscale, &color_boost);
    cvk_mat_t *out[2] = {grayscale, color_boost};
    return handles_to_array(env, out, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_seamlessClone(JNIEnv *, jobject, jlong src, jlong dst,
                                             jlong mask, jint p_x, jint p_y, jint flags) {
    return as_handle(cvk_seamless_clone(as_mat(src), as_mat(dst), as_mat(mask), p_x, p_y,
                                        flags));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorChange(JNIEnv *, jobject, jlong src, jlong mask,
                                           jfloat red_mul, jfloat green_mul,
                                           jfloat blue_mul) {
    return as_handle(cvk_color_change(as_mat(src), as_mat(mask), red_mul, green_mul,
                                      blue_mul));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_illuminationChange(JNIEnv *, jobject, jlong src, jlong mask,
                                                  jfloat alpha, jfloat beta) {
    return as_handle(cvk_illumination_change(as_mat(src), as_mat(mask), alpha, beta));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_textureFlattening(JNIEnv *, jobject, jlong src, jlong mask,
                                                 jfloat low_threshold, jfloat high_threshold,
                                                 jint kernel_size) {
    return as_handle(cvk_texture_flattening(as_mat(src), as_mat(mask), low_threshold,
                                            high_threshold, kernel_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_edgePreservingFilter(JNIEnv *, jobject, jlong src, jint flags,
                                                    jfloat sigma_s, jfloat sigma_r) {
    return as_handle(cvk_edge_preserving_filter(as_mat(src), flags, sigma_s, sigma_r));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_detailEnhance(JNIEnv *, jobject, jlong src, jfloat sigma_s,
                                             jfloat sigma_r) {
    return as_handle(cvk_detail_enhance(as_mat(src), sigma_s, sigma_r));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPhoto_pencilSketch(JNIEnv *env, jobject, jlong src, jfloat sigma_s,
                                            jfloat sigma_r, jfloat shade_factor) {
    cvk_mat_t *dst1 = nullptr;
    cvk_mat_t *dst2 = nullptr;
    cvk_pencil_sketch(as_mat(src), &dst1, &dst2, sigma_s, sigma_r, shade_factor);
    cvk_mat_t *out[2] = {dst1, dst2};
    return handles_to_array(env, out, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_stylization(JNIEnv *, jobject, jlong src, jfloat sigma_s,
                                           jfloat sigma_r) {
    return as_handle(cvk_stylization(as_mat(src), sigma_s, sigma_r));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_correctChromaticAberration(JNIEnv *, jobject, jlong input,
                                                          jlong coefficients, jint width,
                                                          jint height, jint calib_degree,
                                                          jint bayer_pattern) {
    return as_handle(cvk_correct_chromatic_aberration(as_mat(input), as_mat(coefficients),
                                                      width, height, calib_degree,
                                                      bayer_pattern));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_gammaCorrection(JNIEnv *, jobject, jlong src, jdouble gamma) {
    return as_handle(cvk_gamma_correction(as_mat(src), gamma));
}

/* =========================================================================
 * Algorithm subclasses: clear/empty/save/getDefaultName + release
 * ========================================================================= */

#define PHOTO_ALG_JNI(kotlin_name, c_prefix)                                          \
    JNIEXPORT void JNICALL                                                            \
    Java_cn_enaium_opencv_JniPhoto_##kotlin_name##Clear(JNIEnv *, jobject, jlong h) { \
        cvk_##c_prefix##_clear(as_##c_prefix(h));                                     \
    }                                                                                 \
    JNIEXPORT jboolean JNICALL                                                        \
    Java_cn_enaium_opencv_JniPhoto_##kotlin_name##Empty(JNIEnv *, jobject, jlong h) { \
        return cvk_##c_prefix##_empty(as_##c_prefix(h)) ? JNI_TRUE : JNI_FALSE;       \
    }                                                                                 \
    JNIEXPORT void JNICALL                                                            \
    Java_cn_enaium_opencv_JniPhoto_##kotlin_name##Save(JNIEnv *env, jobject, jlong h, \
                                                       jstring filename) {            \
        const char *name = env->GetStringUTFChars(filename, nullptr);                 \
        if (name != nullptr) {                                                        \
            cvk_##c_prefix##_save(as_##c_prefix(h), name);                            \
            env->ReleaseStringUTFChars(filename, name);                               \
        }                                                                             \
    }                                                                                 \
    JNIEXPORT jstring JNICALL                                                         \
    Java_cn_enaium_opencv_JniPhoto_##kotlin_name##GetDefaultName(JNIEnv *env,         \
                                                                 jobject, jlong h) {  \
        const char *name = cvk_##c_prefix##_get_default_name(as_##c_prefix(h));       \
        return name != nullptr ? env->NewStringUTF(name) : nullptr;                   \
    }                                                                                 \
    JNIEXPORT void JNICALL                                                            \
    Java_cn_enaium_opencv_JniPhoto_##kotlin_name##Release(JNIEnv *, jobject, jlong h) { \
        cvk_##c_prefix##_release(as_##c_prefix(h));                                   \
    }

PHOTO_ALG_JNI(tonemap, tonemap)
PHOTO_ALG_JNI(tonemapDrago, tonemap_drago)
PHOTO_ALG_JNI(tonemapMantiuk, tonemap_mantiuk)
PHOTO_ALG_JNI(tonemapReinhard, tonemap_reinhard)
PHOTO_ALG_JNI(alignMtb, align_mtb)
PHOTO_ALG_JNI(calibrateDebevec, calibrate_debevec)
PHOTO_ALG_JNI(calibrateRobertson, calibrate_robertson)
PHOTO_ALG_JNI(mergeDebevec, merge_debevec)
PHOTO_ALG_JNI(mergeMertens, merge_mertens)
PHOTO_ALG_JNI(mergeRobertson, merge_robertson)

#undef PHOTO_ALG_JNI

/* =========================================================================
 * Tonemap
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapCreate(JNIEnv *, jobject, jfloat gamma) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_tonemap_create(gamma)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapProcess(JNIEnv *, jobject, jlong h, jlong src) {
    return as_handle(cvk_tonemap_process(as_tonemap(h), as_mat(src)));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapGetGamma(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_get_gamma(as_tonemap(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapSetGamma(JNIEnv *, jobject, jlong h, jfloat gamma) {
    cvk_tonemap_set_gamma(as_tonemap(h), gamma);
}

/* =========================================================================
 * TonemapDrago
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoCreate(JNIEnv *, jobject, jfloat gamma,
                                                  jfloat saturation, jfloat bias) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_tonemap_drago_create(gamma, saturation, bias)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoProcess(JNIEnv *, jobject, jlong h, jlong src) {
    return as_handle(cvk_tonemap_drago_process(as_tonemap_drago(h), as_mat(src)));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoGetGamma(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_drago_get_gamma(as_tonemap_drago(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoSetGamma(JNIEnv *, jobject, jlong h, jfloat gamma) {
    cvk_tonemap_drago_set_gamma(as_tonemap_drago(h), gamma);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoGetSaturation(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_drago_get_saturation(as_tonemap_drago(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoSetSaturation(JNIEnv *, jobject, jlong h,
                                                         jfloat saturation) {
    cvk_tonemap_drago_set_saturation(as_tonemap_drago(h), saturation);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoGetBias(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_drago_get_bias(as_tonemap_drago(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapDragoSetBias(JNIEnv *, jobject, jlong h, jfloat bias) {
    cvk_tonemap_drago_set_bias(as_tonemap_drago(h), bias);
}

/* =========================================================================
 * TonemapMantiuk
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukCreate(JNIEnv *, jobject, jfloat gamma,
                                                    jfloat scale, jfloat saturation) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_tonemap_mantiuk_create(gamma, scale, saturation)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukProcess(JNIEnv *, jobject, jlong h, jlong src) {
    return as_handle(cvk_tonemap_mantiuk_process(as_tonemap_mantiuk(h), as_mat(src)));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukGetGamma(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_mantiuk_get_gamma(as_tonemap_mantiuk(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukSetGamma(JNIEnv *, jobject, jlong h,
                                                      jfloat gamma) {
    cvk_tonemap_mantiuk_set_gamma(as_tonemap_mantiuk(h), gamma);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukGetScale(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_mantiuk_get_scale(as_tonemap_mantiuk(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukSetScale(JNIEnv *, jobject, jlong h,
                                                      jfloat scale) {
    cvk_tonemap_mantiuk_set_scale(as_tonemap_mantiuk(h), scale);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukGetSaturation(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_mantiuk_get_saturation(as_tonemap_mantiuk(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapMantiukSetSaturation(JNIEnv *, jobject, jlong h,
                                                           jfloat saturation) {
    cvk_tonemap_mantiuk_set_saturation(as_tonemap_mantiuk(h), saturation);
}

/* =========================================================================
 * TonemapReinhard
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardCreate(JNIEnv *, jobject, jfloat gamma,
                                                     jfloat intensity, jfloat light_adapt,
                                                     jfloat color_adapt) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_tonemap_reinhard_create(gamma, intensity, light_adapt, color_adapt)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardProcess(JNIEnv *, jobject, jlong h, jlong src) {
    return as_handle(cvk_tonemap_reinhard_process(as_tonemap_reinhard(h), as_mat(src)));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardGetGamma(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_reinhard_get_gamma(as_tonemap_reinhard(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardSetGamma(JNIEnv *, jobject, jlong h,
                                                       jfloat gamma) {
    cvk_tonemap_reinhard_set_gamma(as_tonemap_reinhard(h), gamma);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardGetIntensity(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_reinhard_get_intensity(as_tonemap_reinhard(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardSetIntensity(JNIEnv *, jobject, jlong h,
                                                           jfloat intensity) {
    cvk_tonemap_reinhard_set_intensity(as_tonemap_reinhard(h), intensity);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardGetLightAdaptation(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_reinhard_get_light_adaptation(as_tonemap_reinhard(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardSetLightAdaptation(JNIEnv *, jobject, jlong h,
                                                                 jfloat light_adapt) {
    cvk_tonemap_reinhard_set_light_adaptation(as_tonemap_reinhard(h), light_adapt);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardGetColorAdaptation(JNIEnv *, jobject, jlong h) {
    return cvk_tonemap_reinhard_get_color_adaptation(as_tonemap_reinhard(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_tonemapReinhardSetColorAdaptation(JNIEnv *, jobject, jlong h,
                                                                 jfloat color_adapt) {
    cvk_tonemap_reinhard_set_color_adaptation(as_tonemap_reinhard(h), color_adapt);
}

/* =========================================================================
 * AlignMTB
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbCreate(JNIEnv *, jobject, jint max_bits,
                                              jint exclude_range, jboolean cut) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_align_mtb_create(max_bits, exclude_range, cut ? 1 : 0)));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbProcess(JNIEnv *env, jobject, jlong h,
                                               jlongArray src) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return nullptr;
    const int count = static_cast<int>(mats.size());
    std::vector<cvk_mat_t *> out(static_cast<size_t>(count), nullptr);
    const int written = cvk_align_mtb_process(as_align_mtb(h), mats.data(), count,
                                              out.data(), count);
    if (written <= 0) return nullptr;
    return handles_to_array(env, out.data(), written);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbProcessTimes(JNIEnv *env, jobject, jlong h,
                                                    jlongArray src, jlong times,
                                                    jlong response) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return nullptr;
    const int count = static_cast<int>(mats.size());
    std::vector<cvk_mat_t *> out(static_cast<size_t>(count), nullptr);
    const int written = cvk_align_mtb_process_times(as_align_mtb(h), mats.data(), count,
                                                    as_mat(times), as_mat(response),
                                                    out.data(), count);
    if (written <= 0) return nullptr;
    return handles_to_array(env, out.data(), written);
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbCalculateShift(JNIEnv *env, jobject, jlong h,
                                                      jlong img0, jlong img1) {
    int out_xy[2] = {0, 0};
    cvk_align_mtb_calculate_shift(as_align_mtb(h), as_mat(img0), as_mat(img1), out_xy);
    jintArray result = env->NewIntArray(2);
    if (result == nullptr) return nullptr;
    env->SetIntArrayRegion(result, 0, 2, out_xy);
    return result;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbShiftMat(JNIEnv *, jobject, jlong h, jlong src,
                                                jint shift_x, jint shift_y) {
    return as_handle(cvk_align_mtb_shift_mat(as_align_mtb(h), as_mat(src), shift_x,
                                             shift_y));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbComputeBitmaps(JNIEnv *env, jobject, jlong h,
                                                      jlong img) {
    cvk_mat_t *tb = nullptr;
    cvk_mat_t *eb = nullptr;
    cvk_align_mtb_compute_bitmaps(as_align_mtb(h), as_mat(img), &tb, &eb);
    cvk_mat_t *out[2] = {tb, eb};
    return handles_to_array(env, out, 2);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbGetMaxBits(JNIEnv *, jobject, jlong h) {
    return cvk_align_mtb_get_max_bits(as_align_mtb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbSetMaxBits(JNIEnv *, jobject, jlong h, jint value) {
    cvk_align_mtb_set_max_bits(as_align_mtb(h), value);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbGetExcludeRange(JNIEnv *, jobject, jlong h) {
    return cvk_align_mtb_get_exclude_range(as_align_mtb(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbSetExcludeRange(JNIEnv *, jobject, jlong h,
                                                       jint value) {
    cvk_align_mtb_set_exclude_range(as_align_mtb(h), value);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbGetCut(JNIEnv *, jobject, jlong h) {
    return cvk_align_mtb_get_cut(as_align_mtb(h)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_alignMtbSetCut(JNIEnv *, jobject, jlong h, jboolean value) {
    cvk_align_mtb_set_cut(as_align_mtb(h), value ? 1 : 0);
}

/* =========================================================================
 * CalibrateDebevec
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecCreate(JNIEnv *, jobject, jint samples,
                                                      jfloat lambda, jboolean random) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_calibrate_debevec_create(samples, lambda, random ? 1 : 0)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecProcess(JNIEnv *env, jobject, jlong h,
                                                       jlongArray src, jlong times) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_calibrate_debevec_process(
            as_calibrate_debevec(h), mats.data(), static_cast<int>(mats.size()),
            as_mat(times)));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecGetLambda(JNIEnv *, jobject, jlong h) {
    return cvk_calibrate_debevec_get_lambda(as_calibrate_debevec(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecSetLambda(JNIEnv *, jobject, jlong h,
                                                         jfloat lambda) {
    cvk_calibrate_debevec_set_lambda(as_calibrate_debevec(h), lambda);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecGetSamples(JNIEnv *, jobject, jlong h) {
    return cvk_calibrate_debevec_get_samples(as_calibrate_debevec(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecSetSamples(JNIEnv *, jobject, jlong h,
                                                          jint samples) {
    cvk_calibrate_debevec_set_samples(as_calibrate_debevec(h), samples);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecGetRandom(JNIEnv *, jobject, jlong h) {
    return cvk_calibrate_debevec_get_random(as_calibrate_debevec(h)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateDebevecSetRandom(JNIEnv *, jobject, jlong h,
                                                         jboolean random) {
    cvk_calibrate_debevec_set_random(as_calibrate_debevec(h), random ? 1 : 0);
}

/* =========================================================================
 * CalibrateRobertson
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateRobertsonCreate(JNIEnv *, jobject, jint max_iter,
                                                        jfloat threshold) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_calibrate_robertson_create(max_iter, threshold)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateRobertsonProcess(JNIEnv *env, jobject, jlong h,
                                                         jlongArray src, jlong times) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_calibrate_robertson_process(
            as_calibrate_robertson(h), mats.data(), static_cast<int>(mats.size()),
            as_mat(times)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateRobertsonGetRadiance(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_calibrate_robertson_get_radiance(as_calibrate_robertson(h)));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateRobertsonGetMaxIter(JNIEnv *, jobject, jlong h) {
    return cvk_calibrate_robertson_get_max_iter(as_calibrate_robertson(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateRobertsonSetMaxIter(JNIEnv *, jobject, jlong h,
                                                            jint max_iter) {
    cvk_calibrate_robertson_set_max_iter(as_calibrate_robertson(h), max_iter);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateRobertsonGetThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_calibrate_robertson_get_threshold(as_calibrate_robertson(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_calibrateRobertsonSetThreshold(JNIEnv *, jobject, jlong h,
                                                              jfloat threshold) {
    cvk_calibrate_robertson_set_threshold(as_calibrate_robertson(h), threshold);
}

/* =========================================================================
 * MergeDebevec
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeDebevecCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_merge_debevec_create()));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeDebevecProcess(JNIEnv *env, jobject, jlong h,
                                                   jlongArray src, jlong times) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_merge_debevec_process(
            as_merge_debevec(h), mats.data(), static_cast<int>(mats.size()),
            as_mat(times)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeDebevecProcessResponse(JNIEnv *env, jobject, jlong h,
                                                           jlongArray src, jlong times,
                                                           jlong response) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_merge_debevec_process_response(
            as_merge_debevec(h), mats.data(), static_cast<int>(mats.size()),
            as_mat(times), as_mat(response)));
}

/* =========================================================================
 * MergeMertens
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensCreate(JNIEnv *, jobject, jfloat contrast_weight,
                                                  jfloat saturation_weight,
                                                  jfloat exposure_weight) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_merge_mertens_create(contrast_weight, saturation_weight, exposure_weight)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensProcess(JNIEnv *env, jobject, jlong h,
                                                   jlongArray src) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_merge_mertens_process(as_merge_mertens(h), mats.data(),
                                               static_cast<int>(mats.size())));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensProcessResponse(JNIEnv *env, jobject, jlong h,
                                                           jlongArray src, jlong times,
                                                           jlong response) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_merge_mertens_process_response(
            as_merge_mertens(h), mats.data(), static_cast<int>(mats.size()),
            as_mat(times), as_mat(response)));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensGetContrastWeight(JNIEnv *, jobject, jlong h) {
    return cvk_merge_mertens_get_contrast_weight(as_merge_mertens(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensSetContrastWeight(JNIEnv *, jobject, jlong h,
                                                             jfloat value) {
    cvk_merge_mertens_set_contrast_weight(as_merge_mertens(h), value);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensGetSaturationWeight(JNIEnv *, jobject, jlong h) {
    return cvk_merge_mertens_get_saturation_weight(as_merge_mertens(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensSetSaturationWeight(JNIEnv *, jobject, jlong h,
                                                               jfloat value) {
    cvk_merge_mertens_set_saturation_weight(as_merge_mertens(h), value);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensGetExposureWeight(JNIEnv *, jobject, jlong h) {
    return cvk_merge_mertens_get_exposure_weight(as_merge_mertens(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeMertensSetExposureWeight(JNIEnv *, jobject, jlong h,
                                                             jfloat value) {
    cvk_merge_mertens_set_exposure_weight(as_merge_mertens(h), value);
}

/* =========================================================================
 * MergeRobertson
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeRobertsonCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_merge_robertson_create()));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeRobertsonProcess(JNIEnv *env, jobject, jlong h,
                                                     jlongArray src, jlong times) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_merge_robertson_process(
            as_merge_robertson(h), mats.data(), static_cast<int>(mats.size()),
            as_mat(times)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_mergeRobertsonProcessResponse(JNIEnv *env, jobject, jlong h,
                                                             jlongArray src, jlong times,
                                                             jlong response) {
    const std::vector<const cvk_mat_t *> mats = mats_from(env, src);
    if (mats.empty()) return 0;
    return as_handle(cvk_merge_robertson_process_response(
            as_merge_robertson(h), mats.data(), static_cast<int>(mats.size()),
            as_mat(times), as_mat(response)));
}

/* =========================================================================
 * ColorCorrectionModel
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelCreate(JNIEnv *, jobject, jlong src,
                                                          jint const_color) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_color_correction_model_create(as_mat(src), const_color)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelCreateEmpty(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_color_correction_model_create_empty()));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetLinearizationGamma(JNIEnv *, jobject,
                                                                         jlong h,
                                                                         jdouble gamma) {
    cvk_color_correction_model_set_linearization_gamma(as_color_correction_model(h), gamma);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetLinearizationDegree(JNIEnv *, jobject,
                                                                          jlong h, jint deg) {
    cvk_color_correction_model_set_linearization_degree(as_color_correction_model(h), deg);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetSaturatedThreshold(JNIEnv *, jobject,
                                                                         jlong h,
                                                                         jdouble lower,
                                                                         jdouble upper) {
    cvk_color_correction_model_set_saturated_threshold(as_color_correction_model(h), lower,
                                                       upper);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetWeightsList(JNIEnv *, jobject, jlong h,
                                                                  jlong weights) {
    cvk_color_correction_model_set_weights_list(as_color_correction_model(h),
                                                as_mat(weights));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetWeightCoeff(JNIEnv *, jobject, jlong h,
                                                                  jdouble coeff) {
    cvk_color_correction_model_set_weight_coeff(as_color_correction_model(h), coeff);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetMaxCount(JNIEnv *, jobject, jlong h,
                                                               jint max_count) {
    cvk_color_correction_model_set_max_count(as_color_correction_model(h), max_count);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetEpsilon(JNIEnv *, jobject, jlong h,
                                                              jdouble epsilon) {
    cvk_color_correction_model_set_epsilon(as_color_correction_model(h), epsilon);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelSetRgb(JNIEnv *, jobject, jlong h,
                                                          jboolean rgb) {
    cvk_color_correction_model_set_rgb(as_color_correction_model(h), rgb ? 1 : 0);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelCompute(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_color_correction_model_compute(as_color_correction_model(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelGetColorCorrectionMatrix(JNIEnv *,
                                                                            jobject,
                                                                            jlong h) {
    return as_handle(cvk_color_correction_model_get_color_correction_matrix(
            as_color_correction_model(h)));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelGetLoss(JNIEnv *, jobject, jlong h) {
    return cvk_color_correction_model_get_loss(as_color_correction_model(h));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelGetSrcLinearRgb(JNIEnv *, jobject,
                                                                   jlong h) {
    return as_handle(cvk_color_correction_model_get_src_linear_rgb(
            as_color_correction_model(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelGetRefLinearRgb(JNIEnv *, jobject,
                                                                   jlong h) {
    return as_handle(cvk_color_correction_model_get_ref_linear_rgb(
            as_color_correction_model(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelGetMask(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_color_correction_model_get_mask(as_color_correction_model(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelGetWeights(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_color_correction_model_get_weights(as_color_correction_model(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelCorrectImage(JNIEnv *, jobject, jlong h,
                                                                jlong src, jboolean islinear) {
    return as_handle(cvk_color_correction_model_correct_image(as_color_correction_model(h),
                                                              as_mat(src),
                                                              islinear ? 1 : 0));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_colorCorrectionModelRelease(JNIEnv *, jobject, jlong h) {
    cvk_color_correction_model_release(as_color_correction_model(h));
}

/* =========================================================================
 * IntelligentScissorsMB
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
            cvk_intelligent_scissors_mb_create()));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsSetWeights(JNIEnv *, jobject, jlong h,
                                                  jfloat weight_non_edge,
                                                  jfloat weight_gradient_direction,
                                                  jfloat weight_gradient_magnitude) {
    cvk_intelligent_scissors_mb_set_weights(as_intelligent_scissors_mb(h), weight_non_edge,
                                            weight_gradient_direction,
                                            weight_gradient_magnitude);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsSetGradientMagnitudeMaxLimit(JNIEnv *, jobject,
                                                                    jlong h, jfloat limit) {
    cvk_intelligent_scissors_mb_set_gradient_magnitude_max_limit(
            as_intelligent_scissors_mb(h), limit);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsSetEdgeFeatureZeroCrossingParameters(JNIEnv *,
                                                                            jobject, jlong h,
                                                                            jfloat min_value) {
    cvk_intelligent_scissors_mb_set_edge_feature_zero_crossing_parameters(
            as_intelligent_scissors_mb(h), min_value);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsSetEdgeFeatureCannyParameters(JNIEnv *, jobject,
                                                                     jlong h, jdouble t1,
                                                                     jdouble t2,
                                                                     jint aperture_size,
                                                                     jboolean l2gradient) {
    cvk_intelligent_scissors_mb_set_edge_feature_canny_parameters(
            as_intelligent_scissors_mb(h), t1, t2, aperture_size, l2gradient ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsApplyImage(JNIEnv *, jobject, jlong h, jlong image) {
    cvk_intelligent_scissors_mb_apply_image(as_intelligent_scissors_mb(h), as_mat(image));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsApplyImageFeatures(JNIEnv *, jobject, jlong h,
                                                          jlong non_edge,
                                                          jlong gradient_direction,
                                                          jlong gradient_magnitude,
                                                          jlong image) {
    cvk_intelligent_scissors_mb_apply_image_features(
            as_intelligent_scissors_mb(h), as_mat(non_edge), as_mat(gradient_direction),
            as_mat(gradient_magnitude), as_mat(image));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsBuildMap(JNIEnv *, jobject, jlong h, jint x, jint y) {
    cvk_intelligent_scissors_mb_build_map(as_intelligent_scissors_mb(h), x, y);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsGetContour(JNIEnv *, jobject, jlong h, jint x,
                                                  jint y, jboolean backward) {
    return as_handle(cvk_intelligent_scissors_mb_get_contour(
            as_intelligent_scissors_mb(h), x, y, backward ? 1 : 0));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniPhoto_scissorsRelease(JNIEnv *, jobject, jlong h) {
    cvk_intelligent_scissors_mb_release(as_intelligent_scissors_mb(h));
}

} /* extern "C" */

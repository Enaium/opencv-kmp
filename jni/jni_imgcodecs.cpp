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
 * JNI bridge for the imgcodecs slice: thin Java_cn_enaium_opencv_JniImgcodecs_*
 * wrappers around the cvk_ multi-image / Animation ABI in shim_imgcodecs.cpp.
 * Mat lists cross the boundary as jlongArray (handles), image lists as
 * jlongArray inputs, and codec parameters as jintArray. No exceptions may
 * cross the JNI boundary — the shim is noexcept and returns NULL/0/false on
 * failure.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_imgcodecs.h"

#include <cstdint>
#include <cstring>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_animation_t *as_animation(jlong handle) {
    return reinterpret_cast<cvk_animation_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_animation_handle(const cvk_animation_t *animation) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(animation));
}

/** Wraps freshly allocated Mat handles into a jlongArray, then frees the list. */
static jlongArray take_mat_list(JNIEnv *env, cvk_mat_list_t *list) {
    if (list == nullptr) return nullptr;
    std::vector<jlong> values(list->count);
    for (size_t i = 0; i < list->count; ++i) {
        values[i] = as_handle(list->items[i]);
    }
    cvk_mat_list_release(list);
    jlongArray result = env->NewLongArray(static_cast<jsize>(values.size()));
    if (result == nullptr) {
        for (size_t i = 0; i < values.size(); ++i) cvk_mat_release(as_mat(values[i]));
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
    return result;
}

/** Copies a jlongArray of Mat handles into a std::vector; empty on null input. */
static std::vector<const cvk_mat_t *> mat_handles(JNIEnv *env, jlongArray handles) {
    std::vector<const cvk_mat_t *> mats;
    if (handles == nullptr) return mats;
    const jsize count = env->GetArrayLength(handles);
    if (count <= 0) return mats;
    jlong *elements = env->GetLongArrayElements(handles, nullptr);
    if (elements == nullptr) return mats;
    mats.reserve(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        mats.push_back(as_mat(elements[i]));
    }
    env->ReleaseLongArrayElements(handles, elements, JNI_ABORT);
    return mats;
}

static std::vector<int> int_params(JNIEnv *env, jintArray params) {
    std::vector<int> values;
    if (params == nullptr) return values;
    const jsize length = env->GetArrayLength(params);
    if (length <= 0) return values;
    values.resize(static_cast<size_t>(length));
    env->GetIntArrayRegion(params, 0, length, values.data());
    return values;
}

/** Copies a malloc'd cvk_ buffer into a fresh jbyteArray, then frees it. */
static jbyteArray take_buffer(JNIEnv *env, unsigned char *buffer, size_t length) {
    if (buffer == nullptr) return nullptr;
    jbyteArray out = env->NewByteArray(static_cast<jsize>(length));
    if (out == nullptr) {
        cvk_free_buffer(buffer);
        return nullptr;
    }
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(length),
                            reinterpret_cast<const jbyte *>(buffer));
    cvk_free_buffer(buffer);
    return out;
}

extern "C" {

// --------------------------------------------------- multi-image codecs

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imreadmulti(JNIEnv *env, jobject, jstring path, jint flags) {
    if (path == nullptr) return nullptr;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return nullptr;
    cvk_mat_list_t *list = cvk_imreadmulti(utf, flags);
    env->ReleaseStringUTFChars(path, utf);
    return take_mat_list(env, list);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imreadmultiRange(JNIEnv *env, jobject, jstring path,
                                                    jint start, jint count, jint flags) {
    if (path == nullptr) return nullptr;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return nullptr;
    cvk_mat_list_t *list = cvk_imreadmulti_range(utf, start, count, flags);
    env->ReleaseStringUTFChars(path, utf);
    return take_mat_list(env, list);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imdecodemulti(JNIEnv *env, jobject, jlong buf,
                                                 jint flags, jint start, jint count) {
    return take_mat_list(env, cvk_imdecodemulti(as_mat(buf), flags, start, count));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imwritemulti(JNIEnv *env, jobject, jstring path,
                                                jlongArray mats, jintArray params) {
    if (path == nullptr) return JNI_FALSE;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return JNI_FALSE;
    std::vector<const cvk_mat_t *> images = mat_handles(env, mats);
    std::vector<int> values = int_params(env, params);
    const int ok = cvk_imwritemulti(utf, images.data(), images.size(),
                                    values.data(), values.size());
    env->ReleaseStringUTFChars(path, utf);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imencodemulti(JNIEnv *env, jobject, jstring ext,
                                                 jlongArray mats, jintArray params) {
    if (ext == nullptr) return nullptr;
    const char *utf = env->GetStringUTFChars(ext, nullptr);
    if (utf == nullptr) return nullptr;
    std::vector<const cvk_mat_t *> images = mat_handles(env, mats);
    std::vector<int> values = int_params(env, params);
    size_t length = 0;
    unsigned char *buffer = cvk_imencodemulti(utf, images.data(), images.size(),
                                              values.data(), values.size(), &length);
    env->ReleaseStringUTFChars(ext, utf);
    return take_buffer(env, buffer, length);
}

// --------------------------------------------------- Animation

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationCreate(JNIEnv *, jobject, jint loopCount,
                                                   jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    return as_animation_handle(cvk_animation_create(loopCount, v0, v1, v2, v3));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationRelease(JNIEnv *, jobject, jlong animation) {
    cvk_animation_release(as_animation(animation));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationGetLoopCount(JNIEnv *, jobject, jlong animation) {
    return cvk_animation_get_loop_count(as_animation(animation));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationSetLoopCount(JNIEnv *, jobject, jlong animation,
                                                         jint loopCount) {
    cvk_animation_set_loop_count(as_animation(animation), loopCount);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationGetBgColor(JNIEnv *env, jobject, jlong animation) {
    const cvk_scalar_t s = cvk_animation_get_bgcolor(as_animation(animation));
    jdouble values[4] = {s.v0, s.v1, s.v2, s.v3};
    jdoubleArray out = env->NewDoubleArray(4);
    if (out == nullptr) return nullptr;
    env->SetDoubleArrayRegion(out, 0, 4, values);
    return out;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationSetBgColor(JNIEnv *, jobject, jlong animation,
                                                       jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    cvk_scalar_t bg;
    bg.v0 = v0;
    bg.v1 = v1;
    bg.v2 = v2;
    bg.v3 = v3;
    cvk_animation_set_bgcolor(as_animation(animation), bg);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationGetDurations(JNIEnv *, jobject, jlong animation) {
    return as_handle(cvk_animation_get_durations(as_animation(animation)));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationSetDurations(JNIEnv *, jobject, jlong animation,
                                                         jlong durations) {
    return cvk_animation_set_durations(as_animation(animation), as_mat(durations)) != 0
                   ? JNI_TRUE
                   : JNI_FALSE;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationGetFrames(JNIEnv *env, jobject, jlong animation) {
    cvk_mat_list_t *list = cvk_animation_get_frames(as_animation(animation));
    return take_mat_list(env, list);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationSetFrames(JNIEnv *env, jobject, jlong animation,
                                                      jlongArray frames) {
    std::vector<const cvk_mat_t *> mats = mat_handles(env, frames);
    return cvk_animation_set_frames(as_animation(animation), mats.data(), mats.size()) != 0
                   ? JNI_TRUE
                   : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationGetStillImage(JNIEnv *, jobject, jlong animation) {
    return as_handle(cvk_animation_get_still_image(as_animation(animation)));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgcodecs_animationSetStillImage(JNIEnv *, jobject, jlong animation,
                                                          jlong image) {
    return cvk_animation_set_still_image(as_animation(animation), as_mat(image)) != 0
                   ? JNI_TRUE
                   : JNI_FALSE;
}

// --------------------------------------------------- animation codecs

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imreadanimation(JNIEnv *env, jobject, jstring path,
                                                   jlong animation, jint start, jint count) {
    if (path == nullptr) return JNI_FALSE;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return JNI_FALSE;
    const int ok = cvk_imreadanimation(utf, as_animation(animation), start, count);
    env->ReleaseStringUTFChars(path, utf);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imdecodeanimation(JNIEnv *env, jobject, jbyteArray data,
                                                     jlong animation, jint start, jint count) {
    if (data == nullptr || animation == 0) return JNI_FALSE;
    const jsize len = env->GetArrayLength(data);
    jbyte *bytes = env->GetByteArrayElements(data, nullptr);
    if (bytes == nullptr) return JNI_FALSE;
    // Wrap the encoded bytes in a CV_8UC1 Mat (type id 0) mirroring
    // cvk_imdecode's internal Mat1b.
    cvk_mat_t *buf = cvk_mat_create(1, len, 0);
    if (buf == nullptr) {
        env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
        return JNI_FALSE;
    }
    unsigned char *pixels = cvk_mat_data(buf);
    if (pixels != nullptr && len > 0) {
        std::memcpy(pixels, bytes, static_cast<size_t>(len));
    }
    env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
    const int ok = cvk_imdecodeanimation(buf, as_animation(animation), start, count);
    cvk_mat_release(buf);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imwriteanimation(JNIEnv *env, jobject, jstring path,
                                                    jlong animation, jintArray params) {
    if (path == nullptr) return JNI_FALSE;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return JNI_FALSE;
    std::vector<int> values = int_params(env, params);
    const int ok = cvk_imwriteanimation(utf, as_animation(animation),
                                        values.data(), values.size());
    env->ReleaseStringUTFChars(path, utf);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniImgcodecs_imencodeanimation(JNIEnv *env, jobject, jstring ext,
                                                     jlong animation, jintArray params) {
    if (ext == nullptr) return nullptr;
    const char *utf = env->GetStringUTFChars(ext, nullptr);
    if (utf == nullptr) return nullptr;
    std::vector<int> values = int_params(env, params);
    size_t length = 0;
    unsigned char *buffer = cvk_imencodeanimation(utf, as_animation(animation),
                                                  values.data(), values.size(), &length);
    env->ReleaseStringUTFChars(ext, utf);
    return take_buffer(env, buffer, length);
}

} /* extern "C" */

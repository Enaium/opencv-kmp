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
 * JNI bridge for the videoio slice: thin Java_cn_enaium_opencv_JniVideoio_*
 * wrappers around the cvk_ C ABI in native/shim_videoio.cpp. Handles arrive
 * as jlong pointers; the shim is noexcept, so no exceptions cross JNI.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_videoio.h"

#include <cstdint>
#include <string>

static inline cvk_video_capture_t *as_capture(jlong handle) {
    return reinterpret_cast<cvk_video_capture_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_capture_handle(const cvk_video_capture_t *capture) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(capture));
}

static inline cvk_video_writer_t *as_writer(jlong handle) {
    return reinterpret_cast<cvk_video_writer_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_writer_handle(const cvk_video_writer_t *writer) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(writer));
}

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

/** Copies a jstring into `out`; returns false when the string is unavailable. */
static bool jstr_copy(JNIEnv *env, jstring s, std::string &out) {
    if (s == nullptr) {
        return false;
    }
    const char *utf = env->GetStringUTFChars(s, nullptr);
    if (utf == nullptr) {
        return false;
    }
    out.assign(utf);
    env->ReleaseStringUTFChars(s, utf);
    return true;
}

extern "C" {

// ---------------------------------------------------------------- VideoCapture

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_captureCreate(JNIEnv *, jobject) {
    return as_capture_handle(cvk_video_capture_create());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_captureCreateIndex(JNIEnv *, jobject, jint index,
                                                    jint apiPreference) {
    return as_capture_handle(cvk_video_capture_create_index(index, apiPreference));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_captureCreateFile(JNIEnv *env, jobject, jstring filename,
                                                   jint apiPreference) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return 0;
    }
    jlong handle = as_capture_handle(cvk_video_capture_create_file(name.c_str(), apiPreference));
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_captureCreateIndexParams(JNIEnv *, jobject, jint index,
                                                          jint apiPreference, jlong params) {
    return as_capture_handle(cvk_video_capture_create_index_params(index, apiPreference,
                                                                   as_mat(params)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_captureCreateFileParams(JNIEnv *env, jobject,
                                                         jstring filename, jint apiPreference,
                                                         jlong params) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return 0;
    }
    jlong handle =
        as_capture_handle(cvk_video_capture_create_file_params(name.c_str(), apiPreference,
                                                               as_mat(params)));
    return handle;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureOpenIndex(JNIEnv *, jobject, jlong h, jint index,
                                                  jint apiPreference) {
    return cvk_video_capture_open_index(as_capture(h), index, apiPreference) != 0
               ? JNI_TRUE
               : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureOpenFile(JNIEnv *env, jobject, jlong h,
                                                 jstring filename, jint apiPreference) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return JNI_FALSE;
    }
    jboolean ok =
        cvk_video_capture_open_file(as_capture(h), name.c_str(), apiPreference) != 0 ? JNI_TRUE
                                                                             : JNI_FALSE;
    return ok;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureOpenIndexParams(JNIEnv *, jobject, jlong h, jint index,
                                                        jint apiPreference, jlong params) {
    return cvk_video_capture_open_index_params(as_capture(h), index, apiPreference,
                                               as_mat(params)) != 0
               ? JNI_TRUE
               : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureOpenFileParams(JNIEnv *env, jobject, jlong h,
                                                       jstring filename, jint apiPreference,
                                                       jlong params) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return JNI_FALSE;
    }
    jboolean ok = cvk_video_capture_open_file_params(as_capture(h), name.c_str(), apiPreference,
                                                     as_mat(params)) != 0
                      ? JNI_TRUE
                      : JNI_FALSE;
    return ok;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureIsOpened(JNIEnv *, jobject, jlong h) {
    return cvk_video_capture_is_opened(as_capture(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideoio_captureRelease(JNIEnv *, jobject, jlong h) {
    cvk_video_capture_release(as_capture(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideoio_captureDelete(JNIEnv *, jobject, jlong h) {
    cvk_video_capture_delete(as_capture(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureGrab(JNIEnv *, jobject, jlong h) {
    return cvk_video_capture_grab(as_capture(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_captureRetrieve(JNIEnv *, jobject, jlong h, jint flag) {
    return as_handle(cvk_video_capture_retrieve(as_capture(h), flag));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureRead(JNIEnv *env, jobject, jlong h, jlongArray out) {
    cvk_mat_t *frame = nullptr;
    jboolean ok = cvk_video_capture_read(as_capture(h), &frame) != 0 ? JNI_TRUE : JNI_FALSE;
    if (out != nullptr) {
        jlong handle = as_handle(frame);
        env->SetLongArrayRegion(out, 0, 1, &handle);
    }
    return ok;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureSet(JNIEnv *, jobject, jlong h, jint propId,
                                            jdouble value) {
    return cvk_video_capture_set(as_capture(h), propId, value) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniVideoio_captureGet(JNIEnv *, jobject, jlong h, jint propId) {
    return cvk_video_capture_get(as_capture(h), propId);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniVideoio_captureBackendName(JNIEnv *env, jobject, jlong h) {
    const char *name = cvk_video_capture_get_backend_name(as_capture(h));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideoio_captureSetExceptionMode(JNIEnv *, jobject, jlong h,
                                                         jboolean enable) {
    cvk_video_capture_set_exception_mode(as_capture(h), enable != JNI_FALSE);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_captureGetExceptionMode(JNIEnv *, jobject, jlong h) {
    return cvk_video_capture_get_exception_mode(as_capture(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

// ---------------------------------------------------------------- VideoWriter

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_writerCreate(JNIEnv *, jobject) {
    return as_writer_handle(cvk_video_writer_create());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_writerCreateFile(JNIEnv *env, jobject, jstring filename,
                                                  jint fourcc, jdouble fps, jint width,
                                                  jint height, jboolean isColor) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return 0;
    }
    jlong handle = as_writer_handle(cvk_video_writer_create_file(
        name.c_str(), fourcc, fps, width, height, isColor != JNI_FALSE));
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_writerCreateFileApi(JNIEnv *env, jobject, jstring filename,
                                                     jint apiPreference, jint fourcc,
                                                     jdouble fps, jint width, jint height,
                                                     jboolean isColor) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return 0;
    }
    jlong handle = as_writer_handle(cvk_video_writer_create_file_api(
        name.c_str(), apiPreference, fourcc, fps, width, height, isColor != JNI_FALSE));
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_writerCreateFileParams(JNIEnv *env, jobject, jstring filename,
                                                        jint fourcc, jdouble fps, jint width,
                                                        jint height, jlong params) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return 0;
    }
    jlong handle = as_writer_handle(cvk_video_writer_create_file_params(
        name.c_str(), fourcc, fps, width, height, as_mat(params)));
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_writerCreateFileApiParams(JNIEnv *env, jobject,
                                                           jstring filename,
                                                           jint apiPreference, jint fourcc,
                                                           jdouble fps, jint width,
                                                           jint height, jlong params) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return 0;
    }
    jlong handle = as_writer_handle(cvk_video_writer_create_file_api_params(
        name.c_str(), apiPreference, fourcc, fps, width, height, as_mat(params)));
    return handle;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_writerOpenFile(JNIEnv *env, jobject, jlong h,
                                                jstring filename, jint fourcc, jdouble fps,
                                                jint width, jint height, jboolean isColor) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return JNI_FALSE;
    }
    jboolean ok = cvk_video_writer_open_file(as_writer(h), name.c_str(), fourcc, fps, width, height,
                                             isColor != JNI_FALSE) != 0
                      ? JNI_TRUE
                      : JNI_FALSE;
    return ok;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_writerOpenFileApi(JNIEnv *env, jobject, jlong h,
                                                   jstring filename, jint apiPreference,
                                                   jint fourcc, jdouble fps, jint width,
                                                   jint height, jboolean isColor) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return JNI_FALSE;
    }
    jboolean ok = cvk_video_writer_open_file_api(as_writer(h), name.c_str(), apiPreference, fourcc,
                                                 fps, width, height, isColor != JNI_FALSE) != 0
                      ? JNI_TRUE
                      : JNI_FALSE;
    return ok;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_writerOpenFileParams(JNIEnv *env, jobject, jlong h,
                                                      jstring filename, jint fourcc,
                                                      jdouble fps, jint width, jint height,
                                                      jlong params) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return JNI_FALSE;
    }
    jboolean ok = cvk_video_writer_open_file_params(as_writer(h), name.c_str(), fourcc, fps, width,
                                                    height, as_mat(params)) != 0
                      ? JNI_TRUE
                      : JNI_FALSE;
    return ok;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_writerOpenFileApiParams(JNIEnv *env, jobject, jlong h,
                                                         jstring filename, jint apiPreference,
                                                         jint fourcc, jdouble fps, jint width,
                                                         jint height, jlong params) {
    std::string name;
    if (!jstr_copy(env, filename, name)) {
        return JNI_FALSE;
    }
    jboolean ok = cvk_video_writer_open_file_api_params(
                      as_writer(h), name.c_str(), apiPreference, fourcc, fps, width, height,
                      as_mat(params)) != 0
                      ? JNI_TRUE
                      : JNI_FALSE;
    return ok;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_writerIsOpened(JNIEnv *, jobject, jlong h) {
    return cvk_video_writer_is_opened(as_writer(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideoio_writerRelease(JNIEnv *, jobject, jlong h) {
    cvk_video_writer_release(as_writer(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideoio_writerDelete(JNIEnv *, jobject, jlong h) {
    cvk_video_writer_delete(as_writer(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_writerWrite(JNIEnv *, jobject, jlong h, jlong mat) {
    return cvk_video_writer_write(as_writer(h), as_mat(mat)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_writerSet(JNIEnv *, jobject, jlong h, jint propId,
                                           jdouble value) {
    return cvk_video_writer_set(as_writer(h), propId, value) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniVideoio_writerGet(JNIEnv *, jobject, jlong h, jint propId) {
    return cvk_video_writer_get(as_writer(h), propId);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniVideoio_writerBackendName(JNIEnv *env, jobject, jlong h) {
    const char *name = cvk_video_writer_get_backend_name(as_writer(h));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

// ------------------------------------------------------- videoio_registry

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioBackendName(JNIEnv *env, jobject, jint api) {
    const char *name = cvk_videoio_get_backend_name(api);
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioBackends(JNIEnv *, jobject) {
    return as_handle(cvk_videoio_get_backends());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioCameraBackends(JNIEnv *, jobject) {
    return as_handle(cvk_videoio_get_camera_backends());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioStreamBackends(JNIEnv *, jobject) {
    return as_handle(cvk_videoio_get_stream_backends());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioStreamBufferedBackends(JNIEnv *, jobject) {
    return as_handle(cvk_videoio_get_stream_buffered_backends());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioWriterBackends(JNIEnv *, jobject) {
    return as_handle(cvk_videoio_get_writer_backends());
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioHasBackend(JNIEnv *, jobject, jint api) {
    return cvk_videoio_has_backend(api) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioIsBackendBuiltIn(JNIEnv *, jobject, jint api) {
    return cvk_videoio_is_backend_built_in(api) != 0 ? JNI_TRUE : JNI_FALSE;
}

static jstring plugin_version(JNIEnv *env, const char *(*fn)(int, int *, int *), jint api,
                              jintArray abi, jintArray apiVersion) {
    int abi_out = -1;
    int api_out = -1;
    const char *desc = fn(api, &abi_out, &api_out);
    if (abi != nullptr) {
        jint values[1] = {abi_out};
        env->SetIntArrayRegion(abi, 0, 1, values);
    }
    if (apiVersion != nullptr) {
        jint values[1] = {api_out};
        env->SetIntArrayRegion(apiVersion, 0, 1, values);
    }
    return desc != nullptr ? env->NewStringUTF(desc) : nullptr;
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioCameraBackendPluginVersion(JNIEnv *env, jobject,
                                                                   jint api,
                                                                   jintArray abi,
                                                                   jintArray apiVersion) {
    return plugin_version(env, cvk_get_camera_plugin_version, api, abi,
                          apiVersion);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioStreamBackendPluginVersion(JNIEnv *env, jobject,
                                                                   jint api,
                                                                   jintArray abi,
                                                                   jintArray apiVersion) {
    return plugin_version(env, cvk_get_stream_plugin_version, api, abi,
                          apiVersion);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioStreamBufferedBackendPluginVersion(JNIEnv *env,
                                                                           jobject, jint api,
                                                                           jintArray abi,
                                                                           jintArray apiVersion) {
    return plugin_version(env, cvk_get_stream_buffered_plugin_version, api,
                          abi, apiVersion);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniVideoio_videoioWriterBackendPluginVersion(JNIEnv *env, jobject,
                                                                   jint api,
                                                                   jintArray abi,
                                                                   jintArray apiVersion) {
    return plugin_version(env, cvk_get_writer_plugin_version, api, abi,
                          apiVersion);
}

} /* extern "C" */

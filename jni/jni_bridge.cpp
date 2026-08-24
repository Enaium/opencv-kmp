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
 * JNI bridge: thin Java_cn_enaium_opencv_Jni_* wrappers around the cvk_ C
 * ABI in native/opencv_shim.cpp. Handles arrive as jlong cv::Mat pointers;
 * structs are expanded into primitive arguments. No exceptions may cross the
 * JNI boundary — the shim is noexcept and returns NULL/0/false on failure.
 */
#include <jni.h>
#include "opencv_kmp.h"

#include <cstdint>
#include <cstring>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static cvk_scalar_t as_scalar(JNIEnv *env, jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    (void) env;
    cvk_scalar_t s;
    s.v0 = v0;
    s.v1 = v1;
    s.v2 = v2;
    s.v3 = v3;
    return s;
}

extern "C" {

// ---------------------------------------------------------------- info

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_Jni_version(JNIEnv *env, jobject) {
    return env->NewStringUTF(cvk_version());
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_Jni_lastError(JNIEnv *env, jobject) {
    const char *message = cvk_last_error();
    return message != nullptr ? env->NewStringUTF(message) : nullptr;
}

// ------------------------------------------------------- mat lifecycle

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matCreate(JNIEnv *, jobject, jint rows, jint cols, jint type) {
    return as_handle(cvk_mat_create(rows, cols, type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matCreateFilled(JNIEnv *, jobject, jint rows, jint cols, jint type,
                                          jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    return as_handle(cvk_mat_create_filled(rows, cols, type, as_scalar(nullptr, v0, v1, v2, v3)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matZeros(JNIEnv *, jobject, jint rows, jint cols, jint type) {
    return as_handle(cvk_mat_zeros(rows, cols, type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matOnes(JNIEnv *, jobject, jint rows, jint cols, jint type) {
    return as_handle(cvk_mat_ones(rows, cols, type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matEye(JNIEnv *, jobject, jint rows, jint cols, jint type) {
    return as_handle(cvk_mat_eye(rows, cols, type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matClone(JNIEnv *, jobject, jlong mat) {
    return as_handle(cvk_mat_clone(as_mat(mat)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matRoi(JNIEnv *, jobject, jlong mat,
                                 jint x, jint y, jint width, jint height) {
    cvk_rect_t rect;
    rect.x = x;
    rect.y = y;
    rect.width = width;
    rect.height = height;
    return as_handle(cvk_mat_roi(as_mat(mat), rect));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_matRelease(JNIEnv *, jobject, jlong mat) {
    cvk_mat_release(as_mat(mat));
}

// ------------------------------------------------------ mat properties

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_matRows(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_rows(as_mat(mat));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_matCols(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_cols(as_mat(mat));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_matType(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_type(as_mat(mat));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_matChannels(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_channels(as_mat(mat));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matElemSize(JNIEnv *, jobject, jlong mat) {
    return static_cast<jlong>(cvk_mat_elem_size(as_mat(mat)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matTotal(JNIEnv *, jobject, jlong mat) {
    return static_cast<jlong>(cvk_mat_total(as_mat(mat)));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_matIsEmpty(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_is_empty(as_mat(mat)) != 0;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_matGet(JNIEnv *, jobject, jlong mat,
                                 jint row, jint col, jint channel) {
    return cvk_mat_get(as_mat(mat), row, col, channel);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_matSet(JNIEnv *, jobject, jlong mat,
                                 jint row, jint col, jint channel, jdouble value) {
    cvk_mat_set(as_mat(mat), row, col, channel, value);
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_Jni_matGetData(JNIEnv *env, jobject, jlong mat) {
    cvk_mat_t *m = as_mat(mat);
    const size_t bytes = cvk_mat_total(m) * cvk_mat_elem_size(m);
    if (bytes == 0) return env->NewByteArray(0);
    unsigned char *data = cvk_mat_data(m);
    if (data == nullptr) return env->NewByteArray(0);
    jbyteArray out = env->NewByteArray(static_cast<jsize>(bytes));
    if (out == nullptr) return nullptr;
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(bytes),
                            reinterpret_cast<const jbyte *>(data));
    return out;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_matSetData(JNIEnv *env, jobject, jlong mat, jbyteArray data) {
    cvk_mat_t *m = as_mat(mat);
    const size_t bytes = cvk_mat_total(m) * cvk_mat_elem_size(m);
    const jsize given = env->GetArrayLength(data);
    if (data == nullptr || static_cast<size_t>(given) != bytes) return;
    unsigned char *target = cvk_mat_data(m);
    if (target == nullptr) return;
    jbyte *elements = env->GetByteArrayElements(data, nullptr);
    if (elements == nullptr) return;
    memcpy(target, elements, bytes);
    env->ReleaseByteArrayElements(data, elements, JNI_ABORT);
}

// ------------------------------------------------- conversions / math

#define UNARY_OP(java_name, c_expr)                                           \
    JNIEXPORT jlong JNICALL                                                   \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong a) {       \
        return as_handle(c_expr);                                             \
    }

#define BINARY_OP(java_name, c_expr)                                          \
    JNIEXPORT jlong JNICALL                                                   \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong a,         \
                                          jlong b) {                          \
        return as_handle(c_expr);                                             \
    }

BINARY_OP(add, cvk_mat_add(as_mat(a), as_mat(b)))
BINARY_OP(subtract, cvk_mat_subtract(as_mat(a), as_mat(b)))
BINARY_OP(absdiff, cvk_mat_absdiff(as_mat(a), as_mat(b)))
BINARY_OP(bitwiseAnd, cvk_mat_bitwise_and(as_mat(a), as_mat(b)))
BINARY_OP(bitwiseOr, cvk_mat_bitwise_or(as_mat(a), as_mat(b)))
BINARY_OP(bitwiseXor, cvk_mat_bitwise_xor(as_mat(a), as_mat(b)))
BINARY_OP(min, cvk_mat_min(as_mat(a), as_mat(b)))
BINARY_OP(max, cvk_mat_max(as_mat(a), as_mat(b)))
BINARY_OP(divide, cvk_mat_divide(as_mat(a), as_mat(b)))

UNARY_OP(bitwiseNot, cvk_mat_bitwise_not(as_mat(a)))
UNARY_OP(transpose, cvk_mat_transpose(as_mat(a)))

#undef BINARY_OP
#undef UNARY_OP

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_multiply(JNIEnv *, jobject, jlong a, jlong b, jdouble scale) {
    return as_handle(cvk_mat_multiply(as_mat(a), as_mat(b), scale));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_scaleAdd(JNIEnv *, jobject, jlong mat, jdouble alpha, jdouble beta) {
    return as_handle(cvk_mat_scale_add(as_mat(mat), alpha, beta));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_convertTo(JNIEnv *, jobject, jlong mat,
                                    jint rtype, jdouble alpha, jdouble beta) {
    return as_handle(cvk_mat_convert_to(as_mat(mat), rtype, alpha, beta));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_inRange(JNIEnv *, jobject, jlong mat,
                                  jdouble l0, jdouble l1, jdouble l2, jdouble l3,
                                  jdouble u0, jdouble u1, jdouble u2, jdouble u3) {
    return as_handle(cvk_mat_in_range(as_mat(mat),
                                      as_scalar(nullptr, l0, l1, l2, l3),
                                      as_scalar(nullptr, u0, u1, u2, u3)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_flip(JNIEnv *, jobject, jlong mat, jint flip_code) {
    return as_handle(cvk_mat_flip(as_mat(mat), flip_code));
}

// ------------------------------------------------ reductions / stats

#define REDUCE_DOUBLE_ARRAY(java_name, c_expr)                                \
    JNIEXPORT jdoubleArray JNICALL                                            \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *env, jobject, jlong mat) {   \
        jdouble out[8] = {0, 0, 0, 0, 0, 0, 0, 0};                            \
        c_expr;                                                               \
        jdoubleArray result = env->NewDoubleArray(8);                         \
        if (result == nullptr) return nullptr;                                \
        env->SetDoubleArrayRegion(result, 0, 8, out);                         \
        return result;                                                        \
    }

REDUCE_DOUBLE_ARRAY(mean, { cvk_scalar_t s = cvk_mat_mean(as_mat(mat)); out[0] = s.v0; out[1] = s.v1; out[2] = s.v2; out[3] = s.v3; })
REDUCE_DOUBLE_ARRAY(sum, { cvk_scalar_t s = cvk_mat_sum(as_mat(mat)); out[0] = s.v0; out[1] = s.v1; out[2] = s.v2; out[3] = s.v3; })
REDUCE_DOUBLE_ARRAY(meanStdDev, cvk_mat_mean_stddev(as_mat(mat), out))
REDUCE_DOUBLE_ARRAY(minMaxLoc, cvk_mat_min_max_loc(as_mat(mat), out))

#undef REDUCE_DOUBLE_ARRAY

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_countNonZero(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_count_non_zero(as_mat(mat));
}

// ------------------------------------------------------------ imgproc

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_cvtColor(JNIEnv *, jobject, jlong mat, jint code) {
    return as_handle(cvk_cvt_color(as_mat(mat), code));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_resize(JNIEnv *, jobject, jlong mat,
                                 jint width, jint height, jint interpolation) {
    return as_handle(cvk_resize(as_mat(mat), width, height, interpolation));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_gaussianBlur(JNIEnv *, jobject, jlong mat,
                                       jint kw, jint kh, jdouble sx, jdouble sy) {
    return as_handle(cvk_gaussian_blur(as_mat(mat), kw, kh, sx, sy));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_medianBlur(JNIEnv *, jobject, jlong mat, jint kernel_size) {
    return as_handle(cvk_median_blur(as_mat(mat), kernel_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_threshold(JNIEnv *, jobject, jlong mat,
                                    jdouble thresh, jdouble max_val, jint type) {
    return as_handle(cvk_threshold(as_mat(mat), thresh, max_val, type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_adaptiveThreshold(JNIEnv *, jobject, jlong mat,
                                            jdouble max_value, jint method, jint type,
                                            jint block_size, jdouble c) {
    return as_handle(cvk_adaptive_threshold(as_mat(mat), max_value, method, type, block_size, c));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_canny(JNIEnv *, jobject, jlong mat,
                                jdouble t1, jdouble t2, jint aperture, jboolean l2) {
    return as_handle(cvk_canny(as_mat(mat), t1, t2, aperture, l2 != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_sobel(JNIEnv *, jobject, jlong mat,
                                jint dx, jint dy, jint kernel_size) {
    return as_handle(cvk_sobel(as_mat(mat), dx, dy, kernel_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_laplacian(JNIEnv *, jobject, jlong mat, jint kernel_size) {
    return as_handle(cvk_laplacian(as_mat(mat), kernel_size));
}

#define DRAW_OP(java_name, body)                                              \
    JNIEXPORT void JNICALL                                                    \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong mat,       \
        jint x1, jint y1, jint x2, jint y2, jdouble v0, jdouble v1,           \
        jdouble v2, jdouble v3, jint thickness) {                             \
        body;                                                                 \
    }

DRAW_OP(rectangle, cvk_rectangle(as_mat(mat), x1, y1, x2, y2,
                                 as_scalar(nullptr, v0, v1, v2, v3), thickness))
DRAW_OP(line, cvk_line(as_mat(mat), x1, y1, x2, y2,
                       as_scalar(nullptr, v0, v1, v2, v3), thickness))

#undef DRAW_OP

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_circle(JNIEnv *, jobject, jlong mat,
                                 jint cx, jint cy, jint radius,
                                 jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                 jint thickness) {
    cvk_circle(as_mat(mat), cx, cy, radius, as_scalar(nullptr, v0, v1, v2, v3), thickness);
}

// --------------------------------------------------------- imgcodecs

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_imread(JNIEnv *env, jobject, jstring path, jint flags) {
    if (path == nullptr) return 0;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return 0;
    cvk_mat_t *result = cvk_imread(utf, flags);
    env->ReleaseStringUTFChars(path, utf);
    return as_handle(result);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_imwrite(JNIEnv *env, jobject, jstring path, jlong mat) {
    if (path == nullptr) return JNI_FALSE;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return JNI_FALSE;
    const int ok = cvk_imwrite(utf, as_mat(mat));
    env->ReleaseStringUTFChars(path, utf);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_Jni_imencode(JNIEnv *env, jobject, jstring ext, jlong mat) {
    if (ext == nullptr) return nullptr;
    const char *utf = env->GetStringUTFChars(ext, nullptr);
    if (utf == nullptr) return nullptr;
    size_t length = 0;
    unsigned char *buffer = cvk_imencode(utf, as_mat(mat), &length);
    env->ReleaseStringUTFChars(ext, utf);
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

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_imdecode(JNIEnv *env, jobject, jbyteArray data, jint flags) {
    if (data == nullptr) return 0;
    const jsize length = env->GetArrayLength(data);
    if (length <= 0) return 0;
    jbyte *elements = env->GetByteArrayElements(data, nullptr);
    if (elements == nullptr) return 0;
    cvk_mat_t *result = cvk_imdecode(reinterpret_cast<const unsigned char *>(elements),
                                     static_cast<size_t>(length), flags);
    env->ReleaseByteArrayElements(data, elements, JNI_ABORT);
    return as_handle(result);
}

} /* extern "C" */

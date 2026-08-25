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
#include <vector>

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

static inline cvk_clahe_t *as_clahe(jlong handle) {
    return reinterpret_cast<cvk_clahe_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_clahe_handle(const cvk_clahe_t *clahe) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(clahe));
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

// ------------------------------------------------- shared output helpers

/** Wraps freshly allocated Mat handles into a jlongArray. */
static jlongArray as_handle_array(JNIEnv *env, cvk_mat_t **handles, jsize count) {
    std::vector<jlong> values(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = as_handle(handles[i]);
    }
    jlongArray result = env->NewLongArray(count);
    if (result == nullptr) {
        for (jsize i = 0; i < count; ++i) cvk_mat_release(handles[i]);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, count, values.data());
    return result;
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

// ------------------------------------------------------ mat members

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matReshape(JNIEnv *, jobject, jlong mat,
                                     jint channels, jint rows) {
    return as_handle(cvk_mat_reshape(as_mat(mat), channels, rows));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matRowRange(JNIEnv *, jobject, jlong mat,
                                      jint start, jint end) {
    return as_handle(cvk_mat_row_range(as_mat(mat), start, end));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matColRange(JNIEnv *, jobject, jlong mat,
                                      jint start, jint end) {
    return as_handle(cvk_mat_col_range(as_mat(mat), start, end));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matDiag(JNIEnv *, jobject, jlong mat, jint d) {
    return as_handle(cvk_mat_diag(as_mat(mat), d));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_matSetIdentity(JNIEnv *, jobject, jlong mat, jdouble scale) {
    cvk_mat_set_identity(as_mat(mat), scale);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_matDot(JNIEnv *, jobject, jlong a, jlong b) {
    return cvk_mat_dot(as_mat(a), as_mat(b));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matInv(JNIEnv *, jobject, jlong mat, jint method) {
    return as_handle(cvk_mat_inv(as_mat(mat), method));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_matDeterminant(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_determinant(as_mat(mat));
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_Jni_matTrace(JNIEnv *env, jobject, jlong mat) {
    const cvk_scalar_t s = cvk_mat_trace(as_mat(mat));
    const jdouble out[4] = {s.v0, s.v1, s.v2, s.v3};
    jdoubleArray result = env->NewDoubleArray(4);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 4, out);
    return result;
}

// ------------------------------------------------- core: array operations

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_splitChannels(JNIEnv *env, jobject, jlong src) {
    std::vector<cvk_mat_t *> handles(512, nullptr);
    const int count = cvk_split(as_mat(src), handles.data(),
                                static_cast<int>(handles.size()));
    if (count <= 0) return env->NewLongArray(0);
    return as_handle_array(env, handles.data(), static_cast<jsize>(count));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_mergeChannels(JNIEnv *env, jobject, jlongArray channels) {
    if (channels == nullptr) return 0;
    const jsize count = env->GetArrayLength(channels);
    if (count <= 0) return 0;
    jlong *elements = env->GetLongArrayElements(channels, nullptr);
    if (elements == nullptr) return 0;
    std::vector<const cvk_mat_t *> mats(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        mats[static_cast<size_t>(i)] = as_mat(elements[i]);
    }
    env->ReleaseLongArrayElements(channels, elements, JNI_ABORT);
    return as_handle(cvk_merge(mats.data(), count));
}

#define MAT2_OP(java_name, c_expr)                                            \
    JNIEXPORT jlong JNICALL                                                   \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong a,         \
                                          jlong b) {                          \
        return as_handle(c_expr);                                             \
    }

MAT2_OP(hconcat, cvk_hconcat(as_mat(a), as_mat(b)))
MAT2_OP(vconcat, cvk_vconcat(as_mat(a), as_mat(b)))
MAT2_OP(lut, cvk_lut(as_mat(a), as_mat(b)))

#undef MAT2_OP

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_compare(JNIEnv *, jobject, jlong a, jlong b, jint op) {
    return as_handle(cvk_compare(as_mat(a), as_mat(b), op));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_norm(JNIEnv *, jobject, jlong src, jint norm_type) {
    return cvk_norm(as_mat(src), norm_type);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_normDiff(JNIEnv *, jobject, jlong a, jlong b,
                                   jint norm_type) {
    return cvk_norm_diff(as_mat(a), as_mat(b), norm_type);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_normalize(JNIEnv *, jobject, jlong src, jdouble alpha,
                                    jdouble beta, jint norm_type, jint dtype) {
    return as_handle(cvk_normalize(as_mat(src), alpha, beta, norm_type, dtype));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_rotate(JNIEnv *, jobject, jlong src, jint code) {
    return as_handle(cvk_rotate(as_mat(src), code));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_copyMakeBorder(JNIEnv *, jobject, jlong src,
                                         jint top, jint bottom, jint left, jint right,
                                         jint border_type,
                                         jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    return as_handle(cvk_copy_make_border(as_mat(src), top, bottom, left, right,
                                          border_type,
                                          as_scalar(nullptr, v0, v1, v2, v3)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_addWeighted(JNIEnv *, jobject, jlong a, jdouble alpha,
                                      jlong b, jdouble beta, jdouble gamma) {
    return as_handle(cvk_add_weighted(as_mat(a), alpha, as_mat(b), beta, gamma));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_convertScaleAbs(JNIEnv *, jobject, jlong src,
                                          jdouble alpha, jdouble beta) {
    return as_handle(cvk_convert_scale_abs(as_mat(src), alpha, beta));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_solve(JNIEnv *, jobject, jlong a, jlong b, jint flags) {
    return as_handle(cvk_solve(as_mat(a), as_mat(b), flags));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_repeat(JNIEnv *, jobject, jlong src, jint nx, jint ny) {
    return as_handle(cvk_repeat(as_mat(src), nx, ny));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_transform(JNIEnv *, jobject, jlong src, jlong m) {
    return as_handle(cvk_transform(as_mat(src), as_mat(m)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_perspectiveTransform(JNIEnv *, jobject, jlong src, jlong m) {
    return as_handle(cvk_perspective_transform(as_mat(src), as_mat(m)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_pow(JNIEnv *, jobject, jlong src, jdouble power) {
    return as_handle(cvk_pow(as_mat(src), power));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_sqrt(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_sqrt(as_mat(src)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_exp(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_exp(as_mat(src)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_log(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_log(as_mat(src)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_magnitude(JNIEnv *, jobject, jlong x, jlong y) {
    return as_handle(cvk_magnitude(as_mat(x), as_mat(y)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_phase(JNIEnv *, jobject, jlong x, jlong y,
                                jboolean angle_in_degrees) {
    return as_handle(cvk_phase(as_mat(x), as_mat(y),
                               angle_in_degrees != JNI_FALSE));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_cartToPolar(JNIEnv *env, jobject, jlong x, jlong y,
                                      jboolean angle_in_degrees) {
    cvk_mat_t *magnitude = nullptr;
    cvk_mat_t *angle = nullptr;
    cvk_cart_to_polar(as_mat(x), as_mat(y), angle_in_degrees != JNI_FALSE,
                      &magnitude, &angle);
    cvk_mat_t *pair[2] = {magnitude, angle};
    return as_handle_array(env, pair, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_polarToCart(JNIEnv *env, jobject, jlong magnitude,
                                      jlong angle, jboolean angle_in_degrees) {
    cvk_mat_t *x = nullptr;
    cvk_mat_t *y = nullptr;
    cvk_polar_to_cart(as_mat(magnitude), as_mat(angle),
                      angle_in_degrees != JNI_FALSE, &x, &y);
    cvk_mat_t *pair[2] = {x, y};
    return as_handle_array(env, pair, 2);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_patchNaNs(JNIEnv *, jobject, jlong mat, jdouble value) {
    cvk_patch_nans(as_mat(mat), value);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_findNonZero(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_find_non_zero(as_mat(src)));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_hasNonZero(JNIEnv *, jobject, jlong src) {
    return cvk_has_non_zero(as_mat(src)) != 0 ? JNI_TRUE : JNI_FALSE;
}

#define FLAGS_OP(java_name, c_expr)                                           \
    JNIEXPORT jlong JNICALL                                                   \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong src,       \
                                          jint flags) {                       \
        return as_handle(c_expr);                                             \
    }

FLAGS_OP(sort, cvk_sort(as_mat(src), flags))
FLAGS_OP(sortIdx, cvk_sort_idx(as_mat(src), flags))
FLAGS_OP(dft, cvk_dft(as_mat(src), flags))
FLAGS_OP(idft, cvk_idft(as_mat(src), flags))
FLAGS_OP(dct, cvk_dct(as_mat(src), flags))
FLAGS_OP(idct, cvk_idct(as_mat(src), flags))

#undef FLAGS_OP

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_reduce(JNIEnv *, jobject, jlong src, jint dim,
                                 jint rtype, jint dtype) {
    return as_handle(cvk_reduce(as_mat(src), dim, rtype, dtype));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_reduceArgMax(JNIEnv *, jobject, jlong src, jint dim) {
    return as_handle(cvk_reduce_arg_max(as_mat(src), dim));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_reduceArgMin(JNIEnv *, jobject, jlong src, jint dim) {
    return as_handle(cvk_reduce_arg_min(as_mat(src), dim));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_extractChannel(JNIEnv *, jobject, jlong src, jint coi) {
    return as_handle(cvk_extract_channel(as_mat(src), coi));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_insertChannel(JNIEnv *, jobject, jlong src, jlong dst,
                                        jint coi) {
    cvk_insert_channel(as_mat(src), as_mat(dst), coi);
}

#define RAND_FILL(java_name, c_expr)                                          \
    JNIEXPORT void JNICALL                                                    \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong dst,       \
        jdouble a0, jdouble a1, jdouble a2, jdouble a3,                       \
        jdouble b0, jdouble b1, jdouble b2, jdouble b3) {                     \
        c_expr;                                                               \
    }

RAND_FILL(randu, cvk_randu(as_mat(dst), as_scalar(nullptr, a0, a1, a2, a3),
                           as_scalar(nullptr, b0, b1, b2, b3)))
RAND_FILL(randn, cvk_randn(as_mat(dst), as_scalar(nullptr, a0, a1, a2, a3),
                           as_scalar(nullptr, b0, b1, b2, b3)))

#undef RAND_FILL

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_psnr(JNIEnv *, jobject, jlong a, jlong b, jdouble r) {
    return cvk_psnr(as_mat(a), as_mat(b), r);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_getOptimalDftSize(JNIEnv *, jobject, jint rowsize) {
    return cvk_get_optimal_dft_size(rowsize);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_mulSpectrums(JNIEnv *, jobject, jlong a, jlong b,
                                       jint conj_flag, jboolean dft_rows) {
    return as_handle(cvk_mul_spectrums(as_mat(a), as_mat(b), conj_flag,
                                       dft_rows != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_divSpectrums(JNIEnv *, jobject, jlong a, jlong b,
                                       jint conj_flag) {
    return as_handle(cvk_div_spectrums(as_mat(a), as_mat(b), conj_flag));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_gemm(JNIEnv *, jobject, jlong a, jlong b, jdouble alpha,
                               jlong c, jdouble gamma) {
    return as_handle(cvk_gemm(as_mat(a), as_mat(b), alpha, as_mat(c), gamma));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_eigen(JNIEnv *env, jobject, jlong src) {
    cvk_mat_t *values = nullptr;
    cvk_mat_t *vectors = nullptr;
    cvk_eigen(as_mat(src), &values, &vectors);
    cvk_mat_t *pair[2] = {values, vectors};
    return as_handle_array(env, pair, 2);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_numThreads(JNIEnv *, jobject) {
    return cvk_num_threads();
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_setNumThreads(JNIEnv *, jobject, jint count) {
    cvk_set_num_threads(count);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_Jni_buildInformation(JNIEnv *env, jobject) {
    return env->NewStringUTF(cvk_build_information());
}

// ------------------------------------------------- imgproc: filters

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_blur(JNIEnv *, jobject, jlong src,
                               jint kernel_width, jint kernel_height) {
    return as_handle(cvk_blur(as_mat(src), kernel_width, kernel_height));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_boxFilter(JNIEnv *, jobject, jlong src, jint ddepth,
                                    jint kernel_width, jint kernel_height,
                                    jboolean normalize) {
    return as_handle(cvk_box_filter(as_mat(src), ddepth, kernel_width,
                                    kernel_height, normalize != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_sqrBoxFilter(JNIEnv *, jobject, jlong src, jint ddepth,
                                       jint kernel_width, jint kernel_height) {
    return as_handle(cvk_sqr_box_filter(as_mat(src), ddepth, kernel_width,
                                        kernel_height));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_bilateralFilter(JNIEnv *, jobject, jlong src, jint d,
                                          jdouble sigma_color, jdouble sigma_space) {
    return as_handle(cvk_bilateral_filter(as_mat(src), d, sigma_color, sigma_space));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_stackBlur(JNIEnv *, jobject, jlong src, jint kernel_size) {
    return as_handle(cvk_stack_blur(as_mat(src), kernel_size));
}

#define MORPH_OP(java_name, c_expr)                                           \
    JNIEXPORT jlong JNICALL                                                   \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong src,       \
        jlong kernel, jint iterations) {                                      \
        return as_handle(c_expr);                                             \
    }

MORPH_OP(erode, cvk_erode(as_mat(src), as_mat(kernel), iterations))
MORPH_OP(dilate, cvk_dilate(as_mat(src), as_mat(kernel), iterations))

#undef MORPH_OP

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_morphologyEx(JNIEnv *, jobject, jlong src, jint op,
                                       jlong kernel, jint iterations) {
    return as_handle(cvk_morphology_ex(as_mat(src), op, as_mat(kernel),
                                       iterations));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_getStructuringElement(JNIEnv *, jobject, jint shape,
                                                jint width, jint height) {
    return as_handle(cvk_get_structuring_element(shape, width, height));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_getGaussianKernel(JNIEnv *, jobject, jint ksize,
                                            jdouble sigma) {
    return as_handle(cvk_get_gaussian_kernel(ksize, sigma));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_filter2D(JNIEnv *, jobject, jlong src, jlong kernel,
                                   jint ddepth, jdouble delta) {
    return as_handle(cvk_filter_2d(as_mat(src), as_mat(kernel), ddepth, delta));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_pyrDown(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_pyr_down(as_mat(src)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_pyrUp(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_pyr_up(as_mat(src)));
}

// ------------------------------------------------- imgproc: geometry / warps

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_warpAffine(JNIEnv *, jobject, jlong src, jlong m,
                                     jint width, jint height, jint flags) {
    return as_handle(cvk_warp_affine(as_mat(src), as_mat(m), width, height, flags));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_warpPerspective(JNIEnv *, jobject, jlong src, jlong m,
                                          jint width, jint height, jint flags) {
    return as_handle(cvk_warp_perspective(as_mat(src), as_mat(m), width, height,
                                          flags));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_remap(JNIEnv *, jobject, jlong src, jlong map1,
                                jlong map2, jint interpolation) {
    return as_handle(cvk_remap(as_mat(src), as_mat(map1), as_mat(map2),
                               interpolation));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_warpPolar(JNIEnv *, jobject, jlong src, jint radius,
                                    jdouble center_x, jdouble center_y,
                                    jdouble max_radius, jint flags) {
    return as_handle(cvk_warp_polar(as_mat(src), radius, center_x, center_y,
                                    max_radius, flags));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_getAffineTransform(JNIEnv *, jobject,
                                             jdouble sx0, jdouble sy0, jdouble sx1,
                                             jdouble sy1, jdouble sx2, jdouble sy2,
                                             jdouble dx0, jdouble dy0, jdouble dx1,
                                             jdouble dy1, jdouble dx2, jdouble dy2) {
    return as_handle(cvk_get_affine_transform(sx0, sy0, sx1, sy1, sx2, sy2,
                                              dx0, dy0, dx1, dy1, dx2, dy2));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_invertAffineTransform(JNIEnv *, jobject, jlong m) {
    return as_handle(cvk_invert_affine_transform(as_mat(m)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_getPerspectiveTransform(JNIEnv *, jobject,
                                                  jdouble sx0, jdouble sy0,
                                                  jdouble sx1, jdouble sy1,
                                                  jdouble sx2, jdouble sy2,
                                                  jdouble sx3, jdouble sy3,
                                                  jdouble dx0, jdouble dy0,
                                                  jdouble dx1, jdouble dy1,
                                                  jdouble dx2, jdouble dy2,
                                                  jdouble dx3, jdouble dy3) {
    return as_handle(cvk_get_perspective_transform(sx0, sy0, sx1, sy1, sx2, sy2,
                                                   sx3, sy3, dx0, dy0, dx1, dy1,
                                                   dx2, dy2, dx3, dy3));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_getRotationMatrix2D(JNIEnv *, jobject, jdouble cx,
                                              jdouble cy, jdouble angle,
                                              jdouble scale) {
    return as_handle(cvk_get_rotation_matrix_2d(cx, cy, angle, scale));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_getRectSubPix(JNIEnv *, jobject, jlong src, jint width,
                                        jint height, jdouble cx, jdouble cy) {
    return as_handle(cvk_get_rect_sub_pix(as_mat(src), width, height, cx, cy));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_undistort(JNIEnv *, jobject, jlong src,
                                    jlong camera_matrix, jlong dist_coeffs) {
    return as_handle(cvk_undistort(as_mat(src), as_mat(camera_matrix),
                                   as_mat(dist_coeffs)));
}

// ------------------------------------------------- imgproc: color / histogram

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_demosaicing(JNIEnv *, jobject, jlong src, jint code) {
    return as_handle(cvk_demosaicing(as_mat(src), code));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_applyColormap(JNIEnv *, jobject, jlong src,
                                        jint colormap) {
    return as_handle(cvk_apply_colormap(as_mat(src), colormap));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_applyColormapUser(JNIEnv *, jobject, jlong src,
                                            jlong user_color) {
    return as_handle(cvk_apply_colormap_user(as_mat(src), as_mat(user_color)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_calcHist(JNIEnv *, jobject, jlong src, jint channel,
                                   jint hist_size, jfloat min_value,
                                   jfloat max_value) {
    return as_handle(cvk_calc_hist(as_mat(src), channel, hist_size, min_value,
                                   max_value));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_calcBackProject(JNIEnv *, jobject, jlong src,
                                          jint channel, jlong hist,
                                          jfloat min_value, jfloat max_value) {
    return as_handle(cvk_calc_back_project(as_mat(src), channel, as_mat(hist),
                                           min_value, max_value));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_compareHist(JNIEnv *, jobject, jlong h1, jlong h2,
                                      jint method) {
    return cvk_compare_hist(as_mat(h1), as_mat(h2), method);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_equalizeHist(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_equalize_hist(as_mat(src)));
}

// ------------------------------------ imgproc: segmentation / contours

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_floodFill(JNIEnv *, jobject, jlong image,
                                    jint seed_x, jint seed_y,
                                    jdouble n0, jdouble n1, jdouble n2, jdouble n3,
                                    jdouble lo0, jdouble lo1, jdouble lo2, jdouble lo3,
                                    jdouble up0, jdouble up1, jdouble up2, jdouble up3,
                                    jint flags) {
    return cvk_flood_fill(as_mat(image), seed_x, seed_y,
                          as_scalar(nullptr, n0, n1, n2, n3),
                          as_scalar(nullptr, lo0, lo1, lo2, lo3),
                          as_scalar(nullptr, up0, up1, up2, up3),
                          flags);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_watershed(JNIEnv *, jobject, jlong image, jlong markers) {
    cvk_watershed(as_mat(image), as_mat(markers));
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_Jni_findContours(JNIEnv *env, jobject, jlong src, jint mode,
                                       jint method) {
    size_t length = 0;
    unsigned char *buffer = cvk_find_contours(as_mat(src), mode, method, &length);
    return take_buffer(env, buffer, length);
}

/** Pins a contour wire-format jbyteArray for the duration of a cvk_ call. */
#define WITH_FLAT(flat_array, body)                                           \
    do {                                                                      \
        if ((flat_array) == nullptr) break;                                   \
        const jsize flat_len = env->GetArrayLength(flat_array);               \
        if (flat_len < 4) break;                                              \
        jbyte *flat_ptr = env->GetByteArrayElements(flat_array, nullptr);     \
        if (flat_ptr == nullptr) break;                                       \
        body;                                                                 \
        env->ReleaseByteArrayElements(flat_array, flat_ptr, JNI_ABORT);       \
    } while (0)

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_drawContours(JNIEnv *env, jobject, jlong image,
                                       jbyteArray flat, jint contour_index,
                                       jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                       jint thickness) {
    WITH_FLAT(flat, cvk_draw_contours(as_mat(image),
                                      reinterpret_cast<const unsigned char *>(flat_ptr),
                                      static_cast<size_t>(flat_len), contour_index,
                                      as_scalar(nullptr, v0, v1, v2, v3),
                                      thickness));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_contourAreaBytes(JNIEnv *env, jobject, jbyteArray flat) {
    double area = 0.0;
    WITH_FLAT(flat, area = cvk_contour_area(
                        reinterpret_cast<const unsigned char *>(flat_ptr),
                        static_cast<size_t>(flat_len)));
    return area;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_arcLengthBytes(JNIEnv *env, jobject, jbyteArray flat,
                                         jboolean closed) {
    double length = 0.0;
    WITH_FLAT(flat, length = cvk_arc_length(
                        reinterpret_cast<const unsigned char *>(flat_ptr),
                        static_cast<size_t>(flat_len), closed != JNI_FALSE));
    return length;
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_Jni_boundingRect(JNIEnv *env, jobject, jbyteArray flat) {
    int out[4] = {0, 0, 0, 0};
    WITH_FLAT(flat, cvk_bounding_rect(reinterpret_cast<const unsigned char *>(flat_ptr),
                                      static_cast<size_t>(flat_len), out));
    jintArray result = env->NewIntArray(4);
    if (result == nullptr) return nullptr;
    env->SetIntArrayRegion(result, 0, 4, out);
    return result;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_Jni_approxPolyDP(JNIEnv *env, jobject, jbyteArray flat,
                                       jdouble epsilon, jboolean closed) {
    if (flat == nullptr) return nullptr;
    const jsize flat_len = env->GetArrayLength(flat);
    if (flat_len < 4) return nullptr;
    jbyte *flat_ptr = env->GetByteArrayElements(flat, nullptr);
    if (flat_ptr == nullptr) return nullptr;
    size_t out_len = 0;
    unsigned char *buffer = cvk_approx_poly_dp(
        reinterpret_cast<const unsigned char *>(flat_ptr),
        static_cast<size_t>(flat_len), epsilon, closed != JNI_FALSE, &out_len);
    env->ReleaseByteArrayElements(flat, flat_ptr, JNI_ABORT);
    return take_buffer(env, buffer, out_len);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_Jni_minAreaRect(JNIEnv *env, jobject, jbyteArray flat) {
    double out[5] = {0, 0, 0, 0, 0};
    WITH_FLAT(flat, cvk_min_area_rect(reinterpret_cast<const unsigned char *>(flat_ptr),
                                      static_cast<size_t>(flat_len), out));
    jdoubleArray result = env->NewDoubleArray(5);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 5, out);
    return result;
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_Jni_minEnclosingCircle(JNIEnv *env, jobject, jbyteArray flat) {
    double out[3] = {0, 0, 0};
    WITH_FLAT(flat, cvk_min_enclosing_circle(
                        reinterpret_cast<const unsigned char *>(flat_ptr),
                        static_cast<size_t>(flat_len), out));
    jdoubleArray result = env->NewDoubleArray(3);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 3, out);
    return result;
}

#undef WITH_FLAT

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_Jni_moments(JNIEnv *env, jobject, jlong arr,
                                  jboolean binary_image) {
    double out[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    cvk_moments(as_mat(arr), binary_image != JNI_FALSE, out);
    jdoubleArray result = env->NewDoubleArray(10);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 10, out);
    return result;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_matchShapes(JNIEnv *, jobject, jlong a, jlong b,
                                      jint method) {
    return cvk_match_shapes(as_mat(a), as_mat(b), method);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_houghLines(JNIEnv *, jobject, jlong src, jdouble rho,
                                     jdouble theta, jint threshold, jdouble srn,
                                     jdouble stn) {
    return as_handle(cvk_hough_lines(as_mat(src), rho, theta, threshold, srn, stn));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_houghLinesP(JNIEnv *, jobject, jlong src, jdouble rho,
                                      jdouble theta, jint threshold,
                                      jdouble min_line_length,
                                      jdouble max_line_gap) {
    return as_handle(cvk_hough_lines_p(as_mat(src), rho, theta, threshold,
                                       min_line_length, max_line_gap));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_houghCircles(JNIEnv *, jobject, jlong src, jint method,
                                       jdouble dp, jdouble min_dist,
                                       jdouble param1, jdouble param2,
                                       jint min_radius, jint max_radius) {
    return as_handle(cvk_hough_circles(as_mat(src), method, dp, min_dist, param1,
                                       param2, min_radius, max_radius));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_cornerHarris(JNIEnv *, jobject, jlong src,
                                       jint block_size, jint ksize, jdouble k) {
    return as_handle(cvk_corner_harris(as_mat(src), block_size, ksize, k));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_cornerMinEigenVal(JNIEnv *, jobject, jlong src,
                                            jint block_size, jint ksize) {
    return as_handle(cvk_corner_min_eigen_val(as_mat(src), block_size, ksize));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_goodFeaturesToTrack(JNIEnv *, jobject, jlong src,
                                              jint max_corners, jdouble quality_level,
                                              jdouble min_distance, jint block_size,
                                              jboolean use_harris_detector,
                                              jdouble k) {
    return as_handle(cvk_good_features_to_track(as_mat(src), max_corners,
                                                quality_level, min_distance,
                                                block_size,
                                                use_harris_detector != JNI_FALSE,
                                                k));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matchTemplate(JNIEnv *, jobject, jlong image,
                                        jlong templ, jint method) {
    return as_handle(cvk_match_template(as_mat(image), as_mat(templ), method));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_distanceTransform(JNIEnv *, jobject, jlong src,
                                            jint distance_type, jint mask_size) {
    return as_handle(cvk_distance_transform(as_mat(src), distance_type, mask_size));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_integral(JNIEnv *, jobject, jlong src, jint sdepth) {
    return as_handle(cvk_integral(as_mat(src), sdepth));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_connectedComponents(JNIEnv *env, jobject, jlong src,
                                              jint connectivity, jint ltype) {
    cvk_mat_t *labels = nullptr;
    const int count = cvk_connected_components(as_mat(src), &labels, connectivity,
                                               ltype);
    const jlong out[2] = {static_cast<jlong>(count), as_handle(labels)};
    jlongArray result = env->NewLongArray(2);
    if (result == nullptr) {
        cvk_mat_release(labels);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 2, out);
    return result;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_connectedComponentsWithStats(JNIEnv *env, jobject,
                                                       jlong src, jint connectivity,
                                                       jint ltype) {
    cvk_mat_t *labels = nullptr;
    cvk_mat_t *stats = nullptr;
    cvk_mat_t *centroids = nullptr;
    const int count = cvk_connected_components_with_stats(as_mat(src), &labels,
                                                          &stats, &centroids,
                                                          connectivity, ltype);
    const jlong out[4] = {static_cast<jlong>(count), as_handle(labels),
                          as_handle(stats), as_handle(centroids)};
    jlongArray result = env->NewLongArray(4);
    if (result == nullptr) {
        cvk_mat_release(labels);
        cvk_mat_release(stats);
        cvk_mat_release(centroids);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 4, out);
    return result;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_pyrMeanShiftFiltering(JNIEnv *, jobject, jlong src,
                                                jdouble sp, jdouble sr,
                                                jint max_level) {
    return as_handle(cvk_pyr_mean_shift_filtering(as_mat(src), sp, sr, max_level));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_thresholdWithMask(JNIEnv *, jobject, jlong src,
                                            jlong mask, jlong dst, jdouble thresh,
                                            jdouble maxval, jint type) {
    return cvk_threshold_with_mask(as_mat(src), as_mat(mask), as_mat(dst),
                                   thresh, maxval, type);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_createHanningWindow(JNIEnv *, jobject, jint width,
                                              jint height, jint type) {
    return as_handle(cvk_create_hanning_window(width, height, type));
}

#define ACCUMULATE_OP(java_name, c_expr)                                      \
    JNIEXPORT void JNICALL                                                    \
    Java_cn_enaium_opencv_Jni_##java_name(JNIEnv *, jobject, jlong src,       \
                                          jlong dst) {                        \
        c_expr;                                                               \
    }

ACCUMULATE_OP(accumulate, cvk_accumulate(as_mat(src), as_mat(dst)))
ACCUMULATE_OP(accumulateSquare, cvk_accumulate_square(as_mat(src), as_mat(dst)))

#undef ACCUMULATE_OP

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_accumulateProduct(JNIEnv *, jobject, jlong a, jlong b,
                                            jlong dst) {
    cvk_accumulate_product(as_mat(a), as_mat(b), as_mat(dst));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_accumulateWeighted(JNIEnv *, jobject, jlong src,
                                             jlong dst, jdouble alpha) {
    cvk_accumulate_weighted(as_mat(src), as_mat(dst), alpha);
}

// ------------------------------------------------- imgproc: drawing

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_arrowedLine(JNIEnv *, jobject, jlong mat,
                                      jint x1, jint y1, jint x2, jint y2,
                                      jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                      jint thickness) {
    cvk_arrowed_line(as_mat(mat), x1, y1, x2, y2,
                     as_scalar(nullptr, v0, v1, v2, v3), thickness);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_drawMarker(JNIEnv *, jobject, jlong mat, jint x, jint y,
                                     jint marker_type, jint size,
                                     jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                     jint thickness) {
    cvk_draw_marker(as_mat(mat), x, y, marker_type, size,
                    as_scalar(nullptr, v0, v1, v2, v3), thickness);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_ellipse(JNIEnv *, jobject, jlong mat, jint cx, jint cy,
                                  jint axes_x, jint axes_y, jdouble angle,
                                  jdouble start_angle, jdouble end_angle,
                                  jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                  jint thickness) {
    cvk_ellipse(as_mat(mat), cx, cy, axes_x, axes_y, angle, start_angle, end_angle,
                as_scalar(nullptr, v0, v1, v2, v3), thickness);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_fillPoly(JNIEnv *env, jobject, jlong mat, jbyteArray flat,
                                   jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                   jint thickness) {
    if (flat == nullptr) return;
    const jsize flat_len = env->GetArrayLength(flat);
    if (flat_len < 4) return;
    jbyte *flat_ptr = env->GetByteArrayElements(flat, nullptr);
    if (flat_ptr == nullptr) return;
    cvk_fill_poly(as_mat(mat), reinterpret_cast<const unsigned char *>(flat_ptr),
                  static_cast<size_t>(flat_len),
                  as_scalar(nullptr, v0, v1, v2, v3), thickness);
    env->ReleaseByteArrayElements(flat, flat_ptr, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_polylines(JNIEnv *env, jobject, jlong mat, jbyteArray flat,
                                    jboolean closed, jdouble v0, jdouble v1,
                                    jdouble v2, jdouble v3, jint thickness) {
    if (flat == nullptr) return;
    const jsize flat_len = env->GetArrayLength(flat);
    if (flat_len < 4) return;
    jbyte *flat_ptr = env->GetByteArrayElements(flat, nullptr);
    if (flat_ptr == nullptr) return;
    cvk_polylines(as_mat(mat), reinterpret_cast<const unsigned char *>(flat_ptr),
                  static_cast<size_t>(flat_len), closed != JNI_FALSE,
                  as_scalar(nullptr, v0, v1, v2, v3), thickness);
    env->ReleaseByteArrayElements(flat, flat_ptr, JNI_ABORT);
}

// ------------------------------------------------- imgproc: CLAHE

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_claheCreate(JNIEnv *, jobject, jdouble clip_limit,
                                      jint tile_width, jint tile_height) {
    return as_clahe_handle(cvk_clahe_create(clip_limit, tile_width, tile_height));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_claheApply(JNIEnv *, jobject, jlong clahe, jlong src) {
    return as_handle(cvk_clahe_apply(as_clahe(clahe), as_mat(src)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_claheSetClipLimit(JNIEnv *, jobject, jlong clahe,
                                            jdouble clip_limit) {
    cvk_clahe_set_clip_limit(as_clahe(clahe), clip_limit);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_claheRelease(JNIEnv *, jobject, jlong clahe) {
    cvk_clahe_release(as_clahe(clahe));
}

// ------------------------------------------------- imgcodecs additions

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_imcount(JNIEnv *env, jobject, jstring path) {
    if (path == nullptr) return 0;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return 0;
    const int count = cvk_imcount(utf);
    env->ReleaseStringUTFChars(path, utf);
    return count;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_haveImageReader(JNIEnv *env, jobject, jstring ext) {
    if (ext == nullptr) return JNI_FALSE;
    const char *utf = env->GetStringUTFChars(ext, nullptr);
    if (utf == nullptr) return JNI_FALSE;
    const int has = cvk_have_image_reader(utf);
    env->ReleaseStringUTFChars(ext, utf);
    return has != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_haveImageWriter(JNIEnv *env, jobject, jstring ext) {
    if (ext == nullptr) return JNI_FALSE;
    const char *utf = env->GetStringUTFChars(ext, nullptr);
    if (utf == nullptr) return JNI_FALSE;
    const int has = cvk_have_image_writer(utf);
    env->ReleaseStringUTFChars(ext, utf);
    return has != 0 ? JNI_TRUE : JNI_FALSE;
}

static size_t copy_int_params(JNIEnv *env, jintArray params, std::vector<int> &out) {
    if (params == nullptr) return 0;
    const jsize length = env->GetArrayLength(params);
    if (length <= 0) return 0;
    out.resize(static_cast<size_t>(length));
    env->GetIntArrayRegion(params, 0, length, out.data());
    return out.size();
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_imwriteParams(JNIEnv *env, jobject, jstring path,
                                        jlong mat, jintArray params) {
    if (path == nullptr) return JNI_FALSE;
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return JNI_FALSE;
    std::vector<int> values;
    const size_t count = copy_int_params(env, params, values);
    const int ok = cvk_imwrite_params(utf, as_mat(mat),
                                      count > 0 ? values.data() : nullptr, count);
    env->ReleaseStringUTFChars(path, utf);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_Jni_imencodeParams(JNIEnv *env, jobject, jstring ext,
                                         jlong mat, jintArray params) {
    if (ext == nullptr) return nullptr;
    const char *utf = env->GetStringUTFChars(ext, nullptr);
    if (utf == nullptr) return nullptr;
    std::vector<int> values;
    const size_t count = copy_int_params(env, params, values);
    size_t length = 0;
    unsigned char *buffer = cvk_imencode_params(utf, as_mat(mat),
                                                count > 0 ? values.data() : nullptr,
                                                count, &length);
    env->ReleaseStringUTFChars(ext, utf);
    return take_buffer(env, buffer, length);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_setRngSeed(JNIEnv *, jobject, jlong seed) {
    cvk_set_rng_seed(static_cast<unsigned long long>(seed));
}

// --------------------------------------- org.opencv.core.Mat parity

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_Jni_matDump(JNIEnv *env, jobject, jlong mat) {
    const char *text = cvk_mat_dump(as_mat(mat));
    return text != nullptr ? env->NewStringUTF(text) : nullptr;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_matIsContinuous(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_is_continuous(as_mat(mat)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_Jni_matIsSubmatrix(JNIEnv *, jobject, jlong mat) {
    return cvk_mat_is_submatrix(as_mat(mat)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matAdjustROI(JNIEnv *, jobject, jlong mat,
                                       jint dtop, jint dbottom,
                                       jint dleft, jint dright) {
    return as_handle(cvk_mat_adjust_roi(as_mat(mat), dtop, dbottom, dleft, dright));
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_Jni_matLocateROI(JNIEnv *env, jobject, jlong mat) {
    int out[4] = {0, 0, 0, 0};
    cvk_mat_locate_roi(as_mat(mat), out);
    jintArray result = env->NewIntArray(4);
    if (result == nullptr) return nullptr;
    env->SetIntArrayRegion(result, 0, 4, out);
    return result;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_matCross(JNIEnv *, jobject, jlong a, jlong b) {
    return as_handle(cvk_mat_cross(as_mat(a), as_mat(b)));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_matPutValues(JNIEnv *env, jobject, jlong mat,
                                       jint row, jint col, jdoubleArray values) {
    if (values == nullptr) return 0;
    const jsize count = env->GetArrayLength(values);
    if (count <= 0) return 0;
    jdouble *elements = env->GetDoubleArrayElements(values, nullptr);
    if (elements == nullptr) return 0;
    const size_t written = cvk_mat_put_values(as_mat(mat), row, col, elements,
                                              static_cast<size_t>(count));
    env->ReleaseDoubleArrayElements(values, elements, JNI_ABORT);
    return static_cast<jint>(written);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_Jni_matGetValues(JNIEnv *env, jobject, jlong mat,
                                       jint row, jint col, jdoubleArray values) {
    if (values == nullptr) return 0;
    const jsize count = env->GetArrayLength(values);
    if (count <= 0) return 0;
    std::vector<jdouble> buffer(static_cast<size_t>(count), 0.0);
    const size_t read = cvk_mat_get_values(as_mat(mat), row, col, buffer.data(),
                                           static_cast<size_t>(count));
    if (read > 0) {
        env->SetDoubleArrayRegion(values, 0, static_cast<jsize>(read), buffer.data());
    }
    return static_cast<jint>(read);
}

// ------------------------------------------------- core: clustering / decomposition

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_kmeans(JNIEnv *env, jobject, jlong data, jint k,
                                 jint crit_type, jint crit_max_count,
                                 jdouble crit_epsilon, jint attempts, jint flags,
                                 jlong labels, jdoubleArray compactness_out) {
    cvk_mat_t *centers = nullptr;
    double compactness = 0.0;
    cvk_kmeans(as_mat(data), k, as_mat(labels),
               crit_type, crit_max_count, crit_epsilon, attempts, flags,
               &centers, &compactness);
    if (compactness_out != nullptr && env->GetArrayLength(compactness_out) > 0) {
        const jdouble value = compactness;
        env->SetDoubleArrayRegion(compactness_out, 0, 1, &value);
    }
    return as_handle(centers);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_svdDecomp(JNIEnv *env, jobject, jlong src, jint flags) {
    cvk_mat_t *w = nullptr;
    cvk_mat_t *u = nullptr;
    cvk_mat_t *vt = nullptr;
    cvk_svd_decomp(as_mat(src), &w, &u, &vt, flags);
    cvk_mat_t *triple[3] = {w, u, vt};
    return as_handle_array(env, triple, 3);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_svdBackSubst(JNIEnv *, jobject, jlong w, jlong u,
                                       jlong vt, jlong b) {
    cvk_mat_t *dst = nullptr;
    cvk_svd_backsubst(as_mat(w), as_mat(u), as_mat(vt), as_mat(b), &dst);
    return as_handle(dst);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_pcaCompute(JNIEnv *env, jobject, jlong data,
                                     jint max_components) {
    cvk_mat_t *mean = nullptr;
    cvk_mat_t *vectors = nullptr;
    cvk_pca_compute(as_mat(data), &mean, &vectors, max_components);
    cvk_mat_t *pair[2] = {mean, vectors};
    return as_handle_array(env, pair, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_Jni_pcaComputeVariance(JNIEnv *env, jobject, jlong data,
                                             jdouble retained_variance) {
    cvk_mat_t *mean = nullptr;
    cvk_mat_t *vectors = nullptr;
    cvk_pca_compute_variance(as_mat(data), &mean, &vectors, retained_variance);
    cvk_mat_t *pair[2] = {mean, vectors};
    return as_handle_array(env, pair, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_pcaProject(JNIEnv *, jobject, jlong data, jlong mean,
                                     jlong vectors) {
    cvk_mat_t *result = nullptr;
    cvk_pca_project(as_mat(data), as_mat(mean), as_mat(vectors), &result);
    return as_handle(result);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_Jni_pcaBackProject(JNIEnv *, jobject, jlong data, jlong mean,
                                         jlong vectors) {
    cvk_mat_t *result = nullptr;
    cvk_pca_backproject(as_mat(data), as_mat(mean), as_mat(vectors), &result);
    return as_handle(result);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_mahalanobis(JNIEnv *, jobject, jlong v1, jlong v2,
                                      jlong icovar) {
    return cvk_mahalanobis(as_mat(v1), as_mat(v2), as_mat(icovar));
}

// ------------------------------------------------- imgproc: features / segmentation

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_Jni_cornerSubPixBytes(JNIEnv *env, jobject, jlong image,
                                            jbyteArray flat,
                                            jint win_w, jint win_h,
                                            jint zero_w, jint zero_h,
                                            jint crit_type, jint crit_max_count,
                                            jdouble crit_epsilon) {
    if (flat == nullptr) return nullptr;
    const jsize flat_len = env->GetArrayLength(flat);
    if (flat_len < 4) return nullptr;
    jbyte *flat_ptr = env->GetByteArrayElements(flat, nullptr);
    if (flat_ptr == nullptr) return nullptr;
    size_t out_len = 0;
    unsigned char *buffer = cvk_corner_sub_pix(
        as_mat(image), reinterpret_cast<const unsigned char *>(flat_ptr),
        static_cast<size_t>(flat_len), win_w, win_h, zero_w, zero_h,
        crit_type, crit_max_count, crit_epsilon, &out_len);
    env->ReleaseByteArrayElements(flat, flat_ptr, JNI_ABORT);
    return take_buffer(env, buffer, out_len);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_Jni_emd(JNIEnv *, jobject, jlong sig1, jlong sig2,
                              jint dist_type) {
    return static_cast<jdouble>(cvk_emd(as_mat(sig1), as_mat(sig2), dist_type));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_Jni_grabCut(JNIEnv *, jobject, jlong img, jlong mask,
                                  jint rx, jint ry, jint rw, jint rh,
                                  jlong bgd_model, jlong fgd_model,
                                  jint iters, jint mode) {
    cvk_grab_cut(as_mat(img), as_mat(mask), rx, ry, rw, rh,
                 as_mat(bgd_model), as_mat(fgd_model), iters, mode);
}
} /* extern "C" */

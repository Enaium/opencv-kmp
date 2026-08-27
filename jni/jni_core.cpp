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
 * JNI bridge for the org.opencv.core.Core statics (core slice): thin
 * Java_cn_enaium_opencv_JniCore_* wrappers around the cvk_ C ABI in
 * native/shim_core.cpp. Handles arrive as jlong pointers; scalars expand to
 * primitives. No exceptions may cross the JNI boundary — the shim is
 * noexcept and returns NULL/0/false on failure.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_core.h"

#include <cstdint>
#include <cstring>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_rng_t *as_rng(jlong handle) {
    return reinterpret_cast<cvk_rng_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_rng_handle(const cvk_rng_t *rng) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(rng));
}

extern "C" {

// ---------------------------------------------------------------- errors

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniCore_lastError(JNIEnv *env, jobject) {
    const char *message = cvk_core_last_error();
    return message != nullptr ? env->NewStringUTF(message) : nullptr;
}

// ----------------------------------------------------------- scalar math

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniCore_cubeRoot(JNIEnv *, jobject, jfloat val) {
    return cvk_cube_root(val);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniCore_fastAtan2(JNIEnv *, jobject, jfloat y, jfloat x) {
    return cvk_fast_atan2(y, x);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_borderInterpolate(JNIEnv *, jobject, jint p,
                                                jint len, jint border_type) {
    return cvk_border_interpolate(p, len, border_type);
}

// ------------------------------------------------------------------- RNG

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_rngFromGlobal(JNIEnv *, jobject) {
    return as_rng_handle(cvk_rng_from_global());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_rngCreate(JNIEnv *, jobject, jlong seed) {
    return as_rng_handle(cvk_rng_create(static_cast<unsigned long long>(seed)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_rngNext(JNIEnv *, jobject, jlong rng) {
    return static_cast<jlong>(cvk_rng_next(as_rng(rng)));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_rngUniformInt(JNIEnv *, jobject, jlong rng,
                                            jint a, jint b) {
    return cvk_rng_uniform_int(as_rng(rng), a, b);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniCore_rngUniformFloat(JNIEnv *, jobject, jlong rng,
                                              jfloat a, jfloat b) {
    return cvk_rng_uniform_float(as_rng(rng), a, b);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCore_rngUniformDouble(JNIEnv *, jobject, jlong rng,
                                               jdouble a, jdouble b) {
    return cvk_rng_uniform_double(as_rng(rng), a, b);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCore_rngGaussian(JNIEnv *, jobject, jlong rng,
                                          jdouble sigma) {
    return cvk_rng_gaussian(as_rng(rng), sigma);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_rngRelease(JNIEnv *, jobject, jlong rng) {
    cvk_rng_release(as_rng(rng));
}

// --------------------------------------------------------- array operations

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_mixChannels(JNIEnv *env, jobject,
                                          jlongArray srcs, jlongArray dsts,
                                          jintArray from_to) {
    jsize nsrcs = env->GetArrayLength(srcs);
    jsize ndsts = env->GetArrayLength(dsts);
    jsize npairs = env->GetArrayLength(from_to);
    std::vector<jlong> src_buf(static_cast<size_t>(nsrcs));
    std::vector<jlong> dst_buf(static_cast<size_t>(ndsts));
    std::vector<jint> pair_buf(static_cast<size_t>(npairs));
    env->GetLongArrayRegion(srcs, 0, nsrcs, src_buf.data());
    env->GetLongArrayRegion(dsts, 0, ndsts, dst_buf.data());
    env->GetIntArrayRegion(from_to, 0, npairs, pair_buf.data());
    std::vector<cvk_mat_t *> src_ptrs(static_cast<size_t>(nsrcs));
    std::vector<cvk_mat_t *> dst_ptrs(static_cast<size_t>(ndsts));
    for (jsize i = 0; i < nsrcs; ++i) src_ptrs[static_cast<size_t>(i)] = as_mat(src_buf[static_cast<size_t>(i)]);
    for (jsize i = 0; i < ndsts; ++i) dst_ptrs[static_cast<size_t>(i)] = as_mat(dst_buf[static_cast<size_t>(i)]);
    cvk_mix_channels(src_ptrs.data(), nsrcs, dst_ptrs.data(), ndsts,
                     reinterpret_cast<const int *>(pair_buf.data()),
                     static_cast<size_t>(npairs));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniCore_batchDistance(JNIEnv *, jobject,
                                            jlong src1, jlong src2,
                                            jlong dist, jint dtype,
                                            jlong nidx, jint norm_type,
                                            jint k, jlong mask, jint update,
                                            jboolean crosscheck) {
    return cvk_batch_distance(as_mat(src1), as_mat(src2), as_mat(dist), dtype,
                              as_mat(nidx), norm_type, k,
                              mask != 0 ? as_mat(mask) : nullptr, update,
                              crosscheck != JNI_FALSE) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_calcCovarMatrix(JNIEnv *, jobject,
                                              jlong samples, jlong covar,
                                              jlong mean, jint flags,
                                              jint ctype) {
    cvk_calc_covar_matrix(as_mat(samples), as_mat(covar), as_mat(mean), flags,
                          ctype);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_completeSymm(JNIEnv *, jobject, jlong m,
                                           jboolean lower_to_upper) {
    cvk_complete_symm(as_mat(m), lower_to_upper != JNI_FALSE);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_solveCubic(JNIEnv *env, jobject, jlong coeffs,
                                         jlongArray rootsOut) {
    cvk_mat_t *roots = nullptr;
    jint count = cvk_solve_cubic(as_mat(coeffs), &roots);
    jlong handle = as_handle(roots);
    if (rootsOut != nullptr) {
        env->SetLongArrayRegion(rootsOut, 0, 1, &handle);
    }
    return count;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCore_solvePoly(JNIEnv *env, jobject, jlong coeffs,
                                        jlongArray rootsOut, jint max_iters) {
    cvk_mat_t *roots = nullptr;
    jdouble eps = cvk_solve_poly(as_mat(coeffs), &roots, max_iters);
    jlong handle = as_handle(roots);
    if (rootsOut != nullptr) {
        env->SetLongArrayRegion(rootsOut, 0, 1, &handle);
    }
    return eps;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_mulTransposed(JNIEnv *, jobject, jlong src,
                                            jboolean a_ta, jlong delta,
                                            jdouble scale, jint dtype) {
    return as_handle(cvk_mul_transposed(as_mat(src), a_ta != JNI_FALSE,
                                        delta != 0 ? as_mat(delta) : nullptr,
                                        scale, dtype));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_flipND(JNIEnv *, jobject, jlong src, jint axis) {
    return as_handle(cvk_flip_nd(as_mat(src), axis));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_broadcast(JNIEnv *, jobject, jlong src,
                                        jlong shape) {
    return as_handle(cvk_broadcast(as_mat(src), as_mat(shape)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_transposeND(JNIEnv *env, jobject, jlong src,
                                          jintArray order) {
    jsize len = env->GetArrayLength(order);
    std::vector<jint> buf(static_cast<size_t>(len));
    env->GetIntArrayRegion(order, 0, len, buf.data());
    return as_handle(cvk_transpose_nd(as_mat(src),
                                      reinterpret_cast<const int *>(buf.data()),
                                      static_cast<size_t>(len)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_copyTo(JNIEnv *, jobject, jlong src, jlong mask) {
    return as_handle(cvk_copy_to(as_mat(src),
                                 mask != 0 ? as_mat(mask) : nullptr));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_scaleAdd(JNIEnv *, jobject, jlong a,
                                       jdouble alpha, jlong b) {
    return as_handle(cvk_scale_add(as_mat(a), alpha, as_mat(b)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_gemmFlags(JNIEnv *, jobject, jlong a, jlong b,
                                        jdouble alpha, jlong c, jdouble gamma,
                                        jint flags) {
    return as_handle(cvk_gemm_flags(as_mat(a), as_mat(b), alpha,
                                    c != 0 ? as_mat(c) : nullptr, gamma,
                                    flags));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniCore_eigenNonSymmetric(JNIEnv *env, jobject,
                                                jlong src) {
    cvk_mat_t *eigenvalues = nullptr;
    cvk_mat_t *eigenvectors = nullptr;
    cvk_eigen_non_symmetric(as_mat(src), &eigenvalues, &eigenvectors);
    jlongArray out = env->NewLongArray(2);
    if (out == nullptr) return nullptr;
    jlong handles[2] = {as_handle(eigenvalues), as_handle(eigenvectors)};
    env->SetLongArrayRegion(out, 0, 2, handles);
    return out;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_finiteMask(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_finite_mask(as_mat(src)));
}

// --------------------------------------------------------- masked statistics

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniCore_minMaxLocMasked(JNIEnv *env, jobject,
                                              jlong src, jlong mask) {
    double out[6];
    cvk_mat_min_max_loc_masked(as_mat(src), as_mat(mask), out);
    jdoubleArray result = env->NewDoubleArray(6);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 6, out);
    return result;
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniCore_meanMasked(JNIEnv *env, jobject, jlong src,
                                         jlong mask) {
    cvk_scalar_t s = cvk_mat_mean_masked(as_mat(src), as_mat(mask));
    jdoubleArray result = env->NewDoubleArray(4);
    if (result == nullptr) return nullptr;
    jdouble vals[4] = {s.v0, s.v1, s.v2, s.v3};
    env->SetDoubleArrayRegion(result, 0, 4, vals);
    return result;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCore_normMasked(JNIEnv *, jobject, jlong src,
                                         jint norm_type, jlong mask) {
    return cvk_norm_masked(as_mat(src), norm_type, as_mat(mask));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCore_normDiffMasked(JNIEnv *, jobject, jlong a,
                                             jlong b, jint norm_type,
                                             jlong mask) {
    return cvk_norm_diff_masked(as_mat(a), as_mat(b), norm_type, as_mat(mask));
}

// ------------------------------------------------- range check / shuffle

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniCore_checkRange(JNIEnv *, jobject, jlong a,
                                         jboolean quiet, jdouble min_val,
                                         jdouble max_val) {
    return cvk_check_range(as_mat(a), quiet != JNI_FALSE, min_val, max_val) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_randShuffle(JNIEnv *, jobject, jlong dst,
                                          jdouble iter_factor) {
    cvk_rand_shuffle(as_mat(dst), iter_factor);
}

// ------------------------------------------------- environment / runtime info

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_getTickCount(JNIEnv *, jobject) {
    return static_cast<jlong>(cvk_get_tick_count());
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCore_getTickFrequency(JNIEnv *, jobject) {
    return cvk_get_tick_frequency();
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_getNumberOfCPUs(JNIEnv *, jobject) {
    return cvk_get_number_of_cpus();
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniCore_checkHardwareSupport(JNIEnv *, jobject,
                                                   jint feature) {
    return cvk_check_hardware_support(feature) != 0;
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniCore_getHardwareFeatureName(JNIEnv *env, jobject,
                                                     jint feature) {
    const char *name = cvk_get_hardware_feature_name(feature);
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniCore_getVersionString(JNIEnv *env, jobject) {
    const char *version = cvk_get_version_string();
    return version != nullptr ? env->NewStringUTF(version) : nullptr;
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_getVersionMajor(JNIEnv *, jobject) {
    return cvk_get_version_major();
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_getVersionMinor(JNIEnv *, jobject) {
    return cvk_get_version_minor();
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_getVersionRevision(JNIEnv *, jobject) {
    return cvk_get_version_revision();
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCore_getCPUTickCount(JNIEnv *, jobject) {
    return static_cast<jlong>(cvk_get_cpu_tick_count());
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_getThreadNum(JNIEnv *, jobject) {
    return cvk_get_thread_num();
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniCore_getDefaultAlgorithmHint(JNIEnv *, jobject) {
    return cvk_get_default_algorithm_hint();
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniCore_useOptimized(JNIEnv *, jobject) {
    return cvk_use_optimized() != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_setUseOptimized(JNIEnv *, jobject,
                                              jboolean onoff) {
    cvk_set_use_optimized(onoff != JNI_FALSE);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniCore_getCPUFeaturesLine(JNIEnv *env, jobject) {
    const char *line = cvk_get_cpu_features_line();
    return line != nullptr ? env->NewStringUTF(line) : nullptr;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniCore_useIPP(JNIEnv *, jobject) {
    return cvk_use_ipp() != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_setUseIPP(JNIEnv *, jobject, jboolean flag) {
    cvk_set_use_ipp(flag != JNI_FALSE);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniCore_getIppVersion(JNIEnv *env, jobject) {
    const char *version = cvk_get_ipp_version();
    return version != nullptr ? env->NewStringUTF(version) : nullptr;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniCore_useIPPNotExact(JNIEnv *, jobject) {
    return cvk_use_ipp_not_exact() != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_setUseIPPNotExact(JNIEnv *, jobject,
                                                jboolean flag) {
    cvk_set_use_ipp_not_exact(flag != JNI_FALSE);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniCore_findFile(JNIEnv *env, jobject,
                                       jstring relative_path,
                                       jboolean required, jboolean silent_mode) {
    const char *path = env->GetStringUTFChars(relative_path, nullptr);
    if (path == nullptr) return nullptr;
    const char *found = cvk_find_file(path, required != JNI_FALSE,
                                      silent_mode != JNI_FALSE);
    env->ReleaseStringUTFChars(relative_path, path);
    return found != nullptr ? env->NewStringUTF(found) : nullptr;
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniCore_findFileOrKeep(JNIEnv *env, jobject,
                                             jstring relative_path,
                                             jboolean silent_mode) {
    const char *path = env->GetStringUTFChars(relative_path, nullptr);
    if (path == nullptr) return nullptr;
    const char *found = cvk_find_file_or_keep(path, silent_mode != JNI_FALSE);
    env->ReleaseStringUTFChars(relative_path, path);
    return found != nullptr ? env->NewStringUTF(found) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_addSamplesDataSearchPath(JNIEnv *env, jobject,
                                                       jstring path) {
    const char *p = env->GetStringUTFChars(path, nullptr);
    if (p == nullptr) return;
    cvk_add_samples_data_search_path(p);
    env->ReleaseStringUTFChars(path, p);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCore_addSamplesDataSearchSubDirectory(JNIEnv *env,
                                                               jobject,
                                                               jstring subdir) {
    const char *s = env->GetStringUTFChars(subdir, nullptr);
    if (s == nullptr) return;
    cvk_add_samples_data_search_sub_directory(s);
    env->ReleaseStringUTFChars(subdir, s);
}

} /* extern "C" */

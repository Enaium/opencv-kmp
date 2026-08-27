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
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package cn.enaium.opencv

import kotlin.concurrent.Volatile
import cvk.cvk_mat_t
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.interpretCPointer
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.rawValue
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.addressOf

import cvk.cvk_add_samples_data_search_path
import cvk.cvk_add_samples_data_search_sub_directory
import cvk.cvk_batch_distance
import cvk.cvk_border_interpolate
import cvk.cvk_broadcast
import cvk.cvk_calc_covar_matrix
import cvk.cvk_check_hardware_support
import cvk.cvk_check_range
import cvk.cvk_complete_symm
import cvk.cvk_copy_to
import cvk.cvk_core_last_error
import cvk.cvk_cube_root
import cvk.cvk_eigen_non_symmetric
import cvk.cvk_fast_atan2
import cvk.cvk_find_file
import cvk.cvk_find_file_or_keep
import cvk.cvk_finite_mask
import cvk.cvk_flip_nd
import cvk.cvk_gemm_flags
import cvk.cvk_get_cpu_features_line
import cvk.cvk_get_cpu_tick_count
import cvk.cvk_get_default_algorithm_hint
import cvk.cvk_get_hardware_feature_name
import cvk.cvk_get_ipp_version
import cvk.cvk_get_number_of_cpus
import cvk.cvk_get_thread_num
import cvk.cvk_get_tick_count
import cvk.cvk_get_tick_frequency
import cvk.cvk_get_version_major
import cvk.cvk_get_version_minor
import cvk.cvk_get_version_revision
import cvk.cvk_get_version_string
import cvk.cvk_hconcat
import cvk.cvk_last_error
import cvk.cvk_mat_mean
import cvk.cvk_mat_mean_masked
import cvk.cvk_mat_min_max_loc
import cvk.cvk_mat_min_max_loc_masked
import cvk.cvk_mix_channels
import cvk.cvk_mul_transposed
import cvk.cvk_norm
import cvk.cvk_norm_diff
import cvk.cvk_norm_diff_masked
import cvk.cvk_norm_masked
import cvk.cvk_rand_shuffle
import cvk.cvk_rng_from_global
import cvk.cvk_rng_gaussian
import cvk.cvk_rng_next
import cvk.cvk_rng_release
import cvk.cvk_rng_t
import cvk.cvk_rng_uniform_double
import cvk.cvk_rng_uniform_float
import cvk.cvk_rng_uniform_int
import cvk.cvk_scale_add
import cvk.cvk_set_use_ipp
import cvk.cvk_set_use_ipp_not_exact
import cvk.cvk_set_use_optimized
import cvk.cvk_solve_cubic
import cvk.cvk_solve_poly
import cvk.cvk_transpose_nd
import cvk.cvk_use_ipp
import cvk.cvk_use_ipp_not_exact
import cvk.cvk_use_optimized
import cvk.cvk_vconcat

import kotlinx.cinterop.ptr
import kotlinx.cinterop.toKString
import kotlinx.cinterop.useContents
import kotlinx.cinterop.value
import platform.posix.size_t

private fun lastNativeError(): String? =
    cvk_core_last_error()?.toKString() ?: cvk_last_error()?.toKString()

// =========================================================================
// scalar math
// =========================================================================

actual fun cubeRoot(value: Float): Float = cvk_cube_root(value)

actual fun fastAtan2(y: Float, x: Float): Float = cvk_fast_atan2(y, x)

actual fun borderInterpolate(p: Int, len: Int, borderType: Int): Int =
    cvk_border_interpolate(p, len, borderType)

// =========================================================================
// RNG
// =========================================================================

internal class NativeRng(@Volatile private var raw: CPointer<cvk_rng_t>?) : RNG {

    private fun check(): CPointer<cvk_rng_t> =
        raw ?: throw IllegalStateException("RNG is closed")

    override fun next(): UInt = cvk_rng_next(check())

    override fun uniform(a: Int, b: Int): Int = cvk_rng_uniform_int(check(), a, b)

    override fun uniform(a: Float, b: Float): Float = cvk_rng_uniform_float(check(), a, b)

    override fun uniform(a: Double, b: Double): Double = cvk_rng_uniform_double(check(), a, b)

    override fun gaussian(sigma: Double): Double = cvk_rng_gaussian(check(), sigma)

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_rng_release(handle)
    }
}

actual fun theRNG(): RNG =
    NativeRng(cvk_rng_from_global() ?: throw OpenCVException("theRNG", lastNativeError()))

// =========================================================================
// array operations
// =========================================================================

internal actual fun hconcatPair(a: Mat, b: Mat): Mat =
    nativeMat(cvk_hconcat(a.nativeHandle(), b.nativeHandle()), "hconcat")

internal actual fun vconcatPair(a: Mat, b: Mat): Mat =
    nativeMat(cvk_vconcat(a.nativeHandle(), b.nativeHandle()), "vconcat")

actual fun mixChannels(src: List<Mat>, dst: List<Mat>, fromTo: IntArray) {
    require(fromTo.size % 2 == 0) { "mixChannels fromTo must hold [srcIdx, dstIdx] pairs" }
    memScoped {
        val srcHandles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
        val dstHandles = allocArray<CPointerVar<cvk_mat_t>>(dst.size)
        src.forEachIndexed { i, m -> srcHandles[i] = m.nativeHandle() }
        dst.forEachIndexed { i, m -> dstHandles[i] = m.nativeHandle() }
        fromTo.usePinned { pinned ->
            cvk_mix_channels(
                srcHandles, src.size, dstHandles, dst.size,
                pinned.addressOf(0), fromTo.size.convert<size_t>(),
            )
        }
    }
}

actual fun batchDistance(
    src1: Mat,
    src2: Mat,
    dtype: Int,
    normType: Int,
    k: Int,
    mask: Mat?,
    update: Int,
    crosscheck: Boolean,
): BatchDistanceResult {
    val dist = mat()
    val nidx = if (k > 0) mat() else null
    val ok = cvk_batch_distance(
        src1.nativeHandle(), src2.nativeHandle(), dist.nativeHandle(), dtype,
        nidx?.nativeHandle(), normType, k, mask?.nativeHandle(), update,
        if (crosscheck) 1 else 0,
    )
    if (ok == 0) {
        dist.close()
        nidx?.close()
        throw OpenCVException("batchDistance", lastNativeError())
    }
    return BatchDistanceResult(dist, nidx ?: mat())
}

actual fun calcCovarMatrix(samples: Mat, flags: Int, ctype: Int): CovarResult {
    val covar = mat()
    val mean = mat()
    cvk_calc_covar_matrix(
        samples.nativeHandle(), covar.nativeHandle(), mean.nativeHandle(),
        flags, ctype,
    )
    return CovarResult(covar, mean)
}

actual fun completeSymm(m: Mat, lowerToUpper: Boolean) {
    cvk_complete_symm(m.nativeHandle(), if (lowerToUpper) 1 else 0)
}

actual fun solveCubic(coeffs: Mat): SolveCubicResult = memScoped {
    val out = alloc<CPointerVar<cvk_mat_t>>()
    val count = cvk_solve_cubic(coeffs.nativeHandle(), out.ptr)
    SolveCubicResult(count, nativeMat(out.value, "solveCubic"))
}

actual fun solvePoly(coeffs: Mat, maxIters: Int): SolvePolyResult = memScoped {
    val out = alloc<CPointerVar<cvk_mat_t>>()
    val epsilon = cvk_solve_poly(coeffs.nativeHandle(), out.ptr, maxIters)
    SolvePolyResult(epsilon, nativeMat(out.value, "solvePoly"))
}

actual fun mulTransposed(src: Mat, aTa: Boolean, delta: Mat?, scale: Double, dtype: Int): Mat =
    nativeMat(
        cvk_mul_transposed(
            src.nativeHandle(), if (aTa) 1 else 0, delta?.nativeHandle(),
            scale, dtype,
        ),
        "mulTransposed",
    )

actual fun flipND(src: Mat, axis: Int): Mat =
    nativeMat(cvk_flip_nd(src.nativeHandle(), axis), "flipND")

actual fun broadcast(src: Mat, shape: Mat): Mat =
    nativeMat(cvk_broadcast(src.nativeHandle(), shape.nativeHandle()), "broadcast")

actual fun transposeND(src: Mat, order: IntArray): Mat = memScoped {
    nativeMat(
        order.usePinned { pinned ->
            cvk_transpose_nd(src.nativeHandle(), pinned.addressOf(0), order.size.convert<size_t>())
        },
        "transposeND",
    )
}

actual fun Mat.copyTo(mask: Mat?): Mat =
    nativeMat(cvk_copy_to(nativeHandle(), mask?.nativeHandle()), "copyTo")

actual fun scaleAdd(src1: Mat, alpha: Double, src2: Mat): Mat =
    nativeMat(cvk_scale_add(src1.nativeHandle(), alpha, src2.nativeHandle()), "scaleAdd")

actual fun gemm(src1: Mat, src2: Mat, alpha: Double, src3: Mat?, beta: Double, flags: Int): Mat =
    nativeMat(
        cvk_gemm_flags(src1.nativeHandle(), src2.nativeHandle(), alpha, src3?.nativeHandle(), beta, flags),
        "gemm",
    )

actual fun eigenNonSymmetric(src: Mat): Pair<Mat, Mat> = memScoped {
    val eigenvalues = alloc<CPointerVar<cvk_mat_t>>()
    val eigenvectors = alloc<CPointerVar<cvk_mat_t>>()
    cvk_eigen_non_symmetric(src.nativeHandle(), eigenvalues.ptr, eigenvectors.ptr)
    nativeMat(eigenvalues.value, "eigenNonSymmetric.eigenvalues") to
        nativeMat(eigenvectors.value, "eigenNonSymmetric.eigenvectors")
}

actual fun finiteMask(src: Mat): Mat =
    nativeMat(cvk_finite_mask(src.nativeHandle()), "finiteMask")

// =========================================================================
// masked statistics
// =========================================================================

actual fun minMaxLoc(src: Mat, mask: Mat?): MinMaxLoc = memScoped {
    val out = allocArray<DoubleVar>(6)
    if (mask != null) {
        cvk_mat_min_max_loc_masked(src.nativeHandle(), mask.nativeHandle(), out)
    } else {
        cvk_mat_min_max_loc(src.nativeHandle(), out)
    }
    MinMaxLoc(
        minVal = out[0], maxVal = out[1],
        minX = out[2].toInt(), minY = out[3].toInt(),
        maxX = out[4].toInt(), maxY = out[5].toInt(),
    )
}

actual fun mean(src: Mat, mask: Mat?): Scalar =
    if (mask != null) {
        cvk_mat_mean_masked(src.nativeHandle(), mask.nativeHandle())
            .useContents { Scalar(v0, v1, v2, v3) }
    } else {
        cvk_mat_mean(src.nativeHandle()).useContents { Scalar(v0, v1, v2, v3) }
    }

actual fun norm(src: Mat, normType: Int, mask: Mat?): Double =
    if (mask != null) {
        cvk_norm_masked(src.nativeHandle(), normType, mask.nativeHandle())
    } else {
        cvk_norm(src.nativeHandle(), normType)
    }

actual fun norm(src1: Mat, src2: Mat, normType: Int, mask: Mat?): Double =
    if (mask != null) {
        cvk_norm_diff_masked(src1.nativeHandle(), src2.nativeHandle(), normType, mask.nativeHandle())
    } else {
        cvk_norm_diff(src1.nativeHandle(), src2.nativeHandle(), normType)
    }

// =========================================================================
// range check / shuffle
// =========================================================================

actual fun checkRange(a: Mat, quiet: Boolean, minVal: Double, maxVal: Double): Boolean =
    cvk_check_range(a.nativeHandle(), if (quiet) 1 else 0, minVal, maxVal) != 0

actual fun randShuffle(dst: Mat, iterFactor: Double) {
    cvk_rand_shuffle(dst.nativeHandle(), iterFactor)
}

// =========================================================================
// environment / runtime info
// =========================================================================

actual fun getTickCount(): Long = cvk_get_tick_count()

actual fun getTickFrequency(): Double = cvk_get_tick_frequency()

actual fun getCPUTickCount(): Long = cvk_get_cpu_tick_count()

actual fun getNumberOfCPUs(): Int = cvk_get_number_of_cpus()

actual fun checkHardwareSupport(feature: Int): Boolean = cvk_check_hardware_support(feature) != 0

actual fun getHardwareFeatureName(feature: Int): String =
    cvk_get_hardware_feature_name(feature)?.toKString() ?: ""

actual fun getVersionString(): String = cvk_get_version_string()?.toKString() ?: ""

actual fun getVersionMajor(): Int = cvk_get_version_major()

actual fun getVersionMinor(): Int = cvk_get_version_minor()

actual fun getVersionRevision(): Int = cvk_get_version_revision()

actual fun getThreadNum(): Int = cvk_get_thread_num()

actual fun getDefaultAlgorithmHint(): Int = cvk_get_default_algorithm_hint()

actual fun useOptimized(): Boolean = cvk_use_optimized() != 0

actual fun setUseOptimized(onoff: Boolean) {
    cvk_set_use_optimized(if (onoff) 1 else 0)
}

actual fun getCPUFeaturesLine(): String = cvk_get_cpu_features_line()?.toKString() ?: ""

actual fun useIPP(): Boolean = cvk_use_ipp() != 0

actual fun setUseIPP(flag: Boolean) {
    cvk_set_use_ipp(if (flag) 1 else 0)
}

actual fun getIppVersion(): String = cvk_get_ipp_version()?.toKString() ?: ""

actual fun useIPP_NotExact(): Boolean = cvk_use_ipp_not_exact() != 0

actual fun setUseIPP_NotExact(flag: Boolean) {
    cvk_set_use_ipp_not_exact(if (flag) 1 else 0)
}

actual fun findFile(relativePath: String, required: Boolean, silentMode: Boolean): String =
    cvk_find_file(relativePath, if (required) 1 else 0, if (silentMode) 1 else 0)?.toKString() ?: ""

actual fun findFileOrKeep(relativePath: String, silentMode: Boolean): String =
    cvk_find_file_or_keep(relativePath, if (silentMode) 1 else 0)?.toKString() ?: relativePath

actual fun addSamplesDataSearchPath(path: String) {
    cvk_add_samples_data_search_path(path)
}

actual fun addSamplesDataSearchSubDirectory(subdir: String) {
    cvk_add_samples_data_search_sub_directory(subdir)
}
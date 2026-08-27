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
package cn.enaium.opencv

// =========================================================================
// org.opencv.core.Core statics — scalar math
// =========================================================================

/** `cv::cubeRoot`: cube root handling negative arguments correctly. */
expect fun cubeRoot(value: Float): Float

/**
 * `cv::fastAtan2`: full-range angle of the vector (y, x) in degrees,
 * varying from 0 to 360.
 */
expect fun fastAtan2(y: Float, x: Float): Float

/** `cv::borderInterpolate`: source coordinate for a mirrored border access. */
expect fun borderInterpolate(p: Int, len: Int, borderType: Int): Int

// =========================================================================
// RNG
// =========================================================================

/**
 * Wrapper over a `cv::RNG` (MWC algorithm) random generator. Every method
 * advances the generator state; [close] releases the native handle.
 */
interface RNG : AutoCloseable {
    /** Next raw 32-bit random word. */
    fun next(): UInt

    /** Uniform integer from [a, b). */
    fun uniform(a: Int, b: Int): Int

    /** Uniform float from [a, b). */
    fun uniform(a: Float, b: Float): Float

    /** Uniform double from [a, b). */
    fun uniform(a: Double, b: Double): Double

    /** Gaussian N(0, [sigma]) sample. */
    fun gaussian(sigma: Double): Double

    override fun close()
}

/**
 * `cv::theRNG`: an owned copy of the thread-local default generator, seeded
 * from its current state. Because the state is copied, drawing from the
 * returned generator does not advance the global one; calling [theRNG]
 * again after [setRNGSeed] yields the same sequence.
 */
expect fun theRNG(): RNG

// =========================================================================
// org.opencv.core.Core statics — array operations
// =========================================================================

/**
 * `cv::hconcat` over a list: concatenates matrices side by side. All inputs
 * must share the same row count and depth.
 */
fun hconcat(src: List<Mat>): Mat {
    require(src.isNotEmpty()) { "hconcat requires at least one matrix" }
    var acc = src.first().clone()
    for (m in src.drop(1)) {
        val next = hconcatPair(acc, m)
        acc.close()
        acc = next
    }
    return acc
}

/**
 * `cv::vconcat` over a list: stacks matrices vertically. All inputs must
 * share the same column count and depth.
 */
fun vconcat(src: List<Mat>): Mat {
    require(src.isNotEmpty()) { "vconcat requires at least one matrix" }
    var acc = src.first().clone()
    for (m in src.drop(1)) {
        val next = vconcatPair(acc, m)
        acc.close()
        acc = next
    }
    return acc
}

/** Two-matrix [hconcat] primitive backed by the native library. */
internal expect fun hconcatPair(a: Mat, b: Mat): Mat

/** Two-matrix [vconcat] primitive backed by the native library. */
internal expect fun vconcatPair(a: Mat, b: Mat): Mat

/**
 * `cv::mixChannels`: copies selected channels between [src] and [dst]
 * matrices. [fromTo] holds `[srcIdx, dstIdx]` channel pairs; every [dst]
 * matrix must be pre-allocated with the source size and depth. The copies
 * happen in place on the [dst] matrices.
 */
expect fun mixChannels(src: List<Mat>, dst: List<Mat>, fromTo: IntArray)

/** Result of [batchDistance]; both Mats must be closed by the caller. */
data class BatchDistanceResult(val dist: Mat, val nidx: Mat)

/**
 * `cv::batchDistance`: naive nearest-neighbor distance matrix between the
 * rows of [src1] (nsrc1 x d) and [src2] (nsrc2 x d). [dist] is nsrc1 x
 * nsrc2 (or nsrc1 x [k] when [k] > 0); [nidx] carries the nearest-neighbor
 * indices and is only produced when [k] > 0.
 *
 * [normType] is a `cv::NormTypes` value (NORM_INF=1, NORM_L1=2, NORM_L2=4,
 * NORM_HAMMING=6, ...); [dtype] may be -1 to pick CV_32F (or CV_32S for
 * Hamming norms).
 */
expect fun batchDistance(
    src1: Mat,
    src2: Mat,
    dtype: Int = -1,
    normType: Int = 4,
    k: Int = 0,
    mask: Mat? = null,
    update: Int = 0,
    crosscheck: Boolean = false,
): BatchDistanceResult

/** Result of [calcCovarMatrix]; both Mats must be closed by the caller. */
data class CovarResult(val covar: Mat, val mean: Mat)

/**
 * `cv::calcCovarMatrix`: covariance matrix (square, of type [ctype]) and
 * mean vector of the rows (or columns, see the COVAR_* flags) of [samples].
 */
expect fun calcCovarMatrix(samples: Mat, flags: Int, ctype: Int = CV_64F): CovarResult

/**
 * `cv::completeSymm`: copies one triangle of the square [m] onto the other,
 * in place. [lowerToUpper] mirrors the lower half up (default mirrors the
 * upper half down).
 */
expect fun completeSymm(m: Mat, lowerToUpper: Boolean = false)

/** Result of [solveCubic]; [roots] must be closed by the caller. */
data class SolveCubicResult(val numRoots: Int, val roots: Mat)

/**
 * `cv::solveCubic`: real roots of a cubic equation. [coeffs] holds 3 (x^3 +
 * a2*x^2 + ...) or 4 coefficients; the returned [SolveCubicResult.roots]
 * matrix is CV_32F with [SolveCubicResult.numRoots] valid entries.
 */
expect fun solveCubic(coeffs: Mat): SolveCubicResult

/** Result of [solvePoly]; [roots] must be closed by the caller. */
data class SolvePolyResult(val epsilon: Double, val roots: Mat)

/**
 * `cv::solvePoly`: real and complex roots of a polynomial. [coeffs] holds
 * the coefficients in ascending order; the returned roots matrix is CV_32FC2
 * (or CV_64FC2) with one (re, im) pair per root.
 */
expect fun solvePoly(coeffs: Mat, maxIters: Int = 300): SolvePolyResult

/**
 * `cv::mulTransposed`: dst = scale * (src - [delta])^T * (src - [delta])
 * when [aTa], otherwise dst = scale * (src - [delta]) * (src - [delta])^T.
 */
expect fun mulTransposed(
    src: Mat,
    aTa: Boolean,
    delta: Mat? = null,
    scale: Double = 1.0,
    dtype: Int = -1,
): Mat

/** `cv::flipND`: flips an N-dimensional array along one [axis]. */
expect fun flipND(src: Mat, axis: Int): Mat

/**
 * `cv::broadcast`: broadcasts [src] to the shape stored in [shape] (a
 * single-row CV_32SC1 matrix, numpy rules).
 */
expect fun broadcast(src: Mat, shape: Mat): Mat

/**
 * `cv::transposeND`: permutes the axes of an N-dimensional array; [order]
 * must be a permutation of 0 until dims-1.
 */
expect fun transposeND(src: Mat, order: IntArray): Mat

/**
 * `cv::copyTo`: copies this matrix into a fresh one. With a [mask], only
 * elements where the mask is non-zero are copied; the remaining elements of
 * the freshly created destination are 0.
 */
expect fun Mat.copyTo(mask: Mat? = null): Mat

/** `cv::scaleAdd`: dst = [src1] * [alpha] + [src2]. */
expect fun scaleAdd(src1: Mat, alpha: Double, src2: Mat): Mat

/**
 * `cv::gemm`: dst = [alpha] * [src1] * [src2] + [beta] * [src3], with
 * [GemmFlags] transposition flags (GEMM_1_T / GEMM_2_T / GEMM_3_T).
 */
expect fun gemm(
    src1: Mat,
    src2: Mat,
    alpha: Double = 1.0,
    src3: Mat? = null,
    beta: Double = 0.0,
    flags: Int = 0,
): Mat

/**
 * `cv::eigenNonSymmetric`: eigenvalues and eigenvectors of a general square
 * matrix; returns (eigenvalues, eigenvectors) as complex CV_32FC2/CV_64FC2
 * matrices.
 */
expect fun eigenNonSymmetric(src: Mat): Pair<Mat, Mat>

/**
 * `cv::finiteMask`: 255 where every channel of [src] is finite (neither NaN
 * nor Inf), else 0; returns a CV_8UC1 matrix of the same size.
 */
expect fun finiteMask(src: Mat): Mat

// =========================================================================
// org.opencv.core.Core statics — masked statistics
// =========================================================================

/** `cv::minMaxLoc` restricted to a [mask]; [mask] null computes over all. */
expect fun minMaxLoc(src: Mat, mask: Mat? = null): MinMaxLoc

/** `cv::mean` over the [mask]ed region; [mask] null computes over all. */
expect fun mean(src: Mat, mask: Mat? = null): Scalar

/**
 * `cv::norm` of [src] under a NormTypes [normType] (default NORM_L2 = 4),
 * restricted to the [mask]ed region when a mask is given.
 */
expect fun norm(src: Mat, normType: Int = 4, mask: Mat? = null): Double

/**
 * `cv::norm` of the difference [src1] - [src2] under a NormTypes
 * [normType] (default NORM_L2 = 4), restricted to the [mask]ed region when
 * a mask is given.
 */
expect fun norm(src1: Mat, src2: Mat, normType: Int = 4, mask: Mat? = null): Double

// =========================================================================
// org.opencv.core.Core statics — range check / shuffle
// =========================================================================

/**
 * `cv::checkRange`: true when every element of [a] is finite and within
 * [minVal]..[maxVal]. When [quiet] is false an out-of-range value raises an
 * exception instead of returning false.
 */
expect fun checkRange(
    a: Mat,
    quiet: Boolean = true,
    minVal: Double = -Double.MAX_VALUE,
    maxVal: Double = Double.MAX_VALUE,
): Boolean

/**
 * `cv::randShuffle`: randomly shuffles the elements of the 1D [dst] in
 * place; [iterFactor] scales the number of swap operations.
 */
expect fun randShuffle(dst: Mat, iterFactor: Double = 1.0)

// =========================================================================
// org.opencv.core.Core statics — environment / runtime info
// =========================================================================

/** `cv::getTickCount`: number of ticks since an arbitrary reference point. */
expect fun getTickCount(): Long

/** `cv::getTickFrequency`: ticks per second of the [getTickCount] clock. */
expect fun getTickFrequency(): Double

/** `cv::getCPUTickCount`: CPU-cycle counter (may be 0 on some platforms). */
expect fun getCPUTickCount(): Long

/** `cv::getNumberOfCPUs`: CPU count OpenCV's thread pool can use. */
expect fun getNumberOfCPUs(): Int

/** `cv::checkHardwareSupport`: whether the CPU feature bit is set. */
expect fun checkHardwareSupport(feature: Int): Boolean

/** `cv::getHardwareFeatureName`: human-readable CPU feature name. */
expect fun getHardwareFeatureName(feature: Int): String

/** `cv::getVersionString`, e.g. "5.0.0". */
expect fun getVersionString(): String

/** `cv::getVersionMajor`. */
expect fun getVersionMajor(): Int

/** `cv::getVersionMinor`. */
expect fun getVersionMinor(): Int

/** `cv::getVersionRevision`. */
expect fun getVersionRevision(): Int

/** `cv::getThreadNum` (deprecated): id of the calling OpenCV thread. */
expect fun getThreadNum(): Int

/** `cv::getDefaultAlgorithmHint`: current default [ALGO_HINT_*] value. */
expect fun getDefaultAlgorithmHint(): Int

/** `cv::useOptimized`: whether OpenCV's optimized routines are enabled. */
expect fun useOptimized(): Boolean

/** `cv::setUseOptimized`: enables or disables optimized routines. */
expect fun setUseOptimized(onoff: Boolean)

/** `cv::getCPUFeaturesLine`: comma-separated list of supported CPU features. */
expect fun getCPUFeaturesLine(): String

/** `cv::ipp::useIPP`: whether IPP acceleration is enabled. */
expect fun useIPP(): Boolean

/** `cv::ipp::setUseIPP`: enables or disables IPP acceleration. */
expect fun setUseIPP(flag: Boolean)

/** `cv::ipp::getIppVersion`: IPP version string, "0.0.0" when unavailable. */
expect fun getIppVersion(): String

/** `cv::ipp::useIPP_NotExact`: whether non-exact IPP results are allowed. */
expect fun useIPP_NotExact(): Boolean

/** `cv::ipp::setUseIPP_NotExact`: allows/disallows non-exact IPP results. */
expect fun setUseIPP_NotExact(flag: Boolean)

/**
 * `cv::samples::findFile`: resolves [relativePath] against the registered
 * sample data search paths. When [required] and the file is missing this
 * throws; with [silentMode] a missing file yields an empty string.
 */
expect fun findFile(
    relativePath: String,
    required: Boolean = true,
    silentMode: Boolean = false,
): String

/** `cv::samples::findFileOrKeep`: like [findFile] but returns the path
 *  unchanged when the file cannot be resolved. */
expect fun findFileOrKeep(relativePath: String, silentMode: Boolean = false): String

/** `cv::samples::addSamplesDataSearchPath`: registers a search root. */
expect fun addSamplesDataSearchPath(path: String)

/** `cv::samples::addSamplesDataSearchSubDirectory`: registers a subdir. */
expect fun addSamplesDataSearchSubDirectory(subdir: String)

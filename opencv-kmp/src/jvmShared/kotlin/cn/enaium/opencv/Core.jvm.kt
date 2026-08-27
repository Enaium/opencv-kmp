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

import kotlin.concurrent.Volatile

private fun lastNativeError(): String? = JniCore.lastError()


// =========================================================================
// scalar math
// =========================================================================

actual fun cubeRoot(value: Float): Float = JniCore.cubeRoot(value)

actual fun fastAtan2(y: Float, x: Float): Float = JniCore.fastAtan2(y, x)

actual fun borderInterpolate(p: Int, len: Int, borderType: Int): Int =
    JniCore.borderInterpolate(p, len, borderType)

// =========================================================================
// RNG
// =========================================================================

internal class JvmRng(@Volatile private var handle: Long) : RNG {

    private fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("RNG is closed")

    override fun next(): UInt = JniCore.rngNext(check()).toUInt()

    override fun uniform(a: Int, b: Int): Int = JniCore.rngUniformInt(check(), a, b)

    override fun uniform(a: Float, b: Float): Float = JniCore.rngUniformFloat(check(), a, b)

    override fun uniform(a: Double, b: Double): Double = JniCore.rngUniformDouble(check(), a, b)

    override fun gaussian(sigma: Double): Double = JniCore.rngGaussian(check(), sigma)

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniCore.rngRelease(h)
        }
    }
}

actual fun theRNG(): RNG =
    JvmRng(JniCore.rngFromGlobal().takeIf { it != 0L }
        ?: throw OpenCVException("theRNG", lastNativeError()))

// =========================================================================
// array operations
// =========================================================================

internal actual fun hconcatPair(a: Mat, b: Mat): Mat =
    jvmMat(Jni.hconcat(handleOf(a), handleOf(b)), "hconcat")

internal actual fun vconcatPair(a: Mat, b: Mat): Mat =
    jvmMat(Jni.vconcat(handleOf(a), handleOf(b)), "vconcat")

actual fun mixChannels(src: List<Mat>, dst: List<Mat>, fromTo: IntArray) {
    require(fromTo.size % 2 == 0) { "mixChannels fromTo must hold [srcIdx, dstIdx] pairs" }
    JniCore.mixChannels(
        src.map(::handleOf).toLongArray(),
        dst.map(::handleOf).toLongArray(),
        fromTo,
    )
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
    val ok = JniCore.batchDistance(
        handleOf(src1), handleOf(src2), handleOf(dist), dtype,
        nidx?.let(::handleOf) ?: 0L, normType, k,
        mask?.let(::handleOf) ?: 0L, update, crosscheck,
    )
    if (!ok) {
        dist.close()
        nidx?.close()
        throw OpenCVException("batchDistance", lastNativeError())
    }
    return BatchDistanceResult(dist, nidx ?: mat())
}

actual fun calcCovarMatrix(samples: Mat, flags: Int, ctype: Int): CovarResult {
    val covar = mat()
    val mean = mat()
    JniCore.calcCovarMatrix(handleOf(samples), handleOf(covar), handleOf(mean), flags, ctype)
    return CovarResult(covar, mean)
}

actual fun completeSymm(m: Mat, lowerToUpper: Boolean) {
    JniCore.completeSymm(handleOf(m), lowerToUpper)
}

actual fun solveCubic(coeffs: Mat): SolveCubicResult {
    val roots = LongArray(1)
    val count = JniCore.solveCubic(handleOf(coeffs), roots)
    return SolveCubicResult(count, jvmMat(roots[0], "solveCubic"))
}

actual fun solvePoly(coeffs: Mat, maxIters: Int): SolvePolyResult {
    val roots = LongArray(1)
    val epsilon = JniCore.solvePoly(handleOf(coeffs), roots, maxIters)
    return SolvePolyResult(epsilon, jvmMat(roots[0], "solvePoly"))
}

actual fun mulTransposed(src: Mat, aTa: Boolean, delta: Mat?, scale: Double, dtype: Int): Mat =
    jvmMat(
        JniCore.mulTransposed(handleOf(src), aTa, delta?.let(::handleOf) ?: 0L, scale, dtype),
        "mulTransposed",
    )

actual fun flipND(src: Mat, axis: Int): Mat =
    jvmMat(JniCore.flipND(handleOf(src), axis), "flipND")

actual fun broadcast(src: Mat, shape: Mat): Mat =
    jvmMat(JniCore.broadcast(handleOf(src), handleOf(shape)), "broadcast")

actual fun transposeND(src: Mat, order: IntArray): Mat =
    jvmMat(JniCore.transposeND(handleOf(src), order), "transposeND")

actual fun Mat.copyTo(mask: Mat?): Mat =
    jvmMat(JniCore.copyTo(handleOf(this), mask?.let(::handleOf) ?: 0L), "copyTo")

actual fun scaleAdd(src1: Mat, alpha: Double, src2: Mat): Mat =
    jvmMat(JniCore.scaleAdd(handleOf(src1), alpha, handleOf(src2)), "scaleAdd")

actual fun gemm(src1: Mat, src2: Mat, alpha: Double, src3: Mat?, beta: Double, flags: Int): Mat =
    jvmMat(
        JniCore.gemmFlags(handleOf(src1), handleOf(src2), alpha, src3?.let(::handleOf) ?: 0L, beta, flags),
        "gemm",
    )

actual fun eigenNonSymmetric(src: Mat): Pair<Mat, Mat> {
    val parts = JniCore.eigenNonSymmetric(handleOf(src))
    return jvmMat(parts[0], "eigenNonSymmetric.eigenvalues") to
        jvmMat(parts[1], "eigenNonSymmetric.eigenvectors")
}

actual fun finiteMask(src: Mat): Mat =
    jvmMat(JniCore.finiteMask(handleOf(src)), "finiteMask")

// =========================================================================
// masked statistics
// =========================================================================

actual fun minMaxLoc(src: Mat, mask: Mat?): MinMaxLoc {
    val out = if (mask != null) {
        JniCore.minMaxLocMasked(handleOf(src), handleOf(mask))
    } else {
        Jni.minMaxLoc(handleOf(src))
    }
    return MinMaxLoc(
        minVal = out[0], maxVal = out[1],
        minX = out[2].toInt(), minY = out[3].toInt(),
        maxX = out[4].toInt(), maxY = out[5].toInt(),
    )
}

actual fun mean(src: Mat, mask: Mat?): Scalar {
    val out = if (mask != null) {
        JniCore.meanMasked(handleOf(src), handleOf(mask))
    } else {
        Jni.mean(handleOf(src))
    }
    return Scalar(out[0], out[1], out[2], out[3])
}

actual fun norm(src: Mat, normType: Int, mask: Mat?): Double =
    if (mask != null) {
        JniCore.normMasked(handleOf(src), normType, handleOf(mask))
    } else {
        Jni.norm(handleOf(src), normType)
    }

actual fun norm(src1: Mat, src2: Mat, normType: Int, mask: Mat?): Double =
    if (mask != null) {
        JniCore.normDiffMasked(handleOf(src1), handleOf(src2), normType, handleOf(mask))
    } else {
        Jni.normDiff(handleOf(src1), handleOf(src2), normType)
    }

// =========================================================================
// range check / shuffle
// =========================================================================

actual fun checkRange(a: Mat, quiet: Boolean, minVal: Double, maxVal: Double): Boolean =
    JniCore.checkRange(handleOf(a), quiet, minVal, maxVal)

actual fun randShuffle(dst: Mat, iterFactor: Double) {
    JniCore.randShuffle(handleOf(dst), iterFactor)
}

// =========================================================================
// environment / runtime info
// =========================================================================

actual fun getTickCount(): Long = JniCore.getTickCount()

actual fun getTickFrequency(): Double = JniCore.getTickFrequency()

actual fun getCPUTickCount(): Long = JniCore.getCPUTickCount()

actual fun getNumberOfCPUs(): Int = JniCore.getNumberOfCPUs()

actual fun checkHardwareSupport(feature: Int): Boolean = JniCore.checkHardwareSupport(feature)

actual fun getHardwareFeatureName(feature: Int): String =
    JniCore.getHardwareFeatureName(feature) ?: ""

actual fun getVersionString(): String = JniCore.getVersionString()

actual fun getVersionMajor(): Int = JniCore.getVersionMajor()

actual fun getVersionMinor(): Int = JniCore.getVersionMinor()

actual fun getVersionRevision(): Int = JniCore.getVersionRevision()

actual fun getThreadNum(): Int = JniCore.getThreadNum()

actual fun getDefaultAlgorithmHint(): Int = JniCore.getDefaultAlgorithmHint()

actual fun useOptimized(): Boolean = JniCore.useOptimized()

actual fun setUseOptimized(onoff: Boolean) {
    JniCore.setUseOptimized(onoff)
}

actual fun getCPUFeaturesLine(): String = JniCore.getCPUFeaturesLine()

actual fun useIPP(): Boolean = JniCore.useIPP()

actual fun setUseIPP(flag: Boolean) {
    JniCore.setUseIPP(flag)
}

actual fun getIppVersion(): String = JniCore.getIppVersion()

actual fun useIPP_NotExact(): Boolean = JniCore.useIPPNotExact()

actual fun setUseIPP_NotExact(flag: Boolean) {
    JniCore.setUseIPPNotExact(flag)
}

actual fun findFile(relativePath: String, required: Boolean, silentMode: Boolean): String =
    JniCore.findFile(relativePath, required, silentMode)
        ?: throw OpenCVException("findFile", lastNativeError())

actual fun findFileOrKeep(relativePath: String, silentMode: Boolean): String =
    JniCore.findFileOrKeep(relativePath, silentMode) ?: relativePath

actual fun addSamplesDataSearchPath(path: String) {
    JniCore.addSamplesDataSearchPath(path)
}

actual fun addSamplesDataSearchSubDirectory(subdir: String) {
    JniCore.addSamplesDataSearchSubDirectory(subdir)
}

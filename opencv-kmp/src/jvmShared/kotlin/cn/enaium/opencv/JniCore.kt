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

/**
 * JNI bridge for the org.opencv.core.Core statics (core slice).
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniCore_<name>`
 * function in jni/jni_core.cpp. All members are public (no `internal`
 * modifier) so their JVM names are not mangled by the Kotlin compiler.
 *
 * Mat handles travel as jlong pointers; scalars expand into primitives.
 * The native library is loaded by [Jni]'s initializer.
 */
internal object JniCore {

    // errors

    external fun lastError(): String?

    // scalar math

    external fun cubeRoot(value: Float): Float
    external fun fastAtan2(y: Float, x: Float): Float
    external fun borderInterpolate(p: Int, len: Int, borderType: Int): Int

    // RNG

    external fun rngFromGlobal(): Long
    external fun rngCreate(seed: Long): Long
    external fun rngNext(rng: Long): Long
    external fun rngUniformInt(rng: Long, a: Int, b: Int): Int
    external fun rngUniformFloat(rng: Long, a: Float, b: Float): Float
    external fun rngUniformDouble(rng: Long, a: Double, b: Double): Double
    external fun rngGaussian(rng: Long, sigma: Double): Double
    external fun rngRelease(rng: Long)

    // array operations

    external fun mixChannels(srcs: LongArray, dsts: LongArray, fromTo: IntArray)
    external fun batchDistance(
        src1: Long, src2: Long, dist: Long, dtype: Int, nidx: Long,
        normType: Int, k: Int, mask: Long, update: Int, crosscheck: Boolean,
    ): Boolean

    external fun calcCovarMatrix(samples: Long, covar: Long, mean: Long, flags: Int, ctype: Int)
    external fun completeSymm(m: Long, lowerToUpper: Boolean)
    external fun solveCubic(coeffs: Long, rootsOut: LongArray): Int
    external fun solvePoly(coeffs: Long, rootsOut: LongArray, maxIters: Int): Double
    external fun mulTransposed(src: Long, aTa: Boolean, delta: Long, scale: Double, dtype: Int): Long
    external fun flipND(src: Long, axis: Int): Long
    external fun broadcast(src: Long, shape: Long): Long
    external fun transposeND(src: Long, order: IntArray): Long
    external fun copyTo(src: Long, mask: Long): Long
    external fun scaleAdd(a: Long, alpha: Double, b: Long): Long
    external fun gemmFlags(a: Long, b: Long, alpha: Double, c: Long, gamma: Double, flags: Int): Long
    external fun eigenNonSymmetric(src: Long): LongArray
    external fun finiteMask(src: Long): Long

    // masked statistics

    external fun minMaxLocMasked(src: Long, mask: Long): DoubleArray
    external fun meanMasked(src: Long, mask: Long): DoubleArray
    external fun normMasked(src: Long, normType: Int, mask: Long): Double
    external fun normDiffMasked(a: Long, b: Long, normType: Int, mask: Long): Double

    // range check / shuffle

    external fun checkRange(a: Long, quiet: Boolean, minVal: Double, maxVal: Double): Boolean
    external fun randShuffle(dst: Long, iterFactor: Double)

    // environment / runtime info

    external fun getTickCount(): Long
    external fun getTickFrequency(): Double
    external fun getNumberOfCPUs(): Int
    external fun checkHardwareSupport(feature: Int): Boolean
    external fun getHardwareFeatureName(feature: Int): String?
    external fun getVersionString(): String
    external fun getVersionMajor(): Int
    external fun getVersionMinor(): Int
    external fun getVersionRevision(): Int
    external fun getCPUTickCount(): Long
    external fun getThreadNum(): Int
    external fun getDefaultAlgorithmHint(): Int
    external fun useOptimized(): Boolean
    external fun setUseOptimized(onoff: Boolean)
    external fun getCPUFeaturesLine(): String
    external fun useIPP(): Boolean
    external fun setUseIPP(flag: Boolean)
    external fun getIppVersion(): String
    external fun useIPPNotExact(): Boolean
    external fun setUseIPPNotExact(flag: Boolean)
    external fun findFile(relativePath: String, required: Boolean, silentMode: Boolean): String?
    external fun findFileOrKeep(relativePath: String, silentMode: Boolean): String?
    external fun addSamplesDataSearchPath(path: String)
    external fun addSamplesDataSearchSubDirectory(subdir: String)
}

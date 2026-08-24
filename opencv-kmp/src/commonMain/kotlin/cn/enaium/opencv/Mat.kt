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
 * A reference-counted OpenCV matrix (`cv::Mat`).
 *
 * Every handle owns one native reference; call [close] (or [Mat.use]) to
 * release it. Operations that return a new [Mat] allocate an independent
 * result unless documented otherwise — only [roi] shares pixels with its
 * source, mirroring `mat(rect)` semantics.
 */
interface Mat : AutoCloseable {
    val rows: Int
    val cols: Int

    /** OpenCV type id (depth + channels); decode with [cvDepthOf]/[cvChannelsOf]. */
    val type: Int
    val channels: Int

    /** Bytes per element across all channels. */
    val elemSize: Int
    val total: Int
    val isEmpty: Boolean

    /** Column-major friendly view: `(cols, rows)`. */
    val size: Size get() = Size(width = cols, height = rows)

    // ---- lifecycle / views -------------------------------------------------

    /** Deep copy with independent pixel storage. */
    fun clone(): Mat

    /**
     * Region of interest sharing this matrix's pixels. Closing either handle
     * is safe; the storage lives until the last reference is released.
     */
    fun roi(rect: Rect): Mat

    fun convertTo(type: Int = this.type, alpha: Double = 1.0, beta: Double = 0.0): Mat

    // ---- element access ----------------------------------------------------

    /** Reads channel 0 of the element at ([row], [col]). */
    operator fun get(row: Int, col: Int): Double

    /** Writes channel 0 of the element at ([row], [col]) with saturation. */
    operator fun set(row: Int, col: Int, value: Double)

    /** Reads any [channel] of the element at ([row], [col]). */
    fun at(row: Int, col: Int, channel: Int = 0): Double

    /** Writes any [channel] of the element at ([row], [col]) with saturation. */
    fun put(row: Int, col: Int, channel: Int = 0, value: Double)

    /** Bulk pixel bytes; size equals [total] * [elemSize]. */
    var pixels: ByteArray

    // ---- arithmetic (each returns a new Mat) --------------------------------

    operator fun plus(other: Mat): Mat
    operator fun minus(other: Mat): Mat

    /** Element-wise product times [scale] (`cv::multiply`). */
    operator fun times(other: Mat): Mat

    /** Divides element-wise by [other] (`cv::divide`). */
    operator fun div(other: Mat): Mat

    /** Scales every element by [scale] (`cv::multiply` with a scalar). */
    operator fun times(scale: Double): Mat

    fun absDiff(other: Mat): Mat
    infix fun bitwiseAnd(other: Mat): Mat
    infix fun bitwiseOr(other: Mat): Mat
    infix fun bitwiseXor(other: Mat): Mat

    fun bitwiseNot(): Mat
    fun min(other: Mat): Mat
    fun max(other: Mat): Mat

    /** Binary mask where lower <= value <= upper component-wise. */
    fun inRange(lower: Scalar, upper: Scalar): Mat

    // ---- geometry ------------------------------------------------------------

    fun transpose(): Mat

    /** [flipCode]: [FlipMode.VERTICAL], [FlipMode.HORIZONTAL] or [FlipMode.BOTH]. */
    fun flip(flipCode: Int): Mat

    // ---- reductions / statistics ----------------------------------------------

    val mean: Scalar
    val sum: Scalar
    fun meanStdDev(): Pair<Scalar, Scalar>
    fun minMaxLoc(): MinMaxLoc
    val nonZeroCount: Int

    // ---- imgproc -----------------------------------------------------------------

    fun cvtColor(code: Int): Mat

    fun resize(
        width: Int,
        height: Int,
        interpolation: Int = InterpolationFlags.LINEAR,
    ): Mat

    fun resize(size: Size, interpolation: Int = InterpolationFlags.LINEAR): Mat =
        resize(width = size.width, height = size.height, interpolation = interpolation)

    fun gaussianBlur(
        kernelWidth: Int,
        kernelHeight: Int,
        sigmaX: Double = 0.0,
        sigmaY: Double = 0.0,
    ): Mat

    fun medianBlur(kernelSize: Int): Mat

    fun threshold(thresh: Double, maxVal: Double, type: Int = ThresholdTypes.BINARY): Mat

    fun adaptiveThreshold(
        maxValue: Double,
        method: Int = AdaptiveThresholdTypes.GAUSSIAN_C,
        type: Int = ThresholdTypes.BINARY,
        blockSize: Int = 3,
        c: Double = 5.0,
    ): Mat

    fun canny(
        threshold1: Double,
        threshold2: Double,
        apertureSize: Int = 3,
        l2Gradient: Boolean = false,
    ): Mat

    fun sobel(dx: Int, dy: Int, kernelSize: Int = 3): Mat
    fun laplacian(kernelSize: Int = 1): Mat

    // ---- drawing (in-place) ------------------------------------------------------

    fun rectangle(from: Point, to: Point, color: Scalar, thickness: Int = LineTypes.LINE_8)
    fun circle(center: Point, radius: Int, color: Scalar, thickness: Int = LineTypes.LINE_8)
    fun line(from: Point, to: Point, color: Scalar, thickness: Int = LineTypes.LINE_8)

    override fun close()
}

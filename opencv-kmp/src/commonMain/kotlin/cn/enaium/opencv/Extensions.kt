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
// Sugar built on top of the [Mat] members; identical on every platform.
// =========================================================================

/** Runs [block] with the matrix and closes it afterwards. */
inline fun <R> Mat.use(block: (Mat) -> R): R {
    try {
        return block(this)
    } finally {
        close()
    }
}

/**
 * Runs [block] on a clone of this matrix, closing both afterwards; useful
 * for chained one-off pipelines: `image.inPlace { it.gaussianBlur(5, 5) }`.
 */
inline fun <R> Mat.withCopy(block: (Mat) -> R): R = use { source ->
    source.clone().use(block)
}

/** 90 degrees clockwise rotation via transpose + horizontal flip. */
fun Mat.rotate90(): Mat = transpose().let { it.use { t -> t.flip(FlipMode.HORIZONTAL) } }

/** 180 degrees rotation (double flip). */
fun Mat.rotate180(): Mat = flip(FlipMode.BOTH)

/** 270 degrees clockwise rotation (= 90 counter-clockwise). */
fun Mat.rotate270(): Mat = transpose().let { it.use { t -> t.flip(FlipMode.VERTICAL) } }

/** Mirror around the y-axis. */
fun Mat.mirror(): Mat = flip(FlipMode.HORIZONTAL)

/** Upside-down flip around the x-axis. */
fun Mat.upsideDown(): Mat = flip(FlipMode.VERTICAL)

/** Converts an 8-bit image to 32-bit floats in 0..255 range. */
fun Mat.toFloat32(): Mat = convertTo(type = cvMakeType(CV_32F, channels))

/** Converts to single-channel grayscale from BGR/BGRA input. */
fun Mat.toGray(): Mat = cvtColor(
    if (channels == 4) ColorConversionCodes.BGRA2GRAY else ColorConversionCodes.BGR2GRAY,
)

/** Reads the element at ([row], [col]) across every channel as a [Scalar]. */
fun Mat.rowColScalar(row: Int, col: Int): Scalar = Scalar(
    v0 = at(row, col, 0),
    v1 = if (channels > 1) at(row, col, 1) else 0.0,
    v2 = if (channels > 2) at(row, col, 2) else 0.0,
    v3 = if (channels > 3) at(row, col, 3) else 0.0,
)

/** Debug-friendly description without touching pixel data. */
val Mat.shape: String get() = "$rows x $cols x $channels"

/**
 * Fills every element by calling [value] per pixel — convenient for tiny
 * test matrices, too slow for production-sized ones ([pixels] instead).
 */
fun Mat.fill(value: (row: Int, col: Int, channel: Int) -> Double): Mat {
    for (r in 0 until rows) {
        for (c in 0 until cols) {
            for (ch in 0 until channels) {
                put(r, c, ch, value(r, c, ch))
            }
        }
    }
    return this
}

/** Element-wise absolute difference as an infix shortcut. */
infix fun Mat.diff(other: Mat): Mat = absDiff(other)

// =========================================================================
// Mat <-> Scalar arithmetic (cv::add/subtract/multiply/divide with a Scalar)
// =========================================================================

/** Whole-image broadcast: adds [value] to every channel (`img + 50`). */
operator fun Mat.plus(value: Double): Mat = addScalar(Scalar.all(value))

/** Whole-image broadcast: subtracts [value] from every channel (`img - 50`). */
operator fun Mat.minus(value: Double): Mat = subtractScalar(Scalar.all(value))

/** Whole-image broadcast: divides every channel by [value] (`img / 50`). */
operator fun Mat.div(value: Double): Mat = divideScalar(Scalar.all(value))

internal expect fun Mat.addScalar(s: Scalar): Mat
internal expect fun Mat.subtractScalar(s: Scalar): Mat
internal expect fun Mat.multiplyScalar(s: Scalar): Mat
internal expect fun Mat.divideScalar(s: Scalar): Mat

// =========================================================================
// Pure-Kotlin Mat convenience sugar (backed by existing native ops)
// =========================================================================

/** Element-wise square: `this * this` (cv::multiply). */
fun Mat.squared(): Mat = times(this)

/** Element-wise absolute value (`cv::abs` semantics via absdiff with zero). */
fun Mat.abs(): Mat = zeros(rows, cols, type).use { absDiff(it) }

/** Reverse subtraction: [other] - this (`cv2.subtract(other, this)`). */
infix fun Mat.rsub(other: Mat): Mat = other.minus(this)

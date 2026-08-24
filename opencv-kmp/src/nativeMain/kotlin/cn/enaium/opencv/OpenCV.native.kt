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
import cvk.cvk_imdecode
import cvk.cvk_imencode
import cvk.cvk_imread
import cvk.cvk_imwrite
import cvk.cvk_last_error
import cvk.cvk_mat_absdiff
import cvk.cvk_mat_add
import cvk.cvk_mat_bitwise_and
import cvk.cvk_mat_bitwise_not
import cvk.cvk_mat_bitwise_or
import cvk.cvk_mat_bitwise_xor
import cvk.cvk_mat_channels
import cvk.cvk_mat_clone
import cvk.cvk_mat_cols
import cvk.cvk_mat_convert_to
import cvk.cvk_mat_count_non_zero
import cvk.cvk_mat_create
import cvk.cvk_mat_create_filled
import cvk.cvk_mat_data
import cvk.cvk_mat_divide
import cvk.cvk_mat_elem_size
import cvk.cvk_mat_eye
import cvk.cvk_mat_flip
import cvk.cvk_mat_get
import cvk.cvk_mat_in_range
import cvk.cvk_mat_is_empty
import cvk.cvk_mat_max
import cvk.cvk_mat_mean
import cvk.cvk_mat_mean_stddev
import cvk.cvk_mat_min
import cvk.cvk_mat_min_max_loc
import cvk.cvk_mat_multiply
import cvk.cvk_mat_ones
import cvk.cvk_mat_release
import cvk.cvk_mat_roi
import cvk.cvk_mat_rows
import cvk.cvk_mat_set
import cvk.cvk_mat_subtract
import cvk.cvk_mat_t
import cvk.cvk_mat_sum
import cvk.cvk_mat_total
import cvk.cvk_mat_transpose
import cvk.cvk_mat_type
import cvk.cvk_adaptive_threshold
import cvk.cvk_canny
import cvk.cvk_circle
import cvk.cvk_cvt_color
import cvk.cvk_free_buffer
import cvk.cvk_gaussian_blur
import cvk.cvk_laplacian
import cvk.cvk_line
import cvk.cvk_median_blur
import cvk.cvk_rectangle
import cvk.cvk_resize
import cvk.cvk_sobel
import cvk.cvk_threshold
import cvk.cvk_rect
import cvk.cvk_scalar
import cvk.cvk_version
import cvk.cvk_mat_zeros
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.CValue
import kotlinx.cinterop.cValue
import kotlinx.cinterop.convert
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.readBytes
import kotlin.concurrent.Volatile
import platform.posix.size_t
import platform.posix.size_tVar
import kotlinx.cinterop.toKString
import kotlinx.cinterop.useContents
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import kotlinx.cinterop.ptr
import platform.posix.memcpy

// =========================================================================
// Helpers shared by the cinterop-backed implementation
// =========================================================================

internal fun Scalar.toCvk(): CValue<cvk_scalar> {
    val value = this
    return cValue<cvk_scalar> {
        v0 = value.v0
        v1 = value.v1
        v2 = value.v2
        v3 = value.v3
    }
}

private fun Rect.toCvk(): CValue<cvk_rect> {
    val region = this
    return cValue<cvk_rect> {
        x = region.x
        y = region.y
        width = region.width
        height = region.height
    }
}

/** Wraps a raw handle; throws with the native error text when it is NULL. */
internal fun nativeMat(ptr: CPointer<cvk_mat_t>?, operation: String): Mat =
    ptr?.let { NativeMat(it) } ?: throw OpenCVException(operation, lastNativeError())

private fun lastNativeError(): String? {
    val message = cvk_last_error() ?: return null
    return message.toKString()
}

// =========================================================================
// Native (cinterop) Mat implementation
// =========================================================================

internal class NativeMat internal constructor(
    @Volatile private var raw: CPointer<cvk_mat_t>?,
) : Mat {

    internal fun check(): CPointer<cvk_mat_t> =
        raw ?: throw IllegalStateException("OpenCV Mat is closed")

    override val rows: Int get() = cvk_mat_rows(check())
    override val cols: Int get() = cvk_mat_cols(check())
    override val type: Int get() = cvk_mat_type(check())
    override val channels: Int get() = cvk_mat_channels(check())
    override val elemSize: Int get() = cvk_mat_elem_size(check()).toInt()
    override val total: Int get() = cvk_mat_total(check()).toInt()
    override val isEmpty: Boolean get() = cvk_mat_is_empty(check()) != 0

    override fun clone(): Mat = nativeMat(cvk_mat_clone(check()), "clone")

    override fun roi(rect: Rect): Mat =
        memScoped { nativeMat(cvk_mat_roi(check(), rect.toCvk()), "roi") }

    override fun convertTo(type: Int, alpha: Double, beta: Double): Mat =
        nativeMat(cvk_mat_convert_to(check(), type, alpha, beta), "convertTo")

    override fun get(row: Int, col: Int): Double = at(row, col, 0)

    override fun set(row: Int, col: Int, value: Double) {
        put(row, col, 0, value)
    }

    override fun at(row: Int, col: Int, channel: Int): Double =
        cvk_mat_get(check(), row, col, channel)

    override fun put(row: Int, col: Int, channel: Int, value: Double) {
        cvk_mat_set(check(), row, col, channel, value)
    }

    override var pixels: ByteArray
        get() {
            val count = elemSize * total
            if (count == 0) return ByteArray(0)
            return cvk_mat_data(check())?.readBytes(count) ?: ByteArray(0)
        }
        set(value) {
            require(value.size == elemSize * total) {
                "pixel buffer holds ${value.size} bytes but the matrix needs ${elemSize * total}"
            }
            if (value.isEmpty()) return
            value.usePinned { pinned ->
                memcpy(cvk_mat_data(check()), pinned.addressOf(0), value.size.convert())
            }
        }

    override fun plus(other: Mat): Mat =
        nativeMat(cvk_mat_add(check(), other.checked), "plus")

    override fun minus(other: Mat): Mat =
        nativeMat(cvk_mat_subtract(check(), other.checked), "minus")

    override fun times(other: Mat): Mat =
        nativeMat(cvk_mat_multiply(check(), other.checked, 1.0), "times")

    override fun div(other: Mat): Mat =
        nativeMat(cvk_mat_divide(check(), other.checked), "div")

    override fun absDiff(other: Mat): Mat =
        nativeMat(cvk_mat_absdiff(check(), other.checked), "absDiff")

    override infix fun bitwiseAnd(other: Mat): Mat =
        nativeMat(cvk_mat_bitwise_and(check(), other.checked), "bitwiseAnd")

    override infix fun bitwiseOr(other: Mat): Mat =
        nativeMat(cvk_mat_bitwise_or(check(), other.checked), "bitwiseOr")

    override infix fun bitwiseXor(other: Mat): Mat =
        nativeMat(cvk_mat_bitwise_xor(check(), other.checked), "bitwiseXor")

    override fun bitwiseNot(): Mat =
        nativeMat(cvk_mat_bitwise_not(check()), "bitwiseNot")

    override fun min(other: Mat): Mat =
        nativeMat(cvk_mat_min(check(), other.checked), "min")

    override fun max(other: Mat): Mat =
        nativeMat(cvk_mat_max(check(), other.checked), "max")

    override fun inRange(lower: Scalar, upper: Scalar): Mat =
        nativeMat(
            cvk_mat_in_range(check(), lower.toCvk(), upper.toCvk()),
            "inRange",
        )

    override fun transpose(): Mat = nativeMat(cvk_mat_transpose(check()), "transpose")

    override fun flip(flipCode: Int): Mat =
        nativeMat(cvk_mat_flip(check(), flipCode), "flip")

    override val mean: Scalar
        get() = cvk_mat_mean(check()).useContents { Scalar(v0, v1, v2, v3) }

    override val sum: Scalar
        get() = cvk_mat_sum(check()).useContents { Scalar(v0, v1, v2, v3) }

    override fun times(scale: Double): Mat =
        // convertTo(-1, scale, 0) is exactly cv::multiply-by-scalar while
        // keeping depth and channels, and avoids a second matrix argument.
        nativeMat(cvk_mat_convert_to(check(), -1, scale, 0.0), "times(scale)")

    override fun meanStdDev(): Pair<Scalar, Scalar> = memScoped {
        val out = allocArray<DoubleVar>(8)
        cvk_mat_mean_stddev(check(), out)
        Scalar(out[0], out[1], out[2], out[3]) to Scalar(out[4], out[5], out[6], out[7])
    }

    override fun minMaxLoc(): MinMaxLoc = memScoped {
        val out = allocArray<DoubleVar>(6)
        cvk_mat_min_max_loc(check(), out)
        MinMaxLoc(
            minVal = out[0], maxVal = out[1],
            minX = out[2].toInt(), minY = out[3].toInt(),
            maxX = out[4].toInt(), maxY = out[5].toInt(),
        )
    }

    override val nonZeroCount: Int get() = cvk_mat_count_non_zero(check())

    override fun cvtColor(code: Int): Mat =
        nativeMat(cvk_cvt_color(check(), code), "cvtColor")

    override fun resize(width: Int, height: Int, interpolation: Int): Mat =
        nativeMat(cvk_resize(check(), width, height, interpolation), "resize")

    override fun gaussianBlur(kernelWidth: Int, kernelHeight: Int, sigmaX: Double, sigmaY: Double): Mat =
        nativeMat(
            cvk_gaussian_blur(check(), kernelWidth, kernelHeight, sigmaX, sigmaY),
            "gaussianBlur",
        )

    override fun medianBlur(kernelSize: Int): Mat =
        nativeMat(cvk_median_blur(check(), kernelSize), "medianBlur")

    override fun threshold(thresh: Double, maxVal: Double, type: Int): Mat =
        nativeMat(cvk_threshold(check(), thresh, maxVal, type), "threshold")

    override fun adaptiveThreshold(
        maxValue: Double,
        method: Int,
        type: Int,
        blockSize: Int,
        c: Double,
    ): Mat = nativeMat(
        cvk_adaptive_threshold(check(), maxValue, method, type, blockSize, c),
        "adaptiveThreshold",
    )

    override fun canny(threshold1: Double, threshold2: Double, apertureSize: Int, l2Gradient: Boolean): Mat =
        nativeMat(
            cvk_canny(check(), threshold1, threshold2, apertureSize, if (l2Gradient) 1 else 0),
            "canny",
        )

    override fun sobel(dx: Int, dy: Int, kernelSize: Int): Mat =
        nativeMat(cvk_sobel(check(), dx, dy, kernelSize), "sobel")

    override fun laplacian(kernelSize: Int): Mat =
        nativeMat(cvk_laplacian(check(), kernelSize), "laplacian")

    override fun rectangle(from: Point, to: Point, color: Scalar, thickness: Int) {
        cvk_rectangle(check(), from.x, from.y, to.x, to.y, color.toCvk(), thickness)
    }

    override fun circle(center: Point, radius: Int, color: Scalar, thickness: Int) {
        cvk_circle(check(), center.x, center.y, radius, color.toCvk(), thickness)
    }

    override fun line(from: Point, to: Point, color: Scalar, thickness: Int) {
        cvk_line(check(), from.x, from.y, to.x, to.y, color.toCvk(), thickness)
    }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_mat_release(handle)
    }

    private val Mat.checked: CPointer<cvk_mat_t>
        get() = (this as? NativeMat)?.check()
            ?: throw IllegalArgumentException("mat belongs to another platform backend")
}

// =========================================================================
// actual declarations
// =========================================================================

actual val opencvVersion: String get() = cvk_version()!!.toKString()

actual val opencvLastError: String? get() = lastNativeError()

actual fun mat(rows: Int, cols: Int, type: Int): Mat =
    nativeMat(cvk_mat_create(rows, cols, type), "mat")

actual fun mat(rows: Int, cols: Int, type: Int, fill: Scalar): Mat =
    nativeMat(cvk_mat_create_filled(rows, cols, type, fill.toCvk()), "mat(fill)")

actual fun zeros(rows: Int, cols: Int, type: Int): Mat =
    nativeMat(cvk_mat_zeros(rows, cols, type), "zeros")

actual fun ones(rows: Int, cols: Int, type: Int): Mat =
    nativeMat(cvk_mat_ones(rows, cols, type), "ones")

actual fun eye(rows: Int, cols: Int, type: Int): Mat =
    nativeMat(cvk_mat_eye(rows, cols, type), "eye")

actual fun imread(path: String, flags: Int): Mat? =
    memScoped { cvk_imread(path, flags)?.let { NativeMat(it) } }

actual fun imwrite(path: String, mat: Mat): Boolean =
    when (val target = mat as? NativeMat) {
        null -> throw IllegalArgumentException("mat belongs to another platform backend")
        else -> cvk_imwrite(path, target.check()) != 0
    }

actual fun imencode(ext: String, mat: Mat): ByteArray = memScoped {
    val target = mat as? NativeMat
        ?: throw IllegalArgumentException("mat belongs to another platform backend")
    val lengthVar = alloc<size_tVar>()
    val buffer = cvk_imencode(normalizeImageExtension(ext), target.check(), lengthVar.ptr)
        ?: throw OpenCVException("imencode", lastNativeError())
    try {
        buffer.readBytes(lengthVar.value.toInt())
    } finally {
        cvk_free_buffer(buffer)
    }
}

actual fun imdecode(data: ByteArray, flags: Int): Mat? =
    data.asUByteArray().usePinned { pinned ->
        cvk_imdecode(pinned.addressOf(0), data.size.convert<size_t>(), flags)?.let(::NativeMat)
    }

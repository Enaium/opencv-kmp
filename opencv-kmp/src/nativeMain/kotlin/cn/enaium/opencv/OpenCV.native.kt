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
import cvk.cvk_accumulate
import cvk.cvk_accumulate_product
import cvk.cvk_accumulate_square
import cvk.cvk_accumulate_weighted
import cvk.cvk_add_weighted
import cvk.cvk_adaptive_threshold
import cvk.cvk_apply_colormap
import cvk.cvk_apply_colormap_user
import cvk.cvk_approx_poly_dp
import cvk.cvk_arc_length
import cvk.cvk_arrowed_line
import cvk.cvk_bilateral_filter
import cvk.cvk_blur
import cvk.cvk_bounding_rect
import cvk.cvk_box_filter
import cvk.cvk_build_information
import cvk.cvk_calc_back_project
import cvk.cvk_calc_hist
import cvk.cvk_canny
import cvk.cvk_cart_to_polar
import cvk.cvk_circle
import cvk.cvk_clahe_apply
import cvk.cvk_clahe_create
import cvk.cvk_clahe_release
import cvk.cvk_clahe_set_clip_limit
import cvk.cvk_clahe_t
import cvk.cvk_compare
import cvk.cvk_connected_components
import cvk.cvk_connected_components_with_stats
import cvk.cvk_contour_area
import cvk.cvk_corner_sub_pix
import cvk.cvk_convert_scale_abs
import cvk.cvk_copy_make_border
import cvk.cvk_corner_harris
import cvk.cvk_corner_min_eigen_val
import cvk.cvk_create_hanning_window
import cvk.cvk_cvt_color
import cvk.cvk_dct
import cvk.cvk_demosaicing
import cvk.cvk_dft
import cvk.cvk_dilate
import cvk.cvk_distance_transform
import cvk.cvk_div_spectrums
import cvk.cvk_emd
import cvk.cvk_draw_contours
import cvk.cvk_draw_marker
import cvk.cvk_eigen
import cvk.cvk_equalize_hist
import cvk.cvk_ellipse
import cvk.cvk_erode
import cvk.cvk_exp
import cvk.cvk_extract_channel
import cvk.cvk_fill_poly
import cvk.cvk_filter_2d
import cvk.cvk_find_contours
import cvk.cvk_find_non_zero
import cvk.cvk_flood_fill
import cvk.cvk_free_buffer
import cvk.cvk_gaussian_blur
import cvk.cvk_gemm
import cvk.cvk_get_affine_transform
import cvk.cvk_get_gaussian_kernel
import cvk.cvk_get_optimal_dft_size
import cvk.cvk_get_perspective_transform
import cvk.cvk_get_rect_sub_pix
import cvk.cvk_get_rotation_matrix_2d
import cvk.cvk_get_structuring_element
import cvk.cvk_good_features_to_track
import cvk.cvk_grab_cut
import cvk.cvk_has_non_zero
import cvk.cvk_hough_circles
import cvk.cvk_hough_lines
import cvk.cvk_hough_lines_p
import cvk.cvk_have_image_reader
import cvk.cvk_have_image_writer
import cvk.cvk_idct
import cvk.cvk_idft
import cvk.cvk_imcount
import cvk.cvk_imdecode
import cvk.cvk_imencode
import cvk.cvk_imencode_params
import cvk.cvk_imread
import cvk.cvk_imwrite
import cvk.cvk_imwrite_params
import cvk.cvk_insert_channel
import cvk.cvk_integral
import cvk.cvk_kmeans
import cvk.cvk_invert_affine_transform
import cvk.cvk_laplacian
import cvk.cvk_last_error
import cvk.cvk_line
import cvk.cvk_log
import cvk.cvk_lut
import cvk.cvk_magnitude
import cvk.cvk_mahalanobis
import cvk.cvk_mat_absdiff
import cvk.cvk_mat_add
import cvk.cvk_mat_adjust_roi
import cvk.cvk_mat_bitwise_and
import cvk.cvk_mat_bitwise_not
import cvk.cvk_mat_bitwise_or
import cvk.cvk_mat_bitwise_xor
import cvk.cvk_mat_channels
import cvk.cvk_mat_clone
import cvk.cvk_mat_cols
import cvk.cvk_mat_col_range
import cvk.cvk_mat_convert_to
import cvk.cvk_mat_count_non_zero
import cvk.cvk_mat_cross
import cvk.cvk_mat_create
import cvk.cvk_mat_create_filled
import cvk.cvk_mat_data
import cvk.cvk_mat_diag
import cvk.cvk_mat_determinant
import cvk.cvk_mat_divide
import cvk.cvk_mat_dot
import cvk.cvk_mat_dump
import cvk.cvk_mat_elem_size
import cvk.cvk_mat_eye
import cvk.cvk_mat_flip
import cvk.cvk_mat_get
import cvk.cvk_mat_get_values
import cvk.cvk_mat_in_range
import cvk.cvk_mat_inv
import cvk.cvk_mat_is_continuous
import cvk.cvk_mat_is_empty
import cvk.cvk_mat_is_submatrix
import cvk.cvk_mat_locate_roi
import cvk.cvk_mat_max
import cvk.cvk_mat_mean
import cvk.cvk_mat_mean_stddev
import cvk.cvk_mat_min
import cvk.cvk_mat_min_max_loc
import cvk.cvk_mat_multiply
import cvk.cvk_mat_ones
import cvk.cvk_mat_put_values
import cvk.cvk_mat_release
import cvk.cvk_mat_reshape
import cvk.cvk_mat_roi
import cvk.cvk_mat_row_range
import cvk.cvk_mat_rows
import cvk.cvk_mat_set
import cvk.cvk_mat_set_identity
import cvk.cvk_mat_subtract
import cvk.cvk_mat_t
import cvk.cvk_mat_sum
import cvk.cvk_mat_total
import cvk.cvk_mat_trace
import cvk.cvk_mat_transpose
import cvk.cvk_mat_type
import cvk.cvk_mat_zeros
import cvk.cvk_match_shapes
import cvk.cvk_match_template
import cvk.cvk_median_blur
import cvk.cvk_min_area_rect
import cvk.cvk_min_enclosing_circle
import cvk.cvk_moments
import cvk.cvk_morphology_ex
import cvk.cvk_mul_spectrums
import cvk.cvk_normalize
import cvk.cvk_num_threads
import cvk.cvk_pca_backproject
import cvk.cvk_pca_compute
import cvk.cvk_pca_compute_variance
import cvk.cvk_pca_project
import cvk.cvk_patch_nans
import cvk.cvk_perspective_transform
import cvk.cvk_phase
import cvk.cvk_polar_to_cart
import cvk.cvk_polylines
import cvk.cvk_pow
import cvk.cvk_psnr
import cvk.cvk_pyr_down
import cvk.cvk_pyr_mean_shift_filtering
import cvk.cvk_pyr_up
import cvk.cvk_randn
import cvk.cvk_randu
import cvk.cvk_rectangle
import cvk.cvk_reduce
import cvk.cvk_reduce_arg_max
import cvk.cvk_reduce_arg_min
import cvk.cvk_remap
import cvk.cvk_repeat
import cvk.cvk_resize
import cvk.cvk_rotate
import cvk.cvk_set_num_threads
import cvk.cvk_set_rng_seed
import cvk.cvk_solve
import cvk.cvk_sort
import cvk.cvk_sort_idx
import cvk.cvk_sobel
import cvk.cvk_split
import cvk.cvk_sqrt
import cvk.cvk_sqr_box_filter
import cvk.cvk_stack_blur
import cvk.cvk_svd_backsubst
import cvk.cvk_svd_decomp
import cvk.cvk_merge
import cvk.cvk_threshold
import cvk.cvk_threshold_with_mask
import cvk.cvk_transform
import cvk.cvk_version
import cvk.cvk_undistort
import cvk.cvk_warp_affine
import cvk.cvk_warp_perspective
import cvk.cvk_warp_polar
import cvk.cvk_watershed
import cvk.cvk_rect
import cvk.cvk_scalar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.CValue
import kotlinx.cinterop.cValue
import kotlinx.cinterop.convert
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.set
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


    // ---- core: shape / algebra ------------------------------------------------

    override fun reshape(channels: Int, rows: Int): Mat =
        nativeMat(cvk_mat_reshape(check(), channels, rows), "reshape")

    override fun rowRange(start: Int, end: Int): Mat =
        nativeMat(cvk_mat_row_range(check(), start, end), "rowRange")

    override fun colRange(start: Int, end: Int): Mat =
        nativeMat(cvk_mat_col_range(check(), start, end), "colRange")

    override fun diag(d: Int): Mat =
        nativeMat(cvk_mat_diag(check(), d), "diag")

    override fun setIdentity(scale: Double) {
        cvk_mat_set_identity(check(), scale)
    }

    override infix fun dot(other: Mat): Double =
        cvk_mat_dot(check(), other.checked)

    override fun inv(method: Int): Mat? {
        val handle = cvk_mat_inv(check(), method) ?: return null
        return NativeMat(handle)
    }

    override val determinant: Double
        get() = cvk_mat_determinant(check())

    override val trace: Scalar
        get() = cvk_mat_trace(check()).useContents { Scalar(v0, v1, v2, v3) }

    // ---- core: array operations -------------------------------------------------

    override fun split(): List<Mat> {
        val count = channels
        if (count == 0) return emptyList()
        return memScoped {
            val handles = allocArray<CPointerVar<cvk_mat_t>>(count)
            val written = cvk_split(check(), handles, count)
            List(written) { index -> nativeMat(handles[index], "split") }
        }
    }

    override fun normalize(alpha: Double, beta: Double, normType: Int, dtype: Int): Mat =
        nativeMat(cvk_normalize(check(), alpha, beta, normType, dtype), "normalize")

    override fun lut(lut: Mat): Mat =
        nativeMat(cvk_lut(check(), lut.checked), "lut")

    override fun rotate(code: Int): Mat =
        nativeMat(cvk_rotate(check(), code), "rotate")

    override fun copyMakeBorder(
        top: Int,
        bottom: Int,
        left: Int,
        right: Int,
        borderType: Int,
        value: Scalar,
    ): Mat = nativeMat(
        cvk_copy_make_border(check(), top, bottom, left, right, borderType, value.toCvk()),
        "copyMakeBorder",
    )

    override fun addWeighted(alpha: Double, other: Mat, beta: Double, gamma: Double): Mat =
        nativeMat(cvk_add_weighted(check(), alpha, other.checked, beta, gamma), "addWeighted")

    override fun convertScaleAbs(alpha: Double, beta: Double): Mat =
        nativeMat(cvk_convert_scale_abs(check(), alpha, beta), "convertScaleAbs")

    override fun compare(other: Mat, op: Int): Mat =
        nativeMat(cvk_compare(check(), other.checked, op), "compare")

    override fun solve(b: Mat, flags: Int): Mat? {
        val handle = cvk_solve(check(), b.checked, flags) ?: return null
        return NativeMat(handle)
    }

    override fun repeat(nx: Int, ny: Int): Mat =
        nativeMat(cvk_repeat(check(), nx, ny), "repeat")

    override fun transform(m: Mat): Mat =
        nativeMat(cvk_transform(check(), m.checked), "transform")

    override fun perspectiveTransform(m: Mat): Mat =
        nativeMat(cvk_perspective_transform(check(), m.checked), "perspectiveTransform")

    override fun pow(power: Double): Mat =
        nativeMat(cvk_pow(check(), power), "pow")

    override fun sqrt(): Mat =
        nativeMat(cvk_sqrt(check()), "sqrt")

    override fun exp(): Mat =
        nativeMat(cvk_exp(check()), "exp")

    override fun log(): Mat =
        nativeMat(cvk_log(check()), "log")

    override fun magnitude(y: Mat): Mat =
        nativeMat(cvk_magnitude(check(), y.checked), "magnitude")

    override fun phase(y: Mat, angleInDegrees: Boolean): Mat =
        nativeMat(cvk_phase(check(), y.checked, if (angleInDegrees) 1 else 0), "phase")

    override fun cartToPolar(y: Mat, angleInDegrees: Boolean): Pair<Mat, Mat> = memScoped {
        val magnitude = alloc<CPointerVar<cvk_mat_t>>()
        val angle = alloc<CPointerVar<cvk_mat_t>>()
        cvk_cart_to_polar(
            check(), y.checked, if (angleInDegrees) 1 else 0,
            magnitude.ptr, angle.ptr,
        )
        nativeMat(magnitude.value, "cartToPolar") to nativeMat(angle.value, "cartToPolar")
    }

    override fun polarToCart(angle: Mat, angleInDegrees: Boolean): Pair<Mat, Mat> = memScoped {
        val x = alloc<CPointerVar<cvk_mat_t>>()
        val y = alloc<CPointerVar<cvk_mat_t>>()
        cvk_polar_to_cart(
            check(), angle.checked, if (angleInDegrees) 1 else 0,
            x.ptr, y.ptr,
        )
        nativeMat(x.value, "polarToCart") to nativeMat(y.value, "polarToCart")
    }

    override fun patchNaNs(value: Double) {
        cvk_patch_nans(check(), value)
    }

    override val hasNonZero: Boolean
        get() = cvk_has_non_zero(check()) != 0

    override fun findNonZero(): Mat =
        nativeMat(cvk_find_non_zero(check()), "findNonZero")

    override fun sort(flags: Int): Mat =
        nativeMat(cvk_sort(check(), flags), "sort")

    override fun sortIdx(flags: Int): Mat =
        nativeMat(cvk_sort_idx(check(), flags), "sortIdx")

    override fun reduce(dim: Int, rtype: Int, dtype: Int): Mat =
        nativeMat(cvk_reduce(check(), dim, rtype, dtype), "reduce")

    override fun reduceArgMax(dim: Int): Mat =
        nativeMat(cvk_reduce_arg_max(check(), dim), "reduceArgMax")

    override fun reduceArgMin(dim: Int): Mat =
        nativeMat(cvk_reduce_arg_min(check(), dim), "reduceArgMin")

    override fun extractChannel(coi: Int): Mat =
        nativeMat(cvk_extract_channel(check(), coi), "extractChannel")

    override fun insertChannel(channel: Mat, coi: Int) {
        cvk_insert_channel(channel.checked, check(), coi)
    }

    override fun randu(low: Scalar, high: Scalar) {
        cvk_randu(check(), low.toCvk(), high.toCvk())
    }

    override fun randn(mean: Scalar, stddev: Scalar) {
        cvk_randn(check(), mean.toCvk(), stddev.toCvk())
    }

    override fun psnr(other: Mat, r: Double): Double =
        cvk_psnr(check(), other.checked, r)

    override fun dft(flags: Int): Mat =
        nativeMat(cvk_dft(check(), flags), "dft")

    override fun idft(flags: Int): Mat =
        nativeMat(cvk_idft(check(), flags), "idft")

    override fun dct(flags: Int): Mat =
        nativeMat(cvk_dct(check(), flags), "dct")

    override fun idct(flags: Int): Mat =
        nativeMat(cvk_idct(check(), flags), "idct")

    override fun mulSpectrums(other: Mat, conjugate: Boolean, dftRows: Boolean): Mat =
        nativeMat(
            cvk_mul_spectrums(
                check(),
                other.checked,
                if (conjugate) 1 else 0,
                if (dftRows) 1 else 0,
            ),
            "mulSpectrums",
        )

    override fun divSpectrums(other: Mat, conjugate: Boolean): Mat =
        nativeMat(
            cvk_div_spectrums(check(), other.checked, if (conjugate) 1 else 0),
            "divSpectrums",
        )

    override fun gemm(other: Mat, alpha: Double, c: Mat?, gamma: Double): Mat =
        nativeMat(cvk_gemm(check(), other.checked, alpha, c?.checked, gamma), "gemm")

    override fun eigen(): Pair<Mat, Mat> = memScoped {
        val values = alloc<CPointerVar<cvk_mat_t>>()
        val vectors = alloc<CPointerVar<cvk_mat_t>>()
        cvk_eigen(check(), values.ptr, vectors.ptr)
        nativeMat(values.value, "eigen") to nativeMat(vectors.value, "eigen")
    }

    // ---- imgproc: filters --------------------------------------------------------

    override fun blur(kernelWidth: Int, kernelHeight: Int): Mat =
        nativeMat(cvk_blur(check(), kernelWidth, kernelHeight), "blur")

    override fun boxFilter(kernelWidth: Int, kernelHeight: Int, ddepth: Int, normalize: Boolean): Mat =
        nativeMat(
            cvk_box_filter(check(), ddepth, kernelWidth, kernelHeight, if (normalize) 1 else 0),
            "boxFilter",
        )

    override fun sqrBoxFilter(kernelWidth: Int, kernelHeight: Int, ddepth: Int): Mat =
        nativeMat(cvk_sqr_box_filter(check(), ddepth, kernelWidth, kernelHeight), "sqrBoxFilter")

    override fun bilateralFilter(d: Int, sigmaColor: Double, sigmaSpace: Double): Mat =
        nativeMat(cvk_bilateral_filter(check(), d, sigmaColor, sigmaSpace), "bilateralFilter")

    override fun stackBlur(kernelSize: Int): Mat =
        nativeMat(cvk_stack_blur(check(), kernelSize), "stackBlur")

    override fun erode(kernel: Mat?, iterations: Int): Mat =
        nativeMat(cvk_erode(check(), kernel?.checked, iterations), "erode")

    override fun dilate(kernel: Mat?, iterations: Int): Mat =
        nativeMat(cvk_dilate(check(), kernel?.checked, iterations), "dilate")

    override fun morphologyEx(op: Int, kernel: Mat?, iterations: Int): Mat =
        nativeMat(cvk_morphology_ex(check(), op, kernel?.checked, iterations), "morphologyEx")

    override fun filter2D(kernel: Mat, ddepth: Int, delta: Double): Mat =
        nativeMat(cvk_filter_2d(check(), kernel.checked, ddepth, delta), "filter2D")

    override fun pyrDown(): Mat =
        nativeMat(cvk_pyr_down(check()), "pyrDown")

    override fun pyrUp(): Mat =
        nativeMat(cvk_pyr_up(check()), "pyrUp")

    // ---- imgproc: geometry ---------------------------------------------------------

    override fun warpAffine(m: Mat, width: Int, height: Int, flags: Int): Mat =
        nativeMat(cvk_warp_affine(check(), m.checked, width, height, flags), "warpAffine")

    override fun warpPerspective(m: Mat, width: Int, height: Int, flags: Int): Mat =
        nativeMat(cvk_warp_perspective(check(), m.checked, width, height, flags), "warpPerspective")

    override fun remap(map1: Mat, map2: Mat, interpolation: Int): Mat =
        nativeMat(cvk_remap(check(), map1.checked, map2.checked, interpolation), "remap")

    override fun warpPolar(
        radius: Int,
        centerX: Double,
        centerY: Double,
        maxRadius: Double,
        flags: Int,
    ): Mat = nativeMat(
        cvk_warp_polar(check(), radius, centerX, centerY, maxRadius, flags),
        "warpPolar",
    )

    override fun undistort(cameraMatrix: Mat, distCoeffs: Mat): Mat =
        nativeMat(cvk_undistort(check(), cameraMatrix.checked, distCoeffs.checked), "undistort")

    override fun getRectSubPix(width: Int, height: Int, centerX: Double, centerY: Double): Mat =
        nativeMat(cvk_get_rect_sub_pix(check(), width, height, centerX, centerY), "getRectSubPix")

    // ---- imgproc: color / histogram --------------------------------------------------

    override fun demosaicing(code: Int): Mat =
        nativeMat(cvk_demosaicing(check(), code), "demosaicing")

    override fun applyColorMap(colormap: Int): Mat =
        nativeMat(cvk_apply_colormap(check(), colormap), "applyColorMap")

    override fun applyColorMap(userColor: Mat): Mat =
        nativeMat(cvk_apply_colormap_user(check(), userColor.checked), "applyColorMap(user)")

    override fun calcHist(channel: Int, histSize: Int, minValue: Float, maxValue: Float): Mat =
        nativeMat(cvk_calc_hist(check(), channel, histSize, minValue, maxValue), "calcHist")

    override fun calcBackProject(hist: Mat, channel: Int, minValue: Float, maxValue: Float): Mat =
        nativeMat(
            cvk_calc_back_project(check(), channel, hist.checked, minValue, maxValue),
            "calcBackProject",
        )

    override fun equalizeHist(): Mat =
        nativeMat(cvk_equalize_hist(check()), "equalizeHist")

    override fun matchShapes(other: Mat, method: Int): Double =
        cvk_match_shapes(check(), other.checked, method)

    override fun moments(binaryImage: Boolean): Moments = memScoped {
        val out = allocArray<DoubleVar>(10)
        cvk_moments(check(), if (binaryImage) 1 else 0, out)
        Moments(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9])
    }

    // ---- imgproc: segmentation / features -----------------------------------------------

    override fun floodFill(
        seedX: Int,
        seedY: Int,
        newValue: Scalar,
        loDiff: Scalar,
        upDiff: Scalar,
        flags: Int,
    ): Int = cvk_flood_fill(
        check(), seedX, seedY,
        newValue.toCvk(), loDiff.toCvk(), upDiff.toCvk(), flags,
    )

    override fun watershed(markers: Mat) {
        cvk_watershed(check(), markers.checked)
    }

    override fun matchTemplate(templ: Mat, method: Int): Mat =
        nativeMat(cvk_match_template(check(), templ.checked, method), "matchTemplate")

    override fun cornerHarris(blockSize: Int, ksize: Int, k: Double): Mat =
        nativeMat(cvk_corner_harris(check(), blockSize, ksize, k), "cornerHarris")

    override fun cornerMinEigenVal(blockSize: Int, ksize: Int): Mat =
        nativeMat(cvk_corner_min_eigen_val(check(), blockSize, ksize), "cornerMinEigenVal")

    override fun goodFeaturesToTrack(
        maxCorners: Int,
        qualityLevel: Double,
        minDistance: Double,
        blockSize: Int,
        useHarrisDetector: Boolean,
        k: Double,
    ): Mat = nativeMat(
        cvk_good_features_to_track(
            check(),
            maxCorners,
            qualityLevel,
            minDistance,
            blockSize,
            if (useHarrisDetector) 1 else 0,
            k,
        ),
        "goodFeaturesToTrack",
    )

    override fun distanceTransform(distanceType: Int, maskSize: Int): Mat =
        nativeMat(cvk_distance_transform(check(), distanceType, maskSize), "distanceTransform")

    override fun integral(sdepth: Int): Mat =
        nativeMat(cvk_integral(check(), sdepth), "integral")

    override fun connectedComponents(connectivity: Int, ltype: Int): Pair<Int, Mat> = memScoped {
        val labels = alloc<CPointerVar<cvk_mat_t>>()
        val count = cvk_connected_components(check(), labels.ptr, connectivity, ltype)
        count to nativeMat(labels.value, "connectedComponents")
    }

    override fun connectedComponentsWithStats(connectivity: Int, ltype: Int): Components = memScoped {
        val labels = alloc<CPointerVar<cvk_mat_t>>()
        val stats = alloc<CPointerVar<cvk_mat_t>>()
        val centroids = alloc<CPointerVar<cvk_mat_t>>()
        val count = cvk_connected_components_with_stats(
            check(), labels.ptr, stats.ptr, centroids.ptr, connectivity, ltype,
        )
        Components(
            count = count,
            labels = nativeMat(labels.value, "connectedComponentsWithStats"),
            stats = nativeMat(stats.value, "connectedComponentsWithStats"),
            centroids = nativeMat(centroids.value, "connectedComponentsWithStats"),
        )
    }

    override fun pyrMeanShiftFiltering(sp: Double, sr: Double, maxLevel: Int): Mat =
        nativeMat(cvk_pyr_mean_shift_filtering(check(), sp, sr, maxLevel), "pyrMeanShiftFiltering")

    override fun thresholdWithMask(thresh: Double, maxVal: Double, type: Int, mask: Mat?): Pair<Double, Mat> {
        // dst must be a caller-provided writable matrix; work on a clone so this
        // stays untouched and the clone keeps the result alive.
        val cloned = clone() as NativeMat
        val computed = cvk_threshold_with_mask(
            check(), mask?.checked, cloned.check(), thresh, maxVal, type,
        )
        return computed to cloned
    }

    // ---- imgproc: hough / accumulators ----------------------------------------------------

    override fun houghLines(rho: Double, theta: Double, threshold: Int, srn: Double, stn: Double): Mat =
        nativeMat(cvk_hough_lines(check(), rho, theta, threshold, srn, stn), "houghLines")

    override fun houghLinesP(
        rho: Double,
        theta: Double,
        threshold: Int,
        minLineLength: Double,
        maxLineGap: Double,
    ): Mat = nativeMat(
        cvk_hough_lines_p(check(), rho, theta, threshold, minLineLength, maxLineGap),
        "houghLinesP",
    )

    override fun houghCircles(
        dp: Double,
        minDist: Double,
        param1: Double,
        param2: Double,
        minRadius: Int,
        maxRadius: Int,
        method: Int,
    ): Mat = nativeMat(
        cvk_hough_circles(check(), method, dp, minDist, param1, param2, minRadius, maxRadius),
        "houghCircles",
    )

    override fun accumulate(src: Mat) {
        cvk_accumulate(src.checked, check())
    }

    override fun accumulateSquare(src: Mat) {
        cvk_accumulate_square(src.checked, check())
    }

    override fun accumulateProduct(a: Mat, b: Mat) {
        cvk_accumulate_product(a.checked, b.checked, check())
    }

    override fun accumulateWeighted(src: Mat, alpha: Double) {
        cvk_accumulate_weighted(src.checked, check(), alpha)
    }

    // ---- imgproc: contours -----------------------------------------------------------------

    override fun findContours(mode: Int, method: Int): List<List<Point>> = memScoped {
        val lengthVar = alloc<size_tVar>()
        val buffer = cvk_find_contours(check(), mode, method, lengthVar.ptr)
            ?: throw OpenCVException("findContours", lastNativeError())
        try {
            ContourCodec.decode(buffer.readBytes(lengthVar.value.toInt()))
        } finally {
            cvk_free_buffer(buffer)
        }
    }

    override fun drawContours(contours: List<List<Point>>, color: Scalar, contourIndex: Int, thickness: Int) {
        val flat = ContourCodec.encode(contours)
        if (flat.isEmpty()) return
        flat.asUByteArray().usePinned { pinned ->
            cvk_draw_contours(
                check(), pinned.addressOf(0), flat.size.convert<size_t>(),
                contourIndex, color.toCvk(), thickness,
            )
        }
    }

    // ---- imgproc: drawing (in-place) --------------------------------------------------------

    override fun arrowedLine(from: Point, to: Point, color: Scalar, thickness: Int) {
        cvk_arrowed_line(check(), from.x, from.y, to.x, to.y, color.toCvk(), thickness)
    }

    override fun drawMarker(pos: Point, color: Scalar, markerType: Int, size: Int, thickness: Int) {
        cvk_draw_marker(check(), pos.x, pos.y, markerType, size, color.toCvk(), thickness)
    }

    override fun ellipse(
        center: Point,
        axes: Size,
        angle: Double,
        startAngle: Double,
        endAngle: Double,
        color: Scalar,
        thickness: Int,
    ) {
        cvk_ellipse(
            check(), center.x, center.y, axes.width, axes.height,
            angle, startAngle, endAngle, color.toCvk(), thickness,
        )
    }

    override fun fillPoly(polygons: List<List<Point>>, color: Scalar, thickness: Int) {
        val flat = ContourCodec.encode(polygons)
        if (flat.isEmpty()) return
        flat.asUByteArray().usePinned { pinned ->
            cvk_fill_poly(
                check(), pinned.addressOf(0), flat.size.convert<size_t>(),
                color.toCvk(), thickness,
            )
        }
    }

    override fun polylines(polylines: List<List<Point>>, closed: Boolean, color: Scalar, thickness: Int) {
        val flat = ContourCodec.encode(polylines)
        if (flat.isEmpty()) return
        flat.asUByteArray().usePinned { pinned ->
            cvk_polylines(
                check(), pinned.addressOf(0), flat.size.convert<size_t>(),
                if (closed) 1 else 0, color.toCvk(), thickness,
            )
        }
    }

    // ---- org.opencv.core.Mat parity ------------------------------------------

    override fun adjustROI(dtop: Int, dbottom: Int, dleft: Int, dright: Int): Mat =
        nativeMat(cvk_mat_adjust_roi(check(), dtop, dbottom, dleft, dright), "adjustROI")

    override fun locateROI(): Pair<Point, Size> = memScoped {
        val out = allocArray<IntVar>(4)
        cvk_mat_locate_roi(check(), out)
        Point(out[0], out[1]) to Size(out[2], out[3])
    }

    override val isContinuous: Boolean
        get() = cvk_mat_is_continuous(check()) != 0

    override val isSubmatrix: Boolean
        get() = cvk_mat_is_submatrix(check()) != 0

    override infix fun cross(other: Mat): Mat =
        nativeMat(cvk_mat_cross(check(), other.checked), "cross")

    override fun dump(): String {
        val text = cvk_mat_dump(check())
            ?: throw OpenCVException("dump", lastNativeError())
        return text.toKString()
    }

    override fun put(row: Int, col: Int, values: DoubleArray): Int {
        if (values.isEmpty()) return 0
        return values.usePinned { pinned ->
            cvk_mat_put_values(
                check(), row, col, pinned.addressOf(0),
                values.size.convert<size_t>(),
            ).toInt()
        }
    }

    override fun get(row: Int, col: Int, values: DoubleArray): Int {
        if (values.isEmpty()) return 0
        return values.usePinned { pinned ->
            cvk_mat_get_values(
                check(), row, col, pinned.addressOf(0),
                values.size.convert<size_t>(),
            ).toInt()
        }
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

/** Unwraps a platform Mat into its raw cvk handle. */
private fun Mat.nativeHandle(): CPointer<cvk_mat_t> =
    (this as? NativeMat)?.check()
        ?: throw IllegalArgumentException("mat belongs to another platform backend")

private fun pinnedContour(contour: List<Point>, body: (data: CPointer<UByteVar>, len: Int) -> Unit) {
    val flat = ContourCodec.encode(listOf(contour))
    if (flat.isEmpty()) return
    flat.asUByteArray().usePinned { pinned ->
        body(pinned.addressOf(0), flat.size)
    }
}

// =========================================================================
// kernels / transforms
// =========================================================================

actual fun getStructuringElement(shape: Int, width: Int, height: Int): Mat =
    nativeMat(cvk_get_structuring_element(shape, width, height), "getStructuringElement")

actual fun getGaussianKernel(ksize: Int, sigma: Double): Mat =
    nativeMat(cvk_get_gaussian_kernel(ksize, sigma), "getGaussianKernel")

actual fun getAffineTransform(src: List<Point>, dst: List<Point>): Mat {
    require(src.size == 3 && dst.size == 3) { "an affine transform needs exactly 3 source and destination points" }
    return nativeMat(
        cvk_get_affine_transform(
            src[0].x.toDouble(), src[0].y.toDouble(),
            src[1].x.toDouble(), src[1].y.toDouble(),
            src[2].x.toDouble(), src[2].y.toDouble(),
            dst[0].x.toDouble(), dst[0].y.toDouble(),
            dst[1].x.toDouble(), dst[1].y.toDouble(),
            dst[2].x.toDouble(), dst[2].y.toDouble(),
        ),
        "getAffineTransform",
    )
}

actual fun invertAffineTransform(transform: Mat): Mat =
    nativeMat(cvk_invert_affine_transform(transform.nativeHandle()), "invertAffineTransform")

actual fun getPerspectiveTransform(src: List<Point>, dst: List<Point>): Mat {
    require(src.size == 4 && dst.size == 4) { "a perspective transform needs exactly 4 source and destination points" }
    return nativeMat(
        cvk_get_perspective_transform(
            src[0].x.toDouble(), src[0].y.toDouble(),
            src[1].x.toDouble(), src[1].y.toDouble(),
            src[2].x.toDouble(), src[2].y.toDouble(),
            src[3].x.toDouble(), src[3].y.toDouble(),
            dst[0].x.toDouble(), dst[0].y.toDouble(),
            dst[1].x.toDouble(), dst[1].y.toDouble(),
            dst[2].x.toDouble(), dst[2].y.toDouble(),
            dst[3].x.toDouble(), dst[3].y.toDouble(),
        ),
        "getPerspectiveTransform",
    )
}

actual fun getRotationMatrix2D(center: Point, angle: Double, scale: Double): Mat =
    nativeMat(
        cvk_get_rotation_matrix_2d(center.x.toDouble(), center.y.toDouble(), angle, scale),
        "getRotationMatrix2D",
    )

actual fun hanningWindow(width: Int, height: Int, type: Int): Mat =
    nativeMat(cvk_create_hanning_window(width, height, type), "hanningWindow")

actual fun merge(channels: List<Mat>): Mat {
    require(channels.isNotEmpty()) { "merge needs at least one channel" }
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(channels.size)
        channels.forEachIndexed { index, channel ->
            handles[index] = channel.nativeHandle()
        }
        nativeMat(cvk_merge(handles, channels.size), "merge")
    }
}

actual val opencvNumThreads: Int
    get() = cvk_num_threads()

actual fun setNumThreads(count: Int) {
    cvk_set_num_threads(count)
}

actual val opencvBuildInformation: String
    get() = cvk_build_information()!!.toKString()

actual fun setRNGSeed(seed: Long) {
    cvk_set_rng_seed(seed.toULong())
}

// =========================================================================
// codecs / environment
// =========================================================================

actual fun imcount(path: String): Int =
    cvk_imcount(path)

actual fun haveImageReader(path: String): Boolean =
    cvk_have_image_reader(path) != 0

actual fun haveImageWriter(path: String): Boolean =
    cvk_have_image_writer(path) != 0

actual fun imencodeParams(ext: String, mat: Mat, params: List<Int>): ByteArray {
    if (params.isEmpty()) return imencode(ext, mat)
    val target = mat as? NativeMat
        ?: throw IllegalArgumentException("mat belongs to another platform backend")
    return memScoped {
        val lengthVar = alloc<size_tVar>()
        val buffer = params.toIntArray().usePinned { pinned ->
            cvk_imencode_params(
                normalizeImageExtension(ext), target.check(),
                pinned.addressOf(0), params.size.convert<size_t>(), lengthVar.ptr,
            )
        } ?: throw OpenCVException("imencodeParams", lastNativeError())
        try {
            buffer.readBytes(lengthVar.value.toInt())
        } finally {
            cvk_free_buffer(buffer)
        }
    }
}

actual fun imwriteParams(path: String, mat: Mat, params: List<Int>): Boolean {
    val target = mat as? NativeMat
        ?: throw IllegalArgumentException("mat belongs to another platform backend")
    if (params.isEmpty()) return cvk_imwrite(path, target.check()) != 0
    return params.toIntArray().usePinned { pinned ->
        cvk_imwrite_params(
            path, target.check(), pinned.addressOf(0), params.size.convert<size_t>(),
        ) != 0
    }
}

actual fun getOptimalDftSize(size: Int): Int =
    cvk_get_optimal_dft_size(size)

// =========================================================================
// contour geometry over the wire format
// =========================================================================

actual fun approxPolyDP(contour: List<Point>, epsilon: Double, closed: Boolean): List<Point> {
    if (contour.isEmpty()) return emptyList()
    val flat = ContourCodec.encode(listOf(contour))
    memScoped {
        val lengthVar = alloc<size_tVar>()
        val buffer = flat.asUByteArray().usePinned { pinned ->
            cvk_approx_poly_dp(
                pinned.addressOf(0), flat.size.convert<size_t>(),
                epsilon, if (closed) 1 else 0, lengthVar.ptr,
            )
        } ?: throw OpenCVException("approxPolyDP", lastNativeError())
        try {
            return ContourCodec.decode(buffer.readBytes(lengthVar.value.toInt()))
                .firstOrNull() ?: emptyList()
        } finally {
            cvk_free_buffer(buffer)
        }
    }
}

actual fun minAreaRect(contour: List<Point>): RotatedRect = memScoped {
    val out = allocArray<DoubleVar>(5)
    pinnedContour(contour) { data, len ->
        cvk_min_area_rect(data, len.convert<size_t>(), out)
    }
    RotatedRect(out[0], out[1], out[2], out[3], out[4])
}

actual fun minEnclosingCircle(contour: List<Point>): Circle = memScoped {
    val out = allocArray<DoubleVar>(3)
    pinnedContour(contour) { data, len ->
        cvk_min_enclosing_circle(data, len.convert<size_t>(), out)
    }
    Circle(out[0], out[1], out[2])
}

internal actual fun contourAreaNative(data: ByteArray): Double =
    data.asUByteArray().usePinned { pinned ->
        cvk_contour_area(pinned.addressOf(0), data.size.convert<size_t>())
    }

internal actual fun arcLengthNative(data: ByteArray, closed: Boolean): Double =
    data.asUByteArray().usePinned { pinned ->
        cvk_arc_length(pinned.addressOf(0), data.size.convert<size_t>(), if (closed) 1 else 0)
    }

internal actual fun contourRect(contour: List<Point>): Rect = memScoped {
    val out = allocArray<IntVar>(4)
    pinnedContour(contour) { data, len ->
        cvk_bounding_rect(data, len.convert<size_t>(), out)
    }
    Rect(out[0], out[1], out[2], out[3])
}

// =========================================================================
// CLAHE
// =========================================================================

internal class NativeClahe(
    @Volatile private var raw: CPointer<cvk_clahe_t>?,
) : CLAHE {

    private fun check(): CPointer<cvk_clahe_t> =
        raw ?: throw IllegalStateException("CLAHE is closed")

    override fun apply(src: Mat): Mat =
        nativeMat(cvk_clahe_apply(check(), src.nativeHandle()), "clahe.apply")

    override fun setClipLimit(clipLimit: Double) {
        cvk_clahe_set_clip_limit(check(), clipLimit)
    }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_clahe_release(handle)
    }
}

actual fun createCLAHE(clipLimit: Double, tileGridSize: Size): CLAHE =
    NativeClahe(
        cvk_clahe_create(clipLimit, tileGridSize.width, tileGridSize.height)
            ?: throw OpenCVException("createCLAHE", lastNativeError()),
    )

// =========================================================================
// org.opencv.core parity: clustering / decomposition
// =========================================================================

actual fun kmeans(
    data: Mat,
    k: Int,
    criteria: TermCriteria,
    attempts: Int,
    flags: Int,
): KmeansResult {
    // Labels are created here so callers receive a fresh Nx1 CV_32S matrix they
    // own; centers are allocated by the shim. data is never closed.
    val labels = zeros(data.rows, 1, CV_32S)
    try {
        memScoped {
            val centers = alloc<CPointerVar<cvk_mat_t>>()
            val compactness = allocArray<DoubleVar>(1)
            cvk_kmeans(
                data.nativeHandle(), k, labels.nativeHandle(),
                criteria.type, criteria.maxCount, criteria.epsilon,
                attempts, flags, centers.ptr, compactness,
            )
            return KmeansResult(
                compactness[0], labels, nativeMat(centers.value, "kmeans"),
            )
        }
    } catch (t: Throwable) {
        labels.close()
        throw t
    }
}

actual fun svDecomp(src: Mat, flags: Int): Svd = memScoped {
    val w = alloc<CPointerVar<cvk_mat_t>>()
    val u = alloc<CPointerVar<cvk_mat_t>>()
    val vt = alloc<CPointerVar<cvk_mat_t>>()
    cvk_svd_decomp(src.nativeHandle(), w.ptr, u.ptr, vt.ptr, flags)
    Svd(
        w = nativeMat(w.value, "svDecomp"),
        u = nativeMat(u.value, "svDecomp"),
        vt = nativeMat(vt.value, "svDecomp"),
    )
}

actual fun svdBackSubst(w: Mat, u: Mat, vt: Mat, b: Mat): Mat = memScoped {
    val dst = alloc<CPointerVar<cvk_mat_t>>()
    cvk_svd_backsubst(
        w.nativeHandle(), u.nativeHandle(), vt.nativeHandle(),
        b.nativeHandle(), dst.ptr,
    )
    nativeMat(dst.value, "svdBackSubst")
}

actual fun pcaCompute(data: Mat, maxComponents: Int): Pca = memScoped {
    val mean = alloc<CPointerVar<cvk_mat_t>>()
    val vectors = alloc<CPointerVar<cvk_mat_t>>()
    cvk_pca_compute(data.nativeHandle(), mean.ptr, vectors.ptr, maxComponents)
    Pca(nativeMat(mean.value, "pcaCompute"), nativeMat(vectors.value, "pcaCompute"))
}

actual fun pcaComputeVariance(data: Mat, retainedVariance: Double): Pca = memScoped {
    val mean = alloc<CPointerVar<cvk_mat_t>>()
    val vectors = alloc<CPointerVar<cvk_mat_t>>()
    cvk_pca_compute_variance(data.nativeHandle(), mean.ptr, vectors.ptr, retainedVariance)
    Pca(nativeMat(mean.value, "pcaComputeVariance"), nativeMat(vectors.value, "pcaComputeVariance"))
}

actual fun pcaProject(data: Mat, mean: Mat, eigenvectors: Mat): Mat = memScoped {
    val result = alloc<CPointerVar<cvk_mat_t>>()
    cvk_pca_project(data.nativeHandle(), mean.nativeHandle(), eigenvectors.nativeHandle(), result.ptr)
    nativeMat(result.value, "pcaProject")
}

actual fun pcaBackProject(data: Mat, mean: Mat, eigenvectors: Mat): Mat = memScoped {
    val result = alloc<CPointerVar<cvk_mat_t>>()
    cvk_pca_backproject(data.nativeHandle(), mean.nativeHandle(), eigenvectors.nativeHandle(), result.ptr)
    nativeMat(result.value, "pcaBackProject")
}

actual fun mahalanobis(v1: Mat, v2: Mat, icovar: Mat): Double =
    cvk_mahalanobis(v1.nativeHandle(), v2.nativeHandle(), icovar.nativeHandle())

// =========================================================================
// org.opencv.imgproc parity
// =========================================================================

actual fun cornerSubPix(
    image: Mat,
    corners: List<Point>,
    winSize: Size,
    zeroZone: Size,
    criteria: TermCriteria,
): List<Point> {
    if (corners.isEmpty()) return emptyList()
    // Corners travel as a single-contour wire buffer; refined points come back
    // in the same flat layout.
    val flat = ContourCodec.encode(listOf(corners))
    return memScoped {
        val outLen = alloc<size_tVar>()
        val buffer = flat.asUByteArray().usePinned { pinned ->
            cvk_corner_sub_pix(
                image.nativeHandle(), pinned.addressOf(0), flat.size.convert<size_t>(),
                winSize.width, winSize.height, zeroZone.width, zeroZone.height,
                criteria.type, criteria.maxCount, criteria.epsilon, outLen.ptr,
            ) ?: throw OpenCVException("cornerSubPix", lastNativeError())
        }
        try {
            ContourCodec.decode(buffer.readBytes(outLen.value.toInt())).firstOrNull() ?: emptyList()
        } finally {
            cvk_free_buffer(buffer)
        }
    }
}

actual fun emd(signature1: Mat, signature2: Mat, distType: Int): Double =
    cvk_emd(signature1.nativeHandle(), signature2.nativeHandle(), distType).toDouble()

actual fun grabCut(image: Mat, mask: Mat, rect: Rect?, iterations: Int, mode: Int) {
    // Stateless convenience wrapper: fresh 1x65 CV_64F models per call
    // (13 * 5 GMM components), freed after.
    val bgdModel = mat(1, 65, CV_64F)
    val fgdModel = mat(1, 65, CV_64F)
    try {
        cvk_grab_cut(
            image.nativeHandle(), mask.nativeHandle(),
            rect?.x ?: 0, rect?.y ?: 0, rect?.width ?: 0, rect?.height ?: 0,
            bgdModel.nativeHandle(), fgdModel.nativeHandle(), iterations, mode,
        )
    } finally {
        bgdModel.close()
        fgdModel.close()
    }
}

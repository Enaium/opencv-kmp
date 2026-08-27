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

import cvk.cvk_stereo_bm_get_pre_filter_cap
import cvk.cvk_stereo_bm_get_pre_filter_size
import cvk.cvk_stereo_bm_get_pre_filter_type
import cvk.cvk_stereo_bm_get_roi1
import cvk.cvk_stereo_bm_get_roi2
import cvk.cvk_stereo_bm_get_smaller_block_size
import cvk.cvk_stereo_bm_get_texture_threshold
import cvk.cvk_stereo_bm_get_uniqueness_ratio
import cvk.cvk_stereo_bm_set_pre_filter_cap
import cvk.cvk_stereo_bm_set_pre_filter_size
import cvk.cvk_stereo_bm_set_pre_filter_type
import cvk.cvk_stereo_bm_set_roi1
import cvk.cvk_stereo_bm_set_roi2
import cvk.cvk_stereo_bm_set_smaller_block_size
import cvk.cvk_stereo_bm_set_texture_threshold
import cvk.cvk_stereo_bm_set_uniqueness_ratio
import cvk.cvk_stereo_get_valid_disparity_roi
import cvk.cvk_stereo_sgbm_get_pre_filter_cap
import cvk.cvk_stereo_sgbm_set_pre_filter_cap
import cvk.cvk_stereo_sgbm_get_uniqueness_ratio
import cvk.cvk_stereo_sgbm_set_uniqueness_ratio
import cvk.cvk_stereo_sgbm_get_p1
import cvk.cvk_stereo_sgbm_set_p1
import cvk.cvk_stereo_sgbm_get_p2
import cvk.cvk_stereo_sgbm_set_p2
import cvk.cvk_stereo_sgbm_get_mode
import cvk.cvk_stereo_sgbm_set_mode
import cvk.cvk_stereo_reproject_image_to_3d
import cvk.cvk_stereo_validate_disparity
import cvk.cvk_stereo_filter_speckles
import cvk.cvk_stereo_filter_speckles_buf
import cvk.cvk_stereo_fisheye_rectify
import cvk.cvk_stereo_matcher_t
import cvk.cvk_stereo_rectify
import cvk.cvk_stereo_rectify_uncalibrated
import cvk.cvk_stereo_matcher_clear
import cvk.cvk_stereo_matcher_compute
import cvk.cvk_stereo_matcher_create_bm
import cvk.cvk_stereo_matcher_create_sgbm
import cvk.cvk_stereo_matcher_empty
import cvk.cvk_stereo_matcher_get_block_size
import cvk.cvk_stereo_matcher_get_default_name
import cvk.cvk_stereo_matcher_get_disp12_max_diff
import cvk.cvk_stereo_matcher_get_min_disparity
import cvk.cvk_stereo_matcher_get_num_disparities
import cvk.cvk_stereo_matcher_get_speckle_range
import cvk.cvk_stereo_matcher_get_speckle_window_size
import cvk.cvk_stereo_matcher_release
import cvk.cvk_stereo_matcher_save
import cvk.cvk_stereo_matcher_set_block_size
import cvk.cvk_stereo_matcher_set_disp12_max_diff
import cvk.cvk_stereo_matcher_set_min_disparity
import cvk.cvk_stereo_matcher_set_num_disparities
import cvk.cvk_stereo_matcher_set_speckle_range
import cvk.cvk_stereo_matcher_set_speckle_window_size
import cvk.cvk_last_error
import cvk.cvk_mat_t
import cvk.cvk_rect_t
import kotlin.concurrent.Volatile
import kotlinx.cinterop.ptr
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.CValue
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.cValue
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.useContents
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

private fun Rect.toCvkRect(): CValue<cvk_rect_t> {
    val region = this
    return cValue {
        x = region.x
        y = region.y
        width = region.width
        height = region.height
    }
}

// =========================================================================
// StereoMatcher
// =========================================================================

internal open class NativeStereoMatcher(
    @Volatile private var raw: CPointer<cvk_stereo_matcher_t>?,
) : StereoMatcher {

    protected fun check(): CPointer<cvk_stereo_matcher_t> =
        raw ?: throw IllegalStateException("StereoMatcher is closed")

    override fun compute(left: Mat, right: Mat): Mat =
        nativeMat(
            cvk_stereo_matcher_compute(check(), left.nativeHandle(), right.nativeHandle()),
            "StereoMatcher.compute",
        )

    override var minDisparity: Int
        get() = cvk_stereo_matcher_get_min_disparity(check())
        set(value) = cvk_stereo_matcher_set_min_disparity(check(), value)

    override var numDisparities: Int
        get() = cvk_stereo_matcher_get_num_disparities(check())
        set(value) = cvk_stereo_matcher_set_num_disparities(check(), value)

    override var blockSize: Int
        get() = cvk_stereo_matcher_get_block_size(check())
        set(value) = cvk_stereo_matcher_set_block_size(check(), value)

    override var speckleWindowSize: Int
        get() = cvk_stereo_matcher_get_speckle_window_size(check())
        set(value) = cvk_stereo_matcher_set_speckle_window_size(check(), value)

    override var speckleRange: Int
        get() = cvk_stereo_matcher_get_speckle_range(check())
        set(value) = cvk_stereo_matcher_set_speckle_range(check(), value)

    override var disp12MaxDiff: Int
        get() = cvk_stereo_matcher_get_disp12_max_diff(check())
        set(value) = cvk_stereo_matcher_set_disp12_max_diff(check(), value)

    override fun clear() {
        cvk_stereo_matcher_clear(check())
    }

    override fun empty(): Boolean = cvk_stereo_matcher_empty(check()) != 0

    override fun save(filename: String) {
        cvk_stereo_matcher_save(check(), filename)
    }

    override fun getDefaultName(): String =
        cvk_stereo_matcher_get_default_name(check())?.toKString().orEmpty()

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_stereo_matcher_release(handle)
    }
}

internal class NativeStereoBm(raw: CPointer<cvk_stereo_matcher_t>?) : NativeStereoMatcher(raw), StereoBM {

    override var preFilterType: Int
        get() = cvk_stereo_bm_get_pre_filter_type(check())
        set(value) = cvk_stereo_bm_set_pre_filter_type(check(), value)

    override var preFilterSize: Int
        get() = cvk_stereo_bm_get_pre_filter_size(check())
        set(value) = cvk_stereo_bm_set_pre_filter_size(check(), value)

    override var preFilterCap: Int
        get() = cvk_stereo_bm_get_pre_filter_cap(check())
        set(value) = cvk_stereo_bm_set_pre_filter_cap(check(), value)

    override var textureThreshold: Int
        get() = cvk_stereo_bm_get_texture_threshold(check())
        set(value) = cvk_stereo_bm_set_texture_threshold(check(), value)

    override var uniquenessRatio: Int
        get() = cvk_stereo_bm_get_uniqueness_ratio(check())
        set(value) = cvk_stereo_bm_set_uniqueness_ratio(check(), value)

    override var smallerBlockSize: Int
        get() = cvk_stereo_bm_get_smaller_block_size(check())
        set(value) = cvk_stereo_bm_set_smaller_block_size(check(), value)

    override var roi1: Rect
        get() = cvk_stereo_bm_get_roi1(check()).useContents { Rect(x, y, width, height) }
        set(value) = cvk_stereo_bm_set_roi1(check(), value.toCvkRect())

    override var roi2: Rect
        get() = cvk_stereo_bm_get_roi2(check()).useContents { Rect(x, y, width, height) }
        set(value) = cvk_stereo_bm_set_roi2(check(), value.toCvkRect())
}

internal class NativeStereoSgbm(raw: CPointer<cvk_stereo_matcher_t>?) : NativeStereoMatcher(raw), StereoSGBM {

    override var preFilterCap: Int
        get() = cvk_stereo_sgbm_get_pre_filter_cap(check())
        set(value) = cvk_stereo_sgbm_set_pre_filter_cap(check(), value)

    override var uniquenessRatio: Int
        get() = cvk_stereo_sgbm_get_uniqueness_ratio(check())
        set(value) = cvk_stereo_sgbm_set_uniqueness_ratio(check(), value)

    override var p1: Int
        get() = cvk_stereo_sgbm_get_p1(check())
        set(value) = cvk_stereo_sgbm_set_p1(check(), value)

    override var p2: Int
        get() = cvk_stereo_sgbm_get_p2(check())
        set(value) = cvk_stereo_sgbm_set_p2(check(), value)

    override var mode: Int
        get() = cvk_stereo_sgbm_get_mode(check())
        set(value) = cvk_stereo_sgbm_set_mode(check(), value)
}

actual fun stereoBMCreate(numDisparities: Int, blockSize: Int): StereoBM =
    NativeStereoBm(cvk_stereo_matcher_create_bm(numDisparities, blockSize)
        ?: throw OpenCVException("stereoBMCreate", lastNativeError()))

actual fun stereoSGBMCreate(
    minDisparity: Int,
    numDisparities: Int,
    blockSize: Int,
    p1: Int,
    p2: Int,
    disp12MaxDiff: Int,
    preFilterCap: Int,
    uniquenessRatio: Int,
    speckleWindowSize: Int,
    speckleRange: Int,
    mode: Int,
): StereoSGBM =
    NativeStereoSgbm(cvk_stereo_matcher_create_sgbm(
        minDisparity, numDisparities, blockSize, p1, p2, disp12MaxDiff, preFilterCap,
        uniquenessRatio, speckleWindowSize, speckleRange, mode,
    ) ?: throw OpenCVException("stereoSGBMCreate", lastNativeError()))

// =========================================================================
// Stereo statics
// =========================================================================

actual fun stereoRectify(
    cameraMatrix1: Mat,
    distCoeffs1: Mat,
    cameraMatrix2: Mat,
    distCoeffs2: Mat,
    imageSize: Size,
    r: Mat,
    t: Mat,
    flags: Int,
    alpha: Double,
    newImageSize: Size,
): StereoRectifyResult = memScoped {
    val roi1 = alloc<cvk_rect_t>()
    val roi2 = alloc<cvk_rect_t>()
    val r1 = alloc<CPointerVar<cvk_mat_t>>()
    val r2 = alloc<CPointerVar<cvk_mat_t>>()
    val p1 = alloc<CPointerVar<cvk_mat_t>>()
    val p2 = alloc<CPointerVar<cvk_mat_t>>()
    val q = alloc<CPointerVar<cvk_mat_t>>()
    val ok = cvk_stereo_rectify(
        cameraMatrix1.nativeHandle(), distCoeffs1.nativeHandle(),
        cameraMatrix2.nativeHandle(), distCoeffs2.nativeHandle(),
        imageSize.width, imageSize.height, r.nativeHandle(), t.nativeHandle(),
        flags, alpha, newImageSize.width, newImageSize.height,
        r1.ptr, r2.ptr, p1.ptr, p2.ptr, q.ptr, roi1.ptr, roi2.ptr,
    )
    if (ok == 0) throw OpenCVException("stereoRectify", lastNativeError())
    StereoRectifyResult(
        r1 = nativeMat(r1.value, "stereoRectify.r1"),
        r2 = nativeMat(r2.value, "stereoRectify.r2"),
        p1 = nativeMat(p1.value, "stereoRectify.p1"),
        p2 = nativeMat(p2.value, "stereoRectify.p2"),
        q = nativeMat(q.value, "stereoRectify.q"),
        validPixROI1 = Rect(roi1.x, roi1.y, roi1.width, roi1.height),
        validPixROI2 = Rect(roi2.x, roi2.y, roi2.width, roi2.height),
    )
}

actual fun stereoRectifyUncalibrated(
    points1: Mat,
    points2: Mat,
    f: Mat,
    imgSize: Size,
    threshold: Double,
): StereoRectifyUncalibratedResult = memScoped {
    val h1 = alloc<CPointerVar<cvk_mat_t>>()
    val h2 = alloc<CPointerVar<cvk_mat_t>>()
    val ok = cvk_stereo_rectify_uncalibrated(
        points1.nativeHandle(), points2.nativeHandle(), f.nativeHandle(),
        imgSize.width, imgSize.height, threshold, h1.ptr, h2.ptr,
    )
    StereoRectifyUncalibratedResult(
        success = ok != 0,
        h1 = nativeMat(h1.value, "stereoRectifyUncalibrated.h1"),
        h2 = nativeMat(h2.value, "stereoRectifyUncalibrated.h2"),
    )
}

actual fun fisheyeStereoRectify(
    k1: Mat,
    d1: Mat,
    k2: Mat,
    d2: Mat,
    imageSize: Size,
    r: Mat,
    tvec: Mat,
    flags: Int,
    newImageSize: Size,
    balance: Double,
    fovScale: Double,
): FisheyeStereoRectifyResult = memScoped {
    val r1 = alloc<CPointerVar<cvk_mat_t>>()
    val r2 = alloc<CPointerVar<cvk_mat_t>>()
    val p1 = alloc<CPointerVar<cvk_mat_t>>()
    val p2 = alloc<CPointerVar<cvk_mat_t>>()
    val q = alloc<CPointerVar<cvk_mat_t>>()
    val ok = cvk_stereo_fisheye_rectify(
        k1.nativeHandle(), d1.nativeHandle(), k2.nativeHandle(), d2.nativeHandle(),
        imageSize.width, imageSize.height, r.nativeHandle(), tvec.nativeHandle(),
        flags, newImageSize.width, newImageSize.height, balance, fovScale,
        r1.ptr, r2.ptr, p1.ptr, p2.ptr, q.ptr,
    )
    if (ok == 0) throw OpenCVException("fisheyeStereoRectify", lastNativeError())
    FisheyeStereoRectifyResult(
        r1 = nativeMat(r1.value, "fisheyeStereoRectify.r1"),
        r2 = nativeMat(r2.value, "fisheyeStereoRectify.r2"),
        p1 = nativeMat(p1.value, "fisheyeStereoRectify.p1"),
        p2 = nativeMat(p2.value, "fisheyeStereoRectify.p2"),
        q = nativeMat(q.value, "fisheyeStereoRectify.q"),
    )
}

actual fun filterSpeckles(
    img: Mat,
    newVal: Double,
    maxSpeckleSize: Int,
    maxDiff: Double,
    buf: Mat?,
) {
    if (buf != null) {
        cvk_stereo_filter_speckles_buf(img.nativeHandle(), newVal, maxSpeckleSize, maxDiff, buf.nativeHandle())
    } else {
        cvk_stereo_filter_speckles(img.nativeHandle(), newVal, maxSpeckleSize, maxDiff)
    }
}

actual fun getValidDisparityROI(
    roi1: Rect,
    roi2: Rect,
    minDisparity: Int,
    numberOfDisparities: Int,
    blockSize: Int,
): Rect = cvk_stereo_get_valid_disparity_roi(
    roi1.toCvkRect(), roi2.toCvkRect(), minDisparity, numberOfDisparities, blockSize,
).useContents { Rect(x, y, width, height) }

actual fun validateDisparity(
    disparity: Mat,
    cost: Mat,
    minDisparity: Int,
    numberOfDisparities: Int,
    disp12MaxDisp: Int,
) {
    cvk_stereo_validate_disparity(
        disparity.nativeHandle(), cost.nativeHandle(), minDisparity, numberOfDisparities,
        disp12MaxDisp,
    )
}

actual fun reprojectImageTo3D(
    disparity: Mat,
    q: Mat,
    handleMissingValues: Boolean,
    ddepth: Int,
): Mat = nativeMat(
    cvk_stereo_reproject_image_to_3d(
        disparity.nativeHandle(), q.nativeHandle(), if (handleMissingValues) 1 else 0, ddepth,
    ),
    "reprojectImageTo3D",
)

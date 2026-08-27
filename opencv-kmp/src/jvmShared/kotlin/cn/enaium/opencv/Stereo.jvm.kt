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

private fun lastNativeError(): String? = Jni.lastError()

/** Validates a fresh factory handle; throws with the native error text when it is 0. */
private fun Long.checkHandle(operation: String): Long =
    takeIf { it != 0L } ?: throw OpenCVException(operation, lastNativeError())

private fun Rect.toIntArray(): IntArray = intArrayOf(x, y, width, height)

private fun IntArray.toRect(): Rect = Rect(this[0], this[1], this[2], this[3])

// =========================================================================
// StereoMatcher
// =========================================================================

internal open class JvmStereoMatcher(@Volatile private var handle: Long) : StereoMatcher {

    protected fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("StereoMatcher is closed")

    override fun compute(left: Mat, right: Mat): Mat =
        jvmMat(
            JniStereo.stereoMatcherCompute(check(), handleOf(left), handleOf(right)),
            "StereoMatcher.compute",
        )

    override var minDisparity: Int
        get() = JniStereo.stereoMatcherGetMinDisparity(check())
        set(value) = JniStereo.stereoMatcherSetMinDisparity(check(), value)

    override var numDisparities: Int
        get() = JniStereo.stereoMatcherGetNumDisparities(check())
        set(value) = JniStereo.stereoMatcherSetNumDisparities(check(), value)

    override var blockSize: Int
        get() = JniStereo.stereoMatcherGetBlockSize(check())
        set(value) = JniStereo.stereoMatcherSetBlockSize(check(), value)

    override var speckleWindowSize: Int
        get() = JniStereo.stereoMatcherGetSpeckleWindowSize(check())
        set(value) = JniStereo.stereoMatcherSetSpeckleWindowSize(check(), value)

    override var speckleRange: Int
        get() = JniStereo.stereoMatcherGetSpeckleRange(check())
        set(value) = JniStereo.stereoMatcherSetSpeckleRange(check(), value)

    override var disp12MaxDiff: Int
        get() = JniStereo.stereoMatcherGetDisp12MaxDiff(check())
        set(value) = JniStereo.stereoMatcherSetDisp12MaxDiff(check(), value)

    override fun clear() {
        JniStereo.stereoMatcherClear(check())
    }

    override fun empty(): Boolean = JniStereo.stereoMatcherEmpty(check())

    override fun save(filename: String) {
        JniStereo.stereoMatcherSave(check(), filename)
    }

    override fun getDefaultName(): String =
        JniStereo.stereoMatcherGetDefaultName(check()).orEmpty()

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniStereo.stereoMatcherRelease(h)
        }
    }
}

internal class JvmStereoBm(handle: Long) : JvmStereoMatcher(handle), StereoBM {

    override var preFilterType: Int
        get() = JniStereo.stereoBmGetPreFilterType(check())
        set(value) = JniStereo.stereoBmSetPreFilterType(check(), value)

    override var preFilterSize: Int
        get() = JniStereo.stereoBmGetPreFilterSize(check())
        set(value) = JniStereo.stereoBmSetPreFilterSize(check(), value)

    override var preFilterCap: Int
        get() = JniStereo.stereoBmGetPreFilterCap(check())
        set(value) = JniStereo.stereoBmSetPreFilterCap(check(), value)

    override var textureThreshold: Int
        get() = JniStereo.stereoBmGetTextureThreshold(check())
        set(value) = JniStereo.stereoBmSetTextureThreshold(check(), value)

    override var uniquenessRatio: Int
        get() = JniStereo.stereoBmGetUniquenessRatio(check())
        set(value) = JniStereo.stereoBmSetUniquenessRatio(check(), value)

    override var smallerBlockSize: Int
        get() = JniStereo.stereoBmGetSmallerBlockSize(check())
        set(value) = JniStereo.stereoBmSetSmallerBlockSize(check(), value)

    override var roi1: Rect
        get() = JniStereo.stereoBmGetRoi1(check()).toRect()
        set(value) = JniStereo.stereoBmSetRoi1(check(), value.toIntArray())

    override var roi2: Rect
        get() = JniStereo.stereoBmGetRoi2(check()).toRect()
        set(value) = JniStereo.stereoBmSetRoi2(check(), value.toIntArray())
}

internal class JvmStereoSgbm(handle: Long) : JvmStereoMatcher(handle), StereoSGBM {

    override var preFilterCap: Int
        get() = JniStereo.stereoSgbmGetPreFilterCap(check())
        set(value) = JniStereo.stereoSgbmSetPreFilterCap(check(), value)

    override var uniquenessRatio: Int
        get() = JniStereo.stereoSgbmGetUniquenessRatio(check())
        set(value) = JniStereo.stereoSgbmSetUniquenessRatio(check(), value)

    override var p1: Int
        get() = JniStereo.stereoSgbmGetP1(check())
        set(value) = JniStereo.stereoSgbmSetP1(check(), value)

    override var p2: Int
        get() = JniStereo.stereoSgbmGetP2(check())
        set(value) = JniStereo.stereoSgbmSetP2(check(), value)

    override var mode: Int
        get() = JniStereo.stereoSgbmGetMode(check())
        set(value) = JniStereo.stereoSgbmSetMode(check(), value)
}

actual fun stereoBMCreate(numDisparities: Int, blockSize: Int): StereoBM =
    JvmStereoBm(JniStereo.stereoBmCreate(numDisparities, blockSize).checkHandle("stereoBMCreate"))

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
    JvmStereoSgbm(JniStereo.stereoSgbmCreate(
        minDisparity, numDisparities, blockSize, p1, p2, disp12MaxDiff, preFilterCap,
        uniquenessRatio, speckleWindowSize, speckleRange, mode,
    ).checkHandle("stereoSGBMCreate"))

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
): StereoRectifyResult {
    val roi1Out = IntArray(4)
    val roi2Out = IntArray(4)
    val handles = JniStereo.stereoRectify(
        handleOf(cameraMatrix1), handleOf(distCoeffs1),
        handleOf(cameraMatrix2), handleOf(distCoeffs2),
        imageSize.width, imageSize.height, handleOf(r), handleOf(t),
        flags, alpha, newImageSize.width, newImageSize.height, roi1Out, roi2Out,
    ) ?: throw OpenCVException("stereoRectify", lastNativeError())
    return StereoRectifyResult(
        r1 = jvmMat(handles[0], "stereoRectify.r1"),
        r2 = jvmMat(handles[1], "stereoRectify.r2"),
        p1 = jvmMat(handles[2], "stereoRectify.p1"),
        p2 = jvmMat(handles[3], "stereoRectify.p2"),
        q = jvmMat(handles[4], "stereoRectify.q"),
        validPixROI1 = roi1Out.toRect(),
        validPixROI2 = roi2Out.toRect(),
    )
}

actual fun stereoRectifyUncalibrated(
    points1: Mat,
    points2: Mat,
    f: Mat,
    imgSize: Size,
    threshold: Double,
): StereoRectifyUncalibratedResult {
    val out = JniStereo.stereoRectifyUncalibrated(
        handleOf(points1), handleOf(points2), handleOf(f),
        imgSize.width, imgSize.height, threshold,
    )
    return StereoRectifyUncalibratedResult(
        success = out[0] != 0L,
        h1 = jvmMat(out[1], "stereoRectifyUncalibrated.h1"),
        h2 = jvmMat(out[2], "stereoRectifyUncalibrated.h2"),
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
): FisheyeStereoRectifyResult {
    val handles = JniStereo.stereoFisheyeRectify(
        handleOf(k1), handleOf(d1), handleOf(k2), handleOf(d2),
        imageSize.width, imageSize.height, handleOf(r), handleOf(tvec),
        flags, newImageSize.width, newImageSize.height, balance, fovScale,
    ) ?: throw OpenCVException("fisheyeStereoRectify", lastNativeError())
    return FisheyeStereoRectifyResult(
        r1 = jvmMat(handles[0], "fisheyeStereoRectify.r1"),
        r2 = jvmMat(handles[1], "fisheyeStereoRectify.r2"),
        p1 = jvmMat(handles[2], "fisheyeStereoRectify.p1"),
        p2 = jvmMat(handles[3], "fisheyeStereoRectify.p2"),
        q = jvmMat(handles[4], "fisheyeStereoRectify.q"),
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
        JniStereo.stereoFilterSpecklesBuf(
            handleOf(img), newVal, maxSpeckleSize, maxDiff, handleOf(buf),
        )
    } else {
        JniStereo.stereoFilterSpeckles(handleOf(img), newVal, maxSpeckleSize, maxDiff)
    }
}

actual fun getValidDisparityROI(
    roi1: Rect,
    roi2: Rect,
    minDisparity: Int,
    numberOfDisparities: Int,
    blockSize: Int,
): Rect = JniStereo.stereoGetValidDisparityROI(
    roi1.toIntArray(), roi2.toIntArray(), minDisparity, numberOfDisparities, blockSize,
).toRect()

actual fun validateDisparity(
    disparity: Mat,
    cost: Mat,
    minDisparity: Int,
    numberOfDisparities: Int,
    disp12MaxDisp: Int,
) {
    JniStereo.stereoValidateDisparity(
        handleOf(disparity), handleOf(cost), minDisparity, numberOfDisparities, disp12MaxDisp,
    )
}

actual fun reprojectImageTo3D(
    disparity: Mat,
    q: Mat,
    handleMissingValues: Boolean,
    ddepth: Int,
): Mat = jvmMat(
    JniStereo.stereoReprojectImageTo3D(
        handleOf(disparity), handleOf(q), handleMissingValues, ddepth,
    ),
    "reprojectImageTo3D",
)

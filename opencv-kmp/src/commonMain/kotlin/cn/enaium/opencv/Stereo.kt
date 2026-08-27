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
 * The base class for stereo correspondence algorithms (`cv::StereoMatcher`).
 *
 * Subclasses ([StereoBM], [StereoSGBM]) compute dense disparity maps for
 * rectified stereo pairs. Fixed-point matchers report disparity values
 * scaled by [DISP_SCALE] (i.e. with [DISP_SHIFT] fractional bits).
 */
interface StereoMatcher : Algorithm {

    companion object {
        /** Number of fractional bits in fixed-point disparity values. */
        const val DISP_SHIFT = 4

        /** Multiplier applied to disparity values (`1 shl DISP_SHIFT`). */
        const val DISP_SCALE = 16
    }

    /**
     * Computes the disparity map for the stereo pair ([left], [right]).
     *
     * @param left Left 8-bit single-channel image.
     * @param right Right image of the same size and type as [left].
     * @return a new disparity Mat (16-bit fixed-point for [StereoBM] /
     *   [StereoSGBM]).
     */
    fun compute(left: Mat, right: Mat): Mat

    /** Minimum possible disparity value (normally zero). */
    var minDisparity: Int

    /** Maximum disparity minus minimum disparity; must be divisible by 16. */
    var numDisparities: Int

    /** Linear size of the matched blocks; should be odd. */
    var blockSize: Int

    /** Maximum size of smooth disparity regions invalidated as speckles. */
    var speckleWindowSize: Int

    /** Maximum disparity variation within a connected speckle component. */
    var speckleRange: Int

    /** Max allowed difference in the left-right disparity check. */
    var disp12MaxDiff: Int

    override fun close()
}

/**
 * Stereo correspondence with the block matching algorithm of K. Konolige
 * (`cv::StereoBM`).
 */
interface StereoBM : StereoMatcher {

    companion object {
        /** Normalized response pre-filter (the default). */
        const val PREFILTER_NORMALIZED_RESPONSE = 0

        /** Sobel x-derivative pre-filter. */
        const val PREFILTER_XSOBEL = 1
    }

    /** Pre-filter type ([PREFILTER_NORMALIZED_RESPONSE] or [PREFILTER_XSOBEL]). */
    var preFilterType: Int

    /** Pre-filter window size (odd, 5..255). */
    var preFilterSize: Int

    /** Truncation value of the pre-filtered pixels (31 default). */
    var preFilterCap: Int

    /** Minimum texture threshold for a pixel to be matched. */
    var textureThreshold: Int

    /** Uniqueness ratio margin (5..15 is a good range). */
    var uniquenessRatio: Int

    /** Smaller block size used by the two-level matching. */
    var smallerBlockSize: Int

    /** Valid ROI in the left image. */
    var roi1: Rect

    /** Valid ROI in the right image. */
    var roi2: Rect

    override fun close()
}

/**
 * The modified H. Hirschmuller algorithm (`cv::StereoSGBM`); semi-global
 * block matching with Birchfield-Tomasi costs.
 */
interface StereoSGBM : StereoMatcher {

    companion object {
        /** Single-pass 5-direction algorithm (the default). */
        const val MODE_SGBM = 0

        /** Full-scale two-pass dynamic programming (8 directions). */
        const val MODE_HH = 1

        /** 3-way SGBM (faster, slightly lower quality). */
        const val MODE_SGBM_3WAY = 2

        /** HH variant optimized for 4-thread execution. */
        const val MODE_HH4 = 3
    }

    /** Truncation value of the pre-filtered image pixels. */
    var preFilterCap: Int

    /** Margin (in percent) the best cost must win the second best by. */
    var uniquenessRatio: Int

    /** Penalty for disparity change by +/-1 between neighbor pixels. */
    var p1: Int

    /** Penalty for disparity change by more than 1 between neighbors; P2 > P1. */
    var p2: Int

    /** Algorithm mode ([MODE_SGBM] .. [MODE_HH4]). */
    var mode: Int

    override fun close()
}

/** Creates a [StereoBM] matcher (`cv::StereoBM::create`). */
expect fun stereoBMCreate(numDisparities: Int = 0, blockSize: Int = 21): StereoBM

/**
 * Creates a [StereoSGBM] matcher (`cv::StereoSGBM::create`); defaults match
 * the C++ defaults (blockSize 3, MODE_SGBM, speckle filtering disabled).
 */
expect fun stereoSGBMCreate(
    minDisparity: Int = 0,
    numDisparities: Int = 16,
    blockSize: Int = 3,
    p1: Int = 0,
    p2: Int = 0,
    disp12MaxDiff: Int = 0,
    preFilterCap: Int = 0,
    uniquenessRatio: Int = 0,
    speckleWindowSize: Int = 0,
    speckleRange: Int = 0,
    mode: Int = StereoSGBM.MODE_SGBM,
): StereoSGBM

// =========================================================================
// Stereo statics (org.opencv.stereo.Stereo parity)
// =========================================================================

/** Stereo utility functions and constants (`cv::Stereo`). */
object Stereo {
    /** Make the principal points of each camera coincide after rectification. */
    const val STEREO_ZERO_DISPARITY = 0x00400
}

/** Outputs of [stereoRectify]; every Mat must be closed by the caller. */
data class StereoRectifyResult(
    /** 3x3 rectification rotation for the first camera. */
    val r1: Mat,
    /** 3x3 rectification rotation for the second camera. */
    val r2: Mat,
    /** 3x4 projection matrix of the rectified first camera. */
    val p1: Mat,
    /** 3x4 projection matrix of the rectified second camera. */
    val p2: Mat,
    /** 4x4 disparity-to-depth mapping matrix. */
    val q: Mat,
    /** Valid pixel ROI inside the first rectified image. */
    val validPixROI1: Rect,
    /** Valid pixel ROI inside the second rectified image. */
    val validPixROI2: Rect,
)

/** Result of [stereoRectifyUncalibrated]; Mats must be closed by the caller. */
data class StereoRectifyUncalibratedResult(
    /** Whether the homographies could be computed. */
    val success: Boolean,
    /** Rectification homography of the first image. */
    val h1: Mat,
    /** Rectification homography of the second image. */
    val h2: Mat,
)

/** Outputs of [fisheyeStereoRectify]; every Mat must be closed by the caller. */
data class FisheyeStereoRectifyResult(
    /** 3x3 rectification rotation for the first camera. */
    val r1: Mat,
    /** 3x3 rectification rotation for the second camera. */
    val r2: Mat,
    /** 3x4 projection matrix of the rectified first camera. */
    val p1: Mat,
    /** 3x4 projection matrix of the rectified second camera. */
    val p2: Mat,
    /** 4x4 disparity-to-depth mapping matrix. */
    val q: Mat,
)

/**
 * Computes rectification transforms for each head of a calibrated stereo
 * camera (`cv::stereoRectify`).
 *
 * @param flags pass [Stereo.STEREO_ZERO_DISPARITY] to make the principal
 *   points of both cameras coincide; [alpha] between 0 and 1 controls the
 *   valid-pixel zoom (or -1 for the default).
 */
expect fun stereoRectify(
    cameraMatrix1: Mat,
    distCoeffs1: Mat,
    cameraMatrix2: Mat,
    distCoeffs2: Mat,
    imageSize: Size,
    r: Mat,
    t: Mat,
    flags: Int = Stereo.STEREO_ZERO_DISPARITY,
    alpha: Double = -1.0,
    newImageSize: Size = Size(0, 0),
): StereoRectifyResult

/**
 * Computes rectification homographies without intrinsic parameters
 * (`cv::stereoRectifyUncalibrated`).
 */
expect fun stereoRectifyUncalibrated(
    points1: Mat,
    points2: Mat,
    f: Mat,
    imgSize: Size,
    threshold: Double = 5.0,
): StereoRectifyUncalibratedResult

/**
 * Stereo rectification for the fisheye camera model
 * (`cv::fisheye::stereoRectify`).
 */
expect fun fisheyeStereoRectify(
    k1: Mat,
    d1: Mat,
    k2: Mat,
    d2: Mat,
    imageSize: Size,
    r: Mat,
    tvec: Mat,
    flags: Int = Stereo.STEREO_ZERO_DISPARITY,
    newImageSize: Size = Size(0, 0),
    balance: Double = 0.0,
    fovScale: Double = 1.0,
): FisheyeStereoRectifyResult

/**
 * Filters off small noise blobs (speckles) in a 16-bit signed disparity
 * image in place (`cv::filterSpeckles`).
 *
 * @param buf optional temporary buffer avoiding allocation inside the call.
 */
expect fun filterSpeckles(
    img: Mat,
    newVal: Double,
    maxSpeckleSize: Int,
    maxDiff: Double,
    buf: Mat? = null,
)

/** Computes the valid disparity ROI from the rectified images' ROIs. */
expect fun getValidDisparityROI(
    roi1: Rect,
    roi2: Rect,
    minDisparity: Int,
    numberOfDisparities: Int,
    blockSize: Int,
): Rect

/**
 * Validates a disparity map with the left-right check in place
 * (`cv::validateDisparity`).
 */
expect fun validateDisparity(
    disparity: Mat,
    cost: Mat,
    minDisparity: Int,
    numberOfDisparities: Int,
    disp12MaxDisp: Int = 1,
)

/**
 * Reprojects a disparity image to 3D space (`cv::reprojectImageTo3D`);
 * returns a new 3-channel image of 3D coordinates.
 */
expect fun reprojectImageTo3D(
    disparity: Mat,
    q: Mat,
    handleMissingValues: Boolean = false,
    ddepth: Int = -1,
): Mat

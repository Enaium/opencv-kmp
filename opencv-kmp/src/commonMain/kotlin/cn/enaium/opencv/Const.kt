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
@file:Suppress("MemberVisibilityCanBePrivate", "PropertyName")

package cn.enaium.opencv

/** Matrix element depth ids; values match OpenCV's `CV_8U`, `CV_32F`, ... */
const val CV_8U: Int = 0
const val CV_8S: Int = 1
const val CV_16U: Int = 2
const val CV_16S: Int = 3
const val CV_32S: Int = 4
const val CV_32F: Int = 5
const val CV_64F: Int = 6
const val CV_16F: Int = 7
const val CV_16BF: Int = 8

// OpenCV 5 encoding: channel count occupies the high bits above a 5-bit
// depth field (interface.h: CV_CN_SHIFT=5, CV_DEPTH_MAX=32).
private const val CN_SHIFT: Int = 5
private const val DEPTH_MASK: Int = 31

/**
 * Builds a Mat type id from a depth and a channel count, matching OpenCV's
 * `CV_MAKETYPE(depth, cn)` macro (e.g. [MatType.CV_8UC3] == 16).
 */
fun cvMakeType(depth: Int, channels: Int): Int =
    (depth and DEPTH_MASK) or ((channels - 1) shl CN_SHIFT)

/** Extracts the depth part of a Mat type id (`CV_MAT_DEPTH`). */
fun cvDepthOf(type: Int): Int = type and DEPTH_MASK

/** Extracts the channel count of a Mat type id (`CV_MAT_CN`). */
fun cvChannelsOf(type: Int): Int = (type shr CN_SHIFT) + 1

/** Frequently used Mat type ids; build arbitrary ones with [cvMakeType]. */
object MatType {
    val CV_8UC1 = cvMakeType(CV_8U, 1)
    val CV_8UC2 = cvMakeType(CV_8U, 2)
    val CV_8UC3 = cvMakeType(CV_8U, 3)
    val CV_8UC4 = cvMakeType(CV_8U, 4)

    val CV_16UC1 = cvMakeType(CV_16U, 1)
    val CV_16UC3 = cvMakeType(CV_16U, 3)
    val CV_16UC4 = cvMakeType(CV_16U, 4)

    val CV_16SC1 = cvMakeType(CV_16S, 1)
    val CV_16SC3 = cvMakeType(CV_16S, 3)

    val CV_32SC1 = cvMakeType(CV_32S, 1)
    val CV_32SC3 = cvMakeType(CV_32S, 3)

    val CV_32FC1 = cvMakeType(CV_32F, 1)
    val CV_32FC3 = cvMakeType(CV_32F, 3)
    val CV_32FC4 = cvMakeType(CV_32F, 4)

    val CV_64FC1 = cvMakeType(CV_64F, 1)
    val CV_64FC3 = cvMakeType(CV_64F, 3)

    /** Builds any type not predefined above. */
    fun of(depth: Int, channels: Int): Int = cvMakeType(depth, channels)
}

/** Human-readable name of a Mat type id, e.g. `"CV_8UC3"`. */
val Int.cvTypeName: String
    get() {
        val depth = when (cvDepthOf(this)) {
            CV_8U -> "8U"
            CV_8S -> "8S"
            CV_16U -> "16U"
            CV_16S -> "16S"
            CV_32S -> "32S"
            CV_32F -> "32F"
            CV_64F -> "64F"
            else -> "16F"
        }
        return "CV_${depth}C${cvChannelsOf(this)}"
    }

// =========================================================================
// imgproc constants (values match the OpenCV enums exactly)
// =========================================================================

/** Subset of OpenCV's `cv::ColorConversionCodes`. */
object ColorConversionCodes {
    const val BGR2BGRA: Int = 0
    const val RGB2RGBA: Int = 0
    const val BGRA2BGR: Int = 1
    const val RGBA2BGR: Int = 1
    const val BGR2RGBA: Int = 2
    const val RGB2BGRA: Int = 2
    const val RGBA2RGB: Int = 3
    const val BGRA2RGB: Int = 3
    const val BGR2RGB: Int = 4
    const val RGB2BGR: Int = 5
    const val BGR2GRAY: Int = 6
    const val RGB2GRAY: Int = 7
    const val GRAY2BGR: Int = 8
    const val GRAY2RGB: Int = 9
    const val BGRA2GRAY: Int = 10
    const val RGBA2GRAY: Int = 11
    const val BGR2XYZ: Int = 32
    const val XYZ2BGR: Int = 34
    const val BGR2YCrCb: Int = 36
    const val YCrCb2BGR: Int = 38
    const val BGR2HSV: Int = 40
    const val HSV2BGR: Int = 54
    const val BGR2Lab: Int = 44
    const val Lab2BGR: Int = 56
    const val BGR2Luv: Int = 50
    const val Luv2BGR: Int = 58
    const val BGR2HLS: Int = 52
    const val HLS2BGR: Int = 60
}

/** OpenCV's `cv::ThresholdTypes`. */
object ThresholdTypes {
    const val BINARY: Int = 0
    const val BINARY_INV: Int = 1
    const val TRUNC: Int = 2
    const val TOZERO: Int = 3
    const val TOZERO_INV: Int = 4
    const val MASK: Int = 7
    const val OTSU: Int = 8
    const val TRIANGLE: Int = 16
}

/** OpenCV's `cv::AdaptiveThresholdTypes`. */
object AdaptiveThresholdTypes {
    const val MEAN_C: Int = 0
    const val GAUSSIAN_C: Int = 1
}

/** OpenCV's `cv::InterpolationFlags`. */
object InterpolationFlags {
    const val NEAREST: Int = 0
    const val LINEAR: Int = 1
    const val CUBIC: Int = 2
    const val AREA: Int = 3
    const val LANCZOS4: Int = 4
    const val LINEAR_EXACT: Int = 5
    const val NEAREST_EXACT: Int = 6
}

/** Flip axes accepted by [Mat.flip]. */
object FlipMode {
    /** Around the x-axis (upside down). */
    const val VERTICAL: Int = 0

    /** Around the y-axis (mirror). */
    const val HORIZONTAL: Int = 1
    const val BOTH: Int = -1
}

/** Subset of OpenCV's `cv::ImreadModes`. */
object ImreadFlags {
    const val GRAYSCALE: Int = 0
    const val COLOR: Int = 1
    const val ANYDEPTH: Int = 2
    const val ANYCOLOR: Int = 4
    const val REDUCED_GRAYSCALE_2: Int = 16
    const val REDUCED_COLOR_2: Int = 17
    const val REDUCED_GRAYSCALE_4: Int = 32
    const val REDUCED_COLOR_4: Int = 33
    const val REDUCED_GRAYSCALE_8: Int = 64
    const val REDUCED_COLOR_8: Int = 65
}

/** OpenCV's `cv::LineTypes`. */
object LineTypes {
    const val FILLED: Int = -1
    const val LINE_4: Int = 4
    const val LINE_8: Int = 8
    const val LINE_AA: Int = 16
}

// =========================================================================
// core constants
// =========================================================================

/** OpenCV's `cv::NormTypes`. */
object NormTypes {
    const val INF: Int = 1
    const val L1: Int = 4
    const val L2: Int = 5
    const val L2SQR: Int = 6
    const val HAMMING: Int = 7
    const val HAMMING2: Int = 8
    const val TYPE_MASK: Int = 7
    const val RELATIVE: Int = 8
    const val MINMAX: Int = 32
}

/** OpenCV's `cv::DecompTypes` (matrix inversion / solving methods). */
object DecompTypes {
    const val LU: Int = 0
    const val SVD: Int = 1
    const val EIG: Int = 2
    const val CHOLESKY: Int = 3
    const val QR: Int = 4
    const val NORMAL: Int = 16
}

/** OpenCV's `cv::RotateFlags`. */
object RotateFlags {
    const val ROTATE_90_CLOCKWISE: Int = 0
    const val ROTATE_180: Int = 1
    const val ROTATE_270: Int = 2
}

/** OpenCV's `cv::BorderTypes`. */
object BorderTypes {
    const val CONSTANT: Int = 0
    const val REPLICATE: Int = 1
    const val REFLECT: Int = 2
    const val WRAP: Int = 3
    const val REFLECT_101: Int = 4
    const val TRANSPARENT: Int = 16
    const val ISOLATED: Int = 8
}

/** OpenCV's `cv::CmpTypes`. */
object CompareOps {
    const val CMP_EQ: Int = 0
    const val CMP_GT: Int = 1
    const val CMP_GE: Int = 2
    const val CMP_LT: Int = 3
    const val CMP_LE: Int = 4
    const val CMP_NE: Int = 5
}

/** OpenCV's sort flags. */
object SortFlags {
    const val EVERY_ROW: Int = 0
    const val EVERY_COLUMN: Int = 1
    const val ASCENDING: Int = 0
    const val DESCENDING: Int = 16
}

/** OpenCV's `cv::ReduceTypes`. */
object ReduceTypes {
    const val SUM: Int = 0
    const val AVG: Int = 1
    const val MAX: Int = 2
    const val MIN: Int = 3
}

/** OpenCV's `cv::DftFlags`. */
object DftFlags {
    const val INVERSE: Int = 1
    const val SCALE: Int = 2
    const val ROWS: Int = 4
    const val COMPLEX_OUTPUT: Int = 16
    const val REAL_OUTPUT: Int = 32
    const val COMPLEX_INPUT: Int = 64
    const val DCT_INVERSE: Int = 256
    const val DCT_ROWS: Int = 512
}

// =========================================================================
// imgproc constants (continued)
// =========================================================================

/** OpenCV's `cv::MorphShapes`. */
object MorphShapes {
    const val RECT: Int = 0
    const val CROSS: Int = 1
    const val ELLIPSE: Int = 2
}

/** OpenCV's `cv::MorphTypes`. */
object MorphTypes {
    const val ERODE: Int = 0
    const val DILATE: Int = 1
    const val OPEN: Int = 2
    const val CLOSE: Int = 3
    const val GRADIENT: Int = 4
    const val TOPHAT: Int = 5
    const val BLACKHAT: Int = 6
    const val HITMISS: Int = 7
}

/** OpenCV's `cv::WarpPolarMode`. */
object WarpPolarMode {
    const val POLAR: Int = 0
    const val LOG: Int = 256
}

/** OpenCV's `cv::ColormapTypes`. */
object ColormapTypes {
    const val AUTUMN: Int = 0
    const val BONE: Int = 1
    const val JET: Int = 2
    const val WINTER: Int = 3
    const val RAINBOW: Int = 4
    const val OCEAN: Int = 5
    const val SUMMER: Int = 6
    const val SPRING: Int = 7
    const val COOL: Int = 8
    const val HSV: Int = 9
    const val PINK: Int = 10
    const val HOT: Int = 11
    const val PARULA: Int = 12
}

/** OpenCV's histogram comparison methods. */
object HistCompMethods {
    const val CORREL: Int = 0
    const val CHISQR: Int = 1
    const val INTERSECT: Int = 2
    const val BHATTACHARYYA: Int = 3

    /** Alias of [BHATTACHARYYA]. */
    const val HELLINGER: Int = 3
    const val CHISQR_ALT: Int = 4
    const val KL_DIV: Int = 5
}

/** OpenCV's Hu moment shape matching methods. */
object ContoursMatchMethods {
    const val CONTOURS_MATCH_I1: Int = 1
    const val CONTOURS_MATCH_I2: Int = 2
    const val CONTOURS_MATCH_I3: Int = 3
}

/** OpenCV's `cv::TemplateMatchModes`. */
object TemplateMatchModes {
    const val SQDIFF: Int = 0
    const val SQDIFF_NORMED: Int = 1
    const val CCORR: Int = 2
    const val CCORR_NORMED: Int = 3
    const val CCOEFF: Int = 4
    const val CCOEFF_NORMED: Int = 5
}

/** OpenCV's contour retrieval modes. */
object RetrievalModes {
    const val RETR_EXTERNAL: Int = 0
    const val RETR_LIST: Int = 1
    const val RETR_CCOMP: Int = 2
    const val RETR_TREE: Int = 3
    const val RETR_FLOODFILL: Int = 4
}

/** OpenCV's contour approximation modes. */
object ContourApproximationModes {
    const val CHAIN_APPROX_NONE: Int = 1
    const val CHAIN_APPROX_SIMPLE: Int = 2
    const val CHAIN_APPROX_TC89_L1: Int = 3
    const val CHAIN_APPROX_TC89_KCOS: Int = 4
}

/** OpenCV's `cv::MarkerTypes`. */
object MarkerTypes {
    const val CROSS: Int = 0
    const val TILTED_CROSS: Int = 1
    const val STAR: Int = 2
    const val DIAMOND: Int = 3
    const val SQUARE: Int = 4
    const val TILTED_SQUARE: Int = 5
}

/** OpenCV's `cv::DistanceTypes`. */
object DistanceTypes {
    const val USER: Int = -1
    const val L1: Int = 1
    const val L2: Int = 2
    const val C: Int = 3
    const val L12: Int = 4
    const val FAIR: Int = 5
    const val WELSCH: Int = 6
    const val HUBER: Int = 7
}

/** Mask sizes accepted by [Mat.distanceTransform]. */
object DistanceTransformMasks {
    const val MASK_3: Int = 3
    const val MASK_5: Int = 5
    const val PRECISE: Int = 0
}

/** Connectivity flags for flood fill. */
object FloodFillFlags {
    const val CONNECTIVITY_4: Int = 4
    const val CONNECTIVITY_8: Int = 8
    const val FIXED_RANGE: Int = 1 shl 16
    const val MASK_ONLY: Int = 1 shl 17
}

/** OpenCV's `cv::HoughTypes`. */
object HoughTypes {
    const val STANDARD: Int = 0
    const val PROBABILISTIC: Int = 1
    const val GRADIENT: Int = 3
    const val GRADIENT_ALT: Int = 4
}

/** Subset of `cv::ImwriteJPEGFlags` / `cv::ImwritePNGFlags` for codec params. */
object ImwriteParams {
    const val JPEG_QUALITY: Int = 1
    const val JPEG_PROGRESSIVE: Int = 2
    const val JPEG_OPTIMIZE: Int = 3
    const val PNG_COMPRESSION: Int = 16
    const val PNG_STRATEGY: Int = 17
    const val PNG_BILEVEL: Int = 18
}

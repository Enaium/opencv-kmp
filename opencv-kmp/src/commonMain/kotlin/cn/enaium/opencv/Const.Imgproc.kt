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

// =========================================================================
// imgproc constants not yet covered by Const.kt (values match the SDK 5.0
// `Imgproc` enum constants exactly)
// =========================================================================

/**
 * OpenCV's `cv::HersheyFonts` (putText font faces).
 *
 * Text rendering is NOT available in this binding: the OpenCV build used by
 * opencv-kmp is compiled without HarfBuzz (`WITH_HARFBUZZ=OFF`), so
 * `putText`, `getTextSize` and `getFontScaleFromHeight` are deliberately
 * not exposed. The constants are kept for source compatibility with code
 * ported from the official SDK.
 */
object HersheyFonts {
    const val FONT_HERSHEY_SIMPLEX: Int = 0
    const val FONT_HERSHEY_PLAIN: Int = 1
    const val FONT_HERSHEY_DUPLEX: Int = 2
    const val FONT_HERSHEY_COMPLEX: Int = 3
    const val FONT_HERSHEY_TRIPLEX: Int = 4
    const val FONT_HERSHEY_COMPLEX_SMALL: Int = 5
    const val FONT_HERSHEY_SCRIPT_SIMPLEX: Int = 6
    const val FONT_HERSHEY_SCRIPT_COMPLEX: Int = 7

    /** Bitwise OR this with a base face for italics. */
    const val FONT_ITALIC: Int = 16
}

/**
 * OpenCV 5's `cv::PutTextFlags` for the FontFace-based `putText` overload.
 *
 * Like [HersheyFonts], these are kept for parity only — the text rendering
 * entry points are unavailable in this build (no HarfBuzz).
 */
object PutTextFlags {
    const val ALIGN_LEFT: Int = 0
    const val ALIGN_CENTER: Int = 1
    const val ALIGN_RIGHT: Int = 2

    /** Bitmask of the alignment bits. */
    const val MASK: Int = 3
    const val ORIGIN_TL: Int = 0
    const val ORIGIN_BL: Int = 32
    const val WRAP: Int = 128
}

/** OpenCV's `cv::ConnectedComponentsAlgorithmsTypes`. */
object ConnectedComponentsAlgorithms {
    const val CCL_DEFAULT: Int = -1
    const val CCL_WU: Int = 0
    const val CCL_GRANA: Int = 1
    const val CCL_BOLELLI: Int = 2
    const val CCL_SAUF: Int = 3
    const val CCL_BBDT: Int = 4
    const val CCL_SPAGHETTI: Int = 5
}

/**
 * Column indices of the per-component `stats` matrix produced by
 * [Mat.connectedComponentsWithStats] (`cv::ConnectedComponentsTypes`).
 */
object ConnectedComponentsTypes {
    const val CC_STAT_LEFT: Int = 0
    const val CC_STAT_TOP: Int = 1
    const val CC_STAT_WIDTH: Int = 2
    const val CC_STAT_HEIGHT: Int = 3
    const val CC_STAT_AREA: Int = 4
    const val CC_STAT_MAX: Int = 5
}

/** OpenCV's `cv::DistanceTransformLabelTypes`. */
object DistanceLabelTypes {
    const val DIST_LABEL_CCOMP: Int = 0
    const val DIST_LABEL_PIXEL: Int = 1
}

/** OpenCV's `cv::WarpFlags`, combined with [InterpolationFlags]. */
object WarpFlags {
    const val FILL_OUTLIERS: Int = 8
    const val INVERSE_MAP: Int = 16
    const val RELATIVE_MAP: Int = 32
}

/**
 * Robust-estimation methods accepted by [estimateAffine2D] /
 * [estimateAffinePartial2D] (`cv::RANSAC`, `cv::LMEDS`, `cv::RHO` and the
 * USAC variants of OpenCV 5).
 */
object RobustMethods {
    const val LMEDS: Int = 4
    const val RANSAC: Int = 8
    const val RHO: Int = 16
    const val USAC_DEFAULT: Int = 32
    const val USAC_PARALLEL: Int = 33
    const val USAC_FAST: Int = 35
    const val USAC_ACCURATE: Int = 36
    const val USAC_PROSAC: Int = 37
    const val USAC_MAGSAC: Int = 38
}

/**
 * Classification returned by [rotatedRectangleIntersection]
 * (`cv::RectIntersectFlags`).
 */
object RectangleIntersectTypes {
    const val NONE: Int = 0
    const val PARTIAL: Int = 1
    const val FULL: Int = 2
}

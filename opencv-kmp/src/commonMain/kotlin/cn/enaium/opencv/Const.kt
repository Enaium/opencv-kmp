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

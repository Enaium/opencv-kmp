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
 * Object wrappers of `org.opencv.imgproc`: [LineSegmentDetector],
 * [GeneralizedHough] (+ [GeneralizedHoughBallard] / [GeneralizedHoughGuil])
 * and the [Filter2DParams] value holder.
 *
 * Not ported: `org.opencv.imgproc.FontFace`. `cv::FontFace` is only compiled
 * into OpenCV 5 when built WITH HarfBuzz (`WITH_HARFBUZZ=ON`), and this
 * binding's native build explicitly disables HarfBuzz (the binding never
 * renders text; a system HarfBuzz would leak into every artifact). There is
 * therefore no native `cv::FontFace` to wrap, so the class is skipped rather
 * than stubbed. When a HarfBuzz-enabled OpenCV build is ever supported, the
 * port is: `class FontFace` with `fun set(fontPathOrName: String): Boolean`,
 * `val name: String`, `fun setInstance(params: MatOfInt): Boolean`,
 * `fun getInstance(params: MatOfInt): Boolean` and two factories.
 */

/** Refinement modes of [LineSegmentDetector], mirroring `cv::LineSegmentDetectorModes`. */
object LineSegmentDetectorModes {
    /** No refinement applied. */
    const val NONE: Int = 0

    /** Standard refinement (default). */
    const val STD: Int = 1

    /** Advanced refinement; the only mode that also fills the NFA vector. */
    const val ADV: Int = 2
}

/**
 * Output of [LineSegmentDetector.detect]: `lines` is an Nx1 CV_32FC4 matrix
 * of Vec4f segments `(x1, y1, x2, y2)`; `width`, `prec` and `nfa` are Nx1
 * CV_64FC1 vectors (empty when the mode does not produce them — `nfa` is
 * only filled for [LineSegmentDetectorModes.ADV]).
 *
 * Every [Mat] is an independent handle owned by the caller: close each one
 * when done.
 */
data class LineSegments(
    val lines: Mat,
    val width: Mat,
    val prec: Mat,
    val nfa: Mat,
)

/**
 * Line segment detector (`cv::LineSegmentDetector`), implementing the
 * algorithm described at CITE: Rafael12.
 */
interface LineSegmentDetector : Algorithm {

    /**
     * Finds lines in a grayscale (CV_8UC1) [image] and returns them together
     * with the per-segment width/precision/NFA vectors.
     */
    fun detect(image: Mat): LineSegments

    /**
     * Draws the line [segments] onto [image] in place (the image should be
     * at least as large as the one the segments were found in).
     */
    fun drawSegments(image: Mat, segments: Mat)

    /**
     * Draws [lines1] in blue and [lines2] in red (into [image] when given,
     * which must be 3-channel) and returns the count of non-overlapping
     * (mismatching) pixels; [size] is the size of the image the lines were
     * found in.
     */
    fun compareSegments(size: Size, lines1: Mat, lines2: Mat, image: Mat? = null): Int
}

/**
 * Creates a [LineSegmentDetector]; defaults match `cv::createLineSegmentDetector()`.
 *
 * @param refine refinement mode, see [LineSegmentDetectorModes]
 * @param scale scale of the image used to find lines, range (0..1]
 * @param sigmaScale sigma for the Gaussian filter (sigma = sigmaScale / scale)
 * @param quant bound to the quantization error on the gradient norm
 * @param angTh gradient angle tolerance in degrees
 * @param logEps detection threshold: -log10(NFA) > logEps
 * @param densityTh minimal density of aligned region points
 * @param nBins number of bins in pseudo-ordering of gradient modulus
 */
expect fun createLineSegmentDetector(
    refine: Int = LineSegmentDetectorModes.STD,
    scale: Double = 0.8,
    sigmaScale: Double = 0.6,
    quant: Double = 2.0,
    angTh: Double = 22.5,
    logEps: Double = 0.0,
    densityTh: Double = 0.7,
    nBins: Int = 1024,
): LineSegmentDetector

/**
 * Result of [GeneralizedHough.detect]: `positions` is a 1xN CV_32FC4 matrix
 * with one `(x, y, angle, scale)` row per detected instance (N = 0 when
 * nothing matched); `votes` is a 1xN CV_32SC3 matrix of accumulated votes
 * and is empty when no votes were accumulated.
 *
 * Both [Mat]s are independent handles owned by the caller: close them when
 * done.
 */
data class GeneralizedHoughResult(
    val positions: Mat,
    val votes: Mat,
)

/**
 * Finds an arbitrary template in a grayscale image using the Generalized
 * Hough Transform (`cv::GeneralizedHough`). Use [createGeneralizedHoughBallard]
 * or [createGeneralizedHoughGuil] to obtain an instance.
 */
interface GeneralizedHough : Algorithm {

    /** Sets the template image; [templCenter] defaults to the template center. */
    fun setTemplate(templ: Mat, templCenter: Point = Point(-1, -1))

    /**
     * Sets the template from precomputed [edges] (CV_8UC1) and gradient
     * images [dx]/[dy] (CV_32FC1); [templCenter] defaults to the template
     * center.
     */
    fun setTemplate(edges: Mat, dx: Mat, dy: Mat, templCenter: Point = Point(-1, -1))

    /** Finds the template in [image]. */
    fun detect(image: Mat): GeneralizedHoughResult

    /** Finds the template using precomputed edge / gradient images. */
    fun detect(edges: Mat, dx: Mat, dy: Mat): GeneralizedHoughResult

    /** Canny low threshold. */
    var cannyLowThresh: Int

    /** Canny high threshold. */
    var cannyHighThresh: Int

    /** Minimum distance between the centers of detected objects. */
    var minDist: Double

    /** Inverse ratio of the accumulator resolution to the image resolution. */
    var dp: Double

    /** Maximal size of inner buffers. */
    var maxBufferSize: Int
}

/**
 * Position-only Generalized Hough detector (`cv::GeneralizedHoughBallard`):
 * detects position without translation and rotation (CITE: Ballard1981).
 */
interface GeneralizedHoughBallard : GeneralizedHough {

    /** R-Table levels. */
    var levels: Int

    /** Accumulator threshold for template centers; smaller = more false positions. */
    var votesThreshold: Int
}

/**
 * Generalized Hough detector for position, rotation and scale
 * (`cv::GeneralizedHoughGuil`, CITE: Guil1999).
 */
interface GeneralizedHoughGuil : GeneralizedHough {

    /** Angle difference in degrees between two points in a feature. */
    var xi: Double

    /** Feature table levels. */
    var levels: Int

    /** Maximal difference between angles treated as equal. */
    var angleEpsilon: Double

    /** Minimal rotation angle to detect, in degrees. */
    var minAngle: Double

    /** Maximal rotation angle to detect, in degrees. */
    var maxAngle: Double

    /** Angle step in degrees. */
    var angleStep: Double

    /** Angle votes threshold. */
    var angleThresh: Int

    /** Minimal scale to detect. */
    var minScale: Double

    /** Maximal scale to detect. */
    var maxScale: Double

    /** Scale step. */
    var scaleStep: Double

    /** Scale votes threshold. */
    var scaleThresh: Int

    /** Position votes threshold. */
    var posThresh: Int
}

/** Creates a position-only generalized Hough detector (`cv::createGeneralizedHoughBallard`). */
expect fun createGeneralizedHoughBallard(): GeneralizedHoughBallard

/** Creates a position/rotation/scale generalized Hough detector (`cv::createGeneralizedHoughGuil`). */
expect fun createGeneralizedHoughGuil(): GeneralizedHoughGuil

/**
 * Parameters of the `filter2D` variant that takes a `Filter2DParams` holder
 * (`cv::Filter2DParams`). Pure Kotlin value holder — no native handle; the
 * kernel travels as a separate argument of the future filter2D-with-params
 * API. Defaults match the C++ struct.
 */
data class Filter2DParams(
    /** Anchor x coordinate; -1 means the kernel center. */
    var anchorX: Int = -1,
    /** Anchor y coordinate; -1 means the kernel center. */
    var anchorY: Int = -1,
    /** Pixel extrapolation method ([BorderTypes]); [BorderTypes.REFLECT_101] is BORDER_DEFAULT. */
    var borderType: Int = BorderTypes.REFLECT_101,
    /** Desired destination depth; -1 keeps the source depth. */
    var ddepth: Int = -1,
    /** Optional scale applied to the filter result. */
    var scale: Double = 1.0,
    /** Optional value added to the filtered pixels before storing them. */
    var shift: Double = 0.0,
)

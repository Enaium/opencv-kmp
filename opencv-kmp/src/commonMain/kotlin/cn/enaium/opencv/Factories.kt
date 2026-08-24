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

/** OpenCV version string backing this binding, e.g. `"5.0.0"`. */
expect val opencvVersion: String

/** Creates an empty matrix when [rows] is 0, otherwise a rows x cols one. */
expect fun mat(rows: Int = 0, cols: Int = 0, type: Int = MatType.CV_8UC1): Mat

/** Creates a rows x cols matrix with every element set to [fill]. */
expect fun mat(rows: Int, cols: Int, type: Int, fill: Scalar): Mat

/** `cv::Mat::zeros`. */
expect fun zeros(rows: Int, cols: Int, type: Int): Mat

/** `cv::Mat::ones` (for multi-channel types only channel 0 is set). */
expect fun ones(rows: Int, cols: Int, type: Int): Mat

/** Identity matrix. */
expect fun eye(rows: Int, cols: Int, type: Int): Mat

/**
 * Loads an image from [path]; returns null when the file cannot be decoded
 * (the native error text is available through [opencvLastError]).
 */
expect fun imread(path: String, flags: Int = ImreadFlags.COLOR): Mat?

/**
 * Saves [mat] to [path], choosing the codec from the extension; true on
 * success.
 */
expect fun imwrite(path: String, mat: Mat): Boolean

/** Encodes [mat] into the format named by [ext] (e.g. "png"). */
expect fun imencode(ext: String, mat: Mat): ByteArray

/** Decodes an encoded image from memory; null on failure. */
expect fun imdecode(data: ByteArray, flags: Int = ImreadFlags.COLOR): Mat?

/** Message of the last native OpenCV error on this thread, or null. */
expect val opencvLastError: String?

/** OpenCV's imencode expects the extension with a leading dot. */
internal fun normalizeImageExtension(ext: String): String =
    if (ext.startsWith('.')) ext else ".$ext"
// =========================================================================
// kernels / transforms
// =========================================================================

/** `cv::getStructuringElement` ([MorphShapes]). */
expect fun getStructuringElement(shape: Int, width: Int, height: Int): Mat

/** `cv::getGaussianKernel`; [ksize] must be odd and positive. */
expect fun getGaussianKernel(ksize: Int, sigma: Double): Mat

/** Affine transform mapping three source points onto three destinations. */
expect fun getAffineTransform(src: List<Point>, dst: List<Point>): Mat

/** Inverts a 2x3 affine transform produced by [getAffineTransform]. */
expect fun invertAffineTransform(transform: Mat): Mat

/** Perspective transform mapping four source points onto four destinations. */
expect fun getPerspectiveTransform(src: List<Point>, dst: List<Point>): Mat

/** Rotation matrix around [center] by [angle] degrees with uniform scale. */
expect fun getRotationMatrix2D(center: Point, angle: Double, scale: Double): Mat

/** `cv::createHanningWindow`. */
expect fun hanningWindow(width: Int, height: Int, type: Int = CV_32F): Mat

/** Merges single-channel matrices into one multi-channel matrix. */
expect fun merge(channels: List<Mat>): Mat

/** Number of worker threads OpenCV will use for parallel loops. */
expect val opencvNumThreads: Int

/** Caps OpenCV's parallel loop threads; 0 or less restores the default. */
expect fun setNumThreads(count: Int)

/** Verbose OpenCV build information (versions, third-party libs). */
expect val opencvBuildInformation: String

/** Seeds the global random generator backing [Mat.randu]/[Mat.randn]. */
expect fun setRNGSeed(seed: Long)

// =========================================================================
// codecs / environment
// =========================================================================

/** Number of frames inside an image file (animated formats), else 1. */
expect fun imcount(path: String): Int

/** Whether the image file at [path] can be decoded. */
expect fun haveImageReader(path: String): Boolean

/** Whether the image file at [path] can be re-encoded. */
expect fun haveImageWriter(path: String): Boolean

/**
 * Encodes [mat] into [ext] format passing codec parameters as a flat
 * `[ImwriteParams.JPEG_QUALITY, 85, ...]` list.
 */
expect fun imencodeParams(ext: String, mat: Mat, params: List<Int>): ByteArray

/** Writes [mat] to [path] with codec parameters; true on success. */
expect fun imwriteParams(path: String, mat: Mat, params: List<Int>): Boolean

/** Smallest efficient DFT size for the given dimension. */
expect fun getOptimalDftSize(size: Int): Int

/** Area enclosed by a contour; sign follows winding order. */
fun contourArea(contour: List<Point>): Double {
    require(contour.size >= 3) { "a contour needs at least 3 points" }
    return contourBuffer(contour, ::contourAreaNative)
}

/** Perimeter of a contour. */
fun arcLength(contour: List<Point>, closed: Boolean): Double =
    contourBuffer(contour) { data -> arcLengthNative(data, closed) }

/** Axis-aligned bounding box of a contour. */
fun boundingRect(contour: List<Point>): Rect = contourRect(contour)

/** Douglas-Peucker simplification of a contour. */
expect fun approxPolyDP(contour: List<Point>, epsilon: Double, closed: Boolean): List<Point>

/** Minimum-area rotated rectangle enclosing a contour. */
expect fun minAreaRect(contour: List<Point>): RotatedRect

/** Minimum enclosing circle of a contour. */
expect fun minEnclosingCircle(contour: List<Point>): Circle

// =========================================================================
// CLAHE
// =========================================================================

/** Contrast Limited Adaptive Histogram Equalization (`cv::CLAHE`). */
interface CLAHE : AutoCloseable {

    /** Equalizes [src] (8-bit single channel) and returns the result. */
    fun apply(src: Mat): Mat

    /** Updates the contrast clipping threshold. */
    fun setClipLimit(clipLimit: Double)

    override fun close()
}

/** Creates a CLAHE processor; defaults match `cv::createCLAHE()`. */
expect fun createCLAHE(
    clipLimit: Double = 2.0,
    tileGridSize: Size = Size(width = 8, height = 8),
): CLAHE

// =========================================================================
// internal helpers shared with the platform implementations
// =========================================================================

/**
 * Runs [body] against the wire-format encoding of exactly one [contour];
 * per-contour geometry is exposed through this flat buffer.
 */
internal inline fun <R> contourBuffer(
    contour: List<Point>,
    body: (data: ByteArray) -> R,
): R = body(ContourCodec.encode(listOf(contour)))

internal expect fun contourAreaNative(data: ByteArray): Double

internal expect fun arcLengthNative(data: ByteArray, closed: Boolean): Double

internal expect fun contourRect(contour: List<Point>): Rect

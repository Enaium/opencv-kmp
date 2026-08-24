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

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

/** Default frame count passed to the animated readers when none is given. */
private const val ANIMATION_DEFAULT_FRAME_COUNT = 32767 // INT16_MAX, like cv::imreadanimation

/**
 * A multi-frame animation (GIF, APNG, WebP, ...) wrapping the native
 * `cv::Animation` struct: frames, per-frame durations in milliseconds, a
 * loop count, a background color and an optional still image.
 *
 * Every getter that returns a [Mat] (or a [MatOfInt]) hands ownership of
 * the wrapped matrix to the caller — close it when done.
 */
interface Animation : AutoCloseable {

    /** Number of times the animation loops; 0 means loop forever. */
    fun getLoop(): Int

    /** Sets the loop count; values outside 0..0xffff are reset to 0. */
    fun setLoop(loop: Int)

    /** Background color in BGRA order (transparent by default). */
    fun getBgColor(): Scalar

    /** Sets the background color in BGRA order. */
    fun setBgColor(bgColor: Scalar)

    /**
     * Per-frame durations in milliseconds as a [MatOfInt]; the wrapped Mat
     * is owned by the caller.
     */
    fun getDurations(): MatOfInt

    /** Sets the per-frame durations in milliseconds. */
    fun setDurations(durations: MatOfInt)

    /**
     * The animation frames; every returned Mat is an independent handle the
     * caller owns.
     */
    fun getImages(): List<Mat>

    /** Replaces the frames (deep-copied into the animation). */
    fun setImages(images: List<Mat>)

    /** Still image used by formats without animation support, or null. */
    fun getStillImage(): Mat?

    /** Sets the still image (deep-copied into the animation). */
    fun setStillImage(image: Mat)

    override fun close()
}

/**
 * Creates an empty [Animation]; [loopCount] 0 means infinite looping and
 * negative values (or values above 0xffff) are reset to 0.
 */
expect fun createAnimation(loopCount: Int = 0, bgColor: Scalar = Scalar()): Animation

// =========================================================================
// multi-page / multi-image codecs
// =========================================================================

/**
 * Loads every page of a multi-page image file (TIFF pages, animated GIF/AVIF
 * frames, ...) into a list of Mats; null when the file cannot be decoded.
 * For single-image formats the list holds exactly one Mat.
 */
expect fun imreadmulti(filename: String, flags: Int = ImreadFlags.COLOR): List<Mat>?

/**
 * Loads [count] pages starting at [start] from a multi-page image file;
 * null when the file cannot be decoded or [start] lies beyond its pages.
 */
expect fun imreadmulti(
    filename: String,
    start: Int,
    count: Int,
    flags: Int = ImreadFlags.ANYCOLOR,
): List<Mat>?

/**
 * Writes every image of [images] into one multi-page file (e.g. TIFF);
 * true on success. Formats without multi-image support write only the first
 * image or fail depending on the codec.
 */
expect fun imwritemulti(
    filename: String,
    images: List<Mat>,
    params: List<Int> = emptyList(),
): Boolean

/**
 * Decodes a multi-page image from the in-memory buffer [buf] (a Mat wrapping
 * the encoded bytes), returning pages in [range]; null on failure.
 */
expect fun imdecodemulti(
    buf: Mat,
    flags: Int,
    range: Range = Range(0, Int.MAX_VALUE),
): List<Mat>?

/**
 * Encodes every image of [images] into a multi-page memory buffer (e.g.
 * TIFF); throws [OpenCVException] when the format does not support it.
 */
expect fun imencodemulti(
    ext: String,
    images: List<Mat>,
    params: List<Int> = emptyList(),
): ByteArray

// =========================================================================
// animated image codecs
// =========================================================================

/**
 * Loads frames from an animated image file into [animation]
 * (cv::imreadanimation); true on success. Frames from [start] up to
 * [start]+[count] are read; count defaults to 32767 (all frames).
 */
expect fun imreadanimation(
    filename: String,
    animation: Animation,
    start: Int = 0,
    count: Int = ANIMATION_DEFAULT_FRAME_COUNT,
): Boolean

/**
 * Decodes frames from an animated image buffer into [animation]
 * (cv::imdecodeanimation); true on success.
 */
expect fun imdecodeanimation(
    data: ByteArray,
    animation: Animation,
    start: Int = 0,
    count: Int = ANIMATION_DEFAULT_FRAME_COUNT,
): Boolean

/**
 * Saves [animation] to a file in an animated format chosen by the extension
 * (GIF, APNG, WebP, ...); true on success.
 */
expect fun imwriteanimation(
    filename: String,
    animation: Animation,
    params: List<Int> = emptyList(),
): Boolean

/**
 * Encodes [animation] into a memory buffer in the format named by [ext]
 * (e.g. "gif", ".png" for APNG); throws [OpenCVException] on failure.
 */
expect fun imencodeanimation(
    ext: String,
    animation: Animation,
    params: List<Int> = emptyList(),
): ByteArray

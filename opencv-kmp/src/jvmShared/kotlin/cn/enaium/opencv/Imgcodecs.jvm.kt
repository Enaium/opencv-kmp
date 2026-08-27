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


/** Unwraps a platform Animation into its raw JNI handle. */
private fun Animation.nativeHandle(): Long =
    (this as? JvmAnimation)?.check()
        ?: throw IllegalArgumentException("animation belongs to another platform backend")

internal class JvmAnimation(@Volatile private var handle: Long) : Animation {

    internal fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Animation is closed")

    override fun getLoop(): Int = JniImgcodecs.animationGetLoopCount(check())

    override fun setLoop(loop: Int) {
        JniImgcodecs.animationSetLoopCount(check(), loop)
    }

    override fun getBgColor(): Scalar {
        val values = JniImgcodecs.animationGetBgColor(check())
        return Scalar(values[0], values[1], values[2], values[3])
    }

    override fun setBgColor(bgColor: Scalar) {
        JniImgcodecs.animationSetBgColor(check(), bgColor.v0, bgColor.v1, bgColor.v2, bgColor.v3)
    }

    override fun getDurations(): MatOfInt =
        MatOfInt(jvmMat(JniImgcodecs.animationGetDurations(check()), "getDurations"))

    override fun setDurations(durations: MatOfInt) {
        if (!JniImgcodecs.animationSetDurations(check(), handleOf(durations.mat))) {
            throw OpenCVException("setDurations", lastNativeError())
        }
    }

    override fun getImages(): List<Mat> =
        JniImgcodecs.animationGetFrames(check())?.map(::JvmMat) ?: emptyList()

    override fun setImages(images: List<Mat>) {
        val handles = images.map { handleOf(it) }.toLongArray()
        if (!JniImgcodecs.animationSetFrames(check(), handles)) {
            throw OpenCVException("setImages", lastNativeError())
        }
    }

    override fun getStillImage(): Mat? =
        JniImgcodecs.animationGetStillImage(check()).takeIf { it != 0L }?.let(::JvmMat)

    override fun setStillImage(image: Mat) {
        if (!JniImgcodecs.animationSetStillImage(check(), handleOf(image))) {
            throw OpenCVException("setStillImage", lastNativeError())
        }
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniImgcodecs.animationRelease(h)
        }
    }
}

actual fun createAnimation(loopCount: Int, bgColor: Scalar): Animation {
    val handle = JniImgcodecs.animationCreate(
        loopCount, bgColor.v0, bgColor.v1, bgColor.v2, bgColor.v3,
    )
    if (handle == 0L) throw OpenCVException("createAnimation", lastNativeError())
    return JvmAnimation(handle)
}

// =========================================================================
// multi-page / multi-image codecs
// =========================================================================

actual fun imreadmulti(filename: String, flags: Int): List<Mat>? =
    JniImgcodecs.imreadmulti(filename, flags)?.map(::JvmMat)

actual fun imreadmulti(filename: String, start: Int, count: Int, flags: Int): List<Mat>? =
    JniImgcodecs.imreadmultiRange(filename, start, count, flags)?.map(::JvmMat)

actual fun imwritemulti(filename: String, images: List<Mat>, params: List<Int>): Boolean =
    JniImgcodecs.imwritemulti(
        filename,
        images.map { handleOf(it) }.toLongArray(),
        params.toIntArray(),
    )

actual fun imdecodemulti(buf: Mat, flags: Int, range: Range): List<Mat>? =
    JniImgcodecs.imdecodemulti(handleOf(buf), flags, range.start, range.end)?.map(::JvmMat)

actual fun imencodemulti(ext: String, images: List<Mat>, params: List<Int>): ByteArray {
    if (images.isEmpty()) throw OpenCVException("imencodemulti", "empty image list")
    return JniImgcodecs.imencodemulti(
        normalizeImageExtension(ext),
        images.map { handleOf(it) }.toLongArray(),
        params.toIntArray(),
    )
}

// =========================================================================
// animated image codecs
// =========================================================================

actual fun imreadanimation(
    filename: String,
    animation: Animation,
    start: Int,
    count: Int,
): Boolean = JniImgcodecs.imreadanimation(filename, animation.nativeHandle(), start, count)

actual fun imdecodeanimation(
    data: ByteArray,
    animation: Animation,
    start: Int,
    count: Int,
): Boolean = JniImgcodecs.imdecodeanimation(data, animation.nativeHandle(), start, count)

actual fun imwriteanimation(
    filename: String,
    animation: Animation,
    params: List<Int>,
): Boolean = JniImgcodecs.imwriteanimation(filename, animation.nativeHandle(), params.toIntArray())

actual fun imencodeanimation(
    ext: String,
    animation: Animation,
    params: List<Int>,
): ByteArray =
    JniImgcodecs.imencodeanimation(
        normalizeImageExtension(ext),
        animation.nativeHandle(),
        params.toIntArray(),
    )

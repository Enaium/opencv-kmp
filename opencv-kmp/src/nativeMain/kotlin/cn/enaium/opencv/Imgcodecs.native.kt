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
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package cn.enaium.opencv

import cvk.cvk_animation_create
import cvk.cvk_animation_get_bgcolor
import cvk.cvk_animation_get_durations
import cvk.cvk_animation_get_frames
import cvk.cvk_animation_get_loop_count
import cvk.cvk_animation_get_still_image
import cvk.cvk_animation_release
import cvk.cvk_animation_set_bgcolor
import cvk.cvk_animation_set_durations
import cvk.cvk_animation_set_frames
import cvk.cvk_animation_set_loop_count
import cvk.cvk_animation_set_still_image
import cvk.cvk_animation_t
import cvk.cvk_imdecodeanimation
import cvk.cvk_imdecodemulti
import cvk.cvk_imencodemulti
import cvk.cvk_imencodeanimation
import cvk.cvk_imreadanimation
import cvk.cvk_imreadmulti
import cvk.cvk_imreadmulti_range
import cvk.cvk_imwriteanimation
import cvk.cvk_imwritemulti
import cvk.cvk_mat_create
import cvk.cvk_mat_data
import cvk.cvk_mat_list
import cvk.cvk_mat_list_release
import cvk.cvk_mat_release
import cvk.cvk_mat_t
import cvk.cvk_free_buffer
import cvk.cvk_last_error
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.convert
import kotlinx.cinterop.get
import kotlinx.cinterop.ptr
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.pointed
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import kotlinx.cinterop.useContents
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import kotlin.concurrent.Volatile
import platform.posix.size_t
import platform.posix.size_tVar

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

/**
 * Wraps a cvk Mat-list into Kotlin Mats: each handle becomes an independent
 * [NativeMat] owned by the caller, then the pointer array is released.
 */
private fun readMatList(list: CPointer<cvk_mat_list>?): List<Mat> {
    if (list == null) return emptyList()
    val items = list.pointed.items
    val count = list.pointed.count.toInt()
    if (items == null || count <= 0) {
        cvk_mat_list_release(list)
        return emptyList()
    }
    val result = ArrayList<Mat>(count)
    for (i in 0 until count) {
        val handle = items[i] ?: continue
        result.add(NativeMat(handle))
    }
    cvk_mat_list_release(list)
    return result
}

/** Runs [body] against the raw pointer/length of [values] (null/0 when empty). */
private inline fun <R> withParams(
    values: List<Int>,
    body: (pointer: CPointer<IntVar>?, length: size_t) -> R,
): R =
    if (values.isEmpty()) {
        body(null, 0.convert<size_t>())
    } else {
        values.toIntArray().usePinned { pinned ->
            body(pinned.addressOf(0), values.size.convert<size_t>())
        }
    }

/** Unwraps a platform Animation into its raw cvk handle. */
private fun Animation.nativeHandle(): CPointer<cvk_animation_t> =
    (this as? NativeAnimation)?.check()
        ?: throw IllegalArgumentException("animation belongs to another platform backend")

internal class NativeAnimation(@Volatile private var raw: CPointer<cvk_animation_t>?) : Animation {

    internal fun check(): CPointer<cvk_animation_t> =
        raw ?: throw IllegalStateException("Animation is closed")

    override fun getLoop(): Int = cvk_animation_get_loop_count(check())

    override fun setLoop(loop: Int) {
        cvk_animation_set_loop_count(check(), loop)
    }

    override fun getBgColor(): Scalar =
        cvk_animation_get_bgcolor(check()).useContents { Scalar(v0, v1, v2, v3) }

    override fun setBgColor(bgColor: Scalar) {
        cvk_animation_set_bgcolor(check(), bgColor.toCvk())
    }

    override fun getDurations(): MatOfInt =
        MatOfInt(nativeMat(cvk_animation_get_durations(check()), "getDurations"))

    override fun setDurations(durations: MatOfInt) {
        if (cvk_animation_set_durations(check(), durations.mat.nativeHandle()) == 0) {
            throw OpenCVException("setDurations", lastNativeError())
        }
    }

    override fun getImages(): List<Mat> = readMatList(cvk_animation_get_frames(check()))

    override fun setImages(images: List<Mat>) {
        val animation = check()
        if (images.isEmpty()) {
            cvk_animation_set_frames(animation, null, 0.convert<size_t>())
            return
        }
        memScoped {
            val handles = allocArray<CPointerVar<cvk_mat_t>>(images.size)
            images.forEachIndexed { index, image -> handles[index] = image.nativeHandle() }
            if (cvk_animation_set_frames(animation, handles, images.size.convert<size_t>()) == 0) {
                throw OpenCVException("setImages", lastNativeError())
            }
        }
    }

    override fun getStillImage(): Mat? =
        cvk_animation_get_still_image(check())?.let { NativeMat(it) }

    override fun setStillImage(image: Mat) {
        if (cvk_animation_set_still_image(check(), image.nativeHandle()) == 0) {
            throw OpenCVException("setStillImage", lastNativeError())
        }
    }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_animation_release(handle)
    }
}

actual fun createAnimation(loopCount: Int, bgColor: Scalar): Animation =
    NativeAnimation(
        cvk_animation_create(loopCount, bgColor.v0, bgColor.v1, bgColor.v2, bgColor.v3)
            ?: throw OpenCVException("createAnimation", lastNativeError()),
    )

// =========================================================================
// multi-page / multi-image codecs
// =========================================================================

actual fun imreadmulti(filename: String, flags: Int): List<Mat>? =
    cvk_imreadmulti(filename, flags)?.let(::readMatList)

actual fun imreadmulti(filename: String, start: Int, count: Int, flags: Int): List<Mat>? =
    cvk_imreadmulti_range(filename, start, count, flags)?.let(::readMatList)

actual fun imwritemulti(filename: String, images: List<Mat>, params: List<Int>): Boolean {
    if (images.isEmpty()) return false
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(images.size)
        images.forEachIndexed { index, image -> handles[index] = image.nativeHandle() }
        withParams(params) { pointer, length ->
            cvk_imwritemulti(filename, handles, images.size.convert<size_t>(), pointer, length) != 0
        }
    }
}

actual fun imdecodemulti(buf: Mat, flags: Int, range: Range): List<Mat>? =
    cvk_imdecodemulti(buf.nativeHandle(), flags, range.start, range.end)?.let(::readMatList)

actual fun imencodemulti(ext: String, images: List<Mat>, params: List<Int>): ByteArray {
    if (images.isEmpty()) throw OpenCVException("imencodemulti", "empty image list")
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(images.size)
        images.forEachIndexed { index, image -> handles[index] = image.nativeHandle() }
        val lengthVar = alloc<size_tVar>()
        val buffer = withParams(params) { pointer, length ->
            cvk_imencodemulti(
                normalizeImageExtension(ext), handles, images.size.convert<size_t>(),
                pointer, length, lengthVar.ptr,
            )
        } ?: throw OpenCVException("imencodemulti", lastNativeError())
        try {
            buffer.readBytes(lengthVar.value.toInt())
        } finally {
            cvk_free_buffer(buffer)
        }
    }
}

// =========================================================================
// animated image codecs
// =========================================================================

actual fun imreadanimation(
    filename: String,
    animation: Animation,
    start: Int,
    count: Int,
): Boolean = cvk_imreadanimation(filename, animation.nativeHandle(), start, count) != 0

actual fun imdecodeanimation(
    data: ByteArray,
    animation: Animation,
    start: Int,
    count: Int,
): Boolean {
    val animationHandle = animation.nativeHandle()
    // Wrap the encoded bytes in a CV_8UC1 Mat (type id 0) mirroring
    // cvk_imdecode's internal Mat1b.
    val buf = cvk_mat_create(1, data.size, 0)
        ?: throw OpenCVException("imdecodeanimation", lastNativeError())
    try {
        if (data.isNotEmpty()) {
            data.usePinned { pinned ->
                platform.posix.memcpy(
                    cvk_mat_data(buf), pinned.addressOf(0), data.size.convert<size_t>(),
                )
            }
        }
        return cvk_imdecodeanimation(buf, animationHandle, start, count) != 0
    } finally {
        cvk_mat_release(buf)
    }
}

actual fun imwriteanimation(
    filename: String,
    animation: Animation,
    params: List<Int>,
): Boolean = withParams(params) { pointer, length ->
    cvk_imwriteanimation(filename, animation.nativeHandle(), pointer, length) != 0
}

actual fun imencodeanimation(
    ext: String,
    animation: Animation,
    params: List<Int>,
): ByteArray = memScoped {
    val lengthVar = alloc<size_tVar>()
    val buffer = withParams(params) { pointer, length ->
        cvk_imencodeanimation(
            normalizeImageExtension(ext), animation.nativeHandle(), pointer, length, lengthVar.ptr,
        )
    } ?: throw OpenCVException("imencodeanimation", lastNativeError())
    try {
        buffer.readBytes(lengthVar.value.toInt())
    } finally {
        cvk_free_buffer(buffer)
    }
}

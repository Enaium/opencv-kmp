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

import cvk.cvk_video_capture_create
import cvk.cvk_video_capture_create_file
import cvk.cvk_video_capture_create_file_params
import cvk.cvk_video_capture_create_index
import cvk.cvk_video_capture_create_index_params
import cvk.cvk_video_capture_delete
import cvk.cvk_video_capture_get
import cvk.cvk_video_capture_get_backend_name
import cvk.cvk_video_capture_get_exception_mode
import cvk.cvk_video_capture_grab
import cvk.cvk_video_capture_is_opened
import cvk.cvk_video_capture_open_file
import cvk.cvk_video_capture_open_file_params
import cvk.cvk_video_capture_open_index
import cvk.cvk_video_capture_open_index_params
import cvk.cvk_video_capture_read
import cvk.cvk_video_capture_release
import cvk.cvk_video_capture_retrieve
import cvk.cvk_video_capture_set
import cvk.cvk_video_capture_set_exception_mode
import cvk.cvk_video_capture_t
import cvk.cvk_video_writer_create
import cvk.cvk_video_writer_create_file
import cvk.cvk_video_writer_create_file_api
import cvk.cvk_video_writer_create_file_api_params
import cvk.cvk_video_writer_create_file_params
import cvk.cvk_video_writer_delete
import cvk.cvk_video_writer_get
import cvk.cvk_video_writer_get_backend_name
import cvk.cvk_video_writer_is_opened
import cvk.cvk_video_writer_open_file
import cvk.cvk_video_writer_open_file_api
import cvk.cvk_video_writer_open_file_api_params
import cvk.cvk_video_writer_open_file_params
import cvk.cvk_video_writer_release
import cvk.cvk_video_writer_set
import cvk.cvk_video_writer_t
import cvk.cvk_video_writer_write
import cvk.cvk_videoio_get_backends
import cvk.cvk_videoio_get_backend_name
import cvk.cvk_get_camera_plugin_version
import cvk.cvk_videoio_get_camera_backends
import cvk.cvk_get_stream_plugin_version
import cvk.cvk_get_stream_buffered_plugin_version
import cvk.cvk_videoio_get_stream_buffered_backends
import cvk.cvk_videoio_get_stream_backends
import cvk.cvk_get_writer_plugin_version
import cvk.cvk_videoio_get_writer_backends
import cvk.cvk_videoio_has_backend
import cvk.cvk_videoio_is_backend_built_in
import cvk.cvk_mat_t
import cvk.cvk_last_error
import kotlin.concurrent.Volatile
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value
import platform.posix.F_OK
import platform.posix.access
import platform.posix.remove

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

// =========================================================================
// VideoCapture
// =========================================================================

internal class NativeVideoCapture(
    @Volatile private var raw: CPointer<cvk_video_capture_t>?,
) : VideoCapture {

    private fun check(): CPointer<cvk_video_capture_t> =
        raw ?: throw IllegalStateException("VideoCapture is closed")

    override val isOpened: Boolean get() = cvk_video_capture_is_opened(check()) != 0

    override fun grab(): Boolean = cvk_video_capture_grab(check()) != 0

    override fun retrieve(flag: Int): Mat? =
        cvk_video_capture_retrieve(check(), flag)?.let { NativeMat(it) }

    override fun read(): Mat? = memScoped {
        val out = alloc<CPointerVar<cvk_mat_t>>()
        if (cvk_video_capture_read(check(), out.ptr) != 0) NativeMat(out.value!!) else null
    }

    override fun set(propId: Int, value: Double): Boolean =
        cvk_video_capture_set(check(), propId, value) != 0

    override fun get(propId: Int): Double = cvk_video_capture_get(check(), propId)

    override val backendName: String
        get() = cvk_video_capture_get_backend_name(check())?.toKString()
            ?: throw OpenCVException("VideoCapture.backendName", lastNativeError())

    override var exceptionMode: Boolean
        get() = cvk_video_capture_get_exception_mode(check()) != 0
        set(value) { cvk_video_capture_set_exception_mode(check(), if (value) 1 else 0) }

    override fun release() {
        raw?.let { cvk_video_capture_release(it) }
    }

    override fun open(index: Int, apiPreference: Int): Boolean =
        cvk_video_capture_open_index(check(), index, apiPreference) != 0

    override fun open(filename: String, apiPreference: Int): Boolean =
        cvk_video_capture_open_file(check(), filename, apiPreference) != 0

    override fun open(index: Int, apiPreference: Int, params: MatOfInt): Boolean =
        cvk_video_capture_open_index_params(check(), index, apiPreference, params.mat.nativeHandle()) != 0

    override fun open(filename: String, apiPreference: Int, params: MatOfInt): Boolean =
        cvk_video_capture_open_file_params(check(), filename, apiPreference, params.mat.nativeHandle()) != 0

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_video_capture_delete(h)
    }
}

// =========================================================================
// VideoWriter
// =========================================================================

internal class NativeVideoWriter(
    @Volatile private var raw: CPointer<cvk_video_writer_t>?,
) : VideoWriter {

    private fun check(): CPointer<cvk_video_writer_t> =
        raw ?: throw IllegalStateException("VideoWriter is closed")

    override val isOpened: Boolean get() = cvk_video_writer_is_opened(check()) != 0

    override fun write(image: Mat): Boolean =
        cvk_video_writer_write(check(), image.nativeHandle()) != 0

    override fun set(propId: Int, value: Double): Boolean =
        cvk_video_writer_set(check(), propId, value) != 0

    override fun get(propId: Int): Double = cvk_video_writer_get(check(), propId)

    override val backendName: String
        get() = cvk_video_writer_get_backend_name(check())?.toKString()
            ?: throw OpenCVException("VideoWriter.backendName", lastNativeError())

    override fun release() {
        raw?.let { cvk_video_writer_release(it) }
    }

    override fun open(filename: String, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): Boolean =
        cvk_video_writer_open_file(check(), filename, fourcc, fps, frameSize.width, frameSize.height, if (isColor) 1 else 0) != 0

    override fun open(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): Boolean =
        cvk_video_writer_open_file_api(check(), filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, if (isColor) 1 else 0) != 0

    override fun open(filename: String, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): Boolean =
        cvk_video_writer_open_file_params(check(), filename, fourcc, fps, frameSize.width, frameSize.height, params.mat.nativeHandle()) != 0

    override fun open(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): Boolean =
        cvk_video_writer_open_file_api_params(check(), filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, params.mat.nativeHandle()) != 0

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_video_writer_delete(h)
    }
}

// =========================================================================
// factories
// =========================================================================

actual fun videoCapture(): VideoCapture =
    NativeVideoCapture(cvk_video_capture_create() ?: throw OpenCVException("videoCapture", lastNativeError()))

actual fun videoCapture(index: Int, apiPreference: Int): VideoCapture =
    NativeVideoCapture(cvk_video_capture_create_index(index, apiPreference) ?: throw OpenCVException("videoCapture(index)", lastNativeError()))

actual fun videoCapture(filename: String, apiPreference: Int): VideoCapture =
    NativeVideoCapture(cvk_video_capture_create_file(filename, apiPreference) ?: throw OpenCVException("videoCapture(filename)", lastNativeError()))

actual fun videoCapture(index: Int, apiPreference: Int, params: MatOfInt): VideoCapture =
    NativeVideoCapture(cvk_video_capture_create_index_params(index, apiPreference, params.mat.nativeHandle()) ?: throw OpenCVException("videoCapture(index, params)", lastNativeError()))

actual fun videoCapture(filename: String, apiPreference: Int, params: MatOfInt): VideoCapture =
    NativeVideoCapture(cvk_video_capture_create_file_params(filename, apiPreference, params.mat.nativeHandle()) ?: throw OpenCVException("videoCapture(filename, params)", lastNativeError()))

actual fun videoWriter(): VideoWriter =
    NativeVideoWriter(cvk_video_writer_create() ?: throw OpenCVException("videoWriter", lastNativeError()))

actual fun videoWriter(filename: String, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): VideoWriter =
    NativeVideoWriter(cvk_video_writer_create_file(filename, fourcc, fps, frameSize.width, frameSize.height, if (isColor) 1 else 0) ?: throw OpenCVException("videoWriter", lastNativeError()))

actual fun videoWriter(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): VideoWriter =
    NativeVideoWriter(cvk_video_writer_create_file_api(filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, if (isColor) 1 else 0) ?: throw OpenCVException("videoWriter", lastNativeError()))

actual fun videoWriter(filename: String, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): VideoWriter =
    NativeVideoWriter(cvk_video_writer_create_file_params(filename, fourcc, fps, frameSize.width, frameSize.height, params.mat.nativeHandle()) ?: throw OpenCVException("videoWriter(params)", lastNativeError()))

actual fun videoWriter(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): VideoWriter =
    NativeVideoWriter(cvk_video_writer_create_file_api_params(filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, params.mat.nativeHandle()) ?: throw OpenCVException("videoWriter(api, params)", lastNativeError()))

// =========================================================================
// videoio_registry
// =========================================================================

actual fun videoioGetBackendName(api: Int): String =
    cvk_videoio_get_backend_name(api)?.toKString()
        ?: throw OpenCVException("videoioGetBackendName", lastNativeError())

private fun backendList(matPtr: CPointer<cvk_mat_t>?, operation: String): List<Int> {
    val m = matPtr ?: throw OpenCVException(operation, lastNativeError())
    return MatOfInt(NativeMat(m)).toList()
}

actual fun videoioGetBackends(): List<Int> =
    backendList(cvk_videoio_get_backends(), "videoioGetBackends")

actual fun videoioGetCameraBackends(): List<Int> =
    backendList(cvk_videoio_get_camera_backends(), "videoioGetCameraBackends")

actual fun videoioGetStreamBackends(): List<Int> =
    backendList(cvk_videoio_get_stream_backends(), "videoioGetStreamBackends")

actual fun videoioGetStreamBufferedBackends(): List<Int> =
    backendList(cvk_videoio_get_stream_buffered_backends(), "videoioGetStreamBufferedBackends")

actual fun videoioGetWriterBackends(): List<Int> =
    backendList(cvk_videoio_get_writer_backends(), "videoioGetWriterBackends")

actual fun videoioHasBackend(api: Int): Boolean = cvk_videoio_has_backend(api) != 0

actual fun videoioIsBackendBuiltIn(api: Int): Boolean = cvk_videoio_is_backend_built_in(api) != 0

private fun pluginVersion(
    query: (api: Int, abi: CPointer<IntVar>?, apiVersion: CPointer<IntVar>?) -> CPointer<ByteVar>?,
    operation: String,
    api: Int,
): BackendPluginVersion = memScoped {
    val abi = alloc<IntVar>()
    val apiVersion = alloc<IntVar>()
    val description = query(api, abi.ptr, apiVersion.ptr)
        ?: throw OpenCVException(operation, lastNativeError())
    BackendPluginVersion(description.toKString(), abi.value, apiVersion.value)
}

actual fun videoioGetCameraBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion({ api, abiP, apiV -> cvk_get_camera_plugin_version(api, abiP, apiV) }, "videoioGetCameraBackendPluginVersion", api)

actual fun videoioGetStreamBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion({ api, abiP, apiV -> cvk_get_stream_plugin_version(api, abiP, apiV) }, "videoioGetStreamBackendPluginVersion", api)

actual fun videoioGetStreamBufferedBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion({ api, abiP, apiV -> cvk_get_stream_buffered_plugin_version(api, abiP, apiV) }, "videoioGetStreamBufferedBackendPluginVersion", api)

actual fun videoioGetWriterBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion({ api, abiP, apiV -> cvk_get_writer_plugin_version(api, abiP, apiV) }, "videoioGetWriterBackendPluginVersion", api)

// =========================================================================
// test helpers
// =========================================================================

internal actual fun fileExists(path: String): Boolean = access(path, F_OK) == 0

internal actual fun deleteFile(path: String) {
    remove(path)
}

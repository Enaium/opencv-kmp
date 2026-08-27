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

import java.io.File

private fun lastNativeError(): String? = Jni.lastError()

/** Wraps a raw handle; throws with the native error text when it is 0. */
private fun matOrThrow(handle: Long, operation: String): Mat =
    if (handle != 0L) JvmMat(handle) else throw OpenCVException(operation, lastNativeError())

// =========================================================================
// VideoCapture
// =========================================================================

internal class JvmVideoCapture(private var handle: Long) : VideoCapture {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("VideoCapture is closed")

    override val isOpened: Boolean get() = JniVideoio.captureIsOpened(check())

    override fun grab(): Boolean = JniVideoio.captureGrab(check())

    override fun retrieve(flag: Int): Mat? {
        val h = JniVideoio.captureRetrieve(check(), flag)
        return if (h != 0L) JvmMat(h) else null
    }

    override fun read(): Mat? {
        val out = LongArray(1)
        return if (JniVideoio.captureRead(check(), out)) JvmMat(out[0]) else null
    }

    override fun set(propId: Int, value: Double): Boolean =
        JniVideoio.captureSet(check(), propId, value)

    override fun get(propId: Int): Double = JniVideoio.captureGet(check(), propId)

    override val backendName: String
        get() = JniVideoio.captureBackendName(check())
            ?: throw OpenCVException("VideoCapture.backendName", lastNativeError())

    override var exceptionMode: Boolean
        get() = JniVideoio.captureGetExceptionMode(check())
        set(value) { JniVideoio.captureSetExceptionMode(check(), value) }

    override fun release() {
        val h = handle
        if (h != 0L) JniVideoio.captureRelease(h)
    }

    override fun open(index: Int, apiPreference: Int): Boolean =
        JniVideoio.captureOpenIndex(check(), index, apiPreference)

    override fun open(filename: String, apiPreference: Int): Boolean =
        JniVideoio.captureOpenFile(check(), filename, apiPreference)

    override fun open(index: Int, apiPreference: Int, params: MatOfInt): Boolean =
        JniVideoio.captureOpenIndexParams(check(), index, apiPreference, handleOf(params.mat))

    override fun open(filename: String, apiPreference: Int, params: MatOfInt): Boolean =
        JniVideoio.captureOpenFileParams(check(), filename, apiPreference, handleOf(params.mat))

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideoio.captureDelete(h)
        }
    }
}

// =========================================================================
// VideoWriter
// =========================================================================

internal class JvmVideoWriter(private var handle: Long) : VideoWriter {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("VideoWriter is closed")

    override val isOpened: Boolean get() = JniVideoio.writerIsOpened(check())

    override fun write(image: Mat): Boolean = JniVideoio.writerWrite(check(), handleOf(image))

    override fun set(propId: Int, value: Double): Boolean =
        JniVideoio.writerSet(check(), propId, value)

    override fun get(propId: Int): Double = JniVideoio.writerGet(check(), propId)

    override val backendName: String
        get() = JniVideoio.writerBackendName(check())
            ?: throw OpenCVException("VideoWriter.backendName", lastNativeError())

    override fun release() {
        val h = handle
        if (h != 0L) JniVideoio.writerRelease(h)
    }

    override fun open(filename: String, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): Boolean =
        JniVideoio.writerOpenFile(check(), filename, fourcc, fps, frameSize.width, frameSize.height, isColor)

    override fun open(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): Boolean =
        JniVideoio.writerOpenFileApi(check(), filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, isColor)

    override fun open(filename: String, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): Boolean =
        JniVideoio.writerOpenFileParams(check(), filename, fourcc, fps, frameSize.width, frameSize.height, handleOf(params.mat))

    override fun open(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): Boolean =
        JniVideoio.writerOpenFileApiParams(check(), filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, handleOf(params.mat))

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideoio.writerDelete(h)
        }
    }
}

// =========================================================================
// factories
// =========================================================================

actual fun videoCapture(): VideoCapture =
    JvmVideoCapture(JniVideoio.captureCreate())

actual fun videoCapture(index: Int, apiPreference: Int): VideoCapture =
    JvmVideoCapture(JniVideoio.captureCreateIndex(index, apiPreference))

actual fun videoCapture(filename: String, apiPreference: Int): VideoCapture =
    JvmVideoCapture(JniVideoio.captureCreateFile(filename, apiPreference))

actual fun videoCapture(index: Int, apiPreference: Int, params: MatOfInt): VideoCapture =
    JvmVideoCapture(JniVideoio.captureCreateIndexParams(index, apiPreference, handleOf(params.mat)))

actual fun videoCapture(filename: String, apiPreference: Int, params: MatOfInt): VideoCapture =
    JvmVideoCapture(JniVideoio.captureCreateFileParams(filename, apiPreference, handleOf(params.mat)))

actual fun videoWriter(): VideoWriter =
    JvmVideoWriter(JniVideoio.writerCreate())

actual fun videoWriter(filename: String, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): VideoWriter =
    JvmVideoWriter(JniVideoio.writerCreateFile(filename, fourcc, fps, frameSize.width, frameSize.height, isColor))

actual fun videoWriter(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, isColor: Boolean): VideoWriter =
    JvmVideoWriter(JniVideoio.writerCreateFileApi(filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, isColor))

actual fun videoWriter(filename: String, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): VideoWriter =
    JvmVideoWriter(JniVideoio.writerCreateFileParams(filename, fourcc, fps, frameSize.width, frameSize.height, handleOf(params.mat)))

actual fun videoWriter(filename: String, apiPreference: Int, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): VideoWriter =
    JvmVideoWriter(JniVideoio.writerCreateFileApiParams(filename, apiPreference, fourcc, fps, frameSize.width, frameSize.height, handleOf(params.mat)))

// =========================================================================
// videoio_registry
// =========================================================================

actual fun videoioGetBackendName(api: Int): String =
    JniVideoio.videoioBackendName(api)
        ?: throw OpenCVException("videoioGetBackendName", lastNativeError())

private fun backendList(handle: Long, operation: String): List<Int> =
    MatOfInt(matOrThrow(handle, operation)).toList()

actual fun videoioGetBackends(): List<Int> =
    backendList(JniVideoio.videoioBackends(), "videoioGetBackends")

actual fun videoioGetCameraBackends(): List<Int> =
    backendList(JniVideoio.videoioCameraBackends(), "videoioGetCameraBackends")

actual fun videoioGetStreamBackends(): List<Int> =
    backendList(JniVideoio.videoioStreamBackends(), "videoioGetStreamBackends")

actual fun videoioGetStreamBufferedBackends(): List<Int> =
    backendList(JniVideoio.videoioStreamBufferedBackends(), "videoioGetStreamBufferedBackends")

actual fun videoioGetWriterBackends(): List<Int> =
    backendList(JniVideoio.videoioWriterBackends(), "videoioGetWriterBackends")

actual fun videoioHasBackend(api: Int): Boolean = JniVideoio.videoioHasBackend(api)

actual fun videoioIsBackendBuiltIn(api: Int): Boolean = JniVideoio.videoioIsBackendBuiltIn(api)

private fun pluginVersion(
    query: (api: Int, abi: IntArray, apiVersion: IntArray) -> String?,
    operation: String,
    api: Int,
): BackendPluginVersion {
    val abi = IntArray(1)
    val apiVersion = IntArray(1)
    val description = query(api, abi, apiVersion)
        ?: throw OpenCVException(operation, lastNativeError())
    return BackendPluginVersion(description, abi[0], apiVersion[0])
}

actual fun videoioGetCameraBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion(JniVideoio::videoioCameraBackendPluginVersion, "videoioGetCameraBackendPluginVersion", api)

actual fun videoioGetStreamBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion(JniVideoio::videoioStreamBackendPluginVersion, "videoioGetStreamBackendPluginVersion", api)

actual fun videoioGetStreamBufferedBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion(JniVideoio::videoioStreamBufferedBackendPluginVersion, "videoioGetStreamBufferedBackendPluginVersion", api)

actual fun videoioGetWriterBackendPluginVersion(api: Int): BackendPluginVersion =
    pluginVersion(JniVideoio::videoioWriterBackendPluginVersion, "videoioGetWriterBackendPluginVersion", api)

// =========================================================================
// test helpers
// =========================================================================

internal actual fun fileExists(path: String): Boolean = File(path).exists()

internal actual fun deleteFile(path: String) {
    File(path).delete()
}

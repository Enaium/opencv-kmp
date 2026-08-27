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
 * Video capturing from video files, image sequences or cameras
 * (`cv::VideoCapture`).
 *
 * A [VideoCapture] owns one native handle; call [close] to free it. The
 * stream is closed by [release], which leaves the object reusable for a
 * later [open]. Reading an unopened capture never throws: [grab],
 * [retrieve] and [read] report failure the same way the C++ API does
 * (`false` / `null`), and [get] returns [VideoCaptureProperties.CAP_PROP_UNKNOWN].
 */
interface VideoCapture : AutoCloseable {
    /** True when video capturing has been initialized. */
    val isOpened: Boolean

    /**
     * Grabs the next frame without decoding it. Use with [retrieve] in
     * multi-camera setups.
     */
    fun grab(): Boolean

    /**
     * Decodes and returns the just-grabbed frame, or `null` when no frame
     * has been grabbed (end of stream, camera disconnected).
     *
     * @param flag frame index or a driver-specific flag (default 0)
     */
    fun retrieve(flag: Int = 0): Mat?

    /**
     * Grabs, decodes and returns the next video frame, or `null` when no
     * frame is available (equivalent to `grab()` + `retrieve()`).
     */
    fun read(): Mat?

    /** Sets property [propId] to [value]; false when unsupported. */
    fun set(propId: Int, value: Double): Boolean

    /**
     * Returns the property [propId] value, or
     * [VideoCaptureProperties.CAP_PROP_UNKNOWN] when unsupported or closed.
     */
    fun get(propId: Int): Double

    /** Used backend API name. Requires an opened stream. */
    val backendName: String

    /**
     * Exceptions mode: when enabled, failing operations raise
     * [OpenCVException] instead of returning failure codes.
     */
    var exceptionMode: Boolean

    /** Closes the stream; the handle stays valid for a later [open]. */
    fun release()

    /** Opens camera [index] (see [VideoCaptureAPIs]). */
    fun open(index: Int, apiPreference: Int = VideoCaptureAPIs.CAP_ANY): Boolean

    /** Opens [filename] (file, image sequence, or URL). */
    fun open(filename: String, apiPreference: Int = VideoCaptureAPIs.CAP_ANY): Boolean

    /** Camera variant with extra [params] (propId/value pairs). */
    fun open(index: Int, apiPreference: Int, params: MatOfInt): Boolean

    /** Filename variant with extra [params] (propId/value pairs). */
    fun open(filename: String, apiPreference: Int, params: MatOfInt): Boolean

    /** Frees the native handle; [close] is idempotent. */
    override fun close()
}

/**
 * Video writer for video files or image sequences (`cv::VideoWriter`).
 *
 * A [VideoWriter] owns one native handle; call [close] to free it. [release]
 * closes the output file and leaves the object reusable for a later [open].
 */
interface VideoWriter : AutoCloseable {
    /** True when the writer has been successfully initialized. */
    val isOpened: Boolean

    /**
     * Writes the next video frame (BGR color images of the size given at
     * open time). Returns false on backend failures or closed writers.
     */
    fun write(image: Mat): Boolean

    /** Sets property [propId] to [value]; false when unsupported. */
    fun set(propId: Int, value: Double): Boolean

    /** Returns the property [propId] value, or [VideoWriterProperties.VIDEOWRITER_PROP_UNKNOWN]. */
    fun get(propId: Int): Double

    /** Used backend API name. Requires an opened writer. */
    val backendName: String

    /** Closes the output file; the handle stays valid for a later [open]. */
    fun release()

    /** Opens [filename] with the given codec/framerate/size. */
    fun open(
        filename: String,
        fourcc: Int,
        fps: Double,
        frameSize: Size,
        isColor: Boolean = true,
    ): Boolean

    /** Backend variant (see [VideoCaptureAPIs]). */
    fun open(
        filename: String,
        apiPreference: Int,
        fourcc: Int,
        fps: Double,
        frameSize: Size,
        isColor: Boolean = true,
    ): Boolean

    /** Variant with extra encoder [params] (propId/value pairs). */
    fun open(filename: String, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): Boolean

    /** Backend + extra encoder [params] variant. */
    fun open(
        filename: String,
        apiPreference: Int,
        fourcc: Int,
        fps: Double,
        frameSize: Size,
        params: MatOfInt,
    ): Boolean

    /** Frees the native handle; [close] is idempotent. */
    override fun close()

    companion object {
        /**
         * Concatenates 4 chars to a fourcc code (`cv::VideoWriter::fourcc`).
         * E.g. `fourcc('M', 'J', 'P', 'G')` selects the motion-jpeg codec.
         */
        fun fourcc(c1: Char, c2: Char, c3: Char, c4: Char): Int =
            (c1.code and 0xFF) or
                ((c2.code and 0xFF) shl 8) or
                ((c3.code and 0xFF) shl 16) or
                ((c4.code and 0xFF) shl 24)
    }
}

/**
 * Result of a videoio plugin version query
 * (Java `Videoio.get*BackendPluginVersion(api, int[] abi, int[] api)`).
 */
data class BackendPluginVersion(
    val description: String,
    val versionAbi: Int,
    val versionApi: Int,
)

/**
 * Static `videoio_registry` queries mirroring the Java `Videoio` class.
 */
object Videoio {
    /** Returns the backend API name or `"UnknownVideoAPI(xxx)"`. */
    fun getBackendName(api: Int): String = videoioGetBackendName(api)

    /** All available backends. */
    fun getBackends(): List<Int> = videoioGetBackends()

    /** Backends usable via `VideoCapture(index)`. */
    fun getCameraBackends(): List<Int> = videoioGetCameraBackends()

    /** Backends usable via `VideoCapture(filename)`. */
    fun getStreamBackends(): List<Int> = videoioGetStreamBackends()

    /** Backends usable via `VideoCapture(buffer)`. */
    fun getStreamBufferedBackends(): List<Int> = videoioGetStreamBufferedBackends()

    /** Backends usable via `VideoWriter()`. */
    fun getWriterBackends(): List<Int> = videoioGetWriterBackends()

    /** True when backend [api] is available. */
    fun hasBackend(api: Int): Boolean = videoioHasBackend(api)

    /** True when backend [api] is built in (false = plugin). */
    fun isBackendBuiltIn(api: Int): Boolean = videoioIsBackendBuiltIn(api)

    /** Description and ABI/API version of the camera interface plugin. */
    fun getCameraBackendPluginVersion(api: Int): BackendPluginVersion =
        videoioGetCameraBackendPluginVersion(api)

    /** Description and ABI/API version of the stream capture interface plugin. */
    fun getStreamBackendPluginVersion(api: Int): BackendPluginVersion =
        videoioGetStreamBackendPluginVersion(api)

    /** Description and ABI/API version of the buffer capture interface plugin. */
    fun getStreamBufferedBackendPluginVersion(api: Int): BackendPluginVersion =
        videoioGetStreamBufferedBackendPluginVersion(api)

    /** Description and ABI/API version of the writer interface plugin. */
    fun getWriterBackendPluginVersion(api: Int): BackendPluginVersion =
        videoioGetWriterBackendPluginVersion(api)
}

// ---------------------------------------------------------------------------
// factories
// ---------------------------------------------------------------------------

/** Default (unopened) [VideoCapture]. */
expect fun videoCapture(): VideoCapture

/** Opens camera [index] with the given backend preference. */
expect fun videoCapture(index: Int, apiPreference: Int = VideoCaptureAPIs.CAP_ANY): VideoCapture

/** Opens [filename] (video file, image sequence, or URL). */
expect fun videoCapture(filename: String, apiPreference: Int = VideoCaptureAPIs.CAP_ANY): VideoCapture

/** Camera variant with extra [params] (propId/value pairs). */
expect fun videoCapture(index: Int, apiPreference: Int, params: MatOfInt): VideoCapture

/** Filename variant with extra [params] (propId/value pairs). */
expect fun videoCapture(filename: String, apiPreference: Int, params: MatOfInt): VideoCapture

/** Default (unopened) [VideoWriter]. */
expect fun videoWriter(): VideoWriter

/** Opens [filename] with the given codec/framerate/size. */
expect fun videoWriter(
    filename: String,
    fourcc: Int,
    fps: Double,
    frameSize: Size,
    isColor: Boolean = true,
): VideoWriter

/** Backend variant (see [VideoCaptureAPIs]). */
expect fun videoWriter(
    filename: String,
    apiPreference: Int,
    fourcc: Int,
    fps: Double,
    frameSize: Size,
    isColor: Boolean = true,
): VideoWriter

/** Variant with extra encoder [params] (propId/value pairs). */
expect fun videoWriter(filename: String, fourcc: Int, fps: Double, frameSize: Size, params: MatOfInt): VideoWriter

/** Backend + extra encoder [params] variant. */
expect fun videoWriter(
    filename: String,
    apiPreference: Int,
    fourcc: Int,
    fps: Double,
    frameSize: Size,
    params: MatOfInt,
): VideoWriter

// ---------------------------------------------------------------------------
// videoio_registry (native-backed statics)
// ---------------------------------------------------------------------------

/** [Videoio.getBackendName] backend. */
expect fun videoioGetBackendName(api: Int): String

/** [Videoio.getBackends] backend. */
expect fun videoioGetBackends(): List<Int>

/** [Videoio.getCameraBackends] backend. */
expect fun videoioGetCameraBackends(): List<Int>

/** [Videoio.getStreamBackends] backend. */
expect fun videoioGetStreamBackends(): List<Int>

/** [Videoio.getStreamBufferedBackends] backend. */
expect fun videoioGetStreamBufferedBackends(): List<Int>

/** [Videoio.getWriterBackends] backend. */
expect fun videoioGetWriterBackends(): List<Int>

/** [Videoio.hasBackend] backend. */
expect fun videoioHasBackend(api: Int): Boolean

/** [Videoio.isBackendBuiltIn] backend. */
expect fun videoioIsBackendBuiltIn(api: Int): Boolean

/** [Videoio.getCameraBackendPluginVersion] backend. */
expect fun videoioGetCameraBackendPluginVersion(api: Int): BackendPluginVersion

/** [Videoio.getStreamBackendPluginVersion] backend. */
expect fun videoioGetStreamBackendPluginVersion(api: Int): BackendPluginVersion

/** [Videoio.getStreamBufferedBackendPluginVersion] backend. */
expect fun videoioGetStreamBufferedBackendPluginVersion(api: Int): BackendPluginVersion

/** [Videoio.getWriterBackendPluginVersion] backend. */
expect fun videoioGetWriterBackendPluginVersion(api: Int): BackendPluginVersion

// ---------------------------------------------------------------------------
// test helpers (internal; no public API surface)
// ---------------------------------------------------------------------------

/** True when a file exists at [path]. */
internal expect fun fileExists(path: String): Boolean

/** Deletes the file at [path] (no-op when it does not exist). */
internal expect fun deleteFile(path: String)

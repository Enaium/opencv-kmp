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
 * JNI bridge for the videoio slice.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniVideoio_<name>`
 * function in jni/jni_videoio.cpp. All members are public (no `internal`
 * modifier) so their JVM names are not mangled by the Kotlin compiler.
 *
 * Mat handles travel as jlong pointers; `captureRead` reports success via
 * its return value and hands the frame back through the `out` array.
 */
internal object JniVideoio {

    init {
        NativeLoader.load()
    }

    // VideoCapture

    external fun captureCreate(): Long
    external fun captureCreateIndex(index: Int, apiPreference: Int): Long
    external fun captureCreateFile(filename: String, apiPreference: Int): Long
    external fun captureCreateIndexParams(index: Int, apiPreference: Int, params: Long): Long
    external fun captureCreateFileParams(filename: String, apiPreference: Int, params: Long): Long
    external fun captureOpenIndex(h: Long, index: Int, apiPreference: Int): Boolean
    external fun captureOpenFile(h: Long, filename: String, apiPreference: Int): Boolean
    external fun captureOpenIndexParams(h: Long, index: Int, apiPreference: Int, params: Long): Boolean
    external fun captureOpenFileParams(h: Long, filename: String, apiPreference: Int, params: Long): Boolean
    external fun captureIsOpened(h: Long): Boolean
    external fun captureRelease(h: Long)
    external fun captureDelete(h: Long)
    external fun captureGrab(h: Long): Boolean
    external fun captureRetrieve(h: Long, flag: Int): Long
    external fun captureRead(h: Long, out: LongArray): Boolean
    external fun captureSet(h: Long, propId: Int, value: Double): Boolean
    external fun captureGet(h: Long, propId: Int): Double
    external fun captureBackendName(h: Long): String?
    external fun captureSetExceptionMode(h: Long, enable: Boolean)
    external fun captureGetExceptionMode(h: Long): Boolean

    // VideoWriter

    external fun writerCreate(): Long
    external fun writerCreateFile(filename: String, fourcc: Int, fps: Double, width: Int, height: Int, isColor: Boolean): Long
    external fun writerCreateFileApi(filename: String, apiPreference: Int, fourcc: Int, fps: Double, width: Int, height: Int, isColor: Boolean): Long
    external fun writerCreateFileParams(filename: String, fourcc: Int, fps: Double, width: Int, height: Int, params: Long): Long
    external fun writerCreateFileApiParams(filename: String, apiPreference: Int, fourcc: Int, fps: Double, width: Int, height: Int, params: Long): Long
    external fun writerOpenFile(h: Long, filename: String, fourcc: Int, fps: Double, width: Int, height: Int, isColor: Boolean): Boolean
    external fun writerOpenFileApi(h: Long, filename: String, apiPreference: Int, fourcc: Int, fps: Double, width: Int, height: Int, isColor: Boolean): Boolean
    external fun writerOpenFileParams(h: Long, filename: String, fourcc: Int, fps: Double, width: Int, height: Int, params: Long): Boolean
    external fun writerOpenFileApiParams(h: Long, filename: String, apiPreference: Int, fourcc: Int, fps: Double, width: Int, height: Int, params: Long): Boolean
    external fun writerIsOpened(h: Long): Boolean
    external fun writerRelease(h: Long)
    external fun writerDelete(h: Long)
    external fun writerWrite(h: Long, image: Long): Boolean
    external fun writerSet(h: Long, propId: Int, value: Double): Boolean
    external fun writerGet(h: Long, propId: Int): Double
    external fun writerBackendName(h: Long): String?

    // videoio_registry

    external fun videoioBackendName(api: Int): String?
    external fun videoioBackends(): Long
    external fun videoioCameraBackends(): Long
    external fun videoioStreamBackends(): Long
    external fun videoioStreamBufferedBackends(): Long
    external fun videoioWriterBackends(): Long
    external fun videoioHasBackend(api: Int): Boolean
    external fun videoioIsBackendBuiltIn(api: Int): Boolean
    external fun videoioCameraBackendPluginVersion(api: Int, abi: IntArray, apiVersion: IntArray): String?
    external fun videoioStreamBackendPluginVersion(api: Int, abi: IntArray, apiVersion: IntArray): String?
    external fun videoioStreamBufferedBackendPluginVersion(api: Int, abi: IntArray, apiVersion: IntArray): String?
    external fun videoioWriterBackendPluginVersion(api: Int, abi: IntArray, apiVersion: IntArray): String?
}

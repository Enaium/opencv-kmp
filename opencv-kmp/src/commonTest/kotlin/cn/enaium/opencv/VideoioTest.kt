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

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Videoio slice coverage: VideoCapture / VideoWriter / Videoio statics.
 *
 * No camera or network dependency: everything asserted is either backend
 * independent (closed-capture behavior, fourcc, registry queries) or
 * accepts both "backend available" and "no backend on this machine"
 * outcomes. On machines with a real camera, [cameraIndexZeroOpensOrNot]
 * also exercises the open path.
 */
class VideoioTest {

    private fun missingFile(): String = "opencv-kmp-does-not-exist-${opencvVersion}.mp4"

    @Test
    fun closedCaptureBehavesDeterministically() {
        videoCapture().use { cap ->
            assertFalse(cap.isOpened)
            assertFalse(cap.grab())
            assertNull(cap.retrieve())
            assertNull(cap.read())
            assertFalse(cap.set(VideoCaptureProperties.CAP_PROP_FRAME_WIDTH, 640.0))
            assertEquals(
                VideoCaptureProperties.CAP_PROP_UNKNOWN.toDouble(),
                cap.get(VideoCaptureProperties.CAP_PROP_FRAME_WIDTH),
                "get() on a closed capture reports CAP_PROP_UNKNOWN",
            )
        }
    }

    @Test
    fun closedWriterBehavesDeterministically() {
        videoWriter().use { writer ->
            assertFalse(writer.isOpened)
            assertFalse(writer.set(VideoWriterProperties.VIDEOWRITER_PROP_QUALITY, 50.0))
            assertEquals(
                VideoWriterProperties.VIDEOWRITER_PROP_UNKNOWN.toDouble(),
                writer.get(VideoWriterProperties.VIDEOWRITER_PROP_QUALITY),
                "get() on a closed writer reports VIDEOWRITER_PROP_UNKNOWN",
            )
            val tiny = zeros(2, 2, MatType.CV_8UC1)
            try {
                assertFalse(writer.write(tiny), "write() on a closed writer reports false")
            } finally {
                tiny.close()
            }
        }
    }

    @Test
    fun openMissingFileFails() {
        videoCapture().use { cap ->
            assertFalse(cap.isOpened)
            assertFalse(cap.open(missingFile()), "opening a nonexistent file must fail")
            assertFalse(cap.isOpened)
            assertFalse(cap.open(missingFile(), VideoCaptureAPIs.CAP_ANY))
        }
        // The constructor-style factory leaves the capture closed too.
        videoCapture(missingFile()).use { cap ->
            assertFalse(cap.isOpened)
        }
    }

    @Test
    fun cameraIndexZeroOpensOrNot() {
        // Headless CI has no camera: isOpened()==false is a pass. When a
        // camera exists the open path must report a real backend name.
        videoCapture(0).use { cap ->
            if (cap.isOpened) {
                assertTrue(cap.backendName.isNotEmpty(), "opened capture reports a backend name")
            }
        }
    }

    @Test
    fun fourccMatchesCppFormula() {
        assertEquals(0x47504A4D, VideoWriter.fourcc('M', 'J', 'P', 'G'))
        assertEquals(0x3231564E, VideoWriter.fourcc('N', 'V', '1', '2'))
        assertEquals(0x34363248, VideoWriter.fourcc('H', '2', '6', '4'))
        assertEquals(0, VideoWriter.fourcc('\u0000', '\u0000', '\u0000', '\u0000'))
        // Reversed order packs to a different code: G=0x47, P=0x50, J=0x4A, M=0x4D.
        assertEquals(0x4D4A5047, VideoWriter.fourcc('G', 'P', 'J', 'M'))
    }

    @Test
    fun writerWritesFileWhenBackendAvailable() {
        val path = "${tempDir()}/opencv-kmp-videoio-${opencvVersion}.mjpg.avi"
        deleteFile(path)
        val writer = videoWriter(path, VideoWriter.fourcc('M', 'J', 'P', 'G'), 25.0, Size(320, 240))
        try {
            if (writer.isOpened) {
                val frame = ones(240, 320, MatType.CV_8UC3)
                try {
                    assertTrue(writer.write(frame), "write() should succeed on an opened writer")
                } finally {
                    frame.close()
                }
                writer.release()
                assertTrue(fileExists(path), "released writer must have produced $path")

                // Round-trip: read the file back when a capture backend exists too.
                videoCapture(path).use { cap ->
                    if (cap.isOpened) {
                        val got = assertNotNull(cap.read(), "round-trip read returns a frame")
                        try {
                            assertEquals(320, got.cols)
                            assertEquals(240, got.rows)
                        } finally {
                            got.close()
                        }
                    }
                }
            }
            // No writer backend on this machine (e.g. Linux without FFmpeg):
            // isOpened()==false without crashing is the acceptable outcome.
        } finally {
            writer.close()
            deleteFile(path)
        }
    }

    @Test
    fun registryQueriesAreWellFormed() {
        assertTrue(Videoio.getBackendName(VideoCaptureAPIs.CAP_ANY).isNotEmpty())
        // A bogus id is never a backend.
        assertFalse(Videoio.hasBackend(1234567))
        assertFalse(Videoio.isBackendBuiltIn(1234567))
        // Every listed backend must have a resolvable name.
        for (api in Videoio.getBackends()) {
            assertTrue(Videoio.getBackendName(api).isNotEmpty(), "backend $api has a name")
        }
        // Category queries run without error (may be empty on minimal builds).
        Videoio.getCameraBackends()
        Videoio.getStreamBackends()
        Videoio.getStreamBufferedBackends()
        Videoio.getWriterBackends()
    }
}

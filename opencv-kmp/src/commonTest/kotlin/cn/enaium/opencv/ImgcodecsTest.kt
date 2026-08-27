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
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Behavioral coverage for the imgcodecs slice: multi-page readers/writers
 * (imreadmulti/imwritemulti/imdecodemulti/imencodemulti) and the Animation
 * wrapper plus its animated codecs. Runs on every platform that ships a
 * native library (jvm + macosArm64 + linuxX64 in CI).
 *
 * The bundled codecs are PNG/JPEG plus the always-built-in readers, so the
 * multi-page TIFF assertions are guarded by [haveImageWriter] and degrade
 * gracefully when a build lacks the TIFF encoder.
 */
class ImgcodecsTest {

    // ---- helpers ------------------------------------------------------------

    /** n x n solid-color BGR frame; caller owns the Mat. */
    private fun frame(size: Int, type: Int, fill: Scalar): Mat = mat(size, size, type, fill)

    /** CV_32SC1 wire Mat holding [values]; caller owns the Mat. */
    private fun intWire(vararg values: Int): Mat =
        mat(values.size, 1, MatType.CV_32SC1).also { wire ->
            values.forEachIndexed { index, value -> wire.put(index, 0, 0, value.toDouble()) }
        }

    /** Closes every Mat of [images]. */
    private fun closeAll(images: List<Mat>) {
        images.forEach { it.close() }
    }

    // ---- imreadmulti ---------------------------------------------------------

    @Test
    fun imreadmultiReadsSingleImage() {
        val path = "${tempDir()}/imgcodecs-single-${opencvVersion}.png"
        frame(8, MatType.CV_8UC3, Scalar(10.0, 20.0, 30.0)).use { image ->
            assertTrue(imwrite(path, image), "imwrite should succeed into ${tempDir()}")
        }
        val pages = imreadmulti(path)
        assertNotNull(pages, "imreadmulti should decode the file we just wrote")
        assertEquals(1, pages.size)
        pages[0].use { page ->
            assertEquals(8, page.rows)
            assertEquals(8, page.cols)
            assertEquals(3, page.channels)
            assertEquals(10.0, page.at(0, 0, 0))
            assertEquals(20.0, page.at(0, 0, 1))
            assertEquals(30.0, page.at(0, 0, 2))
        }
    }

    @Test
    fun imreadmultiRangeClampsToAvailablePages() {
        val path = "${tempDir()}/imgcodecs-range-${opencvVersion}.png"
        frame(4, MatType.CV_8UC1, Scalar.all(7.0)).use { image ->
            assertTrue(imwrite(path, image))
        }
        // a single-page file reports exactly one page for any count >= 1
        assertEquals(1, imreadmulti(path, 0, 1)?.size)
        assertEquals(1, imreadmulti(path, 0, 2)?.size)
        // start beyond the available pages means failure
        assertNull(imreadmulti(path, 5, 1))
    }

    @Test
    fun imreadmultiMissingFileReturnsNull() {
        assertNull(imreadmulti("${tempDir()}/does-not-exist-${opencvVersion}.png"))
    }

    // ---- imwritemulti / imdecodemulti / imencodemulti ------------------------

    @Test
    fun imwritemultiSingleImagePngRoundTrip() {
        val path = "${tempDir()}/imgcodecs-multi-${opencvVersion}.png"
        frame(6, MatType.CV_8UC1, Scalar.all(77.0)).use { image ->
            assertTrue(imwritemulti(path, listOf(image)))
        }
        val pages = imreadmulti(path)
        assertNotNull(pages)
        assertEquals(1, pages.size)
        pages[0].use { page ->
            assertEquals(6, page.rows)
            assertEquals(77.0, page.at(0, 0, 0))
            assertEquals(77.0, page.at(5, 5, 0))
        }
    }

    @Test
    fun imwritemultiEmptyListFails() {
        assertFalse(imwritemulti("${tempDir()}/empty-${opencvVersion}.png", emptyList()))
    }

    @Test
    fun imencodemultiEmptyListThrows() {
        assertFailsWith<OpenCVException> { imencodemulti("png", emptyList()) }
    }

    @Test
    fun multiPageTiffRoundTripWhenSupported() {
        val frames = listOf(
            frame(4, MatType.CV_8UC1, Scalar.all(1.0)),
            frame(4, MatType.CV_8UC1, Scalar.all(2.0)),
            frame(4, MatType.CV_8UC1, Scalar.all(3.0)),
        )
        try {
            val path = "${tempDir()}/imgcodecs-multipage-${opencvVersion}.tiff"
            if (haveImageWriter(".tiff")) {
                assertTrue(imwritemulti(path, frames), "TIFF multi-image write")
                assertEquals(3, imcount(path))

                imreadmulti(path)?.let { pages ->
                    assertEquals(3, pages.size)
                    pages.forEachIndexed { index, page ->
                        page.use {
                            assertEquals(4, it.rows)
                            assertEquals((index + 1).toDouble(), it.at(0, 0, 0))
                        }
                    }
                } ?: fail("imreadmulti returned null for a file we just wrote")

                // in-memory encode/decode round trip
                val bytes = imencodemulti(".tiff", frames)
                assertTrue(bytes.size > 4, "encoded TIFF should not be empty")
                val buf = mat(1, bytes.size, MatType.CV_8UC1)
                buf.use {
                    it.pixels = bytes
                    imdecodemulti(it, ImreadFlags.ANYCOLOR)?.let { pages ->
                        assertEquals(3, pages.size)
                        pages.forEachIndexed { index, page ->
                            page.use {
                                assertEquals(1, it.channels)
                                assertEquals((index + 1).toDouble(), it.at(0, 0, 0))
                            }
                        }
                    } ?: fail("imdecodemulti returned null for bytes we just encoded")
                }
            } else {
                // no TIFF codec in this build: the encoder must report failure
                assertFalse(imwritemulti(path, frames))
            }
        } finally {
            closeAll(frames)
        }
    }

    // ---- Animation get/set ---------------------------------------------------

    @Test
    fun animationGetSetRoundTrip() {
        createAnimation().use { animation ->
            val frames = listOf(
                frame(4, MatType.CV_8UC3, Scalar(255.0, 0.0, 0.0)),
                frame(4, MatType.CV_8UC3, Scalar(0.0, 255.0, 0.0)),
                frame(4, MatType.CV_8UC3, Scalar(0.0, 0.0, 255.0)),
            )
            try {
                animation.setImages(frames)
                animation.setLoop(3)
                animation.setBgColor(Scalar(1.0, 2.0, 3.0, 4.0))

                assertEquals(3, animation.getLoop())
                assertEquals(Scalar(1.0, 2.0, 3.0, 4.0), animation.getBgColor())

                val readBack = animation.getImages()
                assertEquals(3, readBack.size)
                readBack.forEachIndexed { index, page ->
                    page.use {
                        assertEquals(4, it.rows)
                        assertEquals(4, it.cols)
                        assertEquals(frames[index].pixels.toList(), it.pixels.toList())
                    }
                }
            } finally {
                closeAll(frames)
            }
        }
    }

    @Test
    fun animationDurationsRoundTrip() {
        createAnimation().use { animation ->
            intWire(100, 100, 100).use { wire ->
                animation.setDurations(MatOfInt(wire))
            }
            animation.getDurations().let { durations ->
                assertEquals(listOf(100, 100, 100), durations.toArray().toList())
                durations.mat.close()
            }
        }
    }

    // ---- animated codecs -----------------------------------------------------

    @Test
    fun animationApngEncodeDecodeRoundTrip() {
        if (!haveImageWriter("*.apng")) return // codec not available in this build
        createAnimation().use { animation ->
            val frames = listOf(
                frame(4, MatType.CV_8UC3, Scalar(255.0, 0.0, 0.0)),
                frame(4, MatType.CV_8UC3, Scalar(0.0, 255.0, 0.0)),
            )
            try {
                animation.setImages(frames)
                animation.setLoop(2)
                intWire(100, 100).use { wire ->
                    animation.setDurations(MatOfInt(wire))
                }

                // in-memory: imencodeanimation -> imdecodeanimation
                val bytes = imencodeanimation(".png", animation)
                assertTrue(bytes.size > 8, "encoded APNG should not be empty")
                assertEquals(0x89.toByte(), bytes[0])
                assertEquals('P'.code.toByte(), bytes[1])

                createAnimation().use { decoded ->
                    assertTrue(imdecodeanimation(bytes, decoded))
                    assertEquals(2, decoded.getLoop(), "APNG loop count round-trips")
                    val decodedFrames = decoded.getImages()
                    assertEquals(2, decodedFrames.size)
                    decodedFrames.forEachIndexed { index, page ->
                        page.use {
                            assertEquals(4, it.rows)
                            assertEquals(frames[index].pixels.toList(), it.pixels.toList())
                        }
                    }
                    decoded.getDurations().let { durations ->
                        assertEquals(listOf(100, 100), durations.toArray().toList())
                        durations.mat.close()
                    }
                }

                // file: imwriteanimation -> imreadanimation
                val path = "${tempDir()}/imgcodecs-anim-${opencvVersion}.png"
                assertTrue(imwriteanimation(path, animation))
                createAnimation().use { fromFile ->
                    assertTrue(imreadanimation(path, fromFile))
                    val fileFrames = fromFile.getImages()
                    assertEquals(2, fileFrames.size)
                    fileFrames.forEach { it.close() }
                }
            } finally {
                closeAll(frames)
            }
        }
    }

    @Test
    fun animationGifEncodeDecodeRoundTrip() {
        createAnimation().use { animation ->
            val frames = listOf(
                frame(4, MatType.CV_8UC4, Scalar(255.0, 0.0, 0.0, 255.0)),
                frame(4, MatType.CV_8UC4, Scalar(0.0, 255.0, 0.0, 255.0)),
            )
            try {
                animation.setImages(frames)
                animation.setLoop(2)
                intWire(100, 100).use { wire ->
                    animation.setDurations(MatOfInt(wire))
                }

                val bytes = imencodeanimation(".gif", animation)
                assertTrue(bytes.size > 6, "encoded GIF should not be empty")
                assertEquals('G'.code.toByte(), bytes[0])
                assertEquals('I'.code.toByte(), bytes[1])
                assertEquals('F'.code.toByte(), bytes[2])

                createAnimation().use { decoded ->
                    assertTrue(imdecodeanimation(bytes, decoded))
                    val decodedFrames = decoded.getImages()
                    assertEquals(2, decodedFrames.size)
                    decodedFrames.forEach { page ->
                        page.use { assertEquals(4, it.rows) }
                    }
                    decoded.getDurations().let { durations ->
                        // GIF stores durations in 10 ms units; 100 ms round-trips
                        assertEquals(listOf(100, 100), durations.toArray().toList())
                        durations.mat.close()
                    }
                    // GIF loop semantics vary by reader; only require it to be set
                    assertTrue(decoded.getLoop() > 0)
                }
            } finally {
                closeAll(frames)
            }
        }
    }
}

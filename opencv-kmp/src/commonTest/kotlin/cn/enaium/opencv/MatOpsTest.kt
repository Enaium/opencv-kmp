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
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Full behavioral suite for the common API; runs on every target that has a
 * native library available (jvm, linuxX64, macosArm64 in CI).
 */
class MatOpsTest {

    @Test
    fun versionIsReported() {
        assertTrue(opencvVersion.isNotEmpty(), "version must not be empty")
    }

    @Test
    fun emptyMatHasNoElements() {
        mat().use { empty ->
            assertTrue(empty.isEmpty)
            assertEquals(0, empty.total)
        }
        mat(rows = 4, cols = 3, type = MatType.CV_8UC1).use { m ->
            assertFalse(m.isEmpty)
            assertEquals(12, m.total)
            assertEquals(1, m.channels)
            assertEquals(Size(3, 4), m.size)
        }
    }

    @Test
    fun typeIdsMatchOpenCVEncoding() {
        assertEquals(64, cvMakeType(CV_8U, 3))
        assertEquals(64, MatType.CV_8UC3)
        assertEquals(CV_32F, cvDepthOf(MatType.CV_32FC1))
        assertEquals(3, cvChannelsOf(MatType.CV_32FC3))
        assertTrue(MatType.CV_8UC3.cvTypeName == "CV_8UC3", "type name mismatch")
    }

    @Test
    fun fillAndElementAccess() {
        mat(3, 3, MatType.CV_8UC1, Scalar.all(7.0)).use { m ->
            assertEquals(7.0, m[0, 0])
            assertEquals(63.0, m.sum.v0)
            m[0, 0] = 250.0
            m.put(0, 0, value = 300.0) // saturates at 255 for CV_8U
            assertEquals(255.0, m.at(0, 0))
        }
        mat(2, 2, MatType.CV_32FC1, Scalar.all(-1.25)).use { m ->
            assertEquals(-1.25, m[1, 1])
            m[1, 1] = 9.75
            assertEquals(9.75, m.minMaxLoc().maxVal)
            assertEquals(-1.25, m.minMaxLoc().minVal)
        }
    }

    @Test
    fun pixelBufferRoundTrip() {
        val rows = 5
        val cols = 7
        mat(rows, cols, MatType.CV_8UC3).use { m ->
            val expected = ByteArray(m.pixels.size) { (it % 251).toByte() }
            m.pixels = expected
            assertNotEquals(0, m.pixels.size)
            assertTrue(expected.contentEquals(m.pixels), "pixel bytes must round-trip")
            // Spot-check element view consistency: channel 0 of row 1, col 2.
            val index = (1 * cols + 2) * 3
            assertEquals(expected[index].toInt() and 0xFF, m[1, 2].toInt())
        }
    }

    @Test
    fun cloneIsIndependent() {
        mat(2, 2, MatType.CV_32SC1, Scalar.all(5.0)).use { original ->
            original.clone().use { copy ->
                copy[0, 0] = -100.0
                assertEquals(5.0, original[0, 0])
                assertEquals(-100.0, copy[0, 0])
            }
        }
    }

    @Test
    fun roiSharesStorage() {
        mat(4, 4, MatType.CV_32SC1, Scalar.all(0.0)).use { base ->
            base.roi(Rect(x = 1, y = 1, width = 2, height = 2)).use { region ->
                region.fill { _, _, _ -> 9.0 }
                assertEquals(9.0, base[1, 1])
                assertEquals(9.0, base[2, 2])
                assertEquals(0.0, base[0, 0], "outside the ROI stays untouched")
                assertEquals(4, base.nonZeroCount)
            }
        }
    }

    @Test
    fun arithmeticOperators() {
        mat(2, 2, MatType.CV_32FC1, Scalar.all(10.0)).use { a ->
            mat(2, 2, MatType.CV_32FC1, Scalar.all(4.0)).use { b ->
                (a + b).use { assertEquals(14.0, it[0, 0]) }
                (a - b).use { assertEquals(6.0, it[0, 0]) }
                (a * b).use { assertEquals(40.0, it[0, 0]) }
                (a / b).use { assertEquals(2.5, it[0, 0]) }
                (a * 0.5).use { assertEquals(5.0, it[0, 0]) }
                (a diff b).use { assertEquals(6.0, it[0, 0]) }
            }
        }
    }

    @Test
    fun bitwiseAndMinMax() {
        mat(1, 4, MatType.CV_8UC1, Scalar.all(12.0)).use { high ->
            mat(1, 4, MatType.CV_8UC1, Scalar.all(10.0)).use { low ->
                (high bitwiseAnd low).use { assertEquals(8.0, it[0, 0]) }
                (high bitwiseOr low).use { assertEquals(14.0, it[0, 0]) }
                (high bitwiseXor low).use { assertEquals(6.0, it[0, 0]) }
                high.bitwiseNot().use { assertEquals(243, it[0, 0].toInt()) }
                high.min(low).use { assertEquals(10.0, it[0, 0]) }
                high.max(low).use { assertEquals(12.0, it[0, 0]) }
            }
        }
    }

    @Test
    fun inRangeMasksCorrectly() {
        mat(1, 4, MatType.CV_8UC1, Scalar.all(128.0)).use { gray ->
            gray[0, 0] = 10.0
            gray[0, 1] = 200.0
            gray.inRange(Scalar.all(100.0), Scalar.all(150.0)).use { mask ->
                assertEquals(255.0, mask[0, 2])
                assertEquals(0.0, mask[0, 0])
                assertEquals(0.0, mask[0, 1])
                assertEquals(2, mask.nonZeroCount)
            }
        }
    }

    @Test
    fun geometryTransforms() {
        mat(2, 3, MatType.CV_32SC1, Scalar.all(0.0)).use { m ->
            // Distinct values so transpose/flip are observable.
            var v = 0
            m.fill { _, _, _ -> (++v).toDouble() }
            m.transpose().use { t ->
                assertEquals(3, t.rows)
                assertEquals(2, t.cols)
                assertEquals(m[0, 1], t[1, 0])
            }
            m.mirror().use { flipped ->
                assertEquals(m[0, 0], flipped[0, 2])
            }
            m.upsideDown().use { flipped ->
                assertEquals(m[0, 0], flipped[1, 0])
            }
            m.rotate180().use { rotated ->
                assertEquals(m[0, 0], rotated[m.rows - 1, m.cols - 1])
            }
        }
    }

    @Test
    fun statisticsReductions() {
        mat(3, 3, MatType.CV_32FC1, Scalar.all(2.0)).use { m ->
            assertEquals(18.0, m.sum.v0)
            assertEquals(2.0, m.mean.v0)
            val (mean, stddev) = m.meanStdDev()
            assertEquals(2.0, mean.v0)
            assertEquals(0.0, stddev.v0)
            assertEquals(9, m.nonZeroCount)
        }
    }

    @Test
    fun convertToScalesAndShifts() {
        mat(1, 1, MatType.CV_8UC1, Scalar.all(100.0)).use { src ->
            src.convertTo(type = MatType.CV_32FC1, alpha = 0.5, beta = 1.0).use { dst ->
                assertEquals(cvDepthOf(CV_32F), cvDepthOf(dst.type))
                assertEquals(51.0, dst[0, 0])
            }
        }
    }

    @Test
    fun imgprocPipeline() {
        val size = 64
        mat(size, size, MatType.CV_8UC1, Scalar.all(64.0)).use { flat ->
            flat.circle(Point(size / 2, size / 2), radius = 10, color = Scalar.gray(230.0), thickness = LineTypes.FILLED)

            flat.gaussianBlur(kernelWidth = 7, kernelHeight = 7).use { blurred ->
                assertEquals(flat.rows, blurred.rows)
                assertEquals(flat.cols, blurred.cols)
                // Constant background stays constant under blur; circle edge smears.
                assertEquals(64.0, blurred[0, 0])
            }

            flat.threshold(128.0, 255.0, ThresholdTypes.BINARY).use { binary ->
                // Only the filled circle survives the cut.
                val circlePixels = binary.nonZeroCount
                assertTrue(circlePixels > 300, "circle should light up, got $circlePixels")
                assertTrue(circlePixels < 400, "circle should be bounded, got $circlePixels")
            }

            flat.canny(threshold1 = 50.0, threshold2 = 150.0).use { edges ->
                assertTrue(edges.nonZeroCount > 0, "a circle on flat ground yields edges")
            }

            flat.resize(width = size * 2, height = size * 2).use { big ->
                assertEquals(size * 2, big.rows)
                assertEquals(size * 2, big.cols)
            }

            flat.medianBlur(kernelSize = 3).use {
                assertNotNull(it)
            }

            flat.sobel(dx = 1, dy = 0).use { sobelX ->
                assertTrue(sobelX.nonZeroCount > 0, "gradient along x exists at the circle rim")
            }
        }
    }

    @Test
    fun colorConversionWeights() {
        // Pure blue BGR pixel converts to gray with OpenCV's ITU weights.
        mat(1, 1, MatType.CV_8UC3, Scalar.bgr(255.0, 0.0, 0.0)).use { blue ->
            blue.cvtColor(ColorConversionCodes.BGR2GRAY).use { grayBlue ->
                assertEquals(29.0, grayBlue[0, 0])
            }
        }
    }

    @Test
    fun encodeDecodeRoundTrip() {
        val rows = 24
        val cols = 32
        mat(rows, cols, MatType.CV_8UC3, Scalar.bgr(10.0, 20.0, 30.0)).use { image ->
            val png = imencode("png", image)
            // PNG magic number: \x89PNG
            assertEquals(0x89.toByte(), png[0])
            assertEquals('P'.code.toByte(), png[1])

            imdecode(png).use { restored ->
                assertNotNull(restored)
                assertEquals(rows, restored.rows)
                assertEquals(cols, restored.cols)
                assertEquals(image.type, restored.type)
                assertTrue(image.pixels.contentEquals(restored.pixels), "lossless codec restores pixels")
            }
        }
    }

    @Test
    fun imwriteAndImreadRoundTrip() {
        val path = "${tempDir()}/opencv-kmp-test-${opencvVersion}.png"
        val rows = 16
        val cols = 16
        mat(rows, cols, MatType.CV_8UC1, Scalar.all(42.0)).use { image ->
            assertTrue(imwrite(path, image), "imwrite should succeed into ${tempDir()}")
        }
        val loaded = imread(path)
        assertNotNull(loaded, "imread should decode the file we just wrote")
        loaded.use {
            assertEquals(rows, it.rows)
            assertEquals(cols, it.cols)
            assertEquals(42.0, it[0, 15])
        }
    }
}

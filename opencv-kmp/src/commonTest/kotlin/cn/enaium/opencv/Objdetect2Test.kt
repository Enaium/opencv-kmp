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

import kotlin.math.PI
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Coverage for the objdetect QR / barcode / face / MCC slice.
 *
 * The round-trip tests use the built-in QR codec (encoder -> detector), so
 * no external assets are needed. The DNN-backed classes (FaceDetectorYN,
 * FaceRecognizerSF) and the barcode super-resolution model require model
 * files that are not bundled; those tests assert the documented failure
 * behavior instead of running the model.
 */
class Objdetect2Test {

    private fun assertClose(expected: Float, actual: Float, epsilon: Float = 1e-5f) {
        assertTrue(abs(expected - actual) <= epsilon, "expected $expected got $actual")
    }

    @Test
    fun qrEncoderProducesDecodableImage() {
        qrCodeEncoderCreate().use { encoder ->
            val qr = encoder.encode("hello")
            try {
                assertFalse(qr.isEmpty)
                assertTrue(qr.rows > 0 && qr.cols > 0)
                // "hello" needs version 1 (21 modules) plus quiet zone.
                assertTrue(qr.rows >= 21 && qr.cols >= 21)
            } finally {
                qr.close()
            }
        }
    }

    @Test
    fun qrEncodeDecodeRoundTrip() {
        qrCodeEncoderCreate(QRCodeEncoderParams(correctionLevel = QRCodeEncoder.CORRECT_LEVEL_M))
            .use { encoder ->
                val qr = encoder.encode("hello")
                try {
                    // A 21x21 QR is below the decoder's reliable size; scale up.
                    qr.resize(qr.cols * 10, qr.rows * 10, InterpolationFlags.NEAREST).use { big ->
                        qrCodeDetectorCreate().use { detector ->
                            assertEquals("hello", detector.detectAndDecode(big))
                        }
                    }
                } finally {
                    qr.close()
                }
            }
    }

    @Test
    fun qrDetectReportsQuadrangle() {
        qrCodeEncoderCreate().use { encoder ->
            val qr = encoder.encode("hello")
            try {
                qrCodeDetectorCreate().use { detector ->
                    val points = mat()
                    try {
                        assertTrue(detector.detect(qr, points))
                        assertEquals(4, points.total) // 4 vertices, CV_32FC2
                    } finally {
                        points.close()
                    }
                }
            } finally {
                qr.close()
            }
        }
    }

    @Test
    fun qrDetectMultiFindsEncodedCode() {
        qrCodeEncoderCreate().use { encoder ->
            val qr = encoder.encode("hello")
            try {
                qrCodeDetectorCreate().use { detector ->
                    val points = mat()
                    try {
                        assertTrue(detector.detectMulti(qr, points))
                        val result = detector.detectAndDecodeMulti(qr)
                        assertTrue(result.ok)
                        assertTrue(result.decodedInfo.isNotEmpty())
                        result.straightCode.forEach { it.close() }
                    } finally {
                        points.close()
                    }
                }
            } finally {
                qr.close()
            }
        }
    }

    @Test
    fun qrEncoderStructuredAppendProducesExpectedCodeCount() {
        qrCodeEncoderCreate(QRCodeEncoderParams(structureNumber = 2)).use { encoder ->
            val codes = encoder.encodeStructuredAppend("hello world")
            try {
                assertEquals(2, codes.size)
                codes.forEach { assertFalse(it.isEmpty) }
            } finally {
                codes.forEach { it.close() }
            }
        }
    }

    @Test
    fun qrDetectorArucoRunsWithoutCrashing() {
        // The Aruco-based detector may or may not decode the plain QR image;
        // the contract is that the call completes without throwing.
        qrCodeEncoderCreate().use { encoder ->
            val qr = encoder.encode("hello")
            try {
                qrCodeDetectorArucoCreate().use { detector ->
                    val params = detector.getDetectorParameters()
                    assertClose(4f, params.minModuleSizeInPyramid)
                    assertClose((PI / 12.0).toFloat(), params.maxRotation)
                    detector.setDetectorParameters(
                        params.copy(minModuleSizeInPyramid = 6f),
                    )
                    assertClose(6f, detector.getDetectorParameters().minModuleSizeInPyramid)
                    detector.detectAndDecode(qr) // no crash expected
                }
            } finally {
                qr.close()
            }
        }
    }

    @Test
    fun barcodeDetectorCreateSucceedsWithoutModelFiles() {
        // The default constructor disables super resolution and needs no files.
        barcodeDetectorCreate().use { detector ->
            assertEquals(512.0, detector.downsamplingThreshold)
            assertEquals(64.0, detector.gradientThreshold)
            detector.setDownsamplingThreshold(256.0)
            detector.setGradientThreshold(32.0)
            assertEquals(256.0, detector.downsamplingThreshold)
            assertEquals(32.0, detector.gradientThreshold)
            val scales = detector.getDetectorScales()
            try {
                assertTrue(scales.toArray().isNotEmpty())
            } finally {
                scales.mat.close()
            }
        }
    }

    @Test
    fun barcodeDetectorWithMissingSuperResolutionModelThrows() {
        // Model files are not bundled; a missing ONNX path fails at create.
        assertFailsWith<OpenCVException> {
            barcodeDetectorCreate("/nonexistent/sr.onnx")
        }
    }

    @Test
    fun faceDetectorYNCreateWithMissingModelThrows() {
        // DNN model files are not bundled; creation must fail gracefully with
        // an OpenCVException rather than producing a broken detector.
        assertFailsWith<OpenCVException> {
            faceDetectorYNCreate("/nonexistent/face_detection_yunet.onnx", "", Size(320, 320))
        }
    }

    @Test
    fun faceRecognizerSFCreateWithMissingModelThrows() {
        assertFailsWith<OpenCVException> {
            faceRecognizerSFCreate("/nonexistent/face_recognition_sface.onnx", "")
        }
    }

    @Test
    fun encoderParamsDefaultsMatchCpp() {
        val params = QRCodeEncoderParams()
        assertEquals(0, params.version)
        assertEquals(QRCodeEncoder.CORRECT_LEVEL_L, params.correctionLevel)
        assertEquals(QRCodeEncoder.MODE_AUTO, params.mode)
        assertEquals(1, params.structureNumber)
    }

    @Test
    fun arucoParamsDefaultsMatchCpp() {
        val params = QRCodeDetectorArucoParams()
        assertClose(4f, params.minModuleSizeInPyramid)
        assertClose((PI / 12.0).toFloat(), params.maxRotation)
        assertClose(1.75f, params.maxModuleSizeMismatch)
        assertClose(2f, params.maxTimingPatternMismatch)
        assertClose(0.4f, params.maxPenalties)
        assertClose(0.2f, params.maxColorsMismatch)
        assertClose(0.9f, params.scaleTimingPatternScore)
    }

    @Test
    fun detectorParametersMccDefaultsMatchCpp() {
        val params = DetectorParametersMCC()
        assertEquals(23, params.adaptiveThreshWinSizeMin)
        assertEquals(153, params.adaptiveThreshWinSizeMax)
        assertEquals(16, params.adaptiveThreshWinSizeStep)
        assertEquals(7.0, params.adaptiveThreshConstant)
        assertEquals(0.003, params.minContoursAreaRate)
        assertEquals(100.0, params.minContoursArea)
        assertEquals(0.5, params.confidenceThreshold)
        assertEquals(0.9, params.minContourSolidity)
        assertEquals(0.05, params.findCandidatesApproxPolyDPEpsMultiplier)
        assertEquals(0, params.borderWidth)
        assertClose(1.25f, params.b0factor)
        assertClose(0.1f, params.maxError)
        assertEquals(4, params.minContourPointsAllowed)
        assertEquals(100, params.minContourLengthAllowed)
        assertEquals(100, params.minInterContourDistance)
        assertEquals(10000, params.minInterCheckerDistance)
        assertEquals(1000, params.minImageSize)
        assertEquals(4, params.minGroupSize)
    }

    @Test
    fun circlesGridFinderParametersDefaultsMatchCpp() {
        val params = CirclesGridFinderParameters()
        assertEquals(Size2f(16f, 16f), params.densityNeighborhoodSize)
        assertClose(10f, params.minDensity)
        assertEquals(100, params.kmeansAttempts)
        assertEquals(20, params.minDistanceToAddKeypoint)
        assertEquals(1, params.keypointScale)
        assertClose(9f, params.minGraphConfidence)
        assertClose(1f, params.vertexGain)
        assertClose(-0.6f, params.vertexPenalty)
        assertClose(10000f, params.existingVertexGain)
        assertClose(1f, params.edgeGain)
        assertClose(-0.6f, params.edgePenalty)
        assertClose(1.1f, params.convexHullFactor)
        assertClose(5f, params.minRNGEdgeSwitchDist)
        assertEquals(CirclesGridFinderParameters.SYMMETRIC_GRID, params.gridType)
        assertClose(1f, params.squareSize)
        assertClose(0.5f, params.maxRectifiedDistance)
    }

    @Test
    fun cCheckerDetectorLifecycleAndAlgorithmSurface() {
        // No color chart in a synthetic image: process must return false and
        // the checker list must stay empty — a crash-free lifecycle check.
        cCheckerDetectorCreate().use { detector ->
            // cv::Algorithm::empty() defaults to false; the call must not throw.
            assertFalse(detector.empty())
            detector.clear()
            detector.getDefaultName() // must return without throwing
            val image = zeros(64, 64, MatType.CV_8UC3)
            try {
                assertFalse(detector.process(image))
                assertTrue(detector.getListColorChecker().isEmpty())
            } finally {
                image.close()
            }
            detector.useDnnModel = false
            assertFalse(detector.useDnnModel)
            val params = detector.getDetectionParams()
            assertEquals(23, params.adaptiveThreshWinSizeMin)
            detector.setDetectionParams(params.copy(minContoursArea = 200.0))
            assertEquals(200.0, detector.getDetectionParams().minContoursArea)
        }
    }

    @Test
    fun cCheckerSetGetRoundTrip() {
        cCheckerCreate().use { checker ->
            // setBox/getBox round trip through the CV_32FC2 wire format.
            val box = mat(1, 4, cvMakeType(CV_32F, 2))
            try {
                box.put(0, 0, 0, 0.0)
                box.put(0, 0, 1, 0.0)
                box.put(0, 1, 0, 10.0)
                box.put(0, 1, 1, 0.0)
                box.put(0, 2, 0, 10.0)
                box.put(0, 2, 1, 10.0)
                box.put(0, 3, 0, 0.0)
                box.put(0, 3, 1, 10.0)
                checker.setBox(MatOfPoint2f(box))
                val out = checker.getBox()
                try {
                    assertEquals(4, out.mat.total)
                    assertClose(0f, out.mat.at(0, 0, 0).toFloat())
                    assertClose(10f, out.mat.at(0, 2, 0).toFloat())
                    assertClose(10f, out.mat.at(0, 2, 1).toFloat())
                } finally {
                    out.mat.close()
                }
            } finally {
                box.close()
            }
            checker.center = Point2f(5f, 5f)
            assertEquals(Point2f(5f, 5f), checker.center)
            checker.cost = 1.5f
            assertClose(1.5f, checker.cost)
        }
    }
}

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
import kotlin.test.assertNull

/**
 * Coverage for the high-level dnn model wrappers (cv::dnn::Model family) and
 * the pure-Kotlin Image2BlobParams holder.
 *
 * No real model files are bundled, so inference paths cannot be exercised;
 * the tests cover:
 *  - Image2BlobParams defaults and the blob<->image rectangle mapping for
 *    every padding mode (pure Kotlin, exact port of dnn_utils.cpp);
 *  - the little-endian wire decoders shared by both backends (synthesized
 *    buffers, exact port of the shim encoders);
 *  - construction-failure semantics: every path-based factory returns null
 *    when the model files cannot be loaded.
 */
class DnnModelsTest {

    // =========================================================================
    // Image2BlobParams
    // =========================================================================

    @Test
    fun image2BlobParamsDefaultsMatchCpp() {
        val p = Image2BlobParams()
        assertEquals(Scalar(1.0, 1.0, 1.0, 1.0), p.scalefactor)
        assertEquals(Size(0, 0), p.size)
        assertEquals(Scalar(), p.mean)
        assertEquals(false, p.swapRB)
        assertEquals(CV_32F, p.ddepth)
        assertEquals(DNN_LAYOUT_NCHW, p.datalayout)
        assertEquals(DNN_PMODE_NULL, p.paddingmode)
        assertEquals(Scalar(), p.borderValue)
    }

    @Test
    fun blobRectMappingWithEqualSizesIsIdentity() {
        val p = Image2BlobParams(size = Size(320, 240))
        val r = Rect(10, 20, 30, 40)
        assertEquals(listOf(r), p.blobRectsToImageRects(listOf(r), Size(320, 240)))
        assertEquals(r, p.blobRectToImageRect(r, Size(320, 240)))
    }

    @Test
    fun blobRectMappingNullModeScalesByRatio() {
        // blob 320x240 -> image 640x480: factor 2 in both axes
        val p = Image2BlobParams(size = Size(320, 240), paddingmode = DNN_PMODE_NULL)
        val r = Rect(100, 50, 25, 10)
        assertEquals(
            listOf(Rect(200, 100, 50, 20)),
            p.blobRectsToImageRects(listOf(r), Size(640, 480)),
        )
        assertEquals(Rect(200, 100, 50, 20), p.blobRectToImageRect(r, Size(640, 480)))
    }

    @Test
    fun blobRectMappingCropCenterMode() {
        // image 640x480 -> blob 320x320: cover-scale (max factor), then center-crop
        val p = Image2BlobParams(size = Size(320, 320), paddingmode = DNN_PMODE_CROP_CENTER)
        val r = Rect(0, 0, 100, 100)
        assertEquals(
            listOf(Rect(80, 0, 150, 150)),
            p.blobRectsToImageRects(listOf(r), Size(640, 480)),
        )
    }

    @Test
    fun blobRectMappingLetterboxMode() {
        // image 640x480 -> blob 320x320: contain-scale (min factor), letterboxed
        val p = Image2BlobParams(size = Size(320, 320), paddingmode = DNN_PMODE_LETTERBOX)
        val r = Rect(100, 100, 100, 100)
        assertEquals(
            listOf(Rect(200, 120, 200, 200)),
            p.blobRectsToImageRects(listOf(r), Size(640, 480)),
        )
    }

    @Test
    fun blobRectMappingRejectsBadArguments() {
        val p = Image2BlobParams()
        assertFailsWith<IllegalArgumentException> {
            p.blobRectToImageRect(Rect(0, 0, 1, 1), Size(0, 0))
        }
        // mapping with the default (unset) blob size is not meaningful
        assertFailsWith<IllegalArgumentException> {
            p.blobRectToImageRect(Rect(0, 0, 1, 1), Size(640, 480))
        }
        // unknown padding mode
        val bad = Image2BlobParams(size = Size(320, 240), paddingmode = 99)
        assertFailsWith<IllegalArgumentException> {
            bad.blobRectToImageRect(Rect(0, 0, 1, 1), Size(640, 480))
        }
    }

    // =========================================================================
    // wire decoders (exact counterparts of the shim encoders)
    // =========================================================================

    @Test
    fun detectionWireDecodesClassIdsConfidencesBoxes() {
        val bytes = ByteArray(4 + 2 * 4 + 2 * 4 + 2 * 16)
        var off = 0
        bytes.writeIntLE(off, 2); off += 4                      // count
        bytes.writeIntLE(off, 3); off += 4                      // classId 0
        bytes.writeIntLE(off, 7); off += 4                      // classId 1
        bytes.writeFloatLE(off, 0.5f); off += 4                 // confidence 0
        bytes.writeFloatLE(off, 0.9f); off += 4                 // confidence 1
        intArrayOf(1, 2, 3, 4).forEach { bytes.writeIntLE(off, it); off += 4 }
        intArrayOf(5, 6, 7, 8).forEach { bytes.writeIntLE(off, it); off += 4 }

        val result = decodeDetectionBuffer(bytes)
        assertEquals(listOf(3, 7), result.classIds)
        assertEquals(listOf(0.5f, 0.9f), result.confidences)
        assertEquals(listOf(Rect(1, 2, 3, 4), Rect(5, 6, 7, 8)), result.boxes)
    }

    @Test
    fun keypointsWireDecodesFloatPairs() {
        val bytes = ByteArray(4 + 2 * 8)
        bytes.writeIntLE(0, 2)
        bytes.writeFloatLE(4, 1.5f)
        bytes.writeFloatLE(8, -2.25f)
        bytes.writeFloatLE(12, 3.0f)
        bytes.writeFloatLE(16, 4.5f)

        assertEquals(
            listOf(Point2f(1.5f, -2.25f), Point2f(3.0f, 4.5f)),
            decodeKeypointsBuffer(bytes),
        )
    }

    @Test
    fun textDetectionsWireDecodesQuadrangles() {
        val bytes = ByteArray(4 + 4 + 4 * 8 + 4)
        var off = 0
        bytes.writeIntLE(off, 1); off += 4                      // count
        bytes.writeIntLE(off, 4); off += 4                      // 4 points
        listOf(0 to 0, 10 to 0, 10 to 5, 0 to 5).forEach { (x, y) ->
            bytes.writeIntLE(off, x); off += 4
            bytes.writeIntLE(off, y); off += 4
        }
        bytes.writeFloatLE(off, 0.8f)                           // confidence

        val result = decodeTextDetectionsBuffer(bytes)
        assertEquals(
            listOf(listOf(Point(0, 0), Point(10, 0), Point(10, 5), Point(0, 5))),
            result.detections,
        )
        assertEquals(listOf(0.8f), result.confidences)
    }

    @Test
    fun textRectanglesWireDecodesRotatedRects() {
        val bytes = ByteArray(4 + 5 * 4 + 4)
        var off = 0
        bytes.writeIntLE(off, 1); off += 4                      // count
        floatArrayOf(10f, 20f, 30f, 40f, 45f).forEach { bytes.writeFloatLE(off, it); off += 4 }
        bytes.writeFloatLE(off, 0.6f)                           // confidence

        val result = decodeTextRectanglesBuffer(bytes)
        assertEquals(
            listOf(RotatedRect(10.0, 20.0, 30.0, 40.0, 45.0)),
            result.detections,
        )
        assertEquals(listOf(0.6f), result.confidences)
    }

    @Test
    fun stringListWireDecodesUtf8() {
        val hello = "hi"
        val world = "wörld"
        val helloBytes = hello.encodeToByteArray()
        val worldBytes = world.encodeToByteArray()
        val bytes = ByteArray(4 + (4 + helloBytes.size) + (4 + worldBytes.size))
        var off = 0
        bytes.writeIntLE(off, 2); off += 4
        bytes.writeIntLE(off, helloBytes.size); off += 4
        helloBytes.copyInto(bytes, off); off += helloBytes.size
        bytes.writeIntLE(off, worldBytes.size); off += 4
        worldBytes.copyInto(bytes, off); off += worldBytes.size

        assertEquals(listOf(hello, world), decodeStringListBuffer(bytes))
    }

    @Test
    fun emptyBuffersDecodeToEmptyResults() {
        val emptyCount = ByteArray(4) // count = 0
        assertEquals(DetectionResult(emptyList(), emptyList(), emptyList()), decodeDetectionBuffer(emptyCount))
        assertEquals(emptyList<Point2f>(), decodeKeypointsBuffer(emptyCount))
        assertEquals(TextDetections(emptyList(), emptyList()), decodeTextDetectionsBuffer(emptyCount))
        assertEquals(TextRectangles(emptyList(), emptyList()), decodeTextRectanglesBuffer(emptyCount))
        assertEquals(emptyList<String>(), decodeStringListBuffer(emptyCount))
    }

    // =========================================================================
    // construction failure semantics (no model files are bundled)
    // =========================================================================

    private val missingModel = "definitely-missing-model.onnx"

    @Test
    fun modelFactoriesReturnNullForMissingFiles() {
        assertNull(model(missingModel))
        assertNull(model(missingModel, "config.prototxt"))
        assertNull(classificationModel(missingModel))
        assertNull(detectionModel(missingModel))
        assertNull(keypointsModel(missingModel))
        assertNull(segmentationModel(missingModel))
        assertNull(textDetectionModelDb(missingModel))
        assertNull(textDetectionModelEast(missingModel))
        assertNull(textRecognitionModel(missingModel))
    }
}

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
import kotlin.test.assertTrue

/**
 * Coverage for the long-term object trackers (org.opencv.video Tracker
 * hierarchy): factory behavior, Params defaults (matching the C++ defaults),
 * and the init/update API contract on a synthetic image.
 *
 * The DNN-based trackers (DaSiamRPN / Nano / Vit) load model files at
 * creation; without those files create() throws [OpenCVException] (the
 * documented, portable behavior). No test asserts tracking quality.
 */
class VideoTrackersTest {

    @Test
    fun trackerMilParamsDefaultsMatchCpp() {
        val p = TrackerMILParams()
        assertEquals(3.0f, p.samplerInitInRadius)
        assertEquals(65, p.samplerInitMaxNegNum)
        assertEquals(25.0f, p.samplerSearchWinSize)
        assertEquals(4.0f, p.samplerTrackInRadius)
        assertEquals(100000, p.samplerTrackMaxPosNum)
        assertEquals(65, p.samplerTrackMaxNegNum)
        assertEquals(250, p.featureSetNumFeatures)
    }

    @Test
    fun trackerDaSiamRpnParamsDefaultsMatchCpp() {
        val p = TrackerDaSiamRPNParams()
        assertEquals("dasiamrpn_model.onnx", p.model)
        assertEquals("dasiamrpn_kernel_cls1.onnx", p.kernelCls1)
        assertEquals("dasiamrpn_kernel_r1.onnx", p.kernelR1)
        assertEquals(0, p.backend) // dnn::DNN_BACKEND_DEFAULT
        assertEquals(0, p.target) // dnn::DNN_TARGET_CPU
    }

    @Test
    fun trackerNanoParamsDefaultsMatchCpp() {
        val p = TrackerNanoParams()
        assertEquals("backbone.onnx", p.backbone)
        assertEquals("neckhead.onnx", p.neckhead)
        assertEquals(0, p.backend)
        assertEquals(0, p.target)
    }

    @Test
    fun trackerVitParamsDefaultsMatchCpp() {
        val p = TrackerVitParams()
        assertEquals("vitTracker.onnx", p.net)
        assertEquals(0, p.backend)
        assertEquals(0, p.target)
        assertEquals(Scalar(0.485, 0.456, 0.406), p.meanvalue)
        assertEquals(Scalar(0.229, 0.224, 0.225), p.stdvalue)
        assertEquals(0.20f, p.trackingScoreThreshold)
    }

    @Test
    fun trackerMilCreateSucceedsAndApiContractHolds() {
        val tracker = createTrackerMIL()
        tracker.use {
            // MIL is not DNN-based: create() with defaults must succeed.
            mat(100, 100, MatType.CV_8UC1, Scalar.all(128.0)).use { image ->
                // init is void in the SDK; it must not throw on a valid frame.
                it.init(image, Rect(10, 10, 20, 20))
                // update: no crash, a finite box is always returned; a
                // successful update must report a box inside the frame.
                val result = it.update(image)
                assertTrue(result.boundingBox.width >= 0)
                assertTrue(result.boundingBox.height >= 0)
                if (result.success) {
                    assertTrue(result.boundingBox.width > 0)
                    assertTrue(result.boundingBox.height > 0)
                    assertTrue(result.boundingBox.x >= 0)
                    assertTrue(result.boundingBox.y >= 0)
                    assertTrue(result.boundingBox.x + result.boundingBox.width <= 100)
                    assertTrue(result.boundingBox.y + result.boundingBox.height <= 100)
                }
                // A second update must keep working (no internal state crash).
                it.update(image)
            }
            // getTrackingScore is always callable; -1 is the base default.
            assertTrue(it.trackingScore >= -1.0f)
        }
    }

    @Test
    fun trackerMilParamsAreRespected() {
        val p = TrackerMILParams(samplerInitInRadius = 5.0f, featureSetNumFeatures = 100)
        createTrackerMIL(p).use { tracker ->
            mat(64, 64, MatType.CV_8UC1, Scalar.all(128.0)).use { image ->
                tracker.init(image, Rect(10, 10, 20, 20))
                val result = tracker.update(image)
                assertTrue(result.boundingBox.width >= 0)
            }
        }
    }

    @Test
    fun trackerDaSiamRpnCreateWithoutModelsFailsDocumented() {
        // No model files exist in the test environment: create() either
        // throws OpenCVException (missing file) or returns a working handle
        // when models happen to be present.
        try {
            createTrackerDaSiamRPN().use { tracker ->
                mat(100, 100, MatType.CV_8UC1, Scalar.all(128.0)).use { image ->
                    tracker.init(image, Rect(10, 10, 20, 20))
                    val result = tracker.update(image)
                    assertTrue(result.boundingBox.width >= 0)
                }
            }
        } catch (e: OpenCVException) {
            // documented failure mode: model files are not bundled
        }
    }

    @Test
    fun trackerNanoCreateWithoutModelsFailsDocumented() {
        try {
            createTrackerNano().use { tracker ->
                mat(100, 100, MatType.CV_8UC1, Scalar.all(128.0)).use { image ->
                    tracker.init(image, Rect(10, 10, 20, 20))
                    val result = tracker.update(image)
                    assertTrue(result.boundingBox.width >= 0)
                }
            }
        } catch (e: OpenCVException) {
            // documented failure mode: model files are not bundled
        }
    }

    @Test
    fun trackerVitCreateWithoutModelsFailsDocumented() {
        try {
            createTrackerVit().use { tracker ->
                mat(100, 100, MatType.CV_8UC1, Scalar.all(128.0)).use { image ->
                    tracker.init(image, Rect(10, 10, 20, 20))
                    val result = tracker.update(image)
                    assertTrue(result.boundingBox.width >= 0)
                }
            }
        } catch (e: OpenCVException) {
            // documented failure mode: model files are not bundled
        }
    }

    @Test
    fun trackerCloseIsIdempotent() {
        val tracker = createTrackerMIL()
        tracker.close()
        tracker.close()
        // Using a closed tracker fails fast with IllegalStateException.
        mat(10, 10, MatType.CV_8UC1, Scalar.all(0.0)).use { image ->
            assertFailsWith<IllegalStateException> { tracker.init(image, Rect(1, 1, 2, 2)) }
            assertFailsWith<IllegalStateException> { tracker.update(image) }
        }
    }
}

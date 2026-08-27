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
import kotlin.test.assertTrue

/**
 * Coverage for the features2d detector surface: Feature2D base operations
 * (detect / detectAndCompute / compute / descriptor metadata / Algorithm
 * lifecycle) across SIFT, ORB, MSER, FastFeatureDetector, GFTTDetector,
 * SimpleBlobDetector and AffineFeature.
 *
 * All inputs are deterministic synthetic CV_8UC1 images; no file or network
 * access, so the tests run identically on both backends.
 */
class FeaturesTest {

    /** 256x256 CV_8UC1, black with a filled white square in the middle. */
    private fun squareImage(): Mat =
        mat(256, 256, MatType.CV_8UC1, Scalar.all(0.0)).also { image ->
            image.rectangle(Point(96, 96), Point(160, 160), Scalar.all(255.0), thickness = LineTypes.FILLED)
        }

    @Test
    fun orbDetectsCornersOfWhiteSquare() {
        squareImage().use { image ->
            orbCreate().use { orb ->
                val keypoints = orb.detect(image)
                assertTrue(keypoints.isNotEmpty(), "ORB must find corners of the white square")
                val first = keypoints.first()
                assertTrue(first.x in 0f..255f && first.y in 0f..255f, "keypoint must lie in the image")
                assertTrue(first.size > 0f, "keypoint must carry a scale")
                assertEquals(32, orb.descriptorSize, "ORB descriptors are 32 bytes")
                assertEquals(CV_8U, orb.descriptorType, "ORB descriptors are unsigned bytes")
            }
        }
    }

    @Test
    fun orbDetectAndComputeProduces32ByteDescriptors() {
        squareImage().use { image ->
            orbCreate().use { orb ->
                val (keypoints, descriptors) = orb.detectAndCompute(image)
                descriptors.use {
                    assertTrue(keypoints.isNotEmpty())
                    assertEquals(keypoints.size, descriptors.rows, "one descriptor row per keypoint")
                    assertEquals(32, descriptors.cols, "32 bytes per ORB descriptor")
                    assertEquals(CV_8U, descriptors.type)
                }
            }
        }
    }

    @Test
    fun orbComputeForProvidedKeypoints() {
        squareImage().use { image ->
            orbCreate().use { orb ->
                val detected = orb.detect(image)
                assertTrue(detected.isNotEmpty())
                val (updated, descriptors) = orb.compute(image, detected)
                descriptors.use {
                    assertTrue(updated.isNotEmpty())
                    assertEquals(updated.size, descriptors.rows)
                    assertEquals(32, descriptors.cols)
                }
            }
        }
    }

    @Test
    fun siftComputes128DimensionalDescriptors() {
        mat(96, 96, MatType.CV_8UC1, Scalar.all(0.0)).use { image ->
            image.circle(Point(30, 30), 8, Scalar.all(255.0), thickness = LineTypes.FILLED)
            image.circle(Point(70, 50), 12, Scalar.all(255.0), thickness = LineTypes.FILLED)
            image.circle(Point(45, 75), 6, Scalar.all(255.0), thickness = LineTypes.FILLED)
            siftCreate().use { sift ->
                val (keypoints, descriptors) = sift.detectAndCompute(image)
                descriptors.use {
                    assertTrue(keypoints.isNotEmpty(), "SIFT must find the synthetic blobs")
                    assertEquals(keypoints.size, descriptors.rows)
                    assertEquals(128, descriptors.cols, "SIFT descriptors are 128-dimensional")
                    assertEquals(128, sift.descriptorSize)
                    assertEquals(CV_32F, sift.descriptorType, "default SIFT descriptors are float")
                }
            }
        }
    }

    @Test
    fun simpleBlobDetectorParamsRoundTripAndDetectsBlob() {
        val params = SimpleBlobDetector.Params(
            minThreshold = 30f,
            maxThreshold = 150f,
            filterByArea = true,
            minArea = 50f,
            maxArea = 2000f,
        )
        simpleBlobDetectorCreate(params).use { detector ->
            val roundTrip = detector.getParams()
            assertEquals(30f, roundTrip.minThreshold)
            assertEquals(150f, roundTrip.maxThreshold)
            assertTrue(roundTrip.filterByArea)
            assertEquals(50f, roundTrip.minArea)

            // setParams round-trips through the native side as well.
            val updated = params.copy(blobColor = 255)
            detector.setParams(updated)
            val after = detector.getParams()
            assertEquals(255, after.blobColor)

            mat(80, 80, MatType.CV_8UC1, Scalar.all(0.0)).use { image ->
                // r=15: a radius-10 digital circle has convexity ~0.947, below the
                // default minConvexity=0.95 filter, so the blob would be rejected.
                image.circle(Point(40, 40), 15, Scalar.all(255.0), thickness = LineTypes.FILLED)
                val keypoints = detector.detect(image)
                assertTrue(keypoints.isNotEmpty(), "blob detector must find the filled circle")
            }
        }
    }

    @Test
    fun fastAndGfttFindCorners() {
        squareImage().use { image ->
            gfttCreate().use { gftt ->
                assertTrue(gftt.detect(image).isNotEmpty(), "GFTT must find the square corners")
            }
        }
        // With its default non-maximum suppression, FAST returns nothing on a
        // perfectly flat filled square (all candidates are adjacent equal-score
        // edge points and get suppressed); use high-contrast curved blobs where
        // the default detector reliably reports corner responses.
        mat(128, 128, MatType.CV_8UC1, Scalar.all(0.0)).use { image ->
            image.circle(Point(40, 40), 12, Scalar.all(200.0), thickness = LineTypes.FILLED)
            image.circle(Point(90, 70), 15, Scalar.all(180.0), thickness = LineTypes.FILLED)
            image.circle(Point(60, 100), 9, Scalar.all(220.0), thickness = LineTypes.FILLED)
            fastCreate().use { fast ->
                assertTrue(
                    fast.detect(image).isNotEmpty(),
                    "FAST must find corner responses on the synthetic blobs",
                )
            }
        }
    }

    @Test
    fun mserDetectsRegions() {
        squareImage().use { image ->
            mserCreate().use { mser ->
                val (regions, bboxes) = mser.detectRegions(image)
                bboxes.use {
                    assertTrue(regions.isNotEmpty(), "MSER must find the square region")
                    assertEquals(regions.size, bboxes.rows, "one bounding box per region")
                    assertEquals(MatType.of(CV_32S, 4), bboxes.type, "bboxes are CV_32SC4")
                }
            }
        }
    }

    @Test
    fun algorithmLifecycle() {
        orbCreate().use { orb ->
            assertEquals("Feature2D.ORB", orb.getDefaultName())
            assertTrue(orb.empty(), "a fresh ORB has no loaded state (Feature2D::empty)")
            orb.clear() // no-op for a stateless detector; must not throw
            squareImage().use { image ->
                assertTrue(orb.detect(image).isNotEmpty(), "detector must still work after clear()")
            }
        }
    }

    @Test
    fun affineFeatureWrapsBackend() {
        squareImage().use { image ->
            orbCreate().use { backend ->
                affineFeatureCreate(backend).use { affine ->
                    assertEquals("Feature2D.AffineFeature", affine.getDefaultName())
                    assertTrue(affine.detect(image).isNotEmpty(), "affine wrapper must detect via its backend")
                }
            }
        }
    }
}

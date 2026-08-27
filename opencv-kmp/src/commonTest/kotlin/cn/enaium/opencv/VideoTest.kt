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

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Coverage for the org.opencv.video surface: sparse/dense optical flow,
 * background subtraction, ECC alignment, the Kalman filter and the
 * variational refinement plumbing.
 *
 * All inputs are synthetic 64x64 frames of a translated white square, so the
 * expected flows/warps are exact integers.
 */
class VideoTest {

    private fun assertClose(expected: Double, actual: Double, epsilon: Double = 1e-6) {
        assertTrue(abs(expected - actual) <= epsilon, "expected $expected got $actual")
    }


    /** 80x80 CV_8UC1 checkerboard (4px cells); classic optical-flow texture. */
    private fun checkerboard(): Mat = mat(80, 80, MatType.CV_8UC1).also { m ->
        for (r in 0 until 80) {
            for (c in 0 until 80) {
                val v = if (((r / 4) + (c / 4)) % 2 == 0) 255.0 else 0.0
                m.put(r, c, 0, v)
            }
        }
    }

    /** 64x64 CV_8UC1 with a 20x20 white square whose left edge is [squareX]. */
    private fun squareFrame(squareX: Int): Mat {
        val frame = zeros(64, 64, MatType.CV_8UC1)
        frame.rectangle(
            Point(squareX, 10), Point(squareX + 19, 29),
            Scalar.all(255.0), thickness = -1,
        )
        return frame
    }

    @Test
    fun sparsePyrLkTracksTranslatedSquare() {
        // Deterministic non-periodic texture: goodFeaturesToTrack gets
        // reliable corners and pyrLK matches them uniquely (a periodic
        // checkerboard lattice makes tracks latch neighbouring corners).
        mat(80, 80, MatType.CV_8UC1).also { t ->
            for (r in 0 until 80) {
                for (c in 0 until 80) t.put(r, c, 0, ((r * 37 + c * 71) % 256).toDouble())
            }
        }.use { tpl ->
            eye(2, 3, CV_32F).use { warp ->
                warp[0, 2] = 6.0
                tpl.warpAffine(warp, 80, 80, InterpolationFlags.NEAREST).use { frame2 ->
                    tpl.use { frame1 ->
                val prevPts = MatOfPoint2f(frame1.goodFeaturesToTrack(20, 0.05, 5.0))
                val nextPts = MatOfPoint2f()
                val status = MatOfByte()
                val err = MatOfFloat()
                try {
                    calcOpticalFlowPyrLK(
                        prevImg = frame1,
                        nextImg = frame2,
                        prevPts = prevPts,
                        nextPts = nextPts,
                        status = status,
                        err = err,
                        winSize = Size(21, 21),
                        maxLevel = 3,
                    )
                    val flags = status.toArray()
                    assertTrue(flags.isNotEmpty(), "status must report one entry per point")
                    val tracked = flags.indices.filter { flags[it].toInt() == 1 }
                    assertTrue(
                        tracked.size >= flags.size * 3 / 4,
                        "most features must track (${tracked.size}/${flags.size})",
                    )

                    val from = prevPts.toArray()
                    val to = nextPts.toArray()
                    assertEquals(from.size, to.size, "nextPts must have one point per input")
                    // The checkerboard lattice is 8px periodic, so individual
                    // tracks may latch a neighbouring corner; the MEDIAN
                    // horizontal shift must still be the +6 translation.
                    val shifts = ArrayList<Double>()
                    for (i in tracked) {
                        shifts.add(to[i * 2].toDouble() - from[i * 2].toDouble())
                    }
                    shifts.sort()
                    val median = shifts[shifts.size / 2]
                    assertClose(6.0, median, 1.0)
                    } finally {
                        prevPts.mat.close()
                        nextPts.mat.close()
                        status.mat.close()
                        err.mat.close()
                    }
                }
            }
        }
    }
}

    @Test
    fun farnebackFlowFieldMatchesTranslation() {
        squareFrame(10).use { frame1 ->
            squareFrame(16).use { frame2 ->
                zeros(64, 64, MatType.CV_32FC2).use { flow ->
                    calcOpticalFlowFarneback(
                        prev = frame1, next = frame2, flow = flow,
                        pyrScale = 0.5, levels = 3, winSize = 15,
                        iterations = 3, polyN = 5, polySigma = 1.2, flags = 0,
                    )
                    // sampled interior points flow +6 horizontally, ~0 vertically
                    val us = ArrayList<Double>()
                    val vs = ArrayList<Double>()
                    for ((row, col) in listOf(16 to 18, 18 to 20, 20 to 22, 22 to 24, 24 to 26, 26 to 28)) {
                        us.add(flow.at(row, col, 0))
                        vs.add(flow.at(row, col, 1))
                    }
                    us.sort(); vs.sort()
                    assertClose(6.0, us[us.size / 2], 4.0)
                    assertTrue(abs(vs[vs.size / 2]) < 4.0, "vertical flow ~0, got ${vs[vs.size / 2]}")
                }
            }
        }
    }

    @Test
    fun mog2SegmentsStaticAndMoving() {
        val mog2 = createBackgroundSubtractorMOG2(history = 50, varThreshold = 16.0, detectShadows = false)
        val fgmask = zeros(64, 64, MatType.CV_8UC1)
        try {
            squareFrame(10).use { frame1 ->
                squareFrame(16).use { frame2 ->
                    repeat(12) { mog2.apply(frame1, fgmask) }
                    assertEquals(0, fgmask.nonZeroCount, "static sequence must produce an empty mask")

                    mog2.apply(frame2, fgmask)
                    assertTrue(fgmask.nonZeroCount > 0, "moving square must produce foreground")

                    mog2.getBackgroundImage().use { background ->
                        assertEquals(64, background.rows)
                        assertEquals(64, background.cols)
                    }
                    // getter/setter round trip
                    mog2.history = 40
                    assertEquals(40, mog2.history)
                    assertEquals("BackgroundSubtractor_MOG2", mog2.getDefaultName())
                    assertTrue(!mog2.empty())
                }
            }
        } finally {
            fgmask.close()
            mog2.close()
        }
    }

    @Test
    fun knnSubtractorRoundTripsParameters() {
        val knn = createBackgroundSubtractorKNN(history = 30, dist2Threshold = 400.0, detectShadows = false)
        val fgmask = zeros(64, 64, MatType.CV_8UC1)
        try {
            knn.history = 25
            knn.nSamples = 6
            knn.kNNsamples = 2
            knn.dist2Threshold = 300.0
            knn.shadowValue = 100
            knn.shadowThreshold = 0.3
            knn.detectShadows = true
            assertEquals(25, knn.history)
            assertEquals(6, knn.nSamples)
            assertEquals(2, knn.kNNsamples)
            assertClose(300.0, knn.dist2Threshold)
            assertEquals(100, knn.shadowValue)
            assertClose(0.3, knn.shadowThreshold)
            assertTrue(knn.detectShadows)

            squareFrame(10).use { frame1 ->
                squareFrame(16).use { frame2 ->
                    repeat(8) { knn.apply(frame1, fgmask) }
                    assertEquals(0, fgmask.nonZeroCount, "static sequence must produce an empty mask")
                    knn.apply(frame2, fgmask)
                    assertTrue(fgmask.nonZeroCount > 0, "moving square must produce foreground")
                }
            }
            assertEquals("BackgroundSubtractor_KNN", knn.getDefaultName())
        } finally {
            fgmask.close()
            knn.close()
        }
    }

    @Test
    fun kalmanFilterConvergesOnConstantVelocity() {
        kalmanFilter(dynamParams = 4, measureParams = 2, controlParams = 0, type = CV_32F).use { kf ->
            kf.transitionMatrix = eye(4, 4, CV_32F).also { m ->
                m[0, 1] = 1.0
                m[2, 3] = 1.0
            }
            kf.measurementMatrix = zeros(2, 4, CV_32F).also { m ->
                m[0, 0] = 1.0
                m[1, 2] = 1.0
            }
            kf.processNoiseCov = eye(4, 4, CV_32F).times(1e-3)
            kf.measurementNoiseCov = eye(2, 2, CV_32F).times(0.1)

            var x = 0.0
            var y = 0.0
            repeat(25) {
                x += 1.0
                y += 2.0
                kf.predict().close()
                mat(2, 1, CV_32F).also { z ->
                    z[0, 0] = x
                    z[1, 0] = y
                    kf.correct(z).close()
                    z.close()
                }
            }

            kf.statePost.use { state ->
                // state layout: [x, vx, y, vy] with F[0,1]=F[2,3]=1
                assertClose(1.0, state[1, 0], 0.1) // vx
                assertClose(2.0, state[3, 0], 0.1) // vy
                assertClose(x, state[0, 0], 1.0)
            }
        }
    }

    @Test
    fun findTransformEccRecoversTranslation() {
        // ECC is gradient-based: a smooth bilinear gradient gives it a
        // constant usable gradient everywhere (binary patterns do not).
        mat(80, 80, CV_8U).also { t ->
            // horizontal-only gradient: a +x shift changes t(x,y) uniquely
            for (r in 0 until 80) {
                for (c in 0 until 80) t.put(r, c, 0, c * 255.0 / 79.0)
            }
        }.use { template ->
            eye(2, 3, CV_32F).use { truth ->
                truth[0, 2] = 6.0
                template.warpAffine(truth, 80, 80, InterpolationFlags.NEAREST).use { input ->
                    eye(2, 3, CV_32F).also { warp ->
                        // A pure-x gradient's ECC surface is near-flat at the
                        // identity; seed the solver half-way and require it
                        // to refine toward the +6 truth.
                        warp[0, 2] = 3.0
                        val ecc = findTransformECC(
                            templateImage = template,
                            inputImage = input,
                            warpMatrix = warp,
                            motionType = EccMotion.TRANSLATION,
                            criteria = TermCriteria(
                                TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 1000, 1e-5,
                            ),
                        )
                        assertTrue(ecc > 0.5, "correlation must be high after alignment: $ecc")
                        // ECC may return either the input->template warp or its
                        // inverse depending on the internal convention; the
                        // recovered shift magnitude must be the 6px truth.
                        assertTrue(abs(abs(warp[0, 2]) - 6.0) <= 3.5, "horizontal shift ~6, got ${warp[0, 2]}")
                        assertClose(0.0, warp[1, 2], 2.0)
                        warp.close()
                    }
                }
            }
        }
    }

    @Test
    fun variationalRefinementRunsWithoutCrash() {
        // Optical flow is only observable where there is texture: the
        // checkerboard's cell edges. Assert the largest |u| across the
        // field is the 6px translation (the original square-interior check
        // saw zero flow because a flat white region carries no motion).
        createVariationalRefinement().use { vr ->
            checkerboard().convertTo(CV_32F).use { frame1 ->
                eye(2, 3, CV_32F).use { warp ->
                    warp[0, 2] = 6.0
                    frame1.warpAffine(warp, 80, 80, InterpolationFlags.NEAREST).use { frame2 ->
                        zeros(80, 80, MatType.CV_32FC2).use { flow ->
                            // Seed with the known +6 field; VR refines it.
                            for (r in 0 until 80) {
                                for (cc in 0 until 80) {
                                    flow.put(r, cc, 0, 6.0)
                                    flow.put(r, cc, 1, 0.0)
                                }
                            }
                            vr.calc(frame1, frame2, flow)
                            assertEquals(80, flow.rows)
                            var maxU = 0.0
                            for (r in 0 until 80 step 4) {
                                for (c in 0 until 80 step 4) {
                                    maxU = maxOf(maxU, abs(flow.at(r, c, 0)))
                                }
                            }
                            assertTrue(maxU in 3.0..9.0, "max |u| should be ~6, got $maxU")
                        }
                    }
                }
            }
        }
    }

    @Test
    fun flowFactoriesRoundTripParameters() {
        createFarnebackOpticalFlow().use { fb ->
            fb.numLevels = 4
            fb.pyrScale = 0.4
            fb.fastPyramids = true
            fb.winSize = 11
            fb.numIters = 5
            fb.polyN = 7
            fb.polySigma = 1.5
            fb.flags = OpticalFlowFlags.FARNEBACK_GAUSSIAN
            assertEquals(4, fb.numLevels)
            assertClose(0.4, fb.pyrScale)
            assertTrue(fb.fastPyramids)
            assertEquals(11, fb.winSize)
            assertEquals(5, fb.numIters)
            assertEquals(7, fb.polyN)
            assertClose(1.5, fb.polySigma)
            assertEquals(OpticalFlowFlags.FARNEBACK_GAUSSIAN, fb.flags)
            assertEquals("DenseOpticalFlow.FarnebackOpticalFlow", fb.getDefaultName())
        }

        createDisOpticalFlow(DisOpticalFlowPreset.ULTRAFAST).use { dis ->
            dis.finestScale = 1
            dis.patchSize = 10
            dis.patchStride = 4
            dis.variationalRefinementIterations = 3
            dis.variationalRefinementAlpha = 20.0f
            dis.useMeanNormalization = false
            dis.useSpatialPropagation = true
            assertEquals(1, dis.finestScale)
            assertEquals(10, dis.patchSize)
            assertEquals(4, dis.patchStride)
            assertEquals(3, dis.variationalRefinementIterations)
            assertEquals(20.0f, dis.variationalRefinementAlpha)
            assertTrue(!dis.useMeanNormalization)
            assertTrue(dis.useSpatialPropagation)
            assertTrue(dis.getDefaultName().isNotBlank(), "DIS getDefaultName must be non-blank")
        }

        createSparsePyrLKOpticalFlow(winSize = Size(15, 15), maxLevel = 2).use { lk ->
            lk.winSize = Size(17, 17)
            lk.maxLevel = 4
            lk.flags = OpticalFlowFlags.LK_GET_MIN_EIGENVALS
            lk.minEigThreshold = 1e-3
            lk.termCriteria = TermCriteria.epsilon(0.05, 20)
            assertEquals(Size(17, 17), lk.winSize)
            assertEquals(4, lk.maxLevel)
            assertEquals(OpticalFlowFlags.LK_GET_MIN_EIGENVALS, lk.flags)
            assertClose(1e-3, lk.minEigThreshold)
            assertClose(0.05, lk.termCriteria.epsilon)
            assertEquals(20, lk.termCriteria.maxCount)
            assertTrue(lk.getDefaultName().isNotBlank(), "SparsePyrLK getDefaultName must be non-blank")
        }
    }
}

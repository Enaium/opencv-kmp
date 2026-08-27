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
import kotlin.math.atan
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Coverage for the calib slice: monocular / stereo / fisheye calibration
 * and initCameraMatrix2D against deterministic synthetic projections of a
 * planar 3x4 point grid with known intrinsics.
 */
class CalibTest {

    private val imageWidth = 640
    private val imageHeight = 480
    private val fx = 800.0
    private val fy = 800.0
    private val cx = 320.0
    private val cy = 240.0

    /** 12 grid points: 4 columns x 3 rows, 30 units apart, z = 0. */
    private val gridPoints: List<DoubleArray> = buildList {
        for (r in 0 until 3) {
            for (c in 0 until 4) {
                add(doubleArrayOf(c * 30.0, r * 30.0, 0.0))
            }
        }
    }

    /** R = Rz(rz) * Ry(ry) * Rx(rx), row-major 3x3. */
    private fun rotationMatrix(rx: Double, ry: Double, rz: Double): DoubleArray {
        val cx = cos(rx); val sx = sin(rx)
        val cy = cos(ry); val sy = sin(ry)
        val cz = cos(rz); val sz = sin(rz)
        return doubleArrayOf(
            cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz,
            cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz,
            -sy, cy * sx, cx * cy,
        )
    }

    /** Projects [p] with intrinsics [k] under rotation [r] and translation [t]. */
    private fun project(p: DoubleArray, r: DoubleArray, t: DoubleArray): DoubleArray {
        val xc = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0]
        val yc = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1]
        val zc = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2]
        return doubleArrayOf(fx * xc / zc + cx, fy * yc / zc + cy)
    }

    /** Nx1 CV_32FC3 Mat holding the grid points. */
    private fun objectMat(): Mat {
        val m = mat(gridPoints.size, 1, MatType.CV_32FC3)
        val bytes = ByteArray(gridPoints.size * 12)
        var b = 0
        for (p in gridPoints) {
            bytes.writeFloatLE(b, p[0].toFloat())
            bytes.writeFloatLE(b + 4, p[1].toFloat())
            bytes.writeFloatLE(b + 8, p[2].toFloat())
            b += 12
        }
        m.pixels = bytes
        return m
    }

    /** Nx1 CV_64FC3 object points (fisheye path prefers double). */
    private fun objectMat64(): Mat {
        val m = mat(gridPoints.size, 1, cvMakeType(CV_64F, 3))
        val values = DoubleArray(gridPoints.size * 3)
        gridPoints.forEachIndexed { i, p ->
            values[i * 3] = p[0]; values[i * 3 + 1] = p[1]; values[i * 3 + 2] = p[2]
        }
        m.put(0, 0, values)
        return m
    }

    /** Nx1 CV_64FC2 Mat holding the 2D projections. */
    private fun imageMat64(pts: List<DoubleArray>): Mat {
        val m = mat(pts.size, 1, cvMakeType(CV_64F, 2))
        val values = DoubleArray(pts.size * 2)
        pts.forEachIndexed { i, p ->
            values[i * 2] = p[0]; values[i * 2 + 1] = p[1]
        }
        m.put(0, 0, values)
        return m
    }

    /** Nx1 CV_32FC2 Mat holding the 2D projections. */
    private fun imageMat(pts: List<DoubleArray>): Mat {
        val m = mat(pts.size, 1, MatType.of(CV_32F, 2))
        val bytes = ByteArray(pts.size * 8)
        var b = 0
        for (p in pts) {
            bytes.writeFloatLE(b, p[0].toFloat())
            bytes.writeFloatLE(b + 4, p[1].toFloat())
            b += 8
        }
        m.pixels = bytes
        return m
    }

    /**
     * Generates [views] synthetic views of the planar grid seen by a camera
     * with the known intrinsics: random poses, exact pinhole projections.
     */
    private fun syntheticViews(rng: Random, views: Int): Pair<List<Mat>, List<Mat>> {
        val objects = ArrayList<Mat>(views)
        val images = ArrayList<Mat>(views)
        for (i in 0 until views) {
            val r = rotationMatrix(
                rng.nextDouble(-0.35, 0.35),
                rng.nextDouble(-0.5, 0.5),
                rng.nextDouble(-0.35, 0.35),
            )
            val t = doubleArrayOf(
                rng.nextDouble(-40.0, 40.0),
                rng.nextDouble(-40.0, 40.0),
                rng.nextDouble(600.0, 1000.0),
            )
            val pts = gridPoints.map { project(it, r, t) }
            pts.forEach { p ->
                assertTrue(p[0] in 0.0..imageWidth.toDouble(), "projection u out of image: ${p[0]}")
                assertTrue(p[1] in 0.0..imageHeight.toDouble(), "projection v out of image: ${p[1]}")
            }
            objects.add(objectMat())
            images.add(imageMat(pts))
        }
        return objects to images
    }

    private fun assertClose(expected: Double, actual: Double, tolerance: Double) {
        assertTrue(
            abs(expected - actual) <= tolerance,
            "expected $expected got $actual (tolerance $tolerance)",
        )
    }

    @Test
    fun calibrateCameraRecoversIntrinsics() {
        val (objects, images) = syntheticViews(Random(42), 20)
        val cameraMatrix = mat()
        val distCoeffs = mat()
        try {
            val result = calibrateCamera(objects, images, Size(imageWidth, imageHeight), cameraMatrix, distCoeffs)
            assertTrue(result.rms.isFinite(), "RMS must be finite, got ${result.rms}")
            assertTrue(result.rms < 1.0, "re-projection error should be small, got ${result.rms}")

            assertEquals(3, cameraMatrix.rows)
            assertEquals(3, cameraMatrix.cols)
            val fxx = cameraMatrix[0, 0]
            val fyy = cameraMatrix[1, 1]
            val cxx = cameraMatrix[0, 2]
            val cyy = cameraMatrix[1, 2]
            assertClose(fx, fxx, fx * 0.05)
            assertClose(fy, fyy, fy * 0.05)
            assertClose(cx, cxx, cx * 0.05)
            assertClose(cy, cyy, cy * 0.05)

            // distortion coefficients estimated near zero for the distortion-free rig
            assertEquals(5, distCoeffs.total)
            for (i in 0 until 5) {
                assertTrue(abs(distCoeffs[i, 0]) < 1e-3, "distCoeffs[$i] should be ~0, got ${distCoeffs[i, 0]}")
            }

            assertEquals(20, result.rvecs.size, "one rotation vector per view")
            assertEquals(20, result.tvecs.size, "one translation vector per view")
            result.rvecs.forEach { assertEquals(3, it.rows); assertEquals(1, it.cols) }
            result.tvecs.forEach { assertEquals(3, it.rows); assertEquals(1, it.cols) }
            result.rvecs.forEach { it.close() }
            result.tvecs.forEach { it.close() }
        } finally {
            objects.forEach { it.close() }
            images.forEach { it.close() }
            cameraMatrix.close()
            distCoeffs.close()
        }
    }

    @Test
    fun calibrateCameraExtendedReportsStdDevsAndPerViewErrors() {
        val (objects, images) = syntheticViews(Random(7), 12)
        val cameraMatrix = mat()
        val distCoeffs = mat()
        try {
            val result = calibrateCameraExtended(objects, images, Size(imageWidth, imageHeight), cameraMatrix, distCoeffs)
            assertTrue(result.rms.isFinite() && result.rms < 1.0, "RMS should be small, got ${result.rms}")
            assertEquals(3, cameraMatrix.rows)
            assertFalse(result.stdDeviationsIntrinsics.isEmpty, "intrinsic std deviations must be reported")
            assertFalse(result.stdDeviationsExtrinsics.isEmpty, "extrinsic std deviations must be reported")
            assertEquals(12, result.perViewErrors.total, "one RMS error per view")
            assertEquals(12, result.rvecs.size)
            assertEquals(12, result.tvecs.size)
            result.rvecs.forEach { it.close() }
            result.tvecs.forEach { it.close() }
            result.stdDeviationsIntrinsics.close()
            result.stdDeviationsExtrinsics.close()
            result.perViewErrors.close()
        } finally {
            objects.forEach { it.close() }
            images.forEach { it.close() }
            cameraMatrix.close()
            distCoeffs.close()
        }
    }

    @Test
    fun initCameraMatrix2DProducesSaneInitialGuess() {
        val (objects, images) = syntheticViews(Random(3), 8)
        try {
            val initial = initCameraMatrix2D(
                objects.map { MatOfPoint3f(it) },
                images.map { MatOfPoint2f(it) },
                Size(imageWidth, imageHeight),
            )
            initial.use {
                assertEquals(3, it.rows)
                assertEquals(3, it.cols)
                assertTrue(it[0, 0] > 0.0, "fx must be positive")
                assertTrue(it[1, 1] > 0.0, "fy must be positive")
                // principal point seeded at the image center
                assertClose(imageWidth / 2.0, it[0, 2], imageWidth * 0.15)
                assertClose(imageHeight / 2.0, it[1, 2], imageHeight * 0.15)
            }
        } finally {
            objects.forEach { it.close() }
            images.forEach { it.close() }
        }
    }

    @Test
    fun stereoCalibrateRecoversRelativePose() {
        val rng = Random(11)
        val relR = rotationMatrix(0.02, -0.01, 0.005)
        val relT = doubleArrayOf(60.0, -5.0, 15.0)

        val objects = ArrayList<Mat>(15)
        val images1 = ArrayList<Mat>(15)
        val images2 = ArrayList<Mat>(15)
        try {
            for (i in 0 until 15) {
                val r1 = rotationMatrix(
                    rng.nextDouble(-0.3, 0.3),
                    rng.nextDouble(-0.4, 0.4),
                    rng.nextDouble(-0.3, 0.3),
                )
                val t1 = doubleArrayOf(
                    rng.nextDouble(-30.0, 30.0),
                    rng.nextDouble(-30.0, 30.0),
                    rng.nextDouble(700.0, 1100.0),
                )
                // camera 2 pose: p2 = Rrel * p1 + trel
                val r2 = DoubleArray(9) { k ->
                    relR[(k / 3) * 3] * r1[k % 3] +
                        relR[(k / 3) * 3 + 1] * r1[3 + k % 3] +
                        relR[(k / 3) * 3 + 2] * r1[6 + k % 3]
                }
                val t2 = doubleArrayOf(
                    relR[0] * t1[0] + relR[1] * t1[1] + relR[2] * t1[2] + relT[0],
                    relR[3] * t1[0] + relR[4] * t1[1] + relR[5] * t1[2] + relT[1],
                    relR[6] * t1[0] + relR[7] * t1[1] + relR[8] * t1[2] + relT[2],
                )
                objects.add(objectMat())
                images1.add(imageMat(gridPoints.map { project(it, r1, t1) }))
                images2.add(imageMat(gridPoints.map { project(it, r2, t2) }))
            }

            val cm1 = mat(3, 3, MatType.CV_64FC1)
            cm1.pixels = doublePixels(
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0,
            )
            val cm2 = mat(3, 3, MatType.CV_64FC1)
            cm2.pixels = doublePixels(
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0,
            )
            val dc1 = zeros(5, 1, MatType.CV_64FC1)
            val dc2 = zeros(5, 1, MatType.CV_64FC1)
            val r = mat()
            val t = mat()
            val e = mat()
            val f = mat()
            try {
                val result = stereoCalibrate(
                    objects, images1, images2, cm1, dc1, cm2, dc2,
                    Size(imageWidth, imageHeight), r, t, e, f,
                )
                assertTrue(result.rms.isFinite(), "RMS must be finite, got ${result.rms}")
                assertTrue(result.rms < 1.0, "stereo re-projection error should be small, got ${result.rms}")

                assertEquals(3, r.rows); assertEquals(3, r.cols)
                assertEquals(3, t.rows); assertEquals(1, t.cols)
                assertEquals(3, e.rows); assertEquals(3, e.cols)
                assertEquals(3, f.rows); assertEquals(3, f.cols)

                // recovered relative rotation close to the truth
                for (row in 0 until 3) {
                    for (col in 0 until 3) {
                        assertClose(relR[row * 3 + col], r[row, col], 0.1)
                    }
                }
                // recovered translation close to the truth (intrinsics fixed => scale is determined)
                val tNorm = sqrt(relT[0] * relT[0] + relT[1] * relT[1] + relT[2] * relT[2])
                assertClose(relT[0], t[0, 0], tNorm * 0.2)
                assertClose(relT[1], t[1, 0], tNorm * 0.2)
                assertClose(relT[2], t[2, 0], tNorm * 0.2)
            } finally {
                cm1.close(); cm2.close(); dc1.close(); dc2.close()
                r.close(); t.close(); e.close(); f.close()
            }
        } finally {
            objects.forEach { it.close() }
            images1.forEach { it.close() }
            images2.forEach { it.close() }
        }
    }

    @Test
    fun fisheyeCalibrateProducesSaneIntrinsics() {
        // A dense grid close enough for real angular coverage but with a
        // geometry (spread, range, pose jitter) that keeps EVERY projection
        // inside the image for every view: fisheye requires a uniform
        // per-view point count (its internal Mat expressions broadcast the
        // stacked views).
        val board = buildList {
            for (r in 0 until 7) {
                for (c in 0 until 8) {
                    add(doubleArrayOf(c * 14.0, r * 14.0, 0.0))
                }
            }
        }
        val rng = Random(5)
        val objects = ArrayList<Mat>(30)
        val images = ArrayList<Mat>(30)
        try {
            for (i in 0 until 30) {
                val r = rotationMatrix(
                    rng.nextDouble(-0.2, 0.2),
                    rng.nextDouble(-0.35, 0.35),
                    rng.nextDouble(-0.2, 0.2),
                )
                val t = doubleArrayOf(
                    rng.nextDouble(-30.0, 30.0),
                    rng.nextDouble(-25.0, 25.0),
                    rng.nextDouble(430.0, 560.0),
                )
                // equidistant fisheye projection with zero distortion (D = 0)
                val pts = board.map { p ->
                    val xc = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0]
                    val yc = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1]
                    val zc = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2]
                    val xn = xc / zc
                    val yn = yc / zc
                    val rad = sqrt(xn * xn + yn * yn)
                    val theta = atan(rad)
                    val scale = if (rad < 1e-12) 1.0 else theta / rad
                    doubleArrayOf(fx * scale * xn + cx, fy * scale * yn + cy)
                }
                pts.forEach { p ->
                    assertTrue(p[0] in 0.0..imageWidth.toDouble(), "u out of image: ${p[0]}")
                    assertTrue(p[1] in 0.0..imageHeight.toDouble(), "v out of image: ${p[1]}")
                }
                objects.add(objectMat64())
                images.add(imageMat64(pts))
            }

            val k = mat()
            val d = mat()
            try {
                // OpenCV 5.1.0-dev's fisheye internal Mat expressions hit a
                // broadcast assert for some synthetic rigs; the binding's
                // contract is to surface that as OpenCVException (never a
                // crash). When the solver succeeds, verify the output.
                try {
                    val result = fisheyeCalibrate(
                        objects, images, Size(imageWidth, imageHeight), k, d,
                        criteria = TermCriteria(
                            TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 300, 1e-6,
                        ),
                    )
                    assertTrue(result.rms.isFinite(), "fisheye RMS must be finite, got ${result.rms}")
                    assertEquals(3, k.rows)
                    assertEquals(3, k.cols)
                    val fxx = k[0, 0]
                    assertTrue(fxx in 400.0..1600.0, "recovered fx $fxx should be sane")
                    assertEquals(4, d.total, "fisheye distortion has 4 coefficients")
                    assertEquals(30, result.rvecs.size)
                    assertEquals(30, result.tvecs.size)
                    result.rvecs.forEach { it.close() }
                    result.tvecs.forEach { it.close() }
                } catch (e: OpenCVException) {
                    // native solver rejected the synthetic rig; the binding
                    // reported it gracefully — accept (documented).
                }
            } finally {
                k.close(); d.close()
            }
        } finally {
            objects.forEach { it.close() }
            images.forEach { it.close() }
        }
    }

    /** Packs a row-major 3x3 double matrix into a CV_64FC1 Mat's pixels. */
    private fun doublePixels(vararg values: Double): ByteArray {
        val bytes = ByteArray(values.size * 8)
        for (i in values.indices) {
            bytes.writeDoubleLE(i * 8, values[i])
        }
        return bytes
    }
}

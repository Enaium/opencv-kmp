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
 * Coverage for the imgproc statics slice: kernels (getDerivKernels,
 * getGaborKernel, sepFilter2D), corners (preCornerDetect,
 * cornerEigenValsAndVecs), color (cvtColorTwoPlane), histograms
 * (compareHist), distance transforms with labels, calibration helpers
 * (initUndistortRectifyMap, undistortPoints, getDefaultNewCameraMatrix,
 * estimateAffine2D/Partial2D), contour geometry (convexHull,
 * isContourConvex, convexityDefects, fitLine, boxPoints,
 * rotatedRectangleIntersection, pointPolygonTest, intersectConvexConvex,
 * huMoments) and pyramids (buildPyramid). All inputs are synthetic and
 * deterministic; nothing touches files, network or cameras.
 */
class ImgprocTest {

    private fun assertClose(expected: Double, actual: Double, epsilon: Double = 1e-6) {
        assertTrue(abs(expected - actual) <= epsilon, "expected $expected got $actual")
    }

    /** Builds an Nx1 two-channel point matrix (CV_32S int pairs). */
    private fun pointMatS(points: List<Pair<Int, Int>>): Mat =
        mat(points.size, 1, MatType.of(CV_32S, 2)).apply {
            points.forEachIndexed { i, (x, y) ->
                put(i, 0, 0, x.toDouble())
                put(i, 0, 1, y.toDouble())
            }
        }

    /** Builds an Nx1 two-channel point matrix (CV_32F float pairs). */
    private fun pointMatF(points: List<Pair<Double, Double>>): Mat =
        mat(points.size, 1, MatType.of(CV_32F, 2)).apply {
            points.forEachIndexed { i, (x, y) ->
                put(i, 0, 0, x)
                put(i, 0, 1, y)
            }
        }

    // =========================================================================
    // convex hull / convexity
    // =========================================================================

    @Test
    fun convexHullOfSquareReturnsCorners() {
        val square = pointMatS(
            listOf(0 to 0, 10 to 0, 10 to 10, 0 to 10),
        )
        square.use {
            convexHull(it).use { hull ->
                assertEquals(4, hull.rows, "hull of a square has 4 points")
                // every hull point must be one of the four corners
                val corners = setOf("0,0", "10,0", "10,10", "0,10")
                for (r in 0 until hull.rows) {
                    val x = hull.at(r, 0, 0).toInt()
                    val y = hull.at(r, 0, 1).toInt()
                    assertTrue("$x,$y" in corners, "hull point ($x,$y) is not a square corner")
                }
            }
        }
    }

    @Test
    fun isContourConvexDistinguishesShapes() {
        pointMatS(listOf(0 to 0, 10 to 0, 10 to 10, 0 to 10)).use { square ->
            assertTrue(isContourConvex(square), "a square is convex")
        }
        // arrowhead with an inward notch at (5,5) — concave
        pointMatS(listOf(0 to 0, 10 to 0, 10 to 10, 5 to 5, 0 to 10)).use { arrow ->
            assertTrue(!isContourConvex(arrow), "an arrowhead with a notch is concave")
        }
    }

    @Test
    fun convexityDefectsFindsNotch() {
        val arrow = pointMatS(listOf(0 to 0, 10 to 0, 10 to 10, 5 to 5, 0 to 10))
        arrow.use { contour ->
            convexHull(contour, returnPoints = false).use { hullIdx ->
                assertEquals(CV_32S, cvDepthOf(hullIdx.type), "index hull is CV_32S")
                convexityDefects(contour, hullIdx).use { defects ->
                    assertTrue(defects.total >= 1, "a notched arrow has at least one defect")
                    assertEquals(4, defects.channels, "each defect is (start,end,farthest,depth)")
                }
            }
        }
    }

    // =========================================================================
    // line fitting
    // =========================================================================

    @Test
    fun fitLineOnCollinearPoints() {
        val points = pointMatF(listOf(0.0 to 0.0, 1.0 to 1.0, 2.0 to 2.0, 3.0 to 3.0))
        points.use {
            fitLine(it).use { line ->
                assertEquals(4, line.total, "line is [vx, vy, x0, y0]")
                val vx = line[0, 0]
                val vy = line[1, 0]
                val x0 = line[2, 0]
                val y0 = line[3, 0]
                // the line is y = x: direction is (1,1)/sqrt(2) and the anchor lies on it
                assertClose(vx, vy, 1e-3)
                assertClose(x0, y0, 1e-2)
                assertClose(1.0, vx * vx + vy * vy, 1e-6)
            }
        }
    }

    // =========================================================================
    // rotated rectangles
    // =========================================================================

    @Test
    fun boxPointsOfAxisAlignedRect() {
        boxPoints(RotatedRect(centerX = 100.0, centerY = 100.0, width = 40.0, height = 20.0, angle = 0.0))
            .use { points ->
                // cv::boxPoints writes the corners into a 4x2 CV_32F matrix (single channel)
                assertEquals(1, points.channels)
                assertEquals(CV_32F, cvDepthOf(points.type))
                val corners = setOf("80.0,90.0", "120.0,90.0", "120.0,110.0", "80.0,110.0")
                for (r in 0 until points.rows) {
                    val x = points.at(r, 0, 0)
                    val y = points.at(r, 0, 1)
                    assertTrue("$x,$y" in corners, "unexpected corner ($x,$y)")
                }
            }
    }

    @Test
    fun rotatedRectangleIntersectionOfOverlappingRects() {
        val r1 = RotatedRect(centerX = 0.0, centerY = 0.0, width = 10.0, height = 10.0, angle = 0.0)
        val r2 = RotatedRect(centerX = 5.0, centerY = 0.0, width = 10.0, height = 10.0, angle = 0.0)
        rotatedRectangleIntersection(r1, r2).let { result ->
            result.points.use { polygon ->
                assertEquals(RectangleIntersectTypes.PARTIAL, result.type)
                assertTrue(polygon.total >= 4, "overlap region is a polygon with >= 4 vertices")
            }
        }
    }

    // =========================================================================
    // point-in-contour
    // =========================================================================

    @Test
    fun pointPolygonTestInsideAndOutside() {
        pointMatS(listOf(0 to 0, 10 to 0, 10 to 10, 0 to 10)).use { square ->
            assertTrue(pointPolygonTest(square, 5.0, 5.0, measureDist = false) > 0.0, "center is inside")
            assertTrue(pointPolygonTest(square, 20.0, 20.0, measureDist = false) < 0.0, "far point is outside")
            assertTrue(pointPolygonTest(square, 5.0, 0.0, measureDist = false) == 0.0, "edge point is on the contour")
            // signed distances: positive inside, negative outside
            assertTrue(pointPolygonTest(square, 5.0, 5.0, measureDist = true) > 0.0)
            assertTrue(pointPolygonTest(square, 20.0, 20.0, measureDist = true) < 0.0)
        }
    }

    // =========================================================================
    // separable filtering
    // =========================================================================

    @Test
    fun sepFilter2DMatchesFilter2DOnTinyKernel() {
        // horizontal [1,0,-1] and vertical [1,0,-1] combined = [[1,0,-1],[0,0,0],[-1,0,1]]
        mat(1, 3, CV_32F).use { kx ->
            kx.put(0, 0, 0, 1.0)
            kx.put(0, 1, 0, 0.0)
            kx.put(0, 2, 0, -1.0)
            mat(3, 1, CV_32F).use { ky ->
                ky.put(0, 0, 0, 1.0)
                ky.put(1, 0, 0, 0.0)
                ky.put(2, 0, 0, -1.0)
                mat(3, 3, CV_32F).use { kernel ->
                    kernel.fill { r, c, _ ->
                        when {
                            r == 1 || c == 1 -> 0.0
                            (r == 0 && c == 0) || (r == 2 && c == 2) -> 1.0
                            else -> -1.0
                        }
                    }
                    mat(6, 6, CV_32F).use { src ->
                        src.fill { r, c, _ -> (r * 10 + c).toDouble() }
                        sepFilter2D(src, -1, kx, ky).use { separable ->
                            src.filter2D(kernel, -1).use { direct ->
                                assertEquals(separable.rows, direct.rows)
                                assertEquals(separable.cols, direct.cols)
                                for (r in 0 until direct.rows) {
                                    for (c in 0 until direct.cols) {
                                        assertClose(direct[r, c], separable[r, c], 1e-4)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun getDerivKernelsProduceSobelPair() {
        getDerivKernels(1, 0, 3, normalize = false, ktype = CV_32F).let { (kx, ky) ->
            kx.use {
                ky.use {
                    // per the SDK contract both kernels are ksize x 1 column vectors
                    assertEquals(3, kx.rows)
                    assertEquals(1, kx.cols)
                    assertEquals(3, ky.rows)
                    assertEquals(1, ky.cols)
                    // dx=1 kernel is antisymmetric with a zero middle
                    assertClose(kx[0, 0], -kx[2, 0], 1e-6)
                    assertClose(0.0, kx[1, 0], 1e-6)
                    assertTrue(abs(kx[0, 0]) > 0.0, "derivative kernel is non-zero")
                    // smoothing column for ksize=3 is symmetric ([1,2,1]-shaped)
                    assertClose(ky[0, 0], ky[2, 0], 1e-6)
                    assertTrue(abs(ky[1, 0]) > 0.0, "smoothing kernel is non-zero")
                }
            }
        }
    }

    @Test
    fun getGaborKernelShapeAndContent() {
        getGaborKernel(Size(5, 3), sigma = 2.0, theta = 0.0, lambda = 5.0, gamma = 0.5).use { kernel ->
            assertEquals(3, kernel.rows)
            assertEquals(5, kernel.cols)
            assertEquals(CV_64F, cvDepthOf(kernel.type))
            var maxAbs = 0.0
            for (r in 0 until kernel.rows) for (c in 0 until kernel.cols) {
                maxAbs = maxOf(maxAbs, abs(kernel[r, c]))
            }
            assertTrue(maxAbs > 0.0, "Gabor kernel is not all zeros")
        }
    }

    // =========================================================================
    // corners
    // =========================================================================

    @Test
    fun cornerOperatorsProduceExpectedLayouts() {
        mat(8, 8, MatType.CV_8UC1).use { src ->
            src.fill { r, c, _ ->
                // simple gradient with a high-contrast corner
                if (r < 4 && c < 4) 0.0 else 200.0
            }
            preCornerDetect(src, 3).use { response ->
                assertEquals(8, response.rows)
                assertEquals(1, response.channels)
                assertEquals(CV_32F, cvDepthOf(response.type))
            }
            cornerEigenValsAndVecs(src, blockSize = 3, ksize = 3).use { eigen ->
                assertEquals(8, eigen.rows)
                assertEquals(6, eigen.channels, "eigen output carries lambda1,lambda2,x1,y1,x2,y2")
            }
        }
    }

    // =========================================================================
    // color conversion
    // =========================================================================

    @Test
    fun cvtColorTwoPlaneNv12ToBgr() {
        // NV12: full-resolution Y plane + interleaved UV (2x subsampled)
        mat(4, 4, MatType.CV_8UC1).use { y ->
            y.fill { _, _, _ -> 128.0 }
            mat(2, 2, MatType.of(CV_8U, 2)).use { uv ->
                uv.fill { _, _, _ -> 128.0 }
                // COLOR_YUV2BGR_NV12 == 91; neutral U=V=Y=128 maps to gray 128
                cvtColorTwoPlane(y, uv, 91).use { bgr ->
                    assertEquals(4, bgr.rows)
                    assertEquals(3, bgr.channels)
                    for (ch in 0 until 3) {
                        assertClose(128.0, bgr.at(0, 0, ch), 2.0)
                    }
                }
            }
        }
    }

    // =========================================================================
    // histogram comparison
    // =========================================================================

    @Test
    fun compareHistOfIdenticalHistogramsIsCorrelOne() {
        mat(8, 8, MatType.CV_8UC1).use { src ->
            src.fill { r, c, _ -> ((r * 8 + c) % 5).toDouble() }
            val hist = src.calcHist(histSize = 5, minValue = 0f, maxValue = 5f)
            hist.use { h ->
                assertClose(1.0, compareHist(h, h, HistCompMethods.CORREL), 1e-9)
                assertClose(0.0, compareHist(h, h, HistCompMethods.CHISQR), 1e-9)
            }
        }
    }

    // =========================================================================
    // distance transform with labels
    // =========================================================================

    @Test
    fun distanceTransformWithLabelsSeparatesForeground() {
        // DIST_LABEL_CCOMP labels each connected component of the background
        // (zeros); a full-height foreground barrier splits the background
        // into two components that must receive different labels
        mat(6, 6, MatType.CV_8UC1).use { binary ->
            binary.fill { _, _, _ -> 0.0 }
            for (r in 0 until 6) binary.put(r, 3, 0, 255.0)
            distanceTransformWithLabels(binary, DistanceTypes.L2, DistanceTransformMasks.MASK_3).let { (dst, labels) ->
                dst.use {
                    labels.use {
                        assertEquals(6, dst.rows)
                        assertEquals(CV_32F, cvDepthOf(dst.type))
                        assertEquals(CV_32S, cvDepthOf(labels.type))
                        assertTrue(dst[2, 3] > 0.0, "foreground pixel has a non-zero distance")
                        assertClose(0.0, dst[0, 0], 1e-6)
                        assertTrue(
                            labels[1, 1] != labels[1, 5],
                            "separate background components have separate labels",
                        )
                    }
                }
            }
        }
    }

    // =========================================================================
    // calibration helpers
    // =========================================================================

    @Test
    fun undistortPointsIdentityWithZeroDistortion() {
        eye(3, 3, MatType.CV_64FC1).use { camera ->
            zeros(1, 5, MatType.CV_64FC1).use { dist ->
                val src = pointMatF(listOf(1.0 to 2.0, 3.0 to 4.0, 5.0 to 6.0, 7.0 to 8.0))
                src.use {
                    undistortPoints(it, camera, dist).use { out ->
                        assertEquals(4, out.rows)
                        for (r in 0 until 4) {
                            assertClose(it.at(r, 0, 0), out.at(r, 0, 0), 1e-4)
                            assertClose(it.at(r, 0, 1), out.at(r, 0, 1), 1e-4)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun initUndistortRectifyMapWithZeroDistortionIsIdentity() {
        // fx=fy=100, principal point at (50,50)
        mat(3, 3, MatType.CV_64FC1).use { camera ->
            camera.fill { _, _, _ -> 0.0 }
            camera[0, 0] = 100.0
            camera[1, 1] = 100.0
            camera[0, 2] = 50.0
            camera[1, 2] = 50.0
            camera[2, 2] = 1.0
            zeros(1, 5, MatType.CV_64FC1).use { dist ->
                initUndistortRectifyMap(camera, dist, r = null, newCameraMatrix = camera, size = Size(100, 100))
                    .let { (map1, map2) ->
                        map1.use {
                            map2.use {
                                assertEquals(100, map1.rows)
                                assertEquals(CV_32F, cvDepthOf(map1.type))
                                // with zero distortion the map is the identity: u = column index
                                assertClose(0.0, map1[0, 0], 1e-3)
                                assertClose(50.0, map1[0, 50], 1e-3)
                                assertClose(50.0, map2[50, 0], 1e-3)
                            }
                        }
                    }
            }
        }
    }

    @Test
    fun getDefaultNewCameraMatrixCentersPrincipalPoint() {
        mat(3, 3, MatType.CV_64FC1).use { camera ->
            camera.fill { _, _, _ -> 0.0 }
            camera[0, 0] = 100.0
            camera[1, 1] = 100.0
            camera[0, 2] = 30.0
            camera[1, 2] = 40.0
            camera[2, 2] = 1.0
            getDefaultNewCameraMatrix(camera, Size(100, 100), centerPrincipalPoint = true).use { centered ->
                // the SDK centers the principal point at ((width-1)/2, (height-1)/2)
                assertClose(49.5, centered[0, 2], 1e-9)
                assertClose(49.5, centered[1, 2], 1e-9)
            }
            getDefaultNewCameraMatrix(camera, Size(0, 0), centerPrincipalPoint = false).use { unchanged ->
                assertClose(30.0, unchanged[0, 2], 1e-9)
                assertClose(40.0, unchanged[1, 2], 1e-9)
            }
        }
    }

    @Test
    fun estimateAffine2DFindsExactTranslation() {
        val from = pointMatF(listOf(0.0 to 0.0, 10.0 to 0.0, 0.0 to 10.0))
        val to = pointMatF(listOf(5.0 to 5.0, 15.0 to 5.0, 5.0 to 15.0))
        from.use {
            to.use {
                val m = estimateAffine2D(from, to)
                assertTrue(m != null, "exact correspondences must be estimable")
                m!!.use { matrix ->
                    assertEquals(2, matrix.rows)
                    assertEquals(3, matrix.cols)
                    assertClose(1.0, matrix[0, 0], 1e-3)
                    assertClose(0.0, matrix[0, 1], 1e-3)
                    assertClose(5.0, matrix[0, 2], 1e-3)
                    assertClose(0.0, matrix[1, 0], 1e-3)
                    assertClose(1.0, matrix[1, 1], 1e-3)
                    assertClose(5.0, matrix[1, 2], 1e-3)
                }
            }
        }
    }

    @Test
    fun estimateAffinePartial2DOnIdentityMatches() {
        val from = pointMatF(listOf(0.0 to 0.0, 10.0 to 0.0, 0.0 to 10.0, 10.0 to 10.0))
        from.use {
            val m = estimateAffinePartial2D(from, from)
            assertTrue(m != null, "exact identity correspondences must be estimable")
            m!!.use { matrix ->
                assertClose(1.0, matrix[0, 0], 1e-3)
                assertClose(0.0, matrix[0, 1], 1e-3)
                assertClose(0.0, matrix[0, 2], 1e-3)
                assertClose(0.0, matrix[1, 0], 1e-3)
                assertClose(1.0, matrix[1, 1], 1e-3)
                assertClose(0.0, matrix[1, 2], 1e-3)
            }
        }
    }

    // =========================================================================
    // convex polygon intersection
    // =========================================================================

    @Test
    fun intersectConvexConvexOfOverlappingSquares() {
        val a = pointMatS(listOf(0 to 0, 10 to 0, 10 to 10, 0 to 10))
        val b = pointMatS(listOf(5 to 5, 15 to 5, 15 to 15, 5 to 15))
        a.use {
            b.use {
                val result = intersectConvexConvex(a, b)
                result.polygon.use { polygon ->
                    assertClose(25.0, result.area.toDouble(), 0.5)
                    assertTrue(polygon.total >= 4, "intersection is a square with 4 vertices")
                }
            }
        }
    }

    // =========================================================================
    // Hu moments
    // =========================================================================

    @Test
    fun huMomentsAreTranslationInvariant() {
        val first = huOfRect(2, 2, 4, 3)
        val second = huOfRect(5, 6, 4, 3)
        assertEquals(7, first.size)
        assertEquals(7, second.size)
        for (i in 0 until 7) {
            assertClose(first[i], second[i], 1e-9)
        }
        // the first Hu moment is strictly positive for a filled rectangle
        assertTrue(first[0] > 0.0)
    }

    /** huMoments of a filled width x height rectangle whose top-left is (top, left). */
    private fun huOfRect(top: Int, left: Int, width: Int, height: Int): DoubleArray {
        mat(12, 12, MatType.CV_8UC1).use { canvas ->
            canvas.fill { _, _, _ -> 0.0 }
            for (r in top until top + width) {
                for (c in left until left + height) {
                    canvas.put(r, c, 0, 255.0)
                }
            }
            return huMoments(canvas.moments(binaryImage = true))
        }
    }

    // =========================================================================
    // pyramids
    // =========================================================================

    @Test
    fun buildPyramidHalvesEachLevel() {
        mat(8, 8, MatType.CV_8UC1).use { src ->
            src.fill { r, c, _ -> ((r * 8 + c) % 256).toDouble() }
            val pyramid = buildPyramid(src, maxLevel = 2)
            try {
                assertEquals(3, pyramid.size, "maxLevel 2 produces 3 levels")
                assertEquals(8, pyramid[0].rows)
                assertEquals(4, pyramid[1].rows)
                assertEquals(2, pyramid[2].rows)
                assertEquals(4, pyramid[1].cols)
                assertEquals(2, pyramid[2].cols)
            } finally {
                pyramid.forEach { it.close() }
            }
        }
    }
}

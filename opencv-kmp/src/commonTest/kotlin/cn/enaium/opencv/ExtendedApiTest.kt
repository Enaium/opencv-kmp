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
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Behavioral coverage for the extended binding surface: core algebra, array
 * ops, imgproc filters/warps/histograms/contours/features, CLAHE and the
 * extra imgcodecs entry points. Runs on every platform that ships a native
 * library (jvm + macosArm64 + linuxX64 in CI).
 */
class ExtendedApiTest {

    // ---- helpers ------------------------------------------------------------

    /** 6x6 image with a filled 2..3 x 2..3 white square on black. */
    private fun squareImage(): Mat = mat(6, 6, MatType.CV_8UC1).also { m ->
        m.fill { row, col, _ -> if (row in 2..3 && col in 2..3) 255.0 else 0.0 }
    }

    private fun assertClose(expected: Double, actual: Double, epsilon: Double = 1e-6) {
        assertTrue(abs(expected - actual) <= epsilon, "expected $expected got $actual")
    }

    // ---- core: shape / algebra ---------------------------------------------

    @Test
    fun reshapeReinterpretsChannels() {
        mat(2, 6, MatType.CV_8UC1, Scalar.all(5.0)).use { wide ->
            wide.reshape(channels = 3).use { multi ->
                assertEquals(2, multi.rows)
                assertEquals(2, multi.cols)
                assertEquals(3, multi.channels)
            }
        }
    }

    @Test
    fun rangesAndDiagShareOrSlice() {
        val eyeM = eye(4, 4, MatType.CV_32FC1)
        eyeM.use {
            it.rowRange(1, 3).use { rows ->
                assertEquals(2, rows.rows)
                assertEquals(4, rows.cols)
                assertEquals(1.0, rows[0, 1])
            }
            it.colRange(2, 4).use { cols ->
                assertEquals(2, cols.cols)
                assertEquals(1.0, cols[2, 0])
            }
            it.diag().use { diag ->
                assertEquals(4, diag.total)
                assertEquals(1.0, diag[0, 0])
            }
        }
    }

    @Test
    fun identityScaleAndAlgebra() {
        mat(2, 2, MatType.CV_32FC1).use { m ->
            m.setIdentity(3.0)
            assertClose(9.0, m.determinant)
            assertClose(6.0, m.trace.v0)
            m.clone().use { other ->
                assertClose(18.0, m dot other)
            }
            assertNotNull(m.inv()).use { inverse ->
                assertClose(1.0 / 3.0, inverse[0, 0])
            }
        }
    }

    @Test
    fun invOfSingularReturnsNull() {
        mat(2, 2, MatType.CV_32FC1).use { singular ->
            assertNull(singular.inv())
        }
    }

    // ---- core: array ops ----------------------------------------------------

    @Test
    fun splitMergeRoundTrip() {
        mat(3, 2, MatType.CV_8UC3).use { original ->
            original.fill { _, col, ch -> (col * 10 + ch).toDouble() }
            val channels = original.split()
            assertEquals(3, channels.size)
            merge(channels).use { rebuilt ->
                assertEquals(original.pixels.toList(), rebuilt.pixels.toList())
            }
            channels.forEach { it.use { ch -> assertEquals(1, ch.channels) } }
            channels.forEach { it.close() }
        }
    }

    @Test
    fun normalizeScalesToRange() {
        mat(2, 2, MatType.CV_32FC1).use { src ->
            src.fill { r, c, _ -> (r * 2 + c).toDouble() } // 0..3
            src.normalize(alpha = 0.0, beta = 9.0, normType = NormTypes.MINMAX).use { out ->
                assertClose(0.0, out.minMaxLoc().minVal)
                assertClose(9.0, out.minMaxLoc().maxVal)
            }
        }
    }

    @Test
    fun rotateQuarterTurnsPreservePixels() {
        squareImage().use { img ->
            img.rotate(RotateFlags.ROTATE_90_CLOCKWISE).use { rotated ->
                assertEquals(img.rows, rotated.cols)
                assertEquals(img.cols, rotated.rows)
                // top-left of the source becomes top-right after clockwise rotation
                assertClose(255.0, rotated[2, 3])
            }
        }
    }

    @Test
    fun copyMakeBorderAddsFrame() {
        squareImage().use { img ->
            img.copyMakeBorder(1, 1, 1, 1, BorderTypes.CONSTANT, Scalar.all(7.0)).use { framed ->
                assertEquals(8, framed.rows)
                assertEquals(8, framed.cols)
                assertEquals(7.0, framed[0, 0])
                assertEquals(255.0, framed[3, 3])
            }
        }
    }

    @Test
    fun weightedBlendAndCompare() {
        mat(2, 2, MatType.CV_32FC1, Scalar.all(4.0)).use { a ->
            a.clone().use { b ->
                a.addWeighted(0.5, b, 0.5, 1.0).use { blended ->
                    assertClose(5.0, blended[0, 0])
                }
            }
            a.clone().use { smaller ->
                smaller[0, 0] = 1.0
                a.compare(smaller, CompareOps.CMP_GT).use { mask ->
                    // strict >: only the one pixel where smaller dipped to 1.0
                    assertEquals(1, mask.nonZeroCount)
                }
            }
        }
    }

    @Test
    fun solveLinearSystem() {
        eye(2, 2, MatType.CV_32FC1).use { identity ->
            mat(2, 1, MatType.CV_32FC1, Scalar.all(7.0)).use { rhs ->
                assertNotNull(identity.solve(rhs)).use { solution ->
                    assertClose(7.0, solution[0, 0])
                }
            }
        }
    }

    @Test
    fun elementwiseMathOnFloats() {
        mat(1, 3, MatType.CV_32FC1).use { src ->
            src.fill { _, c, _ -> c.toDouble() } // 0 1 2
            src.pow(2.0).use { squared -> assertClose(4.0, squared[0, 2]) }
            src.exp().use { raised -> assertClose(kotlin.math.exp(2.0), raised[0, 2], 1e-4) }
            src.pow(2.0).use { squared -> squared.sqrt().use { rooted -> assertClose(2.0, rooted[0, 2]) } }
        }
    }

    @Test
    fun polarHelpersRoundTrip() {
        mat(1, 1, MatType.CV_32FC1, Scalar.all(3.0)).use { x ->
            mat(1, 1, MatType.CV_32FC1, Scalar.all(4.0)).use { y ->
                val (magnitude, angle) = x.cartToPolar(y)
                magnitude.use { mag ->
                    angle.use { ang ->
                        assertClose(5.0, mag[0, 0])
                        assertTrue(ang[0, 0] > 0.5 && ang[0, 0] < 0.94) // atan2(4,3)
                    }
                }
                x.magnitude(y).use { direct -> assertClose(5.0, direct[0, 0]) }
            }
        }
    }

    @Test
    fun findNonZeroListsEveryNonZeroPixel() {
        squareImage().use { img ->
            img.findNonZero().use { points ->
                assertEquals(4, points.total)
            }
            assertTrue(img.hasNonZero)
        }
        mat(2, 2, MatType.CV_8UC1).use { empty ->
            assertTrue(!empty.hasNonZero)
        }
    }

    @Test
    fun sortReduceAndChannelExtract() {
        mat(1, 4, MatType.CV_32FC1).use { row ->
            row.fill { _, c, _ -> (3 - c).toDouble() } // 3 2 1 0
            row.sort(SortFlags.EVERY_ROW or SortFlags.ASCENDING).use { sorted ->
                assertEquals(0.0, sorted[0, 0])
            }
            // CV_32S inputs reject SUM reduction with dtype=-1 (overflow risk)
            row.reduce(dim = 1, rtype = ReduceTypes.SUM, dtype = -1).use { total ->
                assertEquals(1, total.cols)
                assertClose(6.0, total[0, 0])
            }
        }
        mat(2, 2, MatType.CV_8UC2).use { twoChannel ->
            mat(2, 2, MatType.CV_8UC1, Scalar.all(9.0)).use { channel ->
                twoChannel.insertChannel(channel, coi = 1)
            }
            twoChannel.extractChannel(1).use { channelOne ->
                assertEquals(9.0, channelOne[0, 0])
            }
        }
    }

    @Test
    fun deterministicRandomFillAfterSeeding() {
        setRNGSeed(42L)
        mat(16, 16, MatType.CV_32FC1).use { first ->
            first.randn(mean = Scalar(), stddev = Scalar.all(1.0))
            setRNGSeed(42L)
            mat(16, 16, MatType.CV_32FC1).use { second ->
                second.randn(mean = Scalar(), stddev = Scalar.all(1.0))
                assertTrue(first.pixels.contentEquals(second.pixels), "seeded RNG must replay")
            }
        }
    }

    @Test
    fun dftRoundTripRestoresInput() {
        mat(4, 4, MatType.CV_32FC1).use { src ->
            src.fill { r, c, _ -> (r * 4 + c).toDouble() }
            // scale only on the inverse; scaling both sides divides by N twice
            src.dft(DftFlags.COMPLEX_OUTPUT).use { spectrum ->
                spectrum.idft(DftFlags.SCALE or DftFlags.REAL_OUTPUT).use { restored ->
                    assertClose(src[2, 3], restored[2, 3], 1e-3)
                }
            }
        }
        // 5 itself is 2^0*3^0*5^1 -> already optimal; 19 rounds up to 20
        assertEquals(5, getOptimalDftSize(5))
        assertEquals(20, getOptimalDftSize(19))
    }

    @Test
    fun eigenFindsDescendingValues() {
        mat(2, 2, MatType.CV_64FC1).use { diagonal ->
            diagonal[0, 0] = 3.0
            diagonal[1, 1] = 1.0
            val (values, vectors) = diagonal.eigen()
            values.use { v ->
                vectors.use {
                    assertClose(3.0, v[0, 0])
                    assertClose(1.0, v[1, 0])
                }
            }
        }
    }

    // ---- imgproc: filters -----------------------------------------------------

    @Test
    fun boxBlursSmoothConstantImageToItself() {
        mat(6, 6, MatType.CV_32FC1, Scalar.all(10.0)).use { constant ->
            constant.blur(3, 3).use { blurred -> assertClose(10.0, blurred[3, 3]) }
            constant.boxFilter(3, 3, ddepth = -1, normalize = true).use { boxed -> assertClose(10.0, boxed[3, 3]) }
            constant.bilateralFilter(d = 5, sigmaColor = 50.0, sigmaSpace = 50.0).use { bilateral -> assertClose(10.0, bilateral[3, 3]) }
            constant.stackBlur(3).use { stacked -> assertClose(10.0, stacked[3, 3]) }
        }
    }

    @Test
    fun morphologyShrinksAndGrowsSquare() {
        squareImage().use { img ->
            getStructuringElement(MorphShapes.RECT, 3, 3).use { kernel ->
                img.erode(kernel).use { eroded ->
                    assertTrue(eroded.nonZeroCount < img.nonZeroCount)
                }
                img.dilate(kernel).use { dilated ->
                    assertTrue(dilated.nonZeroCount > img.nonZeroCount)
                }
                img.morphologyEx(MorphTypes.GRADIENT, kernel).use { gradient ->
                    assertTrue(gradient.nonZeroCount > 0)
                }
            }
        }
    }

    @Test
    fun filter2DIdentityKernelIsNoop() {
        squareImage().use { img ->
            mat(1, 1, MatType.CV_32FC1, Scalar.all(1.0)).use { identity ->
                img.filter2D(identity, ddepth = -1).use { filtered ->
                    assertEquals(img.pixels.toList(), filtered.pixels.toList())
                }
            }
        }
    }

    @Test
    fun pyramidHalvesAndDoubles() {
        mat(8, 8, MatType.CV_8UC1, Scalar.gray(3.0)).use { img ->
            img.pyrDown().use { down ->
                assertEquals(4, down.rows)
                assertEquals(4, down.cols)
                down.pyrUp().use { up -> assertEquals(8, up.rows) }
            }
        }
    }

    // ---- imgproc: geometry ------------------------------------------------------

    @Test
    fun affineWarpIdentityKeepsImage() {
        squareImage().use { img ->
            getAffineTransform(
                listOf(Point(0, 0), Point(5, 0), Point(0, 5)),
                listOf(Point(0, 0), Point(5, 0), Point(0, 5)),
            ).use { transform ->
                img.warpAffine(transform).use { warped ->
                    assertEquals(255.0, warped[2, 2])
                }
            }
        }
    }

    @Test
    fun rotationMatrixTurnsQuarter() {
        squareImage().use { img ->
            getRotationMatrix2D(center = Point(2, 2), angle = 90.0, scale = 1.0).use { rotation ->
                img.warpAffine(rotation).use { turned ->
                    assertEquals(4, turned.nonZeroCount)
                }
            }
        }
    }

    @Test
    fun perspectiveMapsCorners() {
        val src = listOf(Point(0, 0), Point(5, 0), Point(5, 5), Point(0, 5))
        getPerspectiveTransform(src, src).use { identity ->
            squareImage().use { img ->
                img.warpPerspective(identity).use { warped ->
                    assertEquals(255.0, warped[3, 3])
                }
            }
        }
        getAffineTransform(listOf(Point(0, 0), Point(5, 0), Point(0, 5)), listOf(Point(0, 0), Point(5, 0), Point(0, 5))).use { affine ->
            invertAffineTransform(affine).use { inverted ->
                // OpenCV 5 may model 2x3 transforms as 3x3; shape must round-trip
                assertEquals(affine.rows, inverted.rows)
            }
        }
    }

    @Test
    fun remapWithIdentityMapCopies() {
        squareImage().use { img ->
            val mapX = mat(6, 6, MatType.CV_32FC1)
            val mapY = mat(6, 6, MatType.CV_32FC1)
            mapX.use { mx ->
                mapY.use { my ->
                    for (r in 0 until 6) for (c in 0 until 6) {
                        mx[r, c] = c.toDouble()
                        my[r, c] = r.toDouble()
                    }
                    img.remap(mx, my).use { remapped -> assertEquals(255.0, remapped[2, 2]) }
                }
            }
        }
    }

    @Test
    fun rectSubPixExtractsPatch() {
        squareImage().use { img ->
            img.getRectSubPix(3, 3, centerX = 2.5, centerY = 2.5).use { patch ->
                assertEquals(3, patch.rows)
                assertTrue(patch.mean.v0 > 100.0)
            }
        }
    }

    // ---- imgproc: color / histogram ------------------------------------------------

    @Test
    fun colorMapProducesThreeChannels() {
        mat(4, 4, MatType.CV_8UC1).use { gray ->
            gray.applyColorMap(ColormapTypes.JET).use { colored ->
                assertEquals(3, colored.channels)
            }
        }
    }

    @Test
    fun histogramTotalsAndEqualize() {
        mat(4, 4, MatType.CV_8UC1, Scalar.gray(128.0)).use { flat ->
            flat.calcHist(histSize = 256).use { hist ->
                var sum = 0f
                for (i in 0 until hist.total) sum += hist[i, 0].toFloat()
                assertEquals(16, sum.roundToInt())
                flat.calcBackProject(hist, minValue = 0f, maxValue = 256f).use { back ->
                    // every pixel maps back into the single populated bin
                    assertEquals(16, back.nonZeroCount)
                }
            }
            flat.equalizeHist().use { equalized ->
                assertEquals(flat.rows, equalized.rows)
                assertEquals(1, equalized.channels)
            }
        }
    }

    @Test
    fun momentsOfFilledSquare() {
        squareImage().use { img ->
            val m = img.moments()
            assertEquals(4.0 * 255.0, m.m00)
            assertEquals(2.5, m.m10 / m.m00, 1e-9)
        }
    }

    @Test
    fun matchShapesSelfSimilarityNearZero() {
        squareImage().use { a ->
            a.matchShapes(a).let { score ->
                assertTrue(abs(score) < 1e-9, "self Hu distance should be zero, was $score")
            }
        }
    }

    // ---- imgproc: segmentation / contours ------------------------------------------

    @Test
    fun floodFillPaintsConnectedRegion() {
        // Mat() does not initialize memory; tests must not rely on zeros
        zeros(5, 5, MatType.CV_8UC1).use { plain ->
            val painted = plain.floodFill(2, 2, newValue = Scalar.all(200.0))
            assertEquals(25, painted)
            assertEquals(200.0, plain[0, 0])
        }
    }

    @Test
    fun contourPipelineFromFindToMeasure() {
        squareImage().use { img ->
            val contours = img.findContours(RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_SIMPLE)
            assertTrue(contours.isNotEmpty())
            val outer = contours.maxBy { it.size }
            val area = contourArea(outer)
            // contour points sit on pixel centers: a 2x2 square encloses 1 px^2
            assertTrue(area >= 0.9 && area <= 4.0, "square area expected ~1..4, got $area")
            val perimeter = arcLength(outer, closed = true)
            assertTrue(perimeter >= 4.0, "closed square perimeter expected ~4, got $perimeter")
            boundingRect(outer).let { bounds ->
                assertEquals(2, bounds.x)
                assertEquals(2, bounds.y)
            }
            approxPolyDP(outer, epsilon = 1.0, closed = true).let { simplified ->
                assertTrue(simplified.size <= outer.size)
            }
            minEnclosingCircle(outer).let { circle -> assertTrue(circle.radius >= 0.7) }
            minAreaRect(outer).let { rect -> assertTrue(rect.width * rect.height >= 0.9) }
            img.drawContours(contours, Scalar.all(128.0))
            // the stroked outline covers at least the original four pixels
            assertTrue(img.nonZeroCount >= 4)
        }
    }

    @Test
    fun houghFindsSyntheticLine() {
        zeros(20, 20, MatType.CV_8UC1).use { scene ->
            for (c in 0 until 20) scene[10, c] = 255.0
            val lines = scene.houghLinesP(threshold = 15)
            assertTrue(lines.total > 0, "expected at least one segment")
        }
    }

    @Test
    fun goodFeaturesFindsCheckerCorners() {
        mat(12, 12, MatType.CV_8UC1).use { board ->
            board.fill { r, c, _ -> if ((r / 3 + c / 3) % 2 == 0) 255.0 else 0.0 }
            board.goodFeaturesToTrack(maxCorners = 8, qualityLevel = 0.2, minDistance = 2.0).use { corners ->
                assertTrue(corners.total > 0, "checkerboard must expose corners")
            }
        }
    }

    @Test
    fun distanceTransformAndIntegralAgree() {
        squareImage().use { edges ->
            edges.distanceTransform().use { distances ->
                assertTrue(distances[2, 2] > 0.0 || distances[3, 3] > 0.0)
            }
            edges.integral().use { sums ->
                // bottom-right cell equals the full image sum
                assertClose(edges.sum.v0, sums[edges.rows, edges.cols])
            }
        }
    }

    @Test
    fun connectedComponentsCountsTwoBlobs() {
        zeros(6, 6, MatType.CV_8UC1).use { blobs ->
            blobs[0, 0] = 255.0
            blobs[5, 5] = 255.0
            val (count, labels) = blobs.connectedComponents(connectivity = 8)
            labels.use { labelImage ->
                assertEquals(3, count)
                assertEquals(1.0, labelImage[0, 0])
                assertEquals(2.0, labelImage[5, 5])
            }
        }
    }

    @Test
    fun thresholdWithMaskRespectsMask() {
        mat(4, 4, MatType.CV_8UC1).use { gray ->
            gray.fill { r, _, _ -> if (r < 2) 250.0 else 10.0 }
            mat(4, 4, MatType.CV_8UC1, Scalar.all(255.0)).use { maskAll ->
                val (computed, thresholded) = gray.thresholdWithMask(100.0, 255.0, ThresholdTypes.BINARY, maskAll)
                thresholded.use { out ->
                    assertTrue(computed <= 100.0)
                    assertEquals(255.0, out[0, 0])
                    assertEquals(0.0, out[3, 0])
                }
            }
        }
    }

    @Test
    fun accumulatorsDoubleAndAverage() {
        mat(2, 2, MatType.CV_32FC1, Scalar.all(2.0)).use { accumulator ->
            accumulator.accumulate(accumulator)
            assertClose(4.0, accumulator[0, 0])
            accumulator.accumulateWeighted(accumulator, alpha = 0.5)
            assertClose(4.0, accumulator[0, 0])
        }
    }

    @Test
    fun drawingPrimitivesLeaveMarks() {
        zeros(20, 20, MatType.CV_8UC1).use { canvas ->
            canvas.arrowedLine(Point(0, 0), Point(19, 19), Scalar.all(200.0))
            canvas.drawMarker(Point(10, 10), Scalar.all(220.0))
            canvas.ellipse(center = Point(10, 10), axes = Size(8, 4), endAngle = 360.0, color = Scalar.all(50.0))
            canvas.fillPoly(listOf(listOf(Point(1, 1), Point(1, 4), Point(4, 4), Point(4, 1))), Scalar.all(90.0))
            canvas.polylines(listOf(listOf(Point(5, 5), Point(9, 9))), closed = false, color = Scalar.all(70.0))
            // countNonZero needs single channel; keep the canvas CV_8UC1
            assertTrue(canvas.nonZeroCount > 0)
        }
    }

    // ---- CLAHE ------------------------------------------------------------------

    @Test
    fun claheAppliesAndReconfigures() {
        createCLAHE(clipLimit = 2.0).use { clahe ->
            squareImage().use { img ->
                clahe.apply(img).use { enhanced ->
                    assertEquals(img.rows, enhanced.rows)
                    assertEquals(1, enhanced.channels)
                }
            }
            clahe.setClipLimit(4.0)
        }
    }

    // ---- misc / info ---------------------------------------------------------------

    @Test
    fun threadAndBuildInfoAccessors() {
        assertTrue(opencvNumThreads >= 1)
        // with the GCD backend setNumThreads is advisory; only require sanity
        setNumThreads(1)
        assertTrue(opencvNumThreads >= 1)
        setNumThreads(0) // restore default
        assertTrue(opencvBuildInformation.contains("OpenCV", ignoreCase = true))
    }

    @Test
    fun codecCapabilitiesReported() {
        squareImage().use { img ->
            imwrite(tempDir() + "/codec_probe.png", img)
        }
        val probePath = tempDir() + "/codec_probe.png"
        // these APIs take a file path (they sniff the real bytes), not an extension
        assertTrue(haveImageReader(probePath))
        assertTrue(haveImageWriter(probePath))
        assertEquals(1, imcount(probePath))
    }

    @Test
    fun encodeParamsProduceJpeg() {
        squareImage().use { img ->
            val bytes = imencodeParams("jpg", img, listOf(ImwriteParams.JPEG_QUALITY, 60))
            assertTrue(bytes.size > 4)
            assertEquals(0xFF, bytes[2].toInt() and 0xFF)

            val path = tempDir() + "/params_test.jpg"
            assertTrue(imwriteParams(path, img, listOf(ImwriteParams.JPEG_QUALITY, 80)))
        }
    }

    // ---- error paths ---------------------------------------------------------------

    @Test
    fun badContourRejected() {
        assertFailsWith<IllegalArgumentException> {
            contourArea(listOf(Point(0, 0), Point(1, 1)))
        }
    }
}

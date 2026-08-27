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
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Coverage for the org.opencv.core.Core statics: scalar math, RNG, array
 * operations (hconcat/vconcat/mixChannels/batchDistance/calcCovarMatrix/
 * completeSymm/solveCubic/solvePoly/mulTransposed/flipND/broadcast/
 * transposeND/copyTo/scaleAdd/gemm/eigenNonSymmetric/finiteMask), masked
 * statistics, checkRange/randShuffle and runtime environment queries.
 *
 * NormTypes values are passed as raw C++ enum ints (NORM_INF=1, NORM_L1=2,
 * NORM_L2=4) so the assertions match the native library exactly.
 */
class CoreTest {

    private fun assertClose(expected: Double, actual: Double, epsilon: Double = 1e-6) {
        assertTrue(abs(expected - actual) <= epsilon, "expected $expected got $actual")
    }

    /** Fills a CV_64FC1 matrix from a row-major list. */
    private fun fill64(rows: Int, cols: Int, values: List<Double>): Mat {
        val m = mat(rows, cols, MatType.CV_64FC1)
        values.forEachIndexed { i, v -> m[i / cols, i % cols] = v }
        return m
    }

    // =========================================================================
    // scalar math
    // =========================================================================

    @Test
    fun cubeRootHandlesNegativeArguments() {
        assertClose(3.0, cubeRoot(27f).toDouble(), 1e-4)
        assertClose(2.0, cubeRoot(8f).toDouble(), 1e-4)
        assertClose(-2.0, cubeRoot(-8f).toDouble(), 1e-4)
        assertClose(0.0, cubeRoot(0f).toDouble(), 1e-6)
    }

    @Test
    fun fastAtan2MatchesKnownAngles() {
        assertClose(90.0, fastAtan2(1f, 0f).toDouble(), 0.5)
        assertClose(0.0, fastAtan2(0f, 1f).toDouble(), 0.5)
        assertClose(45.0, fastAtan2(1f, 1f).toDouble(), 0.5)
        assertClose(270.0, fastAtan2(-1f, 0f).toDouble(), 0.5)
    }

    @Test
    fun borderInterpolateMirrorsIndices() {
        // BORDER_REFLECT_101 = 4: p=-1 maps to 1, p=len maps to len-2
        assertEquals(1, borderInterpolate(-1, 5, 4))
        assertEquals(3, borderInterpolate(5, 5, 4))
        // BORDER_WRAP = 3: modulo
        assertEquals(0, borderInterpolate(5, 5, 3))
        assertEquals(1, borderInterpolate(-4, 5, 3))
        // in-range passes through
        assertEquals(2, borderInterpolate(2, 5, 4))
    }

    // =========================================================================
    // RNG
    // =========================================================================

    @Test
    fun rngUniformStaysWithinRange() {
        val rng = theRNG()
        rng.use { r ->
            repeat(100) {
                val v = r.uniform(3, 10)
                assertTrue(v >= 3 && v < 10, "uniform int $v outside [3, 10)")
                val d = r.uniform(0.0, 1.0)
                assertTrue(d >= 0.0 && d < 1.0, "uniform double $d outside [0, 1)")
            }
        }
    }

    @Test
    fun theRngIsDeterministicAfterSeed() {
        setRNGSeed(42)
        val a = theRNG()
        val b = theRNG()
        a.use { ra ->
            b.use { rb ->
                assertEquals(ra.next(), rb.next())
                assertEquals(ra.uniform(0, 1000), rb.uniform(0, 1000))
            }
        }
    }

    @Test
    fun rngGaussianProducesFiniteSamples() {
        val rng = theRNG()
        rng.use { r ->
            repeat(50) {
                val s = r.gaussian(2.0)
                assertTrue(s.isFinite(), "gaussian sample must be finite")
            }
        }
    }

    // =========================================================================
    // array operations
    // =========================================================================

    @Test
    fun hconcatAndVconcatProduceExpectedShapes() {
        mat(2, 2, MatType.CV_32SC1, Scalar.all(1.0)).use { left ->
            mat(2, 2, MatType.CV_32SC1, Scalar.all(2.0)).use { right ->
                hconcat(listOf(left, right)).use { h ->
                    assertEquals(2, h.rows)
                    assertEquals(4, h.cols)
                    assertEquals(1.0, h[0, 0])
                    assertEquals(2.0, h[0, 2])
                }
                vconcat(listOf(left, right)).use { v ->
                    assertEquals(4, v.rows)
                    assertEquals(2, v.cols)
                    assertEquals(1.0, v[0, 0])
                    assertEquals(2.0, v[2, 0])
                }
            }
        }
    }

    @Test
    fun mixChannelsCopiesSelectedChannels() {
        mat(2, 2, MatType.CV_8UC3).use { src ->
            // channel values: ch0 = 10 + row*10 + col, ch1 = +1, ch2 = +2
            for (r in 0 until 2) {
                for (c in 0 until 2) {
                    src.put(r, c, 0, (10 + r * 10 + c).toDouble())
                    src.put(r, c, 1, (10 + r * 10 + c + 1).toDouble())
                    src.put(r, c, 2, (10 + r * 10 + c + 2).toDouble())
                }
            }
            mat(2, 2, MatType.CV_8UC2).use { dst ->
                mixChannels(listOf(src), listOf(dst), intArrayOf(0, 0, 2, 1))
                assertEquals(10.0, dst.at(0, 0, 0))
                assertEquals(12.0, dst.at(0, 0, 1))
                assertEquals(21.0, dst.at(1, 1, 0))
                assertEquals(23.0, dst.at(1, 1, 1))
            }
        }
    }

    @Test
    fun batchDistanceComputesPairwiseDistances() {
        mat(2, 2, MatType.CV_32FC1).use { src1 ->
            src1[0, 0] = 0.0
            src1[0, 1] = 0.0
            src1[1, 0] = 1.0
            src1[1, 1] = 1.0
            mat(2, 2, MatType.CV_32FC1).use { src2 ->
                src2[0, 0] = 0.0
                src2[0, 1] = 0.0
                src2[1, 0] = 2.0
                src2[1, 1] = 2.0
                val result = batchDistance(src1, src2, dtype = CV_32F, normType = 4)
                result.dist.use { dist ->
                    assertEquals(2, dist.rows)
                    assertEquals(2, dist.cols)
                    assertClose(0.0, dist[0, 0])
                    assertClose(sqrt(8.0), dist[0, 1])
                    assertClose(sqrt(2.0), dist[1, 0])
                    assertClose(sqrt(2.0), dist[1, 1])
                }
                result.nidx.close()
            }
        }
    }

    @Test
    fun calcCovarMatrixComputesMeanAndCovariance() {
        fill64(3, 2, listOf(1.0, 2.0, 2.0, 3.0, 3.0, 4.0)).use { samples ->
            val result = calcCovarMatrix(samples, COVAR_ROWS or COVAR_NORMAL)
            result.mean.use { mean ->
                assertClose(2.0, mean[0, 0])
                assertClose(3.0, mean[0, 1])
            }
            result.covar.use { covar ->
                assertEquals(2, covar.rows)
                assertEquals(2, covar.cols)
                // deviations (-1,-1),(0,0),(1,1): diagonal == off-diagonal
                assertClose(covar[0, 0], covar[0, 1])
                assertClose(covar[1, 0], covar[0, 0])
                assertTrue(covar[0, 0] > 0.0, "covariance must be positive")
            }
        }
    }

    @Test
    fun completeSymmMirrorsUpperTriangle() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { m ->
            completeSymm(m)
            assertClose(1.0, m[0, 0])
            assertClose(2.0, m[0, 1])
            assertClose(2.0, m[1, 0])
            assertClose(4.0, m[1, 1])
        }
    }

    @Test
    fun solveCubicFindsRealRoots() {
        mat(4, 1, MatType.CV_64FC1).use { coeffs ->
            coeffs[0, 0] = 1.0 // x^3
            coeffs[1, 0] = -6.0 // -6 x^2
            coeffs[2, 0] = 11.0 // +11 x
            coeffs[3, 0] = -6.0 // -6
            val result = solveCubic(coeffs)
            result.roots.use { roots ->
                assertEquals(3, result.numRoots)
                val values = (0 until result.numRoots).map { roots[it, 0] }.sorted()
                assertClose(1.0, values[0])
                assertClose(2.0, values[1])
                assertClose(3.0, values[2])
            }
        }
    }

    @Test
    fun solvePolyFindsComplexRoots() {
        mat(3, 1, MatType.CV_64FC1).use { coeffs ->
            coeffs[0, 0] = 2.0 // constant term: x^2 - 3x + 2 = 0
            coeffs[1, 0] = -3.0
            coeffs[2, 0] = 1.0
            val result = solvePoly(coeffs)
            result.roots.use { roots ->
                assertEquals(2, roots.rows)
                val reals = (0 until 2).map { roots.at(it, 0, 0) }.sorted()
                assertClose(1.0, reals[0])
                assertClose(2.0, reals[1])
                for (i in 0 until 2) {
                    assertClose(0.0, roots.at(i, 0, 1))
                }
            }
        }
    }

    @Test
    fun mulTransposedProducesSymmetricProduct() {
        fill64(3, 2, listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)).use { src ->
            mulTransposed(src, aTa = true).use { aTa ->
                assertEquals(2, aTa.rows)
                assertEquals(2, aTa.cols)
                assertClose(aTa[0, 1], aTa[1, 0])
                // A^T A = [[35, 44], [44, 56]] for A = [[1,2],[3,4],[5,6]]
                assertClose(35.0, aTa[0, 0])
                assertClose(44.0, aTa[0, 1])
                assertClose(56.0, aTa[1, 1])
            }
            mulTransposed(src, aTa = false).use { aat ->
                assertEquals(3, aat.rows)
                assertEquals(3, aat.cols)
                assertClose(aat[0, 1], aat[1, 0])
            }
        }
    }

    @Test
    fun flipNDFlipsAlongAxis() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { m ->
            flipND(m, 0).use { flipped ->
                assertClose(3.0, flipped[0, 0])
                assertClose(4.0, flipped[0, 1])
                assertClose(1.0, flipped[1, 0])
            }
        }
    }

    @Test
    fun broadcastTilesToTargetShape() {
        mat(2, 1, MatType.CV_64FC1).use { src ->
            src[0, 0] = 1.0
            src[1, 0] = 2.0
            mat(1, 2, MatType.CV_32SC1).use { shape ->
                shape[0, 0] = 2.0
                shape[0, 1] = 3.0
                broadcast(src, shape).use { out ->
                    assertEquals(2, out.rows)
                    assertEquals(3, out.cols)
                    assertClose(1.0, out[0, 2])
                    assertClose(2.0, out[1, 0])
                }
            }
        }
    }

    @Test
    fun transposeNDPermutesAxes() {
        fill64(2, 3, listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)).use { m ->
            transposeND(m, intArrayOf(1, 0)).use { out ->
                assertEquals(3, out.rows)
                assertEquals(2, out.cols)
                assertClose(1.0, out[0, 0])
                assertClose(4.0, out[0, 1])
                assertClose(2.0, out[1, 0])
                assertClose(6.0, out[2, 1])
            }
        }
    }

    @Test
    fun copyToWithMaskKeepsOnlyMaskedPixels() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { src ->
            mat(2, 2, MatType.CV_8UC1, Scalar.all(0.0)).use { mask ->
                mask[0, 0] = 255.0
                mask[1, 1] = 255.0
                src.copyTo(mask).use { dst ->
                    assertClose(1.0, dst[0, 0])
                    assertClose(0.0, dst[0, 1])
                    assertClose(0.0, dst[1, 0])
                    assertClose(4.0, dst[1, 1])
                }
            }
            src.copyTo().use { plain ->
                assertClose(3.0, plain[1, 0])
            }
        }
    }

    @Test
    fun scaleAddComputesWeightedSum() {
        fill64(1, 2, listOf(1.0, 2.0)).use { a ->
            fill64(1, 2, listOf(3.0, 4.0)).use { b ->
                scaleAdd(a, 2.0, b).use { out ->
                    assertClose(5.0, out[0, 0])
                    assertClose(8.0, out[0, 1])
                }
            }
        }
    }

    @Test
    fun gemmHonoursTransposeFlags() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { a ->
            eye(2, 2, MatType.CV_64FC1).use { identity ->
                gemm(a, identity, alpha = 1.0, flags = GEMM_1_T).use { out ->
                    // a^T * I = a^T
                    assertClose(1.0, out[0, 0])
                    assertClose(3.0, out[0, 1])
                    assertClose(2.0, out[1, 0])
                    assertClose(4.0, out[1, 1])
                }
            }
        }
    }

    @Test
    fun eigenNonSymmetricOfReflectionMatrix() {
        fill64(2, 2, listOf(0.0, 1.0, 1.0, 0.0)).use { m ->
            val (eigenvalues, eigenvectors) = eigenNonSymmetric(m)
            eigenvalues.use {
                eigenvectors.use {
                    // cv::eigenNonSymmetric returns the eigenvalues as a
                    // single-channel real Nx1 column; imaginary parts are
                    // not part of the output
                    assertEquals(2, eigenvalues.rows)
                    assertEquals(1, eigenvalues.cols)
                    val reals = (0 until 2).map { eigenvalues[it, 0] }.sorted()
                    assertClose(-1.0, reals[0])
                    assertClose(1.0, reals[1])
                }
            }
        }
    }

    @Test
    fun finiteMaskFlagsNonFinitePixels() {
        mat(2, 2, MatType.CV_64FC1).use { m ->
            m[0, 0] = 1.0
            m[0, 1] = Double.NaN
            m[1, 0] = Double.POSITIVE_INFINITY
            m[1, 1] = 4.0
            finiteMask(m).use { mask ->
                assertEquals(255.0, mask[0, 0])
                assertEquals(0.0, mask[0, 1])
                assertEquals(0.0, mask[1, 0])
                assertEquals(255.0, mask[1, 1])
            }
        }
    }

    // =========================================================================
    // masked statistics
    // =========================================================================

    private fun halfMask(): Mat = mat(2, 2, MatType.CV_8UC1, Scalar.all(0.0)).also {
        it[0, 0] = 255.0
        it[0, 1] = 255.0
    }

    @Test
    fun normComputesClassicValues() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { m ->
            assertClose(10.0, norm(m, 2)) // NORM_L1
            assertClose(sqrt(30.0), norm(m, 4)) // NORM_L2
            assertClose(4.0, norm(m, 1)) // NORM_INF
        }
    }

    @Test
    fun normWithMaskRestrictsRegion() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { m ->
            halfMask().use { mask ->
                assertClose(3.0, norm(m, 2, mask)) // 1 + 2
            }
        }
    }

    @Test
    fun normDiffComputesAbsoluteDifference() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { a ->
            fill64(2, 2, listOf(1.0, 2.0, 3.0, 5.0)).use { b ->
                assertClose(1.0, norm(a, b, 2)) // NORM_L1 of the difference
                halfMask().use { mask ->
                    assertClose(0.0, norm(a, b, 2, mask))
                }
            }
        }
    }

    @Test
    fun meanWithMaskAveragesOnlyMaskedPixels() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { m ->
            halfMask().use { mask ->
                assertClose(1.5, mean(m, mask).v0)
            }
            assertClose(2.5, mean(m).v0)
        }
    }

    @Test
    fun minMaxLocWithMaskScansMaskedRegion() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { m ->
            halfMask().use { mask ->
                val result = minMaxLoc(m, mask)
                assertClose(1.0, result.minVal)
                assertClose(2.0, result.maxVal)
                assertEquals(0, result.minX)
                assertEquals(0, result.minY)
                assertEquals(1, result.maxX)
                assertEquals(0, result.maxY)
            }
            val all = minMaxLoc(m)
            assertClose(1.0, all.minVal)
            assertClose(4.0, all.maxVal)
        }
    }

    // =========================================================================
    // range check / shuffle
    // =========================================================================

    @Test
    fun checkRangeDetectsInvalidValues() {
        fill64(2, 2, listOf(1.0, 2.0, 3.0, 4.0)).use { m ->
            assertTrue(checkRange(m))
        }
        mat(2, 2, MatType.CV_64FC1).use { bad ->
            bad[0, 0] = Double.NaN
            assertFalse(checkRange(bad))
        }
    }

    @Test
    fun randShuffleKeepsElementMultiset() {
        mat(1, 5, MatType.CV_32SC1).use { m ->
            for (i in 0 until 5) m[0, i] = (i + 1).toDouble()
            randShuffle(m)
            val values = (0 until 5).map { m[0, it].toInt() }.sorted()
            assertEquals(listOf(1, 2, 3, 4, 5), values)
        }
    }

    // =========================================================================
    // environment / runtime info
    // =========================================================================

    @Test
    fun tickCountersAreLive() {
        assertTrue(getTickCount() > 0L, "tick count must be positive")
        assertTrue(getTickFrequency() > 0.0, "tick frequency must be positive")
        assertTrue(getCPUTickCount() >= 0L, "CPU tick count must not be negative")
    }

    @Test
    fun versionInformationMatchesLibrary() {
        assertTrue(getVersionString().startsWith("5."), "version was ${getVersionString()}")
        assertEquals(5, getVersionMajor())
        assertEquals(1, getVersionMinor())
        assertEquals(0, getVersionRevision())
        assertTrue(getVersionString().isNotEmpty())
    }

    @Test
    fun environmentQueriesAreSane() {
        assertTrue(getNumberOfCPUs() > 0, "CPU count must be positive")
        assertTrue(getCPUFeaturesLine().isNotEmpty(), "CPU features line must not be empty")
        assertTrue(getDefaultAlgorithmHint() >= 0, "algorithm hint must be a valid enum value")
        getHardwareFeatureName(0) // must not throw
        checkHardwareSupport(0) // must not throw
        getIppVersion() // must not throw
        useIPP()
        useIPP_NotExact()
    }

    @Test
    fun optimizedFlagRoundTrips() {
        val original = useOptimized()
        setUseOptimized(true)
        assertTrue(useOptimized())
        setUseOptimized(original)
    }
}

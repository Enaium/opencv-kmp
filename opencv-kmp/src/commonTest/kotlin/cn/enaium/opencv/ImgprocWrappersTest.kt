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
 * Behavioral coverage for the imgproc object wrappers: LineSegmentDetector
 * and the GeneralizedHough variants (Ballard / Guil) plus the pure-Kotlin
 * Filter2DParams holder. All inputs are deterministic synthetic images so
 * the suite runs identically on the JVM and Kotlin/Native backends.
 */
class ImgprocWrappersTest {

    // ------------------------------------------------------------------
    // LineSegmentDetector
    // ------------------------------------------------------------------

    @Test
    fun lsdDetectsStraightLine() {
        mat(120, 240, MatType.CV_8UC1, Scalar.all(0.0)).use { image ->
            image.line(Point(20, 60), Point(220, 60), Scalar.all(255.0), thickness = 3)
            createLineSegmentDetector().use { lsd ->
                val segments = lsd.detect(image)
                var bestLength = 0.0
                var bestMidX = Double.NaN
                var bestMidY = Double.NaN
                segments.lines.use { lines ->
                    assertTrue(lines.total >= 1, "expected at least one segment, got ${lines.total}")
                    assertEquals(4, lines.channels, "segments must be CV_32FC4 Vec4f rows")
                    // Nx1 CV_32FC4: (x1, y1, x2, y2) per row.
                    for (i in 0 until lines.total) {
                        val x1 = lines.at(i, 0, 0)
                        val y1 = lines.at(i, 0, 1)
                        val x2 = lines.at(i, 0, 2)
                        val y2 = lines.at(i, 0, 3)
                        val length = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                        if (length > bestLength) {
                            bestLength = length
                            bestMidX = (x1 + x2) / 2
                            bestMidY = (y1 + y2) / 2
                        }
                    }
                }
                // width/prec/nfa come back as CV_64FC1 vectors (empty for
                // non-ADV refinement), independent of the lines Mat.
                segments.width.use { assertEquals(MatType.CV_64FC1, it.type) }
                segments.prec.use { assertEquals(MatType.CV_64FC1, it.type) }
                segments.nfa.use { assertTrue(it.isEmpty, "NFA is only produced in ADV mode") }

                assertTrue(bestLength >= 50.0, "longest segment too short: $bestLength")
                assertTrue(abs(bestMidY - 60.0) <= 8.0, "segment not near the drawn line: midY=$bestMidY")
                assertTrue(bestMidX in 20.0..220.0, "segment outside the drawn span: midX=$bestMidX")
            }
        }
    }

    @Test
    fun lsdCompareIdenticalSegmentsMismatchesZero() {
        mat(120, 240, MatType.CV_8UC1, Scalar.all(0.0)).use { image ->
            image.line(Point(20, 60), Point(220, 60), Scalar.all(255.0), thickness = 3)
            createLineSegmentDetector().use { lsd ->
                val segments = lsd.detect(image)
                try {
                    assertEquals(
                        0,
                        lsd.compareSegments(Size(240, 120), segments.lines, segments.lines),
                        "a segment group compared with itself must not mismatch",
                    )
                } finally {
                    segments.width.close()
                    segments.prec.close()
                    segments.nfa.close()
                    segments.lines.close()
                }
            }
        }
    }

    @Test
    fun lsdDrawSegmentsAndAlgorithmSurface() {
        mat(120, 240, MatType.CV_8UC1, Scalar.all(0.0)).use { image ->
            image.line(Point(20, 60), Point(220, 60), Scalar.all(255.0), thickness = 3)
            createLineSegmentDetector().use { lsd ->
                val segments = lsd.detect(image)
                try {
                    mat(120, 240, MatType.CV_8UC3, Scalar.all(0.0)).use { canvas ->
                        lsd.drawSegments(canvas, segments.lines)
                        // drawSegments paints red (0,0,255); countNonZero is
                        // single-channel only, so count through the red channel.
                        val channels = canvas.split()
                        try {
                            assertTrue(channels[2].nonZeroCount > 0, "drawSegments must paint the segments")
                        } finally {
                            channels.forEach { it.close() }
                        }
                    }
                    assertFalse(lsd.empty(), "a configured LineSegmentDetector reports empty()==false")
                    assertTrue(lsd.getDefaultName().isNotBlank(), "getDefaultName must not be blank")
                    lsd.clear() // must not throw
                } finally {
                    segments.width.close()
                    segments.prec.close()
                    segments.nfa.close()
                    segments.lines.close()
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // GeneralizedHoughBallard
    // ------------------------------------------------------------------

    @Test
    fun ghBallardSettersRoundTrip() {
        createGeneralizedHoughBallard().use { gh ->
            gh.cannyLowThresh = 50
            gh.cannyHighThresh = 150
            gh.minDist = 10.0
            gh.dp = 1.5
            gh.levels = 3
            gh.votesThreshold = 40

            assertEquals(50, gh.cannyLowThresh)
            assertEquals(150, gh.cannyHighThresh)
            assertEquals(10.0, gh.minDist)
            assertEquals(1.5, gh.dp)
            assertEquals(3, gh.levels)
            assertEquals(40, gh.votesThreshold)
            // NB: OpenCV's GeneralizedHoughBallard ignores maxBufferSize (the
            // setter is a no-op and the getter always reports 0); the parameter
            // only affects the Guil variant, so it is not round-tripped here.
        }
    }

    @Test
    fun ghBallardDetectOnSyntheticScene() {
        // 50x50 template: filled 30x30 square centered at (25, 25).
        mat(50, 50, MatType.CV_8UC1, Scalar.all(0.0)).use { templ ->
            templ.rectangle(Point(10, 10), Point(40, 40), Scalar.all(255.0), LineTypes.FILLED)
            createGeneralizedHoughBallard().use { gh ->
                gh.setTemplate(templ)
                gh.votesThreshold = 15
                // 150x150 scene with the same square at (50,50)-(80,80): center (65,65).
                mat(150, 150, MatType.CV_8UC1, Scalar.all(0.0)).use { scene ->
                    scene.rectangle(Point(50, 50), Point(80, 80), Scalar.all(255.0), LineTypes.FILLED)
                    val result = gh.detect(scene)
                    result.positions.use { positions ->
                        assertEquals(MatType.CV_32FC4, positions.type, "positions are Vec4f")
                        assertEquals(1, positions.rows, "positions are a 1xN row vector")
                        assertTrue(positions.total >= 1, "expected at least one detection")
                        // 1xN CV_32FC4: (x, y, angle, scale) per column.
                        var best = Double.MAX_VALUE
                        for (i in 0 until positions.total) {
                            val x = positions.at(0, i, 0)
                            val y = positions.at(0, i, 1)
                            val d = sqrt((x - 65.0) * (x - 65.0) + (y - 65.0) * (y - 65.0))
                            if (d < best) best = d
                        }
                        assertTrue(best < 10.0, "detected center too far from (65,65): $best")
                    }
                    result.votes.use { votes ->
                        assertTrue(
                            votes.isEmpty || (votes.rows == 1 && votes.type == MatType.CV_32SC3),
                            "votes must be empty or a 1xN CV_32SC3 row vector",
                        )
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // GeneralizedHoughGuil
    // ------------------------------------------------------------------

    @Test
    fun ghGuilSettersRoundTrip() {
        createGeneralizedHoughGuil().use { gh ->
            gh.cannyLowThresh = 40
            gh.cannyHighThresh = 160
            gh.minDist = 20.0
            gh.dp = 2.0
            gh.maxBufferSize = 4096
            gh.xi = 45.0
            gh.levels = 4
            gh.angleEpsilon = 1.0
            gh.minAngle = 0.0
            gh.maxAngle = 180.0
            gh.angleStep = 1.0
            gh.angleThresh = 100
            gh.minScale = 0.5
            gh.maxScale = 2.0
            gh.scaleStep = 0.05
            gh.scaleThresh = 50
            gh.posThresh = 50

            assertEquals(40, gh.cannyLowThresh)
            assertEquals(160, gh.cannyHighThresh)
            assertEquals(20.0, gh.minDist)
            assertEquals(2.0, gh.dp)
            assertEquals(4096, gh.maxBufferSize)
            assertEquals(45.0, gh.xi)
            assertEquals(4, gh.levels)
            assertEquals(1.0, gh.angleEpsilon)
            assertEquals(0.0, gh.minAngle)
            assertEquals(180.0, gh.maxAngle)
            assertEquals(1.0, gh.angleStep)
            assertEquals(100, gh.angleThresh)
            assertEquals(0.5, gh.minScale)
            assertEquals(2.0, gh.maxScale)
            assertEquals(0.05, gh.scaleStep)
            assertEquals(50, gh.scaleThresh)
            assertEquals(50, gh.posThresh)
        }
    }

    @Test
    fun ghGuilDetectNoCrashAndResultShape() {
        // Template: filled square; scene: the same square shifted. Position
        // accuracy is not asserted (Guil needs many votes), only that detect
        // runs and produces the documented result shapes.
        mat(40, 40, MatType.CV_8UC1, Scalar.all(0.0)).use { templ ->
            templ.rectangle(Point(10, 10), Point(30, 30), Scalar.all(255.0), LineTypes.FILLED)
            createGeneralizedHoughGuil().use { gh ->
                gh.setTemplate(templ)
                gh.angleThresh = 50
                gh.posThresh = 20
                gh.scaleThresh = 20
                mat(120, 120, MatType.CV_8UC1, Scalar.all(0.0)).use { scene ->
                    scene.rectangle(Point(40, 40), Point(60, 60), Scalar.all(255.0), LineTypes.FILLED)
                    val result = gh.detect(scene)
                    result.positions.use { positions ->
                        assertEquals(MatType.CV_32FC4, positions.type)
                        assertEquals(1, positions.rows)
                    }
                    result.votes.use { votes ->
                        assertTrue(
                            votes.isEmpty || (votes.rows == 1 && votes.type == MatType.CV_32SC3),
                            "votes must be empty or a 1xN CV_32SC3 row vector",
                        )
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Filter2DParams (pure Kotlin holder)
    // ------------------------------------------------------------------

    @Test
    fun filter2DParamsDefaultsMatchOpenCV() {
        val params = Filter2DParams()
        assertEquals(-1, params.anchorX)
        assertEquals(-1, params.anchorY)
        assertEquals(BorderTypes.REFLECT_101, params.borderType) // BORDER_DEFAULT
        assertEquals(-1, params.ddepth)
        assertEquals(1.0, params.scale)
        assertEquals(0.0, params.shift)

        params.anchorX = 2
        params.anchorY = 3
        params.ddepth = CV_32F
        assertEquals(2, params.anchorX)
        assertEquals(3, params.anchorY)
        assertEquals(CV_32F, params.ddepth)
    }
}

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

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Core value types + MatOf* wire views + TickMeter; runs on every target.
 */
class CoreTypesTest {

    // ---------------------------------------------------------------- value types

    @Test
    fun rangeEmptyAllAndContains() {
        assertTrue(Range.empty().empty)
        assertEquals(0, Range.empty().size())
        assertEquals(Range(0, 0), Range.empty())

        val all = Range.all()
        assertEquals(Int.MIN_VALUE, all.start)
        assertEquals(Int.MAX_VALUE, all.end)
        assertFalse(all.empty)
        assertTrue(all.contains(0))
        assertTrue(all.contains(Int.MIN_VALUE))
        assertFalse(all.contains(Int.MAX_VALUE), "end is exclusive")

        val r = Range(2, 5)
        assertEquals(3, r.size())
        assertTrue(r.contains(2))
        assertTrue(r.contains(4))
        assertFalse(r.contains(5))
        assertFalse(r.contains(1))
        assertTrue(Range(5, 2).empty, "end <= start is empty")
        assertTrue(Range(5, 5).empty)

        assertEquals(Range(5, 10), Range(0, 10).intersection(Range(5, 15)))
        val disjoint = Range(0, 2).intersection(Range(5, 7))
        assertTrue(disjoint.empty)
        assertEquals(Range(3, 5), Range(1, 3).shift(2))
    }

    @Test
    fun point3DotAndCross() {
        val a = Point3(1.0, 2.0, 3.0)
        val b = Point3(4.0, -5.0, 6.0)
        assertEquals(12.0, a.dot(b), absoluteTolerance = 1e-12)

        assertEquals(Point3(0.0, 0.0, 1.0), Point3(1.0, 0.0, 0.0).cross(Point3(0.0, 1.0, 0.0)))
        assertEquals(Point3(0.0, 0.0, -1.0), Point3(0.0, 1.0, 0.0).cross(Point3(1.0, 0.0, 0.0)))
        val c = a.cross(b)
        assertEquals(0.0, a.dot(c), absoluteTolerance = 1e-12, message = "cross must be orthogonal")
        assertEquals(0.0, b.dot(c), absoluteTolerance = 1e-12)
    }

    @Test
    fun dMatchAndRect2dBasics() {
        assertTrue(DMatch(0, 1, 0, 1.5f).lessThan(DMatch(0, 1, 0, 2.0f)))
        assertFalse(DMatch(0, 1, 0, 2.0f).lessThan(DMatch(0, 1, 0, 1.5f)))

        val r = Rect2d(1.0, 2.0, 3.0, 4.0)
        assertEquals(12.0, r.area, absoluteTolerance = 1e-12)
        assertFalse(r.empty)
        assertTrue(r.contains(Point(1, 2)))
        assertTrue(r.contains(Point(3, 5)))
        assertFalse(r.contains(Point(4, 2)), "x + width is exclusive")
        assertTrue(Rect2d(0.0, 0.0, 0.0, 0.0).empty)
        assertTrue(Rect2d(0.0, 0.0, -1.0, 2.0).empty)
    }

    // ---------------------------------------------------------------- MatOf* wire round trips

    @Test
    fun matOfByteRoundTrip() {
        val view = MatOfByte()
        try {
            assertTrue(view.empty)
            assertEquals(0, view.total())

            val bytes = byteArrayOf(1, -2, 3, 0, 127, -128)
            view.fromArray(bytes)
            assertEquals(cvMakeType(CV_8U, 1), view.mat.type)
            assertEquals(6, view.total())
            assertContentEquals(bytes, view.toArray())
            assertEquals(bytes[2], view[2])
            assertEquals(bytes.toList(), view.toList())

            view.fromList(listOf(9, 8, 7).map { it.toByte() })
            assertContentEquals(byteArrayOf(9, 8, 7), view.toArray())
            assertEquals(3, view.total())

            view.fromArray(1, 2, byteArrayOf(10, 20, 30))
            assertContentEquals(byteArrayOf(20, 30), view.toArray())
            assertEquals(2, view.total())

            val varargView = MatOfByte(5, 6, 7)
            assertContentEquals(byteArrayOf(5, 6, 7), varargView.toArray())
            varargView.mat.close()
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfDoubleRoundTrip() {
        val view = MatOfDouble()
        try {
            val values = doubleArrayOf(0.0, -1.25, 3.5, 1e300, -2.5e-9)
            view.fromArray(values)
            assertEquals(cvMakeType(CV_64F, 1), view.mat.type)
            assertEquals(5, view.total())
            assertContentEquals(values, view.toArray())
            assertEquals(values[3], view[3])
            assertEquals(values.toList(), view.toList())

            view.fromList(listOf(1.0, 2.0))
            assertContentEquals(doubleArrayOf(1.0, 2.0), view.toArray())
            assertEquals(2.0, view[1])
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfFloatRoundTrip() {
        val view = MatOfFloat()
        try {
            val values = floatArrayOf(0f, -1.25f, 3.5f, Float.MAX_VALUE, 1e-30f)
            view.fromArray(values)
            assertEquals(cvMakeType(CV_32F, 1), view.mat.type)
            assertContentEquals(values, view.toArray())
            assertEquals(values[2], view[2])
            assertEquals(values.toList(), view.toList())

            view.fromList(listOf(7f, 8f, 9f))
            assertContentEquals(floatArrayOf(7f, 8f, 9f), view.toArray())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfFloat4RoundTrip() {
        val view = MatOfFloat4()
        try {
            val values = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
            view.fromArray(values)
            assertEquals(cvMakeType(CV_32F, 4), view.mat.type)
            assertEquals(2, view.total())
            assertContentEquals(values, view.toArray())
            assertEquals(values.toList(), view.toList())
            assertContentEquals(floatArrayOf(5f, 6f, 7f, 8f), view[1])

            // trailing values past a whole element are dropped (SDK behavior)
            view.fromArray(floatArrayOf(1f, 2f, 3f, 4f, 5f))
            assertEquals(1, view.total())
            assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), view.toArray())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfFloat6RoundTrip() {
        val view = MatOfFloat6()
        try {
            val values = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
            view.fromArray(values)
            assertEquals(cvMakeType(CV_32F, 6), view.mat.type)
            assertEquals(2, view.total())
            assertContentEquals(values, view.toArray())
            assertEquals(values.toList(), view.toList())
            assertContentEquals(floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f), view[1])
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfIntRoundTrip() {
        val view = MatOfInt()
        try {
            val values = intArrayOf(0, -1, Int.MAX_VALUE, Int.MIN_VALUE, 42)
            view.fromArray(values)
            assertEquals(cvMakeType(CV_32S, 1), view.mat.type)
            assertContentEquals(values, view.toArray())
            assertEquals(values[3], view[3])
            assertEquals(values.toList(), view.toList())

            val varargView = MatOfInt(1, 2, 3)
            assertContentEquals(intArrayOf(1, 2, 3), varargView.toArray())
            varargView.mat.close()

            view.fromList(listOf(7, 8))
            assertContentEquals(intArrayOf(7, 8), view.toArray())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfInt4RoundTrip() {
        val view = MatOfInt4()
        try {
            val values = intArrayOf(1, 2, 3, 4, -1, -2, -3, -4)
            view.fromArray(values)
            assertEquals(cvMakeType(CV_32S, 4), view.mat.type)
            assertEquals(2, view.total())
            assertContentEquals(values, view.toArray())
            assertEquals(values.toList(), view.toList())
            assertContentEquals(intArrayOf(-1, -2, -3, -4), view[1])
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfPointRoundTrip() {
        val view = MatOfPoint()
        try {
            val points = listOf(Point(1, 2), Point(-3, 4), Point(0, 0))
            view.fromList(points)
            assertEquals(cvMakeType(CV_32S, 2), view.mat.type)
            assertEquals(3, view.total())
            assertEquals(points, view.toList())
            assertEquals(Point(-3, 4), view[1])
            assertContentEquals(intArrayOf(1, 2, -3, 4, 0, 0), view.toArray())

            view.fromArray(intArrayOf(9, 8, 7, 6))
            assertEquals(listOf(Point(9, 8), Point(7, 6)), view.toList())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfPoint2fRoundTrip() {
        val view = MatOfPoint2f()
        try {
            val points = listOf(Point(1, 2), Point(-3, 4), Point(0, 0))
            view.fromList(points)
            assertEquals(cvMakeType(CV_32F, 2), view.mat.type)
            assertEquals(3, view.total())
            assertEquals(points, view.toList())
            assertEquals(Point(-3, 4), view[1])
            assertContentEquals(floatArrayOf(1f, 2f, -3f, 4f, 0f, 0f), view.toArray())

            view.fromArray(floatArrayOf(9f, 8f, 7f, 6f))
            assertEquals(listOf(Point(9, 8), Point(7, 6)), view.toList())
            assertEquals(2, view.total())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfPoint3RoundTrip() {
        val view = MatOfPoint3()
        try {
            val points = listOf(Point3(1.0, 2.0, 3.0), Point3(-4.0, 5.0, -6.0))
            view.fromList(points)
            assertEquals(cvMakeType(CV_32S, 3), view.mat.type)
            assertEquals(2, view.total())
            assertEquals(points, view.toList())
            assertEquals(Point3(-4.0, 5.0, -6.0), view[1])
            assertContentEquals(intArrayOf(1, 2, 3, -4, 5, -6), view.toArray())

            view.fromArray(intArrayOf(9, 8, 7))
            assertEquals(listOf(Point3(9.0, 8.0, 7.0)), view.toList())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfPoint3fRoundTrip() {
        val view = MatOfPoint3f()
        try {
            // values exactly representable in float survive the double<->float hop
            val points = listOf(Point3(1.5, -2.25, 0.5), Point3(0.0, 0.0, 0.0))
            view.fromList(points)
            assertEquals(cvMakeType(CV_32F, 3), view.mat.type)
            assertEquals(2, view.total())
            assertEquals(points, view.toList())
            assertEquals(Point3(0.0, 0.0, 0.0), view[1])
            assertContentEquals(floatArrayOf(1.5f, -2.25f, 0.5f, 0f, 0f, 0f), view.toArray())

            view.fromArray(floatArrayOf(9.5f, 8.25f, 7.0f))
            assertEquals(listOf(Point3(9.5, 8.25, 7.0)), view.toList())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfRectRoundTrip() {
        val view = MatOfRect()
        try {
            val rects = listOf(Rect(1, 2, 3, 4), Rect(-1, -2, 0, 0))
            view.fromList(rects)
            assertEquals(cvMakeType(CV_32S, 4), view.mat.type)
            assertEquals(2, view.total())
            assertEquals(rects, view.toList())
            assertEquals(Rect(-1, -2, 0, 0), view[1])
            assertContentEquals(intArrayOf(1, 2, 3, 4, -1, -2, 0, 0), view.toArray())

            view.fromArray(intArrayOf(9, 8, 7, 6))
            assertEquals(listOf(Rect(9, 8, 7, 6)), view.toList())
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfRect2dRoundTrip() {
        val view = MatOfRect2d()
        try {
            val rects = listOf(Rect2d(1.5, -2.25, 3.0, 0.5), Rect2d(0.0, 0.0, 0.0, 0.0))
            view.fromList(rects)
            assertEquals(cvMakeType(CV_64F, 4), view.mat.type)
            assertEquals(2, view.total())
            assertEquals(rects, view.toList())
            assertEquals(Rect2d(0.0, 0.0, 0.0, 0.0), view[1])
            assertContentEquals(
                doubleArrayOf(1.5, -2.25, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0),
                view.toArray(),
            )
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfRotatedRectRoundTrip() {
        val view = MatOfRotatedRect()
        try {
            val rects = listOf(
                RotatedRect(centerX = 1.5, centerY = -2.0, width = 3.25, height = 0.5, angle = 90.0),
                RotatedRect(centerX = 0.0, centerY = 0.0, width = 0.0, height = 0.0, angle = 0.0),
            )
            view.fromList(rects)
            assertEquals(cvMakeType(CV_32F, 5), view.mat.type)
            assertEquals(2, view.total())
            assertEquals(rects, view.toList())
            assertEquals(rects[0], view[0])
            assertContentEquals(
                floatArrayOf(1.5f, -2f, 3.25f, 0.5f, 90f, 0f, 0f, 0f, 0f, 0f),
                view.toArray(),
            )
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfKeyPointRoundTrip() {
        val view = MatOfKeyPoint()
        try {
            val keypoints = listOf(
                KeyPoint(x = 1.5f, y = 2.25f, size = 3.5f, angle = -1f, response = 0.75f, octave = 2, classId = -1),
                KeyPoint(x = 0f, y = 0f, size = 0f, angle = 0f, response = 0f, octave = 0, classId = 0),
            )
            view.fromList(keypoints)
            assertEquals(cvMakeType(CV_32F, 7), view.mat.type)
            assertEquals(2, view.total())
            assertEquals(keypoints, view.toList())
            assertEquals(keypoints[0], view[0])
            assertContentEquals(
                floatArrayOf(1.5f, 2.25f, 3.5f, -1f, 0.75f, 2f, -1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
                view.toArray(),
            )
        } finally {
            view.mat.close()
        }
    }

    @Test
    fun matOfDMatchRoundTrip() {
        val view = MatOfDMatch()
        try {
            val matches = listOf(
                DMatch(queryIdx = 7, trainIdx = 42, imgIdx = -1, distance = 1.5f),
                DMatch(queryIdx = 0, trainIdx = 0, imgIdx = 0, distance = Float.MAX_VALUE),
            )
            view.fromList(matches)
            assertEquals(cvMakeType(CV_32F, 4), view.mat.type)
            assertEquals(2, view.total())
            assertEquals(matches, view.toList())
            assertEquals(matches[1], view[1])
            assertContentEquals(
                floatArrayOf(7f, 42f, -1f, 1.5f, 0f, 0f, 0f, Float.MAX_VALUE),
                view.toArray(),
            )
        } finally {
            view.mat.close()
        }
    }

    // ---------------------------------------------------------------- MatOf* validation & lifecycle

    @Test
    fun matOfRejectsIncompatibleBackingMat() {
        mat(1, 1, MatType.CV_8UC1).use { wrong ->
            assertFailsWith<IllegalArgumentException> { MatOfPoint2f(wrong) }
            assertFailsWith<IllegalArgumentException> { MatOfKeyPoint(wrong) }
            assertFailsWith<IllegalArgumentException> { MatOfDMatch(wrong) }
        }

        // an empty Mat of any type is accepted (SDK checks only non-empty Mats)
        val empty = MatOfPoint2f(mat(0, 0, MatType.CV_8UC1))
        assertTrue(empty.empty)
        empty.mat.close()

        // a compatible non-empty Mat is adopted as the backing store
        val right = mat(2, 1, cvMakeType(CV_32F, 2))
        val view = MatOfPoint2f(right)
        assertEquals(2, view.total())
        view.mat.close()
    }

    @Test
    fun matOfFromListReplacesBacking() {
        val view = MatOfPoint2f()
        try {
            view.fromArray(floatArrayOf(1f, 2f, 3f, 4f))
            assertEquals(2, view.total())

            // empty writes are no-ops and keep the current contents
            view.fromList(emptyList())
            assertEquals(2, view.total())
            assertEquals(listOf(Point(1, 2), Point(3, 4)), view.toList())

            view.fromList(listOf(Point(9, 9)))
            assertEquals(1, view.total())
            assertEquals(Point(9, 9), view[0])
        } finally {
            view.mat.close()
        }
    }

    // ---------------------------------------------------------------- TickMeter

    @Test
    fun tickMeterLifecycle() {
        tickMeter().use { tm ->
            assertEquals(0.0, tm.timeSec)
            assertEquals(0.0, tm.timeTicks)
            assertEquals(0.0, tm.timeSum)
            assertEquals(0, tm.counter)
            assertEquals(0.0, tm.avgTime)
            assertTrue(tm.freq > 0.0, "tick frequency must be positive")

            tm.start()
            burnTime()
            tm.stop()

            assertEquals(1, tm.counter)
            val first = tm.timeSec
            assertTrue(first >= 0.0, "measured time must be non-negative")
            assertEquals(tm.timeTicks, tm.timeSum, absoluteTolerance = 0.0)
            assertEquals(first, tm.avgTime, absoluteTolerance = 1e-12, message = "avg of one sample equals total")
            assertEquals(first, tm.timeSum / tm.freq, absoluteTolerance = 1e-9, message = "sec = sum / freq")

            tm.start()
            burnTime()
            tm.stop()

            assertEquals(2, tm.counter)
            assertTrue(tm.timeSec >= first, "timeSec must be monotonic across intervals")

            tm.reset()
            assertEquals(0, tm.counter)
            assertEquals(0.0, tm.timeSec)
            assertEquals(0.0, tm.timeSum)
            assertEquals(0.0, tm.avgTime)
        }
    }

    // ---------------------------------------------------------------- helpers

    /** A deterministic, platform-independent busy loop that cannot be elided. */
    private fun burnTime() {
        var acc = 0.0
        for (i in 1..2_000_000) {
            acc += sqrt(i.toDouble())
        }
        assertTrue(acc > 0.0, "burn loop must run")
    }
}

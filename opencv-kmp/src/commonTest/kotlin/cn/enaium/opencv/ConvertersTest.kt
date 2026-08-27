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
 * Round-trip coverage for the org.opencv.utils.Converters port: every typed
 * List <-> Mat conversion must survive a full encode/decode cycle on both
 * backends, and invalid Mats must be rejected like the SDK does.
 */
class ConvertersTest {

    /** Encodes [values], decodes the result and closes the Mat. */
    private fun <T> roundTrip(toMat: (List<T>) -> Mat, fromMat: (Mat) -> List<T>, values: List<T>): List<T> {
        val m = toMat(values)
        try {
            return fromMat(m)
        } finally {
            m.close()
        }
    }

    @Test
    fun pointRoundTripIsExact() {
        val pts = listOf(Point(1, 2), Point(-3, 4), Point(0, 0), Point(Int.MAX_VALUE, Int.MIN_VALUE))
        assertEquals(pts, roundTrip(::vectorPointToMat, ::matToVectorPoint, pts))
    }

    @Test
    fun point2fExactRoundTrip() {
        val pts = listOf(Point2f(1.5f, 2.5f), Point2f(-3.25f, 4.0f), Point2f(0f, -0.5f))
        assertEquals(pts, roundTrip(::vectorPoint2fToMat, ::matToVectorPoint2f, pts))
    }

    @Test
    fun point2fTruncatesThroughIntPoint() {
        // matToVectorPoint truncates CV_32FC2 to the Int-based Point, like
        // MatOfPoint2f's own toList().
        val pts = listOf(Point2f(1.5f, 2.5f), Point2f(-3.25f, 4.0f))
        vectorPoint2fToMat(pts).use { m ->
            assertEquals(listOf(Point(1, 2), Point(-3, 4)), matToVectorPoint(m))
        }
    }

    @Test
    fun point2dDecodesThroughPoint() {
        val pts = listOf(Point(1, 2), Point(-3, 4))
        assertEquals(pts, roundTrip(::vectorPoint2dToMat, ::matToVectorPoint, pts))
    }

    @Test
    fun pointDepthVariantsUseExpectedWireTypes() {
        vectorPointToMat(listOf(Point(1, 2))).use { assertEquals(cvMakeType(CV_32S, 2), it.type) }
        vectorPoint2fToMat(listOf(Point2f(1f, 2f))).use { assertEquals(cvMakeType(CV_32F, 2), it.type) }
        vectorPoint2dToMat(listOf(Point(1, 2))).use { assertEquals(cvMakeType(CV_64F, 2), it.type) }
    }

    @Test
    fun pointTypeDepthOverloadMatchesConvenience() {
        val pts = listOf(Point(1, 2), Point(-5, 6))
        vectorPointToMat(pts, CV_32S).use { explicit ->
            vectorPointToMat(pts).use { default ->
                assertEquals(explicit.type, default.type)
                assertEquals(matToVectorPoint(explicit), matToVectorPoint(default))
            }
        }
    }

    @Test
    fun pointTypeDepthRejectsUnknownDepth() {
        assertFailsWith<IllegalArgumentException> { vectorPointToMat(listOf(Point(1, 2)), CV_16U) }
        assertFailsWith<IllegalArgumentException> { vectorPoint3ToMat(listOf(Point3(1.0, 2.0, 3.0)), CV_16U) }
    }

    @Test
    fun matToVectorPointDecodesManuallyBuilt64FC2() {
        mat(2, 1, cvMakeType(CV_64F, 2)).use { m ->
            m.put(0, 0, doubleArrayOf(1.5, 2.5, -3.0, 4.25))
            assertEquals(listOf(Point(1, 2), Point(-3, 4)), matToVectorPoint(m))
        }
    }

    @Test
    fun matToVectorPointRejectsWideOrWrongType() {
        mat(1, 2, cvMakeType(CV_32S, 2)).use { wide ->
            assertFailsWith<IllegalArgumentException> { matToVectorPoint(wide) }
        }
        mat(2, 1, cvMakeType(CV_64F, 1)).use { wrong ->
            assertFailsWith<IllegalArgumentException> { matToVectorPoint(wrong) }
        }
        // An explicitly zeroed CV_32SC2 Mat decodes to zero points (a bare
        // mat() allocation is uninitialized storage, so zero it first).
        mat(2, 1, cvMakeType(CV_32S, 2), Scalar.all(0.0)).use { zeroed ->
            assertEquals(listOf(Point(0, 0), Point(0, 0)), matToVectorPoint(zeroed))
        }
    }

    @Test
    fun point3iTruncatesLikeSdkCast() {
        val pts = listOf(Point3(1.9, -2.9, 3.5), Point3(0.0, 0.0, 0.0))
        val decoded = roundTrip(::vectorPoint3iToMat, ::matToVectorPoint3, pts)
        assertEquals(listOf(Point3(1.0, -2.0, 3.0), Point3(0.0, 0.0, 0.0)), decoded)
    }

    @Test
    fun point3fAndPoint3dRoundTrip() {
        val pts = listOf(Point3(1.5, -2.25, 3.0), Point3(-0.5, 0.0, 100.125))
        assertEquals(pts, roundTrip(::vectorPoint3fToMat, ::matToVectorPoint3, pts))
        assertEquals(pts, roundTrip(::vectorPoint3dToMat, ::matToVectorPoint3, pts))
    }

    @Test
    fun floatRoundTrip() {
        val fs = listOf(1.5f, -2.25f, 0f, 100.5f, -0.5f)
        assertEquals(fs, roundTrip(::vectorFloatToMat, ::matToVectorFloat, fs))
    }

    @Test
    fun doubleRoundTrip() {
        val ds = listOf(1.5, -2.25, 0.0, 123456.789, -1e300)
        assertEquals(ds, roundTrip(::vectorDoubleToMat, ::matToVectorDouble, ds))
    }

    @Test
    fun intRoundTrip() {
        val ints = listOf(0, 1, -1, Int.MAX_VALUE, Int.MIN_VALUE)
        assertEquals(ints, roundTrip(::vectorIntToMat, ::matToVectorInt, ints))
    }

    @Test
    fun ucharRoundTrip() {
        val bytes = byteArrayOf(0, 1, -1, 127, -128).toList()
        assertEquals(bytes, roundTrip(::vectorUcharToMat, ::matToVectorUchar, bytes))
    }

    @Test
    fun charRoundTrip() {
        val bytes = byteArrayOf(-128, -1, 0, 1, 127).toList()
        assertEquals(bytes, roundTrip(::vectorCharToMat, ::matToVectorChar, bytes))
    }

    @Test
    fun rectRoundTrip() {
        val rs = listOf(Rect(1, 2, 3, 4), Rect(-5, -6, 7, 8))
        assertEquals(rs, roundTrip(::vectorRectToMat, ::matToVectorRect, rs))
    }

    @Test
    fun rect2dRoundTrip() {
        val rs = listOf(Rect2d(1.5, 2.5, 3.0, 4.5), Rect2d(-1.0, 0.0, 0.5, 100.25))
        assertEquals(rs, roundTrip(::vectorRect2dToMat, ::matToVectorRect2d, rs))
    }

    @Test
    fun rotatedRectRoundTrip() {
        val rs = listOf(RotatedRect(1.5, 2.5, 3.0, 4.0, 30.0), RotatedRect(-1.0, 0.0, 0.5, 2.0, -45.0))
        assertEquals(rs, roundTrip(::vectorRotatedRectToMat, ::matToVectorRotatedRect, rs))
    }

    @Test
    fun keyPointRoundTrip() {
        val kps = listOf(
            KeyPoint(1.5f, 2.5f, 3.0f, 30.0f, 0.25f, 1, 2),
            KeyPoint(-0.5f, 0.0f, 10.0f, -90.0f, 1.0f, 0, -1),
        )
        assertEquals(kps, roundTrip(::vectorKeyPointToMat, ::matToVectorKeyPoint, kps))
    }

    @Test
    fun dMatchRoundTrip() {
        val ds = listOf(DMatch(1, 2, 3, 0.5f), DMatch(-1, 0, 2, 1.25f))
        assertEquals(ds, roundTrip(::vectorDMatchToMat, ::matToVectorDMatch, ds))
    }

    @Test
    fun featureWireTypes() {
        val kp = KeyPoint(1f, 1f, 1f, 0f, 0f, 0, 0)
        vectorKeyPointToMat(listOf(kp)).use { assertEquals(cvMakeType(CV_32F, 7), it.type) }
        vectorDMatchToMat(listOf(DMatch(0, 0, 0, 0f))).use { assertEquals(cvMakeType(CV_32F, 4), it.type) }
    }

    @Test
    fun emptyListRoundTrips() {
        // every family: empty list -> empty Mat -> empty list
        fun <T> check(toMat: (List<T>) -> Mat, fromMat: (Mat) -> List<T>) {
            val m = toMat(emptyList())
            try {
                assertTrue(m.isEmpty, "expected an empty Mat")
                assertEquals(emptyList<T>(), fromMat(m))
            } finally {
                m.close()
            }
        }
        check(::vectorPointToMat, ::matToVectorPoint)
        check(::vectorPoint2fToMat, ::matToVectorPoint2f)
        check(::vectorPoint2dToMat, ::matToVectorPoint)
        check(::vectorPoint3iToMat, ::matToVectorPoint3)
        check(::vectorPoint3fToMat, ::matToVectorPoint3)
        check(::vectorPoint3dToMat, ::matToVectorPoint3)
        check(::vectorFloatToMat, ::matToVectorFloat)
        check(::vectorUcharToMat, ::matToVectorUchar)
        check(::vectorCharToMat, ::matToVectorChar)
        check(::vectorIntToMat, ::matToVectorInt)
        check(::vectorDoubleToMat, ::matToVectorDouble)
        check(::vectorRectToMat, ::matToVectorRect)
        check(::vectorRect2dToMat, ::matToVectorRect2d)
        check(::vectorRotatedRectToMat, ::matToVectorRotatedRect)
        check(::vectorKeyPointToMat, ::matToVectorKeyPoint)
        check(::vectorDMatchToMat, ::matToVectorDMatch)
    }

    @Test
    fun decodeRejectsInvalidMats() {
        fun <T> check(fromMat: (Mat) -> List<T>) {
            mat(2, 1, cvMakeType(CV_8U, 1)).use { wrongType ->
                assertFailsWith<IllegalArgumentException> { fromMat(wrongType) }
            }
            mat(1, 2, cvMakeType(CV_32F, 1)).use { wide ->
                assertFailsWith<IllegalArgumentException> { fromMat(wide) }
            }
        }
        check(::matToVectorPoint)
        check(::matToVectorPoint2f)
        check(::matToVectorPoint3)
        check(::matToVectorFloat)
        // matToVectorUchar intentionally accepts any CV_8UC1 mat (it IS the
        // uchar wire type), so it is not in the rejection list.
        check(::matToVectorChar)
        check(::matToVectorInt)
        check(::matToVectorDouble)
        check(::matToVectorRect)
        check(::matToVectorRect2d)
        check(::matToVectorRotatedRect)
        check(::matToVectorKeyPoint)
        check(::matToVectorDMatch)
    }

    @Test
    fun decodeHandlesManuallyBuiltMats() {
        // CV_32SC2 built outside the converters
        mat(2, 1, cvMakeType(CV_32S, 2)).use { m ->
            m.put(0, 0, doubleArrayOf(7.0, 8.0, -1.0, -2.0))
            assertEquals(listOf(Point(7, 8), Point(-1, -2)), matToVectorPoint(m))
        }
        // CV_64FC3
        mat(1, 1, cvMakeType(CV_64F, 3)).use { m ->
            m.put(0, 0, doubleArrayOf(1.5, 2.5, 3.5))
            assertEquals(listOf(Point3(1.5, 2.5, 3.5)), matToVectorPoint3(m))
        }
        // CV_8SC1
        mat(3, 1, cvMakeType(CV_8S, 1)).use { m ->
            m.pixels = byteArrayOf(0x10, 0x80.toByte(), 0xFF.toByte())
            assertEquals(listOf(0x10.toByte(), 0x80.toByte(), 0xFF.toByte()), matToVectorChar(m))
        }
    }
}

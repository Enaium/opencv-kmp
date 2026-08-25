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

/**
 * Coverage for the Mat <-> Scalar arithmetic operators and the pure-Kotlin
 * [Scalar] arithmetic sugar. Runs wherever a native library is available.
 */
class ScalarArithmeticTest {

    @Test
    fun matScalarOperatorsArePerChannel() {
        // CV_32FC3 so each channel has independent values, proving the scalar
        // arithmetic is per-channel rather than a flat broadcast.
        mat(1, 1, MatType.CV_32FC3).use { m ->
            m[0, 0] = 1.0
            m.put(0, 0, channel = 1, value = 2.0)
            m.put(0, 0, channel = 2, value = 4.0)

            val s = Scalar(10.0, 20.0, 40.0, 80.0)
            (m + s).use { r ->
                assertEquals(11.0, r.at(0, 0, 0))
                assertEquals(22.0, r.at(0, 0, 1))
                assertEquals(44.0, r.at(0, 0, 2))
            }
            (m - s).use { r ->
                assertEquals(-9.0, r.at(0, 0, 0))
                assertEquals(-18.0, r.at(0, 0, 1))
                assertEquals(-36.0, r.at(0, 0, 2))
            }
            (m * s).use { r ->
                assertEquals(10.0, r.at(0, 0, 0))
                assertEquals(40.0, r.at(0, 0, 1))
                assertEquals(160.0, r.at(0, 0, 2))
            }
            // CV_32F division is float-inexact; allow ~1e-4.
            (m / s).use { r ->
                assertEquals(0.1, r.at(0, 0, 0), 1e-4)
                assertEquals(0.1, r.at(0, 0, 1), 1e-4)
                assertEquals(0.1, r.at(0, 0, 2), 1e-4)
            }
        }
    }

    @Test
    fun matScalarBroadcastOperator() {
        mat(2, 2, MatType.CV_32FC1, Scalar.all(10.0)).use { m ->
            (m + 5.0).use { assertEquals(15.0, it[0, 0]) }
            (m - 2.5).use { assertEquals(7.5, it[0, 0], 1e-4) }
            (m * 2.0).use { assertEquals(20.0, it[0, 0]) }
            (m / 4.0).use { assertEquals(2.5, it[0, 0], 1e-4) }
        }
    }

    @Test
    fun scalarArithmeticOperators() {
        val a = Scalar(1.0, 2.0, 4.0, 8.0)
        val b = Scalar(10.0, 5.0, 2.0, 1.0)

        assertEquals(Scalar(11.0, 7.0, 6.0, 9.0), a + b)
        assertEquals(Scalar(-9.0, -3.0, 2.0, 7.0), a - b)
        assertEquals(Scalar(10.0, 10.0, 8.0, 8.0), a * b)
        assertEquals(Scalar(0.1, 0.4, 2.0, 8.0), a / b)

        assertEquals(Scalar(2.0, 4.0, 8.0, 16.0), a * 2.0)
        assertEquals(Scalar(0.5, 1.0, 2.0, 4.0), a / 2.0)
        assertEquals(Scalar(6.0, 7.0, 9.0, 13.0), a + 5.0)
        assertEquals(Scalar(-1.0, 0.0, 2.0, 6.0), a - 2.0)

        assertEquals(Scalar(-1.0, -2.0, -4.0, -8.0), -a)
        assertEquals(1.0, Scalar(1.0, 0.0, 0.0, 0.0) dist Scalar(), 1e-4)
    }

    @Test
    fun scalarCompanionHelpers() {
        assertEquals(Scalar(3.0, 3.0, 3.0, 3.0), Scalar.all(3.0))
        val bgr = Scalar.bgr(blue = 1.0, green = 2.0, red = 3.0)
        assertEquals(1.0, bgr[0])
        assertEquals(2.0, bgr[1])
        assertEquals(3.0, bgr[2])
    }

    @Test
    fun pureKotlinMatSugar() {
        mat(2, 2, MatType.CV_32FC1, Scalar.all(3.0)).use { m ->
            m.squared().use { assertEquals(9.0, it[0, 0]) }
        }
        // a rsub b == b - a
        mat(2, 2, MatType.CV_32FC1, Scalar.all(3.0)).use { small ->
            mat(2, 2, MatType.CV_32FC1, Scalar.all(10.0)).use { big ->
                (big rsub small).use { assertEquals(-7.0, it[0, 0]) }
                (small rsub big).use { assertEquals(7.0, it[0, 0]) }
            }
        }
        mat(2, 2, MatType.CV_32FC1, Scalar.all(-5.0)).use { neg ->
            neg.abs().use { assertEquals(5.0, it[0, 0]) }
        }
    }
}
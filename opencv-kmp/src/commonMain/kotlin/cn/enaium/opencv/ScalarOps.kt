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

// =========================================================================
// Scalar arithmetic (component-wise, mirrors cv::Scalar / numpy broadcast)
// =========================================================================

/** Component-wise addition. */
operator fun Scalar.plus(other: Scalar): Scalar =
    Scalar(v0 + other.v0, v1 + other.v1, v2 + other.v2, v3 + other.v3)

/** Component-wise subtraction. */
operator fun Scalar.minus(other: Scalar): Scalar =
    Scalar(v0 - other.v0, v1 - other.v1, v2 - other.v2, v3 - other.v3)

/** Component-wise multiplication. */
operator fun Scalar.times(other: Scalar): Scalar =
    Scalar(v0 * other.v0, v1 * other.v1, v2 * other.v2, v3 * other.v3)

/** Component-wise division. */
operator fun Scalar.div(other: Scalar): Scalar =
    Scalar(v0 / other.v0, v1 / other.v1, v2 / other.v2, v3 / other.v3)

/** Scales every component by [scale]. */
operator fun Scalar.times(scale: Double): Scalar =
    Scalar(v0 * scale, v1 * scale, v2 * scale, v3 * scale)

/** Divides every component by [scale]. */
operator fun Scalar.div(scale: Double): Scalar =
    Scalar(v0 / scale, v1 / scale, v2 / scale, v3 / scale)

/** Broadcasts [value] into every component and adds. */
operator fun Scalar.plus(value: Double): Scalar =
    Scalar(v0 + value, v1 + value, v2 + value, v3 + value)

/** Broadcasts [value] into every component and subtracts. */
operator fun Scalar.minus(value: Double): Scalar =
    Scalar(v0 - value, v1 - value, v2 - value, v3 - value)

/** Negates every component. */
operator fun Scalar.unaryMinus(): Scalar =
    Scalar(-v0, -v1, -v2, -v3)

/** L1 distance (sum of absolute component differences). */
infix fun Scalar.dist(other: Scalar): Double =
    kotlin.math.abs(v0 - other.v0) + kotlin.math.abs(v1 - other.v1) +
        kotlin.math.abs(v2 - other.v2) + kotlin.math.abs(v3 - other.v3)
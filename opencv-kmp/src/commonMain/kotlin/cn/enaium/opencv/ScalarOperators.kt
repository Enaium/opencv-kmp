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

/** Adds a scalar to every element per channel (`cv::add` with a scalar). */
operator fun Mat.plus(s: Scalar): Mat = addScalar(s)

/** Subtracts a scalar from every element per channel (`cv::subtract` with a scalar). */
operator fun Mat.minus(s: Scalar): Mat = subtractScalar(s)

/** Multiplies every element by the scalar per channel (`cv::multiply` with a scalar). */
operator fun Mat.times(s: Scalar): Mat = multiplyScalar(s)

/** Divides every element by the scalar per channel (`cv::divide` with a scalar). */
operator fun Mat.div(s: Scalar): Mat = divideScalar(s)

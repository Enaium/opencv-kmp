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
package cn.enaium.opencv.example

import cn.enaium.opencv.MatType
import cn.enaium.opencv.Scalar
import cn.enaium.opencv.eye
import cn.enaium.opencv.mat
import cn.enaium.opencv.ones
import cn.enaium.opencv.opencvVersion
import cn.enaium.opencv.shape

/** Headless demo exercising core Mat operations. */
fun runBasicDemo(): String = buildString {
    appendLine("OpenCV version: $opencvVersion")

    mat(rows = 2, cols = 2, type = MatType.CV_32FC1).use { empty ->
        appendLine("empty 2x2   -> isEmpty=${empty.isEmpty}, shape=${empty.shape}")
    }

    mat(rows = 2, cols = 2, type = MatType.CV_32FC1, fill = Scalar.all(1.5)).use { a ->
        eye(2, 2, MatType.CV_32FC1).use { identity ->
            (a + identity).use { sum ->
                appendLine("a + I       -> [${sum[0, 0]}, ${sum[0, 1]}] (expect [2.5, 1.5])")
                appendLine("sum.mean    -> ${sum.mean.v0} (expect 2.0)")
            }
            (a * 3.0).use { scaled ->
                appendLine("a * 3.0     -> [${scaled[0, 0]}] (expect [4.5])")
            }
            ones(2, 2, MatType.CV_32FC1).use { unit ->
                (a / unit).use { quotient ->
                    appendLine("a / 1.0     -> [${quotient[0, 0]}] (expect [1.5])")
                }
                (identity * unit).use { product ->
                    appendLine("I * 1       -> [${product[0, 0]}, ${product[1, 0]}] (expect [1.0, 0.0])")
                }
            }
        }
    }
}

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
import cn.enaium.opencv.ColorConversionCodes
import cn.enaium.opencv.LineTypes
import cn.enaium.opencv.MatType
import cn.enaium.opencv.Point
import cn.enaium.opencv.Scalar
import cn.enaium.opencv.ThresholdTypes
import cn.enaium.opencv.cvTypeName
import cn.enaium.opencv.imdecode
import cn.enaium.opencv.imencode
import cn.enaium.opencv.imwrite
import cn.enaium.opencv.mat
import cn.enaium.opencv.opencvLastError
import cn.enaium.opencv.shape

/**
 * Synthesizes a gradient scene, runs a small imgproc pipeline and an encode/
 * decode round trip; returns a report string.
 */
fun runImageProcessingDemo(outDir: String): String = buildString {
    val width = 128
    val height = 96

    mat(height, width, MatType.CV_8UC3).use { canvas ->
        // Diagonal blue->white gradient through the green channel plus shapes.
        for (row in 0 until height) {
            for (col in 0 until width) {
                canvas.put(row, col, 0, 255.0 * col / (width - 1))
                canvas.put(row, col, 1, 255.0 * row / (height - 1))
            }
        }
        canvas.rectangle(
            from = Point(8, 8),
            to = Point(width / 2, height / 2),
            color = Scalar.bgr(blue = 255.0, green = 0.0, red = 0.0),
            thickness = LineTypes.LINE_8,
        )
        canvas.circle(
            center = Point(width - 32, height - 24),
            radius = 16,
            color = Scalar.bgr(0.0, 0.0, 255.0),
            thickness = LineTypes.FILLED,
        )
        appendLine("canvas      -> ${canvas.shape}, blueMean=${canvas.mean.v0}")

        canvas.gaussianBlur(kernelWidth = 7, kernelHeight = 7).use { blurred ->
            blurred.cvtColor(ColorConversionCodes.BGR2GRAY).use { gray ->
                gray.threshold(128.0, 255.0, ThresholdTypes.BINARY).use { binary ->
                    val edges = binary.canny(threshold1 = 50.0, threshold2 = 150.0)
                    edges.use {
                        appendLine("binary mask -> nonZero=${binary.nonZeroCount}")
                        appendLine("edges       -> ${edges.shape}, nonZero=${edges.nonZeroCount}")

                        val png = imencode("png", edges)
                        appendLine("encoded png -> ${png.size} bytes, magic=${png.take(4).map { it.toUByte() }}")

                        imdecode(png)?.use { restored ->
                            appendLine("decoded     -> ${restored.shape} (${restored.type.cvTypeName})")
                            check(restored.rows == height && restored.cols == width) {
                                "decode changed dimensions"
                            }
                        }
                    }
                }
            }

            canvas.resize(width / 2, height / 2).use { small ->
                appendLine("resized     -> ${small.shape}")
            }

            val path = "$outDir/demo_canvas.png"
            if (imwrite(path, canvas)) {
                appendLine("wrote       -> $path")
            } else {
                appendLine("imwrite failed: ${opencvLastError}")
            }
        }
    }
}

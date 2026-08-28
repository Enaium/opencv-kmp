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
package cn.enaium.opencv.tutorial

import cn.enaium.opencv.FILLED
import cn.enaium.opencv.Mat
import cn.enaium.opencv.MatType
import cn.enaium.opencv.Point
import cn.enaium.opencv.Scalar
import cn.enaium.opencv.photoFastNlMeansDenoising
import cn.enaium.opencv.photoInpaint
import cn.enaium.opencv.zeros

/**
 * Ports of `tutorial_code/photo`:
 *  - Non-local means denoising on a synthetic noisy image.
 *  - Inpainting a synthetic scratch over a banded pattern.
 */
fun runPhotoTutorials(): String = buildString {
    appendLine("-- denoising --")
    denoisingDemo().also { append(it) }

    appendLine("-- inpainting --")
    inpaintingDemo().also { append(it) }
}

private fun bandedPattern(): Mat {
    val m = zeros(72, 72, MatType.CV_8UC1)
    for (r in 0 until m.rows) {
        val v = (r * 255 / m.rows).toDouble()
        for (c in 0 until m.cols) m[r, c] = v
    }
    return m
}

private fun denoisingDemo(): String = buildString {
    bandedPattern().use { clean ->
        val noisy = clean.clone().addSpeckle(seed = 7, sigma = 30.0)
        noisy.use { n ->
            photoFastNlMeansDenoising(n, h = 12f).use { denoised ->
                line("noise delta   (source)", "${avgAbsDelta(clean, n)}")
                line("denoised delta", "${avgAbsDelta(clean, denoised)}")
            }
        }
        noisy.close()
    }
}

private fun inpaintingDemo(): String = buildString {
    bandedPattern().use { clean ->
        val damaged = clean.clone()
        damaged.rectangle(Point(12, 20), Point(60, 26), Scalar.all(0.0), FILLED)
        damaged.rectangle(Point(30, 8), Point(36, 64), Scalar.all(0.0), FILLED)

        val mask = zeros(clean.rows, clean.cols, MatType.CV_8UC1)
        mask.rectangle(Point(12, 20), Point(60, 26), Scalar.all(255.0), FILLED)
        mask.rectangle(Point(30, 8), Point(36, 64), Scalar.all(255.0), FILLED)

        mask.use { m ->
            damaged.use { d ->
                photoInpaint(d, m, 3.0).use { restored ->
                    line("damaged delta", "${avgAbsDelta(clean, d)}")
                    line("restored delta", "${avgAbsDelta(clean, restored)}")
                }
            }
        }
        damaged.close()
        mask.close()
    }
}

private fun avgAbsDelta(a: Mat, b: Mat): Double =
    a.absDiff(b).use { it.mean.v0 }
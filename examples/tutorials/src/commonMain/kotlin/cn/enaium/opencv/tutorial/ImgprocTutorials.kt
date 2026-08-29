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

import cn.enaium.opencv.Mat
import cn.enaium.opencv.Point
import cn.enaium.opencv.Rect
import cn.enaium.opencv.Scalar
import cn.enaium.opencv.TemplateMatchModes
import cn.enaium.opencv.zeros

/**
 * Ports of `tutorial_code/ImgProc`:
 *  - Template Matching (templateMatching.cpp)
 *  - Smoothing (Smoothing.cpp)
 *  - Pyramids (Pyramids.cpp)
 * Every demo uses a synthetic scene or the bundled [assetsDir] images.
 */
fun runImgprocTutorials(assetsDir: String): String = buildString {
    appendLine("-- template matching --")
    templateMatchingDemo(assetsDir).also { append(it) }

    appendLine("-- smoothing --")
    smoothingDemo().also { append(it) }

    appendLine("-- pyramids --")
    pyramidsDemo().also { append(it) }
}

private fun templateMatchingDemo(assetsDir: String): String = buildString {
    // The template is cut from the same synthetic scene (the 24x16 bright
    // block at 12,12), so the normalized match must peak at ~1.0.
    val scene = syntheticScene()
    val templ = scene.roi(Rect(12, 12, 24, 16)).clone()
    scene.use { src ->
        templ.use { t ->
            src.matchTemplate(t, TemplateMatchModes.CCOEFF_NORMED).use { res ->
                val mm = res.minMaxLoc()
                expectClose(1.0, mm.maxVal, "template match score")
                line("best score", "${mm.maxVal}")
                line("best location", "${mm.maxPoint}")
                line("map size", "${res.rows}x${res.cols}")
            }
        }
    }
}

private fun smoothingDemo(): String = buildString {
    syntheticScene().use { src ->
        src.gaussianBlur(5, 5, 1.5).use { g -> line("gaussian delta", "${avgDelta(src, g)}") }
        src.medianBlur(5).use { m -> line("median   delta", "${avgDelta(src, m)}") }
        src.bilateralFilter(9, 90.0, 90.0).use { b -> line("bilateral delta", "${avgDelta(src, b)}") }
    }
}

private fun pyramidsDemo(): String = buildString {
    syntheticScene().use { src ->
        src.resize(32, 24).use { down ->
            line("reduced size", "${down.rows}x${down.cols} (was ${src.rows}x${src.cols})")
            down.resize(64, 48).use { up ->
                line("restored size", "${up.rows}x${up.cols}")
                line("roundtrip delta", "${avgDelta(src, up)}")
            }
        }
    }
}

/** Mean absolute pixel delta between two same-sized Mats. */
private fun avgDelta(a: Mat, b: Mat): Double =
    a.absDiff(b).use { it.mean.v0 }

/** Black canvas with a bright rectangle plus a vertical line of pixels. */
internal fun syntheticScene(rows: Int = 48, cols: Int = 64): Mat {
    val canvas = cn.enaium.opencv.zeros(rows, cols, cn.enaium.opencv.MatType.CV_8UC1)
    val w = cols / 8
    val h = rows / 8
    canvas.rectangle(Point(w, h), Point(cols - w, rows - h), Scalar.all(255.0), cn.enaium.opencv.FILLED)
    val step = (rows / 7 + 1).coerceAtLeast(1)
    for (y in 0 until rows step step) canvas.line(Point(2, y), Point(2, y), Scalar.all(255.0), 1)
    return canvas
}

private fun expectClose(expected: Double, actual: Double, what: String) {
    if (expected - actual > 1e-6) {
        throw AssertionError("$what: expected ~$expected, got $actual")
    }
}


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

import kotlin.math.PI
import kotlin.math.abs
import cn.enaium.opencv.Mat
import cn.enaium.opencv.MatType
import cn.enaium.opencv.MorphShapes
import cn.enaium.opencv.MorphTypes
import cn.enaium.opencv.Point
import cn.enaium.opencv.Scalar
import cn.enaium.opencv.ThresholdTypes
import cn.enaium.opencv.getRotationMatrix2D
import cn.enaium.opencv.getStructuringElement
import cn.enaium.opencv.mat
import cn.enaium.opencv.shape
import cn.enaium.opencv.toGray
import cn.enaium.opencv.zeros

/**
 * Headless ports of the classic OpenCV "Image Processing" tutorials. Every
 * demo builds its own synthetic input and returns a short report so they run
 * identically on desktop JVM, native targets, and Android.
 */

/** Morphology_1/2: erosion, dilation, opening and closing on a synthetic scene. */
fun morphologyDemo(): String = buildString {
    syntheticScene().use { src ->
        getStructuringElement(MorphShapes.RECT, 3, 3).use { kernel ->
            src.erode(kernel).use { eroded ->
                appendLine("erode   -> nonZero ${eroded.nonZeroCount} (was ${src.nonZeroCount})")
            }
            src.dilate(kernel).use { dilated ->
                appendLine("dilate  -> nonZero ${dilated.nonZeroCount}")
            }
            src.morphologyEx(MorphTypes.OPEN, kernel).use { opened ->
                appendLine("open    -> nonZero ${opened.nonZeroCount} (speckles removed)")
            }
            src.morphologyEx(MorphTypes.CLOSE, kernel).use { closed ->
                appendLine("close   -> nonZero ${closed.nonZeroCount} (pinholes filled)")
            }
        }
    }
}

/** Threshold tutorial: binary, inverse-binary and tozero over a gradient. */
fun thresholdDemo(): String = buildString {
    gradientRamp().use { gray ->
        gray.threshold(128.0, 255.0, ThresholdTypes.BINARY).use { binary ->
            var white = 0
            for (c in 0 until binary.cols) if (binary[0, c] > 0) white++
            appendLine("binary  -> white pixels: $white (expect ~128)")
        }
        gray.threshold(128.0, 255.0, ThresholdTypes.BINARY_INV).use { inv ->
            var white = 0
            for (c in 0 until inv.cols) if (inv[0, c] > 0) white++
            appendLine("inverse -> white pixels: $white")
        }
        gray.threshold(64.0, 255.0, ThresholdTypes.TOZERO).use { toZero ->
            appendLine("tozero  -> [0,0]=${toZero[0, 0]} stays, [200]=${toZero[0, 200]} kept as-is above threshold")
        }
    }
}

/** Color spaces: BGR -> GRAY plus channel split/merge round trip. */
fun colorSpacesDemo(): String = buildString {
    syntheticScene().use { bgr ->
        bgr.toGray().use { gray ->
            appendLine("gray    -> shape=${gray.shape}, channels=${gray.channels}")
        }
        val planes = bgr.split()
        try {
            cn.enaium.opencv.merge(planes).use { merged ->
                var diff = 0.0
                for (r in 0 until bgr.rows) for (c in 0 until bgr.cols) for (ch in 0 until bgr.channels) {
                    diff += abs(bgr.at(r, c, ch) - merged.at(r, c, ch))
                }
                appendLine("split+merge -> diff=$diff (expect 0)")
            }
        } finally {
            planes.forEach(cn.enaium.opencv.Mat::close)
        }
    }
}

/** Hough transform: Canny edges fed to standard and probabilistic Hough. */
fun houghLinesDemo(): String = buildString {
    syntheticScene().use { scene ->
        scene.toGray().use { gray ->
            gray.canny(threshold1 = 50.0, threshold2 = 150.0).use { edges ->
                edges.houghLinesP(rho = 1.0, theta = PI / 180.0, threshold = 40).use { segments ->
                    appendLine("HoughLinesP -> ${segments.rows} segment rows (${segments.shape})")
                }
                edges.houghLines(rho = 1.0, theta = PI / 180.0, threshold = 50).use { lines ->
                    appendLine("houghLines  -> ${lines.rows} line rows")
                }
            }
        }
    }
}

/** Geometric transforms: rotate about the center, then a pyramid round trip. */
fun geometricTransformsDemo(): String = buildString {
    syntheticScene().use { scene ->
        getRotationMatrix2D(
            center = Point(scene.cols / 2, scene.rows / 2),
            angle = 45.0,
            scale = 1.0,
        ).use { rotation ->
            scene.warpAffine(rotation, width = scene.cols, height = scene.rows).use { rotated ->
                appendLine("rotated 45° -> shape=${rotated.shape}")
            }
        }
        scene.pyrDown().use { down ->
            down.pyrUp().use { up ->
                appendLine("pyramid     -> ${scene.shape} -> ${down.shape} -> ${up.shape}")
            }
        }
    }
}

/** Histogram equalization widens a low-contrast grayscale ramp. */
fun histogramEqualizationDemo(): String = buildString {
    lowContrastRamp().use { flat ->
        flat.equalizeHist().use { equalized ->
            var min = 255.0
            var max = 0.0
            for (c in 0 until equalized.cols) {
                min = minOf(min, equalized[0, c])
                max = maxOf(max, equalized[0, c])
            }
            appendLine("contrast    -> was [16, 143], equalized [$min, $max]")
        }
    }
}

/** Drawing primitives summary: rect + circle + line on a canvas. */
fun drawingDemo(): String = buildString {
    zeros(240, 320, MatType.CV_8UC3).use { canvas ->
        canvas.rectangle(from = Point(20, 20), to = Point(140, 100), color = Scalar.bgr(255.0, 0.0, 0.0))
        canvas.circle(center = Point(220, 70), radius = 45, color = Scalar.bgr(0.0, 255.0, 0.0))
        canvas.line(from = Point(20, 200), to = Point(300, 200), color = Scalar.bgr(0.0, 0.0, 255.0))
        appendLine("drawing     -> blue rect, green circle, red line")
        appendLine("mean color  -> B=${canvas.mean.v0} G=${canvas.mean.v1} R=${canvas.mean.v2}")
    }
}

/** Scalar arithmetic: brightness shift, per-channel gain and dimming. */
fun scalarArithmeticDemo(): String = buildString {
    mat(2, 2, MatType.CV_32FC3, fill = Scalar(10.0, 60.0, 110.0)).use { image ->
        // brightness shift: dst = 1*src + 15
        image.addWeighted(alpha = 1.0, other = image, beta = 0.0, gamma = 15.0).use { shifted ->
            appendLine("shift +15   -> [${shifted.at(0, 0, 0)}, ${shifted.at(0, 0, 1)}, ${shifted.at(0, 0, 2)}]")
        }
        // global gain and dimming via addWeighted scaling
        image.addWeighted(alpha = 2.0, other = image, beta = 0.0, gamma = 0.0).use { gained ->
            appendLine("*2.0 gain   -> [${gained.at(0, 0, 0)}] (expect 20)")
        }
        image.addWeighted(alpha = 0.25, other = image, beta = 0.0, gamma = 0.0).use { dimmed ->
            appendLine("/4.0 dim    -> [${dimmed.at(0, 0, 0)}] (expect 2.5)")
        }
    }
}

/** Runs every tutorial demo and returns the combined report. */
fun runTutorialDemos(): String = buildString {
    appendLine("-- morphology --"); appendLine(morphologyDemo())
    appendLine("-- threshold --"); appendLine(thresholdDemo())
    appendLine("-- color spaces --"); appendLine(colorSpacesDemo())
    appendLine("-- hough lines --"); appendLine(houghLinesDemo())
    appendLine("-- geometric transforms --"); appendLine(geometricTransformsDemo())
    appendLine("-- histogram equalization --"); appendLine(histogramEqualizationDemo())
    appendLine("-- drawing --"); appendLine(drawingDemo())
    appendLine("-- scalar arithmetic --"); appendLine(scalarArithmeticDemo())
}

// ------------------------------------------------------------------ inputs

/** Black canvas with a bright rectangle plus deterministic speckle noise. */
private fun syntheticScene(): Mat {
    zeros(48, 64, MatType.CV_8UC3).use { canvas ->
        canvas.rectangle(
            from = Point(12, 12),
            to = Point(52, 36),
            color = Scalar.all(255.0),
        )
        for (y in 0 until canvas.rows step 7) canvas.put(y, 2, value = 255.0)
        return canvas.clone()
    }
}

/** Horizontal 0..255 ramp in a single-channel 8-bit row. */
private fun gradientRamp(): Mat {
    val base = mat(1, 256, MatType.CV_8UC1)
    for (c in 0 until 256) base.put(0, c, value = c.toDouble())
    return base.clone().also { base.close() }
}

/** 64-wide grayscale row squeezed into a narrow [16, 143] contrast band. */
private fun lowContrastRamp(): Mat {
    val base = mat(1, 64, MatType.CV_8UC1)
    for (c in 0 until 64) base.put(0, c, value = (16 + c * 2).toDouble())
    return base.clone().also { base.close() }
}
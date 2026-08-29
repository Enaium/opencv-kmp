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
import cn.enaium.opencv.createBackgroundSubtractorKNN
import cn.enaium.opencv.createBackgroundSubtractorMOG2
import cn.enaium.opencv.createFarnebackOpticalFlow
import cn.enaium.opencv.mat
import cn.enaium.opencv.zeros

/**
 * Ports of `tutorial_code/video`:
 *  - bg_sub.cpp (MOG2 / KNN background subtraction)
 *  - optical_flow_dense.cpp (Farneback dense flow)
 * Frames are synthesized (a moving disk on a static background) so the demo
 * runs without a camera or a video file.
 */
fun runVideoTutorials(): String = buildString {
    appendLine("-- background subtraction --")
    backgroundSubtractionDemo().also { append(it) }

    appendLine("-- dense optical flow --")
    farnebackDemo().also { append(it) }
}

/**
 * A static bright disk plus a second disk moving right along the row
 * [motionY]; frame i places the moving disk at [motionX] + [step]*i.
 */
private fun syntheticFrames(count: Int, motionX: Int, motionY: Int, step: Int): List<Mat> =
    List(count) { i ->
        val f = zeros(96, 96, MatType.CV_8UC1)
        f.circle(Point(24, 24), 12, Scalar.all(160.0), FILLED)
        val x = (motionX + step * i).coerceIn(8, 88)
        f.circle(Point(x, motionY), 8, Scalar.all(255.0), FILLED)
        f
    }

private fun backgroundSubtractionDemo(): String = buildString {
    val frames = syntheticFrames(count = 12, motionX = 16, motionY = 64, step = 5)
    try {
        createBackgroundSubtractorMOG2().use { mog2 ->
            createBackgroundSubtractorKNN().use { knn ->
                var mogPixels = 0
                var knnPixels = 0
                frames.forEach { frame ->
                    mat(96, 96, MatType.CV_8UC1).use { mogFg -> mog2.apply(frame, mogFg); mogPixels += mogFg.nonZeroCount }
                    mat(96, 96, MatType.CV_8UC1).use { knnFg -> knn.apply(frame, knnFg); knnPixels += knnFg.nonZeroCount }
                }
                line("MOG2 cumulative foreground", "${mogPixels}")
                line("KNN cumulative foreground", "${knnPixels}")
                line("MOG2 tracked disk pixels", "${frames.last().let { last ->
                    mat(96, 96, MatType.CV_8UC1).use { fg -> mog2.apply(last, fg); fg.nonZeroCount }
                }}")
            }
        }
    } finally {
        frames.forEach { it.close() }
    }
}

private fun farnebackDemo(): String = buildString {
    val frames = syntheticFrames(count = 3, motionX = 16, motionY = 64, step = 8)
    try {
        val a = frames[0]
        val b = frames[1]
        createFarnebackOpticalFlow().use { flow ->
            mat(96, 96, MatType.CV_32FC2).use { out ->
                flow.calc(a, b, out)
                var right = 0
                var largeMag = 0
                for (r in 0 until out.rows) {
                    for (c in 0 until out.cols) {
                        val vx = out.at(r, c, 0).toFloat()
                        val vy = out.at(r, c, 1).toFloat()
                        val mag = kotlin.math.sqrt(vx * vx + vy * vy)
                        if (mag > 0.5) largeMag++
                        if (vx > 0.5) right++
                    }
                }
                line("flow vectors pointing right", "${right}")
                line("flow vectors with mag > 0.5", "${largeMag}")
            }
        }
    } finally {
        frames.forEach { it.close() }
    }
}
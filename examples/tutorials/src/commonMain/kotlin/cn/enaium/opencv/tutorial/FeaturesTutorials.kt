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
import cn.enaium.opencv.bfMatcherCreate
import cn.enaium.opencv.orbCreate
import cn.enaium.opencv.zeros

/**
 * Port of the keypoint-matching tutorials (feature_description.cpp /
 * AKAZE_match.cpp): ORB keypoints + descriptors, brute-force matching with
 * a ratio test.
 */
fun runFeaturesTutorials(assetsDir: String): String = buildString {
    appendLine("-- ORB detection & matching --")

    val img1 = loadOr(assetsDir, "box.png") { syntheticShape(64) }
    val img2 = loadOr(assetsDir, "box_in_scene.png") { syntheticShape(64).also { s -> s.shiftInPlace(4, 3) } }

    img1.use { a ->
        img2.use { b ->
            orbCreate().use { orb ->
                val (kp1, desc1) = orb.detectAndCompute(a)
                val (kp2, desc2) = orb.detectAndCompute(b)
                line("keypoints img1", "${kp1.size}")
                line("keypoints img2", "${kp2.size}")
                line("desc1 shape", "${desc1.rows}x${desc1.cols}")
                line("desc2 shape", "${desc2.rows}x${desc2.cols}")

                bfMatcherCreate(cn.enaium.opencv.MatcherNorms.HAMMING).use { matcher ->
                    val knn = matcher.knnMatch(desc1, desc2, k = 2)
                    var good = 0
                    for (pair in knn) {
                        if (pair.size >= 2 && pair[0].distance < 0.75f * pair[1].distance) good++
                    }
                    line("good ratio matches", "$good")
                }
            }
        }
    }
}

/** 64x64 dark canvas with a bright disk and a bright square. */
private fun syntheticShape(n: Int): Mat {
    val m = zeros(n, n, MatType.CV_8UC1)
    m.circle(Point(n / 2, n / 2), n / 5, Scalar.all(255.0), FILLED)
    m.rectangle(Point(6, 6), Point(14, 14), Scalar.all(255.0), FILLED)
    return m
}

/** Shifts every pixel down-right by [dy]/[dx]; the top/left band is zeroed. */
private fun Mat.shiftInPlace(dx: Int, dy: Int) {
    val copy = clone()
    for (r in 0 until rows) {
        for (c in 0 until cols) {
            val sy = r - dy
            val sx = c - dx
            this[r, c] = if (sy >= 0 && sx >= 0) copy[sy, sx] else 0.0
        }
    }
    copy.close()
}
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

import cn.enaium.opencv.imshow
import cn.enaium.opencv.destroyAllWindows
import cn.enaium.opencv.namedWindow
import cn.enaium.opencv.waitKey

/**
 * Port of the HighGUI window tutorials (AddingImagesTrackbar.cpp /
 * BasicLinearTransformsTrackbar.cpp): create a window, show a synthetic
 * image, pump a single waitKey so the image is painted, then tear the
 * window down again. Headless builtin (Linux CI) reports to stderr instead
 * of opening a window; macOS/Windows open real windows briefly.
 */
fun runHighGuiTutorials(): String = buildString {
    appendLine("-- imshow / namedWindow / waitKey --")
    syntheticScene().use { img ->
        namedWindow("tutorial")
        imshow("tutorial", img)
        val key = waitKey(1)
        destroyAllWindows()
        line("waitKey returned", "$key")
        line("window shown", "true")
    }
}
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
 * BasicLinearTransformsTrackbar.cpp): create a window, show an image, pump
 * waitKey so the window is painted, then tear it down. macOS/Windows open
 * real windows; the builtin headless backend (Linux CI) reports to stderr.
 */
fun runHighGuiTutorials(): String = buildString {
    appendLine("-- imshow / namedWindow / waitKey --")
    // Demo-size image (480x640) so the window is clearly visible; the raw
    // scene is small (48x64), which would open a tiny window.
    syntheticScene(rows = 480, cols = 640).use { img ->
        namedWindow("tutorial")
        imshow("tutorial", img)
        // Hold for a few seconds so the user can see the window before it
        // closes; -1 means no key was pressed.
        val key = waitKey(3000)
        destroyAllWindows()
        line("waitKey returned", "$key")
        line("window shown", "true")
        line("image size", "${img.rows}x${img.cols}")
    }
}
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

/**
 * Aggregates every module tutorial into one report. [assetsDir] points at
 * the directory holding the optional sample images (blank to skip assets).
 */
fun runTutorials(assetsDir: String): String = buildString {
    appendLine("# imgproc")
    append(runImgprocTutorials(assetsDir))

    appendLine("# video")
    append(runVideoTutorials())

    appendLine("# features")
    append(runFeaturesTutorials(assetsDir))

    appendLine("# objdetect")
    append(runObjdetectTutorials())

    appendLine("# photo")
    append(runPhotoTutorials())

    appendLine("# calib3d")
    append(runCalib3dTutorials())

    appendLine("# highgui")
    append(runHighGuiTutorials())
}
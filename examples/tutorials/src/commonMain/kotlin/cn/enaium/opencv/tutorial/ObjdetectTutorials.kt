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
import cn.enaium.opencv.qrCodeDetectorCreate
import cn.enaium.opencv.qrCodeEncoderCreate

/**
 * Ports of the object-detection tutorials:
 *  - QR code encode (qrCodeEncoderCreate) -> draw onto a canvas ->
 *    [qrCodeDetectorCreate] detect + decode round-trip.
 * Everything is synthesized; no external model files are downloaded.
 */
fun runObjdetectTutorials(): String = buildString {
    appendLine("-- QR encode/detect/decode round trip --")
    qrDemo().also { append(it) }
}

private fun qrDemo(): String = buildString {
    // QRCodeEncoder emits a tight grayscale pattern; upscale it to give the
    // detector a realistic input size and verify the payload survives.
    qrCodeEncoderCreate().use { encoder ->
        encoder.encode("opencv-kmp tutorial").use { qr ->
            qr.resize(300, 300).use { large ->
                line("encoded size", "${qr.rows}x${qr.cols} -> ${large.rows}x${large.cols}")

                qrCodeDetectorCreate().use { detector ->
                    val points = cn.enaium.opencv.mat(4, 1, cn.enaium.opencv.MatType.CV_32FC2)
                    val found = detector.detect(large, points)
                    line("detected", "$found")
                    val decoded = detector.decode(large, points)
                    line("decoded", "'$decoded'")
                    if (decoded != "opencv-kmp tutorial") {
                        throw AssertionError("QR round trip failed: '$decoded'")
                    }
                    points.close()
                }
            }
        }
    }
}


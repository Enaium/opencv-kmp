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
package cn.enaium.opencv

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Coverage for the org.opencv.photo module: inpainting, non-local means
 * denoising, seamless cloning, tonemapping, exposure merging/alignment,
 * color correction and pencil sketch. All inputs are deterministic
 * synthetic Mats so the suite runs on both backends.
 */
class PhotoTest {

    /** Mean squared error between two same-shaped CV_8U Mats. */
    private fun mse(a: Mat, b: Mat): Double {
        val pa = a.pixels
        val pb = b.pixels
        var sum = 0.0
        for (i in pa.indices) {
            val d = (pa[i].toInt() and 0xFF) - (pb[i].toInt() and 0xFF)
            sum += d * d
        }
        return sum / pa.size
    }

    @Test
    fun inpaintFillsScratch() {
        mat(32, 32, MatType.CV_8UC1).use { src ->
            // smooth vertical gradient
            for (r in 0 until 32) {
                for (c in 0 until 32) src.put(r, c, 0, ((r * 8) % 256).toDouble())
            }
            // 1px white scratch in the middle of the gradient
            src.put(16, 16, 0, 255.0)
            zeros(32, 32, MatType.CV_8UC1).use { mask ->
                mask.put(16, 16, 0, 255.0)
                photoInpaint(src, mask, 3.0, Photo.INPAINT_TELEA).use { dst ->
                    assertEquals(32, dst.rows)
                    assertEquals(32, dst.cols)
                    assertEquals(MatType.CV_8UC1, dst.type)
                    val repaired = dst[16, 16]
                    assertNotEquals(255.0, repaired, "inpainted pixel must not keep the scratch value")
                    // gradient value at (16,16) is 128; the repair should blend with the neighborhood
                    assertTrue(
                        abs(repaired - 128.0) < 40.0,
                        "inpainted pixel should blend with surroundings, got $repaired",
                    )
                }
            }
        }
    }

    @Test
    fun seamlessClonePastesStructuredPatch() {
        // 12x12 patch: left half red, right half blue (strong internal gradient)
        mat(12, 12, MatType.CV_8UC3).use { patch ->
            for (r in 0 until 12) {
                for (c in 0 until 12) {
                    if (c < 6) {
                        patch.put(r, c, 0, 0.0)
                        patch.put(r, c, 1, 0.0)
                        patch.put(r, c, 2, 255.0)
                    } else {
                        patch.put(r, c, 0, 255.0)
                        patch.put(r, c, 1, 0.0)
                        patch.put(r, c, 2, 0.0)
                    }
                }
            }
            mat(32, 32, MatType.CV_8UC3, Scalar.all(128.0)).use { dst ->
                mat(12, 12, MatType.CV_8UC1, Scalar.all(255.0)).use { mask ->
                    photoSeamlessClone(patch, dst, mask, Point(16, 16), Photo.NORMAL_CLONE).use { blend ->
                        assertEquals(32, blend.rows)
                        assertEquals(32, blend.cols)
                        assertEquals(MatType.CV_8UC3, blend.type)
                        // left-of-seam pixel keeps the red tint (R much stronger than B)
                        val red = blend.at(16, 13, 2)
                        val blue = blend.at(16, 13, 0)
                        assertTrue(
                            red - blue > 30.0,
                            "cloned red half must stay redder than blue, R=$red B=$blue",
                        )
                        // right-of-seam pixel keeps the blue tint
                        val blueSideB = blend.at(16, 19, 0)
                        val blueSideR = blend.at(16, 19, 2)
                        assertTrue(
                            blueSideB - blueSideR > 30.0,
                            "cloned blue half must stay bluer than red, B=$blueSideB R=$blueSideR",
                        )
                    }
                }
            }
        }
    }

    @Test
    fun fastNlMeansDenoisingApproachesCleanImage() {
        mat(64, 64, MatType.CV_8UC1).use { clean ->
            for (r in 0 until 64) {
                for (c in 0 until 64) clean.put(r, c, 0, (((r + c) * 2) % 256).toDouble())
            }
            zeros(64, 64, MatType.CV_8UC1).use { noise ->
                noise.randn(Scalar(), Scalar(25.0))
                (clean + noise).use { noisy ->
                    photoFastNlMeansDenoising(noisy, 10f, 7, 21).use { denoised ->
                        assertEquals(64, denoised.rows)
                        assertEquals(64, denoised.cols)
                        assertEquals(MatType.CV_8UC1, denoised.type)
                        val noisyMse = mse(noisy, clean)
                        val denoisedMse = mse(denoised, clean)
                        assertTrue(
                            denoisedMse < noisyMse,
                            "denoising must move the image toward the clean signal: " +
                                "noisyMSE=$noisyMse denoisedMSE=$denoisedMse",
                        )
                    }
                }
            }
        }
    }

    @Test
    fun tonemapReinhardMapsToUnitRange() {
        mat(16, 16, MatType.CV_32FC3).use { hdr ->
            // HDR-ish values from 0 up to 9
            for (r in 0 until 16) {
                for (c in 0 until 16) {
                    val v = ((r * 16 + c) % 10).toDouble()
                    hdr.put(r, c, 0, v)
                    hdr.put(r, c, 1, v * 0.5)
                    hdr.put(r, c, 2, v * 0.25)
                }
            }
            createTonemapReinhard().use { tm ->
                tm.process(hdr).use { out ->
                    assertEquals(16, out.rows)
                    assertEquals(16, out.cols)
                    assertEquals(MatType.CV_32FC3, out.type)
                    var min = Double.MAX_VALUE
                    var max = -Double.MAX_VALUE
                    for (r in 0 until 16) {
                        for (c in 0 until 16) {
                            for (ch in 0 until 3) {
                                val v = out.at(r, c, ch)
                                if (v < min) min = v
                                if (v > max) max = v
                            }
                        }
                    }
                    assertTrue(
                        min >= 0.0 && max <= 1.0,
                        "tonemapped output must lie in [0, 1], got min=$min max=$max",
                    )
                    assertTrue(max > 0.5, "bright HDR content must stay bright, got max=$max")
                }
            }
        }
    }

    @Test
    fun mergeMertensFusesExposures() {
        mat(20, 20, MatType.CV_8UC3, Scalar.all(20.0)).use { dark ->
            mat(20, 20, MatType.CV_8UC3, Scalar.all(128.0)).use { mid ->
                mat(20, 20, MatType.CV_8UC3, Scalar.all(240.0)).use { bright ->
                    createMergeMertens().use { merger ->
                        merger.process(listOf(dark, mid, bright)).use { fused ->
                            assertEquals(20, fused.rows)
                            assertEquals(20, fused.cols)
                            assertEquals(MatType.CV_32FC3, fused.type)
                            val v = fused.at(10, 10, 0)
                            assertTrue(
                                v >= 0.0 && v <= 1.0,
                                "fused exposure value must be in [0, 1], got $v",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun alignMtbProcessAlignsSequence() {
        // AlignMTB's median-threshold bitmap needs real texture; use a
        // deterministic pattern shifted by (0,0), (3,0) and (6,0).
        fun textured(shift: Int): Mat = mat(32, 32, MatType.CV_8UC3).also { m ->
            for (r in 0 until 32) {
                for (c in 0 until 32) {
                    val v = if (c >= shift) ((r * 37 + (c - shift) * 71) % 256).toDouble() else 30.0
                    m.put(r, c, 0, v); m.put(r, c, 1, v); m.put(r, c, 2, v)
                }
            }
        }
        val images = listOf(textured(0), textured(3), textured(6))
        try {
            createAlignMTB().use { aligner ->
                val aligned = aligner.process(images)
                try {
                    assertEquals(3, aligned.size)
                    val targetRows = aligned.first().rows
                    val targetCols = aligned.first().cols
                    assertTrue(targetRows >= 1 && targetCols >= 1, "aligned frame must not be empty")
                    aligned.forEach { m ->
                        assertEquals(targetRows, m.rows)
                        assertEquals(targetCols, m.cols)
                        assertEquals(MatType.CV_8UC3, m.type)
                    }

                } finally {
                    aligned.forEach { it.close() }
                }
            }
        } finally {
            images.forEach { it.close() }
        }
    }

    @Test
    fun colorCorrectionModelCorrectsImage() {
        // 24 synthetic Macbeth-like patches (RGB, values in [0, 1])
        mat(24, 1, cvMakeType(CV_64F, 3)).use { patches ->
            for (i in 0 until 24) {
                val v = i / 23.0
                patches.put(i, 0, 0, v)
                patches.put(i, 0, 1, 1.0 - v)
                patches.put(i, 0, 2, (v + 0.5) % 1.0)
            }
            colorCorrectionModel(patches, Photo.COLORCHECKER_MACBETH).use { model ->
                model.setMaxCount(300)
                model.setEpsilon(1e-3)
                model.compute().use { ccm ->
                    assertEquals(3, ccm.rows, "linear CCM must be 3 rows")
                    assertTrue(
                        ccm.cols == 3 || ccm.cols == 4,
                        "CCM must be 3x3 or 4x3, got ${ccm.rows}x${ccm.cols}",
                    )
                }
                mat(8, 8, MatType.CV_8UC3, Scalar(100.0, 150.0, 200.0)).use { img ->
                    model.correctImage(img).use { corrected ->
                        assertEquals(img.rows, corrected.rows)
                        assertEquals(img.cols, corrected.cols)
                        assertEquals(img.type, corrected.type)
                    }
                }
            }
        }
    }

    @Test
    fun pencilSketchReturnsPair() {
        mat(32, 32, MatType.CV_8UC3).use { img ->
            for (r in 0 until 32) {
                for (c in 0 until 32) {
                    img.put(r, c, 0, (((r + c) * 4) % 256).toDouble())
                    img.put(r, c, 1, (((r * 2 + c) * 3) % 256).toDouble())
                    img.put(r, c, 2, (((r + c * 3) * 2) % 256).toDouble())
                }
            }
            val (sketch, color) = photoPencilSketch(img)
            try {
                assertEquals(32, sketch.rows)
                assertEquals(32, sketch.cols)
                assertEquals(MatType.CV_8UC1, sketch.type)
                assertEquals(32, color.rows)
                assertEquals(32, color.cols)
                assertEquals(MatType.CV_8UC3, color.type)
            } finally {
                sketch.close()
                color.close()
            }
        }
    }
}

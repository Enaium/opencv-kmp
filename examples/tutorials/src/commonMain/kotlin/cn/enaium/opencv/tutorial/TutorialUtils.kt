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

/**
 * Shared helpers for the tutorial demos. Every demo builds its own input or
 * loads it from the OpenCV submodule's samples/data directory, so the demos
 * run with zero network access and no model downloads.
 */

/** Appends an indented `label: value` line to a report builder. */
internal fun StringBuilder.line(label: String, value: String) {
    append("  $label: $value\n")
}

/**
 * Loads [name] from the OpenCV submodule's samples/data directory (or from
 * [dir] when set); falls back to [fallback] when the file is unavailable.
 */
internal fun loadOr(dir: String, name: String, fallback: () -> Mat): Mat {
    val candidates = listOf(
        dir.takeIf { it.isNotBlank() }?.let { "$it/$name" },
        "opencv/samples/data/$name",
        "../opencv/samples/data/$name",
        "../../opencv/samples/data/$name",
    ).filterNotNull()
    for (path in candidates) {
        val m = cn.enaium.opencv.imread(path)
        if (m != null) return m
    }
    return fallback()
}

/** Adds deterministic uniform speckle noise (value `sigma * lcg`) to [this]. */
internal fun Mat.addSpeckle(seed: Int, sigma: Double = 12.0): Mat {
    var state = seed
    for (r in 0 until rows) {
        for (c in 0 until cols) {
            state = state * 1103515245 + 12345
            val noise = ((state ushr 8) % 1000 / 1000.0 - 0.5) * 2.0 * sigma
            this[r, c] = (this[r, c] + noise).coerceIn(0.0, 255.0)
        }
    }
    return this
}
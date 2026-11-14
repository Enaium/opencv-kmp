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
import cn.enaium.opencv.destroyAllWindows
import cn.enaium.opencv.imshow
import cn.enaium.opencv.waitKey

/** One named picture a tutorial produced. */
class DemoImage(val title: String, val mat: Mat)

/**
 * Collects the pictures the tutorials want to present.
 *
 * The tutorials stay headless and report text; the pictures they mark with
 * [add] are shown by the platform: the desktop entry points through
 * [imshowAll], the Android app on a Compose canvas. [add] stores a copy, so
 * the demo can keep releasing its intermediates with `use {}`.
 */
class Gallery : AutoCloseable {
    private val items = mutableListOf<DemoImage>()

    val images: List<DemoImage> get() = items

    fun add(title: String, mat: Mat): Gallery = apply { items += DemoImage(title, mat.clone()) }

    override fun close() {
        items.forEach { it.mat.close() }
        items.clear()
    }
}

/** `--hold=<ms>` from the command line; `0` (wait for a key) when absent. */
fun Array<String>.imshowHoldMillis(): Int =
    firstOrNull { it.startsWith("--hold=") }?.substringAfter('=')?.toIntOrNull() ?: 0

/**
 * Desktop presentation of every collected picture: one highgui window each,
 * then a key press. [holdMillis] `0` waits for the user; a positive value
 * closes the windows after that many milliseconds, which keeps scripted runs
 * from blocking forever.
 */
fun Gallery.imshowAll(holdMillis: Int = 0) {
    if (images.isEmpty()) return
    images.forEach { imshow(it.title, it.mat) }
    waitKey(holdMillis)
    destroyAllWindows()
}

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
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

import cn.enaium.opencv.example.Gallery
import cn.enaium.opencv.example.imshowAll
import cn.enaium.opencv.example.imshowHoldMillis
import cn.enaium.opencv.example.runBasicDemo
import cn.enaium.opencv.example.runImageProcessingDemo
import cn.enaium.opencv.example.runTutorialDemos
import kotlinx.cinterop.toKString
import platform.posix.getenv

fun main(args: Array<String>) {
    // POSIX mkdir takes a mode argument while MinGW does not; writing straight
    // into the temporary directory keeps this source set portable.
    val outDir = getenv("TMPDIR")?.toKString() ?: getenv("TEMP")?.toKString() ?: "/tmp"
    Gallery().use { gallery ->
        println("== basic ==")
        println(runBasicDemo())
        println("== image processing ==")
        println(runImageProcessingDemo(outDir, gallery))
        println("== tutorials ==")
        println(runTutorialDemos(gallery))
        gallery.imshowAll(args.imshowHoldMillis())
    }
}

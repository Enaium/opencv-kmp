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
@file:JvmName("BitmapMatConversions")

package cn.enaium.opencv

import android.graphics.Bitmap

/**
 * Android Bitmap <-> Mat interop, mirroring `org.opencv.android.Utils`
 * (`bitmapToMat` / `matToBitmap`) as Kotlin extensions instead of statics.
 *
 * Conventions match the official Utils: bitmaps are ARGB_8888, Mats are
 * CV_8UC4 holding RGBA bytes.
 */

/** ARGB_8888 int (0xAARRGGBB) -> RGBA byte quadruple. */
private fun argbToRgba(argb: Int, rgba: ByteArray, at: Int) {
    rgba[at] = ((argb ushr 16) and 0xFF).toByte() // R
    rgba[at + 1] = ((argb ushr 8) and 0xFF).toByte() // G
    rgba[at + 2] = (argb and 0xFF).toByte() // B
    rgba[at + 3] = ((argb ushr 24) and 0xFF).toByte() // A
}

/** RGBA byte quadruple -> ARGB_8888 int (0xAARRGGBB). */
private fun rgbaToArgb(rgba: ByteArray, at: Int): Int =
    ((rgba[at + 3].toInt() and 0xFF) shl 24) or
        ((rgba[at].toInt() and 0xFF) shl 16) or
        ((rgba[at + 1].toInt() and 0xFF) shl 8) or
        (rgba[at + 2].toInt() and 0xFF)

private fun Bitmap.readPixels(): Pair<IntArray, Int> {
    val width = width
    val pixels = IntArray(width * height)
    getPixels(pixels, 0, width, 0, 0, width, height)
    return pixels to width
}

/**
 * Converts this ARGB_8888 bitmap into a new CV_8UC4 [Mat] holding RGBA bytes,
 * like `Utils.bitmapToMat`. The bitmap is left untouched; use [Bitmap.copyTo]
 * to write into an existing Mat instead.
 */
fun Bitmap.toMat(): Mat {
    val (pixels, width) = readPixels()
    val mat = mat(rows = height, cols = width, type = MatType.CV_8UC4)
    val rgba = mat.pixels
    var p = 0
    for (argb in pixels) {
        argbToRgba(argb, rgba, p)
        p += 4
    }
    mat.pixels = rgba
    return mat
}

/**
 * Copies this bitmap's pixels into an existing CV_8UC4 [mat]; shape-checked
 * counterpart of `Utils.bitmapToMat`. Returns false on dimension mismatch.
 */
fun Bitmap.copyTo(mat: Mat): Boolean {
    if (mat.rows != height || mat.cols != width || mat.type != MatType.CV_8UC4) {
        return false
    }
    val (pixels, width) = readPixels()
    val rgba = mat.pixels
    var p = 0
    for (argb in pixels) {
        argbToRgba(argb, rgba, p)
        p += 4
    }
    mat.pixels = rgba
    return true
}

/**
 * Renders this CV_8UC4 RGBA matrix into a new ARGB_8888 bitmap, like
 * `Utils.matToBitmap`. The matrix is left untouched; use [Mat.copyTo] to
 * draw into an existing bitmap instead.
 */
fun Mat.toBitmap(config: Bitmap.Config = Bitmap.Config.ARGB_8888): Bitmap =
    Bitmap.createBitmap(cols, rows, config).also { it.copyFrom(this) }

/**
 * Draws this CV_8UC4 RGBA matrix onto an existing bitmap of the same
 * dimensions. Returns false on dimension mismatch.
 */
fun Mat.copyTo(bitmap: Bitmap): Boolean {
    if (type != MatType.CV_8UC4 || bitmap.width != cols || bitmap.height != rows) {
        return false
    }
    if (!bitmap.isMutable) return false
    val rgba = pixels
    val out = IntArray(rows * cols)
    var p = 0
    for (i in out.indices) {
        out[i] = rgbaToArgb(rgba, p)
        p += 4
    }
    bitmap.setPixels(out, 0, cols, 0, 0, cols, rows)
    return true
}

private fun Bitmap.copyFrom(mat: Mat): Bitmap = also { mat.copyTo(it) }

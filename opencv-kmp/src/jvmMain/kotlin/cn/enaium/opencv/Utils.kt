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

/**
 * Reflection-based parity with `org.opencv.android.Utils`.
 *
 * `android.graphics.Bitmap` is reached purely through [Class.forName] so this
 * source also compiles and links on desktop JVM. There the class lookup fails,
 * which both functions report as `false`: on desktop JVM they are no-ops that
 * never touch the [Mat]. On Android, pass any `android.graphics.Bitmap`.
 *
 * Both directions assume `Bitmap.Config.ARGB_8888` and a [Mat] whose type is
 * [MatType.CV_8UC4] with `rows == bitmap.height`, `cols == bitmap.width`;
 * the Mat holds RGBA bytes while bitmap pixels are ARGB ints.
 */
object AndroidUtils {

    private const val BITMAP_CLASS = "android.graphics.Bitmap"
    private const val CONFIG_CLASS = "android.graphics.Bitmap\$Config"

    private val INT: Class<*> = Int::class.javaPrimitiveType!!

    /** Cached reflection handles; null when android.graphics is absent. */
    private val bitmapClass: Class<*>? by lazy { runCatching { Class.forName(BITMAP_CLASS) }.getOrNull() }

    private val configClass: Class<*>? by lazy { runCatching { Class.forName(CONFIG_CLASS) }.getOrNull() }

    private fun argb8888(): Any? =
        configClass?.enumConstants?.singleOrNull { it.toString() == "ARGB_8888" }

    /**
     * Fills [mat] (`CV_8UC4`, RGBA) from an `ARGB_8888` bitmap's pixels,
     * like `org.opencv.android.Utils.bitmapToMat`. Returns `false` when
     * reflection fails or the shapes do not match.
     */
    fun bitmapToMat(bitmap: Any, mat: Mat): Boolean = try {
        val clazz = bitmapClass ?: return false
        if (!clazz.isInstance(bitmap)) return false
        val width = clazz.getMethod("getWidth").invoke(bitmap) as Int
        val height = clazz.getMethod("getHeight").invoke(bitmap) as Int
        val config = runCatching { clazz.getMethod("getConfig").invoke(bitmap) }.getOrNull()
        if (config == null || config != argb8888()) return false
        if (mat.rows != height || mat.cols != width || mat.type != MatType.CV_8UC4) return false

        val pixels = IntArray(width * height)
        clazz.getMethod(
            "getPixels",
            IntArray::class.java, INT, INT, INT, INT, INT, INT,
        ).invoke(bitmap, pixels, 0, width, 0, 0, width, height)

        val bytes = ByteArray(pixels.size * 4)
        for (i in pixels.indices) {
            val p = pixels[i]
            val o = i * 4
            // ARGB int -> RGBA byte order
            bytes[o] = ((p shr 16) and 0xFF).toByte()
            bytes[o + 1] = ((p shr 8) and 0xFF).toByte()
            bytes[o + 2] = (p and 0xFF).toByte()
            bytes[o + 3] = ((p ushr 24) and 0xFF).toByte()
        }
        mat.pixels = bytes
        true
    } catch (_: Throwable) {
        false
    }

    /**
     * Writes [mat]'s RGBA bytes into an `ARGB_8888` bitmap,
     * like `org.opencv.android.Utils.matToBitmap`. Returns `false` when
     * reflection fails or the shapes do not match.
     */
    fun matToBitmap(mat: Mat, bitmap: Any): Boolean = try {
        val clazz = bitmapClass ?: return false
        if (!clazz.isInstance(bitmap)) return false
        if (mat.type != MatType.CV_8UC4) return false
        val width = clazz.getMethod("getWidth").invoke(bitmap) as Int
        val height = clazz.getMethod("getHeight").invoke(bitmap) as Int
        if (mat.rows != height || mat.cols != width) return false

        val bytes = mat.pixels
        if (bytes.size != width * height * 4) return false

        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val o = i * 4
            // RGBA byte order -> ARGB int
            pixels[i] = ((bytes[o + 3].toInt() and 0xFF) shl 24) or
                ((bytes[o].toInt() and 0xFF) shl 16) or
                ((bytes[o + 1].toInt() and 0xFF) shl 8) or
                (bytes[o + 2].toInt() and 0xFF)
        }
        clazz.getMethod(
            "setPixels",
            IntArray::class.java, INT, INT, INT, INT, INT, INT,
        ).invoke(bitmap, pixels, 0, width, 0, 0, width, height)
        true
    } catch (_: Throwable) {
        false
    }
}

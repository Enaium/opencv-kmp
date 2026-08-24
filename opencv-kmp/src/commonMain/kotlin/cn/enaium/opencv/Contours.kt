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
 * Pure-Kotlin codec for the contour wire format shared by every platform:
 *
 * ```
 * uint32 count
 * per contour: uint32 point-count, then int32 x,y pairs (little endian)
 * ```
 *
 * The same layout backs `cvk_find_contours` / `cvk_draw_contours` /
 * `cvk_approx_poly_dp` in native/include/opencv_kmp.h.
 */
internal object ContourCodec {

    fun encode(contours: List<List<Point>>): ByteArray {
        var pointCount = 0
        for (contour in contours) pointCount += contour.size
        val buffer = ByteArray(4 + contours.size * 4 + pointCount * 8)
        var offset = writeUInt(buffer, 0, contours.size)
        for (contour in contours) {
            offset = writeUInt(buffer, offset, contour.size)
            for (point in contour) {
                offset = writeInt(buffer, offset, point.x)
                offset = writeInt(buffer, offset, point.y)
            }
        }
        return buffer
    }

    fun decode(data: ByteArray): List<List<Point>> {
        if (data.size < 4) return emptyList()
        var offset = 0
        val count = readUInt(data, offset).also { offset += 4 }
        val result = ArrayList<List<Point>>(count)
        repeat(count) {
            if (offset + 4 > data.size) return result
            val points = readUInt(data, offset).also { offset += 4 }
            val contour = ArrayList<Point>(points)
            repeat(points) {
                if (offset + 8 > data.size) return result
                val x = readInt(data, offset).also { offset += 4 }
                val y = readInt(data, offset).also { offset += 4 }
                contour.add(Point(x, y))
            }
            result.add(contour)
        }
        return result
    }

    private fun writeUInt(target: ByteArray, at: Int, value: Int): Int {
        target[at] = (value and 0xFF).toByte()
        target[at + 1] = ((value ushr 8) and 0xFF).toByte()
        target[at + 2] = ((value ushr 16) and 0xFF).toByte()
        target[at + 3] = ((value ushr 24) and 0xFF).toByte()
        return at + 4
    }

    private fun writeInt(target: ByteArray, at: Int, value: Int): Int = writeUInt(target, at, value)

    private fun readUInt(source: ByteArray, at: Int): Int =
        (source[at].toInt() and 0xFF) or
            ((source[at + 1].toInt() and 0xFF) shl 8) or
            ((source[at + 2].toInt() and 0xFF) shl 16) or
            ((source[at + 3].toInt() and 0xFF) shl 24)

    private fun readInt(source: ByteArray, at: Int): Int = readUInt(source, at)
}

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
 * JNI bridge for the JVM target.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_Jni_<name>`
 * function in jni_bridge.cpp. All members are public (no `internal`
 * modifier) so their JVM names are not mangled by the Kotlin compiler.
 *
 * Mat handles travel as jlong pointers; cvk scalar/rect structs are expanded
 * into primitive arguments so no marshaling structs are needed here.
 */
internal object Jni {

    init {
        NativeLoader.load()
    }

    // Info / errors

    external fun version(): String
    external fun lastError(): String?

    // Mat lifecycle / properties

    external fun matCreate(rows: Int, cols: Int, type: Int): Long
    external fun matCreateFilled(rows: Int, cols: Int, type: Int, v0: Double, v1: Double, v2: Double, v3: Double): Long
    external fun matZeros(rows: Int, cols: Int, type: Int): Long
    external fun matOnes(rows: Int, cols: Int, type: Int): Long
    external fun matEye(rows: Int, cols: Int, type: Int): Long
    external fun matClone(mat: Long): Long
    external fun matRoi(mat: Long, x: Int, y: Int, width: Int, height: Int): Long
    external fun matRelease(mat: Long)

    external fun matRows(mat: Long): Int
    external fun matCols(mat: Long): Int
    external fun matType(mat: Long): Int
    external fun matChannels(mat: Long): Int
    external fun matElemSize(mat: Long): Long
    external fun matTotal(mat: Long): Long
    external fun matIsEmpty(mat: Long): Boolean
    external fun matGet(mat: Long, row: Int, col: Int, channel: Int): Double
    external fun matSet(mat: Long, row: Int, col: Int, channel: Int, value: Double)
    external fun matGetData(mat: Long): ByteArray
    external fun matSetData(mat: Long, data: ByteArray)

    // Conversions / arithmetic

    external fun convertTo(mat: Long, rtype: Int, alpha: Double, beta: Double): Long
    external fun add(a: Long, b: Long): Long
    external fun subtract(a: Long, b: Long): Long
    external fun multiply(a: Long, b: Long, scale: Double): Long
    external fun divide(a: Long, b: Long): Long
    external fun scaleAdd(mat: Long, alpha: Double, beta: Double): Long
    external fun absdiff(a: Long, b: Long): Long
    external fun bitwiseAnd(a: Long, b: Long): Long
    external fun bitwiseOr(a: Long, b: Long): Long
    external fun bitwiseXor(a: Long, b: Long): Long
    external fun bitwiseNot(a: Long): Long
    external fun min(a: Long, b: Long): Long
    external fun max(a: Long, b: Long): Long
    external fun inRange(
        mat: Long,
        l0: Double, l1: Double, l2: Double, l3: Double,
        u0: Double, u1: Double, u2: Double, u3: Double,
    ): Long

    external fun transpose(mat: Long): Long
    external fun flip(mat: Long, flipCode: Int): Long

    // Reductions / statistics

    external fun mean(mat: Long): DoubleArray
    external fun sum(mat: Long): DoubleArray
    external fun meanStdDev(mat: Long): DoubleArray
    external fun minMaxLoc(mat: Long): DoubleArray
    external fun countNonZero(mat: Long): Int

    // imgproc

    external fun cvtColor(mat: Long, code: Int): Long
    external fun resize(mat: Long, width: Int, height: Int, interpolation: Int): Long
    external fun gaussianBlur(mat: Long, kw: Int, kh: Int, sigmaX: Double, sigmaY: Double): Long
    external fun medianBlur(mat: Long, kernelSize: Int): Long
    external fun threshold(mat: Long, thresh: Double, maxVal: Double, type: Int): Long
    external fun adaptiveThreshold(
        mat: Long, maxValue: Double, method: Int, type: Int, blockSize: Int, c: Double,
    ): Long

    external fun canny(mat: Long, threshold1: Double, threshold2: Double, apertureSize: Int, l2Gradient: Boolean): Long
    external fun sobel(mat: Long, dx: Int, dy: Int, kernelSize: Int): Long
    external fun laplacian(mat: Long, kernelSize: Int): Long

    external fun rectangle(
        mat: Long, x1: Int, y1: Int, x2: Int, y2: Int,
        v0: Double, v1: Double, v2: Double, v3: Double, thickness: Int,
    )

    external fun circle(
        mat: Long, centerX: Int, centerY: Int, radius: Int,
        v0: Double, v1: Double, v2: Double, v3: Double, thickness: Int,
    )

    external fun line(
        mat: Long, x1: Int, y1: Int, x2: Int, y2: Int,
        v0: Double, v1: Double, v2: Double, v3: Double, thickness: Int,
    )

    // imgcodecs

    external fun imread(path: String, flags: Int): Long
    external fun imwrite(path: String, mat: Long): Boolean
    external fun imencode(ext: String, mat: Long): ByteArray
    external fun imdecode(data: ByteArray, flags: Int): Long
}

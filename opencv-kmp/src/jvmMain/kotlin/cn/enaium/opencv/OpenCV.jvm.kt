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

// =========================================================================
// JVM (JNI-backed) implementation of the common API.
// Mat handles are jlong pointers to cv::Mat owned by the native side; every
// operation goes through [Jni], which forwards to the same cvk_ shim the
// native targets bind via cinterop.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()

/** Wraps a raw handle; throws with the native error text when it is 0. */
private fun jvmMat(ptr: Long, operation: String): Mat =
    if (ptr != 0L) JvmMat(ptr) else throw OpenCVException(operation, lastNativeError())

internal class JvmMat internal constructor(
    @Volatile private var ptr: Long,
) : Mat {

    internal fun check(): Long =
        if (ptr != 0L) ptr else throw IllegalStateException("OpenCV Mat is closed")

    override val rows: Int get() = Jni.matRows(check())
    override val cols: Int get() = Jni.matCols(check())
    override val type: Int get() = Jni.matType(check())
    override val channels: Int get() = Jni.matChannels(check())
    override val elemSize: Int get() = Jni.matElemSize(check()).toInt()
    override val total: Int get() = Jni.matTotal(check()).toInt()
    override val isEmpty: Boolean get() = Jni.matIsEmpty(check())

    override fun clone(): Mat = jvmMat(Jni.matClone(check()), "clone")

    override fun roi(rect: Rect): Mat =
        jvmMat(Jni.matRoi(check(), rect.x, rect.y, rect.width, rect.height), "roi")

    override fun convertTo(type: Int, alpha: Double, beta: Double): Mat =
        jvmMat(Jni.convertTo(check(), type, alpha, beta), "convertTo")

    override fun get(row: Int, col: Int): Double = at(row, col, 0)

    override fun set(row: Int, col: Int, value: Double) {
        put(row, col, 0, value)
    }

    override fun at(row: Int, col: Int, channel: Int): Double =
        Jni.matGet(check(), row, col, channel)

    override fun put(row: Int, col: Int, channel: Int, value: Double) {
        Jni.matSet(check(), row, col, channel, value)
    }

    override var pixels: ByteArray
        get() = Jni.matGetData(check())
        set(value) {
            require(value.size == elemSize * total) {
                "pixel buffer holds ${value.size} bytes but the matrix needs ${elemSize * total}"
            }
            Jni.matSetData(check(), value)
        }

    private fun rawOf(mat: Mat): Long =
        (mat as? JvmMat)?.check()
            ?: throw IllegalArgumentException("mat belongs to another platform backend")

    override fun plus(other: Mat): Mat = jvmMat(Jni.add(check(), rawOf(other)), "plus")
    override fun minus(other: Mat): Mat = jvmMat(Jni.subtract(check(), rawOf(other)), "minus")
    override fun times(other: Mat): Mat = jvmMat(Jni.multiply(check(), rawOf(other), 1.0), "times")
    override fun div(other: Mat): Mat = jvmMat(Jni.divide(check(), rawOf(other)), "div")

    override fun times(scale: Double): Mat = jvmMat(Jni.scaleAdd(check(), scale, 0.0), "times(scale)")

    override fun absDiff(other: Mat): Mat = jvmMat(Jni.absdiff(check(), rawOf(other)), "absDiff")

    override infix fun bitwiseAnd(other: Mat): Mat =
        jvmMat(Jni.bitwiseAnd(check(), rawOf(other)), "bitwiseAnd")

    override infix fun bitwiseOr(other: Mat): Mat =
        jvmMat(Jni.bitwiseOr(check(), rawOf(other)), "bitwiseOr")

    override infix fun bitwiseXor(other: Mat): Mat =
        jvmMat(Jni.bitwiseXor(check(), rawOf(other)), "bitwiseXor")

    override fun bitwiseNot(): Mat = jvmMat(Jni.bitwiseNot(check()), "bitwiseNot")

    override fun min(other: Mat): Mat = jvmMat(Jni.min(check(), rawOf(other)), "min")
    override fun max(other: Mat): Mat = jvmMat(Jni.max(check(), rawOf(other)), "max")

    override fun inRange(lower: Scalar, upper: Scalar): Mat = jvmMat(
        Jni.inRange(
            check(),
            lower.v0, lower.v1, lower.v2, lower.v3,
            upper.v0, upper.v1, upper.v2, upper.v3,
        ),
        "inRange",
    )

    override fun transpose(): Mat = jvmMat(Jni.transpose(check()), "transpose")

    override fun flip(flipCode: Int): Mat = jvmMat(Jni.flip(check(), flipCode), "flip")

    override val mean: Scalar
        get() {
            val m = Jni.mean(check())
            return Scalar(m[0], m[1], m[2], m[3])
        }

    override val sum: Scalar
        get() {
            val s = Jni.sum(check())
            return Scalar(s[0], s[1], s[2], s[3])
        }

    override fun meanStdDev(): Pair<Scalar, Scalar> {
        val out = Jni.meanStdDev(check())
        return Scalar(out[0], out[1], out[2], out[3]) to Scalar(out[4], out[5], out[6], out[7])
    }

    override fun minMaxLoc(): MinMaxLoc {
        val out = Jni.minMaxLoc(check())
        return MinMaxLoc(
            minVal = out[0], maxVal = out[1],
            minX = out[2].toInt(), minY = out[3].toInt(),
            maxX = out[4].toInt(), maxY = out[5].toInt(),
        )
    }

    override val nonZeroCount: Int get() = Jni.countNonZero(check())

    override fun cvtColor(code: Int): Mat = jvmMat(Jni.cvtColor(check(), code), "cvtColor")

    override fun resize(width: Int, height: Int, interpolation: Int): Mat =
        jvmMat(Jni.resize(check(), width, height, interpolation), "resize")

    override fun gaussianBlur(kernelWidth: Int, kernelHeight: Int, sigmaX: Double, sigmaY: Double): Mat =
        jvmMat(Jni.gaussianBlur(check(), kernelWidth, kernelHeight, sigmaX, sigmaY), "gaussianBlur")

    override fun medianBlur(kernelSize: Int): Mat =
        jvmMat(Jni.medianBlur(check(), kernelSize), "medianBlur")

    override fun threshold(thresh: Double, maxVal: Double, type: Int): Mat =
        jvmMat(Jni.threshold(check(), thresh, maxVal, type), "threshold")

    override fun adaptiveThreshold(
        maxValue: Double,
        method: Int,
        type: Int,
        blockSize: Int,
        c: Double,
    ): Mat = jvmMat(Jni.adaptiveThreshold(check(), maxValue, method, type, blockSize, c), "adaptiveThreshold")

    override fun canny(threshold1: Double, threshold2: Double, apertureSize: Int, l2Gradient: Boolean): Mat =
        jvmMat(Jni.canny(check(), threshold1, threshold2, apertureSize, l2Gradient), "canny")

    override fun sobel(dx: Int, dy: Int, kernelSize: Int): Mat =
        jvmMat(Jni.sobel(check(), dx, dy, kernelSize), "sobel")

    override fun laplacian(kernelSize: Int): Mat =
        jvmMat(Jni.laplacian(check(), kernelSize), "laplacian")

    override fun rectangle(from: Point, to: Point, color: Scalar, thickness: Int) {
        Jni.rectangle(
            check(), from.x, from.y, to.x, to.y,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    override fun circle(center: Point, radius: Int, color: Scalar, thickness: Int) {
        Jni.circle(
            check(), center.x, center.y, radius,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    override fun line(from: Point, to: Point, color: Scalar, thickness: Int) {
        Jni.line(
            check(), from.x, from.y, to.x, to.y,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    override fun close() {
        val handle = ptr
        if (handle == 0L) return
        ptr = 0L
        Jni.matRelease(handle)
    }
}

// =========================================================================
// actual declarations
// =========================================================================

actual val opencvVersion: String get() = Jni.version()

actual val opencvLastError: String? get() = lastNativeError()

actual fun mat(rows: Int, cols: Int, type: Int): Mat =
    jvmMat(Jni.matCreate(rows, cols, type), "mat")

actual fun mat(rows: Int, cols: Int, type: Int, fill: Scalar): Mat =
    jvmMat(
        Jni.matCreateFilled(rows, cols, type, fill.v0, fill.v1, fill.v2, fill.v3),
        "mat(fill)",
    )

actual fun zeros(rows: Int, cols: Int, type: Int): Mat =
    jvmMat(Jni.matZeros(rows, cols, type), "zeros")

actual fun ones(rows: Int, cols: Int, type: Int): Mat =
    jvmMat(Jni.matOnes(rows, cols, type), "ones")

actual fun eye(rows: Int, cols: Int, type: Int): Mat =
    jvmMat(Jni.matEye(rows, cols, type), "eye")

actual fun imread(path: String, flags: Int): Mat? =
    Jni.imread(path, flags).takeIf { it != 0L }?.let(::JvmMat)

actual fun imwrite(path: String, mat: Mat): Boolean =
    when (val target = mat as? JvmMat) {
        null -> throw IllegalArgumentException("mat belongs to another platform backend")
        else -> Jni.imwrite(path, target.check())
    }

actual fun imencode(ext: String, mat: Mat): ByteArray =
    when (val target = mat as? JvmMat) {
        null -> throw IllegalArgumentException("mat belongs to another platform backend")
        else -> Jni.imencode(normalizeImageExtension(ext), target.check())
    }

actual fun imdecode(data: ByteArray, flags: Int): Mat? =
    Jni.imdecode(data, flags).takeIf { it != 0L }?.let(::JvmMat)

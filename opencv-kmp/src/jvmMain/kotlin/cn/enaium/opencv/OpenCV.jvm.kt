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

/** Unpacks a fixed-order two-handle jlongArray output into a Mat pair. */
private fun matPair(handles: LongArray, operation: String): Pair<Mat, Mat> =
    if (handles.size == 2 && handles[0] != 0L && handles[1] != 0L) {
        JvmMat(handles[0]) to JvmMat(handles[1])
    } else {
        throw OpenCVException(operation, lastNativeError())
    }

/** Raw handle of any JVM-backed Mat argument. */
private fun handleOf(mat: Mat): Long =
    (mat as? JvmMat)?.check()
        ?: throw IllegalArgumentException("mat belongs to another platform backend")

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

    // ---- core: shape / algebra

    override fun reshape(channels: Int, rows: Int): Mat =
        jvmMat(Jni.matReshape(check(), channels, rows), "reshape")

    override fun rowRange(start: Int, end: Int): Mat =
        jvmMat(Jni.matRowRange(check(), start, end), "rowRange")

    override fun colRange(start: Int, end: Int): Mat =
        jvmMat(Jni.matColRange(check(), start, end), "colRange")

    override fun diag(d: Int): Mat = jvmMat(Jni.matDiag(check(), d), "diag")

    override fun setIdentity(scale: Double) {
        Jni.matSetIdentity(check(), scale)
    }

    override infix fun dot(other: Mat): Double = Jni.matDot(check(), rawOf(other))

    override fun inv(method: Int): Mat? =
        Jni.matInv(check(), method).takeIf { it != 0L }?.let(::JvmMat)

    override val determinant: Double get() = Jni.matDeterminant(check())

    override val trace: Scalar
        get() {
            val t = Jni.matTrace(check())
            return Scalar(t[0], t[1], t[2], t[3])
        }

    // ---- core: array operations

    override fun split(): List<Mat> =
        Jni.splitChannels(check()).map { handle -> jvmMat(handle, "split") }

    override fun normalize(
        alpha: Double,
        beta: Double,
        normType: Int,
        dtype: Int,
    ): Mat = jvmMat(Jni.normalize(check(), alpha, beta, normType, dtype), "normalize")

    override fun lut(lut: Mat): Mat = jvmMat(Jni.lut(check(), rawOf(lut)), "lut")

    override fun rotate(code: Int): Mat = jvmMat(Jni.rotate(check(), code), "rotate")

    override fun copyMakeBorder(
        top: Int,
        bottom: Int,
        left: Int,
        right: Int,
        borderType: Int,
        value: Scalar,
    ): Mat = jvmMat(
        Jni.copyMakeBorder(
            check(), top, bottom, left, right, borderType,
            value.v0, value.v1, value.v2, value.v3,
        ),
        "copyMakeBorder",
    )

    override fun addWeighted(alpha: Double, other: Mat, beta: Double, gamma: Double): Mat =
        jvmMat(Jni.addWeighted(check(), alpha, rawOf(other), beta, gamma), "addWeighted")

    override fun convertScaleAbs(alpha: Double, beta: Double): Mat =
        jvmMat(Jni.convertScaleAbs(check(), alpha, beta), "convertScaleAbs")

    override fun compare(other: Mat, op: Int): Mat =
        jvmMat(Jni.compare(check(), rawOf(other), op), "compare")

    override fun solve(b: Mat, flags: Int): Mat? =
        Jni.solve(check(), rawOf(b), flags).takeIf { it != 0L }?.let(::JvmMat)

    override fun repeat(nx: Int, ny: Int): Mat =
        jvmMat(Jni.repeat(check(), nx, ny), "repeat")

    override fun transform(m: Mat): Mat =
        jvmMat(Jni.transform(check(), rawOf(m)), "transform")

    override fun perspectiveTransform(m: Mat): Mat =
        jvmMat(Jni.perspectiveTransform(check(), rawOf(m)), "perspectiveTransform")

    override fun pow(power: Double): Mat = jvmMat(Jni.pow(check(), power), "pow")
    override fun sqrt(): Mat = jvmMat(Jni.sqrt(check()), "sqrt")
    override fun exp(): Mat = jvmMat(Jni.exp(check()), "exp")
    override fun log(): Mat = jvmMat(Jni.log(check()), "log")

    override fun magnitude(y: Mat): Mat =
        jvmMat(Jni.magnitude(check(), rawOf(y)), "magnitude")

    override fun phase(y: Mat, angleInDegrees: Boolean): Mat =
        jvmMat(Jni.phase(check(), rawOf(y), angleInDegrees), "phase")

    override fun cartToPolar(y: Mat, angleInDegrees: Boolean): Pair<Mat, Mat> =
        matPair(Jni.cartToPolar(check(), rawOf(y), angleInDegrees), "cartToPolar")

    override fun polarToCart(angle: Mat, angleInDegrees: Boolean): Pair<Mat, Mat> =
        matPair(Jni.polarToCart(check(), rawOf(angle), angleInDegrees), "polarToCart")

    override fun patchNaNs(value: Double) {
        Jni.patchNaNs(check(), value)
    }

    override val hasNonZero: Boolean get() = Jni.hasNonZero(check())

    override fun findNonZero(): Mat = jvmMat(Jni.findNonZero(check()), "findNonZero")

    override fun sort(flags: Int): Mat = jvmMat(Jni.sort(check(), flags), "sort")

    override fun sortIdx(flags: Int): Mat = jvmMat(Jni.sortIdx(check(), flags), "sortIdx")

    override fun reduce(dim: Int, rtype: Int, dtype: Int): Mat =
        jvmMat(Jni.reduce(check(), dim, rtype, dtype), "reduce")

    override fun reduceArgMax(dim: Int): Mat =
        jvmMat(Jni.reduceArgMax(check(), dim), "reduceArgMax")

    override fun reduceArgMin(dim: Int): Mat =
        jvmMat(Jni.reduceArgMin(check(), dim), "reduceArgMin")

    override fun extractChannel(coi: Int): Mat =
        jvmMat(Jni.extractChannel(check(), coi), "extractChannel")

    override fun insertChannel(channel: Mat, coi: Int) {
        Jni.insertChannel(rawOf(channel), check(), coi)
    }

    override fun randu(low: Scalar, high: Scalar) {
        Jni.randu(
            check(),
            low.v0, low.v1, low.v2, low.v3,
            high.v0, high.v1, high.v2, high.v3,
        )
    }

    override fun randn(mean: Scalar, stddev: Scalar) {
        Jni.randn(
            check(),
            mean.v0, mean.v1, mean.v2, mean.v3,
            stddev.v0, stddev.v1, stddev.v2, stddev.v3,
        )
    }

    override fun psnr(other: Mat, r: Double): Double = Jni.psnr(check(), rawOf(other), r)

    override fun dft(flags: Int): Mat = jvmMat(Jni.dft(check(), flags), "dft")
    override fun idft(flags: Int): Mat = jvmMat(Jni.idft(check(), flags), "idft")
    override fun dct(flags: Int): Mat = jvmMat(Jni.dct(check(), flags), "dct")
    override fun idct(flags: Int): Mat = jvmMat(Jni.idct(check(), flags), "idct")

    override fun mulSpectrums(other: Mat, conjugate: Boolean, dftRows: Boolean): Mat =
        jvmMat(Jni.mulSpectrums(check(), rawOf(other), conjugate, dftRows), "mulSpectrums")

    override fun divSpectrums(other: Mat, conjugate: Boolean): Mat =
        jvmMat(Jni.divSpectrums(check(), rawOf(other), conjugate), "divSpectrums")

    override fun gemm(other: Mat, alpha: Double, c: Mat?, gamma: Double): Mat =
        jvmMat(
            Jni.gemm(check(), rawOf(other), alpha, c?.let { rawOf(it) } ?: 0L, gamma),
            "gemm",
        )

    override fun eigen(): Pair<Mat, Mat> = matPair(Jni.eigen(check()), "eigen")

    // ---- imgproc: filters

    override fun blur(kernelWidth: Int, kernelHeight: Int): Mat =
        jvmMat(Jni.blur(check(), kernelWidth, kernelHeight), "blur")

    override fun boxFilter(
        kernelWidth: Int,
        kernelHeight: Int,
        ddepth: Int,
        normalize: Boolean,
    ): Mat = jvmMat(Jni.boxFilter(check(), ddepth, kernelWidth, kernelHeight, normalize), "boxFilter")

    override fun sqrBoxFilter(kernelWidth: Int, kernelHeight: Int, ddepth: Int): Mat =
        jvmMat(Jni.sqrBoxFilter(check(), ddepth, kernelWidth, kernelHeight), "sqrBoxFilter")

    override fun bilateralFilter(d: Int, sigmaColor: Double, sigmaSpace: Double): Mat =
        jvmMat(Jni.bilateralFilter(check(), d, sigmaColor, sigmaSpace), "bilateralFilter")

    override fun stackBlur(kernelSize: Int): Mat =
        jvmMat(Jni.stackBlur(check(), kernelSize), "stackBlur")

    override fun erode(kernel: Mat?, iterations: Int): Mat =
        jvmMat(Jni.erode(check(), kernel?.let { rawOf(it) } ?: 0L, iterations), "erode")

    override fun dilate(kernel: Mat?, iterations: Int): Mat =
        jvmMat(Jni.dilate(check(), kernel?.let { rawOf(it) } ?: 0L, iterations), "dilate")

    override fun morphologyEx(op: Int, kernel: Mat?, iterations: Int): Mat =
        jvmMat(
            Jni.morphologyEx(check(), op, kernel?.let { rawOf(it) } ?: 0L, iterations),
            "morphologyEx",
        )

    override fun filter2D(kernel: Mat, ddepth: Int, delta: Double): Mat =
        jvmMat(Jni.filter2D(check(), rawOf(kernel), ddepth, delta), "filter2D")

    override fun pyrDown(): Mat = jvmMat(Jni.pyrDown(check()), "pyrDown")
    override fun pyrUp(): Mat = jvmMat(Jni.pyrUp(check()), "pyrUp")

    // ---- imgproc: geometry

    override fun warpAffine(m: Mat, width: Int, height: Int, flags: Int): Mat =
        jvmMat(Jni.warpAffine(check(), rawOf(m), width, height, flags), "warpAffine")

    override fun warpPerspective(m: Mat, width: Int, height: Int, flags: Int): Mat =
        jvmMat(Jni.warpPerspective(check(), rawOf(m), width, height, flags), "warpPerspective")

    override fun remap(map1: Mat, map2: Mat, interpolation: Int): Mat =
        jvmMat(Jni.remap(check(), rawOf(map1), rawOf(map2), interpolation), "remap")

    override fun warpPolar(
        radius: Int,
        centerX: Double,
        centerY: Double,
        maxRadius: Double,
        flags: Int,
    ): Mat = jvmMat(Jni.warpPolar(check(), radius, centerX, centerY, maxRadius, flags), "warpPolar")

    override fun undistort(cameraMatrix: Mat, distCoeffs: Mat): Mat =
        jvmMat(Jni.undistort(check(), rawOf(cameraMatrix), rawOf(distCoeffs)), "undistort")

    override fun getRectSubPix(width: Int, height: Int, centerX: Double, centerY: Double): Mat =
        jvmMat(Jni.getRectSubPix(check(), width, height, centerX, centerY), "getRectSubPix")

    // ---- imgproc: color / histogram

    override fun demosaicing(code: Int): Mat =
        jvmMat(Jni.demosaicing(check(), code), "demosaicing")

    override fun applyColorMap(colormap: Int): Mat =
        jvmMat(Jni.applyColormap(check(), colormap), "applyColorMap")

    override fun applyColorMap(userColor: Mat): Mat =
        jvmMat(Jni.applyColormapUser(check(), rawOf(userColor)), "applyColorMap(user)")

    override fun calcHist(
        channel: Int,
        histSize: Int,
        minValue: Float,
        maxValue: Float,
    ): Mat = jvmMat(Jni.calcHist(check(), channel, histSize, minValue, maxValue), "calcHist")

    override fun calcBackProject(
        hist: Mat,
        channel: Int,
        minValue: Float,
        maxValue: Float,
    ): Mat = jvmMat(
        Jni.calcBackProject(check(), channel, rawOf(hist), minValue, maxValue),
        "calcBackProject",
    )

    override fun equalizeHist(): Mat = jvmMat(Jni.equalizeHist(check()), "equalizeHist")

    override fun matchShapes(other: Mat, method: Int): Double =
        Jni.matchShapes(check(), rawOf(other), method)

    override fun moments(binaryImage: Boolean): Moments {
        val m = Jni.moments(check(), binaryImage)
        return Moments(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9])
    }

    // ---- imgproc: segmentation / features

    override fun floodFill(
        seedX: Int,
        seedY: Int,
        newValue: Scalar,
        loDiff: Scalar,
        upDiff: Scalar,
        flags: Int,
    ): Int = Jni.floodFill(
        check(), seedX, seedY,
        newValue.v0, newValue.v1, newValue.v2, newValue.v3,
        loDiff.v0, loDiff.v1, loDiff.v2, loDiff.v3,
        upDiff.v0, upDiff.v1, upDiff.v2, upDiff.v3,
        flags,
    )

    override fun watershed(markers: Mat) {
        Jni.watershed(check(), rawOf(markers))
    }

    override fun matchTemplate(templ: Mat, method: Int): Mat =
        jvmMat(Jni.matchTemplate(check(), rawOf(templ), method), "matchTemplate")

    override fun cornerHarris(blockSize: Int, ksize: Int, k: Double): Mat =
        jvmMat(Jni.cornerHarris(check(), blockSize, ksize, k), "cornerHarris")

    override fun cornerMinEigenVal(blockSize: Int, ksize: Int): Mat =
        jvmMat(Jni.cornerMinEigenVal(check(), blockSize, ksize), "cornerMinEigenVal")

    override fun goodFeaturesToTrack(
        maxCorners: Int,
        qualityLevel: Double,
        minDistance: Double,
        blockSize: Int,
        useHarrisDetector: Boolean,
        k: Double,
    ): Mat = jvmMat(
        Jni.goodFeaturesToTrack(
            check(), maxCorners, qualityLevel, minDistance,
            blockSize, useHarrisDetector, k,
        ),
        "goodFeaturesToTrack",
    )

    override fun distanceTransform(distanceType: Int, maskSize: Int): Mat =
        jvmMat(Jni.distanceTransform(check(), distanceType, maskSize), "distanceTransform")

    override fun integral(sdepth: Int): Mat = jvmMat(Jni.integral(check(), sdepth), "integral")

    override fun connectedComponents(connectivity: Int, ltype: Int): Pair<Int, Mat> {
        val out = Jni.connectedComponents(check(), connectivity, ltype)
        return out[0].toInt() to jvmMat(out[1], "connectedComponents")
    }

    override fun connectedComponentsWithStats(connectivity: Int, ltype: Int): Components {
        val out = Jni.connectedComponentsWithStats(check(), connectivity, ltype)
        return Components(
            count = out[0].toInt(),
            labels = jvmMat(out[1], "connectedComponentsWithStats"),
            stats = jvmMat(out[2], "connectedComponentsWithStats"),
            centroids = jvmMat(out[3], "connectedComponentsWithStats"),
        )
    }

    override fun pyrMeanShiftFiltering(sp: Double, sr: Double, maxLevel: Int): Mat =
        jvmMat(Jni.pyrMeanShiftFiltering(check(), sp, sr, maxLevel), "pyrMeanShiftFiltering")

    override fun thresholdWithMask(
        thresh: Double,
        maxVal: Double,
        type: Int,
        mask: Mat?,
    ): Pair<Double, Mat> {
        val cloned = rawOf(clone())
        val maskHandle = mask?.let { rawOf(it) } ?: 0L
        val computed =
            Jni.thresholdWithMask(check(), maskHandle, cloned, thresh, maxVal, type)
        return computed to jvmMat(cloned, "thresholdWithMask")
    }

    // ---- imgproc: hough / accumulators

    override fun houghLines(rho: Double, theta: Double, threshold: Int, srn: Double, stn: Double): Mat =
        jvmMat(Jni.houghLines(check(), rho, theta, threshold, srn, stn), "houghLines")

    override fun houghLinesP(rho: Double, theta: Double, threshold: Int,
                             minLineLength: Double, maxLineGap: Double): Mat =
        jvmMat(
            Jni.houghLinesP(check(), rho, theta, threshold, minLineLength, maxLineGap),
            "houghLinesP",
        )

    override fun houghCircles(
        dp: Double,
        minDist: Double,
        param1: Double,
        param2: Double,
        minRadius: Int,
        maxRadius: Int,
        method: Int,
    ): Mat = jvmMat(
        Jni.houghCircles(check(), method, dp, minDist, param1, param2, minRadius, maxRadius),
        "houghCircles",
    )

    override fun accumulate(src: Mat) {
        Jni.accumulate(rawOf(src), check())
    }

    override fun accumulateSquare(src: Mat) {
        Jni.accumulateSquare(rawOf(src), check())
    }

    override fun accumulateProduct(a: Mat, b: Mat) {
        Jni.accumulateProduct(rawOf(a), rawOf(b), check())
    }

    override fun accumulateWeighted(src: Mat, alpha: Double) {
        Jni.accumulateWeighted(rawOf(src), check(), alpha)
    }

    // ---- imgproc: contours

    override fun findContours(mode: Int, method: Int): List<List<Point>> =
        ContourCodec.decode(Jni.findContours(check(), mode, method))

    override fun drawContours(
        contours: List<List<Point>>,
        color: Scalar,
        contourIndex: Int,
        thickness: Int,
    ) {
        Jni.drawContours(
            check(), ContourCodec.encode(contours), contourIndex,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    // ---- imgproc: drawing (in-place)

    override fun arrowedLine(from: Point, to: Point, color: Scalar, thickness: Int) {
        Jni.arrowedLine(
            check(), from.x, from.y, to.x, to.y,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    override fun drawMarker(
        pos: Point,
        color: Scalar,
        markerType: Int,
        size: Int,
        thickness: Int,
    ) {
        Jni.drawMarker(
            check(), pos.x, pos.y, markerType, size,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    override fun ellipse(
        center: Point,
        axes: Size,
        angle: Double,
        startAngle: Double,
        endAngle: Double,
        color: Scalar,
        thickness: Int,
    ) {
        Jni.ellipse(
            check(), center.x, center.y, axes.width, axes.height,
            angle, startAngle, endAngle,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    override fun fillPoly(polygons: List<List<Point>>, color: Scalar, thickness: Int) {
        Jni.fillPoly(
            check(), ContourCodec.encode(polygons),
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    override fun polylines(polylines: List<List<Point>>, closed: Boolean, color: Scalar, thickness: Int) {
        Jni.polylines(
            check(), ContourCodec.encode(polylines), closed,
            color.v0, color.v1, color.v2, color.v3, thickness,
        )
    }

    // ---- org.opencv.core.Mat parity ------------------------------------------

    override fun adjustROI(dtop: Int, dbottom: Int, dleft: Int, dright: Int): Mat =
        jvmMat(Jni.matAdjustROI(check(), dtop, dbottom, dleft, dright), "adjustROI")

    override fun locateROI(): Pair<Point, Size> {
        val out = Jni.matLocateROI(check())
        return Point(out[0], out[1]) to Size(out[2], out[3])
    }

    override val isContinuous: Boolean get() = Jni.matIsContinuous(check())

    override val isSubmatrix: Boolean get() = Jni.matIsSubmatrix(check())

    override infix fun cross(other: Mat): Mat =
        jvmMat(Jni.matCross(check(), rawOf(other)), "cross")

    override fun dump(): String = Jni.matDump(check())

    override fun put(row: Int, col: Int, values: DoubleArray): Int =
        Jni.matPutValues(check(), row, col, values)

    override fun get(row: Int, col: Int, values: DoubleArray): Int =
        Jni.matGetValues(check(), row, col, values)

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

// =========================================================================
// CLAHE
// =========================================================================

/** JNI-backed [CLAHE] wrapping a `cv::Ptr<cv::CLAHE>` handle. */
internal class JvmClahe(private var handle: Long) : CLAHE {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("CLAHE is closed")

    override fun apply(src: Mat): Mat =
        jvmMat(Jni.claheApply(check(), handleOf(src)), "claheApply")

    override fun setClipLimit(clipLimit: Double) {
        Jni.claheSetClipLimit(check(), clipLimit)
    }

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        Jni.claheRelease(current)
    }
}

// =========================================================================
// kernel / transform / info factories
// =========================================================================

actual fun getStructuringElement(shape: Int, width: Int, height: Int): Mat =
    jvmMat(Jni.getStructuringElement(shape, width, height), "getStructuringElement")

actual fun getGaussianKernel(ksize: Int, sigma: Double): Mat =
    jvmMat(Jni.getGaussianKernel(ksize, sigma), "getGaussianKernel")

actual fun getAffineTransform(src: List<Point>, dst: List<Point>): Mat {
    require(src.size == 3 && dst.size == 3) {
        "exactly three source and destination points are required"
    }
    return jvmMat(
        Jni.getAffineTransform(
            src[0].x.toDouble(), src[0].y.toDouble(),
            src[1].x.toDouble(), src[1].y.toDouble(),
            src[2].x.toDouble(), src[2].y.toDouble(),
            dst[0].x.toDouble(), dst[0].y.toDouble(),
            dst[1].x.toDouble(), dst[1].y.toDouble(),
            dst[2].x.toDouble(), dst[2].y.toDouble(),
        ),
        "getAffineTransform",
    )
}

actual fun invertAffineTransform(transform: Mat): Mat =
    jvmMat(Jni.invertAffineTransform(handleOf(transform)), "invertAffineTransform")

actual fun getPerspectiveTransform(src: List<Point>, dst: List<Point>): Mat {
    require(src.size == 4 && dst.size == 4) {
        "exactly four source and destination points are required"
    }
    return jvmMat(
        Jni.getPerspectiveTransform(
            src[0].x.toDouble(), src[0].y.toDouble(),
            src[1].x.toDouble(), src[1].y.toDouble(),
            src[2].x.toDouble(), src[2].y.toDouble(),
            src[3].x.toDouble(), src[3].y.toDouble(),
            dst[0].x.toDouble(), dst[0].y.toDouble(),
            dst[1].x.toDouble(), dst[1].y.toDouble(),
            dst[2].x.toDouble(), dst[2].y.toDouble(),
            dst[3].x.toDouble(), dst[3].y.toDouble(),
        ),
        "getPerspectiveTransform",
    )
}

actual fun getRotationMatrix2D(center: Point, angle: Double, scale: Double): Mat =
    jvmMat(
        Jni.getRotationMatrix2D(center.x.toDouble(), center.y.toDouble(), angle, scale),
        "getRotationMatrix2D",
    )

actual fun hanningWindow(width: Int, height: Int, type: Int): Mat =
    jvmMat(Jni.createHanningWindow(width, height, type), "hanningWindow")

actual fun merge(channels: List<Mat>): Mat =
    jvmMat(Jni.mergeChannels(channels.map { handleOf(it) }.toLongArray()), "merge")

actual val opencvNumThreads: Int get() = Jni.numThreads()

actual fun setNumThreads(count: Int) {
    Jni.setNumThreads(count)
}

actual val opencvBuildInformation: String get() = Jni.buildInformation()

actual fun setRNGSeed(seed: Long) {
    Jni.setRngSeed(seed)
}

// =========================================================================
// contour geometry
// =========================================================================

actual fun approxPolyDP(contour: List<Point>, epsilon: Double, closed: Boolean): List<Point> =
    contourBuffer(contour) { data ->
        ContourCodec.decode(Jni.approxPolyDP(data, epsilon, closed)).firstOrNull().orEmpty()
    }

actual fun minAreaRect(contour: List<Point>): RotatedRect =
    contourBuffer(contour) { data ->
        val out = Jni.minAreaRect(data)
        RotatedRect(out[0], out[1], out[2], out[3], out[4])
    }

actual fun minEnclosingCircle(contour: List<Point>): Circle =
    contourBuffer(contour) { data ->
        val out = Jni.minEnclosingCircle(data)
        Circle(out[0], out[1], out[2])
    }

internal actual fun contourAreaNative(data: ByteArray): Double =
    Jni.contourAreaBytes(data)

internal actual fun arcLengthNative(data: ByteArray, closed: Boolean): Double =
    Jni.arcLengthBytes(data, closed)

internal actual fun contourRect(contour: List<Point>): Rect {
    val out = contourBuffer(contour) { data -> Jni.boundingRect(data) }
    return Rect(out[0], out[1], out[2], out[3])
}

// =========================================================================
// CLAHE factory
// =========================================================================

actual fun createCLAHE(clipLimit: Double, tileGridSize: Size): CLAHE =
    JvmClahe(Jni.claheCreate(clipLimit, tileGridSize.width, tileGridSize.height))

// =========================================================================
// imgcodecs additions / DFT helpers
// =========================================================================

actual fun imcount(path: String): Int = Jni.imcount(path)

actual fun haveImageReader(path: String): Boolean = Jni.haveImageReader(path)

actual fun haveImageWriter(path: String): Boolean = Jni.haveImageWriter(path)

actual fun imencodeParams(ext: String, mat: Mat, params: List<Int>): ByteArray =
    Jni.imencodeParams(normalizeImageExtension(ext), handleOf(mat), params.toIntArray())

actual fun imwriteParams(path: String, mat: Mat, params: List<Int>): Boolean =
    Jni.imwriteParams(path, handleOf(mat), params.toIntArray())

actual fun getOptimalDftSize(size: Int): Int = Jni.getOptimalDftSize(size)

// =========================================================================
// org.opencv.core / imgproc parity
// =========================================================================

actual fun kmeans(
    data: Mat,
    k: Int,
    criteria: TermCriteria,
    attempts: Int,
    flags: Int,
): KmeansResult {
    zeros(data.rows, 1, CV_32S).use { labels ->
        val compactness = DoubleArray(1)
        val centersHandle = Jni.kmeans(
            handleOf(data), k, criteria.type, criteria.maxCount, criteria.epsilon,
            attempts, flags, handleOf(labels), compactness,
        )
        return KmeansResult(compactness[0], labels.clone(), jvmMat(centersHandle, "kmeans"))
    }
}

actual fun svDecomp(src: Mat, flags: Int): Svd {
    val parts = Jni.svdDecomp(handleOf(src), flags)
    return Svd(jvmMat(parts[0], "svd.w"), jvmMat(parts[1], "svd.u"), jvmMat(parts[2], "svd.vt"))
}

actual fun svdBackSubst(w: Mat, u: Mat, vt: Mat, b: Mat): Mat =
    jvmMat(Jni.svdBackSubst(handleOf(w), handleOf(u), handleOf(vt), handleOf(b)), "svdBackSubst")

actual fun pcaCompute(data: Mat, maxComponents: Int): Pca {
    val parts = Jni.pcaCompute(handleOf(data), maxComponents)
    return Pca(jvmMat(parts[0], "pca.mean"), jvmMat(parts[1], "pca.vectors"))
}

actual fun pcaComputeVariance(data: Mat, retainedVariance: Double): Pca {
    val parts = Jni.pcaComputeVariance(handleOf(data), retainedVariance)
    return Pca(jvmMat(parts[0], "pca.mean"), jvmMat(parts[1], "pca.vectors"))
}

actual fun pcaProject(data: Mat, mean: Mat, eigenvectors: Mat): Mat =
    jvmMat(Jni.pcaProject(handleOf(data), handleOf(mean), handleOf(eigenvectors)), "pcaProject")

actual fun pcaBackProject(data: Mat, mean: Mat, eigenvectors: Mat): Mat =
    jvmMat(Jni.pcaBackProject(handleOf(data), handleOf(mean), handleOf(eigenvectors)), "pcaBackProject")

actual fun mahalanobis(v1: Mat, v2: Mat, icovar: Mat): Double =
    Jni.mahalanobis(handleOf(v1), handleOf(v2), handleOf(icovar))

actual fun cornerSubPix(
    image: Mat,
    corners: List<Point>,
    winSize: Size,
    zeroZone: Size,
    criteria: TermCriteria,
): List<Point> {
    require(corners.isNotEmpty()) { "cornerSubPix needs at least one point" }
    val refined = Jni.cornerSubPixBytes(
        handleOf(image), ContourCodec.encode(listOf(corners)),
        winSize.width, winSize.height, zeroZone.width, zeroZone.height,
        criteria.type, criteria.maxCount, criteria.epsilon,
    )
    return ContourCodec.decode(refined).firstOrNull() ?: corners
}

actual fun emd(signature1: Mat, signature2: Mat, distType: Int): Double =
    Jni.emd(handleOf(signature1), handleOf(signature2), distType)

actual fun grabCut(
    image: Mat,
    mask: Mat,
    rect: Rect?,
    iterations: Int,
    mode: Int,
) {
        mat(1, 65, MatType.CV_64FC1).use { bgdModel ->
            mat(1, 65, MatType.CV_64FC1).use { fgdModel ->
            val r = rect ?: Rect(0, 0, 0, 0)
            Jni.grabCut(
                handleOf(image), handleOf(mask),
                r.x, r.y, r.width, r.height,
                handleOf(bgdModel), handleOf(fgdModel),
                iterations, mode,
            )
        }
    }
}

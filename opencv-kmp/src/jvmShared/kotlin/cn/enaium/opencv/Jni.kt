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
    external fun addScalar(mat: Long, v0: Double, v1: Double, v2: Double, v3: Double): Long
    external fun subtractScalar(mat: Long, v0: Double, v1: Double, v2: Double, v3: Double): Long
    external fun multiplyScalar(mat: Long, v0: Double, v1: Double, v2: Double, v3: Double): Long
    external fun divideScalar(mat: Long, v0: Double, v1: Double, v2: Double, v3: Double): Long
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

    // ---- core: Mat shape / algebra

    external fun matReshape(mat: Long, channels: Int, rows: Int): Long
    external fun matRowRange(mat: Long, start: Int, end: Int): Long
    external fun matColRange(mat: Long, start: Int, end: Int): Long
    external fun matDiag(mat: Long, d: Int): Long
    external fun matSetIdentity(mat: Long, scale: Double)
    external fun matDot(a: Long, b: Long): Double
    external fun matInv(mat: Long, method: Int): Long
    external fun matDeterminant(mat: Long): Double
    external fun matTrace(mat: Long): DoubleArray

    // ---- core: array operations

    external fun splitChannels(src: Long): LongArray
    external fun mergeChannels(channels: LongArray): Long
    external fun hconcat(a: Long, b: Long): Long
    external fun vconcat(a: Long, b: Long): Long
    external fun norm(src: Long, normType: Int): Double
    external fun normDiff(a: Long, b: Long, normType: Int): Double
    external fun normalize(
        src: Long,
        alpha: Double,
        beta: Double,
        normType: Int,
        dtype: Int,
    ): Long

    external fun lut(src: Long, lut: Long): Long
    external fun rotate(src: Long, code: Int): Long
    external fun copyMakeBorder(
        src: Long,
        top: Int,
        bottom: Int,
        left: Int,
        right: Int,
        borderType: Int,
        v0: Double, v1: Double, v2: Double, v3: Double,
    ): Long

    external fun addWeighted(a: Long, alpha: Double, b: Long, beta: Double, gamma: Double): Long
    external fun convertScaleAbs(src: Long, alpha: Double, beta: Double): Long
    external fun compare(a: Long, b: Long, op: Int): Long
    external fun solve(a: Long, b: Long, flags: Int): Long
    external fun repeat(src: Long, nx: Int, ny: Int): Long
    external fun transform(src: Long, m: Long): Long
    external fun perspectiveTransform(src: Long, m: Long): Long
    external fun pow(src: Long, power: Double): Long
    external fun sqrt(src: Long): Long
    external fun exp(src: Long): Long
    external fun log(src: Long): Long
    external fun magnitude(x: Long, y: Long): Long
    external fun phase(x: Long, y: Long, angleInDegrees: Boolean): Long
    external fun cartToPolar(x: Long, y: Long, angleInDegrees: Boolean): LongArray
    external fun polarToCart(magnitude: Long, angle: Long, angleInDegrees: Boolean): LongArray
    external fun patchNaNs(mat: Long, value: Double)
    external fun findNonZero(src: Long): Long
    external fun hasNonZero(src: Long): Boolean
    external fun sort(src: Long, flags: Int): Long
    external fun sortIdx(src: Long, flags: Int): Long
    external fun reduce(src: Long, dim: Int, rtype: Int, dtype: Int): Long
    external fun reduceArgMax(src: Long, dim: Int): Long
    external fun reduceArgMin(src: Long, dim: Int): Long
    external fun extractChannel(src: Long, coi: Int): Long
    external fun insertChannel(src: Long, dst: Long, coi: Int)
    external fun randu(dst: Long, a0: Double, a1: Double, a2: Double, a3: Double,
                       b0: Double, b1: Double, b2: Double, b3: Double)
    external fun randn(dst: Long, a0: Double, a1: Double, a2: Double, a3: Double,
                       b0: Double, b1: Double, b2: Double, b3: Double)
    external fun psnr(a: Long, b: Long, r: Double): Double
    external fun dft(src: Long, flags: Int): Long
    external fun idft(src: Long, flags: Int): Long
    external fun dct(src: Long, flags: Int): Long
    external fun idct(src: Long, flags: Int): Long
    external fun getOptimalDftSize(rowsize: Int): Int
    external fun mulSpectrums(a: Long, b: Long, conjFlag: Boolean, dftRows: Boolean): Long
    external fun divSpectrums(a: Long, b: Long, conjFlag: Boolean): Long
    external fun gemm(a: Long, b: Long, alpha: Double, c: Long, gamma: Double): Long
    external fun eigen(src: Long): LongArray
    external fun numThreads(): Int
    external fun setNumThreads(count: Int)
    external fun buildInformation(): String

    // ---- imgproc: filters

    external fun blur(src: Long, kernelWidth: Int, kernelHeight: Int): Long
    external fun boxFilter(
        src: Long,
        ddepth: Int,
        kernelWidth: Int,
        kernelHeight: Int,
        normalize: Boolean,
    ): Long

    external fun sqrBoxFilter(src: Long, ddepth: Int, kernelWidth: Int, kernelHeight: Int): Long
    external fun bilateralFilter(src: Long, d: Int, sigmaColor: Double, sigmaSpace: Double): Long
    external fun stackBlur(src: Long, kernelSize: Int): Long
    external fun erode(src: Long, kernel: Long, iterations: Int): Long
    external fun dilate(src: Long, kernel: Long, iterations: Int): Long
    external fun morphologyEx(src: Long, op: Int, kernel: Long, iterations: Int): Long
    external fun getStructuringElement(shape: Int, width: Int, height: Int): Long
    external fun getGaussianKernel(ksize: Int, sigma: Double): Long
    external fun filter2D(src: Long, kernel: Long, ddepth: Int, delta: Double): Long
    external fun pyrDown(src: Long): Long
    external fun pyrUp(src: Long): Long

    // ---- imgproc: geometry / warps

    external fun warpAffine(src: Long, m: Long, width: Int, height: Int, flags: Int): Long
    external fun warpPerspective(src: Long, m: Long, width: Int, height: Int, flags: Int): Long
    external fun remap(src: Long, map1: Long, map2: Long, interpolation: Int): Long
    external fun warpPolar(
        src: Long,
        radius: Int,
        centerX: Double,
        centerY: Double,
        maxRadius: Double,
        flags: Int,
    ): Long

    external fun getAffineTransform(
        sx0: Double, sy0: Double, sx1: Double, sy1: Double, sx2: Double, sy2: Double,
        dx0: Double, dy0: Double, dx1: Double, dy1: Double, dx2: Double, dy2: Double,
    ): Long

    external fun invertAffineTransform(m: Long): Long
    external fun getPerspectiveTransform(
        sx0: Double, sy0: Double, sx1: Double, sy1: Double,
        sx2: Double, sy2: Double, sx3: Double, sy3: Double,
        dx0: Double, dy0: Double, dx1: Double, dy1: Double,
        dx2: Double, dy2: Double, dx3: Double, dy3: Double,
    ): Long

    external fun getRotationMatrix2D(cx: Double, cy: Double, angle: Double, scale: Double): Long
    external fun getRectSubPix(src: Long, width: Int, height: Int, cx: Double, cy: Double): Long
    external fun undistort(src: Long, cameraMatrix: Long, distCoeffs: Long): Long

    // ---- imgproc: color / histogram

    external fun demosaicing(src: Long, code: Int): Long
    external fun applyColormap(src: Long, colormap: Int): Long
    external fun applyColormapUser(src: Long, userColor: Long): Long
    external fun calcHist(src: Long, channel: Int, histSize: Int, minValue: Float, maxValue: Float): Long
    external fun calcBackProject(
        src: Long,
        channel: Int,
        hist: Long,
        minValue: Float,
        maxValue: Float,
    ): Long

    external fun compareHist(h1: Long, h2: Long, method: Int): Double
    external fun equalizeHist(src: Long): Long

    // ---- imgproc: segmentation / contours / features

    external fun floodFill(
        image: Long,
        seedX: Int,
        seedY: Int,
        n0: Double, n1: Double, n2: Double, n3: Double,
        lo0: Double, lo1: Double, lo2: Double, lo3: Double,
        up0: Double, up1: Double, up2: Double, up3: Double,
        flags: Int,
    ): Int

    external fun watershed(image: Long, markers: Long)
    external fun findContours(src: Long, mode: Int, method: Int): ByteArray
    external fun drawContours(
        image: Long,
        flat: ByteArray,
        contourIndex: Int,
        v0: Double, v1: Double, v2: Double, v3: Double,
        thickness: Int,
    )

    external fun contourAreaBytes(flat: ByteArray): Double
    external fun arcLengthBytes(flat: ByteArray, closed: Boolean): Double
    external fun boundingRect(flat: ByteArray): IntArray
    external fun approxPolyDP(flat: ByteArray, epsilon: Double, closed: Boolean): ByteArray
    external fun minAreaRect(flat: ByteArray): DoubleArray
    external fun minEnclosingCircle(flat: ByteArray): DoubleArray
    external fun moments(arr: Long, binaryImage: Boolean): DoubleArray
    external fun matchShapes(a: Long, b: Long, method: Int): Double
    external fun houghLines(src: Long, rho: Double, theta: Double, threshold: Int, srn: Double, stn: Double): Long
    external fun houghLinesP(
        src: Long,
        rho: Double,
        theta: Double,
        threshold: Int,
        minLineLength: Double,
        maxLineGap: Double,
    ): Long

    external fun houghCircles(
        src: Long,
        method: Int,
        dp: Double,
        minDist: Double,
        param1: Double,
        param2: Double,
        minRadius: Int,
        maxRadius: Int,
    ): Long

    external fun cornerHarris(src: Long, blockSize: Int, ksize: Int, k: Double): Long
    external fun cornerMinEigenVal(src: Long, blockSize: Int, ksize: Int): Long
    external fun goodFeaturesToTrack(
        src: Long,
        maxCorners: Int,
        qualityLevel: Double,
        minDistance: Double,
        blockSize: Int,
        useHarrisDetector: Boolean,
        k: Double,
    ): Long

    external fun matchTemplate(image: Long, templ: Long, method: Int): Long
    external fun distanceTransform(src: Long, distanceType: Int, maskSize: Int): Long
    external fun integral(src: Long, sdepth: Int): Long
    external fun connectedComponents(src: Long, connectivity: Int, ltype: Int): LongArray
    external fun connectedComponentsWithStats(src: Long, connectivity: Int, ltype: Int): LongArray
    external fun pyrMeanShiftFiltering(src: Long, sp: Double, sr: Double, maxLevel: Int): Long
    external fun thresholdWithMask(src: Long, mask: Long, dst: Long, thresh: Double, maxVal: Double, type: Int): Double
    external fun createHanningWindow(width: Int, height: Int, type: Int): Long
    external fun accumulate(src: Long, dst: Long)
    external fun accumulateSquare(src: Long, dst: Long)
    external fun accumulateProduct(a: Long, b: Long, dst: Long)
    external fun accumulateWeighted(src: Long, dst: Long, alpha: Double)

    // ---- imgproc: drawing (in-place)

    external fun arrowedLine(
        mat: Long,
        x1: Int, y1: Int, x2: Int, y2: Int,
        v0: Double, v1: Double, v2: Double, v3: Double,
        thickness: Int,
    )

    external fun drawMarker(
        mat: Long,
        x: Int,
        y: Int,
        markerType: Int,
        size: Int,
        v0: Double, v1: Double, v2: Double, v3: Double,
        thickness: Int,
    )

    external fun ellipse(
        mat: Long,
        cx: Int,
        cy: Int,
        axesX: Int,
        axesY: Int,
        angle: Double,
        startAngle: Double,
        endAngle: Double,
        v0: Double, v1: Double, v2: Double, v3: Double,
        thickness: Int,
    )

    external fun fillPoly(
        mat: Long,
        flat: ByteArray,
        v0: Double, v1: Double, v2: Double, v3: Double,
        thickness: Int,
    )

    external fun polylines(
        mat: Long,
        flat: ByteArray,
        closed: Boolean,
        v0: Double, v1: Double, v2: Double, v3: Double,
        thickness: Int,
    )

    // ---- imgproc: CLAHE

    external fun claheCreate(clipLimit: Double, tileWidth: Int, tileHeight: Int): Long
    external fun claheApply(clahe: Long, src: Long): Long
    external fun claheSetClipLimit(clahe: Long, clipLimit: Double)
    external fun claheRelease(clahe: Long)

    // ---- imgcodecs additions

    external fun imcount(path: String): Int
    external fun haveImageReader(ext: String): Boolean
    external fun haveImageWriter(ext: String): Boolean
    external fun imwriteParams(path: String, mat: Long, params: IntArray): Boolean
    external fun imencodeParams(ext: String, mat: Long, params: IntArray): ByteArray
    external fun setRngSeed(seed: Long)

    // ---- org.opencv.core.Mat parity

    external fun matDump(mat: Long): String
    external fun matIsContinuous(mat: Long): Boolean
    external fun matIsSubmatrix(mat: Long): Boolean
    external fun matAdjustROI(mat: Long, dtop: Int, dbottom: Int, dleft: Int, dright: Int): Long
    external fun matLocateROI(mat: Long): IntArray
    external fun matCross(a: Long, b: Long): Long
    external fun matPutValues(mat: Long, row: Int, col: Int, values: DoubleArray): Int
    external fun matGetValues(mat: Long, row: Int, col: Int, values: DoubleArray): Int

    // ---- core: clustering / decomposition (Core parity)

    external fun kmeans(
        data: Long,
        k: Int,
        critType: Int,
        critMax: Int,
        critEps: Double,
        attempts: Int,
        flags: Int,
        labels: Long,
        compactnessOut: DoubleArray,
    ): Long

    external fun svdDecomp(src: Long, flags: Int): LongArray
    external fun svdBackSubst(w: Long, u: Long, vt: Long, b: Long): Long
    external fun pcaCompute(data: Long, maxComponents: Int): LongArray
    external fun pcaComputeVariance(data: Long, retainedVariance: Double): LongArray
    external fun pcaProject(data: Long, mean: Long, vectors: Long): Long
    external fun pcaBackProject(data: Long, mean: Long, vectors: Long): Long
    external fun mahalanobis(v1: Long, v2: Long, icovar: Long): Double

    // ---- imgproc parity additions

    external fun cornerSubPixBytes(
        image: Long,
        flat: ByteArray,
        winW: Int,
        winH: Int,
        zeroW: Int,
        zeroH: Int,
        critType: Int,
        critMax: Int,
        critEps: Double,
    ): ByteArray

    external fun emd(sig1: Long, sig2: Long, distType: Int): Double
    external fun grabCut(
        img: Long,
        mask: Long,
        rx: Int,
        ry: Int,
        rw: Int,
        rh: Int,
        bgdModel: Long,
        fgdModel: Long,
        iters: Int,
        mode: Int,
    )

    // highgui (desktop window backends)

    external fun namedWindow(winname: String, flags: Int)
    external fun resizeWindow(winname: String, width: Int, height: Int)
    external fun imshow(winname: String, mat: Long)
    external fun waitKey(delayMs: Int): Int
    external fun destroyWindow(winname: String)
    external fun destroyAllWindows()
}

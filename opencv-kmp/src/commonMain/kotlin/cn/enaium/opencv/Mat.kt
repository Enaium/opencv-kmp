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

import kotlin.math.PI

/**
 * A reference-counted OpenCV matrix (`cv::Mat`).
 *
 * Every handle owns one native reference; call [close] (or [Mat.use]) to
 * release it. Operations that return a new [Mat] allocate an independent
 * result unless documented otherwise — only [roi] shares pixels with its
 * source, mirroring `mat(rect)` semantics.
 */
interface Mat : AutoCloseable {
    val rows: Int
    val cols: Int

    /** OpenCV type id (depth + channels); decode with [cvDepthOf]/[cvChannelsOf]. */
    val type: Int
    val channels: Int

    /** Bytes per element across all channels. */
    val elemSize: Int
    val total: Int
    val isEmpty: Boolean

    /** Column-major friendly view: `(cols, rows)`. */
    val size: Size get() = Size(width = cols, height = rows)

    // ---- lifecycle / views -------------------------------------------------

    /** Deep copy with independent pixel storage. */
    fun clone(): Mat

    /**
     * Region of interest sharing this matrix's pixels. Closing either handle
     * is safe; the storage lives until the last reference is released.
     */
    fun roi(rect: Rect): Mat

    fun convertTo(type: Int = this.type, alpha: Double = 1.0, beta: Double = 0.0): Mat

    // ---- element access ----------------------------------------------------

    /** Reads channel 0 of the element at ([row], [col]). */
    operator fun get(row: Int, col: Int): Double

    /** Writes channel 0 of the element at ([row], [col]) with saturation. */
    operator fun set(row: Int, col: Int, value: Double)

    /** Reads any [channel] of the element at ([row], [col]). */
    fun at(row: Int, col: Int, channel: Int = 0): Double

    /** Writes any [channel] of the element at ([row], [col]) with saturation. */
    fun put(row: Int, col: Int, channel: Int = 0, value: Double)

    /** Bulk pixel bytes; size equals [total] * [elemSize]. */
    var pixels: ByteArray

    // ---- arithmetic (each returns a new Mat) --------------------------------

    operator fun plus(other: Mat): Mat
    operator fun minus(other: Mat): Mat

    /** Element-wise product times [scale] (`cv::multiply`). */
    operator fun times(other: Mat): Mat

    /** Divides element-wise by [other] (`cv::divide`). */
    operator fun div(other: Mat): Mat

    /** Scales every element by [scale] (`cv::multiply` with a scalar). */
    operator fun times(scale: Double): Mat

    fun absDiff(other: Mat): Mat
    infix fun bitwiseAnd(other: Mat): Mat
    infix fun bitwiseOr(other: Mat): Mat
    infix fun bitwiseXor(other: Mat): Mat

    fun bitwiseNot(): Mat
    fun min(other: Mat): Mat
    fun max(other: Mat): Mat

    /** Binary mask where lower <= value <= upper component-wise. */
    fun inRange(lower: Scalar, upper: Scalar): Mat

    // ---- geometry ------------------------------------------------------------

    fun transpose(): Mat

    /** [flipCode]: [FlipMode.VERTICAL], [FlipMode.HORIZONTAL] or [FlipMode.BOTH]. */
    fun flip(flipCode: Int): Mat

    // ---- reductions / statistics ----------------------------------------------

    val mean: Scalar
    val sum: Scalar
    fun meanStdDev(): Pair<Scalar, Scalar>
    fun minMaxLoc(): MinMaxLoc
    val nonZeroCount: Int

    // ---- imgproc -----------------------------------------------------------------

    fun cvtColor(code: Int): Mat

    fun resize(
        width: Int,
        height: Int,
        interpolation: Int = InterpolationFlags.LINEAR,
    ): Mat

    fun resize(size: Size, interpolation: Int = InterpolationFlags.LINEAR): Mat =
        resize(width = size.width, height = size.height, interpolation = interpolation)

    fun gaussianBlur(
        kernelWidth: Int,
        kernelHeight: Int,
        sigmaX: Double = 0.0,
        sigmaY: Double = 0.0,
    ): Mat

    fun medianBlur(kernelSize: Int): Mat

    fun threshold(thresh: Double, maxVal: Double, type: Int = ThresholdTypes.BINARY): Mat

    fun adaptiveThreshold(
        maxValue: Double,
        method: Int = AdaptiveThresholdTypes.GAUSSIAN_C,
        type: Int = ThresholdTypes.BINARY,
        blockSize: Int = 3,
        c: Double = 5.0,
    ): Mat

    fun canny(
        threshold1: Double,
        threshold2: Double,
        apertureSize: Int = 3,
        l2Gradient: Boolean = false,
    ): Mat

    fun sobel(dx: Int, dy: Int, kernelSize: Int = 3): Mat
    fun laplacian(kernelSize: Int = 1): Mat

    // ---- drawing (in-place) ------------------------------------------------------

    fun rectangle(from: Point, to: Point, color: Scalar, thickness: Int = LineTypes.LINE_8)
    fun circle(center: Point, radius: Int, color: Scalar, thickness: Int = LineTypes.LINE_8)
    fun line(from: Point, to: Point, color: Scalar, thickness: Int = LineTypes.LINE_8)

    // ---- core: shape / algebra ------------------------------------------------

    /** Same data reinterpreted with different [channels]/[rows] (`cv::Mat::reshape`). */
    fun reshape(channels: Int = 1, rows: Int = 0): Mat

    /** Row slice sharing pixels; [end] is exclusive. */
    fun rowRange(start: Int, end: Int): Mat

    /** Column slice sharing pixels; [end] is exclusive. */
    fun colRange(start: Int, end: Int): Mat

    /** Diagonal as a column matrix ([d]: 0 main, >0 below, <0 above). */
    fun diag(d: Int = 0): Mat

    /** Scales the diagonal in place (`cv::setIdentity`). */
    fun setIdentity(scale: Double = 1.0)

    /** Dot product of two same-shaped single-channel matrices. */
    infix fun dot(other: Mat): Double

    /** Matrix inverse ([DecompTypes]); null when singular for [DecompTypes.LU]. */
    fun inv(method: Int = DecompTypes.LU): Mat?

    /** Determinant (square single-channel matrices). */
    val determinant: Double

    /** Sum of diagonal elements. */
    val trace: Scalar

    // ---- core: array operations ----------------------------------------------

    /** Splits channels into single-channel matrices. */
    fun split(): List<Mat>

    /** Normalizes to [alpha]..[beta] range under [normType]. */
    fun normalize(
        alpha: Double = 1.0,
        beta: Double = 0.0,
        normType: Int = NormTypes.L2,
        dtype: Int = -1,
    ): Mat

    /** LUT remap for CV_8U sources. */
    fun lut(lut: Mat): Mat

    /** Rotates by multiples of 90 degrees ([RotateFlags]). */
    fun rotate(code: Int): Mat

    /** Adds a border around the matrix. */
    fun copyMakeBorder(
        top: Int,
        bottom: Int,
        left: Int,
        right: Int,
        borderType: Int = BorderTypes.CONSTANT,
        value: Scalar = Scalar(),
    ): Mat

    /** Weighted blend: alpha*this + beta*[other] + gamma. */
    fun addWeighted(alpha: Double, other: Mat, beta: Double, gamma: Double): Mat

    /** convertScaleAbs with absolute values into CV_8U. */
    fun convertScaleAbs(alpha: Double = 1.0, beta: Double = 0.0): Mat

    /** Element-wise comparison producing an 8-bit mask ([CompareOps]). */
    fun compare(other: Mat, op: Int): Mat

    /** Solves this * x = [b]; null when no solution exists. */
    fun solve(b: Mat, flags: Int = DecompTypes.LU): Mat?

    /** Tiles the matrix nx times horizontally and ny times vertically. */
    fun repeat(nx: Int, ny: Int): Mat

    /** Linear per-element transform by an m-rows x n-cols matrix. */
    fun transform(m: Mat): Mat

    /** Transforms points through a 3x3/4x4 projection matrix. */
    fun perspectiveTransform(m: Mat): Mat

    fun pow(power: Double): Mat
    fun sqrt(): Mat
    fun exp(): Mat
    fun log(): Mat

    /** Magnitude of the vector pair (this=x, [y]=y). */
    fun magnitude(y: Mat): Mat

    /** Angle of the vector pair (this=x, [y]=y). */
    fun phase(y: Mat, angleInDegrees: Boolean = false): Mat

    /** Polar decomposition of cartesian coordinates ([y]=y component). */
    fun cartToPolar(y: Mat, angleInDegrees: Boolean = false): Pair<Mat, Mat>

    /** Cartesian coordinates from magnitude (this) and [angle]. */
    fun polarToCart(angle: Mat, angleInDegrees: Boolean = false): Pair<Mat, Mat>

    /** Replaces NaNs with [value] in place. */
    fun patchNaNs(value: Double = 0.0)

    /** Whether any element is non-zero. */
    val hasNonZero: Boolean

    /** Non-zero coordinates as Nx1 CV_32SC2. */
    fun findNonZero(): Mat

    fun sort(flags: Int = SortFlags.EVERY_ROW): Mat
    fun sortIdx(flags: Int = SortFlags.EVERY_ROW): Mat

    fun reduce(dim: Int, rtype: Int, dtype: Int = -1): Mat
    fun reduceArgMax(dim: Int): Mat
    fun reduceArgMin(dim: Int): Mat

    /** Channel [coi] as its own matrix. */
    fun extractChannel(coi: Int): Mat

    /** Copies [channel] into channel [coi] of this matrix in place. */
    fun insertChannel(channel: Mat, coi: Int)

    /** Fills with uniformly distributed values in place. */
    fun randu(low: Scalar, high: Scalar)

    /** Fills with normally distributed values in place. */
    fun randn(mean: Scalar, stddev: Scalar)

    /** Peak signal-to-noise ratio against [other]. */
    fun psnr(other: Mat, r: Double = 255.0): Double

    /** Forward DFT; combine [DftFlags] for inverse/scale variants. */
    fun dft(flags: Int = 0): Mat

    /** Inverse DFT (`DftFlags.INVERSE` applied internally). */
    fun idft(flags: Int = DftFlags.SCALE): Mat
    fun dct(flags: Int = 0): Mat
    fun idct(flags: Int = DftFlags.INVERSE or DftFlags.SCALE): Mat

    /** Element-wise spectrum product (CV_32FC2/CV_64FC2 inputs). */
    fun mulSpectrums(other: Mat, conjugate: Boolean = false, dftRows: Boolean = false): Mat

    /** Element-wise spectrum division. */
    fun divSpectrums(other: Mat, conjugate: Boolean = false): Mat

    /** Generalized product alpha*this*[other] + gamma*[c]. */
    fun gemm(other: Mat, alpha: Double = 1.0, c: Mat? = null, gamma: Double = 0.0): Mat

    /** Symmetric eigen decomposition: eigenvalues to eigenvectors. */
    fun eigen(): Pair<Mat, Mat>

    // ---- imgproc: filters ------------------------------------------------------

    fun blur(kernelWidth: Int, kernelHeight: Int): Mat

    fun boxFilter(
        kernelWidth: Int,
        kernelHeight: Int,
        ddepth: Int = -1,
        normalize: Boolean = true,
    ): Mat

    fun sqrBoxFilter(kernelWidth: Int, kernelHeight: Int, ddepth: Int = -1): Mat

    fun bilateralFilter(d: Int, sigmaColor: Double, sigmaSpace: Double): Mat

    fun stackBlur(kernelSize: Int): Mat

    /** Null [kernel] means a default 3x3 rectangular structuring element. */
    fun erode(kernel: Mat? = null, iterations: Int = 1): Mat

    /** Null [kernel] means a default 3x3 rectangular structuring element. */
    fun dilate(kernel: Mat? = null, iterations: Int = 1): Mat

    /** Full morphology pipeline ([MorphTypes]). */
    fun morphologyEx(op: Int, kernel: Mat? = null, iterations: Int = 1): Mat

    fun filter2D(kernel: Mat, ddepth: Int = -1, delta: Double = 0.0): Mat

    fun pyrDown(): Mat
    fun pyrUp(): Mat

    // ---- imgproc: geometry -------------------------------------------------------

    fun warpAffine(m: Mat, width: Int = cols, height: Int = rows, flags: Int = InterpolationFlags.LINEAR): Mat

    fun warpPerspective(m: Mat, width: Int = cols, height: Int = rows, flags: Int = InterpolationFlags.LINEAR): Mat

    fun remap(map1: Mat, map2: Mat, interpolation: Int = InterpolationFlags.LINEAR): Mat

    /** Polar/log-polar unwrapping around a center ([WarpPolarMode]). */
    fun warpPolar(
        radius: Int,
        centerX: Double,
        centerY: Double,
        maxRadius: Double,
        flags: Int = WarpPolarMode.POLAR,
    ): Mat

    /** Lens undistortion with camera intrinsics. */
    fun undistort(cameraMatrix: Mat, distCoeffs: Mat): Mat

    /** Sub-pixel centered patch extraction. */
    fun getRectSubPix(width: Int, height: Int, centerX: Double, centerY: Double): Mat

    // ---- imgproc: color / histogram -------------------------------------------------

    fun demosaicing(code: Int): Mat

    /** Pseudo-coloring with a built-in palette ([ColormapTypes]). */
    fun applyColorMap(colormap: Int): Mat

    /** Pseudo-coloring through a user lookup table. */
    fun applyColorMap(userColor: Mat): Mat

    /**
     * One-dimensional histogram of [channel]; bins span [minValue] until
     * [maxValue]. Returns histSize x 1 CV_32F.
     */
    fun calcHist(
        channel: Int = 0,
        histSize: Int = 256,
        minValue: Float = 0f,
        maxValue: Float = 256f,
    ): Mat

    /** Back-projects [hist] (from [calcHist]) onto this image. */
    fun calcBackProject(
        hist: Mat,
        channel: Int = 0,
        minValue: Float = 0f,
        maxValue: Float = 256f,
    ): Mat

    fun equalizeHist(): Mat

    /** Hu-moment shape similarity ([ContoursMatchMethods]). */
    fun matchShapes(other: Mat, method: Int = ContoursMatchMethods.CONTOURS_MATCH_I1): Double

    /** Image moments of a rasterized shape. */
    fun moments(binaryImage: Boolean = false): Moments

    // ---- imgproc: segmentation / features ----------------------------------------------

    /** Flood-fills from the seed; returns the number of repainted pixels. */
    fun floodFill(
        seedX: Int,
        seedY: Int,
        newValue: Scalar,
        loDiff: Scalar = Scalar(),
        upDiff: Scalar = Scalar(),
        flags: Int = FloodFillFlags.CONNECTIVITY_4,
    ): Int

    /** Graph-cut segmentation driven by labeled [markers] (in place). */
    fun watershed(markers: Mat)

    fun matchTemplate(templ: Mat, method: Int = TemplateMatchModes.CCOEFF_NORMED): Mat

    fun cornerHarris(blockSize: Int = 2, ksize: Int = 3, k: Double = 0.04): Mat

    fun cornerMinEigenVal(blockSize: Int = 3, ksize: Int = 3): Mat

    /** Strong corners as Nx1 CV_32FC2 points. */
    fun goodFeaturesToTrack(
        maxCorners: Int,
        qualityLevel: Double = 0.01,
        minDistance: Double = 10.0,
        blockSize: Int = 3,
        useHarrisDetector: Boolean = false,
        k: Double = 0.04,
    ): Mat

    fun distanceTransform(
        distanceType: Int = DistanceTypes.L2,
        maskSize: Int = DistanceTransformMasks.MASK_3,
    ): Mat

    fun integral(sdepth: Int = -1): Mat

    /** Labels connected components; count includes background label 0. */
    fun connectedComponents(connectivity: Int = 8, ltype: Int = CV_32S): Pair<Int, Mat>

    /** Like [connectedComponents] plus bounding-box stats and centroids. */
    fun connectedComponentsWithStats(connectivity: Int = 8, ltype: Int = CV_32S): Components

    fun pyrMeanShiftFiltering(sp: Double, sr: Double, maxLevel: Int = 1): Mat

    /** Mask-aware threshold; returns the computed threshold (Otsu/Triangle). */
    fun thresholdWithMask(
        thresh: Double,
        maxVal: Double,
        type: Int = ThresholdTypes.BINARY,
        mask: Mat? = null,
    ): Pair<Double, Mat>

    // ---- imgproc: hough / accumulators ---------------------------------------------------

    fun houghCircles(
        dp: Double = 1.0,
        minDist: Double = 1.0,
        param1: Double = 100.0,
        param2: Double = 100.0,
        minRadius: Int = 0,
        maxRadius: Int = 0,
        method: Int = HoughTypes.GRADIENT,
    ): Mat

    fun accumulate(src: Mat)
    fun accumulateSquare(src: Mat)
    fun accumulateProduct(a: Mat, b: Mat)
    fun accumulateWeighted(src: Mat, alpha: Double)

    // ---- imgproc: contours ------------------------------------------------------------

    /** Contours of a binary image ([RetrievalModes], [ContourApproximationModes]). */
    fun findContours(mode: Int = RetrievalModes.RETR_LIST, method: Int = ContourApproximationModes.CHAIN_APPROX_SIMPLE): List<List<Point>>

    /** Draws contours ([contourIndex] -1 draws all); thickness<0 fills. */
    fun drawContours(
        contours: List<List<Point>>,
        color: Scalar,
        contourIndex: Int = -1,
        thickness: Int = LineTypes.LINE_8,
    )

    // ---- imgproc: drawing (in-place) -----------------------------------------------------

    fun arrowedLine(from: Point, to: Point, color: Scalar, thickness: Int = LineTypes.LINE_8)

    fun drawMarker(
        pos: Point,
        color: Scalar,
        markerType: Int = MarkerTypes.CROSS,
        size: Int = 20,
        thickness: Int = LineTypes.LINE_8,
    )

    fun ellipse(
        center: Point,
        axes: Size,
        angle: Double = 0.0,
        startAngle: Double = 0.0,
        endAngle: Double = 360.0,
        color: Scalar,
        thickness: Int = LineTypes.LINE_8,
    )

    fun houghLines(rho: Double = 1.0, theta: Double = PI / 180.0, threshold: Int = 80, srn: Double = 0.0, stn: Double = 0.0): Mat

    fun houghLinesP(rho: Double = 1.0, theta: Double = PI / 180.0, threshold: Int = 80, minLineLength: Double = 0.0, maxLineGap: Double = 0.0): Mat

    /** Fills/strokes polygons ([thickness] < 0 fills). */
    fun fillPoly(polygons: List<List<Point>>, color: Scalar, thickness: Int = LineTypes.FILLED)

    fun polylines(polylines: List<List<Point>>, closed: Boolean, color: Scalar, thickness: Int = LineTypes.LINE_8)

    // ---- org.opencv.core.Mat parity ------------------------------------------

    /** ROI view alias matching the official `Mat.submat`. */
    fun submat(rect: Rect): Mat = roi(rect)

    /** Grows/shrinks this ROI by the given deltas, like `Mat.adjustROI`. */
    fun adjustROI(dtop: Int, dbottom: Int, dleft: Int, dright: Int): Mat

    /** Offset and size of this ROI inside its parent. */
    fun locateROI(): Pair<Point, Size>

    val isContinuous: Boolean
    val isSubmatrix: Boolean

    /** 3-element cross product (1x3/3x1 CV_32FC1). */
    infix fun cross(other: Mat): Mat

    /** Content dump mirroring the official `Mat.dump()`. */
    fun dump(): String

    /**
     * Writes [values] starting at ([row], [col]) running along the row,
     * exactly like the official typed `Mat.put`; returns elements written.
     */
    fun put(row: Int, col: Int, values: DoubleArray): Int

    /**
     * Reads at most [values].size elements starting at ([row], [col]) into
     * [values], like the official typed `Mat.get`; returns elements read.
     */
    fun get(row: Int, col: Int, values: DoubleArray): Int

    override fun close()
}

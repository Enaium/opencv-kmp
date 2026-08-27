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

private fun lastNativeError(): String? = Jni.lastError()


/** Unpacks a two-handle output; the caller must close both Mats. */
private fun matPair(handles: LongArray, operation: String): Pair<Mat, Mat> {
    require(handles.size >= 2) { "$operation returned no result" }
    return jvmMat(handles[0], operation) to jvmMat(handles[1], operation)
}

// =========================================================================
// kernels / filters
// =========================================================================

actual fun getDerivKernels(
    dx: Int,
    dy: Int,
    ksize: Int,
    normalize: Boolean,
    ktype: Int,
): Pair<Mat, Mat> =
    matPair(JniImgproc.getDerivKernels(dx, dy, ksize, normalize, ktype), "getDerivKernels")

actual fun getGaborKernel(
    ksize: Size,
    sigma: Double,
    theta: Double,
    lambda: Double,
    gamma: Double,
    psi: Double,
    ktype: Int,
): Mat = jvmMat(
    JniImgproc.getGaborKernel(ksize.width, ksize.height, sigma, theta, lambda, gamma, psi, ktype),
    "getGaborKernel",
)

actual fun sepFilter2D(
    src: Mat,
    ddepth: Int,
    kernelX: Mat,
    kernelY: Mat,
    anchorX: Int,
    anchorY: Int,
    delta: Double,
    borderType: Int,
): Mat = jvmMat(
    JniImgproc.sepFilter2D(
        handleOf(src), ddepth, handleOf(kernelX), handleOf(kernelY),
        anchorX, anchorY, delta, borderType,
    ),
    "sepFilter2D",
)

// =========================================================================
// corners
// =========================================================================

actual fun preCornerDetect(src: Mat, ksize: Int, borderType: Int): Mat =
    jvmMat(JniImgproc.preCornerDetect(handleOf(src), ksize, borderType), "preCornerDetect")

actual fun cornerEigenValsAndVecs(src: Mat, blockSize: Int, ksize: Int, borderType: Int): Mat =
    jvmMat(
        JniImgproc.cornerEigenValsAndVecs(handleOf(src), blockSize, ksize, borderType),
        "cornerEigenValsAndVecs",
    )

// =========================================================================
// color conversion
// =========================================================================

actual fun cvtColorTwoPlane(src1: Mat, src2: Mat, code: Int): Mat =
    jvmMat(JniImgproc.cvtColorTwoPlane(handleOf(src1), handleOf(src2), code), "cvtColorTwoPlane")

// =========================================================================
// histogram
// =========================================================================

actual fun compareHist(h1: Mat, h2: Mat, method: Int): Double =
    Jni.compareHist(handleOf(h1), handleOf(h2), method)

// =========================================================================
// distance transform
// =========================================================================

actual fun distanceTransformWithLabels(
    src: Mat,
    distanceType: Int,
    maskSize: Int,
    labelType: Int,
): Pair<Mat, Mat> =
    matPair(
        JniImgproc.distanceTransformWithLabels(handleOf(src), distanceType, maskSize, labelType),
        "distanceTransformWithLabels",
    )

// =========================================================================
// calibration helpers
// =========================================================================

actual fun initUndistortRectifyMap(
    cameraMatrix: Mat,
    distCoeffs: Mat,
    r: Mat?,
    newCameraMatrix: Mat?,
    size: Size,
    m1type: Int,
): Pair<Mat, Mat> = matPair(
    JniImgproc.initUndistortRectifyMap(
        handleOf(cameraMatrix), handleOf(distCoeffs),
        r?.let { handleOf(it) } ?: 0L,
        newCameraMatrix?.let { handleOf(it) } ?: 0L,
        size.width, size.height, m1type,
    ),
    "initUndistortRectifyMap",
)

actual fun undistortPoints(src: Mat, cameraMatrix: Mat, distCoeffs: Mat, r: Mat?, p: Mat?): Mat =
    jvmMat(
        JniImgproc.undistortPoints(
            handleOf(src), handleOf(cameraMatrix), handleOf(distCoeffs),
            r?.let { handleOf(it) } ?: 0L,
            p?.let { handleOf(it) } ?: 0L,
        ),
        "undistortPoints",
    )

actual fun getDefaultNewCameraMatrix(
    cameraMatrix: Mat,
    imgsize: Size,
    centerPrincipalPoint: Boolean,
): Mat = jvmMat(
    JniImgproc.getDefaultNewCameraMatrix(
        handleOf(cameraMatrix), imgsize.width, imgsize.height, centerPrincipalPoint,
    ),
    "getDefaultNewCameraMatrix",
)

actual fun estimateAffine2D(
    from: Mat,
    to: Mat,
    method: Int,
    ransacReprojThreshold: Double,
    maxIters: Int,
    confidence: Double,
    refineIters: Int,
): Mat? = JniImgproc.estimateAffine2D(
    handleOf(from), handleOf(to), method, ransacReprojThreshold,
    maxIters.toLong(), confidence, refineIters.toLong(),
).takeIf { it != 0L }?.let { jvmMat(it, "estimateAffine2D") }

actual fun estimateAffinePartial2D(
    from: Mat,
    to: Mat,
    method: Int,
    ransacReprojThreshold: Double,
    maxIters: Int,
    confidence: Double,
    refineIters: Int,
): Mat? = JniImgproc.estimateAffinePartial2D(
    handleOf(from), handleOf(to), method, ransacReprojThreshold,
    maxIters.toLong(), confidence, refineIters.toLong(),
).takeIf { it != 0L }?.let { jvmMat(it, "estimateAffinePartial2D") }

// =========================================================================
// contours / geometry
// =========================================================================

actual fun convexHull(points: Mat, clockwise: Boolean, returnPoints: Boolean): Mat =
    jvmMat(
        JniImgproc.convexHull(handleOf(points), clockwise, returnPoints),
        "convexHull",
    )

actual fun isContourConvex(points: Mat): Boolean =
    JniImgproc.isContourConvex(handleOf(points))

actual fun convexityDefects(contour: Mat, convexHullIdx: Mat): Mat =
    jvmMat(JniImgproc.convexityDefects(handleOf(contour), handleOf(convexHullIdx)), "convexityDefects")

actual fun fitLine(points: Mat, distType: Int, param: Double, reps: Double, aeps: Double): Mat =
    jvmMat(JniImgproc.fitLine(handleOf(points), distType, param, reps, aeps), "fitLine")

actual fun boxPoints(box: RotatedRect): Mat = jvmMat(
    JniImgproc.boxPoints(box.centerX, box.centerY, box.width, box.height, box.angle),
    "boxPoints",
)

actual fun rotatedRectangleIntersection(rect1: RotatedRect, rect2: RotatedRect): RotatedRectIntersection {
    val out = JniImgproc.rotatedRectangleIntersection(
        rect1.centerX, rect1.centerY, rect1.width, rect1.height, rect1.angle,
        rect2.centerX, rect2.centerY, rect2.width, rect2.height, rect2.angle,
    )
    require(out.size >= 2) { "rotatedRectangleIntersection returned no result" }
    return RotatedRectIntersection(
        type = out[0].toInt(),
        points = jvmMat(out[1], "rotatedRectangleIntersection"),
    )
}

actual fun pointPolygonTest(contour: Mat, x: Double, y: Double, measureDist: Boolean): Double =
    JniImgproc.pointPolygonTest(handleOf(contour), x, y, measureDist)

actual fun intersectConvexConvex(p1: Mat, p2: Mat, handleNested: Boolean): ConvexIntersection {
    val out = JniImgproc.intersectConvexConvex(handleOf(p1), handleOf(p2), handleNested)
    require(out.size >= 2) { "intersectConvexConvex returned no result" }
    return ConvexIntersection(
        area = Float.fromBits(out[1].toInt()),
        polygon = jvmMat(out[0], "intersectConvexConvex"),
    )
}

actual fun huMoments(moments: Moments): DoubleArray = JniImgproc.huMoments(
    moments.m00, moments.m10, moments.m01, moments.m20, moments.m11,
    moments.m02, moments.m30, moments.m21, moments.m12, moments.m03,
)

// =========================================================================
// pyramids
// =========================================================================

actual fun buildPyramid(src: Mat, maxLevel: Int, borderType: Int): List<Mat> =
    JniImgproc.buildPyramid(handleOf(src), maxLevel, borderType).map { handle ->
        jvmMat(handle, "buildPyramid")
    }

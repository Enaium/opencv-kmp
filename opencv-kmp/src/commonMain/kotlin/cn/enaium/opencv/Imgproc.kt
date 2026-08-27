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
 * Static (factory-style) operations of `org.opencv.imgproc.Imgproc` that
 * return fresh matrices or plain values and were not already covered by
 * [Mat] members or the existing top-level factories ([getStructuringElement],
 * [getGaussianKernel], [minAreaRect], ...).
 *
 * Every function allocates its own result; the caller owns and must close
 * any returned [Mat]. Functions with two matrix outputs return a `Pair` of
 * freshly allocated matrices.
 *
 * NOT provided: `putText` / `getTextSize` / `getFontScaleFromHeight` (and
 * the FontFace overloads). OpenCV-kmp's native OpenCV is built without
 * HarfBuzz (`WITH_HARFBUZZ=OFF`), so the text-rendering entry points are
 * unavailable in the binary; the [HersheyFonts] / [PutTextFlags] constants
 * are still exposed for source compatibility.
 */

// =========================================================================
// kernels / filters
// =========================================================================

/**
 * `cv::getDerivKernels`: filter coefficients for spatial image derivatives.
 *
 * Returns `(kx, ky)` where `kx` is the row filter and `ky` the column filter
 * (both `ktype`-typed, e.g. CV_32F); pass them to [sepFilter2D]. With
 * [normalize] the kernels are scaled to unit L1 norm.
 */
expect fun getDerivKernels(
    dx: Int,
    dy: Int,
    ksize: Int,
    normalize: Boolean = false,
    ktype: Int = CV_32F,
): Pair<Mat, Mat>

/**
 * `cv::getGaborKernel`: a Gabor filter coefficient matrix of [ksize]
 * (rows = height, cols = width) with wavelength [lambda] and phase [psi].
 */
expect fun getGaborKernel(
    ksize: Size,
    sigma: Double,
    theta: Double,
    lambda: Double,
    gamma: Double,
    psi: Double = PI / 2,
    ktype: Int = CV_64F,
): Mat

/**
 * `cv::sepFilter2D`: separable convolution with a horizontal kernel
 * [kernelX] and a vertical kernel [kernelY]; equivalent to [Mat.filter2D]
 * with their outer product but faster. Returns a new matrix.
 */
expect fun sepFilter2D(
    src: Mat,
    ddepth: Int,
    kernelX: Mat,
    kernelY: Mat,
    anchorX: Int = -1,
    anchorY: Int = -1,
    delta: Double = 0.0,
    borderType: Int = BorderTypes.REFLECT_101,
): Mat

// =========================================================================
// corners
// =========================================================================

/** `cv::preCornerDetect`: corner-preprocessor response as a CV_32F matrix. */
expect fun preCornerDetect(
    src: Mat,
    ksize: Int,
    borderType: Int = BorderTypes.REFLECT_101,
): Mat

/**
 * `cv::cornerEigenValsAndVecs`: per-pixel 6-channel CV_32FC6 output —
 * (lambda1, lambda2, x1, y1, x2, y2) of the gradient covariance matrix.
 */
expect fun cornerEigenValsAndVecs(
    src: Mat,
    blockSize: Int,
    ksize: Int,
    borderType: Int = BorderTypes.REFLECT_101,
): Mat

// =========================================================================
// color conversion
// =========================================================================

/**
 * `cv::cvtColorTwoPlane`: converts a two-plane YUV source (Y in [src1],
 * interleaved UV in [src2]) — e.g. NV12/NV21 — into BGR/RGB ([code] is a
 * `COLOR_YUV2*` constant).
 */
expect fun cvtColorTwoPlane(src1: Mat, src2: Mat, code: Int): Mat

// =========================================================================
// histogram
// =========================================================================

/** `cv::compareHist`: similarity of two histograms ([HistCompMethods]). */
expect fun compareHist(h1: Mat, h2: Mat, method: Int = HistCompMethods.CORREL): Double

// =========================================================================
// distance transform
// =========================================================================

/**
 * `cv::distanceTransform` with an output label matrix: returns `(dst,
 * labels)` where `dst` is the CV_32F distance map and `labels` the CV_32S
 * connected-component labels ([DistanceLabelTypes]).
 */
expect fun distanceTransformWithLabels(
    src: Mat,
    distanceType: Int,
    maskSize: Int,
    labelType: Int = DistanceLabelTypes.DIST_LABEL_CCOMP,
): Pair<Mat, Mat>

// =========================================================================
// calibration helpers
// =========================================================================

/**
 * `cv::initUndistortRectifyMap`: undistortion + rectification maps for
 * [Mat.remap]. Returns `(map1, map2)` of type [m1type] (CV_32FC1, CV_32FC2 or
 * CV_16SC2). Pass null [r] / [newCameraMatrix] for identity rotation and
 * the original camera matrix.
 */
expect fun initUndistortRectifyMap(
    cameraMatrix: Mat,
    distCoeffs: Mat,
    r: Mat?,
    newCameraMatrix: Mat?,
    size: Size,
    m1type: Int = MatType.CV_32FC1,
): Pair<Mat, Mat>

/**
 * `cv::undistortPoints`: undistorts a set of image points ([src], Nx1/Nx2
 * CV_32FC2 or CV_64FC2) and returns points of the same layout. Without [p]
 * the output is in normalized coordinates; pass a projection matrix to map
 * back to pixels.
 */
expect fun undistortPoints(
    src: Mat,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    r: Mat? = null,
    p: Mat? = null,
): Mat

/**
 * `cv::getDefaultNewCameraMatrix`: camera matrix adjusted so the principal
 * point sits at the image center of [imgsize] (when [centerPrincipalPoint]).
 * An empty [imgsize] keeps the principal point unchanged.
 */
expect fun getDefaultNewCameraMatrix(
    cameraMatrix: Mat,
    imgsize: Size = Size(0, 0),
    centerPrincipalPoint: Boolean = false,
): Mat

/**
 * `cv::estimateAffine2D`: best-fit 2x3 affine transform mapping [from]
 * points onto [to] points ([RobustMethods]); returns null when no
 * transformation could be estimated.
 */
expect fun estimateAffine2D(
    from: Mat,
    to: Mat,
    method: Int = RobustMethods.RANSAC,
    ransacReprojThreshold: Double = 3.0,
    maxIters: Int = 2000,
    confidence: Double = 0.99,
    refineIters: Int = 10,
): Mat?

/**
 * `cv::estimateAffinePartial2D`: like [estimateAffine2D] but restricted to
 * translation, rotation and uniform scale (4 degrees of freedom).
 */
expect fun estimateAffinePartial2D(
    from: Mat,
    to: Mat,
    method: Int = RobustMethods.RANSAC,
    ransacReprojThreshold: Double = 3.0,
    maxIters: Int = 2000,
    confidence: Double = 0.99,
    refineIters: Int = 10,
): Mat?

// =========================================================================
// contours / geometry
// =========================================================================

/**
 * `cv::convexHull`: convex hull of a 2D point set ([points], CV_32S or
 * CV_32F, Nx2). With [returnPoints] the hull points are returned (same
 * depth); otherwise a CV_32S index vector into [points] — the form
 * [convexityDefects] expects.
 */
expect fun convexHull(
    points: Mat,
    clockwise: Boolean = false,
    returnPoints: Boolean = true,
): Mat

/** `cv::isContourConvex`: whether the 2D point set forms a convex contour. */
expect fun isContourConvex(points: Mat): Boolean

/**
 * `cv::convexityDefects`: Nx4 CV_32S matrix of convexity defects —
 * `(startIdx, endIdx, farthestIdx, fixptDepth)` referencing the original
 * contour. [convexHullIdx] must come from [convexHull] with
 * `returnPoints = false`.
 */
expect fun convexityDefects(contour: Mat, convexHullIdx: Mat): Mat

/**
 * `cv::fitLine`: least-squares (or M-estimator, [distType]) line fit of a
 * 2D/3D point set; returns a 4x1 CV_32F vector `[vx, vy, x0, y0]` — a
 * normalized direction and a point on the line.
 */
expect fun fitLine(
    points: Mat,
    distType: Int = DistanceTypes.L2,
    param: Double = 0.0,
    reps: Double = 0.01,
    aeps: Double = 0.01,
): Mat

/**
 * `cv::boxPoints`: four corners of a [RotatedRect] as a 4x2 CV_32F matrix
 * (useful for drawing the rectangle from [minAreaRect]).
 */
expect fun boxPoints(box: RotatedRect): Mat

/** Result of [rotatedRectangleIntersection]; close [points] when done. */
data class RotatedRectIntersection(
    /** One of [RectangleIntersectTypes]. */
    val type: Int,
    /** Intersection polygon, Nx2 CV_32F (empty for [RectangleIntersectTypes.NONE]). */
    val points: Mat,
)

/**
 * `cv::rotatedRectangleIntersection`: intersection of two rotated
 * rectangles; the classification is one of [RectangleIntersectTypes].
 */
expect fun rotatedRectangleIntersection(rect1: RotatedRect, rect2: RotatedRect): RotatedRectIntersection

/**
 * `cv::pointPolygonTest`: signed distance from (x, y) to the nearest
 * contour edge when [measureDist]; otherwise +1 inside / -1 outside / 0 on
 * the contour.
 */
expect fun pointPolygonTest(
    contour: Mat,
    x: Double,
    y: Double,
    measureDist: Boolean = true,
): Double

/** Result of [intersectConvexConvex]; close [polygon] when done. */
data class ConvexIntersection(
    /** Area of the intersection polygon. */
    val area: Float,
    /** Intersection polygon, Nx2 CV_32F (empty when the polygons don't overlap). */
    val polygon: Mat,
)

/**
 * `cv::intersectConvexConvex`: intersection of two convex polygons ([p1],
 * [p2], Nx2 CV_32S/CV_32F); [handleNested] treats containment as a full
 * intersection.
 */
expect fun intersectConvexConvex(
    p1: Mat,
    p2: Mat,
    handleNested: Boolean = true,
): ConvexIntersection

/**
 * `cv::HuMoments`: seven translation/rotation/scale-invariant shape
 * descriptors computed from [Moments].
 */
expect fun huMoments(moments: Moments): DoubleArray

// =========================================================================
// pyramids
// =========================================================================

/**
 * `cv::buildPyramid`: Gaussian pyramid with [maxLevel] downsampling steps;
 * the returned list has `maxLevel + 1` entries (level 0 is a copy of
 * [src]).
 */
expect fun buildPyramid(
    src: Mat,
    maxLevel: Int,
    borderType: Int = BorderTypes.REFLECT_101,
): List<Mat>

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
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package cn.enaium.opencv

import cvk.cvk_box_points
import cvk.cvk_build_pyramid
import cvk.cvk_compare_hist
import cvk.cvk_convex_hull
import cvk.cvk_convexity_defects
import cvk.cvk_corner_eigen_vals_and_vecs
import cvk.cvk_cvt_color_two_plane
import cvk.cvk_distance_transform_with_labels
import cvk.cvk_estimate_affine_2d
import cvk.cvk_estimate_affine_partial_2d
import cvk.cvk_fit_line
import cvk.cvk_get_default_new_camera_matrix
import cvk.cvk_get_deriv_kernels
import cvk.cvk_get_gabor_kernel
import cvk.cvk_hu_moments
import cvk.cvk_init_undistort_rectify_map
import cvk.cvk_intersect_convex_convex
import cvk.cvk_is_contour_convex
import cvk.cvk_last_error
import cvk.cvk_mat_t
import cvk.cvk_moments
import cvk.cvk_point_polygon_test
import cvk.cvk_pre_corner_detect
import cvk.cvk_rotated_rectangle_intersection
import cvk.cvk_rotated_rect
import cvk.cvk_sep_filter_2d
import cvk.cvk_undistort_points
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.cValue
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value
import kotlinx.cinterop.memScoped

private fun lastNativeError(): String? {
    val message = cvk_last_error() ?: return null
    return message.toKString()
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
): Pair<Mat, Mat> = memScoped {
    val kx = alloc<CPointerVar<cvk_mat_t>>()
    val ky = alloc<CPointerVar<cvk_mat_t>>()
    if (cvk_get_deriv_kernels(dx, dy, ksize, if (normalize) 1 else 0, ktype, kx.ptr, ky.ptr) == 0) {
        throw OpenCVException("getDerivKernels", lastNativeError())
    }
    nativeMat(kx.value, "getDerivKernels") to nativeMat(ky.value, "getDerivKernels")
}

actual fun getGaborKernel(
    ksize: Size,
    sigma: Double,
    theta: Double,
    lambda: Double,
    gamma: Double,
    psi: Double,
    ktype: Int,
): Mat = nativeMat(
    cvk_get_gabor_kernel(ksize.width, ksize.height, sigma, theta, lambda, gamma, psi, ktype),
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
): Mat = nativeMat(
    cvk_sep_filter_2d(
        src.nativeHandle(), ddepth, kernelX.nativeHandle(), kernelY.nativeHandle(),
        anchorX, anchorY, delta, borderType,
    ),
    "sepFilter2D",
)

// =========================================================================
// corners
// =========================================================================

actual fun preCornerDetect(src: Mat, ksize: Int, borderType: Int): Mat =
    nativeMat(cvk_pre_corner_detect(src.nativeHandle(), ksize, borderType), "preCornerDetect")

actual fun cornerEigenValsAndVecs(src: Mat, blockSize: Int, ksize: Int, borderType: Int): Mat =
    nativeMat(
        cvk_corner_eigen_vals_and_vecs(src.nativeHandle(), blockSize, ksize, borderType),
        "cornerEigenValsAndVecs",
    )

// =========================================================================
// color conversion
// =========================================================================

actual fun cvtColorTwoPlane(src1: Mat, src2: Mat, code: Int): Mat =
    nativeMat(cvk_cvt_color_two_plane(src1.nativeHandle(), src2.nativeHandle(), code), "cvtColorTwoPlane")

// =========================================================================
// histogram
// =========================================================================

actual fun compareHist(h1: Mat, h2: Mat, method: Int): Double =
    cvk_compare_hist(h1.nativeHandle(), h2.nativeHandle(), method)

// =========================================================================
// distance transform
// =========================================================================

actual fun distanceTransformWithLabels(
    src: Mat,
    distanceType: Int,
    maskSize: Int,
    labelType: Int,
): Pair<Mat, Mat> = memScoped {
    val dst = alloc<CPointerVar<cvk_mat_t>>()
    val labels = alloc<CPointerVar<cvk_mat_t>>()
    if (cvk_distance_transform_with_labels(src.nativeHandle(), distanceType, maskSize, labelType, dst.ptr, labels.ptr) == 0) {
        throw OpenCVException("distanceTransformWithLabels", lastNativeError())
    }
    nativeMat(dst.value, "distanceTransformWithLabels") to nativeMat(labels.value, "distanceTransformWithLabels")
}

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
): Pair<Mat, Mat> = memScoped {
    val map1 = alloc<CPointerVar<cvk_mat_t>>()
    val map2 = alloc<CPointerVar<cvk_mat_t>>()
    val ok = cvk_init_undistort_rectify_map(
        cameraMatrix.nativeHandle(), distCoeffs.nativeHandle(),
        r?.nativeHandle(), newCameraMatrix?.nativeHandle(),
        size.width, size.height, m1type, map1.ptr, map2.ptr,
    )
    if (ok == 0) throw OpenCVException("initUndistortRectifyMap", lastNativeError())
    nativeMat(map1.value, "initUndistortRectifyMap") to nativeMat(map2.value, "initUndistortRectifyMap")
}

actual fun undistortPoints(src: Mat, cameraMatrix: Mat, distCoeffs: Mat, r: Mat?, p: Mat?): Mat =
    nativeMat(
        cvk_undistort_points(
            src.nativeHandle(), cameraMatrix.nativeHandle(), distCoeffs.nativeHandle(),
            r?.nativeHandle(), p?.nativeHandle(),
        ),
        "undistortPoints",
    )

actual fun getDefaultNewCameraMatrix(
    cameraMatrix: Mat,
    imgsize: Size,
    centerPrincipalPoint: Boolean,
): Mat = nativeMat(
    cvk_get_default_new_camera_matrix(
        cameraMatrix.nativeHandle(), imgsize.width, imgsize.height,
        if (centerPrincipalPoint) 1 else 0,
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
): Mat? = cvk_estimate_affine_2d(
    from.nativeHandle(), to.nativeHandle(), method, ransacReprojThreshold,
    maxIters.toLong(), confidence, refineIters.toLong(),
)?.let { nativeMat(it, "estimateAffine2D") }

actual fun estimateAffinePartial2D(
    from: Mat,
    to: Mat,
    method: Int,
    ransacReprojThreshold: Double,
    maxIters: Int,
    confidence: Double,
    refineIters: Int,
): Mat? = cvk_estimate_affine_partial_2d(
    from.nativeHandle(), to.nativeHandle(), method, ransacReprojThreshold,
    maxIters.toLong(), confidence, refineIters.toLong(),
)?.let { nativeMat(it, "estimateAffinePartial2D") }

// =========================================================================
// contours / geometry
// =========================================================================

actual fun convexHull(points: Mat, clockwise: Boolean, returnPoints: Boolean): Mat =
    nativeMat(
        cvk_convex_hull(points.nativeHandle(), if (clockwise) 1 else 0, if (returnPoints) 1 else 0),
        "convexHull",
    )

actual fun isContourConvex(points: Mat): Boolean =
    cvk_is_contour_convex(points.nativeHandle()) != 0

actual fun convexityDefects(contour: Mat, convexHullIdx: Mat): Mat =
    nativeMat(cvk_convexity_defects(contour.nativeHandle(), convexHullIdx.nativeHandle()), "convexityDefects")

actual fun fitLine(points: Mat, distType: Int, param: Double, reps: Double, aeps: Double): Mat =
    nativeMat(cvk_fit_line(points.nativeHandle(), distType, param, reps, aeps), "fitLine")

actual fun boxPoints(box: RotatedRect): Mat {
    val rect = cValue<cvk_rotated_rect> {
        cx = box.centerX
        cy = box.centerY
        width = box.width
        height = box.height
        angle = box.angle
    }
    return nativeMat(cvk_box_points(rect), "boxPoints")
}

actual fun rotatedRectangleIntersection(rect1: RotatedRect, rect2: RotatedRect): RotatedRectIntersection = memScoped {
    val first = cValue<cvk_rotated_rect> {
        cx = rect1.centerX
        cy = rect1.centerY
        width = rect1.width
        height = rect1.height
        angle = rect1.angle
    }
    val second = cValue<cvk_rotated_rect> {
        cx = rect2.centerX
        cy = rect2.centerY
        width = rect2.width
        height = rect2.height
        angle = rect2.angle
    }
    val polygon = alloc<CPointerVar<cvk_mat_t>>()
    val type = cvk_rotated_rectangle_intersection(first, second, polygon.ptr)
    RotatedRectIntersection(type, nativeMat(polygon.value, "rotatedRectangleIntersection"))
}

actual fun pointPolygonTest(contour: Mat, x: Double, y: Double, measureDist: Boolean): Double =
    cvk_point_polygon_test(contour.nativeHandle(), x, y, if (measureDist) 1 else 0)

actual fun intersectConvexConvex(p1: Mat, p2: Mat, handleNested: Boolean): ConvexIntersection = memScoped {
    val polygon = alloc<CPointerVar<cvk_mat_t>>()
    val area = cvk_intersect_convex_convex(
        p1.nativeHandle(), p2.nativeHandle(), if (handleNested) 1 else 0, polygon.ptr,
    )
    ConvexIntersection(area, nativeMat(polygon.value, "intersectConvexConvex"))
}

actual fun huMoments(moments: Moments): DoubleArray = memScoped {
    val out = allocArray<DoubleVar>(7)
    val m = cValue<cvk_moments> {
        m00 = moments.m00
        m10 = moments.m10
        m01 = moments.m01
        m20 = moments.m20
        m11 = moments.m11
        m02 = moments.m02
        m30 = moments.m30
        m21 = moments.m21
        m12 = moments.m12
        m03 = moments.m03
    }
    cvk_hu_moments(m, out)
    DoubleArray(7) { index -> out[index] }
}

// =========================================================================
// pyramids
// =========================================================================

actual fun buildPyramid(src: Mat, maxLevel: Int, borderType: Int): List<Mat> = memScoped {
    val count = maxLevel + 1
    val levels = allocArray<CPointerVar<cvk_mat_t>>(count)
    val written = cvk_build_pyramid(src.nativeHandle(), maxLevel, borderType, levels, count)
    if (written <= 0) throw OpenCVException("buildPyramid", lastNativeError())
    List(written) { index -> nativeMat(levels[index], "buildPyramid") }
}

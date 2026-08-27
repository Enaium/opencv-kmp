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
 * JNI bridge for the imgproc statics slice (JVM target).
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniImgproc_<name>`
 * function in jni/jni_imgproc.cpp. Mat handles travel as jlong pointers;
 * functions producing several matrices return a `LongArray` of handles.
 *
 * No `init` block: the shared [Jni] object already loaded the native library.
 */
internal object JniImgproc {

    // kernels / filters

    external fun getDerivKernels(dx: Int, dy: Int, ksize: Int, normalize: Boolean, ktype: Int): LongArray
    external fun getGaborKernel(
        width: Int,
        height: Int,
        sigma: Double,
        theta: Double,
        lambda: Double,
        gamma: Double,
        psi: Double,
        ktype: Int,
    ): Long

    external fun sepFilter2D(
        src: Long,
        ddepth: Int,
        kernelX: Long,
        kernelY: Long,
        anchorX: Int,
        anchorY: Int,
        delta: Double,
        borderType: Int,
    ): Long

    // corners

    external fun preCornerDetect(src: Long, ksize: Int, borderType: Int): Long
    external fun cornerEigenValsAndVecs(src: Long, blockSize: Int, ksize: Int, borderType: Int): Long

    // color conversion

    external fun cvtColorTwoPlane(src1: Long, src2: Long, code: Int): Long

    // calibration helpers

    external fun initUndistortRectifyMap(
        cameraMatrix: Long,
        distCoeffs: Long,
        r: Long,
        newCameraMatrix: Long,
        width: Int,
        height: Int,
        m1type: Int,
    ): LongArray

    external fun undistortPoints(src: Long, cameraMatrix: Long, distCoeffs: Long, r: Long, p: Long): Long
    external fun getDefaultNewCameraMatrix(cameraMatrix: Long, width: Int, height: Int, centerPrincipalPoint: Boolean): Long
    external fun estimateAffine2D(
        from: Long,
        to: Long,
        method: Int,
        ransacReprojThreshold: Double,
        maxIters: Long,
        confidence: Double,
        refineIters: Long,
    ): Long

    external fun estimateAffinePartial2D(
        from: Long,
        to: Long,
        method: Int,
        ransacReprojThreshold: Double,
        maxIters: Long,
        confidence: Double,
        refineIters: Long,
    ): Long

    // distance transform

    external fun distanceTransformWithLabels(src: Long, distanceType: Int, maskSize: Int, labelType: Int): LongArray

    // contours / geometry

    external fun convexHull(points: Long, clockwise: Boolean, returnPoints: Boolean): Long
    external fun isContourConvex(points: Long): Boolean
    external fun convexityDefects(contour: Long, hullIdx: Long): Long
    external fun fitLine(points: Long, distType: Int, param: Double, reps: Double, aeps: Double): Long
    external fun boxPoints(cx: Double, cy: Double, width: Double, height: Double, angle: Double): Long
    external fun rotatedRectangleIntersection(
        r1cx: Double, r1cy: Double, r1w: Double, r1h: Double, r1a: Double,
        r2cx: Double, r2cy: Double, r2w: Double, r2h: Double, r2a: Double,
    ): LongArray

    external fun pointPolygonTest(contour: Long, x: Double, y: Double, measureDist: Boolean): Double

    /** Returns `[polygon handle, float area bits]`. */
    external fun intersectConvexConvex(p1: Long, p2: Long, handleNested: Boolean): LongArray
    external fun huMoments(
        m00: Double, m10: Double, m01: Double, m20: Double, m11: Double,
        m02: Double, m30: Double, m21: Double, m12: Double, m03: Double,
    ): DoubleArray

    // pyramids

    external fun buildPyramid(src: Long, maxLevel: Int, borderType: Int): LongArray
}

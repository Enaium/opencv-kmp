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
 * JNI bridge for the calib slice. Every `external fun` maps 1:1 to a
 * `Java_cn_enaium_opencv_JniCalib_<name>` function in jni/jni_calib.cpp.
 * Mat handles travel as jlong pointers; calibration failures surface as
 * NaN return values (the shim never throws across the boundary).
 */
internal object JniCalib {

    external fun initCameraMatrix2D(
        objectPoints: Long, imagePoints: Long,
        imageWidth: Int, imageHeight: Int, aspectRatio: Double,
    ): Long

    external fun calibrateCamera(
        objectPoints: Long, imagePoints: Long,
        imageWidth: Int, imageHeight: Int,
        cameraMatrix: Long, distCoeffs: Long, rvecs: Long, tvecs: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun calibrateCameraExtended(
        objectPoints: Long, imagePoints: Long,
        imageWidth: Int, imageHeight: Int,
        cameraMatrix: Long, distCoeffs: Long, rvecs: Long, tvecs: Long,
        stdDeviationsIntrinsics: Long, stdDeviationsExtrinsics: Long, perViewErrors: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun calibrateCameraRO(
        objectPoints: Long, imagePoints: Long,
        imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
        cameraMatrix: Long, distCoeffs: Long, rvecs: Long, tvecs: Long, newObjPoints: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun calibrateCameraROExtended(
        objectPoints: Long, imagePoints: Long,
        imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
        cameraMatrix: Long, distCoeffs: Long, rvecs: Long, tvecs: Long, newObjPoints: Long,
        stdDeviationsIntrinsics: Long, stdDeviationsExtrinsics: Long,
        stdDeviationsObjPoints: Long, perViewErrors: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun stereoCalibrate(
        objectPoints: Long, imagePoints1: Long, imagePoints2: Long,
        imageWidth: Int, imageHeight: Int,
        cameraMatrix1: Long, distCoeffs1: Long, cameraMatrix2: Long, distCoeffs2: Long,
        r: Long, t: Long, e: Long, f: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun stereoCalibratePerView(
        objectPoints: Long, imagePoints1: Long, imagePoints2: Long,
        imageWidth: Int, imageHeight: Int,
        cameraMatrix1: Long, distCoeffs1: Long, cameraMatrix2: Long, distCoeffs2: Long,
        r: Long, t: Long, e: Long, f: Long, perViewErrors: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun stereoCalibrateExtended(
        objectPoints: Long, imagePoints1: Long, imagePoints2: Long,
        imageWidth: Int, imageHeight: Int,
        cameraMatrix1: Long, distCoeffs1: Long, cameraMatrix2: Long, distCoeffs2: Long,
        r: Long, t: Long, e: Long, f: Long, rvecs: Long, tvecs: Long, perViewErrors: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun registerCameras(
        objectPoints1: Long, objectPoints2: Long, imagePoints1: Long, imagePoints2: Long,
        cameraMatrix1: Long, distCoeffs1: Long, cameraModel1: Int,
        cameraMatrix2: Long, distCoeffs2: Long, cameraModel2: Int,
        r: Long, t: Long, e: Long, f: Long, perViewErrors: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun registerCamerasExtended(
        objectPoints1: Long, objectPoints2: Long, imagePoints1: Long, imagePoints2: Long,
        cameraMatrix1: Long, distCoeffs1: Long, cameraModel1: Int,
        cameraMatrix2: Long, distCoeffs2: Long, cameraModel2: Int,
        r: Long, t: Long, e: Long, f: Long, rvecs: Long, tvecs: Long, perViewErrors: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun fisheyeCalibrate(
        objectPoints: Long, imagePoints: Long,
        imageWidth: Int, imageHeight: Int,
        k: Long, d: Long, rvecs: Long, tvecs: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun fisheyeStereoCalibrate(
        objectPoints: Long, imagePoints1: Long, imagePoints2: Long,
        imageWidth: Int, imageHeight: Int,
        k1: Long, d1: Long, k2: Long, d2: Long, r: Long, t: Long, rvecs: Long, tvecs: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double

    external fun fisheyeStereoCalibratePose(
        objectPoints: Long, imagePoints1: Long, imagePoints2: Long,
        imageWidth: Int, imageHeight: Int,
        k1: Long, d1: Long, k2: Long, d2: Long, r: Long, t: Long,
        flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
    ): Double
}

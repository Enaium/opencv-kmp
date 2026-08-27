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
// JVM (JNI-backed) actuals for the calib slice. Mat handles are jlong
// pointers to cv::Mat; every call goes through [JniCalib], which forwards
// to the same cvk_ shim the native targets bind via cinterop.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()


internal actual fun matListToWire(mats: List<Mat>): Mat {
    val wire = mat(mats.size, 1, MatType.of(CV_32S, 2))
    if (mats.isEmpty()) return wire
    val bytes = ByteArray(mats.size * 8)
    var b = 0
    for (m in mats) {
        val addr = handleOf(m)
        bytes.writeIntLE(b, (addr ushr 32).toInt())
        bytes.writeIntLE(b + 4, (addr and 0xFFFF_FFFFL).toInt())
        b += 8
    }
    wire.pixels = bytes
    return wire
}

internal actual fun wireToMatList(wire: Mat): List<Mat> {
    val bytes = wire.pixels
    val count = bytes.size / 8
    val out = ArrayList<Mat>(count)
    var b = 0
    repeat(count) {
        val hi = bytes.readIntLE(b).toLong() and 0xFFFF_FFFFL
        val lo = bytes.readIntLE(b + 4).toLong() and 0xFFFF_FFFFL
        out.add(JvmMat((hi shl 32) or lo))
        b += 8
    }
    return out
}

internal actual fun initCameraMatrix2DNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, aspectRatio: Double,
): Mat = jvmMat(
    JniCalib.initCameraMatrix2D(handleOf(objectPoints), handleOf(imagePoints), imageWidth, imageHeight, aspectRatio),
    "initCameraMatrix2D",
)

internal actual fun calibrateCameraNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.calibrateCamera(
    handleOf(objectPoints), handleOf(imagePoints), imageWidth, imageHeight,
    handleOf(cameraMatrix), handleOf(distCoeffs), handleOf(rvecs), handleOf(tvecs),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun calibrateCameraExtendedNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat,
    stdDeviationsIntrinsics: Mat, stdDeviationsExtrinsics: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.calibrateCameraExtended(
    handleOf(objectPoints), handleOf(imagePoints), imageWidth, imageHeight,
    handleOf(cameraMatrix), handleOf(distCoeffs), handleOf(rvecs), handleOf(tvecs),
    handleOf(stdDeviationsIntrinsics), handleOf(stdDeviationsExtrinsics), handleOf(perViewErrors),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun calibrateCameraRoNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, newObjPoints: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.calibrateCameraRO(
    handleOf(objectPoints), handleOf(imagePoints), imageWidth, imageHeight, iFixedPoint,
    handleOf(cameraMatrix), handleOf(distCoeffs), handleOf(rvecs), handleOf(tvecs), handleOf(newObjPoints),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun calibrateCameraRoExtendedNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, newObjPoints: Mat,
    stdDeviationsIntrinsics: Mat, stdDeviationsExtrinsics: Mat,
    stdDeviationsObjPoints: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.calibrateCameraROExtended(
    handleOf(objectPoints), handleOf(imagePoints), imageWidth, imageHeight, iFixedPoint,
    handleOf(cameraMatrix), handleOf(distCoeffs), handleOf(rvecs), handleOf(tvecs), handleOf(newObjPoints),
    handleOf(stdDeviationsIntrinsics), handleOf(stdDeviationsExtrinsics),
    handleOf(stdDeviationsObjPoints), handleOf(perViewErrors),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun stereoCalibrateNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.stereoCalibrate(
    handleOf(objectPoints), handleOf(imagePoints1), handleOf(imagePoints2), imageWidth, imageHeight,
    handleOf(cameraMatrix1), handleOf(distCoeffs1), handleOf(cameraMatrix2), handleOf(distCoeffs2),
    handleOf(r), handleOf(t), handleOf(e), handleOf(f),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun stereoCalibratePerViewNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.stereoCalibratePerView(
    handleOf(objectPoints), handleOf(imagePoints1), handleOf(imagePoints2), imageWidth, imageHeight,
    handleOf(cameraMatrix1), handleOf(distCoeffs1), handleOf(cameraMatrix2), handleOf(distCoeffs2),
    handleOf(r), handleOf(t), handleOf(e), handleOf(f), handleOf(perViewErrors),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun stereoCalibrateExtendedNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat, rvecs: Mat, tvecs: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.stereoCalibrateExtended(
    handleOf(objectPoints), handleOf(imagePoints1), handleOf(imagePoints2), imageWidth, imageHeight,
    handleOf(cameraMatrix1), handleOf(distCoeffs1), handleOf(cameraMatrix2), handleOf(distCoeffs2),
    handleOf(r), handleOf(t), handleOf(e), handleOf(f), handleOf(rvecs), handleOf(tvecs), handleOf(perViewErrors),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun registerCamerasNative(
    objectPoints1: Mat, objectPoints2: Mat, imagePoints1: Mat, imagePoints2: Mat,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraModel1: Int,
    cameraMatrix2: Mat, distCoeffs2: Mat, cameraModel2: Int,
    r: Mat, t: Mat, e: Mat, f: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.registerCameras(
    handleOf(objectPoints1), handleOf(objectPoints2), handleOf(imagePoints1), handleOf(imagePoints2),
    handleOf(cameraMatrix1), handleOf(distCoeffs1), cameraModel1,
    handleOf(cameraMatrix2), handleOf(distCoeffs2), cameraModel2,
    handleOf(r), handleOf(t), handleOf(e), handleOf(f), handleOf(perViewErrors),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun registerCamerasExtendedNative(
    objectPoints1: Mat, objectPoints2: Mat, imagePoints1: Mat, imagePoints2: Mat,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraModel1: Int,
    cameraMatrix2: Mat, distCoeffs2: Mat, cameraModel2: Int,
    r: Mat, t: Mat, e: Mat, f: Mat, rvecs: Mat, tvecs: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.registerCamerasExtended(
    handleOf(objectPoints1), handleOf(objectPoints2), handleOf(imagePoints1), handleOf(imagePoints2),
    handleOf(cameraMatrix1), handleOf(distCoeffs1), cameraModel1,
    handleOf(cameraMatrix2), handleOf(distCoeffs2), cameraModel2,
    handleOf(r), handleOf(t), handleOf(e), handleOf(f), handleOf(rvecs), handleOf(tvecs), handleOf(perViewErrors),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun fisheyeCalibrateNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    k: Mat, d: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.fisheyeCalibrate(
    handleOf(objectPoints), handleOf(imagePoints), imageWidth, imageHeight,
    handleOf(k), handleOf(d), handleOf(rvecs), handleOf(tvecs),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun fisheyeStereoCalibrateNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    k1: Mat, d1: Mat, k2: Mat, d2: Mat, r: Mat, t: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.fisheyeStereoCalibrate(
    handleOf(objectPoints), handleOf(imagePoints1), handleOf(imagePoints2), imageWidth, imageHeight,
    handleOf(k1), handleOf(d1), handleOf(k2), handleOf(d2), handleOf(r), handleOf(t),
    handleOf(rvecs), handleOf(tvecs),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun fisheyeStereoCalibratePoseNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    k1: Mat, d1: Mat, k2: Mat, d2: Mat, r: Mat, t: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = JniCalib.fisheyeStereoCalibratePose(
    handleOf(objectPoints), handleOf(imagePoints1), handleOf(imagePoints2), imageWidth, imageHeight,
    handleOf(k1), handleOf(d1), handleOf(k2), handleOf(d2), handleOf(r), handleOf(t),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

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

import cvk.cvk_calibrate_camera
import cvk.cvk_calibrate_camera_extended
import cvk.cvk_calibrate_camera_ro
import cvk.cvk_calibrate_camera_ro_extended
import cvk.cvk_fisheye_calibrate
import cvk.cvk_fisheye_stereo_calibrate
import cvk.cvk_fisheye_stereo_calibrate_pose
import cvk.cvk_init_camera_matrix_2d
import cvk.cvk_mat_t
import cvk.cvk_register_cameras
import cvk.cvk_register_cameras_extended
import cvk.cvk_stereo_calibrate
import cvk.cvk_stereo_calibrate_extended
import cvk.cvk_stereo_calibrate_per_view
import cvk.cvk_last_error
import kotlinx.cinterop.toCPointer
import kotlinx.cinterop.toKString

// =========================================================================
// Native (cinterop) actuals for the calib slice.
// =========================================================================

private fun lastNativeError(): String? {
    val message = cvk_last_error() ?: return null
    return message.toKString()
}

internal actual fun matListToWire(mats: List<Mat>): Mat {
    val wire = mat(mats.size, 1, MatType.of(CV_32S, 2))
    if (mats.isEmpty()) return wire
    val bytes = ByteArray(mats.size * 8)
    var b = 0
    for (m in mats) {
        val addr = m.nativeHandle().rawValue.toLong()
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
        val addr = (hi shl 32) or lo
        out.add(NativeMat(addr.toCPointer<cvk_mat_t>() ?: throw OpenCVException("wireToMatList", lastNativeError())))
        b += 8
    }
    return out
}

internal actual fun initCameraMatrix2DNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, aspectRatio: Double,
): Mat = nativeMat(
    cvk_init_camera_matrix_2d(objectPoints.nativeHandle(), imagePoints.nativeHandle(),
                              imageWidth, imageHeight, aspectRatio),
    "initCameraMatrix2D",
)

internal actual fun calibrateCameraNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_calibrate_camera(
    objectPoints.nativeHandle(), imagePoints.nativeHandle(), imageWidth, imageHeight,
    cameraMatrix.nativeHandle(), distCoeffs.nativeHandle(),
    rvecs.nativeHandle(), tvecs.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun calibrateCameraExtendedNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat,
    stdDeviationsIntrinsics: Mat, stdDeviationsExtrinsics: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_calibrate_camera_extended(
    objectPoints.nativeHandle(), imagePoints.nativeHandle(), imageWidth, imageHeight,
    cameraMatrix.nativeHandle(), distCoeffs.nativeHandle(),
    rvecs.nativeHandle(), tvecs.nativeHandle(),
    stdDeviationsIntrinsics.nativeHandle(), stdDeviationsExtrinsics.nativeHandle(),
    perViewErrors.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun calibrateCameraRoNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, newObjPoints: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_calibrate_camera_ro(
    objectPoints.nativeHandle(), imagePoints.nativeHandle(), imageWidth, imageHeight, iFixedPoint,
    cameraMatrix.nativeHandle(), distCoeffs.nativeHandle(),
    rvecs.nativeHandle(), tvecs.nativeHandle(), newObjPoints.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun calibrateCameraRoExtendedNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, newObjPoints: Mat,
    stdDeviationsIntrinsics: Mat, stdDeviationsExtrinsics: Mat,
    stdDeviationsObjPoints: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_calibrate_camera_ro_extended(
    objectPoints.nativeHandle(), imagePoints.nativeHandle(), imageWidth, imageHeight, iFixedPoint,
    cameraMatrix.nativeHandle(), distCoeffs.nativeHandle(),
    rvecs.nativeHandle(), tvecs.nativeHandle(), newObjPoints.nativeHandle(),
    stdDeviationsIntrinsics.nativeHandle(), stdDeviationsExtrinsics.nativeHandle(),
    stdDeviationsObjPoints.nativeHandle(), perViewErrors.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun stereoCalibrateNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_stereo_calibrate(
    objectPoints.nativeHandle(), imagePoints1.nativeHandle(), imagePoints2.nativeHandle(),
    imageWidth, imageHeight,
    cameraMatrix1.nativeHandle(), distCoeffs1.nativeHandle(),
    cameraMatrix2.nativeHandle(), distCoeffs2.nativeHandle(),
    r.nativeHandle(), t.nativeHandle(), e.nativeHandle(), f.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun stereoCalibratePerViewNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_stereo_calibrate_per_view(
    objectPoints.nativeHandle(), imagePoints1.nativeHandle(), imagePoints2.nativeHandle(),
    imageWidth, imageHeight,
    cameraMatrix1.nativeHandle(), distCoeffs1.nativeHandle(),
    cameraMatrix2.nativeHandle(), distCoeffs2.nativeHandle(),
    r.nativeHandle(), t.nativeHandle(), e.nativeHandle(), f.nativeHandle(),
    perViewErrors.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun stereoCalibrateExtendedNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat, rvecs: Mat, tvecs: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_stereo_calibrate_extended(
    objectPoints.nativeHandle(), imagePoints1.nativeHandle(), imagePoints2.nativeHandle(),
    imageWidth, imageHeight,
    cameraMatrix1.nativeHandle(), distCoeffs1.nativeHandle(),
    cameraMatrix2.nativeHandle(), distCoeffs2.nativeHandle(),
    r.nativeHandle(), t.nativeHandle(), e.nativeHandle(), f.nativeHandle(),
    rvecs.nativeHandle(), tvecs.nativeHandle(), perViewErrors.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun registerCamerasNative(
    objectPoints1: Mat, objectPoints2: Mat, imagePoints1: Mat, imagePoints2: Mat,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraModel1: Int,
    cameraMatrix2: Mat, distCoeffs2: Mat, cameraModel2: Int,
    r: Mat, t: Mat, e: Mat, f: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_register_cameras(
    objectPoints1.nativeHandle(), objectPoints2.nativeHandle(),
    imagePoints1.nativeHandle(), imagePoints2.nativeHandle(),
    cameraMatrix1.nativeHandle(), distCoeffs1.nativeHandle(), cameraModel1,
    cameraMatrix2.nativeHandle(), distCoeffs2.nativeHandle(), cameraModel2,
    r.nativeHandle(), t.nativeHandle(), e.nativeHandle(), f.nativeHandle(),
    perViewErrors.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun registerCamerasExtendedNative(
    objectPoints1: Mat, objectPoints2: Mat, imagePoints1: Mat, imagePoints2: Mat,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraModel1: Int,
    cameraMatrix2: Mat, distCoeffs2: Mat, cameraModel2: Int,
    r: Mat, t: Mat, e: Mat, f: Mat, rvecs: Mat, tvecs: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_register_cameras_extended(
    objectPoints1.nativeHandle(), objectPoints2.nativeHandle(),
    imagePoints1.nativeHandle(), imagePoints2.nativeHandle(),
    cameraMatrix1.nativeHandle(), distCoeffs1.nativeHandle(), cameraModel1,
    cameraMatrix2.nativeHandle(), distCoeffs2.nativeHandle(), cameraModel2,
    r.nativeHandle(), t.nativeHandle(), e.nativeHandle(), f.nativeHandle(),
    rvecs.nativeHandle(), tvecs.nativeHandle(), perViewErrors.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun fisheyeCalibrateNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    k: Mat, d: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_fisheye_calibrate(
    objectPoints.nativeHandle(), imagePoints.nativeHandle(), imageWidth, imageHeight,
    k.nativeHandle(), d.nativeHandle(), rvecs.nativeHandle(), tvecs.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun fisheyeStereoCalibrateNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    k1: Mat, d1: Mat, k2: Mat, d2: Mat, r: Mat, t: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_fisheye_stereo_calibrate(
    objectPoints.nativeHandle(), imagePoints1.nativeHandle(), imagePoints2.nativeHandle(),
    imageWidth, imageHeight,
    k1.nativeHandle(), d1.nativeHandle(), k2.nativeHandle(), d2.nativeHandle(),
    r.nativeHandle(), t.nativeHandle(), rvecs.nativeHandle(), tvecs.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

internal actual fun fisheyeStereoCalibratePoseNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    k1: Mat, d1: Mat, k2: Mat, d2: Mat, r: Mat, t: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double = cvk_fisheye_stereo_calibrate_pose(
    objectPoints.nativeHandle(), imagePoints1.nativeHandle(), imagePoints2.nativeHandle(),
    imageWidth, imageHeight,
    k1.nativeHandle(), d1.nativeHandle(), k2.nativeHandle(), d2.nativeHandle(),
    r.nativeHandle(), t.nativeHandle(),
    flags, criteriaType, criteriaMaxCount, criteriaEpsilon,
)

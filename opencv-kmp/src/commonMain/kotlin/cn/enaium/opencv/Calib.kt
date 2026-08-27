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
 * Camera-calibration utilities (`org.opencv.calib.Calib`): monocular,
 * stereo and fisheye calibration over planar calibration patterns.
 *
 * Point-list arguments travel as CV_32FC3 (object) / CV_32FC2 (image) Mats
 * — one per view, e.g. `MatOfPoint3f`/`MatOfPoint2f` instances. Output
 * Mats are written into the handles the caller supplies (mirroring the SDK's
 * in/out parameters); multi-Mat outputs are additionally bundled into the
 * returned result data classes. Per-view rotation/translation vectors come
 * back as `List<Mat>` of 3x1 CV_64F Mats (each owned by the caller).
 *
 * Every function throws [OpenCVException] when the native call fails
 * (under-constrained input, mismatched point counts, ...); the returned RMS
 * re-projection error is otherwise finite and non-negative.
 */
object Calib {

    /** `cv::CALIB_USE_INTRINSIC_GUESS`: cameraMatrix holds valid initial fx, fy, cx, cy. */
    const val CALIB_USE_INTRINSIC_GUESS: Int = 0x00001

    /** `cv::CALIB_FIX_ASPECT_RATIO`: only fy is a free parameter; fx/fy stays fixed. */
    const val CALIB_FIX_ASPECT_RATIO: Int = 0x00002

    /** `cv::CALIB_FIX_PRINCIPAL_POINT`: the principal point is not changed. */
    const val CALIB_FIX_PRINCIPAL_POINT: Int = 0x00004

    /** `cv::CALIB_ZERO_TANGENT_DIST`: tangential distortion coefficients stay zero. */
    const val CALIB_ZERO_TANGENT_DIST: Int = 0x00008

    /** `cv::CALIB_FIX_FOCAL_LENGTH`: focal length is not changed during optimization. */
    const val CALIB_FIX_FOCAL_LENGTH: Int = 0x00010

    /** `cv::CALIB_FIX_K1`: radial coefficient k1 is not changed. */
    const val CALIB_FIX_K1: Int = 0x00020

    /** `cv::CALIB_FIX_K2`: radial coefficient k2 is not changed. */
    const val CALIB_FIX_K2: Int = 0x00040

    /** `cv::CALIB_FIX_K3`: radial coefficient k3 is not changed. */
    const val CALIB_FIX_K3: Int = 0x00080

    /** `cv::CALIB_FIX_INTRINSIC`: fix cameraMatrix/distCoeffs, estimate only R, T, E, F. */
    const val CALIB_FIX_INTRINSIC: Int = 0x00100

    /** `cv::CALIB_SAME_FOCAL_LENGTH`: enforce fx1=fx2 and fy1=fy2 for a stereo pair. */
    const val CALIB_SAME_FOCAL_LENGTH: Int = 0x00200

    /** `cv::CALIB_ZERO_DISPARITY`: (stereo rectification) zero-disparity flag. */
    const val CALIB_ZERO_DISPARITY: Int = 0x00400

    /** `cv::CALIB_FIX_K4`: radial coefficient k4 is not changed. */
    const val CALIB_FIX_K4: Int = 0x00800

    /** `cv::CALIB_FIX_K5`: radial coefficient k5 is not changed. */
    const val CALIB_FIX_K5: Int = 0x01000

    /** `cv::CALIB_FIX_K6`: radial coefficient k6 is not changed. */
    const val CALIB_FIX_K6: Int = 0x02000

    /** `cv::CALIB_RATIONAL_MODEL`: enable k4, k5, k6 (8+ distortion coefficients). */
    const val CALIB_RATIONAL_MODEL: Int = 0x04000

    /** `cv::CALIB_THIN_PRISM_MODEL`: enable s1..s4 (12+ distortion coefficients). */
    const val CALIB_THIN_PRISM_MODEL: Int = 0x08000

    /** `cv::CALIB_FIX_S1_S2_S3_S4`: thin-prism coefficients are not changed. */
    const val CALIB_FIX_S1_S2_S3_S4: Int = 0x10000

    /** `cv::CALIB_TILTED_MODEL`: enable tauX/tauY (14 distortion coefficients). */
    const val CALIB_TILTED_MODEL: Int = 0x40000

    /** `cv::CALIB_FIX_TAUX_TAUY`: tilted-sensor coefficients are not changed. */
    const val CALIB_FIX_TAUX_TAUY: Int = 0x80000

    /** `cv::CALIB_USE_QR`: use the QR solver instead of the default Cholesky. */
    const val CALIB_USE_QR: Int = 0x100000

    /** `cv::CALIB_FIX_TANGENT_DIST`: tangential distortion coefficients are not changed. */
    const val CALIB_FIX_TANGENT_DIST: Int = 0x200000

    /** `cv::CALIB_USE_LU`: use the LU solver instead of the default Cholesky. */
    const val CALIB_USE_LU: Int = 1 shl 17

    /** `cv::CALIB_DISABLE_SCHUR_COMPLEMENT`: use the Bouguet calibration engine. */
    const val CALIB_DISABLE_SCHUR_COMPLEMENT: Int = 1 shl 18

    /** `cv::CALIB_USE_EXTRINSIC_GUESS`: R and T contain valid initial values. */
    const val CALIB_USE_EXTRINSIC_GUESS: Int = 1 shl 22

    /** `cv::CALIB_RECOMPUTE_EXTRINSIC` (fisheye): recompute extrinsics each iteration. */
    const val CALIB_RECOMPUTE_EXTRINSIC: Int = 1 shl 23

    /** `cv::CALIB_CHECK_COND` (fisheye): check SVD condition number per frame. */
    const val CALIB_CHECK_COND: Int = 1 shl 24

    /** `cv::CALIB_FIX_SKEW` (fisheye): skew coefficient is set to zero and stays zero. */
    const val CALIB_FIX_SKEW: Int = 1 shl 25

    /** `cv::CALIB_STEREO_REGISTRATION`: (stereo) registration flag. */
    const val CALIB_STEREO_REGISTRATION: Int = 1 shl 26

    /** `cv::CameraModel::CALIB_MODEL_PINHOLE`: pinhole camera model (registerCameras). */
    const val CALIB_MODEL_PINHOLE: Int = 0

    /** `cv::CameraModel::CALIB_MODEL_FISHEYE`: fisheye camera model (registerCameras). */
    const val CALIB_MODEL_FISHEYE: Int = 1

    /** `cv::HandEyeCalibrationMethod::CALIB_HAND_EYE_TSAI`. */
    const val CALIB_HAND_EYE_TSAI: Int = 0

    /** `cv::HandEyeCalibrationMethod::CALIB_HAND_EYE_PARK`. */
    const val CALIB_HAND_EYE_PARK: Int = 1

    /** `cv::HandEyeCalibrationMethod::CALIB_HAND_EYE_HORAUD`. */
    const val CALIB_HAND_EYE_HORAUD: Int = 2

    /** `cv::HandEyeCalibrationMethod::CALIB_HAND_EYE_ANDREFF`. */
    const val CALIB_HAND_EYE_ANDREFF: Int = 3

    /** `cv::HandEyeCalibrationMethod::CALIB_HAND_EYE_DANIILIDIS`. */
    const val CALIB_HAND_EYE_DANIILIDIS: Int = 4

    /** `cv::RobotWorldHandEyeCalibrationMethod::CALIB_ROBOT_WORLD_HAND_EYE_SHAH`. */
    const val CALIB_ROBOT_WORLD_HAND_EYE_SHAH: Int = 0

    /** `cv::RobotWorldHandEyeCalibrationMethod::CALIB_ROBOT_WORLD_HAND_EYE_LI`. */
    const val CALIB_ROBOT_WORLD_HAND_EYE_LI: Int = 1
}

// =========================================================================
// Result data classes
// =========================================================================

/**
 * Output of [calibrateCamera].
 *
 * [cameraMatrix] and [distCoeffs] are the same Mats passed in (filled in
 * place); [rvecs]/[tvecs] are per-view 3x1 CV_64F Mats owned by the caller.
 */
data class CalibrateCameraResult(
    /** Overall RMS re-projection error. */
    val rms: Double,
    /** Output 3x3 camera intrinsic matrix. */
    val cameraMatrix: Mat,
    /** Output vector of distortion coefficients. */
    val distCoeffs: Mat,
    /** Per-view rotation vectors (Rodrigues). */
    val rvecs: List<Mat>,
    /** Per-view translation vectors. */
    val tvecs: List<Mat>,
)

/**
 * Output of [calibrateCameraExtended]: [CalibrateCameraResult] plus the
 * per-parameter standard deviations and per-view RMS errors.
 */
data class CalibrateCameraExtendedResult(
    /** Overall RMS re-projection error. */
    val rms: Double,
    /** Output 3x3 camera intrinsic matrix. */
    val cameraMatrix: Mat,
    /** Output vector of distortion coefficients. */
    val distCoeffs: Mat,
    /** Per-view rotation vectors (Rodrigues). */
    val rvecs: List<Mat>,
    /** Per-view translation vectors. */
    val tvecs: List<Mat>,
    /** Standard deviations of the intrinsic parameters (fx, fy, cx, cy, k1..k6, p1, p2, ...). */
    val stdDeviationsIntrinsics: Mat,
    /** Standard deviations of the extrinsic parameters (R0, T0, ..., R(M-1), T(M-1)). */
    val stdDeviationsExtrinsics: Mat,
    /** RMS re-projection error of each pattern view. */
    val perViewErrors: Mat,
)

/**
 * Output of [calibrateCameraRO]: [CalibrateCameraResult] plus the refined
 * object points produced by the object-releasing method.
 */
data class CalibrateCameraRoResult(
    /** Overall RMS re-projection error. */
    val rms: Double,
    /** Output 3x3 camera intrinsic matrix. */
    val cameraMatrix: Mat,
    /** Output vector of distortion coefficients. */
    val distCoeffs: Mat,
    /** Per-view rotation vectors (Rodrigues). */
    val rvecs: List<Mat>,
    /** Per-view translation vectors. */
    val tvecs: List<Mat>,
    /** Refined calibration pattern points (object-releasing method only). */
    val newObjPoints: Mat,
)

/**
 * Output of [calibrateCameraROExtended]: [CalibrateCameraRoResult] plus
 * standard deviations and per-view errors.
 */
data class CalibrateCameraRoExtendedResult(
    /** Overall RMS re-projection error. */
    val rms: Double,
    /** Output 3x3 camera intrinsic matrix. */
    val cameraMatrix: Mat,
    /** Output vector of distortion coefficients. */
    val distCoeffs: Mat,
    /** Per-view rotation vectors (Rodrigues). */
    val rvecs: List<Mat>,
    /** Per-view translation vectors. */
    val tvecs: List<Mat>,
    /** Refined calibration pattern points (object-releasing method only). */
    val newObjPoints: Mat,
    /** Standard deviations of the intrinsic parameters. */
    val stdDeviationsIntrinsics: Mat,
    /** Standard deviations of the extrinsic parameters. */
    val stdDeviationsExtrinsics: Mat,
    /** Standard deviations of the refined object points. */
    val stdDeviationsObjPoints: Mat,
    /** RMS re-projection error of each pattern view. */
    val perViewErrors: Mat,
)

/**
 * Output of [stereoCalibrate]: the two cameras' intrinsics (filled in
 * place), the relative pose [R]/[T] and the [E]ssential/[F]undamental
 * matrices.
 */
data class StereoCalibrateResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Camera 1 intrinsic matrix (input/output). */
    val cameraMatrix1: Mat,
    /** Camera 1 distortion coefficients (input/output). */
    val distCoeffs1: Mat,
    /** Camera 2 intrinsic matrix (input/output). */
    val cameraMatrix2: Mat,
    /** Camera 2 distortion coefficients (input/output). */
    val distCoeffs2: Mat,
    /** Rotation from camera 1 to camera 2 coordinate systems (3x3). */
    val R: Mat,
    /** Translation from camera 1 to camera 2 coordinate systems (3x1). */
    val T: Mat,
    /** Essential matrix (3x3). */
    val E: Mat,
    /** Fundamental matrix (3x3). */
    val F: Mat,
)

/**
 * Output of [stereoCalibratePerView]: [StereoCalibrateResult] plus the
 * per-view RMS re-projection errors.
 */
data class StereoCalibratePerViewResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Camera 1 intrinsic matrix (input/output). */
    val cameraMatrix1: Mat,
    /** Camera 1 distortion coefficients (input/output). */
    val distCoeffs1: Mat,
    /** Camera 2 intrinsic matrix (input/output). */
    val cameraMatrix2: Mat,
    /** Camera 2 distortion coefficients (input/output). */
    val distCoeffs2: Mat,
    /** Rotation from camera 1 to camera 2 coordinate systems (3x3). */
    val R: Mat,
    /** Translation from camera 1 to camera 2 coordinate systems (3x1). */
    val T: Mat,
    /** Essential matrix (3x3). */
    val E: Mat,
    /** Fundamental matrix (3x3). */
    val F: Mat,
    /** RMS re-projection error of each pattern view. */
    val perViewErrors: Mat,
)

/**
 * Output of [stereoCalibrateExtended]: [StereoCalibrateResult] plus the
 * per-view poses and per-view errors.
 */
data class StereoCalibrateExtendedResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Camera 1 intrinsic matrix (input/output). */
    val cameraMatrix1: Mat,
    /** Camera 1 distortion coefficients (input/output). */
    val distCoeffs1: Mat,
    /** Camera 2 intrinsic matrix (input/output). */
    val cameraMatrix2: Mat,
    /** Camera 2 distortion coefficients (input/output). */
    val distCoeffs2: Mat,
    /** Rotation from camera 1 to camera 2 coordinate systems (3x3). */
    val R: Mat,
    /** Translation from camera 1 to camera 2 coordinate systems (3x1). */
    val T: Mat,
    /** Essential matrix (3x3). */
    val E: Mat,
    /** Fundamental matrix (3x3). */
    val F: Mat,
    /** Per-view rotation vectors in camera 1's coordinate system. */
    val rvecs: List<Mat>,
    /** Per-view translation vectors in camera 1's coordinate system. */
    val tvecs: List<Mat>,
    /** RMS re-projection error of each pattern view. */
    val perViewErrors: Mat,
)

/**
 * Output of [registerCameras]: the relative pose [R]/[T] of a camera pair
 * with known intrinsics plus the [E]ssential/[F]undamental matrices and
 * per-view errors.
 */
data class RegisterCamerasResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Rotation from camera 1 to camera 2 coordinate systems (3x3). */
    val R: Mat,
    /** Translation from camera 1 to camera 2 coordinate systems (3x1). */
    val T: Mat,
    /** Essential matrix (3x3). */
    val E: Mat,
    /** Fundamental matrix (3x3). */
    val F: Mat,
    /** RMS re-projection error of each pattern view. */
    val perViewErrors: Mat,
)

/**
 * Output of [registerCamerasExtended]: [RegisterCamerasResult] plus the
 * per-view poses.
 */
data class RegisterCamerasExtendedResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Rotation from camera 1 to camera 2 coordinate systems (3x3). */
    val R: Mat,
    /** Translation from camera 1 to camera 2 coordinate systems (3x1). */
    val T: Mat,
    /** Essential matrix (3x3). */
    val E: Mat,
    /** Fundamental matrix (3x3). */
    val F: Mat,
    /** Per-view rotation vectors in camera 1's coordinate system. */
    val rvecs: List<Mat>,
    /** Per-view translation vectors in camera 1's coordinate system. */
    val tvecs: List<Mat>,
    /** RMS re-projection error of each pattern view. */
    val perViewErrors: Mat,
)

/**
 * Output of [fisheyeCalibrate]: the fisheye intrinsics [K]/[D] and the
 * per-view poses.
 */
data class FisheyeCalibrateResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Output 3x3 fisheye camera intrinsic matrix. */
    val K: Mat,
    /** Output 4-element fisheye distortion vector. */
    val D: Mat,
    /** Per-view rotation vectors (Rodrigues). */
    val rvecs: List<Mat>,
    /** Per-view translation vectors. */
    val tvecs: List<Mat>,
)

/**
 * Output of [fisheyeStereoCalibrate]: the fisheye intrinsics of both
 * cameras, the relative pose [R]/[T] and the per-view poses.
 */
data class FisheyeStereoCalibrateResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Camera 1 fisheye intrinsic matrix (input/output). */
    val K1: Mat,
    /** Camera 1 fisheye distortion vector (input/output). */
    val D1: Mat,
    /** Camera 2 fisheye intrinsic matrix (input/output). */
    val K2: Mat,
    /** Camera 2 fisheye distortion vector (input/output). */
    val D2: Mat,
    /** Rotation from camera 1 to camera 2 coordinate systems (3x3). */
    val R: Mat,
    /** Translation from camera 1 to camera 2 coordinate systems (3x1). */
    val T: Mat,
    /** Per-view rotation vectors in camera 1's coordinate system. */
    val rvecs: List<Mat>,
    /** Per-view translation vectors in camera 1's coordinate system. */
    val tvecs: List<Mat>,
)

/**
 * Output of [fisheyeStereoCalibratePose]: [FisheyeStereoCalibrateResult]
 * without the per-view poses.
 */
data class FisheyeStereoCalibratePoseResult(
    /** Final re-projection error. */
    val rms: Double,
    /** Camera 1 fisheye intrinsic matrix (input/output). */
    val K1: Mat,
    /** Camera 1 fisheye distortion vector (input/output). */
    val D1: Mat,
    /** Camera 2 fisheye intrinsic matrix (input/output). */
    val K2: Mat,
    /** Camera 2 fisheye distortion vector (input/output). */
    val D2: Mat,
    /** Rotation from camera 1 to camera 2 coordinate systems (3x3). */
    val R: Mat,
    /** Translation from camera 1 to camera 2 coordinate systems (3x1). */
    val T: Mat,
)

// =========================================================================
// Platform bridge: vector-of-Mat wire encoding and the raw native calls.
// =========================================================================

/**
 * Encodes [mats] into the CV_32SC2 Nx1 wire Mat whose i-th row carries the
 * high/low 32 bits of the i-th Mat handle (Converters.vector_Mat_to_Mat
 * format). The returned wire Mat is owned by the caller.
 */
internal expect fun matListToWire(mats: List<Mat>): Mat

/**
 * Decodes a CV_32SC2 Nx1 wire Mat produced by the native side into a list of
 * Mat handles; each element is owned by the caller and must be closed.
 */
internal expect fun wireToMatList(wire: Mat): List<Mat>

/** Raw cv::initCameraMatrix2D; returns a new Mat or throws [OpenCVException]. */
internal expect fun initCameraMatrix2DNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, aspectRatio: Double,
): Mat

/** Raw cv::calibrateCamera; returns the RMS error, NaN on failure. */
internal expect fun calibrateCameraNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::calibrateCamera (Extended); returns the RMS error, NaN on failure. */
internal expect fun calibrateCameraExtendedNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat,
    stdDeviationsIntrinsics: Mat, stdDeviationsExtrinsics: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::calibrateCameraRO; returns the RMS error, NaN on failure. */
internal expect fun calibrateCameraRoNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, newObjPoints: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::calibrateCameraRO (Extended); returns the RMS error, NaN on failure. */
internal expect fun calibrateCameraRoExtendedNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int, iFixedPoint: Int,
    cameraMatrix: Mat, distCoeffs: Mat, rvecs: Mat, tvecs: Mat, newObjPoints: Mat,
    stdDeviationsIntrinsics: Mat, stdDeviationsExtrinsics: Mat,
    stdDeviationsObjPoints: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::stereoCalibrate; returns the RMS error, NaN on failure. */
internal expect fun stereoCalibrateNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::stereoCalibrate (perViewErrors overload); returns the RMS error, NaN on failure. */
internal expect fun stereoCalibratePerViewNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::stereoCalibrate (Extended); returns the RMS error, NaN on failure. */
internal expect fun stereoCalibrateExtendedNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraMatrix2: Mat, distCoeffs2: Mat,
    r: Mat, t: Mat, e: Mat, f: Mat, rvecs: Mat, tvecs: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::registerCameras; returns the RMS error, NaN on failure. */
internal expect fun registerCamerasNative(
    objectPoints1: Mat, objectPoints2: Mat, imagePoints1: Mat, imagePoints2: Mat,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraModel1: Int,
    cameraMatrix2: Mat, distCoeffs2: Mat, cameraModel2: Int,
    r: Mat, t: Mat, e: Mat, f: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::registerCameras (Extended); returns the RMS error, NaN on failure. */
internal expect fun registerCamerasExtendedNative(
    objectPoints1: Mat, objectPoints2: Mat, imagePoints1: Mat, imagePoints2: Mat,
    cameraMatrix1: Mat, distCoeffs1: Mat, cameraModel1: Int,
    cameraMatrix2: Mat, distCoeffs2: Mat, cameraModel2: Int,
    r: Mat, t: Mat, e: Mat, f: Mat, rvecs: Mat, tvecs: Mat, perViewErrors: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::fisheye::calibrate; returns the RMS error, NaN on failure. */
internal expect fun fisheyeCalibrateNative(
    objectPoints: Mat, imagePoints: Mat, imageWidth: Int, imageHeight: Int,
    k: Mat, d: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::fisheye::stereoCalibrate; returns the RMS error, NaN on failure. */
internal expect fun fisheyeStereoCalibrateNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    k1: Mat, d1: Mat, k2: Mat, d2: Mat, r: Mat, t: Mat, rvecs: Mat, tvecs: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

/** Raw cv::fisheye::stereoCalibrate (pose-only); returns the RMS error, NaN on failure. */
internal expect fun fisheyeStereoCalibratePoseNative(
    objectPoints: Mat, imagePoints1: Mat, imagePoints2: Mat, imageWidth: Int, imageHeight: Int,
    k1: Mat, d1: Mat, k2: Mat, d2: Mat, r: Mat, t: Mat,
    flags: Int, criteriaType: Int, criteriaMaxCount: Int, criteriaEpsilon: Double,
): Double

// =========================================================================
// Public API
// =========================================================================

private const val DBL_EPSILON: Double = 2.220446049250313e-16

/** Runs [block], closing every scratch [wires] Mat afterwards. */
private fun <T> withWires(wires: List<Mat>, block: () -> T): T {
    try {
        return block()
    } finally {
        wires.forEach { it.close() }
    }
}

/** Throws when the native calibration reported failure (NaN RMS). */
private fun checkCalibError(rms: Double, operation: String): Double {
    if (rms.isNaN() || rms <= 0.0) {
        throw OpenCVException(operation, "rms=$rms err=${opencvLastError ?: "none"}")
    }
    return rms
}

/**
 * Finds an initial camera intrinsic matrix from 3D-2D point correspondences
 * (cv::initCameraMatrix2D). Only planar patterns (z = 0) are supported.
 *
 * @param objectPoints per-view CV_32FC3 Mats (e.g. [MatOfPoint3f]).
 * @param imagePoints per-view CV_32FC2 Mats (e.g. [MatOfPoint2f]).
 * @param imageSize image size used to initialize the principal point.
 * @param aspectRatio fx/fy when positive; otherwise both are estimated independently.
 * @return the initial 3x3 camera intrinsic matrix.
 */
fun initCameraMatrix2D(
    objectPoints: List<MatOfPoint3f>,
    imagePoints: List<MatOfPoint2f>,
    imageSize: Size,
    aspectRatio: Double = 1.0,
): Mat {
    val objWire = matListToWire(objectPoints.map { it.mat })
    val imgWire = matListToWire(imagePoints.map { it.mat })
    return withWires(listOf(objWire, imgWire)) {
        initCameraMatrix2DNative(objWire, imgWire, imageSize.width, imageSize.height, aspectRatio)
    }
}

/**
 * Finds the camera intrinsic and extrinsic parameters from several views of
 * a calibration pattern (cv::calibrateCamera).
 *
 * @param objectPoints per-view CV_32FC3 Mats of calibration pattern points.
 * @param imagePoints per-view CV_32FC2 Mats of pattern projections.
 * @param imageSize image size used to initialize the intrinsics.
 * @param cameraMatrix input (with [Calib.CALIB_USE_INTRINSIC_GUESS]) / output 3x3 matrix.
 * @param distCoeffs input / output distortion coefficient vector.
 * @param flags zero or a combination of [Calib] constants.
 * @param criteria termination criteria for the iterative optimization.
 * @return the overall RMS re-projection error and all outputs.
 */
fun calibrateCamera(
    objectPoints: List<Mat>,
    imagePoints: List<Mat>,
    imageSize: Size,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    flags: Int = 0,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 500, DBL_EPSILON),
): CalibrateCameraResult {
    val objWire = matListToWire(objectPoints)
    val imgWire = matListToWire(imagePoints)
    val rvecsWire = mat()
    val tvecsWire = mat()
    return withWires(listOf(objWire, imgWire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            calibrateCameraNative(
                objWire, imgWire, imageSize.width, imageSize.height,
                cameraMatrix, distCoeffs, rvecsWire, tvecsWire,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "calibrateCamera",
        )
        CalibrateCameraResult(rms, cameraMatrix, distCoeffs, wireToMatList(rvecsWire), wireToMatList(tvecsWire))
    }
}

/**
 * [calibrateCamera] extended with per-parameter standard deviations and
 * per-view errors (cv::calibrateCamera with stdDeviationsIntrinsics,
 * stdDeviationsExtrinsics and perViewErrors outputs).
 */
fun calibrateCameraExtended(
    objectPoints: List<Mat>,
    imagePoints: List<Mat>,
    imageSize: Size,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    flags: Int = 0,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 500, DBL_EPSILON),
): CalibrateCameraExtendedResult {
    val objWire = matListToWire(objectPoints)
    val imgWire = matListToWire(imagePoints)
    val rvecsWire = mat()
    val tvecsWire = mat()
    val stdDevIntr = mat()
    val stdDevExtr = mat()
    val perView = mat()
    return withWires(listOf(objWire, imgWire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            calibrateCameraExtendedNative(
                objWire, imgWire, imageSize.width, imageSize.height,
                cameraMatrix, distCoeffs, rvecsWire, tvecsWire,
                stdDevIntr, stdDevExtr, perView,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "calibrateCameraExtended",
        )
        CalibrateCameraExtendedResult(
            rms, cameraMatrix, distCoeffs,
            wireToMatList(rvecsWire), wireToMatList(tvecsWire),
            stdDevIntr, stdDevExtr, perView,
        )
    }
}

/**
 * [calibrateCamera] with the object-releasing method (cv::calibrateCameraRO).
 *
 * @param iFixedPoint index of the 3D object point in `objectPoints[0]` to
 *   fix; a value inside [1, objectPoints[0].size()-2] selects the
 *   object-releasing method, anything else the standard method.
 * @param newObjPoints refined output object points (object-releasing method only).
 */
fun calibrateCameraRO(
    objectPoints: List<Mat>,
    imagePoints: List<Mat>,
    imageSize: Size,
    iFixedPoint: Int,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    flags: Int = 0,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 500, DBL_EPSILON),
): CalibrateCameraRoResult {
    val objWire = matListToWire(objectPoints)
    val imgWire = matListToWire(imagePoints)
    val rvecsWire = mat()
    val tvecsWire = mat()
    val newObjPoints = mat()
    return withWires(listOf(objWire, imgWire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            calibrateCameraRoNative(
                objWire, imgWire, imageSize.width, imageSize.height, iFixedPoint,
                cameraMatrix, distCoeffs, rvecsWire, tvecsWire, newObjPoints,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "calibrateCameraRO",
        )
        CalibrateCameraRoResult(rms, cameraMatrix, distCoeffs, wireToMatList(rvecsWire), wireToMatList(tvecsWire), newObjPoints)
    }
}

/**
 * [calibrateCameraRO] extended with standard deviations and per-view errors
 * (cv::calibrateCameraRO with all stdDeviation/perViewErrors outputs).
 */
fun calibrateCameraROExtended(
    objectPoints: List<Mat>,
    imagePoints: List<Mat>,
    imageSize: Size,
    iFixedPoint: Int,
    cameraMatrix: Mat,
    distCoeffs: Mat,
    flags: Int = 0,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 500, DBL_EPSILON),
): CalibrateCameraRoExtendedResult {
    val objWire = matListToWire(objectPoints)
    val imgWire = matListToWire(imagePoints)
    val rvecsWire = mat()
    val tvecsWire = mat()
    val newObjPoints = mat()
    val stdDevIntr = mat()
    val stdDevExtr = mat()
    val stdDevObj = mat()
    val perView = mat()
    return withWires(listOf(objWire, imgWire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            calibrateCameraRoExtendedNative(
                objWire, imgWire, imageSize.width, imageSize.height, iFixedPoint,
                cameraMatrix, distCoeffs, rvecsWire, tvecsWire, newObjPoints,
                stdDevIntr, stdDevExtr, stdDevObj, perView,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "calibrateCameraROExtended",
        )
        CalibrateCameraRoExtendedResult(
            rms, cameraMatrix, distCoeffs,
            wireToMatList(rvecsWire), wireToMatList(tvecsWire),
            newObjPoints, stdDevIntr, stdDevExtr, stdDevObj, perView,
        )
    }
}

/**
 * Calibrates a stereo camera pair: intrinsic parameters of both cameras and
 * the extrinsic transformation between them (cv::stereoCalibrate).
 *
 * @param objectPoints per-view CV_32FC3 Mats (shared by both cameras).
 * @param imagePoints1 per-view CV_32FC2 Mats observed by camera 1.
 * @param imagePoints2 per-view CV_32FC2 Mats observed by camera 2.
 * @param cameraMatrix1/distCoeffs1/cameraMatrix2/distCoeffs2 input/output intrinsics.
 * @param R/T output relative pose (camera 1 -> camera 2).
 * @param E/F output essential/fundamental matrices.
 * @param flags zero or a combination of [Calib] constants (default [Calib.CALIB_FIX_INTRINSIC]).
 */
fun stereoCalibrate(
    objectPoints: List<Mat>,
    imagePoints1: List<Mat>,
    imagePoints2: List<Mat>,
    cameraMatrix1: Mat,
    distCoeffs1: Mat,
    cameraMatrix2: Mat,
    distCoeffs2: Mat,
    imageSize: Size,
    R: Mat,
    T: Mat,
    E: Mat,
    F: Mat,
    flags: Int = Calib.CALIB_FIX_INTRINSIC,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 100, 1e-6),
): StereoCalibrateResult {
    val objWire = matListToWire(objectPoints)
    val img1Wire = matListToWire(imagePoints1)
    val img2Wire = matListToWire(imagePoints2)
    return withWires(listOf(objWire, img1Wire, img2Wire)) {
        val rms = checkCalibError(
            stereoCalibrateNative(
                objWire, img1Wire, img2Wire, imageSize.width, imageSize.height,
                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                R, T, E, F,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "stereoCalibrate",
        )
        StereoCalibrateResult(rms, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F)
    }
}

/**
 * [stereoCalibrate] that also reports the RMS re-projection error of each
 * pattern view (cv::stereoCalibrate with a perViewErrors output).
 */
fun stereoCalibratePerView(
    objectPoints: List<Mat>,
    imagePoints1: List<Mat>,
    imagePoints2: List<Mat>,
    cameraMatrix1: Mat,
    distCoeffs1: Mat,
    cameraMatrix2: Mat,
    distCoeffs2: Mat,
    imageSize: Size,
    R: Mat,
    T: Mat,
    E: Mat,
    F: Mat,
    flags: Int = Calib.CALIB_FIX_INTRINSIC,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 30, 1e-6),
): StereoCalibratePerViewResult {
    val objWire = matListToWire(objectPoints)
    val img1Wire = matListToWire(imagePoints1)
    val img2Wire = matListToWire(imagePoints2)
    val perView = mat()
    return withWires(listOf(objWire, img1Wire, img2Wire)) {
        val rms = checkCalibError(
            stereoCalibratePerViewNative(
                objWire, img1Wire, img2Wire, imageSize.width, imageSize.height,
                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                R, T, E, F, perView,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "stereoCalibratePerView",
        )
        StereoCalibratePerViewResult(rms, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perView)
    }
}

/**
 * [stereoCalibrate] extended with per-view poses and per-view errors
 * (cv::stereoCalibrate with rvecs/tvecs/perViewErrors outputs).
 */
fun stereoCalibrateExtended(
    objectPoints: List<Mat>,
    imagePoints1: List<Mat>,
    imagePoints2: List<Mat>,
    cameraMatrix1: Mat,
    distCoeffs1: Mat,
    cameraMatrix2: Mat,
    distCoeffs2: Mat,
    imageSize: Size,
    R: Mat,
    T: Mat,
    E: Mat,
    F: Mat,
    flags: Int = Calib.CALIB_FIX_INTRINSIC,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 100, 1e-6),
): StereoCalibrateExtendedResult {
    val objWire = matListToWire(objectPoints)
    val img1Wire = matListToWire(imagePoints1)
    val img2Wire = matListToWire(imagePoints2)
    val rvecsWire = mat()
    val tvecsWire = mat()
    val perView = mat()
    return withWires(listOf(objWire, img1Wire, img2Wire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            stereoCalibrateExtendedNative(
                objWire, img1Wire, img2Wire, imageSize.width, imageSize.height,
                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                R, T, E, F, rvecsWire, tvecsWire, perView,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "stereoCalibrateExtended",
        )
        StereoCalibrateExtendedResult(
            rms, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F,
            wireToMatList(rvecsWire), wireToMatList(tvecsWire), perView,
        )
    }
}

/**
 * Estimates the extrinsic transformation between two cameras with known
 * intrinsics; the cameras need not share a field of view as long as each
 * observes the same calibration target (cv::registerCameras).
 *
 * @param cameraModel1/cameraModel2 [Calib.CALIB_MODEL_PINHOLE] or [Calib.CALIB_MODEL_FISHEYE].
 */
fun registerCameras(
    objectPoints1: List<Mat>,
    objectPoints2: List<Mat>,
    imagePoints1: List<Mat>,
    imagePoints2: List<Mat>,
    cameraMatrix1: Mat,
    distCoeffs1: Mat,
    cameraModel1: Int,
    cameraMatrix2: Mat,
    distCoeffs2: Mat,
    cameraModel2: Int,
    R: Mat,
    T: Mat,
    E: Mat,
    F: Mat,
    flags: Int = 0,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 100, 1e-6),
): RegisterCamerasResult {
    val op1Wire = matListToWire(objectPoints1)
    val op2Wire = matListToWire(objectPoints2)
    val ip1Wire = matListToWire(imagePoints1)
    val ip2Wire = matListToWire(imagePoints2)
    val perView = mat()
    return withWires(listOf(op1Wire, op2Wire, ip1Wire, ip2Wire)) {
        val rms = checkCalibError(
            registerCamerasNative(
                op1Wire, op2Wire, ip1Wire, ip2Wire,
                cameraMatrix1, distCoeffs1, cameraModel1,
                cameraMatrix2, distCoeffs2, cameraModel2,
                R, T, E, F, perView,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "registerCameras",
        )
        RegisterCamerasResult(rms, R, T, E, F, perView)
    }
}

/**
 * [registerCameras] extended with per-view poses
 * (cv::registerCameras with rvecs/tvecs/perViewErrors outputs).
 */
fun registerCamerasExtended(
    objectPoints1: List<Mat>,
    objectPoints2: List<Mat>,
    imagePoints1: List<Mat>,
    imagePoints2: List<Mat>,
    cameraMatrix1: Mat,
    distCoeffs1: Mat,
    cameraModel1: Int,
    cameraMatrix2: Mat,
    distCoeffs2: Mat,
    cameraModel2: Int,
    R: Mat,
    T: Mat,
    E: Mat,
    F: Mat,
    flags: Int = 0,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 100, 1e-6),
): RegisterCamerasExtendedResult {
    val op1Wire = matListToWire(objectPoints1)
    val op2Wire = matListToWire(objectPoints2)
    val ip1Wire = matListToWire(imagePoints1)
    val ip2Wire = matListToWire(imagePoints2)
    val rvecsWire = mat()
    val tvecsWire = mat()
    val perView = mat()
    return withWires(listOf(op1Wire, op2Wire, ip1Wire, ip2Wire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            registerCamerasExtendedNative(
                op1Wire, op2Wire, ip1Wire, ip2Wire,
                cameraMatrix1, distCoeffs1, cameraModel1,
                cameraMatrix2, distCoeffs2, cameraModel2,
                R, T, E, F, rvecsWire, tvecsWire, perView,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "registerCamerasExtended",
        )
        RegisterCamerasExtendedResult(rms, R, T, E, F, wireToMatList(rvecsWire), wireToMatList(tvecsWire), perView)
    }
}

/**
 * Performs fisheye camera calibration (cv::fisheye::calibrate).
 *
 * @param K input (with [Calib.CALIB_USE_INTRINSIC_GUESS]) / output 3x3 matrix.
 * @param D output 4-element fisheye distortion vector.
 */
fun fisheyeCalibrate(
    objectPoints: List<Mat>,
    imagePoints: List<Mat>,
    imageSize: Size,
    K: Mat,
    D: Mat,
    flags: Int = 0,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 100, DBL_EPSILON),
): FisheyeCalibrateResult {
    val objWire = matListToWire(objectPoints)
    val imgWire = matListToWire(imagePoints)
    val rvecsWire = mat()
    val tvecsWire = mat()
    return withWires(listOf(objWire, imgWire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            fisheyeCalibrateNative(
                objWire, imgWire, imageSize.width, imageSize.height,
                K, D, rvecsWire, tvecsWire,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "fisheyeCalibrate",
        )
        FisheyeCalibrateResult(rms, K, D, wireToMatList(rvecsWire), wireToMatList(tvecsWire))
    }
}

/**
 * Performs fisheye stereo calibration (cv::fisheye::stereoCalibrate).
 *
 * @param K1/D1/K2/D2 input/output fisheye intrinsics and distortions.
 * @param R/T output relative pose (camera 1 -> camera 2).
 */
fun fisheyeStereoCalibrate(
    objectPoints: List<Mat>,
    imagePoints1: List<Mat>,
    imagePoints2: List<Mat>,
    K1: Mat,
    D1: Mat,
    K2: Mat,
    D2: Mat,
    imageSize: Size,
    R: Mat,
    T: Mat,
    flags: Int = Calib.CALIB_FIX_INTRINSIC,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 100, DBL_EPSILON),
): FisheyeStereoCalibrateResult {
    val objWire = matListToWire(objectPoints)
    val img1Wire = matListToWire(imagePoints1)
    val img2Wire = matListToWire(imagePoints2)
    val rvecsWire = mat()
    val tvecsWire = mat()
    return withWires(listOf(objWire, img1Wire, img2Wire, rvecsWire, tvecsWire)) {
        val rms = checkCalibError(
            fisheyeStereoCalibrateNative(
                objWire, img1Wire, img2Wire, imageSize.width, imageSize.height,
                K1, D1, K2, D2, R, T, rvecsWire, tvecsWire,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "fisheyeStereoCalibrate",
        )
        FisheyeStereoCalibrateResult(rms, K1, D1, K2, D2, R, T, wireToMatList(rvecsWire), wireToMatList(tvecsWire))
    }
}

/**
 * Fisheye stereo calibration of the relative pose only
 * (cv::fisheye::stereoCalibrate without rvecs/tvecs outputs).
 */
fun fisheyeStereoCalibratePose(
    objectPoints: List<Mat>,
    imagePoints1: List<Mat>,
    imagePoints2: List<Mat>,
    K1: Mat,
    D1: Mat,
    K2: Mat,
    D2: Mat,
    imageSize: Size,
    R: Mat,
    T: Mat,
    flags: Int = Calib.CALIB_FIX_INTRINSIC,
    criteria: TermCriteria = TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 100, DBL_EPSILON),
): FisheyeStereoCalibratePoseResult {
    val objWire = matListToWire(objectPoints)
    val img1Wire = matListToWire(imagePoints1)
    val img2Wire = matListToWire(imagePoints2)
    return withWires(listOf(objWire, img1Wire, img2Wire)) {
        val rms = checkCalibError(
            fisheyeStereoCalibratePoseNative(
                objWire, img1Wire, img2Wire, imageSize.width, imageSize.height,
                K1, D1, K2, D2, R, T,
                flags, criteria.type, criteria.maxCount, criteria.epsilon,
            ),
            "fisheyeStereoCalibratePose",
        )
        FisheyeStereoCalibratePoseResult(rms, K1, D1, K2, D2, R, T)
    }
}

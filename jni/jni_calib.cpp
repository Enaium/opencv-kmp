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

/*
 * JNI bridge for the calib slice: thin Java_cn_enaium_opencv_JniCalib_*
 * wrappers around the cvk_ C ABI in native/shim_calib.cpp. Mat handles
 * travel as jlong cv::Mat pointers; the shim is noexcept so nothing throws
 * across the boundary (calibration failures surface as NaN return values).
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_calib.h"

#include <cstdint>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCalib_initCameraMatrix2D(JNIEnv *, jobject,
                                                   jlong objectPoints, jlong imagePoints,
                                                   jint imageWidth, jint imageHeight,
                                                   jdouble aspectRatio) {
    return as_handle(cvk_init_camera_matrix_2d(as_mat(objectPoints), as_mat(imagePoints),
                                               imageWidth, imageHeight, aspectRatio));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_calibrateCamera(JNIEnv *, jobject,
                                                jlong objectPoints, jlong imagePoints,
                                                jint imageWidth, jint imageHeight,
                                                jlong cameraMatrix, jlong distCoeffs,
                                                jlong rvecs, jlong tvecs,
                                                jint flags, jint criteriaType,
                                                jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_calibrate_camera(as_mat(objectPoints), as_mat(imagePoints),
                                imageWidth, imageHeight, as_mat(cameraMatrix), as_mat(distCoeffs),
                                as_mat(rvecs), as_mat(tvecs), flags, criteriaType,
                                criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_calibrateCameraExtended(JNIEnv *, jobject,
                                                        jlong objectPoints, jlong imagePoints,
                                                        jint imageWidth, jint imageHeight,
                                                        jlong cameraMatrix, jlong distCoeffs,
                                                        jlong rvecs, jlong tvecs,
                                                        jlong stdDeviationsIntrinsics,
                                                        jlong stdDeviationsExtrinsics,
                                                        jlong perViewErrors,
                                                        jint flags, jint criteriaType,
                                                        jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_calibrate_camera_extended(as_mat(objectPoints), as_mat(imagePoints),
                                         imageWidth, imageHeight, as_mat(cameraMatrix),
                                         as_mat(distCoeffs), as_mat(rvecs), as_mat(tvecs),
                                         as_mat(stdDeviationsIntrinsics),
                                         as_mat(stdDeviationsExtrinsics),
                                         as_mat(perViewErrors), flags, criteriaType,
                                         criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_calibrateCameraRO(JNIEnv *, jobject,
                                                  jlong objectPoints, jlong imagePoints,
                                                  jint imageWidth, jint imageHeight,
                                                  jint iFixedPoint,
                                                  jlong cameraMatrix, jlong distCoeffs,
                                                  jlong rvecs, jlong tvecs,
                                                  jlong newObjPoints,
                                                  jint flags, jint criteriaType,
                                                  jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_calibrate_camera_ro(as_mat(objectPoints), as_mat(imagePoints),
                                   imageWidth, imageHeight, iFixedPoint,
                                   as_mat(cameraMatrix), as_mat(distCoeffs),
                                   as_mat(rvecs), as_mat(tvecs), as_mat(newObjPoints),
                                   flags, criteriaType, criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_calibrateCameraROExtended(JNIEnv *, jobject,
                                                          jlong objectPoints, jlong imagePoints,
                                                          jint imageWidth, jint imageHeight,
                                                          jint iFixedPoint,
                                                          jlong cameraMatrix, jlong distCoeffs,
                                                          jlong rvecs, jlong tvecs,
                                                          jlong newObjPoints,
                                                          jlong stdDeviationsIntrinsics,
                                                          jlong stdDeviationsExtrinsics,
                                                          jlong stdDeviationsObjPoints,
                                                          jlong perViewErrors,
                                                          jint flags, jint criteriaType,
                                                          jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_calibrate_camera_ro_extended(as_mat(objectPoints), as_mat(imagePoints),
                                            imageWidth, imageHeight, iFixedPoint,
                                            as_mat(cameraMatrix), as_mat(distCoeffs),
                                            as_mat(rvecs), as_mat(tvecs), as_mat(newObjPoints),
                                            as_mat(stdDeviationsIntrinsics),
                                            as_mat(stdDeviationsExtrinsics),
                                            as_mat(stdDeviationsObjPoints),
                                            as_mat(perViewErrors), flags, criteriaType,
                                            criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_stereoCalibrate(JNIEnv *, jobject,
                                                jlong objectPoints, jlong imagePoints1, jlong imagePoints2,
                                                jint imageWidth, jint imageHeight,
                                                jlong cameraMatrix1, jlong distCoeffs1,
                                                jlong cameraMatrix2, jlong distCoeffs2,
                                                jlong r, jlong t, jlong e, jlong f,
                                                jint flags, jint criteriaType,
                                                jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_stereo_calibrate(as_mat(objectPoints), as_mat(imagePoints1), as_mat(imagePoints2),
                                imageWidth, imageHeight, as_mat(cameraMatrix1), as_mat(distCoeffs1),
                                as_mat(cameraMatrix2), as_mat(distCoeffs2),
                                as_mat(r), as_mat(t), as_mat(e), as_mat(f),
                                flags, criteriaType, criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_stereoCalibratePerView(JNIEnv *, jobject,
                                                       jlong objectPoints, jlong imagePoints1, jlong imagePoints2,
                                                       jint imageWidth, jint imageHeight,
                                                       jlong cameraMatrix1, jlong distCoeffs1,
                                                       jlong cameraMatrix2, jlong distCoeffs2,
                                                       jlong r, jlong t, jlong e, jlong f,
                                                       jlong perViewErrors,
                                                       jint flags, jint criteriaType,
                                                       jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_stereo_calibrate_per_view(as_mat(objectPoints), as_mat(imagePoints1), as_mat(imagePoints2),
                                         imageWidth, imageHeight, as_mat(cameraMatrix1), as_mat(distCoeffs1),
                                         as_mat(cameraMatrix2), as_mat(distCoeffs2),
                                         as_mat(r), as_mat(t), as_mat(e), as_mat(f),
                                         as_mat(perViewErrors), flags, criteriaType,
                                         criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_stereoCalibrateExtended(JNIEnv *, jobject,
                                                        jlong objectPoints, jlong imagePoints1, jlong imagePoints2,
                                                        jint imageWidth, jint imageHeight,
                                                        jlong cameraMatrix1, jlong distCoeffs1,
                                                        jlong cameraMatrix2, jlong distCoeffs2,
                                                        jlong r, jlong t, jlong e, jlong f,
                                                        jlong rvecs, jlong tvecs,
                                                        jlong perViewErrors,
                                                        jint flags, jint criteriaType,
                                                        jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_stereo_calibrate_extended(as_mat(objectPoints), as_mat(imagePoints1), as_mat(imagePoints2),
                                         imageWidth, imageHeight, as_mat(cameraMatrix1), as_mat(distCoeffs1),
                                         as_mat(cameraMatrix2), as_mat(distCoeffs2),
                                         as_mat(r), as_mat(t), as_mat(e), as_mat(f),
                                         as_mat(rvecs), as_mat(tvecs), as_mat(perViewErrors),
                                         flags, criteriaType, criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_registerCameras(JNIEnv *, jobject,
                                                jlong objectPoints1, jlong objectPoints2,
                                                jlong imagePoints1, jlong imagePoints2,
                                                jlong cameraMatrix1, jlong distCoeffs1, jint cameraModel1,
                                                jlong cameraMatrix2, jlong distCoeffs2, jint cameraModel2,
                                                jlong r, jlong t, jlong e, jlong f,
                                                jlong perViewErrors,
                                                jint flags, jint criteriaType,
                                                jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_register_cameras(as_mat(objectPoints1), as_mat(objectPoints2),
                                as_mat(imagePoints1), as_mat(imagePoints2),
                                as_mat(cameraMatrix1), as_mat(distCoeffs1), cameraModel1,
                                as_mat(cameraMatrix2), as_mat(distCoeffs2), cameraModel2,
                                as_mat(r), as_mat(t), as_mat(e), as_mat(f),
                                as_mat(perViewErrors), flags, criteriaType,
                                criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_registerCamerasExtended(JNIEnv *, jobject,
                                                        jlong objectPoints1, jlong objectPoints2,
                                                        jlong imagePoints1, jlong imagePoints2,
                                                        jlong cameraMatrix1, jlong distCoeffs1, jint cameraModel1,
                                                        jlong cameraMatrix2, jlong distCoeffs2, jint cameraModel2,
                                                        jlong r, jlong t, jlong e, jlong f,
                                                        jlong rvecs, jlong tvecs,
                                                        jlong perViewErrors,
                                                        jint flags, jint criteriaType,
                                                        jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_register_cameras_extended(as_mat(objectPoints1), as_mat(objectPoints2),
                                         as_mat(imagePoints1), as_mat(imagePoints2),
                                         as_mat(cameraMatrix1), as_mat(distCoeffs1), cameraModel1,
                                         as_mat(cameraMatrix2), as_mat(distCoeffs2), cameraModel2,
                                         as_mat(r), as_mat(t), as_mat(e), as_mat(f),
                                         as_mat(rvecs), as_mat(tvecs), as_mat(perViewErrors),
                                         flags, criteriaType, criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_fisheyeCalibrate(JNIEnv *, jobject,
                                                 jlong objectPoints, jlong imagePoints,
                                                 jint imageWidth, jint imageHeight,
                                                 jlong k, jlong d,
                                                 jlong rvecs, jlong tvecs,
                                                 jint flags, jint criteriaType,
                                                 jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_fisheye_calibrate(as_mat(objectPoints), as_mat(imagePoints),
                                 imageWidth, imageHeight, as_mat(k), as_mat(d),
                                 as_mat(rvecs), as_mat(tvecs), flags, criteriaType,
                                 criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_fisheyeStereoCalibrate(JNIEnv *, jobject,
                                                       jlong objectPoints, jlong imagePoints1, jlong imagePoints2,
                                                       jint imageWidth, jint imageHeight,
                                                       jlong k1, jlong d1, jlong k2, jlong d2,
                                                       jlong r, jlong t,
                                                       jlong rvecs, jlong tvecs,
                                                       jint flags, jint criteriaType,
                                                       jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_fisheye_stereo_calibrate(as_mat(objectPoints), as_mat(imagePoints1), as_mat(imagePoints2),
                                        imageWidth, imageHeight, as_mat(k1), as_mat(d1),
                                        as_mat(k2), as_mat(d2), as_mat(r), as_mat(t),
                                        as_mat(rvecs), as_mat(tvecs), flags, criteriaType,
                                        criteriaMaxCount, criteriaEpsilon);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCalib_fisheyeStereoCalibratePose(JNIEnv *, jobject,
                                                           jlong objectPoints, jlong imagePoints1, jlong imagePoints2,
                                                           jint imageWidth, jint imageHeight,
                                                           jlong k1, jlong d1, jlong k2, jlong d2,
                                                           jlong r, jlong t,
                                                           jint flags, jint criteriaType,
                                                           jint criteriaMaxCount, jdouble criteriaEpsilon) {
    return cvk_fisheye_stereo_calibrate_pose(as_mat(objectPoints), as_mat(imagePoints1), as_mat(imagePoints2),
                                             imageWidth, imageHeight, as_mat(k1), as_mat(d1),
                                             as_mat(k2), as_mat(d2), as_mat(r), as_mat(t),
                                             flags, criteriaType, criteriaMaxCount, criteriaEpsilon);
}

} /* extern "C" */

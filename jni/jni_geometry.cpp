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
 * JNI bridge for the geometry slice: Java_cn_enaium_opencv_JniGeometry_*
 * wrappers around the cvk_ geometry ABI (native/shim_geometry.cpp).
 *
 * Handle conventions mirror jni_bridge.cpp: Mats travel as jlong cv::Mat
 * pointers; multi-result functions return jlongArrays of handles (the first
 * element carries an int result count where the C function returns one);
 * Subdiv2D handles travel as jlong cvk_subdiv2d_t pointers.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_geometry.h"

#include <cstdint>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_subdiv2d_t *as_subdiv2d(jlong handle) {
    return reinterpret_cast<cvk_subdiv2d_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_subdiv2d_handle(const cvk_subdiv2d_t *h) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(h));
}

/** Wraps freshly allocated Mat handles into a jlongArray. */
static jlongArray as_handle_array(JNIEnv *env, cvk_mat_t **handles, jsize count) {
    std::vector<jlong> values(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = as_handle(handles[i]);
    }
    jlongArray result = env->NewLongArray(count);
    if (result == nullptr) {
        for (jsize i = 0; i < count; ++i) cvk_mat_release(handles[i]);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, count, values.data());
    return result;
}

extern "C" {

/* ---------------------------------------------------------- Rodrigues */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_rodrigues(JNIEnv *, jobject, jlong src) {
    return as_handle(cvk_rodrigues(as_mat(src)));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_rodriguesJacobian(JNIEnv *env, jobject, jlong src) {
    cvk_mat_t *dst = nullptr;
    cvk_mat_t *jacobian = nullptr;
    dst = cvk_rodrigues_jacobian(as_mat(src), &jacobian);
    if (dst == nullptr || jacobian == nullptr) {
        cvk_mat_release(dst);
        cvk_mat_release(jacobian);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {dst, jacobian};
    return as_handle_array(env, handles, 2);
}

/* ------------------------------------------------------ findHomography */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_findHomography(JNIEnv *, jobject, jlong src, jlong dst,
                                                 jint method, jdouble threshold, jint maxIters,
                                                 jdouble confidence) {
    return as_handle(cvk_find_homography(as_mat(src), as_mat(dst), method, threshold, maxIters,
                                         confidence));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findHomographyMasked(JNIEnv *env, jobject, jlong src, jlong dst,
                                                       jint method, jdouble threshold,
                                                       jint maxIters, jdouble confidence) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *h = cvk_find_homography_masked(as_mat(src), as_mat(dst), method, threshold, maxIters,
                                              confidence, &mask);
    if (h == nullptr || mask == nullptr) {
        cvk_mat_release(h);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {h, mask};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findHomographyUsac(
    JNIEnv *env, jobject, jlong src, jlong dst,
    jdouble confidence, jboolean isParallel, jint loIterations, jint loMethod, jint loSampleSize,
    jint maxIterations, jint neighborsSearch, jint randomGeneratorState, jint sampler, jint score,
    jdouble threshold, jint finalPolisher, jint finalPolisherIterations) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *h = cvk_find_homography_usac(
        as_mat(src), as_mat(dst), &mask, confidence, isParallel, loIterations, loMethod,
        loSampleSize, maxIterations, neighborsSearch, randomGeneratorState, sampler, score,
        threshold, finalPolisher, finalPolisherIterations);
    if (h == nullptr || mask == nullptr) {
        cvk_mat_release(h);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {h, mask};
    return as_handle_array(env, handles, 2);
}

/* ---------------------------------------------------------- RQDecomp3x3 */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_rqDecomp3x3(JNIEnv *env, jobject, jlong src, jboolean wantAxes,
                                              jdoubleArray eulerOut) {
    cvk_mat_t *r = nullptr;
    cvk_mat_t *q = nullptr;
    cvk_mat_t *qx = nullptr;
    cvk_mat_t *qy = nullptr;
    cvk_mat_t *qz = nullptr;
    double euler[3] = {0.0, 0.0, 0.0};
    cvk_rq_decomp_3x3(as_mat(src), wantAxes, &r, &q, &qx, &qy, &qz, euler);
    if (r == nullptr || q == nullptr) {
        cvk_mat_release(r);
        cvk_mat_release(q);
        cvk_mat_release(qx);
        cvk_mat_release(qy);
        cvk_mat_release(qz);
        return nullptr;
    }
    env->SetDoubleArrayRegion(eulerOut, 0, 3, euler);
    if (wantAxes) {
        cvk_mat_t *handles[5] = {r, q, qx, qy, qz};
        return as_handle_array(env, handles, 5);
    }
    cvk_mat_t *handles[2] = {r, q};
    return as_handle_array(env, handles, 2);
}

/* ----------------------------------------------------------- matMulDeriv */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_matMulDeriv(JNIEnv *env, jobject, jlong a, jlong b) {
    cvk_mat_t *dabda = nullptr;
    cvk_mat_t *dabdb = nullptr;
    cvk_mat_mul_deriv(as_mat(a), as_mat(b), &dabda, &dabdb);
    if (dabda == nullptr || dabdb == nullptr) {
        cvk_mat_release(dabda);
        cvk_mat_release(dabdb);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {dabda, dabdb};
    return as_handle_array(env, handles, 2);
}

/* ------------------------------------------------------------- composeRT */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_composeRt(JNIEnv *env, jobject, jlong rvec1, jlong tvec1,
                                            jlong rvec2, jlong tvec2, jboolean wantDerivatives) {
    cvk_mat_t *rvec3 = nullptr;
    cvk_mat_t *tvec3 = nullptr;
    cvk_mat_t *dr3dr1 = nullptr, *dr3dt1 = nullptr, *dr3dr2 = nullptr, *dr3dt2 = nullptr;
    cvk_mat_t *dt3dr1 = nullptr, *dt3dt1 = nullptr, *dt3dr2 = nullptr, *dt3dt2 = nullptr;
    cvk_compose_rt(as_mat(rvec1), as_mat(tvec1), as_mat(rvec2), as_mat(tvec2), wantDerivatives,
                   &rvec3, &tvec3, &dr3dr1, &dr3dt1, &dr3dr2, &dr3dt2, &dt3dr1, &dt3dt1, &dt3dr2,
                   &dt3dt2);
    if (rvec3 == nullptr || tvec3 == nullptr) {
        cvk_mat_release(rvec3);
        cvk_mat_release(tvec3);
        cvk_mat_release(dr3dr1);
        cvk_mat_release(dr3dt1);
        cvk_mat_release(dr3dr2);
        cvk_mat_release(dr3dt2);
        cvk_mat_release(dt3dr1);
        cvk_mat_release(dt3dt1);
        cvk_mat_release(dt3dr2);
        cvk_mat_release(dt3dt2);
        return nullptr;
    }
    if (wantDerivatives) {
        cvk_mat_t *handles[10] = {rvec3,  tvec3,  dr3dr1, dr3dt1, dr3dr2,
                                  dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2};
        return as_handle_array(env, handles, 10);
    }
    cvk_mat_t *handles[2] = {rvec3, tvec3};
    return as_handle_array(env, handles, 2);
}

/* --------------------------------------------------------- projectPoints */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_projectPoints(JNIEnv *, jobject, jlong objectPoints, jlong rvec,
                                                jlong tvec, jlong cameraMatrix, jlong distCoeffs) {
    return as_handle(cvk_project_points(as_mat(objectPoints), as_mat(rvec), as_mat(tvec),
                                        as_mat(cameraMatrix), as_mat(distCoeffs)));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_projectPointsJacobian(JNIEnv *env, jobject, jlong objectPoints,
                                                        jlong rvec, jlong tvec, jlong cameraMatrix,
                                                        jlong distCoeffs, jdouble aspectRatio) {
    cvk_mat_t *jacobian = nullptr;
    cvk_mat_t *image_points =
        cvk_project_points_jacobian(as_mat(objectPoints), as_mat(rvec), as_mat(tvec),
                                    as_mat(cameraMatrix), as_mat(distCoeffs), aspectRatio,
                                    &jacobian);
    if (image_points == nullptr || jacobian == nullptr) {
        cvk_mat_release(image_points);
        cvk_mat_release(jacobian);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {image_points, jacobian};
    return as_handle_array(env, handles, 2);
}

/* --------------------------------------------------------------- solvePnP */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_solvePnP(JNIEnv *env, jobject, jlong objectPoints,
                                           jlong imagePoints, jlong cameraMatrix, jlong distCoeffs,
                                           jboolean useExtrinsicGuess, jint flags) {
    cvk_mat_t *rvec = nullptr;
    cvk_mat_t *tvec = nullptr;
    if (!cvk_solve_pnp(as_mat(objectPoints), as_mat(imagePoints), as_mat(cameraMatrix),
                       as_mat(distCoeffs), useExtrinsicGuess, flags, &rvec, &tvec)) {
        cvk_mat_release(rvec);
        cvk_mat_release(tvec);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {rvec, tvec};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_solvePnPRansac(JNIEnv *env, jobject, jlong objectPoints,
                                                 jlong imagePoints, jlong cameraMatrix,
                                                 jlong distCoeffs, jboolean useExtrinsicGuess,
                                                 jint iterationsCount, jfloat reprojectionError,
                                                 jdouble confidence, jint flags) {
    cvk_mat_t *rvec = nullptr;
    cvk_mat_t *tvec = nullptr;
    cvk_mat_t *inliers = nullptr;
    if (!cvk_solve_pnp_ransac(as_mat(objectPoints), as_mat(imagePoints), as_mat(cameraMatrix),
                              as_mat(distCoeffs), useExtrinsicGuess, iterationsCount,
                              reprojectionError, confidence, flags, &rvec, &tvec, &inliers)) {
        cvk_mat_release(rvec);
        cvk_mat_release(tvec);
        cvk_mat_release(inliers);
        return nullptr;
    }
    cvk_mat_t *handles[3] = {rvec, tvec, inliers};
    return as_handle_array(env, handles, 3);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_solvePnPRansacUsac(
    JNIEnv *env, jobject, jlong objectPoints, jlong imagePoints, jlong cameraMatrix,
    jlong distCoeffs, jdouble confidence, jboolean isParallel, jint loIterations, jint loMethod,
    jint loSampleSize, jint maxIterations, jint neighborsSearch, jint randomGeneratorState,
    jint sampler, jint score, jdouble threshold, jint finalPolisher,
    jint finalPolisherIterations) {
    cvk_mat_t *rvec = nullptr;
    cvk_mat_t *tvec = nullptr;
    cvk_mat_t *inliers = nullptr;
    if (!cvk_solve_pnp_ransac_usac(
            as_mat(objectPoints), as_mat(imagePoints), as_mat(cameraMatrix), as_mat(distCoeffs),
            &rvec, &tvec, &inliers, confidence, isParallel, loIterations, loMethod, loSampleSize,
            maxIterations, neighborsSearch, randomGeneratorState, sampler, score, threshold,
            finalPolisher, finalPolisherIterations)) {
        cvk_mat_release(rvec);
        cvk_mat_release(tvec);
        cvk_mat_release(inliers);
        return nullptr;
    }
    cvk_mat_t *handles[3] = {rvec, tvec, inliers};
    return as_handle_array(env, handles, 3);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_solvePnPGeneric(JNIEnv *env, jobject, jlong objectPoints,
                                                  jlong imagePoints, jlong cameraMatrix,
                                                  jlong distCoeffs, jboolean useExtrinsicGuess,
                                                  jint flags, jlong rvec, jlong tvec,
                                                  jboolean wantReprojectionError) {
    cvk_mat_t *rvecs = nullptr;
    cvk_mat_t *tvecs = nullptr;
    cvk_mat_t *reprojection_error = nullptr;
    const int count = cvk_solve_pnp_generic(
        as_mat(objectPoints), as_mat(imagePoints), as_mat(cameraMatrix), as_mat(distCoeffs),
        useExtrinsicGuess, flags, as_mat(rvec), as_mat(tvec), &rvecs, &tvecs,
        wantReprojectionError ? &reprojection_error : nullptr);
    if (rvecs == nullptr || tvecs == nullptr) {
        cvk_mat_release(rvecs);
        cvk_mat_release(tvecs);
        cvk_mat_release(reprojection_error);
        return nullptr;
    }
    if (wantReprojectionError) {
        cvk_mat_t *handles[4] = {nullptr, rvecs, tvecs, reprojection_error};
        handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
        return as_handle_array(env, handles, 4);
    }
    cvk_mat_t *handles[3] = {nullptr, rvecs, tvecs};
    handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
    return as_handle_array(env, handles, 3);
}

/* ---------------------------------------------------- convert points */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_convertPointsToHomogeneous(JNIEnv *, jobject, jlong src,
                                                             jint dtype) {
    return as_handle(cvk_convert_points_to_homogeneous(as_mat(src), dtype));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_convertPointsFromHomogeneous(JNIEnv *, jobject, jlong src,
                                                               jint dtype) {
    return as_handle(cvk_convert_points_from_homogeneous(as_mat(src), dtype));
}

/* ------------------------------------------------------ findFundamentalMat */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_findFundamentalMat(JNIEnv *, jobject, jlong points1,
                                                     jlong points2, jint method,
                                                     jdouble ransacThreshold, jdouble confidence,
                                                     jint maxIters) {
    return as_handle(cvk_find_fundamental_mat(as_mat(points1), as_mat(points2), method,
                                              ransacThreshold, confidence, maxIters));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findFundamentalMatMasked(JNIEnv *env, jobject, jlong points1,
                                                           jlong points2, jint method,
                                                           jdouble ransacThreshold,
                                                           jdouble confidence, jint maxIters) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *f = cvk_find_fundamental_mat_masked(as_mat(points1), as_mat(points2), method,
                                                   ransacThreshold, confidence, maxIters, &mask);
    if (f == nullptr || mask == nullptr) {
        cvk_mat_release(f);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {f, mask};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findFundamentalMatUsac(
    JNIEnv *env, jobject, jlong points1, jlong points2, jdouble confidence, jboolean isParallel,
    jint loIterations, jint loMethod, jint loSampleSize, jint maxIterations, jint neighborsSearch,
    jint randomGeneratorState, jint sampler, jint score, jdouble threshold, jint finalPolisher,
    jint finalPolisherIterations) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *f = cvk_find_fundamental_mat_usac(
        as_mat(points1), as_mat(points2), &mask, confidence, isParallel, loIterations, loMethod,
        loSampleSize, maxIterations, neighborsSearch, randomGeneratorState, sampler, score,
        threshold, finalPolisher, finalPolisherIterations);
    if (f == nullptr || mask == nullptr) {
        cvk_mat_release(f);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {f, mask};
    return as_handle_array(env, handles, 2);
}

/* ------------------------------------------------------ findEssentialMat */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_findEssentialMat(JNIEnv *, jobject, jlong points1,
                                                   jlong points2, jlong cameraMatrix, jint method,
                                                   jdouble prob, jdouble threshold, jint maxIters) {
    return as_handle(cvk_find_essential_mat(as_mat(points1), as_mat(points2),
                                            as_mat(cameraMatrix), method, prob, threshold,
                                            maxIters));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findEssentialMatMasked(JNIEnv *env, jobject, jlong points1,
                                                         jlong points2, jlong cameraMatrix,
                                                         jint method, jdouble prob,
                                                         jdouble threshold, jint maxIters) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *e = cvk_find_essential_mat_masked(as_mat(points1), as_mat(points2),
                                                 as_mat(cameraMatrix), method, prob, threshold,
                                                 maxIters, &mask);
    if (e == nullptr || mask == nullptr) {
        cvk_mat_release(e);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {e, mask};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_findEssentialMatFocal(JNIEnv *, jobject, jlong points1,
                                                        jlong points2, jdouble focal,
                                                        jdouble ppX, jdouble ppY, jint method,
                                                        jdouble prob, jdouble threshold,
                                                        jint maxIters) {
    return as_handle(cvk_find_essential_mat_focal(as_mat(points1), as_mat(points2), focal, ppX,
                                                  ppY, method, prob, threshold, maxIters));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findEssentialMatFocalMasked(JNIEnv *env, jobject, jlong points1,
                                                              jlong points2, jdouble focal,
                                                              jdouble ppX, jdouble ppY,
                                                              jint method, jdouble prob,
                                                              jdouble threshold, jint maxIters) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *e = cvk_find_essential_mat_focal_masked(as_mat(points1), as_mat(points2), focal,
                                                       ppX, ppY, method, prob, threshold, maxIters,
                                                       &mask);
    if (e == nullptr || mask == nullptr) {
        cvk_mat_release(e);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {e, mask};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_findEssentialMatStereo(JNIEnv *, jobject, jlong points1,
                                                         jlong points2, jlong cameraMatrix1,
                                                         jlong distCoeffs1, jlong cameraMatrix2,
                                                         jlong distCoeffs2, jint method,
                                                         jdouble prob, jdouble threshold) {
    return as_handle(cvk_find_essential_mat_stereo(
        as_mat(points1), as_mat(points2), as_mat(cameraMatrix1), as_mat(distCoeffs1),
        as_mat(cameraMatrix2), as_mat(distCoeffs2), method, prob, threshold));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findEssentialMatStereoMasked(JNIEnv *env, jobject,
                                                               jlong points1, jlong points2,
                                                               jlong cameraMatrix1,
                                                               jlong distCoeffs1,
                                                               jlong cameraMatrix2,
                                                               jlong distCoeffs2, jint method,
                                                               jdouble prob, jdouble threshold) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *e = cvk_find_essential_mat_stereo_masked(
        as_mat(points1), as_mat(points2), as_mat(cameraMatrix1), as_mat(distCoeffs1),
        as_mat(cameraMatrix2), as_mat(distCoeffs2), method, prob, threshold, &mask);
    if (e == nullptr || mask == nullptr) {
        cvk_mat_release(e);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {e, mask};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_findEssentialMatStereoUsac(
    JNIEnv *env, jobject, jlong points1, jlong points2, jlong cameraMatrix1, jlong cameraMatrix2,
    jlong distCoeffs1, jlong distCoeffs2, jdouble confidence, jboolean isParallel,
    jint loIterations, jint loMethod, jint loSampleSize, jint maxIterations, jint neighborsSearch,
    jint randomGeneratorState, jint sampler, jint score, jdouble threshold, jint finalPolisher,
    jint finalPolisherIterations) {
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *e = cvk_find_essential_mat_stereo_usac(
        as_mat(points1), as_mat(points2), as_mat(cameraMatrix1), as_mat(cameraMatrix2),
        as_mat(distCoeffs1), as_mat(distCoeffs2), &mask, confidence, isParallel, loIterations,
        loMethod, loSampleSize, maxIterations, neighborsSearch, randomGeneratorState, sampler,
        score, threshold, finalPolisher, finalPolisherIterations);
    if (e == nullptr || mask == nullptr) {
        cvk_mat_release(e);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {e, mask};
    return as_handle_array(env, handles, 2);
}

/* --------------------------------------------------- decomposeEssentialMat */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_decomposeEssentialMat(JNIEnv *env, jobject, jlong e) {
    cvk_mat_t *r1 = nullptr;
    cvk_mat_t *r2 = nullptr;
    cvk_mat_t *t = nullptr;
    cvk_decompose_essential_mat(as_mat(e), &r1, &r2, &t);
    if (r1 == nullptr || r2 == nullptr || t == nullptr) {
        cvk_mat_release(r1);
        cvk_mat_release(r2);
        cvk_mat_release(t);
        return nullptr;
    }
    cvk_mat_t *handles[3] = {r1, r2, t};
    return as_handle_array(env, handles, 3);
}

/* ------------------------------------------------------------ recoverPose */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_recoverPose(JNIEnv *env, jobject, jlong points1, jlong points2,
                                              jlong cameraMatrix1, jlong distCoeffs1,
                                              jlong cameraMatrix2, jlong distCoeffs2, jint method,
                                              jdouble prob, jdouble threshold, jlong maskIn) {
    cvk_mat_t *e = nullptr;
    cvk_mat_t *r = nullptr;
    cvk_mat_t *t = nullptr;
    cvk_mat_t *mask = nullptr;
    const int count = cvk_recover_pose(
        as_mat(points1), as_mat(points2), as_mat(cameraMatrix1), as_mat(distCoeffs1),
        as_mat(cameraMatrix2), as_mat(distCoeffs2), method, prob, threshold, as_mat(maskIn), &e,
        &r, &t, &mask);
    if (e == nullptr || r == nullptr || t == nullptr || mask == nullptr) {
        cvk_mat_release(e);
        cvk_mat_release(r);
        cvk_mat_release(t);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[5] = {nullptr, e, r, t, mask};
    handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
    return as_handle_array(env, handles, 5);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_recoverPoseE(JNIEnv *env, jobject, jlong e, jlong points1,
                                               jlong points2, jlong cameraMatrix, jlong maskIn) {
    cvk_mat_t *r = nullptr;
    cvk_mat_t *t = nullptr;
    cvk_mat_t *mask = nullptr;
    const int count = cvk_recover_pose_e(as_mat(e), as_mat(points1), as_mat(points2),
                                         as_mat(cameraMatrix), as_mat(maskIn), &r, &t, &mask);
    if (r == nullptr || t == nullptr || mask == nullptr) {
        cvk_mat_release(r);
        cvk_mat_release(t);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[4] = {nullptr, r, t, mask};
    handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
    return as_handle_array(env, handles, 4);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_recoverPoseEFocal(JNIEnv *env, jobject, jlong e, jlong points1,
                                                    jlong points2, jdouble focal, jdouble ppX,
                                                    jdouble ppY, jlong maskIn) {
    cvk_mat_t *r = nullptr;
    cvk_mat_t *t = nullptr;
    cvk_mat_t *mask = nullptr;
    const int count = cvk_recover_pose_e_focal(as_mat(e), as_mat(points1), as_mat(points2), focal,
                                               ppX, ppY, as_mat(maskIn), &r, &t, &mask);
    if (r == nullptr || t == nullptr || mask == nullptr) {
        cvk_mat_release(r);
        cvk_mat_release(t);
        cvk_mat_release(mask);
        return nullptr;
    }
    cvk_mat_t *handles[4] = {nullptr, r, t, mask};
    handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
    return as_handle_array(env, handles, 4);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_recoverPoseEDistance(JNIEnv *env, jobject, jlong e,
                                                       jlong points1, jlong points2,
                                                       jlong cameraMatrix, jdouble distanceThresh,
                                                       jlong maskIn,
                                                       jboolean wantTriangulatedPoints) {
    cvk_mat_t *r = nullptr;
    cvk_mat_t *t = nullptr;
    cvk_mat_t *mask = nullptr;
    cvk_mat_t *triangulated = nullptr;
    const int count = cvk_recover_pose_e_distance(
        as_mat(e), as_mat(points1), as_mat(points2), as_mat(cameraMatrix), distanceThresh,
        as_mat(maskIn), &r, &t, &mask, wantTriangulatedPoints ? &triangulated : nullptr);
    if (r == nullptr || t == nullptr || mask == nullptr) {
        cvk_mat_release(r);
        cvk_mat_release(t);
        cvk_mat_release(mask);
        cvk_mat_release(triangulated);
        return nullptr;
    }
    if (wantTriangulatedPoints) {
        cvk_mat_t *handles[5] = {nullptr, r, t, mask, triangulated};
        handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
        return as_handle_array(env, handles, 5);
    }
    cvk_mat_t *handles[4] = {nullptr, r, t, mask};
    handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
    return as_handle_array(env, handles, 4);
}

/* -------------------------------------------------------- triangulatePoints */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_triangulatePoints(JNIEnv *, jobject, jlong projMatr1,
                                                    jlong projMatr2, jlong projPoints1,
                                                    jlong projPoints2) {
    return as_handle(cvk_triangulate_points(as_mat(projMatr1), as_mat(projMatr2),
                                            as_mat(projPoints1), as_mat(projPoints2)));
}

/* ---------------------------------------------------------- correctMatches */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_correctMatches(JNIEnv *env, jobject, jlong f, jlong points1,
                                                 jlong points2) {
    cvk_mat_t *np1 = nullptr;
    cvk_mat_t *np2 = nullptr;
    cvk_correct_matches(as_mat(f), as_mat(points1), as_mat(points2), &np1, &np2);
    if (np1 == nullptr || np2 == nullptr) {
        cvk_mat_release(np1);
        cvk_mat_release(np2);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {np1, np2};
    return as_handle_array(env, handles, 2);
}

/* ---------------------------------------------------------- estimateAffine3D */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_estimateAffine3D(JNIEnv *env, jobject, jlong src, jlong dst,
                                                   jdouble ransacThreshold, jdouble confidence) {
    cvk_mat_t *transform = nullptr;
    cvk_mat_t *inliers = nullptr;
    if (!cvk_estimate_affine_3d(as_mat(src), as_mat(dst), ransacThreshold, confidence, &transform,
                                &inliers)) {
        cvk_mat_release(transform);
        cvk_mat_release(inliers);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {transform, inliers};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_estimateAffine3DUmeyama(JNIEnv *env, jobject, jlong src,
                                                          jlong dst, jboolean forceRotation,
                                                          jdoubleArray scaleOut) {
    double scale = 1.0;
    cvk_mat_t *transform =
        cvk_estimate_affine_3d_umeyama(as_mat(src), as_mat(dst), forceRotation, &scale);
    if (transform != nullptr) {
        env->SetDoubleArrayRegion(scaleOut, 0, 1, &scale);
    }
    return as_handle(transform);
}

/* ---------------------------------------------------------- estimateAffine2D (USAC) */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_estimateAffine2DUsac(
    JNIEnv *env, jobject, jlong points1, jlong points2, jdouble confidence, jboolean isParallel,
    jint loIterations, jint loMethod, jint loSampleSize, jint maxIterations, jint neighborsSearch,
    jint randomGeneratorState, jint sampler, jint score, jdouble threshold, jint finalPolisher,
    jint finalPolisherIterations) {
    cvk_mat_t *inliers = nullptr;
    cvk_mat_t *transform = cvk_estimate_affine_2d_usac(
        as_mat(points1), as_mat(points2), &inliers, confidence, isParallel, loIterations, loMethod,
        loSampleSize, maxIterations, neighborsSearch, randomGeneratorState, sampler, score,
        threshold, finalPolisher, finalPolisherIterations);
    if (transform == nullptr || inliers == nullptr) {
        cvk_mat_release(transform);
        cvk_mat_release(inliers);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {transform, inliers};
    return as_handle_array(env, handles, 2);
}

/* ----------------------------------------------------- decomposeHomographyMat */

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_decomposeHomographyMat(JNIEnv *env, jobject, jlong h, jlong k) {
    cvk_mat_t *rotations = nullptr;
    cvk_mat_t *translations = nullptr;
    cvk_mat_t *normals = nullptr;
    const int count = cvk_decompose_homography_mat(as_mat(h), as_mat(k), &rotations,
                                                   &translations, &normals);
    if (rotations == nullptr || translations == nullptr || normals == nullptr) {
        cvk_mat_release(rotations);
        cvk_mat_release(translations);
        cvk_mat_release(normals);
        return nullptr;
    }
    cvk_mat_t *handles[4] = {nullptr, rotations, translations, normals};
    handles[0] = reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(count));
    return as_handle_array(env, handles, 4);
}

/* ------------------------------------------------- filterHomographyDecomp */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_filterHomographyDecomp(JNIEnv *, jobject, jlong rotations,
                                                         jlong normals, jlong beforePoints,
                                                         jlong afterPoints, jlong pointsMask) {
    cvk_mat_t *solutions = nullptr;
    cvk_filter_homography_decomp(as_mat(rotations), as_mat(normals), as_mat(beforePoints),
                                 as_mat(afterPoints), as_mat(pointsMask), &solutions);
    return as_handle(solutions);
}

/* =========================================================================
 * Subdiv2D
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dCreate(JNIEnv *, jobject) {
    return as_subdiv2d_handle(cvk_subdiv2d_create());
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dCreateWithRect(JNIEnv *, jobject, jint x, jint y,
                                                         jint width, jint height) {
    return as_subdiv2d_handle(cvk_subdiv2d_create_with_rect(x, y, width, height));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dInitDelaunay(JNIEnv *, jobject, jlong handle, jint x,
                                                       jint y, jint width, jint height) {
    cvk_subdiv2d_init_delaunay(as_subdiv2d(handle), x, y, width, height);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dInsertPoint(JNIEnv *, jobject, jlong handle,
                                                      jdouble x, jdouble y) {
    return cvk_subdiv2d_insert_point(as_subdiv2d(handle), x, y);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dInsertPoints(JNIEnv *, jobject, jlong handle,
                                                       jlong points) {
    cvk_subdiv2d_insert_points(as_subdiv2d(handle), as_mat(points));
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dLocate(JNIEnv *env, jobject, jlong handle, jdouble x,
                                                 jdouble y) {
    int edge = 0;
    int vertex = 0;
    const int code = cvk_subdiv2d_locate(as_subdiv2d(handle), x, y, &edge, &vertex);
    const jint values[3] = {code, edge, vertex};
    jintArray result = env->NewIntArray(3);
    if (result == nullptr) return nullptr;
    env->SetIntArrayRegion(result, 0, 3, values);
    return result;
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dFindNearest(JNIEnv *env, jobject, jlong handle,
                                                      jdouble x, jdouble y) {
    double nx = 0.0;
    double ny = 0.0;
    const int vertex = cvk_subdiv2d_find_nearest(as_subdiv2d(handle), x, y, &nx, &ny);
    const jdouble values[3] = {static_cast<jdouble>(vertex), nx, ny};
    jdoubleArray result = env->NewDoubleArray(3);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 3, values);
    return result;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dGetEdgeList(JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_subdiv2d_get_edge_list(as_subdiv2d(handle)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dGetLeadingEdgeList(JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_subdiv2d_get_leading_edge_list(as_subdiv2d(handle)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dGetTriangleList(JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_subdiv2d_get_triangle_list(as_subdiv2d(handle)));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dGetVoronoiFacetList(JNIEnv *env, jobject, jlong handle,
                                                              jlong idx) {
    cvk_mat_t *centers = nullptr;
    cvk_mat_t *facets = cvk_subdiv2d_get_voronoi_facet_list(as_subdiv2d(handle), as_mat(idx),
                                                            &centers);
    if (facets == nullptr || centers == nullptr) {
        cvk_mat_release(facets);
        cvk_mat_release(centers);
        return nullptr;
    }
    cvk_mat_t *handles[2] = {facets, centers};
    return as_handle_array(env, handles, 2);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dGetVertex(JNIEnv *env, jobject, jlong handle,
                                                    jint vertex) {
    double x = 0.0;
    double y = 0.0;
    int first_edge = 0;
    cvk_subdiv2d_get_vertex(as_subdiv2d(handle), vertex, &x, &y, &first_edge);
    const jdouble values[3] = {x, y, static_cast<jdouble>(first_edge)};
    jdoubleArray result = env->NewDoubleArray(3);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 3, values);
    return result;
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dGetEdge(JNIEnv *, jobject, jlong handle, jint edge,
                                                  jint nextEdgeType) {
    return cvk_subdiv2d_get_edge(as_subdiv2d(handle), edge, nextEdgeType);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dNextEdge(JNIEnv *, jobject, jlong handle, jint edge) {
    return cvk_subdiv2d_next_edge(as_subdiv2d(handle), edge);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dRotateEdge(JNIEnv *, jobject, jlong handle, jint edge,
                                                     jint rotate) {
    return cvk_subdiv2d_rotate_edge(as_subdiv2d(handle), edge, rotate);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dSymEdge(JNIEnv *, jobject, jlong handle, jint edge) {
    return cvk_subdiv2d_sym_edge(as_subdiv2d(handle), edge);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dEdgeOrg(JNIEnv *env, jobject, jlong handle, jint edge) {
    double x = 0.0;
    double y = 0.0;
    const int vertex = cvk_subdiv2d_edge_org(as_subdiv2d(handle), edge, &x, &y);
    const jdouble values[3] = {static_cast<jdouble>(vertex), x, y};
    jdoubleArray result = env->NewDoubleArray(3);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 3, values);
    return result;
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dEdgeDst(JNIEnv *env, jobject, jlong handle, jint edge) {
    double x = 0.0;
    double y = 0.0;
    const int vertex = cvk_subdiv2d_edge_dst(as_subdiv2d(handle), edge, &x, &y);
    const jdouble values[3] = {static_cast<jdouble>(vertex), x, y};
    jdoubleArray result = env->NewDoubleArray(3);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 3, values);
    return result;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniGeometry_subdiv2dRelease(JNIEnv *, jobject, jlong handle) {
    cvk_subdiv2d_release(as_subdiv2d(handle));
}

} /* extern "C" */

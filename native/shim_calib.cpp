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
 * C shim for the OpenCV "calib" module (camera calibration).
 *
 * Every function is noexcept: bodies run inside guarded(), which catches
 * cv::Exception/std::exception and reports through the thread-local error
 * (mirroring opencv_shim.cpp). Pointers default to NULL and doubles default
 * to NaN on failure so callers can distinguish a failed calibration from a
 * legitimate zero RMS error.
 *
 * vector-of-Mat wire format: a CV_32SC2 Nx1 Mat whose i-th row holds the
 * high/low 32 bits of the i-th cv::Mat address. Input lists are decoded to
 * std::vector<cv::Mat> (shallow copies sharing pixels); output lists are
 * encoded by heap-allocating per-view Mats whose handles the Kotlin layer
 * wraps and releases.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_calib.h"

#include <opencv2/calib.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

thread_local std::string g_calib_last_error;

void record_error(const char *message) {
    g_calib_last_error = message != nullptr ? message : "unknown error";
}

/** Default failure value per return type: NULL for pointers, NaN for doubles. */
template <typename T>
struct guarded_default {
    static T value() { return T(); }
};
template <>
struct guarded_default<double> {
    static double value() { return std::numeric_limits<double>::quiet_NaN(); }
};

template <typename F>
auto guarded(F &&body) -> decltype(body()) {
    try {
        return body();
    } catch (const cv::Exception &e) {
        record_error(e.what());
    } catch (const std::exception &e) {
        record_error(e.what());
    } catch (...) {
        record_error("unknown native error");
    }
    return guarded_default<decltype(body())>::value();
}

cv::Mat *require(cvk_mat_t *mat) {
    if (mat == nullptr) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<cv::Mat *>(mat);
}

const cv::Mat *require_const(const cvk_mat_t *mat) {
    return require(const_cast<cvk_mat_t *>(mat));
}

/**
 * Decodes a CV_32SC2 Nx1 wire Mat into a vector of shallow Mat copies
 * (shared pixel data). An empty wire yields an empty vector.
 */
bool wire_to_mats(const cv::Mat &wire, std::vector<cv::Mat> &out) {
    out.clear();
    if (wire.empty()) return true;
    if (wire.type() != CV_32SC2 || wire.cols != 1) {
        record_error("expected CV_32SC2 Nx1 vector-of-Mat wire");
        return false;
    }
    out.reserve(static_cast<size_t>(wire.rows));
    for (int i = 0; i < wire.rows; i++) {
        const int *p = wire.ptr<int>(i);
        uint64_t addr = (static_cast<uint64_t>(static_cast<uint32_t>(p[0])) << 32) |
                        static_cast<uint64_t>(static_cast<uint32_t>(p[1]));
        auto *m = reinterpret_cast<cv::Mat *>(static_cast<uintptr_t>(addr));
        if (m == nullptr) {
            record_error("null Mat handle inside vector-of-Mat wire");
            return false;
        }
        out.push_back(*m);
    }
    return true;
}

/**
 * Encodes a vector of Mats into the CV_32SC2 Nx1 wire format, allocating a
 * heap copy per Mat so the handles survive the local vector's destruction.
 * An empty vector releases the output Mat.
 */
bool mats_to_wire(std::vector<cv::Mat> &src, cv::Mat &out) {
    if (src.empty()) {
        out.release();
        return true;
    }
    out.create(static_cast<int>(src.size()), 1, CV_32SC2);
    for (size_t i = 0; i < src.size(); i++) {
        auto *h = new cv::Mat(src[i]);
        int64_t addr = static_cast<int64_t>(reinterpret_cast<uintptr_t>(h));
        int *p = out.ptr<int>(static_cast<int>(i));
        p[0] = static_cast<int>(addr >> 32);
        p[1] = static_cast<int>(addr & 0xffffffffLL);
    }
    return true;
}

} // namespace

extern "C" {

cvk_mat_t *cvk_init_camera_matrix_2d(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points,
                                     int image_width, int image_height,
                                     double aspect_ratio) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img = require_const(image_points);
        if (obj == nullptr || img == nullptr) return nullptr;
        std::vector<cv::Mat> objMats, imgMats;
        if (!wire_to_mats(*obj, objMats)) return nullptr;
        if (!wire_to_mats(*img, imgMats)) return nullptr;
        cv::Mat result = cv::initCameraMatrix2D(objMats, imgMats,
                                                cv::Size(image_width, image_height),
                                                aspect_ratio);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(result));
    });
}

double cvk_calibrate_camera(const cvk_mat_t *object_points,
                            const cvk_mat_t *image_points,
                            int image_width, int image_height,
                            cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                            cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                            int flags, int criteria_type, int criteria_max_count,
                            double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img = require_const(image_points);
        cv::Mat *cm = require(camera_matrix);
        cv::Mat *dc = require(dist_coeffs);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        if (obj == nullptr || img == nullptr || cm == nullptr || dc == nullptr ||
            rv == nullptr || tv == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, imgMats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img, imgMats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::calibrateCamera(
            objMats, imgMats, cv::Size(image_width, image_height), *cm, *dc,
            rvecsOut, tvecsOut, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_calibrate_camera_extended(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points,
                                     int image_width, int image_height,
                                     cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                                     cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                     cvk_mat_t *std_deviations_intrinsics,
                                     cvk_mat_t *std_deviations_extrinsics,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img = require_const(image_points);
        cv::Mat *cm = require(camera_matrix);
        cv::Mat *dc = require(dist_coeffs);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        cv::Mat *sdi = require(std_deviations_intrinsics);
        cv::Mat *sde = require(std_deviations_extrinsics);
        cv::Mat *pve = require(per_view_errors);
        if (obj == nullptr || img == nullptr || cm == nullptr || dc == nullptr ||
            rv == nullptr || tv == nullptr || sdi == nullptr || sde == nullptr ||
            pve == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, imgMats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img, imgMats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::calibrateCamera(
            objMats, imgMats, cv::Size(image_width, image_height), *cm, *dc,
            rvecsOut, tvecsOut, *sdi, *sde, *pve, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_calibrate_camera_ro(const cvk_mat_t *object_points,
                               const cvk_mat_t *image_points,
                               int image_width, int image_height, int i_fixed_point,
                               cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                               cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                               cvk_mat_t *new_obj_points,
                               int flags, int criteria_type, int criteria_max_count,
                               double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img = require_const(image_points);
        cv::Mat *cm = require(camera_matrix);
        cv::Mat *dc = require(dist_coeffs);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        cv::Mat *nop = require(new_obj_points);
        if (obj == nullptr || img == nullptr || cm == nullptr || dc == nullptr ||
            rv == nullptr || tv == nullptr || nop == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, imgMats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img, imgMats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::calibrateCameraRO(
            objMats, imgMats, cv::Size(image_width, image_height), i_fixed_point,
            *cm, *dc, rvecsOut, tvecsOut, *nop, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_calibrate_camera_ro_extended(const cvk_mat_t *object_points,
                                        const cvk_mat_t *image_points,
                                        int image_width, int image_height, int i_fixed_point,
                                        cvk_mat_t *camera_matrix, cvk_mat_t *dist_coeffs,
                                        cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                        cvk_mat_t *new_obj_points,
                                        cvk_mat_t *std_deviations_intrinsics,
                                        cvk_mat_t *std_deviations_extrinsics,
                                        cvk_mat_t *std_deviations_obj_points,
                                        cvk_mat_t *per_view_errors,
                                        int flags, int criteria_type, int criteria_max_count,
                                        double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img = require_const(image_points);
        cv::Mat *cm = require(camera_matrix);
        cv::Mat *dc = require(dist_coeffs);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        cv::Mat *nop = require(new_obj_points);
        cv::Mat *sdi = require(std_deviations_intrinsics);
        cv::Mat *sde = require(std_deviations_extrinsics);
        cv::Mat *sdop = require(std_deviations_obj_points);
        cv::Mat *pve = require(per_view_errors);
        if (obj == nullptr || img == nullptr || cm == nullptr || dc == nullptr ||
            rv == nullptr || tv == nullptr || nop == nullptr || sdi == nullptr ||
            sde == nullptr || sdop == nullptr || pve == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, imgMats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img, imgMats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::calibrateCameraRO(
            objMats, imgMats, cv::Size(image_width, image_height), i_fixed_point,
            *cm, *dc, rvecsOut, tvecsOut, *nop, *sdi, *sde, *sdop, *pve, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_stereo_calibrate(const cvk_mat_t *object_points,
                            const cvk_mat_t *image_points1,
                            const cvk_mat_t *image_points2,
                            int image_width, int image_height,
                            cvk_mat_t *camera_matrix1, cvk_mat_t *dist_coeffs1,
                            cvk_mat_t *camera_matrix2, cvk_mat_t *dist_coeffs2,
                            cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                            int flags, int criteria_type, int criteria_max_count,
                            double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img1 = require_const(image_points1);
        const cv::Mat *img2 = require_const(image_points2);
        cv::Mat *cm1 = require(camera_matrix1);
        cv::Mat *dc1 = require(dist_coeffs1);
        cv::Mat *cm2 = require(camera_matrix2);
        cv::Mat *dc2 = require(dist_coeffs2);
        cv::Mat *R = require(r);
        cv::Mat *T = require(t);
        cv::Mat *E = require(e);
        cv::Mat *F = require(f);
        if (obj == nullptr || img1 == nullptr || img2 == nullptr || cm1 == nullptr ||
            dc1 == nullptr || cm2 == nullptr || dc2 == nullptr || R == nullptr ||
            T == nullptr || E == nullptr || F == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, img1Mats, img2Mats;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img1, img1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img2, img2Mats)) return std::numeric_limits<double>::quiet_NaN();
        return cv::stereoCalibrate(
            objMats, img1Mats, img2Mats, *cm1, *dc1, *cm2, *dc2,
            cv::Size(image_width, image_height), *R, *T, *E, *F, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
    });
}

double cvk_stereo_calibrate_per_view(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points1,
                                     const cvk_mat_t *image_points2,
                                     int image_width, int image_height,
                                     cvk_mat_t *camera_matrix1, cvk_mat_t *dist_coeffs1,
                                     cvk_mat_t *camera_matrix2, cvk_mat_t *dist_coeffs2,
                                     cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img1 = require_const(image_points1);
        const cv::Mat *img2 = require_const(image_points2);
        cv::Mat *cm1 = require(camera_matrix1);
        cv::Mat *dc1 = require(dist_coeffs1);
        cv::Mat *cm2 = require(camera_matrix2);
        cv::Mat *dc2 = require(dist_coeffs2);
        cv::Mat *R = require(r);
        cv::Mat *T = require(t);
        cv::Mat *E = require(e);
        cv::Mat *F = require(f);
        cv::Mat *pve = require(per_view_errors);
        if (obj == nullptr || img1 == nullptr || img2 == nullptr || cm1 == nullptr ||
            dc1 == nullptr || cm2 == nullptr || dc2 == nullptr || R == nullptr ||
            T == nullptr || E == nullptr || F == nullptr || pve == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, img1Mats, img2Mats;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img1, img1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img2, img2Mats)) return std::numeric_limits<double>::quiet_NaN();
        return cv::stereoCalibrate(
            objMats, img1Mats, img2Mats, *cm1, *dc1, *cm2, *dc2,
            cv::Size(image_width, image_height), *R, *T, *E, *F, *pve, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
    });
}

double cvk_stereo_calibrate_extended(const cvk_mat_t *object_points,
                                     const cvk_mat_t *image_points1,
                                     const cvk_mat_t *image_points2,
                                     int image_width, int image_height,
                                     cvk_mat_t *camera_matrix1, cvk_mat_t *dist_coeffs1,
                                     cvk_mat_t *camera_matrix2, cvk_mat_t *dist_coeffs2,
                                     cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                                     cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img1 = require_const(image_points1);
        const cv::Mat *img2 = require_const(image_points2);
        cv::Mat *cm1 = require(camera_matrix1);
        cv::Mat *dc1 = require(dist_coeffs1);
        cv::Mat *cm2 = require(camera_matrix2);
        cv::Mat *dc2 = require(dist_coeffs2);
        cv::Mat *R = require(r);
        cv::Mat *T = require(t);
        cv::Mat *E = require(e);
        cv::Mat *F = require(f);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        cv::Mat *pve = require(per_view_errors);
        if (obj == nullptr || img1 == nullptr || img2 == nullptr || cm1 == nullptr ||
            dc1 == nullptr || cm2 == nullptr || dc2 == nullptr || R == nullptr ||
            T == nullptr || E == nullptr || F == nullptr || rv == nullptr ||
            tv == nullptr || pve == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, img1Mats, img2Mats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img1, img1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img2, img2Mats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::stereoCalibrate(
            objMats, img1Mats, img2Mats, *cm1, *dc1, *cm2, *dc2,
            cv::Size(image_width, image_height), *R, *T, *E, *F, rvecsOut, tvecsOut,
            *pve, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_register_cameras(const cvk_mat_t *object_points1,
                            const cvk_mat_t *object_points2,
                            const cvk_mat_t *image_points1,
                            const cvk_mat_t *image_points2,
                            const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                            int camera_model1,
                            const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                            int camera_model2,
                            cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                            cvk_mat_t *per_view_errors,
                            int flags, int criteria_type, int criteria_max_count,
                            double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *op1 = require_const(object_points1);
        const cv::Mat *op2 = require_const(object_points2);
        const cv::Mat *ip1 = require_const(image_points1);
        const cv::Mat *ip2 = require_const(image_points2);
        const cv::Mat *cm1 = require_const(camera_matrix1);
        const cv::Mat *dc1 = require_const(dist_coeffs1);
        const cv::Mat *cm2 = require_const(camera_matrix2);
        const cv::Mat *dc2 = require_const(dist_coeffs2);
        cv::Mat *R = require(r);
        cv::Mat *T = require(t);
        cv::Mat *E = require(e);
        cv::Mat *F = require(f);
        cv::Mat *pve = require(per_view_errors);
        if (op1 == nullptr || op2 == nullptr || ip1 == nullptr || ip2 == nullptr ||
            cm1 == nullptr || dc1 == nullptr || cm2 == nullptr || dc2 == nullptr ||
            R == nullptr || T == nullptr || E == nullptr || F == nullptr || pve == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> op1Mats, op2Mats, ip1Mats, ip2Mats;
        if (!wire_to_mats(*op1, op1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*op2, op2Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*ip1, ip1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*ip2, ip2Mats)) return std::numeric_limits<double>::quiet_NaN();
        return cv::registerCameras(
            op1Mats, op2Mats, ip1Mats, ip2Mats, *cm1, *dc1,
            static_cast<cv::CameraModel>(camera_model1), *cm2, *dc2,
            static_cast<cv::CameraModel>(camera_model2), *R, *T, *E, *F, *pve, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
    });
}

double cvk_register_cameras_extended(const cvk_mat_t *object_points1,
                                     const cvk_mat_t *object_points2,
                                     const cvk_mat_t *image_points1,
                                     const cvk_mat_t *image_points2,
                                     const cvk_mat_t *camera_matrix1, const cvk_mat_t *dist_coeffs1,
                                     int camera_model1,
                                     const cvk_mat_t *camera_matrix2, const cvk_mat_t *dist_coeffs2,
                                     int camera_model2,
                                     cvk_mat_t *r, cvk_mat_t *t, cvk_mat_t *e, cvk_mat_t *f,
                                     cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                     cvk_mat_t *per_view_errors,
                                     int flags, int criteria_type, int criteria_max_count,
                                     double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *op1 = require_const(object_points1);
        const cv::Mat *op2 = require_const(object_points2);
        const cv::Mat *ip1 = require_const(image_points1);
        const cv::Mat *ip2 = require_const(image_points2);
        const cv::Mat *cm1 = require_const(camera_matrix1);
        const cv::Mat *dc1 = require_const(dist_coeffs1);
        const cv::Mat *cm2 = require_const(camera_matrix2);
        const cv::Mat *dc2 = require_const(dist_coeffs2);
        cv::Mat *R = require(r);
        cv::Mat *T = require(t);
        cv::Mat *E = require(e);
        cv::Mat *F = require(f);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        cv::Mat *pve = require(per_view_errors);
        if (op1 == nullptr || op2 == nullptr || ip1 == nullptr || ip2 == nullptr ||
            cm1 == nullptr || dc1 == nullptr || cm2 == nullptr || dc2 == nullptr ||
            R == nullptr || T == nullptr || E == nullptr || F == nullptr ||
            rv == nullptr || tv == nullptr || pve == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> op1Mats, op2Mats, ip1Mats, ip2Mats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*op1, op1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*op2, op2Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*ip1, ip1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*ip2, ip2Mats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::registerCameras(
            op1Mats, op2Mats, ip1Mats, ip2Mats, *cm1, *dc1,
            static_cast<cv::CameraModel>(camera_model1), *cm2, *dc2,
            static_cast<cv::CameraModel>(camera_model2), *R, *T, *E, *F,
            rvecsOut, tvecsOut, *pve, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_fisheye_calibrate(const cvk_mat_t *object_points,
                             const cvk_mat_t *image_points,
                             int image_width, int image_height,
                             cvk_mat_t *k, cvk_mat_t *d,
                             cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                             int flags, int criteria_type, int criteria_max_count,
                             double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img = require_const(image_points);
        cv::Mat *K = require(k);
        cv::Mat *D = require(d);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        if (obj == nullptr || img == nullptr || K == nullptr || D == nullptr ||
            rv == nullptr || tv == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, imgMats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img, imgMats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::fisheye::calibrate(
            objMats, imgMats, cv::Size(image_width, image_height), *K, *D,
            rvecsOut, tvecsOut, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_fisheye_stereo_calibrate(const cvk_mat_t *object_points,
                                    const cvk_mat_t *image_points1,
                                    const cvk_mat_t *image_points2,
                                    int image_width, int image_height,
                                    cvk_mat_t *k1, cvk_mat_t *d1,
                                    cvk_mat_t *k2, cvk_mat_t *d2,
                                    cvk_mat_t *r, cvk_mat_t *t,
                                    cvk_mat_t *rvecs, cvk_mat_t *tvecs,
                                    int flags, int criteria_type, int criteria_max_count,
                                    double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img1 = require_const(image_points1);
        const cv::Mat *img2 = require_const(image_points2);
        cv::Mat *K1 = require(k1);
        cv::Mat *D1 = require(d1);
        cv::Mat *K2 = require(k2);
        cv::Mat *D2 = require(d2);
        cv::Mat *R = require(r);
        cv::Mat *T = require(t);
        cv::Mat *rv = require(rvecs);
        cv::Mat *tv = require(tvecs);
        if (obj == nullptr || img1 == nullptr || img2 == nullptr || K1 == nullptr ||
            D1 == nullptr || K2 == nullptr || D2 == nullptr || R == nullptr ||
            T == nullptr || rv == nullptr || tv == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, img1Mats, img2Mats, rvecsOut, tvecsOut;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img1, img1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img2, img2Mats)) return std::numeric_limits<double>::quiet_NaN();
        double err = cv::fisheye::stereoCalibrate(
            objMats, img1Mats, img2Mats, *K1, *D1, *K2, *D2,
            cv::Size(image_width, image_height), *R, *T, rvecsOut, tvecsOut, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
        if (!mats_to_wire(rvecsOut, *rv)) return std::numeric_limits<double>::quiet_NaN();
        if (!mats_to_wire(tvecsOut, *tv)) return std::numeric_limits<double>::quiet_NaN();
        return err;
    });
}

double cvk_fisheye_stereo_calibrate_pose(const cvk_mat_t *object_points,
                                         const cvk_mat_t *image_points1,
                                         const cvk_mat_t *image_points2,
                                         int image_width, int image_height,
                                         cvk_mat_t *k1, cvk_mat_t *d1,
                                         cvk_mat_t *k2, cvk_mat_t *d2,
                                         cvk_mat_t *r, cvk_mat_t *t,
                                         int flags, int criteria_type, int criteria_max_count,
                                         double criteria_epsilon) {
    return guarded([&]() -> double {
        const cv::Mat *obj = require_const(object_points);
        const cv::Mat *img1 = require_const(image_points1);
        const cv::Mat *img2 = require_const(image_points2);
        cv::Mat *K1 = require(k1);
        cv::Mat *D1 = require(d1);
        cv::Mat *K2 = require(k2);
        cv::Mat *D2 = require(d2);
        cv::Mat *R = require(r);
        cv::Mat *T = require(t);
        if (obj == nullptr || img1 == nullptr || img2 == nullptr || K1 == nullptr ||
            D1 == nullptr || K2 == nullptr || D2 == nullptr || R == nullptr ||
            T == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<cv::Mat> objMats, img1Mats, img2Mats;
        if (!wire_to_mats(*obj, objMats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img1, img1Mats)) return std::numeric_limits<double>::quiet_NaN();
        if (!wire_to_mats(*img2, img2Mats)) return std::numeric_limits<double>::quiet_NaN();
        return cv::fisheye::stereoCalibrate(
            objMats, img1Mats, img2Mats, *K1, *D1, *K2, *D2,
            cv::Size(image_width, image_height), *R, *T, flags,
            cv::TermCriteria(criteria_type, criteria_max_count, criteria_epsilon));
    });
}

} /* extern "C" */

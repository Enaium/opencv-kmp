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
 * cvk_ ABI implementation for the objdetect module: ArUco (dictionary, boards,
 * detectors, params) plus the chessboard/circles-grid statics.
 *
 * List<Mat> marshalling uses the CV_32SC2 address-container wire format (see
 * opencv_kmp_objdetect.h): input containers decode into std::vector<cv::Mat>
 * sharing the caller's data; output containers heap-allocate each Mat and
 * encode its address, transferring ownership to the caller.
 *
 * GridBoard/CharucoBoard handles are distinct opaque types whose first member
 * is the Board base subobject, so a cvk_grid_board_t* / cvk_charuco_board_t*
 * may be passed where a cvk_board_t* is expected (offset-0 single inheritance).
 */
#include "opencv_kmp.h"
#include "opencv_kmp_objdetect.h"
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct cvk_dictionary { cv::aruco::Dictionary d; };
struct cvk_board { cv::aruco::Board b; };
struct cvk_grid_board { cv::aruco::GridBoard b; };
struct cvk_charuco_board { cv::aruco::CharucoBoard b; };
struct cvk_aruco_detector { cv::Ptr<cv::aruco::ArucoDetector> ptr; };
struct cvk_charuco_detector { cv::Ptr<cv::aruco::CharucoDetector> ptr; };

namespace {

thread_local std::string g_objdetect_str;
thread_local std::string g_last_error;

void record_error(const char *m) { g_last_error = m ? m : "unknown error"; }

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
    return decltype(body())();
}

cv::Mat *require(cvk_mat_t *m) {
    if (!m) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<cv::Mat *>(m);
}

const cv::Mat *require_const(const cvk_mat_t *m) {
    if (!m) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<const cv::Mat *>(m);
}

/* ---- vector<Mat> <-> CV_32SC2 address container ------------------------ */

/** Decodes a container into Mat copies sharing the source data. */
bool mats_from_container(const cvk_mat_t *container, std::vector<cv::Mat> *out) {
    out->clear();
    if (container == nullptr) return true; /* NULL container == empty list */
    const cv::Mat *m = require_const(container);
    if (m == nullptr) return false;
    if (m->empty()) return true;
    if (m->type() != CV_32SC2 || m->cols != 1) {
        record_error("vector-of-Mat container must be CV_32SC2 with one column");
        return false;
    }
    const int count = m->rows;
    for (int i = 0; i < count; i++) {
        const cv::Vec2i cell = m->at<cv::Vec2i>(i, 0);
        const uintptr_t addr =
            (static_cast<uintptr_t>(static_cast<uint32_t>(cell[0])) << 32) |
            static_cast<uint32_t>(cell[1]);
        auto *mat = reinterpret_cast<cv::Mat *>(addr);
        if (mat == nullptr) {
            record_error("null Mat address in vector-of-Mat container");
            return false;
        }
        out->push_back(*mat);
    }
    return true;
}

/**
 * Encodes mats into a fresh container Mat. Every element is deep-copied to
 * the heap (new cv::Mat) and owned by whoever decodes the container.
 */
cvk_mat_t *container_of_mats(const std::vector<cv::Mat> &mats) {
    if (mats.empty()) return reinterpret_cast<cvk_mat_t *>(new cv::Mat());
    cv::Mat container(static_cast<int>(mats.size()), 1, CV_32SC2);
    for (size_t i = 0; i < mats.size(); i++) {
        auto *copy = new cv::Mat(mats[i]);
        const uintptr_t addr = reinterpret_cast<uintptr_t>(copy);
        container.at<cv::Vec2i>(static_cast<int>(i), 0) =
            cv::Vec2i(static_cast<int>(addr >> 32), static_cast<int>(addr & 0xffffffffu));
    }
    return reinterpret_cast<cvk_mat_t *>(new cv::Mat(container));
}

/* ---- DetectorParameters <-> cvk struct ---------------------------------- */

cv::aruco::DetectorParameters to_cv(const cvk_detector_params_t &p) {
    cv::aruco::DetectorParameters d;
    d.adaptiveThreshWinSizeMin = p.adaptive_thresh_win_size_min;
    d.adaptiveThreshWinSizeMax = p.adaptive_thresh_win_size_max;
    d.adaptiveThreshWinSizeStep = p.adaptive_thresh_win_size_step;
    d.adaptiveThreshConstant = p.adaptive_thresh_constant;
    d.minMarkerPerimeterRate = p.min_marker_perimeter_rate;
    d.maxMarkerPerimeterRate = p.max_marker_perimeter_rate;
    d.polygonalApproxAccuracyRate = p.polygonal_approx_accuracy_rate;
    d.minCornerDistanceRate = p.min_corner_distance_rate;
    d.minDistanceToBorder = p.min_distance_to_border;
    d.minMarkerDistanceRate = p.min_marker_distance_rate;
    d.minGroupDistance = p.min_group_distance;
    d.cornerRefinementMethod = p.corner_refinement_method;
    d.cornerRefinementWinSize = p.corner_refinement_win_size;
    d.relativeCornerRefinmentWinSize = p.relative_corner_refinment_win_size;
    d.cornerRefinementMaxIterations = p.corner_refinement_max_iterations;
    d.cornerRefinementMinAccuracy = p.corner_refinement_min_accuracy;
    d.markerBorderBits = p.marker_border_bits;
    d.perspectiveRemovePixelPerCell = p.perspective_remove_pixel_per_cell;
    d.perspectiveRemoveIgnoredMarginPerCell = p.perspective_remove_ignored_margin_per_cell;
    d.maxErroneousBitsInBorderRate = p.max_erroneous_bits_in_border_rate;
    d.minOtsuStdDev = p.min_otsu_std_dev;
    d.errorCorrectionRate = p.error_correction_rate;
    d.aprilTagQuadDecimate = p.april_tag_quad_decimate;
    d.aprilTagQuadSigma = p.april_tag_quad_sigma;
    d.aprilTagMinClusterPixels = p.april_tag_min_cluster_pixels;
    d.aprilTagMaxNmaxima = p.april_tag_max_nmaxima;
    d.aprilTagCriticalRad = p.april_tag_critical_rad;
    d.aprilTagMaxLineFitMse = p.april_tag_max_line_fit_mse;
    d.aprilTagMinWhiteBlackDiff = p.april_tag_min_white_black_diff;
    d.aprilTagDeglitch = p.april_tag_deglitch;
    d.detectInvertedMarker = p.detect_inverted_marker != 0;
    d.useAruco3Detection = p.use_aruco != 0;
    d.minSideLengthCanonicalImg = p.min_side_length_canonical_img;
    d.minMarkerLengthRatioOriginalImg = p.min_marker_length_ratio_original_img;
    d.validBitIdThreshold = p.valid_bit_id_threshold;
    return d;
}

void from_cv(const cv::aruco::DetectorParameters &d, cvk_detector_params_t *p) {
    p->adaptive_thresh_win_size_min = d.adaptiveThreshWinSizeMin;
    p->adaptive_thresh_win_size_max = d.adaptiveThreshWinSizeMax;
    p->adaptive_thresh_win_size_step = d.adaptiveThreshWinSizeStep;
    p->adaptive_thresh_constant = d.adaptiveThreshConstant;
    p->min_marker_perimeter_rate = d.minMarkerPerimeterRate;
    p->max_marker_perimeter_rate = d.maxMarkerPerimeterRate;
    p->polygonal_approx_accuracy_rate = d.polygonalApproxAccuracyRate;
    p->min_corner_distance_rate = d.minCornerDistanceRate;
    p->min_distance_to_border = d.minDistanceToBorder;
    p->min_marker_distance_rate = d.minMarkerDistanceRate;
    p->min_group_distance = d.minGroupDistance;
    p->corner_refinement_method = d.cornerRefinementMethod;
    p->corner_refinement_win_size = d.cornerRefinementWinSize;
    p->relative_corner_refinment_win_size = d.relativeCornerRefinmentWinSize;
    p->corner_refinement_max_iterations = d.cornerRefinementMaxIterations;
    p->corner_refinement_min_accuracy = d.cornerRefinementMinAccuracy;
    p->marker_border_bits = d.markerBorderBits;
    p->perspective_remove_pixel_per_cell = d.perspectiveRemovePixelPerCell;
    p->perspective_remove_ignored_margin_per_cell = d.perspectiveRemoveIgnoredMarginPerCell;
    p->max_erroneous_bits_in_border_rate = d.maxErroneousBitsInBorderRate;
    p->min_otsu_std_dev = d.minOtsuStdDev;
    p->error_correction_rate = d.errorCorrectionRate;
    p->april_tag_quad_decimate = d.aprilTagQuadDecimate;
    p->april_tag_quad_sigma = d.aprilTagQuadSigma;
    p->april_tag_min_cluster_pixels = d.aprilTagMinClusterPixels;
    p->april_tag_max_nmaxima = d.aprilTagMaxNmaxima;
    p->april_tag_critical_rad = d.aprilTagCriticalRad;
    p->april_tag_max_line_fit_mse = d.aprilTagMaxLineFitMse;
    p->april_tag_min_white_black_diff = d.aprilTagMinWhiteBlackDiff;
    p->april_tag_deglitch = d.aprilTagDeglitch;
    p->detect_inverted_marker = d.detectInvertedMarker ? 1 : 0;
    p->use_aruco = d.useAruco3Detection ? 1 : 0;
    p->min_side_length_canonical_img = d.minSideLengthCanonicalImg;
    p->min_marker_length_ratio_original_img = d.minMarkerLengthRatioOriginalImg;
    p->valid_bit_id_threshold = d.validBitIdThreshold;
}

cv::aruco::RefineParameters to_cv(const cvk_refine_params_t &p) {
    return cv::aruco::RefineParameters(p.min_rep_distance, p.error_correction_rate,
                                       p.check_all_orders != 0);
}

void from_cv(const cv::aruco::RefineParameters &d, cvk_refine_params_t *p) {
    p->min_rep_distance = d.minRepDistance;
    p->error_correction_rate = d.errorCorrectionRate;
    p->check_all_orders = d.checkAllOrders ? 1 : 0;
}

cv::aruco::CharucoParameters to_cv(const cvk_charuco_params_t &p) {
    cv::aruco::CharucoParameters cp;
    cp.cameraMatrix = p.camera_matrix ? *require_const(p.camera_matrix) : cv::Mat();
    cp.distCoeffs = p.dist_coeffs ? *require_const(p.dist_coeffs) : cv::Mat();
    cp.minMarkers = p.min_markers;
    cp.tryRefineMarkers = p.try_refine_markers != 0;
    cp.checkMarkers = p.check_markers != 0;
    return cp;
}

void from_cv(const cv::aruco::CharucoParameters &d, cvk_charuco_params_t *p) {
    p->camera_matrix =
        d.cameraMatrix.empty() ? nullptr : reinterpret_cast<cvk_mat_t *>(new cv::Mat(d.cameraMatrix));
    p->dist_coeffs =
        d.distCoeffs.empty() ? nullptr : reinterpret_cast<cvk_mat_t *>(new cv::Mat(d.distCoeffs));
    p->min_markers = d.minMarkers;
    p->try_refine_markers = d.tryRefineMarkers ? 1 : 0;
    p->check_markers = d.checkMarkers ? 1 : 0;
}

/* Algorithm clear/empty/save/getDefaultName for Ptr-based handle structs. */
#define CVK_ALG_FUNCS(T)                                                       \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                     \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        if (!p || !p->ptr) { record_error("null " #T " handle"); return; }     \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                  \
    }                                                                          \
    int cvk_##T##_empty(cvk_##T##_t *h) {                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        if (!p || !p->ptr) { record_error("null " #T " handle"); return 1; }   \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });      \
    }                                                                          \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {                \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        if (!p || !p->ptr) { record_error("null " #T " handle"); return; }     \
        guarded([&]() -> int { p->ptr->save(filename ? filename : ""); return 0; }); \
    }                                                                          \
    const char *cvk_##T##_get_default_name(cvk_##T##_t *h) {                   \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        if (!p || !p->ptr) { record_error("null " #T " handle"); return nullptr; } \
        return guarded([&]() -> const char * {                                 \
            g_objdetect_str = p->ptr->getDefaultName();                        \
            return g_objdetect_str.c_str();                                    \
        });                                                                    \
    }

} /* namespace */

extern "C" {

/* =========================================================================
 * Dictionary
 * ========================================================================= */

cvk_dictionary_t *cvk_dictionary_create(const cvk_mat_t *bytes_list, int marker_size,
                                        int max_correction_bits) {
    if (bytes_list == nullptr) {
        record_error("null bytesList Mat");
        return nullptr;
    }
    return guarded([&]() -> cvk_dictionary_t * {
        auto *handle = new cvk_dictionary;
        handle->d = cv::aruco::Dictionary(*require_const(bytes_list), marker_size,
                                          max_correction_bits);
        return reinterpret_cast<cvk_dictionary_t *>(handle);
    });
}

void cvk_dictionary_release(cvk_dictionary_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_dictionary *>(h);
        return nullptr;
    });
}

cvk_mat_t *cvk_dictionary_get_bytes_list(const cvk_dictionary_t *h) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(h->d.bytesList));
    });
}

void cvk_dictionary_set_bytes_list(cvk_dictionary_t *h, const cvk_mat_t *bytes_list) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return;
    }
    guarded([&]() -> void * {
        h->d.bytesList = *require_const(bytes_list);
        return nullptr;
    });
}

int cvk_dictionary_get_marker_size(const cvk_dictionary_t *h) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return 0;
    }
    return guarded([&]() -> int { return h->d.markerSize; });
}

void cvk_dictionary_set_marker_size(cvk_dictionary_t *h, int marker_size) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return;
    }
    guarded([&]() -> void * {
        h->d.markerSize = marker_size;
        return nullptr;
    });
}

int cvk_dictionary_get_max_correction_bits(const cvk_dictionary_t *h) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return 0;
    }
    return guarded([&]() -> int { return h->d.maxCorrectionBits; });
}

void cvk_dictionary_set_max_correction_bits(cvk_dictionary_t *h, int max_correction_bits) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return;
    }
    guarded([&]() -> void * {
        h->d.maxCorrectionBits = max_correction_bits;
        return nullptr;
    });
}

int cvk_dictionary_identify(const cvk_dictionary_t *h, const cvk_mat_t *only_bits,
                            double max_correction_rate, int *idx_out, int *rotation_out) {
    if (idx_out) *idx_out = -1;
    if (rotation_out) *rotation_out = 0;
    if (h == nullptr || only_bits == nullptr) {
        record_error("null Dictionary or bits handle");
        return 0;
    }
    return guarded([&]() -> int {
        int idx = -1, rotation = 0;
        const bool found = h->d.identify(*require_const(only_bits), idx, rotation,
                                         max_correction_rate);
        if (idx_out) *idx_out = idx;
        if (rotation_out) *rotation_out = rotation;
        return found ? 1 : 0;
    });
}

int cvk_dictionary_identify_pixel_ratio(const cvk_dictionary_t *h,
                                        const cvk_mat_t *only_cell_pixel_ratio,
                                        double max_correction_rate,
                                        float valid_bit_id_threshold,
                                        int *idx_out, int *rotation_out) {
    if (idx_out) *idx_out = -1;
    if (rotation_out) *rotation_out = 0;
    if (h == nullptr || only_cell_pixel_ratio == nullptr) {
        record_error("null Dictionary or pixel-ratio handle");
        return 0;
    }
    return guarded([&]() -> int {
        int idx = -1, rotation = 0;
        const bool found = h->d.identify(*require_const(only_cell_pixel_ratio), idx, rotation,
                                         max_correction_rate, valid_bit_id_threshold);
        if (idx_out) *idx_out = idx;
        if (rotation_out) *rotation_out = rotation;
        return found ? 1 : 0;
    });
}

int cvk_dictionary_get_distance_to_id(const cvk_dictionary_t *h, const cvk_mat_t *bits,
                                      int id, int all_rotations) {
    if (h == nullptr || bits == nullptr) {
        record_error("null Dictionary or bits handle");
        return -1;
    }
    return guarded([&]() -> int {
        return h->d.getDistanceToId(*require_const(bits), id, all_rotations != 0);
    });
}

cvk_mat_t *cvk_dictionary_generate_image_marker(const cvk_dictionary_t *h, int id,
                                                int side_pixels, int border_bits) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat img;
        h->d.generateImageMarker(id, side_pixels, img, border_bits);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(img));
    });
}

cvk_mat_t *cvk_dictionary_get_marker_bits(const cvk_dictionary_t *h, int marker_id,
                                          int rotation_id) {
    if (h == nullptr) {
        record_error("null Dictionary handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat bits = h->d.getMarkerBits(marker_id, rotation_id);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(bits));
    });
}

cvk_mat_t *cvk_dictionary_get_byte_list_from_bits(const cvk_mat_t *bits) {
    if (bits == nullptr) {
        record_error("null bits Mat");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out = cv::aruco::Dictionary::getByteListFromBits(*require_const(bits));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

cvk_mat_t *cvk_dictionary_get_bits_from_byte_list(const cvk_mat_t *byte_list,
                                                  int marker_size, int rotation_id) {
    if (byte_list == nullptr) {
        record_error("null byteList Mat");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out = cv::aruco::Dictionary::getBitsFromByteList(*require_const(byte_list),
                                                                 marker_size, rotation_id);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

cvk_dictionary_t *cvk_get_predefined_dictionary(int dict) {
    return guarded([&]() -> cvk_dictionary_t * {
        auto *handle = new cvk_dictionary;
        handle->d = cv::aruco::getPredefinedDictionary(dict);
        return reinterpret_cast<cvk_dictionary_t *>(handle);
    });
}

cvk_dictionary_t *cvk_extend_dictionary(int n_markers, int marker_size,
                                        const cvk_dictionary_t *base_dictionary,
                                        int random_seed) {
    return guarded([&]() -> cvk_dictionary_t * {
        auto *handle = new cvk_dictionary;
        const cv::aruco::Dictionary &base =
            base_dictionary != nullptr ? base_dictionary->d : cv::aruco::Dictionary();
        handle->d = cv::aruco::extendDictionary(n_markers, marker_size, base, random_seed);
        return reinterpret_cast<cvk_dictionary_t *>(handle);
    });
}

/* =========================================================================
 * Board
 * ========================================================================= */

cvk_board_t *cvk_board_create(const cvk_mat_t *obj_points, const cvk_dictionary_t *dictionary,
                              const cvk_mat_t *ids) {
    if (dictionary == nullptr) {
        record_error("null Dictionary handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_board_t * {
        std::vector<cv::Mat> obj;
        if (!mats_from_container(obj_points, &obj)) return nullptr;
        auto *handle = new cvk_board;
        cv::Mat id_mat;
        if (ids != nullptr) id_mat = *require_const(ids);
        handle->b = cv::aruco::Board(obj, dictionary->d, id_mat);
        return reinterpret_cast<cvk_board_t *>(handle);
    });
}

void cvk_board_release(cvk_board_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_board *>(h);
        return nullptr;
    });
}

cvk_dictionary_t *cvk_board_get_dictionary(const cvk_board_t *h) {
    if (h == nullptr) {
        record_error("null Board handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_dictionary_t * {
        auto *handle = new cvk_dictionary;
        handle->d = h->b.getDictionary();
        return reinterpret_cast<cvk_dictionary_t *>(handle);
    });
}

cvk_mat_t *cvk_board_get_obj_points(const cvk_board_t *h) {
    if (h == nullptr) {
        record_error("null Board handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        const std::vector<std::vector<cv::Point3f> > &obj = h->b.getObjPoints();
        std::vector<cv::Mat> mats;
        mats.reserve(obj.size());
        for (size_t i = 0; i < obj.size(); i++) {
            const std::vector<cv::Point3f> &corners = obj[i];
            cv::Mat m(static_cast<int>(corners.size()), 1, CV_32FC3);
            for (size_t j = 0; j < corners.size(); j++) {
                m.at<cv::Point3f>(static_cast<int>(j), 0) = corners[j];
            }
            mats.push_back(m);
        }
        return container_of_mats(mats);
    });
}

cvk_mat_t *cvk_board_get_ids(const cvk_board_t *h) {
    if (h == nullptr) {
        record_error("null Board handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        const std::vector<int> &ids = h->b.getIds();
        cv::Mat m(static_cast<int>(ids.size()), 1, CV_32SC1);
        for (size_t i = 0; i < ids.size(); i++) {
            m.at<int>(static_cast<int>(i), 0) = ids[i];
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
    });
}

void cvk_board_get_right_bottom_corner(const cvk_board_t *h, double *out3) {
    if (out3) {
        out3[0] = 0.0;
        out3[1] = 0.0;
        out3[2] = 0.0;
    }
    if (h == nullptr) {
        record_error("null Board handle");
        return;
    }
    guarded([&]() -> void * {
        const cv::Point3f &corner = h->b.getRightBottomCorner();
        if (out3) {
            out3[0] = corner.x;
            out3[1] = corner.y;
            out3[2] = corner.z;
        }
        return nullptr;
    });
}

int cvk_board_match_image_points(const cvk_board_t *h, const cvk_mat_t *detected_corners,
                                 const cvk_mat_t *detected_ids, cvk_mat_t **obj_points_out,
                                 cvk_mat_t **img_points_out) {
    if (obj_points_out) *obj_points_out = nullptr;
    if (img_points_out) *img_points_out = nullptr;
    if (h == nullptr) {
        record_error("null Board handle");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> corners;
        if (!mats_from_container(detected_corners, &corners)) return 0;
        cv::Mat ids;
        if (detected_ids != nullptr) ids = *require_const(detected_ids);
        cv::Mat obj, img;
        h->b.matchImagePoints(corners, ids, obj, img);
        if (obj_points_out) *obj_points_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(obj));
        if (img_points_out) *img_points_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(img));
        return 1;
    });
}

cvk_mat_t *cvk_board_generate_image(const cvk_board_t *h, double out_size_width,
                                    double out_size_height, int margin_size, int border_bits) {
    if (h == nullptr) {
        record_error("null Board handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat img;
        h->b.generateImage(cv::Size((int)out_size_width, (int)out_size_height), img,
                           margin_size, border_bits);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(img));
    });
}

/* =========================================================================
 * GridBoard / CharucoBoard
 * ========================================================================= */

cvk_grid_board_t *cvk_grid_board_create(double size_width, double size_height,
                                        float marker_length, float marker_separation,
                                        const cvk_dictionary_t *dictionary,
                                        const cvk_mat_t *ids) {
    if (dictionary == nullptr) {
        record_error("null Dictionary handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_grid_board_t * {
        auto *handle = new cvk_grid_board;
        cv::Mat id_mat;
        if (ids != nullptr) id_mat = *require_const(ids);
        handle->b = cv::aruco::GridBoard(cv::Size((int)size_width, (int)size_height),
                                         marker_length, marker_separation, dictionary->d, id_mat);
        return reinterpret_cast<cvk_grid_board_t *>(handle);
    });
}

void cvk_grid_board_release(cvk_grid_board_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_grid_board *>(h);
        return nullptr;
    });
}

void cvk_grid_board_get_grid_size(const cvk_grid_board_t *h, int *out2) {
    if (out2) {
        out2[0] = 0;
        out2[1] = 0;
    }
    if (h == nullptr) {
        record_error("null GridBoard handle");
        return;
    }
    guarded([&]() -> void * {
        const cv::Size size = h->b.getGridSize();
        if (out2) {
            out2[0] = size.width;
            out2[1] = size.height;
        }
        return nullptr;
    });
}

float cvk_grid_board_get_marker_length(const cvk_grid_board_t *h) {
    if (h == nullptr) {
        record_error("null GridBoard handle");
        return 0.f;
    }
    return guarded([&]() -> float { return h->b.getMarkerLength(); });
}

float cvk_grid_board_get_marker_separation(const cvk_grid_board_t *h) {
    if (h == nullptr) {
        record_error("null GridBoard handle");
        return 0.f;
    }
    return guarded([&]() -> float { return h->b.getMarkerSeparation(); });
}

cvk_charuco_board_t *cvk_charuco_board_create(double size_width, double size_height,
                                              float square_length, float marker_length,
                                              const cvk_dictionary_t *dictionary,
                                              const cvk_mat_t *ids) {
    if (dictionary == nullptr) {
        record_error("null Dictionary handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_charuco_board_t * {
        auto *handle = new cvk_charuco_board;
        cv::Mat id_mat;
        if (ids != nullptr) id_mat = *require_const(ids);
        handle->b = cv::aruco::CharucoBoard(cv::Size((int)size_width, (int)size_height),
                                            square_length, marker_length, dictionary->d, id_mat);
        return reinterpret_cast<cvk_charuco_board_t *>(handle);
    });
}

void cvk_charuco_board_release(cvk_charuco_board_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_charuco_board *>(h);
        return nullptr;
    });
}

void cvk_charuco_board_set_legacy_pattern(cvk_charuco_board_t *h, int legacy_pattern) {
    if (h == nullptr) {
        record_error("null CharucoBoard handle");
        return;
    }
    guarded([&]() -> void * {
        h->b.setLegacyPattern(legacy_pattern != 0);
        return nullptr;
    });
}

int cvk_charuco_board_get_legacy_pattern(const cvk_charuco_board_t *h) {
    if (h == nullptr) {
        record_error("null CharucoBoard handle");
        return 0;
    }
    return guarded([&]() -> int { return h->b.getLegacyPattern() ? 1 : 0; });
}

void cvk_charuco_board_get_chessboard_size(const cvk_charuco_board_t *h, int *out2) {
    if (out2) {
        out2[0] = 0;
        out2[1] = 0;
    }
    if (h == nullptr) {
        record_error("null CharucoBoard handle");
        return;
    }
    guarded([&]() -> void * {
        const cv::Size size = h->b.getChessboardSize();
        if (out2) {
            out2[0] = size.width;
            out2[1] = size.height;
        }
        return nullptr;
    });
}

float cvk_charuco_board_get_square_length(const cvk_charuco_board_t *h) {
    if (h == nullptr) {
        record_error("null CharucoBoard handle");
        return 0.f;
    }
    return guarded([&]() -> float { return h->b.getSquareLength(); });
}

float cvk_charuco_board_get_marker_length(const cvk_charuco_board_t *h) {
    if (h == nullptr) {
        record_error("null CharucoBoard handle");
        return 0.f;
    }
    return guarded([&]() -> float { return h->b.getMarkerLength(); });
}

cvk_mat_t *cvk_charuco_board_get_chessboard_corners(const cvk_charuco_board_t *h) {
    if (h == nullptr) {
        record_error("null CharucoBoard handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        const std::vector<cv::Point3f> corners = h->b.getChessboardCorners();
        cv::Mat m(static_cast<int>(corners.size()), 1, CV_32FC3);
        for (size_t i = 0; i < corners.size(); i++) {
            m.at<cv::Point3f>(static_cast<int>(i), 0) = corners[i];
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
    });
}

int cvk_charuco_board_check_charuco_corners_collinear(const cvk_charuco_board_t *h,
                                                      const cvk_mat_t *charuco_ids) {
    if (h == nullptr || charuco_ids == nullptr) {
        record_error("null CharucoBoard or ids handle");
        return 0;
    }
    return guarded([&]() -> int {
        return h->b.checkCharucoCornersCollinear(*require_const(charuco_ids)) ? 1 : 0;
    });
}

/* =========================================================================
 * ArucoDetector
 * ========================================================================= */

cvk_aruco_detector_t *cvk_aruco_detector_create(const cvk_dictionary_t *dictionary,
                                                const cvk_detector_params_t *detector_params,
                                                const cvk_refine_params_t *refine_params) {
    if (dictionary == nullptr || detector_params == nullptr || refine_params == nullptr) {
        record_error("null ArucoDetector constructor argument");
        return nullptr;
    }
    return guarded([&]() -> cvk_aruco_detector_t * {
        auto *handle = new cvk_aruco_detector;
        handle->ptr = cv::makePtr<cv::aruco::ArucoDetector>(dictionary->d,
                                                            to_cv(*detector_params),
                                                            to_cv(*refine_params));
        return reinterpret_cast<cvk_aruco_detector_t *>(handle);
    });
}

void cvk_aruco_detector_release(cvk_aruco_detector_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_aruco_detector *>(h);
        return nullptr;
    });
}

int cvk_aruco_detector_detect_markers(const cvk_aruco_detector_t *h, const cvk_mat_t *image,
                                      cvk_mat_t **corners_out, cvk_mat_t **ids_out,
                                      cvk_mat_t **rejected_out) {
    if (corners_out) *corners_out = nullptr;
    if (ids_out) *ids_out = nullptr;
    if (rejected_out) *rejected_out = nullptr;
    if (h == nullptr || image == nullptr) {
        record_error("null ArucoDetector or image handle");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> corners;
        cv::Mat ids;
        std::vector<cv::Mat> rejected;
        h->ptr->detectMarkers(*require_const(image), corners, ids, rejected);
        if (corners_out) *corners_out = container_of_mats(corners);
        if (ids_out) *ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(ids));
        if (rejected_out) *rejected_out = container_of_mats(rejected);
        return 1;
    });
}

int cvk_aruco_detector_detect_markers_with_confidence(const cvk_aruco_detector_t *h,
                                                      const cvk_mat_t *image,
                                                      cvk_mat_t **corners_out,
                                                      cvk_mat_t **ids_out,
                                                      cvk_mat_t **confidence_out,
                                                      cvk_mat_t **rejected_out) {
    if (corners_out) *corners_out = nullptr;
    if (ids_out) *ids_out = nullptr;
    if (confidence_out) *confidence_out = nullptr;
    if (rejected_out) *rejected_out = nullptr;
    if (h == nullptr || image == nullptr) {
        record_error("null ArucoDetector or image handle");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> corners;
        cv::Mat ids;
        cv::Mat confidence;
        std::vector<cv::Mat> rejected;
        h->ptr->detectMarkersWithConfidence(*require_const(image), corners, ids, confidence,
                                            rejected);
        if (corners_out) *corners_out = container_of_mats(corners);
        if (ids_out) *ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(ids));
        if (confidence_out) *confidence_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(confidence));
        if (rejected_out) *rejected_out = container_of_mats(rejected);
        return 1;
    });
}

int cvk_aruco_detector_refine_detected_markers(const cvk_aruco_detector_t *h,
                                               const cvk_mat_t *image,
                                               const cvk_board_t *board,
                                               const cvk_mat_t *detected_corners,
                                               const cvk_mat_t *detected_ids,
                                               const cvk_mat_t *rejected_corners,
                                               const cvk_mat_t *camera_matrix,
                                               const cvk_mat_t *dist_coeffs,
                                               cvk_mat_t **corners_out, cvk_mat_t **ids_out,
                                               cvk_mat_t **rejected_out,
                                               cvk_mat_t **recovered_idxs_out) {
    if (corners_out) *corners_out = nullptr;
    if (ids_out) *ids_out = nullptr;
    if (rejected_out) *rejected_out = nullptr;
    if (recovered_idxs_out) *recovered_idxs_out = nullptr;
    if (h == nullptr || image == nullptr || board == nullptr) {
        record_error("null ArucoDetector/image/board handle");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> detected;
        if (!mats_from_container(detected_corners, &detected)) return 0;
        cv::Mat detected_ids_mat;
        if (detected_ids != nullptr) detected_ids_mat = *require_const(detected_ids);
        std::vector<cv::Mat> rejected;
        if (!mats_from_container(rejected_corners, &rejected)) return 0;
        cv::Mat camera, dist;
        if (camera_matrix != nullptr) camera = *require_const(camera_matrix);
        if (dist_coeffs != nullptr) dist = *require_const(dist_coeffs);
        cv::Mat recovered;
        h->ptr->refineDetectedMarkers(*require_const(image), board->b, detected,
                                      detected_ids_mat, rejected, camera, dist, recovered);
        if (corners_out) *corners_out = container_of_mats(detected);
        if (ids_out) *ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(detected_ids_mat));
        if (rejected_out) *rejected_out = container_of_mats(rejected);
        if (recovered_idxs_out) *recovered_idxs_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(recovered));
        return 1;
    });
}

int cvk_aruco_detector_detect_markers_multi_dict(const cvk_aruco_detector_t *h,
                                                 const cvk_mat_t *image,
                                                 cvk_mat_t **corners_out, cvk_mat_t **ids_out,
                                                 cvk_mat_t **rejected_out,
                                                 cvk_mat_t **dict_indices_out) {
    if (corners_out) *corners_out = nullptr;
    if (ids_out) *ids_out = nullptr;
    if (rejected_out) *rejected_out = nullptr;
    if (dict_indices_out) *dict_indices_out = nullptr;
    if (h == nullptr || image == nullptr) {
        record_error("null ArucoDetector or image handle");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> corners;
        cv::Mat ids;
        std::vector<cv::Mat> rejected;
        cv::Mat dict_indices;
        h->ptr->detectMarkersMultiDict(*require_const(image), corners, ids, rejected,
                                       dict_indices);
        if (corners_out) *corners_out = container_of_mats(corners);
        if (ids_out) *ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(ids));
        if (rejected_out) *rejected_out = container_of_mats(rejected);
        if (dict_indices_out) *dict_indices_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dict_indices));
        return 1;
    });
}

cvk_dictionary_t *cvk_aruco_detector_get_dictionary(const cvk_aruco_detector_t *h) {
    if (h == nullptr) {
        record_error("null ArucoDetector handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_dictionary_t * {
        auto *handle = new cvk_dictionary;
        handle->d = h->ptr->getDictionary();
        return reinterpret_cast<cvk_dictionary_t *>(handle);
    });
}

void cvk_aruco_detector_set_dictionary(cvk_aruco_detector_t *h,
                                       const cvk_dictionary_t *dictionary) {
    if (h == nullptr) {
        record_error("null ArucoDetector handle");
        return;
    }
    guarded([&]() -> void * {
        h->ptr->setDictionary(dictionary->d);
        return nullptr;
    });
}

void cvk_aruco_detector_get_detector_params(const cvk_aruco_detector_t *h,
                                            cvk_detector_params_t *out) {
    if (h == nullptr || out == nullptr) {
        record_error("null ArucoDetector or params out");
        return;
    }
    guarded([&]() -> void * {
        from_cv(h->ptr->getDetectorParameters(), out);
        return nullptr;
    });
}

void cvk_aruco_detector_set_detector_params(cvk_aruco_detector_t *h,
                                            const cvk_detector_params_t *params) {
    if (h == nullptr || params == nullptr) {
        record_error("null ArucoDetector or params");
        return;
    }
    guarded([&]() -> void * {
        cv::aruco::DetectorParameters dp = to_cv(*params);
        h->ptr->setDetectorParameters(dp);
        return nullptr;
    });
}

void cvk_aruco_detector_get_refine_params(const cvk_aruco_detector_t *h,
                                          cvk_refine_params_t *out) {
    if (h == nullptr || out == nullptr) {
        record_error("null ArucoDetector or params out");
        return;
    }
    guarded([&]() -> void * {
        from_cv(h->ptr->getRefineParameters(), out);
        return nullptr;
    });
}

void cvk_aruco_detector_set_refine_params(cvk_aruco_detector_t *h,
                                          const cvk_refine_params_t *params) {
    if (h == nullptr || params == nullptr) {
        record_error("null ArucoDetector or params");
        return;
    }
    guarded([&]() -> void * {
        cv::aruco::RefineParameters rp = to_cv(*params);
        h->ptr->setRefineParameters(rp);
        return nullptr;
    });
}

CVK_ALG_FUNCS(aruco_detector)

/* =========================================================================
 * CharucoDetector
 * ========================================================================= */

cvk_charuco_detector_t *cvk_charuco_detector_create(const cvk_charuco_board_t *board,
                                                    const cvk_charuco_params_t *charuco_params,
                                                    const cvk_detector_params_t *detector_params,
                                                    const cvk_refine_params_t *refine_params) {
    if (board == nullptr || charuco_params == nullptr || detector_params == nullptr ||
        refine_params == nullptr) {
        record_error("null CharucoDetector constructor argument");
        return nullptr;
    }
    return guarded([&]() -> cvk_charuco_detector_t * {
        auto *handle = new cvk_charuco_detector;
        handle->ptr = cv::makePtr<cv::aruco::CharucoDetector>(board->b,
                                                              to_cv(*charuco_params),
                                                              to_cv(*detector_params),
                                                              to_cv(*refine_params));
        return reinterpret_cast<cvk_charuco_detector_t *>(handle);
    });
}

void cvk_charuco_detector_release(cvk_charuco_detector_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_charuco_detector *>(h);
        return nullptr;
    });
}

cvk_charuco_board_t *cvk_charuco_detector_get_board(const cvk_charuco_detector_t *h) {
    if (h == nullptr) {
        record_error("null CharucoDetector handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_charuco_board_t * {
        auto *handle = new cvk_charuco_board;
        handle->b = h->ptr->getBoard();
        return reinterpret_cast<cvk_charuco_board_t *>(handle);
    });
}

void cvk_charuco_detector_set_board(cvk_charuco_detector_t *h,
                                    const cvk_charuco_board_t *board) {
    if (h == nullptr || board == nullptr) {
        record_error("null CharucoDetector or board handle");
        return;
    }
    guarded([&]() -> void * {
        h->ptr->setBoard(board->b);
        return nullptr;
    });
}

void cvk_charuco_detector_get_charuco_params(const cvk_charuco_detector_t *h,
                                             cvk_charuco_params_t *out) {
    if (h == nullptr || out == nullptr) {
        record_error("null CharucoDetector or params out");
        return;
    }
    guarded([&]() -> void * {
        from_cv(h->ptr->getCharucoParameters(), out);
        return nullptr;
    });
}

void cvk_charuco_detector_set_charuco_params(cvk_charuco_detector_t *h,
                                             const cvk_charuco_params_t *params) {
    if (h == nullptr || params == nullptr) {
        record_error("null CharucoDetector or params");
        return;
    }
    guarded([&]() -> void * {
        cv::aruco::CharucoParameters cp = to_cv(*params);
        h->ptr->setCharucoParameters(cp);
        return nullptr;
    });
}

void cvk_charuco_detector_get_detector_params(const cvk_charuco_detector_t *h,
                                              cvk_detector_params_t *out) {
    if (h == nullptr || out == nullptr) {
        record_error("null CharucoDetector or params out");
        return;
    }
    guarded([&]() -> void * {
        from_cv(h->ptr->getDetectorParameters(), out);
        return nullptr;
    });
}

void cvk_charuco_detector_set_detector_params(cvk_charuco_detector_t *h,
                                              const cvk_detector_params_t *params) {
    if (h == nullptr || params == nullptr) {
        record_error("null CharucoDetector or params");
        return;
    }
    guarded([&]() -> void * {
        cv::aruco::DetectorParameters dp = to_cv(*params);
        h->ptr->setDetectorParameters(dp);
        return nullptr;
    });
}

void cvk_charuco_detector_get_refine_params(const cvk_charuco_detector_t *h,
                                            cvk_refine_params_t *out) {
    if (h == nullptr || out == nullptr) {
        record_error("null CharucoDetector or params out");
        return;
    }
    guarded([&]() -> void * {
        from_cv(h->ptr->getRefineParameters(), out);
        return nullptr;
    });
}

void cvk_charuco_detector_set_refine_params(cvk_charuco_detector_t *h,
                                            const cvk_refine_params_t *params) {
    if (h == nullptr || params == nullptr) {
        record_error("null CharucoDetector or params");
        return;
    }
    guarded([&]() -> void * {
        cv::aruco::RefineParameters rp = to_cv(*params);
        h->ptr->setRefineParameters(rp);
        return nullptr;
    });
}

int cvk_charuco_detector_detect_board(const cvk_charuco_detector_t *h, const cvk_mat_t *image,
                                      const cvk_mat_t *marker_corners,
                                      const cvk_mat_t *marker_ids,
                                      cvk_mat_t **charuco_corners_out,
                                      cvk_mat_t **charuco_ids_out,
                                      cvk_mat_t **marker_corners_out,
                                      cvk_mat_t **marker_ids_out) {
    if (charuco_corners_out) *charuco_corners_out = nullptr;
    if (charuco_ids_out) *charuco_ids_out = nullptr;
    if (marker_corners_out) *marker_corners_out = nullptr;
    if (marker_ids_out) *marker_ids_out = nullptr;
    if (h == nullptr || image == nullptr) {
        record_error("null CharucoDetector or image handle");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> corners;
        if (!mats_from_container(marker_corners, &corners)) return 0;
        cv::Mat ids;
        if (marker_ids != nullptr) ids = *require_const(marker_ids);
        cv::Mat charuco_corners, charuco_ids;
        h->ptr->detectBoard(*require_const(image), charuco_corners, charuco_ids, corners, ids);
        if (charuco_corners_out) {
            *charuco_corners_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(charuco_corners));
        }
        if (charuco_ids_out) {
            *charuco_ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(charuco_ids));
        }
        if (marker_corners_out) *marker_corners_out = container_of_mats(corners);
        if (marker_ids_out) *marker_ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(ids));
        return 1;
    });
}

int cvk_charuco_detector_detect_diamonds(const cvk_charuco_detector_t *h, const cvk_mat_t *image,
                                         const cvk_mat_t *marker_corners,
                                         const cvk_mat_t *marker_ids,
                                         cvk_mat_t **diamond_corners_out,
                                         cvk_mat_t **diamond_ids_out,
                                         cvk_mat_t **marker_corners_out,
                                         cvk_mat_t **marker_ids_out) {
    if (diamond_corners_out) *diamond_corners_out = nullptr;
    if (diamond_ids_out) *diamond_ids_out = nullptr;
    if (marker_corners_out) *marker_corners_out = nullptr;
    if (marker_ids_out) *marker_ids_out = nullptr;
    if (h == nullptr || image == nullptr) {
        record_error("null CharucoDetector or image handle");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> corners;
        if (!mats_from_container(marker_corners, &corners)) return 0;
        cv::Mat ids;
        if (marker_ids != nullptr) ids = *require_const(marker_ids);
        std::vector<cv::Mat> diamond_corners;
        cv::Mat diamond_ids;
        h->ptr->detectDiamonds(*require_const(image), diamond_corners, diamond_ids, corners, ids);
        if (diamond_corners_out) *diamond_corners_out = container_of_mats(diamond_corners);
        if (diamond_ids_out) *diamond_ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(diamond_ids));
        if (marker_corners_out) *marker_corners_out = container_of_mats(corners);
        if (marker_ids_out) *marker_ids_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(ids));
        return 1;
    });
}

CVK_ALG_FUNCS(charuco_detector)

/* =========================================================================
 * Objdetect statics
 * ========================================================================= */

int cvk_find_chessboard_corners(const cvk_mat_t *image, double pattern_width,
                                double pattern_height, int flags, cvk_mat_t **corners_out) {
    if (corners_out) *corners_out = nullptr;
    if (image == nullptr) {
        record_error("null image handle");
        return 0;
    }
    return guarded([&]() -> int {
        cv::Mat corners;
        const bool found = cv::findChessboardCorners(
            *require_const(image), cv::Size((int)pattern_width, (int)pattern_height),
            corners, flags);
        if (corners_out) *corners_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(corners));
        return found ? 1 : 0;
    });
}

int cvk_check_chessboard(const cvk_mat_t *image, double size_width, double size_height) {
    if (image == nullptr) {
        record_error("null image handle");
        return 0;
    }
    return guarded([&]() -> int {
        return cv::checkChessboard(*require_const(image),
                                   cv::Size((int)size_width, (int)size_height)) ? 1 : 0;
    });
}

int cvk_find_chessboard_corners_sb_with_meta(const cvk_mat_t *image, double pattern_width,
                                             double pattern_height, int flags,
                                             cvk_mat_t **corners_out, cvk_mat_t **meta_out) {
    if (corners_out) *corners_out = nullptr;
    if (meta_out) *meta_out = nullptr;
    if (image == nullptr) {
        record_error("null image handle");
        return 0;
    }
    return guarded([&]() -> int {
        cv::Mat corners, meta;
        const bool found = cv::findChessboardCornersSB(
            *require_const(image), cv::Size((int)pattern_width, (int)pattern_height),
            corners, flags, meta);
        if (corners_out) *corners_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(corners));
        if (meta_out) *meta_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(meta));
        return found ? 1 : 0;
    });
}

int cvk_find_chessboard_corners_sb(const cvk_mat_t *image, double pattern_width,
                                   double pattern_height, int flags, cvk_mat_t **corners_out) {
    if (corners_out) *corners_out = nullptr;
    if (image == nullptr) {
        record_error("null image handle");
        return 0;
    }
    return guarded([&]() -> int {
        cv::Mat corners;
        const bool found = cv::findChessboardCornersSB(
            *require_const(image), cv::Size((int)pattern_width, (int)pattern_height),
            corners, flags);
        if (corners_out) *corners_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(corners));
        return found ? 1 : 0;
    });
}

int cvk_estimate_chessboard_sharpness(const cvk_mat_t *image, double pattern_width,
                                      double pattern_height, const cvk_mat_t *corners,
                                      float rise_distance, int vertical,
                                      cvk_scalar_t *scalar_out, cvk_mat_t **sharpness_out) {
    if (scalar_out) {
        scalar_out->v0 = 0.0;
        scalar_out->v1 = 0.0;
        scalar_out->v2 = 0.0;
        scalar_out->v3 = 0.0;
    }
    if (sharpness_out) *sharpness_out = nullptr;
    if (image == nullptr || corners == nullptr) {
        record_error("null image/corners handle");
        return 0;
    }
    return guarded([&]() -> int {
        cv::Mat sharpness;
        const cv::Scalar s = cv::estimateChessboardSharpness(
            *require_const(image), cv::Size((int)pattern_width, (int)pattern_height),
            *require_const(corners), rise_distance, vertical != 0, sharpness);
        if (scalar_out) {
            scalar_out->v0 = s[0];
            scalar_out->v1 = s[1];
            scalar_out->v2 = s[2];
            scalar_out->v3 = s[3];
        }
        if (sharpness_out) *sharpness_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(sharpness));
        return 1;
    });
}

int cvk_find4_quad_corner_subpix(const cvk_mat_t *image, cvk_mat_t *corners,
                                 double region_width, double region_height) {
    if (image == nullptr || corners == nullptr) {
        record_error("null image/corners handle");
        return 0;
    }
    return guarded([&]() -> int {
        return cv::find4QuadCornerSubpix(*require_const(image), *require(corners),
                                         cv::Size((int)region_width, (int)region_height)) ? 1 : 0;
    });
}

void cvk_draw_chessboard_corners(cvk_mat_t *image, double pattern_width,
                                 double pattern_height, const cvk_mat_t *corners,
                                 int pattern_was_found) {
    if (image == nullptr) {
        record_error("null image handle");
        return;
    }
    guarded([&]() -> void * {
        cv::drawChessboardCorners(*require(image),
                                  cv::Size((int)pattern_width, (int)pattern_height),
                                  *require_const(corners), pattern_was_found != 0);
        return nullptr;
    });
}

int cvk_find_circles_grid(const cvk_mat_t *image, double pattern_width, double pattern_height,
                          int flags, cvk_mat_t **centers_out) {
    if (centers_out) *centers_out = nullptr;
    if (image == nullptr) {
        record_error("null image handle");
        return 0;
    }
    return guarded([&]() -> int {
        cv::Mat centers;
        const bool found = cv::findCirclesGrid(
            *require_const(image), cv::Size((int)pattern_width, (int)pattern_height),
            centers, flags);
        if (centers_out) *centers_out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(centers));
        return found ? 1 : 0;
    });
}

void cvk_draw_detected_markers(cvk_mat_t *image, const cvk_mat_t *corners,
                               const cvk_mat_t *ids, cvk_scalar_t border_color) {
    if (image == nullptr) {
        record_error("null image handle");
        return;
    }
    guarded([&]() -> void * {
        std::vector<cv::Mat> corner_mats;
        if (!mats_from_container(corners, &corner_mats)) return nullptr;
        cv::Mat id_mat;
        if (ids != nullptr) id_mat = *require_const(ids);
        cv::aruco::drawDetectedMarkers(*require(image), corner_mats, id_mat,
                                       cv::Scalar(border_color.v0, border_color.v1,
                                                  border_color.v2, border_color.v3));
        return nullptr;
    });
}

cvk_mat_t *cvk_generate_image_marker(const cvk_dictionary_t *dictionary, int id,
                                     int side_pixels, int border_bits) {
    if (dictionary == nullptr) {
        record_error("null Dictionary handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat img;
        cv::aruco::generateImageMarker(dictionary->d, id, side_pixels, img, border_bits);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(img));
    });
}

void cvk_draw_detected_corners_charuco(cvk_mat_t *image, const cvk_mat_t *charuco_corners,
                                       const cvk_mat_t *charuco_ids, cvk_scalar_t corner_color) {
    if (image == nullptr) {
        record_error("null image handle");
        return;
    }
    guarded([&]() -> void * {
        cv::Mat id_mat;
        if (charuco_ids != nullptr) id_mat = *require_const(charuco_ids);
        cv::aruco::drawDetectedCornersCharuco(*require(image), *require_const(charuco_corners),
                                              id_mat,
                                              cv::Scalar(corner_color.v0, corner_color.v1,
                                                         corner_color.v2, corner_color.v3));
        return nullptr;
    });
}

void cvk_draw_detected_diamonds(cvk_mat_t *image, const cvk_mat_t *diamond_corners,
                                const cvk_mat_t *diamond_ids, cvk_scalar_t border_color) {
    if (image == nullptr) {
        record_error("null image handle");
        return;
    }
    guarded([&]() -> void * {
        std::vector<cv::Mat> corner_mats;
        if (!mats_from_container(diamond_corners, &corner_mats)) return nullptr;
        cv::Mat id_mat;
        if (diamond_ids != nullptr) id_mat = *require_const(diamond_ids);
        cv::aruco::drawDetectedDiamonds(*require(image), corner_mats, id_mat,
                                        cv::Scalar(border_color.v0, border_color.v1,
                                                   border_color.v2, border_color.v3));
        return nullptr;
    });
}

} /* extern "C" */

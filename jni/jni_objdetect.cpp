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
 * JNI bridge for the objdetect slice: Java_cn_enaium_opencv_JniObjdetect_*
 * wrappers around the cvk_ ABI in native/shim_objdetect.cpp.
 *
 * DetectorParameters/RefineParameters travel as DoubleArray in the fixed
 * field order documented in opencv_kmp_objdetect.h (which matches the Kotlin
 * data class declaration order in Objdetect.kt). Charuco parameters travel
 * as (cameraMatrix, distCoeffs) jlong handles plus expanded scalar args.
 *
 * Board-level cvk_ functions take cvk_board_t*; grid/charuco board handles
 * are reinterpreted to it (their Board base subobject sits at offset 0).
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_objdetect.h"

#include <cstdint>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}

static inline cvk_dictionary_t *as_dict(jlong handle) {
    return reinterpret_cast<cvk_dictionary_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_dict_handle(const cvk_dictionary_t *d) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(d));
}

static inline cvk_board_t *as_board(jlong handle) {
    return reinterpret_cast<cvk_board_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_board_handle(const cvk_board_t *b) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(b));
}

static inline cvk_grid_board_t *as_grid_board(jlong handle) {
    return reinterpret_cast<cvk_grid_board_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_grid_board_handle(const cvk_grid_board_t *b) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(b));
}

static inline cvk_charuco_board_t *as_charuco_board(jlong handle) {
    return reinterpret_cast<cvk_charuco_board_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_charuco_board_handle(const cvk_charuco_board_t *b) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(b));
}

static inline cvk_aruco_detector_t *as_aruco(jlong handle) {
    return reinterpret_cast<cvk_aruco_detector_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_aruco_handle(const cvk_aruco_detector_t *d) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(d));
}

static inline cvk_charuco_detector_t *as_charuco(jlong handle) {
    return reinterpret_cast<cvk_charuco_detector_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_charuco_handle(const cvk_charuco_detector_t *d) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(d));
}

/* ---- params wire format (fixed field order, see opencv_kmp_objdetect.h) -- */

static const int DETECTOR_PARAMS_COUNT = 35;

static void fill_detector_params(JNIEnv *env, jdoubleArray arr, cvk_detector_params_t *p) {
    memset(p, 0, sizeof(*p));
    if (arr == nullptr) return;
    const jsize len = env->GetArrayLength(arr);
    if (len <= 0) return;
    std::vector<jdouble> buf(static_cast<size_t>(len));
    env->GetDoubleArrayRegion(arr, 0, len, buf.data());
    auto get = [&](int i, double fallback) -> double {
        return i < len ? buf[i] : fallback;
    };
    p->adaptive_thresh_win_size_min = (int)get(0, 3);
    p->adaptive_thresh_win_size_max = (int)get(1, 23);
    p->adaptive_thresh_win_size_step = (int)get(2, 10);
    p->adaptive_thresh_constant = get(3, 7);
    p->min_marker_perimeter_rate = get(4, 0.03);
    p->max_marker_perimeter_rate = get(5, 4.0);
    p->polygonal_approx_accuracy_rate = get(6, 0.03);
    p->min_corner_distance_rate = get(7, 0.05);
    p->min_distance_to_border = (int)get(8, 3);
    p->min_marker_distance_rate = get(9, 0.125);
    p->min_group_distance = (float)get(10, 0.21);
    p->corner_refinement_method = (int)get(11, 0);
    p->corner_refinement_win_size = (int)get(12, 5);
    p->relative_corner_refinment_win_size = (float)get(13, 0.3);
    p->corner_refinement_max_iterations = (int)get(14, 30);
    p->corner_refinement_min_accuracy = get(15, 0.1);
    p->marker_border_bits = (int)get(16, 1);
    p->perspective_remove_pixel_per_cell = (int)get(17, 4);
    p->perspective_remove_ignored_margin_per_cell = get(18, 0.13);
    p->max_erroneous_bits_in_border_rate = get(19, 0.35);
    p->min_otsu_std_dev = get(20, 5.0);
    p->error_correction_rate = get(21, 0.6);
    p->april_tag_quad_decimate = (float)get(22, 0.0);
    p->april_tag_quad_sigma = (float)get(23, 0.0);
    p->april_tag_min_cluster_pixels = (int)get(24, 5);
    p->april_tag_max_nmaxima = (int)get(25, 10);
    p->april_tag_critical_rad = (float)get(26, 10.0 * 3.14159265358979323846 / 180.0);
    p->april_tag_max_line_fit_mse = (float)get(27, 10.0);
    p->april_tag_min_white_black_diff = (int)get(28, 5);
    p->april_tag_deglitch = (int)get(29, 0);
    p->detect_inverted_marker = get(30, 0.0) != 0.0 ? 1 : 0;
    p->use_aruco = get(31, 0.0) != 0.0 ? 1 : 0;
    p->min_side_length_canonical_img = (int)get(32, 32);
    p->min_marker_length_ratio_original_img = (float)get(33, 0.0);
    p->valid_bit_id_threshold = (float)get(34, 0.49);
}

static jdoubleArray detector_params_array(JNIEnv *env, const cvk_detector_params_t &p) {
    const jdouble values[DETECTOR_PARAMS_COUNT] = {
        (jdouble)p.adaptive_thresh_win_size_min, (jdouble)p.adaptive_thresh_win_size_max,
        (jdouble)p.adaptive_thresh_win_size_step, (jdouble)p.adaptive_thresh_constant,
        (jdouble)p.min_marker_perimeter_rate, (jdouble)p.max_marker_perimeter_rate,
        (jdouble)p.polygonal_approx_accuracy_rate, (jdouble)p.min_corner_distance_rate,
        (jdouble)p.min_distance_to_border, (jdouble)p.min_marker_distance_rate,
        (jdouble)p.min_group_distance, (jdouble)p.corner_refinement_method,
        (jdouble)p.corner_refinement_win_size, (jdouble)p.relative_corner_refinment_win_size,
        (jdouble)p.corner_refinement_max_iterations, (jdouble)p.corner_refinement_min_accuracy,
        (jdouble)p.marker_border_bits, (jdouble)p.perspective_remove_pixel_per_cell,
        (jdouble)p.perspective_remove_ignored_margin_per_cell,
        (jdouble)p.max_erroneous_bits_in_border_rate, (jdouble)p.min_otsu_std_dev,
        (jdouble)p.error_correction_rate, (jdouble)p.april_tag_quad_decimate, (jdouble)p.april_tag_quad_sigma,
        (jdouble)p.april_tag_min_cluster_pixels, (jdouble)p.april_tag_max_nmaxima,
        (jdouble)p.april_tag_critical_rad, (jdouble)p.april_tag_max_line_fit_mse,
        (jdouble)p.april_tag_min_white_black_diff, (jdouble)p.april_tag_deglitch,
        (jdouble)p.detect_inverted_marker, (jdouble)p.use_aruco,
        (jdouble)p.min_side_length_canonical_img, (jdouble)p.min_marker_length_ratio_original_img,
        (jdouble)p.valid_bit_id_threshold,
    };
    jdoubleArray out = env->NewDoubleArray(DETECTOR_PARAMS_COUNT);
    if (out == nullptr) return nullptr;
    env->SetDoubleArrayRegion(out, 0, DETECTOR_PARAMS_COUNT, values);
    return out;
}

static const int REFINE_PARAMS_COUNT = 3;

static void fill_refine_params(JNIEnv *env, jdoubleArray arr, cvk_refine_params_t *p) {
    p->min_rep_distance = 10.f;
    p->error_correction_rate = 3.f;
    p->check_all_orders = 1;
    if (arr == nullptr) return;
    const jsize len = env->GetArrayLength(arr);
    if (len <= 0) return;
    std::vector<jdouble> buf(static_cast<size_t>(len));
    env->GetDoubleArrayRegion(arr, 0, len, buf.data());
    auto get = [&](int i, double fallback) -> double {
        return i < len ? buf[i] : fallback;
    };
    p->min_rep_distance = (float)get(0, 10.f);
    p->error_correction_rate = (float)get(1, 3.f);
    p->check_all_orders = get(2, 1.0) != 0.0 ? 1 : 0;
}

static jdoubleArray refine_params_array(JNIEnv *env, const cvk_refine_params_t &p) {
    const jdouble values[REFINE_PARAMS_COUNT] = {
        (jdouble)p.min_rep_distance, (jdouble)p.error_correction_rate, (jdouble)p.check_all_orders,
    };
    jdoubleArray out = env->NewDoubleArray(REFINE_PARAMS_COUNT);
    if (out == nullptr) return nullptr;
    env->SetDoubleArrayRegion(out, 0, REFINE_PARAMS_COUNT, values);
    return out;
}

extern "C" {

/* =========================================================================
 * Dictionary
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryCreate(JNIEnv *, jobject, jlong bytesList,
                                                    jint markerSize, jint maxCorrectionBits) {
    return as_dict_handle(cvk_dictionary_create(as_mat(bytesList), markerSize, maxCorrectionBits));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryRelease(JNIEnv *, jobject, jlong handle) {
    cvk_dictionary_release(as_dict(handle));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGetBytesList(JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_dictionary_get_bytes_list(as_dict(handle)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionarySetBytesList(JNIEnv *, jobject, jlong handle,
                                                          jlong bytesList) {
    cvk_dictionary_set_bytes_list(as_dict(handle), as_mat(bytesList));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGetMarkerSize(JNIEnv *, jobject, jlong handle) {
    return cvk_dictionary_get_marker_size(as_dict(handle));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionarySetMarkerSize(JNIEnv *, jobject, jlong handle,
                                                           jint markerSize) {
    cvk_dictionary_set_marker_size(as_dict(handle), markerSize);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGetMaxCorrectionBits(JNIEnv *, jobject, jlong handle) {
    return cvk_dictionary_get_max_correction_bits(as_dict(handle));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionarySetMaxCorrectionBits(JNIEnv *, jobject, jlong handle,
                                                                  jint maxCorrectionBits) {
    cvk_dictionary_set_max_correction_bits(as_dict(handle), maxCorrectionBits);
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryIdentify(JNIEnv *env, jobject, jlong handle,
                                                      jlong onlyBits, jdouble maxCorrectionRate) {
    int idx = -1, rotation = 0;
    const int found = cvk_dictionary_identify(as_dict(handle), as_mat(onlyBits),
                                              maxCorrectionRate, &idx, &rotation);
    jintArray out = env->NewIntArray(3);
    if (out == nullptr) return nullptr;
    const jint values[3] = {found, idx, rotation};
    env->SetIntArrayRegion(out, 0, 3, values);
    return out;
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryIdentifyPixelRatio(JNIEnv *env, jobject,
                                                                jlong handle,
                                                                jlong onlyCellPixelRatio,
                                                                jdouble maxCorrectionRate,
                                                                jfloat validBitIdThreshold) {
    int idx = -1, rotation = 0;
    const int found = cvk_dictionary_identify_pixel_ratio(
        as_dict(handle), as_mat(onlyCellPixelRatio), maxCorrectionRate, validBitIdThreshold,
        &idx, &rotation);
    jintArray out = env->NewIntArray(3);
    if (out == nullptr) return nullptr;
    const jint values[3] = {found, idx, rotation};
    env->SetIntArrayRegion(out, 0, 3, values);
    return out;
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGetDistanceToId(JNIEnv *, jobject, jlong handle,
                                                             jlong bits, jint id,
                                                             jboolean allRotations) {
    return cvk_dictionary_get_distance_to_id(as_dict(handle), as_mat(bits), id,
                                             allRotations ? 1 : 0);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGenerateImageMarker(JNIEnv *, jobject, jlong handle,
                                                                 jint id, jint sidePixels,
                                                                 jint borderBits) {
    return as_handle(cvk_dictionary_generate_image_marker(as_dict(handle), id, sidePixels,
                                                          borderBits));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGetMarkerBits(JNIEnv *, jobject, jlong handle,
                                                           jint markerId, jint rotationId) {
    return as_handle(cvk_dictionary_get_marker_bits(as_dict(handle), markerId, rotationId));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGetByteListFromBits(JNIEnv *, jobject, jlong bits) {
    return as_handle(cvk_dictionary_get_byte_list_from_bits(as_mat(bits)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_dictionaryGetBitsFromByteList(JNIEnv *, jobject, jlong byteList,
                                                                 jint markerSize,
                                                                 jint rotationId) {
    return as_handle(cvk_dictionary_get_bits_from_byte_list(as_mat(byteList), markerSize,
                                                            rotationId));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_getPredefinedDictionary(JNIEnv *, jobject, jint dict) {
    return as_dict_handle(cvk_get_predefined_dictionary(dict));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_extendDictionary(JNIEnv *, jobject, jint nMarkers,
                                                    jint markerSize, jlong baseDictionary,
                                                    jint randomSeed) {
    return as_dict_handle(cvk_extend_dictionary(nMarkers, markerSize,
                                                as_dict(baseDictionary), randomSeed));
}

/* =========================================================================
 * Board
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardCreate(JNIEnv *, jobject, jlong objPoints,
                                               jlong dictionary, jlong ids) {
    return as_board_handle(cvk_board_create(as_mat(objPoints), as_dict(dictionary),
                                            as_mat(ids)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardRelease(JNIEnv *, jobject, jlong handle) {
    cvk_board_release(as_board(handle));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardGetDictionary(JNIEnv *, jobject, jlong handle) {
    return as_dict_handle(cvk_board_get_dictionary(as_board(handle)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardGetObjPoints(JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_board_get_obj_points(as_board(handle)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardGetIds(JNIEnv *, jobject, jlong handle) {
    return as_handle(cvk_board_get_ids(as_board(handle)));
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardGetRightBottomCorner(JNIEnv *env, jobject, jlong handle) {
    double out3[3] = {0.0, 0.0, 0.0};
    cvk_board_get_right_bottom_corner(as_board(handle), out3);
    jdoubleArray result = env->NewDoubleArray(3);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 3, out3);
    return result;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardMatchImagePoints(JNIEnv *env, jobject, jlong handle,
                                                         jlong detectedCorners,
                                                         jlong detectedIds) {
    cvk_mat_t *obj = nullptr;
    cvk_mat_t *img = nullptr;
    if (!cvk_board_match_image_points(as_board(handle), as_mat(detectedCorners),
                                      as_mat(detectedIds), &obj, &img)) {
        return nullptr;
    }
    jlongArray out = env->NewLongArray(2);
    if (out == nullptr) return nullptr;
    const jlong values[2] = {as_handle(obj), as_handle(img)};
    env->SetLongArrayRegion(out, 0, 2, values);
    return out;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_boardGenerateImage(JNIEnv *, jobject, jlong handle,
                                                      jdouble outSizeWidth, jdouble outSizeHeight,
                                                      jint marginSize, jint borderBits) {
    return as_handle(cvk_board_generate_image(as_board(handle), outSizeWidth, outSizeHeight,
                                              marginSize, borderBits));
}

/* =========================================================================
 * GridBoard
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_gridBoardCreate(JNIEnv *, jobject, jdouble sizeWidth,
                                                   jdouble sizeHeight, jfloat markerLength,
                                                   jfloat markerSeparation, jlong dictionary,
                                                   jlong ids) {
    return as_grid_board_handle(cvk_grid_board_create(sizeWidth, sizeHeight, markerLength,
                                                      markerSeparation, as_dict(dictionary),
                                                      as_mat(ids)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_gridBoardRelease(JNIEnv *, jobject, jlong handle) {
    cvk_grid_board_release(as_grid_board(handle));
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_gridBoardGetGridSize(JNIEnv *env, jobject, jlong handle) {
    int out2[2] = {0, 0};
    cvk_grid_board_get_grid_size(as_grid_board(handle), out2);
    jintArray result = env->NewIntArray(2);
    if (result == nullptr) return nullptr;
    env->SetIntArrayRegion(result, 0, 2, out2);
    return result;
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniObjdetect_gridBoardGetMarkerLength(JNIEnv *, jobject, jlong handle) {
    return cvk_grid_board_get_marker_length(as_grid_board(handle));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniObjdetect_gridBoardGetMarkerSeparation(JNIEnv *, jobject, jlong handle) {
    return cvk_grid_board_get_marker_separation(as_grid_board(handle));
}

/* =========================================================================
 * CharucoBoard
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardCreate(JNIEnv *, jobject, jdouble sizeWidth,
                                                      jdouble sizeHeight, jfloat squareLength,
                                                      jfloat markerLength, jlong dictionary,
                                                      jlong ids) {
    return as_charuco_board_handle(cvk_charuco_board_create(sizeWidth, sizeHeight, squareLength,
                                                            markerLength, as_dict(dictionary),
                                                            as_mat(ids)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardRelease(JNIEnv *, jobject, jlong handle) {
    cvk_charuco_board_release(as_charuco_board(handle));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardSetLegacyPattern(JNIEnv *, jobject, jlong handle,
                                                                jboolean legacyPattern) {
    cvk_charuco_board_set_legacy_pattern(as_charuco_board(handle), legacyPattern ? 1 : 0);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardGetLegacyPattern(JNIEnv *, jobject, jlong handle) {
    return cvk_charuco_board_get_legacy_pattern(as_charuco_board(handle)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardGetChessboardSize(JNIEnv *env, jobject,
                                                                 jlong handle) {
    int out2[2] = {0, 0};
    cvk_charuco_board_get_chessboard_size(as_charuco_board(handle), out2);
    jintArray result = env->NewIntArray(2);
    if (result == nullptr) return nullptr;
    env->SetIntArrayRegion(result, 0, 2, out2);
    return result;
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardGetSquareLength(JNIEnv *, jobject, jlong handle) {
    return cvk_charuco_board_get_square_length(as_charuco_board(handle));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardGetMarkerLength(JNIEnv *, jobject, jlong handle) {
    return cvk_charuco_board_get_marker_length(as_charuco_board(handle));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardGetChessboardCorners(JNIEnv *, jobject,
                                                                    jlong handle) {
    return as_handle(cvk_charuco_board_get_chessboard_corners(as_charuco_board(handle)));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoBoardCheckCharucoCornersCollinear(
    JNIEnv *, jobject, jlong handle, jlong charucoIds) {
    return cvk_charuco_board_check_charuco_corners_collinear(as_charuco_board(handle),
                                                             as_mat(charucoIds))
               ? JNI_TRUE
               : JNI_FALSE;
}

/* =========================================================================
 * ArucoDetector
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorCreate(JNIEnv *env, jobject, jlong dictionary,
                                                       jdoubleArray detectorParams,
                                                       jdoubleArray refineParams) {
    cvk_detector_params_t dp;
    fill_detector_params(env, detectorParams, &dp);
    cvk_refine_params_t rp;
    fill_refine_params(env, refineParams, &rp);
    return as_aruco_handle(cvk_aruco_detector_create(as_dict(dictionary), &dp, &rp));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorRelease(JNIEnv *, jobject, jlong handle) {
    cvk_aruco_detector_release(as_aruco(handle));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorDetectMarkers(JNIEnv *env, jobject, jlong handle,
                                                              jlong image) {
    cvk_mat_t *corners = nullptr;
    cvk_mat_t *ids = nullptr;
    cvk_mat_t *rejected = nullptr;
    if (!cvk_aruco_detector_detect_markers(as_aruco(handle), as_mat(image), &corners, &ids,
                                           &rejected)) {
        return nullptr;
    }
    jlongArray out = env->NewLongArray(3);
    if (out == nullptr) return nullptr;
    const jlong values[3] = {as_handle(corners), as_handle(ids), as_handle(rejected)};
    env->SetLongArrayRegion(out, 0, 3, values);
    return out;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorDetectMarkersWithConfidence(JNIEnv *env,
                                                                            jobject,
                                                                            jlong handle,
                                                                            jlong image) {
    cvk_mat_t *corners = nullptr;
    cvk_mat_t *ids = nullptr;
    cvk_mat_t *confidence = nullptr;
    cvk_mat_t *rejected = nullptr;
    if (!cvk_aruco_detector_detect_markers_with_confidence(as_aruco(handle), as_mat(image),
                                                           &corners, &ids, &confidence,
                                                           &rejected)) {
        return nullptr;
    }
    jlongArray out = env->NewLongArray(4);
    if (out == nullptr) return nullptr;
    const jlong values[4] = {as_handle(corners), as_handle(ids), as_handle(confidence),
                             as_handle(rejected)};
    env->SetLongArrayRegion(out, 0, 4, values);
    return out;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorRefineDetectedMarkers(
    JNIEnv *env, jobject, jlong handle, jlong image, jlong board, jlong detectedCorners,
    jlong detectedIds, jlong rejectedCorners, jlong cameraMatrix, jlong distCoeffs) {
    cvk_mat_t *corners = nullptr;
    cvk_mat_t *ids = nullptr;
    cvk_mat_t *rejected = nullptr;
    cvk_mat_t *recovered = nullptr;
    if (!cvk_aruco_detector_refine_detected_markers(
            as_aruco(handle), as_mat(image), as_board(board), as_mat(detectedCorners),
            as_mat(detectedIds), as_mat(rejectedCorners), as_mat(cameraMatrix),
            as_mat(distCoeffs), &corners, &ids, &rejected, &recovered)) {
        return nullptr;
    }
    jlongArray out = env->NewLongArray(4);
    if (out == nullptr) return nullptr;
    const jlong values[4] = {as_handle(corners), as_handle(ids), as_handle(rejected),
                             as_handle(recovered)};
    env->SetLongArrayRegion(out, 0, 4, values);
    return out;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorDetectMarkersMultiDict(JNIEnv *env, jobject,
                                                                       jlong handle,
                                                                       jlong image) {
    cvk_mat_t *corners = nullptr;
    cvk_mat_t *ids = nullptr;
    cvk_mat_t *rejected = nullptr;
    cvk_mat_t *dictIndices = nullptr;
    if (!cvk_aruco_detector_detect_markers_multi_dict(as_aruco(handle), as_mat(image),
                                                      &corners, &ids, &rejected,
                                                      &dictIndices)) {
        return nullptr;
    }
    jlongArray out = env->NewLongArray(4);
    if (out == nullptr) return nullptr;
    const jlong values[4] = {as_handle(corners), as_handle(ids), as_handle(rejected),
                             as_handle(dictIndices)};
    env->SetLongArrayRegion(out, 0, 4, values);
    return out;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorGetDictionary(JNIEnv *, jobject, jlong handle) {
    return as_dict_handle(cvk_aruco_detector_get_dictionary(as_aruco(handle)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorSetDictionary(JNIEnv *, jobject, jlong handle,
                                                              jlong dictionary) {
    cvk_aruco_detector_set_dictionary(as_aruco(handle), as_dict(dictionary));
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorGetDetectorParams(JNIEnv *env, jobject,
                                                                  jlong handle) {
    cvk_detector_params_t p;
    memset(&p, 0, sizeof(p));
    cvk_aruco_detector_get_detector_params(as_aruco(handle), &p);
    return detector_params_array(env, p);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorSetDetectorParams(JNIEnv *env, jobject,
                                                                  jlong handle,
                                                                  jdoubleArray detectorParams) {
    cvk_detector_params_t dp;
    fill_detector_params(env, detectorParams, &dp);
    cvk_aruco_detector_set_detector_params(as_aruco(handle), &dp);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorGetRefineParams(JNIEnv *env, jobject,
                                                                jlong handle) {
    cvk_refine_params_t p;
    p.min_rep_distance = 0.f;
    p.error_correction_rate = 0.f;
    p.check_all_orders = 0;
    cvk_aruco_detector_get_refine_params(as_aruco(handle), &p);
    return refine_params_array(env, p);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorSetRefineParams(JNIEnv *env, jobject,
                                                                jlong handle,
                                                                jdoubleArray refineParams) {
    cvk_refine_params_t rp;
    fill_refine_params(env, refineParams, &rp);
    cvk_aruco_detector_set_refine_params(as_aruco(handle), &rp);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorClear(JNIEnv *, jobject, jlong handle) {
    cvk_aruco_detector_clear(as_aruco(handle));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorEmpty(JNIEnv *, jobject, jlong handle) {
    return cvk_aruco_detector_empty(as_aruco(handle)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorSave(JNIEnv *env, jobject, jlong handle,
                                                     jstring filename) {
    const char *name = filename != nullptr ? env->GetStringUTFChars(filename, nullptr) : nullptr;
    cvk_aruco_detector_save(as_aruco(handle), name);
    if (name != nullptr) env->ReleaseStringUTFChars(filename, name);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniObjdetect_arucoDetectorGetDefaultName(JNIEnv *env, jobject,
                                                               jlong handle) {
    const char *name = cvk_aruco_detector_get_default_name(as_aruco(handle));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

/* =========================================================================
 * CharucoDetector
 * ========================================================================= */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorCreate(
    JNIEnv *env, jobject, jlong board, jlong cameraMatrix, jlong distCoeffs, jint minMarkers,
    jboolean tryRefineMarkers, jboolean checkMarkers, jdoubleArray detectorParams,
    jdoubleArray refineParams) {
    cvk_charuco_params_t cp;
    cp.camera_matrix = as_mat(cameraMatrix);
    cp.dist_coeffs = as_mat(distCoeffs);
    cp.min_markers = minMarkers;
    cp.try_refine_markers = tryRefineMarkers ? 1 : 0;
    cp.check_markers = checkMarkers ? 1 : 0;
    cvk_detector_params_t dp;
    fill_detector_params(env, detectorParams, &dp);
    cvk_refine_params_t rp;
    fill_refine_params(env, refineParams, &rp);
    return as_charuco_handle(cvk_charuco_detector_create(as_charuco_board(board), &cp, &dp, &rp));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorRelease(JNIEnv *, jobject, jlong handle) {
    cvk_charuco_detector_release(as_charuco(handle));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorGetBoard(JNIEnv *, jobject, jlong handle) {
    return as_charuco_board_handle(cvk_charuco_detector_get_board(as_charuco(handle)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorSetBoard(JNIEnv *, jobject, jlong handle,
                                                           jlong board) {
    cvk_charuco_detector_set_board(as_charuco(handle), as_charuco_board(board));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorGetCharucoParams(JNIEnv *env, jobject,
                                                                   jlong handle) {
    cvk_charuco_params_t p;
    p.camera_matrix = nullptr;
    p.dist_coeffs = nullptr;
    p.min_markers = 0;
    p.try_refine_markers = 0;
    p.check_markers = 0;
    cvk_charuco_detector_get_charuco_params(as_charuco(handle), &p);
    jlongArray out = env->NewLongArray(5);
    if (out == nullptr) return nullptr;
    const jlong values[5] = {as_handle(p.camera_matrix), as_handle(p.dist_coeffs),
                             p.min_markers, p.try_refine_markers, p.check_markers};
    env->SetLongArrayRegion(out, 0, 5, values);
    return out;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorSetCharucoParams(
    JNIEnv *, jobject, jlong handle, jlong cameraMatrix, jlong distCoeffs, jint minMarkers,
    jboolean tryRefineMarkers, jboolean checkMarkers) {
    cvk_charuco_params_t cp;
    cp.camera_matrix = as_mat(cameraMatrix);
    cp.dist_coeffs = as_mat(distCoeffs);
    cp.min_markers = minMarkers;
    cp.try_refine_markers = tryRefineMarkers ? 1 : 0;
    cp.check_markers = checkMarkers ? 1 : 0;
    cvk_charuco_detector_set_charuco_params(as_charuco(handle), &cp);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorGetDetectorParams(JNIEnv *env, jobject,
                                                                    jlong handle) {
    cvk_detector_params_t p;
    memset(&p, 0, sizeof(p));
    cvk_charuco_detector_get_detector_params(as_charuco(handle), &p);
    return detector_params_array(env, p);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorSetDetectorParams(JNIEnv *env, jobject,
                                                                    jlong handle,
                                                                    jdoubleArray detectorParams) {
    cvk_detector_params_t dp;
    fill_detector_params(env, detectorParams, &dp);
    cvk_charuco_detector_set_detector_params(as_charuco(handle), &dp);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorGetRefineParams(JNIEnv *env, jobject,
                                                                  jlong handle) {
    cvk_refine_params_t p;
    p.min_rep_distance = 0.f;
    p.error_correction_rate = 0.f;
    p.check_all_orders = 0;
    cvk_charuco_detector_get_refine_params(as_charuco(handle), &p);
    return refine_params_array(env, p);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorSetRefineParams(JNIEnv *env, jobject,
                                                                  jlong handle,
                                                                  jdoubleArray refineParams) {
    cvk_refine_params_t rp;
    fill_refine_params(env, refineParams, &rp);
    cvk_charuco_detector_set_refine_params(as_charuco(handle), &rp);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorDetectBoard(JNIEnv *env, jobject, jlong handle,
                                                              jlong image, jlong markerCorners,
                                                              jlong markerIds) {
    cvk_mat_t *charucoCorners = nullptr;
    cvk_mat_t *charucoIds = nullptr;
    cvk_mat_t *markerCornersOut = nullptr;
    cvk_mat_t *markerIdsOut = nullptr;
    if (!cvk_charuco_detector_detect_board(as_charuco(handle), as_mat(image),
                                           as_mat(markerCorners), as_mat(markerIds),
                                           &charucoCorners, &charucoIds, &markerCornersOut,
                                           &markerIdsOut)) {
        return nullptr;
    }
    jlongArray out = env->NewLongArray(4);
    if (out == nullptr) return nullptr;
    const jlong values[4] = {as_handle(charucoCorners), as_handle(charucoIds),
                             as_handle(markerCornersOut), as_handle(markerIdsOut)};
    env->SetLongArrayRegion(out, 0, 4, values);
    return out;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorDetectDiamonds(JNIEnv *env, jobject,
                                                                 jlong handle, jlong image,
                                                                 jlong markerCorners,
                                                                 jlong markerIds) {
    cvk_mat_t *diamondCorners = nullptr;
    cvk_mat_t *diamondIds = nullptr;
    cvk_mat_t *markerCornersOut = nullptr;
    cvk_mat_t *markerIdsOut = nullptr;
    if (!cvk_charuco_detector_detect_diamonds(as_charuco(handle), as_mat(image),
                                              as_mat(markerCorners), as_mat(markerIds),
                                              &diamondCorners, &diamondIds, &markerCornersOut,
                                              &markerIdsOut)) {
        return nullptr;
    }
    jlongArray out = env->NewLongArray(4);
    if (out == nullptr) return nullptr;
    const jlong values[4] = {as_handle(diamondCorners), as_handle(diamondIds),
                             as_handle(markerCornersOut), as_handle(markerIdsOut)};
    env->SetLongArrayRegion(out, 0, 4, values);
    return out;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorClear(JNIEnv *, jobject, jlong handle) {
    cvk_charuco_detector_clear(as_charuco(handle));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorEmpty(JNIEnv *, jobject, jlong handle) {
    return cvk_charuco_detector_empty(as_charuco(handle)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorSave(JNIEnv *env, jobject, jlong handle,
                                                       jstring filename) {
    const char *name = filename != nullptr ? env->GetStringUTFChars(filename, nullptr) : nullptr;
    cvk_charuco_detector_save(as_charuco(handle), name);
    if (name != nullptr) env->ReleaseStringUTFChars(filename, name);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniObjdetect_charucoDetectorGetDefaultName(JNIEnv *env, jobject,
                                                                 jlong handle) {
    const char *name = cvk_charuco_detector_get_default_name(as_charuco(handle));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

/* =========================================================================
 * Objdetect statics
 * ========================================================================= */

/* Returns the fresh corners Mat handle; writes `found` into foundOut[0]. */
JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_findChessboardCorners(JNIEnv *env, jobject, jlong image,
                                                         jdouble patternWidth,
                                                         jdouble patternHeight, jint flags,
                                                         jbooleanArray foundOut) {
    cvk_mat_t *cornersOut = nullptr;
    const int found = cvk_find_chessboard_corners(as_mat(image), patternWidth, patternHeight,
                                                  flags, &cornersOut);
    if (foundOut != nullptr && env->GetArrayLength(foundOut) > 0) {
        const jboolean value = found ? JNI_TRUE : JNI_FALSE;
        env->SetBooleanArrayRegion(foundOut, 0, 1, &value);
    }
    return as_handle(cornersOut);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect_checkChessboard(JNIEnv *, jobject, jlong image,
                                                   jdouble sizeWidth, jdouble sizeHeight) {
    return cvk_check_chessboard(as_mat(image), sizeWidth, sizeHeight) ? JNI_TRUE : JNI_FALSE;
}

/* Returns the fresh corners Mat handle; writes `found` into foundOut[0]. */
JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_findChessboardCornersSB(JNIEnv *env, jobject, jlong image,
                                                           jdouble patternWidth,
                                                           jdouble patternHeight, jint flags,
                                                           jbooleanArray foundOut) {
    cvk_mat_t *cornersOut = nullptr;
    const int found = cvk_find_chessboard_corners_sb(as_mat(image), patternWidth, patternHeight,
                                                     flags, &cornersOut);
    if (foundOut != nullptr && env->GetArrayLength(foundOut) > 0) {
        const jboolean value = found ? JNI_TRUE : JNI_FALSE;
        env->SetBooleanArrayRegion(foundOut, 0, 1, &value);
    }
    return as_handle(cornersOut);
}

/* Returns [corners, meta] Mat handles; writes `found` into foundOut[0]. */
JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect_findChessboardCornersSBWithMeta(
    JNIEnv *env, jobject, jlong image, jdouble patternWidth, jdouble patternHeight, jint flags,
    jbooleanArray foundOut) {
    cvk_mat_t *cornersOut = nullptr;
    cvk_mat_t *metaOut = nullptr;
    const int found = cvk_find_chessboard_corners_sb_with_meta(
        as_mat(image), patternWidth, patternHeight, flags, &cornersOut, &metaOut);
    if (foundOut != nullptr && env->GetArrayLength(foundOut) > 0) {
        const jboolean value = found ? JNI_TRUE : JNI_FALSE;
        env->SetBooleanArrayRegion(foundOut, 0, 1, &value);
    }
    jlongArray out = env->NewLongArray(2);
    if (out == nullptr) return nullptr;
    const jlong values[2] = {as_handle(cornersOut), as_handle(metaOut)};
    env->SetLongArrayRegion(out, 0, 2, values);
    return out;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_estimateChessboardSharpness(
    JNIEnv *env, jobject, jlong image, jdouble patternWidth, jdouble patternHeight,
    jlong corners, jfloat riseDistance, jboolean vertical, jdoubleArray scalarOut) {
    cvk_scalar_t s;
    cvk_mat_t *sharpness = nullptr;
    const int ok = cvk_estimate_chessboard_sharpness(
        as_mat(image), patternWidth, patternHeight, as_mat(corners), riseDistance,
        vertical ? 1 : 0, &s, &sharpness);
    if (!ok) return 0L;
    if (scalarOut != nullptr && env->GetArrayLength(scalarOut) >= 4) {
        const jdouble values[4] = {s.v0, s.v1, s.v2, s.v3};
        env->SetDoubleArrayRegion(scalarOut, 0, 4, values);
    }
    return as_handle(sharpness);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect_find4QuadCornerSubpix(JNIEnv *, jobject, jlong image,
                                                         jlong corners, jdouble regionWidth,
                                                         jdouble regionHeight) {
    return cvk_find4_quad_corner_subpix(as_mat(image), as_mat(corners), regionWidth, regionHeight)
               ? JNI_TRUE
               : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_drawChessboardCorners(JNIEnv *, jobject, jlong image,
                                                         jdouble patternWidth,
                                                         jdouble patternHeight, jlong corners,
                                                         jboolean patternWasFound) {
    cvk_draw_chessboard_corners(as_mat(image), patternWidth, patternHeight, as_mat(corners),
                                patternWasFound ? 1 : 0);
}

/* Returns the fresh centers Mat handle; writes `found` into foundOut[0]. */
JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_findCirclesGrid(JNIEnv *env, jobject, jlong image,
                                                   jdouble patternWidth, jdouble patternHeight,
                                                   jint flags, jbooleanArray foundOut) {
    cvk_mat_t *centersOut = nullptr;
    const int found = cvk_find_circles_grid(as_mat(image), patternWidth, patternHeight, flags,
                                            &centersOut);
    if (foundOut != nullptr && env->GetArrayLength(foundOut) > 0) {
        const jboolean value = found ? JNI_TRUE : JNI_FALSE;
        env->SetBooleanArrayRegion(foundOut, 0, 1, &value);
    }
    return as_handle(centersOut);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_drawDetectedMarkers(JNIEnv *, jobject, jlong image,
                                                       jlong corners, jlong ids,
                                                       jdouble v0, jdouble v1, jdouble v2,
                                                       jdouble v3) {
    cvk_scalar_t color;
    color.v0 = v0;
    color.v1 = v1;
    color.v2 = v2;
    color.v3 = v3;
    cvk_draw_detected_markers(as_mat(image), as_mat(corners), as_mat(ids), color);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect_generateImageMarker(JNIEnv *, jobject, jlong dictionary,
                                                       jint id, jint sidePixels,
                                                       jint borderBits) {
    return as_handle(cvk_generate_image_marker(as_dict(dictionary), id, sidePixels, borderBits));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_drawDetectedCornersCharuco(JNIEnv *, jobject, jlong image,
                                                              jlong charucoCorners,
                                                              jlong charucoIds, jdouble v0,
                                                              jdouble v1, jdouble v2,
                                                              jdouble v3) {
    cvk_scalar_t color;
    color.v0 = v0;
    color.v1 = v1;
    color.v2 = v2;
    color.v3 = v3;
    cvk_draw_detected_corners_charuco(as_mat(image), as_mat(charucoCorners), as_mat(charucoIds),
                                      color);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect_drawDetectedDiamonds(JNIEnv *, jobject, jlong image,
                                                        jlong diamondCorners, jlong diamondIds,
                                                        jdouble v0, jdouble v1, jdouble v2,
                                                        jdouble v3) {
    cvk_scalar_t color;
    color.v0 = v0;
    color.v1 = v1;
    color.v2 = v2;
    color.v3 = v3;
    cvk_draw_detected_diamonds(as_mat(image), as_mat(diamondCorners), as_mat(diamondIds), color);
}

} /* extern "C" */

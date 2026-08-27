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
 * cvk_ C ABI declarations for the OpenCV "objdetect" module — QR / barcode /
 * face / MCC classes (QRCodeDetector, QRCodeDetectorAruco, QRCodeEncoder,
 * GraphicalCodeDetector, BarcodeDetector, FaceDetectorYN, FaceRecognizerSF,
 * CChecker, CCheckerDetector).
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here back both Kotlin/Native (cinterop) and the
 * JVM (JNI).
 *
 * Conventions (same as the rest of the cvk_ ABI):
 *  - every function is noexcept; failures return NULL / 0 / false and are
 *    reported through cvk_last_error();
 *  - `cvk_mat_t *` outputs are pre-allocated by the Kotlin layer (like the
 *    Java SDK's caller-allocated `Mat points = new Mat()`) and filled by the
 *    callee;
 *  - vector<string> results travel as a malloc'd buffer released with
 *    cvk_free_buffer(): [u32le ok][u32le count][u32le byteLen][utf8 bytes]...
 *    with additional packed string lists appended after the first one;
 *  - vector<Mat> results travel as a CV_8UC1 wire Mat:
 *    [u32le count][per mat: u32le rows, u32le cols, u32le type, u32le dataLen,
 *    raw continuous bytes];
 *  - vector<Point2f> / vector<float> / vector<Rect> / vector<uchar> args and
 *    results use the MatOf* wire types (CV_32FC2 / CV_32FC1 / CV_32SC4 /
 *    CV_8UC1).
 */
#ifndef OPENCV_KMP_OBJDETECT2_H
#define OPENCV_KMP_OBJDETECT2_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- opaque handles ---------------------------------------------------- */

typedef struct cvk_qr_code_detector cvk_qr_code_detector_t;
typedef struct cvk_qr_code_detector_aruco cvk_qr_code_detector_aruco_t;
typedef struct cvk_qr_code_encoder cvk_qr_code_encoder_t;
typedef struct cvk_barcode_detector cvk_barcode_detector_t;
typedef struct cvk_face_detector_yn cvk_face_detector_yn_t;
typedef struct cvk_face_recognizer_sf cvk_face_recognizer_sf_t;
typedef struct cvk_c_checker cvk_c_checker_t;
typedef struct cvk_c_checker_detector cvk_c_checker_detector_t;

/* ---- cv::QRCodeDetector -------------------------------------------------- */

/** cv::QRCodeDetector::QRCodeDetector(). */
cvk_qr_code_detector_t *cvk_qr_code_detector_create(void);

/** In-place cv::QRCodeDetector::setEpsX/setEpsY/setUseAlignmentMarkers. */
void cvk_qr_code_detector_set_eps_x(cvk_qr_code_detector_t *h, double eps_x);
void cvk_qr_code_detector_set_eps_y(cvk_qr_code_detector_t *h, double eps_y);
void cvk_qr_code_detector_set_use_alignment_markers(cvk_qr_code_detector_t *h, int use);

/** cv::QRCodeDetector::decodeCurved / detectAndDecodeCurved (UTF-8, "" if none).
 *  Note: the underlying C++ methods are non-const, so the handle is not const. */
const char *cvk_qr_code_detector_decode_curved(cvk_qr_code_detector_t *h,
                                               const cvk_mat_t *img,
                                               const cvk_mat_t *points,
                                               cvk_mat_t *straight_qrcode);
const char *cvk_qr_code_detector_detect_and_decode_curved(cvk_qr_code_detector_t *h,
                                                          const cvk_mat_t *img,
                                                          cvk_mat_t *points,
                                                          cvk_mat_t *straight_qrcode);

/** cv::QRCodeDetector::getEncoding(codeIdx). */
int cvk_qr_code_detector_get_encoding(cvk_qr_code_detector_t *h, int code_idx);

/* GraphicalCodeDetector surface (per concrete detector type). */
int cvk_qr_code_detector_detect(const cvk_qr_code_detector_t *h, const cvk_mat_t *img,
                                cvk_mat_t *points);
const char *cvk_qr_code_detector_decode(const cvk_qr_code_detector_t *h, const cvk_mat_t *img,
                                        const cvk_mat_t *points, cvk_mat_t *straight_code);
const char *cvk_qr_code_detector_detect_and_decode(const cvk_qr_code_detector_t *h,
                                                   const cvk_mat_t *img,
                                                   cvk_mat_t *points, cvk_mat_t *straight_code);
int cvk_qr_code_detector_detect_multi(const cvk_qr_code_detector_t *h, const cvk_mat_t *img,
                                      cvk_mat_t *points);
unsigned char *cvk_qr_code_detector_decode_multi(const cvk_qr_code_detector_t *h,
                                                 const cvk_mat_t *img, const cvk_mat_t *points,
                                                 cvk_mat_t *straight_code, size_t *out_len);
unsigned char *cvk_qr_code_detector_detect_and_decode_multi(const cvk_qr_code_detector_t *h,
                                                            const cvk_mat_t *img,
                                                            cvk_mat_t *points,
                                                            cvk_mat_t *straight_code,
                                                            size_t *out_len);

void cvk_qr_code_detector_release(cvk_qr_code_detector_t *h);

/* ---- cv::QRCodeDetectorAruco ---------------------------------------------- */

/** cv::QRCodeDetectorAruco::QRCodeDetectorAruco(). */
cvk_qr_code_detector_aruco_t *cvk_qr_code_detector_aruco_create(void);

/** cv::QRCodeDetectorAruco::QRCodeDetectorAruco(params) — 7 Params fields. */
cvk_qr_code_detector_aruco_t *cvk_qr_code_detector_aruco_create_with_params(
    float min_module_size_in_pyramid, float max_rotation, float max_module_size_mismatch,
    float max_timing_pattern_mismatch, float max_penalties, float max_colors_mismatch,
    float scale_timing_pattern_score);

/** cv::QRCodeDetectorAruco::getDetectorParameters/setDetectorParameters. */
void cvk_qr_code_detector_aruco_get_detector_params(const cvk_qr_code_detector_aruco_t *h,
                                                    float *out7);
void cvk_qr_code_detector_aruco_set_detector_params(cvk_qr_code_detector_aruco_t *h,
                                                    float min_module_size_in_pyramid,
                                                    float max_rotation,
                                                    float max_module_size_mismatch,
                                                    float max_timing_pattern_mismatch,
                                                    float max_penalties,
                                                    float max_colors_mismatch,
                                                    float scale_timing_pattern_score);

/* GraphicalCodeDetector surface. */
int cvk_qr_code_detector_aruco_detect(const cvk_qr_code_detector_aruco_t *h, const cvk_mat_t *img,
                                      cvk_mat_t *points);
const char *cvk_qr_code_detector_aruco_decode(const cvk_qr_code_detector_aruco_t *h,
                                              const cvk_mat_t *img, const cvk_mat_t *points,
                                              cvk_mat_t *straight_code);
const char *cvk_qr_code_detector_aruco_detect_and_decode(const cvk_qr_code_detector_aruco_t *h,
                                                         const cvk_mat_t *img,
                                                         cvk_mat_t *points,
                                                         cvk_mat_t *straight_code);
int cvk_qr_code_detector_aruco_detect_multi(const cvk_qr_code_detector_aruco_t *h,
                                            const cvk_mat_t *img, cvk_mat_t *points);
unsigned char *cvk_qr_code_detector_aruco_decode_multi(const cvk_qr_code_detector_aruco_t *h,
                                                       const cvk_mat_t *img,
                                                       const cvk_mat_t *points,
                                                       cvk_mat_t *straight_code,
                                                       size_t *out_len);
unsigned char *cvk_qr_code_detector_aruco_detect_and_decode_multi(
    const cvk_qr_code_detector_aruco_t *h, const cvk_mat_t *img, cvk_mat_t *points,
    cvk_mat_t *straight_code, size_t *out_len);

void cvk_qr_code_detector_aruco_release(cvk_qr_code_detector_aruco_t *h);

/* ---- cv::QRCodeEncoder ----------------------------------------------------- */

/** cv::QRCodeEncoder::create(). */
cvk_qr_code_encoder_t *cvk_qr_code_encoder_create(void);

/** cv::QRCodeEncoder::create(params) — 4 Params fields. */
cvk_qr_code_encoder_t *cvk_qr_code_encoder_create_with_params(int version,
                                                              int correction_level,
                                                              int mode,
                                                              int structure_number);

/** cv::QRCodeEncoder::encode(String, Mat&) — fills the caller's qrcode Mat.
 *  Returns 1 on success, 0 on failure (cvk_last_error() describes it). */
int cvk_qr_code_encoder_encode(const cvk_qr_code_encoder_t *h, const char *encoded_info,
                               cvk_mat_t *qrcode);

/** cv::QRCodeEncoder::encode(byte[], Mat&) — binary payload overload. */
int cvk_qr_code_encoder_encode_bytes(const cvk_qr_code_encoder_t *h,
                                     const unsigned char *data, size_t len,
                                     cvk_mat_t *qrcode);

/** cv::QRCodeEncoder::encodeStructuredAppend(String, vector<Mat>&) — wire Mat out. */
int cvk_qr_code_encoder_encode_structured_append(const cvk_qr_code_encoder_t *h,
                                                 const char *encoded_info,
                                                 cvk_mat_t *qrcodes);

void cvk_qr_code_encoder_release(cvk_qr_code_encoder_t *h);

/* ---- cv::barcode::BarcodeDetector ------------------------------------------- */

/** cv::barcode::BarcodeDetector::BarcodeDetector() — super resolution disabled. */
cvk_barcode_detector_t *cvk_barcode_detector_create(void);

/** cv::barcode::BarcodeDetector::BarcodeDetector(ONNX super-resolution model path). */
cvk_barcode_detector_t *cvk_barcode_detector_create_with_model(const char *model_path);

/* GraphicalCodeDetector surface. */
int cvk_barcode_detector_detect(const cvk_barcode_detector_t *h, const cvk_mat_t *img,
                                cvk_mat_t *points);
const char *cvk_barcode_detector_decode(const cvk_barcode_detector_t *h, const cvk_mat_t *img,
                                        const cvk_mat_t *points, cvk_mat_t *straight_code);
const char *cvk_barcode_detector_detect_and_decode(const cvk_barcode_detector_t *h,
                                                   const cvk_mat_t *img,
                                                   cvk_mat_t *points, cvk_mat_t *straight_code);
int cvk_barcode_detector_detect_multi(const cvk_barcode_detector_t *h, const cvk_mat_t *img,
                                      cvk_mat_t *points);
unsigned char *cvk_barcode_detector_decode_multi(const cvk_barcode_detector_t *h,
                                                 const cvk_mat_t *img, const cvk_mat_t *points,
                                                 cvk_mat_t *straight_code, size_t *out_len);
unsigned char *cvk_barcode_detector_detect_and_decode_multi(const cvk_barcode_detector_t *h,
                                                            const cvk_mat_t *img,
                                                            cvk_mat_t *points,
                                                            cvk_mat_t *straight_code,
                                                            size_t *out_len);

/** cv::barcode::BarcodeDetector::decodeWithType/detectAndDecodeWithType — buffer holds
 *  [u32le ok][string list: decoded_info][string list: decoded_type]. */
unsigned char *cvk_barcode_detector_decode_with_type(const cvk_barcode_detector_t *h,
                                                     const cvk_mat_t *img,
                                                     const cvk_mat_t *points, size_t *out_len);
unsigned char *cvk_barcode_detector_detect_and_decode_with_type(const cvk_barcode_detector_t *h,
                                                                const cvk_mat_t *img,
                                                                cvk_mat_t *points,
                                                                size_t *out_len);

double cvk_barcode_detector_get_downsampling_threshold(const cvk_barcode_detector_t *h);
void cvk_barcode_detector_set_downsampling_threshold(cvk_barcode_detector_t *h, double thresh);
void cvk_barcode_detector_get_detector_scales(const cvk_barcode_detector_t *h, cvk_mat_t *sizes);
void cvk_barcode_detector_set_detector_scales(cvk_barcode_detector_t *h, const cvk_mat_t *sizes);
double cvk_barcode_detector_get_gradient_threshold(const cvk_barcode_detector_t *h);
void cvk_barcode_detector_set_gradient_threshold(cvk_barcode_detector_t *h, double thresh);

void cvk_barcode_detector_release(cvk_barcode_detector_t *h);

/* ---- cv::FaceDetectorYN ------------------------------------------------------ */

/** cv::FaceDetectorYN::create(model, config, size, ...). */
cvk_face_detector_yn_t *cvk_face_detector_yn_create(const char *model, const char *config,
                                                    int input_width, int input_height,
                                                    float score_threshold, float nms_threshold,
                                                    int top_k, int backend_id, int target_id);

/** cv::FaceDetectorYN::create(framework, bufferModel, bufferConfig, size, ...). */
cvk_face_detector_yn_t *cvk_face_detector_yn_create_from_buffers(
    const char *framework, const cvk_mat_t *buffer_model, const cvk_mat_t *buffer_config,
    int input_width, int input_height, float score_threshold, float nms_threshold,
    int top_k, int backend_id, int target_id);

void cvk_face_detector_yn_set_input_size(cvk_face_detector_yn_t *h, int width, int height);
void cvk_face_detector_yn_get_input_size(const cvk_face_detector_yn_t *h, int *out2);
void cvk_face_detector_yn_set_score_threshold(cvk_face_detector_yn_t *h, float threshold);
float cvk_face_detector_yn_get_score_threshold(const cvk_face_detector_yn_t *h);
void cvk_face_detector_yn_set_nms_threshold(cvk_face_detector_yn_t *h, float threshold);
float cvk_face_detector_yn_get_nms_threshold(const cvk_face_detector_yn_t *h);
void cvk_face_detector_yn_set_top_k(cvk_face_detector_yn_t *h, int top_k);
int cvk_face_detector_yn_get_top_k(const cvk_face_detector_yn_t *h);

/** cv::FaceDetectorYN::detect(image, faces) — fills faces [num_faces, 15], returns count. */
int cvk_face_detector_yn_detect(const cvk_face_detector_yn_t *h, const cvk_mat_t *image,
                                cvk_mat_t *faces);

void cvk_face_detector_yn_release(cvk_face_detector_yn_t *h);

/* ---- cv::FaceRecognizerSF ----------------------------------------------------- */

/** cv::FaceRecognizerSF::create(model, config, ...). */
cvk_face_recognizer_sf_t *cvk_face_recognizer_sf_create(const char *model, const char *config,
                                                        int backend_id, int target_id);

/** cv::FaceRecognizerSF::create(framework, bufferModel, bufferConfig, ...). */
cvk_face_recognizer_sf_t *cvk_face_recognizer_sf_create_from_buffers(
    const char *framework, const cvk_mat_t *buffer_model, const cvk_mat_t *buffer_config,
    int backend_id, int target_id);

/** cv::FaceRecognizerSF::alignCrop(src_img, face_box, aligned_img&). */
void cvk_face_recognizer_sf_align_crop(const cvk_face_recognizer_sf_t *h,
                                       const cvk_mat_t *src_img, const cvk_mat_t *face_box,
                                       cvk_mat_t *aligned_img);

/** cv::FaceRecognizerSF::feature(aligned_img, face_feature&). */
void cvk_face_recognizer_sf_feature(const cvk_face_recognizer_sf_t *h,
                                    const cvk_mat_t *aligned_img, cvk_mat_t *face_feature);

/** cv::FaceRecognizerSF::match(f1, f2, disType). */
double cvk_face_recognizer_sf_match(const cvk_face_recognizer_sf_t *h,
                                    const cvk_mat_t *feature1, const cvk_mat_t *feature2,
                                    int dis_type);

void cvk_face_recognizer_sf_release(cvk_face_recognizer_sf_t *h);

/* ---- cv::mcc::CChecker --------------------------------------------------------- */

/** cv::mcc::CChecker::create(). */
cvk_c_checker_t *cvk_c_checker_create(void);

void cvk_c_checker_set_box(cvk_c_checker_t *h, const cvk_mat_t *box);          /* CV_32FC2 */
void cvk_c_checker_set_charts_rgb(cvk_c_checker_t *h, const cvk_mat_t *charts);
void cvk_c_checker_set_charts_y_cb_cr(cvk_c_checker_t *h, const cvk_mat_t *charts);
void cvk_c_checker_set_cost(cvk_c_checker_t *h, float cost);
void cvk_c_checker_set_center(cvk_c_checker_t *h, double x, double y);

cvk_mat_t *cvk_c_checker_get_box(const cvk_c_checker_t *h);                    /* CV_32FC2 */
cvk_mat_t *cvk_c_checker_get_color_charts(const cvk_c_checker_t *h);           /* CV_32FC2 */
cvk_mat_t *cvk_c_checker_get_charts_rgb(const cvk_c_checker_t *h, int get_stats);
cvk_mat_t *cvk_c_checker_get_charts_y_cb_cr(const cvk_c_checker_t *h);
float cvk_c_checker_get_cost(const cvk_c_checker_t *h);
void cvk_c_checker_get_center(const cvk_c_checker_t *h, double *out2);

/* Algorithm surface (clear/empty/save/getDefaultName). */
void cvk_c_checker_clear(cvk_c_checker_t *h);
int cvk_c_checker_empty(const cvk_c_checker_t *h);
void cvk_c_checker_save(cvk_c_checker_t *h, const char *filename);
const char *cvk_c_checker_get_default_name(const cvk_c_checker_t *h);

void cvk_c_checker_release(cvk_c_checker_t *h);

/* ---- cv::mcc::CCheckerDetector --------------------------------------------------- */

/** cv::mcc::CCheckerDetector::create(). */
cvk_c_checker_detector_t *cvk_c_checker_detector_create(void);

/** cv::mcc::CCheckerDetector::process(image, regionsOfInterest, nc). */
int cvk_c_checker_detector_process_with_roi(const cvk_c_checker_detector_t *h,
                                            const cvk_mat_t *image, const cvk_mat_t *roi,
                                            int nc);

/** cv::mcc::CCheckerDetector::process(image, nc). */
int cvk_c_checker_detector_process(const cvk_c_checker_detector_t *h, const cvk_mat_t *image,
                                   int nc);

/** cv::mcc::CCheckerDetector::getListColorChecker() — writes up to `capacity`
 *  handles into out[] and returns the total count. */
int cvk_c_checker_detector_get_list(const cvk_c_checker_detector_t *h,
                                    cvk_c_checker_t **out, int capacity);

/** cv::mcc::CCheckerDetector::getRefColors(). */
cvk_mat_t *cvk_c_checker_detector_get_ref_colors(const cvk_c_checker_detector_t *h);

/** cv::mcc::CCheckerDetector::setDetectionParams(params) — 18 DetectorParametersMCC
 *  fields (ints and doubles). */
void cvk_c_checker_detector_set_detection_params(cvk_c_checker_detector_t *h,
                                                 int adaptive_thresh_win_size_min,
                                                 int adaptive_thresh_win_size_max,
                                                 int adaptive_thresh_win_size_step,
                                                 double adaptive_thresh_constant,
                                                 double min_contours_area_rate,
                                                 double min_contours_area,
                                                 double confidence_threshold,
                                                 double min_contour_solidity,
                                                 double find_candidates_approx_poly_d_peps_multiplier,
                                                 int border_width,
                                                 float b0factor, float max_error,
                                                 int min_contour_points_allowed,
                                                 int min_contour_length_allowed,
                                                 int min_inter_contour_distance,
                                                 int min_inter_checker_distance,
                                                 int min_image_size,
                                                 int min_group_size);

/** cv::mcc::CCheckerDetector::getDetectionParams() — doubles, index order matches
 *  set_detection_params (ints/floats widened to double). */
void cvk_c_checker_detector_get_detection_params(const cvk_c_checker_detector_t *h,
                                                 double *out18);

/** cv::mcc::CCheckerDetector::setUseDnnModel/getUseDnnModel. */
void cvk_c_checker_detector_set_use_dnn_model(cvk_c_checker_detector_t *h, int use_dnn);
int cvk_c_checker_detector_get_use_dnn_model(const cvk_c_checker_detector_t *h);

/* Algorithm surface (clear/empty/save/getDefaultName). */
void cvk_c_checker_detector_clear(cvk_c_checker_detector_t *h);
int cvk_c_checker_detector_empty(const cvk_c_checker_detector_t *h);
void cvk_c_checker_detector_save(cvk_c_checker_detector_t *h, const char *filename);
const char *cvk_c_checker_detector_get_default_name(const cvk_c_checker_detector_t *h);

void cvk_c_checker_detector_release(cvk_c_checker_detector_t *h);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_OBJDETECT2_H */

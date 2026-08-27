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
 * JNI bridge for the objdetect QR / barcode / face / MCC slice: thin
 * Java_cn_enaium_opencv_JniObjdetect2_* wrappers around the cvk_ C ABI in
 * native/shim_objdetect2.cpp. Mat handles travel as jlong pointers; string
 * and string-list results are converted to jstring / jbyteArray (the packed
 * buffer format documented in opencv_kmp_objdetect2.h).
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_objdetect2.h"

#include <cstdint>
#include <string>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_qr_code_detector_t *as_qr_code_detector(jlong h) {
    return reinterpret_cast<cvk_qr_code_detector_t *>(static_cast<uintptr_t>(h));
}
static inline cvk_qr_code_detector_aruco_t *as_qr_code_detector_aruco(jlong h) {
    return reinterpret_cast<cvk_qr_code_detector_aruco_t *>(static_cast<uintptr_t>(h));
}
static inline cvk_qr_code_encoder_t *as_qr_code_encoder(jlong h) {
    return reinterpret_cast<cvk_qr_code_encoder_t *>(static_cast<uintptr_t>(h));
}
static inline cvk_barcode_detector_t *as_barcode_detector(jlong h) {
    return reinterpret_cast<cvk_barcode_detector_t *>(static_cast<uintptr_t>(h));
}
static inline cvk_face_detector_yn_t *as_face_detector_yn(jlong h) {
    return reinterpret_cast<cvk_face_detector_yn_t *>(static_cast<uintptr_t>(h));
}
static inline cvk_face_recognizer_sf_t *as_face_recognizer_sf(jlong h) {
    return reinterpret_cast<cvk_face_recognizer_sf_t *>(static_cast<uintptr_t>(h));
}
static inline cvk_c_checker_t *as_c_checker(jlong h) {
    return reinterpret_cast<cvk_c_checker_t *>(static_cast<uintptr_t>(h));
}
static inline cvk_c_checker_detector_t *as_c_checker_detector(jlong h) {
    return reinterpret_cast<cvk_c_checker_detector_t *>(static_cast<uintptr_t>(h));
}

static std::string jstring_to_std(JNIEnv *env, jstring s) {
    if (s == nullptr) return {};
    const char *chars = env->GetStringUTFChars(s, nullptr);
    if (chars == nullptr) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

extern "C" {

/* ---- cv::QRCodeDetector -------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_qr_code_detector_create()));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorSetEpsX(JNIEnv *, jobject, jlong h, jdouble eps) {
    cvk_qr_code_detector_set_eps_x(as_qr_code_detector(h), eps);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorSetEpsY(JNIEnv *, jobject, jlong h, jdouble eps) {
    cvk_qr_code_detector_set_eps_y(as_qr_code_detector(h), eps);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorSetUseAlignmentMarkers(JNIEnv *, jobject,
                                                                         jlong h, jboolean use) {
    cvk_qr_code_detector_set_use_alignment_markers(as_qr_code_detector(h), use != JNI_FALSE);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorDecodeCurved(JNIEnv *env, jobject, jlong h,
                                                               jlong img, jlong points,
                                                               jlong straight) {
    const char *s = cvk_qr_code_detector_decode_curved(as_qr_code_detector(h), as_mat(img),
                                                       as_mat(points), as_mat(straight));
    return s != nullptr ? env->NewStringUTF(s) : nullptr;
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorDetectAndDecodeCurved(JNIEnv *env, jobject,
                                                                        jlong h, jlong img,
                                                                        jlong points,
                                                                        jlong straight) {
    const char *s = cvk_qr_code_detector_detect_and_decode_curved(as_qr_code_detector(h),
                                                                  as_mat(img), as_mat(points),
                                                                  as_mat(straight));
    return s != nullptr ? env->NewStringUTF(s) : nullptr;
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorGetEncoding(JNIEnv *, jobject, jlong h,
                                                              jint code_idx) {
    return cvk_qr_code_detector_get_encoding(as_qr_code_detector(h), code_idx);
}

/* ---- cv::QRCodeDetectorAruco ---------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorArucoCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_qr_code_detector_aruco_create()));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorArucoCreateWithParams(
    JNIEnv *, jobject, jfloat min_module, jfloat max_rot, jfloat max_mismatch,
    jfloat max_timing, jfloat max_penalties, jfloat max_colors, jfloat scale_timing) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
        cvk_qr_code_detector_aruco_create_with_params(
            min_module, max_rot, max_mismatch, max_timing, max_penalties, max_colors,
            scale_timing)));
}

JNIEXPORT jfloatArray JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorArucoGetDetectorParams(JNIEnv *env, jobject,
                                                                         jlong h) {
    float out[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    cvk_qr_code_detector_aruco_get_detector_params(as_qr_code_detector_aruco(h), out);
    jfloatArray arr = env->NewFloatArray(7);
    env->SetFloatArrayRegion(arr, 0, 7, out);
    return arr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeDetectorArucoSetDetectorParams(
    JNIEnv *, jobject, jlong h, jfloat min_module, jfloat max_rot, jfloat max_mismatch,
    jfloat max_timing, jfloat max_penalties, jfloat max_colors, jfloat scale_timing) {
    cvk_qr_code_detector_aruco_set_detector_params(
        as_qr_code_detector_aruco(h), min_module, max_rot, max_mismatch, max_timing,
        max_penalties, max_colors, scale_timing);
}

/* ---- cv::QRCodeEncoder ----------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeEncoderCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_qr_code_encoder_create()));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeEncoderCreateWithParams(JNIEnv *, jobject, jint version,
                                                                  jint correction,
                                                                  jint mode, jint structure) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
        cvk_qr_code_encoder_create_with_params(version, correction, mode, structure)));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeEncoderEncode(JNIEnv *env, jobject, jlong h,
                                                        jstring info, jlong qrcode) {
    const std::string s = jstring_to_std(env, info);
    return cvk_qr_code_encoder_encode(as_qr_code_encoder(h), s.c_str(), as_mat(qrcode));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeEncoderEncodeBytes(JNIEnv *env, jobject, jlong h,
                                                             jbyteArray data, jlong qrcode) {
    const jsize len = env->GetArrayLength(data);
    std::vector<unsigned char> bytes(static_cast<size_t>(len));
    env->GetByteArrayRegion(data, 0, len, reinterpret_cast<jbyte *>(bytes.data()));
    return cvk_qr_code_encoder_encode_bytes(as_qr_code_encoder(h), bytes.data(), bytes.size(),
                                            as_mat(qrcode));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeEncoderEncodeStructuredAppend(JNIEnv *env, jobject,
                                                                        jlong h, jstring info,
                                                                        jlong qrcodes) {
    const std::string s = jstring_to_std(env, info);
    return cvk_qr_code_encoder_encode_structured_append(as_qr_code_encoder(h), s.c_str(),
                                                        as_mat(qrcodes));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_qrCodeEncoderRelease(JNIEnv *, jobject, jlong h) {
    cvk_qr_code_encoder_release(as_qr_code_encoder(h));
}

/* ---- cv::barcode::BarcodeDetector ------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_barcode_detector_create()));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorCreateWithModel(JNIEnv *env, jobject,
                                                                   jstring model_path) {
    const std::string path = jstring_to_std(env, model_path);
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_barcode_detector_create_with_model(path.c_str())));
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorDecodeWithType(JNIEnv *env, jobject, jlong h,
                                                                  jlong img, jlong points) {
    size_t len = 0;
    unsigned char *buf = cvk_barcode_detector_decode_with_type(as_barcode_detector(h),
                                                               as_mat(img), as_mat(points),
                                                               &len);
    if (buf == nullptr) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(len));
    if (arr != nullptr) {
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(len),
                                reinterpret_cast<const jbyte *>(buf));
    }
    cvk_free_buffer(buf);
    return arr;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorDetectAndDecodeWithType(JNIEnv *env, jobject,
                                                                           jlong h, jlong img,
                                                                           jlong points) {
    size_t len = 0;
    unsigned char *buf = cvk_barcode_detector_detect_and_decode_with_type(
        as_barcode_detector(h), as_mat(img), as_mat(points), &len);
    if (buf == nullptr) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(len));
    if (arr != nullptr) {
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(len),
                                reinterpret_cast<const jbyte *>(buf));
    }
    cvk_free_buffer(buf);
    return arr;
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorGetDownsamplingThreshold(JNIEnv *, jobject,
                                                                            jlong h) {
    return cvk_barcode_detector_get_downsampling_threshold(as_barcode_detector(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorSetDownsamplingThreshold(JNIEnv *, jobject,
                                                                            jlong h,
                                                                            jdouble thresh) {
    cvk_barcode_detector_set_downsampling_threshold(as_barcode_detector(h), thresh);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorGetDetectorScales(JNIEnv *, jobject, jlong h,
                                                                     jlong sizes) {
    cvk_barcode_detector_get_detector_scales(as_barcode_detector(h), as_mat(sizes));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorSetDetectorScales(JNIEnv *, jobject, jlong h,
                                                                     jlong sizes) {
    cvk_barcode_detector_set_detector_scales(as_barcode_detector(h), as_mat(sizes));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorGetGradientThreshold(JNIEnv *, jobject,
                                                                        jlong h) {
    return cvk_barcode_detector_get_gradient_threshold(as_barcode_detector(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_barcodeDetectorSetGradientThreshold(JNIEnv *, jobject,
                                                                        jlong h, jdouble thresh) {
    cvk_barcode_detector_set_gradient_threshold(as_barcode_detector(h), thresh);
}

/* ---- cv::FaceDetectorYN ------------------------------------------------------ */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNCreate(JNIEnv *env, jobject, jstring model,
                                                         jstring config, jint w, jint h,
                                                         jfloat score, jfloat nms, jint top_k,
                                                         jint backend, jint target) {
    const std::string m = jstring_to_std(env, model);
    const std::string c = jstring_to_std(env, config);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
        cvk_face_detector_yn_create(m.c_str(), c.c_str(), w, h, score, nms, top_k, backend,
                                    target)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNCreateFromBuffers(
    JNIEnv *env, jobject, jstring framework, jlong buffer_model, jlong buffer_config,
    jint w, jint h, jfloat score, jfloat nms, jint top_k, jint backend, jint target) {
    const std::string f = jstring_to_std(env, framework);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
        cvk_face_detector_yn_create_from_buffers(f.c_str(), as_mat(buffer_model),
                                                 as_mat(buffer_config), w, h, score, nms,
                                                 top_k, backend, target)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNSetInputSize(JNIEnv *, jobject, jlong h,
                                                               jint w, jint h2) {
    cvk_face_detector_yn_set_input_size(as_face_detector_yn(h), w, h2);
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNGetInputSize(JNIEnv *env, jobject, jlong h) {
    int out[2] = {0, 0};
    cvk_face_detector_yn_get_input_size(as_face_detector_yn(h), out);
    jintArray arr = env->NewIntArray(2);
    env->SetIntArrayRegion(arr, 0, 2, out);
    return arr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNSetScoreThreshold(JNIEnv *, jobject, jlong h,
                                                                    jfloat t) {
    cvk_face_detector_yn_set_score_threshold(as_face_detector_yn(h), t);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNGetScoreThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_face_detector_yn_get_score_threshold(as_face_detector_yn(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNSetNmsThreshold(JNIEnv *, jobject, jlong h,
                                                                  jfloat t) {
    cvk_face_detector_yn_set_nms_threshold(as_face_detector_yn(h), t);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNGetNmsThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_face_detector_yn_get_nms_threshold(as_face_detector_yn(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNSetTopK(JNIEnv *, jobject, jlong h, jint top_k) {
    cvk_face_detector_yn_set_top_k(as_face_detector_yn(h), top_k);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNGetTopK(JNIEnv *, jobject, jlong h) {
    return cvk_face_detector_yn_get_top_k(as_face_detector_yn(h));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNDetect(JNIEnv *, jobject, jlong h,
                                                         jlong image, jlong faces) {
    return cvk_face_detector_yn_detect(as_face_detector_yn(h), as_mat(image), as_mat(faces));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceDetectorYNRelease(JNIEnv *, jobject, jlong h) {
    cvk_face_detector_yn_release(as_face_detector_yn(h));
}

/* ---- cv::FaceRecognizerSF ----------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceRecognizerSFCreate(JNIEnv *env, jobject, jstring model,
                                                           jstring config, jint backend,
                                                           jint target) {
    const std::string m = jstring_to_std(env, model);
    const std::string c = jstring_to_std(env, config);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
        cvk_face_recognizer_sf_create(m.c_str(), c.c_str(), backend, target)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceRecognizerSFCreateFromBuffers(
    JNIEnv *env, jobject, jstring framework, jlong buffer_model, jlong buffer_config,
    jint backend, jint target) {
    const std::string f = jstring_to_std(env, framework);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(
        cvk_face_recognizer_sf_create_from_buffers(f.c_str(), as_mat(buffer_model),
                                                   as_mat(buffer_config), backend, target)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceRecognizerSFAlignCrop(JNIEnv *, jobject, jlong h,
                                                              jlong src, jlong box,
                                                              jlong aligned) {
    cvk_face_recognizer_sf_align_crop(as_face_recognizer_sf(h), as_mat(src), as_mat(box),
                                      as_mat(aligned));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceRecognizerSFFeature(JNIEnv *, jobject, jlong h,
                                                            jlong aligned, jlong feature) {
    cvk_face_recognizer_sf_feature(as_face_recognizer_sf(h), as_mat(aligned), as_mat(feature));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceRecognizerSFMatch(JNIEnv *, jobject, jlong h,
                                                          jlong f1, jlong f2, jint dis_type) {
    return cvk_face_recognizer_sf_match(as_face_recognizer_sf(h), as_mat(f1), as_mat(f2),
                                        dis_type);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_faceRecognizerSFRelease(JNIEnv *, jobject, jlong h) {
    cvk_face_recognizer_sf_release(as_face_recognizer_sf(h));
}

/* ---- cv::mcc::CChecker --------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_c_checker_create()));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerSetBox(JNIEnv *, jobject, jlong h, jlong box) {
    cvk_c_checker_set_box(as_c_checker(h), as_mat(box));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerSetChartsRgb(JNIEnv *, jobject, jlong h,
                                                         jlong charts) {
    cvk_c_checker_set_charts_rgb(as_c_checker(h), as_mat(charts));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerSetChartsYCbCr(JNIEnv *, jobject, jlong h,
                                                           jlong charts) {
    cvk_c_checker_set_charts_y_cb_cr(as_c_checker(h), as_mat(charts));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerSetCost(JNIEnv *, jobject, jlong h, jfloat cost) {
    cvk_c_checker_set_cost(as_c_checker(h), cost);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerSetCenter(JNIEnv *, jobject, jlong h, jdouble x,
                                                      jdouble y) {
    cvk_c_checker_set_center(as_c_checker(h), x, y);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerGetBox(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_c_checker_get_box(as_c_checker(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerGetColorCharts(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_c_checker_get_color_charts(as_c_checker(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerGetChartsRgb(JNIEnv *, jobject, jlong h,
                                                         jboolean get_stats) {
    return as_handle(cvk_c_checker_get_charts_rgb(as_c_checker(h), get_stats != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerGetChartsYCbCr(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_c_checker_get_charts_y_cb_cr(as_c_checker(h)));
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerGetCost(JNIEnv *, jobject, jlong h) {
    return cvk_c_checker_get_cost(as_c_checker(h));
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerGetCenter(JNIEnv *env, jobject, jlong h) {
    double out[2] = {0.0, 0.0};
    cvk_c_checker_get_center(as_c_checker(h), out);
    jdoubleArray arr = env->NewDoubleArray(2);
    env->SetDoubleArrayRegion(arr, 0, 2, out);
    return arr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerClear(JNIEnv *, jobject, jlong h) {
    cvk_c_checker_clear(as_c_checker(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerEmpty(JNIEnv *, jobject, jlong h) {
    return cvk_c_checker_empty(as_c_checker(h)) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerSave(JNIEnv *env, jobject, jlong h, jstring filename) {
    const std::string f = jstring_to_std(env, filename);
    cvk_c_checker_save(as_c_checker(h), f.c_str());
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerGetDefaultName(JNIEnv *env, jobject, jlong h) {
    const char *name = cvk_c_checker_get_default_name(as_c_checker(h));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerRelease(JNIEnv *, jobject, jlong h) {
    cvk_c_checker_release(as_c_checker(h));
}

/* ---- cv::mcc::CCheckerDetector --------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorCreate(JNIEnv *, jobject) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_c_checker_detector_create()));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorProcessWithRoi(JNIEnv *, jobject, jlong h,
                                                                   jlong image, jlong roi,
                                                                   jint nc) {
    return cvk_c_checker_detector_process_with_roi(as_c_checker_detector(h), as_mat(image),
                                                   as_mat(roi), nc) != 0;
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorProcess(JNIEnv *, jobject, jlong h,
                                                            jlong image, jint nc) {
    return cvk_c_checker_detector_process(as_c_checker_detector(h), as_mat(image), nc) != 0;
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorGetList(JNIEnv *env, jobject, jlong h) {
    const int count = cvk_c_checker_detector_get_list(as_c_checker_detector(h), nullptr, 0);
    if (count < 0) return nullptr;
    std::vector<cvk_c_checker_t *> items(static_cast<size_t>(count));
    const int filled = cvk_c_checker_detector_get_list(as_c_checker_detector(h), items.data(),
                                                       count);
    if (filled < 0) return nullptr;
    std::vector<jlong> handles(static_cast<size_t>(filled));
    for (int i = 0; i < filled; ++i) {
        handles[static_cast<size_t>(i)] =
            static_cast<jlong>(reinterpret_cast<uintptr_t>(items[static_cast<size_t>(i)]));
    }
    jlongArray arr = env->NewLongArray(filled);
    if (arr != nullptr) {
        env->SetLongArrayRegion(arr, 0, filled, handles.data());
    }
    return arr;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorGetRefColors(JNIEnv *, jobject, jlong h) {
    return as_handle(cvk_c_checker_detector_get_ref_colors(as_c_checker_detector(h)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorSetDetectionParams(
    JNIEnv *, jobject, jlong h, jint win_min, jint win_max, jint win_step,
    jdouble thresh_constant, jdouble area_rate, jdouble min_area, jdouble confidence,
    jdouble solidity, jdouble approx_eps, jint border_width, jfloat b0factor, jfloat max_error,
    jint min_points, jint min_length, jint inter_contour, jint inter_checker, jint min_image,
    jint min_group) {
    cvk_c_checker_detector_set_detection_params(
        as_c_checker_detector(h), win_min, win_max, win_step, thresh_constant, area_rate,
        min_area, confidence, solidity, approx_eps, border_width, b0factor, max_error,
        min_points, min_length, inter_contour, inter_checker, min_image, min_group);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorGetDetectionParams(JNIEnv *env, jobject,
                                                                       jlong h) {
    double out[18] = {0.0};
    cvk_c_checker_detector_get_detection_params(as_c_checker_detector(h), out);
    jdoubleArray arr = env->NewDoubleArray(18);
    env->SetDoubleArrayRegion(arr, 0, 18, out);
    return arr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorSetUseDnnModel(JNIEnv *, jobject, jlong h,
                                                                   jboolean use) {
    cvk_c_checker_detector_set_use_dnn_model(as_c_checker_detector(h), use != JNI_FALSE);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorGetUseDnnModel(JNIEnv *, jobject, jlong h) {
    return cvk_c_checker_detector_get_use_dnn_model(as_c_checker_detector(h)) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorClear(JNIEnv *, jobject, jlong h) {
    cvk_c_checker_detector_clear(as_c_checker_detector(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorEmpty(JNIEnv *, jobject, jlong h) {
    return cvk_c_checker_detector_empty(as_c_checker_detector(h)) != 0;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorSave(JNIEnv *env, jobject, jlong h,
                                                         jstring filename) {
    const std::string f = jstring_to_std(env, filename);
    cvk_c_checker_detector_save(as_c_checker_detector(h), f.c_str());
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorGetDefaultName(JNIEnv *env, jobject,
                                                                   jlong h) {
    const char *name = cvk_c_checker_detector_get_default_name(as_c_checker_detector(h));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniObjdetect2_cCheckerDetectorRelease(JNIEnv *, jobject, jlong h) {
    cvk_c_checker_detector_release(as_c_checker_detector(h));
}

/* ---- GraphicalCodeDetector surface (per concrete detector) ------------------- */

#define JNI_GCD_FUNCS(J, C, AS)                                                       \
    JNIEXPORT jboolean JNICALL                                                        \
    Java_cn_enaium_opencv_JniObjdetect2_##J##Detect(JNIEnv *, jobject, jlong h,       \
                                                    jlong img, jlong points) {         \
        return cvk_##C##_detect(AS(h), as_mat(img), as_mat(points)) != 0;             \
    }                                                                                 \
    JNIEXPORT jstring JNICALL                                                         \
    Java_cn_enaium_opencv_JniObjdetect2_##J##Decode(JNIEnv *env, jobject, jlong h,    \
                                                    jlong img, jlong points,           \
                                                    jlong straight) {                  \
        const char *s = cvk_##C##_decode(AS(h), as_mat(img), as_mat(points),           \
                                         as_mat(straight));                            \
        return s != nullptr ? env->NewStringUTF(s) : nullptr;                         \
    }                                                                                 \
    JNIEXPORT jstring JNICALL                                                         \
    Java_cn_enaium_opencv_JniObjdetect2_##J##DetectAndDecode(JNIEnv *env, jobject,    \
                                                             jlong h, jlong img,       \
                                                             jlong points,             \
                                                             jlong straight) {          \
        const char *s = cvk_##C##_detect_and_decode(AS(h), as_mat(img), as_mat(points), \
                                                    as_mat(straight));                 \
        return s != nullptr ? env->NewStringUTF(s) : nullptr;                         \
    }                                                                                 \
    JNIEXPORT jboolean JNICALL                                                        \
    Java_cn_enaium_opencv_JniObjdetect2_##J##DetectMulti(JNIEnv *, jobject, jlong h,  \
                                                         jlong img, jlong points) {    \
        return cvk_##C##_detect_multi(AS(h), as_mat(img), as_mat(points)) != 0;       \
    }                                                                                 \
    JNIEXPORT jbyteArray JNICALL                                                      \
    Java_cn_enaium_opencv_JniObjdetect2_##J##DecodeMulti(JNIEnv *env, jobject, jlong h, \
                                                         jlong img, jlong points,      \
                                                         jlong straight) {             \
        size_t len = 0;                                                               \
        unsigned char *buf = cvk_##C##_decode_multi(AS(h), as_mat(img), as_mat(points), \
                                                    as_mat(straight), &len);           \
        if (buf == nullptr) return nullptr;                                           \
        jbyteArray arr = env->NewByteArray(static_cast<jsize>(len));                  \
        if (arr != nullptr) {                                                         \
            env->SetByteArrayRegion(arr, 0, static_cast<jsize>(len),                  \
                                    reinterpret_cast<const jbyte *>(buf));            \
        }                                                                             \
        cvk_free_buffer(buf);                                                         \
        return arr;                                                                   \
    }                                                                                 \
    JNIEXPORT jbyteArray JNICALL                                                      \
    Java_cn_enaium_opencv_JniObjdetect2_##J##DetectAndDecodeMulti(                   \
        JNIEnv *env, jobject, jlong h, jlong img, jlong points, jlong straight) {      \
        size_t len = 0;                                                               \
        unsigned char *buf = cvk_##C##_detect_and_decode_multi(                       \
            AS(h), as_mat(img), as_mat(points), as_mat(straight), &len);               \
        if (buf == nullptr) return nullptr;                                           \
        jbyteArray arr = env->NewByteArray(static_cast<jsize>(len));                  \
        if (arr != nullptr) {                                                         \
            env->SetByteArrayRegion(arr, 0, static_cast<jsize>(len),                  \
                                    reinterpret_cast<const jbyte *>(buf));            \
        }                                                                             \
        cvk_free_buffer(buf);                                                         \
        return arr;                                                                   \
    }                                                                                 \
    JNIEXPORT void JNICALL                                                            \
    Java_cn_enaium_opencv_JniObjdetect2_##J##Release(JNIEnv *, jobject, jlong h) {    \
        cvk_##C##_release(AS(h));                                                     \
    }

JNI_GCD_FUNCS(qrCodeDetector, qr_code_detector, as_qr_code_detector)
JNI_GCD_FUNCS(qrCodeDetectorAruco, qr_code_detector_aruco, as_qr_code_detector_aruco)
JNI_GCD_FUNCS(barcodeDetector, barcode_detector, as_barcode_detector)

} /* extern "C" */

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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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
 * JNI bridge for the high-level dnn model wrappers (JniDnn2). Thin
 * Java_cn_enaium_opencv_JniDnn2_* wrappers around the cvk_ ABI in
 * native/shim_dnn2.cpp. All model handles are cvk_model_t* travelling as
 * jlong; the Net handle arrives from the DnnCore slice as a cvk_net_t*
 * jlong.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_dnn2.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_model_t *as_model(jlong handle) {
    return reinterpret_cast<cvk_model_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_net_t *as_net(jlong handle) {
    return reinterpret_cast<cvk_net_t *>(static_cast<uintptr_t>(handle));
}

static cvk_scalar_t as_scalar(jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    cvk_scalar_t s;
    s.v0 = v0;
    s.v1 = v1;
    s.v2 = v2;
    s.v3 = v3;
    return s;
}

/** Copies a malloc'd cvk_ buffer into a fresh jbyteArray, then frees it. */
static jbyteArray take_buffer(JNIEnv *env, unsigned char *buffer, size_t length) {
    if (buffer == nullptr) return nullptr;
    jbyteArray out = env->NewByteArray(static_cast<jsize>(length));
    if (out == nullptr) {
        std::free(buffer);
        return nullptr;
    }
    if (length > 0) {
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(length),
                                reinterpret_cast<const jbyte *>(buffer));
    }
    std::free(buffer);
    return out;
}

static unsigned int read_u32le(const unsigned char *p) {
    return static_cast<unsigned int>(p[0]) |
           (static_cast<unsigned int>(p[1]) << 8) |
           (static_cast<unsigned int>(p[2]) << 16) |
           (static_cast<unsigned int>(p[3]) << 24);
}

/** Decodes a [u32 count][per: u32 len + len bytes] wire buffer into Strings. */
static jobjectArray strings_to_java(JNIEnv *env, unsigned char *buffer, size_t length) {
    if (buffer == nullptr) return nullptr;
    if (length < 4) {
        std::free(buffer);
        return nullptr;
    }
    const unsigned int count = read_u32le(buffer);
    jobjectArray result = env->NewObjectArray(static_cast<jsize>(count),
                                              env->FindClass("java/lang/String"),
                                              nullptr);
    if (result == nullptr) {
        std::free(buffer);
        return nullptr;
    }
    size_t off = 4;
    for (unsigned int i = 0; i < count; ++i) {
        if (off + 4 > length) {
            std::free(buffer);
            return nullptr;
        }
        const unsigned int len = read_u32le(buffer + off);
        off += 4;
        if (off + len > length) {
            std::free(buffer);
            return nullptr;
        }
        const std::string s(reinterpret_cast<const char *>(buffer + off), len);
        jstring jstr = env->NewStringUTF(s.c_str());
        if (jstr == nullptr) {
            std::free(buffer);
            return nullptr;
        }
        env->SetObjectArrayElement(result, static_cast<jsize>(i), jstr);
        env->DeleteLocalRef(jstr);
        off += len;
    }
    std::free(buffer);
    return result;
}

/** Copies a Java String array into a std::vector<std::string>. */
static bool copy_string_array(JNIEnv *env, jobjectArray arr, std::vector<std::string> &out) {
    if (arr == nullptr) return false;
    const jsize n = env->GetArrayLength(arr);
    out.reserve(static_cast<size_t>(n));
    for (jsize i = 0; i < n; ++i) {
        jstring s = static_cast<jstring>(env->GetObjectArrayElement(arr, i));
        if (s == nullptr) return false;
        const char *utf = env->GetStringUTFChars(s, nullptr);
        if (utf == nullptr) return false;
        out.emplace_back(utf);
        env->ReleaseStringUTFChars(s, utf);
        env->DeleteLocalRef(s);
    }
    return true;
}

extern "C" {

// ---------------------------------------------------------------- Model

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_modelCreate(JNIEnv *env, jobject, jstring model, jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_model_create(model_utf, config_utf != nullptr ? config_utf : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_modelCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(cvk_model_create_from_net(as_net(net))));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetInputSize(JNIEnv *, jobject, jlong model,
                                                jint width, jint height) {
    cvk_model_set_input_size(as_model(model), width, height);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetInputMean(JNIEnv *, jobject, jlong model,
                                                jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    cvk_model_set_input_mean(as_model(model), as_scalar(v0, v1, v2, v3));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetInputScale(JNIEnv *, jobject, jlong model,
                                                 jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    cvk_model_set_input_scale(as_model(model), as_scalar(v0, v1, v2, v3));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetInputCrop(JNIEnv *, jobject, jlong model, jboolean crop) {
    cvk_model_set_input_crop(as_model(model), crop != JNI_FALSE);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetInputSwapRB(JNIEnv *, jobject, jlong model, jboolean swap) {
    cvk_model_set_input_swap_rb(as_model(model), swap != JNI_FALSE);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetOutputNames(JNIEnv *env, jobject, jlong model,
                                                  jobjectArray names) {
    std::vector<std::string> values;
    if (!copy_string_array(env, names, values)) return;
    std::vector<const char *> ptrs;
    ptrs.reserve(values.size());
    for (const std::string &v : values) ptrs.push_back(v.c_str());
    cvk_model_set_output_names(as_model(model), ptrs.data(), static_cast<int>(ptrs.size()));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetInputParams(JNIEnv *, jobject, jlong model,
                                                  jdouble scale, jint width, jint height,
                                                  jdouble m0, jdouble m1, jdouble m2, jdouble m3,
                                                  jboolean swap_rb, jboolean crop) {
    cvk_model_set_input_params(as_model(model), scale, width, height,
                               as_scalar(m0, m1, m2, m3), swap_rb != JNI_FALSE, crop != JNI_FALSE);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniDnn2_modelPredict(JNIEnv *env, jobject, jlong model, jlong frame) {
    size_t count = 0;
    cvk_mat_t **handles = cvk_model_predict(as_model(model), as_mat(frame), &count);
    if (handles == nullptr) return nullptr;
    jlongArray out = env->NewLongArray(static_cast<jsize>(count));
    if (out == nullptr) {
        for (size_t i = 0; i < count; ++i) cvk_mat_release(handles[i]);
        std::free(handles);
        return nullptr;
    }
    std::vector<jlong> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = as_handle(handles[i]);
    env->SetLongArrayRegion(out, 0, static_cast<jsize>(count), values.data());
    // The Mat handles themselves transfer to the JVM; only the array is ours.
    std::free(handles);
    return out;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetPreferableBackend(JNIEnv *, jobject, jlong model,
                                                        jint backend) {
    cvk_model_set_preferable_backend(as_model(model), backend);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelSetPreferableTarget(JNIEnv *, jobject, jlong model,
                                                       jint target) {
    cvk_model_set_preferable_target(as_model(model), target);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelEnableWinograd(JNIEnv *, jobject, jlong model,
                                                  jboolean use) {
    cvk_model_enable_winograd(as_model(model), use != JNI_FALSE);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_modelRelease(JNIEnv *, jobject, jlong model) {
    cvk_model_release(as_model(model));
}

// ------------------------------------------------ ClassificationModel

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_classificationModelCreate(JNIEnv *env, jobject, jstring model,
                                                        jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_classification_model_create(model_utf,
                                                          config_utf != nullptr ? config_utf : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_classificationModelCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_classification_model_create_from_net(as_net(net))));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_classificationModelSetEnableSoftmaxPostProcessing(
    JNIEnv *, jobject, jlong model, jboolean enable) {
    cvk_classification_model_set_enable_softmax_post_processing(as_model(model),
                                                                enable != JNI_FALSE);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniDnn2_classificationModelGetEnableSoftmaxPostProcessing(
    JNIEnv *, jobject, jlong model) {
    return cvk_classification_model_get_enable_softmax_post_processing(as_model(model))
               ? JNI_TRUE
               : JNI_FALSE;
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniDnn2_classificationModelClassify(JNIEnv *env, jobject, jlong model,
                                                          jlong frame) {
    int class_id = 0;
    float confidence = 0.0f;
    if (cvk_classification_model_classify(as_model(model), as_mat(frame),
                                          &class_id, &confidence) == 0) {
        return nullptr;
    }
    const jdouble out[2] = {static_cast<jdouble>(class_id), static_cast<jdouble>(confidence)};
    jdoubleArray result = env->NewDoubleArray(2);
    if (result != nullptr) env->SetDoubleArrayRegion(result, 0, 2, out);
    return result;
}

// ---------------------------------------------------- DetectionModel

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_detectionModelCreate(JNIEnv *env, jobject, jstring model,
                                                   jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_detection_model_create(model_utf,
                                                     config_utf != nullptr ? config_utf : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_detectionModelCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_detection_model_create_from_net(as_net(net))));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_detectionModelSetNmsAcrossClasses(JNIEnv *, jobject, jlong model,
                                                                jboolean value) {
    cvk_detection_model_set_nms_across_classes(as_model(model), value != JNI_FALSE);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniDnn2_detectionModelGetNmsAcrossClasses(JNIEnv *, jobject, jlong model) {
    return cvk_detection_model_get_nms_across_classes(as_model(model)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn2_detectionModelDetect(JNIEnv *env, jobject, jlong model, jlong frame,
                                                   jfloat conf_threshold, jfloat nms_threshold) {
    size_t length = 0;
    unsigned char *buffer = cvk_detection_model_detect(as_model(model), as_mat(frame),
                                                       conf_threshold, nms_threshold, &length);
    return take_buffer(env, buffer, length);
}

// ---------------------------------------------------- KeypointsModel

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_keypointsModelCreate(JNIEnv *env, jobject, jstring model,
                                                   jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_keypoints_model_create(model_utf,
                                                     config_utf != nullptr ? config_utf : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_keypointsModelCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_keypoints_model_create_from_net(as_net(net))));
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn2_keypointsModelEstimate(JNIEnv *env, jobject, jlong model,
                                                     jlong frame, jfloat thresh) {
    size_t length = 0;
    unsigned char *buffer = cvk_keypoints_model_estimate(as_model(model), as_mat(frame),
                                                         thresh, &length);
    return take_buffer(env, buffer, length);
}

// -------------------------------------------------- SegmentationModel

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_segmentationModelCreate(JNIEnv *env, jobject, jstring model,
                                                      jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_segmentation_model_create(model_utf,
                                                        config_utf != nullptr ? config_utf : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_segmentationModelCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_segmentation_model_create_from_net(as_net(net))));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_segmentationModelSegment(JNIEnv *, jobject, jlong model,
                                                       jlong frame) {
    return as_handle(cvk_segmentation_model_segment(as_model(model), as_mat(frame)));
}

// --------------------------------------------- TextDetectionModel base

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDetect(JNIEnv *env, jobject, jlong model,
                                                       jlong frame) {
    size_t length = 0;
    unsigned char *buffer = cvk_text_detection_model_detect(as_model(model), as_mat(frame),
                                                            &length);
    return take_buffer(env, buffer, length);
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDetectTextRectangles(JNIEnv *env, jobject,
                                                                     jlong model, jlong frame) {
    size_t length = 0;
    unsigned char *buffer = cvk_text_detection_model_detect_text_rectangles(as_model(model),
                                                                            as_mat(frame),
                                                                            &length);
    return take_buffer(env, buffer, length);
}

// ------------------------------------------------ TextDetectionModel_DB

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbCreate(JNIEnv *env, jobject, jstring model,
                                                         jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_text_detection_model_db_create(model_utf,
                                                             config_utf != nullptr ? config_utf
                                                                                   : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_text_detection_model_db_create_from_net(as_net(net))));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbSetBinaryThreshold(JNIEnv *, jobject,
                                                                     jlong model, jfloat value) {
    cvk_text_detection_model_db_set_binary_threshold(as_model(model), value);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbGetBinaryThreshold(JNIEnv *, jobject,
                                                                     jlong model) {
    return cvk_text_detection_model_db_get_binary_threshold(as_model(model));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbSetPolygonThreshold(JNIEnv *, jobject,
                                                                      jlong model, jfloat value) {
    cvk_text_detection_model_db_set_polygon_threshold(as_model(model), value);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbGetPolygonThreshold(JNIEnv *, jobject,
                                                                      jlong model) {
    return cvk_text_detection_model_db_get_polygon_threshold(as_model(model));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbSetUnclipRatio(JNIEnv *, jobject, jlong model,
                                                                 jdouble value) {
    cvk_text_detection_model_db_set_unclip_ratio(as_model(model), value);
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbGetUnclipRatio(JNIEnv *, jobject, jlong model) {
    return cvk_text_detection_model_db_get_unclip_ratio(as_model(model));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbSetMaxCandidates(JNIEnv *, jobject, jlong model,
                                                                   jint value) {
    cvk_text_detection_model_db_set_max_candidates(as_model(model), value);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelDbGetMaxCandidates(JNIEnv *, jobject,
                                                                   jlong model) {
    return cvk_text_detection_model_db_get_max_candidates(as_model(model));
}

// ---------------------------------------------- TextDetectionModel_EAST

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelEastCreate(JNIEnv *env, jobject, jstring model,
                                                           jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_text_detection_model_east_create(model_utf,
                                                               config_utf != nullptr ? config_utf
                                                                                     : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelEastCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_text_detection_model_east_create_from_net(as_net(net))));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelEastSetConfidenceThreshold(JNIEnv *, jobject,
                                                                           jlong model,
                                                                           jfloat value) {
    cvk_text_detection_model_east_set_confidence_threshold(as_model(model), value);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelEastGetConfidenceThreshold(JNIEnv *, jobject,
                                                                           jlong model) {
    return cvk_text_detection_model_east_get_confidence_threshold(as_model(model));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelEastSetNmsThreshold(JNIEnv *, jobject, jlong model,
                                                                    jfloat value) {
    cvk_text_detection_model_east_set_nms_threshold(as_model(model), value);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniDnn2_textDetectionModelEastGetNmsThreshold(JNIEnv *, jobject,
                                                                    jlong model) {
    return cvk_text_detection_model_east_get_nms_threshold(as_model(model));
}

// ---------------------------------------------- TextRecognitionModel

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelCreate(JNIEnv *env, jobject, jstring model,
                                                         jstring config) {
    if (model == nullptr) return 0;
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    if (model_utf == nullptr) return 0;
    const char *config_utf = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_model_t *result = cvk_text_recognition_model_create(model_utf,
                                                            config_utf != nullptr ? config_utf
                                                                                  : "");
    if (config_utf != nullptr) env->ReleaseStringUTFChars(config, config_utf);
    env->ReleaseStringUTFChars(model, model_utf);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(result));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelCreateFromNet(JNIEnv *, jobject, jlong net) {
    return static_cast<jlong>(
        reinterpret_cast<uintptr_t>(cvk_text_recognition_model_create_from_net(as_net(net))));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelSetDecodeType(JNIEnv *env, jobject,
                                                                jlong model, jstring decode_type) {
    if (decode_type == nullptr) return;
    const char *utf = env->GetStringUTFChars(decode_type, nullptr);
    if (utf == nullptr) return;
    cvk_text_recognition_model_set_decode_type(as_model(model), utf);
    env->ReleaseStringUTFChars(decode_type, utf);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelGetDecodeType(JNIEnv *env, jobject,
                                                                jlong model) {
    const char *text = cvk_text_recognition_model_get_decode_type(as_model(model));
    return text != nullptr ? env->NewStringUTF(text) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelSetDecodeOptsCtcPrefixBeamSearch(
    JNIEnv *, jobject, jlong model, jint beam_size, jint voc_prune_size) {
    cvk_text_recognition_model_set_decode_opts_ctc_prefix_beam_search(as_model(model),
                                                                      beam_size, voc_prune_size);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelSetVocabulary(JNIEnv *env, jobject, jlong model,
                                                                jobjectArray vocabulary) {
    std::vector<std::string> values;
    if (!copy_string_array(env, vocabulary, values)) return;
    std::vector<const char *> ptrs;
    ptrs.reserve(values.size());
    for (const std::string &v : values) ptrs.push_back(v.c_str());
    cvk_text_recognition_model_set_vocabulary(as_model(model), ptrs.data(),
                                              static_cast<int>(ptrs.size()));
}

JNIEXPORT jobjectArray JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelGetVocabulary(JNIEnv *env, jobject,
                                                                jlong model) {
    size_t length = 0;
    unsigned char *buffer = cvk_text_recognition_model_get_vocabulary(as_model(model), &length);
    return strings_to_java(env, buffer, length);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelRecognize(JNIEnv *env, jobject, jlong model,
                                                            jlong frame) {
    const char *text = cvk_text_recognition_model_recognize(as_model(model), as_mat(frame));
    return text != nullptr ? env->NewStringUTF(text) : nullptr;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn2_textRecognitionModelRecognizeRois(JNIEnv *env, jobject, jlong model,
                                                                jlong frame, jintArray rois) {
    jint *elements = rois != nullptr ? env->GetIntArrayElements(rois, nullptr) : nullptr;
    const jsize length = rois != nullptr ? env->GetArrayLength(rois) : 0;
    if (rois != nullptr && elements == nullptr) return nullptr;
    size_t out_len = 0;
    unsigned char *buffer = cvk_text_recognition_model_recognize_rois(
        as_model(model), as_mat(frame), elements, static_cast<int>(length / 4), &out_len);
    if (rois != nullptr) env->ReleaseIntArrayElements(rois, elements, JNI_ABORT);
    return take_buffer(env, buffer, out_len);
}

} /* extern "C" */

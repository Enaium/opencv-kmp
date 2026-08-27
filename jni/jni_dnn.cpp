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
 * JNI bridge for the org.opencv.dnn surface: Java_cn_enaium_opencv_JniDnn_*
 * wrappers around the cvk_ C ABI in native/shim_dnn.cpp. Handles arrive as
 * jlong pointers; the shim is noexcept and reports failures via
 * cvk_last_error() (surfaced by the Kotlin layer as OpenCVException).
 *
 * String arguments are copied into the shim by the call (cv::String), so
 * GetStringUTFChars/ReleaseStringUTFChars pairs are scoped per call.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_dnn.h"

#include <cstdint>
#include <cstring>

static inline cvk_mat_t *as_mat(jlong handle) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_handle(const cvk_mat_t *mat) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(mat));
}

static inline cvk_net_t *as_net(jlong handle) {
    return reinterpret_cast<cvk_net_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_layer_t *as_layer(jlong handle) {
    return reinterpret_cast<cvk_layer_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_tokenizer_t *as_tokenizer(jlong handle) {
    return reinterpret_cast<cvk_tokenizer_t *>(static_cast<uintptr_t>(handle));
}

static inline cvk_scalar_t as_scalar(jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    cvk_scalar_t s;
    s.v0 = v0;
    s.v1 = v1;
    s.v2 = v2;
    s.v3 = v3;
    return s;
}

static jbyteArray flat_to_java(JNIEnv *env, unsigned char *flat, size_t len) {
    if (flat == nullptr) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(len));
    if (arr != nullptr) {
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(len),
                                reinterpret_cast<const jbyte *>(flat));
    }
    cvk_free_buffer(flat);
    return arr;
}

static jstring str_or_null(JNIEnv *env, const char *text) {
    return text != nullptr ? env->NewStringUTF(text) : nullptr;
}

extern "C" {

// ---------------------------------------------------------------- dnn statics

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNet(JNIEnv *env, jobject, jstring model, jstring config,
                                        jstring framework, jint engine) {
    const char *model_s = model != nullptr ? env->GetStringUTFChars(model, nullptr) : nullptr;
    const char *config_s = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    const char *framework_s = framework != nullptr ? env->GetStringUTFChars(framework, nullptr) : nullptr;
    cvk_net_t *net = cvk_dnn_read_net(model_s, config_s, framework_s, engine);
    if (model_s != nullptr) env->ReleaseStringUTFChars(model, model_s);
    if (config_s != nullptr) env->ReleaseStringUTFChars(config, config_s);
    if (framework_s != nullptr) env->ReleaseStringUTFChars(framework, framework_s);
    return reinterpret_cast<jlong>(net);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetBuffer(JNIEnv *env, jobject, jstring framework,
                                              jlong buffer_model, jlong buffer_config,
                                              jint engine) {
    const char *framework_s = framework != nullptr ? env->GetStringUTFChars(framework, nullptr) : nullptr;
    cvk_net_t *net = cvk_dnn_read_net_buffer(framework_s, as_mat(buffer_model),
                                             buffer_config != 0 ? as_mat(buffer_config) : nullptr,
                                             engine);
    if (framework_s != nullptr) env->ReleaseStringUTFChars(framework, framework_s);
    return reinterpret_cast<jlong>(net);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromOnnx(JNIEnv *env, jobject, jstring onnx_file,
                                                jint engine) {
    const char *path = env->GetStringUTFChars(onnx_file, nullptr);
    cvk_net_t *net = cvk_dnn_read_net_from_onnx(path, engine);
    env->ReleaseStringUTFChars(onnx_file, path);
    return reinterpret_cast<jlong>(net);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromOnnxBuffer(JNIEnv *, jobject, jlong buffer,
                                                      jint engine) {
    return reinterpret_cast<jlong>(cvk_dnn_read_net_from_onnx_buffer(as_mat(buffer), engine));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromTensorflow(JNIEnv *env, jobject, jstring model,
                                                      jstring config, jint engine) {
    const char *model_s = env->GetStringUTFChars(model, nullptr);
    const char *config_s = config != nullptr ? env->GetStringUTFChars(config, nullptr) : nullptr;
    cvk_net_t *net = cvk_dnn_read_net_from_tensorflow(model_s, config_s, engine);
    env->ReleaseStringUTFChars(model, model_s);
    if (config_s != nullptr) env->ReleaseStringUTFChars(config, config_s);
    return reinterpret_cast<jlong>(net);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromTensorflowBuffer(JNIEnv *, jobject, jlong buffer_model,
                                                            jlong buffer_config, jint engine) {
    return reinterpret_cast<jlong>(cvk_dnn_read_net_from_tensorflow_buffer(
        as_mat(buffer_model), buffer_config != 0 ? as_mat(buffer_config) : nullptr, engine));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromTflite(JNIEnv *env, jobject, jstring model,
                                                  jint engine) {
    const char *model_s = env->GetStringUTFChars(model, nullptr);
    cvk_net_t *net = cvk_dnn_read_net_from_tflite(model_s, engine);
    env->ReleaseStringUTFChars(model, model_s);
    return reinterpret_cast<jlong>(net);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromTfliteBuffer(JNIEnv *, jobject, jlong buffer_model,
                                                        jint engine) {
    return reinterpret_cast<jlong>(cvk_dnn_read_net_from_tflite_buffer(as_mat(buffer_model), engine));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromModelOptimizer(JNIEnv *env, jobject, jstring xml,
                                                          jstring bin) {
    const char *xml_s = env->GetStringUTFChars(xml, nullptr);
    const char *bin_s = bin != nullptr ? env->GetStringUTFChars(bin, nullptr) : nullptr;
    cvk_net_t *net = cvk_dnn_read_net_from_model_optimizer(xml_s, bin_s);
    env->ReleaseStringUTFChars(xml, xml_s);
    if (bin_s != nullptr) env->ReleaseStringUTFChars(bin, bin_s);
    return reinterpret_cast<jlong>(net);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadNetFromModelOptimizerBuffer(JNIEnv *, jobject,
                                                                jlong buffer_config,
                                                                jlong buffer_weights) {
    return reinterpret_cast<jlong>(cvk_dnn_read_net_from_model_optimizer_buffer(
        as_mat(buffer_config), as_mat(buffer_weights)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnReadTensorFromOnnx(JNIEnv *env, jobject, jstring path) {
    const char *path_s = env->GetStringUTFChars(path, nullptr);
    cvk_mat_t *mat = cvk_dnn_read_tensor_from_onnx(path_s);
    env->ReleaseStringUTFChars(path, path_s);
    return as_handle(mat);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnBlobFromImage(JNIEnv *, jobject, jlong image,
                                              jdouble scalefactor, jint size_w, jint size_h,
                                              jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                              jboolean swap_rb, jboolean crop, jint ddepth) {
    return as_handle(cvk_dnn_blob_from_image(as_mat(image), scalefactor, size_w, size_h,
                                             as_scalar(v0, v1, v2, v3), swap_rb ? 1 : 0,
                                             crop ? 1 : 0, ddepth));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnBlobFromImages(JNIEnv *, jobject, jlong images,
                                               jdouble scalefactor, jint size_w, jint size_h,
                                               jdouble v0, jdouble v1, jdouble v2, jdouble v3,
                                               jboolean swap_rb, jboolean crop, jint ddepth) {
    return as_handle(cvk_dnn_blob_from_images(as_mat(images), scalefactor, size_w, size_h,
                                              as_scalar(v0, v1, v2, v3), swap_rb ? 1 : 0,
                                              crop ? 1 : 0, ddepth));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnImagesFromBlob(JNIEnv *, jobject, jlong blob) {
    return as_handle(cvk_dnn_images_from_blob(as_mat(blob)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_dnnNmsBoxes(JNIEnv *, jobject, jlong bboxes, jlong scores,
                                         jfloat score_threshold, jfloat nms_threshold,
                                         jlong indices, jfloat eta, jint top_k) {
    cvk_dnn_nms_boxes(as_mat(bboxes), as_mat(scores), score_threshold, nms_threshold,
                      as_mat(indices), eta, top_k);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_dnnNmsBoxesRotated(JNIEnv *, jobject, jlong bboxes, jlong scores,
                                                jfloat score_threshold, jfloat nms_threshold,
                                                jlong indices, jfloat eta, jint top_k) {
    cvk_dnn_nms_boxes_rotated(as_mat(bboxes), as_mat(scores), score_threshold, nms_threshold,
                              as_mat(indices), eta, top_k);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_dnnNmsBoxesBatched(JNIEnv *, jobject, jlong bboxes, jlong scores,
                                                jlong class_ids, jfloat score_threshold,
                                                jfloat nms_threshold, jlong indices,
                                                jfloat eta, jint top_k) {
    cvk_dnn_nms_boxes_batched(as_mat(bboxes), as_mat(scores), as_mat(class_ids),
                              score_threshold, nms_threshold, as_mat(indices), eta, top_k);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_dnnSoftNmsBoxes(JNIEnv *, jobject, jlong bboxes, jlong scores,
                                             jlong updated_scores, jfloat score_threshold,
                                             jfloat nms_threshold, jlong indices, jlong top_k,
                                             jfloat sigma, jint method) {
    cvk_dnn_soft_nms_boxes(as_mat(bboxes), as_mat(scores), as_mat(updated_scores),
                           score_threshold, nms_threshold, as_mat(indices),
                           static_cast<int>(top_k), sigma, method);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnGetAvailableTargets(JNIEnv *, jobject, jint backend) {
    return as_handle(cvk_dnn_get_available_targets(backend));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_dnnGetAvailableBackends(JNIEnv *, jobject) {
    return as_handle(cvk_dnn_get_available_backends());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_dnnWriteTextGraph(JNIEnv *env, jobject, jstring model,
                                               jstring output) {
    const char *model_s = env->GetStringUTFChars(model, nullptr);
    const char *output_s = env->GetStringUTFChars(output, nullptr);
    cvk_dnn_write_text_graph(model_s, output_s);
    env->ReleaseStringUTFChars(model, model_s);
    env->ReleaseStringUTFChars(output, output_s);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netCreate(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(cvk_net_create());
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniDnn_netEmpty(JNIEnv *, jobject, jlong net) {
    return cvk_net_empty(as_net(net)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniDnn_netDump(JNIEnv *env, jobject, jlong net) {
    return str_or_null(env, cvk_net_dump(as_net(net)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netDumpToFile(JNIEnv *env, jobject, jlong net, jstring path) {
    const char *path_s = env->GetStringUTFChars(path, nullptr);
    cvk_net_dump_to_file(as_net(net), path_s);
    env->ReleaseStringUTFChars(path, path_s);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netDumpToPbtxt(JNIEnv *env, jobject, jlong net, jstring path) {
    const char *path_s = env->GetStringUTFChars(path, nullptr);
    cvk_net_dump_to_pbtxt(as_net(net), path_s);
    env->ReleaseStringUTFChars(path, path_s);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniDnn_netGetLayerId(JNIEnv *env, jobject, jlong net, jstring layer) {
    const char *layer_s = env->GetStringUTFChars(layer, nullptr);
    jint id = cvk_net_get_layer_id(as_net(net), layer_s);
    env->ReleaseStringUTFChars(layer, layer_s);
    return id;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn_netGetLayerNames(JNIEnv *env, jobject, jlong net) {
    size_t len = 0;
    unsigned char *flat = cvk_net_get_layer_names(as_net(net), &len);
    return flat_to_java(env, flat, len);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netGetLayer(JNIEnv *, jobject, jlong net, jint layer_id) {
    return reinterpret_cast<jlong>(cvk_net_get_layer(as_net(net), layer_id));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netGetLayerByName(JNIEnv *env, jobject, jlong net,
                                               jstring layer_name) {
    const char *name_s = env->GetStringUTFChars(layer_name, nullptr);
    cvk_layer_t *layer = cvk_net_get_layer_by_name(as_net(net), name_s);
    env->ReleaseStringUTFChars(layer_name, name_s);
    return reinterpret_cast<jlong>(layer);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netConnect(JNIEnv *env, jobject, jlong net, jstring out_pin,
                                        jstring inp_pin) {
    const char *out_s = env->GetStringUTFChars(out_pin, nullptr);
    const char *inp_s = env->GetStringUTFChars(inp_pin, nullptr);
    cvk_net_connect(as_net(net), out_s, inp_s);
    env->ReleaseStringUTFChars(out_pin, out_s);
    env->ReleaseStringUTFChars(inp_pin, inp_s);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniDnn_netRegisterOutput(JNIEnv *env, jobject, jlong net,
                                               jstring output_name, jint layer_id,
                                               jint output_port) {
    const char *name_s = env->GetStringUTFChars(output_name, nullptr);
    jint index = cvk_net_register_output(as_net(net), name_s, layer_id, output_port);
    env->ReleaseStringUTFChars(output_name, name_s);
    return index;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netSetInputsNames(JNIEnv *env, jobject, jlong net,
                                               jbyteArray names) {
    jsize len = env->GetArrayLength(names);
    jbyte *flat = env->GetByteArrayElements(names, nullptr);
    cvk_net_set_inputs_names(as_net(net), reinterpret_cast<unsigned char *>(flat),
                             static_cast<size_t>(len));
    env->ReleaseByteArrayElements(names, flat, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netSetInputShape(JNIEnv *env, jobject, jlong net,
                                              jstring input_name, jlong shape) {
    const char *name_s = env->GetStringUTFChars(input_name, nullptr);
    cvk_net_set_input_shape(as_net(net), name_s, as_mat(shape));
    env->ReleaseStringUTFChars(input_name, name_s);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netForward(JNIEnv *env, jobject, jlong net, jstring output_name) {
    const char *name_s = output_name != nullptr ? env->GetStringUTFChars(output_name, nullptr) : nullptr;
    cvk_mat_t *mat = cvk_net_forward(as_net(net), name_s);
    if (name_s != nullptr) env->ReleaseStringUTFChars(output_name, name_s);
    return as_handle(mat);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netForwardLayer(JNIEnv *env, jobject, jlong net,
                                             jstring output_name) {
    const char *name_s = output_name != nullptr ? env->GetStringUTFChars(output_name, nullptr) : nullptr;
    cvk_mat_t *mat = cvk_net_forward_layer(as_net(net), name_s);
    if (name_s != nullptr) env->ReleaseStringUTFChars(output_name, name_s);
    return as_handle(mat);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netForwardNames(JNIEnv *env, jobject, jlong net,
                                             jbyteArray names) {
    jsize len = env->GetArrayLength(names);
    jbyte *flat = env->GetByteArrayElements(names, nullptr);
    cvk_mat_t *mat = cvk_net_forward_names(as_net(net),
                                           reinterpret_cast<unsigned char *>(flat),
                                           static_cast<size_t>(len));
    env->ReleaseByteArrayElements(names, flat, JNI_ABORT);
    return as_handle(mat);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netForwardAndRetrieve(JNIEnv *env, jobject, jlong net,
                                                   jbyteArray names) {
    jsize len = env->GetArrayLength(names);
    jbyte *flat = env->GetByteArrayElements(names, nullptr);
    cvk_mat_t *mat = cvk_net_forward_and_retrieve(as_net(net),
                                                  reinterpret_cast<unsigned char *>(flat),
                                                  static_cast<size_t>(len));
    env->ReleaseByteArrayElements(names, flat, JNI_ABORT);
    return as_handle(mat);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netSetPreferableBackend(JNIEnv *, jobject, jlong net,
                                                     jint backend_id) {
    cvk_net_set_preferable_backend(as_net(net), backend_id);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netSetPreferableTarget(JNIEnv *, jobject, jlong net,
                                                    jint target_id) {
    cvk_net_set_preferable_target(as_net(net), target_id);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netFinalize(JNIEnv *, jobject, jlong net) {
    cvk_net_finalize(as_net(net));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netSetInput(JNIEnv *env, jobject, jlong net, jlong blob,
                                         jstring name, jdouble scalefactor,
                                         jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    const char *name_s = name != nullptr ? env->GetStringUTFChars(name, nullptr) : nullptr;
    cvk_net_set_input(as_net(net), as_mat(blob), name_s, scalefactor,
                      as_scalar(v0, v1, v2, v3));
    if (name_s != nullptr) env->ReleaseStringUTFChars(name, name_s);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netSetParam(JNIEnv *, jobject, jlong net, jint layer,
                                         jint num_param, jlong blob) {
    cvk_net_set_param(as_net(net), layer, num_param, as_mat(blob));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netSetParamByName(JNIEnv *env, jobject, jlong net,
                                               jstring layer_name, jint num_param, jlong blob) {
    const char *name_s = env->GetStringUTFChars(layer_name, nullptr);
    cvk_net_set_param_by_name(as_net(net), name_s, num_param, as_mat(blob));
    env->ReleaseStringUTFChars(layer_name, name_s);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netGetParam(JNIEnv *, jobject, jlong net, jint layer,
                                         jint num_param) {
    return as_handle(cvk_net_get_param(as_net(net), layer, num_param));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netGetParamByName(JNIEnv *env, jobject, jlong net,
                                               jstring layer_name, jint num_param) {
    const char *name_s = env->GetStringUTFChars(layer_name, nullptr);
    cvk_mat_t *mat = cvk_net_get_param_by_name(as_net(net), name_s, num_param);
    env->ReleaseStringUTFChars(layer_name, name_s);
    return as_handle(mat);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netGetUnconnectedOutLayers(JNIEnv *, jobject, jlong net) {
    return as_handle(cvk_net_get_unconnected_out_layers(as_net(net)));
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn_netGetUnconnectedOutLayersNames(JNIEnv *env, jobject, jlong net) {
    size_t len = 0;
    unsigned char *flat = cvk_net_get_unconnected_out_layers_names(as_net(net), &len);
    return flat_to_java(env, flat, len);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netGetFlops(JNIEnv *, jobject, jlong net, jlong input_shapes,
                                         jlong input_types) {
    return static_cast<jlong>(cvk_net_get_flops(as_net(net), as_mat(input_shapes),
                                                as_mat(input_types)));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniDnn_netGetMemoryConsumption(JNIEnv *env, jobject, jlong net,
                                                     jlong input_shapes, jlong input_types) {
    unsigned long long out[2] = {0, 0};
    cvk_net_get_memory_consumption(as_net(net), as_mat(input_shapes), as_mat(input_types), out);
    jlongArray arr = env->NewLongArray(2);
    if (arr != nullptr) {
        jlong values[2] = {static_cast<jlong>(out[0]), static_cast<jlong>(out[1])};
        env->SetLongArrayRegion(arr, 0, 2, values);
    }
    return arr;
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn_netGetLayerTypes(JNIEnv *env, jobject, jlong net) {
    size_t len = 0;
    unsigned char *flat = nullptr;
    cvk_net_get_layer_types(as_net(net), &flat, &len);
    return flat_to_java(env, flat, len);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniDnn_netGetLayersCount(JNIEnv *env, jobject, jlong net,
                                               jstring layer_type) {
    const char *type_s = env->GetStringUTFChars(layer_type, nullptr);
    jint count = cvk_net_get_layers_count(as_net(net), type_s);
    env->ReleaseStringUTFChars(layer_type, type_s);
    return count;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netEnableFusion(JNIEnv *, jobject, jlong net, jboolean fusion) {
    cvk_net_enable_fusion(as_net(net), fusion ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netEnableWinograd(JNIEnv *, jobject, jlong net,
                                               jboolean use_winograd) {
    cvk_net_enable_winograd(as_net(net), use_winograd ? 1 : 0);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_netGetPerfProfile(JNIEnv *, jobject, jlong net, jlong timings) {
    return static_cast<jlong>(cvk_net_get_perf_profile(as_net(net), as_mat(timings)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netEnableKvCache(JNIEnv *, jobject, jlong net) {
    cvk_net_enable_kv_cache(as_net(net));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netDisableKvCache(JNIEnv *, jobject, jlong net) {
    cvk_net_disable_kv_cache(as_net(net));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netResetKvCache(JNIEnv *, jobject, jlong net) {
    cvk_net_reset_kv_cache(as_net(net));
}

JNIEXPORT jbyteArray JNICALL
Java_cn_enaium_opencv_JniDnn_netGetPerfProfileNames(JNIEnv *env, jobject, jlong net) {
    size_t len = 0;
    unsigned char *flat = nullptr;
    cvk_net_get_perf_profile_names(as_net(net), &flat, &len);
    return flat_to_java(env, flat, len);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netPrintPerfProfile(JNIEnv *, jobject, jlong net) {
    cvk_net_print_perf_profile(as_net(net));
}

// ------------------------------------------------------------------ Layer

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_layerRelease(JNIEnv *, jobject, jlong layer) {
    cvk_layer_release(as_layer(layer));
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniDnn_layerName(JNIEnv *env, jobject, jlong layer) {
    return str_or_null(env, cvk_layer_name(as_layer(layer)));
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniDnn_layerType(JNIEnv *env, jobject, jlong layer) {
    return str_or_null(env, cvk_layer_type(as_layer(layer)));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniDnn_layerPreferableTarget(JNIEnv *, jobject, jlong layer) {
    return cvk_layer_preferable_target(as_layer(layer));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniDnn_layerOutputNameToIndex(JNIEnv *env, jobject, jlong layer,
                                                    jstring output_name) {
    const char *name_s = env->GetStringUTFChars(output_name, nullptr);
    jint index = cvk_layer_output_name_to_index(as_layer(layer), name_s);
    env->ReleaseStringUTFChars(output_name, name_s);
    return index;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_layerGetBlobs(JNIEnv *, jobject, jlong layer) {
    return as_handle(cvk_layer_get_blobs(as_layer(layer)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_layerSetBlobs(JNIEnv *, jobject, jlong layer, jlong blobs) {
    cvk_layer_set_blobs(as_layer(layer), as_mat(blobs));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_layerFinalize(JNIEnv *, jobject, jlong layer, jlong inputs,
                                           jlong outputs) {
    cvk_layer_finalize(as_layer(layer), as_mat(inputs), as_mat(outputs));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_layerRun(JNIEnv *, jobject, jlong layer, jlong inputs,
                                      jlong outputs, jlong internals) {
    cvk_layer_run(as_layer(layer), as_mat(inputs), as_mat(outputs), as_mat(internals));
}

// --------------------------------------------------------------- Tokenizer

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_tokenizerLoad(JNIEnv *env, jobject, jstring model_config) {
    const char *path_s = env->GetStringUTFChars(model_config, nullptr);
    cvk_tokenizer_t *tok = cvk_tokenizer_load(path_s);
    env->ReleaseStringUTFChars(model_config, path_s);
    return reinterpret_cast<jlong>(tok);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_tokenizerRelease(JNIEnv *, jobject, jlong tokenizer) {
    cvk_tokenizer_release(as_tokenizer(tokenizer));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniDnn_tokenizerEncode(JNIEnv *env, jobject, jlong tokenizer,
                                             jstring text) {
    const char *text_s = env->GetStringUTFChars(text, nullptr);
    cvk_mat_t *mat = cvk_tokenizer_encode(as_tokenizer(tokenizer), text_s);
    env->ReleaseStringUTFChars(text, text_s);
    return as_handle(mat);
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniDnn_tokenizerDecode(JNIEnv *env, jobject, jlong tokenizer,
                                             jlong tokens) {
    return str_or_null(env, cvk_tokenizer_decode(as_tokenizer(tokenizer), as_mat(tokens)));
}

// ------------------------------------------------- dnn Mat introspection

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniDnn_dnnMatDims(JNIEnv *, jobject, jlong mat) {
    return cvk_dnn_mat_dims(as_mat(mat));
}

JNIEXPORT jintArray JNICALL
Java_cn_enaium_opencv_JniDnn_dnnMatShape(JNIEnv *env, jobject, jlong mat) {
    const int dims = cvk_dnn_mat_dims(as_mat(mat));
    if (dims <= 0) return nullptr;
    jintArray arr = env->NewIntArray(dims);
    if (arr == nullptr) return nullptr;
    int shape[32] = {0};
    const int reported = cvk_dnn_mat_shape(as_mat(mat), shape, dims);
    (void) reported;
    env->SetIntArrayRegion(arr, 0, dims, shape);
    return arr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniDnn_netRelease(JNIEnv *, jobject, jlong net) {
    cvk_net_release(reinterpret_cast<cvk_net_t *>(static_cast<uintptr_t>(net)));
}

} /* extern "C" */

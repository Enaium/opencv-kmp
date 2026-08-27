/*
 * cvk_ C ABI declarations for the OpenCV "dnn" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * The org.opencv.dnn surface: Dnn statics, Net, Layer, Tokenizer.
 * DictValue is a pure-Kotlin value type (no native handle), so it has no
 * declarations here.
 *
 * Wire formats used by this slice (matching the official Java bindings):
 *  - vector<Mat> travels as a CV_32SC2 Mat whose rows are the 64-bit heap
 *    addresses of the member Mats ("Mat of Mat pointers"). Member Mats are
 *    owned by the caller and released with cvk_mat_release; the container
 *    Mat is a plain cvk_mat_t released the same way.
 *  - vector<vector<Mat>> is the same encoding applied twice (the outer
 *    addresses point at inner CV_32SC2 Mats).
 *  - std::vector<std::string> travels as a malloc'd flat buffer:
 *      [u32le count][per string: u32le byte_len + UTF-8 bytes]
 *    released with cvk_free_buffer.
 *  - MatShape (std::vector<int>) travels as a CV_32S Mat; lists of shapes
 *    as a CV_32SC2 Mat of addresses of such Mats.
 *  - NMS outputs mutate the caller's CV_32S indices Mat in place.
 */
#ifndef OPENCV_KMP_DNN_H
#define OPENCV_KMP_DNN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvk_net cvk_net_t;
typedef struct cvk_layer cvk_layer_t;
typedef struct cvk_tokenizer cvk_tokenizer_t;

/* =========================================================================
 * dnn statics
 * ========================================================================= */

/**
 * cv::dnn::readNet(model, config, framework, engine). All model-file readers
 * are guarded: a missing/unreadable file reports via cvk_last_error and
 * returns NULL.
 */
cvk_net_t *cvk_dnn_read_net(const char *model, const char *config,
                            const char *framework, int engine);

/** readNet(framework, bufferModel, bufferConfig, engine); buffers are CV_8UC1 Mats. */
cvk_net_t *cvk_dnn_read_net_buffer(const char *framework,
                                   const cvk_mat_t *buffer_model,
                                   const cvk_mat_t *buffer_config,
                                   int engine);

cvk_net_t *cvk_dnn_read_net_from_onnx(const char *onnx_file, int engine);
cvk_net_t *cvk_dnn_read_net_from_onnx_buffer(const cvk_mat_t *buffer, int engine);
cvk_net_t *cvk_dnn_read_net_from_tensorflow(const char *model, const char *config,
                                            int engine);
cvk_net_t *cvk_dnn_read_net_from_tensorflow_buffer(const cvk_mat_t *buffer_model,
                                                   const cvk_mat_t *buffer_config,
                                                   int engine);
cvk_net_t *cvk_dnn_read_net_from_tflite(const char *model, int engine);
cvk_net_t *cvk_dnn_read_net_from_tflite_buffer(const cvk_mat_t *buffer_model, int engine);
cvk_net_t *cvk_dnn_read_net_from_model_optimizer(const char *xml, const char *bin);
cvk_net_t *cvk_dnn_read_net_from_model_optimizer_buffer(const cvk_mat_t *buffer_config,
                                                        const cvk_mat_t *buffer_weights);

/** cv::dnn::readTensorFromONNX(path); returns a new Mat or NULL. */
cvk_mat_t *cvk_dnn_read_tensor_from_onnx(const char *path);

/**
 * cv::dnn::blobFromImage. size==(0,0) keeps the input spatial size;
 * ddepth is CV_32F (5) or CV_8U (0).
 */
cvk_mat_t *cvk_dnn_blob_from_image(const cvk_mat_t *image, double scalefactor,
                                   int size_width, int size_height,
                                   cvk_scalar_t mean, int swap_rb, int crop,
                                   int ddepth);

/** cv::dnn::blobFromImages; `images` is a CV_32SC2 Mat-of-Mat-addresses. */
cvk_mat_t *cvk_dnn_blob_from_images(const cvk_mat_t *images, double scalefactor,
                                    int size_width, int size_height,
                                    cvk_scalar_t mean, int swap_rb, int crop,
                                    int ddepth);

/**
 * cv::dnn::imagesFromBlob. Returns the extracted images as a CV_32SC2
 * Mat-of-Mats (each member a CV_32FC3 2D image); NULL on failure.
 */
cvk_mat_t *cvk_dnn_images_from_blob(const cvk_mat_t *blob);

/** NMS family; indices Mat (CV_32S) is resized and filled in place. */
void cvk_dnn_nms_boxes(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                       float score_threshold, float nms_threshold,
                       cvk_mat_t *indices, float eta, int top_k);
void cvk_dnn_nms_boxes_rotated(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                               float score_threshold, float nms_threshold,
                               cvk_mat_t *indices, float eta, int top_k);
void cvk_dnn_nms_boxes_batched(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                               const cvk_mat_t *class_ids, float score_threshold,
                               float nms_threshold, cvk_mat_t *indices,
                               float eta, int top_k);

/** softNMSBoxes; updated_scores (CV_32F) and indices (CV_32S) filled in place. */
void cvk_dnn_soft_nms_boxes(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                            cvk_mat_t *updated_scores, float score_threshold,
                            float nms_threshold, cvk_mat_t *indices,
                            int top_k, float sigma, int method);

/** cv::dnn::getAvailableTargets(backend); returns an Nx1 CV_32S Mat or NULL. */
cvk_mat_t *cvk_dnn_get_available_targets(int backend);

/** cv::dnn::getAvailableBackends(); returns an Nx2 CV_32SC2 Mat (backend, target) per row. */
cvk_mat_t *cvk_dnn_get_available_backends(void);

/** cv::dnn::writeTextGraph(model, output). */
void cvk_dnn_write_text_graph(const char *model, const char *output);

/* =========================================================================
 * Net
 * ========================================================================= */

/** cv::dnn::Net() default constructor (empty network). */
cvk_net_t *cvk_net_create(void);

/** Frees a cvk_dnn_read_net* / cvk_net_create result (NULL tolerated). */
void cvk_net_release(cvk_net_t *net);

int cvk_net_empty(const cvk_net_t *net);
const char *cvk_net_dump(const cvk_net_t *net);
void cvk_net_dump_to_file(const cvk_net_t *net, const char *path);
void cvk_net_dump_to_pbtxt(const cvk_net_t *net, const char *path);
int cvk_net_get_layer_id(const cvk_net_t *net, const char *layer);
unsigned char *cvk_net_get_layer_names(const cvk_net_t *net, size_t *out_len);
cvk_layer_t *cvk_net_get_layer(const cvk_net_t *net, int layer_id);
cvk_layer_t *cvk_net_get_layer_by_name(const cvk_net_t *net, const char *layer_name);
void cvk_net_connect(const cvk_net_t *net, const char *out_pin, const char *inp_pin);
int cvk_net_register_output(const cvk_net_t *net, const char *output_name,
                            int layer_id, int output_port);
void cvk_net_set_inputs_names(const cvk_net_t *net, const unsigned char *flat,
                              size_t len);
void cvk_net_set_input_shape(const cvk_net_t *net, const char *input_name,
                             const cvk_mat_t *shape);

/** forward(outputName); whole network when outputName is empty. */
cvk_mat_t *cvk_net_forward(const cvk_net_t *net, const char *output_name);

/** forward() returning every output blob as a CV_32SC2 Mat-of-Mats. */
cvk_mat_t *cvk_net_forward_all(const cvk_net_t *net);

/**
 * forward(outputBlobs, outputName): every output blob of the layer named
 * [output_name] (whole network when empty) as a CV_32SC2 Mat-of-Mats.
 */
cvk_mat_t *cvk_net_forward_layer(const cvk_net_t *net, const char *output_name);

/** forward(outputBlobs, outBlobNames); returns a CV_32SC2 Mat-of-Mats. */
cvk_mat_t *cvk_net_forward_names(const cvk_net_t *net, const unsigned char *flat,
                                 size_t len);

/** forwardAndRetrieve: vector<vector<Mat>> as a doubly-encoded Mat-of-Mats. */
cvk_mat_t *cvk_net_forward_and_retrieve(const cvk_net_t *net,
                                        const unsigned char *flat, size_t len);

void cvk_net_set_preferable_backend(const cvk_net_t *net, int backend_id);
void cvk_net_set_preferable_target(const cvk_net_t *net, int target_id);
void cvk_net_finalize(const cvk_net_t *net);
void cvk_net_set_input(const cvk_net_t *net, const cvk_mat_t *blob,
                       const char *name, double scalefactor, cvk_scalar_t mean);
void cvk_net_set_param(const cvk_net_t *net, int layer, int num_param,
                       const cvk_mat_t *blob);
void cvk_net_set_param_by_name(const cvk_net_t *net, const char *layer_name,
                               int num_param, const cvk_mat_t *blob);
cvk_mat_t *cvk_net_get_param(const cvk_net_t *net, int layer, int num_param);
cvk_mat_t *cvk_net_get_param_by_name(const cvk_net_t *net, const char *layer_name,
                                     int num_param);

/** getUnconnectedOutLayers(); returns an Nx1 CV_32S Mat. */
cvk_mat_t *cvk_net_get_unconnected_out_layers(const cvk_net_t *net);
unsigned char *cvk_net_get_unconnected_out_layers_names(const cvk_net_t *net,
                                                        size_t *out_len);

/**
 * getFLOPS(netInputShapes, netInputTypes). input_shapes is a CV_32SC2
 * Mat-of-Mats of CV_32S shape Mats; input_types is an Nx1 CV_32S Mat.
 */
long long cvk_net_get_flops(const cvk_net_t *net, const cvk_mat_t *input_shapes,
                            const cvk_mat_t *input_types);

/** getMemoryConsumption; writes weights bytes into out[0], blobs into out[1]. */
void cvk_net_get_memory_consumption(const cvk_net_t *net,
                                    const cvk_mat_t *input_shapes,
                                    const cvk_mat_t *input_types,
                                    unsigned long long *out);

void cvk_net_get_layer_types(const cvk_net_t *net, unsigned char **flat,
                             size_t *out_len);
int cvk_net_get_layers_count(const cvk_net_t *net, const char *layer_type);
void cvk_net_enable_fusion(const cvk_net_t *net, int fusion);
void cvk_net_enable_winograd(const cvk_net_t *net, int use_winograd);

/** getPerfProfile(timings); overall ticks returned, timings Mat filled in place. */
long long cvk_net_get_perf_profile(const cvk_net_t *net, cvk_mat_t *timings);

void cvk_net_enable_kv_cache(const cvk_net_t *net);
void cvk_net_disable_kv_cache(const cvk_net_t *net);
void cvk_net_reset_kv_cache(const cvk_net_t *net);
/**
 * getPerfProfile(names, timems, counts) packed into one flat buffer:
 * [u32le count] then per entry: [u32le name_len, bytes, u32le time_len,
 * bytes, u32le count_len, bytes]. Released with cvk_free_buffer.
 */
void cvk_net_get_perf_profile_names(const cvk_net_t *net, unsigned char **flat,
                                    size_t *out_len);
void cvk_net_print_perf_profile(const cvk_net_t *net);

/* =========================================================================
 * Layer
 * ========================================================================= */

void cvk_layer_release(cvk_layer_t *layer);
const char *cvk_layer_name(const cvk_layer_t *layer);
const char *cvk_layer_type(const cvk_layer_t *layer);
int cvk_layer_preferable_target(const cvk_layer_t *layer);
int cvk_layer_output_name_to_index(const cvk_layer_t *layer,
                                   const char *output_name);

/** get_blobs(); CV_32SC2 Mat-of-Mats (caller owns members and container). */
cvk_mat_t *cvk_layer_get_blobs(const cvk_layer_t *layer);
void cvk_layer_set_blobs(cvk_layer_t *layer, const cvk_mat_t *blobs);

/** finalize(inputs, outputs): outputs Mat-of-Mats resized/filled in place. */
void cvk_layer_finalize(const cvk_layer_t *layer, const cvk_mat_t *inputs,
                        cvk_mat_t *outputs);
/** run(inputs, outputs, internals): outputs/internals are in/out Mat-of-Mats. */
void cvk_layer_run(const cvk_layer_t *layer, const cvk_mat_t *inputs,
                   cvk_mat_t *outputs, cvk_mat_t *internals);

/* =========================================================================
 * Tokenizer
 * ========================================================================= */

/** Tokenizer::load(model_config); NULL (with error) when files are missing. */
cvk_tokenizer_t *cvk_tokenizer_load(const char *model_config);
void cvk_tokenizer_release(cvk_tokenizer_t *tokenizer);

/** encode(text); returns an Nx1 CV_32S Mat of token ids. */
cvk_mat_t *cvk_tokenizer_encode(const cvk_tokenizer_t *tokenizer, const char *text);

/** decode(tokens); tokens is an Nx1 CV_32S Mat; static string buffer. */
const char *cvk_tokenizer_decode(const cvk_tokenizer_t *tokenizer,
                                 const cvk_mat_t *tokens);

/* =========================================================================
 * dnn helpers (N-dim Mat introspection for blob tests)
 * ========================================================================= */

/** Number of dimensions of the Mat (2 for the common case). */
int cvk_dnn_mat_dims(const cvk_mat_t *mat);

/** Writes the Mat's dimensions into out[] (up to max_count); returns dims. */
int cvk_dnn_mat_shape(const cvk_mat_t *mat, int *out, int max_count);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_DNN_H */

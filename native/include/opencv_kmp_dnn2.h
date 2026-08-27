/*
 * cvk_ C ABI declarations for opencv-kmp slice "dnn2".
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here back both Kotlin/Native (cinterop) and the
 * JVM (JNI).
 *
 * High-level cv::dnn model wrappers (Model, ClassificationModel,
 * DetectionModel, KeypointsModel, SegmentationModel, TextDetectionModel_DB,
 * TextDetectionModel_EAST, TextRecognitionModel). Every handle is a single
 * cvk_model_t: the shim stores whichever concrete cv::dnn model the handle
 * was created with inside one struct, so base-Model operations and the
 * per-class operations all take cvk_model_t *.
 */
#ifndef OPENCV_KMP_DNN2_H
#define OPENCV_KMP_DNN2_H

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration matching the DnnCore slice's handle
 * (opencv_kmp_dnn.h is included after this header by the umbrella; the
 * identical typedef is legal in C11 and C++). */
typedef struct cvk_net cvk_net_t;

typedef struct cvk_model cvk_model_t;

/* =========================================================================
 * Model (cv::dnn::Model) — base API, valid for every model handle
 * ========================================================================= */

/** Loads a model from trained-weights [model] and optional [config] files. */
cvk_model_t *cvk_model_create(const char *model, const char *config);

/** Wraps an already-loaded [net] (cv::dnn::Model(const Net&)). */
cvk_model_t *cvk_model_create_from_net(cvk_net_t *net);

int cvk_model_set_input_size(cvk_model_t *h, int width, int height);
int cvk_model_set_input_mean(cvk_model_t *h, cvk_scalar_t mean);
int cvk_model_set_input_scale(cvk_model_t *h, cvk_scalar_t scale);
int cvk_model_set_input_crop(cvk_model_t *h, int crop);
int cvk_model_set_input_swap_rb(cvk_model_t *h, int swap_rb);
int cvk_model_set_output_names(cvk_model_t *h, const char *const *names, int count);

/** scale * resize(frame) - mean with the given blob size and flags. */
int cvk_model_set_input_params(cvk_model_t *h, double scale, int width, int height,
                               cvk_scalar_t mean, int swap_rb, int crop);

/**
 * Runs the net on [frame]; returns a malloc'd array of out_count new Mat
 * handles (each released with cvk_mat_release, the array with
 * cvk_free_buffer). NULL on failure.
 */
cvk_mat_t **cvk_model_predict(const cvk_model_t *h, const cvk_mat_t *frame,
                              size_t *out_count);

int cvk_model_set_preferable_backend(cvk_model_t *h, int backend_id);
int cvk_model_set_preferable_target(cvk_model_t *h, int target_id);
int cvk_model_enable_winograd(cvk_model_t *h, int use_winograd);

/** Frees the handle (exactly once); works for every model subclass. */
void cvk_model_release(cvk_model_t *h);

/* =========================================================================
 * ClassificationModel
 * ========================================================================= */

cvk_model_t *cvk_classification_model_create(const char *model, const char *config);
cvk_model_t *cvk_classification_model_create_from_net(cvk_net_t *net);
int cvk_classification_model_set_enable_softmax_post_processing(cvk_model_t *h, int enable);
int cvk_classification_model_get_enable_softmax_post_processing(const cvk_model_t *h);

/** Writes top-1 classId and confidence; returns 1 on success. */
int cvk_classification_model_classify(const cvk_model_t *h, const cvk_mat_t *frame,
                                      int *class_id, float *confidence);

/* =========================================================================
 * DetectionModel
 * ========================================================================= */

cvk_model_t *cvk_detection_model_create(const char *model, const char *config);
cvk_model_t *cvk_detection_model_create_from_net(cvk_net_t *net);
int cvk_detection_model_set_nms_across_classes(cvk_model_t *h, int value);
int cvk_detection_model_get_nms_across_classes(const cvk_model_t *h);

/**
 * Runs detection. Returns a malloc'd little-endian wire buffer
 * ([u32 count][count x i32 classId][count x f32 confidence]
 *  [count x 4 x i32 box x,y,w,h]); NULL on failure. Free with
 * cvk_free_buffer.
 */
unsigned char *cvk_detection_model_detect(const cvk_model_t *h, const cvk_mat_t *frame,
                                          float conf_threshold, float nms_threshold,
                                          size_t *out_len);

/* =========================================================================
 * KeypointsModel
 * ========================================================================= */

cvk_model_t *cvk_keypoints_model_create(const char *model, const char *config);
cvk_model_t *cvk_keypoints_model_create_from_net(cvk_net_t *net);

/** Returns a malloc'd wire buffer ([u32 count][count x 2 x f32 x,y]). */
unsigned char *cvk_keypoints_model_estimate(const cvk_model_t *h, const cvk_mat_t *frame,
                                            float thresh, size_t *out_len);

/* =========================================================================
 * SegmentationModel
 * ========================================================================= */

cvk_model_t *cvk_segmentation_model_create(const char *model, const char *config);
cvk_model_t *cvk_segmentation_model_create_from_net(cvk_net_t *net);

/** Runs segmentation; returns the class-prediction mask as a new Mat. */
cvk_mat_t *cvk_segmentation_model_segment(const cvk_model_t *h, const cvk_mat_t *frame);

/* =========================================================================
 * TextDetectionModel — base ops (handles created as _DB or _EAST)
 * ========================================================================= */

/**
 * Quadrangle detection. Wire buffer:
 * [u32 count][per detection: u32 npts + npts x 2 x i32 x,y]
 * [count x f32 confidence]. Free with cvk_free_buffer.
 */
unsigned char *cvk_text_detection_model_detect(const cvk_model_t *h, const cvk_mat_t *frame,
                                               size_t *out_len);

/**
 * Rotated-rectangle detection. Wire buffer:
 * [u32 count][count x 5 x f32 cx,cy,w,h,angle][count x f32 confidence].
 */
unsigned char *cvk_text_detection_model_detect_text_rectangles(const cvk_model_t *h,
                                                               const cvk_mat_t *frame,
                                                               size_t *out_len);

/* =========================================================================
 * TextDetectionModel_DB
 * ========================================================================= */

cvk_model_t *cvk_text_detection_model_db_create(const char *model, const char *config);
cvk_model_t *cvk_text_detection_model_db_create_from_net(cvk_net_t *net);
int cvk_text_detection_model_db_set_binary_threshold(cvk_model_t *h, float value);
float cvk_text_detection_model_db_get_binary_threshold(const cvk_model_t *h);
int cvk_text_detection_model_db_set_polygon_threshold(cvk_model_t *h, float value);
float cvk_text_detection_model_db_get_polygon_threshold(const cvk_model_t *h);
int cvk_text_detection_model_db_set_unclip_ratio(cvk_model_t *h, double value);
double cvk_text_detection_model_db_get_unclip_ratio(const cvk_model_t *h);
int cvk_text_detection_model_db_set_max_candidates(cvk_model_t *h, int value);
int cvk_text_detection_model_db_get_max_candidates(const cvk_model_t *h);

/* =========================================================================
 * TextDetectionModel_EAST
 * ========================================================================= */

cvk_model_t *cvk_text_detection_model_east_create(const char *model, const char *config);
cvk_model_t *cvk_text_detection_model_east_create_from_net(cvk_net_t *net);
int cvk_text_detection_model_east_set_confidence_threshold(cvk_model_t *h, float value);
float cvk_text_detection_model_east_get_confidence_threshold(const cvk_model_t *h);
int cvk_text_detection_model_east_set_nms_threshold(cvk_model_t *h, float value);
float cvk_text_detection_model_east_get_nms_threshold(const cvk_model_t *h);

/* =========================================================================
 * TextRecognitionModel
 * ========================================================================= */

cvk_model_t *cvk_text_recognition_model_create(const char *model, const char *config);
cvk_model_t *cvk_text_recognition_model_create_from_net(cvk_net_t *net);
int cvk_text_recognition_model_set_decode_type(cvk_model_t *h, const char *decode_type);
const char *cvk_text_recognition_model_get_decode_type(const cvk_model_t *h);
int cvk_text_recognition_model_set_decode_opts_ctc_prefix_beam_search(cvk_model_t *h,
                                                                      int beam_size,
                                                                      int voc_prune_size);
int cvk_text_recognition_model_set_vocabulary(cvk_model_t *h, const char *const *vocab,
                                              int count);

/** Wire buffer: [u32 count][per string: u32 len + len bytes]. */
unsigned char *cvk_text_recognition_model_get_vocabulary(const cvk_model_t *h,
                                                         size_t *out_len);

/** Recognizes the whole [frame]; thread-local string result. */
const char *cvk_text_recognition_model_recognize(const cvk_model_t *h, const cvk_mat_t *frame);

/**
 * Recognizes each of roi_count ROIs from the flat [x,y,w,h] rect array.
 * Wire buffer: [u32 count][per string: u32 len + len bytes].
 */
unsigned char *cvk_text_recognition_model_recognize_rois(const cvk_model_t *h,
                                                         const cvk_mat_t *frame,
                                                         const int *roi_rects,
                                                         int roi_count,
                                                         size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_DNN2_H */

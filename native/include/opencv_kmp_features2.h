/*
 * cvk_ C ABI declarations for opencv-kmp slice "features2".
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here back both Kotlin/Native (cinterop) and the
 * JVM (JNI).
 *
 * Slice surface: descriptor matchers (DescriptorMatcher/BFMatcher/
 * FlannBasedMatcher/LightGlueMatcher), the Annoy-backed ANNIndex, the
 * learned Feature2D extractors (ALIKED, DISK) and the Features static
 * utilities (goodFeaturesToTrack family, drawKeypoints/drawMatches).
 *
 * Wire formats (SDK Converters parity):
 *  - DMatch results: fresh Nx1 CV_32FC4 Mats (queryIdx, trainIdx, imgIdx,
 *    distance).
 *  - List-of-DMatch results (knnMatch/radiusMatch): a fresh Nx1 CV_32SC2
 *    Mat whose rows pack the 64-bit addresses of fresh Nx1 CV_32FC4 Mats.
 *  - List<Mat> arguments (add/masks/drawMatchesKnn): an Nx1 CV_32SC2 Mat
 *    with the same 64-bit address packing, produced by the Kotlin layer.
 */
#ifndef OPENCV_KMP_FEATURES2_H
#define OPENCV_KMP_FEATURES2_H

#ifdef __cplusplus
extern "C" {
#endif

/* Descriptor matchers. All four share one underlying cv::Ptr<cv::DescriptorMatcher>. */
typedef struct cvk_descriptor_matcher cvk_descriptor_matcher_t;
typedef struct cvk_descriptor_matcher cvk_bf_matcher_t;
typedef struct cvk_descriptor_matcher cvk_flann_matcher_t;
typedef struct cvk_descriptor_matcher cvk_lightglue_matcher_t;

typedef struct cvk_ann_index cvk_ann_index_t;
typedef struct cvk_aliked cvk_aliked_t;
typedef struct cvk_disk cvk_disk_t;

/* ---- factories -------------------------------------------------------- */

/** cv::DescriptorMatcher::create(descriptorMatcherType), e.g. "BruteForce". */
cvk_descriptor_matcher_t *cvk_descriptor_matcher_create(const char *type);

/** cv::DescriptorMatcher::create(MatcherType) with the SDK's int enum. */
cvk_descriptor_matcher_t *cvk_descriptor_matcher_create_type(int matcher_type);

/** cv::BFMatcher::create(normType, crossCheck). */
cvk_bf_matcher_t *cvk_bf_matcher_create(int norm_type, int cross_check);

/**
 * cv::FlannBasedMatcher ctor with an index-params selector string:
 * "kdtree" (default), "linear", "kmeans", "lsh", "composite",
 * "autotuned" or "hierarchical"; unknown/empty selects KD-tree.
 */
cvk_flann_matcher_t *cvk_flann_matcher_create(const char *index_params);

/** cv::LightGlueMatcher::create(modelPath, scoreThreshold, backend, target). */
cvk_lightglue_matcher_t *cvk_lightglue_matcher_create(
    const char *model_path, float score_threshold, int backend, int target);

/** cv::LightGlueMatcher::create(vector<uchar> modelData, ...). */
cvk_lightglue_matcher_t *cvk_lightglue_matcher_create_from_memory(
    const unsigned char *model_data, size_t model_len,
    float score_threshold, int backend, int target);

/** cv::DescriptorMatcher::clone(emptyTrainData). */
cvk_descriptor_matcher_t *cvk_descriptor_matcher_clone(
    const cvk_descriptor_matcher_t *h, int empty_train_data);

/* ---- train collection ------------------------------------------------- */

/** Adds `descriptors_wire` (CV_32SC2 handle packing) to the train collection. */
void cvk_descriptor_matcher_add(cvk_descriptor_matcher_t *h,
                                const cvk_mat_t *descriptors_wire);

/** Returns the train collection as a CV_32SC2 wire Mat of fresh Mat clones. */
cvk_mat_t *cvk_descriptor_matcher_get_train_descriptors(
    const cvk_descriptor_matcher_t *h);

/** cv::DescriptorMatcher::isMaskSupported(). */
int cvk_descriptor_matcher_is_mask_supported(const cvk_descriptor_matcher_t *h);

/** cv::DescriptorMatcher::train(). */
void cvk_descriptor_matcher_train(cvk_descriptor_matcher_t *h);

/** cv::DescriptorMatcher::write(fileName). */
void cvk_descriptor_matcher_write(cvk_descriptor_matcher_t *h,
                                  const char *filename);

/** cv::DescriptorMatcher::read(fileName). */
void cvk_descriptor_matcher_read(cvk_descriptor_matcher_t *h,
                                 const char *filename);

/* ---- matching --------------------------------------------------------- */

/**
 * match() against an explicit train matrix; returns a fresh Nx1 CV_32FC4
 * Mat of DMatch rows. `mask` may be NULL for no mask.
 */
cvk_mat_t *cvk_descriptor_matcher_match_train(const cvk_descriptor_matcher_t *h,
                                              const cvk_mat_t *query,
                                              const cvk_mat_t *train,
                                              const cvk_mat_t *mask);

/** match() against the stored train collection; `masks_wire` may be NULL. */
cvk_mat_t *cvk_descriptor_matcher_match(const cvk_descriptor_matcher_t *h,
                                        const cvk_mat_t *query,
                                        const cvk_mat_t *masks_wire);

/**
 * knnMatch() against an explicit train matrix; returns a CV_32SC2 wire Mat
 * of fresh Nx1 CV_32FC4 Mats (one per query row).
 */
cvk_mat_t *cvk_descriptor_matcher_knn_match_train(
    const cvk_descriptor_matcher_t *h, const cvk_mat_t *query,
    const cvk_mat_t *train, int k, const cvk_mat_t *mask, int compact_result);

/** knnMatch() against the stored train collection. */
cvk_mat_t *cvk_descriptor_matcher_knn_match(const cvk_descriptor_matcher_t *h,
                                            const cvk_mat_t *query, int k,
                                            const cvk_mat_t *masks_wire,
                                            int compact_result);

/** radiusMatch() against an explicit train matrix (wire Mat of DMatch Mats). */
cvk_mat_t *cvk_descriptor_matcher_radius_match_train(
    const cvk_descriptor_matcher_t *h, const cvk_mat_t *query,
    const cvk_mat_t *train, float max_distance, const cvk_mat_t *mask,
    int compact_result);

/** radiusMatch() against the stored train collection. */
cvk_mat_t *cvk_descriptor_matcher_radius_match(
    const cvk_descriptor_matcher_t *h, const cvk_mat_t *query,
    float max_distance, const cvk_mat_t *masks_wire, int compact_result);

/* ---- Algorithm surface ------------------------------------------------ */

void cvk_descriptor_matcher_clear(cvk_descriptor_matcher_t *h);
int cvk_descriptor_matcher_empty(cvk_descriptor_matcher_t *h);
void cvk_descriptor_matcher_save(cvk_descriptor_matcher_t *h,
                                 const char *filename);
const char *cvk_descriptor_matcher_get_default_name(cvk_descriptor_matcher_t *h);
void cvk_descriptor_matcher_release(cvk_descriptor_matcher_t *h);

/* ---- LightGlueMatcher extras ------------------------------------------ */

/** cv::LightGlueMatcher::setPairInfo(queryKpts, trainKpts, sizes). */
void cvk_lightglue_matcher_set_pair_info(cvk_lightglue_matcher_t *h,
                                         const cvk_mat_t *query_kpts,
                                         const cvk_mat_t *train_kpts,
                                         double query_width, double query_height,
                                         double train_width, double train_height);

/** cv::LightGlueMatcher::clearPairInfo(). */
void cvk_lightglue_matcher_clear_pair_info(cvk_lightglue_matcher_t *h);

/* ---- ANNIndex (Annoy) ------------------------------------------------- */

/** cv::ANNIndex::create(dim, distType). */
cvk_ann_index_t *cvk_ann_index_create(int dim, int dist_type);

/** cv::ANNIndex::addItems(features). */
void cvk_ann_index_add_items(cvk_ann_index_t *h, const cvk_mat_t *features);

/** cv::ANNIndex::build(trees); -1 picks the tree count automatically. */
void cvk_ann_index_build(cvk_ann_index_t *h, int trees);

/** knnSearch; writes fresh indices (CV_32S) and dists (CV_32F) handles. */
void cvk_ann_index_knn_search(const cvk_ann_index_t *h, const cvk_mat_t *query,
                              int knn, int search_k, cvk_mat_t **indices,
                              cvk_mat_t **dists);

/** cv::ANNIndex::save/load with prefault flags. */
void cvk_ann_index_save(const cvk_ann_index_t *h, const char *filename,
                        int prefault);
void cvk_ann_index_load(const cvk_ann_index_t *h, const char *filename,
                        int prefault);

int cvk_ann_index_tree_number(const cvk_ann_index_t *h);
int cvk_ann_index_item_number(const cvk_ann_index_t *h);

/** cv::ANNIndex::setOnDiskBuild(filename); returns success. */
int cvk_ann_index_set_on_disk_build(const cvk_ann_index_t *h,
                                    const char *filename);

/** cv::ANNIndex::setSeed(seed). */
void cvk_ann_index_set_seed(const cvk_ann_index_t *h, int seed);

void cvk_ann_index_release(cvk_ann_index_t *h);

/* ---- ALIKED (Feature2D) ----------------------------------------------- */

/** cv::ALIKED::create(modelPath, expanded Params). */
cvk_aliked_t *cvk_aliked_create(const char *model_path, int input_width,
                                int input_height, int normalize_descriptors,
                                int engine, int backend, int target);

/** detect; returns a fresh CV_32FC7 keypoint Mat (mask may be NULL). */
cvk_mat_t *cvk_aliked_detect(const cvk_aliked_t *h, const cvk_mat_t *image,
                             const cvk_mat_t *mask);

/** compute; keypoints CV_32FC7 in/out; writes fresh keypoints + CV_32FC1 descriptors. */
void cvk_aliked_compute(const cvk_aliked_t *h, const cvk_mat_t *image,
                        const cvk_mat_t *keypoints, cvk_mat_t **out_keypoints,
                        cvk_mat_t **out_descriptors);

/** detectAndCompute; writes fresh keypoints + descriptor handles. */
void cvk_aliked_detect_and_compute(const cvk_aliked_t *h, const cvk_mat_t *image,
                                   const cvk_mat_t *mask,
                                   int use_provided_keypoints,
                                   cvk_mat_t **out_keypoints,
                                   cvk_mat_t **out_descriptors);

int cvk_aliked_descriptor_size(const cvk_aliked_t *h);
int cvk_aliked_descriptor_type(const cvk_aliked_t *h);
int cvk_aliked_default_norm(const cvk_aliked_t *h);
void cvk_aliked_write(const cvk_aliked_t *h, const char *filename);
void cvk_aliked_read(cvk_aliked_t *h, const char *filename);
void cvk_aliked_clear(cvk_aliked_t *h);
int cvk_aliked_empty(cvk_aliked_t *h);
void cvk_aliked_save(cvk_aliked_t *h, const char *filename);
const char *cvk_aliked_get_default_name(cvk_aliked_t *h);
void cvk_aliked_release(cvk_aliked_t *h);

/* ---- DISK (Feature2D) ------------------------------------------------- */

/** cv::DISK::create(modelPath, maxKeypoints, scoreThreshold, imageSize, ...). */
cvk_disk_t *cvk_disk_create(const char *model_path, int max_keypoints,
                            float score_threshold, double image_width,
                            double image_height, int backend_id, int target_id);

/** cv::DISK::createFromMemory(modelData, ...). */
cvk_disk_t *cvk_disk_create_from_memory(const unsigned char *model_data,
                                        size_t model_len, int max_keypoints,
                                        float score_threshold,
                                        double image_width, double image_height,
                                        int backend_id, int target_id);

cvk_mat_t *cvk_disk_detect(const cvk_disk_t *h, const cvk_mat_t *image,
                           const cvk_mat_t *mask);
void cvk_disk_compute(const cvk_disk_t *h, const cvk_mat_t *image,
                      const cvk_mat_t *keypoints, cvk_mat_t **out_keypoints,
                      cvk_mat_t **out_descriptors);
void cvk_disk_detect_and_compute(const cvk_disk_t *h, const cvk_mat_t *image,
                                 const cvk_mat_t *mask,
                                 int use_provided_keypoints,
                                 cvk_mat_t **out_keypoints,
                                 cvk_mat_t **out_descriptors);
int cvk_disk_descriptor_size(const cvk_disk_t *h);
int cvk_disk_descriptor_type(const cvk_disk_t *h);
int cvk_disk_default_norm(const cvk_disk_t *h);
void cvk_disk_write(const cvk_disk_t *h, const char *filename);
void cvk_disk_read(cvk_disk_t *h, const char *filename);
void cvk_disk_set_max_keypoints(cvk_disk_t *h, int max_keypoints);
int cvk_disk_get_max_keypoints(const cvk_disk_t *h);
void cvk_disk_set_score_threshold(cvk_disk_t *h, float threshold);
float cvk_disk_get_score_threshold(const cvk_disk_t *h);
void cvk_disk_set_image_size(cvk_disk_t *h, double width, double height);
void cvk_disk_image_size(const cvk_disk_t *h, double out[2]);
void cvk_disk_clear(cvk_disk_t *h);
int cvk_disk_empty(cvk_disk_t *h);
void cvk_disk_save(cvk_disk_t *h, const char *filename);
const char *cvk_disk_get_default_name(cvk_disk_t *h);
void cvk_disk_release(cvk_disk_t *h);

/* ---- Features statics ------------------------------------------------- */

/**
 * cv::goodFeaturesToTrack (vector<Point> output). Returns a fresh Nx1
 * CV_32SC2 Mat of int point pairs; `mask` may be NULL.
 */
cvk_mat_t *cvk_features_good_features_to_track(
    const cvk_mat_t *image, int max_corners, double quality_level,
    double min_distance, const cvk_mat_t *mask, int block_size,
    int use_harris_detector, double k);

/** cv::goodFeaturesToTrack with an explicit gradientSize aperture. */
cvk_mat_t *cvk_features_good_features_to_track_gradient(
    const cvk_mat_t *image, int max_corners, double quality_level,
    double min_distance, const cvk_mat_t *mask, int block_size,
    int gradient_size, int use_harris_detector, double k);

/** goodFeaturesToTrackWithQuality; writes fresh corners (CV_32SC2) and quality (CV_32FC1). */
void cvk_features_good_features_to_track_quality(
    const cvk_mat_t *image, int max_corners, double quality_level,
    double min_distance, const cvk_mat_t *mask, int block_size,
    int gradient_size, int use_harris_detector, double k,
    cvk_mat_t **corners, cvk_mat_t **quality);

/** cv::drawKeypoints into a fresh output image (keypoints: CV_32FC7). */
cvk_mat_t *cvk_draw_keypoints(const cvk_mat_t *image,
                              const cvk_mat_t *keypoints, cvk_scalar_t color,
                              int flags);

/** cv::drawKeypoints drawing over the caller's [out_image] in place. */
void cvk_draw_keypoints_over(const cvk_mat_t *image,
                             const cvk_mat_t *keypoints, cvk_mat_t *out_image,
                             cvk_scalar_t color, int flags);

/** cv::drawMatches into a fresh output image (matches: CV_32FC4). */
cvk_mat_t *cvk_draw_matches(const cvk_mat_t *img1, const cvk_mat_t *keypoints1,
                            const cvk_mat_t *img2, const cvk_mat_t *keypoints2,
                            const cvk_mat_t *matches, cvk_scalar_t match_color,
                            cvk_scalar_t single_point_color,
                            const cvk_mat_t *matches_mask, int flags);

/** cv::drawMatches drawing over the caller's [out_img] in place. */
void cvk_draw_matches_over(const cvk_mat_t *img1, const cvk_mat_t *keypoints1,
                           const cvk_mat_t *img2, const cvk_mat_t *keypoints2,
                           const cvk_mat_t *matches, cvk_mat_t *out_img,
                           cvk_scalar_t match_color,
                           cvk_scalar_t single_point_color,
                           const cvk_mat_t *matches_mask, int flags);

/** cv::drawMatches with an explicit matchesThickness, fresh output image. */
cvk_mat_t *cvk_draw_matches_thickness(
    const cvk_mat_t *img1, const cvk_mat_t *keypoints1, const cvk_mat_t *img2,
    const cvk_mat_t *keypoints2, const cvk_mat_t *matches, int matches_thickness,
    cvk_scalar_t match_color, cvk_scalar_t single_point_color,
    const cvk_mat_t *matches_mask, int flags);

/**
 * cv::drawMatches (KNN variant). `matches_wire` and `masks_wire` are
 * CV_32SC2 wire Mats of CV_32FC4 / CV_8SC1 Mats respectively; either may be
 * NULL. Returns a fresh output image.
 */
cvk_mat_t *cvk_draw_matches_knn(const cvk_mat_t *img1,
                                const cvk_mat_t *keypoints1,
                                const cvk_mat_t *img2,
                                const cvk_mat_t *keypoints2,
                                const cvk_mat_t *matches_wire,
                                cvk_scalar_t match_color,
                                cvk_scalar_t single_point_color,
                                const cvk_mat_t *masks_wire, int flags);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_FEATURES2_H */

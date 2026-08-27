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
 * JNI bridge for the "features2" slice: thin
 * Java_cn_enaium_opencv_JniFeatures2_* wrappers around the cvk_ C ABI in
 * native/shim_features2.cpp. Mat handles travel as jlong pointers; scalars
 * are expanded into primitive arguments. No exceptions cross the boundary.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_features2.h"

#include <cstdint>
#include <string>
#include <vector>

static inline cvk_mat_t *as_mat(jlong h) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(h));
}
static inline jlong as_handle(const cvk_mat_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}
static inline cvk_descriptor_matcher_t *as_matcher(jlong h) {
    return reinterpret_cast<cvk_descriptor_matcher_t *>(static_cast<uintptr_t>(h));
}
static inline jlong as_matcher_handle(const cvk_descriptor_matcher_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}
static inline cvk_ann_index_t *as_ann(jlong h) {
    return reinterpret_cast<cvk_ann_index_t *>(static_cast<uintptr_t>(h));
}
static inline jlong as_ann_handle(const cvk_ann_index_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}
static inline cvk_aliked_t *as_aliked(jlong h) {
    return reinterpret_cast<cvk_aliked_t *>(static_cast<uintptr_t>(h));
}
static inline jlong as_aliked_handle(const cvk_aliked_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}
static inline cvk_disk_t *as_disk(jlong h) {
    return reinterpret_cast<cvk_disk_t *>(static_cast<uintptr_t>(h));
}
static inline jlong as_disk_handle(const cvk_disk_t *m) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(m));
}

static inline cvk_scalar_t as_scalar(jdouble v0, jdouble v1, jdouble v2, jdouble v3) {
    cvk_scalar_t s;
    s.v0 = v0;
    s.v1 = v1;
    s.v2 = v2;
    s.v3 = v3;
    return s;
}

static std::string jstr(JNIEnv *env, jstring s) {
    if (s == nullptr) return {};
    const char *utf = env->GetStringUTFChars(s, nullptr);
    if (utf == nullptr) return {};
    std::string out(utf);
    env->ReleaseStringUTFChars(s, utf);
    return out;
}

static jlongArray handle_pair(JNIEnv *env, cvk_mat_t *a, cvk_mat_t *b) {
    jlong values[2] = {as_handle(a), as_handle(b)};
    jlongArray result = env->NewLongArray(2);
    if (result == nullptr) {
        cvk_mat_release(a);
        cvk_mat_release(b);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 2, values);
    return result;
}

extern "C" {

/* ---- DescriptorMatcher factories -------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherCreate(JNIEnv *env, jobject,
                                                           jstring type) {
    return as_matcher_handle(cvk_descriptor_matcher_create(jstr(env, type).c_str()));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherCreateType(JNIEnv *, jobject,
                                                               jint type) {
    return as_matcher_handle(cvk_descriptor_matcher_create_type(type));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_bfMatcherCreate(JNIEnv *, jobject, jint norm_type,
                                                   jboolean cross_check) {
    return as_matcher_handle(cvk_bf_matcher_create(norm_type, cross_check != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_flannMatcherCreate(JNIEnv *env, jobject,
                                                      jstring index_params) {
    return as_matcher_handle(cvk_flann_matcher_create(jstr(env, index_params).c_str()));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_lightGlueMatcherCreate(JNIEnv *env, jobject,
                                                          jstring model_path,
                                                          jfloat score_threshold,
                                                          jint backend, jint target) {
    return as_matcher_handle(cvk_lightglue_matcher_create(
        jstr(env, model_path).c_str(), score_threshold, backend, target));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_lightGlueMatcherCreateFromMemory(
    JNIEnv *env, jobject, jbyteArray model_data, jfloat score_threshold, jint backend,
    jint target) {
    if (model_data == nullptr) return 0;
    const jsize length = env->GetArrayLength(model_data);
    jbyte *elements = env->GetByteArrayElements(model_data, nullptr);
    if (elements == nullptr) return 0;
    const auto *bytes = reinterpret_cast<const unsigned char *>(elements);
    const jlong handle = as_matcher_handle(cvk_lightglue_matcher_create_from_memory(
        bytes, static_cast<size_t>(length), score_threshold, backend, target));
    env->ReleaseByteArrayElements(model_data, elements, JNI_ABORT);
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherClone(JNIEnv *, jobject, jlong h,
                                                          jboolean empty_train_data) {
    return as_matcher_handle(cvk_descriptor_matcher_clone(
        as_matcher(h), empty_train_data != JNI_FALSE));
}

/* ---- train collection ------------------------------------------------- */

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherAdd(JNIEnv *, jobject, jlong h,
                                                        jlong descriptors_wire) {
    cvk_descriptor_matcher_add(as_matcher(h), as_mat(descriptors_wire));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherGetTrainDescriptors(JNIEnv *,
                                                                        jobject,
                                                                        jlong h) {
    return as_handle(cvk_descriptor_matcher_get_train_descriptors(as_matcher(h)));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherIsMaskSupported(JNIEnv *, jobject,
                                                                    jlong h) {
    return cvk_descriptor_matcher_is_mask_supported(as_matcher(h)) != 0 ? JNI_TRUE
                                                                        : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherTrain(JNIEnv *, jobject, jlong h) {
    cvk_descriptor_matcher_train(as_matcher(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherWrite(JNIEnv *env, jobject,
                                                          jlong h, jstring filename) {
    cvk_descriptor_matcher_write(as_matcher(h), jstr(env, filename).c_str());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherRead(JNIEnv *env, jobject, jlong h,
                                                         jstring filename) {
    cvk_descriptor_matcher_read(as_matcher(h), jstr(env, filename).c_str());
}

/* ---- matching --------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherMatchTrain(JNIEnv *, jobject,
                                                               jlong h, jlong query,
                                                               jlong train, jlong mask) {
    return as_handle(cvk_descriptor_matcher_match_train(
        as_matcher(h), as_mat(query), as_mat(train), mask != 0 ? as_mat(mask) : nullptr));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherMatch(JNIEnv *, jobject, jlong h,
                                                          jlong query,
                                                          jlong masks_wire) {
    return as_handle(cvk_descriptor_matcher_match(
        as_matcher(h), as_mat(query), masks_wire != 0 ? as_mat(masks_wire) : nullptr));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherKnnMatchTrain(
    JNIEnv *, jobject, jlong h, jlong query, jlong train, jint k, jlong mask,
    jboolean compact_result) {
    return as_handle(cvk_descriptor_matcher_knn_match_train(
        as_matcher(h), as_mat(query), as_mat(train), k,
        mask != 0 ? as_mat(mask) : nullptr, compact_result != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherKnnMatch(JNIEnv *, jobject,
                                                             jlong h, jlong query,
                                                             jint k, jlong masks_wire,
                                                             jboolean compact_result) {
    return as_handle(cvk_descriptor_matcher_knn_match(
        as_matcher(h), as_mat(query), k,
        masks_wire != 0 ? as_mat(masks_wire) : nullptr, compact_result != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherRadiusMatchTrain(
    JNIEnv *, jobject, jlong h, jlong query, jlong train, jfloat max_distance,
    jlong mask, jboolean compact_result) {
    return as_handle(cvk_descriptor_matcher_radius_match_train(
        as_matcher(h), as_mat(query), as_mat(train), max_distance,
        mask != 0 ? as_mat(mask) : nullptr, compact_result != JNI_FALSE));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherRadiusMatch(
    JNIEnv *, jobject, jlong h, jlong query, jfloat max_distance, jlong masks_wire,
    jboolean compact_result) {
    return as_handle(cvk_descriptor_matcher_radius_match(
        as_matcher(h), as_mat(query), max_distance,
        masks_wire != 0 ? as_mat(masks_wire) : nullptr, compact_result != JNI_FALSE));
}

/* ---- Algorithm surface ------------------------------------------------ */

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherClear(JNIEnv *, jobject, jlong h) {
    cvk_descriptor_matcher_clear(as_matcher(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherEmpty(JNIEnv *, jobject, jlong h) {
    return cvk_descriptor_matcher_empty(as_matcher(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherSave(JNIEnv *env, jobject, jlong h,
                                                         jstring filename) {
    cvk_descriptor_matcher_save(as_matcher(h), jstr(env, filename).c_str());
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherGetDefaultName(JNIEnv *env, jobject,
                                                                   jlong h) {
    const char *name = cvk_descriptor_matcher_get_default_name(as_matcher(h));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_descriptorMatcherRelease(JNIEnv *, jobject, jlong h) {
    cvk_descriptor_matcher_release(as_matcher(h));
}

/* ---- LightGlueMatcher extras ------------------------------------------ */

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_lightGlueMatcherSetPairInfo(
    JNIEnv *, jobject, jlong h, jlong query_kpts, jlong train_kpts, jdouble qw,
    jdouble qh, jdouble tw, jdouble th) {
    cvk_lightglue_matcher_set_pair_info(as_matcher(h), as_mat(query_kpts),
                                        as_mat(train_kpts), qw, qh, tw, th);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_lightGlueMatcherClearPairInfo(JNIEnv *, jobject,
                                                                 jlong h) {
    cvk_lightglue_matcher_clear_pair_info(as_matcher(h));
}

/* ---- ANNIndex --------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexCreate(JNIEnv *, jobject, jint dim,
                                                  jint dist_type) {
    return as_ann_handle(cvk_ann_index_create(dim, dist_type));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexAddItems(JNIEnv *, jobject, jlong h,
                                                    jlong features) {
    cvk_ann_index_add_items(as_ann(h), as_mat(features));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexBuild(JNIEnv *, jobject, jlong h,
                                                 jint trees) {
    cvk_ann_index_build(as_ann(h), trees);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexKnnSearch(JNIEnv *env, jobject, jlong h,
                                                     jlong query, jint knn,
                                                     jint search_k) {
    cvk_mat_t *indices = nullptr;
    cvk_mat_t *dists = nullptr;
    cvk_ann_index_knn_search(as_ann(h), as_mat(query), knn, search_k, &indices, &dists);
    return handle_pair(env, indices, dists);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexSave(JNIEnv *env, jobject, jlong h,
                                                jstring filename, jboolean prefault) {
    cvk_ann_index_save(as_ann(h), jstr(env, filename).c_str(), prefault != JNI_FALSE);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexLoad(JNIEnv *env, jobject, jlong h,
                                                jstring filename, jboolean prefault) {
    cvk_ann_index_load(as_ann(h), jstr(env, filename).c_str(), prefault != JNI_FALSE);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexTreeNumber(JNIEnv *, jobject, jlong h) {
    return cvk_ann_index_tree_number(as_ann(h));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexItemNumber(JNIEnv *, jobject, jlong h) {
    return cvk_ann_index_item_number(as_ann(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexSetOnDiskBuild(JNIEnv *env, jobject,
                                                          jlong h, jstring filename) {
    return cvk_ann_index_set_on_disk_build(as_ann(h), jstr(env, filename).c_str()) != 0
               ? JNI_TRUE
               : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexSetSeed(JNIEnv *, jobject, jlong h,
                                                   jint seed) {
    cvk_ann_index_set_seed(as_ann(h), seed);
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_annIndexRelease(JNIEnv *, jobject, jlong h) {
    cvk_ann_index_release(as_ann(h));
}

/* ---- ALIKED ----------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedCreate(JNIEnv *env, jobject, jstring model_path,
                                                jint input_width, jint input_height,
                                                jboolean normalize_descriptors,
                                                jint engine, jint backend, jint target) {
    return as_aliked_handle(cvk_aliked_create(
        jstr(env, model_path).c_str(), input_width, input_height,
        normalize_descriptors != JNI_FALSE, engine, backend, target));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedDetect(JNIEnv *, jobject, jlong h, jlong image,
                                                jlong mask) {
    return as_handle(cvk_aliked_detect(as_aliked(h), as_mat(image),
                                       mask != 0 ? as_mat(mask) : nullptr));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedCompute(JNIEnv *env, jobject, jlong h,
                                                 jlong image, jlong keypoints) {
    cvk_mat_t *kps = nullptr;
    cvk_mat_t *descriptors = nullptr;
    cvk_aliked_compute(as_aliked(h), as_mat(image), as_mat(keypoints), &kps,
                       &descriptors);
    return handle_pair(env, kps, descriptors);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedDetectAndCompute(JNIEnv *env, jobject, jlong h,
                                                          jlong image, jlong mask,
                                                          jboolean use_provided) {
    cvk_mat_t *kps = nullptr;
    cvk_mat_t *descriptors = nullptr;
    cvk_aliked_detect_and_compute(as_aliked(h), as_mat(image),
                                  mask != 0 ? as_mat(mask) : nullptr,
                                  use_provided != JNI_FALSE, &kps, &descriptors);
    return handle_pair(env, kps, descriptors);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedDescriptorSize(JNIEnv *, jobject, jlong h) {
    return cvk_aliked_descriptor_size(as_aliked(h));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedDescriptorType(JNIEnv *, jobject, jlong h) {
    return cvk_aliked_descriptor_type(as_aliked(h));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedDefaultNorm(JNIEnv *, jobject, jlong h) {
    return cvk_aliked_default_norm(as_aliked(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedWrite(JNIEnv *env, jobject, jlong h,
                                               jstring filename) {
    cvk_aliked_write(as_aliked(h), jstr(env, filename).c_str());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedRead(JNIEnv *env, jobject, jlong h,
                                              jstring filename) {
    cvk_aliked_read(as_aliked(h), jstr(env, filename).c_str());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedClear(JNIEnv *, jobject, jlong h) {
    cvk_aliked_clear(as_aliked(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedEmpty(JNIEnv *, jobject, jlong h) {
    return cvk_aliked_empty(as_aliked(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedSave(JNIEnv *env, jobject, jlong h,
                                              jstring filename) {
    cvk_aliked_save(as_aliked(h), jstr(env, filename).c_str());
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedGetDefaultName(JNIEnv *env, jobject, jlong h) {
    const char *name = cvk_aliked_get_default_name(as_aliked(h));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_alikedRelease(JNIEnv *, jobject, jlong h) {
    cvk_aliked_release(as_aliked(h));
}

/* ---- DISK ------------------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskCreate(JNIEnv *env, jobject, jstring model_path,
                                              jint max_keypoints, jfloat score_threshold,
                                              jdouble image_width, jdouble image_height,
                                              jint backend_id, jint target_id) {
    return as_disk_handle(cvk_disk_create(
        jstr(env, model_path).c_str(), max_keypoints, score_threshold, image_width,
        image_height, backend_id, target_id));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskCreateFromMemory(JNIEnv *env, jobject,
                                                        jbyteArray model_data,
                                                        jint max_keypoints,
                                                        jfloat score_threshold,
                                                        jdouble image_width,
                                                        jdouble image_height,
                                                        jint backend_id,
                                                        jint target_id) {
    if (model_data == nullptr) return 0;
    const jsize length = env->GetArrayLength(model_data);
    jbyte *elements = env->GetByteArrayElements(model_data, nullptr);
    if (elements == nullptr) return 0;
    const auto *bytes = reinterpret_cast<const unsigned char *>(elements);
    const jlong handle = as_disk_handle(cvk_disk_create_from_memory(
        bytes, static_cast<size_t>(length), max_keypoints, score_threshold, image_width,
        image_height, backend_id, target_id));
    env->ReleaseByteArrayElements(model_data, elements, JNI_ABORT);
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskDetect(JNIEnv *, jobject, jlong h, jlong image,
                                              jlong mask) {
    return as_handle(cvk_disk_detect(as_disk(h), as_mat(image),
                                     mask != 0 ? as_mat(mask) : nullptr));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskCompute(JNIEnv *env, jobject, jlong h,
                                               jlong image, jlong keypoints) {
    cvk_mat_t *kps = nullptr;
    cvk_mat_t *descriptors = nullptr;
    cvk_disk_compute(as_disk(h), as_mat(image), as_mat(keypoints), &kps, &descriptors);
    return handle_pair(env, kps, descriptors);
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskDetectAndCompute(JNIEnv *env, jobject, jlong h,
                                                        jlong image, jlong mask,
                                                        jboolean use_provided) {
    cvk_mat_t *kps = nullptr;
    cvk_mat_t *descriptors = nullptr;
    cvk_disk_detect_and_compute(as_disk(h), as_mat(image),
                                mask != 0 ? as_mat(mask) : nullptr,
                                use_provided != JNI_FALSE, &kps, &descriptors);
    return handle_pair(env, kps, descriptors);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskDescriptorSize(JNIEnv *, jobject, jlong h) {
    return cvk_disk_descriptor_size(as_disk(h));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskDescriptorType(JNIEnv *, jobject, jlong h) {
    return cvk_disk_descriptor_type(as_disk(h));
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskDefaultNorm(JNIEnv *, jobject, jlong h) {
    return cvk_disk_default_norm(as_disk(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskWrite(JNIEnv *env, jobject, jlong h,
                                             jstring filename) {
    cvk_disk_write(as_disk(h), jstr(env, filename).c_str());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskRead(JNIEnv *env, jobject, jlong h,
                                            jstring filename) {
    cvk_disk_read(as_disk(h), jstr(env, filename).c_str());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskSetMaxKeypoints(JNIEnv *, jobject, jlong h,
                                                       jint max_keypoints) {
    cvk_disk_set_max_keypoints(as_disk(h), max_keypoints);
}

JNIEXPORT jint JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskGetMaxKeypoints(JNIEnv *, jobject, jlong h) {
    return cvk_disk_get_max_keypoints(as_disk(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskSetScoreThreshold(JNIEnv *, jobject, jlong h,
                                                         jfloat threshold) {
    cvk_disk_set_score_threshold(as_disk(h), threshold);
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskGetScoreThreshold(JNIEnv *, jobject, jlong h) {
    return cvk_disk_get_score_threshold(as_disk(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskSetImageSize(JNIEnv *, jobject, jlong h,
                                                    jdouble width, jdouble height) {
    cvk_disk_set_image_size(as_disk(h), width, height);
}

JNIEXPORT jdoubleArray JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskImageSize(JNIEnv *env, jobject, jlong h) {
    double out[2] = {0, 0};
    cvk_disk_image_size(as_disk(h), out);
    jdoubleArray result = env->NewDoubleArray(2);
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, 2, out);
    return result;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskClear(JNIEnv *, jobject, jlong h) {
    cvk_disk_clear(as_disk(h));
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskEmpty(JNIEnv *, jobject, jlong h) {
    return cvk_disk_empty(as_disk(h)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskSave(JNIEnv *env, jobject, jlong h,
                                            jstring filename) {
    cvk_disk_save(as_disk(h), jstr(env, filename).c_str());
}

JNIEXPORT jstring JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskGetDefaultName(JNIEnv *env, jobject, jlong h) {
    const char *name = cvk_disk_get_default_name(as_disk(h));
    return name != nullptr ? env->NewStringUTF(name) : nullptr;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_diskRelease(JNIEnv *, jobject, jlong h) {
    cvk_disk_release(as_disk(h));
}

/* ---- Features statics ------------------------------------------------- */

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_featuresGoodFeaturesToTrack(
    JNIEnv *, jobject, jlong image, jint max_corners, jdouble quality_level,
    jdouble min_distance, jlong mask, jint block_size, jboolean use_harris, jdouble k) {
    return as_handle(cvk_features_good_features_to_track(
        as_mat(image), max_corners, quality_level, min_distance,
        mask != 0 ? as_mat(mask) : nullptr, block_size, use_harris != JNI_FALSE, k));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_featuresGoodFeaturesToTrackGradient(
    JNIEnv *, jobject, jlong image, jint max_corners, jdouble quality_level,
    jdouble min_distance, jlong mask, jint block_size, jint gradient_size,
    jboolean use_harris, jdouble k) {
    return as_handle(cvk_features_good_features_to_track_gradient(
        as_mat(image), max_corners, quality_level, min_distance, as_mat(mask),
        block_size, gradient_size, use_harris != JNI_FALSE, k));
}

JNIEXPORT jlongArray JNICALL
Java_cn_enaium_opencv_JniFeatures2_featuresGoodFeaturesToTrackQuality(
    JNIEnv *env, jobject, jlong image, jint max_corners, jdouble quality_level,
    jdouble min_distance, jlong mask, jint block_size, jint gradient_size,
    jboolean use_harris, jdouble k) {
    cvk_mat_t *corners = nullptr;
    cvk_mat_t *quality = nullptr;
    cvk_features_good_features_to_track_quality(
        as_mat(image), max_corners, quality_level, min_distance,
        mask != 0 ? as_mat(mask) : nullptr, block_size, gradient_size,
        use_harris != JNI_FALSE, k, &corners, &quality);
    return handle_pair(env, corners, quality);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_drawKeypoints(JNIEnv *, jobject, jlong image,
                                                 jlong keypoints, jdouble c0, jdouble c1,
                                                 jdouble c2, jdouble c3, jint flags) {
    return as_handle(cvk_draw_keypoints(as_mat(image), as_mat(keypoints),
                                        as_scalar(c0, c1, c2, c3), flags));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_drawKeypointsOver(JNIEnv *, jobject, jlong image,
                                                     jlong keypoints, jlong out_image,
                                                     jdouble c0, jdouble c1, jdouble c2,
                                                     jdouble c3, jint flags) {
    cvk_draw_keypoints_over(as_mat(image), as_mat(keypoints), as_mat(out_image),
                            as_scalar(c0, c1, c2, c3), flags);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_drawMatches(
    JNIEnv *, jobject, jlong img1, jlong keypoints1, jlong img2, jlong keypoints2,
    jlong matches, jdouble mc0, jdouble mc1, jdouble mc2, jdouble mc3, jdouble sc0,
    jdouble sc1, jdouble sc2, jdouble sc3, jlong matches_mask, jint flags) {
    return as_handle(cvk_draw_matches(
        as_mat(img1), as_mat(keypoints1), as_mat(img2), as_mat(keypoints2),
        as_mat(matches), as_scalar(mc0, mc1, mc2, mc3), as_scalar(sc0, sc1, sc2, sc3),
        matches_mask != 0 ? as_mat(matches_mask) : nullptr, flags));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniFeatures2_drawMatchesOver(
    JNIEnv *, jobject, jlong img1, jlong keypoints1, jlong img2, jlong keypoints2,
    jlong matches, jlong out_img, jdouble mc0, jdouble mc1, jdouble mc2, jdouble mc3,
    jdouble sc0, jdouble sc1, jdouble sc2, jdouble sc3, jlong matches_mask, jint flags) {
    cvk_draw_matches_over(as_mat(img1), as_mat(keypoints1), as_mat(img2),
                          as_mat(keypoints2), as_mat(matches), as_mat(out_img),
                          as_scalar(mc0, mc1, mc2, mc3), as_scalar(sc0, sc1, sc2, sc3),
                          matches_mask != 0 ? as_mat(matches_mask) : nullptr, flags);
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_drawMatchesThickness(
    JNIEnv *, jobject, jlong img1, jlong keypoints1, jlong img2, jlong keypoints2,
    jlong matches, jint matches_thickness, jdouble mc0, jdouble mc1, jdouble mc2,
    jdouble mc3, jdouble sc0, jdouble sc1, jdouble sc2, jdouble sc3, jlong matches_mask,
    jint flags) {
    return as_handle(cvk_draw_matches_thickness(
        as_mat(img1), as_mat(keypoints1), as_mat(img2), as_mat(keypoints2),
        as_mat(matches), matches_thickness, as_scalar(mc0, mc1, mc2, mc3),
        as_scalar(sc0, sc1, sc2, sc3), matches_mask != 0 ? as_mat(matches_mask) : nullptr,
        flags));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniFeatures2_drawMatchesKnn(
    JNIEnv *, jobject, jlong img1, jlong keypoints1, jlong img2, jlong keypoints2,
    jlong matches_wire, jdouble mc0, jdouble mc1, jdouble mc2, jdouble mc3, jdouble sc0,
    jdouble sc1, jdouble sc2, jdouble sc3, jlong masks_wire, jint flags) {
    return as_handle(cvk_draw_matches_knn(
        as_mat(img1), as_mat(keypoints1), as_mat(img2), as_mat(keypoints2),
        matches_wire != 0 ? as_mat(matches_wire) : nullptr,
        as_scalar(mc0, mc1, mc2, mc3), as_scalar(sc0, sc1, sc2, sc3),
        masks_wire != 0 ? as_mat(masks_wire) : nullptr, flags));
}

} /* extern "C" */

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
 * JNI bridge for the "video2" slice (long-term object trackers). Each
 * function forwards to the cvk_ C ABI and is registered as a member of the
 * Kotlin `internal object JniVideo2` (Java_cn_enaium_opencv_JniVideo2_*).
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_video2.h"
#include <cstdint>

static inline cvk_mat_t *as_mat(jlong h) {
    return reinterpret_cast<cvk_mat_t *>(static_cast<uintptr_t>(h));
}

static inline cvk_tracker_t *as_tracker(jlong h) {
    return reinterpret_cast<cvk_tracker_t *>(static_cast<uintptr_t>(h));
}

static inline jlong as_handle(const cvk_tracker_t *t) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(t));
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerMILCreate(
    JNIEnv *, jobject,
    jfloat samplerInitInRadius, jint samplerInitMaxNegNum,
    jfloat samplerSearchWinSize, jfloat samplerTrackInRadius,
    jint samplerTrackMaxPosNum, jint samplerTrackMaxNegNum,
    jint featureSetNumFeatures) {
    return as_handle(cvk_tracker_mil_create(
        samplerInitInRadius, samplerInitMaxNegNum,
        samplerSearchWinSize, samplerTrackInRadius,
        samplerTrackMaxPosNum, samplerTrackMaxNegNum,
        featureSetNumFeatures));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerDaSiamRPNCreate(
    JNIEnv *env, jobject,
    jstring model, jstring kernelCls1, jstring kernelR1,
    jint backend, jint target) {
    if (model == nullptr || kernelCls1 == nullptr || kernelR1 == nullptr) {
        return 0;
    }
    const char *model_utf = env->GetStringUTFChars(model, nullptr);
    const char *cls1_utf = env->GetStringUTFChars(kernelCls1, nullptr);
    const char *r1_utf = env->GetStringUTFChars(kernelR1, nullptr);
    if (model_utf == nullptr || cls1_utf == nullptr || r1_utf == nullptr) {
        if (model_utf != nullptr) env->ReleaseStringUTFChars(model, model_utf);
        if (cls1_utf != nullptr) env->ReleaseStringUTFChars(kernelCls1, cls1_utf);
        if (r1_utf != nullptr) env->ReleaseStringUTFChars(kernelR1, r1_utf);
        return 0;
    }
    const jlong handle = as_handle(cvk_tracker_dasiamrpn_create(
        model_utf, cls1_utf, r1_utf, backend, target));
    env->ReleaseStringUTFChars(model, model_utf);
    env->ReleaseStringUTFChars(kernelCls1, cls1_utf);
    env->ReleaseStringUTFChars(kernelR1, r1_utf);
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerNanoCreate(
    JNIEnv *env, jobject,
    jstring backbone, jstring neckhead, jint backend, jint target) {
    if (backbone == nullptr || neckhead == nullptr) {
        return 0;
    }
    const char *backbone_utf = env->GetStringUTFChars(backbone, nullptr);
    const char *neckhead_utf = env->GetStringUTFChars(neckhead, nullptr);
    if (backbone_utf == nullptr || neckhead_utf == nullptr) {
        if (backbone_utf != nullptr) env->ReleaseStringUTFChars(backbone, backbone_utf);
        if (neckhead_utf != nullptr) env->ReleaseStringUTFChars(neckhead, neckhead_utf);
        return 0;
    }
    const jlong handle = as_handle(cvk_tracker_nano_create(
        backbone_utf, neckhead_utf, backend, target));
    env->ReleaseStringUTFChars(backbone, backbone_utf);
    env->ReleaseStringUTFChars(neckhead, neckhead_utf);
    return handle;
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerVitCreate(
    JNIEnv *env, jobject,
    jstring net, jint backend, jint target,
    jdouble meanV0, jdouble meanV1, jdouble meanV2, jdouble meanV3,
    jdouble stdV0, jdouble stdV1, jdouble stdV2, jdouble stdV3,
    jfloat trackingScoreThreshold) {
    if (net == nullptr) {
        return 0;
    }
    const char *net_utf = env->GetStringUTFChars(net, nullptr);
    if (net_utf == nullptr) {
        return 0;
    }
    const cvk_scalar_t meanvalue = {meanV0, meanV1, meanV2, meanV3};
    const cvk_scalar_t stdvalue = {stdV0, stdV1, stdV2, stdV3};
    const jlong handle = as_handle(cvk_tracker_vit_create(
        net_utf, backend, target, meanvalue, stdvalue, trackingScoreThreshold));
    env->ReleaseStringUTFChars(net, net_utf);
    return handle;
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerInit(
    JNIEnv *, jobject,
    jlong tracker, jlong image,
    jint x, jint y, jint width, jint height) {
    const cvk_rect_t box = {x, y, width, height};
    cvk_tracker_init(as_tracker(tracker), as_mat(image), box);
}

JNIEXPORT jboolean JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerUpdate(
    JNIEnv *env, jobject,
    jlong tracker, jlong image, jintArray outRect) {
    cvk_rect_t box = {0, 0, 0, 0};
    const int ok = cvk_tracker_update(as_tracker(tracker), as_mat(image), &box);
    if (outRect != nullptr) {
        const jint rect[4] = {box.x, box.y, box.width, box.height};
        env->SetIntArrayRegion(outRect, 0, 4, rect);
    }
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jfloat JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerGetTrackingScore(
    JNIEnv *, jobject, jlong tracker) {
    return static_cast<jfloat>(cvk_tracker_get_tracking_score(as_tracker(tracker)));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniVideo2_trackerRelease(
    JNIEnv *, jobject, jlong tracker) {
    cvk_tracker_release(as_tracker(tracker));
}

} /* extern "C" */

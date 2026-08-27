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
 * cvk_ C ABI declarations for opencv-kmp slice "video2".
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here back both Kotlin/Native (cinterop) and the
 * JVM (JNI).
 *
 * This slice covers the long-term object trackers (cv::Tracker and the
 * cv::TrackerMIL / TrackerDaSiamRPN / TrackerNano / TrackerVit subclasses).
 * The *_Params holders are pure Kotlin (per the port contract), so every
 * create() below takes the Params fields as expanded scalar arguments.
 * The DaSiamRPN / Nano / Vit trackers are DNN-based: their create() reads
 * model files through cv::dnn::readNet, which throws (reported through
 * cvk_last_error(), NULL handle returned) when the model files are missing.
 */
#ifndef OPENCV_KMP_VIDEO2_H
#define OPENCV_KMP_VIDEO2_H

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to a cv::Ptr<cv::Tracker> of any concrete type. */
typedef struct cvk_tracker cvk_tracker_t;

/** Creates a MIL tracker (cv::TrackerMIL::create) from expanded Params. */
cvk_tracker_t *cvk_tracker_mil_create(
    float samplerInitInRadius, int samplerInitMaxNegNum,
    float samplerSearchWinSize, float samplerTrackInRadius,
    int samplerTrackMaxPosNum, int samplerTrackMaxNegNum,
    int featureSetNumFeatures);

/** Creates a DaSiamRPN tracker; `model`/`kernel_cls1`/`kernel_r1` are model
 *  file paths. NULL when the models cannot be loaded. */
cvk_tracker_t *cvk_tracker_dasiamrpn_create(
    const char *model, const char *kernel_cls1, const char *kernel_r1,
    int backend, int target);

/** Creates a Nano tracker; `backbone`/`neckhead` are model file paths.
 *  NULL when the models cannot be loaded. */
cvk_tracker_t *cvk_tracker_nano_create(
    const char *backbone, const char *neckhead, int backend, int target);

/** Creates a VIT tracker; `net` is the model file path. NULL when the model
 *  cannot be loaded. */
cvk_tracker_t *cvk_tracker_vit_create(
    const char *net, int backend, int target,
    cvk_scalar_t meanvalue, cvk_scalar_t stdvalue,
    float tracking_score_threshold);

/** cv::Tracker::init: initializes the tracker with the initial frame and
 *  bounding box. No-op (with recorded error) on invalid handles. */
void cvk_tracker_init(cvk_tracker_t *h, const cvk_mat_t *image,
                      cvk_rect_t bounding_box);

/** cv::Tracker::update: writes the estimated bounding box into
 *  *out_bounding_box (always, zeroed first) and returns 1 when the target
 *  was located, 0 otherwise. */
int cvk_tracker_update(cvk_tracker_t *h, const cvk_mat_t *image,
                       cvk_rect_t *out_bounding_box);

/** cv::Tracker::getTrackingScore; -1 when no score is available. */
float cvk_tracker_get_tracking_score(const cvk_tracker_t *h);

/** Frees the handle (exactly once). */
void cvk_tracker_release(cvk_tracker_t *h);

#ifdef __cplusplus
}
#endif
#endif /* OPENCV_KMP_VIDEO2_H */

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
 * cvk_ C ABI implementation for the "video2" slice: long-term object
 * trackers (cv::TrackerMIL / TrackerDaSiamRPN / TrackerNano / TrackerVit).
 * Every exported function is noexcept: cv::Exception is caught and reported
 * through cvk_last_error(); create() returns NULL and update() returns 0 on
 * failure.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_video2.h"
#include <opencv2/video/tracking.hpp>

/** Opaque tracker handle backing cvk_tracker_t; holds any concrete tracker. */
struct cvk_tracker { cv::Ptr<cv::Tracker> ptr; };

namespace {

thread_local std::string g_last_error;

void record_error(const char *message) {
    g_last_error = message != nullptr ? message : "unknown error";
}

template <typename F>
auto guarded(F &&body) -> decltype(body()) {
    try {
        return body();
    } catch (const cv::Exception &e) {
        record_error(e.what());
    } catch (const std::exception &e) {
        record_error(e.what());
    } catch (...) {
        record_error("unknown native error");
    }
    return decltype(body())();
}

cv::Mat *require(cvk_mat_t *mat) {
    if (mat == nullptr) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<cv::Mat *>(mat);
}

cv::Rect cv_rect(cvk_rect_t r) { return {r.x, r.y, r.width, r.height}; }

} /* namespace */

extern "C" {

cvk_tracker_t *cvk_tracker_mil_create(
    float samplerInitInRadius, int samplerInitMaxNegNum,
    float samplerSearchWinSize, float samplerTrackInRadius,
    int samplerTrackMaxPosNum, int samplerTrackMaxNegNum,
    int featureSetNumFeatures) {
    return guarded([&]() -> cvk_tracker_t * {
        cv::TrackerMIL::Params params;
        params.samplerInitInRadius = samplerInitInRadius;
        params.samplerInitMaxNegNum = samplerInitMaxNegNum;
        params.samplerSearchWinSize = samplerSearchWinSize;
        params.samplerTrackInRadius = samplerTrackInRadius;
        params.samplerTrackMaxPosNum = samplerTrackMaxPosNum;
        params.samplerTrackMaxNegNum = samplerTrackMaxNegNum;
        params.featureSetNumFeatures = featureSetNumFeatures;
        auto *h = new cvk_tracker;
        h->ptr = cv::TrackerMIL::create(params);
        return reinterpret_cast<cvk_tracker_t *>(h);
    });
}

cvk_tracker_t *cvk_tracker_dasiamrpn_create(
    const char *model, const char *kernel_cls1, const char *kernel_r1,
    int backend, int target) {
    return guarded([&]() -> cvk_tracker_t * {
        cv::TrackerDaSiamRPN::Params params;
        params.model = model != nullptr ? model : "";
        params.kernel_cls1 = kernel_cls1 != nullptr ? kernel_cls1 : "";
        params.kernel_r1 = kernel_r1 != nullptr ? kernel_r1 : "";
        params.backend = backend;
        params.target = target;
        auto *h = new cvk_tracker;
        h->ptr = cv::TrackerDaSiamRPN::create(params);
        return reinterpret_cast<cvk_tracker_t *>(h);
    });
}

cvk_tracker_t *cvk_tracker_nano_create(
    const char *backbone, const char *neckhead, int backend, int target) {
    return guarded([&]() -> cvk_tracker_t * {
        cv::TrackerNano::Params params;
        params.backbone = backbone != nullptr ? backbone : "";
        params.neckhead = neckhead != nullptr ? neckhead : "";
        params.backend = backend;
        params.target = target;
        auto *h = new cvk_tracker;
        h->ptr = cv::TrackerNano::create(params);
        return reinterpret_cast<cvk_tracker_t *>(h);
    });
}

cvk_tracker_t *cvk_tracker_vit_create(
    const char *net, int backend, int target,
    cvk_scalar_t meanvalue, cvk_scalar_t stdvalue,
    float tracking_score_threshold) {
    return guarded([&]() -> cvk_tracker_t * {
        cv::TrackerVit::Params params;
        params.net = net != nullptr ? net : "";
        params.backend = backend;
        params.target = target;
        params.meanvalue = cv::Scalar(meanvalue.v0, meanvalue.v1,
                                      meanvalue.v2, meanvalue.v3);
        params.stdvalue = cv::Scalar(stdvalue.v0, stdvalue.v1,
                                     stdvalue.v2, stdvalue.v3);
        params.tracking_score_threshold = tracking_score_threshold;
        auto *h = new cvk_tracker;
        h->ptr = cv::TrackerVit::create(params);
        return reinterpret_cast<cvk_tracker_t *>(h);
    });
}

void cvk_tracker_init(cvk_tracker_t *h, const cvk_mat_t *image,
                      cvk_rect_t bounding_box) {
    auto *p = reinterpret_cast<cvk_tracker *>(h);
    if (p == nullptr || !p->ptr) {
        record_error("null tracker handle");
        return;
    }
    cv::Mat *img = require(const_cast<cvk_mat_t *>(image));
    if (img == nullptr) {
        return;
    }
    guarded([&]() -> int {
        p->ptr->init(*img, cv_rect(bounding_box));
        return 0;
    });
}

int cvk_tracker_update(cvk_tracker_t *h, const cvk_mat_t *image,
                       cvk_rect_t *out_bounding_box) {
    auto *p = reinterpret_cast<cvk_tracker *>(h);
    if (p == nullptr || !p->ptr) {
        record_error("null tracker handle");
        return 0;
    }
    cv::Mat *img = require(const_cast<cvk_mat_t *>(image));
    if (img == nullptr) {
        return 0;
    }
    if (out_bounding_box != nullptr) {
        *out_bounding_box = cvk_rect_t{0, 0, 0, 0};
    }
    return guarded([&]() -> int {
        cv::Rect rect;
        const bool ok = p->ptr->update(*img, rect);
        if (out_bounding_box != nullptr) {
            out_bounding_box->x = rect.x;
            out_bounding_box->y = rect.y;
            out_bounding_box->width = rect.width;
            out_bounding_box->height = rect.height;
        }
        return ok ? 1 : 0;
    });
}

float cvk_tracker_get_tracking_score(const cvk_tracker_t *h) {
    auto *p = reinterpret_cast<cvk_tracker *>(const_cast<cvk_tracker_t *>(h));
    if (p == nullptr || !p->ptr) {
        record_error("null tracker handle");
        return -1.0f;
    }
    return guarded([&]() -> float { return p->ptr->getTrackingScore(); });
}

void cvk_tracker_release(cvk_tracker_t *h) {
    delete reinterpret_cast<cvk_tracker *>(h);
}

} /* extern "C" */

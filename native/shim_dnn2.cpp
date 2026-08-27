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
 * cvk_ shim for the high-level cv::dnn model wrappers. Every model class is
 * a value type deriving from cv::dnn::Model; one cvk_model_t handle stores a
 * std::variant holding whichever concrete model was created, so base-Model
 * operations and per-class operations share a single handle type and one
 * cvk_model_release() deletes the right object.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_dnn2.h"
#include <opencv2/dnn.hpp>

#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

/* Identical definition to the DnnCore slice's cvk_net (same complete type,
 * legal under the ODR because the two definitions are token-for-token the
 * same). */
struct cvk_net { cv::Ptr<cv::dnn::Net> ptr; };

/* One handle type for every cv::dnn model wrapper; the active alternative
 * is created via placement-new in model_new(). Declared at file scope so the
 * struct tag matches the header's `typedef struct cvk_model cvk_model_t;`
 * (an anonymous-namespace definition would be ambiguous with it). */
using model_variant = std::variant<
    cv::dnn::Model,
    cv::dnn::ClassificationModel,
    cv::dnn::DetectionModel,
    cv::dnn::KeypointsModel,
    cv::dnn::SegmentationModel,
    cv::dnn::TextDetectionModel_DB,
    cv::dnn::TextDetectionModel_EAST,
    cv::dnn::TextRecognitionModel>;

struct cvk_model { model_variant obj; };

namespace {

thread_local std::string g_dnn2_str;
thread_local std::string g_dnn2_last_error;

void record_error(const char *message) {
    g_dnn2_last_error = message != nullptr ? message : "unknown error";
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

const cv::Mat *require_const(const cvk_mat_t *mat) {
    return require(const_cast<cvk_mat_t *>(mat));
}

cvk_scalar_t scalar_of(const cv::Scalar &s) {
    return {s[0], s[1], s[2], s[3]};
}

cv::Scalar cv_scalar(cvk_scalar_t s) {
    return cv::Scalar(s.v0, s.v1, s.v2, s.v3);
}

cv::dnn::Net *net_of(cvk_net_t *h) {
    if (h == nullptr) {
        record_error("null Net handle");
        return nullptr;
    }
    return reinterpret_cast<cvk_net *>(h)->ptr.get();
}

// ---------------------------------------------------------------------------
// model handle: one struct, one variant, every cv::dnn model class inside
// ---------------------------------------------------------------------------

/* Placement-news the variant so the deprecated default Model() ctor is
 * never called; delete (via cvk_model_release) destroys the active
 * alternative properly. */
template <typename T, typename... Args>
cvk_model *model_new(Args &&...args) {
    void *mem = std::malloc(sizeof(cvk_model));
    if (mem == nullptr) throw std::bad_alloc();
    cvk_model *h = static_cast<cvk_model *>(mem);
    new (&h->obj) model_variant(std::in_place_type<T>, std::forward<Args>(args)...);
    return h;
}

/* The model object at offset 0 is also a cv::dnn::Model base subobject for
 * every alternative, so a visitor upcast yields the base API. */
cv::dnn::Model *model_base(cvk_model_t *h) {
    if (h == nullptr) {
        record_error("null model handle");
        return nullptr;
    }
    return &std::visit([](auto &m) -> cv::dnn::Model & { return m; },
                       reinterpret_cast<cvk_model *>(h)->obj);
}

const cv::dnn::Model *model_base_const(const cvk_model_t *h) {
    if (h == nullptr) {
        record_error("null model handle");
        return nullptr;
    }
    return &std::visit([](const auto &m) -> const cv::dnn::Model & { return m; },
                       reinterpret_cast<const cvk_model *>(h)->obj);
}

/* The C ABI treats handles as opaque pointers; getters and setters both
 * arrive with const handles, and several OpenCV getters are not const, so
 * the cast is intentionally const-unfriendly. */
template <typename T>
T *model_as(const cvk_model_t *h) {
    if (h == nullptr) {
        record_error("null model handle");
        return nullptr;
    }
    auto *p = std::get_if<T>(&reinterpret_cast<cvk_model *>(
            const_cast<cvk_model_t *>(h))->obj);
    if (p == nullptr) record_error("model handle has a different type");
    return p;
}

template <typename T>
const T *model_as_const(const cvk_model_t *h) {
    return model_as<T>(h);
}

const cv::dnn::TextDetectionModel *text_detection_base(const cvk_model_t *h) {
    if (h == nullptr) {
        record_error("null model handle");
        return nullptr;
    }
    return std::visit(
        [](const auto &m) -> const cv::dnn::TextDetectionModel * {
            if constexpr (std::is_base_of_v<cv::dnn::TextDetectionModel,
                                            std::decay_t<decltype(m)>>)
                return &m;
            else
                return nullptr;
        },
        reinterpret_cast<const cvk_model *>(h)->obj);
}

// ---------------------------------------------------------------------------
// little-endian flat wire buffers (freed with cvk_free_buffer)
// ---------------------------------------------------------------------------

static void put_u32le(unsigned char *p, unsigned int v) {
    p[0] = static_cast<unsigned char>(v & 0xFFu);
    p[1] = static_cast<unsigned char>((v >> 8) & 0xFFu);
    p[2] = static_cast<unsigned char>((v >> 16) & 0xFFu);
    p[3] = static_cast<unsigned char>((v >> 24) & 0xFFu);
}

struct flat_buf {
    std::vector<unsigned char> data;

    void i32(int v) {
        unsigned char b[4];
        put_u32le(b, static_cast<unsigned int>(v));
        data.insert(data.end(), b, b + 4);
    }

    void f32(float v) {
        unsigned int bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        unsigned char b[4];
        put_u32le(b, bits);
        data.insert(data.end(), b, b + 4);
    }

    void raw(const unsigned char *p, size_t n) {
        data.insert(data.end(), p, p + n);
    }

    unsigned char *release(size_t *out_len) {
        // malloc(0) may return NULL, which callers treat as failure; hand out
        // a 1-byte allocation for empty payloads so "no results" stays
        // distinguishable from an error.
        const size_t n = data.size();
        auto *buf = static_cast<unsigned char *>(std::malloc(n > 0 ? n : 1));
        if (buf == nullptr) throw std::bad_alloc();
        if (n > 0) std::memcpy(buf, data.data(), n);
        if (out_len != nullptr) *out_len = n;
        return buf;
    }
};

/* [u32 count][per string: u32 len + len bytes] */
unsigned char *encode_strings(const std::vector<std::string> &strings, size_t *out_len) {
    flat_buf buf;
    buf.i32(static_cast<int>(strings.size()));
    for (const std::string &s : strings) {
        buf.i32(static_cast<int>(s.size()));
        buf.raw(reinterpret_cast<const unsigned char *>(s.data()), s.size());
    }
    return buf.release(out_len);
}

}  // namespace

extern "C" {

// =========================================================================
// Model base
// =========================================================================

cvk_model_t *cvk_model_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::Model>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_model_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::Model>(*n));
    });
}

int cvk_model_set_input_size(cvk_model_t *h, int width, int height) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setInputSize(width, height); return 1; });
}

int cvk_model_set_input_mean(cvk_model_t *h, cvk_scalar_t mean) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setInputMean(cv_scalar(mean)); return 1; });
}

int cvk_model_set_input_scale(cvk_model_t *h, cvk_scalar_t scale) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setInputScale(cv_scalar(scale)); return 1; });
}

int cvk_model_set_input_crop(cvk_model_t *h, int crop) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setInputCrop(crop != 0); return 1; });
}

int cvk_model_set_input_swap_rb(cvk_model_t *h, int swap_rb) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setInputSwapRB(swap_rb != 0); return 1; });
}

int cvk_model_set_output_names(cvk_model_t *h, const char *const *names, int count) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    if (names == nullptr || count < 0) {
        record_error("bad output names");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::String> out;
        out.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            if (names[i] == nullptr) {
                record_error("null output name");
                return 0;
            }
            out.emplace_back(names[i]);
        }
        m->setOutputNames(out);
        return 1;
    });
}

int cvk_model_set_input_params(cvk_model_t *h, double scale, int width, int height,
                               cvk_scalar_t mean, int swap_rb, int crop) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] {
        m->setInputParams(scale, cv::Size(width, height), cv_scalar(mean), swap_rb != 0,
                          crop != 0);
        return 1;
    });
}

cvk_mat_t **cvk_model_predict(const cvk_model_t *h, const cvk_mat_t *frame,
                              size_t *out_count) {
    if (out_count != nullptr) *out_count = 0;
    const cv::dnn::Model *m = model_base_const(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t ** {
        std::vector<cv::Mat> outs;
        m->predict(*f, outs);
        auto **handles = static_cast<cvk_mat_t **>(std::malloc(outs.size() * sizeof(cvk_mat_t *)));
        if (handles == nullptr && !outs.empty()) throw std::bad_alloc();
        for (size_t i = 0; i < outs.size(); ++i) {
            handles[i] = reinterpret_cast<cvk_mat_t *>(new cv::Mat(outs[i]));
        }
        if (out_count != nullptr) *out_count = outs.size();
        return handles;
    });
}

int cvk_model_set_preferable_backend(cvk_model_t *h, int backend_id) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] {
        m->setPreferableBackend(static_cast<cv::dnn::Backend>(backend_id));
        return 1;
    });
}

int cvk_model_set_preferable_target(cvk_model_t *h, int target_id) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] {
        m->setPreferableTarget(static_cast<cv::dnn::Target>(target_id));
        return 1;
    });
}

int cvk_model_enable_winograd(cvk_model_t *h, int use_winograd) {
    cv::dnn::Model *m = model_base(h);
    if (m == nullptr) return 0;
    return guarded([&] {
        m->enableWinograd(use_winograd != 0);
        return 1;
    });
}

void cvk_model_release(cvk_model_t *h) {
    delete reinterpret_cast<cvk_model *>(h);
}

// =========================================================================
// ClassificationModel
// =========================================================================

cvk_model_t *cvk_classification_model_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::ClassificationModel>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_classification_model_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(
            model_new<cv::dnn::ClassificationModel>(*n));
    });
}

int cvk_classification_model_set_enable_softmax_post_processing(cvk_model_t *h, int enable) {
    auto *m = model_as<cv::dnn::ClassificationModel>(h);
    if (m == nullptr) return 0;
    return guarded([&] {
        m->setEnableSoftmaxPostProcessing(enable != 0);
        return 1;
    });
}

int cvk_classification_model_get_enable_softmax_post_processing(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::ClassificationModel>(h);
    if (m == nullptr) return 0;
    return guarded([&] { return m->getEnableSoftmaxPostProcessing() ? 1 : 0; });
}

int cvk_classification_model_classify(const cvk_model_t *h, const cvk_mat_t *frame,
                                      int *class_id, float *confidence) {
    auto *m = model_as<cv::dnn::ClassificationModel>(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr || class_id == nullptr || confidence == nullptr) {
        if (m != nullptr && f != nullptr) record_error("null classify out-param");
        return 0;
    }
    return guarded([&]() -> int {
        std::pair<int, float> r = m->classify(*f);
        *class_id = r.first;
        *confidence = r.second;
        return 1;
    });
}

// =========================================================================
// DetectionModel
// =========================================================================

cvk_model_t *cvk_detection_model_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::DetectionModel>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_detection_model_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::DetectionModel>(*n));
    });
}

int cvk_detection_model_set_nms_across_classes(cvk_model_t *h, int value) {
    auto *m = model_as<cv::dnn::DetectionModel>(h);
    if (m == nullptr) return 0;
    return guarded([&] {
        m->setNmsAcrossClasses(value != 0);
        return 1;
    });
}

int cvk_detection_model_get_nms_across_classes(const cvk_model_t *h) {
    auto *m = model_as<cv::dnn::DetectionModel>(h);
    if (m == nullptr) return 0;
    return guarded([&] { return m->getNmsAcrossClasses() ? 1 : 0; });
}

unsigned char *cvk_detection_model_detect(const cvk_model_t *h, const cvk_mat_t *frame,
                                          float conf_threshold, float nms_threshold,
                                          size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    auto *m = model_as<cv::dnn::DetectionModel>(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        m->detect(*f, class_ids, confidences, boxes, conf_threshold, nms_threshold);
        flat_buf buf;
        buf.i32(static_cast<int>(class_ids.size()));
        for (int id : class_ids) buf.i32(id);
        for (float c : confidences) buf.f32(c);
        for (const cv::Rect &r : boxes) {
            buf.i32(r.x);
            buf.i32(r.y);
            buf.i32(r.width);
            buf.i32(r.height);
        }
        return buf.release(out_len);
    });
}

// =========================================================================
// KeypointsModel
// =========================================================================

cvk_model_t *cvk_keypoints_model_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::KeypointsModel>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_keypoints_model_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::KeypointsModel>(*n));
    });
}

unsigned char *cvk_keypoints_model_estimate(const cvk_model_t *h, const cvk_mat_t *frame,
                                            float thresh, size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    auto *m = model_as<cv::dnn::KeypointsModel>(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<cv::Point2f> points = m->estimate(*f, thresh);
        flat_buf buf;
        buf.i32(static_cast<int>(points.size()));
        for (const cv::Point2f &pt : points) {
            buf.f32(pt.x);
            buf.f32(pt.y);
        }
        return buf.release(out_len);
    });
}

// =========================================================================
// SegmentationModel
// =========================================================================

cvk_model_t *cvk_segmentation_model_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::SegmentationModel>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_segmentation_model_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::SegmentationModel>(*n));
    });
}

cvk_mat_t *cvk_segmentation_model_segment(const cvk_model_t *h, const cvk_mat_t *frame) {
    auto *m = model_as<cv::dnn::SegmentationModel>(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *mask = new cv::Mat();
        m->segment(*f, *mask);
        return reinterpret_cast<cvk_mat_t *>(mask);
    });
}

// =========================================================================
// TextDetectionModel base
// =========================================================================

unsigned char *cvk_text_detection_model_detect(const cvk_model_t *h, const cvk_mat_t *frame,
                                               size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    const cv::dnn::TextDetectionModel *m = text_detection_base(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr) {
        if (m == nullptr && h != nullptr) record_error("model handle is not a text detection model");
        return nullptr;
    }
    return guarded([&]() -> unsigned char * {
        std::vector<std::vector<cv::Point>> detections;
        std::vector<float> confidences;
        m->detect(*f, detections, confidences);
        flat_buf buf;
        buf.i32(static_cast<int>(detections.size()));
        for (const auto &quad : detections) {
            buf.i32(static_cast<int>(quad.size()));
            for (const cv::Point &pt : quad) {
                buf.i32(pt.x);
                buf.i32(pt.y);
            }
        }
        for (float c : confidences) buf.f32(c);
        return buf.release(out_len);
    });
}

unsigned char *cvk_text_detection_model_detect_text_rectangles(const cvk_model_t *h,
                                                               const cvk_mat_t *frame,
                                                               size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    const cv::dnn::TextDetectionModel *m = text_detection_base(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr) {
        if (m == nullptr && h != nullptr) record_error("model handle is not a text detection model");
        return nullptr;
    }
    return guarded([&]() -> unsigned char * {
        std::vector<cv::RotatedRect> detections;
        std::vector<float> confidences;
        m->detectTextRectangles(*f, detections, confidences);
        flat_buf buf;
        buf.i32(static_cast<int>(detections.size()));
        for (const cv::RotatedRect &r : detections) {
            buf.f32(r.center.x);
            buf.f32(r.center.y);
            buf.f32(r.size.width);
            buf.f32(r.size.height);
            buf.f32(r.angle);
        }
        for (float c : confidences) buf.f32(c);
        return buf.release(out_len);
    });
}

// =========================================================================
// TextDetectionModel_DB
// =========================================================================

cvk_model_t *cvk_text_detection_model_db_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::TextDetectionModel_DB>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_text_detection_model_db_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(
            model_new<cv::dnn::TextDetectionModel_DB>(*n));
    });
}

int cvk_text_detection_model_db_set_binary_threshold(cvk_model_t *h, float value) {
    auto *m = model_as<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setBinaryThreshold(value); return 1; });
}

float cvk_text_detection_model_db_get_binary_threshold(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0.0f;
    return guarded([&] { return m->getBinaryThreshold(); });
}

int cvk_text_detection_model_db_set_polygon_threshold(cvk_model_t *h, float value) {
    auto *m = model_as<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setPolygonThreshold(value); return 1; });
}

float cvk_text_detection_model_db_get_polygon_threshold(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0.0f;
    return guarded([&] { return m->getPolygonThreshold(); });
}

int cvk_text_detection_model_db_set_unclip_ratio(cvk_model_t *h, double value) {
    auto *m = model_as<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setUnclipRatio(value); return 1; });
}

double cvk_text_detection_model_db_get_unclip_ratio(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0.0;
    return guarded([&] { return m->getUnclipRatio(); });
}

int cvk_text_detection_model_db_set_max_candidates(cvk_model_t *h, int value) {
    auto *m = model_as<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setMaxCandidates(value); return 1; });
}

int cvk_text_detection_model_db_get_max_candidates(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::TextDetectionModel_DB>(h);
    if (m == nullptr) return 0;
    return guarded([&] { return m->getMaxCandidates(); });
}

// =========================================================================
// TextDetectionModel_EAST
// =========================================================================

cvk_model_t *cvk_text_detection_model_east_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::TextDetectionModel_EAST>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_text_detection_model_east_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(
            model_new<cv::dnn::TextDetectionModel_EAST>(*n));
    });
}

int cvk_text_detection_model_east_set_confidence_threshold(cvk_model_t *h, float value) {
    auto *m = model_as<cv::dnn::TextDetectionModel_EAST>(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setConfidenceThreshold(value); return 1; });
}

float cvk_text_detection_model_east_get_confidence_threshold(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::TextDetectionModel_EAST>(h);
    if (m == nullptr) return 0.0f;
    return guarded([&] { return m->getConfidenceThreshold(); });
}

int cvk_text_detection_model_east_set_nms_threshold(cvk_model_t *h, float value) {
    auto *m = model_as<cv::dnn::TextDetectionModel_EAST>(h);
    if (m == nullptr) return 0;
    return guarded([&] { m->setNMSThreshold(value); return 1; });
}

float cvk_text_detection_model_east_get_nms_threshold(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::TextDetectionModel_EAST>(h);
    if (m == nullptr) return 0.0f;
    return guarded([&] { return m->getNMSThreshold(); });
}

// =========================================================================
// TextRecognitionModel
// =========================================================================

cvk_model_t *cvk_text_recognition_model_create(const char *model, const char *config) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(model_new<cv::dnn::TextRecognitionModel>(
            std::string(model), config != nullptr ? std::string(config) : std::string()));
    });
}

cvk_model_t *cvk_text_recognition_model_create_from_net(cvk_net_t *net) {
    cv::dnn::Net *n = net_of(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_model_t * {
        return reinterpret_cast<cvk_model_t *>(
            model_new<cv::dnn::TextRecognitionModel>(*n));
    });
}

int cvk_text_recognition_model_set_decode_type(cvk_model_t *h, const char *decode_type) {
    auto *m = model_as<cv::dnn::TextRecognitionModel>(h);
    if (m == nullptr) return 0;
    if (decode_type == nullptr) {
        record_error("null decode type");
        return 0;
    }
    return guarded([&] { m->setDecodeType(decode_type); return 1; });
}

const char *cvk_text_recognition_model_get_decode_type(const cvk_model_t *h) {
    const auto *m = model_as_const<cv::dnn::TextRecognitionModel>(h);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> const char * {
        g_dnn2_str = m->getDecodeType();
        return g_dnn2_str.c_str();
    });
}

int cvk_text_recognition_model_set_decode_opts_ctc_prefix_beam_search(cvk_model_t *h,
                                                                      int beam_size,
                                                                      int voc_prune_size) {
    auto *m = model_as<cv::dnn::TextRecognitionModel>(h);
    if (m == nullptr) return 0;
    return guarded([&] {
        m->setDecodeOptsCTCPrefixBeamSearch(beam_size, voc_prune_size);
        return 1;
    });
}

int cvk_text_recognition_model_set_vocabulary(cvk_model_t *h, const char *const *vocab,
                                              int count) {
    auto *m = model_as<cv::dnn::TextRecognitionModel>(h);
    if (m == nullptr) return 0;
    if (vocab == nullptr || count < 0) {
        record_error("bad vocabulary");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<std::string> words;
        words.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            if (vocab[i] == nullptr) {
                record_error("null vocabulary entry");
                return 0;
            }
            words.emplace_back(vocab[i]);
        }
        m->setVocabulary(words);
        return 1;
    });
}

unsigned char *cvk_text_recognition_model_get_vocabulary(const cvk_model_t *h,
                                                         size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    const auto *m = model_as_const<cv::dnn::TextRecognitionModel>(h);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        return encode_strings(m->getVocabulary(), out_len);
    });
}

const char *cvk_text_recognition_model_recognize(const cvk_model_t *h, const cvk_mat_t *frame) {
    auto *m = model_as<cv::dnn::TextRecognitionModel>(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr) return nullptr;
    return guarded([&]() -> const char * {
        g_dnn2_str = m->recognize(*f);
        return g_dnn2_str.c_str();
    });
}

unsigned char *cvk_text_recognition_model_recognize_rois(const cvk_model_t *h,
                                                         const cvk_mat_t *frame,
                                                         const int *roi_rects,
                                                         int roi_count,
                                                         size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    auto *m = model_as<cv::dnn::TextRecognitionModel>(h);
    const cv::Mat *f = require_const(frame);
    if (m == nullptr || f == nullptr || roi_rects == nullptr || roi_count < 0) {
        if (m != nullptr && f != nullptr && roi_rects == nullptr) record_error("null ROI rects");
        return nullptr;
    }
    return guarded([&]() -> unsigned char * {
        // Mirrors TextRecognitionModel_Impl::recognize(frame, rois, results):
        // each ROI is cropped and recognized with the single-frame path
        // (the vector<Mat> overload's copyTo path is not wired up in this
        // OpenCV build, so going through it would throw StsNotImplemented).
        std::vector<std::string> results;
        const cv::Mat input = *f;
        for (int i = 0; i < roi_count; ++i) {
            const int *r = roi_rects + i * 4;
            cv::Rect roi(r[0], r[1], r[2], r[3]);
            results.push_back(m->recognize(input(roi)));
        }
        return encode_strings(results, out_len);
    });
}

} /* extern "C" */

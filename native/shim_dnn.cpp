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
 * Implementation of the dnn cvk_ C ABI (see opencv_kmp_dnn.h).
 *
 * Every exported function is noexcept via the local `guarded` helper:
 * cv::Exception is caught and reported through cvk_last_error(), failures
 * return NULL / 0 / false. Model-file readers never crash on missing files.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_dnn.h"

#include <opencv2/dnn.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

/** Opaque handles; complete the types the header forward-declared. */
struct cvk_net { cv::Ptr<cv::dnn::Net> ptr; };
struct cvk_layer { cv::Ptr<cv::dnn::Layer> ptr; };
struct cvk_tokenizer { cv::dnn::Tokenizer tok; };

namespace {

thread_local std::string g_dnn_error;
thread_local std::string g_dnn_str;

void record_error(const char *message) {
    try {
        g_dnn_error = message != nullptr ? message : "unknown error";
    } catch (...) {
    }
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

cvk_net *require_net(const cvk_net_t *net) {
    if (net == nullptr || reinterpret_cast<const cvk_net *>(net)->ptr.empty()) {
        record_error("null or released Net handle");
        return nullptr;
    }
    return reinterpret_cast<cvk_net *>(const_cast<cvk_net_t *>(net));
}

cvk_layer *require_layer(const cvk_layer_t *layer) {
    if (layer == nullptr || reinterpret_cast<const cvk_layer *>(layer)->ptr.empty()) {
        record_error("null or released Layer handle");
        return nullptr;
    }
    return reinterpret_cast<cvk_layer *>(const_cast<cvk_layer_t *>(layer));
}

cvk_tokenizer *require_tokenizer(const cvk_tokenizer_t *tok) {
    if (tok == nullptr) {
        record_error("null Tokenizer handle");
        return nullptr;
    }
    return reinterpret_cast<cvk_tokenizer *>(const_cast<cvk_tokenizer_t *>(tok));
}

cv::Scalar cv_scalar(cvk_scalar_t s) { return {s.v0, s.v1, s.v2, s.v3}; }

/* ---- vector<Mat> wire format: CV_32SC2 Mat of Mat addresses --------------
 * Matches the official Java Converters layout: each row is one 64-bit heap
 * address of a cv::Mat object owned by the caller.
 */

std::vector<cv::Mat> decode_mat_vector(const cv::Mat &addr_mat) {
    std::vector<cv::Mat> out;
    if (addr_mat.empty()) return out;
    if (addr_mat.type() != CV_32SC2 || addr_mat.cols != 1) {
        throw cv::Exception(cv::Error::StsBadArg, "expected CV_32SC2 Mat of Mat addresses",
                            __func__, __FILE__, __LINE__);
    }
    const auto *data = addr_mat.ptr<const int64_t>();
    out.reserve(static_cast<size_t>(addr_mat.rows));
    for (int i = 0; i < addr_mat.rows; ++i) {
        auto *m = reinterpret_cast<cv::Mat *>(data[i]);
        if (m == nullptr) {
            throw cv::Exception(cv::Error::StsBadArg, "null Mat address in vector",
                                __func__, __FILE__, __LINE__);
        }
        out.emplace_back(*m);  // shares the pixel refcount
    }
    return out;
}

/** Encodes into a fresh CV_32SC2 Mat; member Mats are heap copies owned by the caller. */
cv::Mat encode_mat_vector(const std::vector<cv::Mat> &mats) {
    cv::Mat addr(static_cast<int>(mats.size()), 1, CV_32SC2);
    auto *data = addr.ptr<int64_t>();
    for (size_t i = 0; i < mats.size(); ++i) {
        data[i] = reinterpret_cast<int64_t>(new cv::Mat(mats[i]));
    }
    return addr;
}

/* ---- std::vector<std::string> wire format --------------------------------
 * malloc'd flat buffer: [u32le count][per string: u32le byte_len + UTF-8].
 * Released by cvk_free_buffer (std::free).
 */

unsigned char *encode_strings(const std::vector<std::string> &strings, size_t *out_len) {
    size_t total = 4;
    for (const auto &s : strings) total += 4 + s.size();
    auto *buf = static_cast<unsigned char *>(std::malloc(total));
    if (buf == nullptr) throw std::bad_alloc();
    auto *p = buf;
    const uint32_t count = static_cast<uint32_t>(strings.size());
    std::memcpy(p, &count, 4);
    p += 4;
    for (const auto &s : strings) {
        const uint32_t len = static_cast<uint32_t>(s.size());
        std::memcpy(p, &len, 4);
        p += 4;
        if (len) std::memcpy(p, s.data(), len);
        p += len;
    }
    if (out_len != nullptr) *out_len = total;
    return buf;
}

std::vector<std::string> decode_strings(const unsigned char *flat, size_t len) {
    std::vector<std::string> out;
    if (flat == nullptr || len < 4) {
        throw cv::Exception(cv::Error::StsParseError, "malformed string buffer",
                            __func__, __FILE__, __LINE__);
    }
    uint32_t count;
    std::memcpy(&count, flat, 4);
    size_t off = 4;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (off + 4 > len) throw cv::Exception(cv::Error::StsParseError, "truncated string buffer",
                                               __func__, __FILE__, __LINE__);
        uint32_t slen;
        std::memcpy(&slen, flat + off, 4);
        off += 4;
        if (off + slen > len) throw cv::Exception(cv::Error::StsParseError, "truncated string buffer",
                                                  __func__, __FILE__, __LINE__);
        out.emplace_back(reinterpret_cast<const char *>(flat + off), slen);
        off += slen;
    }
    return out;
}

/* ---- int vector helpers -------------------------------------------------- */

std::vector<int> decode_int_mat(const cv::Mat &m, const char *what) {
    std::vector<int> out;
    if (m.empty()) return out;
    if (m.type() != CV_32S) {
        throw cv::Exception(cv::Error::StsBadArg,
                            std::string(what) + " must be a CV_32S Mat", __func__, __FILE__, __LINE__);
    }
    const auto *p = m.ptr<const int>();
    out.assign(p, p + m.total());
    return out;
}

std::vector<float> decode_float_mat(const cv::Mat &m, const char *what) {
    std::vector<float> out;
    if (m.empty()) return out;
    if (m.type() != CV_32F) {
        throw cv::Exception(cv::Error::StsBadArg,
                            std::string(what) + " must be a CV_32F Mat", __func__, __FILE__, __LINE__);
    }
    const auto *p = m.ptr<const float>();
    out.assign(p, p + m.total());
    return out;
}

std::vector<uchar> decode_byte_mat(const cv::Mat &m, const char *what) {
    std::vector<uchar> out;
    if (m.empty()) return out;
    if (m.type() != CV_8UC1) {
        throw cv::Exception(cv::Error::StsBadArg,
                            std::string(what) + " must be a CV_8UC1 Mat", __func__, __FILE__, __LINE__);
    }
    const auto *p = m.ptr<const uchar>();
    out.assign(p, p + m.total());
    return out;
}

/** Overwrites `dst` with an Nx1 CV_32S Mat holding the given ints (in place). */
void fill_int_mat(cv::Mat &dst, const std::vector<int> &ints) {
    dst.create(static_cast<int>(ints.size()), 1, CV_32S);
    if (!ints.empty()) {
        std::memcpy(dst.data, ints.data(), ints.size() * sizeof(int));
    }
}

/** Overwrites `dst` with an Nx1 CV_32F Mat holding the given floats (in place). */
void fill_float_mat(cv::Mat &dst, const std::vector<float> &floats) {
    dst.create(static_cast<int>(floats.size()), 1, CV_32F);
    if (!floats.empty()) {
        std::memcpy(dst.data, floats.data(), floats.size() * sizeof(float));
    }
}

/* ---- MatShape lists ------------------------------------------------------ */

/** vector<MatShape> (list of CV_32S Mats) from a CV_32SC2 Mat-of-Mats. */
std::vector<cv::MatShape> decode_shape_list(const cv::Mat &addr_mat) {
    std::vector<cv::MatShape> out;
    for (const cv::Mat &m : decode_mat_vector(addr_mat)) {
        // OpenCV 5's MatShape is a struct (not a vector<int> alias); the
        // vector constructor is explicit.
        out.push_back(cv::MatShape(decode_int_mat(m, "shape")));
    }
    return out;
}

}  // namespace

extern "C" {

/* =========================================================================
 * dnn statics
 * ========================================================================= */

cvk_net_t *cvk_dnn_read_net(const char *model, const char *config,
                            const char *framework, int engine) {
    return guarded([&]() -> cvk_net_t * {
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNet(
            model != nullptr ? model : "", config != nullptr ? config : "",
            framework != nullptr ? framework : "", engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_buffer(const char *framework,
                                   const cvk_mat_t *buffer_model,
                                   const cvk_mat_t *buffer_config,
                                   int engine) {
    return guarded([&]() -> cvk_net_t * {
        const cv::Mat &model = *require_const(buffer_model);
        std::vector<uchar> model_bytes = decode_byte_mat(model, "model buffer");
        std::vector<uchar> config_bytes;
        if (buffer_config != nullptr) {
            config_bytes = decode_byte_mat(*require_const(buffer_config), "config buffer");
        }
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNet(
            framework != nullptr ? framework : "", model_bytes, config_bytes, engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_onnx(const char *onnx_file, int engine) {
    return guarded([&]() -> cvk_net_t * {
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNetFromONNX(cv::String(onnx_file), engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_onnx_buffer(const cvk_mat_t *buffer, int engine) {
    return guarded([&]() -> cvk_net_t * {
        const cv::Mat &m = *require_const(buffer);
        std::vector<uchar> bytes = decode_byte_mat(m, "ONNX buffer");
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNetFromONNX(bytes, engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_tensorflow(const char *model, const char *config,
                                            int engine) {
    return guarded([&]() -> cvk_net_t * {
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNetFromTensorflow(
            model, config != nullptr ? config : "", engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_tensorflow_buffer(const cvk_mat_t *buffer_model,
                                                   const cvk_mat_t *buffer_config,
                                                   int engine) {
    return guarded([&]() -> cvk_net_t * {
        const cv::Mat &model = *require_const(buffer_model);
        std::vector<uchar> model_bytes = decode_byte_mat(model, "model buffer");
        std::vector<uchar> config_bytes;
        if (buffer_config != nullptr) {
            config_bytes = decode_byte_mat(*require_const(buffer_config), "config buffer");
        }
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(
            cv::dnn::readNetFromTensorflow(model_bytes, config_bytes, engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_tflite(const char *model, int engine) {
    return guarded([&]() -> cvk_net_t * {
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNetFromTFLite(cv::String(model), engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_tflite_buffer(const cvk_mat_t *buffer_model, int engine) {
    return guarded([&]() -> cvk_net_t * {
        const cv::Mat &m = *require_const(buffer_model);
        std::vector<uchar> bytes = decode_byte_mat(m, "TFLite buffer");
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNetFromTFLite(bytes, engine)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_model_optimizer(const char *xml, const char *bin) {
    return guarded([&]() -> cvk_net_t * {
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNetFromModelOptimizer(
            xml, bin != nullptr ? bin : "")));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_net_t *cvk_dnn_read_net_from_model_optimizer_buffer(const cvk_mat_t *buffer_config,
                                                        const cvk_mat_t *buffer_weights) {
    return guarded([&]() -> cvk_net_t * {
        const cv::Mat &cfg = *require_const(buffer_config);
        const cv::Mat &w = *require_const(buffer_weights);
        std::vector<uchar> cfg_bytes = decode_byte_mat(cfg, "config buffer");
        std::vector<uchar> w_bytes = decode_byte_mat(w, "weights buffer");
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net(cv::dnn::readNetFromModelOptimizer(cfg_bytes, w_bytes)));
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

cvk_mat_t *cvk_dnn_read_tensor_from_onnx(const char *path) {
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = cv::dnn::readTensorFromONNX(path);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

cvk_mat_t *cvk_dnn_blob_from_image(const cvk_mat_t *image, double scalefactor,
                                   int size_width, int size_height,
                                   cvk_scalar_t mean, int swap_rb, int crop,
                                   int ddepth) {
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = cv::dnn::blobFromImage(*require_const(image), scalefactor,
                                           cv::Size(size_width, size_height),
                                           cv_scalar(mean), swap_rb != 0, crop != 0, ddepth);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

cvk_mat_t *cvk_dnn_blob_from_images(const cvk_mat_t *images, double scalefactor,
                                    int size_width, int size_height,
                                    cvk_scalar_t mean, int swap_rb, int crop,
                                    int ddepth) {
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Mat> imgs = decode_mat_vector(*require_const(images));
        cv::Mat m = cv::dnn::blobFromImages(imgs, scalefactor,
                                            cv::Size(size_width, size_height),
                                            cv_scalar(mean), swap_rb != 0, crop != 0, ddepth);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

cvk_mat_t *cvk_dnn_images_from_blob(const cvk_mat_t *blob) {
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Mat> images;
        cv::dnn::imagesFromBlob(*require_const(blob), images);
        cv::Mat addr = encode_mat_vector(images);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(addr)));
    });
}

namespace {

void nms_boxes_impl(const cv::Mat &bboxes, const cv::Mat &scores,
                    float score_threshold, float nms_threshold,
                    cv::Mat &indices, float eta, int top_k) {
    if (bboxes.type() != CV_64FC4) {
        throw cv::Exception(cv::Error::StsBadArg, "bboxes must be a CV_64FC4 Mat",
                            __func__, __FILE__, __LINE__);
    }
    std::vector<cv::Rect2d> boxes;
    boxes.reserve(static_cast<size_t>(bboxes.rows));
    for (int i = 0; i < bboxes.rows; ++i) {
        const auto *p = bboxes.ptr<const double>(i, 0);
        boxes.push_back(cv::Rect2d(p[0], p[1], p[2], p[3]));
    }
    std::vector<float> score_vec = decode_float_mat(scores, "scores");
    std::vector<int> out;
    cv::dnn::NMSBoxes(boxes, score_vec, score_threshold, nms_threshold, out, eta, top_k);
    fill_int_mat(indices, out);
}

}  // namespace

void cvk_dnn_nms_boxes(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                       float score_threshold, float nms_threshold,
                       cvk_mat_t *indices, float eta, int top_k) {
    guarded([&]() -> int {
        nms_boxes_impl(*require_const(bboxes), *require_const(scores), score_threshold,
                       nms_threshold, *require(indices), eta, top_k);
        return 0;
    });
}

void cvk_dnn_nms_boxes_rotated(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                               float score_threshold, float nms_threshold,
                               cvk_mat_t *indices, float eta, int top_k) {
    guarded([&]() -> int {
        const cv::Mat &b = *require_const(bboxes);
        if (b.type() != CV_MAKETYPE(CV_32F, 5)) {
            throw cv::Exception(cv::Error::StsBadArg, "bboxes must be a CV_32FC5 Mat",
                                __func__, __FILE__, __LINE__);
        }
        std::vector<cv::RotatedRect> boxes;
        boxes.reserve(static_cast<size_t>(b.rows));
        for (int i = 0; i < b.rows; ++i) {
            const auto *p = b.ptr<const float>(i, 0);
            boxes.push_back(cv::RotatedRect(cv::Point2f(p[0], p[1]),
                                            cv::Size2f(p[2], p[3]), p[4]));
        }
        std::vector<float> score_vec = decode_float_mat(*require_const(scores), "scores");
        std::vector<int> out;
        cv::dnn::NMSBoxes(boxes, score_vec, score_threshold, nms_threshold, out, eta, top_k);
        fill_int_mat(*require(indices), out);
        return 0;
    });
}

void cvk_dnn_nms_boxes_batched(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                               const cvk_mat_t *class_ids, float score_threshold,
                               float nms_threshold, cvk_mat_t *indices,
                               float eta, int top_k) {
    guarded([&]() -> int {
        const cv::Mat &b = *require_const(bboxes);
        if (b.type() != CV_64FC4) {
            throw cv::Exception(cv::Error::StsBadArg, "bboxes must be a CV_64FC4 Mat",
                                __func__, __FILE__, __LINE__);
        }
        std::vector<cv::Rect2d> boxes;
        boxes.reserve(static_cast<size_t>(b.rows));
        for (int i = 0; i < b.rows; ++i) {
            const auto *p = b.ptr<const double>(i, 0);
            boxes.push_back(cv::Rect2d(p[0], p[1], p[2], p[3]));
        }
        std::vector<float> score_vec = decode_float_mat(*require_const(scores), "scores");
        std::vector<int> ids = decode_int_mat(*require_const(class_ids), "class_ids");
        std::vector<int> out;
        cv::dnn::NMSBoxesBatched(boxes, score_vec, ids, score_threshold, nms_threshold,
                                 out, eta, top_k);
        fill_int_mat(*require(indices), out);
        return 0;
    });
}

void cvk_dnn_soft_nms_boxes(const cvk_mat_t *bboxes, const cvk_mat_t *scores,
                            cvk_mat_t *updated_scores, float score_threshold,
                            float nms_threshold, cvk_mat_t *indices,
                            int top_k, float sigma, int method) {
    guarded([&]() -> int {
        const cv::Mat &b = *require_const(bboxes);
        if (b.type() != CV_32SC4) {
            throw cv::Exception(cv::Error::StsBadArg, "bboxes must be a CV_32SC4 Mat",
                                __func__, __FILE__, __LINE__);
        }
        std::vector<cv::Rect> boxes;
        boxes.reserve(static_cast<size_t>(b.rows));
        for (int i = 0; i < b.rows; ++i) {
            const auto *p = b.ptr<const int>(i, 0);
            boxes.push_back(cv::Rect(p[0], p[1], p[2], p[3]));
        }
        std::vector<float> score_vec = decode_float_mat(*require_const(scores), "scores");
        std::vector<float> updated;
        std::vector<int> out;
        cv::dnn::softNMSBoxes(boxes, score_vec, updated, score_threshold, nms_threshold,
                              out, static_cast<size_t>(top_k), sigma,
                              static_cast<cv::dnn::SoftNMSMethod>(method));
        fill_float_mat(*require(updated_scores), updated);
        fill_int_mat(*require(indices), out);
        return 0;
    });
}

cvk_mat_t *cvk_dnn_get_available_targets(int backend) {
    return guarded([&]() -> cvk_mat_t * {
        std::vector<int> targets;
        for (cv::dnn::Target t : cv::dnn::getAvailableTargets(
                 static_cast<cv::dnn::Backend>(backend))) {
            targets.push_back(static_cast<int>(t));
        }
        cv::Mat m(static_cast<int>(targets.size()), 1, CV_32S);
        if (!targets.empty()) std::memcpy(m.data, targets.data(), targets.size() * sizeof(int));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

cvk_mat_t *cvk_dnn_get_available_backends(void) {
    return guarded([&]() -> cvk_mat_t * {
        std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> pairs =
            cv::dnn::getAvailableBackends();
        cv::Mat m(static_cast<int>(pairs.size()), 1, CV_32SC2);
        auto *data = m.ptr<int>();
        for (size_t i = 0; i < pairs.size(); ++i) {
            data[2 * i] = static_cast<int>(pairs[i].first);
            data[2 * i + 1] = static_cast<int>(pairs[i].second);
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

void cvk_dnn_write_text_graph(const char *model, const char *output) {
    guarded([&]() -> int {
        cv::dnn::writeTextGraph(model, output);
        return 0;
    });
}

/* =========================================================================
 * Net
 * ========================================================================= */

cvk_net_t *cvk_net_create(void) {
    return guarded([&]() -> cvk_net_t * {
        auto *h = new cvk_net;
        h->ptr.reset(new cv::dnn::Net());
        return reinterpret_cast<cvk_net_t *>(h);
    });
}

void cvk_net_release(cvk_net_t *net) {
    delete reinterpret_cast<cvk_net *>(net);
}

int cvk_net_empty(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    return n == nullptr ? 1 : (n->ptr->empty() ? 1 : 0);
}

const char *cvk_net_dump(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> const char * {
        g_dnn_str = n->ptr->dump();
        return g_dnn_str.c_str();
    });
}

void cvk_net_dump_to_file(const cvk_net_t *net, const char *path) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->dumpToFile(path);
        return 0;
    });
}

void cvk_net_dump_to_pbtxt(const cvk_net_t *net, const char *path) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->dumpToPbtxt(path);
        return 0;
    });
}

int cvk_net_get_layer_id(const cvk_net_t *net, const char *layer) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return -1;
    return guarded([&]() -> int { return n->ptr->getLayerId(layer); });
}

unsigned char *cvk_net_get_layer_names(const cvk_net_t *net, size_t *out_len) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<cv::String> names = n->ptr->getLayerNames();
        std::vector<std::string> strings(names.begin(), names.end());
        return encode_strings(strings, out_len);
    });
}

cvk_layer_t *cvk_net_get_layer(const cvk_net_t *net, int layer_id) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_layer_t * {
        auto *h = new cvk_layer;
        h->ptr = n->ptr->getLayer(layer_id);
        return reinterpret_cast<cvk_layer_t *>(h);
    });
}

cvk_layer_t *cvk_net_get_layer_by_name(const cvk_net_t *net, const char *layer_name) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_layer_t * {
        auto *h = new cvk_layer;
        h->ptr = n->ptr->getLayer(layer_name);
        return reinterpret_cast<cvk_layer_t *>(h);
    });
}

void cvk_net_connect(const cvk_net_t *net, const char *out_pin, const char *inp_pin) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->connect(out_pin, inp_pin);
        return 0;
    });
}

int cvk_net_register_output(const cvk_net_t *net, const char *output_name,
                            int layer_id, int output_port) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return -1;
    return guarded([&]() -> int {
        return n->ptr->registerOutput(output_name, layer_id, output_port);
    });
}

void cvk_net_set_inputs_names(const cvk_net_t *net, const unsigned char *flat, size_t len) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        std::vector<std::string> names = decode_strings(flat, len);
        std::vector<cv::String> cnames(names.begin(), names.end());
        n->ptr->setInputsNames(cnames);
        return 0;
    });
}

void cvk_net_set_input_shape(const cvk_net_t *net, const char *input_name,
                             const cvk_mat_t *shape) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        std::vector<int> shape_vec = decode_int_mat(*require_const(shape), "shape");
        n->ptr->setInputShape(input_name, cv::MatShape(shape_vec.begin(), shape_vec.end()));
        return 0;
    });
}

cvk_mat_t *cvk_net_forward(const cvk_net_t *net, const char *output_name) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = n->ptr->forward(output_name != nullptr ? output_name : "");
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

cvk_mat_t *cvk_net_forward_all(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Mat> outputs;
        n->ptr->forward(outputs);
        cv::Mat addr = encode_mat_vector(outputs);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(addr)));
    });
}

cvk_mat_t *cvk_net_forward_layer(const cvk_net_t *net, const char *output_name) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Mat> outputs;
        n->ptr->forward(outputs, output_name != nullptr ? output_name : "");
        cv::Mat addr = encode_mat_vector(outputs);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(addr)));
    });
}

cvk_mat_t *cvk_net_forward_names(const cvk_net_t *net, const unsigned char *flat, size_t len) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<std::string> names = decode_strings(flat, len);
        std::vector<cv::String> cnames(names.begin(), names.end());
        std::vector<cv::Mat> outputs;
        n->ptr->forward(outputs, cnames);
        cv::Mat addr = encode_mat_vector(outputs);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(addr)));
    });
}

cvk_mat_t *cvk_net_forward_and_retrieve(const cvk_net_t *net,
                                        const unsigned char *flat, size_t len) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<std::string> names = decode_strings(flat, len);
        std::vector<cv::String> cnames(names.begin(), names.end());
        std::vector<std::vector<cv::Mat>> outputs;
        n->ptr->forward(outputs, cnames);
        // outer Mat-of-Mats: each member is the encoded inner vector
        cv::Mat outer(static_cast<int>(outputs.size()), 1, CV_32SC2);
        auto *data = outer.ptr<int64_t>();
        for (size_t i = 0; i < outputs.size(); ++i) {
            data[i] = reinterpret_cast<int64_t>(new cv::Mat(encode_mat_vector(outputs[i])));
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(outer)));
    });
}

void cvk_net_set_preferable_backend(const cvk_net_t *net, int backend_id) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->setPreferableBackend(backend_id);
        return 0;
    });
}

void cvk_net_set_preferable_target(const cvk_net_t *net, int target_id) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->setPreferableTarget(target_id);
        return 0;
    });
}

void cvk_net_finalize(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->finalizeNet();
        return 0;
    });
}

void cvk_net_set_input(const cvk_net_t *net, const cvk_mat_t *blob,
                       const char *name, double scalefactor, cvk_scalar_t mean) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->setInput(*require_const(blob), name != nullptr ? name : "",
                         scalefactor, cv_scalar(mean));
        return 0;
    });
}

void cvk_net_set_param(const cvk_net_t *net, int layer, int num_param,
                       const cvk_mat_t *blob) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->setParam(layer, num_param, *require_const(blob));
        return 0;
    });
}

void cvk_net_set_param_by_name(const cvk_net_t *net, const char *layer_name,
                               int num_param, const cvk_mat_t *blob) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->setParam(layer_name, num_param, *require_const(blob));
        return 0;
    });
}

cvk_mat_t *cvk_net_get_param(const cvk_net_t *net, int layer, int num_param) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = n->ptr->getParam(layer, num_param);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

cvk_mat_t *cvk_net_get_param_by_name(const cvk_net_t *net, const char *layer_name,
                                     int num_param) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat m = n->ptr->getParam(layer_name, num_param);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

cvk_mat_t *cvk_net_get_unconnected_out_layers(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<int> layers = n->ptr->getUnconnectedOutLayers();
        cv::Mat m(static_cast<int>(layers.size()), 1, CV_32S);
        if (!layers.empty()) std::memcpy(m.data, layers.data(), layers.size() * sizeof(int));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

unsigned char *cvk_net_get_unconnected_out_layers_names(const cvk_net_t *net,
                                                        size_t *out_len) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<cv::String> names = n->ptr->getUnconnectedOutLayersNames();
        std::vector<std::string> strings(names.begin(), names.end());
        return encode_strings(strings, out_len);
    });
}

long long cvk_net_get_flops(const cvk_net_t *net, const cvk_mat_t *input_shapes,
                            const cvk_mat_t *input_types) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return -1;
    return guarded([&]() -> long long {
        std::vector<cv::MatShape> shapes = decode_shape_list(*require_const(input_shapes));
        std::vector<int> types = decode_int_mat(*require_const(input_types), "input types");
        return static_cast<long long>(n->ptr->getFLOPS(shapes, types));
    });
}

void cvk_net_get_memory_consumption(const cvk_net_t *net,
                                    const cvk_mat_t *input_shapes,
                                    const cvk_mat_t *input_types,
                                    unsigned long long *out) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        std::vector<cv::MatShape> shapes = decode_shape_list(*require_const(input_shapes));
        std::vector<int> types = decode_int_mat(*require_const(input_types), "input types");
        size_t weights = 0;
        size_t blobs = 0;
        n->ptr->getMemoryConsumption(shapes, types, weights, blobs);
        out[0] = static_cast<unsigned long long>(weights);
        out[1] = static_cast<unsigned long long>(blobs);
        return 0;
    });
}

void cvk_net_get_layer_types(const cvk_net_t *net, unsigned char **flat, size_t *out_len) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        std::vector<cv::String> types;
        n->ptr->getLayerTypes(types);
        std::vector<std::string> strings(types.begin(), types.end());
        *flat = encode_strings(strings, out_len);
        return 0;
    });
}

int cvk_net_get_layers_count(const cvk_net_t *net, const char *layer_type) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return 0;
    return guarded([&]() -> int { return n->ptr->getLayersCount(layer_type); });
}

void cvk_net_enable_fusion(const cvk_net_t *net, int fusion) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->enableFusion(fusion != 0);
        return 0;
    });
}

void cvk_net_enable_winograd(const cvk_net_t *net, int use_winograd) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->enableWinograd(use_winograd != 0);
        return 0;
    });
}

long long cvk_net_get_perf_profile(const cvk_net_t *net, cvk_mat_t *timings) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return -1;
    return guarded([&]() -> long long {
        std::vector<double> timing_vec;
        int64_t total = n->ptr->getPerfProfile(timing_vec);
        cv::Mat &m = *require(timings);
        m.create(static_cast<int>(timing_vec.size()), 1, CV_64F);
        if (!timing_vec.empty()) {
            std::memcpy(m.data, timing_vec.data(), timing_vec.size() * sizeof(double));
        }
        return static_cast<long long>(total);
    });
}

void cvk_net_enable_kv_cache(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->enableKVCache();
        return 0;
    });
}

void cvk_net_disable_kv_cache(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->disableKVCache();
        return 0;
    });
}

void cvk_net_reset_kv_cache(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->resetKVCache();
        return 0;
    });
}

void cvk_net_get_perf_profile_names(const cvk_net_t *net, unsigned char **flat,
                                    size_t *out_len) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        std::vector<std::string> names, timems, counts;
        n->ptr->getPerfProfile(names, timems, counts);
        size_t total = 4;
        for (size_t i = 0; i < names.size(); ++i) {
            total += 12 + names[i].size() + timems[i].size() + counts[i].size();
        }
        auto *buf = static_cast<unsigned char *>(std::malloc(total));
        if (buf == nullptr) throw std::bad_alloc();
        auto *p = buf;
        const uint32_t count = static_cast<uint32_t>(names.size());
        std::memcpy(p, &count, 4);
        p += 4;
        for (size_t i = 0; i < names.size(); ++i) {
            for (const std::string &s : {names[i], timems[i], counts[i]}) {
                const uint32_t len = static_cast<uint32_t>(s.size());
                std::memcpy(p, &len, 4);
                p += 4;
                if (len) std::memcpy(p, s.data(), len);
                p += len;
            }
        }
        *flat = buf;
        *out_len = total;
        return 0;
    });
}

void cvk_net_print_perf_profile(const cvk_net_t *net) {
    cvk_net *n = require_net(net);
    if (n == nullptr) return;
    guarded([&]() -> int {
        n->ptr->printPerfProfile();
        return 0;
    });
}

/* =========================================================================
 * Layer
 * ========================================================================= */

void cvk_layer_release(cvk_layer_t *layer) {
    delete reinterpret_cast<cvk_layer *>(layer);
}

const char *cvk_layer_name(const cvk_layer_t *layer) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return nullptr;
    return guarded([&]() -> const char * {
        g_dnn_str = l->ptr->name;
        return g_dnn_str.c_str();
    });
}

const char *cvk_layer_type(const cvk_layer_t *layer) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return nullptr;
    return guarded([&]() -> const char * {
        g_dnn_str = l->ptr->type;
        return g_dnn_str.c_str();
    });
}

int cvk_layer_preferable_target(const cvk_layer_t *layer) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return 0;
    return guarded([&]() -> int { return l->ptr->preferableTarget; });
}

int cvk_layer_output_name_to_index(const cvk_layer_t *layer, const char *output_name) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return -1;
    return guarded([&]() -> int { return l->ptr->outputNameToIndex(output_name); });
}

cvk_mat_t *cvk_layer_get_blobs(const cvk_layer_t *layer) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat addr = encode_mat_vector(l->ptr->blobs);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(addr)));
    });
}

void cvk_layer_set_blobs(cvk_layer_t *layer, const cvk_mat_t *blobs) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return;
    guarded([&]() -> int {
        l->ptr->blobs = decode_mat_vector(*require_const(blobs));
        return 0;
    });
}

void cvk_layer_finalize(const cvk_layer_t *layer, const cvk_mat_t *inputs,
                        cvk_mat_t *outputs) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return;
    guarded([&]() -> int {
        std::vector<cv::Mat> in_vec = decode_mat_vector(*require_const(inputs));
        std::vector<cv::Mat> out_vec;
        l->ptr->finalize(in_vec, out_vec);
        cv::Mat &out_mat = *require(outputs);
        out_mat = encode_mat_vector(out_vec);
        return 0;
    });
}

void cvk_layer_run(const cvk_layer_t *layer, const cvk_mat_t *inputs,
                   cvk_mat_t *outputs, cvk_mat_t *internals) {
    cvk_layer *l = require_layer(layer);
    if (l == nullptr) return;
    guarded([&]() -> int {
        std::vector<cv::Mat> in_vec = decode_mat_vector(*require_const(inputs));
        std::vector<cv::Mat> out_vec;
        std::vector<cv::Mat> int_vec;
        l->ptr->run(in_vec, out_vec, int_vec);
        cv::Mat &out_mat = *require(outputs);
        cv::Mat &int_mat = *require(internals);
        out_mat = encode_mat_vector(out_vec);
        int_mat = encode_mat_vector(int_vec);
        return 0;
    });
}

/* =========================================================================
 * Tokenizer
 * ========================================================================= */

cvk_tokenizer_t *cvk_tokenizer_load(const char *model_config) {
    return guarded([&]() -> cvk_tokenizer_t * {
        return reinterpret_cast<cvk_tokenizer_t *>(
            new cvk_tokenizer{cv::dnn::Tokenizer::load(model_config)});
    });
}

void cvk_tokenizer_release(cvk_tokenizer_t *tokenizer) {
    delete reinterpret_cast<cvk_tokenizer *>(tokenizer);
}

cvk_mat_t *cvk_tokenizer_encode(const cvk_tokenizer_t *tokenizer, const char *text) {
    cvk_tokenizer *t = require_tokenizer(tokenizer);
    if (t == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<int> ids = t->tok.encode(text);
        cv::Mat m(static_cast<int>(ids.size()), 1, CV_32S);
        if (!ids.empty()) std::memcpy(m.data, ids.data(), ids.size() * sizeof(int));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(m)));
    });
}

const char *cvk_tokenizer_decode(const cvk_tokenizer_t *tokenizer, const cvk_mat_t *tokens) {
    cvk_tokenizer *t = require_tokenizer(tokenizer);
    if (t == nullptr) return nullptr;
    return guarded([&]() -> const char * {
        std::vector<int> ids = decode_int_mat(*require_const(tokens), "tokens");
        g_dnn_str = t->tok.decode(ids);
        return g_dnn_str.c_str();
    });
}

/* =========================================================================
 * dnn helpers (N-dim Mat introspection)
 * ========================================================================= */

int cvk_dnn_mat_dims(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m == nullptr ? 0 : m->dims;
}

int cvk_dnn_mat_shape(const cvk_mat_t *mat, int *out, int max_count) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return 0;
    const int dims = m->dims;
    const int n = dims < max_count ? dims : max_count;
    for (int i = 0; i < n; ++i) {
        out[i] = m->size.p[i];
    }
    return dims;
}

} /* extern "C" */

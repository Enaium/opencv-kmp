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
 * cvk_ C ABI implementation for the objdetect QR / barcode / face / MCC
 * classes (see opencv_kmp_objdetect2.h).
 *
 * String-vector results are packed into malloc'd buffers released with
 * cvk_free_buffer(); vector<Mat> results are packed into a CV_8UC1 wire Mat
 * (both formats documented in the header). Output Mats are caller-allocated
 * and filled in place, mirroring the Java SDK's out-parameter style.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_objdetect2.h"

#include <opencv2/objdetect.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ---- opaque handles ------------------------------------------------------ */

struct cvk_qr_code_detector { cv::QRCodeDetector impl; };
struct cvk_qr_code_detector_aruco { cv::QRCodeDetectorAruco impl; };
struct cvk_qr_code_encoder { cv::Ptr<cv::QRCodeEncoder> ptr; };
struct cvk_barcode_detector { cv::barcode::BarcodeDetector impl; };
struct cvk_face_detector_yn { cv::Ptr<cv::FaceDetectorYN> ptr; };
struct cvk_face_recognizer_sf { cv::Ptr<cv::FaceRecognizerSF> ptr; };
struct cvk_c_checker { cv::Ptr<cv::mcc::CChecker> ptr; };
struct cvk_c_checker_detector { cv::Ptr<cv::mcc::CCheckerDetector> ptr; };

namespace {

thread_local std::string g_od2_last_error;
thread_local std::string g_od2_str;

void record_error(const char *message) {
    try {
        g_od2_last_error = message != nullptr ? message : "unknown error";
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

/* ---- wire-format helpers -------------------------------------------------- */

struct ByteWriter {
    std::vector<unsigned char> buf;

    void put_u32le(unsigned int v) {
        buf.push_back(static_cast<unsigned char>(v & 0xFFu));
        buf.push_back(static_cast<unsigned char>((v >> 8) & 0xFFu));
        buf.push_back(static_cast<unsigned char>((v >> 16) & 0xFFu));
        buf.push_back(static_cast<unsigned char>((v >> 24) & 0xFFu));
    }

    void put_bytes(const void *data, size_t n) {
        const auto *p = static_cast<const unsigned char *>(data);
        buf.insert(buf.end(), p, p + n);
    }

    void put_string_list(const std::vector<std::string> &list) {
        put_u32le(static_cast<unsigned int>(list.size()));
        for (const auto &s : list) {
            put_u32le(static_cast<unsigned int>(s.size()));
            put_bytes(s.data(), s.size());
        }
    }

    unsigned char *finish(size_t *out_len) {
        auto *out = static_cast<unsigned char *>(std::malloc(buf.size()));
        if (out == nullptr) throw std::bad_alloc();
        std::memcpy(out, buf.data(), buf.size());
        if (out_len != nullptr) *out_len = buf.size();
        return out;
    }
};

/** Packs a vector of Mats into the CV_8UC1 wire format (see header). */
void pack_mat_list_into(cv::Mat &dst, const std::vector<cv::Mat> &mats) {
    size_t total = 4;
    std::vector<size_t> lens;
    lens.reserve(mats.size());
    for (const auto &m : mats) {
        const size_t len = m.empty() ? 0 : m.total() * m.elemSize();
        lens.push_back(len);
        total += 16 + len;
    }
    dst.create(1, static_cast<int>(total), CV_8UC1);
    unsigned char *p = dst.ptr<unsigned char>();
    auto put32 = [&p](unsigned int v) {
        p[0] = static_cast<unsigned char>(v & 0xFFu);
        p[1] = static_cast<unsigned char>((v >> 8) & 0xFFu);
        p[2] = static_cast<unsigned char>((v >> 16) & 0xFFu);
        p[3] = static_cast<unsigned char>((v >> 24) & 0xFFu);
        p += 4;
    };
    put32(static_cast<unsigned int>(mats.size()));
    for (size_t i = 0; i < mats.size(); ++i) {
        const cv::Mat &m = mats[i];
        put32(static_cast<unsigned int>(m.rows));
        put32(static_cast<unsigned int>(m.cols));
        put32(static_cast<unsigned int>(m.type()));
        put32(static_cast<unsigned int>(lens[i]));
        if (lens[i] > 0) {
            const cv::Mat cont = m.isContinuous() ? m : m.clone();
            std::memcpy(p, cont.data, lens[i]);
            p += lens[i];
        }
    }
}

cv::Mat continuous_of(const cv::Mat &m) { return m.isContinuous() ? m : m.clone(); }

std::vector<cv::Point2f> mat_to_points2f(const cv::Mat &m) {
    if (m.empty()) return {};
    if (m.type() != CV_32FC2) {
        throw cv::Exception(cv::Error::StsBadArg, "expected CV_32FC2 Mat for point list",
                            __func__, __FILE__, __LINE__);
    }
    const cv::Mat src = continuous_of(m);
    const float *data = src.ptr<float>();
    std::vector<cv::Point2f> out;
    out.reserve(src.total());
    for (size_t i = 0; i < src.total(); ++i) {
        out.emplace_back(data[2 * i], data[2 * i + 1]);
    }
    return out;
}

std::vector<cv::Rect> mat_to_rects(const cv::Mat &m) {
    if (m.empty()) return {};
    if (m.type() != CV_32SC4) {
        throw cv::Exception(cv::Error::StsBadArg, "expected CV_32SC4 Mat for rect list",
                            __func__, __FILE__, __LINE__);
    }
    const cv::Mat src = continuous_of(m);
    const int *data = src.ptr<int>();
    std::vector<cv::Rect> out;
    out.reserve(src.total());
    for (size_t i = 0; i < src.total(); ++i) {
        out.emplace_back(data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]);
    }
    return out;
}

std::vector<float> mat_to_floats(const cv::Mat &m) {
    if (m.empty()) return {};
    if (m.type() != CV_32FC1) {
        throw cv::Exception(cv::Error::StsBadArg, "expected CV_32FC1 Mat for float list",
                            __func__, __FILE__, __LINE__);
    }
    const cv::Mat src = continuous_of(m);
    const float *data = src.ptr<float>();
    return std::vector<float>(data, data + src.total());
}

std::vector<unsigned char> mat_to_bytes(const cv::Mat &m) {
    if (m.empty()) return {};
    if (m.type() != CV_8UC1) {
        throw cv::Exception(cv::Error::StsBadArg, "expected CV_8UC1 Mat for byte buffer",
                            __func__, __FILE__, __LINE__);
    }
    const cv::Mat src = continuous_of(m);
    const unsigned char *data = src.ptr<unsigned char>();
    return std::vector<unsigned char>(data, data + src.total());
}

cvk_mat_t *points2f_to_mat(const std::vector<cv::Point2f> &pts) {
    auto *m = new cv::Mat(1, static_cast<int>(pts.size()), CV_32FC2);
    if (!pts.empty()) {
        std::memcpy(m->data, pts.data(), pts.size() * sizeof(cv::Point2f));
    }
    return reinterpret_cast<cvk_mat_t *>(m);
}

void fill_floats(cv::Mat &dst, const std::vector<float> &v) {
    dst.create(1, static_cast<int>(v.size()), CV_32FC1);
    if (!v.empty()) {
        std::memcpy(dst.data, v.data(), v.size() * sizeof(float));
    }
}

/* ---- GraphicalCodeDetector surface, generated per concrete detector ------- */

#define CVK_GCD_FUNCS(T)                                                        \
    int cvk_##T##_detect(const cvk_##T##_t *h, const cvk_mat_t *img,            \
                         cvk_mat_t *points) {                                   \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;        \
        cv::Mat *pts = p != nullptr ? require(points) : nullptr;                \
        if (p == nullptr || im == nullptr || pts == nullptr) return 0;          \
        return guarded([&]() -> int {                                           \
            return p->impl.detect(*im, *pts) ? 1 : 0;                           \
        });                                                                     \
    }                                                                           \
    const char *cvk_##T##_decode(const cvk_##T##_t *h, const cvk_mat_t *img,    \
                                 const cvk_mat_t *points, cvk_mat_t *straight_code) { \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;        \
        const cv::Mat *pts = p != nullptr ? require_const(points) : nullptr;    \
        cv::Mat *sc = p != nullptr ? require(straight_code) : nullptr;          \
        if (p == nullptr || im == nullptr || pts == nullptr || sc == nullptr) { \
            return nullptr;                                                     \
        }                                                                       \
        return guarded([&]() -> const char * {                                  \
            g_od2_str = p->impl.decode(*im, *pts, *sc);                         \
            return g_od2_str.c_str();                                           \
        });                                                                     \
    }                                                                           \
    const char *cvk_##T##_detect_and_decode(const cvk_##T##_t *h,               \
                                            const cvk_mat_t *img,               \
                                            cvk_mat_t *points,                  \
                                            cvk_mat_t *straight_code) {         \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;        \
        cv::Mat *pts = p != nullptr ? require(points) : nullptr;                \
        cv::Mat *sc = p != nullptr ? require(straight_code) : nullptr;          \
        if (p == nullptr || im == nullptr || pts == nullptr || sc == nullptr) { \
            return nullptr;                                                     \
        }                                                                       \
        return guarded([&]() -> const char * {                                  \
            g_od2_str = p->impl.detectAndDecode(*im, *pts, *sc);                \
            return g_od2_str.c_str();                                           \
        });                                                                     \
    }                                                                           \
    int cvk_##T##_detect_multi(const cvk_##T##_t *h, const cvk_mat_t *img,      \
                               cvk_mat_t *points) {                             \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;        \
        cv::Mat *pts = p != nullptr ? require(points) : nullptr;                \
        if (p == nullptr || im == nullptr || pts == nullptr) return 0;          \
        return guarded([&]() -> int {                                           \
            return p->impl.detectMulti(*im, *pts) ? 1 : 0;                      \
        });                                                                     \
    }                                                                           \
    unsigned char *cvk_##T##_decode_multi(const cvk_##T##_t *h,                 \
                                          const cvk_mat_t *img,                 \
                                          const cvk_mat_t *points,              \
                                          cvk_mat_t *straight_code,             \
                                          size_t *out_len) {                    \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;        \
        const cv::Mat *pts = p != nullptr ? require_const(points) : nullptr;    \
        cv::Mat *sc = p != nullptr ? require(straight_code) : nullptr;          \
        if (p == nullptr || im == nullptr || pts == nullptr || sc == nullptr) { \
            return nullptr;                                                     \
        }                                                                       \
        return guarded([&]() -> unsigned char * {                               \
            std::vector<std::string> info;                                      \
            std::vector<cv::Mat> straight;                                      \
            const bool ok = p->impl.decodeMulti(*im, *pts, info, straight);     \
            pack_mat_list_into(*sc, straight);                                  \
            ByteWriter w;                                                       \
            w.put_u32le(ok ? 1u : 0u);                                          \
            w.put_string_list(info);                                            \
            return w.finish(out_len);                                           \
        });                                                                     \
    }                                                                           \
    unsigned char *cvk_##T##_detect_and_decode_multi(const cvk_##T##_t *h,      \
                                                     const cvk_mat_t *img,      \
                                                     cvk_mat_t *points,         \
                                                     cvk_mat_t *straight_code,  \
                                                     size_t *out_len) {         \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;        \
        cv::Mat *pts = p != nullptr ? require(points) : nullptr;                \
        cv::Mat *sc = p != nullptr ? require(straight_code) : nullptr;          \
        if (p == nullptr || im == nullptr || pts == nullptr || sc == nullptr) { \
            return nullptr;                                                     \
        }                                                                       \
        return guarded([&]() -> unsigned char * {                               \
            std::vector<std::string> info;                                      \
            std::vector<cv::Mat> straight;                                      \
            const bool ok = p->impl.detectAndDecodeMulti(*im, info, *pts, straight); \
            pack_mat_list_into(*sc, straight);                                  \
            ByteWriter w;                                                       \
            w.put_u32le(ok ? 1u : 0u);                                          \
            w.put_string_list(info);                                            \
            return w.finish(out_len);                                           \
        });                                                                     \
    }

/* Algorithm surface for Ptr-based Algorithm subclasses. */
#define CVK_ALG_FUNCS(T)                                                        \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                               \
        if (p == nullptr) return;                                               \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                   \
    }                                                                           \
    int cvk_##T##_empty(const cvk_##T##_t *h) {                                 \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        if (p == nullptr) return 1;                                             \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });       \
    }                                                                           \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {                 \
        auto *p = reinterpret_cast<cvk_##T *>(h);                               \
        if (p == nullptr || filename == nullptr) return;                        \
        guarded([&]() -> int { p->ptr->save(filename); return 0; });            \
    }                                                                           \
    const char *cvk_##T##_get_default_name(const cvk_##T##_t *h) {              \
        const auto *p = reinterpret_cast<const cvk_##T *>(h);                   \
        if (p == nullptr) return nullptr;                                       \
        return guarded([&]() -> const char * {                                  \
            g_od2_str = p->ptr->getDefaultName();                               \
            return g_od2_str.c_str();                                           \
        });                                                                     \
    }

} // namespace

extern "C" {

/* =========================================================================
 * cv::QRCodeDetector
 * ========================================================================= */

cvk_qr_code_detector_t *cvk_qr_code_detector_create(void) {
    return guarded([&]() -> cvk_qr_code_detector_t * {
        auto *h = new cvk_qr_code_detector;
        return reinterpret_cast<cvk_qr_code_detector_t *>(h);
    });
}

void cvk_qr_code_detector_set_eps_x(cvk_qr_code_detector_t *h, double eps_x) {
    auto *p = reinterpret_cast<cvk_qr_code_detector *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->impl.setEpsX(eps_x);
        return 0;
    });
}

void cvk_qr_code_detector_set_eps_y(cvk_qr_code_detector_t *h, double eps_y) {
    auto *p = reinterpret_cast<cvk_qr_code_detector *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->impl.setEpsY(eps_y);
        return 0;
    });
}

void cvk_qr_code_detector_set_use_alignment_markers(cvk_qr_code_detector_t *h, int use) {
    auto *p = reinterpret_cast<cvk_qr_code_detector *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->impl.setUseAlignmentMarkers(use != 0);
        return 0;
    });
}

const char *cvk_qr_code_detector_decode_curved(cvk_qr_code_detector_t *h,
                                               const cvk_mat_t *img,
                                               const cvk_mat_t *points,
                                               cvk_mat_t *straight_qrcode) {
    auto *p = reinterpret_cast<cvk_qr_code_detector *>(h);
    const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;
    const cv::Mat *pts = p != nullptr ? require_const(points) : nullptr;
    cv::Mat *sq = p != nullptr ? require(straight_qrcode) : nullptr;
    if (p == nullptr || im == nullptr || pts == nullptr || sq == nullptr) {
        return nullptr;
    }
    return guarded([&]() -> const char * {
        g_od2_str = p->impl.decodeCurved(*im, *pts, *sq);
        return g_od2_str.c_str();
    });
}

const char *cvk_qr_code_detector_detect_and_decode_curved(cvk_qr_code_detector_t *h,
                                                          const cvk_mat_t *img,
                                                          cvk_mat_t *points,
                                                          cvk_mat_t *straight_qrcode) {
    auto *p = reinterpret_cast<cvk_qr_code_detector *>(h);
    const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;
    cv::Mat *pts = p != nullptr ? require(points) : nullptr;
    cv::Mat *sq = p != nullptr ? require(straight_qrcode) : nullptr;
    if (p == nullptr || im == nullptr || pts == nullptr || sq == nullptr) {
        return nullptr;
    }
    return guarded([&]() -> const char * {
        g_od2_str = p->impl.detectAndDecodeCurved(*im, *pts, *sq);
        return g_od2_str.c_str();
    });
}

int cvk_qr_code_detector_get_encoding(cvk_qr_code_detector_t *h, int code_idx) {
    auto *p = reinterpret_cast<cvk_qr_code_detector *>(h);
    if (p == nullptr) return 0;
    return guarded([&]() -> int { return static_cast<int>(p->impl.getEncoding(code_idx)); });
}

void cvk_qr_code_detector_release(cvk_qr_code_detector_t *h) {
    delete reinterpret_cast<cvk_qr_code_detector *>(h);
}

CVK_GCD_FUNCS(qr_code_detector)

/* =========================================================================
 * cv::QRCodeDetectorAruco
 * ========================================================================= */

cvk_qr_code_detector_aruco_t *cvk_qr_code_detector_aruco_create(void) {
    return guarded([&]() -> cvk_qr_code_detector_aruco_t * {
        auto *h = new cvk_qr_code_detector_aruco;
        return reinterpret_cast<cvk_qr_code_detector_aruco_t *>(h);
    });
}

cvk_qr_code_detector_aruco_t *cvk_qr_code_detector_aruco_create_with_params(
    float min_module_size_in_pyramid, float max_rotation, float max_module_size_mismatch,
    float max_timing_pattern_mismatch, float max_penalties, float max_colors_mismatch,
    float scale_timing_pattern_score) {
    return guarded([&]() -> cvk_qr_code_detector_aruco_t * {
        cv::QRCodeDetectorAruco::Params params;
        params.minModuleSizeInPyramid = min_module_size_in_pyramid;
        params.maxRotation = max_rotation;
        params.maxModuleSizeMismatch = max_module_size_mismatch;
        params.maxTimingPatternMismatch = max_timing_pattern_mismatch;
        params.maxPenalties = max_penalties;
        params.maxColorsMismatch = max_colors_mismatch;
        params.scaleTimingPatternScore = scale_timing_pattern_score;
        auto *h = new cvk_qr_code_detector_aruco;
        h->impl = cv::QRCodeDetectorAruco(params);
        return reinterpret_cast<cvk_qr_code_detector_aruco_t *>(h);
    });
}

void cvk_qr_code_detector_aruco_get_detector_params(const cvk_qr_code_detector_aruco_t *h,
                                                    float *out7) {
    const auto *p = reinterpret_cast<const cvk_qr_code_detector_aruco *>(h);
    if (p == nullptr || out7 == nullptr) return;
    guarded([&]() -> int {
        const cv::QRCodeDetectorAruco::Params &params = p->impl.getDetectorParameters();
        out7[0] = params.minModuleSizeInPyramid;
        out7[1] = params.maxRotation;
        out7[2] = params.maxModuleSizeMismatch;
        out7[3] = params.maxTimingPatternMismatch;
        out7[4] = params.maxPenalties;
        out7[5] = params.maxColorsMismatch;
        out7[6] = params.scaleTimingPatternScore;
        return 0;
    });
}

void cvk_qr_code_detector_aruco_set_detector_params(cvk_qr_code_detector_aruco_t *h,
                                                    float min_module_size_in_pyramid,
                                                    float max_rotation,
                                                    float max_module_size_mismatch,
                                                    float max_timing_pattern_mismatch,
                                                    float max_penalties,
                                                    float max_colors_mismatch,
                                                    float scale_timing_pattern_score) {
    auto *p = reinterpret_cast<cvk_qr_code_detector_aruco *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        cv::QRCodeDetectorAruco::Params params;
        params.minModuleSizeInPyramid = min_module_size_in_pyramid;
        params.maxRotation = max_rotation;
        params.maxModuleSizeMismatch = max_module_size_mismatch;
        params.maxTimingPatternMismatch = max_timing_pattern_mismatch;
        params.maxPenalties = max_penalties;
        params.maxColorsMismatch = max_colors_mismatch;
        params.scaleTimingPatternScore = scale_timing_pattern_score;
        p->impl.setDetectorParameters(params);
        return 0;
    });
}

void cvk_qr_code_detector_aruco_release(cvk_qr_code_detector_aruco_t *h) {
    delete reinterpret_cast<cvk_qr_code_detector_aruco *>(h);
}

CVK_GCD_FUNCS(qr_code_detector_aruco)

/* =========================================================================
 * cv::QRCodeEncoder
 * ========================================================================= */

cvk_qr_code_encoder_t *cvk_qr_code_encoder_create(void) {
    return guarded([&]() -> cvk_qr_code_encoder_t * {
        auto *h = new cvk_qr_code_encoder;
        h->ptr = cv::QRCodeEncoder::create();
        return reinterpret_cast<cvk_qr_code_encoder_t *>(h);
    });
}

cvk_qr_code_encoder_t *cvk_qr_code_encoder_create_with_params(int version,
                                                              int correction_level,
                                                              int mode,
                                                              int structure_number) {
    return guarded([&]() -> cvk_qr_code_encoder_t * {
        cv::QRCodeEncoder::Params params;
        params.version = version;
        params.correction_level = static_cast<cv::QRCodeEncoder::CorrectionLevel>(correction_level);
        params.mode = static_cast<cv::QRCodeEncoder::EncodeMode>(mode);
        params.structure_number = structure_number;
        auto *h = new cvk_qr_code_encoder;
        h->ptr = cv::QRCodeEncoder::create(params);
        return reinterpret_cast<cvk_qr_code_encoder_t *>(h);
    });
}

int cvk_qr_code_encoder_encode(const cvk_qr_code_encoder_t *h, const char *encoded_info,
                               cvk_mat_t *qrcode) {
    const auto *p = reinterpret_cast<const cvk_qr_code_encoder *>(h);
    cv::Mat *qr = p != nullptr ? require(qrcode) : nullptr;
    if (p == nullptr || encoded_info == nullptr || qr == nullptr) return 0;
    return guarded([&]() -> int {
        p->ptr->encode(std::string(encoded_info), *qr);
        return 1;
    });
}

int cvk_qr_code_encoder_encode_bytes(const cvk_qr_code_encoder_t *h,
                                     const unsigned char *data, size_t len,
                                     cvk_mat_t *qrcode) {
    const auto *p = reinterpret_cast<const cvk_qr_code_encoder *>(h);
    cv::Mat *qr = p != nullptr ? require(qrcode) : nullptr;
    if (p == nullptr || qr == nullptr) return 0;
    return guarded([&]() -> int {
        const char *text = data != nullptr ? reinterpret_cast<const char *>(data) : "";
        p->ptr->encode(std::string(text, len), *qr);
        return 1;
    });
}

int cvk_qr_code_encoder_encode_structured_append(const cvk_qr_code_encoder_t *h,
                                                 const char *encoded_info,
                                                 cvk_mat_t *qrcodes) {
    const auto *p = reinterpret_cast<const cvk_qr_code_encoder *>(h);
    cv::Mat *qr = p != nullptr ? require(qrcodes) : nullptr;
    if (p == nullptr || encoded_info == nullptr || qr == nullptr) return 0;
    return guarded([&]() -> int {
        std::vector<cv::Mat> codes;
        p->ptr->encodeStructuredAppend(std::string(encoded_info), codes);
        pack_mat_list_into(*qr, codes);
        return 1;
    });
}

void cvk_qr_code_encoder_release(cvk_qr_code_encoder_t *h) {
    delete reinterpret_cast<cvk_qr_code_encoder *>(h);
}

/* =========================================================================
 * cv::barcode::BarcodeDetector
 * ========================================================================= */

cvk_barcode_detector_t *cvk_barcode_detector_create(void) {
    return guarded([&]() -> cvk_barcode_detector_t * {
        auto *h = new cvk_barcode_detector;
        return reinterpret_cast<cvk_barcode_detector_t *>(h);
    });
}

cvk_barcode_detector_t *cvk_barcode_detector_create_with_model(const char *model_path) {
    if (model_path == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_barcode_detector_t * {
        auto *h = new cvk_barcode_detector;
        h->impl = cv::barcode::BarcodeDetector(std::string(model_path));
        return reinterpret_cast<cvk_barcode_detector_t *>(h);
    });
}

unsigned char *cvk_barcode_detector_decode_with_type(const cvk_barcode_detector_t *h,
                                                     const cvk_mat_t *img,
                                                     const cvk_mat_t *points, size_t *out_len) {
    const auto *p = reinterpret_cast<const cvk_barcode_detector *>(h);
    const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;
    const cv::Mat *pts = p != nullptr ? require_const(points) : nullptr;
    if (p == nullptr || im == nullptr || pts == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<std::string> info;
        std::vector<std::string> type;
        const bool ok = p->impl.decodeWithType(*im, *pts, info, type);
        ByteWriter w;
        w.put_u32le(ok ? 1u : 0u);
        w.put_string_list(info);
        w.put_string_list(type);
        return w.finish(out_len);
    });
}

unsigned char *cvk_barcode_detector_detect_and_decode_with_type(const cvk_barcode_detector_t *h,
                                                                const cvk_mat_t *img,
                                                                cvk_mat_t *points,
                                                                size_t *out_len) {
    const auto *p = reinterpret_cast<const cvk_barcode_detector *>(h);
    const cv::Mat *im = p != nullptr ? require_const(img) : nullptr;
    cv::Mat *pts = p != nullptr ? require(points) : nullptr;
    if (p == nullptr || im == nullptr || pts == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<std::string> info;
        std::vector<std::string> type;
        const bool ok = p->impl.detectAndDecodeWithType(*im, info, type, *pts);
        ByteWriter w;
        w.put_u32le(ok ? 1u : 0u);
        w.put_string_list(info);
        w.put_string_list(type);
        return w.finish(out_len);
    });
}

double cvk_barcode_detector_get_downsampling_threshold(const cvk_barcode_detector_t *h) {
    const auto *p = reinterpret_cast<const cvk_barcode_detector *>(h);
    if (p == nullptr) return 0.0;
    return guarded([&]() -> double { return p->impl.getDownsamplingThreshold(); });
}

void cvk_barcode_detector_set_downsampling_threshold(cvk_barcode_detector_t *h, double thresh) {
    auto *p = reinterpret_cast<cvk_barcode_detector *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->impl.setDownsamplingThreshold(thresh);
        return 0;
    });
}

void cvk_barcode_detector_get_detector_scales(const cvk_barcode_detector_t *h, cvk_mat_t *sizes) {
    const auto *p = reinterpret_cast<const cvk_barcode_detector *>(h);
    cv::Mat *s = p != nullptr ? require(sizes) : nullptr;
    if (p == nullptr || s == nullptr) return;
    guarded([&]() -> int {
        std::vector<float> scales;
        p->impl.getDetectorScales(scales);
        fill_floats(*s, scales);
        return 0;
    });
}

void cvk_barcode_detector_set_detector_scales(cvk_barcode_detector_t *h, const cvk_mat_t *sizes) {
    auto *p = reinterpret_cast<cvk_barcode_detector *>(h);
    const cv::Mat *s = p != nullptr ? require_const(sizes) : nullptr;
    if (p == nullptr || s == nullptr) return;
    guarded([&]() -> int {
        p->impl.setDetectorScales(mat_to_floats(*s));
        return 0;
    });
}

double cvk_barcode_detector_get_gradient_threshold(const cvk_barcode_detector_t *h) {
    const auto *p = reinterpret_cast<const cvk_barcode_detector *>(h);
    if (p == nullptr) return 0.0;
    return guarded([&]() -> double { return p->impl.getGradientThreshold(); });
}

void cvk_barcode_detector_set_gradient_threshold(cvk_barcode_detector_t *h, double thresh) {
    auto *p = reinterpret_cast<cvk_barcode_detector *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->impl.setGradientThreshold(thresh);
        return 0;
    });
}

void cvk_barcode_detector_release(cvk_barcode_detector_t *h) {
    delete reinterpret_cast<cvk_barcode_detector *>(h);
}

CVK_GCD_FUNCS(barcode_detector)

/* =========================================================================
 * cv::FaceDetectorYN
 * ========================================================================= */

cvk_face_detector_yn_t *cvk_face_detector_yn_create(const char *model, const char *config,
                                                    int input_width, int input_height,
                                                    float score_threshold, float nms_threshold,
                                                    int top_k, int backend_id, int target_id) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_face_detector_yn_t * {
        auto *h = new cvk_face_detector_yn;
        h->ptr = cv::FaceDetectorYN::create(
            std::string(model), config != nullptr ? std::string(config) : std::string(),
            cv::Size(input_width, input_height), score_threshold, nms_threshold, top_k,
            backend_id, target_id);
        return reinterpret_cast<cvk_face_detector_yn_t *>(h);
    });
}

cvk_face_detector_yn_t *cvk_face_detector_yn_create_from_buffers(
    const char *framework, const cvk_mat_t *buffer_model, const cvk_mat_t *buffer_config,
    int input_width, int input_height, float score_threshold, float nms_threshold,
    int top_k, int backend_id, int target_id) {
    const cv::Mat *bm = require_const(buffer_model);
    const cv::Mat *bc = require_const(buffer_config);
    if (framework == nullptr || bm == nullptr || bc == nullptr) return nullptr;
    return guarded([&]() -> cvk_face_detector_yn_t * {
        std::vector<unsigned char> model = mat_to_bytes(*bm);
        std::vector<unsigned char> config = mat_to_bytes(*bc);
        auto *h = new cvk_face_detector_yn;
        h->ptr = cv::FaceDetectorYN::create(
            std::string(framework), model, config, cv::Size(input_width, input_height),
            score_threshold, nms_threshold, top_k, backend_id, target_id);
        return reinterpret_cast<cvk_face_detector_yn_t *>(h);
    });
}

void cvk_face_detector_yn_set_input_size(cvk_face_detector_yn_t *h, int width, int height) {
    auto *p = reinterpret_cast<cvk_face_detector_yn *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setInputSize(cv::Size(width, height));
        return 0;
    });
}

void cvk_face_detector_yn_get_input_size(const cvk_face_detector_yn_t *h, int *out2) {
    const auto *p = reinterpret_cast<const cvk_face_detector_yn *>(h);
    if (p == nullptr || out2 == nullptr) return;
    guarded([&]() -> int {
        const cv::Size size = p->ptr->getInputSize();
        out2[0] = size.width;
        out2[1] = size.height;
        return 0;
    });
}

void cvk_face_detector_yn_set_score_threshold(cvk_face_detector_yn_t *h, float threshold) {
    auto *p = reinterpret_cast<cvk_face_detector_yn *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setScoreThreshold(threshold);
        return 0;
    });
}

float cvk_face_detector_yn_get_score_threshold(const cvk_face_detector_yn_t *h) {
    const auto *p = reinterpret_cast<const cvk_face_detector_yn *>(h);
    if (p == nullptr) return 0.0f;
    return guarded([&]() -> float { return p->ptr->getScoreThreshold(); });
}

void cvk_face_detector_yn_set_nms_threshold(cvk_face_detector_yn_t *h, float threshold) {
    auto *p = reinterpret_cast<cvk_face_detector_yn *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setNMSThreshold(threshold);
        return 0;
    });
}

float cvk_face_detector_yn_get_nms_threshold(const cvk_face_detector_yn_t *h) {
    const auto *p = reinterpret_cast<const cvk_face_detector_yn *>(h);
    if (p == nullptr) return 0.0f;
    return guarded([&]() -> float { return p->ptr->getNMSThreshold(); });
}

void cvk_face_detector_yn_set_top_k(cvk_face_detector_yn_t *h, int top_k) {
    auto *p = reinterpret_cast<cvk_face_detector_yn *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setTopK(top_k);
        return 0;
    });
}

int cvk_face_detector_yn_get_top_k(const cvk_face_detector_yn_t *h) {
    const auto *p = reinterpret_cast<const cvk_face_detector_yn *>(h);
    if (p == nullptr) return 0;
    return guarded([&]() -> int { return p->ptr->getTopK(); });
}

int cvk_face_detector_yn_detect(const cvk_face_detector_yn_t *h, const cvk_mat_t *image,
                                cvk_mat_t *faces) {
    const auto *p = reinterpret_cast<const cvk_face_detector_yn *>(h);
    const cv::Mat *im = p != nullptr ? require_const(image) : nullptr;
    cv::Mat *f = p != nullptr ? require(faces) : nullptr;
    if (p == nullptr || im == nullptr || f == nullptr) return 0;
    return guarded([&]() -> int { return p->ptr->detect(*im, *f); });
}

void cvk_face_detector_yn_release(cvk_face_detector_yn_t *h) {
    delete reinterpret_cast<cvk_face_detector_yn *>(h);
}

/* =========================================================================
 * cv::FaceRecognizerSF
 * ========================================================================= */

cvk_face_recognizer_sf_t *cvk_face_recognizer_sf_create(const char *model, const char *config,
                                                        int backend_id, int target_id) {
    if (model == nullptr) {
        record_error("null model path");
        return nullptr;
    }
    return guarded([&]() -> cvk_face_recognizer_sf_t * {
        auto *h = new cvk_face_recognizer_sf;
        h->ptr = cv::FaceRecognizerSF::create(
            std::string(model), config != nullptr ? std::string(config) : std::string(),
            backend_id, target_id);
        return reinterpret_cast<cvk_face_recognizer_sf_t *>(h);
    });
}

cvk_face_recognizer_sf_t *cvk_face_recognizer_sf_create_from_buffers(
    const char *framework, const cvk_mat_t *buffer_model, const cvk_mat_t *buffer_config,
    int backend_id, int target_id) {
    const cv::Mat *bm = require_const(buffer_model);
    const cv::Mat *bc = require_const(buffer_config);
    if (framework == nullptr || bm == nullptr || bc == nullptr) return nullptr;
    return guarded([&]() -> cvk_face_recognizer_sf_t * {
        std::vector<unsigned char> model = mat_to_bytes(*bm);
        std::vector<unsigned char> config = mat_to_bytes(*bc);
        auto *h = new cvk_face_recognizer_sf;
        h->ptr = cv::FaceRecognizerSF::create(std::string(framework), model, config,
                                              backend_id, target_id);
        return reinterpret_cast<cvk_face_recognizer_sf_t *>(h);
    });
}

void cvk_face_recognizer_sf_align_crop(const cvk_face_recognizer_sf_t *h,
                                       const cvk_mat_t *src_img, const cvk_mat_t *face_box,
                                       cvk_mat_t *aligned_img) {
    const auto *p = reinterpret_cast<const cvk_face_recognizer_sf *>(h);
    const cv::Mat *src = p != nullptr ? require_const(src_img) : nullptr;
    const cv::Mat *box = p != nullptr ? require_const(face_box) : nullptr;
    cv::Mat *aligned = p != nullptr ? require(aligned_img) : nullptr;
    if (p == nullptr || src == nullptr || box == nullptr || aligned == nullptr) return;
    guarded([&]() -> int {
        p->ptr->alignCrop(*src, *box, *aligned);
        return 0;
    });
}

void cvk_face_recognizer_sf_feature(const cvk_face_recognizer_sf_t *h,
                                    const cvk_mat_t *aligned_img, cvk_mat_t *face_feature) {
    const auto *p = reinterpret_cast<const cvk_face_recognizer_sf *>(h);
    const cv::Mat *aligned = p != nullptr ? require_const(aligned_img) : nullptr;
    cv::Mat *feature = p != nullptr ? require(face_feature) : nullptr;
    if (p == nullptr || aligned == nullptr || feature == nullptr) return;
    guarded([&]() -> int {
        p->ptr->feature(*aligned, *feature);
        return 0;
    });
}

double cvk_face_recognizer_sf_match(const cvk_face_recognizer_sf_t *h,
                                    const cvk_mat_t *feature1, const cvk_mat_t *feature2,
                                    int dis_type) {
    const auto *p = reinterpret_cast<const cvk_face_recognizer_sf *>(h);
    const cv::Mat *f1 = p != nullptr ? require_const(feature1) : nullptr;
    const cv::Mat *f2 = p != nullptr ? require_const(feature2) : nullptr;
    if (p == nullptr || f1 == nullptr || f2 == nullptr) return 0.0;
    return guarded([&]() -> double { return p->ptr->match(*f1, *f2, dis_type); });
}

void cvk_face_recognizer_sf_release(cvk_face_recognizer_sf_t *h) {
    delete reinterpret_cast<cvk_face_recognizer_sf *>(h);
}

/* =========================================================================
 * cv::mcc::CChecker
 * ========================================================================= */

cvk_c_checker_t *cvk_c_checker_create(void) {
    return guarded([&]() -> cvk_c_checker_t * {
        auto *h = new cvk_c_checker;
        h->ptr = cv::mcc::CChecker::create();
        return reinterpret_cast<cvk_c_checker_t *>(h);
    });
}

void cvk_c_checker_set_box(cvk_c_checker_t *h, const cvk_mat_t *box) {
    auto *p = reinterpret_cast<cvk_c_checker *>(h);
    const cv::Mat *b = p != nullptr ? require_const(box) : nullptr;
    if (p == nullptr || b == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setBox(mat_to_points2f(*b));
        return 0;
    });
}

void cvk_c_checker_set_charts_rgb(cvk_c_checker_t *h, const cvk_mat_t *charts) {
    auto *p = reinterpret_cast<cvk_c_checker *>(h);
    const cv::Mat *c = p != nullptr ? require_const(charts) : nullptr;
    if (p == nullptr || c == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setChartsRGB(*c);
        return 0;
    });
}

void cvk_c_checker_set_charts_y_cb_cr(cvk_c_checker_t *h, const cvk_mat_t *charts) {
    auto *p = reinterpret_cast<cvk_c_checker *>(h);
    const cv::Mat *c = p != nullptr ? require_const(charts) : nullptr;
    if (p == nullptr || c == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setChartsYCbCr(*c);
        return 0;
    });
}

void cvk_c_checker_set_cost(cvk_c_checker_t *h, float cost) {
    auto *p = reinterpret_cast<cvk_c_checker *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setCost(cost);
        return 0;
    });
}

void cvk_c_checker_set_center(cvk_c_checker_t *h, double x, double y) {
    auto *p = reinterpret_cast<cvk_c_checker *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setCenter(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
        return 0;
    });
}

cvk_mat_t *cvk_c_checker_get_box(const cvk_c_checker_t *h) {
    const auto *p = reinterpret_cast<const cvk_c_checker *>(h);
    if (p == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        return points2f_to_mat(p->ptr->getBox());
    });
}

cvk_mat_t *cvk_c_checker_get_color_charts(const cvk_c_checker_t *h) {
    const auto *p = reinterpret_cast<const cvk_c_checker *>(h);
    if (p == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        return points2f_to_mat(p->ptr->getColorCharts());
    });
}

cvk_mat_t *cvk_c_checker_get_charts_rgb(const cvk_c_checker_t *h, int get_stats) {
    const auto *p = reinterpret_cast<const cvk_c_checker *>(h);
    if (p == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat charts = p->ptr->getChartsRGB(get_stats != 0);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(charts));
    });
}

cvk_mat_t *cvk_c_checker_get_charts_y_cb_cr(const cvk_c_checker_t *h) {
    const auto *p = reinterpret_cast<const cvk_c_checker *>(h);
    if (p == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat charts = p->ptr->getChartsYCbCr();
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(charts));
    });
}

float cvk_c_checker_get_cost(const cvk_c_checker_t *h) {
    const auto *p = reinterpret_cast<const cvk_c_checker *>(h);
    if (p == nullptr) return 0.0f;
    return guarded([&]() -> float { return p->ptr->getCost(); });
}

void cvk_c_checker_get_center(const cvk_c_checker_t *h, double *out2) {
    const auto *p = reinterpret_cast<const cvk_c_checker *>(h);
    if (p == nullptr || out2 == nullptr) return;
    guarded([&]() -> int {
        const cv::Point2f center = p->ptr->getCenter();
        out2[0] = center.x;
        out2[1] = center.y;
        return 0;
    });
}

void cvk_c_checker_release(cvk_c_checker_t *h) {
    delete reinterpret_cast<cvk_c_checker *>(h);
}

CVK_ALG_FUNCS(c_checker)

/* =========================================================================
 * cv::mcc::CCheckerDetector
 * ========================================================================= */

cvk_c_checker_detector_t *cvk_c_checker_detector_create(void) {
    return guarded([&]() -> cvk_c_checker_detector_t * {
        auto *h = new cvk_c_checker_detector;
        h->ptr = cv::mcc::CCheckerDetector::create();
        return reinterpret_cast<cvk_c_checker_detector_t *>(h);
    });
}

int cvk_c_checker_detector_process_with_roi(const cvk_c_checker_detector_t *h,
                                            const cvk_mat_t *image, const cvk_mat_t *roi,
                                            int nc) {
    const auto *p = reinterpret_cast<const cvk_c_checker_detector *>(h);
    const cv::Mat *im = p != nullptr ? require_const(image) : nullptr;
    const cv::Mat *r = p != nullptr ? require_const(roi) : nullptr;
    if (p == nullptr || im == nullptr || r == nullptr) return 0;
    return guarded([&]() -> int {
        return p->ptr->process(*im, mat_to_rects(*r), nc) ? 1 : 0;
    });
}

int cvk_c_checker_detector_process(const cvk_c_checker_detector_t *h, const cvk_mat_t *image,
                                   int nc) {
    const auto *p = reinterpret_cast<const cvk_c_checker_detector *>(h);
    const cv::Mat *im = p != nullptr ? require_const(image) : nullptr;
    if (p == nullptr || im == nullptr) return 0;
    return guarded([&]() -> int {
        return p->ptr->process(*im, nc) ? 1 : 0;
    });
}

int cvk_c_checker_detector_get_list(const cvk_c_checker_detector_t *h,
                                    cvk_c_checker_t **out, int capacity) {
    const auto *p = reinterpret_cast<const cvk_c_checker_detector *>(h);
    if (p == nullptr) return -1;
    return guarded([&]() -> int {
        const std::vector<cv::Ptr<cv::mcc::CChecker>> list = p->ptr->getListColorChecker();
        const int n = static_cast<int>(list.size());
        if (out != nullptr && capacity > 0) {
            const int filled = n < capacity ? n : capacity;
            for (int i = 0; i < filled; ++i) {
                auto *c = new cvk_c_checker;
                c->ptr = list[static_cast<size_t>(i)];
                out[i] = reinterpret_cast<cvk_c_checker_t *>(c);
            }
        }
        return n;
    });
}

cvk_mat_t *cvk_c_checker_detector_get_ref_colors(const cvk_c_checker_detector_t *h) {
    const auto *p = reinterpret_cast<const cvk_c_checker_detector *>(h);
    if (p == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat colors = p->ptr->getRefColors();
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(colors));
    });
}

void cvk_c_checker_detector_set_detection_params(cvk_c_checker_detector_t *h,
                                                 int adaptive_thresh_win_size_min,
                                                 int adaptive_thresh_win_size_max,
                                                 int adaptive_thresh_win_size_step,
                                                 double adaptive_thresh_constant,
                                                 double min_contours_area_rate,
                                                 double min_contours_area,
                                                 double confidence_threshold,
                                                 double min_contour_solidity,
                                                 double find_candidates_approx_poly_d_peps_multiplier,
                                                 int border_width,
                                                 float b0factor, float max_error,
                                                 int min_contour_points_allowed,
                                                 int min_contour_length_allowed,
                                                 int min_inter_contour_distance,
                                                 int min_inter_checker_distance,
                                                 int min_image_size,
                                                 int min_group_size) {
    auto *p = reinterpret_cast<cvk_c_checker_detector *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        cv::mcc::DetectorParametersMCC params;
        params.adaptiveThreshWinSizeMin = adaptive_thresh_win_size_min;
        params.adaptiveThreshWinSizeMax = adaptive_thresh_win_size_max;
        params.adaptiveThreshWinSizeStep = adaptive_thresh_win_size_step;
        params.adaptiveThreshConstant = adaptive_thresh_constant;
        params.minContoursAreaRate = min_contours_area_rate;
        params.minContoursArea = min_contours_area;
        params.confidenceThreshold = confidence_threshold;
        params.minContourSolidity = min_contour_solidity;
        params.findCandidatesApproxPolyDPEpsMultiplier = find_candidates_approx_poly_d_peps_multiplier;
        params.borderWidth = border_width;
        params.B0factor = b0factor;
        params.maxError = max_error;
        params.minContourPointsAllowed = min_contour_points_allowed;
        params.minContourLengthAllowed = min_contour_length_allowed;
        params.minInterContourDistance = min_inter_contour_distance;
        params.minInterCheckerDistance = min_inter_checker_distance;
        params.minImageSize = min_image_size;
        params.minGroupSize = min_group_size;
        p->ptr->setDetectionParams(params);
        return 0;
    });
}

void cvk_c_checker_detector_get_detection_params(const cvk_c_checker_detector_t *h,
                                                 double *out18) {
    const auto *p = reinterpret_cast<const cvk_c_checker_detector *>(h);
    if (p == nullptr || out18 == nullptr) return;
    guarded([&]() -> int {
        const cv::mcc::DetectorParametersMCC &d = p->ptr->getDetectionParams();
        out18[0] = d.adaptiveThreshWinSizeMin;
        out18[1] = d.adaptiveThreshWinSizeMax;
        out18[2] = d.adaptiveThreshWinSizeStep;
        out18[3] = d.adaptiveThreshConstant;
        out18[4] = d.minContoursAreaRate;
        out18[5] = d.minContoursArea;
        out18[6] = d.confidenceThreshold;
        out18[7] = d.minContourSolidity;
        out18[8] = d.findCandidatesApproxPolyDPEpsMultiplier;
        out18[9] = d.borderWidth;
        out18[10] = d.B0factor;
        out18[11] = d.maxError;
        out18[12] = d.minContourPointsAllowed;
        out18[13] = d.minContourLengthAllowed;
        out18[14] = d.minInterContourDistance;
        out18[15] = d.minInterCheckerDistance;
        out18[16] = d.minImageSize;
        out18[17] = d.minGroupSize;
        return 0;
    });
}

void cvk_c_checker_detector_set_use_dnn_model(cvk_c_checker_detector_t *h, int use_dnn) {
    auto *p = reinterpret_cast<cvk_c_checker_detector *>(h);
    if (p == nullptr) return;
    guarded([&]() -> int {
        p->ptr->setUseDnnModel(use_dnn != 0);
        return 0;
    });
}

int cvk_c_checker_detector_get_use_dnn_model(const cvk_c_checker_detector_t *h) {
    const auto *p = reinterpret_cast<const cvk_c_checker_detector *>(h);
    if (p == nullptr) return 0;
    return guarded([&]() -> int { return p->ptr->getUseDnnModel() ? 1 : 0; });
}

void cvk_c_checker_detector_release(cvk_c_checker_detector_t *h) {
    delete reinterpret_cast<cvk_c_checker_detector *>(h);
}

CVK_ALG_FUNCS(c_checker_detector)

} /* extern "C" */

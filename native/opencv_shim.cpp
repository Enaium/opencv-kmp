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
 * Implementation of the cvk_ C ABI over C++ OpenCV (see opencv_kmp.h).
 *
 * Every exported function runs inside a catch-all guard so no C++ exception
 * ever escapes into cinterop or JNI callers; failures are reported through
 * cvk_last_error() plus the documented NULL / NaN / zero conventions.
 */
#include "opencv_kmp.h"

#include <opencv2/core.hpp>
#include <opencv2/core/bindings_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/features.hpp>
#if !defined(__ANDROID__)
#include <opencv2/highgui.hpp>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>
/** Opaque CLAHE handle backing cvk_clahe_t; defined at file scope so it
 *  completes the type the C header forward-declared. */
struct cvk_clahe { cv::Ptr<cv::CLAHE> ptr; };


namespace {

thread_local std::string g_last_error;
thread_local bool g_has_error = false;

void record_error(const char *message) {
    g_has_error = true;
    try {
        g_last_error = message != nullptr ? message : "unknown error";
    } catch (...) {
        g_has_error = false;
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
    // Default-initialized result (NULL pointers / zeros); callers document
    // their own failure convention on top of cvk_last_error().
    return decltype(body())();
}

cvk_scalar_t scalar_of(const cv::Scalar &s) {
    cvk_scalar_t out;
    out.v0 = s[0];
    out.v1 = s[1];
    out.v2 = s[2];
    out.v3 = s[3];
    return out;
}

cv::Scalar cv_scalar(cvk_scalar_t s) { return {s.v0, s.v1, s.v2, s.v3}; }

cv::Rect cv_rect(cvk_rect_t r) { return {r.x, r.y, r.width, r.height}; }

/** Requires a live handle; records an error and returns nullptr otherwise. */
cv::Mat *require(cvk_mat_t *mat) {
    if (mat == nullptr) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<cv::Mat *>(mat);
}

const cv::Mat *require_const(const cvk_mat_t *mat) { return require(const_cast<cvk_mat_t *>(mat)); }

template <typename T>
double read_at(const cv::Mat &m, int row, int col, int channel) {
    const T *p = m.ptr<T>(row, col);
    return static_cast<double>(p[channel]);
}

template <typename T>
void write_at(cv::Mat &m, int row, int col, int channel, double value) {
    T *p = m.ptr<T>(row, col);
    p[channel] = cv::saturate_cast<T>(value);
}



/* ---- Contour flat-buffer helpers -------------------------------------
 * Wire format: [uint32 le count][per contour: uint32 le npoints][int32 le x,y ...]
 */

void put_u32le(unsigned char *p, unsigned int value) {
    p[0] = static_cast<unsigned char>(value & 0xFFu);
    p[1] = static_cast<unsigned char>((value >> 8) & 0xFFu);
    p[2] = static_cast<unsigned char>((value >> 16) & 0xFFu);
    p[3] = static_cast<unsigned char>((value >> 24) & 0xFFu);
}

unsigned int get_u32le(const unsigned char *p) {
    return static_cast<unsigned int>(p[0]) |
           (static_cast<unsigned int>(p[1]) << 8) |
           (static_cast<unsigned int>(p[2]) << 16) |
           (static_cast<unsigned int>(p[3]) << 24);
}

unsigned char *encode_contours(const std::vector<std::vector<cv::Point>> &contours,
                               size_t *out_len) {
    size_t total = 4;
    for (const auto &pts : contours) total += 4 + pts.size() * 8;
    auto *buf = static_cast<unsigned char *>(std::malloc(total));
    if (buf == nullptr) throw std::bad_alloc();
    put_u32le(buf, static_cast<unsigned int>(contours.size()));
    size_t off = 4;
    for (const auto &pts : contours) {
        put_u32le(buf + off, static_cast<unsigned int>(pts.size()));
        off += 4;
        for (const cv::Point &pt : pts) {
            put_u32le(buf + off, static_cast<unsigned int>(pt.x));
            put_u32le(buf + off + 4, static_cast<unsigned int>(pt.y));
            off += 8;
        }
    }
    if (out_len != nullptr) *out_len = total;
    return buf;
}

std::vector<std::vector<cv::Point>> decode_contours(const unsigned char *flat, size_t len) {
    if (flat == nullptr || len < 4) {
        throw cv::Exception(cv::Error::StsParseError, "malformed contour buffer", __func__, __FILE__, __LINE__);
    }
    const unsigned int count = get_u32le(flat);
    std::vector<std::vector<cv::Point>> contours;
    contours.reserve(count);
    size_t off = 4;
    for (unsigned int i = 0; i < count; ++i) {
        if (off + 4 > len) {
            throw cv::Exception(cv::Error::StsParseError, "truncated contour buffer", __func__, __FILE__, __LINE__);
        }
        const unsigned int n = get_u32le(flat + off);
        off += 4;
        if (off + static_cast<size_t>(n) * 8 > len) {
            throw cv::Exception(cv::Error::StsParseError, "truncated contour points", __func__, __FILE__, __LINE__);
        }
        std::vector<cv::Point> pts;
        pts.reserve(n);
        for (unsigned int j = 0; j < n; ++j) {
            const int x = static_cast<int>(get_u32le(flat + off));
            const int y = static_cast<int>(get_u32le(flat + off + 4));
            off += 8;
            pts.emplace_back(x, y);
        }
        contours.push_back(std::move(pts));
    }
    return contours;
}

/** Decodes a buffer holding exactly one contour. */
std::vector<cv::Point> single_contour(const unsigned char *flat, size_t len) {
    std::vector<std::vector<cv::Point>> contours = decode_contours(flat, len);
    if (contours.size() != 1) {
        throw cv::Exception(cv::Error::StsBadArg, "expected exactly one contour", __func__, __FILE__, __LINE__);
    }
    return std::move(contours[0]);
}

/** Default morphology kernel when the caller passes NULL. */
cv::Mat default_kernel() { return cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); }
} // namespace

extern "C" {

/* =========================================================================
 * Info / errors
 * ========================================================================= */

const char *cvk_version(void) {
    static const char *version = CV_VERSION;
    return version;
}

const char *cvk_last_error(void) { return g_has_error ? g_last_error.c_str() : nullptr; }

void cvk_clear_error(void) {
    g_has_error = false;
    g_last_error.clear();
}

/* =========================================================================
 * Mat lifecycle
 * ========================================================================= */

cvk_mat_t *cvk_mat_create(int rows, int cols, int type) {
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(rows, cols, type));
    });
}

cvk_mat_t *cvk_mat_create_filled(int rows, int cols, int type, cvk_scalar_t value) {
    return guarded([&]() -> cvk_mat_t * {
        // Mat(rows, cols, type, Scalar) routes through scalarToRawData,
        // which rejects multi-channel fills on this OpenCV build; setTo
        // handles every depth/channel combination.
        auto *m = new cv::Mat(rows, cols, type);
        m->setTo(cv_scalar(value));
        return reinterpret_cast<cvk_mat_t *>(m);
    });
}

cvk_mat_t *cvk_mat_zeros(int rows, int cols, int type) {
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(cv::Mat::zeros(rows, cols, type)));
    });
}

cvk_mat_t *cvk_mat_ones(int rows, int cols, int type) {
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(cv::Mat::ones(rows, cols, type)));
    });
}

cvk_mat_t *cvk_mat_eye(int rows, int cols, int type) {
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(cv::Mat::eye(rows, cols, type)));
    });
}

cvk_mat_t *cvk_mat_clone(const cvk_mat_t *mat) {
    const cv::Mat *src = require_const(mat);
    if (src == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(src->clone()));
    });
}

cvk_mat_t *cvk_mat_roi(const cvk_mat_t *mat, cvk_rect_t rect) {
    const cv::Mat *src = require_const(mat);
    if (src == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        // operator()(Rect) shares pixels and increments the reference count,
        // so releasing the ROI never frees the parent's buffer early.
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat((*src)(cv_rect(rect))));
    });
}

void cvk_mat_release(cvk_mat_t *mat) {
    if (mat == nullptr) return;
    auto *m = reinterpret_cast<cv::Mat *>(mat);
    guarded([&]() { delete m; });
}

/* =========================================================================
 * Mat properties
 * ========================================================================= */

int cvk_mat_rows(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->rows; }) : -1;
}

int cvk_mat_cols(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->cols; }) : -1;
}

int cvk_mat_type(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->type(); }) : -1;
}

int cvk_mat_channels(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->channels(); }) : -1;
}

size_t cvk_mat_elem_size(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->elemSize(); }) : size_t(0);
}

size_t cvk_mat_total(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->total(); }) : size_t(0);
}

int cvk_mat_is_empty(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return 1;
    return guarded([&] { return m->empty() ? 1 : 0; });
}

unsigned char *cvk_mat_data(cvk_mat_t *mat) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        if (m->empty()) return nullptr;
        return m->data;
    });
}

double cvk_mat_get(const cvk_mat_t *mat, int row, int col, int channel) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return std::nan("");
    return guarded([&]() -> double {
        switch (m->depth()) {
            case CV_8U: return read_at<uchar>(*m, row, col, channel);
            case CV_8S: return read_at<schar>(*m, row, col, channel);
            case CV_16U: return read_at<ushort>(*m, row, col, channel);
            case CV_16S: return read_at<short>(*m, row, col, channel);
            case CV_32S: return read_at<int>(*m, row, col, channel);
            case CV_32F: return read_at<float>(*m, row, col, channel);
            case CV_64F: return read_at<double>(*m, row, col, channel);
            default: throw cv::Exception(cv::Error::StsUnsupportedFormat, "unsupported matrix depth", __func__, __FILE__, __LINE__);
        }
    });
}

void cvk_mat_set(cvk_mat_t *mat, int row, int col, int channel, double value) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        switch (m->depth()) {
            case CV_8U: write_at<uchar>(*m, row, col, channel, value); break;
            case CV_8S: write_at<schar>(*m, row, col, channel, value); break;
            case CV_16U: write_at<ushort>(*m, row, col, channel, value); break;
            case CV_16S: write_at<short>(*m, row, col, channel, value); break;
            case CV_32S: write_at<int>(*m, row, col, channel, value); break;
            case CV_32F: write_at<float>(*m, row, col, channel, value); break;
            case CV_64F: write_at<double>(*m, row, col, channel, value); break;
            default: throw cv::Exception(cv::Error::StsUnsupportedFormat, "unsupported matrix depth", __func__, __FILE__, __LINE__);
        }
        return nullptr;
    });
}

/* =========================================================================
 * Conversions / arithmetic
 * ========================================================================= */

cvk_mat_t *cvk_mat_convert_to(const cvk_mat_t *mat, int rtype, double alpha, double beta) {
    const cv::Mat *src = require_const(mat);
    if (src == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        src->convertTo(*dst, rtype, alpha, beta);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

#define BINARY_OP(name, expr)                                                          \
    cvk_mat_t *name(const cvk_mat_t *a, const cvk_mat_t *b) {                          \
        const cv::Mat *ma = require_const(a);                                          \
        const cv::Mat *mb = require_const(b);                                          \
        if (ma == nullptr || mb == nullptr) return nullptr;                            \
        return guarded([&]() -> cvk_mat_t * {                                          \
            auto *dst = new cv::Mat();                                                 \
            (void)expr;                                                                \
            return reinterpret_cast<cvk_mat_t *>(dst);                                 \
        });                                                                   \
    }

BINARY_OP(cvk_mat_add, cv::add(*ma, *mb, *dst))
BINARY_OP(cvk_mat_subtract, cv::subtract(*ma, *mb, *dst))
BINARY_OP(cvk_mat_absdiff, cv::absdiff(*ma, *mb, *dst))
BINARY_OP(cvk_mat_bitwise_and, cv::bitwise_and(*ma, *mb, *dst))
BINARY_OP(cvk_mat_bitwise_or, cv::bitwise_or(*ma, *mb, *dst))
BINARY_OP(cvk_mat_bitwise_xor, cv::bitwise_xor(*ma, *mb, *dst))
BINARY_OP(cvk_mat_min, cv::min(*ma, *mb, *dst))
BINARY_OP(cvk_mat_max, cv::max(*ma, *mb, *dst))

#undef BINARY_OP

cvk_mat_t *cvk_mat_multiply(const cvk_mat_t *a, const cvk_mat_t *b, double scale) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::multiply(*ma, *mb, *dst, scale);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_mat_divide(const cvk_mat_t *a, const cvk_mat_t *b) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::divide(*ma, *mb, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/** Per-channel arithmetic with a scalar (cv::add/subtract/multiply/divide). */
#define SCALAR_OP(name, expr)                                                    \
    cvk_mat_t *name(const cvk_mat_t *a, cvk_scalar_t s) {                       \
        const cv::Mat *ma = require_const(a);                                   \
        if (ma == nullptr) return nullptr;                                      \
        return guarded([&]() -> cvk_mat_t * {                                   \
            auto *dst = new cv::Mat();                                          \
            (void)expr;                                                         \
            return reinterpret_cast<cvk_mat_t *>(dst);                          \
        });                                                                      \
    }

SCALAR_OP(cvk_mat_add_scalar, cv::add(*ma, cv_scalar(s), *dst))
SCALAR_OP(cvk_mat_subtract_scalar, cv::subtract(*ma, cv_scalar(s), *dst))
SCALAR_OP(cvk_mat_multiply_scalar, cv::multiply(*ma, cv_scalar(s), *dst))
SCALAR_OP(cvk_mat_divide_scalar, cv::divide(*ma, cv_scalar(s), *dst))

#undef SCALAR_OP

cvk_mat_t *cvk_mat_scale_add(const cvk_mat_t *mat, double alpha, double beta) {
    const cv::Mat *src = require_const(mat);
    if (src == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        src->convertTo(*dst, -1, alpha, beta);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_mat_bitwise_not(const cvk_mat_t *a) {
    const cv::Mat *ma = require_const(a);
    if (ma == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::bitwise_not(*ma, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_mat_in_range(const cvk_mat_t *mat, cvk_scalar_t lower, cvk_scalar_t upper) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::inRange(*m, cv_scalar(lower), cv_scalar(upper), *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_mat_transpose(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::transpose(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_mat_flip(const cvk_mat_t *mat, int flip_code) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::flip(*m, *dst, flip_code);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_scalar_t cvk_mat_mean(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return cvk_scalar_t{};
    return guarded([&] { return scalar_of(cv::mean(*m)); });
}

cvk_scalar_t cvk_mat_sum(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return cvk_scalar_t{};
    return guarded([&] { return scalar_of(cv::sum(*m)); });
}

void cvk_mat_mean_stddev(const cvk_mat_t *mat, double *out8) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr || out8 == nullptr) return;
    guarded([&]() -> void * {
        cv::Mat mean;
        cv::Mat stddev;
        cv::meanStdDev(*m, mean, stddev);
        for (int i = 0; i < 4; ++i) {
            out8[i] = i < mean.total() ? mean.at<double>(i) : 0.0;
            out8[4 + i] = i < stddev.total() ? stddev.at<double>(i) : 0.0;
        }
        return nullptr;
    });
}

void cvk_mat_min_max_loc(const cvk_mat_t *mat, double *out6) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr || out6 == nullptr) return;
    guarded([&]() -> void * {
        double min_val = 0.0;
        double max_val = 0.0;
        cv::Point min_loc;
        cv::Point max_loc;
        cv::minMaxLoc(*m, &min_val, &max_val, &min_loc, &max_loc);
        out6[0] = min_val;
        out6[1] = max_val;
        out6[2] = min_loc.x;
        out6[3] = min_loc.y;
        out6[4] = max_loc.x;
        out6[5] = max_loc.y;
        return nullptr;
    });
}

int cvk_mat_count_non_zero(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return -1;
    return guarded([&] { return static_cast<int>(cv::countNonZero(*m)); });
}

/* =========================================================================
 * imgproc
 * ========================================================================= */

cvk_mat_t *cvk_cvt_color(const cvk_mat_t *src, int code) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::cvtColor(*m, *dst, code);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_resize(const cvk_mat_t *src, int width, int height, int interpolation) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::resize(*m, *dst, cv::Size(width, height), 0.0, 0.0, interpolation);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_gaussian_blur(const cvk_mat_t *src, int kernel_width, int kernel_height,
                             double sigma_x, double sigma_y) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::GaussianBlur(*m, *dst, cv::Size(kernel_width, kernel_height), sigma_x, sigma_y,
                         cv::BORDER_DEFAULT);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_median_blur(const cvk_mat_t *src, int kernel_size) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::medianBlur(*m, *dst, kernel_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_threshold(const cvk_mat_t *src, double thresh, double maxval, int type) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::threshold(*m, *dst, thresh, maxval, type);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_adaptive_threshold(const cvk_mat_t *src, double max_value, int method,
                                  int type, int block_size, double c) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::adaptiveThreshold(*m, *dst, max_value, method, type, block_size, c);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_canny(const cvk_mat_t *src, double threshold1, double threshold2,
                     int aperture_size, int l2_gradient) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::Canny(*m, *dst, threshold1, threshold2, aperture_size, l2_gradient != 0);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_sobel(const cvk_mat_t *src, int dx, int dy, int kernel_size) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::Sobel(*m, *dst, -1, dx, dy, kernel_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_laplacian(const cvk_mat_t *src, int kernel_size) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::Laplacian(*m, *dst, -1, kernel_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_rectangle(cvk_mat_t *mat, int x1, int y1, int x2, int y2,
                   cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::rectangle(*m, cv::Point(x1, y1), cv::Point(x2, y2), cv_scalar(color), thickness);
        return nullptr;
    });
}

void cvk_circle(cvk_mat_t *mat, int center_x, int center_y, int radius,
                cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::circle(*m, cv::Point(center_x, center_y), radius, cv_scalar(color), thickness);
        return nullptr;
    });
}

void cvk_line(cvk_mat_t *mat, int x1, int y1, int x2, int y2,
              cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::line(*m, cv::Point(x1, y1), cv::Point(x2, y2), cv_scalar(color), thickness);
        return nullptr;
    });
}

/* =========================================================================
 * imgcodecs
 * ========================================================================= */

cvk_mat_t *cvk_imread(const char *filename, int flags) {
    if (filename == nullptr) {
        record_error("null filename");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat(cv::imread(filename, flags));
        if (dst->empty()) {
            delete dst;
            record_error("imread produced an empty image");
            return static_cast<cvk_mat_t *>(nullptr);
        }
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

int cvk_imwrite(const char *filename, const cvk_mat_t *mat) {
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return 0;
    return guarded([&] { return cv::imwrite(filename, *m) ? 1 : 0; });
}

unsigned char *cvk_imencode(const char *ext, const cvk_mat_t *mat, size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    if (ext == nullptr) {
        record_error("null extension");
        return nullptr;
    }
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<uchar> buf;
        if (!cv::imencode(ext, *m, buf)) {
            record_error("imencode failed");
            return static_cast<unsigned char *>(nullptr);
        }
        auto *copy = static_cast<unsigned char *>(std::malloc(buf.size()));
        if (copy == nullptr) throw std::bad_alloc();
        std::memcpy(copy, buf.data(), buf.size());
        if (out_len != nullptr) *out_len = buf.size();
        return copy;
    });
}

cvk_mat_t *cvk_imdecode(const unsigned char *data, size_t len, int flags) {
    if (data == nullptr && len > 0) {
        record_error("null encoded data");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat(
                cv::imdecode(cv::Mat1b(1, static_cast<int>(len),
                                       const_cast<uchar *>(data)),
                             flags));
        if (dst->empty()) {
            delete dst;
            record_error("imdecode produced an empty image");
            return static_cast<cvk_mat_t *>(nullptr);
        }
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_free_buffer(unsigned char *buffer) { std::free(buffer); }

/* =========================================================================
 * Mat members (core)
 * ========================================================================= */

cvk_mat_t *cvk_mat_reshape(const cvk_mat_t *mat, int channels, int rows) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m->reshape(channels, rows)));
    });
}

cvk_mat_t *cvk_mat_row_range(const cvk_mat_t *mat, int start, int end) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        // rowRange shares pixels and increments the reference count.
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m->rowRange(start, end)));
    });
}

cvk_mat_t *cvk_mat_col_range(const cvk_mat_t *mat, int start, int end) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m->colRange(start, end)));
    });
}

cvk_mat_t *cvk_mat_diag(const cvk_mat_t *mat, int d) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m->diag(d)));
    });
}

void cvk_mat_set_identity(cvk_mat_t *mat, double scale) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::setIdentity(*m, cv::Scalar(scale));
        return nullptr;
    });
}

double cvk_mat_dot(const cvk_mat_t *a, const cvk_mat_t *b) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return -1.0;
    return guarded([&] { return ma->dot(*mb); });
}

cvk_mat_t *cvk_mat_inv(const cvk_mat_t *mat, int method) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        const double result = cv::invert(*m, *dst, method);
        if (result == 0.0) {
            delete dst;
            record_error("matrix is singular and cannot be inverted");
            return static_cast<cvk_mat_t *>(nullptr);
        }
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

double cvk_mat_determinant(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return std::nan("");
    return guarded([&] { return cv::determinant(*m); });
}

cvk_scalar_t cvk_mat_trace(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return cvk_scalar_t{};
    return guarded([&] { return scalar_of(cv::trace(*m)); });
}

/* =========================================================================
 * core: array operations
 * ========================================================================= */

int cvk_split(const cvk_mat_t *src, cvk_mat_t **out, int max_count) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr || out == nullptr || max_count <= 0) return -1;
    return guarded([&]() -> int {
        for (int i = 0; i < max_count; ++i) out[i] = nullptr;
        std::vector<cv::Mat> channels;
        cv::split(*m, channels);
        const int filled = static_cast<int>(channels.size()) < max_count
                                   ? static_cast<int>(channels.size())
                                   : max_count;
        for (int i = 0; i < filled; ++i) {
            out[i] = reinterpret_cast<cvk_mat_t *>(new cv::Mat(channels[static_cast<size_t>(i)]));
        }
        return static_cast<int>(channels.size());
    });
}

cvk_mat_t *cvk_merge(const cvk_mat_t **mv, int count) {
    if (mv == nullptr || count <= 0) {
        record_error("null merge inputs");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Mat> channels;
        channels.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const cv::Mat *m = require_const(mv[i]);
            if (m == nullptr) throw cv::Exception(cv::Error::StsBadArg, "null Mat in merge list", __func__, __FILE__, __LINE__);
            channels.push_back(*m);
        }
        auto *dst = new cv::Mat();
        cv::merge(channels, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_hconcat(const cvk_mat_t *a, const cvk_mat_t *b) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::hconcat(*ma, *mb, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_vconcat(const cvk_mat_t *a, const cvk_mat_t *b) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::vconcat(*ma, *mb, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

double cvk_norm(const cvk_mat_t *src, int norm_type) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return -1.0;
    return guarded([&] { return cv::norm(*m, norm_type); });
}

double cvk_norm_diff(const cvk_mat_t *a, const cvk_mat_t *b, int norm_type) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return -1.0;
    return guarded([&] { return cv::norm(*ma, *mb, norm_type); });
}

cvk_mat_t *cvk_normalize(const cvk_mat_t *src, double alpha, double beta,
                         int norm_type, int dtype) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::normalize(*m, *dst, alpha, beta, norm_type, dtype);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_lut(const cvk_mat_t *src, const cvk_mat_t *lut) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *lm = require_const(lut);
    if (m == nullptr || lm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::LUT(*m, *lm, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_rotate(const cvk_mat_t *src, int code) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::rotate(*m, *dst, code);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_copy_make_border(const cvk_mat_t *src, int top, int bottom,
                                int left, int right, int border_type,
                                cvk_scalar_t value) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::copyMakeBorder(*m, *dst, top, bottom, left, right, border_type, cv_scalar(value));
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_add_weighted(const cvk_mat_t *a, double alpha,
                            const cvk_mat_t *b, double beta, double gamma) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::addWeighted(*ma, alpha, *mb, beta, gamma, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_convert_scale_abs(const cvk_mat_t *src, double alpha, double beta) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::convertScaleAbs(*m, *dst, alpha, beta);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_compare(const cvk_mat_t *a, const cvk_mat_t *b, int cmp_op) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::compare(*ma, *mb, *dst, cmp_op);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_solve(const cvk_mat_t *a, const cvk_mat_t *b, int flags) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        if (!cv::solve(*ma, *mb, *dst, flags)) {
            delete dst;
            record_error("solve found no solution");
            return static_cast<cvk_mat_t *>(nullptr);
        }
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_repeat(const cvk_mat_t *src, int nx, int ny) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::repeat(*m, ny, nx, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_transform(const cvk_mat_t *src, const cvk_mat_t *m) {
    const cv::Mat *sm = require_const(src);
    const cv::Mat *mm = require_const(m);
    if (sm == nullptr || mm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::transform(*sm, *dst, *mm);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_perspective_transform(const cvk_mat_t *src, const cvk_mat_t *m) {
    const cv::Mat *sm = require_const(src);
    const cv::Mat *mm = require_const(m);
    if (sm == nullptr || mm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::perspectiveTransform(*sm, *dst, *mm);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_pow(const cvk_mat_t *src, double power) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::pow(*m, power, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_sqrt(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::sqrt(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_exp(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::exp(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_log(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::log(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_magnitude(const cvk_mat_t *x, const cvk_mat_t *y) {
    const cv::Mat *mx = require_const(x);
    const cv::Mat *my = require_const(y);
    if (mx == nullptr || my == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::magnitude(*mx, *my, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_phase(const cvk_mat_t *x, const cvk_mat_t *y, int angle_in_degrees) {
    const cv::Mat *mx = require_const(x);
    const cv::Mat *my = require_const(y);
    if (mx == nullptr || my == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::phase(*mx, *my, *dst, angle_in_degrees != 0);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_cart_to_polar(const cvk_mat_t *x, const cvk_mat_t *y,
                       int angle_in_degrees, cvk_mat_t **magnitude,
                       cvk_mat_t **angle) {
    if (magnitude != nullptr) *magnitude = nullptr;
    if (angle != nullptr) *angle = nullptr;
    const cv::Mat *mx = require_const(x);
    const cv::Mat *my = require_const(y);
    if (mx == nullptr || my == nullptr || magnitude == nullptr || angle == nullptr) return;
    guarded([&]() -> void * {
        auto *mag = new cv::Mat();
        auto *ang = new cv::Mat();
        cv::cartToPolar(*mx, *my, *mag, *ang, angle_in_degrees != 0);
        *magnitude = reinterpret_cast<cvk_mat_t *>(mag);
        *angle = reinterpret_cast<cvk_mat_t *>(ang);
        return nullptr;
    });
}

void cvk_polar_to_cart(const cvk_mat_t *magnitude, const cvk_mat_t *angle,
                       int angle_in_degrees, cvk_mat_t **x, cvk_mat_t **y) {
    if (x != nullptr) *x = nullptr;
    if (y != nullptr) *y = nullptr;
    const cv::Mat *mag = require_const(magnitude);
    const cv::Mat *ang = require_const(angle);
    if (mag == nullptr || ang == nullptr || x == nullptr || y == nullptr) return;
    guarded([&]() -> void * {
        auto *px = new cv::Mat();
        auto *py = new cv::Mat();
        cv::polarToCart(*mag, *ang, *px, *py, angle_in_degrees != 0);
        *x = reinterpret_cast<cvk_mat_t *>(px);
        *y = reinterpret_cast<cvk_mat_t *>(py);
        return nullptr;
    });
}

void cvk_patch_nans(cvk_mat_t *mat, double value) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::patchNaNs(*m, value);
        return nullptr;
    });
}

cvk_mat_t *cvk_find_non_zero(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::findNonZero(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

int cvk_has_non_zero(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return -1;
    return guarded([&] { return cv::countNonZero(*m) > 0 ? 1 : 0; });
}

cvk_mat_t *cvk_sort(const cvk_mat_t *src, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::sort(*m, *dst, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_sort_idx(const cvk_mat_t *src, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::sortIdx(*m, *dst, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_reduce(const cvk_mat_t *src, int dim, int rtype, int dtype) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::reduce(*m, *dst, dim, rtype, dtype);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_reduce_arg_max(const cvk_mat_t *src, int dim) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::reduceArgMax(*m, *dst, dim);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_reduce_arg_min(const cvk_mat_t *src, int dim) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::reduceArgMin(*m, *dst, dim);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_extract_channel(const cvk_mat_t *src, int coi) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::extractChannel(*m, *dst, coi);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_insert_channel(const cvk_mat_t *src, cvk_mat_t *dst, int coi) {
    const cv::Mat *sm = require_const(src);
    cv::Mat *dm = require(dst);
    if (sm == nullptr || dm == nullptr) return;
    guarded([&]() -> void * {
        cv::insertChannel(*sm, *dm, coi);
        return nullptr;
    });
}

void cvk_randu(cvk_mat_t *dst, cvk_scalar_t low, cvk_scalar_t high) {
    cv::Mat *m = require(dst);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::randu(*m, cv_scalar(low), cv_scalar(high));
        return nullptr;
    });
}

void cvk_randn(cvk_mat_t *dst, cvk_scalar_t mean, cvk_scalar_t stddev) {
    cv::Mat *m = require(dst);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::randn(*m, cv_scalar(mean), cv_scalar(stddev));
        return nullptr;
    });
}

void cvk_set_rng_seed(unsigned long long seed) {
    guarded([&]() -> void * {
        cv::theRNG().state = static_cast<uint64>(seed);
        return nullptr;
    });
}

double cvk_psnr(const cvk_mat_t *a, const cvk_mat_t *b, double r) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return -1.0;
    return guarded([&] { return cv::PSNR(*ma, *mb, r); });
}

cvk_mat_t *cvk_dft(const cvk_mat_t *src, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::dft(*m, *dst, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_idft(const cvk_mat_t *src, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::idft(*m, *dst, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_dct(const cvk_mat_t *src, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::dct(*m, *dst, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_idct(const cvk_mat_t *src, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::idct(*m, *dst, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

int cvk_get_optimal_dft_size(int rowsize) {
    return guarded([&] { return cv::getOptimalDFTSize(rowsize); });
}

cvk_mat_t *cvk_mul_spectrums(const cvk_mat_t *a, const cvk_mat_t *b,
                             int conj_flag, int dft_rows) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::mulSpectrums(*ma, *mb, *dst, conj_flag, dft_rows != 0);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_div_spectrums(const cvk_mat_t *a, const cvk_mat_t *b,
                             int conj_flag) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::divSpectrums(*ma, *mb, *dst, conj_flag);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_gemm(const cvk_mat_t *a, const cvk_mat_t *b, double alpha,
                    const cvk_mat_t *c, double gamma) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    const cv::Mat *mc = c != nullptr ? require_const(c) : nullptr;
    if (c != nullptr && mc == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::gemm(*ma, *mb, alpha, mc != nullptr ? *mc : cv::noArray(), gamma, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_eigen(const cvk_mat_t *src, cvk_mat_t **eigenvalues,
               cvk_mat_t **eigenvectors) {
    if (eigenvalues != nullptr) *eigenvalues = nullptr;
    if (eigenvectors != nullptr) *eigenvectors = nullptr;
    const cv::Mat *m = require_const(src);
    if (m == nullptr || eigenvalues == nullptr || eigenvectors == nullptr) return;
    guarded([&]() -> void * {
        auto *values = new cv::Mat();
        auto *vectors = new cv::Mat();
        cv::eigen(*m, *values, *vectors);
        *eigenvalues = reinterpret_cast<cvk_mat_t *>(values);
        *eigenvectors = reinterpret_cast<cvk_mat_t *>(vectors);
        return nullptr;
    });
}

int cvk_num_threads(void) {
    return guarded([] { return cv::getNumThreads(); });
}

void cvk_set_num_threads(int count) {
    guarded([&]() -> void * {
        cv::setNumThreads(count);
        return nullptr;
    });
}

const char *cvk_build_information(void) {
    // getBuildInformation() returns a fresh string each call; cache one copy.
    static const std::string info = cv::getBuildInformation();
    return info.c_str();
}

cvk_mat_t *cvk_remap(const cvk_mat_t *src, const cvk_mat_t *map1,
                     const cvk_mat_t *map2, int interpolation) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *m1 = require_const(map1);
    if (m == nullptr || m1 == nullptr) return nullptr;
    const cv::Mat *m2 = map2 != nullptr ? require_const(map2) : nullptr;
    if (map2 != nullptr && m2 == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::remap(*m, *dst, *m1, m2 != nullptr ? *m2 : cv::noArray(), interpolation);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * imgproc: filters
 * ========================================================================= */

cvk_mat_t *cvk_blur(const cvk_mat_t *src, int kernel_width, int kernel_height) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::blur(*m, *dst, cv::Size(kernel_width, kernel_height));
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_box_filter(const cvk_mat_t *src, int ddepth, int kernel_width,
                          int kernel_height, int normalize) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::boxFilter(*m, *dst, ddepth, cv::Size(kernel_width, kernel_height),
                      cv::Point(-1, -1), normalize != 0, cv::BORDER_DEFAULT);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_sqr_box_filter(const cvk_mat_t *src, int ddepth,
                              int kernel_width, int kernel_height) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::sqrBoxFilter(*m, *dst, ddepth, cv::Size(kernel_width, kernel_height),
                         cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_bilateral_filter(const cvk_mat_t *src, int d,
                                double sigma_color, double sigma_space) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::bilateralFilter(*m, *dst, d, sigma_color, sigma_space, cv::BORDER_DEFAULT);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_stack_blur(const cvk_mat_t *src, int kernel_size) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::stackBlur(*m, *dst, cv::Size(kernel_size, kernel_size));
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_erode(const cvk_mat_t *src, const cvk_mat_t *kernel, int iterations) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    const cv::Mat *k = kernel != nullptr ? require_const(kernel) : nullptr;
    if (kernel != nullptr && k == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::erode(*m, *dst, k != nullptr ? *k : default_kernel(),
                  cv::Point(-1, -1), iterations);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_dilate(const cvk_mat_t *src, const cvk_mat_t *kernel, int iterations) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    const cv::Mat *k = kernel != nullptr ? require_const(kernel) : nullptr;
    if (kernel != nullptr && k == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::dilate(*m, *dst, k != nullptr ? *k : default_kernel(),
                   cv::Point(-1, -1), iterations);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_morphology_ex(const cvk_mat_t *src, int op,
                             const cvk_mat_t *kernel, int iterations) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    const cv::Mat *k = kernel != nullptr ? require_const(kernel) : nullptr;
    if (kernel != nullptr && k == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::morphologyEx(*m, *dst, op, k != nullptr ? *k : default_kernel(),
                         cv::Point(-1, -1), iterations);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_get_structuring_element(int shape, int width, int height) {
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(
                new cv::Mat(cv::getStructuringElement(shape, cv::Size(width, height))));
    });
}

cvk_mat_t *cvk_get_gaussian_kernel(int ksize, double sigma) {
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(
                new cv::Mat(cv::getGaussianKernel(ksize, sigma)));
    });
}

cvk_mat_t *cvk_filter_2d(const cvk_mat_t *src, const cvk_mat_t *kernel,
                         int ddepth, double delta) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *k = require_const(kernel);
    if (m == nullptr || k == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::filter2D(*m, *dst, ddepth, *k, cv::Point(-1, -1), delta, cv::BORDER_DEFAULT);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_pyr_down(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::pyrDown(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_pyr_up(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::pyrUp(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * imgproc: geometry / warps
 * ========================================================================= */

cvk_mat_t *cvk_warp_affine(const cvk_mat_t *src, const cvk_mat_t *m,
                           int width, int height, int flags) {
    const cv::Mat *sm = require_const(src);
    const cv::Mat *mm = require_const(m);
    if (sm == nullptr || mm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::warpAffine(*sm, *dst, *mm, cv::Size(width, height), flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_warp_perspective(const cvk_mat_t *src, const cvk_mat_t *m,
                                int width, int height, int flags) {
    const cv::Mat *sm = require_const(src);
    const cv::Mat *mm = require_const(m);
    if (sm == nullptr || mm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::warpPerspective(*sm, *dst, *mm, cv::Size(width, height), flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}


cvk_mat_t *cvk_warp_polar(const cvk_mat_t *src, int radius,
                          double center_x, double center_y,
                          double max_radius, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::warpPolar(*m, *dst, cv::Size(radius, m->rows),
                      cv::Point2f(static_cast<float>(center_x), static_cast<float>(center_y)),
                      max_radius, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_get_affine_transform(double sx0, double sy0, double sx1,
                                    double sy1, double sx2, double sy2,
                                    double dx0, double dy0, double dx1,
                                    double dy1, double dx2, double dy2) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Point2f src[] = {{static_cast<float>(sx0), static_cast<float>(sy0)},
                                   {static_cast<float>(sx1), static_cast<float>(sy1)},
                                   {static_cast<float>(sx2), static_cast<float>(sy2)}};
        const cv::Point2f dst[] = {{static_cast<float>(dx0), static_cast<float>(dy0)},
                                   {static_cast<float>(dx1), static_cast<float>(dy1)},
                                   {static_cast<float>(dx2), static_cast<float>(dy2)}};
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(cv::getAffineTransform(src, dst)));
    });
}

cvk_mat_t *cvk_invert_affine_transform(const cvk_mat_t *m) {
    const cv::Mat *mm = require_const(m);
    if (mm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::invertAffineTransform(*mm, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_get_perspective_transform(double sx0, double sy0, double sx1,
                                         double sy1, double sx2, double sy2,
                                         double sx3, double sy3, double dx0,
                                         double dy0, double dx1, double dy1,
                                         double dx2, double dy2, double dx3,
                                         double dy3) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Point2f src[] = {{static_cast<float>(sx0), static_cast<float>(sy0)},
                                   {static_cast<float>(sx1), static_cast<float>(sy1)},
                                   {static_cast<float>(sx2), static_cast<float>(sy2)},
                                   {static_cast<float>(sx3), static_cast<float>(sy3)}};
        const cv::Point2f dst[] = {{static_cast<float>(dx0), static_cast<float>(dy0)},
                                   {static_cast<float>(dx1), static_cast<float>(dy1)},
                                   {static_cast<float>(dx2), static_cast<float>(dy2)},
                                   {static_cast<float>(dx3), static_cast<float>(dy3)}};
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(cv::getPerspectiveTransform(src, dst)));
    });
}

cvk_mat_t *cvk_get_rotation_matrix_2d(double cx, double cy, double angle,
                                      double scale) {
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(
                cv::getRotationMatrix2D(cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
                                        angle, scale)));
    });
}

cvk_mat_t *cvk_get_rect_sub_pix(const cvk_mat_t *src, int width, int height,
                                double cx, double cy) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::getRectSubPix(*m, cv::Size(width, height),
                          cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)), *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_undistort(const cvk_mat_t *src, const cvk_mat_t *camera_matrix,
                         const cvk_mat_t *dist_coeffs) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *cam = require_const(camera_matrix);
    if (m == nullptr || cam == nullptr) return nullptr;
    const cv::Mat *dist = dist_coeffs != nullptr ? require_const(dist_coeffs) : nullptr;
    if (dist_coeffs != nullptr && dist == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::undistort(*m, *dst, *cam, dist != nullptr ? *dist : cv::noArray());
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * imgproc: color / histogram
 * ========================================================================= */

cvk_mat_t *cvk_demosaicing(const cvk_mat_t *src, int code) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::demosaicing(*m, *dst, code);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_apply_colormap(const cvk_mat_t *src, int colormap) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::applyColorMap(*m, *dst, colormap);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_apply_colormap_user(const cvk_mat_t *src, const cvk_mat_t *user_color) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *uc = require_const(user_color);
    if (m == nullptr || uc == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::applyColorMap(*m, *dst, *uc);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_calc_hist(const cvk_mat_t *src, int channel, int hist_size,
                         float min_value, float max_value) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Mat> images{*m};
        std::vector<int> channels{channel};
        std::vector<int> hist_sizes{hist_size};
        std::vector<float> ranges{min_value, max_value};
        auto *hist = new cv::Mat();
        cv::calcHist(images, channels, cv::noArray(), *hist, hist_sizes, ranges);
        // calcHist returns a dims==1 array; flatten it to hist_size x 1 so
        // row/col element access works uniformly.
        if (hist->dims == 1) {
            *hist = hist->reshape(1, hist_size);
        }
        return reinterpret_cast<cvk_mat_t *>(hist);
    });
}

cvk_mat_t *cvk_calc_back_project(const cvk_mat_t *src, int channel,
                                 const cvk_mat_t *hist, float min_value,
                                 float max_value) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *hm = require_const(hist);
    if (m == nullptr || hm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        std::vector<cv::Mat> images{*m};
        std::vector<int> channels{channel};
        std::vector<float> ranges{min_value, max_value};
        auto *dst = new cv::Mat();
        cv::calcBackProject(images, channels, *hm, *dst, ranges, 1.0);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

double cvk_compare_hist(const cvk_mat_t *h1, const cvk_mat_t *h2, int method) {
    const cv::Mat *mh1 = require_const(h1);
    const cv::Mat *mh2 = require_const(h2);
    if (mh1 == nullptr || mh2 == nullptr) return -1.0;
    return guarded([&] { return cv::compareHist(*mh1, *mh2, method); });
}

cvk_mat_t *cvk_equalize_hist(const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::equalizeHist(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * imgproc: segmentation / contours / features
 * ========================================================================= */

int cvk_flood_fill(cvk_mat_t *image, int seed_x, int seed_y,
                   cvk_scalar_t new_value, cvk_scalar_t lo_diff,
                   cvk_scalar_t up_diff, int flags) {
    cv::Mat *m = require(image);
    if (m == nullptr) return -1;
    return guarded([&] {
        return cv::floodFill(*m, cv::Point(seed_x, seed_y), cv_scalar(new_value),
                             static_cast<cv::Rect *>(nullptr), cv_scalar(lo_diff),
                             cv_scalar(up_diff), flags);
    });
}

void cvk_watershed(cvk_mat_t *image, cvk_mat_t *markers) {
    cv::Mat *im = require(image);
    cv::Mat *mk = require(markers);
    if (im == nullptr || mk == nullptr) return;
    guarded([&]() -> void * {
        cv::watershed(*im, *mk);
        return nullptr;
    });
}

unsigned char *cvk_find_contours(const cvk_mat_t *src, int mode, int method,
                                 size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(*m, contours, mode, method);
        return encode_contours(contours, out_len);
    });
}

void cvk_draw_contours(cvk_mat_t *image, const unsigned char *flat, size_t len,
                       int contour_index, cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(image);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        std::vector<std::vector<cv::Point>> contours = decode_contours(flat, len);
        cv::drawContours(*m, contours, contour_index, cv_scalar(color), thickness, cv::LINE_8);
        return nullptr;
    });
}

double cvk_contour_area(const unsigned char *flat, size_t len) {
    return guarded([&] { return cv::contourArea(single_contour(flat, len)); });
}

double cvk_arc_length(const unsigned char *flat, size_t len, int closed) {
    return guarded([&] {
        return cv::arcLength(single_contour(flat, len), closed != 0);
    });
}

void cvk_bounding_rect(const unsigned char *flat, size_t len, int out[4]) {
    if (out == nullptr) return;
    guarded([&]() -> void * {
        const cv::Rect r = cv::boundingRect(single_contour(flat, len));
        out[0] = r.x;
        out[1] = r.y;
        out[2] = r.width;
        out[3] = r.height;
        return nullptr;
    });
}

unsigned char *cvk_approx_poly_dp(const unsigned char *flat, size_t len,
                                  double epsilon, int closed, size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    return guarded([&]() -> unsigned char * {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(single_contour(flat, len), approx, epsilon, closed != 0);
        return encode_contours({approx}, out_len);
    });
}

void cvk_min_area_rect(const unsigned char *flat, size_t len, double out[5]) {
    if (out == nullptr) return;
    guarded([&]() -> void * {
        const cv::RotatedRect r = cv::minAreaRect(single_contour(flat, len));
        out[0] = r.center.x;
        out[1] = r.center.y;
        out[2] = r.size.width;
        out[3] = r.size.height;
        out[4] = r.angle;
        return nullptr;
    });
}

void cvk_min_enclosing_circle(const unsigned char *flat, size_t len, double out[3]) {
    if (out == nullptr) return;
    guarded([&]() -> void * {
        cv::Point2f center;
        float radius = 0.0f;
        cv::minEnclosingCircle(single_contour(flat, len), center, radius);
        out[0] = center.x;
        out[1] = center.y;
        out[2] = radius;
        return nullptr;
    });
}

void cvk_moments(const cvk_mat_t *arr, int binary_image, double out[10]) {
    if (out == nullptr) return;
    const cv::Mat *m = require_const(arr);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        const cv::Moments mo = cv::moments(*m, binary_image != 0);
        out[0] = mo.m00;
        out[1] = mo.m10;
        out[2] = mo.m01;
        out[3] = mo.m20;
        out[4] = mo.m11;
        out[5] = mo.m02;
        out[6] = mo.m30;
        out[7] = mo.m21;
        out[8] = mo.m12;
        out[9] = mo.m03;
        return nullptr;
    });
}

double cvk_match_shapes(const cvk_mat_t *a, const cvk_mat_t *b, int method) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return -1.0;
    return guarded([&] { return cv::matchShapes(*ma, *mb, method, 0.0); });
}

cvk_mat_t *cvk_hough_lines_p(const cvk_mat_t *src, double rho, double theta,
                             int threshold, double min_line_length,
                             double max_line_gap) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *lines = new cv::Mat();
        cv::HoughLinesP(*m, *lines, rho, theta, threshold, min_line_length, max_line_gap);
        return reinterpret_cast<cvk_mat_t *>(lines);
    });
}

cvk_mat_t *cvk_hough_lines(const cvk_mat_t *src, double rho, double theta,
                           int threshold, double srn, double stn) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *lines = new cv::Mat();
        cv::HoughLines(*m, *lines, rho, theta, threshold, srn, stn);
        return reinterpret_cast<cvk_mat_t *>(lines);
    });
}

cvk_mat_t *cvk_hough_circles(const cvk_mat_t *src, int method, double dp,
                             double min_dist, double param1, double param2,
                             int min_radius, int max_radius) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *circles = new cv::Mat();
        cv::HoughCircles(*m, *circles, method, dp, min_dist, param1, param2,
                         min_radius, max_radius);
        return reinterpret_cast<cvk_mat_t *>(circles);
    });
}

cvk_mat_t *cvk_corner_harris(const cvk_mat_t *src, int block_size, int ksize,
                             double k) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::cornerHarris(*m, *dst, block_size, ksize, k, cv::BORDER_DEFAULT);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_corner_min_eigen_val(const cvk_mat_t *src, int block_size,
                                    int ksize) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::cornerMinEigenVal(*m, *dst, block_size, ksize, cv::BORDER_DEFAULT);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_good_features_to_track(const cvk_mat_t *src, int max_corners,
                                      double quality_level, double min_distance,
                                      int block_size, int use_harris_detector,
                                      double k) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *corners = new cv::Mat();
        cv::goodFeaturesToTrack(*m, *corners, max_corners, quality_level,
                                min_distance, cv::noArray(), block_size,
                                use_harris_detector != 0, k);
        return reinterpret_cast<cvk_mat_t *>(corners);
    });
}

cvk_mat_t *cvk_match_template(const cvk_mat_t *image, const cvk_mat_t *templ,
                              int method) {
    const cv::Mat *mi = require_const(image);
    const cv::Mat *mt = require_const(templ);
    if (mi == nullptr || mt == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::matchTemplate(*mi, *mt, *dst, method);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_distance_transform(const cvk_mat_t *src, int distance_type,
                                  int mask_size) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::distanceTransform(*m, *dst, distance_type, mask_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_integral(const cvk_mat_t *src, int sdepth) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *sum = new cv::Mat();
        cv::integral(*m, *sum, sdepth);
        return reinterpret_cast<cvk_mat_t *>(sum);
    });
}

int cvk_connected_components(const cvk_mat_t *src, cvk_mat_t **labels,
                             int connectivity, int ltype) {
    if (labels != nullptr) *labels = nullptr;
    const cv::Mat *m = require_const(src);
    if (m == nullptr || labels == nullptr) return -1;
    return guarded([&]() -> int {
        auto *lab = new cv::Mat();
        const int count = cv::connectedComponents(*m, *lab, connectivity, ltype);
        *labels = reinterpret_cast<cvk_mat_t *>(lab);
        return count;
    });
}

int cvk_connected_components_with_stats(const cvk_mat_t *src, cvk_mat_t **labels,
                                        cvk_mat_t **stats, cvk_mat_t **centroids,
                                        int connectivity, int ltype) {
    if (labels != nullptr) *labels = nullptr;
    if (stats != nullptr) *stats = nullptr;
    if (centroids != nullptr) *centroids = nullptr;
    const cv::Mat *m = require_const(src);
    if (m == nullptr || labels == nullptr || stats == nullptr || centroids == nullptr) return -1;
    return guarded([&]() -> int {
        auto *lab = new cv::Mat();
        auto *st = new cv::Mat();
        auto *ce = new cv::Mat();
        const int count =
                cv::connectedComponentsWithStats(*m, *lab, *st, *ce, connectivity, ltype);
        *labels = reinterpret_cast<cvk_mat_t *>(lab);
        *stats = reinterpret_cast<cvk_mat_t *>(st);
        *centroids = reinterpret_cast<cvk_mat_t *>(ce);
        return count;
    });
}

cvk_mat_t *cvk_pyr_mean_shift_filtering(const cvk_mat_t *src, double sp,
                                        double sr, int max_level) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::pyrMeanShiftFiltering(*m, *dst, sp, sr, max_level);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

double cvk_threshold_with_mask(const cvk_mat_t *src, const cvk_mat_t *mask,
                               cvk_mat_t *dst, double thresh, double maxval,
                               int type) {
    const cv::Mat *m = require_const(src);
    cv::Mat *dm = require(dst);
    if (m == nullptr || dm == nullptr) return -1.0;
    const cv::Mat *mk = mask != nullptr ? require_const(mask) : nullptr;
    if (mask != nullptr && mk == nullptr) return -1.0;
    return guarded([&] {
        return cv::thresholdWithMask(*m, *dm, mk != nullptr ? *mk : cv::noArray(),
                                     thresh, maxval, type);
    });
}

cvk_mat_t *cvk_create_hanning_window(int width, int height, int type) {
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::createHanningWindow(*dst, cv::Size(width, height), type);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_accumulate(const cvk_mat_t *src, cvk_mat_t *dst) {
    const cv::Mat *m = require_const(src);
    cv::Mat *d = require(dst);
    if (m == nullptr || d == nullptr) return;
    guarded([&]() -> void * {
        cv::accumulate(*m, *d);
        return nullptr;
    });
}

void cvk_accumulate_square(const cvk_mat_t *src, cvk_mat_t *dst) {
    const cv::Mat *m = require_const(src);
    cv::Mat *d = require(dst);
    if (m == nullptr || d == nullptr) return;
    guarded([&]() -> void * {
        cv::accumulateSquare(*m, *d);
        return nullptr;
    });
}

void cvk_accumulate_product(const cvk_mat_t *a, const cvk_mat_t *b, cvk_mat_t *dst) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    cv::Mat *d = require(dst);
    if (ma == nullptr || mb == nullptr || d == nullptr) return;
    guarded([&]() -> void * {
        cv::accumulateProduct(*ma, *mb, *d);
        return nullptr;
    });
}

void cvk_accumulate_weighted(const cvk_mat_t *src, cvk_mat_t *dst, double alpha) {
    const cv::Mat *m = require_const(src);
    cv::Mat *d = require(dst);
    if (m == nullptr || d == nullptr) return;
    guarded([&]() -> void * {
        cv::accumulateWeighted(*m, *d, alpha);
        return nullptr;
    });
}

/* =========================================================================
 * imgproc: drawing
 * ========================================================================= */

void cvk_arrowed_line(cvk_mat_t *mat, int x1, int y1, int x2, int y2,
                      cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::arrowedLine(*m, cv::Point(x1, y1), cv::Point(x2, y2), cv_scalar(color),
                        thickness, cv::LINE_8);
        return nullptr;
    });
}

void cvk_draw_marker(cvk_mat_t *mat, int x, int y, int marker_type, int size,
                     cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::drawMarker(*m, cv::Point(x, y), cv_scalar(color), marker_type, size,
                       thickness, cv::LINE_8);
        return nullptr;
    });
}

void cvk_ellipse(cvk_mat_t *mat, int cx, int cy, int axes_x, int axes_y,
                 double angle, double start_angle, double end_angle,
                 cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        cv::ellipse(*m, cv::Point(cx, cy), cv::Size(axes_x, axes_y), angle,
                    start_angle, end_angle, cv_scalar(color), thickness, cv::LINE_8);
        return nullptr;
    });
}

void cvk_fill_poly(cvk_mat_t *mat, const unsigned char *flat, size_t len,
                   cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        std::vector<std::vector<cv::Point>> polygons = decode_contours(flat, len);
        if (thickness < 0) {
            cv::fillPoly(*m, polygons, cv_scalar(color), cv::LINE_8);
        } else {
            cv::polylines(*m, polygons, true, cv_scalar(color), thickness, cv::LINE_8);
        }
        return nullptr;
    });
}

void cvk_polylines(cvk_mat_t *mat, const unsigned char *flat, size_t len,
                   int closed, cvk_scalar_t color, int thickness) {
    cv::Mat *m = require(mat);
    if (m == nullptr) return;
    guarded([&]() -> void * {
        std::vector<std::vector<cv::Point>> pts = decode_contours(flat, len);
        cv::polylines(*m, pts, closed != 0, cv_scalar(color), thickness, cv::LINE_8);
        return nullptr;
    });
}

/* =========================================================================
 * imgproc: CLAHE
 * ========================================================================= */

cvk_clahe_t *cvk_clahe_create(double clip_limit, int tile_width, int tile_height) {
    return guarded([&]() -> cvk_clahe_t * {
        auto *handle = new cvk_clahe;
        handle->ptr = cv::createCLAHE(clip_limit, cv::Size(tile_width, tile_height));
        return reinterpret_cast<cvk_clahe_t *>(handle);
    });
}

cvk_mat_t *cvk_clahe_apply(cvk_clahe_t *clahe, const cvk_mat_t *src) {
    const cv::Mat *m = require_const(src);
    if (clahe == nullptr || clahe->ptr.empty() || m == nullptr) {
        record_error("null CLAHE handle or source");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        clahe->ptr->apply(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_clahe_set_clip_limit(cvk_clahe_t *clahe, double clip_limit) {
    if (clahe == nullptr || clahe->ptr.empty()) {
        record_error("null CLAHE handle");
        return;
    }
    guarded([&]() -> void * {
        clahe->ptr->setClipLimit(clip_limit);
        return nullptr;
    });
}

void cvk_clahe_release(cvk_clahe_t *clahe) {
    if (clahe == nullptr) return;
    guarded([&]() -> void * {
        delete clahe;
        return nullptr;
    });
}

/* =========================================================================
 * imgcodecs additions
 * ========================================================================= */

int cvk_imcount(const char *filename) {
    if (filename == nullptr) {
        record_error("null filename");
        return -1;
    }
    return guarded([&] { return static_cast<int>(cv::imcount(filename)); });
}

int cvk_have_image_reader(const char *ext) {
    if (ext == nullptr) {
        record_error("null extension");
        return 0;
    }
    return guarded([&] { return cv::haveImageReader(ext) ? 1 : 0; });
}

int cvk_have_image_writer(const char *ext) {
    if (ext == nullptr) {
        record_error("null extension");
        return 0;
    }
    return guarded([&] { return cv::haveImageWriter(ext) ? 1 : 0; });
}

int cvk_imwrite_params(const char *filename, const cvk_mat_t *mat,
                       const int *params, size_t params_len) {
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return 0;
    return guarded([&] {
        const std::vector<int> p(params, params + params_len);
        return cv::imwrite(filename, *m, p) ? 1 : 0;
    });
}

unsigned char *cvk_imencode_params(const char *ext, const cvk_mat_t *mat,
                                   const int *params, size_t params_len,
                                   size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    if (ext == nullptr) {
        record_error("null extension");
        return nullptr;
    }
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        const std::vector<int> p(params, params + params_len);
        std::vector<uchar> buf;
        if (!cv::imencode(ext, *m, buf, p)) {
            record_error("imencode failed");
            return static_cast<unsigned char *>(nullptr);
        }
        auto *copy = static_cast<unsigned char *>(std::malloc(buf.size()));
        if (copy == nullptr) throw std::bad_alloc();
        std::memcpy(copy, buf.data(), buf.size());
        if (out_len != nullptr) *out_len = buf.size();
        return copy;
    });
}

/* =========================================================================
 * Official Java/Python SDK parity
 * ========================================================================= */

const char *cvk_mat_dump(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> const char * {
        // Mirrors Java Mat.dump(): shape header then every element value.
        std::ostringstream os;
        os << "mat(" << m->rows << ", " << m->cols << ", type=" << m->type()
           << ") = [";
        for (int r = 0; r < m->rows; ++r) {
            os << "\n  ";
            for (int c = 0; c < m->cols; ++c) {
                for (int ch = 0; ch < m->channels(); ++ch) {
                    os << ' ' << cvk_mat_get(mat, r, c, ch);
                }
            }
        }
        os << "\n]";
        static std::string content;
        content = os.str();
        return content.c_str();
    });
}

int cvk_mat_is_continuous(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->isContinuous() ? 1 : 0; }) : 0;
}

int cvk_mat_is_submatrix(const cvk_mat_t *mat) {
    const cv::Mat *m = require_const(mat);
    return m != nullptr ? guarded([&] { return m->isSubmatrix() ? 1 : 0; }) : 0;
}

cvk_mat_t *cvk_mat_adjust_roi(const cvk_mat_t *mat, int dtop, int dbottom,
                              int dleft, int dright) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        // adjustROI mutates the header in place; work on a header copy that
        // shares the same pixel buffer, mirroring the official semantics.
        auto *view = new cv::Mat(*m);
        view->adjustROI(dtop, dbottom, dleft, dright);
        return reinterpret_cast<cvk_mat_t *>(view);
    });
}

void cvk_mat_locate_roi(const cvk_mat_t *mat, int out[4]) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr || out == nullptr) return;
    guarded([&]() -> void * {
        cv::Size whole;
        cv::Point offset;
        m->locateROI(whole, offset);
        out[0] = offset.x;
        out[1] = offset.y;
        out[2] = m->cols;
        out[3] = m->rows;
        return nullptr;
    });
}


cvk_mat_t *cvk_mat_cross(const cvk_mat_t *a, const cvk_mat_t *b) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    if (ma == nullptr || mb == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        *dst = ma->cross(*mb);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

size_t cvk_mat_put_values(cvk_mat_t *mat, int row, int col,
                          const double *values, size_t count) {
    cv::Mat *m = require(mat);
    if (m == nullptr || values == nullptr) return 0;
    return guarded([&]() -> size_t {
        const int chs = m->channels();
        size_t written = 0;
        // Element-wise along the row starting at (row, col), interleaving
        // channels exactly like Java Mat.put(row, col, double[]); stops at
        // the row end or once count values are consumed.
        for (int c = col; c < m->cols && written < count; ++c) {
            for (int ch = 0; ch < chs && written < count; ++ch, ++written) {
                switch (m->depth()) {
                    case CV_8U: write_at<uchar>(*m, row, c, ch, values[written]); break;
                    case CV_8S: write_at<schar>(*m, row, c, ch, values[written]); break;
                    case CV_16U: write_at<ushort>(*m, row, c, ch, values[written]); break;
                    case CV_16S: write_at<short>(*m, row, c, ch, values[written]); break;
                    case CV_32S: write_at<int>(*m, row, c, ch, values[written]); break;
                    case CV_32F: write_at<float>(*m, row, c, ch, values[written]); break;
                    case CV_64F: write_at<double>(*m, row, c, ch, values[written]); break;
                    default: throw cv::Exception(cv::Error::StsUnsupportedFormat, "unsupported matrix depth", __func__, __FILE__, __LINE__);
                }
            }
        }
        return written;
    });
}

size_t cvk_mat_get_values(const cvk_mat_t *mat, int row, int col,
                          double *out, size_t count) {
    const cv::Mat *m = require_const(mat);
    if (m == nullptr || out == nullptr) return 0;
    return guarded([&]() -> size_t {
        const int chs = m->channels();
        size_t read = 0;
        // Mirror of put_values: element-wise along the row starting at
        // (row, col), stopping at the row end.
        for (int c = col; c < m->cols && read < count; ++c) {
            for (int ch = 0; ch < chs && read < count; ++ch, ++read) {
                switch (m->depth()) {
                    case CV_8U: out[read] = read_at<uchar>(*m, row, c, ch); break;
                    case CV_8S: out[read] = read_at<schar>(*m, row, c, ch); break;
                    case CV_16U: out[read] = read_at<ushort>(*m, row, c, ch); break;
                    case CV_16S: out[read] = read_at<short>(*m, row, c, ch); break;
                    case CV_32S: out[read] = read_at<int>(*m, row, c, ch); break;
                    case CV_32F: out[read] = read_at<float>(*m, row, c, ch); break;
                    case CV_64F: out[read] = read_at<double>(*m, row, c, ch); break;
                    default: throw cv::Exception(cv::Error::StsUnsupportedFormat, "unsupported matrix depth", __func__, __FILE__, __LINE__);
                }
            }
        }
        return read;
    });
}

void cvk_kmeans(const cvk_mat_t *data, int k, cvk_mat_t *best_labels,
                int crit_type, int crit_max_count, double crit_epsilon,
                int attempts, int flags, cvk_mat_t **centers,
                double *out_compactness) {
    const cv::Mat *d = require_const(data);
    cv::Mat *labels = require(best_labels);
    if (d == nullptr || labels == nullptr || centers == nullptr) return;
    *centers = nullptr;
    guarded([&]() -> void * {
        auto *ctr = new cv::Mat();
        const double compactness =
                cv::kmeans(*d, k, *labels,
                           cv::TermCriteria(crit_type, crit_max_count, crit_epsilon),
                           attempts, flags, *ctr);
        *centers = reinterpret_cast<cvk_mat_t *>(ctr);
        if (out_compactness != nullptr) *out_compactness = compactness;
        return nullptr;
    });
}

void cvk_svd_decomp(const cvk_mat_t *src, cvk_mat_t **w, cvk_mat_t **u,
                    cvk_mat_t **vt, int flags) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr || w == nullptr || u == nullptr || vt == nullptr) return;
    *w = nullptr;
    *u = nullptr;
    *vt = nullptr;
    guarded([&]() -> void * {
        auto *wm = new cv::Mat();
        auto *um = new cv::Mat();
        auto *vtm = new cv::Mat();
        cv::SVDecomp(*m, *wm, *um, *vtm, flags);
        *w = reinterpret_cast<cvk_mat_t *>(wm);
        *u = reinterpret_cast<cvk_mat_t *>(um);
        *vt = reinterpret_cast<cvk_mat_t *>(vtm);
        return nullptr;
    });
}

void cvk_svd_backsubst(const cvk_mat_t *w, const cvk_mat_t *u,
                       const cvk_mat_t *vt, const cvk_mat_t *b,
                       cvk_mat_t **dst) {
    const cv::Mat *mw = require_const(w);
    const cv::Mat *mu = require_const(u);
    const cv::Mat *mvt = require_const(vt);
    const cv::Mat *mb = require_const(b);
    if (mw == nullptr || mu == nullptr || mvt == nullptr || mb == nullptr ||
        dst == nullptr) return;
    *dst = nullptr;
    guarded([&]() -> void * {
        auto *out = new cv::Mat();
        cv::SVBackSubst(*mw, *mu, *mvt, *mb, *out);
        *dst = reinterpret_cast<cvk_mat_t *>(out);
        return nullptr;
    });
}

void cvk_pca_compute(const cvk_mat_t *data, cvk_mat_t **mean,
                     cvk_mat_t **vectors, int max_components) {
    const cv::Mat *d = require_const(data);
    if (d == nullptr || mean == nullptr || vectors == nullptr) return;
    *mean = nullptr;
    *vectors = nullptr;
    guarded([&]() -> void * {
        auto *mu = new cv::Mat();
        auto *vec = new cv::Mat();
        cv::PCACompute(*d, *mu, *vec, max_components);
        *mean = reinterpret_cast<cvk_mat_t *>(mu);
        *vectors = reinterpret_cast<cvk_mat_t *>(vec);
        return nullptr;
    });
}

void cvk_pca_compute_variance(const cvk_mat_t *data, cvk_mat_t **mean,
                              cvk_mat_t **vectors, double retained_variance) {
    const cv::Mat *d = require_const(data);
    if (d == nullptr || mean == nullptr || vectors == nullptr) return;
    *mean = nullptr;
    *vectors = nullptr;
    guarded([&]() -> void * {
        auto *mu = new cv::Mat();
        auto *vec = new cv::Mat();
        cv::PCACompute(*d, *mu, *vec, retained_variance);
        *mean = reinterpret_cast<cvk_mat_t *>(mu);
        *vectors = reinterpret_cast<cvk_mat_t *>(vec);
        return nullptr;
    });
}

void cvk_pca_project(const cvk_mat_t *data, const cvk_mat_t *mean,
                     const cvk_mat_t *vectors, cvk_mat_t **result) {
    const cv::Mat *d = require_const(data);
    const cv::Mat *mu = require_const(mean);
    const cv::Mat *vec = require_const(vectors);
    if (d == nullptr || mu == nullptr || vec == nullptr || result == nullptr) return;
    *result = nullptr;
    guarded([&]() -> void * {
        // Rebuild the PCA from caller-supplied mean and eigenvectors rather
        // than recomputing them, matching Java Core.PCAProject.
        cv::PCA pca;
        pca.mean = *mu;
        pca.eigenvectors = *vec;
        auto *out = new cv::Mat();
        pca.project(*d, *out);
        *result = reinterpret_cast<cvk_mat_t *>(out);
        return nullptr;
    });
}

void cvk_pca_backproject(const cvk_mat_t *data, const cvk_mat_t *mean,
                         const cvk_mat_t *vectors, cvk_mat_t **result) {
    const cv::Mat *d = require_const(data);
    const cv::Mat *mu = require_const(mean);
    const cv::Mat *vec = require_const(vectors);
    if (d == nullptr || mu == nullptr || vec == nullptr || result == nullptr) return;
    *result = nullptr;
    guarded([&]() -> void * {
        cv::PCA pca;
        pca.mean = *mu;
        pca.eigenvectors = *vec;
        auto *out = new cv::Mat();
        pca.backProject(*d, *out);
        *result = reinterpret_cast<cvk_mat_t *>(out);
        return nullptr;
    });
}

double cvk_mahalanobis(const cvk_mat_t *v1, const cvk_mat_t *v2,
                       const cvk_mat_t *icovar) {
    const cv::Mat *mv1 = require_const(v1);
    const cv::Mat *mv2 = require_const(v2);
    const cv::Mat *micovar = require_const(icovar);
    if (mv1 == nullptr || mv2 == nullptr || micovar == nullptr) return -1.0;
    return guarded([&] { return cv::Mahalanobis(*mv1, *mv2, *micovar); });
}

unsigned char *cvk_corner_sub_pix(const cvk_mat_t *image,
                                  const unsigned char *flat, size_t len,
                                  int win_w, int win_h,
                                  int zero_w, int zero_h,
                                  int crit_type, int crit_max_count,
                                  double crit_epsilon, size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    const cv::Mat *m = require_const(image);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<cv::Point> corners = single_contour(flat, len);
        std::vector<cv::Point2f> pts;
        pts.reserve(corners.size());
        for (const cv::Point &pt : corners) {
            pts.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
        }
        cv::cornerSubPix(*m, pts, cv::Size(win_w, win_h), cv::Size(zero_w, zero_h),
                         cv::TermCriteria(crit_type, crit_max_count, crit_epsilon));
        // OpenCV Java keeps integer points: round the refined coordinates
        // back to int pairs and re-encode the single contour buffer.
        std::vector<cv::Point> refined;
        refined.reserve(pts.size());
        for (const cv::Point2f &pt : pts) {
            refined.emplace_back(cvRound(pt.x), cvRound(pt.y));
        }
        return encode_contours({refined}, out_len);
    });
}

float cvk_emd(const cvk_mat_t *signature1, const cvk_mat_t *signature2,
              int dist_type) {
    const cv::Mat *m1 = require_const(signature1);
    const cv::Mat *m2 = require_const(signature2);
    if (m1 == nullptr || m2 == nullptr) return -1.0f;
    return guarded([&] { return cv::EMD(*m1, *m2, dist_type); });
}

void cvk_grab_cut(const cvk_mat_t *img, cvk_mat_t *mask,
                  int rx, int ry, int rw, int rh,
                  cvk_mat_t *bgd_model, cvk_mat_t *fgd_model,
                  int iterations, int mode) {
    const cv::Mat *im = require_const(img);
    cv::Mat *mk = require(mask);
    cv::Mat *bgd = require(bgd_model);
    cv::Mat *fgd = require(fgd_model);
    if (im == nullptr || mk == nullptr || bgd == nullptr || fgd == nullptr) return;
    guarded([&]() -> void * {
        // The rect only participates in the INIT_WITH_RECT pass.
        const cv::Rect rect = mode == cv::GC_INIT_WITH_RECT
                                      ? cv::Rect(rx, ry, rw, rh)
                                      : cv::Rect();
        cv::grabCut(*im, *mk, rect, *bgd, *fgd, iterations, mode);
        return nullptr;
    });
}

} /* extern "C" */


/* =========================================================================
 * highgui (desktop backends: Win32UI / Cocoa; headless Linux and Android
 * get no-op stubs that report on stderr so callers can detect it)
 * ========================================================================= */
#if defined(__ANDROID__)
#define CVK_HIGHGUI_UNAVAILABLE 1
#endif

void cvk_named_window(const char *winname, int flags) {
#ifdef CVK_HIGHGUI_UNAVAILABLE
    (void)winname; (void)flags;
    fprintf(stderr, "opencv-kmp: highgui is not available on this platform\n");
#else
    const cv::Mat *m = nullptr; (void)m;
    guarded([&]() -> cvk_mat_t * {
        cv::namedWindow(winname, flags);
        return nullptr;
    });
#endif
}

void cvk_resize_window(const char *winname, int width, int height) {
#ifdef CVK_HIGHGUI_UNAVAILABLE
    (void)winname; (void)width; (void)height;
    fprintf(stderr, "opencv-kmp: highgui is not available on this platform\n");
#else
    guarded([&]() -> cvk_mat_t * {
        cv::resizeWindow(winname, width, height);
        return nullptr;
    });
#endif
}

void cvk_imshow(const char *winname, const cvk_mat_t *mat) {
#ifdef CVK_HIGHGUI_UNAVAILABLE
    (void)winname; (void)mat;
    fprintf(stderr, "opencv-kmp: highgui is not available on this platform\n");
#else
    const cv::Mat *m = require_const(mat);
    if (m == nullptr) return;
    guarded([&]() -> cvk_mat_t * {
        cv::imshow(winname, *m);
        return nullptr;
    });
#endif
}

int cvk_wait_key(int delay_ms) {
#ifdef CVK_HIGHGUI_UNAVAILABLE
    (void)delay_ms;
    fprintf(stderr, "opencv-kmp: highgui is not available on this platform\n");
    return -1;
#else
    int key = guarded([&]() -> int {
        return cv::waitKey(delay_ms);
    });
    return key;
#endif
}

void cvk_destroy_window(const char *winname) {
#ifdef CVK_HIGHGUI_UNAVAILABLE
    (void)winname;
#else
    guarded([&]() -> cvk_mat_t * {
        cv::destroyWindow(winname);
        return nullptr;
    });
#endif
}

void cvk_destroy_all_windows(void) {
#ifndef CVK_HIGHGUI_UNAVAILABLE
    guarded([&]() -> cvk_mat_t * {
        cv::destroyAllWindows();
        return nullptr;
    });
#endif
}

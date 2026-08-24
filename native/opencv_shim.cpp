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
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

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

} /* extern "C" */

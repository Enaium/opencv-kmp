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
 * cvk_ C ABI implementation for the OpenCV "videoio" module
 * (see opencv_kmp_videoio.h). VideoCapture / VideoWriter are held by value
 * inside the opaque handles so release() (stream close) and open() (reuse)
 * map 1:1 onto the C++ API.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_videoio.h"

#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct cvk_video_capture {
    cv::VideoCapture cap;
};

struct cvk_video_writer {
    cv::VideoWriter writer;
};

namespace {

thread_local std::string g_videoio_str;

void record_error(const char *message) {
    try {
        g_videoio_str = message != nullptr ? message : "unknown error";
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

/** Decodes a CV_32SC1 Mat into the vector<int> the C++ API wants. */
std::vector<int> vec_int_of(const cvk_mat_t *params) {
    std::vector<int> out;
    if (params == nullptr) {
        return out;
    }
    const cv::Mat *m = require_const(params);
    if (m == nullptr || m->empty()) {
        return out;
    }
    if (m->type() != CV_32SC1) {
        record_error("params Mat must be CV_32SC1");
        return out;
    }
    m->copyTo(out);
    return out;
}

/** Encodes a vector<int> as a 1xN CV_32SC1 Mat (MatOfInt wire format). */
cvk_mat_t *vec_int_to_mat(const std::vector<int> &values) {
    cv::Mat m(1, static_cast<int>(values.size()), CV_32SC1);
    if (!values.empty()) {
        std::memcpy(m.data, values.data(), values.size() * sizeof(int));
    }
    return reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
}

/** Encodes the backend-id enum list the registry returns. */
cvk_mat_t *vec_int_to_mat(const std::vector<cv::VideoCaptureAPIs> &values) {
    std::vector<int> ints;
    ints.reserve(values.size());
    for (cv::VideoCaptureAPIs api : values) {
        ints.push_back(static_cast<int>(api));
    }
    return vec_int_to_mat(ints);
}

} // namespace

extern "C" {

/* ---------------------------------------------------------------------------
 * VideoCapture
 * ------------------------------------------------------------------------- */

cvk_video_capture_t *cvk_video_capture_create(void) {
    return guarded([&]() -> cvk_video_capture_t * {
        auto *h = new cvk_video_capture;
        return reinterpret_cast<cvk_video_capture_t *>(h);
    });
}

cvk_video_capture_t *cvk_video_capture_create_index(int index, int api_preference) {
    return guarded([&]() -> cvk_video_capture_t * {
        auto *h = new cvk_video_capture;
        h->cap.open(index, api_preference);
        return reinterpret_cast<cvk_video_capture_t *>(h);
    });
}

cvk_video_capture_t *cvk_video_capture_create_file(const char *filename, int api_preference) {
    return guarded([&]() -> cvk_video_capture_t * {
        if (filename == nullptr) {
            record_error("null filename");
            return nullptr;
        }
        auto *h = new cvk_video_capture;
        h->cap.open(cv::String(filename), api_preference);
        return reinterpret_cast<cvk_video_capture_t *>(h);
    });
}

cvk_video_capture_t *cvk_video_capture_create_index_params(int index, int api_preference,
                                                           const cvk_mat_t *params) {
    return guarded([&]() -> cvk_video_capture_t * {
        auto *h = new cvk_video_capture;
        h->cap.open(index, api_preference, vec_int_of(params));
        return reinterpret_cast<cvk_video_capture_t *>(h);
    });
}

cvk_video_capture_t *cvk_video_capture_create_file_params(const char *filename,
                                                          int api_preference,
                                                          const cvk_mat_t *params) {
    return guarded([&]() -> cvk_video_capture_t * {
        if (filename == nullptr) {
            record_error("null filename");
            return nullptr;
        }
        auto *h = new cvk_video_capture;
        h->cap.open(cv::String(filename), api_preference, vec_int_of(params));
        return reinterpret_cast<cvk_video_capture_t *>(h);
    });
}

int cvk_video_capture_open_index(cvk_video_capture_t *h, int index, int api_preference) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> int { return p->cap.open(index, api_preference) ? 1 : 0; });
}

int cvk_video_capture_open_file(cvk_video_capture_t *h, const char *filename,
                                int api_preference) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> int { return p->cap.open(cv::String(filename), api_preference) ? 1 : 0; });
}

int cvk_video_capture_open_index_params(cvk_video_capture_t *h, int index, int api_preference,
                                        const cvk_mat_t *params) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> int {
        return p->cap.open(index, api_preference, vec_int_of(params)) ? 1 : 0;
    });
}

int cvk_video_capture_open_file_params(cvk_video_capture_t *h, const char *filename,
                                       int api_preference, const cvk_mat_t *params) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> int {
        return p->cap.open(cv::String(filename), api_preference, vec_int_of(params)) ? 1 : 0;
    });
}

int cvk_video_capture_is_opened(const cvk_video_capture_t *h) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    auto *p = reinterpret_cast<const cvk_video_capture *>(h);
    return guarded([&]() -> int { return p->cap.isOpened() ? 1 : 0; });
}

void cvk_video_capture_release(cvk_video_capture_t *h) {
    if (h == nullptr) {
        return;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    guarded([&]() -> int {
        p->cap.release();
        return 0;
    });
}

void cvk_video_capture_delete(cvk_video_capture_t *h) {
    delete reinterpret_cast<cvk_video_capture *>(h);
}

int cvk_video_capture_grab(cvk_video_capture_t *h) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> int { return p->cap.grab() ? 1 : 0; });
}

cvk_mat_t *cvk_video_capture_retrieve(cvk_video_capture_t *h, int flag) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return nullptr;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat frame;
        if (!p->cap.retrieve(frame, flag)) {
            return nullptr;
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(frame)));
    });
}

int cvk_video_capture_read(cvk_video_capture_t *h, cvk_mat_t **out) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    if (out == nullptr) {
        record_error("null out-param");
        return 0;
    }
    *out = nullptr;
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> int {
        cv::Mat frame;
        if (!p->cap.read(frame)) {
            return 0;
        }
        *out = reinterpret_cast<cvk_mat_t *>(new cv::Mat(std::move(frame)));
        return 1;
    });
}

int cvk_video_capture_set(cvk_video_capture_t *h, int prop_id, double value) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    return guarded([&]() -> int { return p->cap.set(prop_id, value) ? 1 : 0; });
}

double cvk_video_capture_get(const cvk_video_capture_t *h, int prop_id) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0.0;
    }
    auto *p = reinterpret_cast<const cvk_video_capture *>(h);
    return guarded([&]() -> double { return p->cap.get(prop_id); });
}

const char *cvk_video_capture_get_backend_name(const cvk_video_capture_t *h) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return nullptr;
    }
    auto *p = reinterpret_cast<const cvk_video_capture *>(h);
    return guarded([&]() -> const char * {
        g_videoio_str = p->cap.getBackendName();
        return g_videoio_str.c_str();
    });
}

void cvk_video_capture_set_exception_mode(cvk_video_capture_t *h, int enable) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return;
    }
    auto *p = reinterpret_cast<cvk_video_capture *>(h);
    guarded([&]() -> int {
        p->cap.setExceptionMode(enable != 0);
        return 0;
    });
}

int cvk_video_capture_get_exception_mode(const cvk_video_capture_t *h) {
    if (h == nullptr) {
        record_error("null VideoCapture handle");
        return 0;
    }
    auto *p = reinterpret_cast<const cvk_video_capture *>(h);
    return guarded([&]() -> int { return p->cap.getExceptionMode() ? 1 : 0; });
}

/* ---------------------------------------------------------------------------
 * VideoWriter
 * ------------------------------------------------------------------------- */

cvk_video_writer_t *cvk_video_writer_create(void) {
    return guarded([&]() -> cvk_video_writer_t * {
        auto *h = new cvk_video_writer;
        return reinterpret_cast<cvk_video_writer_t *>(h);
    });
}

static cvk_video_writer_t *writer_open_file(const char *filename, int api_preference,
                                            int fourcc, double fps, int width, int height,
                                            int is_color) {
    if (filename == nullptr) {
        record_error("null filename");
        return nullptr;
    }
    auto *h = new cvk_video_writer;
    h->writer.open(cv::String(filename), api_preference, fourcc, fps, cv::Size(width, height), is_color != 0);
    return reinterpret_cast<cvk_video_writer_t *>(h);
}

cvk_video_writer_t *cvk_video_writer_create_file(const char *filename, int fourcc, double fps,
                                                 int width, int height, int is_color) {
    return guarded([&]() -> cvk_video_writer_t * {
        return writer_open_file(filename, cv::CAP_ANY, fourcc, fps, width, height, is_color);
    });
}

cvk_video_writer_t *cvk_video_writer_create_file_api(const char *filename, int api_preference,
                                                     int fourcc, double fps, int width,
                                                     int height, int is_color) {
    return guarded([&]() -> cvk_video_writer_t * {
        return writer_open_file(filename, api_preference, fourcc, fps, width, height, is_color);
    });
}

static cvk_video_writer_t *writer_open_file_params(const char *filename, int api_preference,
                                                   int fourcc, double fps, int width,
                                                   int height, const cvk_mat_t *params) {
    if (filename == nullptr) {
        record_error("null filename");
        return nullptr;
    }
    auto *h = new cvk_video_writer;
    h->writer.open(cv::String(filename), api_preference, fourcc, fps, cv::Size(width, height),
                   vec_int_of(params));
    return reinterpret_cast<cvk_video_writer_t *>(h);
}

cvk_video_writer_t *cvk_video_writer_create_file_params(const char *filename, int fourcc,
                                                        double fps, int width, int height,
                                                        const cvk_mat_t *params) {
    return guarded([&]() -> cvk_video_writer_t * {
        return writer_open_file_params(filename, cv::CAP_ANY, fourcc, fps, width, height,
                                       params);
    });
}

cvk_video_writer_t *cvk_video_writer_create_file_api_params(const char *filename,
                                                            int api_preference, int fourcc,
                                                            double fps, int width, int height,
                                                            const cvk_mat_t *params) {
    return guarded([&]() -> cvk_video_writer_t * {
        return writer_open_file_params(filename, api_preference, fourcc, fps, width, height,
                                       params);
    });
}

int cvk_video_writer_open_file(cvk_video_writer_t *h, const char *filename, int fourcc,
                               double fps, int width, int height, int is_color) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0;
    }
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_writer *>(h);
    return guarded([&]() -> int {
        return p->writer.open(cv::String(filename), cv::CAP_ANY, fourcc, fps, cv::Size(width, height),
                              is_color != 0)
                   ? 1
                   : 0;
    });
}

int cvk_video_writer_open_file_api(cvk_video_writer_t *h, const char *filename,
                                   int api_preference, int fourcc, double fps, int width,
                                   int height, int is_color) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0;
    }
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_writer *>(h);
    return guarded([&]() -> int {
        return p->writer.open(cv::String(filename), api_preference, fourcc, fps, cv::Size(width, height),
                              is_color != 0)
                   ? 1
                   : 0;
    });
}

int cvk_video_writer_open_file_params(cvk_video_writer_t *h, const char *filename, int fourcc,
                                      double fps, int width, int height,
                                      const cvk_mat_t *params) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0;
    }
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_writer *>(h);
    return guarded([&]() -> int {
        return p->writer.open(cv::String(filename), cv::CAP_ANY, fourcc, fps, cv::Size(width, height),
                              vec_int_of(params))
                   ? 1
                   : 0;
    });
}

int cvk_video_writer_open_file_api_params(cvk_video_writer_t *h, const char *filename,
                                          int api_preference, int fourcc, double fps,
                                          int width, int height, const cvk_mat_t *params) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0;
    }
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_writer *>(h);
    return guarded([&]() -> int {
        return p->writer.open(cv::String(filename), api_preference, fourcc, fps, cv::Size(width, height),
                              vec_int_of(params))
                   ? 1
                   : 0;
    });
}

int cvk_video_writer_is_opened(const cvk_video_writer_t *h) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0;
    }
    auto *p = reinterpret_cast<const cvk_video_writer *>(h);
    return guarded([&]() -> int { return p->writer.isOpened() ? 1 : 0; });
}

void cvk_video_writer_release(cvk_video_writer_t *h) {
    if (h == nullptr) {
        return;
    }
    auto *p = reinterpret_cast<cvk_video_writer *>(h);
    guarded([&]() -> int {
        p->writer.release();
        return 0;
    });
}

void cvk_video_writer_delete(cvk_video_writer_t *h) {
    delete reinterpret_cast<cvk_video_writer *>(h);
}

int cvk_video_writer_write(cvk_video_writer_t *h, const cvk_mat_t *image) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0;
    }
    const cv::Mat *m = require_const(image);
    if (m == nullptr) {
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_writer *>(h);
    return guarded([&]() -> int { return p->writer.write(*m) ? 1 : 0; });
}

int cvk_video_writer_set(cvk_video_writer_t *h, int prop_id, double value) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0;
    }
    auto *p = reinterpret_cast<cvk_video_writer *>(h);
    return guarded([&]() -> int { return p->writer.set(prop_id, value) ? 1 : 0; });
}

double cvk_video_writer_get(const cvk_video_writer_t *h, int prop_id) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return 0.0;
    }
    auto *p = reinterpret_cast<const cvk_video_writer *>(h);
    return guarded([&]() -> double { return p->writer.get(prop_id); });
}

const char *cvk_video_writer_get_backend_name(const cvk_video_writer_t *h) {
    if (h == nullptr) {
        record_error("null VideoWriter handle");
        return nullptr;
    }
    auto *p = reinterpret_cast<const cvk_video_writer *>(h);
    return guarded([&]() -> const char * {
        g_videoio_str = p->writer.getBackendName();
        return g_videoio_str.c_str();
    });
}

/* ---------------------------------------------------------------------------
 * videoio_registry
 * ------------------------------------------------------------------------- */

const char *cvk_videoio_get_backend_name(int api) {
    return guarded([&]() -> const char * {
        g_videoio_str =
            cv::videoio_registry::getBackendName(static_cast<cv::VideoCaptureAPIs>(api));
        return g_videoio_str.c_str();
    });
}

cvk_mat_t *cvk_videoio_get_backends(void) {
    return guarded([&]() -> cvk_mat_t * {
        return vec_int_to_mat(cv::videoio_registry::getBackends());
    });
}

cvk_mat_t *cvk_videoio_get_camera_backends(void) {
    return guarded([&]() -> cvk_mat_t * {
        return vec_int_to_mat(cv::videoio_registry::getCameraBackends());
    });
}

cvk_mat_t *cvk_videoio_get_stream_backends(void) {
    return guarded([&]() -> cvk_mat_t * {
        return vec_int_to_mat(cv::videoio_registry::getStreamBackends());
    });
}

cvk_mat_t *cvk_videoio_get_stream_buffered_backends(void) {
    return guarded([&]() -> cvk_mat_t * {
        return vec_int_to_mat(cv::videoio_registry::getStreamBufferedBackends());
    });
}

cvk_mat_t *cvk_videoio_get_writer_backends(void) {
    return guarded([&]() -> cvk_mat_t * {
        return vec_int_to_mat(cv::videoio_registry::getWriterBackends());
    });
}

int cvk_videoio_has_backend(int api) {
    return guarded([&]() -> int {
        return cv::videoio_registry::hasBackend(static_cast<cv::VideoCaptureAPIs>(api)) ? 1 : 0;
    });
}

int cvk_videoio_is_backend_built_in(int api) {
    return guarded([&]() -> int {
        return cv::videoio_registry::isBackendBuiltIn(static_cast<cv::VideoCaptureAPIs>(api))
                   ? 1
                   : 0;
    });
}

const char *cvk_get_camera_plugin_version(int api, int *version_abi,
                                                          int *version_api) {
    return guarded([&]() -> const char * {
        int abi = -1;
        int api_version = -1;
        std::string desc = cv::videoio_registry::getCameraBackendPluginVersion(
            static_cast<cv::VideoCaptureAPIs>(api), abi, api_version);
        if (version_abi != nullptr) {
            *version_abi = abi;
        }
        if (version_api != nullptr) {
            *version_api = api_version;
        }
        g_videoio_str = std::move(desc);
        return g_videoio_str.c_str();
    });
}

const char *cvk_get_stream_plugin_version(int api, int *version_abi,
                                                          int *version_api) {
    return guarded([&]() -> const char * {
        int abi = -1;
        int api_version = -1;
        std::string desc = cv::videoio_registry::getStreamBackendPluginVersion(
            static_cast<cv::VideoCaptureAPIs>(api), abi, api_version);
        if (version_abi != nullptr) {
            *version_abi = abi;
        }
        if (version_api != nullptr) {
            *version_api = api_version;
        }
        g_videoio_str = std::move(desc);
        return g_videoio_str.c_str();
    });
}

const char *cvk_get_stream_buffered_plugin_version(int api, int *version_abi,
                                                                   int *version_api) {
    return guarded([&]() -> const char * {
        int abi = -1;
        int api_version = -1;
        std::string desc = cv::videoio_registry::getStreamBufferedBackendPluginVersion(
            static_cast<cv::VideoCaptureAPIs>(api), abi, api_version);
        if (version_abi != nullptr) {
            *version_abi = abi;
        }
        if (version_api != nullptr) {
            *version_api = api_version;
        }
        g_videoio_str = std::move(desc);
        return g_videoio_str.c_str();
    });
}

const char *cvk_get_writer_plugin_version(int api, int *version_abi,
                                                          int *version_api) {
    return guarded([&]() -> const char * {
        int abi = -1;
        int api_version = -1;
        std::string desc = cv::videoio_registry::getWriterBackendPluginVersion(
            static_cast<cv::VideoCaptureAPIs>(api), abi, api_version);
        if (version_abi != nullptr) {
            *version_abi = abi;
        }
        if (version_api != nullptr) {
            *version_api = api_version;
        }
        g_videoio_str = std::move(desc);
        return g_videoio_str.c_str();
    });
}

} /* extern "C" */

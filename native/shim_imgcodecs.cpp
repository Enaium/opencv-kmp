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
 * imgcodecs slice of the cvk_ C ABI: multi-image codecs (imreadmulti /
 * imwritemulti / imdecodemulti / imencodemulti) and the cv::Animation
 * wrapper plus its animated codec functions (imreadanimation /
 * imdecodeanimation / imwriteanimation / imencodeanimation).
 *
 * Every exported function is noexcept: cv::Exception and std::exception are
 * caught and reported through cvk_last_error(), failures return NULL/0/false.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_imgcodecs.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

/* The struct the C header forward-declared: a plain cv::Animation value. */
struct cvk_animation {
    cv::Animation value;
};

namespace {

thread_local std::string g_imgcodecs_error;

void record_error(const char *message) {
    g_imgcodecs_error = message != nullptr ? message : "unknown error";
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

cv::Animation *require_animation(cvk_animation_t *h) {
    if (h == nullptr) {
        record_error("null Animation handle");
        return nullptr;
    }
    return &reinterpret_cast<cvk_animation *>(h)->value;
}

const cv::Animation *require_animation_const(const cvk_animation_t *h) {
    if (h == nullptr) {
        record_error("null Animation handle");
        return nullptr;
    }
    return &reinterpret_cast<const cvk_animation *>(h)->value;
}

/* Copies `mats` into a fresh cvk_mat_list_t; each page becomes its own
 * `new cv::Mat` handle owned by the caller. */
cvk_mat_list_t *make_mat_list(const std::vector<cv::Mat> &mats) {
    auto *list = new cvk_mat_list;
    list->count = mats.size();
    list->items = new cvk_mat_t *[mats.size()];
    for (size_t i = 0; i < mats.size(); ++i) {
        list->items[i] = reinterpret_cast<cvk_mat_t *>(new cv::Mat(mats[i]));
    }
    return list;
}

/* Builds the params vector from a C array; tolerates null/empty. */
std::vector<int> param_vector(const int *params, size_t params_len) {
    std::vector<int> p;
    if (params != nullptr && params_len > 0) {
        p.assign(params, params + params_len);
    }
    return p;
}

} /* namespace */

extern "C" {

void cvk_mat_list_release(cvk_mat_list_t *list) {
    if (list == nullptr) return;
    delete[] list->items;
    delete list;
}

/* ------------------------------------------------------------------ *
 * multi-page / multi-image codecs
 * ------------------------------------------------------------------ */

cvk_mat_list_t *cvk_imreadmulti(const char *filename, int flags) {
    if (filename == nullptr) {
        record_error("null filename");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_list_t * {
        std::vector<cv::Mat> mats;
        if (!cv::imreadmulti(filename, mats, flags)) {
            record_error("imreadmulti failed");
            return static_cast<cvk_mat_list_t *>(nullptr);
        }
        return make_mat_list(mats);
    });
}

cvk_mat_list_t *cvk_imreadmulti_range(const char *filename, int start, int count, int flags) {
    if (filename == nullptr) {
        record_error("null filename");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_list_t * {
        std::vector<cv::Mat> mats;
        if (!cv::imreadmulti(filename, mats, start, count, flags)) {
            record_error("imreadmulti(range) failed");
            return static_cast<cvk_mat_list_t *>(nullptr);
        }
        return make_mat_list(mats);
    });
}

cvk_mat_list_t *cvk_imdecodemulti(const cvk_mat_t *buf, int flags, int start, int count) {
    const cv::Mat *m = require_const(buf);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_list_t * {
        std::vector<cv::Mat> mats;
        if (!cv::imdecodemulti(*m, flags, mats, cv::Range(start, count))) {
            record_error("imdecodemulti failed");
            return static_cast<cvk_mat_list_t *>(nullptr);
        }
        return make_mat_list(mats);
    });
}

int cvk_imwritemulti(const char *filename, const cvk_mat_t *const *mats, size_t count,
                     const int *params, size_t params_len) {
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    if (mats == nullptr || count == 0) {
        record_error("empty image list");
        return 0;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> images;
        images.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const cv::Mat *m = require_const(mats[i]);
            if (m == nullptr) {
                throw cv::Exception(cv::Error::StsBadArg, "null Mat in image list",
                                    __func__, __FILE__, __LINE__);
            }
            images.push_back(*m);
        }
        return cv::imwritemulti(filename, images, param_vector(params, params_len)) ? 1 : 0;
    });
}

unsigned char *cvk_imencodemulti(const char *ext, const cvk_mat_t *const *mats, size_t count,
                                 const int *params, size_t params_len, size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    if (ext == nullptr) {
        record_error("null extension");
        return nullptr;
    }
    if (mats == nullptr || count == 0) {
        record_error("empty image list");
        return nullptr;
    }
    return guarded([&]() -> unsigned char * {
        std::vector<cv::Mat> images;
        images.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const cv::Mat *m = require_const(mats[i]);
            if (m == nullptr) {
                throw cv::Exception(cv::Error::StsBadArg, "null Mat in image list",
                                    __func__, __FILE__, __LINE__);
            }
            images.push_back(*m);
        }
        std::vector<uchar> buf;
        if (!cv::imencodemulti(ext, images, buf, param_vector(params, params_len))) {
            record_error("imencodemulti failed");
            return static_cast<unsigned char *>(nullptr);
        }
        auto *copy = static_cast<unsigned char *>(std::malloc(buf.size()));
        if (copy == nullptr) throw std::bad_alloc();
        std::memcpy(copy, buf.data(), buf.size());
        if (out_len != nullptr) *out_len = buf.size();
        return copy;
    });
}

/* ------------------------------------------------------------------ *
 * Animation wrapper
 * ------------------------------------------------------------------ */

cvk_animation_t *cvk_animation_create(int loop_count, double bg0, double bg1, double bg2, double bg3) {
    return guarded([&]() -> cvk_animation_t * {
        auto *h = new cvk_animation;
        h->value = cv::Animation(loop_count, cv::Scalar(bg0, bg1, bg2, bg3));
        return reinterpret_cast<cvk_animation_t *>(h);
    });
}

void cvk_animation_release(cvk_animation_t *h) {
    delete reinterpret_cast<cvk_animation *>(h);
}

int cvk_animation_get_loop_count(const cvk_animation_t *h) {
    const cv::Animation *a = require_animation_const(h);
    if (a == nullptr) return 0;
    return guarded([&] { return a->loop_count; });
}

void cvk_animation_set_loop_count(cvk_animation_t *h, int loop_count) {
    cv::Animation *a = require_animation(h);
    if (a == nullptr) return;
    guarded([&]() -> int {
        a->loop_count = loop_count;
        return 0;
    });
}

cvk_scalar_t cvk_animation_get_bgcolor(const cvk_animation_t *h) {
    const cv::Animation *a = require_animation_const(h);
    if (a == nullptr) return {0.0, 0.0, 0.0, 0.0};
    return guarded([&]() -> cvk_scalar_t {
        const cv::Scalar &s = a->bgcolor;
        cvk_scalar_t out;
        out.v0 = s[0];
        out.v1 = s[1];
        out.v2 = s[2];
        out.v3 = s[3];
        return out;
    });
}

void cvk_animation_set_bgcolor(cvk_animation_t *h, cvk_scalar_t bg) {
    cv::Animation *a = require_animation(h);
    if (a == nullptr) return;
    guarded([&]() -> int {
        a->bgcolor = cv::Scalar(bg.v0, bg.v1, bg.v2, bg.v3);
        return 0;
    });
}

cvk_mat_t *cvk_animation_get_durations(const cvk_animation_t *h) {
    const cv::Animation *a = require_animation_const(h);
    if (a == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        const std::vector<int> &d = a->durations;
        auto *m = new cv::Mat(static_cast<int>(d.size()), 1, CV_32SC1);
        for (size_t i = 0; i < d.size(); ++i) {
            m->at<int>(static_cast<int>(i)) = d[i];
        }
        return reinterpret_cast<cvk_mat_t *>(m);
    });
}

int cvk_animation_set_durations(cvk_animation_t *h, const cvk_mat_t *durations) {
    cv::Animation *a = require_animation(h);
    if (a == nullptr) return 0;
    const cv::Mat *m = require_const(durations);
    if (m == nullptr) return 0;
    return guarded([&]() -> int {
        std::vector<int> values;
        values.reserve(static_cast<size_t>(m->total()));
        if (m->rows == 1) {
            for (int c = 0; c < m->cols; ++c) values.push_back(m->at<int>(0, c));
        } else {
            for (int r = 0; r < m->rows; ++r) values.push_back(m->at<int>(r, 0));
        }
        a->durations = std::move(values);
        return 1;
    });
}

cvk_mat_list_t *cvk_animation_get_frames(const cvk_animation_t *h) {
    const cv::Animation *a = require_animation_const(h);
    if (a == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_list_t * { return make_mat_list(a->frames); });
}

int cvk_animation_set_frames(cvk_animation_t *h, const cvk_mat_t *const *mats, size_t count) {
    cv::Animation *a = require_animation(h);
    if (a == nullptr) return 0;
    if (mats == nullptr || count == 0) {
        guarded([&]() -> int {
            a->frames.clear();
            return 0;
        });
        return 1;
    }
    return guarded([&]() -> int {
        std::vector<cv::Mat> frames;
        frames.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const cv::Mat *m = require_const(mats[i]);
            if (m == nullptr) {
                throw cv::Exception(cv::Error::StsBadArg, "null Mat in frame list",
                                    __func__, __FILE__, __LINE__);
            }
            frames.push_back(*m);
        }
        a->frames = std::move(frames);
        return 1;
    });
}

cvk_mat_t *cvk_animation_get_still_image(const cvk_animation_t *h) {
    const cv::Animation *a = require_animation_const(h);
    if (a == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        if (a->still_image.empty()) {
            return static_cast<cvk_mat_t *>(nullptr);
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(a->still_image));
    });
}

int cvk_animation_set_still_image(cvk_animation_t *h, const cvk_mat_t *img) {
    cv::Animation *a = require_animation(h);
    if (a == nullptr) return 0;
    const cv::Mat *m = require_const(img);
    if (m == nullptr) return 0;
    return guarded([&]() -> int {
        a->still_image = *m;
        return 1;
    });
}

/* ------------------------------------------------------------------ *
 * animation codecs
 * ------------------------------------------------------------------ */

int cvk_imreadanimation(const char *filename, cvk_animation_t *h, int start, int count) {
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    cv::Animation *a = require_animation(h);
    if (a == nullptr) return 0;
    return guarded([&] { return cv::imreadanimation(filename, *a, start, count) ? 1 : 0; });
}

int cvk_imdecodeanimation(const cvk_mat_t *buf, cvk_animation_t *h, int start, int count) {
    const cv::Mat *m = require_const(buf);
    if (m == nullptr) return 0;
    cv::Animation *a = require_animation(h);
    if (a == nullptr) return 0;
    return guarded([&] { return cv::imdecodeanimation(*m, *a, start, count) ? 1 : 0; });
}

int cvk_imwriteanimation(const char *filename, const cvk_animation_t *h,
                         const int *params, size_t params_len) {
    if (filename == nullptr) {
        record_error("null filename");
        return 0;
    }
    const cv::Animation *a = require_animation_const(h);
    if (a == nullptr) return 0;
    return guarded([&] {
        return cv::imwriteanimation(filename, *a, param_vector(params, params_len)) ? 1 : 0;
    });
}

unsigned char *cvk_imencodeanimation(const char *ext, const cvk_animation_t *h,
                                     const int *params, size_t params_len, size_t *out_len) {
    if (out_len != nullptr) *out_len = 0;
    if (ext == nullptr) {
        record_error("null extension");
        return nullptr;
    }
    const cv::Animation *a = require_animation_const(h);
    if (a == nullptr) return nullptr;
    return guarded([&]() -> unsigned char * {
        std::vector<uchar> buf;
        if (!cv::imencodeanimation(ext, *a, buf, param_vector(params, params_len))) {
            record_error("imencodeanimation failed");
            return static_cast<unsigned char *>(nullptr);
        }
        auto *copy = static_cast<unsigned char *>(std::malloc(buf.size()));
        if (copy == nullptr) throw std::bad_alloc();
        std::memcpy(copy, buf.data(), buf.size());
        if (out_len != nullptr) *out_len = buf.size();
        return copy;
    });
}

} /* extern "C" */

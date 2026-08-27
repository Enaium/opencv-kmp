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
 * cvk_ implementations for the "features2" slice: descriptor matchers
 * (DescriptorMatcher/BFMatcher/FlannBasedMatcher/LightGlueMatcher), the
 * Annoy-backed ANNIndex, the learned Feature2D extractors (ALIKED, DISK)
 * and the Features static utilities. Every exported function is noexcept
 * and reports failures through cvk_last_error().
 *
 * Wire formats (see opencv_kmp_features2.h):
 *   - DMatch rows: Nx1 CV_32FC4 (queryIdx, trainIdx, imgIdx, distance).
 *   - List<Mat>: Nx1 CV_32SC2, each row packing a 64-bit cv::Mat address.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_features2.h"

#include <opencv2/features.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct cvk_descriptor_matcher { cv::Ptr<cv::DescriptorMatcher> ptr; };
struct cvk_ann_index { cv::Ptr<cv::ANNIndex> ptr; };
struct cvk_aliked { cv::Ptr<cv::ALIKED> ptr; };
struct cvk_disk { cv::Ptr<cv::DISK> ptr; };

namespace {

thread_local std::string g_f2_str;
thread_local std::string g_f2_error;

void record_error(const char *message) {
    g_f2_error = message != nullptr ? message : "unknown error";
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

cv::Scalar cv_scalar(cvk_scalar_t s) { return {s.v0, s.v1, s.v2, s.v3}; }

/* ---- wire-format helpers --------------------------------------------- */

/** Decodes an Nx1 CV_32SC2 handle wire Mat into a vector of Mat copies. */
bool wire_to_mats(const cv::Mat &wire, std::vector<cv::Mat> &out) {
    if (wire.empty()) return true;
    if (wire.type() != CV_32SC2 || wire.cols != 1) {
        record_error("expected a CV_32SC2 single-column Mat of Mat handles");
        return false;
    }
    const int count = wire.rows;
    const int *data = wire.ptr<int>();
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const uint64_t lo = static_cast<uint32_t>(data[i * 2]);
        const uint64_t hi = static_cast<uint32_t>(data[i * 2 + 1]);
        const uint64_t addr = (hi << 32) | lo;
        out.emplace_back(*reinterpret_cast<cv::Mat *>(static_cast<uintptr_t>(addr)));
    }
    return true;
}

/** Encodes freshly allocated Mat handles into an Nx1 CV_32SC2 wire Mat. */
cvk_mat_t *mats_to_wire(std::vector<cvk_mat_t *> &&mats) {
    auto *out = new cv::Mat(static_cast<int>(mats.size()), 1, CV_32SC2);
    int *data = out->ptr<int>();
    for (size_t i = 0; i < mats.size(); ++i) {
        const uint64_t addr = reinterpret_cast<uintptr_t>(mats[i]);
        data[i * 2] = static_cast<int>(addr >> 32);
        data[i * 2 + 1] = static_cast<int>(addr & 0xFFFFFFFFu);
    }
    return reinterpret_cast<cvk_mat_t *>(out);
}

/** Fresh Nx1 CV_32FC4 Mat of DMatch rows. */
cvk_mat_t *dmatches_to_mat(const std::vector<cv::DMatch> &matches) {
    auto *out = new cv::Mat(static_cast<int>(matches.size()), 1, CV_32FC4);
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Vec4f &v = out->at<cv::Vec4f>(static_cast<int>(i), 0);
        v[0] = static_cast<float>(matches[i].queryIdx);
        v[1] = static_cast<float>(matches[i].trainIdx);
        v[2] = static_cast<float>(matches[i].imgIdx);
        v[3] = matches[i].distance;
    }
    return reinterpret_cast<cvk_mat_t *>(out);
}

/** Decodes an Nx1 CV_32FC4 Mat into DMatch rows. */
bool mat_to_dmatches(const cv::Mat &m, std::vector<cv::DMatch> &out) {
    if (m.empty()) return true;
    if (m.type() != CV_32FC4 || m.cols != 1) {
        record_error("expected a CV_32FC4 single-column Mat of DMatch rows");
        return false;
    }
    const int count = m.rows;
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const cv::Vec4f &v = m.at<cv::Vec4f>(i, 0);
        out.emplace_back(static_cast<int>(v[0]), static_cast<int>(v[1]),
                         static_cast<int>(v[2]), v[3]);
    }
    return true;
}

/** Fresh wire Mat of fresh Nx1 CV_32FC4 Mats (one per DMatch group). */
cvk_mat_t *dmatches_wire(const std::vector<std::vector<cv::DMatch>> &groups) {
    std::vector<cvk_mat_t *> mats;
    mats.reserve(groups.size());
    for (const auto &g : groups) mats.push_back(dmatches_to_mat(g));
    return mats_to_wire(std::move(mats));
}


/** Fresh Nx1 CV_32FC7 Mat of keypoints. */
cvk_mat_t *keypoints_to_mat(const std::vector<cv::KeyPoint> &kps) {
    auto *out = new cv::Mat(static_cast<int>(kps.size()), 1, CV_32FC(7));
    for (size_t i = 0; i < kps.size(); ++i) {
        cv::Vec<float, 7> &v = out->at<cv::Vec<float, 7>>(static_cast<int>(i), 0);
        v[0] = kps[i].pt.x;
        v[1] = kps[i].pt.y;
        v[2] = kps[i].size;
        v[3] = kps[i].angle;
        v[4] = kps[i].response;
        v[5] = static_cast<float>(kps[i].octave);
        v[6] = static_cast<float>(kps[i].class_id);
    }
    return reinterpret_cast<cvk_mat_t *>(out);
}

/** Decodes an Nx1 CV_32FC7 Mat into keypoints. */
bool mat_to_keypoints(const cv::Mat &m, std::vector<cv::KeyPoint> &out) {
    if (m.empty()) return true;
    if (m.type() != CV_32FC(7) || m.cols != 1) {
        record_error("expected a CV_32FC7 single-column Mat of keypoints");
        return false;
    }
    const int count = m.rows;
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const cv::Vec<float, 7> &v = m.at<cv::Vec<float, 7>>(i, 0);
        out.emplace_back(v[0], v[1], v[2], v[3], v[4],
                         static_cast<int>(v[5]), static_cast<int>(v[6]));
    }
    return true;
}

/** Decodes an Nx1 CV_8SC1 Mat into chars (drawMatches mask). */
bool mat_to_chars(const cv::Mat &m, std::vector<char> &out) {
    if (m.empty()) return true;
    if (m.type() != CV_8SC1 || m.cols != 1) {
        record_error("expected a CV_8SC1 single-column Mat of mask bytes");
        return false;
    }
    const char *data = m.ptr<char>();
    out.assign(data, data + m.rows);
    return true;
}

/* Algorithm clear/empty/save/getDefaultName for Ptr<cv::T> handle structs. */
#define CVK_ALG_FUNCS(T)                                                      \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                    \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        if (!p) { record_error("null " #T " handle"); return; }               \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                 \
    }                                                                         \
    int cvk_##T##_empty(cvk_##T##_t *h) {                                     \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        if (!p) { record_error("null " #T " handle"); return 0; }             \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });     \
    }                                                                         \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {               \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        if (!p) { record_error("null " #T " handle"); return; }               \
        guarded([&]() -> int { p->ptr->save(filename); return 0; });          \
    }                                                                         \
    const char *cvk_##T##_get_default_name(cvk_##T##_t *h) {                  \
        auto *p = reinterpret_cast<cvk_##T *>(h);                             \
        if (!p) { record_error("null " #T " handle"); return nullptr; }       \
        return guarded([&]() -> const char * {                                \
            g_f2_str = p->ptr->getDefaultName();                              \
            return g_f2_str.c_str();                                          \
        });                                                                   \
    }

} /* namespace */

extern "C" {

/* =========================================================================
 * DescriptorMatcher factories
 * ========================================================================= */

cvk_descriptor_matcher_t *cvk_descriptor_matcher_create(const char *type) {
    return guarded([&]() -> cvk_descriptor_matcher_t * {
        auto *h = new cvk_descriptor_matcher;
        h->ptr = cv::DescriptorMatcher::create(type != nullptr ? type : "");
        return reinterpret_cast<cvk_descriptor_matcher_t *>(h);
    });
}

void cvk_descriptor_matcher_release(cvk_descriptor_matcher_t *h) {
    delete reinterpret_cast<cvk_descriptor_matcher *>(h);
}

cvk_descriptor_matcher_t *cvk_descriptor_matcher_create_type(int matcher_type) {
    return guarded([&]() -> cvk_descriptor_matcher_t * {
        auto *h = new cvk_descriptor_matcher;
        h->ptr = cv::DescriptorMatcher::create(
            static_cast<cv::DescriptorMatcher::MatcherType>(matcher_type));
        return reinterpret_cast<cvk_descriptor_matcher_t *>(h);
    });
}

cvk_bf_matcher_t *cvk_bf_matcher_create(int norm_type, int cross_check) {
    return guarded([&]() -> cvk_bf_matcher_t * {
        auto *h = new cvk_descriptor_matcher;
        h->ptr = cv::BFMatcher::create(norm_type, cross_check != 0);
        return reinterpret_cast<cvk_bf_matcher_t *>(h);
    });
}

cvk_flann_matcher_t *cvk_flann_matcher_create(const char *index_params) {
    return guarded([&]() -> cvk_flann_matcher_t * {
        std::string p = index_params != nullptr ? index_params : "";
        std::string lc;
        lc.reserve(p.size());
        for (char c : p) {
            lc.push_back(static_cast<char>(::tolower(static_cast<unsigned char>(c))));
        }
        cv::Ptr<cv::flann::IndexParams> index;
        if (lc == "linear") {
            index = cv::makePtr<cv::flann::LinearIndexParams>();
        } else if (lc == "kmeans") {
            index = cv::makePtr<cv::flann::KMeansIndexParams>();
        } else if (lc == "composite") {
            index = cv::makePtr<cv::flann::CompositeIndexParams>();
        } else if (lc == "autotuned") {
            index = cv::makePtr<cv::flann::AutotunedIndexParams>();
        } else if (lc == "hierarchical") {
            index = cv::makePtr<cv::flann::HierarchicalClusteringIndexParams>();
        } else if (lc == "lsh") {
            index = cv::makePtr<cv::flann::LshIndexParams>(20, 10, 2);
        } else { /* "" | "kdtree" | unknown -> KD-tree */
            index = cv::makePtr<cv::flann::KDTreeIndexParams>();
        }
        auto *h = new cvk_descriptor_matcher;
        h->ptr = cv::makePtr<cv::FlannBasedMatcher>(index);
        return reinterpret_cast<cvk_flann_matcher_t *>(h);
    });
}

cvk_lightglue_matcher_t *cvk_lightglue_matcher_create(const char *model_path,
                                                      float score_threshold,
                                                      int backend, int target) {
    return guarded([&]() -> cvk_lightglue_matcher_t * {
        auto *h = new cvk_descriptor_matcher;
        h->ptr = cv::LightGlueMatcher::create(model_path, score_threshold, backend, target);
        return reinterpret_cast<cvk_lightglue_matcher_t *>(h);
    });
}

cvk_lightglue_matcher_t *cvk_lightglue_matcher_create_from_memory(
    const unsigned char *model_data, size_t model_len, float score_threshold,
    int backend, int target) {
    return guarded([&]() -> cvk_lightglue_matcher_t * {
        std::vector<unsigned char> data;
        if (model_data != nullptr && model_len > 0) {
            data.assign(model_data, model_data + model_len);
        }
        auto *h = new cvk_descriptor_matcher;
        h->ptr = cv::LightGlueMatcher::create(data, score_threshold, backend, target);
        return reinterpret_cast<cvk_lightglue_matcher_t *>(h);
    });
}

cvk_descriptor_matcher_t *cvk_descriptor_matcher_clone(
    const cvk_descriptor_matcher_t *h, int empty_train_data) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_descriptor_matcher_t * {
        auto *out = new cvk_descriptor_matcher;
        out->ptr = p->ptr->clone(empty_train_data != 0);
        return reinterpret_cast<cvk_descriptor_matcher_t *>(out);
    });
}

/* =========================================================================
 * DescriptorMatcher train collection / Algorithm surface
 * ========================================================================= */

void cvk_descriptor_matcher_add(cvk_descriptor_matcher_t *h,
                                const cvk_mat_t *descriptors_wire) {
    auto *p = reinterpret_cast<cvk_descriptor_matcher *>(h);
    guarded([&]() -> int {
        const cv::Mat *wire = require_const(descriptors_wire);
        if (wire == nullptr) return 0;
        std::vector<cv::Mat> descriptors;
        if (!wire_to_mats(*wire, descriptors)) return 0;
        p->ptr->add(descriptors);
        return 0;
    });
}

cvk_mat_t *cvk_descriptor_matcher_get_train_descriptors(
    const cvk_descriptor_matcher_t *h) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const std::vector<cv::Mat> &train = p->ptr->getTrainDescriptors();
        std::vector<cvk_mat_t *> clones;
        clones.reserve(train.size());
        for (const auto &m : train) {
            clones.push_back(reinterpret_cast<cvk_mat_t *>(new cv::Mat(m.clone())));
        }
        return mats_to_wire(std::move(clones));
    });
}

int cvk_descriptor_matcher_is_mask_supported(const cvk_descriptor_matcher_t *h) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> int { return p->ptr->isMaskSupported() ? 1 : 0; });
}

void cvk_descriptor_matcher_train(cvk_descriptor_matcher_t *h) {
    auto *p = reinterpret_cast<cvk_descriptor_matcher *>(h);
    guarded([&]() -> int { p->ptr->train(); return 0; });
}

void cvk_descriptor_matcher_write(cvk_descriptor_matcher_t *h,
                                  const char *filename) {
    auto *p = reinterpret_cast<cvk_descriptor_matcher *>(h);
    guarded([&]() -> int { p->ptr->write(filename); return 0; });
}

void cvk_descriptor_matcher_read(cvk_descriptor_matcher_t *h,
                                 const char *filename) {
    auto *p = reinterpret_cast<cvk_descriptor_matcher *>(h);
    guarded([&]() -> int { p->ptr->read(filename); return 0; });
}

/* =========================================================================
 * DescriptorMatcher matching
 * ========================================================================= */

cvk_mat_t *cvk_descriptor_matcher_match_train(const cvk_descriptor_matcher_t *h,
                                              const cvk_mat_t *query,
                                              const cvk_mat_t *train,
                                              const cvk_mat_t *mask) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *q = require_const(query);
        if (q == nullptr) return nullptr;
        const cv::Mat *t = require_const(train);
        if (t == nullptr) return nullptr;
        std::vector<cv::DMatch> matches;
        if (mask == nullptr) {
            p->ptr->match(*q, *t, matches);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return nullptr;
            p->ptr->match(*q, *t, matches, *m);
        }
        return dmatches_to_mat(matches);
    });
}

cvk_mat_t *cvk_descriptor_matcher_match(const cvk_descriptor_matcher_t *h,
                                        const cvk_mat_t *query,
                                        const cvk_mat_t *masks_wire) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *q = require_const(query);
        if (q == nullptr) return nullptr;
        std::vector<cv::DMatch> matches;
        if (masks_wire == nullptr) {
            p->ptr->match(*q, matches);
        } else {
            const cv::Mat *mw = require_const(masks_wire);
            if (mw == nullptr) return nullptr;
            std::vector<cv::Mat> masks;
            if (!wire_to_mats(*mw, masks)) return nullptr;
            p->ptr->match(*q, matches, masks);
        }
        return dmatches_to_mat(matches);
    });
}

cvk_mat_t *cvk_descriptor_matcher_knn_match_train(
    const cvk_descriptor_matcher_t *h, const cvk_mat_t *query,
    const cvk_mat_t *train, int k, const cvk_mat_t *mask, int compact_result) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *q = require_const(query);
        if (q == nullptr) return nullptr;
        const cv::Mat *t = require_const(train);
        if (t == nullptr) return nullptr;
        std::vector<std::vector<cv::DMatch>> matches;
        if (mask == nullptr) {
            p->ptr->knnMatch(*q, *t, matches, k, cv::noArray(), compact_result != 0);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return nullptr;
            p->ptr->knnMatch(*q, *t, matches, k, *m, compact_result != 0);
        }
        return dmatches_wire(matches);
    });
}

cvk_mat_t *cvk_descriptor_matcher_knn_match(const cvk_descriptor_matcher_t *h,
                                            const cvk_mat_t *query, int k,
                                            const cvk_mat_t *masks_wire,
                                            int compact_result) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *q = require_const(query);
        if (q == nullptr) return nullptr;
        std::vector<std::vector<cv::DMatch>> matches;
        if (masks_wire == nullptr) {
            p->ptr->knnMatch(*q, matches, k, cv::noArray(), compact_result != 0);
        } else {
            const cv::Mat *mw = require_const(masks_wire);
            if (mw == nullptr) return nullptr;
            std::vector<cv::Mat> masks;
            if (!wire_to_mats(*mw, masks)) return nullptr;
            p->ptr->knnMatch(*q, matches, k, masks, compact_result != 0);
        }
        return dmatches_wire(matches);
    });
}

cvk_mat_t *cvk_descriptor_matcher_radius_match_train(
    const cvk_descriptor_matcher_t *h, const cvk_mat_t *query,
    const cvk_mat_t *train, float max_distance, const cvk_mat_t *mask,
    int compact_result) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *q = require_const(query);
        if (q == nullptr) return nullptr;
        const cv::Mat *t = require_const(train);
        if (t == nullptr) return nullptr;
        std::vector<std::vector<cv::DMatch>> matches;
        if (mask == nullptr) {
            p->ptr->radiusMatch(*q, *t, matches, max_distance, cv::noArray(),
                                compact_result != 0);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return nullptr;
            p->ptr->radiusMatch(*q, *t, matches, max_distance, *m,
                                compact_result != 0);
        }
        return dmatches_wire(matches);
    });
}

cvk_mat_t *cvk_descriptor_matcher_radius_match(const cvk_descriptor_matcher_t *h,
                                               const cvk_mat_t *query,
                                               float max_distance,
                                               const cvk_mat_t *masks_wire,
                                               int compact_result) {
    auto *p = reinterpret_cast<const cvk_descriptor_matcher *>(h);
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *q = require_const(query);
        if (q == nullptr) return nullptr;
        std::vector<std::vector<cv::DMatch>> matches;
        if (masks_wire == nullptr) {
            p->ptr->radiusMatch(*q, matches, max_distance, cv::noArray(),
                                compact_result != 0);
        } else {
            const cv::Mat *mw = require_const(masks_wire);
            if (mw == nullptr) return nullptr;
            std::vector<cv::Mat> masks;
            if (!wire_to_mats(*mw, masks)) return nullptr;
            p->ptr->radiusMatch(*q, matches, max_distance, masks,
                                compact_result != 0);
        }
        return dmatches_wire(matches);
    });
}

/* =========================================================================
 * LightGlueMatcher extras
 * ========================================================================= */

void cvk_lightglue_matcher_set_pair_info(cvk_lightglue_matcher_t *h,
                                         const cvk_mat_t *query_kpts,
                                         const cvk_mat_t *train_kpts,
                                         double query_width, double query_height,
                                         double train_width, double train_height) {
    auto *p = reinterpret_cast<cvk_descriptor_matcher *>(h);
    guarded([&]() -> int {
        const cv::Mat *q = require_const(query_kpts);
        if (q == nullptr) return 0;
        const cv::Mat *t = require_const(train_kpts);
        if (t == nullptr) return 0;
        cv::Ptr<cv::LightGlueMatcher> lg = p->ptr.dynamicCast<cv::LightGlueMatcher>();
        if (lg.empty()) {
            record_error("matcher is not a LightGlueMatcher");
            return 0;
        }
        lg->setPairInfo(*q, *t, cv::Size(static_cast<int>(query_width),
                                         static_cast<int>(query_height)),
                        cv::Size(static_cast<int>(train_width),
                                 static_cast<int>(train_height)));
        return 0;
    });
}

void cvk_lightglue_matcher_clear_pair_info(cvk_lightglue_matcher_t *h) {
    auto *p = reinterpret_cast<cvk_descriptor_matcher *>(h);
    guarded([&]() -> int {
        cv::Ptr<cv::LightGlueMatcher> lg = p->ptr.dynamicCast<cv::LightGlueMatcher>();
        if (lg.empty()) {
            record_error("matcher is not a LightGlueMatcher");
            return 0;
        }
        lg->clearPairInfo();
        return 0;
    });
}

/* =========================================================================
 * ANNIndex (Annoy)
 * ========================================================================= */

cvk_ann_index_t *cvk_ann_index_create(int dim, int dist_type) {
    return guarded([&]() -> cvk_ann_index_t * {
        auto *h = new cvk_ann_index;
        h->ptr = cv::ANNIndex::create(dim,
                                      static_cast<cv::ANNIndex::Distance>(dist_type));
        return reinterpret_cast<cvk_ann_index_t *>(h);
    });
}

void cvk_ann_index_add_items(cvk_ann_index_t *h, const cvk_mat_t *features) {
    auto *p = reinterpret_cast<cvk_ann_index *>(h);
    guarded([&]() -> int {
        const cv::Mat *f = require_const(features);
        if (f == nullptr) return 0;
        p->ptr->addItems(*f);
        return 0;
    });
}

void cvk_ann_index_build(cvk_ann_index_t *h, int trees) {
    auto *p = reinterpret_cast<cvk_ann_index *>(h);
    guarded([&]() -> int { p->ptr->build(trees); return 0; });
}

void cvk_ann_index_knn_search(const cvk_ann_index_t *h, const cvk_mat_t *query,
                              int knn, int search_k, cvk_mat_t **indices,
                              cvk_mat_t **dists) {
    auto *p = reinterpret_cast<const cvk_ann_index *>(h);
    guarded([&]() -> int {
        const cv::Mat *q = require_const(query);
        if (q == nullptr) return 0;
        cv::Mat idx, dst;
        p->ptr->knnSearch(*q, idx, dst, knn, search_k);
        *indices = reinterpret_cast<cvk_mat_t *>(new cv::Mat(idx));
        *dists = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dst));
        return 0;
    });
}

void cvk_ann_index_save(const cvk_ann_index_t *h, const char *filename,
                        int prefault) {
    auto *p = reinterpret_cast<const cvk_ann_index *>(h);
    guarded([&]() -> int { p->ptr->save(filename, prefault != 0); return 0; });
}

void cvk_ann_index_load(const cvk_ann_index_t *h, const char *filename,
                        int prefault) {
    auto *p = reinterpret_cast<const cvk_ann_index *>(h);
    guarded([&]() -> int { p->ptr->load(filename, prefault != 0); return 0; });
}

int cvk_ann_index_tree_number(const cvk_ann_index_t *h) {
    auto *p = reinterpret_cast<const cvk_ann_index *>(h);
    return guarded([&]() -> int { return p->ptr->getTreeNumber(); });
}

int cvk_ann_index_item_number(const cvk_ann_index_t *h) {
    auto *p = reinterpret_cast<const cvk_ann_index *>(h);
    return guarded([&]() -> int { return p->ptr->getItemNumber(); });
}

int cvk_ann_index_set_on_disk_build(const cvk_ann_index_t *h,
                                    const char *filename) {
    auto *p = reinterpret_cast<const cvk_ann_index *>(h);
    return guarded([&]() -> int {
        return p->ptr->setOnDiskBuild(filename) ? 1 : 0;
    });
}

void cvk_ann_index_set_seed(const cvk_ann_index_t *h, int seed) {
    auto *p = reinterpret_cast<const cvk_ann_index *>(h);
    guarded([&]() -> int { p->ptr->setSeed(seed); return 0; });
}

void cvk_ann_index_release(cvk_ann_index_t *h) {
    delete reinterpret_cast<cvk_ann_index *>(h);
}

/* =========================================================================
 * ALIKED (Feature2D)
 * ========================================================================= */

cvk_aliked_t *cvk_aliked_create(const char *model_path, int input_width,
                                int input_height, int normalize_descriptors,
                                int engine, int backend, int target) {
    return guarded([&]() -> cvk_aliked_t * {
        cv::ALIKED::Params params;
        params.inputSize = cv::Size(input_width, input_height);
        params.normalizeDescriptors = normalize_descriptors != 0;
        params.engine = engine;
        params.backend = backend;
        params.target = target;
        auto *h = new cvk_aliked;
        h->ptr = cv::ALIKED::create(model_path, params);
        return reinterpret_cast<cvk_aliked_t *>(h);
    });
}

cvk_mat_t *cvk_aliked_detect(const cvk_aliked_t *h, const cvk_mat_t *image,
                             const cvk_mat_t *mask) {
    auto *p = reinterpret_cast<cvk_aliked *>(const_cast<cvk_aliked_t *>(h));
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return nullptr;
        std::vector<cv::KeyPoint> keypoints;
        if (mask == nullptr) {
            p->ptr->detect(*img, keypoints);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return nullptr;
            p->ptr->detect(*img, keypoints, *m);
        }
        return keypoints_to_mat(keypoints);
    });
}

void cvk_aliked_compute(const cvk_aliked_t *h, const cvk_mat_t *image,
                        const cvk_mat_t *keypoints, cvk_mat_t **out_keypoints,
                        cvk_mat_t **out_descriptors) {
    auto *p = reinterpret_cast<cvk_aliked *>(const_cast<cvk_aliked_t *>(h));
    guarded([&]() -> int {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return 0;
        const cv::Mat *kps = require_const(keypoints);
        if (kps == nullptr) return 0;
        std::vector<cv::KeyPoint> keypoints_vec;
        if (!mat_to_keypoints(*kps, keypoints_vec)) return 0;
        cv::Mat descriptors;
        p->ptr->compute(*img, keypoints_vec, descriptors);
        *out_keypoints = keypoints_to_mat(keypoints_vec);
        *out_descriptors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(descriptors));
        return 0;
    });
}

void cvk_aliked_detect_and_compute(const cvk_aliked_t *h, const cvk_mat_t *image,
                                   const cvk_mat_t *mask,
                                   int use_provided_keypoints,
                                   cvk_mat_t **out_keypoints,
                                   cvk_mat_t **out_descriptors) {
    auto *p = reinterpret_cast<cvk_aliked *>(const_cast<cvk_aliked_t *>(h));
    guarded([&]() -> int {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return 0;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        if (mask == nullptr) {
            p->ptr->detectAndCompute(*img, cv::noArray(), keypoints, descriptors,
                                     use_provided_keypoints != 0);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return 0;
            p->ptr->detectAndCompute(*img, *m, keypoints, descriptors,
                                     use_provided_keypoints != 0);
        }
        *out_keypoints = keypoints_to_mat(keypoints);
        *out_descriptors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(descriptors));
        return 0;
    });
}

int cvk_aliked_descriptor_size(const cvk_aliked_t *h) {
    auto *p = reinterpret_cast<cvk_aliked *>(const_cast<cvk_aliked_t *>(h));
    return guarded([&]() -> int { return p->ptr->descriptorSize(); });
}

int cvk_aliked_descriptor_type(const cvk_aliked_t *h) {
    auto *p = reinterpret_cast<cvk_aliked *>(const_cast<cvk_aliked_t *>(h));
    return guarded([&]() -> int { return p->ptr->descriptorType(); });
}

int cvk_aliked_default_norm(const cvk_aliked_t *h) {
    auto *p = reinterpret_cast<cvk_aliked *>(const_cast<cvk_aliked_t *>(h));
    return guarded([&]() -> int { return p->ptr->defaultNorm(); });
}

void cvk_aliked_write(const cvk_aliked_t *h, const char *filename) {
    auto *p = reinterpret_cast<cvk_aliked *>(const_cast<cvk_aliked_t *>(h));
    guarded([&]() -> int { p->ptr->write(filename); return 0; });
}

void cvk_aliked_read(cvk_aliked_t *h, const char *filename) {
    auto *p = reinterpret_cast<cvk_aliked *>(h);
    guarded([&]() -> int { p->ptr->read(filename); return 0; });
}

void cvk_aliked_release(cvk_aliked_t *h) {
    delete reinterpret_cast<cvk_aliked *>(h);
}

/* =========================================================================
 * DISK (Feature2D)
 * ========================================================================= */

cvk_disk_t *cvk_disk_create(const char *model_path, int max_keypoints,
                            float score_threshold, double image_width,
                            double image_height, int backend_id, int target_id) {
    return guarded([&]() -> cvk_disk_t * {
        auto *h = new cvk_disk;
        h->ptr = cv::DISK::create(model_path, max_keypoints, score_threshold,
                                  cv::Size(static_cast<int>(image_width),
                                           static_cast<int>(image_height)),
                                  backend_id, target_id);
        return reinterpret_cast<cvk_disk_t *>(h);
    });
}

cvk_disk_t *cvk_disk_create_from_memory(const unsigned char *model_data,
                                        size_t model_len, int max_keypoints,
                                        float score_threshold,
                                        double image_width, double image_height,
                                        int backend_id, int target_id) {
    return guarded([&]() -> cvk_disk_t * {
        std::vector<unsigned char> data;
        if (model_data != nullptr && model_len > 0) {
            data.assign(model_data, model_data + model_len);
        }
        auto *h = new cvk_disk;
        h->ptr = cv::DISK::create(data, max_keypoints, score_threshold,
                                  cv::Size(static_cast<int>(image_width),
                                           static_cast<int>(image_height)),
                                  backend_id, target_id);
        return reinterpret_cast<cvk_disk_t *>(h);
    });
}

cvk_mat_t *cvk_disk_detect(const cvk_disk_t *h, const cvk_mat_t *image,
                           const cvk_mat_t *mask) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return nullptr;
        std::vector<cv::KeyPoint> keypoints;
        if (mask == nullptr) {
            p->ptr->detect(*img, keypoints);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return nullptr;
            p->ptr->detect(*img, keypoints, *m);
        }
        return keypoints_to_mat(keypoints);
    });
}

void cvk_disk_compute(const cvk_disk_t *h, const cvk_mat_t *image,
                      const cvk_mat_t *keypoints, cvk_mat_t **out_keypoints,
                      cvk_mat_t **out_descriptors) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    guarded([&]() -> int {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return 0;
        const cv::Mat *kps = require_const(keypoints);
        if (kps == nullptr) return 0;
        std::vector<cv::KeyPoint> keypoints_vec;
        if (!mat_to_keypoints(*kps, keypoints_vec)) return 0;
        cv::Mat descriptors;
        p->ptr->compute(*img, keypoints_vec, descriptors);
        *out_keypoints = keypoints_to_mat(keypoints_vec);
        *out_descriptors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(descriptors));
        return 0;
    });
}

void cvk_disk_detect_and_compute(const cvk_disk_t *h, const cvk_mat_t *image,
                                 const cvk_mat_t *mask,
                                 int use_provided_keypoints,
                                 cvk_mat_t **out_keypoints,
                                 cvk_mat_t **out_descriptors) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    guarded([&]() -> int {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return 0;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        if (mask == nullptr) {
            p->ptr->detectAndCompute(*img, cv::noArray(), keypoints, descriptors,
                                     use_provided_keypoints != 0);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return 0;
            p->ptr->detectAndCompute(*img, *m, keypoints, descriptors,
                                     use_provided_keypoints != 0);
        }
        *out_keypoints = keypoints_to_mat(keypoints);
        *out_descriptors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(descriptors));
        return 0;
    });
}

int cvk_disk_descriptor_size(const cvk_disk_t *h) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    return guarded([&]() -> int { return p->ptr->descriptorSize(); });
}

int cvk_disk_descriptor_type(const cvk_disk_t *h) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    return guarded([&]() -> int { return p->ptr->descriptorType(); });
}

int cvk_disk_default_norm(const cvk_disk_t *h) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    return guarded([&]() -> int { return p->ptr->defaultNorm(); });
}

void cvk_disk_write(const cvk_disk_t *h, const char *filename) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    guarded([&]() -> int { p->ptr->write(filename); return 0; });
}

void cvk_disk_read(cvk_disk_t *h, const char *filename) {
    auto *p = reinterpret_cast<cvk_disk *>(h);
    guarded([&]() -> int { p->ptr->read(filename); return 0; });
}

void cvk_disk_set_max_keypoints(cvk_disk_t *h, int max_keypoints) {
    auto *p = reinterpret_cast<cvk_disk *>(h);
    guarded([&]() -> int { p->ptr->setMaxKeypoints(max_keypoints); return 0; });
}

int cvk_disk_get_max_keypoints(const cvk_disk_t *h) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    return guarded([&]() -> int { return p->ptr->getMaxKeypoints(); });
}

void cvk_disk_set_score_threshold(cvk_disk_t *h, float threshold) {
    auto *p = reinterpret_cast<cvk_disk *>(h);
    guarded([&]() -> int { p->ptr->setScoreThreshold(threshold); return 0; });
}

float cvk_disk_get_score_threshold(const cvk_disk_t *h) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    return guarded([&]() -> float { return p->ptr->getScoreThreshold(); });
}

void cvk_disk_set_image_size(cvk_disk_t *h, double width, double height) {
    auto *p = reinterpret_cast<cvk_disk *>(h);
    guarded([&]() -> int {
        p->ptr->setImageSize(cv::Size(static_cast<int>(width),
                                      static_cast<int>(height)));
        return 0;
    });
}

void cvk_disk_image_size(const cvk_disk_t *h, double out[2]) {
    auto *p = reinterpret_cast<cvk_disk *>(const_cast<cvk_disk_t *>(h));
    guarded([&]() -> int {
        const cv::Size s = p->ptr->getImageSize();
        out[0] = static_cast<double>(s.width);
        out[1] = static_cast<double>(s.height);
        return 0;
    });
}

void cvk_disk_release(cvk_disk_t *h) {
    delete reinterpret_cast<cvk_disk *>(h);
}

/* =========================================================================
 * Features statics
 * ========================================================================= */

static cvk_mat_t *points_to_mat(const std::vector<cv::Point> &points) {
    auto *out = new cv::Mat(static_cast<int>(points.size()), 1, CV_32SC2);
    int *data = out->ptr<int>();
    for (size_t i = 0; i < points.size(); ++i) {
        data[i * 2] = points[i].x;
        data[i * 2 + 1] = points[i].y;
    }
    return reinterpret_cast<cvk_mat_t *>(out);
}

cvk_mat_t *cvk_features_good_features_to_track(
    const cvk_mat_t *image, int max_corners, double quality_level,
    double min_distance, const cvk_mat_t *mask, int block_size,
    int use_harris_detector, double k) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return nullptr;
        std::vector<cv::Point> corners;
        if (mask == nullptr) {
            cv::goodFeaturesToTrack(*img, corners, max_corners, quality_level,
                                    min_distance, cv::noArray(), block_size,
                                    use_harris_detector != 0, k);
        } else {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return nullptr;
            cv::goodFeaturesToTrack(*img, corners, max_corners, quality_level,
                                    min_distance, *m, block_size,
                                    use_harris_detector != 0, k);
        }
        return points_to_mat(corners);
    });
}

cvk_mat_t *cvk_features_good_features_to_track_gradient(
    const cvk_mat_t *image, int max_corners, double quality_level,
    double min_distance, const cvk_mat_t *mask, int block_size,
    int gradient_size, int use_harris_detector, double k) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return nullptr;
        const cv::Mat *m = require_const(mask);
        if (m == nullptr) return nullptr;
        std::vector<cv::Point> corners;
        cv::goodFeaturesToTrack(*img, corners, max_corners, quality_level,
                                min_distance, *m, block_size, gradient_size,
                                use_harris_detector != 0, k);
        return points_to_mat(corners);
    });
}

void cvk_features_good_features_to_track_quality(
    const cvk_mat_t *image, int max_corners, double quality_level,
    double min_distance, const cvk_mat_t *mask, int block_size,
    int gradient_size, int use_harris_detector, double k,
    cvk_mat_t **corners, cvk_mat_t **quality) {
    guarded([&]() -> int {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return 0;
        cv::Mat mask_mat;
        if (mask != nullptr) {
            const cv::Mat *m = require_const(mask);
            if (m == nullptr) return 0;
            mask_mat = *m;
        }
        std::vector<cv::Point> corner_vec;
        cv::Mat quality_mat;
        cv::goodFeaturesToTrack(*img, corner_vec, max_corners, quality_level,
                                min_distance, mask_mat, quality_mat, block_size,
                                gradient_size, use_harris_detector != 0, k);
        *corners = points_to_mat(corner_vec);
        *quality = reinterpret_cast<cvk_mat_t *>(new cv::Mat(quality_mat));
        return 0;
    });
}

cvk_mat_t *cvk_draw_keypoints(const cvk_mat_t *image,
                              const cvk_mat_t *keypoints, cvk_scalar_t color,
                              int flags) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return nullptr;
        const cv::Mat *kps = require_const(keypoints);
        if (kps == nullptr) return nullptr;
        std::vector<cv::KeyPoint> keypoints_vec;
        if (!mat_to_keypoints(*kps, keypoints_vec)) return nullptr;
        cv::Mat out;
        cv::drawKeypoints(*img, keypoints_vec, out, cv_scalar(color),
                          static_cast<cv::DrawMatchesFlags>(flags));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

void cvk_draw_keypoints_over(const cvk_mat_t *image,
                             const cvk_mat_t *keypoints, cvk_mat_t *out_image,
                             cvk_scalar_t color, int flags) {
    guarded([&]() -> int {
        const cv::Mat *img = require_const(image);
        if (img == nullptr) return 0;
        const cv::Mat *kps = require_const(keypoints);
        if (kps == nullptr) return 0;
        cv::Mat *out = require(out_image);
        if (out == nullptr) return 0;
        std::vector<cv::KeyPoint> keypoints_vec;
        if (!mat_to_keypoints(*kps, keypoints_vec)) return 0;
        cv::drawKeypoints(*img, keypoints_vec, *out, cv_scalar(color),
                          static_cast<cv::DrawMatchesFlags>(flags));
        return 0;
    });
}

static cvk_mat_t *draw_matches_impl(const cvk_mat_t *img1,
                                    const cvk_mat_t *keypoints1,
                                    const cvk_mat_t *img2,
                                    const cvk_mat_t *keypoints2,
                                    const cvk_mat_t *matches,
                                    const cvk_mat_t *matches_mask,
                                    int matches_thickness, int has_thickness,
                                    cvk_scalar_t match_color,
                                    cvk_scalar_t single_point_color, int flags,
                                    cv::Mat *out_img) {
    const cv::Mat *im1 = require_const(img1);
    if (im1 == nullptr) return nullptr;
    const cv::Mat *im2 = require_const(img2);
    if (im2 == nullptr) return nullptr;
    const cv::Mat *m1 = require_const(keypoints1);
    if (m1 == nullptr) return nullptr;
    const cv::Mat *m2 = require_const(keypoints2);
    if (m2 == nullptr) return nullptr;
    const cv::Mat *mts = require_const(matches);
    if (mts == nullptr) return nullptr;
    std::vector<cv::KeyPoint> kps1;
    std::vector<cv::KeyPoint> kps2;
    if (!mat_to_keypoints(*m1, kps1)) return nullptr;
    if (!mat_to_keypoints(*m2, kps2)) return nullptr;
    std::vector<cv::DMatch> dm;
    if (!mat_to_dmatches(*mts, dm)) return nullptr;
    std::vector<char> mask;
    if (matches_mask != nullptr) {
        const cv::Mat *mm = require_const(matches_mask);
        if (mm == nullptr) return nullptr;
        if (!mat_to_chars(*mm, mask)) return nullptr;
    }
    if (has_thickness) {
        cv::drawMatches(*im1, kps1, *im2, kps2, dm, *out_img, matches_thickness,
                        cv_scalar(match_color), cv_scalar(single_point_color),
                        mask, static_cast<cv::DrawMatchesFlags>(flags));
    } else {
        cv::drawMatches(*im1, kps1, *im2, kps2, dm, *out_img,
                        cv_scalar(match_color), cv_scalar(single_point_color),
                        mask, static_cast<cv::DrawMatchesFlags>(flags));
    }
    return reinterpret_cast<cvk_mat_t *>(new cv::Mat(*out_img));
}

cvk_mat_t *cvk_draw_matches(const cvk_mat_t *img1, const cvk_mat_t *keypoints1,
                            const cvk_mat_t *img2, const cvk_mat_t *keypoints2,
                            const cvk_mat_t *matches, cvk_scalar_t match_color,
                            cvk_scalar_t single_point_color,
                            const cvk_mat_t *matches_mask, int flags) {
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out;
        return draw_matches_impl(img1, keypoints1, img2, keypoints2, matches,
                                 matches_mask, 0, 0, match_color,
                                 single_point_color, flags, &out);
    });
}

void cvk_draw_matches_over(const cvk_mat_t *img1, const cvk_mat_t *keypoints1,
                           const cvk_mat_t *img2, const cvk_mat_t *keypoints2,
                           const cvk_mat_t *matches, cvk_mat_t *out_img,
                           cvk_scalar_t match_color,
                           cvk_scalar_t single_point_color,
                           const cvk_mat_t *matches_mask, int flags) {
    guarded([&]() -> int {
        cv::Mat *out = require(out_img);
        if (out == nullptr) return 0;
        draw_matches_impl(img1, keypoints1, img2, keypoints2, matches,
                          matches_mask, 0, 0, match_color, single_point_color,
                          flags, out);
        return 0;
    });
}

cvk_mat_t *cvk_draw_matches_thickness(
    const cvk_mat_t *img1, const cvk_mat_t *keypoints1, const cvk_mat_t *img2,
    const cvk_mat_t *keypoints2, const cvk_mat_t *matches, int matches_thickness,
    cvk_scalar_t match_color, cvk_scalar_t single_point_color,
    const cvk_mat_t *matches_mask, int flags) {
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out;
        return draw_matches_impl(img1, keypoints1, img2, keypoints2, matches,
                                 matches_mask, matches_thickness, 1, match_color,
                                 single_point_color, flags, &out);
    });
}

cvk_mat_t *cvk_draw_matches_knn(const cvk_mat_t *img1,
                                const cvk_mat_t *keypoints1,
                                const cvk_mat_t *img2,
                                const cvk_mat_t *keypoints2,
                                const cvk_mat_t *matches_wire,
                                cvk_scalar_t match_color,
                                cvk_scalar_t single_point_color,
                                const cvk_mat_t *masks_wire, int flags) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *im1 = require_const(img1);
        if (im1 == nullptr) return nullptr;
        const cv::Mat *im2 = require_const(img2);
        if (im2 == nullptr) return nullptr;
        const cv::Mat *m1 = require_const(keypoints1);
        if (m1 == nullptr) return nullptr;
        const cv::Mat *m2 = require_const(keypoints2);
        if (m2 == nullptr) return nullptr;
        std::vector<cv::KeyPoint> kps1;
        std::vector<cv::KeyPoint> kps2;
        if (!mat_to_keypoints(*m1, kps1)) return nullptr;
        if (!mat_to_keypoints(*m2, kps2)) return nullptr;
        std::vector<std::vector<cv::DMatch>> groups;
        std::vector<std::vector<char>> masks;
        if (matches_wire != nullptr) {
            const cv::Mat *mw = require_const(matches_wire);
            if (mw == nullptr) return nullptr;
            std::vector<cv::Mat> mats;
            if (!wire_to_mats(*mw, mats)) return nullptr;
            groups.reserve(mats.size());
            for (const auto &m : mats) {
                std::vector<cv::DMatch> group;
                if (!mat_to_dmatches(m, group)) return nullptr;
                groups.push_back(std::move(group));
            }
        }
        if (masks_wire != nullptr) {
            const cv::Mat *mw = require_const(masks_wire);
            if (mw == nullptr) return nullptr;
            std::vector<cv::Mat> mats;
            if (!wire_to_mats(*mw, mats)) return nullptr;
            masks.reserve(mats.size());
            for (const auto &m : mats) {
                std::vector<char> group;
                if (!mat_to_chars(m, group)) return nullptr;
                masks.push_back(std::move(group));
            }
        }
        cv::Mat out;
        cv::drawMatches(*im1, kps1, *im2, kps2, groups, out,
                        cv_scalar(match_color), cv_scalar(single_point_color),
                        masks, static_cast<cv::DrawMatchesFlags>(flags));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

CVK_ALG_FUNCS(descriptor_matcher)
CVK_ALG_FUNCS(aliked)
CVK_ALG_FUNCS(disk)

} /* extern "C" */

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
 * cvk_ C ABI implementation for the imgproc2 slice: LineSegmentDetector and
 * the two GeneralizedHough variants (Ballard, Guil). Every exported function
 * is noexcept (guarded), failures surface through cvk_last_error() plus the
 * documented NULL / 0 return conventions.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_imgproc2.h"

#include <opencv2/imgproc.hpp>

#include <string>

/** Opaque handle structs completing the header's forward declarations. */
struct cvk_line_segment_detector { cv::Ptr<cv::LineSegmentDetector> ptr; };
struct cvk_gh_ballard { cv::Ptr<cv::GeneralizedHoughBallard> ptr; };
struct cvk_gh_guil { cv::Ptr<cv::GeneralizedHoughGuil> ptr; };

namespace {

thread_local std::string g_imgproc2_error;
thread_local std::string g_imgproc2_str;

void record_error(const char *message) {
    try {
        g_imgproc2_error = message != nullptr ? message : "unknown error";
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

/* Emits set_/get_ pairs for one integer field of a GH variant. */
#define CVK_GH_INT_FIELD(T, Camel, snake)                                      \
    void cvk_##T##_set_##snake(cvk_##T##_t *h, int v) {                        \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return;                                                            \
        }                                                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        guarded([&]() -> int { p->ptr->set##Camel(v); return 0; });            \
    }                                                                          \
    int cvk_##T##_get_##snake(const cvk_##T##_t *h) {                          \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return 0;                                                          \
        }                                                                      \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                        \
        return guarded([&]() -> int { return p->ptr->get##Camel(); });         \
    }

/* Emits set_/get_ pairs for one double field of a GH variant. */
#define CVK_GH_DBL_FIELD(T, Camel, snake)                                     \
    void cvk_##T##_set_##snake(cvk_##T##_t *h, double v) {                     \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return;                                                            \
        }                                                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        guarded([&]() -> int { p->ptr->set##Camel(v); return 0; });            \
    }                                                                          \
    double cvk_##T##_get_##snake(const cvk_##T##_t *h) {                       \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return 0.0;                                                        \
        }                                                                      \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                        \
        return guarded([&]() -> double { return p->ptr->get##Camel(); });      \
    }

/* Emits the four cv::Algorithm member wrappers (clear/empty/save/getDefaultName). */
#define CVK_ALG_FUNCS(T)                                                       \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                     \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return;                                                            \
        }                                                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                  \
    }                                                                          \
    int cvk_##T##_empty(cvk_##T##_t *h) {                                      \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return 1;                                                          \
        }                                                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });      \
    }                                                                          \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {                \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return;                                                            \
        }                                                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        guarded([&]() -> int {                                                 \
            p->ptr->save(filename != nullptr ? filename : "");                 \
            return 0;                                                          \
        });                                                                    \
    }                                                                          \
    const char *cvk_##T##_get_default_name(cvk_##T##_t *h) {                   \
        if (h == nullptr) {                                                    \
            record_error("null " #T " handle");                                \
            return nullptr;                                                    \
        }                                                                      \
        auto *p = reinterpret_cast<cvk_##T *>(h);                              \
        return guarded([&]() -> const char * {                                 \
            g_imgproc2_str = p->ptr->getDefaultName();                         \
            return g_imgproc2_str.c_str();                                     \
        });                                                                    \
    }

} // namespace

extern "C" {

/* =========================================================================
 * LineSegmentDetector
 * ========================================================================= */

cvk_line_segment_detector_t *cvk_line_segment_detector_create(
    int refine, double scale, double sigma_scale, double quant, double ang_th,
    double log_eps, double density_th, int n_bins) {
    return guarded([&]() -> cvk_line_segment_detector_t * {
        auto *h = new cvk_line_segment_detector;
        h->ptr = cv::createLineSegmentDetector(
            static_cast<cv::LineSegmentDetectorModes>(refine), scale, sigma_scale,
            quant, ang_th, log_eps, density_th, n_bins);
        return reinterpret_cast<cvk_line_segment_detector_t *>(h);
    });
}

cvk_mat_t *cvk_line_segment_detector_detect(
    const cvk_line_segment_detector_t *h, const cvk_mat_t *image,
    cvk_mat_t **width, cvk_mat_t **prec, cvk_mat_t **nfa) {
    if (h == nullptr) {
        record_error("null LineSegmentDetector handle");
        return nullptr;
    }
    const cv::Mat *img = require_const(image);
    if (img == nullptr) return nullptr;
    auto *p = reinterpret_cast<const cvk_line_segment_detector *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat lines, width_out, prec_out, nfa_out;
        p->ptr->detect(*img, lines, width_out, prec_out, nfa_out);
        if (width != nullptr) {
            *width = reinterpret_cast<cvk_mat_t *>(new cv::Mat(width_out));
        }
        if (prec != nullptr) {
            *prec = reinterpret_cast<cvk_mat_t *>(new cv::Mat(prec_out));
        }
        if (nfa != nullptr) {
            *nfa = reinterpret_cast<cvk_mat_t *>(new cv::Mat(nfa_out));
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(lines));
    });
}

void cvk_line_segment_detector_draw_segments(
    const cvk_line_segment_detector_t *h, cvk_mat_t *image, const cvk_mat_t *lines) {
    if (h == nullptr) {
        record_error("null LineSegmentDetector handle");
        return;
    }
    cv::Mat *img = require(image);
    if (img == nullptr) return;
    const cv::Mat *l = require_const(lines);
    if (l == nullptr) return;
    auto *p = reinterpret_cast<const cvk_line_segment_detector *>(h);
    guarded([&]() -> int {
        p->ptr->drawSegments(*img, *l);
        return 0;
    });
}

int cvk_line_segment_detector_compare_segments(
    const cvk_line_segment_detector_t *h, int size_width, int size_height,
    const cvk_mat_t *lines1, const cvk_mat_t *lines2, cvk_mat_t *image) {
    if (h == nullptr) {
        record_error("null LineSegmentDetector handle");
        return 0;
    }
    const cv::Mat *l1 = require_const(lines1);
    if (l1 == nullptr) return 0;
    const cv::Mat *l2 = require_const(lines2);
    if (l2 == nullptr) return 0;
    cv::Mat *img = nullptr;
    if (image != nullptr) {
        img = require(image);
        if (img == nullptr) return 0;
    }
    auto *p = reinterpret_cast<const cvk_line_segment_detector *>(h);
    return guarded([&]() -> int {
        cv::Mat empty;
        return p->ptr->compareSegments(cv::Size(size_width, size_height), *l1, *l2,
                                       image != nullptr ? *img : empty);
    });
}

void cvk_line_segment_detector_release(cvk_line_segment_detector_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_line_segment_detector *>(h);
        return nullptr;
    });
}

CVK_ALG_FUNCS(line_segment_detector)

/* =========================================================================
 * GeneralizedHoughBallard
 * ========================================================================= */

cvk_gh_ballard_t *cvk_gh_ballard_create(void) {
    return guarded([&]() -> cvk_gh_ballard_t * {
        auto *h = new cvk_gh_ballard;
        h->ptr = cv::createGeneralizedHoughBallard();
        return reinterpret_cast<cvk_gh_ballard_t *>(h);
    });
}

void cvk_gh_ballard_set_template(const cvk_gh_ballard_t *h, const cvk_mat_t *templ,
                                 double center_x, double center_y) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughBallard handle");
        return;
    }
    const cv::Mat *t = require_const(templ);
    if (t == nullptr) return;
    auto *p = reinterpret_cast<const cvk_gh_ballard *>(h);
    guarded([&]() -> int {
        p->ptr->setTemplate(*t, cv::Point(center_x, center_y));
        return 0;
    });
}

void cvk_gh_ballard_set_template_edges(const cvk_gh_ballard_t *h, const cvk_mat_t *edges,
                                       const cvk_mat_t *dx, const cvk_mat_t *dy,
                                       double center_x, double center_y) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughBallard handle");
        return;
    }
    const cv::Mat *e = require_const(edges);
    if (e == nullptr) return;
    const cv::Mat *dxm = require_const(dx);
    if (dxm == nullptr) return;
    const cv::Mat *dym = require_const(dy);
    if (dym == nullptr) return;
    auto *p = reinterpret_cast<const cvk_gh_ballard *>(h);
    guarded([&]() -> int {
        p->ptr->setTemplate(*e, *dxm, *dym, cv::Point(center_x, center_y));
        return 0;
    });
}

cvk_mat_t *cvk_gh_ballard_detect(const cvk_gh_ballard_t *h, const cvk_mat_t *image,
                                 cvk_mat_t **votes) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughBallard handle");
        return nullptr;
    }
    const cv::Mat *img = require_const(image);
    if (img == nullptr) return nullptr;
    auto *p = reinterpret_cast<const cvk_gh_ballard *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat positions, votes_out;
        p->ptr->detect(*img, positions, votes_out);
        if (votes != nullptr) {
            *votes = reinterpret_cast<cvk_mat_t *>(new cv::Mat(votes_out));
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(positions));
    });
}

cvk_mat_t *cvk_gh_ballard_detect_edges(const cvk_gh_ballard_t *h, const cvk_mat_t *edges,
                                       const cvk_mat_t *dx, const cvk_mat_t *dy,
                                       cvk_mat_t **votes) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughBallard handle");
        return nullptr;
    }
    const cv::Mat *e = require_const(edges);
    if (e == nullptr) return nullptr;
    const cv::Mat *dxm = require_const(dx);
    if (dxm == nullptr) return nullptr;
    const cv::Mat *dym = require_const(dy);
    if (dym == nullptr) return nullptr;
    auto *p = reinterpret_cast<const cvk_gh_ballard *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat positions, votes_out;
        p->ptr->detect(*e, *dxm, *dym, positions, votes_out);
        if (votes != nullptr) {
            *votes = reinterpret_cast<cvk_mat_t *>(new cv::Mat(votes_out));
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(positions));
    });
}

void cvk_gh_ballard_release(cvk_gh_ballard_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_gh_ballard *>(h);
        return nullptr;
    });
}

CVK_ALG_FUNCS(gh_ballard)

CVK_GH_INT_FIELD(gh_ballard, CannyLowThresh, canny_low_thresh)
CVK_GH_INT_FIELD(gh_ballard, CannyHighThresh, canny_high_thresh)
CVK_GH_DBL_FIELD(gh_ballard, MinDist, min_dist)
CVK_GH_DBL_FIELD(gh_ballard, Dp, dp)
CVK_GH_INT_FIELD(gh_ballard, MaxBufferSize, max_buffer_size)
CVK_GH_INT_FIELD(gh_ballard, Levels, levels)
CVK_GH_INT_FIELD(gh_ballard, VotesThreshold, votes_threshold)

/* =========================================================================
 * GeneralizedHoughGuil
 * ========================================================================= */

cvk_gh_guil_t *cvk_gh_guil_create(void) {
    return guarded([&]() -> cvk_gh_guil_t * {
        auto *h = new cvk_gh_guil;
        h->ptr = cv::createGeneralizedHoughGuil();
        return reinterpret_cast<cvk_gh_guil_t *>(h);
    });
}

void cvk_gh_guil_set_template(const cvk_gh_guil_t *h, const cvk_mat_t *templ,
                              double center_x, double center_y) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughGuil handle");
        return;
    }
    const cv::Mat *t = require_const(templ);
    if (t == nullptr) return;
    auto *p = reinterpret_cast<const cvk_gh_guil *>(h);
    guarded([&]() -> int {
        p->ptr->setTemplate(*t, cv::Point(center_x, center_y));
        return 0;
    });
}

void cvk_gh_guil_set_template_edges(const cvk_gh_guil_t *h, const cvk_mat_t *edges,
                                    const cvk_mat_t *dx, const cvk_mat_t *dy,
                                    double center_x, double center_y) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughGuil handle");
        return;
    }
    const cv::Mat *e = require_const(edges);
    if (e == nullptr) return;
    const cv::Mat *dxm = require_const(dx);
    if (dxm == nullptr) return;
    const cv::Mat *dym = require_const(dy);
    if (dym == nullptr) return;
    auto *p = reinterpret_cast<const cvk_gh_guil *>(h);
    guarded([&]() -> int {
        p->ptr->setTemplate(*e, *dxm, *dym, cv::Point(center_x, center_y));
        return 0;
    });
}

cvk_mat_t *cvk_gh_guil_detect(const cvk_gh_guil_t *h, const cvk_mat_t *image,
                              cvk_mat_t **votes) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughGuil handle");
        return nullptr;
    }
    const cv::Mat *img = require_const(image);
    if (img == nullptr) return nullptr;
    auto *p = reinterpret_cast<const cvk_gh_guil *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat positions, votes_out;
        p->ptr->detect(*img, positions, votes_out);
        if (votes != nullptr) {
            *votes = reinterpret_cast<cvk_mat_t *>(new cv::Mat(votes_out));
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(positions));
    });
}

cvk_mat_t *cvk_gh_guil_detect_edges(const cvk_gh_guil_t *h, const cvk_mat_t *edges,
                                    const cvk_mat_t *dx, const cvk_mat_t *dy,
                                    cvk_mat_t **votes) {
    if (h == nullptr) {
        record_error("null GeneralizedHoughGuil handle");
        return nullptr;
    }
    const cv::Mat *e = require_const(edges);
    if (e == nullptr) return nullptr;
    const cv::Mat *dxm = require_const(dx);
    if (dxm == nullptr) return nullptr;
    const cv::Mat *dym = require_const(dy);
    if (dym == nullptr) return nullptr;
    auto *p = reinterpret_cast<const cvk_gh_guil *>(h);
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat positions, votes_out;
        p->ptr->detect(*e, *dxm, *dym, positions, votes_out);
        if (votes != nullptr) {
            *votes = reinterpret_cast<cvk_mat_t *>(new cv::Mat(votes_out));
        }
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(positions));
    });
}

void cvk_gh_guil_release(cvk_gh_guil_t *h) {
    if (h == nullptr) return;
    guarded([&]() -> void * {
        delete reinterpret_cast<cvk_gh_guil *>(h);
        return nullptr;
    });
}

CVK_ALG_FUNCS(gh_guil)

CVK_GH_INT_FIELD(gh_guil, CannyLowThresh, canny_low_thresh)
CVK_GH_INT_FIELD(gh_guil, CannyHighThresh, canny_high_thresh)
CVK_GH_DBL_FIELD(gh_guil, MinDist, min_dist)
CVK_GH_DBL_FIELD(gh_guil, Dp, dp)
CVK_GH_INT_FIELD(gh_guil, MaxBufferSize, max_buffer_size)
CVK_GH_DBL_FIELD(gh_guil, Xi, xi)
CVK_GH_INT_FIELD(gh_guil, Levels, levels)
CVK_GH_DBL_FIELD(gh_guil, AngleEpsilon, angle_epsilon)
CVK_GH_DBL_FIELD(gh_guil, MinAngle, min_angle)
CVK_GH_DBL_FIELD(gh_guil, MaxAngle, max_angle)
CVK_GH_DBL_FIELD(gh_guil, AngleStep, angle_step)
CVK_GH_INT_FIELD(gh_guil, AngleThresh, angle_thresh)
CVK_GH_DBL_FIELD(gh_guil, MinScale, min_scale)
CVK_GH_DBL_FIELD(gh_guil, MaxScale, max_scale)
CVK_GH_DBL_FIELD(gh_guil, ScaleStep, scale_step)
CVK_GH_INT_FIELD(gh_guil, ScaleThresh, scale_thresh)
CVK_GH_INT_FIELD(gh_guil, PosThresh, pos_thresh)

} /* extern "C" */

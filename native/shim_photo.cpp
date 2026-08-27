/*
 * C shim over the OpenCV "photo" module (and its ccm / segmentation
 * submodules), implementing the cvk_ ABI declared in opencv_kmp_photo.h.
 *
 * Every exported function is noexcept: cv::Exception is caught and reported
 * through cvk_last_error(), Mat-producing calls return NULL instead, and
 * two-output filters leave the out handles NULL.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_photo.h"

#include <opencv2/photo.hpp>
#include <opencv2/photo/ccm.hpp>
#include <opencv2/photo/segmentation.hpp>

#include <string>
#include <vector>

struct cvk_tonemap { cv::Ptr<cv::Tonemap> ptr; };
struct cvk_tonemap_drago { cv::Ptr<cv::TonemapDrago> ptr; };
struct cvk_tonemap_mantiuk { cv::Ptr<cv::TonemapMantiuk> ptr; };
struct cvk_tonemap_reinhard { cv::Ptr<cv::TonemapReinhard> ptr; };
struct cvk_align_mtb { cv::Ptr<cv::AlignMTB> ptr; };
struct cvk_calibrate_debevec { cv::Ptr<cv::CalibrateDebevec> ptr; };
struct cvk_calibrate_robertson { cv::Ptr<cv::CalibrateRobertson> ptr; };
struct cvk_merge_debevec { cv::Ptr<cv::MergeDebevec> ptr; };
struct cvk_merge_mertens { cv::Ptr<cv::MergeMertens> ptr; };
struct cvk_merge_robertson { cv::Ptr<cv::MergeRobertson> ptr; };
struct cvk_color_correction_model { cv::ccm::ColorCorrectionModel *ptr; };
struct cvk_intelligent_scissors_mb { cv::segmentation::IntelligentScissorsMB *ptr; };

namespace {

thread_local std::string g_last_error;
thread_local std::string g_photo_str;

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

const cv::Mat *require_const(const cvk_mat_t *mat) {
    return require(const_cast<cvk_mat_t *>(mat));
}

/** Converts a handle list into std::vector<cv::Mat>; false on invalid input. */
bool collect_mats(const cvk_mat_t *const *srcs, int count, std::vector<cv::Mat> &out) {
    if (srcs == nullptr || count <= 0) {
        record_error("null or empty Mat list");
        return false;
    }
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const cv::Mat *m = require_const(srcs[i]);
        if (m == nullptr) return false;
        out.push_back(*m);
    }
    return true;
}

/** Reads a CV_32FC1 Mat as std::vector<float> (empty when not CV_32FC1). */
std::vector<float> floats_of(const cv::Mat &m) {
    std::vector<float> out;
    if (m.type() != CV_32FC1) return out;
    cv::Mat flat = m.reshape(1, 1);
    const float *data = flat.ptr<float>();
    out.assign(data, data + flat.total());
    return out;
}

/** Wraps a std::vector<Mat> into caller-owned handles; returns the count. */
int store_mats(const std::vector<cv::Mat> &src, cvk_mat_t **out, int max_out) {
    const size_t n = src.size();
    if (n > static_cast<size_t>(max_out)) {
        record_error("output Mat list too small");
        return 0;
    }
    int written = 0;
    for (const cv::Mat &m : src) {
        out[written++] = reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
    }
    return written;
}

}  // namespace

extern "C" {

/* =========================================================================
 * Photo statics
 * ========================================================================= */

cvk_mat_t *cvk_inpaint(const cvk_mat_t *src, const cvk_mat_t *inpaint_mask,
                       double inpaint_radius, int flags) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *mask = require_const(inpaint_mask);
    if (m == nullptr || mask == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::inpaint(*m, *mask, *dst, inpaint_radius, flags);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_fast_nl_means_denoising(const cvk_mat_t *src, float h,
                                       int template_window_size, int search_window_size) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::fastNlMeansDenoising(*m, *dst, h, template_window_size, search_window_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_fast_nl_means_denoising_h(const cvk_mat_t *src, const cvk_mat_t *h,
                                         int template_window_size, int search_window_size,
                                         int norm_type) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *hm = require_const(h);
    if (m == nullptr || hm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        const std::vector<float> hv = floats_of(*hm);
        if (hv.empty()) {
            throw cv::Exception(cv::Error::StsBadArg, "h must be a non-empty CV_32FC1 Mat",
                                __func__, __FILE__, __LINE__);
        }
        auto *dst = new cv::Mat();
        cv::fastNlMeansDenoising(*m, *dst, hv, template_window_size, search_window_size,
                                 norm_type);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_fast_nl_means_denoising_multi(const cvk_mat_t *const *src, int count,
                                             int img_to_denoise_index,
                                             int temporal_window_size, float h,
                                             int template_window_size, int search_window_size) {
    std::vector<cv::Mat> srcs;
    if (!collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::fastNlMeansDenoisingMulti(srcs, *dst, img_to_denoise_index, temporal_window_size,
                                      h, template_window_size, search_window_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_fast_nl_means_denoising_multi_h(const cvk_mat_t *const *src, int count,
                                               int img_to_denoise_index,
                                               int temporal_window_size,
                                               const cvk_mat_t *h,
                                               int template_window_size,
                                               int search_window_size, int norm_type) {
    std::vector<cv::Mat> srcs;
    const cv::Mat *hm = require_const(h);
    if (!collect_mats(src, count, srcs) || hm == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        const std::vector<float> hv = floats_of(*hm);
        if (hv.empty()) {
            throw cv::Exception(cv::Error::StsBadArg, "h must be a non-empty CV_32FC1 Mat",
                                __func__, __FILE__, __LINE__);
        }
        auto *dst = new cv::Mat();
        cv::fastNlMeansDenoisingMulti(srcs, *dst, img_to_denoise_index, temporal_window_size,
                                      hv, template_window_size, search_window_size, norm_type);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_fast_nl_means_denoising_colored(const cvk_mat_t *src, float h, float h_color,
                                               int template_window_size, int search_window_size) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::fastNlMeansDenoisingColored(*m, *dst, h, h_color, template_window_size,
                                        search_window_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_fast_nl_means_denoising_colored_multi(const cvk_mat_t *const *src, int count,
                                                     int img_to_denoise_index,
                                                     int temporal_window_size,
                                                     float h, float h_color,
                                                     int template_window_size,
                                                     int search_window_size) {
    std::vector<cv::Mat> srcs;
    if (!collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::fastNlMeansDenoisingColoredMulti(srcs, *dst, img_to_denoise_index,
                                             temporal_window_size, h, h_color,
                                             template_window_size, search_window_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_denoise_tvl1(const cvk_mat_t *const *src, int count, double lambda, int niters) {
    std::vector<cv::Mat> srcs;
    if (!collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::denoise_TVL1(srcs, *dst, lambda, niters);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_decolor(const cvk_mat_t *src, cvk_mat_t **grayscale, cvk_mat_t **color_boost) {
    if (grayscale != nullptr) *grayscale = nullptr;
    if (color_boost != nullptr) *color_boost = nullptr;
    const cv::Mat *m = require_const(src);
    if (m == nullptr || grayscale == nullptr || color_boost == nullptr) return;
    guarded([&]() -> void * {
        auto *g = new cv::Mat();
        auto *cb = new cv::Mat();
        cv::decolor(*m, *g, *cb);
        *grayscale = reinterpret_cast<cvk_mat_t *>(g);
        *color_boost = reinterpret_cast<cvk_mat_t *>(cb);
        return nullptr;
    });
}

cvk_mat_t *cvk_seamless_clone(const cvk_mat_t *src, const cvk_mat_t *dst,
                              const cvk_mat_t *mask, int p_x, int p_y, int flags) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *d = require_const(dst);
    const cv::Mat *mk = require_const(mask);
    if (m == nullptr || d == nullptr || mk == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *blend = new cv::Mat();
        cv::seamlessClone(*m, *d, *mk, cv::Point(p_x, p_y), *blend, flags);
        return reinterpret_cast<cvk_mat_t *>(blend);
    });
}

cvk_mat_t *cvk_color_change(const cvk_mat_t *src, const cvk_mat_t *mask,
                            float red_mul, float green_mul, float blue_mul) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *mk = require_const(mask);
    if (m == nullptr || mk == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::colorChange(*m, *mk, *dst, red_mul, green_mul, blue_mul);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_illumination_change(const cvk_mat_t *src, const cvk_mat_t *mask,
                                   float alpha, float beta) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *mk = require_const(mask);
    if (m == nullptr || mk == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::illuminationChange(*m, *mk, *dst, alpha, beta);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_texture_flattening(const cvk_mat_t *src, const cvk_mat_t *mask,
                                  float low_threshold, float high_threshold,
                                  int kernel_size) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *mk = require_const(mask);
    if (m == nullptr || mk == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::textureFlattening(*m, *mk, *dst, low_threshold, high_threshold, kernel_size);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_edge_preserving_filter(const cvk_mat_t *src, int flags,
                                      float sigma_s, float sigma_r) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::edgePreservingFilter(*m, *dst, flags, sigma_s, sigma_r);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_detail_enhance(const cvk_mat_t *src, float sigma_s, float sigma_r) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::detailEnhance(*m, *dst, sigma_s, sigma_r);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_pencil_sketch(const cvk_mat_t *src, cvk_mat_t **dst1, cvk_mat_t **dst2,
                       float sigma_s, float sigma_r, float shade_factor) {
    if (dst1 != nullptr) *dst1 = nullptr;
    if (dst2 != nullptr) *dst2 = nullptr;
    const cv::Mat *m = require_const(src);
    if (m == nullptr || dst1 == nullptr || dst2 == nullptr) return;
    guarded([&]() -> void * {
        auto *d1 = new cv::Mat();
        auto *d2 = new cv::Mat();
        cv::pencilSketch(*m, *d1, *d2, sigma_s, sigma_r, shade_factor);
        *dst1 = reinterpret_cast<cvk_mat_t *>(d1);
        *dst2 = reinterpret_cast<cvk_mat_t *>(d2);
        return nullptr;
    });
}

cvk_mat_t *cvk_stylization(const cvk_mat_t *src, float sigma_s, float sigma_r) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::stylization(*m, *dst, sigma_s, sigma_r);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_correct_chromatic_aberration(const cvk_mat_t *input,
                                            const cvk_mat_t *coefficients,
                                            int image_width, int image_height,
                                            int calib_degree, int bayer_pattern) {
    const cv::Mat *in = require_const(input);
    const cv::Mat *coeff = require_const(coefficients);
    if (in == nullptr || coeff == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::correctChromaticAberration(*in, *coeff, *dst, cv::Size(image_width, image_height),
                                       calib_degree, bayer_pattern);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_gamma_correction(const cvk_mat_t *src, double gamma) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        cv::ccm::gammaCorrection(*m, *dst, gamma);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * Algorithm helpers (clear/empty/save/getDefaultName) for the handle types
 * whose classes derive from cv::Algorithm.
 * ========================================================================= */

#define CVK_PHOTO_ALG_FUNCS(T)                                                          \
    void cvk_##T##_clear(cvk_##T##_t *h) {                                              \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                       \
        if (p == nullptr) { record_error("null " #T " handle"); return; }               \
        guarded([&]() -> int { p->ptr->clear(); return 0; });                           \
    }                                                                                   \
    int cvk_##T##_empty(const cvk_##T##_t *h) {                                         \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                                 \
        if (p == nullptr) { record_error("null " #T " handle"); return 1; }             \
        return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });               \
    }                                                                                   \
    void cvk_##T##_save(cvk_##T##_t *h, const char *filename) {                         \
        auto *p = reinterpret_cast<cvk_##T *>(h);                                       \
        if (p == nullptr || filename == nullptr) {                                      \
            record_error("null " #T " handle or filename");                             \
            return;                                                                     \
        }                                                                               \
        guarded([&]() -> int { p->ptr->save(filename); return 0; });                    \
    }                                                                                   \
    const char *cvk_##T##_get_default_name(const cvk_##T##_t *h) {                      \
        auto *p = reinterpret_cast<const cvk_##T *>(h);                                 \
        if (p == nullptr) { record_error("null " #T " handle"); return nullptr; }       \
        return guarded([&]() -> const char * {                                          \
            g_photo_str = p->ptr->getDefaultName();                                     \
            return g_photo_str.c_str();                                                 \
        });                                                                             \
    }

CVK_PHOTO_ALG_FUNCS(tonemap)
CVK_PHOTO_ALG_FUNCS(tonemap_drago)
CVK_PHOTO_ALG_FUNCS(tonemap_mantiuk)
CVK_PHOTO_ALG_FUNCS(tonemap_reinhard)
CVK_PHOTO_ALG_FUNCS(align_mtb)
CVK_PHOTO_ALG_FUNCS(calibrate_debevec)
CVK_PHOTO_ALG_FUNCS(calibrate_robertson)
CVK_PHOTO_ALG_FUNCS(merge_debevec)
CVK_PHOTO_ALG_FUNCS(merge_mertens)
CVK_PHOTO_ALG_FUNCS(merge_robertson)

#undef CVK_PHOTO_ALG_FUNCS

/* =========================================================================
 * Tonemap
 * ========================================================================= */

cvk_tonemap_t *cvk_tonemap_create(float gamma) {
    return guarded([&]() -> cvk_tonemap_t * {
        auto *h = new cvk_tonemap;
        h->ptr = cv::createTonemap(gamma);
        return reinterpret_cast<cvk_tonemap_t *>(h);
    });
}

void cvk_tonemap_release(cvk_tonemap_t *h) {
    delete reinterpret_cast<cvk_tonemap *>(h);
}

cvk_mat_t *cvk_tonemap_process(const cvk_tonemap_t *h, const cvk_mat_t *src) {
    const auto *p = reinterpret_cast<const cvk_tonemap *>(h);
    const cv::Mat *m = require_const(src);
    if (p == nullptr || m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

float cvk_tonemap_get_gamma(const cvk_tonemap_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap *>(h);
    if (p == nullptr) { record_error("null tonemap handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getGamma(); });
}

void cvk_tonemap_set_gamma(cvk_tonemap_t *h, float gamma) {
    auto *p = reinterpret_cast<cvk_tonemap *>(h);
    if (p == nullptr) { record_error("null tonemap handle"); return; }
    guarded([&]() -> int { p->ptr->setGamma(gamma); return 0; });
}

/* =========================================================================
 * TonemapDrago
 * ========================================================================= */

cvk_tonemap_drago_t *cvk_tonemap_drago_create(float gamma, float saturation, float bias) {
    return guarded([&]() -> cvk_tonemap_drago_t * {
        auto *h = new cvk_tonemap_drago;
        h->ptr = cv::createTonemapDrago(gamma, saturation, bias);
        return reinterpret_cast<cvk_tonemap_drago_t *>(h);
    });
}

void cvk_tonemap_drago_release(cvk_tonemap_drago_t *h) {
    delete reinterpret_cast<cvk_tonemap_drago *>(h);
}

cvk_mat_t *cvk_tonemap_drago_process(const cvk_tonemap_drago_t *h, const cvk_mat_t *src) {
    const auto *p = reinterpret_cast<const cvk_tonemap_drago *>(h);
    const cv::Mat *m = require_const(src);
    if (p == nullptr || m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

float cvk_tonemap_drago_get_gamma(const cvk_tonemap_drago_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_drago *>(h);
    if (p == nullptr) { record_error("null tonemapDrago handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getGamma(); });
}

void cvk_tonemap_drago_set_gamma(cvk_tonemap_drago_t *h, float gamma) {
    auto *p = reinterpret_cast<cvk_tonemap_drago *>(h);
    if (p == nullptr) { record_error("null tonemapDrago handle"); return; }
    guarded([&]() -> int { p->ptr->setGamma(gamma); return 0; });
}

float cvk_tonemap_drago_get_saturation(const cvk_tonemap_drago_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_drago *>(h);
    if (p == nullptr) { record_error("null tonemapDrago handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getSaturation(); });
}

void cvk_tonemap_drago_set_saturation(cvk_tonemap_drago_t *h, float saturation) {
    auto *p = reinterpret_cast<cvk_tonemap_drago *>(h);
    if (p == nullptr) { record_error("null tonemapDrago handle"); return; }
    guarded([&]() -> int { p->ptr->setSaturation(saturation); return 0; });
}

float cvk_tonemap_drago_get_bias(const cvk_tonemap_drago_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_drago *>(h);
    if (p == nullptr) { record_error("null tonemapDrago handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getBias(); });
}

void cvk_tonemap_drago_set_bias(cvk_tonemap_drago_t *h, float bias) {
    auto *p = reinterpret_cast<cvk_tonemap_drago *>(h);
    if (p == nullptr) { record_error("null tonemapDrago handle"); return; }
    guarded([&]() -> int { p->ptr->setBias(bias); return 0; });
}

/* =========================================================================
 * TonemapMantiuk
 * ========================================================================= */

cvk_tonemap_mantiuk_t *cvk_tonemap_mantiuk_create(float gamma, float scale, float saturation) {
    return guarded([&]() -> cvk_tonemap_mantiuk_t * {
        auto *h = new cvk_tonemap_mantiuk;
        h->ptr = cv::createTonemapMantiuk(gamma, scale, saturation);
        return reinterpret_cast<cvk_tonemap_mantiuk_t *>(h);
    });
}

void cvk_tonemap_mantiuk_release(cvk_tonemap_mantiuk_t *h) {
    delete reinterpret_cast<cvk_tonemap_mantiuk *>(h);
}

cvk_mat_t *cvk_tonemap_mantiuk_process(const cvk_tonemap_mantiuk_t *h, const cvk_mat_t *src) {
    const auto *p = reinterpret_cast<const cvk_tonemap_mantiuk *>(h);
    const cv::Mat *m = require_const(src);
    if (p == nullptr || m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

float cvk_tonemap_mantiuk_get_gamma(const cvk_tonemap_mantiuk_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_mantiuk *>(h);
    if (p == nullptr) { record_error("null tonemapMantiuk handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getGamma(); });
}

void cvk_tonemap_mantiuk_set_gamma(cvk_tonemap_mantiuk_t *h, float gamma) {
    auto *p = reinterpret_cast<cvk_tonemap_mantiuk *>(h);
    if (p == nullptr) { record_error("null tonemapMantiuk handle"); return; }
    guarded([&]() -> int { p->ptr->setGamma(gamma); return 0; });
}

float cvk_tonemap_mantiuk_get_scale(const cvk_tonemap_mantiuk_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_mantiuk *>(h);
    if (p == nullptr) { record_error("null tonemapMantiuk handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getScale(); });
}

void cvk_tonemap_mantiuk_set_scale(cvk_tonemap_mantiuk_t *h, float scale) {
    auto *p = reinterpret_cast<cvk_tonemap_mantiuk *>(h);
    if (p == nullptr) { record_error("null tonemapMantiuk handle"); return; }
    guarded([&]() -> int { p->ptr->setScale(scale); return 0; });
}

float cvk_tonemap_mantiuk_get_saturation(const cvk_tonemap_mantiuk_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_mantiuk *>(h);
    if (p == nullptr) { record_error("null tonemapMantiuk handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getSaturation(); });
}

void cvk_tonemap_mantiuk_set_saturation(cvk_tonemap_mantiuk_t *h, float saturation) {
    auto *p = reinterpret_cast<cvk_tonemap_mantiuk *>(h);
    if (p == nullptr) { record_error("null tonemapMantiuk handle"); return; }
    guarded([&]() -> int { p->ptr->setSaturation(saturation); return 0; });
}

/* =========================================================================
 * TonemapReinhard
 * ========================================================================= */

cvk_tonemap_reinhard_t *cvk_tonemap_reinhard_create(float gamma, float intensity,
                                                    float light_adapt, float color_adapt) {
    return guarded([&]() -> cvk_tonemap_reinhard_t * {
        auto *h = new cvk_tonemap_reinhard;
        h->ptr = cv::createTonemapReinhard(gamma, intensity, light_adapt, color_adapt);
        return reinterpret_cast<cvk_tonemap_reinhard_t *>(h);
    });
}

void cvk_tonemap_reinhard_release(cvk_tonemap_reinhard_t *h) {
    delete reinterpret_cast<cvk_tonemap_reinhard *>(h);
}

cvk_mat_t *cvk_tonemap_reinhard_process(const cvk_tonemap_reinhard_t *h, const cvk_mat_t *src) {
    const auto *p = reinterpret_cast<const cvk_tonemap_reinhard *>(h);
    const cv::Mat *m = require_const(src);
    if (p == nullptr || m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(*m, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

float cvk_tonemap_reinhard_get_gamma(const cvk_tonemap_reinhard_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getGamma(); });
}

void cvk_tonemap_reinhard_set_gamma(cvk_tonemap_reinhard_t *h, float gamma) {
    auto *p = reinterpret_cast<cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return; }
    guarded([&]() -> int { p->ptr->setGamma(gamma); return 0; });
}

float cvk_tonemap_reinhard_get_intensity(const cvk_tonemap_reinhard_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getIntensity(); });
}

void cvk_tonemap_reinhard_set_intensity(cvk_tonemap_reinhard_t *h, float intensity) {
    auto *p = reinterpret_cast<cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return; }
    guarded([&]() -> int { p->ptr->setIntensity(intensity); return 0; });
}

float cvk_tonemap_reinhard_get_light_adaptation(const cvk_tonemap_reinhard_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getLightAdaptation(); });
}

void cvk_tonemap_reinhard_set_light_adaptation(cvk_tonemap_reinhard_t *h, float light_adapt) {
    auto *p = reinterpret_cast<cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return; }
    guarded([&]() -> int { p->ptr->setLightAdaptation(light_adapt); return 0; });
}

float cvk_tonemap_reinhard_get_color_adaptation(const cvk_tonemap_reinhard_t *h) {
    const auto *p = reinterpret_cast<const cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getColorAdaptation(); });
}

void cvk_tonemap_reinhard_set_color_adaptation(cvk_tonemap_reinhard_t *h, float color_adapt) {
    auto *p = reinterpret_cast<cvk_tonemap_reinhard *>(h);
    if (p == nullptr) { record_error("null tonemapReinhard handle"); return; }
    guarded([&]() -> int { p->ptr->setColorAdaptation(color_adapt); return 0; });
}

/* =========================================================================
 * AlignMTB
 * ========================================================================= */

cvk_align_mtb_t *cvk_align_mtb_create(int max_bits, int exclude_range, int cut) {
    return guarded([&]() -> cvk_align_mtb_t * {
        auto *h = new cvk_align_mtb;
        h->ptr = cv::createAlignMTB(max_bits, exclude_range, cut != 0);
        return reinterpret_cast<cvk_align_mtb_t *>(h);
    });
}

void cvk_align_mtb_release(cvk_align_mtb_t *h) {
    delete reinterpret_cast<cvk_align_mtb *>(h);
}

int cvk_align_mtb_process(const cvk_align_mtb_t *h, const cvk_mat_t *const *src, int count,
                          cvk_mat_t **out, int max_out) {
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    if (p == nullptr || out == nullptr || max_out <= 0) {
        record_error("null alignMTB handle or output list");
        return 0;
    }
    std::vector<cv::Mat> srcs;
    if (!collect_mats(src, count, srcs)) return 0;
    return guarded([&]() -> int {
        std::vector<cv::Mat> dst(srcs.size());
        p->ptr->process(srcs, dst);
        return store_mats(dst, out, max_out);
    });
}

int cvk_align_mtb_process_times(const cvk_align_mtb_t *h, const cvk_mat_t *const *src, int count,
                                const cvk_mat_t *times, const cvk_mat_t *response,
                                cvk_mat_t **out, int max_out) {
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    const cv::Mat *t = require_const(times);
    const cv::Mat *r = require_const(response);
    if (p == nullptr || out == nullptr || max_out <= 0 || t == nullptr || r == nullptr) {
        if (p == nullptr || out == nullptr || max_out <= 0) record_error("null alignMTB handle or output list");
        return 0;
    }
    std::vector<cv::Mat> srcs;
    if (!collect_mats(src, count, srcs)) return 0;
    return guarded([&]() -> int {
        std::vector<cv::Mat> dst(srcs.size());
        p->ptr->process(srcs, dst, *t, *r);
        return store_mats(dst, out, max_out);
    });
}

void cvk_align_mtb_calculate_shift(const cvk_align_mtb_t *h, const cvk_mat_t *img0,
                                   const cvk_mat_t *img1, int *out_xy) {
    if (out_xy != nullptr) { out_xy[0] = 0; out_xy[1] = 0; }
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    const cv::Mat *m0 = require_const(img0);
    const cv::Mat *m1 = require_const(img1);
    if (p == nullptr || m0 == nullptr || m1 == nullptr || out_xy == nullptr) return;
    guarded([&]() -> void * {
        const cv::Point shift = p->ptr->calculateShift(*m0, *m1);
        out_xy[0] = shift.x;
        out_xy[1] = shift.y;
        return nullptr;
    });
}

cvk_mat_t *cvk_align_mtb_shift_mat(const cvk_align_mtb_t *h, const cvk_mat_t *src,
                                   int shift_x, int shift_y) {
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    const cv::Mat *m = require_const(src);
    if (p == nullptr || m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->shiftMat(*m, *dst, cv::Point(shift_x, shift_y));
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

void cvk_align_mtb_compute_bitmaps(const cvk_align_mtb_t *h, const cvk_mat_t *img,
                                   cvk_mat_t **tb, cvk_mat_t **eb) {
    if (tb != nullptr) *tb = nullptr;
    if (eb != nullptr) *eb = nullptr;
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    const cv::Mat *m = require_const(img);
    if (p == nullptr || m == nullptr || tb == nullptr || eb == nullptr) return;
    guarded([&]() -> void * {
        auto *t = new cv::Mat();
        auto *e = new cv::Mat();
        p->ptr->computeBitmaps(*m, *t, *e);
        *tb = reinterpret_cast<cvk_mat_t *>(t);
        *eb = reinterpret_cast<cvk_mat_t *>(e);
        return nullptr;
    });
}

int cvk_align_mtb_get_max_bits(const cvk_align_mtb_t *h) {
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    if (p == nullptr) { record_error("null alignMTB handle"); return 0; }
    return guarded([&]() -> int { return p->ptr->getMaxBits(); });
}

void cvk_align_mtb_set_max_bits(cvk_align_mtb_t *h, int max_bits) {
    auto *p = reinterpret_cast<cvk_align_mtb *>(h);
    if (p == nullptr) { record_error("null alignMTB handle"); return; }
    guarded([&]() -> int { p->ptr->setMaxBits(max_bits); return 0; });
}

int cvk_align_mtb_get_exclude_range(const cvk_align_mtb_t *h) {
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    if (p == nullptr) { record_error("null alignMTB handle"); return 0; }
    return guarded([&]() -> int { return p->ptr->getExcludeRange(); });
}

void cvk_align_mtb_set_exclude_range(cvk_align_mtb_t *h, int exclude_range) {
    auto *p = reinterpret_cast<cvk_align_mtb *>(h);
    if (p == nullptr) { record_error("null alignMTB handle"); return; }
    guarded([&]() -> int { p->ptr->setExcludeRange(exclude_range); return 0; });
}

int cvk_align_mtb_get_cut(const cvk_align_mtb_t *h) {
    const auto *p = reinterpret_cast<const cvk_align_mtb *>(h);
    if (p == nullptr) { record_error("null alignMTB handle"); return 0; }
    return guarded([&]() -> int { return p->ptr->getCut() ? 1 : 0; });
}

void cvk_align_mtb_set_cut(cvk_align_mtb_t *h, int cut) {
    auto *p = reinterpret_cast<cvk_align_mtb *>(h);
    if (p == nullptr) { record_error("null alignMTB handle"); return; }
    guarded([&]() -> int { p->ptr->setCut(cut != 0); return 0; });
}

/* =========================================================================
 * CalibrateDebevec
 * ========================================================================= */

cvk_calibrate_debevec_t *cvk_calibrate_debevec_create(int samples, float lambda, int random) {
    return guarded([&]() -> cvk_calibrate_debevec_t * {
        auto *h = new cvk_calibrate_debevec;
        h->ptr = cv::createCalibrateDebevec(samples, lambda, random != 0);
        return reinterpret_cast<cvk_calibrate_debevec_t *>(h);
    });
}

void cvk_calibrate_debevec_release(cvk_calibrate_debevec_t *h) {
    delete reinterpret_cast<cvk_calibrate_debevec *>(h);
}

cvk_mat_t *cvk_calibrate_debevec_process(const cvk_calibrate_debevec_t *h,
                                         const cvk_mat_t *const *src, int count,
                                         const cvk_mat_t *times) {
    const auto *p = reinterpret_cast<const cvk_calibrate_debevec *>(h);
    const cv::Mat *t = require_const(times);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || t == nullptr || !collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst, *t);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

float cvk_calibrate_debevec_get_lambda(const cvk_calibrate_debevec_t *h) {
    const auto *p = reinterpret_cast<const cvk_calibrate_debevec *>(h);
    if (p == nullptr) { record_error("null calibrateDebevec handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getLambda(); });
}

void cvk_calibrate_debevec_set_lambda(cvk_calibrate_debevec_t *h, float lambda) {
    auto *p = reinterpret_cast<cvk_calibrate_debevec *>(h);
    if (p == nullptr) { record_error("null calibrateDebevec handle"); return; }
    guarded([&]() -> int { p->ptr->setLambda(lambda); return 0; });
}

int cvk_calibrate_debevec_get_samples(const cvk_calibrate_debevec_t *h) {
    const auto *p = reinterpret_cast<const cvk_calibrate_debevec *>(h);
    if (p == nullptr) { record_error("null calibrateDebevec handle"); return 0; }
    return guarded([&]() -> int { return p->ptr->getSamples(); });
}

void cvk_calibrate_debevec_set_samples(cvk_calibrate_debevec_t *h, int samples) {
    auto *p = reinterpret_cast<cvk_calibrate_debevec *>(h);
    if (p == nullptr) { record_error("null calibrateDebevec handle"); return; }
    guarded([&]() -> int { p->ptr->setSamples(samples); return 0; });
}

int cvk_calibrate_debevec_get_random(const cvk_calibrate_debevec_t *h) {
    const auto *p = reinterpret_cast<const cvk_calibrate_debevec *>(h);
    if (p == nullptr) { record_error("null calibrateDebevec handle"); return 0; }
    return guarded([&]() -> int { return p->ptr->getRandom() ? 1 : 0; });
}

void cvk_calibrate_debevec_set_random(cvk_calibrate_debevec_t *h, int random) {
    auto *p = reinterpret_cast<cvk_calibrate_debevec *>(h);
    if (p == nullptr) { record_error("null calibrateDebevec handle"); return; }
    guarded([&]() -> int { p->ptr->setRandom(random != 0); return 0; });
}

/* =========================================================================
 * CalibrateRobertson
 * ========================================================================= */

cvk_calibrate_robertson_t *cvk_calibrate_robertson_create(int max_iter, float threshold) {
    return guarded([&]() -> cvk_calibrate_robertson_t * {
        auto *h = new cvk_calibrate_robertson;
        h->ptr = cv::createCalibrateRobertson(max_iter, threshold);
        return reinterpret_cast<cvk_calibrate_robertson_t *>(h);
    });
}

void cvk_calibrate_robertson_release(cvk_calibrate_robertson_t *h) {
    delete reinterpret_cast<cvk_calibrate_robertson *>(h);
}

cvk_mat_t *cvk_calibrate_robertson_process(const cvk_calibrate_robertson_t *h,
                                           const cvk_mat_t *const *src, int count,
                                           const cvk_mat_t *times) {
    const auto *p = reinterpret_cast<const cvk_calibrate_robertson *>(h);
    const cv::Mat *t = require_const(times);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || t == nullptr || !collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst, *t);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_calibrate_robertson_get_radiance(const cvk_calibrate_robertson_t *h) {
    const auto *p = reinterpret_cast<const cvk_calibrate_robertson *>(h);
    if (p == nullptr) { record_error("null calibrateRobertson handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->ptr->getRadiance()));
    });
}

int cvk_calibrate_robertson_get_max_iter(const cvk_calibrate_robertson_t *h) {
    const auto *p = reinterpret_cast<const cvk_calibrate_robertson *>(h);
    if (p == nullptr) { record_error("null calibrateRobertson handle"); return 0; }
    return guarded([&]() -> int { return p->ptr->getMaxIter(); });
}

void cvk_calibrate_robertson_set_max_iter(cvk_calibrate_robertson_t *h, int max_iter) {
    auto *p = reinterpret_cast<cvk_calibrate_robertson *>(h);
    if (p == nullptr) { record_error("null calibrateRobertson handle"); return; }
    guarded([&]() -> int { p->ptr->setMaxIter(max_iter); return 0; });
}

float cvk_calibrate_robertson_get_threshold(const cvk_calibrate_robertson_t *h) {
    const auto *p = reinterpret_cast<const cvk_calibrate_robertson *>(h);
    if (p == nullptr) { record_error("null calibrateRobertson handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getThreshold(); });
}

void cvk_calibrate_robertson_set_threshold(cvk_calibrate_robertson_t *h, float threshold) {
    auto *p = reinterpret_cast<cvk_calibrate_robertson *>(h);
    if (p == nullptr) { record_error("null calibrateRobertson handle"); return; }
    guarded([&]() -> int { p->ptr->setThreshold(threshold); return 0; });
}

/* =========================================================================
 * MergeDebevec
 * ========================================================================= */

cvk_merge_debevec_t *cvk_merge_debevec_create(void) {
    return guarded([&]() -> cvk_merge_debevec_t * {
        auto *h = new cvk_merge_debevec;
        h->ptr = cv::createMergeDebevec();
        return reinterpret_cast<cvk_merge_debevec_t *>(h);
    });
}

void cvk_merge_debevec_release(cvk_merge_debevec_t *h) {
    delete reinterpret_cast<cvk_merge_debevec *>(h);
}

cvk_mat_t *cvk_merge_debevec_process(const cvk_merge_debevec_t *h,
                                     const cvk_mat_t *const *src, int count,
                                     const cvk_mat_t *times) {
    const auto *p = reinterpret_cast<const cvk_merge_debevec *>(h);
    const cv::Mat *t = require_const(times);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || t == nullptr || !collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst, *t);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_merge_debevec_process_response(const cvk_merge_debevec_t *h,
                                              const cvk_mat_t *const *src, int count,
                                              const cvk_mat_t *times,
                                              const cvk_mat_t *response) {
    const auto *p = reinterpret_cast<const cvk_merge_debevec *>(h);
    const cv::Mat *t = require_const(times);
    const cv::Mat *r = require_const(response);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || t == nullptr || r == nullptr || !collect_mats(src, count, srcs)) {
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst, *t, *r);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * MergeMertens
 * ========================================================================= */

cvk_merge_mertens_t *cvk_merge_mertens_create(float contrast_weight, float saturation_weight,
                                              float exposure_weight) {
    return guarded([&]() -> cvk_merge_mertens_t * {
        auto *h = new cvk_merge_mertens;
        h->ptr = cv::createMergeMertens(contrast_weight, saturation_weight, exposure_weight);
        return reinterpret_cast<cvk_merge_mertens_t *>(h);
    });
}

void cvk_merge_mertens_release(cvk_merge_mertens_t *h) {
    delete reinterpret_cast<cvk_merge_mertens *>(h);
}

cvk_mat_t *cvk_merge_mertens_process(const cvk_merge_mertens_t *h,
                                     const cvk_mat_t *const *src, int count) {
    const auto *p = reinterpret_cast<const cvk_merge_mertens *>(h);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || !collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_merge_mertens_process_response(const cvk_merge_mertens_t *h,
                                              const cvk_mat_t *const *src, int count,
                                              const cvk_mat_t *times,
                                              const cvk_mat_t *response) {
    const auto *p = reinterpret_cast<const cvk_merge_mertens *>(h);
    const cv::Mat *t = require_const(times);
    const cv::Mat *r = require_const(response);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || t == nullptr || r == nullptr || !collect_mats(src, count, srcs)) {
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst, *t, *r);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

float cvk_merge_mertens_get_contrast_weight(const cvk_merge_mertens_t *h) {
    const auto *p = reinterpret_cast<const cvk_merge_mertens *>(h);
    if (p == nullptr) { record_error("null mergeMertens handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getContrastWeight(); });
}

void cvk_merge_mertens_set_contrast_weight(cvk_merge_mertens_t *h, float contrast_weight) {
    auto *p = reinterpret_cast<cvk_merge_mertens *>(h);
    if (p == nullptr) { record_error("null mergeMertens handle"); return; }
    guarded([&]() -> int { p->ptr->setContrastWeight(contrast_weight); return 0; });
}

float cvk_merge_mertens_get_saturation_weight(const cvk_merge_mertens_t *h) {
    const auto *p = reinterpret_cast<const cvk_merge_mertens *>(h);
    if (p == nullptr) { record_error("null mergeMertens handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getSaturationWeight(); });
}

void cvk_merge_mertens_set_saturation_weight(cvk_merge_mertens_t *h, float saturation_weight) {
    auto *p = reinterpret_cast<cvk_merge_mertens *>(h);
    if (p == nullptr) { record_error("null mergeMertens handle"); return; }
    guarded([&]() -> int { p->ptr->setSaturationWeight(saturation_weight); return 0; });
}

float cvk_merge_mertens_get_exposure_weight(const cvk_merge_mertens_t *h) {
    const auto *p = reinterpret_cast<const cvk_merge_mertens *>(h);
    if (p == nullptr) { record_error("null mergeMertens handle"); return 0.0f; }
    return guarded([&]() -> float { return p->ptr->getExposureWeight(); });
}

void cvk_merge_mertens_set_exposure_weight(cvk_merge_mertens_t *h, float exposure_weight) {
    auto *p = reinterpret_cast<cvk_merge_mertens *>(h);
    if (p == nullptr) { record_error("null mergeMertens handle"); return; }
    guarded([&]() -> int { p->ptr->setExposureWeight(exposure_weight); return 0; });
}

/* =========================================================================
 * MergeRobertson
 * ========================================================================= */

cvk_merge_robertson_t *cvk_merge_robertson_create(void) {
    return guarded([&]() -> cvk_merge_robertson_t * {
        auto *h = new cvk_merge_robertson;
        h->ptr = cv::createMergeRobertson();
        return reinterpret_cast<cvk_merge_robertson_t *>(h);
    });
}

void cvk_merge_robertson_release(cvk_merge_robertson_t *h) {
    delete reinterpret_cast<cvk_merge_robertson *>(h);
}

cvk_mat_t *cvk_merge_robertson_process(const cvk_merge_robertson_t *h,
                                       const cvk_mat_t *const *src, int count,
                                       const cvk_mat_t *times) {
    const auto *p = reinterpret_cast<const cvk_merge_robertson *>(h);
    const cv::Mat *t = require_const(times);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || t == nullptr || !collect_mats(src, count, srcs)) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst, *t);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

cvk_mat_t *cvk_merge_robertson_process_response(const cvk_merge_robertson_t *h,
                                                const cvk_mat_t *const *src, int count,
                                                const cvk_mat_t *times,
                                                const cvk_mat_t *response) {
    const auto *p = reinterpret_cast<const cvk_merge_robertson *>(h);
    const cv::Mat *t = require_const(times);
    const cv::Mat *r = require_const(response);
    std::vector<cv::Mat> srcs;
    if (p == nullptr || t == nullptr || r == nullptr || !collect_mats(src, count, srcs)) {
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->process(srcs, *dst, *t, *r);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * ColorCorrectionModel (cv::ccm)
 * ========================================================================= */

cvk_color_correction_model_t *cvk_color_correction_model_create(const cvk_mat_t *src,
                                                                int const_color) {
    const cv::Mat *m = require_const(src);
    if (m == nullptr) return nullptr;
    return guarded([&]() -> cvk_color_correction_model_t * {
        auto *h = new cvk_color_correction_model;
        h->ptr = new cv::ccm::ColorCorrectionModel(
                *m, static_cast<cv::ccm::ColorCheckerType>(const_color));
        return reinterpret_cast<cvk_color_correction_model_t *>(h);
    });
}

cvk_color_correction_model_t *cvk_color_correction_model_create_empty(void) {
    return guarded([&]() -> cvk_color_correction_model_t * {
        auto *h = new cvk_color_correction_model;
        h->ptr = new cv::ccm::ColorCorrectionModel();
        return reinterpret_cast<cvk_color_correction_model_t *>(h);
    });
}

void cvk_color_correction_model_release(cvk_color_correction_model_t *h) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) return;
    delete p->ptr;
    delete p;
}

void cvk_color_correction_model_set_linearization_gamma(cvk_color_correction_model_t *h,
                                                        double gamma) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return; }
    guarded([&]() -> int { p->ptr->setLinearizationGamma(gamma); return 0; });
}

void cvk_color_correction_model_set_linearization_degree(cvk_color_correction_model_t *h,
                                                         int deg) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return; }
    guarded([&]() -> int { p->ptr->setLinearizationDegree(deg); return 0; });
}

void cvk_color_correction_model_set_saturated_threshold(cvk_color_correction_model_t *h,
                                                        double lower, double upper) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return; }
    guarded([&]() -> int { p->ptr->setSaturatedThreshold(lower, upper); return 0; });
}

void cvk_color_correction_model_set_weights_list(cvk_color_correction_model_t *h,
                                                 const cvk_mat_t *weights) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    const cv::Mat *w = require_const(weights);
    if (p == nullptr || w == nullptr) return;
    guarded([&]() -> int { p->ptr->setWeightsList(*w); return 0; });
}

void cvk_color_correction_model_set_weight_coeff(cvk_color_correction_model_t *h,
                                                 double weights_coeff) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return; }
    guarded([&]() -> int { p->ptr->setWeightCoeff(weights_coeff); return 0; });
}

void cvk_color_correction_model_set_max_count(cvk_color_correction_model_t *h, int max_count) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return; }
    guarded([&]() -> int { p->ptr->setMaxCount(max_count); return 0; });
}

void cvk_color_correction_model_set_epsilon(cvk_color_correction_model_t *h, double epsilon) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return; }
    guarded([&]() -> int { p->ptr->setEpsilon(epsilon); return 0; });
}

void cvk_color_correction_model_set_rgb(cvk_color_correction_model_t *h, int rgb) {
    auto *p = reinterpret_cast<cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return; }
    guarded([&]() -> int { p->ptr->setRGB(rgb != 0); return 0; });
}

cvk_mat_t *cvk_color_correction_model_compute(const cvk_color_correction_model_t *h) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->ptr->compute()));
    });
}

cvk_mat_t *cvk_color_correction_model_get_color_correction_matrix(
        const cvk_color_correction_model_t *h) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->ptr->getColorCorrectionMatrix()));
    });
}

double cvk_color_correction_model_get_loss(const cvk_color_correction_model_t *h) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return 0.0; }
    return guarded([&]() -> double { return p->ptr->getLoss(); });
}

cvk_mat_t *cvk_color_correction_model_get_src_linear_rgb(const cvk_color_correction_model_t *h) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->ptr->getSrcLinearRGB()));
    });
}

cvk_mat_t *cvk_color_correction_model_get_ref_linear_rgb(const cvk_color_correction_model_t *h) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->ptr->getRefLinearRGB()));
    });
}

cvk_mat_t *cvk_color_correction_model_get_mask(const cvk_color_correction_model_t *h) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->ptr->getMask()));
    });
}

cvk_mat_t *cvk_color_correction_model_get_weights(const cvk_color_correction_model_t *h) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    if (p == nullptr) { record_error("null colorCorrectionModel handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(p->ptr->getWeights()));
    });
}

cvk_mat_t *cvk_color_correction_model_correct_image(const cvk_color_correction_model_t *h,
                                                    const cvk_mat_t *src, int islinear) {
    const auto *p = reinterpret_cast<const cvk_color_correction_model *>(h);
    const cv::Mat *m = require_const(src);
    if (p == nullptr || m == nullptr) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        auto *dst = new cv::Mat();
        p->ptr->correctImage(*m, *dst, islinear != 0);
        return reinterpret_cast<cvk_mat_t *>(dst);
    });
}

/* =========================================================================
 * IntelligentScissorsMB (cv::segmentation)
 * ========================================================================= */

cvk_intelligent_scissors_mb_t *cvk_intelligent_scissors_mb_create(void) {
    return guarded([&]() -> cvk_intelligent_scissors_mb_t * {
        auto *h = new cvk_intelligent_scissors_mb;
        h->ptr = new cv::segmentation::IntelligentScissorsMB();
        return reinterpret_cast<cvk_intelligent_scissors_mb_t *>(h);
    });
}

void cvk_intelligent_scissors_mb_release(cvk_intelligent_scissors_mb_t *h) {
    auto *p = reinterpret_cast<cvk_intelligent_scissors_mb *>(h);
    if (p == nullptr) return;
    delete p->ptr;
    delete p;
}

void cvk_intelligent_scissors_mb_set_weights(cvk_intelligent_scissors_mb_t *h,
                                             float weight_non_edge,
                                             float weight_gradient_direction,
                                             float weight_gradient_magnitude) {
    auto *p = reinterpret_cast<cvk_intelligent_scissors_mb *>(h);
    if (p == nullptr) { record_error("null scissors handle"); return; }
    guarded([&]() -> int {
        p->ptr->setWeights(weight_non_edge, weight_gradient_direction,
                           weight_gradient_magnitude);
        return 0;
    });
}

void cvk_intelligent_scissors_mb_set_gradient_magnitude_max_limit(
        cvk_intelligent_scissors_mb_t *h, float gradient_magnitude_threshold_max) {
    auto *p = reinterpret_cast<cvk_intelligent_scissors_mb *>(h);
    if (p == nullptr) { record_error("null scissors handle"); return; }
    guarded([&]() -> int {
        p->ptr->setGradientMagnitudeMaxLimit(gradient_magnitude_threshold_max);
        return 0;
    });
}

void cvk_intelligent_scissors_mb_set_edge_feature_zero_crossing_parameters(
        cvk_intelligent_scissors_mb_t *h, float gradient_magnitude_min_value) {
    auto *p = reinterpret_cast<cvk_intelligent_scissors_mb *>(h);
    if (p == nullptr) { record_error("null scissors handle"); return; }
    guarded([&]() -> int {
        p->ptr->setEdgeFeatureZeroCrossingParameters(gradient_magnitude_min_value);
        return 0;
    });
}

void cvk_intelligent_scissors_mb_set_edge_feature_canny_parameters(
        cvk_intelligent_scissors_mb_t *h, double threshold1, double threshold2,
        int aperture_size, int l2gradient) {
    auto *p = reinterpret_cast<cvk_intelligent_scissors_mb *>(h);
    if (p == nullptr) { record_error("null scissors handle"); return; }
    guarded([&]() -> int {
        p->ptr->setEdgeFeatureCannyParameters(threshold1, threshold2, aperture_size,
                                              l2gradient != 0);
        return 0;
    });
}

void cvk_intelligent_scissors_mb_apply_image(cvk_intelligent_scissors_mb_t *h,
                                             const cvk_mat_t *image) {
    auto *p = reinterpret_cast<cvk_intelligent_scissors_mb *>(h);
    const cv::Mat *m = require_const(image);
    if (p == nullptr || m == nullptr) return;
    guarded([&]() -> int { p->ptr->applyImage(*m); return 0; });
}

void cvk_intelligent_scissors_mb_apply_image_features(cvk_intelligent_scissors_mb_t *h,
                                                      const cvk_mat_t *non_edge,
                                                      const cvk_mat_t *gradient_direction,
                                                      const cvk_mat_t *gradient_magnitude,
                                                      const cvk_mat_t *image) {
    auto *p = reinterpret_cast<cvk_intelligent_scissors_mb *>(h);
    const cv::Mat *ne = require_const(non_edge);
    const cv::Mat *gd = require_const(gradient_direction);
    const cv::Mat *gm = require_const(gradient_magnitude);
    const cv::Mat *img = image != nullptr ? require_const(image) : nullptr;
    if (p == nullptr || ne == nullptr || gd == nullptr || gm == nullptr) return;
    guarded([&]() -> int {
        if (img != nullptr) {
            p->ptr->applyImageFeatures(*ne, *gd, *gm, *img);
        } else {
            p->ptr->applyImageFeatures(*ne, *gd, *gm);
        }
        return 0;
    });
}

void cvk_intelligent_scissors_mb_build_map(const cvk_intelligent_scissors_mb_t *h,
                                           int source_x, int source_y) {
    const auto *p = reinterpret_cast<const cvk_intelligent_scissors_mb *>(h);
    if (p == nullptr) { record_error("null scissors handle"); return; }
    guarded([&]() -> int {
        p->ptr->buildMap(cv::Point(source_x, source_y));
        return 0;
    });
}

cvk_mat_t *cvk_intelligent_scissors_mb_get_contour(const cvk_intelligent_scissors_mb_t *h,
                                                   int target_x, int target_y, int backward) {
    const auto *p = reinterpret_cast<const cvk_intelligent_scissors_mb *>(h);
    if (p == nullptr) { record_error("null scissors handle"); return nullptr; }
    return guarded([&]() -> cvk_mat_t * {
        auto *contour = new cv::Mat();
        p->ptr->getContour(cv::Point(target_x, target_y), *contour, backward != 0);
        return reinterpret_cast<cvk_mat_t *>(contour);
    });
}

} /* extern "C" */

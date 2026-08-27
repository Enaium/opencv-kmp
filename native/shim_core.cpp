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
 * Implementation of the org.opencv.core.Core statics (core slice) over the
 * cvk_ C ABI. Every exported function is noexcept: cv::Exception is caught
 * and reported through cvk_core_last_error() / cvk_last_error(); functions
 * that return a new Mat return NULL on failure instead.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_core.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

/** Opaque RNG handle backing cvk_rng_t; defined at file scope so it
 *  completes the type the C header forward-declared. */
struct cvk_rng { cv::RNG rng; };

namespace {

thread_local std::string g_core_error;
thread_local std::string g_core_str;

void record_error(const char *message) {
    try {
        g_core_error = message != nullptr ? message : "unknown error";
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
    // Default-initialized result (NULL pointers / zeros).
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

cvk_scalar_t scalar_of(const cv::Scalar &s) {
    cvk_scalar_t out;
    out.v0 = s[0];
    out.v1 = s[1];
    out.v2 = s[2];
    out.v3 = s[3];
    return out;
}

} // namespace

extern "C" {

const char *cvk_core_last_error(void) {
    return g_core_error.empty() ? nullptr : g_core_error.c_str();
}

/* ---- scalar math ------------------------------------------------------ */

float cvk_cube_root(float val) {
    return guarded([&]() -> float { return cv::cubeRoot(val); });
}

float cvk_fast_atan2(float y, float x) {
    return guarded([&]() -> float { return cv::fastAtan2(y, x); });
}

int cvk_border_interpolate(int p, int len, int border_type) {
    return guarded([&]() -> int { return cv::borderInterpolate(p, len, border_type); });
}

/* ---- RNG -------------------------------------------------------------- */

cvk_rng_t *cvk_rng_from_global(void) {
    return guarded([&]() -> cvk_rng_t * {
        auto *h = new cvk_rng;
        h->rng = cv::RNG(cv::theRNG().state);
        return reinterpret_cast<cvk_rng_t *>(h);
    });
}

cvk_rng_t *cvk_rng_create(unsigned long long seed) {
    return guarded([&]() -> cvk_rng_t * {
        auto *h = new cvk_rng;
        h->rng = cv::RNG(static_cast<uint64>(seed));
        return reinterpret_cast<cvk_rng_t *>(h);
    });
}

unsigned int cvk_rng_next(cvk_rng_t *rng) {
    auto *p = reinterpret_cast<cvk_rng *>(rng);
    if (!p) {
        record_error("null RNG handle");
        return 0;
    }
    return guarded([&]() -> unsigned int { return p->rng.next(); });
}

int cvk_rng_uniform_int(cvk_rng_t *rng, int a, int b) {
    auto *p = reinterpret_cast<cvk_rng *>(rng);
    if (!p) {
        record_error("null RNG handle");
        return 0;
    }
    return guarded([&]() -> int { return p->rng.uniform(a, b); });
}

float cvk_rng_uniform_float(cvk_rng_t *rng, float a, float b) {
    auto *p = reinterpret_cast<cvk_rng *>(rng);
    if (!p) {
        record_error("null RNG handle");
        return 0.0f;
    }
    return guarded([&]() -> float { return p->rng.uniform(a, b); });
}

double cvk_rng_uniform_double(cvk_rng_t *rng, double a, double b) {
    auto *p = reinterpret_cast<cvk_rng *>(rng);
    if (!p) {
        record_error("null RNG handle");
        return 0.0;
    }
    return guarded([&]() -> double { return p->rng.uniform(a, b); });
}

double cvk_rng_gaussian(cvk_rng_t *rng, double sigma) {
    auto *p = reinterpret_cast<cvk_rng *>(rng);
    if (!p) {
        record_error("null RNG handle");
        return 0.0;
    }
    return guarded([&]() -> double { return p->rng.gaussian(sigma); });
}

void cvk_rng_release(cvk_rng_t *rng) {
    delete reinterpret_cast<cvk_rng *>(rng);
}

/* ---- array operations ------------------------------------------------- */

void cvk_mix_channels(cvk_mat_t **srcs, int nsrcs, cvk_mat_t **dsts,
                      int ndsts, const int *from_to, size_t from_to_len) {
    guarded([&]() -> int {
        if (srcs == nullptr || dsts == nullptr || from_to == nullptr ||
            nsrcs <= 0 || ndsts <= 0) {
            record_error("null mixChannels argument");
            return 0;
        }
        std::vector<cv::Mat> src_mats(nsrcs), dst_mats(ndsts);
        for (int i = 0; i < nsrcs; ++i) {
            cv::Mat *m = require(srcs[i]);
            if (m == nullptr) return 0;
            src_mats[i] = *m;
        }
        for (int i = 0; i < ndsts; ++i) {
            cv::Mat *m = require(dsts[i]);
            if (m == nullptr) return 0;
            dst_mats[i] = *m;
        }
        cv::mixChannels(src_mats.data(), nsrcs, dst_mats.data(), ndsts,
                        from_to, from_to_len / 2);
        return 0;
    });
}

int cvk_batch_distance(const cvk_mat_t *src1, const cvk_mat_t *src2,
                       cvk_mat_t *dist, int dtype, cvk_mat_t *nidx,
                       int norm_type, int k, const cvk_mat_t *mask,
                       int update, int crosscheck) {
    return guarded([&]() -> int {
        const cv::Mat *m1 = require_const(src1);
        const cv::Mat *m2 = require_const(src2);
        cv::Mat *d = require(dist);
        if (m1 == nullptr || m2 == nullptr || d == nullptr) {
            return 0;
        }
        cv::Mat *n = require(nidx);
        const cv::Mat *mk = require_const(mask);
        cv::batchDistance(*m1, *m2, *d, dtype,
                          n != nullptr ? *n : cv::noArray(), norm_type, k,
                          mk != nullptr ? *mk : cv::noArray(), update,
                          crosscheck != 0);
        return 1;
    });
}

void cvk_calc_covar_matrix(const cvk_mat_t *samples, cvk_mat_t *covar,
                           cvk_mat_t *mean, int flags, int ctype) {
    guarded([&]() -> int {
        const cv::Mat *s = require_const(samples);
        cv::Mat *c = require(covar);
        cv::Mat *m = require(mean);
        if (s == nullptr || c == nullptr || m == nullptr) return 0;
        cv::calcCovarMatrix(*s, *c, *m, flags, ctype);
        return 0;
    });
}

void cvk_complete_symm(cvk_mat_t *m, int lower_to_upper) {
    guarded([&]() -> int {
        cv::Mat *mat = require(m);
        if (mat == nullptr) return 0;
        cv::completeSymm(*mat, lower_to_upper != 0);
        return 0;
    });
}

int cvk_solve_cubic(const cvk_mat_t *coeffs, cvk_mat_t **roots) {
    return guarded([&]() -> int {
        const cv::Mat *c = require_const(coeffs);
        if (c == nullptr || roots == nullptr) return 0;
        auto *r = new cv::Mat();
        int count = cv::solveCubic(*c, *r);
        *roots = reinterpret_cast<cvk_mat_t *>(r);
        return count;
    });
}

double cvk_solve_poly(const cvk_mat_t *coeffs, cvk_mat_t **roots,
                      int max_iters) {
    return guarded([&]() -> double {
        const cv::Mat *c = require_const(coeffs);
        if (c == nullptr || roots == nullptr) return 0.0;
        auto *r = new cv::Mat();
        double eps = cv::solvePoly(*c, *r, max_iters);
        *roots = reinterpret_cast<cvk_mat_t *>(r);
        return eps;
    });
}

cvk_mat_t *cvk_mul_transposed(const cvk_mat_t *src, int a_ta,
                              const cvk_mat_t *delta, double scale,
                              int dtype) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *s = require_const(src);
        if (s == nullptr) return nullptr;
        const cv::Mat *d = require_const(delta);
        auto *out = new cv::Mat();
        cv::mulTransposed(*s, *out, a_ta != 0,
                          d != nullptr ? *d : cv::noArray(), scale, dtype);
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

cvk_mat_t *cvk_flip_nd(const cvk_mat_t *src, int axis) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *s = require_const(src);
        if (s == nullptr) return nullptr;
        auto *out = new cv::Mat();
        cv::flipND(*s, *out, axis);
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

cvk_mat_t *cvk_broadcast(const cvk_mat_t *src, const cvk_mat_t *shape) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *s = require_const(src);
        const cv::Mat *sh = require_const(shape);
        if (s == nullptr || sh == nullptr) return nullptr;
        auto *out = new cv::Mat();
        cv::broadcast(*s, *sh, *out);
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

cvk_mat_t *cvk_transpose_nd(const cvk_mat_t *src, const int *order,
                            size_t order_len) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *s = require_const(src);
        if (s == nullptr || order == nullptr) return nullptr;
        std::vector<int> ord(order, order + order_len);
        auto *out = new cv::Mat();
        cv::transposeND(*s, ord, *out);
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

cvk_mat_t *cvk_copy_to(const cvk_mat_t *src, const cvk_mat_t *mask) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *s = require_const(src);
        if (s == nullptr) return nullptr;
        const cv::Mat *mk = require_const(mask);
        auto *out = new cv::Mat();
        cv::copyTo(*s, *out, mk != nullptr ? *mk : cv::noArray());
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

cvk_mat_t *cvk_scale_add(const cvk_mat_t *a, double alpha, const cvk_mat_t *b) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *ma = require_const(a);
        const cv::Mat *mb = require_const(b);
        if (ma == nullptr || mb == nullptr) return nullptr;
        auto *out = new cv::Mat();
        cv::scaleAdd(*ma, alpha, *mb, *out);
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

cvk_mat_t *cvk_gemm_flags(const cvk_mat_t *a, const cvk_mat_t *b,
                          double alpha, const cvk_mat_t *c, double gamma,
                          int flags) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *ma = require_const(a);
        const cv::Mat *mb = require_const(b);
        if (ma == nullptr || mb == nullptr) return nullptr;
        const cv::Mat *mc = require_const(c);
        auto *out = new cv::Mat();
        cv::gemm(*ma, *mb, alpha, mc != nullptr ? *mc : cv::noArray(), gamma,
                 *out, flags);
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

void cvk_eigen_non_symmetric(const cvk_mat_t *src, cvk_mat_t **eigenvalues,
                             cvk_mat_t **eigenvectors) {
    guarded([&]() -> int {
        const cv::Mat *s = require_const(src);
        if (s == nullptr || eigenvalues == nullptr || eigenvectors == nullptr) {
            return 0;
        }
        auto *vals = new cv::Mat();
        auto *vecs = new cv::Mat();
        cv::eigenNonSymmetric(*s, *vals, *vecs);
        *eigenvalues = reinterpret_cast<cvk_mat_t *>(vals);
        *eigenvectors = reinterpret_cast<cvk_mat_t *>(vecs);
        return 0;
    });
}

cvk_mat_t *cvk_finite_mask(const cvk_mat_t *src) {
    return guarded([&]() -> cvk_mat_t * {
        const cv::Mat *s = require_const(src);
        if (s == nullptr) return nullptr;
        auto *out = new cv::Mat();
        cv::finiteMask(*s, *out);
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

/* ---- masked statistics ------------------------------------------------ */

void cvk_mat_min_max_loc_masked(const cvk_mat_t *src, const cvk_mat_t *mask,
                                double *out6) {
    guarded([&]() -> int {
        const cv::Mat *m = require_const(src);
        const cv::Mat *mk = require_const(mask);
        if (m == nullptr || mk == nullptr || out6 == nullptr) return 0;
        double min_val = 0, max_val = 0;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(*m, &min_val, &max_val, &min_loc, &max_loc, *mk);
        out6[0] = min_val;
        out6[1] = max_val;
        out6[2] = static_cast<double>(min_loc.x);
        out6[3] = static_cast<double>(min_loc.y);
        out6[4] = static_cast<double>(max_loc.x);
        out6[5] = static_cast<double>(max_loc.y);
        return 0;
    });
}

cvk_scalar_t cvk_mat_mean_masked(const cvk_mat_t *src, const cvk_mat_t *mask) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *mk = require_const(mask);
    if (m == nullptr || mk == nullptr) return cvk_scalar_t{};
    return guarded([&]() -> cvk_scalar_t {
        return scalar_of(cv::mean(*m, *mk));
    });
}

double cvk_norm_masked(const cvk_mat_t *src, int norm_type,
                       const cvk_mat_t *mask) {
    const cv::Mat *m = require_const(src);
    const cv::Mat *mk = require_const(mask);
    if (m == nullptr || mk == nullptr) return -1.0;
    return guarded([&]() -> double { return cv::norm(*m, norm_type, *mk); });
}

double cvk_norm_diff_masked(const cvk_mat_t *a, const cvk_mat_t *b,
                            int norm_type, const cvk_mat_t *mask) {
    const cv::Mat *ma = require_const(a);
    const cv::Mat *mb = require_const(b);
    const cv::Mat *mk = require_const(mask);
    if (ma == nullptr || mb == nullptr || mk == nullptr) return -1.0;
    return guarded([&]() -> double {
        return cv::norm(*ma, *mb, norm_type, *mk);
    });
}

/* ---- range check / shuffle -------------------------------------------- */

int cvk_check_range(const cvk_mat_t *a, int quiet, double min_val,
                    double max_val) {
    const cv::Mat *m = require_const(a);
    if (m == nullptr) return 0;
    return guarded([&]() -> int {
        return cv::checkRange(*m, quiet != 0, 0, min_val, max_val) ? 1 : 0;
    });
}

void cvk_rand_shuffle(cvk_mat_t *dst, double iter_factor) {
    guarded([&]() -> int {
        cv::Mat *m = require(dst);
        if (m == nullptr) return 0;
        cv::randShuffle(*m, iter_factor);
        return 0;
    });
}

/* ---- environment / runtime info --------------------------------------- */

long long cvk_get_tick_count(void) {
    return guarded([&]() -> long long { return static_cast<long long>(cv::getTickCount()); });
}

double cvk_get_tick_frequency(void) {
    return guarded([&]() -> double { return cv::getTickFrequency(); });
}

int cvk_get_number_of_cpus(void) {
    return guarded([&]() -> int { return cv::getNumberOfCPUs(); });
}

int cvk_check_hardware_support(int feature) {
    return guarded([&]() -> int { return cv::checkHardwareSupport(feature) ? 1 : 0; });
}

const char *cvk_get_hardware_feature_name(int feature) {
    return guarded([&]() -> const char * {
        g_core_str = cv::getHardwareFeatureName(feature);
        return g_core_str.c_str();
    });
}

const char *cvk_get_version_string(void) {
    return guarded([&]() -> const char * {
        g_core_str = cv::getVersionString();
        return g_core_str.c_str();
    });
}

int cvk_get_version_major(void) {
    return guarded([&]() -> int { return cv::getVersionMajor(); });
}

int cvk_get_version_minor(void) {
    return guarded([&]() -> int { return cv::getVersionMinor(); });
}

int cvk_get_version_revision(void) {
    return guarded([&]() -> int { return cv::getVersionRevision(); });
}

long long cvk_get_cpu_tick_count(void) {
    return guarded([&]() -> long long { return static_cast<long long>(cv::getCPUTickCount()); });
}

int cvk_get_thread_num(void) {
    return guarded([&]() -> int { return cv::getThreadNum(); });
}

int cvk_get_default_algorithm_hint(void) {
    return guarded([&]() -> int {
        return static_cast<int>(cv::getDefaultAlgorithmHint());
    });
}

int cvk_use_optimized(void) {
    return guarded([&]() -> int { return cv::useOptimized() ? 1 : 0; });
}

void cvk_set_use_optimized(int onoff) {
    guarded([&]() -> int {
        cv::setUseOptimized(onoff != 0);
        return 0;
    });
}

const char *cvk_get_cpu_features_line(void) {
    return guarded([&]() -> const char * {
        g_core_str = cv::getCPUFeaturesLine();
        return g_core_str.c_str();
    });
}

int cvk_use_ipp(void) {
    return guarded([&]() -> int { return cv::ipp::useIPP() ? 1 : 0; });
}

void cvk_set_use_ipp(int flag) {
    guarded([&]() -> int {
        cv::ipp::setUseIPP(flag != 0);
        return 0;
    });
}

const char *cvk_get_ipp_version(void) {
    return guarded([&]() -> const char * {
        g_core_str = cv::ipp::getIppVersion();
        return g_core_str.c_str();
    });
}

int cvk_use_ipp_not_exact(void) {
    return guarded([&]() -> int { return cv::ipp::useIPP_NotExact() ? 1 : 0; });
}

void cvk_set_use_ipp_not_exact(int flag) {
    guarded([&]() -> int {
        cv::ipp::setUseIPP_NotExact(flag != 0);
        return 0;
    });
}

const char *cvk_find_file(const char *relative_path, int required,
                          int silent_mode) {
    return guarded([&]() -> const char * {
        if (relative_path == nullptr) {
            record_error("null path");
            return nullptr;
        }
        g_core_str = cv::samples::findFile(relative_path, required != 0,
                                           silent_mode != 0);
        return g_core_str.c_str();
    });
}

const char *cvk_find_file_or_keep(const char *relative_path, int silent_mode) {
    return guarded([&]() -> const char * {
        if (relative_path == nullptr) {
            record_error("null path");
            return nullptr;
        }
        g_core_str = cv::samples::findFileOrKeep(relative_path,
                                                 silent_mode != 0);
        return g_core_str.c_str();
    });
}

void cvk_add_samples_data_search_path(const char *path) {
    guarded([&]() -> int {
        if (path == nullptr) {
            record_error("null path");
            return 0;
        }
        cv::samples::addSamplesDataSearchPath(path);
        return 0;
    });
}

void cvk_add_samples_data_search_sub_directory(const char *subdir) {
    guarded([&]() -> int {
        if (subdir == nullptr) {
            record_error("null subdir");
            return 0;
        }
        cv::samples::addSamplesDataSearchSubDirectory(subdir);
        return 0;
    });
}

} /* extern "C" */

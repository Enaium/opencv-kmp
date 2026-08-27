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
 * cv::TickMeter wrapper (coretypes slice).
 *
 * cv::TickMeter keeps its accumulated tick total private in OpenCV 5, so the
 * sum is read through getTimeTicks() (which returns sumTime) and the clock
 * rate through cv::getTickFrequency(). All exported functions are noexcept
 * via guarded(); on failure they record the message locally and return the
 * documented zero values.
 */
#include "opencv_kmp.h"
#include "opencv_kmp_coretypes.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

struct cvk_tick_meter { cv::TickMeter tm; };

namespace {

thread_local std::string g_coretypes_error;

void record_error(const char *message) {
    try {
        g_coretypes_error = message != nullptr ? message : "unknown error";
    } catch (...) {
        /* ignore allocation failures while reporting an error */
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

} // namespace

extern "C" {

cvk_tick_meter_t *cvk_tick_meter_create(void) {
    return guarded([&]() -> cvk_tick_meter_t * {
        auto *h = new cvk_tick_meter;
        return reinterpret_cast<cvk_tick_meter_t *>(h);
    });
}

void cvk_tick_meter_start(cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return;
    }
    guarded([&]() -> int {
        p->tm.start();
        return 0;
    });
}

void cvk_tick_meter_stop(cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return;
    }
    guarded([&]() -> int {
        p->tm.stop();
        return 0;
    });
}

void cvk_tick_meter_reset(cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return;
    }
    guarded([&]() -> int {
        p->tm.reset();
        return 0;
    });
}

int64_t cvk_tick_meter_get_time_ticks(const cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<const cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return 0;
    }
    return guarded([&]() -> int64_t { return p->tm.getTimeTicks(); });
}

double cvk_tick_meter_get_time_sec(const cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<const cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return 0.0;
    }
    return guarded([&]() -> double { return p->tm.getTimeSec(); });
}

int64_t cvk_tick_meter_get_time_sum(const cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<const cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return 0;
    }
    return guarded([&]() -> int64_t { return p->tm.getTimeTicks(); });
}

int64_t cvk_tick_meter_get_counter(const cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<const cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return 0;
    }
    return guarded([&]() -> int64_t { return p->tm.getCounter(); });
}

double cvk_tick_meter_get_avg_time(const cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<const cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return 0.0;
    }
    return guarded([&]() -> double { return p->tm.getAvgTimeSec(); });
}

double cvk_tick_meter_get_freq(const cvk_tick_meter_t *h) {
    auto *p = reinterpret_cast<const cvk_tick_meter *>(h);
    if (!p) {
        record_error("null TickMeter handle");
        return 0.0;
    }
    return guarded([&]() -> double { return cv::getTickFrequency(); });
}

void cvk_tick_meter_release(cvk_tick_meter_t *h) {
    delete reinterpret_cast<cvk_tick_meter *>(h);
}

} /* extern "C" */

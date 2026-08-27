/*
 * cvk_ C ABI declarations for opencv-kmp slice "coretypes".
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here back both Kotlin/Native (cinterop) and the
 * JVM (JNI).
 */
#ifndef OPENCV_KMP_CORETYPES_H
#define OPENCV_KMP_CORETYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to a cv::TickMeter. */
typedef struct cvk_tick_meter cvk_tick_meter_t;

/** Creates a stopped cv::TickMeter (counter and accumulated time zeroed). */
cvk_tick_meter_t *cvk_tick_meter_create(void);

/** Starts counting ticks; a second start without stop restarts the interval. */
void cvk_tick_meter_start(cvk_tick_meter_t *h);

/** Stops counting and accumulates the interval; no-op when not running. */
void cvk_tick_meter_stop(cvk_tick_meter_t *h);

/** Resets the counter and accumulated time to zero. */
void cvk_tick_meter_reset(cvk_tick_meter_t *h);

/** Total accumulated ticks across stopped intervals. */
int64_t cvk_tick_meter_get_time_ticks(const cvk_tick_meter_t *h);

/** Total accumulated time in seconds. */
double cvk_tick_meter_get_time_sec(const cvk_tick_meter_t *h);

/** Sum of ticks across stopped intervals (same value as get_time_ticks). */
int64_t cvk_tick_meter_get_time_sum(const cvk_tick_meter_t *h);

/** Number of completed start/stop intervals. */
int64_t cvk_tick_meter_get_counter(const cvk_tick_meter_t *h);

/** Average seconds per interval (0 when nothing was measured yet). */
double cvk_tick_meter_get_avg_time(const cvk_tick_meter_t *h);

/** Ticks per second of the platform clock (cv::getTickFrequency). */
double cvk_tick_meter_get_freq(const cvk_tick_meter_t *h);

/** Frees the handle (exactly once). */
void cvk_tick_meter_release(cvk_tick_meter_t *h);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_CORETYPES_H */

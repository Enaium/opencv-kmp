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
 * JNI bridge for the coretypes slice: thin Java_cn_enaium_opencv_JniCoreTypes_*
 * wrappers around the cvk_tick_meter_* C ABI. The TickMeter handle travels as
 * a jlong pointer; the shim is noexcept so nothing throws across the boundary.
 */
#include <jni.h>
#include "opencv_kmp.h"
#include "opencv_kmp_coretypes.h"

#include <cstdint>

static inline cvk_tick_meter_t *as_tick_meter(jlong handle) {
    return reinterpret_cast<cvk_tick_meter_t *>(static_cast<uintptr_t>(handle));
}

static inline jlong as_tick_meter_handle(const cvk_tick_meter_t *tm) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(tm));
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterCreate(JNIEnv *, jobject) {
    return as_tick_meter_handle(cvk_tick_meter_create());
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterRelease(JNIEnv *, jobject, jlong h) {
    cvk_tick_meter_release(as_tick_meter(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterStart(JNIEnv *, jobject, jlong h) {
    cvk_tick_meter_start(as_tick_meter(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterStop(JNIEnv *, jobject, jlong h) {
    cvk_tick_meter_stop(as_tick_meter(h));
}

JNIEXPORT void JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterReset(JNIEnv *, jobject, jlong h) {
    cvk_tick_meter_reset(as_tick_meter(h));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterGetTimeTicks(JNIEnv *, jobject, jlong h) {
    return static_cast<jlong>(cvk_tick_meter_get_time_ticks(as_tick_meter(h)));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterGetTimeSec(JNIEnv *, jobject, jlong h) {
    return cvk_tick_meter_get_time_sec(as_tick_meter(h));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterGetTimeSum(JNIEnv *, jobject, jlong h) {
    return static_cast<jlong>(cvk_tick_meter_get_time_sum(as_tick_meter(h)));
}

JNIEXPORT jlong JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterGetCounter(JNIEnv *, jobject, jlong h) {
    return static_cast<jlong>(cvk_tick_meter_get_counter(as_tick_meter(h)));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterGetAvgTime(JNIEnv *, jobject, jlong h) {
    return cvk_tick_meter_get_avg_time(as_tick_meter(h));
}

JNIEXPORT jdouble JNICALL
Java_cn_enaium_opencv_JniCoreTypes_tickMeterGetFreq(JNIEnv *, jobject, jlong h) {
    return cvk_tick_meter_get_freq(as_tick_meter(h));
}

} /* extern "C" */

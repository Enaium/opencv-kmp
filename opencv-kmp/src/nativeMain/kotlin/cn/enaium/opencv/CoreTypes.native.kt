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
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package cn.enaium.opencv

import cvk.cvk_last_error
import cvk.cvk_tick_meter_create
import cvk.cvk_tick_meter_get_avg_time
import cvk.cvk_tick_meter_get_counter
import cvk.cvk_tick_meter_get_freq
import cvk.cvk_tick_meter_get_time_sec
import cvk.cvk_tick_meter_get_time_sum
import cvk.cvk_tick_meter_get_time_ticks
import cvk.cvk_tick_meter_release
import cvk.cvk_tick_meter_reset
import cvk.cvk_tick_meter_start
import cvk.cvk_tick_meter_stop
import cvk.cvk_tick_meter_t
import kotlin.concurrent.Volatile
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.toKString

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

/**
 * cinterop-backed [TickMeter] wrapping a raw `cv::TickMeter` handle.
 */
internal class NativeTickMeter(@Volatile private var raw: CPointer<cvk_tick_meter_t>?) : TickMeter {

    private fun check(): CPointer<cvk_tick_meter_t> =
        raw ?: throw IllegalStateException("TickMeter is closed")

    override val timeSum: Double get() = cvk_tick_meter_get_time_sum(check()).toDouble()
    override val counter: Int get() = cvk_tick_meter_get_counter(check()).toInt()
    override val avgTime: Double get() = cvk_tick_meter_get_avg_time(check())
    override val freq: Double get() = cvk_tick_meter_get_freq(check())
    override val timeTicks: Double get() = cvk_tick_meter_get_time_ticks(check()).toDouble()
    override val timeSec: Double get() = cvk_tick_meter_get_time_sec(check())

    override fun start() {
        cvk_tick_meter_start(check())
    }

    override fun stop() {
        cvk_tick_meter_stop(check())
    }

    override fun reset() {
        cvk_tick_meter_reset(check())
    }

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_tick_meter_release(h)
    }
}

/** Creates a stopped [TickMeter] backed by a native `cv::TickMeter`. */
actual fun tickMeter(): TickMeter =
    NativeTickMeter(
        cvk_tick_meter_create() ?: throw OpenCVException("tickMeter", lastNativeError()),
    )

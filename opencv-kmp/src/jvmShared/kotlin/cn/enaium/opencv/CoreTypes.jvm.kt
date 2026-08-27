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
package cn.enaium.opencv

private fun lastNativeError(): String? = Jni.lastError()

/**
 * JNI-backed [TickMeter] wrapping a `cv::TickMeter` handle.
 */
internal class JvmTickMeter(private var handle: Long) : TickMeter {

    private fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("TickMeter is closed")

    override val timeSum: Double get() = JniCoreTypes.tickMeterGetTimeSum(check()).toDouble()
    override val counter: Int get() = JniCoreTypes.tickMeterGetCounter(check()).toInt()
    override val avgTime: Double get() = JniCoreTypes.tickMeterGetAvgTime(check())
    override val freq: Double get() = JniCoreTypes.tickMeterGetFreq(check())
    override val timeTicks: Double get() = JniCoreTypes.tickMeterGetTimeTicks(check()).toDouble()
    override val timeSec: Double get() = JniCoreTypes.tickMeterGetTimeSec(check())

    override fun start() {
        JniCoreTypes.tickMeterStart(check())
    }

    override fun stop() {
        JniCoreTypes.tickMeterStop(check())
    }

    override fun reset() {
        JniCoreTypes.tickMeterReset(check())
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniCoreTypes.tickMeterRelease(h)
        }
    }
}

/** Creates a stopped [TickMeter] backed by a native `cv::TickMeter`. */
actual fun tickMeter(): TickMeter =
    JvmTickMeter(
        JniCoreTypes.tickMeterCreate().takeIf { it != 0L }
            ?: throw OpenCVException("tickMeter", lastNativeError()),
    )

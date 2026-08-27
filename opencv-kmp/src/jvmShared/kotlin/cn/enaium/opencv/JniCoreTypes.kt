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

/**
 * JNI bridge for the coretypes slice.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniCoreTypes_<name>`
 * function in jni/jni_coretypes.cpp. The TickMeter handle travels as a jlong
 * pointer. Members stay public (no `internal` modifier) so their JVM names
 * are not mangled, mirroring [Jni].
 */
internal object JniCoreTypes {

    init {
        NativeLoader.load()
    }

    external fun tickMeterCreate(): Long
    external fun tickMeterRelease(handle: Long)
    external fun tickMeterStart(handle: Long)
    external fun tickMeterStop(handle: Long)
    external fun tickMeterReset(handle: Long)
    external fun tickMeterGetTimeTicks(handle: Long): Long
    external fun tickMeterGetTimeSec(handle: Long): Double
    external fun tickMeterGetTimeSum(handle: Long): Long
    external fun tickMeterGetCounter(handle: Long): Long
    external fun tickMeterGetAvgTime(handle: Long): Double
    external fun tickMeterGetFreq(handle: Long): Double
}

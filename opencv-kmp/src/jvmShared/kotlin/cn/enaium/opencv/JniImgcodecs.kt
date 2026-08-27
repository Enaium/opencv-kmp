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
 * JNI bridge for the imgcodecs slice.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniImgcodecs_<name>`
 * function in jni/jni_imgcodecs.cpp. Mat handles travel as jlong pointers;
 * lists of Mat handles travel as jlongArray; codec parameters as jintArray.
 *
 * No init block: [Jni]'s initializer already loaded the native library.
 */
internal object JniImgcodecs {

    // Multi-page / multi-image codecs; null means the operation failed.

    external fun imreadmulti(path: String, flags: Int): LongArray?
    external fun imreadmultiRange(path: String, start: Int, count: Int, flags: Int): LongArray?
    external fun imdecodemulti(buf: Long, flags: Int, start: Int, end: Int): LongArray?
    external fun imwritemulti(path: String, mats: LongArray, params: IntArray): Boolean
    external fun imencodemulti(ext: String, mats: LongArray, params: IntArray): ByteArray

    // Animation

    external fun animationCreate(loopCount: Int, v0: Double, v1: Double, v2: Double, v3: Double): Long
    external fun animationRelease(animation: Long)
    external fun animationGetLoopCount(animation: Long): Int
    external fun animationSetLoopCount(animation: Long, loopCount: Int)
    external fun animationGetBgColor(animation: Long): DoubleArray
    external fun animationSetBgColor(animation: Long, v0: Double, v1: Double, v2: Double, v3: Double)
    external fun animationGetDurations(animation: Long): Long
    external fun animationSetDurations(animation: Long, durations: Long): Boolean
    external fun animationGetFrames(animation: Long): LongArray?
    external fun animationSetFrames(animation: Long, frames: LongArray): Boolean
    external fun animationGetStillImage(animation: Long): Long
    external fun animationSetStillImage(animation: Long, image: Long): Boolean

    // Animated image codecs

    external fun imreadanimation(path: String, animation: Long, start: Int, count: Int): Boolean
    external fun imdecodeanimation(data: ByteArray, animation: Long, start: Int, count: Int): Boolean
    external fun imwriteanimation(path: String, animation: Long, params: IntArray): Boolean
    external fun imencodeanimation(ext: String, animation: Long, params: IntArray): ByteArray
}

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
 * JNI bridge for the "video2" slice (long-term object trackers).
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniVideo2_<name>`
 * function in jni_video2.cpp. Tracker handles travel as jlong pointers; the
 * rect is expanded into primitives (init) or marshalled through an int[4]
 * out-array (update).
 */
internal object JniVideo2 {

    external fun trackerMILCreate(
        samplerInitInRadius: Float,
        samplerInitMaxNegNum: Int,
        samplerSearchWinSize: Float,
        samplerTrackInRadius: Float,
        samplerTrackMaxPosNum: Int,
        samplerTrackMaxNegNum: Int,
        featureSetNumFeatures: Int,
    ): Long

    external fun trackerDaSiamRPNCreate(
        model: String,
        kernelCls1: String,
        kernelR1: String,
        backend: Int,
        target: Int,
    ): Long

    external fun trackerNanoCreate(
        backbone: String,
        neckhead: String,
        backend: Int,
        target: Int,
    ): Long

    external fun trackerVitCreate(
        net: String,
        backend: Int,
        target: Int,
        meanV0: Double, meanV1: Double, meanV2: Double, meanV3: Double,
        stdV0: Double, stdV1: Double, stdV2: Double, stdV3: Double,
        trackingScoreThreshold: Float,
    ): Long

    external fun trackerInit(
        tracker: Long,
        image: Long,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
    )

    /** Runs tracker.update; fills [outRect] with x,y,width,height; returns success. */
    external fun trackerUpdate(tracker: Long, image: Long, outRect: IntArray): Boolean

    external fun trackerGetTrackingScore(tracker: Long): Float

    external fun trackerRelease(tracker: Long)
}

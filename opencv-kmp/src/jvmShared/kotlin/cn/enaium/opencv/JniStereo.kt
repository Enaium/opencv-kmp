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
 * JNI bridge for the stereo module.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniStereo_<name>`
 * function in jni/jni_stereo.cpp. Mat and StereoMatcher handles travel as
 * jlongs; rects travel as 4-element IntArrays.
 */

internal object JniStereo {

    init {
        // Trigger the canonical native-library load via Jni.
        Jni.lastError()
    }

    // factories

    external fun stereoBmCreate(numDisparities: Int, blockSize: Int): Long

    external fun stereoSgbmCreate(
        minDisparity: Int,
        numDisparities: Int,
        blockSize: Int,
        p1: Int,
        p2: Int,
        disp12MaxDiff: Int,
        preFilterCap: Int,
        uniquenessRatio: Int,
        speckleWindowSize: Int,
        speckleRange: Int,
        mode: Int,
    ): Long

    // StereoMatcher

    external fun stereoMatcherCompute(matcher: Long, left: Long, right: Long): Long
    external fun stereoMatcherClear(matcher: Long)
    external fun stereoMatcherEmpty(matcher: Long): Boolean
    external fun stereoMatcherSave(matcher: Long, filename: String)
    external fun stereoMatcherGetDefaultName(matcher: Long): String?

    external fun stereoMatcherGetMinDisparity(matcher: Long): Int
    external fun stereoMatcherSetMinDisparity(matcher: Long, value: Int)
    external fun stereoMatcherGetNumDisparities(matcher: Long): Int
    external fun stereoMatcherSetNumDisparities(matcher: Long, value: Int)
    external fun stereoMatcherGetBlockSize(matcher: Long): Int
    external fun stereoMatcherSetBlockSize(matcher: Long, value: Int)
    external fun stereoMatcherGetSpeckleWindowSize(matcher: Long): Int
    external fun stereoMatcherSetSpeckleWindowSize(matcher: Long, value: Int)
    external fun stereoMatcherGetSpeckleRange(matcher: Long): Int
    external fun stereoMatcherSetSpeckleRange(matcher: Long, value: Int)
    external fun stereoMatcherGetDisp12MaxDiff(matcher: Long): Int
    external fun stereoMatcherSetDisp12MaxDiff(matcher: Long, value: Int)
    external fun stereoMatcherRelease(matcher: Long)

    // StereoBM

    external fun stereoBmGetPreFilterType(matcher: Long): Int
    external fun stereoBmSetPreFilterType(matcher: Long, value: Int)
    external fun stereoBmGetPreFilterSize(matcher: Long): Int
    external fun stereoBmSetPreFilterSize(matcher: Long, value: Int)
    external fun stereoBmGetPreFilterCap(matcher: Long): Int
    external fun stereoBmSetPreFilterCap(matcher: Long, value: Int)
    external fun stereoBmGetTextureThreshold(matcher: Long): Int
    external fun stereoBmSetTextureThreshold(matcher: Long, value: Int)
    external fun stereoBmGetUniquenessRatio(matcher: Long): Int
    external fun stereoBmSetUniquenessRatio(matcher: Long, value: Int)
    external fun stereoBmGetSmallerBlockSize(matcher: Long): Int
    external fun stereoBmSetSmallerBlockSize(matcher: Long, value: Int)
    external fun stereoBmGetRoi1(matcher: Long): IntArray
    external fun stereoBmSetRoi1(matcher: Long, roi: IntArray)
    external fun stereoBmGetRoi2(matcher: Long): IntArray
    external fun stereoBmSetRoi2(matcher: Long, roi: IntArray)

    // StereoSGBM

    external fun stereoSgbmGetPreFilterCap(matcher: Long): Int
    external fun stereoSgbmSetPreFilterCap(matcher: Long, value: Int)
    external fun stereoSgbmGetUniquenessRatio(matcher: Long): Int
    external fun stereoSgbmSetUniquenessRatio(matcher: Long, value: Int)
    external fun stereoSgbmGetP1(matcher: Long): Int
    external fun stereoSgbmSetP1(matcher: Long, value: Int)
    external fun stereoSgbmGetP2(matcher: Long): Int
    external fun stereoSgbmSetP2(matcher: Long, value: Int)
    external fun stereoSgbmGetMode(matcher: Long): Int
    external fun stereoSgbmSetMode(matcher: Long, value: Int)

    // Stereo statics

    /** 5 handles [r1, r2, p1, p2, q]; fills [roi1Out]/[roi2Out] in place. */
    external fun stereoRectify(
        cm1: Long,
        dc1: Long,
        cm2: Long,
        dc2: Long,
        imageWidth: Int,
        imageHeight: Int,
        r: Long,
        t: Long,
        flags: Int,
        alpha: Double,
        newWidth: Int,
        newHeight: Int,
        roi1Out: IntArray,
        roi2Out: IntArray,
    ): LongArray?

    /** 3 longs [ok, h1, h2]. */
    external fun stereoRectifyUncalibrated(
        points1: Long,
        points2: Long,
        f: Long,
        imgWidth: Int,
        imgHeight: Int,
        threshold: Double,
    ): LongArray

    /** 5 handles [r1, r2, p1, p2, q]. */
    external fun stereoFisheyeRectify(
        k1: Long,
        d1: Long,
        k2: Long,
        d2: Long,
        imageWidth: Int,
        imageHeight: Int,
        r: Long,
        tvec: Long,
        flags: Int,
        newWidth: Int,
        newHeight: Int,
        balance: Double,
        fovScale: Double,
    ): LongArray?

    external fun stereoFilterSpeckles(img: Long, newVal: Double, maxSpeckleSize: Int, maxDiff: Double)
    external fun stereoFilterSpecklesBuf(img: Long, newVal: Double, maxSpeckleSize: Int, maxDiff: Double, buf: Long)
    external fun stereoGetValidDisparityROI(roi1: IntArray, roi2: IntArray, minDisparity: Int, numberOfDisparities: Int, blockSize: Int): IntArray
    external fun stereoValidateDisparity(disparity: Long, cost: Long, minDisparity: Int, numberOfDisparities: Int, disp12MaxDisp: Int)
    external fun stereoReprojectImageTo3D(disparity: Long, q: Long, handleMissingValues: Boolean, ddepth: Int): Long
}

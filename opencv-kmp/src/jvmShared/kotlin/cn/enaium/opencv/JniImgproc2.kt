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
 * JNI bridge for the imgproc2 slice (LineSegmentDetector / GeneralizedHough).
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniImgproc2_<name>`
 * function in jni/jni_imgproc2.cpp. Mat handles travel as jlong pointers;
 * multi-Mat outputs (detect) come back as fixed-order jlongArrays.
 *
 * No init block is needed: the main `Jni` object's init already loaded the
 * native library.
 */
internal object JniImgproc2 {

    // LineSegmentDetector

    external fun lsdCreate(
        refine: Int,
        scale: Double,
        sigmaScale: Double,
        quant: Double,
        angTh: Double,
        logEps: Double,
        densityTh: Double,
        nBins: Int,
    ): Long

    /** Returns [lines, width, prec, nfa] handles. */
    external fun lsdDetect(h: Long, image: Long): LongArray
    external fun lsdDrawSegments(h: Long, image: Long, lines: Long)
    external fun lsdCompareSegments(h: Long, width: Int, height: Int, lines1: Long, lines2: Long, image: Long): Int
    external fun lsdClear(h: Long)
    external fun lsdEmpty(h: Long): Boolean
    external fun lsdSave(h: Long, filename: String)
    external fun lsdGetDefaultName(h: Long): String
    external fun lsdRelease(h: Long)

    // GeneralizedHoughBallard

    external fun ghBallardCreate(): Long
    external fun ghBallardSetTemplate(h: Long, templ: Long, cx: Double, cy: Double)
    external fun ghBallardSetTemplateEdges(h: Long, edges: Long, dx: Long, dy: Long, cx: Double, cy: Double)

    /** Returns [positions, votes] handles. */
    external fun ghBallardDetect(h: Long, image: Long): LongArray

    /** Returns [positions, votes] handles. */
    external fun ghBallardDetectEdges(h: Long, edges: Long, dx: Long, dy: Long): LongArray

    external fun ghBallardSetCannyLowThresh(h: Long, v: Int)
    external fun ghBallardGetCannyLowThresh(h: Long): Int
    external fun ghBallardSetCannyHighThresh(h: Long, v: Int)
    external fun ghBallardGetCannyHighThresh(h: Long): Int
    external fun ghBallardSetMinDist(h: Long, v: Double)
    external fun ghBallardGetMinDist(h: Long): Double
    external fun ghBallardSetDp(h: Long, v: Double)
    external fun ghBallardGetDp(h: Long): Double
    external fun ghBallardSetMaxBufferSize(h: Long, v: Int)
    external fun ghBallardGetMaxBufferSize(h: Long): Int
    external fun ghBallardSetLevels(h: Long, v: Int)
    external fun ghBallardGetLevels(h: Long): Int
    external fun ghBallardSetVotesThreshold(h: Long, v: Int)
    external fun ghBallardGetVotesThreshold(h: Long): Int
    external fun ghBallardClear(h: Long)
    external fun ghBallardEmpty(h: Long): Boolean
    external fun ghBallardSave(h: Long, filename: String)
    external fun ghBallardGetDefaultName(h: Long): String
    external fun ghBallardRelease(h: Long)

    // GeneralizedHoughGuil

    external fun ghGuilCreate(): Long
    external fun ghGuilSetTemplate(h: Long, templ: Long, cx: Double, cy: Double)
    external fun ghGuilSetTemplateEdges(h: Long, edges: Long, dx: Long, dy: Long, cx: Double, cy: Double)

    /** Returns [positions, votes] handles. */
    external fun ghGuilDetect(h: Long, image: Long): LongArray

    /** Returns [positions, votes] handles. */
    external fun ghGuilDetectEdges(h: Long, edges: Long, dx: Long, dy: Long): LongArray

    external fun ghGuilSetCannyLowThresh(h: Long, v: Int)
    external fun ghGuilGetCannyLowThresh(h: Long): Int
    external fun ghGuilSetCannyHighThresh(h: Long, v: Int)
    external fun ghGuilGetCannyHighThresh(h: Long): Int
    external fun ghGuilSetMinDist(h: Long, v: Double)
    external fun ghGuilGetMinDist(h: Long): Double
    external fun ghGuilSetDp(h: Long, v: Double)
    external fun ghGuilGetDp(h: Long): Double
    external fun ghGuilSetMaxBufferSize(h: Long, v: Int)
    external fun ghGuilGetMaxBufferSize(h: Long): Int
    external fun ghGuilSetXi(h: Long, v: Double)
    external fun ghGuilGetXi(h: Long): Double
    external fun ghGuilSetLevels(h: Long, v: Int)
    external fun ghGuilGetLevels(h: Long): Int
    external fun ghGuilSetAngleEpsilon(h: Long, v: Double)
    external fun ghGuilGetAngleEpsilon(h: Long): Double
    external fun ghGuilSetMinAngle(h: Long, v: Double)
    external fun ghGuilGetMinAngle(h: Long): Double
    external fun ghGuilSetMaxAngle(h: Long, v: Double)
    external fun ghGuilGetMaxAngle(h: Long): Double
    external fun ghGuilSetAngleStep(h: Long, v: Double)
    external fun ghGuilGetAngleStep(h: Long): Double
    external fun ghGuilSetAngleThresh(h: Long, v: Int)
    external fun ghGuilGetAngleThresh(h: Long): Int
    external fun ghGuilSetMinScale(h: Long, v: Double)
    external fun ghGuilGetMinScale(h: Long): Double
    external fun ghGuilSetMaxScale(h: Long, v: Double)
    external fun ghGuilGetMaxScale(h: Long): Double
    external fun ghGuilSetScaleStep(h: Long, v: Double)
    external fun ghGuilGetScaleStep(h: Long): Double
    external fun ghGuilSetScaleThresh(h: Long, v: Int)
    external fun ghGuilGetScaleThresh(h: Long): Int
    external fun ghGuilSetPosThresh(h: Long, v: Int)
    external fun ghGuilGetPosThresh(h: Long): Int
    external fun ghGuilClear(h: Long)
    external fun ghGuilEmpty(h: Long): Boolean
    external fun ghGuilSave(h: Long, filename: String)
    external fun ghGuilGetDefaultName(h: Long): String
    external fun ghGuilRelease(h: Long)
}

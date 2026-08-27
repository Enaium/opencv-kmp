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
 * JNI bridge for the org.opencv.video surface.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniVideo_<name>`
 * function in jni/jni_video.cpp. Mat and algorithm handles travel as jlong
 * pointers; TermCriteria and Size are expanded into primitives.
 */
internal object JniVideo {

    // ---- video statics -------------------------------------------------

    external fun calcOpticalFlowFarneback(
        prev: Long,
        next: Long,
        flow: Long,
        pyrScale: Double,
        levels: Int,
        winSize: Int,
        iterations: Int,
        polyN: Int,
        polySigma: Double,
        flags: Int,
    )

    external fun calcOpticalFlowPyrLK(
        prevImg: Long,
        nextImg: Long,
        prevPts: Long,
        nextPts: Long,
        status: Long,
        err: Long,
        winW: Int,
        winH: Int,
        maxLevel: Int,
        tcType: Int,
        tcMaxCount: Int,
        tcEpsilon: Double,
        flags: Int,
        minEigThreshold: Double,
    )

    external fun computeECC(templateImage: Long, inputImage: Long, inputMask: Long): Double

    external fun findTransformECC(
        templateImage: Long,
        inputImage: Long,
        warpMatrix: Long,
        motionType: Int,
        tcType: Int,
        tcMaxCount: Int,
        tcEpsilon: Double,
        inputMask: Long,
        gaussFiltSize: Int,
    ): Double

    external fun findTransformECCWithMask(
        templateImage: Long,
        inputImage: Long,
        templateMask: Long,
        inputMask: Long,
        warpMatrix: Long,
        motionType: Int,
        tcType: Int,
        tcMaxCount: Int,
        tcEpsilon: Double,
        gaussFiltSize: Int,
    ): Double

    external fun findTransformECCMultiScale(
        reference: Long,
        sample: Long,
        warpMatrix: Long,
        motionType: Int,
        tcType: Int,
        tcMaxCount: Int,
        tcEpsilon: Double,
        itersPerLevel: Long,
        gaussFiltSize: Int,
        nlevels: Int,
        interpolation: Int,
        referenceMask: Long,
        sampleMask: Long,
    ): Double

    // ---- BackgroundSubtractorKNN ----------------------------------------

    external fun createBackgroundSubtractorKNN(
        history: Int,
        dist2Threshold: Double,
        detectShadows: Boolean,
    ): Long

    external fun backgroundSubtractorKNNRelease(handle: Long)
    external fun backgroundSubtractorKNNApply(handle: Long, image: Long, fgmask: Long, learningRate: Double)
    external fun backgroundSubtractorKNNApplyMask(
        handle: Long,
        image: Long,
        knownForegroundMask: Long,
        fgmask: Long,
        learningRate: Double,
    )

    external fun backgroundSubtractorKNNGetBackgroundImage(handle: Long): Long

    external fun backgroundSubtractorKNNGetHistory(handle: Long): Int
    external fun backgroundSubtractorKNNSetHistory(handle: Long, value: Int)
    external fun backgroundSubtractorKNNGetNSamples(handle: Long): Int
    external fun backgroundSubtractorKNNSetNSamples(handle: Long, value: Int)
    external fun backgroundSubtractorKNNGetDist2Threshold(handle: Long): Double
    external fun backgroundSubtractorKNNSetDist2Threshold(handle: Long, value: Double)
    external fun backgroundSubtractorKNNGetkNNSamples(handle: Long): Int
    external fun backgroundSubtractorKNNSetkNNSamples(handle: Long, value: Int)
    external fun backgroundSubtractorKNNGetDetectShadows(handle: Long): Boolean
    external fun backgroundSubtractorKNNSetDetectShadows(handle: Long, value: Boolean)
    external fun backgroundSubtractorKNNGetShadowValue(handle: Long): Int
    external fun backgroundSubtractorKNNSetShadowValue(handle: Long, value: Int)
    external fun backgroundSubtractorKNNGetShadowThreshold(handle: Long): Double
    external fun backgroundSubtractorKNNSetShadowThreshold(handle: Long, value: Double)

    external fun backgroundSubtractorKNNClear(handle: Long)
    external fun backgroundSubtractorKNNEmpty(handle: Long): Boolean
    external fun backgroundSubtractorKNNSave(handle: Long, filename: String)
    external fun backgroundSubtractorKNNGetDefaultName(handle: Long): String

    // ---- BackgroundSubtractorMOG2 ---------------------------------------

    external fun createBackgroundSubtractorMOG2(
        history: Int,
        varThreshold: Double,
        detectShadows: Boolean,
    ): Long

    external fun backgroundSubtractorMOG2Release(handle: Long)
    external fun backgroundSubtractorMOG2Apply(handle: Long, image: Long, fgmask: Long, learningRate: Double)
    external fun backgroundSubtractorMOG2ApplyMask(
        handle: Long,
        image: Long,
        knownForegroundMask: Long,
        fgmask: Long,
        learningRate: Double,
    )

    external fun backgroundSubtractorMOG2GetBackgroundImage(handle: Long): Long

    external fun backgroundSubtractorMOG2GetHistory(handle: Long): Int
    external fun backgroundSubtractorMOG2SetHistory(handle: Long, value: Int)
    external fun backgroundSubtractorMOG2GetNMixtures(handle: Long): Int
    external fun backgroundSubtractorMOG2SetNMixtures(handle: Long, value: Int)
    external fun backgroundSubtractorMOG2GetBackgroundRatio(handle: Long): Double
    external fun backgroundSubtractorMOG2SetBackgroundRatio(handle: Long, value: Double)
    external fun backgroundSubtractorMOG2GetVarThreshold(handle: Long): Double
    external fun backgroundSubtractorMOG2SetVarThreshold(handle: Long, value: Double)
    external fun backgroundSubtractorMOG2GetVarThresholdGen(handle: Long): Double
    external fun backgroundSubtractorMOG2SetVarThresholdGen(handle: Long, value: Double)
    external fun backgroundSubtractorMOG2GetVarInit(handle: Long): Double
    external fun backgroundSubtractorMOG2SetVarInit(handle: Long, value: Double)
    external fun backgroundSubtractorMOG2GetVarMin(handle: Long): Double
    external fun backgroundSubtractorMOG2SetVarMin(handle: Long, value: Double)
    external fun backgroundSubtractorMOG2GetVarMax(handle: Long): Double
    external fun backgroundSubtractorMOG2SetVarMax(handle: Long, value: Double)
    external fun backgroundSubtractorMOG2GetComplexityReductionThreshold(handle: Long): Double
    external fun backgroundSubtractorMOG2SetComplexityReductionThreshold(handle: Long, value: Double)
    external fun backgroundSubtractorMOG2GetDetectShadows(handle: Long): Boolean
    external fun backgroundSubtractorMOG2SetDetectShadows(handle: Long, value: Boolean)
    external fun backgroundSubtractorMOG2GetShadowValue(handle: Long): Int
    external fun backgroundSubtractorMOG2SetShadowValue(handle: Long, value: Int)
    external fun backgroundSubtractorMOG2GetShadowThreshold(handle: Long): Double
    external fun backgroundSubtractorMOG2SetShadowThreshold(handle: Long, value: Double)

    external fun backgroundSubtractorMOG2Clear(handle: Long)
    external fun backgroundSubtractorMOG2Empty(handle: Long): Boolean
    external fun backgroundSubtractorMOG2Save(handle: Long, filename: String)
    external fun backgroundSubtractorMOG2GetDefaultName(handle: Long): String

    // ---- FarnebackOpticalFlow -------------------------------------------

    external fun createFarnebackOpticalFlow(
        numLevels: Int,
        pyrScale: Double,
        fastPyramids: Boolean,
        winSize: Int,
        numIters: Int,
        polyN: Int,
        polySigma: Double,
        flags: Int,
    ): Long

    external fun farnebackOpticalFlowRelease(handle: Long)
    external fun farnebackOpticalFlowCalc(handle: Long, i0: Long, i1: Long, flow: Long)
    external fun farnebackOpticalFlowCollectGarbage(handle: Long)

    external fun farnebackOpticalFlowGetNumLevels(handle: Long): Int
    external fun farnebackOpticalFlowSetNumLevels(handle: Long, value: Int)
    external fun farnebackOpticalFlowGetPyrScale(handle: Long): Double
    external fun farnebackOpticalFlowSetPyrScale(handle: Long, value: Double)
    external fun farnebackOpticalFlowGetFastPyramids(handle: Long): Boolean
    external fun farnebackOpticalFlowSetFastPyramids(handle: Long, value: Boolean)
    external fun farnebackOpticalFlowGetWinSize(handle: Long): Int
    external fun farnebackOpticalFlowSetWinSize(handle: Long, value: Int)
    external fun farnebackOpticalFlowGetNumIters(handle: Long): Int
    external fun farnebackOpticalFlowSetNumIters(handle: Long, value: Int)
    external fun farnebackOpticalFlowGetPolyN(handle: Long): Int
    external fun farnebackOpticalFlowSetPolyN(handle: Long, value: Int)
    external fun farnebackOpticalFlowGetPolySigma(handle: Long): Double
    external fun farnebackOpticalFlowSetPolySigma(handle: Long, value: Double)
    external fun farnebackOpticalFlowGetFlags(handle: Long): Int
    external fun farnebackOpticalFlowSetFlags(handle: Long, value: Int)

    external fun farnebackOpticalFlowClear(handle: Long)
    external fun farnebackOpticalFlowEmpty(handle: Long): Boolean
    external fun farnebackOpticalFlowSave(handle: Long, filename: String)
    external fun farnebackOpticalFlowGetDefaultName(handle: Long): String

    // ---- DISOpticalFlow -------------------------------------------------

    external fun createDisOpticalFlow(preset: Int): Long

    external fun disOpticalFlowRelease(handle: Long)
    external fun disOpticalFlowCalc(handle: Long, i0: Long, i1: Long, flow: Long)
    external fun disOpticalFlowCollectGarbage(handle: Long)

    external fun disOpticalFlowGetFinestScale(handle: Long): Int
    external fun disOpticalFlowSetFinestScale(handle: Long, value: Int)
    external fun disOpticalFlowGetCoarsestScale(handle: Long): Int
    external fun disOpticalFlowSetCoarsestScale(handle: Long, value: Int)
    external fun disOpticalFlowGetPatchSize(handle: Long): Int
    external fun disOpticalFlowSetPatchSize(handle: Long, value: Int)
    external fun disOpticalFlowGetPatchStride(handle: Long): Int
    external fun disOpticalFlowSetPatchStride(handle: Long, value: Int)
    external fun disOpticalFlowGetGradientDescentIterations(handle: Long): Int
    external fun disOpticalFlowSetGradientDescentIterations(handle: Long, value: Int)
    external fun disOpticalFlowGetVariationalRefinementIterations(handle: Long): Int
    external fun disOpticalFlowSetVariationalRefinementIterations(handle: Long, value: Int)
    external fun disOpticalFlowGetVariationalRefinementAlpha(handle: Long): Float
    external fun disOpticalFlowSetVariationalRefinementAlpha(handle: Long, value: Float)
    external fun disOpticalFlowGetVariationalRefinementDelta(handle: Long): Float
    external fun disOpticalFlowSetVariationalRefinementDelta(handle: Long, value: Float)
    external fun disOpticalFlowGetVariationalRefinementGamma(handle: Long): Float
    external fun disOpticalFlowSetVariationalRefinementGamma(handle: Long, value: Float)
    external fun disOpticalFlowGetVariationalRefinementEpsilon(handle: Long): Float
    external fun disOpticalFlowSetVariationalRefinementEpsilon(handle: Long, value: Float)
    external fun disOpticalFlowGetUseMeanNormalization(handle: Long): Boolean
    external fun disOpticalFlowSetUseMeanNormalization(handle: Long, value: Boolean)
    external fun disOpticalFlowGetUseSpatialPropagation(handle: Long): Boolean
    external fun disOpticalFlowSetUseSpatialPropagation(handle: Long, value: Boolean)

    external fun disOpticalFlowClear(handle: Long)
    external fun disOpticalFlowEmpty(handle: Long): Boolean
    external fun disOpticalFlowSave(handle: Long, filename: String)
    external fun disOpticalFlowGetDefaultName(handle: Long): String

    // ---- SparsePyrLKOpticalFlow -----------------------------------------

    external fun createSparsePyrLKOpticalFlow(
        winW: Int,
        winH: Int,
        maxLevel: Int,
        tcType: Int,
        tcMaxCount: Int,
        tcEpsilon: Double,
        flags: Int,
        minEigThreshold: Double,
    ): Long

    external fun sparsePyrLKOpticalFlowRelease(handle: Long)
    external fun sparsePyrLKOpticalFlowCalc(
        handle: Long,
        prevImg: Long,
        nextImg: Long,
        prevPts: Long,
        nextPts: Long,
        status: Long,
        err: Long,
    )

    external fun sparsePyrLKOpticalFlowGetWinW(handle: Long): Int
    external fun sparsePyrLKOpticalFlowGetWinH(handle: Long): Int
    external fun sparsePyrLKOpticalFlowSetWinSize(handle: Long, winW: Int, winH: Int)
    external fun sparsePyrLKOpticalFlowGetMaxLevel(handle: Long): Int
    external fun sparsePyrLKOpticalFlowSetMaxLevel(handle: Long, value: Int)

    /** Returns [type, maxCount, epsilon]. */
    external fun sparsePyrLKOpticalFlowGetTermCriteria(handle: Long): DoubleArray

    external fun sparsePyrLKOpticalFlowSetTermCriteria(
        handle: Long,
        tcType: Int,
        tcMaxCount: Int,
        tcEpsilon: Double,
    )

    external fun sparsePyrLKOpticalFlowGetFlags(handle: Long): Int
    external fun sparsePyrLKOpticalFlowSetFlags(handle: Long, value: Int)
    external fun sparsePyrLKOpticalFlowGetMinEigThreshold(handle: Long): Double
    external fun sparsePyrLKOpticalFlowSetMinEigThreshold(handle: Long, value: Double)

    external fun sparsePyrLKOpticalFlowClear(handle: Long)
    external fun sparsePyrLKOpticalFlowEmpty(handle: Long): Boolean
    external fun sparsePyrLKOpticalFlowSave(handle: Long, filename: String)
    external fun sparsePyrLKOpticalFlowGetDefaultName(handle: Long): String

    // ---- VariationalRefinement ------------------------------------------

    external fun createVariationalRefinement(): Long

    external fun variationalRefinementRelease(handle: Long)
    external fun variationalRefinementCalc(handle: Long, i0: Long, i1: Long, flow: Long)
    external fun variationalRefinementCalcUV(handle: Long, i0: Long, i1: Long, flowU: Long, flowV: Long)
    external fun variationalRefinementCollectGarbage(handle: Long)

    external fun variationalRefinementGetFixedPointIterations(handle: Long): Int
    external fun variationalRefinementSetFixedPointIterations(handle: Long, value: Int)
    external fun variationalRefinementGetSorIterations(handle: Long): Int
    external fun variationalRefinementSetSorIterations(handle: Long, value: Int)
    external fun variationalRefinementGetOmega(handle: Long): Float
    external fun variationalRefinementSetOmega(handle: Long, value: Float)
    external fun variationalRefinementGetAlpha(handle: Long): Float
    external fun variationalRefinementSetAlpha(handle: Long, value: Float)
    external fun variationalRefinementGetDelta(handle: Long): Float
    external fun variationalRefinementSetDelta(handle: Long, value: Float)
    external fun variationalRefinementGetGamma(handle: Long): Float
    external fun variationalRefinementSetGamma(handle: Long, value: Float)
    external fun variationalRefinementGetEpsilon(handle: Long): Float
    external fun variationalRefinementSetEpsilon(handle: Long, value: Float)

    external fun variationalRefinementClear(handle: Long)
    external fun variationalRefinementEmpty(handle: Long): Boolean
    external fun variationalRefinementSave(handle: Long, filename: String)
    external fun variationalRefinementGetDefaultName(handle: Long): String

    // ---- KalmanFilter ---------------------------------------------------

    external fun kalmanFilterCreate(
        dynamParams: Int,
        measureParams: Int,
        controlParams: Int,
        type: Int,
    ): Long

    external fun kalmanFilterRelease(handle: Long)
    external fun kalmanFilterPredict(handle: Long, control: Long): Long
    external fun kalmanFilterCorrect(handle: Long, measurement: Long): Long

    external fun kalmanFilterGetStatePre(handle: Long): Long
    external fun kalmanFilterSetStatePre(handle: Long, mat: Long)
    external fun kalmanFilterGetStatePost(handle: Long): Long
    external fun kalmanFilterSetStatePost(handle: Long, mat: Long)
    external fun kalmanFilterGetTransitionMatrix(handle: Long): Long
    external fun kalmanFilterSetTransitionMatrix(handle: Long, mat: Long)
    external fun kalmanFilterGetControlMatrix(handle: Long): Long
    external fun kalmanFilterSetControlMatrix(handle: Long, mat: Long)
    external fun kalmanFilterGetMeasurementMatrix(handle: Long): Long
    external fun kalmanFilterSetMeasurementMatrix(handle: Long, mat: Long)
    external fun kalmanFilterGetProcessNoiseCov(handle: Long): Long
    external fun kalmanFilterSetProcessNoiseCov(handle: Long, mat: Long)
    external fun kalmanFilterGetMeasurementNoiseCov(handle: Long): Long
    external fun kalmanFilterSetMeasurementNoiseCov(handle: Long, mat: Long)
    external fun kalmanFilterGetErrorCovPre(handle: Long): Long
    external fun kalmanFilterSetErrorCovPre(handle: Long, mat: Long)
    external fun kalmanFilterGetGain(handle: Long): Long
    external fun kalmanFilterSetGain(handle: Long, mat: Long)
    external fun kalmanFilterGetErrorCovPost(handle: Long): Long
    external fun kalmanFilterSetErrorCovPost(handle: Long, mat: Long)
}

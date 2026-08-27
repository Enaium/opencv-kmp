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

/** Message of the last native OpenCV error on this thread, or null. */
private fun lastNativeError(): String? = Jni.lastError()


// =========================================================================
// video statics
// =========================================================================

actual fun calcOpticalFlowFarneback(
    prev: Mat,
    next: Mat,
    flow: Mat,
    pyrScale: Double,
    levels: Int,
    winSize: Int,
    iterations: Int,
    polyN: Int,
    polySigma: Double,
    flags: Int,
) = JniVideo.calcOpticalFlowFarneback(
    handleOf(prev), handleOf(next), handleOf(flow),
    pyrScale, levels, winSize, iterations, polyN, polySigma, flags,
)

actual fun calcOpticalFlowPyrLK(
    prevImg: Mat,
    nextImg: Mat,
    prevPts: MatOfPoint2f,
    nextPts: MatOfPoint2f,
    status: MatOfByte,
    err: MatOfFloat,
    winSize: Size,
    maxLevel: Int,
    criteria: TermCriteria,
    flags: Int,
    minEigThreshold: Double,
) = JniVideo.calcOpticalFlowPyrLK(
    handleOf(prevImg), handleOf(nextImg), handleOf(prevPts.mat), handleOf(nextPts.mat),
    handleOf(status.mat), handleOf(err.mat),
    winSize.width, winSize.height, maxLevel, criteria.type, criteria.maxCount,
    criteria.epsilon, flags, minEigThreshold,
)

actual fun computeECC(templateImage: Mat, inputImage: Mat, inputMask: Mat?): Double =
    JniVideo.computeECC(handleOf(templateImage), handleOf(inputImage), inputMask?.let(::handleOf) ?: 0L)

actual fun findTransformECC(
    templateImage: Mat,
    inputImage: Mat,
    warpMatrix: Mat,
    motionType: Int,
    criteria: TermCriteria,
    inputMask: Mat?,
    gaussFiltSize: Int,
): Double = JniVideo.findTransformECC(
    handleOf(templateImage), handleOf(inputImage), handleOf(warpMatrix), motionType,
    criteria.type, criteria.maxCount, criteria.epsilon, inputMask?.let(::handleOf) ?: 0L,
    gaussFiltSize,
)

actual fun findTransformECCWithMask(
    templateImage: Mat,
    inputImage: Mat,
    templateMask: Mat,
    inputMask: Mat,
    warpMatrix: Mat,
    motionType: Int,
    criteria: TermCriteria,
    gaussFiltSize: Int,
): Double = JniVideo.findTransformECCWithMask(
    handleOf(templateImage), handleOf(inputImage), handleOf(templateMask),
    handleOf(inputMask), handleOf(warpMatrix), motionType, criteria.type,
    criteria.maxCount, criteria.epsilon, gaussFiltSize,
)

actual fun findTransformECCMultiScale(
    reference: Mat,
    sample: Mat,
    warpMatrix: Mat,
    eccParams: ECCParameters,
    referenceMask: Mat?,
    sampleMask: Mat?,
): Double = JniVideo.findTransformECCMultiScale(
    handleOf(reference), handleOf(sample), handleOf(warpMatrix), eccParams.motionType,
    eccParams.criteria.type, eccParams.criteria.maxCount, eccParams.criteria.epsilon,
    eccParams.itersPerLevel?.let { handleOf(it.mat) } ?: 0L,
    eccParams.gaussFiltSize, eccParams.nlevels, eccParams.interpolation,
    referenceMask?.let(::handleOf) ?: 0L, sampleMask?.let(::handleOf) ?: 0L,
)

// =========================================================================
// background subtraction
// =========================================================================

internal class JvmBackgroundSubtractorKNN(private var handle: Long) : BackgroundSubtractorKNN {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("BackgroundSubtractorKNN is closed")

    override fun apply(image: Mat, fgmask: Mat, learningRate: Double) {
        JniVideo.backgroundSubtractorKNNApply(check(), handleOf(image), handleOf(fgmask), learningRate)
    }

    override fun apply(image: Mat, knownForegroundMask: Mat, fgmask: Mat, learningRate: Double) {
        JniVideo.backgroundSubtractorKNNApplyMask(
            check(), handleOf(image), handleOf(knownForegroundMask), handleOf(fgmask), learningRate,
        )
    }

    override fun getBackgroundImage(): Mat =
        jvmMat(JniVideo.backgroundSubtractorKNNGetBackgroundImage(check()), "getBackgroundImage")

    override var history: Int
        get() = JniVideo.backgroundSubtractorKNNGetHistory(check())
        set(value) = JniVideo.backgroundSubtractorKNNSetHistory(check(), value)

    override var nSamples: Int
        get() = JniVideo.backgroundSubtractorKNNGetNSamples(check())
        set(value) = JniVideo.backgroundSubtractorKNNSetNSamples(check(), value)

    override var dist2Threshold: Double
        get() = JniVideo.backgroundSubtractorKNNGetDist2Threshold(check())
        set(value) = JniVideo.backgroundSubtractorKNNSetDist2Threshold(check(), value)

    override var kNNsamples: Int
        get() = JniVideo.backgroundSubtractorKNNGetkNNSamples(check())
        set(value) = JniVideo.backgroundSubtractorKNNSetkNNSamples(check(), value)

    override var detectShadows: Boolean
        get() = JniVideo.backgroundSubtractorKNNGetDetectShadows(check())
        set(value) = JniVideo.backgroundSubtractorKNNSetDetectShadows(check(), value)

    override var shadowValue: Int
        get() = JniVideo.backgroundSubtractorKNNGetShadowValue(check())
        set(value) = JniVideo.backgroundSubtractorKNNSetShadowValue(check(), value)

    override var shadowThreshold: Double
        get() = JniVideo.backgroundSubtractorKNNGetShadowThreshold(check())
        set(value) = JniVideo.backgroundSubtractorKNNSetShadowThreshold(check(), value)

    override fun clear() = JniVideo.backgroundSubtractorKNNClear(check())

    override fun empty(): Boolean = JniVideo.backgroundSubtractorKNNEmpty(check())

    override fun save(filename: String) = JniVideo.backgroundSubtractorKNNSave(check(), filename)

    override fun getDefaultName(): String = JniVideo.backgroundSubtractorKNNGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo.backgroundSubtractorKNNRelease(h)
        }
    }
}

internal class JvmBackgroundSubtractorMOG2(private var handle: Long) : BackgroundSubtractorMOG2 {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("BackgroundSubtractorMOG2 is closed")

    override fun apply(image: Mat, fgmask: Mat, learningRate: Double) {
        JniVideo.backgroundSubtractorMOG2Apply(check(), handleOf(image), handleOf(fgmask), learningRate)
    }

    override fun apply(image: Mat, knownForegroundMask: Mat, fgmask: Mat, learningRate: Double) {
        JniVideo.backgroundSubtractorMOG2ApplyMask(
            check(), handleOf(image), handleOf(knownForegroundMask), handleOf(fgmask), learningRate,
        )
    }

    override fun getBackgroundImage(): Mat =
        jvmMat(JniVideo.backgroundSubtractorMOG2GetBackgroundImage(check()), "getBackgroundImage")

    override var history: Int
        get() = JniVideo.backgroundSubtractorMOG2GetHistory(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetHistory(check(), value)

    override var nMixtures: Int
        get() = JniVideo.backgroundSubtractorMOG2GetNMixtures(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetNMixtures(check(), value)

    override var backgroundRatio: Double
        get() = JniVideo.backgroundSubtractorMOG2GetBackgroundRatio(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetBackgroundRatio(check(), value)

    override var varThreshold: Double
        get() = JniVideo.backgroundSubtractorMOG2GetVarThreshold(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetVarThreshold(check(), value)

    override var varThresholdGen: Double
        get() = JniVideo.backgroundSubtractorMOG2GetVarThresholdGen(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetVarThresholdGen(check(), value)

    override var varInit: Double
        get() = JniVideo.backgroundSubtractorMOG2GetVarInit(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetVarInit(check(), value)

    override var varMin: Double
        get() = JniVideo.backgroundSubtractorMOG2GetVarMin(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetVarMin(check(), value)

    override var varMax: Double
        get() = JniVideo.backgroundSubtractorMOG2GetVarMax(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetVarMax(check(), value)

    override var complexityReductionThreshold: Double
        get() = JniVideo.backgroundSubtractorMOG2GetComplexityReductionThreshold(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetComplexityReductionThreshold(check(), value)

    override var detectShadows: Boolean
        get() = JniVideo.backgroundSubtractorMOG2GetDetectShadows(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetDetectShadows(check(), value)

    override var shadowValue: Int
        get() = JniVideo.backgroundSubtractorMOG2GetShadowValue(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetShadowValue(check(), value)

    override var shadowThreshold: Double
        get() = JniVideo.backgroundSubtractorMOG2GetShadowThreshold(check())
        set(value) = JniVideo.backgroundSubtractorMOG2SetShadowThreshold(check(), value)

    override fun clear() = JniVideo.backgroundSubtractorMOG2Clear(check())

    override fun empty(): Boolean = JniVideo.backgroundSubtractorMOG2Empty(check())

    override fun save(filename: String) = JniVideo.backgroundSubtractorMOG2Save(check(), filename)

    override fun getDefaultName(): String = JniVideo.backgroundSubtractorMOG2GetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo.backgroundSubtractorMOG2Release(h)
        }
    }
}

actual fun createBackgroundSubtractorKNN(
    history: Int,
    dist2Threshold: Double,
    detectShadows: Boolean,
): BackgroundSubtractorKNN =
    JvmBackgroundSubtractorKNN(JniVideo.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows))

actual fun createBackgroundSubtractorMOG2(
    history: Int,
    varThreshold: Double,
    detectShadows: Boolean,
): BackgroundSubtractorMOG2 =
    JvmBackgroundSubtractorMOG2(JniVideo.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows))

// =========================================================================
// optical flow
// =========================================================================

internal class JvmFarnebackOpticalFlow(private var handle: Long) : FarnebackOpticalFlow {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("FarnebackOpticalFlow is closed")

    override fun calc(i0: Mat, i1: Mat, flow: Mat) {
        JniVideo.farnebackOpticalFlowCalc(check(), handleOf(i0), handleOf(i1), handleOf(flow))
    }

    override fun collectGarbage() = JniVideo.farnebackOpticalFlowCollectGarbage(check())

    override var numLevels: Int
        get() = JniVideo.farnebackOpticalFlowGetNumLevels(check())
        set(value) = JniVideo.farnebackOpticalFlowSetNumLevels(check(), value)

    override var pyrScale: Double
        get() = JniVideo.farnebackOpticalFlowGetPyrScale(check())
        set(value) = JniVideo.farnebackOpticalFlowSetPyrScale(check(), value)

    override var fastPyramids: Boolean
        get() = JniVideo.farnebackOpticalFlowGetFastPyramids(check())
        set(value) = JniVideo.farnebackOpticalFlowSetFastPyramids(check(), value)

    override var winSize: Int
        get() = JniVideo.farnebackOpticalFlowGetWinSize(check())
        set(value) = JniVideo.farnebackOpticalFlowSetWinSize(check(), value)

    override var numIters: Int
        get() = JniVideo.farnebackOpticalFlowGetNumIters(check())
        set(value) = JniVideo.farnebackOpticalFlowSetNumIters(check(), value)

    override var polyN: Int
        get() = JniVideo.farnebackOpticalFlowGetPolyN(check())
        set(value) = JniVideo.farnebackOpticalFlowSetPolyN(check(), value)

    override var polySigma: Double
        get() = JniVideo.farnebackOpticalFlowGetPolySigma(check())
        set(value) = JniVideo.farnebackOpticalFlowSetPolySigma(check(), value)

    override var flags: Int
        get() = JniVideo.farnebackOpticalFlowGetFlags(check())
        set(value) = JniVideo.farnebackOpticalFlowSetFlags(check(), value)

    override fun clear() = JniVideo.farnebackOpticalFlowClear(check())

    override fun empty(): Boolean = JniVideo.farnebackOpticalFlowEmpty(check())

    override fun save(filename: String) = JniVideo.farnebackOpticalFlowSave(check(), filename)

    override fun getDefaultName(): String = JniVideo.farnebackOpticalFlowGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo.farnebackOpticalFlowRelease(h)
        }
    }
}

internal class JvmDisOpticalFlow(private var handle: Long) : DISOpticalFlow {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("DISOpticalFlow is closed")

    override fun calc(i0: Mat, i1: Mat, flow: Mat) {
        JniVideo.disOpticalFlowCalc(check(), handleOf(i0), handleOf(i1), handleOf(flow))
    }

    override fun collectGarbage() = JniVideo.disOpticalFlowCollectGarbage(check())

    override var finestScale: Int
        get() = JniVideo.disOpticalFlowGetFinestScale(check())
        set(value) = JniVideo.disOpticalFlowSetFinestScale(check(), value)

    override var coarsestScale: Int
        get() = JniVideo.disOpticalFlowGetCoarsestScale(check())
        set(value) = JniVideo.disOpticalFlowSetCoarsestScale(check(), value)

    override var patchSize: Int
        get() = JniVideo.disOpticalFlowGetPatchSize(check())
        set(value) = JniVideo.disOpticalFlowSetPatchSize(check(), value)

    override var patchStride: Int
        get() = JniVideo.disOpticalFlowGetPatchStride(check())
        set(value) = JniVideo.disOpticalFlowSetPatchStride(check(), value)

    override var gradientDescentIterations: Int
        get() = JniVideo.disOpticalFlowGetGradientDescentIterations(check())
        set(value) = JniVideo.disOpticalFlowSetGradientDescentIterations(check(), value)

    override var variationalRefinementIterations: Int
        get() = JniVideo.disOpticalFlowGetVariationalRefinementIterations(check())
        set(value) = JniVideo.disOpticalFlowSetVariationalRefinementIterations(check(), value)

    override var variationalRefinementAlpha: Float
        get() = JniVideo.disOpticalFlowGetVariationalRefinementAlpha(check())
        set(value) = JniVideo.disOpticalFlowSetVariationalRefinementAlpha(check(), value)

    override var variationalRefinementDelta: Float
        get() = JniVideo.disOpticalFlowGetVariationalRefinementDelta(check())
        set(value) = JniVideo.disOpticalFlowSetVariationalRefinementDelta(check(), value)

    override var variationalRefinementGamma: Float
        get() = JniVideo.disOpticalFlowGetVariationalRefinementGamma(check())
        set(value) = JniVideo.disOpticalFlowSetVariationalRefinementGamma(check(), value)

    override var variationalRefinementEpsilon: Float
        get() = JniVideo.disOpticalFlowGetVariationalRefinementEpsilon(check())
        set(value) = JniVideo.disOpticalFlowSetVariationalRefinementEpsilon(check(), value)

    override var useMeanNormalization: Boolean
        get() = JniVideo.disOpticalFlowGetUseMeanNormalization(check())
        set(value) = JniVideo.disOpticalFlowSetUseMeanNormalization(check(), value)

    override var useSpatialPropagation: Boolean
        get() = JniVideo.disOpticalFlowGetUseSpatialPropagation(check())
        set(value) = JniVideo.disOpticalFlowSetUseSpatialPropagation(check(), value)

    override fun clear() = JniVideo.disOpticalFlowClear(check())

    override fun empty(): Boolean = JniVideo.disOpticalFlowEmpty(check())

    override fun save(filename: String) = JniVideo.disOpticalFlowSave(check(), filename)

    override fun getDefaultName(): String = JniVideo.disOpticalFlowGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo.disOpticalFlowRelease(h)
        }
    }
}

internal class JvmSparsePyrLKOpticalFlow(private var handle: Long) : SparsePyrLKOpticalFlow {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("SparsePyrLKOpticalFlow is closed")

    override fun calc(prevImg: Mat, nextImg: Mat, prevPts: Mat, nextPts: Mat, status: Mat, err: Mat?) {
        JniVideo.sparsePyrLKOpticalFlowCalc(
            check(), handleOf(prevImg), handleOf(nextImg), handleOf(prevPts), handleOf(nextPts),
            handleOf(status), err?.let(::handleOf) ?: 0L,
        )
    }

    override var winSize: Size
        get() = Size(JniVideo.sparsePyrLKOpticalFlowGetWinW(check()), JniVideo.sparsePyrLKOpticalFlowGetWinH(check()))
        set(value) = JniVideo.sparsePyrLKOpticalFlowSetWinSize(check(), value.width, value.height)

    override var maxLevel: Int
        get() = JniVideo.sparsePyrLKOpticalFlowGetMaxLevel(check())
        set(value) = JniVideo.sparsePyrLKOpticalFlowSetMaxLevel(check(), value)

    override var termCriteria: TermCriteria
        get() {
            val v = JniVideo.sparsePyrLKOpticalFlowGetTermCriteria(check())
            return TermCriteria(v[0].toInt(), v[1].toInt(), v[2])
        }
        set(value) = JniVideo.sparsePyrLKOpticalFlowSetTermCriteria(check(), value.type, value.maxCount, value.epsilon)

    override var flags: Int
        get() = JniVideo.sparsePyrLKOpticalFlowGetFlags(check())
        set(value) = JniVideo.sparsePyrLKOpticalFlowSetFlags(check(), value)

    override var minEigThreshold: Double
        get() = JniVideo.sparsePyrLKOpticalFlowGetMinEigThreshold(check())
        set(value) = JniVideo.sparsePyrLKOpticalFlowSetMinEigThreshold(check(), value)

    override fun clear() = JniVideo.sparsePyrLKOpticalFlowClear(check())

    override fun empty(): Boolean = JniVideo.sparsePyrLKOpticalFlowEmpty(check())

    override fun save(filename: String) = JniVideo.sparsePyrLKOpticalFlowSave(check(), filename)

    override fun getDefaultName(): String = JniVideo.sparsePyrLKOpticalFlowGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo.sparsePyrLKOpticalFlowRelease(h)
        }
    }
}

internal class JvmVariationalRefinement(private var handle: Long) : VariationalRefinement {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("VariationalRefinement is closed")

    override fun calc(i0: Mat, i1: Mat, flow: Mat) {
        JniVideo.variationalRefinementCalc(check(), handleOf(i0), handleOf(i1), handleOf(flow))
    }

    override fun calcUV(i0: Mat, i1: Mat, flowU: Mat, flowV: Mat) {
        JniVideo.variationalRefinementCalcUV(check(), handleOf(i0), handleOf(i1), handleOf(flowU), handleOf(flowV))
    }

    override fun collectGarbage() = JniVideo.variationalRefinementCollectGarbage(check())

    override var fixedPointIterations: Int
        get() = JniVideo.variationalRefinementGetFixedPointIterations(check())
        set(value) = JniVideo.variationalRefinementSetFixedPointIterations(check(), value)

    override var sorIterations: Int
        get() = JniVideo.variationalRefinementGetSorIterations(check())
        set(value) = JniVideo.variationalRefinementSetSorIterations(check(), value)

    override var omega: Float
        get() = JniVideo.variationalRefinementGetOmega(check())
        set(value) = JniVideo.variationalRefinementSetOmega(check(), value)

    override var alpha: Float
        get() = JniVideo.variationalRefinementGetAlpha(check())
        set(value) = JniVideo.variationalRefinementSetAlpha(check(), value)

    override var delta: Float
        get() = JniVideo.variationalRefinementGetDelta(check())
        set(value) = JniVideo.variationalRefinementSetDelta(check(), value)

    override var gamma: Float
        get() = JniVideo.variationalRefinementGetGamma(check())
        set(value) = JniVideo.variationalRefinementSetGamma(check(), value)

    override var epsilon: Float
        get() = JniVideo.variationalRefinementGetEpsilon(check())
        set(value) = JniVideo.variationalRefinementSetEpsilon(check(), value)

    override fun clear() = JniVideo.variationalRefinementClear(check())

    override fun empty(): Boolean = JniVideo.variationalRefinementEmpty(check())

    override fun save(filename: String) = JniVideo.variationalRefinementSave(check(), filename)

    override fun getDefaultName(): String = JniVideo.variationalRefinementGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo.variationalRefinementRelease(h)
        }
    }
}

actual fun createFarnebackOpticalFlow(
    numLevels: Int,
    pyrScale: Double,
    fastPyramids: Boolean,
    winSize: Int,
    numIters: Int,
    polyN: Int,
    polySigma: Double,
    flags: Int,
): FarnebackOpticalFlow = JvmFarnebackOpticalFlow(
    JniVideo.createFarnebackOpticalFlow(numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, flags),
)

actual fun createDisOpticalFlow(preset: Int): DISOpticalFlow =
    JvmDisOpticalFlow(JniVideo.createDisOpticalFlow(preset))

actual fun createVariationalRefinement(): VariationalRefinement =
    JvmVariationalRefinement(JniVideo.createVariationalRefinement())

actual fun createSparsePyrLKOpticalFlow(
    winSize: Size,
    maxLevel: Int,
    criteria: TermCriteria,
    flags: Int,
    minEigThreshold: Double,
): SparsePyrLKOpticalFlow = JvmSparsePyrLKOpticalFlow(
    JniVideo.createSparsePyrLKOpticalFlow(
        winSize.width, winSize.height, maxLevel, criteria.type, criteria.maxCount,
        criteria.epsilon, flags, minEigThreshold,
    ),
)

// =========================================================================
// Kalman filter
// =========================================================================

internal class JvmKalmanFilter(private var handle: Long) : KalmanFilter {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("KalmanFilter is closed")

    override fun predict(control: Mat?): Mat =
        jvmMat(JniVideo.kalmanFilterPredict(check(), control?.let(::handleOf) ?: 0L), "predict")

    override fun correct(measurement: Mat): Mat =
        jvmMat(JniVideo.kalmanFilterCorrect(check(), handleOf(measurement)), "correct")

    override var statePre: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetStatePre(check()), "statePre")
        set(value) = JniVideo.kalmanFilterSetStatePre(check(), handleOf(value))

    override var statePost: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetStatePost(check()), "statePost")
        set(value) = JniVideo.kalmanFilterSetStatePost(check(), handleOf(value))

    override var transitionMatrix: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetTransitionMatrix(check()), "transitionMatrix")
        set(value) = JniVideo.kalmanFilterSetTransitionMatrix(check(), handleOf(value))

    override var controlMatrix: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetControlMatrix(check()), "controlMatrix")
        set(value) = JniVideo.kalmanFilterSetControlMatrix(check(), handleOf(value))

    override var measurementMatrix: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetMeasurementMatrix(check()), "measurementMatrix")
        set(value) = JniVideo.kalmanFilterSetMeasurementMatrix(check(), handleOf(value))

    override var processNoiseCov: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetProcessNoiseCov(check()), "processNoiseCov")
        set(value) = JniVideo.kalmanFilterSetProcessNoiseCov(check(), handleOf(value))

    override var measurementNoiseCov: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetMeasurementNoiseCov(check()), "measurementNoiseCov")
        set(value) = JniVideo.kalmanFilterSetMeasurementNoiseCov(check(), handleOf(value))

    override var errorCovPre: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetErrorCovPre(check()), "errorCovPre")
        set(value) = JniVideo.kalmanFilterSetErrorCovPre(check(), handleOf(value))

    override var gain: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetGain(check()), "gain")
        set(value) = JniVideo.kalmanFilterSetGain(check(), handleOf(value))

    override var errorCovPost: Mat
        get() = jvmMat(JniVideo.kalmanFilterGetErrorCovPost(check()), "errorCovPost")
        set(value) = JniVideo.kalmanFilterSetErrorCovPost(check(), handleOf(value))

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo.kalmanFilterRelease(h)
        }
    }
}

actual fun kalmanFilter(
    dynamParams: Int,
    measureParams: Int,
    controlParams: Int,
    type: Int,
): KalmanFilter = JvmKalmanFilter(JniVideo.kalmanFilterCreate(dynamParams, measureParams, controlParams, type))

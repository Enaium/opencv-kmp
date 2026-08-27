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

// =========================================================================
// org.opencv.video parity: motion analysis (background subtraction, optical
// flow, ECC alignment, Kalman filter). Trackers live in the tracker slice.
// =========================================================================

/** Flags accepted by [calcOpticalFlowPyrLK] and [calcOpticalFlowFarneback]. */
object OpticalFlowFlags {
    /** Uses the values already stored in `nextPts`/`flow` as initial estimates. */
    const val USE_INITIAL_FLOW: Int = 4

    /** LK: minimum eigen values are used as the per-point error measure. */
    const val LK_GET_MIN_EIGENVALS: Int = 8

    /** Farneback: use a Gaussian window instead of a box window. */
    const val FARNEBACK_GAUSSIAN: Int = 256
}

/** Motion models accepted by [findTransformECC] and [findTransformECCMultiScale]. */
object EccMotion {
    /** Translational model; warp is 2x3 with a unity 2x2 block. */
    const val TRANSLATION: Int = 0

    /** Euclidean (rigid) model; 3 parameters, 2x3 warp. */
    const val EUCLIDEAN: Int = 1

    /** Affine model (default); 6 parameters, 2x3 warp. */
    const val AFFINE: Int = 2

    /** Homography model; 8 parameters, 3x3 warp. */
    const val HOMOGRAPHY: Int = 3
}

/** Presets for [createDisOpticalFlow] (`cv::DISOpticalFlow::Preset`). */
object DisOpticalFlowPreset {
    /** Fastest preset with the lowest quality. */
    const val ULTRAFAST: Int = 0

    /** Default preset balancing speed and quality. */
    const val FAST: Int = 1

    /** Slowest preset with the best quality. */
    const val MEDIUM: Int = 2
}

/**
 * Parameter holder for [findTransformECCMultiScale] (`cv::ECCParameters`).
 *
 * Pure Kotlin: the fields are expanded into the native call on use.
 * [itersPerLevel] optionally distributes the iteration budget over pyramid
 * levels (CV_32SC1 ints); null means the criteria budget is used on every
 * level.
 */
data class ECCParameters(
    var motionType: Int = EccMotion.AFFINE,
    var criteria: TermCriteria =
        TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 50, 0.001),
    var itersPerLevel: MatOfInt? = null,
    var gaussFiltSize: Int = 5,
    var nlevels: Int = 4,
    var interpolation: Int = InterpolationFlags.LINEAR,
)

// =========================================================================
// video statics
// =========================================================================

/**
 * Computes a dense optical flow using the Gunnar Farneback's algorithm
 * (`cv::calcOpticalFlowFarneback`); writes the CV_32FC2 flow into [flow].
 */
expect fun calcOpticalFlowFarneback(
    prev: Mat,
    next: Mat,
    flow: Mat,
    pyrScale: Double = 0.5,
    levels: Int = 5,
    winSize: Int = 13,
    iterations: Int = 10,
    polyN: Int = 5,
    polySigma: Double = 1.1,
    flags: Int = 0,
)

/**
 * Sparse iterative Lucas-Kanade optical flow with pyramids
 * (`cv::calcOpticalFlowPyrLK`). [nextPts], [status] and [err] are written
 * in place; [err] receives a per-point error when non-empty, pass
 * [MatOfFloat]() to request it.
 */
expect fun calcOpticalFlowPyrLK(
    prevImg: Mat,
    nextImg: Mat,
    prevPts: MatOfPoint2f,
    nextPts: MatOfPoint2f,
    status: MatOfByte,
    err: MatOfFloat,
    winSize: Size = Size(21, 21),
    maxLevel: Int = 3,
    criteria: TermCriteria =
        TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 30, 0.01),
    flags: Int = 0,
    minEigThreshold: Double = 1e-4,
)

/** Enhanced Correlation Coefficient between two images (`cv::computeECC`). */
expect fun computeECC(templateImage: Mat, inputImage: Mat, inputMask: Mat? = null): Double

/**
 * Finds the geometric transform (warp) between two images under the ECC
 * criterion (`cv::findTransformECC`); [warpMatrix] is refined in place and
 * the final correlation coefficient is returned.
 */
expect fun findTransformECC(
    templateImage: Mat,
    inputImage: Mat,
    warpMatrix: Mat,
    motionType: Int = EccMotion.AFFINE,
    criteria: TermCriteria =
        TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 50, 0.001),
    inputMask: Mat? = null,
    gaussFiltSize: Int = 5,
): Double

/**
 * ECC alignment with validity masks for both the template and the input
 * image (`cv::findTransformECCWithMask`); [warpMatrix] is refined in place.
 */
expect fun findTransformECCWithMask(
    templateImage: Mat,
    inputImage: Mat,
    templateMask: Mat,
    inputMask: Mat,
    warpMatrix: Mat,
    motionType: Int = EccMotion.AFFINE,
    criteria: TermCriteria =
        TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 50, 1e-6),
    gaussFiltSize: Int = 5,
): Double

/**
 * Pyramid-based ECC alignment (`cv::findTransformECCMultiScale`), more
 * stable than [findTransformECC]; [warpMatrix] is refined in place.
 */
expect fun findTransformECCMultiScale(
    reference: Mat,
    sample: Mat,
    warpMatrix: Mat,
    eccParams: ECCParameters = ECCParameters(),
    referenceMask: Mat? = null,
    sampleMask: Mat? = null,
): Double

// =========================================================================
// background subtraction
// =========================================================================

/**
 * Base interface for background/foreground segmentation
 * (`cv::BackgroundSubtractor`).
 */
interface BackgroundSubtractor : Algorithm {

    /**
     * Computes a foreground mask ([fgmask], 8-bit binary image) from the
     * next video frame. [learningRate] between 0 and 1; negative lets the
     * algorithm pick a rate automatically.
     */
    fun apply(image: Mat, fgmask: Mat, learningRate: Double = -1.0)

    /**
     * Computes a foreground mask while ignoring pixels already known to be
     * foreground in [knownForegroundMask].
     */
    fun apply(image: Mat, knownForegroundMask: Mat, fgmask: Mat, learningRate: Double = -1.0)

    /** Computes the estimated background image. */
    fun getBackgroundImage(): Mat
}

/** K-nearest neighbours background subtraction (`cv::BackgroundSubtractorKNN`). */
interface BackgroundSubtractorKNN : BackgroundSubtractor {

    /** Number of last frames that affect the background model. */
    var history: Int

    /** Number of data samples in the background model. */
    var nSamples: Int

    /** Threshold on the squared distance between a pixel and a sample. */
    var dist2Threshold: Double

    /** Number of neighbours (k) that need to match. */
    var kNNsamples: Int

    /** Shadow detection flag; shadows are marked with [shadowValue]. */
    var detectShadows: Boolean

    /** Value used to mark shadows in the foreground mask. */
    var shadowValue: Int

    /** Shadow threshold (Tau in the paper); darker pixels are not shadows. */
    var shadowThreshold: Double

    override fun close()
}

/** Gaussian mixture-based background subtraction (`cv::BackgroundSubtractorMOG2`). */
interface BackgroundSubtractorMOG2 : BackgroundSubtractor {

    /** Number of last frames that affect the background model. */
    var history: Int

    /** Number of gaussian components in the background model. */
    var nMixtures: Int

    /** "Background ratio" parameter (TB in the paper). */
    var backgroundRatio: Double

    /** Variance threshold for the pixel-model match (Cthr in the paper). */
    var varThreshold: Double

    /** Variance threshold for new mixture component generation (Tg). */
    var varThresholdGen: Double

    /** Initial variance of each gaussian component. */
    var varInit: Double

    /** Minimum variance of each gaussian component. */
    var varMin: Double

    /** Maximum variance of each gaussian component. */
    var varMax: Double

    /** Complexity reduction threshold (CT). */
    var complexityReductionThreshold: Double

    /** Shadow detection flag; shadows are marked with [shadowValue]. */
    var detectShadows: Boolean

    /** Value used to mark shadows in the foreground mask. */
    var shadowValue: Int

    /** Shadow threshold (Tau in the paper). */
    var shadowThreshold: Double

    override fun close()
}

/** Creates a KNN background subtractor (`cv::createBackgroundSubtractorKNN`). */
expect fun createBackgroundSubtractorKNN(
    history: Int = 500,
    dist2Threshold: Double = 400.0,
    detectShadows: Boolean = true,
): BackgroundSubtractorKNN

/** Creates a MOG2 background subtractor (`cv::createBackgroundSubtractorMOG2`). */
expect fun createBackgroundSubtractorMOG2(
    history: Int = 500,
    varThreshold: Double = 16.0,
    detectShadows: Boolean = true,
): BackgroundSubtractorMOG2

// =========================================================================
// optical flow
// =========================================================================

/** Base interface for dense optical flow algorithms (`cv::DenseOpticalFlow`). */
interface DenseOpticalFlow : Algorithm {

    /**
     * Calculates an optical flow between [i0] and [i1] (same size/type);
     * writes a CV_32FC2 flow into [flow].
     */
    fun calc(i0: Mat, i1: Mat, flow: Mat)

    /** Releases all inner buffers. */
    fun collectGarbage()
}

/** Dense flow using the Gunnar Farneback's algorithm. */
interface FarnebackOpticalFlow : DenseOpticalFlow {
    var numLevels: Int
    var pyrScale: Double
    var fastPyramids: Boolean
    var winSize: Int
    var numIters: Int
    var polyN: Int
    var polySigma: Double
    var flags: Int
    override fun close()
}

/** Dense Inverse Search (DIS) optical flow algorithm. */
interface DISOpticalFlow : DenseOpticalFlow {
    var finestScale: Int
    var coarsestScale: Int
    var patchSize: Int
    var patchStride: Int
    var gradientDescentIterations: Int
    var variationalRefinementIterations: Int
    var variationalRefinementAlpha: Float
    var variationalRefinementDelta: Float
    var variationalRefinementGamma: Float
    var variationalRefinementEpsilon: Float
    var useMeanNormalization: Boolean
    var useSpatialPropagation: Boolean
    override fun close()
}

/** Base interface for sparse optical flow algorithms (`cv::SparseOpticalFlow`). */
interface SparseOpticalFlow : Algorithm {

    /**
     * Calculates a sparse optical flow between [prevImg] and [nextImg].
     * [prevPts] holds the input CV_32FC2 points; [nextPts] and [status] are
     * written in place; [err] receives a per-point error when non-null.
     */
    fun calc(
        prevImg: Mat,
        nextImg: Mat,
        prevPts: Mat,
        nextPts: Mat,
        status: Mat,
        err: Mat? = null,
    )
}

/** Iterative Lucas-Kanade optical flow with pyramids. */
interface SparsePyrLKOpticalFlow : SparseOpticalFlow {
    var winSize: Size
    var maxLevel: Int
    var termCriteria: TermCriteria
    var flags: Int
    var minEigThreshold: Double
    override fun close()
}

/** Variational optical flow refinement (`cv::VariationalRefinement`). */
interface VariationalRefinement : DenseOpticalFlow {

    /**
     * Refines separate horizontal ([flowU]) and vertical ([flowV]) flow
     * components to avoid extra splits/merges.
     */
    fun calcUV(i0: Mat, i1: Mat, flowU: Mat, flowV: Mat)

    var fixedPointIterations: Int
    var sorIterations: Int
    var omega: Float
    var alpha: Float
    var delta: Float
    var gamma: Float
    var epsilon: Float
    override fun close()
}

/** Creates a Farneback dense optical flow computer. */
expect fun createFarnebackOpticalFlow(
    numLevels: Int = 5,
    pyrScale: Double = 0.5,
    fastPyramids: Boolean = false,
    winSize: Int = 13,
    numIters: Int = 10,
    polyN: Int = 5,
    polySigma: Double = 1.1,
    flags: Int = 0,
): FarnebackOpticalFlow

/** Creates a DIS dense optical flow computer with the given [preset]. */
expect fun createDisOpticalFlow(preset: Int = DisOpticalFlowPreset.FAST): DISOpticalFlow

/** Creates a variational flow refinement instance. */
expect fun createVariationalRefinement(): VariationalRefinement

/** Creates a sparse iterative Lucas-Kanade optical flow computer. */
expect fun createSparsePyrLKOpticalFlow(
    winSize: Size = Size(21, 21),
    maxLevel: Int = 3,
    criteria: TermCriteria =
        TermCriteria(TermCriteriaTypes.COUNT or TermCriteriaTypes.EPS, 30, 0.01),
    flags: Int = 0,
    minEigThreshold: Double = 1e-4,
): SparsePyrLKOpticalFlow

// =========================================================================
// Kalman filter
// =========================================================================

/**
 * Standard Kalman filter (`cv::KalmanFilter`). [predict] advances the
 * state, [correct] fuses a measurement; the state matrices are exposed for
 * direct manipulation (modify [transitionMatrix] etc. for an extended
 * filter).
 */
interface KalmanFilter : AutoCloseable {

    /**
     * Computes the predicted state; returns a Mat sharing the updated
     * [statePre] data.
     */
    fun predict(control: Mat? = null): Mat

    /**
     * Updates the predicted state from [measurement]; returns a Mat sharing
     * the updated [statePost] data.
     */
    fun correct(measurement: Mat): Mat

    /** Predicted state x'(k). */
    var statePre: Mat

    /** Corrected state x(k). */
    var statePost: Mat

    /** State transition matrix (A). */
    var transitionMatrix: Mat

    /** Control matrix (B), unused when there is no control input. */
    var controlMatrix: Mat

    /** Measurement matrix (H). */
    var measurementMatrix: Mat

    /** Process noise covariance (Q). */
    var processNoiseCov: Mat

    /** Measurement noise covariance (R). */
    var measurementNoiseCov: Mat

    /** Priori error estimate covariance P'(k). */
    var errorCovPre: Mat

    /** Kalman gain K(k). */
    var gain: Mat

    /** Posteriori error estimate covariance P(k). */
    var errorCovPost: Mat

    override fun close()
}

/**
 * Creates a Kalman filter ([cv::KalmanFilter] constructor). [type] is
 * CV_32F or CV_64F; pass 0 dims (the defaults) for the empty filter that is
 * configured through the matrix properties afterwards.
 */
expect fun kalmanFilter(
    dynamParams: Int = 0,
    measureParams: Int = 0,
    controlParams: Int = 0,
    type: Int = CV_32F,
): KalmanFilter

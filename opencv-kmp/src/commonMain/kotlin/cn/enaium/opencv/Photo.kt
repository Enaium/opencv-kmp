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
 * Constants and flags of the OpenCV photo module, mirroring the public
 * static fields of `org.opencv.photo.Photo`.
 */
object Photo {
    /** `cv::INPAINT_NS`: Navier-Stokes inpainting. */
    const val INPAINT_NS: Int = 0

    /** `cv::INPAINT_TELEA`: fast marching method inpainting. */
    const val INPAINT_TELEA: Int = 1

    /** `cv::LDR_SIZE`: default size of low-dynamic-range outputs. */
    const val LDR_SIZE: Int = 256

    /** `cv::RECURS_FILTER`: recursive edge-preserving filter. */
    const val RECURS_FILTER: Int = 1

    /** `cv::NORMCONV_FILTER`: normalized-convolution edge-preserving filter. */
    const val NORMCONV_FILTER: Int = 2

    /** `cv::SeamlessCloneFlags::NORMAL_CLONE`. */
    const val NORMAL_CLONE: Int = 1

    /** `cv::SeamlessCloneFlags::MIXED_CLONE`. */
    const val MIXED_CLONE: Int = 2

    /** `cv::SeamlessCloneFlags::MONOCHROME_TRANSFER`. */
    const val MONOCHROME_TRANSFER: Int = 3

    /** `cv::SeamlessCloneFlags::NORMAL_CLONE_WIDE`. */
    const val NORMAL_CLONE_WIDE: Int = 9

    /** `cv::SeamlessCloneFlags::MIXED_CLONE_WIDE`. */
    const val MIXED_CLONE_WIDE: Int = 10

    /** `cv::SeamlessCloneFlags::MONOCHROME_TRANSFER_WIDE`. */
    const val MONOCHROME_TRANSFER_WIDE: Int = 11

    /** `cv::ccm::CcmType::CCM_LINEAR`: 3x3 correction matrix. */
    const val CCM_LINEAR: Int = 0

    /** `cv::ccm::CcmType::CCM_AFFINE`: 4x3 correction matrix with offset. */
    const val CCM_AFFINE: Int = 1

    /** `cv::ccm::ColorCheckerType::COLORCHECKER_MACBETH`. */
    const val COLORCHECKER_MACBETH: Int = 0

    /** `cv::ccm::ColorCheckerType::COLORCHECKER_VINYL`. */
    const val COLORCHECKER_VINYL: Int = 1

    /** `cv::ccm::ColorCheckerType::COLORCHECKER_DIGITAL_SG`. */
    const val COLORCHECKER_DIGITAL_SG: Int = 2

    /** `cv::ccm::ColorSpace::COLOR_SPACE_SRGB`. */
    const val COLOR_SPACE_SRGB: Int = 0

    /** `cv::ccm::ColorSpace::COLOR_SPACE_SRGBL`. */
    const val COLOR_SPACE_SRGBL: Int = 1

    /** `cv::ccm::ColorSpace::COLOR_SPACE_ADOBE_RGB`. */
    const val COLOR_SPACE_ADOBE_RGB: Int = 2

    /** `cv::ccm::ColorSpace::COLOR_SPACE_ADOBE_RGBL`. */
    const val COLOR_SPACE_ADOBE_RGBL: Int = 3

    /** `cv::ccm::ColorSpace::COLOR_SPACE_WIDE_GAMUT_RGB`. */
    const val COLOR_SPACE_WIDE_GAMUT_RGB: Int = 4

    /** `cv::ccm::ColorSpace::COLOR_SPACE_WIDE_GAMUT_RGBL`. */
    const val COLOR_SPACE_WIDE_GAMUT_RGBL: Int = 5

    /** `cv::ccm::ColorSpace::COLOR_SPACE_PRO_PHOTO_RGB`. */
    const val COLOR_SPACE_PRO_PHOTO_RGB: Int = 6

    /** `cv::ccm::ColorSpace::COLOR_SPACE_PRO_PHOTO_RGBL`. */
    const val COLOR_SPACE_PRO_PHOTO_RGBL: Int = 7

    /** `cv::ccm::ColorSpace::COLOR_SPACE_DCI_P3_RGB`. */
    const val COLOR_SPACE_DCI_P3_RGB: Int = 8

    /** `cv::ccm::ColorSpace::COLOR_SPACE_DCI_P3_RGBL`. */
    const val COLOR_SPACE_DCI_P3_RGBL: Int = 9

    /** `cv::ccm::ColorSpace::COLOR_SPACE_APPLE_RGB`. */
    const val COLOR_SPACE_APPLE_RGB: Int = 10

    /** `cv::ccm::ColorSpace::COLOR_SPACE_APPLE_RGBL`. */
    const val COLOR_SPACE_APPLE_RGBL: Int = 11

    /** `cv::ccm::ColorSpace::COLOR_SPACE_REC_709_RGB`. */
    const val COLOR_SPACE_REC_709_RGB: Int = 12

    /** `cv::ccm::ColorSpace::COLOR_SPACE_REC_709_RGBL`. */
    const val COLOR_SPACE_REC_709_RGBL: Int = 13

    /** `cv::ccm::ColorSpace::COLOR_SPACE_REC_2020_RGB`. */
    const val COLOR_SPACE_REC_2020_RGB: Int = 14

    /** `cv::ccm::ColorSpace::COLOR_SPACE_REC_2020_RGBL`. */
    const val COLOR_SPACE_REC_2020_RGBL: Int = 15

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D65_2`. */
    const val COLOR_SPACE_XYZ_D65_2: Int = 16

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D50_2`. */
    const val COLOR_SPACE_XYZ_D50_2: Int = 17

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D65_10`. */
    const val COLOR_SPACE_XYZ_D65_10: Int = 18

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D50_10`. */
    const val COLOR_SPACE_XYZ_D50_10: Int = 19

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_A_2`. */
    const val COLOR_SPACE_XYZ_A_2: Int = 20

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_A_10`. */
    const val COLOR_SPACE_XYZ_A_10: Int = 21

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D55_2`. */
    const val COLOR_SPACE_XYZ_D55_2: Int = 22

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D55_10`. */
    const val COLOR_SPACE_XYZ_D55_10: Int = 23

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D75_2`. */
    const val COLOR_SPACE_XYZ_D75_2: Int = 24

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_D75_10`. */
    const val COLOR_SPACE_XYZ_D75_10: Int = 25

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_E_2`. */
    const val COLOR_SPACE_XYZ_E_2: Int = 26

    /** `cv::ccm::ColorSpace::COLOR_SPACE_XYZ_E_10`. */
    const val COLOR_SPACE_XYZ_E_10: Int = 27

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D65_2`. */
    const val COLOR_SPACE_LAB_D65_2: Int = 28

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D50_2`. */
    const val COLOR_SPACE_LAB_D50_2: Int = 29

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D65_10`. */
    const val COLOR_SPACE_LAB_D65_10: Int = 30

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D50_10`. */
    const val COLOR_SPACE_LAB_D50_10: Int = 31

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_A_2`. */
    const val COLOR_SPACE_LAB_A_2: Int = 32

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_A_10`. */
    const val COLOR_SPACE_LAB_A_10: Int = 33

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D55_2`. */
    const val COLOR_SPACE_LAB_D55_2: Int = 34

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D55_10`. */
    const val COLOR_SPACE_LAB_D55_10: Int = 35

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D75_2`. */
    const val COLOR_SPACE_LAB_D75_2: Int = 36

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_D75_10`. */
    const val COLOR_SPACE_LAB_D75_10: Int = 37

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_E_2`. */
    const val COLOR_SPACE_LAB_E_2: Int = 38

    /** `cv::ccm::ColorSpace::COLOR_SPACE_LAB_E_10`. */
    const val COLOR_SPACE_LAB_E_10: Int = 39

    /** `cv::ccm::DistanceType::DISTANCE_CIE76`. */
    const val DISTANCE_CIE76: Int = 0

    /** `cv::ccm::DistanceType::DISTANCE_CIE94_GRAPHIC_ARTS`. */
    const val DISTANCE_CIE94_GRAPHIC_ARTS: Int = 1

    /** `cv::ccm::DistanceType::DISTANCE_CIE94_TEXTILES`. */
    const val DISTANCE_CIE94_TEXTILES: Int = 2

    /** `cv::ccm::DistanceType::DISTANCE_CIE2000`. */
    const val DISTANCE_CIE2000: Int = 3

    /** `cv::ccm::DistanceType::DISTANCE_CMC_1TO1`. */
    const val DISTANCE_CMC_1TO1: Int = 4

    /** `cv::ccm::DistanceType::DISTANCE_CMC_2TO1`. */
    const val DISTANCE_CMC_2TO1: Int = 5

    /** `cv::ccm::DistanceType::DISTANCE_RGB`. */
    const val DISTANCE_RGB: Int = 6

    /** `cv::ccm::DistanceType::DISTANCE_RGBL`. */
    const val DISTANCE_RGBL: Int = 7

    /** `cv::ccm::InitialMethodType::INITIAL_METHOD_WHITE_BALANCE`. */
    const val INITIAL_METHOD_WHITE_BALANCE: Int = 0

    /** `cv::ccm::InitialMethodType::INITIAL_METHOD_LEAST_SQUARE`. */
    const val INITIAL_METHOD_LEAST_SQUARE: Int = 1

    /** `cv::ccm::LinearizationType::LINEARIZATION_IDENTITY`. */
    const val LINEARIZATION_IDENTITY: Int = 0

    /** `cv::ccm::LinearizationType::LINEARIZATION_GAMMA`. */
    const val LINEARIZATION_GAMMA: Int = 1

    /** `cv::ccm::LinearizationType::LINEARIZATION_COLORPOLYFIT`. */
    const val LINEARIZATION_COLORPOLYFIT: Int = 2

    /** `cv::ccm::LinearizationType::LINEARIZATION_COLORLOGPOLYFIT`. */
    const val LINEARIZATION_COLORLOGPOLYFIT: Int = 3

    /** `cv::ccm::LinearizationType::LINEARIZATION_GRAYPOLYFIT`. */
    const val LINEARIZATION_GRAYPOLYFIT: Int = 4

    /** `cv::ccm::LinearizationType::LINEARIZATION_GRAYLOGPOLYFIT`. */
    const val LINEARIZATION_GRAYLOGPOLYFIT: Int = 5
}

// =========================================================================
// Photo statics (org.opencv.photo.Photo)
// =========================================================================

/**
 * Restores the selected region in an image using the region neighborhood
 * (`cv::inpaint`). [inpaintMask] is an 8-bit single-channel image whose
 * non-zero pixels mark the area to reconstruct.
 */
expect fun photoInpaint(
    src: Mat,
    inpaintMask: Mat,
    inpaintRadius: Double,
    flags: Int = Photo.INPAINT_TELEA,
): Mat

/**
 * Non-local means denoising of a grayscale image with a scalar strength
 * [h] (`cv::fastNlMeansDenoising`).
 */
expect fun photoFastNlMeansDenoising(
    src: Mat,
    h: Float = 3f,
    templateWindowSize: Int = 7,
    searchWindowSize: Int = 21,
): Mat

/**
 * Non-local means denoising with per-channel strength values packed as a
 * CV_32FC1 [h] (`cv::fastNlMeansDenoising` vector variant).
 */
expect fun photoFastNlMeansDenoising(
    src: Mat,
    h: MatOfFloat,
    templateWindowSize: Int = 7,
    searchWindowSize: Int = 21,
    normType: Int = NormTypes.L2,
): Mat

/**
 * Non-local means denoising of a color image; [h] applies to the luminance
 * component and [hColor] to the chroma components
 * (`cv::fastNlMeansDenoisingColored`).
 */
expect fun photoFastNlMeansDenoisingColored(
    src: Mat,
    h: Float = 3f,
    hColor: Float = 3f,
    templateWindowSize: Int = 7,
    searchWindowSize: Int = 21,
): Mat

/**
 * Temporal non-local means denoising of a grayscale image sequence
 * (`cv::fastNlMeansDenoisingMulti`). [temporalWindowSize] must be odd.
 */
expect fun photoFastNlMeansDenoisingMulti(
    src: List<Mat>,
    imgToDenoiseIndex: Int,
    temporalWindowSize: Int,
    h: Float = 3f,
    templateWindowSize: Int = 7,
    searchWindowSize: Int = 21,
): Mat

/**
 * Temporal non-local means denoising of a grayscale image sequence with
 * per-channel strength values (`cv::fastNlMeansDenoisingMulti` vector h).
 */
expect fun photoFastNlMeansDenoisingMulti(
    src: List<Mat>,
    imgToDenoiseIndex: Int,
    temporalWindowSize: Int,
    h: MatOfFloat,
    templateWindowSize: Int = 7,
    searchWindowSize: Int = 21,
    normType: Int = NormTypes.L2,
): Mat

/**
 * Temporal non-local means denoising of a color image sequence
 * (`cv::fastNlMeansDenoisingColoredMulti`).
 */
expect fun photoFastNlMeansDenoisingColoredMulti(
    src: List<Mat>,
    imgToDenoiseIndex: Int,
    temporalWindowSize: Int,
    h: Float = 3f,
    hColor: Float = 3f,
    templateWindowSize: Int = 7,
    searchWindowSize: Int = 21,
): Mat

/**
 * Total-variation L1 denoising of a set of noisy observations of the same
 * scene (`cv::denoise_TVL1`); the result is CV_32FC3.
 */
expect fun photoDenoiseTvl1(
    observations: List<Mat>,
    lambda: Double = 1.0,
    niters: Int = 30,
): Mat

/**
 * Converts a color image to a quality grayscale and a boosted-color image
 * (`cv::decolor`); returns the `(grayscale, colorBoost)` pair.
 */
expect fun photoDecolor(src: Mat): Pair<Mat, Mat>

/**
 * Seamless image cloning: pastes [src] onto [dst] through [mask] centered
 * at [p] (`cv::seamlessClone`). [flags] is one of the `Photo.NORMAL_CLONE`
 * family.
 */
expect fun photoSeamlessClone(
    src: Mat,
    dst: Mat,
    mask: Mat,
    p: Point,
    flags: Int = Photo.NORMAL_CLONE,
): Mat

/**
 * Mixes differently colored versions of an image guided by [mask]
 * (`cv::colorChange`).
 */
expect fun photoColorChange(
    src: Mat,
    mask: Mat,
    redMul: Float = 1.0f,
    greenMul: Float = 1.0f,
    blueMul: Float = 1.0f,
): Mat

/**
 * Relights the [mask] region of an image (`cv::illuminationChange`).
 */
expect fun photoIlluminationChange(
    src: Mat,
    mask: Mat,
    alpha: Float = 0.2f,
    beta: Float = 0.4f,
): Mat

/**
 * Flattens texture inside the [mask] region while keeping edges
 * (`cv::textureFlattening`).
 */
expect fun photoTextureFlattening(
    src: Mat,
    mask: Mat,
    lowThreshold: Float = 30f,
    highThreshold: Float = 45f,
    kernelSize: Int = 3,
): Mat

/**
 * Edge-preserving smoothing (`cv::edgePreservingFilter`); [flags] is
 * [Photo.RECURS_FILTER] or [Photo.NORMCONV_FILTER].
 */
expect fun photoEdgePreservingFilter(
    src: Mat,
    flags: Int = Photo.RECURS_FILTER,
    sigmaS: Float = 60f,
    sigmaR: Float = 0.4f,
): Mat

/**
 * Enhances the details of an image (`cv::detailEnhance`).
 */
expect fun photoDetailEnhance(
    src: Mat,
    sigmaS: Float = 10f,
    sigmaR: Float = 0.15f,
): Mat

/**
 * Produces a pencil sketch of [src] (`cv::pencilSketch`); returns the
 * `(grayscale sketch, color sketch)` pair.
 */
expect fun photoPencilSketch(
    src: Mat,
    sigmaS: Float = 60f,
    sigmaR: Float = 0.07f,
    shadeFactor: Float = 0.02f,
): Pair<Mat, Mat>

/**
 * Stylizes an image with an abstraction effect (`cv::stylization`).
 */
expect fun photoStylization(
    src: Mat,
    sigmaS: Float = 60f,
    sigmaR: Float = 0.45f,
): Mat

/**
 * Corrects chromatic aberration using [coefficients] calibrated for
 * [imageSize] (`cv::correctChromaticAberration`).
 */
expect fun photoCorrectChromaticAberration(
    input: Mat,
    coefficients: Mat,
    imageSize: Size,
    calibDegree: Int,
    bayerPattern: Int = -1,
): Mat

/**
 * Applies gamma correction (`cv::ccm::gammaCorrection`).
 */
expect fun photoGammaCorrection(src: Mat, gamma: Double): Mat

// =========================================================================
// Tonemapping (org.opencv.photo.Tonemap)
// =========================================================================

/**
 * Base class for tonemapping algorithms mapping an HDR image into the
 * [0, 1] range (`cv::Tonemap`).
 */
interface Tonemap : Algorithm {

    /** Tonemaps [src] (CV_32FC3) into a CV_32FC3 result in [0, 1]. */
    fun process(src: Mat): Mat

    /** Gamma correction factor. */
    var gamma: Float

    override fun close()
}

/**
 * Adaptive logarithmic tonemapping (`cv::TonemapDrago`).
 */
interface TonemapDrago : Tonemap {
    /** Color saturation. */
    var saturation: Float

    /** Bias of the logarithmic mapping. */
    var bias: Float
}

/**
 * Gradient-domain tonemapping (`cv::TonemapMantiuk`).
 */
interface TonemapMantiuk : Tonemap {
    /** Contrast scale factor. */
    var scale: Float

    /** Color saturation. */
    var saturation: Float
}

/**
 * Photoreceptor-model tonemapping (`cv::TonemapReinhard`).
 */
interface TonemapReinhard : Tonemap {
    /** Result intensity in [0, 1]. */
    var intensity: Float

    /** Light adaptation strength; 0 disables it, 1 fully adapts. */
    var lightAdaptation: Float

    /** Color adaptation strength; 0 disables it, 1 fully adapts. */
    var colorAdaptation: Float
}

/** Creates a simple linear gamma-corrected tonemapper. */
expect fun createTonemap(gamma: Float = 1.0f): Tonemap

/** Creates an adaptive-logarithmic (Drago) tonemapper. */
expect fun createTonemapDrago(
    gamma: Float = 1.0f,
    saturation: Float = 1.0f,
    bias: Float = 0.85f,
): TonemapDrago

/** Creates a photoreceptor-model (Reinhard) tonemapper. */
expect fun createTonemapReinhard(
    gamma: Float = 1.0f,
    intensity: Float = 0.0f,
    lightAdaptation: Float = 1.0f,
    colorAdaptation: Float = 0.0f,
): TonemapReinhard

/** Creates a gradient-domain (Mantiuk) tonemapper. */
expect fun createTonemapMantiuk(
    gamma: Float = 1.0f,
    scale: Float = 0.7f,
    saturation: Float = 1.0f,
): TonemapMantiuk

// =========================================================================
// Exposure alignment / merging (org.opencv.photo.*)
// =========================================================================

/**
 * Base class for algorithms aligning images of the same scene captured
 * with different exposures (`cv::AlignExposures`).
 */
interface AlignExposures : Algorithm {

    /**
     * Aligns [src] into one output image per input; the caller owns and
     * must close every returned Mat.
     */
    fun process(src: List<Mat>, times: Mat, response: Mat): List<Mat>

    override fun close()
}

/**
 * Median threshold bitmap alignment (`cv::AlignMTB`).
 */
interface AlignMTB : AlignExposures {

    /** Aligns the exposure sequence without response calibration. */
    fun process(src: List<Mat>): List<Mat>

    /** Shift needed to make [img1] correspond to [img0]. */
    fun calculateShift(img0: Mat, img1: Mat): Point

    /** Shifts [src] by [shift], filling new regions with zeros. */
    fun shiftMat(src: Mat, shift: Point): Mat

    /** Median threshold bitmap and exclude bitmap of [img]. */
    fun computeBitmaps(img: Mat): Pair<Mat, Mat>

    /** Number of bits used in the median threshold bitmap. */
    var maxBits: Int

    /** Range of excluded intensity levels around the median. */
    var excludeRange: Int

    /** Whether images are cut before computing bitmaps. */
    var cut: Boolean
}

/**
 * Base class for camera response calibration algorithms
 * (`cv::CalibrateCRF`).
 */
interface CalibrateCRF : Algorithm {

    /**
     * Recovers the inverse camera response from [src] exposures taken at
     * [times]; returns the CV_32FC1 response function.
     */
    fun process(src: List<Mat>, times: Mat): Mat

    override fun close()
}

/**
 * Debevec camera response calibration (`cv::CalibrateDebevec`).
 */
interface CalibrateDebevec : CalibrateCRF {
    /** Smoothness regularization weight. */
    var lambda: Float

    /** Number of sampled image points used. */
    var samples: Int

    /** Whether sample points are picked randomly. */
    var random: Boolean
}

/**
 * Robertson camera response calibration (`cv::CalibrateRobertson`).
 */
interface CalibrateRobertson : CalibrateCRF {
    /** Maximum number of iterations. */
    var maxIter: Int

    /** Convergence threshold. */
    var threshold: Float

    /** Last computed radiance estimate (owned by the caller). */
    val radiance: Mat
}

/**
 * Base class for merging exposure sequences into one HDR image
 * (`cv::MergeExposures`).
 */
interface MergeExposures : Algorithm {

    /** Merges [src] using [times] and the calibrated [response]. */
    fun process(src: List<Mat>, times: Mat, response: Mat): Mat

    override fun close()
}

/**
 * Debevec HDR merging (`cv::MergeDebevec`).
 */
interface MergeDebevec : MergeExposures {
    /** Merges using exposure [times] without a response function. */
    fun process(src: List<Mat>, times: Mat): Mat
}

/**
 * Mertens exposure fusion (`cv::MergeMertens`); produces an LDR result in
 * [0, 1] without a response calibration.
 */
interface MergeMertens : MergeExposures {
    /** Merges the exposure sequence without times or response. */
    fun process(src: List<Mat>): Mat

    /** Contrast weight. */
    var contrastWeight: Float

    /** Saturation weight. */
    var saturationWeight: Float

    /** Well-exposedness weight. */
    var exposureWeight: Float
}

/**
 * Robertson HDR merging (`cv::MergeRobertson`).
 */
interface MergeRobertson : MergeExposures {
    /** Merges using exposure [times] without a response function. */
    fun process(src: List<Mat>, times: Mat): Mat
}

/** Creates an AlignMTB exposure aligner. */
expect fun createAlignMTB(
    maxBits: Int = 6,
    excludeRange: Int = 4,
    cut: Boolean = true,
): AlignMTB

/** Creates a Debevec camera response calibrator. */
expect fun createCalibrateDebevec(
    samples: Int = 70,
    lambda: Float = 10.0f,
    random: Boolean = false,
): CalibrateDebevec

/** Creates a Robertson camera response calibrator. */
expect fun createCalibrateRobertson(
    maxIter: Int = 30,
    threshold: Float = 0.01f,
): CalibrateRobertson

/** Creates a Debevec HDR merger. */
expect fun createMergeDebevec(): MergeDebevec

/** Creates a Mertens exposure fuser. */
expect fun createMergeMertens(
    contrastWeight: Float = 1.0f,
    saturationWeight: Float = 1.0f,
    exposureWeight: Float = 0.0f,
): MergeMertens

/** Creates a Robertson HDR merger. */
expect fun createMergeRobertson(): MergeRobertson

// =========================================================================
// Color correction / interactive segmentation
// =========================================================================

/**
 * Core class of the color correction model (`cv::ccm::ColorCorrectionModel`):
 * fits a color correction matrix to detected color card patches and applies
 * it to images.
 */
interface ColorCorrectionModel : AutoCloseable {

    /** Sets the gamma of the gamma linearization (default 2.2). */
    fun setLinearizationGamma(gamma: Double)

    /** Sets the polynomial degree of the (log)polyfit linearization. */
    fun setLinearizationDegree(deg: Int)

    /** Restricts loss-function participation to colors in [lower, upper]. */
    fun setSaturatedThreshold(lower: Double, upper: Double)

    /** Sets the per-patch weight list. */
    fun setWeightsList(weights: Mat)

    /** Sets the exponent of the Lab L* weight. */
    fun setWeightCoeff(weightsCoeff: Double)

    /** Caps DownhillSolver iterations (default 5000). */
    fun setMaxCount(maxCount: Int)

    /** Sets the DownhillSolver terminal epsilon (default 1e-4). */
    fun setEpsilon(epsilon: Double)

    /** Whether input images are RGB; false (default) means BGR. */
    fun setRGB(rgb: Boolean)

    /** Fits the model and returns the color correction matrix. */
    fun compute(): Mat

    /** The fitted 3x3 (linear) or 4x3 (affine) correction matrix. */
    val colorCorrectionMatrix: Mat

    /** Value of the loss function after [compute]. */
    val loss: Double

    /** Linearized RGB values of the detected patches. */
    val srcLinearRGB: Mat

    /** Linearized RGB values of the reference color card. */
    val refLinearRGB: Mat

    /** Mask of patches participating in the fit. */
    val mask: Mat

    /** Per-patch weights used by the fit. */
    val weights: Mat

    /**
     * Applies the fitted matrix to [src] and returns the corrected image of
     * the same size and type.
     */
    fun correctImage(src: Mat, islinear: Boolean = false): Mat

    override fun close()
}

/** Creates an empty color correction model. */
expect fun colorCorrectionModel(): ColorCorrectionModel

/**
 * Creates a color correction model from detected [src] color card patches
 * (RGB, values in [0, 1]) and a built-in [constColor] card.
 */
expect fun colorCorrectionModel(
    src: Mat,
    constColor: Int = Photo.COLORCHECKER_MACBETH,
): ColorCorrectionModel

/**
 * Interactive image segmentation with intelligent scissors
 * (`cv::segmentation::IntelligentScissorsMB`).
 */
interface IntelligentScissorsMB : AutoCloseable {

    /** Weights of the non-edge, gradient-direction and gradient-magnitude features. */
    fun setWeights(
        weightNonEdge: Float,
        weightGradientDirection: Float,
        weightGradientMagnitude: Float,
    )

    /** Caps the gradient magnitude feature (0 disables the limit). */
    fun setGradientMagnitudeMaxLimit(gradientMagnitudeThresholdMax: Float = 0.0f)

    /** Switches to the Laplacian zero-crossing edge feature. */
    fun setEdgeFeatureZeroCrossingParameters(gradientMagnitudeMinValue: Float = 0.0f)

    /** Switches to the Canny edge feature. */
    fun setEdgeFeatureCannyParameters(
        threshold1: Double,
        threshold2: Double,
        apertureSize: Int = 3,
        l2gradient: Boolean = false,
    )

    /** Loads [image] and extracts its edge features. */
    fun applyImage(image: Mat)

    /** Uses caller-provided edge features instead of computing them. */
    fun applyImageFeatures(
        nonEdge: Mat,
        gradientDirection: Mat,
        gradientMagnitude: Mat,
        image: Mat? = null,
    )

    /** Builds the optimal-path map from [sourcePt]. */
    fun buildMap(sourcePt: Point)

    /**
     * Extracts the optimal contour from [sourcePt] (set by [buildMap]) to
     * [targetPt]; the result is a CV_32SC2 list of pixel points.
     */
    fun getContour(targetPt: Point, backward: Boolean = false): Mat

    override fun close()
}

/** Creates an intelligent scissors segmenter. */
expect fun intelligentScissorsMB(): IntelligentScissorsMB

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
// Feature2D and the concrete detectors (org.opencv.features)
// =========================================================================

/**
 * Abstract base class for 2D image feature detectors and descriptor
 * extractors ([SIFT], [ORB], [MSER], [FastFeatureDetector], [GFTTDetector],
 * [SimpleBlobDetector], [AffineFeature]).
 *
 * KeyPoint collections travel through the [MatOfKeyPoint] wire format
 * (CV_32FC(7): x, y, size, angle, response, octave, classId); descriptors
 * are returned as a [Mat] with one row per keypoint.
 */
interface Feature2D : Algorithm {

    /**
     * Detects keypoints in [image]. [mask] is an optional 8-bit matrix with
     * non-zero values in the region of interest.
     */
    fun detect(image: Mat, mask: Mat? = null): List<KeyPoint>

    /**
     * Detects keypoints and computes their descriptors in one pass.
     * Returns the keypoints and a descriptor [Mat] (rows = keypoints,
     * cols = [descriptorSize]).
     */
    fun detectAndCompute(image: Mat, mask: Mat? = null): Pair<List<KeyPoint>, Mat>

    /**
     * Computes descriptors for [keypoints] detected in [image]. Keypoints
     * for which no descriptor can be computed may be removed, and some
     * detectors (e.g. SIFT) may add keypoints (multiple orientations);
     * the updated keypoint list is returned alongside the descriptor [Mat].
     */
    fun compute(image: Mat, keypoints: List<KeyPoint>): Pair<List<KeyPoint>, Mat>

    /** Columns of the descriptor matrix; 0 for detectors without descriptors. */
    val descriptorSize: Int

    /** OpenCV Mat depth of the descriptors ([CV_8U], [CV_32F], ...). */
    val descriptorType: Int

    /** Norm type used to compare descriptors ([NormTypes]). */
    val defaultNorm: Int

    /** Persists the detector parameters to [fileName] (YAML/XML). */
    fun write(fileName: String)

    /** Loads detector parameters from [fileName] (YAML/XML). */
    fun read(fileName: String)

    override fun close()
}

// =========================================================================
// SIFT
// =========================================================================

/**
 * Keypoint detector and descriptor extractor using the Scale Invariant
 * Feature Transform (SIFT) algorithm.
 */
interface SIFT : Feature2D {

    /** The number of best features to retain. */
    var nFeatures: Int

    /** The number of layers in each octave (3 is the D. Lowe paper value). */
    var nOctaveLayers: Int

    /** Contrast threshold filtering weak features in low-contrast regions. */
    var contrastThreshold: Double

    /** Edge threshold; larger keeps more edge-like features. */
    var edgeThreshold: Double

    /** Sigma of the Gaussian applied at octave 0. */
    var sigma: Double
}

/**
 * Creates a [SIFT] detector. [descriptorType] selects the descriptor depth
 * ([CV_32F] or [CV_8U]); pass -1 (the default) for the standard [CV_32F]
 * descriptors. [enablePreciseUpscale] maps index `x` to `2x` in the scale
 * pyramid, preventing localization bias.
 */
expect fun siftCreate(
    nfeatures: Int = 0,
    nOctaveLayers: Int = 3,
    contrastThreshold: Double = 0.04,
    edgeThreshold: Double = 10.0,
    sigma: Double = 1.6,
    descriptorType: Int = -1,
    enablePreciseUpscale: Boolean = false,
): SIFT

// =========================================================================
// ORB
// =========================================================================

/**
 * Keypoint detector and descriptor extractor using the ORB (*oriented
 * BRIEF*) algorithm.
 */
interface ORB : Feature2D {

    companion object {
        /** Keypoints ranked by the Harris corner response. */
        const val HARRIS_SCORE: Int = 0

        /** Keypoints ranked by the FAST score (slightly less stable). */
        const val FAST_SCORE: Int = 1
    }

    /** Maximum number of features to retain. */
    var maxFeatures: Int

    /** Pyramid decimation ratio, greater than 1. */
    var scaleFactor: Double

    /** Number of pyramid levels. */
    var nLevels: Int

    /** Border size where no features are detected (matches patch size). */
    var edgeThreshold: Int

    /** Pyramid level the source image is placed on. */
    var firstLevel: Int

    /** Points per oriented-BRIEF element (2, 3 or 4). */
    var wtaK: Int

    /** [HARRIS_SCORE] or [FAST_SCORE]. */
    var scoreType: Int

    /** Size of the patch used by the oriented BRIEF descriptor. */
    var patchSize: Int

    /** FAST threshold used to extract keypoints. */
    var fastThreshold: Int
}

/** Creates an [ORB] detector with the classic default parameters. */
expect fun orbCreate(
    nfeatures: Int = 500,
    scaleFactor: Float = 1.2f,
    nlevels: Int = 8,
    edgeThreshold: Int = 31,
    firstLevel: Int = 0,
    wtaK: Int = 2,
    scoreType: Int = ORB.HARRIS_SCORE,
    patchSize: Int = 31,
    fastThreshold: Int = 20,
): ORB

// =========================================================================
// MSER
// =========================================================================

/**
 * Maximally Stable Extremal Region extractor. Besides the generic
 * [Feature2D] surface it can return the full region point-sets via
 * [detectRegions].
 */
interface MSER : Feature2D {

    /** Comparison delta `(size_i - size_{i-delta}) / size_{i-delta}`. */
    var delta: Int

    /** Prunes regions smaller than this area. */
    var minArea: Int

    /** Prunes regions larger than this area. */
    var maxArea: Int

    /** Prunes regions with size similar to their children. */
    var maxVariation: Double

    /** Color-image trace-back cutoff for MSER diversity. */
    var minDiversity: Double

    /** Color-image evolution steps. */
    var maxEvolution: Int

    /** Color-image area threshold that causes re-initialization. */
    var areaThreshold: Double

    /** Color-image minimum margin. */
    var minMargin: Double

    /** Color-image aperture size for edge blur. */
    var edgeBlurSize: Int

    /** Whether to skip the second (stability) pass. */
    var pass2Only: Boolean

    /**
     * Detects MSER regions in [image] (8UC1/8UC3/8UC4, at least 3x3).
     * Returns the region point-sets and an Nx1 CV_32SC4 bounding-box [Mat]
     * ([x, y, width, height] per row) that the caller must close.
     */
    fun detectRegions(image: Mat): Pair<List<List<Point>>, Mat>
}

/** Creates an [MSER] detector with the default parameters. */
expect fun mserCreate(
    delta: Int = 5,
    minArea: Int = 60,
    maxArea: Int = 14400,
    maxVariation: Double = 0.25,
    minDiversity: Double = 0.2,
    maxEvolution: Int = 200,
    areaThreshold: Double = 1.01,
    minMargin: Double = 0.003,
    edgeBlurSize: Int = 5,
): MSER

// =========================================================================
// FastFeatureDetector
// =========================================================================

/** Wrapping class for feature detection using the FAST method. */
interface FastFeatureDetector : Feature2D {

    companion object {
        /** Type id of the `threshold` property. */
        const val THRESHOLD: Int = 10000

        /** Type id of the `nonmaxSuppression` property. */
        const val NONMAX_SUPPRESSION: Int = 10001

        /** Type id of the `type` property. */
        const val FAST_N: Int = 10002

        /** 5-8 point neighborhood. */
        const val TYPE_5_8: Int = 0

        /** 7-12 point neighborhood. */
        const val TYPE_7_12: Int = 1

        /** 9-16 point neighborhood (default). */
        const val TYPE_9_16: Int = 2
    }

    /** Threshold on the intensity difference to the circle pixels. */
    var threshold: Int

    /** Whether non-maximum suppression is applied to detected corners. */
    var nonmaxSuppression: Boolean

    /** Neighborhood type ([TYPE_5_8], [TYPE_7_12] or [TYPE_9_16]). */
    var type: Int
}

/** Creates a [FastFeatureDetector]. */
expect fun fastCreate(
    threshold: Int = 10,
    nonmaxSuppression: Boolean = true,
    type: Int = FastFeatureDetector.TYPE_9_16,
): FastFeatureDetector

// =========================================================================
// GFTTDetector
// =========================================================================

/**
 * Wrapping class for feature detection using the `goodFeaturesToTrack`
 * corner detector.
 */
interface GFTTDetector : Feature2D {

    /** Maximum number of corners to return. */
    var maxFeatures: Int

    /** Minimal accepted quality relative to the best corner. */
    var qualityLevel: Double

    /** Minimum possible Euclidean distance between returned corners. */
    var minDistance: Double

    /** Size of the averaging block for the derivative covariation matrix. */
    var blockSize: Int

    /** Sobel aperture used for the derivatives (3 by default). */
    var gradientSize: Int

    /** Whether to use the Harris detector instead of the min-eigenvalue one. */
    var harrisDetector: Boolean

    /** Free parameter of the Harris detector. */
    var k: Double
}

/**
 * Creates a [GFTTDetector]. Pass [gradientSize] = -1 (the default) to use
 * the classic blockSize-only overload (internally a 3x3 Sobel aperture);
 * pass an explicit aperture (3, 5 or 7) to select the gradient-size
 * overload.
 */
expect fun gfttCreate(
    maxCorners: Int = 1000,
    qualityLevel: Double = 0.01,
    minDistance: Double = 1.0,
    blockSize: Int = 3,
    gradientSize: Int = -1,
    useHarrisDetector: Boolean = false,
    k: Double = 0.04,
): GFTTDetector

// =========================================================================
// SimpleBlobDetector
// =========================================================================

/**
 * Class for extracting blobs from an image by thresholding, extracting
 * connected components and filtering them by color, area, circularity,
 * inertia ratio and convexity. Default parameters extract dark circular
 * blobs.
 */
interface SimpleBlobDetector : Feature2D {

    /**
     * Pure-Kotlin configuration holder mirroring
     * `cv::SimpleBlobDetector::Params`; defaults match the C++ header.
     */
    data class Params(
        /** Distance between neighboring thresholds. */
        var thresholdStep: Float = 10f,
        /** Lowest threshold (inclusive). */
        var minThreshold: Float = 50f,
        /** Highest threshold (exclusive). */
        var maxThreshold: Float = 220f,
        /** Minimum number of times a blob must appear to be kept. */
        var minRepeatability: Long = 2L,
        /** Minimum distance between blob centers. */
        var minDistBetweenBlobs: Float = 10f,
        /** Whether to filter by blob color. */
        var filterByColor: Boolean = true,
        /** Blob color to keep (0 = dark, 255 = light); 0..255. */
        var blobColor: Int = 0,
        /** Whether to filter by blob area. */
        var filterByArea: Boolean = true,
        /** Minimum blob area (inclusive). */
        var minArea: Float = 25f,
        /** Maximum blob area (exclusive). */
        var maxArea: Float = 5000f,
        /** Whether to filter by circularity. */
        var filterByCircularity: Boolean = false,
        /** Minimum circularity (inclusive). */
        var minCircularity: Float = 0.8f,
        /** Maximum circularity (exclusive). */
        var maxCircularity: Float = Float.MAX_VALUE,
        /** Whether to filter by the min/max inertia ratio. */
        var filterByInertia: Boolean = true,
        /** Minimum inertia ratio (inclusive). */
        var minInertiaRatio: Float = 0.1f,
        /** Maximum inertia ratio (exclusive). */
        var maxInertiaRatio: Float = Float.MAX_VALUE,
        /** Whether to filter by convexity. */
        var filterByConvexity: Boolean = true,
        /** Minimum convexity (inclusive). */
        var minConvexity: Float = 0.95f,
        /** Maximum convexity (exclusive). */
        var maxConvexity: Float = Float.MAX_VALUE,
        /** Whether to store the blob contours for [SimpleBlobDetector.getBlobContours]. */
        var collectContours: Boolean = false,
    )

    /** Replaces the detector's parameters. */
    fun setParams(params: Params)

    /** Returns a copy of the detector's current parameters. */
    fun getParams(): Params

    /**
     * Returns the contours of the blobs detected during the last
     * [Feature2D.detect] call; requires [Params.collectContours] = true.
     */
    fun getBlobContours(): List<List<Point>>
}

/** Creates a [SimpleBlobDetector] from [params] (defaults if omitted). */
expect fun simpleBlobDetectorCreate(
    params: SimpleBlobDetector.Params = SimpleBlobDetector.Params(),
): SimpleBlobDetector

// =========================================================================
// AffineFeature
// =========================================================================

/**
 * Wrapper making a detector/extractor affine invariant, described as ASIFT.
 * The backend runs over tilted and rotated views of the image.
 */
interface AffineFeature : Feature2D {

    /** Sets the tilt/roll view parameters used by the detector. */
    fun setViewParams(tilts: FloatArray, rolls: FloatArray)

    /** Returns the current tilt/roll view parameters. */
    fun getViewParams(): Pair<FloatArray, FloatArray>
}

/**
 * Creates an [AffineFeature] wrapping [backend]. [maxTilt] is the highest
 * power index of the tilt factor (5 in the paper), [minTilt] the lowest
 * (0), [tiltStep] the tilt sampling step and [rotateStepBase] the rotation
 * sampling step factor.
 */
expect fun affineFeatureCreate(
    backend: Feature2D,
    maxTilt: Int = 5,
    minTilt: Int = 0,
    tiltStep: Float = 1.4142136f,
    rotateStepBase: Float = 72f,
): AffineFeature

// =========================================================================
// internal helpers shared with the platform implementations
// =========================================================================

/**
 * Reads the keypoints out of a CV_32FC(7) keypoint [Mat] (the
 * [MatOfKeyPoint] wire format) and closes it. The Mat is consumed.
 */
internal fun keypointsOf(mat: Mat): List<KeyPoint> =
    try {
        MatOfKeyPoint(mat).toList()
    } finally {
        mat.close()
    }

/** Encodes [keypoints] into a fresh Nx1 CV_32FC(7) Mat owned by the caller. */
internal fun keypointsMat(keypoints: List<KeyPoint>): Mat {
    val m = mat(keypoints.size, 1, cvMakeType(CV_32F, 7))
    val view = MatOfKeyPoint(m)
    view.fromList(keypoints)
    // fromList() may swap the backing Mat (MatOfStore.adopt closes the
    // original); hand back the live backing Mat instead of the closed one.
    return view.mat
}

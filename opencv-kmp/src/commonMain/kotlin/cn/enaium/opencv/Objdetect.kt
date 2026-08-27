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

import kotlin.math.PI

/**
 * ArUco + calibration-pattern bindings for `org.opencv.objdetect`.
 *
 * Covers [Dictionary], [Board]/[GridBoard]/[CharucoBoard],
 * [ArucoDetector], [CharucoDetector] and the chessboard/circles-grid statics
 * of the SDK's `Objdetect` class (the QR/face/barcode surface lives in the
 * objdetect2 slice). The `*_Params` holders are pure Kotlin data classes;
 * the native detectors accept and expose them as plain values.
 *
 * List<Mat> arguments/results travel through the SDK's `vector_Mat` wire
 * format: a CV_32SC2 Nx1 Mat whose row i holds the hi/lo 32-bit halves of
 * the i-th Mat address. Every [Mat] inside a returned list is an independent
 * handle that must be closed by the caller.
 */
object Objdetect {
    /** `CALIB_CB_ADAPTIVE_THRESH`: adaptive thresholding before contour search. */
    const val CALIB_CB_ADAPTIVE_THRESH = 1
    const val CALIB_CB_NORMALIZE_IMAGE = 2
    const val CALIB_CB_FILTER_QUADS = 4
    const val CALIB_CB_FAST_CHECK = 8
    const val CALIB_CB_EXHAUSTIVE = 16
    const val CALIB_CB_ACCURACY = 32
    const val CALIB_CB_LARGER = 64
    const val CALIB_CB_MARKER = 128
    const val CALIB_CB_PLAIN = 256
    const val CALIB_CB_SYMMETRIC_GRID = 1
    const val CALIB_CB_ASYMMETRIC_GRID = 2
    const val CALIB_CB_CLUSTERING = 4

    /** ArUco corner refinement strategies ([DetectorParameters.cornerRefinementMethod]). */
    const val CORNER_REFINE_NONE = 0
    const val CORNER_REFINE_SUBPIX = 1
    const val CORNER_REFINE_CONTOUR = 2
    const val CORNER_REFINE_APRILTAG = 3

    const val DICT_4X4_50 = 0
    const val DICT_4X4_100 = 1
    const val DICT_4X4_250 = 2
    const val DICT_4X4_1000 = 3
    const val DICT_5X5_50 = 4
    const val DICT_5X5_100 = 5
    const val DICT_5X5_250 = 6
    const val DICT_5X5_1000 = 7
    const val DICT_6X6_50 = 8
    const val DICT_6X6_100 = 9
    const val DICT_6X6_250 = 10
    const val DICT_6X6_1000 = 11
    const val DICT_7X7_50 = 12
    const val DICT_7X7_100 = 13
    const val DICT_7X7_250 = 14
    const val DICT_7X7_1000 = 15
    const val DICT_ARUCO_ORIGINAL = 16
    const val DICT_APRILTAG_16h5 = 17
    const val DICT_APRILTAG_25h9 = 18
    const val DICT_APRILTAG_36h10 = 19
    const val DICT_APRILTAG_36h11 = 20
    const val DICT_ARUCO_MIP_36h12 = 21
}

/**
 * Marker detection parameters (`cv::aruco::DetectorParameters`).
 *
 * All defaults match the OpenCV C++ struct. Field order is the wire order
 * used by the native transports: it MUST stay in sync with
 * `cvk_detector_params_t` (opencv_kmp_objdetect.h) and the JNI DoubleArray
 * conversion (jni_objdetect.cpp).
 */
data class DetectorParameters(
    /** Minimum window size for adaptive thresholding (default 3). */
    var adaptiveThreshWinSizeMin: Int = 3,
    /** Maximum window size for adaptive thresholding (default 23). */
    var adaptiveThreshWinSizeMax: Int = 23,
    /** Window-size increments during thresholding (default 10). */
    var adaptiveThreshWinSizeStep: Int = 10,
    /** Constant for adaptive thresholding (default 7). */
    var adaptiveThreshConstant: Double = 7.0,
    /** Minimum marker perimeter as a rate of the image dimension (default 0.03). */
    var minMarkerPerimeterRate: Double = 0.03,
    /** Maximum marker perimeter rate (default 4.0). */
    var maxMarkerPerimeterRate: Double = 4.0,
    /** Polygonal approximation accuracy rate (default 0.03). */
    var polygonalApproxAccuracyRate: Double = 0.03,
    /** Minimum corner distance relative to the marker perimeter (default 0.05). */
    var minCornerDistanceRate: Double = 0.05,
    /** Minimum distance of any corner to the image border, in pixels (default 3). */
    var minDistanceToBorder: Int = 3,
    /** Grouping distance rate between marker corners (default 0.125). */
    var minMarkerDistanceRate: Double = 0.125,
    /** Minimum average corner distance inside a group, relative to module size (default 0.21). */
    var minGroupDistance: Float = 0.21f,
    /** Corner refinement strategy, see [Objdetect.CORNER_REFINE_NONE] (default 0). */
    var cornerRefinementMethod: Int = Objdetect.CORNER_REFINE_NONE,
    /** Maximum corner refinement window size in pixels (default 5). */
    var cornerRefinementWinSize: Int = 5,
    /** Dynamic refinement window relative to the marker module size (default 0.3). */
    var relativeCornerRefinmentWinSize: Float = 0.3f,
    /** Maximum iterations of the corner refinement stop criteria (default 30). */
    var cornerRefinementMaxIterations: Int = 30,
    /** Minimum error for the corner refinement stop criteria (default 0.1). */
    var cornerRefinementMinAccuracy: Double = 0.1,
    /** Width of the marker border in bits (default 1). */
    var markerBorderBits: Int = 1,
    /** Cells per dimension when removing perspective (default 4). */
    var perspectiveRemovePixelPerCell: Int = 4,
    /** Ignored cell-margin rate during perspective removal (default 0.13). */
    var perspectiveRemoveIgnoredMarginPerCell: Double = 0.13,
    /** Allowed erroneous border bits as a rate of the marker bits (default 0.35). */
    var maxErroneousBitsInBorderRate: Double = 0.35,
    /** Minimum pixel stddev before Otsu thresholding during decoding (default 5.0). */
    var minOtsuStdDev: Double = 5.0,
    /** Error-correction rate relative to the dictionary capability (default 0.6). */
    var errorCorrectionRate: Double = 0.6,
    /** AprilTag: quad detection decimation factor (default 0.0). */
    var aprilTagQuadDecimate: Float = 0.0f,
    /** AprilTag: Gaussian blur sigma for quad detection (default 0.0). */
    var aprilTagQuadSigma: Float = 0.0f,
    /** AprilTag: reject quads with too few pixels (default 5). */
    var aprilTagMinClusterPixels: Int = 5,
    /** AprilTag: corner candidates considered per segment (default 10). */
    var aprilTagMaxNmaxima: Int = 10,
    /** AprilTag: reject quads with edges closer than this angle, in radians (default 10*PI/180). */
    var aprilTagCriticalRad: Float = (10.0 * PI / 180.0).toFloat(),
    /** AprilTag: maximum line-fit mean squared error (default 10.0). */
    var aprilTagMaxLineFitMse: Float = 10.0f,
    /** AprilTag: required brightness difference between white and black models (default 5). */
    var aprilTagMinWhiteBlackDiff: Int = 5,
    /** AprilTag: deglitch the thresholded image (default 0). */
    var aprilTagDeglitch: Int = 0,
    /** Also search inverted (white) markers (default false). */
    var detectInvertedMarker: Boolean = false,
    /** Enable the Aruco3 detection strategy (`useAruco3Detection`, default false). */
    var useAruco: Boolean = false,
    /** Minimum marker side length in the canonical image (default 32). */
    var minSideLengthCanonicalImg: Int = 32,
    /** Marker length ratio in the original image, tau_i from the Aruco3 paper (default 0.0). */
    var minMarkerLengthRatioOriginalImg: Float = 0.0f,
    /** Accepted threshold when comparing a detection to the dictionary (default 0.49). */
    var validBitIdThreshold: Float = 0.49f,
)

/**
 * Refinement parameters (`cv::aruco::RefineParameters`).
 *
 * Field order is the native wire order (see [DetectorParameters]).
 */
data class RefineParameters(
    /** Minimum distance between rejected-candidate and reprojected corners (default 10). */
    var minRepDistance: Float = 10.0f,
    /** Allowed erroneous-bits rate vs the dictionary capability; -1 skips correction (default 3). */
    var errorCorrectionRate: Float = 3.0f,
    /** Consider all four corner orders of the rejected candidates (default true). */
    var checkAllOrders: Boolean = true,
)

/**
 * ChArUco interpolation parameters (`cv::aruco::CharucoParameters`).
 */
data class CharucoParameters(
    /** Optional 3x3 camera matrix used for pose-based interpolation. */
    var cameraMatrix: Mat? = null,
    /** Optional distortion coefficients used for pose-based interpolation. */
    var distCoeffs: Mat? = null,
    /** Adjacent markers required to return a charuco corner (default 2). */
    var minMarkers: Int = 2,
    /** Try refineBoard to recover missing markers (default false). */
    var tryRefineMarkers: Boolean = false,
    /** Verify that the markers belong to the same board (default true). */
    var checkMarkers: Boolean = true,
)

/** Result of [Dictionary.identify]. */
data class IdentifyResult(
    /** Whether the marker was identified. */
    val found: Boolean,
    /** Marker id in the dictionary, -1 when not identified. */
    val idx: Int,
    /** Rotation of the marker (0..3). */
    val rotation: Int,
)

/** Result of [ArucoDetector.detectMarkers]; every Mat must be closed by the caller. */
data class MarkerDetection(
    /** Detected marker corners; for N markers, N Mats of CV_32FC2 shape 4x1, clockwise. */
    val corners: List<Mat>,
    /** CV_32SC1 Nx1 identifiers, same order as [corners]. */
    val ids: Mat,
    /** Rejected square candidates in the same format as [corners]. */
    val rejectedImgPoints: List<Mat>,
)

/** Result of [ArucoDetector.detectMarkersWithConfidence]. */
data class MarkerDetectionWithConfidence(
    val corners: List<Mat>,
    val ids: Mat,
    /** CV_32FC1 Nx1 normalized confidence in [0;1] per detected marker. */
    val markersConfidence: Mat,
    val rejectedImgPoints: List<Mat>,
)

/** Result of [ArucoDetector.refineDetectedMarkers]. */
data class RefinedMarkers(
    /** Refined marker corners. */
    val corners: List<Mat>,
    /** Refined marker identifiers. */
    val ids: Mat,
    /** Remaining rejected candidates. */
    val rejectedCorners: List<Mat>,
    /** Indexes into the original rejectedCorners array of recovered candidates. */
    val recoveredIdxs: Mat,
)

/** Result of [ArucoDetector.detectMarkersMultiDict]. */
data class MultiDictDetection(
    val corners: List<Mat>,
    val ids: Mat,
    val rejectedImgPoints: List<Mat>,
    /** CV_32SC1 dictionary index per detected marker. */
    val dictIndices: Mat,
)

/** Result of [Board.matchImagePoints], usable with solvePnP. */
data class BoardMatchPoints(
    /** CV_32FC3 marker points in board coordinates. */
    val objPoints: Mat,
    /** CV_32FC2 marker points in image coordinates. */
    val imgPoints: Mat,
)

/** Result of [CharucoDetector.detectBoard]. */
data class CharucoBoardDetection(
    /** CV_32FC2 interpolated chessboard corners. */
    val charucoCorners: Mat,
    /** CV_32SC1 identifiers of the returned charuco corners. */
    val charucoIds: Mat,
    /** Detected (or caller-provided) marker corners. */
    val markerCorners: List<Mat>,
    /** CV_32SC1 marker identifiers matching [markerCorners]. */
    val markerIds: Mat,
)

/** Result of [CharucoDetector.detectDiamonds]. */
data class DiamondDetection(
    /** CV_32FC2 4x1 corner Mats, one per diamond. */
    val diamondCorners: List<Mat>,
    /** CV_32SC4 diamonds ids (4 marker ids per diamond). */
    val diamondIds: Mat,
    /** Detected (or caller-provided) marker corners. */
    val markerCorners: List<Mat>,
    /** CV_32SC1 marker identifiers matching [markerCorners]. */
    val markerIds: Mat,
)

/** Result of [findChessboardCorners] / [findChessboardCornersSB]. */
data class ChessboardCornersResult(
    /** True when the whole pattern was found and ordered. */
    val found: Boolean,
    /** CV_32FC2 Nx1 detected corners (empty when not found). */
    val corners: Mat,
)

/** Result of [findChessboardCornersSBWithMeta]. */
data class ChessboardSbMetaResult(
    val found: Boolean,
    /** CV_32FC2 Nx1 detected corners. */
    val corners: Mat,
    /** CV_8UC1 patternSize cell metadata. */
    val meta: Mat,
)

/** Result of [findCirclesGrid]. */
data class CirclesGridResult(
    val found: Boolean,
    /** CV_32FC2 Nx1 circle centers (empty when not found). */
    val centers: Mat,
)

/** Result of [estimateChessboardSharpness]. */
data class ChessboardSharpnessResult(
    /** (average sharpness, average min brightness, average max brightness, 0). */
    val sharpness: Scalar,
    /** CV_32FC1 per-edge profiles: x, y, sharpness, black, white signal. */
    val sharpnessMap: Mat,
)

/**
 * A set of unique ArUco markers of the same size (`cv::aruco::Dictionary`).
 *
 * Every handle owns one native reference; call [close] to release it.
 * [bytesList] and [getMarkerBits] return independent Mat handles that must
 * also be closed by the caller.
 */
interface Dictionary : AutoCloseable {
    /** Marker codewords: CV_8UC4 with `rows * 4 * nbytes` bytes per row. */
    val bytesList: Mat
    /** Number of bits per marker dimension. */
    val markerSize: Int
    /** Maximum number of bits that can be corrected. */
    val maxCorrectionBits: Int

    fun setBytesList(bytesList: Mat)
    fun setMarkerSize(markerSize: Int)
    fun setMaxCorrectionBits(maxCorrectionBits: Int)

    /** Identifies a marker bit matrix; [IdentifyResult.idx] is -1 when unknown. */
    fun identify(onlyBits: Mat, maxCorrectionRate: Double): IdentifyResult

    /** Identifies a [0;1] per-cell pixel-ratio matrix. */
    fun identify(onlyCellPixelRatio: Mat, maxCorrectionRate: Double, validBitIdThreshold: Float): IdentifyResult

    /** Hamming distance of [bits] to the marker [id], all rotations when [allRotations]. */
    fun getDistanceToId(bits: Mat, id: Int, allRotations: Boolean = true): Int

    /** Renders the canonical marker image (side_pixels x side_pixels, CV_8UC1). */
    fun generateImageMarker(id: Int, sidePixels: Int, borderBits: Int = 1): Mat

    /** Ground-truth bit matrix (markerSize x markerSize) for a marker id. */
    fun getMarkerBits(markerId: Int, rotationId: Int = 0): Mat

    override fun close()
}

/** A board of ArUco markers in a common 3D coordinate system (`cv::aruco::Board`). */
interface Board : AutoCloseable {
    /** New handle holding a copy of the board's dictionary. */
    val dictionary: Dictionary
    /** Marker corners in board coordinates; each CV_32FC3 4x1 Mat is caller-owned. */
    val objPoints: List<Mat>
    /** CV_32SC1 marker identifiers, same size as [objPoints]. */
    val ids: Mat
    /** Bottom-right corner of the board. */
    val rightBottomCorner: Point3

    /**
     * Matches detected markers against the board layout for solvePnP.
     * [detectedCorners] must be the marker corners returned by
     * [ArucoDetector.detectMarkers] and [detectedIds] the matching ids.
     */
    fun matchImagePoints(detectedCorners: List<Mat>, detectedIds: Mat): BoardMatchPoints

    /** Draws the planar board centered on a new outSize image. */
    fun generateImage(outSize: Size, marginSize: Int = 0, borderBits: Int = 1): Mat

    override fun close()
}

/** Planar board with a grid arrangement of markers (`cv::aruco::GridBoard`). */
interface GridBoard : Board {
    /** Marker count per (width, height) direction. */
    val gridSize: Size
    /** Marker side length (normally in meters). */
    val markerLength: Float
    /** Separation between two markers (same unit as [markerLength]). */
    val markerSeparation: Float
}

/** ChArUco board: markers inside the white squares of a chessboard. */
interface CharucoBoard : Board {
    /** Legacy (pre-4.6.0) chessboard pattern. */
    var legacyPattern: Boolean
    /** Chessboard square count per (width, height) direction. */
    val chessboardSize: Size
    /** Chessboard square side length. */
    val squareLength: Float
    /** Marker side length (same unit as [squareLength]). */
    val markerLength: Float
    /** Chessboard corners as a CV_32FC3 Nx1 Mat (caller-owned). */
    val chessboardCorners: Mat

    /** True when the given charuco corner ids form a straight line. */
    fun checkCharucoCornersCollinear(charucoIds: Mat): Boolean
}

/**
 * ArUco marker detector (`cv::aruco::ArucoDetector`).
 *
 * [dictionary], [detectorParameters] and [refineParameters] are plain value
 * holders; reading [dictionary] creates a fresh Dictionary handle the caller
 * must close.
 */
interface ArucoDetector : Algorithm {
    var dictionary: Dictionary
    var detectorParameters: DetectorParameters
    var refineParameters: RefineParameters

    /** Basic marker detection; see [MarkerDetection]. */
    fun detectMarkers(image: Mat): MarkerDetection

    /** Marker detection with per-marker confidence; see [MarkerDetectionWithConfidence]. */
    fun detectMarkersWithConfidence(image: Mat): MarkerDetectionWithConfidence

    /**
     * Recovers markers that were missed by [detectMarkers] using the board
     * layout. [detectedCorners]/[detectedIds] come from [detectMarkers],
     * [rejectedCorners] from its rejected list. [cameraMatrix]/[distCoeffs]
     * may be null for homography-based interpolation.
     */
    fun refineDetectedMarkers(
        image: Mat,
        board: Board,
        detectedCorners: List<Mat>,
        detectedIds: Mat,
        rejectedCorners: List<Mat>,
        cameraMatrix: Mat? = null,
        distCoeffs: Mat? = null,
    ): RefinedMarkers

    /** detectMarkers over the configured dictionary list, with dict indices. */
    fun detectMarkersMultiDict(image: Mat): MultiDictDetection

    override fun close()
}

/**
 * ChArUco corner interpolator (`cv::aruco::CharucoDetector`).
 *
 * [detectBoard] and [detectDiamonds] detect the markers themselves when
 * [CharucoBoardDetection.markerCorners] / [DiamondDetection.markerCorners]
 * are passed empty; otherwise the caller-supplied detections are refined.
 * [detectDiamonds] requires a 3x3 board.
 */
interface CharucoDetector : Algorithm {
    var board: CharucoBoard
    var charucoParameters: CharucoParameters
    var detectorParameters: DetectorParameters
    var refineParameters: RefineParameters

    /**
     * Interpolates charuco corners from detected markers.
     * When [markerCorners] is empty (and [markerIds] null) the detector runs
     * its own marker detection first; otherwise both must be provided and
     * [markerIds] total must equal [markerCorners] size.
     */
    fun detectBoard(
        image: Mat,
        markerCorners: List<Mat> = emptyList(),
        markerIds: Mat? = null,
    ): CharucoBoardDetection

    /** Detects ChArUco diamond markers (board must be 3x3). */
    fun detectDiamonds(
        image: Mat,
        markerCorners: List<Mat> = emptyList(),
        markerIds: Mat? = null,
    ): DiamondDetection

    override fun close()
}

// =========================================================================
// factories
// =========================================================================

/** Builds a [Dictionary] from its marker codewords; release with [Dictionary.close]. */
expect fun dictionary(bytesList: Mat, markerSize: Int, maxCorrectionBits: Int = 0): Dictionary

/** Returns one of the predefined dictionaries referenced by the [Objdetect.DICT_*] constants. */
expect fun getPredefinedDictionary(dict: Int): Dictionary

/** Named overload: `getPredefinedDictionary("DICT_4X4_50")` etc. */
fun getPredefinedDictionary(name: String): Dictionary = getPredefinedDictionary(predefinedDictionaryId(name))

/**
 * Extends [baseDictionary] (or creates a fresh one when null) to
 * [nMarkers] markers of [markerSize] x [markerSize] bits.
 */
expect fun extendDictionary(
    nMarkers: Int,
    markerSize: Int,
    baseDictionary: Dictionary? = null,
    randomSeed: Int = 0,
): Dictionary

/** Common board constructor; each Mat in [objPoints] is the CV_32FC3 4x1 corners of a marker. */
expect fun board(objPoints: List<Mat>, dictionary: Dictionary, ids: Mat): Board

/** GridBoard constructor; [ids] defaults to the first N dictionary ids. */
expect fun gridBoard(
    size: Size,
    markerLength: Float,
    markerSeparation: Float,
    dictionary: Dictionary,
    ids: Mat? = null,
): GridBoard

/** CharucoBoard constructor; [ids] defaults to the ids filling the white squares. */
expect fun charucoBoard(
    size: Size,
    squareLength: Float,
    markerLength: Float,
    dictionary: Dictionary,
    ids: Mat? = null,
): CharucoBoard

/** Basic ArucoDetector constructor. */
expect fun arucoDetector(
    dictionary: Dictionary,
    detectorParams: DetectorParameters = DetectorParameters(),
    refineParams: RefineParameters = RefineParameters(),
): ArucoDetector

/** Basic CharucoDetector constructor. */
expect fun charucoDetector(
    board: CharucoBoard,
    charucoParams: CharucoParameters = CharucoParameters(),
    detectorParams: DetectorParameters = DetectorParameters(),
    refineParams: RefineParameters = RefineParameters(),
): CharucoDetector

// =========================================================================
// Objdetect statics
// =========================================================================

/** Finds the internal corners of a chessboard pattern. */
expect fun findChessboardCorners(
    image: Mat,
    patternSize: Size,
    flags: Int = Objdetect.CALIB_CB_ADAPTIVE_THRESH + Objdetect.CALIB_CB_NORMALIZE_IMAGE,
): ChessboardCornersResult

/** Checks whether the image contains a chessboard of the specific size. */
expect fun checkChessboard(img: Mat, size: Size): Boolean

/** Sector-based chessboard corner detection (radon transform). */
expect fun findChessboardCornersSB(
    image: Mat,
    patternSize: Size,
    flags: Int = 0,
): ChessboardCornersResult

/** Sector-based chessboard detection with the per-corner metadata output. */
expect fun findChessboardCornersSBWithMeta(
    image: Mat,
    patternSize: Size,
    flags: Int,
): ChessboardSbMetaResult

/** Estimates the sharpness of a detected chessboard from edge profiles. */
expect fun estimateChessboardSharpness(
    image: Mat,
    patternSize: Size,
    corners: Mat,
    riseDistance: Float = 0.8f,
    vertical: Boolean = false,
): ChessboardSharpnessResult

/** Refines chessboard corners in place ([corners] is mutated); true on success. */
expect fun find4QuadCornerSubpix(img: Mat, corners: Mat, regionSize: Size): Boolean

/** Renders the detected chessboard corners onto [image]. */
expect fun drawChessboardCorners(
    image: Mat,
    patternSize: Size,
    corners: Mat,
    patternWasFound: Boolean,
)

/** Finds the centers of a grid of circles. */
expect fun findCirclesGrid(
    image: Mat,
    patternSize: Size,
    flags: Int = Objdetect.CALIB_CB_SYMMETRIC_GRID,
): CirclesGridResult

/** Draws detected ArUco markers (corners from [ArucoDetector.detectMarkers]). */
expect fun drawDetectedMarkers(
    image: Mat,
    corners: List<Mat>,
    ids: Mat? = null,
    borderColor: Scalar = Scalar(0.0, 255.0, 0.0),
)

/** Renders a canonical marker image for [dictionary]. */
expect fun generateImageMarker(
    dictionary: Dictionary,
    id: Int,
    sidePixels: Int,
    borderBits: Int = 1,
): Mat

/** Draws a set of detected Charuco corners. */
expect fun drawDetectedCornersCharuco(
    image: Mat,
    charucoCorners: Mat,
    charucoIds: Mat? = null,
    cornerColor: Scalar = Scalar(255.0, 0.0, 0.0),
)

/** Draws a set of detected ChArUco Diamond markers. */
expect fun drawDetectedDiamonds(
    image: Mat,
    diamondCorners: List<Mat>,
    diamondIds: Mat? = null,
    borderColor: Scalar = Scalar(0.0, 0.0, 255.0),
)

// =========================================================================
// internal helpers shared with the platform implementations
// =========================================================================

/** Wire type of vector-of-Mat containers: CV_32SC2. */
internal val vectorMatContainerType: Int = cvMakeType(CV_32S, 2)

internal const val DETECTOR_PARAM_COUNT = 35
internal const val REFINE_PARAM_COUNT = 3

/** Packed wire order — must match `cvk_detector_params_t` / the JNI conversion. */
internal fun DetectorParameters.toParamArray(): DoubleArray = doubleArrayOf(
    adaptiveThreshWinSizeMin.toDouble(),
    adaptiveThreshWinSizeMax.toDouble(),
    adaptiveThreshWinSizeStep.toDouble(),
    adaptiveThreshConstant,
    minMarkerPerimeterRate,
    maxMarkerPerimeterRate,
    polygonalApproxAccuracyRate,
    minCornerDistanceRate,
    minDistanceToBorder.toDouble(),
    minMarkerDistanceRate,
    minGroupDistance.toDouble(),
    cornerRefinementMethod.toDouble(),
    cornerRefinementWinSize.toDouble(),
    relativeCornerRefinmentWinSize.toDouble(),
    cornerRefinementMaxIterations.toDouble(),
    cornerRefinementMinAccuracy,
    markerBorderBits.toDouble(),
    perspectiveRemovePixelPerCell.toDouble(),
    perspectiveRemoveIgnoredMarginPerCell,
    maxErroneousBitsInBorderRate,
    minOtsuStdDev,
    errorCorrectionRate,
    aprilTagQuadDecimate.toDouble(),
    aprilTagQuadSigma.toDouble(),
    aprilTagMinClusterPixels.toDouble(),
    aprilTagMaxNmaxima.toDouble(),
    aprilTagCriticalRad.toDouble(),
    aprilTagMaxLineFitMse.toDouble(),
    aprilTagMinWhiteBlackDiff.toDouble(),
    aprilTagDeglitch.toDouble(),
    if (detectInvertedMarker) 1.0 else 0.0,
    if (useAruco) 1.0 else 0.0,
    minSideLengthCanonicalImg.toDouble(),
    minMarkerLengthRatioOriginalImg.toDouble(),
    validBitIdThreshold.toDouble(),
)

internal fun DoubleArray.toDetectorParameters(): DetectorParameters {
    require(size >= DETECTOR_PARAM_COUNT) { "detector params array too short: $size" }
    return DetectorParameters(
        this[0].toInt(), this[1].toInt(), this[2].toInt(), this[3],
        this[4], this[5], this[6], this[7], this[8].toInt(), this[9],
        this[10].toFloat(), this[11].toInt(), this[12].toInt(), this[13].toFloat(),
        this[14].toInt(), this[15], this[16].toInt(), this[17].toInt(), this[18],
        this[19], this[20], this[21], this[22].toFloat(), this[23].toFloat(),
        this[24].toInt(), this[25].toInt(), this[26].toFloat(), this[27].toFloat(),
        this[28].toInt(), this[29].toInt(), this[30] != 0.0, this[31] != 0.0,
        this[32].toInt(), this[33].toFloat(), this[34].toFloat(),
    )
}

internal fun RefineParameters.toParamArray(): DoubleArray = doubleArrayOf(
    minRepDistance.toDouble(),
    errorCorrectionRate.toDouble(),
    if (checkAllOrders) 1.0 else 0.0,
)

internal fun DoubleArray.toRefineParameters(): RefineParameters {
    require(size >= REFINE_PARAM_COUNT) { "refine params array too short: $size" }
    return RefineParameters(this[0].toFloat(), this[1].toFloat(), this[2] != 0.0)
}

/** Maps a DICT_* constant name to its [Objdetect] id. */
internal fun predefinedDictionaryId(name: String): Int = when (name) {
    "DICT_4X4_50" -> Objdetect.DICT_4X4_50
    "DICT_4X4_100" -> Objdetect.DICT_4X4_100
    "DICT_4X4_250" -> Objdetect.DICT_4X4_250
    "DICT_4X4_1000" -> Objdetect.DICT_4X4_1000
    "DICT_5X5_50" -> Objdetect.DICT_5X5_50
    "DICT_5X5_100" -> Objdetect.DICT_5X5_100
    "DICT_5X5_250" -> Objdetect.DICT_5X5_250
    "DICT_5X5_1000" -> Objdetect.DICT_5X5_1000
    "DICT_6X6_50" -> Objdetect.DICT_6X6_50
    "DICT_6X6_100" -> Objdetect.DICT_6X6_100
    "DICT_6X6_250" -> Objdetect.DICT_6X6_250
    "DICT_6X6_1000" -> Objdetect.DICT_6X6_1000
    "DICT_7X7_50" -> Objdetect.DICT_7X7_50
    "DICT_7X7_100" -> Objdetect.DICT_7X7_100
    "DICT_7X7_250" -> Objdetect.DICT_7X7_250
    "DICT_7X7_1000" -> Objdetect.DICT_7X7_1000
    "DICT_ARUCO_ORIGINAL" -> Objdetect.DICT_ARUCO_ORIGINAL
    "DICT_APRILTAG_16h5" -> Objdetect.DICT_APRILTAG_16h5
    "DICT_APRILTAG_25h9" -> Objdetect.DICT_APRILTAG_25h9
    "DICT_APRILTAG_36h10" -> Objdetect.DICT_APRILTAG_36h10
    "DICT_APRILTAG_36h11" -> Objdetect.DICT_APRILTAG_36h11
    "DICT_ARUCO_MIP_36h12" -> Objdetect.DICT_ARUCO_MIP_36h12
    else -> throw IllegalArgumentException("unknown predefined dictionary name: $name")
}

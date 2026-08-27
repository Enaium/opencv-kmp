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
 * Base interface for graphical-code detectors (`cv::GraphicalCodeDetector`):
 * QR codes and barcodes. Out-parameter [Mat]s (points / straight code) are
 * caller-allocated and filled by the native side, mirroring the Java SDK.
 */
interface GraphicalCodeDetector : AutoCloseable {

    /**
     * Detects a graphical code in [img] and writes the quadrangle vertices
     * into [points]. Returns true when a code was found.
     */
    fun detect(img: Mat, points: Mat): Boolean

    /** Decodes the code in [img] located at [points], writing the binarized
     * code into [straightCode]. Returns "" when no code can be decoded. */
    fun decode(img: Mat, points: Mat, straightCode: Mat): String

    /** [decode] without the straightened output image. */
    fun decode(img: Mat, points: Mat): String {
        val straight = mat()
        try {
            return decode(img, points, straight)
        } finally {
            straight.close()
        }
    }

    /** Both detects and decodes a graphical code in [img]. */
    fun detectAndDecode(img: Mat, points: Mat, straightCode: Mat): String

    /** [detectAndDecode] without the straightened output image. */
    fun detectAndDecode(img: Mat, points: Mat): String {
        val straight = mat()
        try {
            return detectAndDecode(img, points, straight)
        } finally {
            straight.close()
        }
    }

    /** [detectAndDecode] discarding the quadrangle and straightened output. */
    fun detectAndDecode(img: Mat): String {
        val points = mat()
        val straight = mat()
        try {
            return detectAndDecode(img, points, straight)
        } finally {
            points.close()
            straight.close()
        }
    }

    /** Detects multiple graphical codes, writing all quadrangles into [points]. */
    fun detectMulti(img: Mat, points: Mat): Boolean

    /**
     * Decodes all codes in [img] located at [points]. Returns the decoded
     * strings plus the straightened code images (see [MultiDecodeResult]).
     */
    fun decodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult

    /** [decodeMulti] discarding the straightened output images. */
    fun decodeMulti(img: Mat, points: Mat): MultiDecodeResult {
        val straight = mat()
        try {
            return decodeMulti(img, points, straight)
        } finally {
            straight.close()
        }
    }

    /** Both detects and decodes multiple graphical codes in [img]. */
    fun detectAndDecodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult

    /** [detectAndDecodeMulti] discarding the straightened output images. */
    fun detectAndDecodeMulti(img: Mat, points: Mat): MultiDecodeResult {
        val straight = mat()
        try {
            return detectAndDecodeMulti(img, points, straight)
        } finally {
            straight.close()
        }
    }

    /** [detectAndDecodeMulti] discarding the quadrangles and straightened images. */
    fun detectAndDecodeMulti(img: Mat): MultiDecodeResult {
        val points = mat()
        val straight = mat()
        try {
            return detectAndDecodeMulti(img, points, straight)
        } finally {
            points.close()
            straight.close()
        }
    }

    override fun close()
}

/** Result of the multi-code detect/decode family. */
data class MultiDecodeResult(
    /** True when at least one code was detected/decoded. */
    val ok: Boolean,
    /** UTF-8 decoded payloads, one entry per detected code. */
    val decodedInfo: List<String>,
    /** Binarized code images produced as a side output; close after use. */
    val straightCode: List<Mat>,
)

/** Result of [BarcodeDetector.decodeWithType] / [BarcodeDetector.detectAndDecodeWithType]. */
data class BarcodeDecodeResult(
    /** True when at least one valid barcode was found. */
    val ok: Boolean,
    /** UTF-8 decoded payloads, one entry per detected barcode. */
    val decodedInfo: List<String>,
    /** Symbology type strings (e.g. "EAN_13"), one per detected barcode. */
    val decodedType: List<String>,
)

// =========================================================================
// QRCodeDetector
// =========================================================================

/** QR code detector (`cv::QRCodeDetector`). */
interface QRCodeDetector : GraphicalCodeDetector {

    /**
     * Sets the epsilon used during the horizontal scan of QR code stop marker
     * detection; returns this detector for chaining.
     */
    fun setEpsX(epsX: Double): QRCodeDetector

    /** Sets the epsilon used during the vertical scan of QR code stop marker
     * detection; returns this detector for chaining. */
    fun setEpsY(epsY: Double): QRCodeDetector

    /** Use alignment markers to improve the corner positions; on by default. */
    fun setUseAlignmentMarkers(useAlignmentMarkers: Boolean): QRCodeDetector

    /** Decodes a QR code on a curved surface once located by [detect]. */
    fun decodeCurved(img: Mat, points: Mat, straightQrcode: Mat): String

    /** [decodeCurved] without the rectified output image. */
    fun decodeCurved(img: Mat, points: Mat): String {
        val straight = mat()
        try {
            return decodeCurved(img, points, straight)
        } finally {
            straight.close()
        }
    }

    /** Both detects and decodes a QR code on a curved surface. */
    fun detectAndDecodeCurved(img: Mat, points: Mat, straightQrcode: Mat): String

    /** [detectAndDecodeCurved] without the rectified output image. */
    fun detectAndDecodeCurved(img: Mat, points: Mat): String {
        val straight = mat()
        try {
            return detectAndDecodeCurved(img, points, straight)
        } finally {
            straight.close()
        }
    }

    /** [detectAndDecodeCurved] discarding the quadrangle and rectified output. */
    fun detectAndDecodeCurved(img: Mat): String {
        val points = mat()
        val straight = mat()
        try {
            return detectAndDecodeCurved(img, points, straight)
        } finally {
            points.close()
            straight.close()
        }
    }

    /**
     * Kind of encoding for the latest decoded info ([QRCodeEncoder.ECI_SHIFT_JIS]
     * or [QRCodeEncoder.ECI_UTF8]); [codeIdx] selects among multi-code results.
     */
    fun getEncoding(codeIdx: Int = 0): Int

    override fun close()
}

/** Creates a [QRCodeDetector] with default parameters. */
expect fun qrCodeDetectorCreate(): QRCodeDetector

// =========================================================================
// QRCodeDetectorAruco
// =========================================================================

/**
 * QR code detector based on ArUco marker detection code
 * (`cv::QRCodeDetectorAruco`).
 */
interface QRCodeDetectorAruco : GraphicalCodeDetector {

    /** Detector parameters getter; see [QRCodeDetectorArucoParams]. */
    fun getDetectorParameters(): QRCodeDetectorArucoParams

    /** Detector parameters setter; returns this detector for chaining. */
    fun setDetectorParameters(params: QRCodeDetectorArucoParams): QRCodeDetectorAruco

    override fun close()
}

/**
 * Configuration for [QRCodeDetectorAruco] (`cv::QRCodeDetectorAruco::Params`).
 * Defaults match the C++ constructor.
 */
data class QRCodeDetectorArucoParams(
    /** Minimum allowed pixel size of a QR module in the smallest pyramid image. */
    var minModuleSizeInPyramid: Float = 4f,
    /** Maximum allowed relative rotation for finder patterns in one QR code. */
    var maxRotation: Float = (PI / 12.0).toFloat(),
    /** Maximum allowed relative module-size mismatch for one QR code. */
    var maxModuleSizeMismatch: Float = 1.75f,
    /** Maximum allowed module relative mismatch for the timing pattern. */
    var maxTimingPatternMismatch: Float = 2f,
    /** Maximum allowed percentage of penalty points in the timing pattern. */
    var maxPenalties: Float = 0.4f,
    /** Maximum allowed relative color mismatch in the timing pattern. */
    var maxColorsMismatch: Float = 0.2f,
    /** Timing-pattern score multiplier used when picking the best QR code. */
    var scaleTimingPatternScore: Float = 0.9f,
)

/** Creates a [QRCodeDetectorAruco]; [params] defaults to the C++ defaults. */
expect fun qrCodeDetectorArucoCreate(
    params: QRCodeDetectorArucoParams = QRCodeDetectorArucoParams(),
): QRCodeDetectorAruco

// =========================================================================
// QRCodeEncoder
// =========================================================================

/** QR code encoder (`cv::QRCodeEncoder`). */
interface QRCodeEncoder : AutoCloseable {

    companion object {
        /** Error correction level L (low, ~7%). */
        const val CORRECT_LEVEL_L = 0
        /** Error correction level M (~15%). */
        const val CORRECT_LEVEL_M = 1
        /** Error correction level Q (~25%). */
        const val CORRECT_LEVEL_Q = 2
        /** Error correction level H (high, ~30%). */
        const val CORRECT_LEVEL_H = 3

        /** ECI encoding: Shift JIS. */
        const val ECI_SHIFT_JIS = 20
        /** ECI encoding: UTF-8. */
        const val ECI_UTF8 = 26

        /** Encoding mode chosen automatically. */
        const val MODE_AUTO = -1
        /** Numeric encoding mode. */
        const val MODE_NUMERIC = 1
        /** Alphanumeric encoding mode. */
        const val MODE_ALPHANUMERIC = 2
        /** Byte encoding mode. */
        const val MODE_BYTE = 4
        /** ECI encoding mode. */
        const val MODE_ECI = 7
        /** Kanji encoding mode. */
        const val MODE_KANJI = 8
        /** Structured Append encoding mode. */
        const val MODE_STRUCTURED_APPEND = 3
    }

    /** Generates a QR code image (grayscale CV_8UC1) from [encodedInfo]. */
    fun encode(encodedInfo: String): Mat

    /** Generates a QR code image from raw [encodedInfo] bytes. */
    fun encode(encodedInfo: ByteArray): Mat

    /** Generates the QR code sequence for [encodedInfo] in Structured Append mode. */
    fun encodeStructuredAppend(encodedInfo: String): List<Mat>

    override fun close()
}

/**
 * QR code encoder parameters (`cv::QRCodeEncoder::Params`). Defaults match
 * the C++ constructor.
 */
data class QRCodeEncoderParams(
    /** QR version; 0 selects the maximum possible for the payload length. */
    var version: Int = 0,
    /** Error correction level, one of [QRCodeEncoder.CORRECT_LEVEL_L]..H. */
    var correctionLevel: Int = QRCodeEncoder.CORRECT_LEVEL_L,
    /** Encoding mode, one of [QRCodeEncoder.MODE_AUTO] and friends. */
    var mode: Int = QRCodeEncoder.MODE_AUTO,
    /** Number of QR codes to generate in Structured Append mode. */
    var structureNumber: Int = 1,
)

/** Creates a [QRCodeEncoder]; [params] defaults to the C++ defaults. */
expect fun qrCodeEncoderCreate(params: QRCodeEncoderParams = QRCodeEncoderParams()): QRCodeEncoder

// =========================================================================
// BarcodeDetector
// =========================================================================

/** 1D barcode detector and decoder (`cv::barcode::BarcodeDetector`). */
interface BarcodeDetector : GraphicalCodeDetector {

    /**
     * Decodes barcodes in [img] located at [points] (quadrangles from
     * [detect]); returns the payloads and symbology types.
     */
    fun decodeWithType(img: Mat, points: Mat): BarcodeDecodeResult

    /** Both detects and decodes barcodes in [img]. */
    fun detectAndDecodeWithType(img: Mat, points: Mat): BarcodeDecodeResult

    /** [detectAndDecodeWithType] discarding the found quadrangles. */
    fun detectAndDecodeWithType(img: Mat): BarcodeDecodeResult {
        val points = mat()
        try {
            return detectAndDecodeWithType(img, points)
        } finally {
            points.close()
        }
    }

    /** Detector downsampling limit (default 512); larger disables downsampling. */
    val downsamplingThreshold: Double

    /** Sets the downsampling threshold; returns this detector for chaining. */
    fun setDownsamplingThreshold(thresh: Double): BarcodeDetector

    /** Detector box filter sizes relative to the minimum image dimension. */
    fun getDetectorScales(): MatOfFloat

    /** Sets the detector box filter sizes; returns this detector for chaining. */
    fun setDetectorScales(sizes: MatOfFloat): BarcodeDetector

    /** Detector gradient magnitude threshold (default 64). */
    val gradientThreshold: Double

    /** Sets the gradient magnitude threshold; returns this detector for chaining. */
    fun setGradientThreshold(thresh: Double): BarcodeDetector

    override fun close()
}

/** Creates a [BarcodeDetector] with super resolution disabled. */
expect fun barcodeDetectorCreate(): BarcodeDetector

/**
 * Creates a [BarcodeDetector] loading an ONNX super-resolution model from
 * [modelPath]. The model file is not bundled; a missing or invalid path makes
 * the native call fail with [OpenCVException].
 */
expect fun barcodeDetectorCreate(modelPath: String): BarcodeDetector

// =========================================================================
// FaceDetectorYN
// =========================================================================

/**
 * DNN-based face detector (`cv::FaceDetectorYN`). Model files are not
 * bundled: [faceDetectorYNCreate] throws [OpenCVException] when the ONNX
 * model cannot be loaded. Models: opencv_zoo `face_detection_yunet`.
 */
interface FaceDetectorYN : AutoCloseable {

    /** Size of the network input; overwrites the size used at creation. */
    var inputSize: Size

    /** Score threshold for filtering out bounding boxes. */
    var scoreThreshold: Float

    /** Non-maximum-suppression IoU threshold. */
    var nmsThreshold: Float

    /** Number of bounding boxes preserved before NMS. */
    var topK: Int

    /**
     * Detects faces in [image], writing a [num_faces, 15] CV_32F result into
     * [faces] (bbox, landmarks, score; see the Java SDK docs). Returns the
     * number of detected faces.
     */
    fun detect(image: Mat, faces: Mat): Int

    override fun close()
}

/**
 * Creates a [FaceDetectorYN] from model files. [model] is an ONNX file path;
 * [config] is unused for ONNX and may be empty. Throws [OpenCVException]
 * when the model file cannot be loaded.
 */
expect fun faceDetectorYNCreate(
    model: String,
    config: String,
    inputSize: Size,
    scoreThreshold: Float = 0.9f,
    nmsThreshold: Float = 0.3f,
    topK: Int = 5000,
    backendId: Int = 0,
    targetId: Int = 0,
): FaceDetectorYN

/**
 * Creates a [FaceDetectorYN] from in-memory model buffers; [framework] names
 * the origin framework (e.g. "onnx").
 */
expect fun faceDetectorYNCreate(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte,
    inputSize: Size,
    scoreThreshold: Float = 0.9f,
    nmsThreshold: Float = 0.3f,
    topK: Int = 5000,
    backendId: Int = 0,
    targetId: Int = 0,
): FaceDetectorYN

// =========================================================================
// FaceRecognizerSF
// =========================================================================

/**
 * DNN-based face recognizer (`cv::FaceRecognizerSF`). Model files are not
 * bundled: [faceRecognizerSFCreate] throws [OpenCVException] when the ONNX
 * model cannot be loaded. Models: opencv_zoo `face_recognition_sface`.
 */
interface FaceRecognizerSF : AutoCloseable {

    companion object {
        /** Cosine distance. */
        const val FR_COSINE = 0
        /** L2 normalized distance. */
        const val FR_NORM_L2 = 1
    }

    /** Aligns the face described by [faceBox] in [srcImg] and crops it. */
    fun alignCrop(srcImg: Mat, faceBox: Mat): Mat

    /** Extracts the face feature vector from an aligned image. */
    fun feature(alignedImg: Mat): Mat

    /** Distance between two face features; [disType] is [FR_COSINE] or [FR_NORM_L2]. */
    fun match(faceFeature1: Mat, faceFeature2: Mat, disType: Int = FR_COSINE): Double

    override fun close()
}

/**
 * Creates a [FaceRecognizerSF] from model files; [config] is unused for ONNX
 * and may be empty. Throws [OpenCVException] when the model cannot be loaded.
 */
expect fun faceRecognizerSFCreate(
    model: String,
    config: String,
    backendId: Int = 0,
    targetId: Int = 0,
): FaceRecognizerSF

/** Creates a [FaceRecognizerSF] from in-memory model buffers. */
expect fun faceRecognizerSFCreate(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte,
    backendId: Int = 0,
    targetId: Int = 0,
): FaceRecognizerSF

// =========================================================================
// MCC: CChecker / CCheckerDetector
// =========================================================================

/**
 * A detected color checker (`cv::mcc::CChecker`): its box corners, color
 * charts and cost. Also an [Algorithm] (clear/empty/save/getDefaultName).
 */
interface CChecker : Algorithm {

    /** Sets the box corners (a CV_32FC2 point list). */
    fun setBox(box: MatOfPoint2f)

    /** Sets the RGB chart colors. */
    fun setChartsRGB(chartsRGB: Mat)

    /** Sets the YCbCr chart colors. */
    fun setChartsYCbCr(chartsYCbCr: Mat)

    /** Detection cost of this checker. */
    var cost: Float

    /** Center point of the checker. */
    var center: Point2f

    /** Box corners as a CV_32FC2 point list. */
    fun getBox(): MatOfPoint2f

    /** Coordinates of the central parts of the chart modules. */
    fun getColorCharts(): MatOfPoint2f

    /** RGB chart colors; [getStats] requests per-chart statistics. */
    fun getChartsRGB(getStats: Boolean = true): Mat

    /** YCbCr chart colors. */
    fun getChartsYCbCr(): Mat

    override fun close()
}

/** Creates a new [CChecker] object. */
expect fun cCheckerCreate(): CChecker

/**
 * Finds the positions of color charts in an image (`cv::mcc::CCheckerDetector`).
 * Detected checkers are stored internally and exposed via
 * [getListColorChecker]. Also an [Algorithm].
 */
interface CCheckerDetector : Algorithm {

    /** Searches [image] within [regionsOfInterest] (a CV_32SC4 rect list). */
    fun processWithROI(image: Mat, regionsOfInterest: MatOfRect, nc: Int = 1): Boolean

    /** Searches the full [image] for [nc] charts. */
    fun process(image: Mat, nc: Int = 1): Boolean

    /** All color checkers detected by the last [process] call. */
    fun getListColorChecker(): List<CChecker>

    /** Reference colors for the detected chart type. */
    fun getRefColors(): Mat

    /** Sets the MCC detection configuration. */
    fun setDetectionParams(params: DetectorParametersMCC)

    /** Enables/disables the neural-network-based detection stage. */
    var useDnnModel: Boolean

    /** Current MCC detection configuration. */
    fun getDetectionParams(): DetectorParametersMCC

    override fun close()
}

/** Creates a [CCheckerDetector]. */
expect fun cCheckerDetectorCreate(): CCheckerDetector

/**
 * Detection configuration for [CCheckerDetector]
 * (`cv::mcc::DetectorParametersMCC`). Defaults match the C++ constructor.
 */
data class DetectorParametersMCC(
    /** Minimum window size for adaptive thresholding. */
    var adaptiveThreshWinSizeMin: Int = 23,
    /** Maximum window size for adaptive thresholding. */
    var adaptiveThreshWinSizeMax: Int = 153,
    /** Window-size step between the min and max thresholds. */
    var adaptiveThreshWinSizeStep: Int = 16,
    /** Constant for adaptive thresholding. */
    var adaptiveThreshConstant: Double = 7.0,
    /** Minimum marker contour area as a rate of the image area (DNN stage). */
    var minContoursAreaRate: Double = 0.003,
    /** Minimum marker contour area in pixels (DNN stage). */
    var minContoursArea: Double = 100.0,
    /** Minimum confidence for a DNN bounding box. */
    var confidenceThreshold: Double = 0.5,
    /** Minimum solidity of a contour to be accepted as a chart square. */
    var minContourSolidity: Double = 0.9,
    /** Multiplier for approxPolyDP during candidate finding. */
    var findCandidatesApproxPolyDPEpsMultiplier: Double = 0.05,
    /** Padding width around initial DNN detections. */
    var borderWidth: Int = 0,
    /** Distance between neighboring squares as a ratio of square size. */
    var b0factor: Float = 1.25f,
    /** Maximum allowed error in chart detection. */
    var maxError: Float = 0.1f,
    /** Minimum points in a detected contour. */
    var minContourPointsAllowed: Int = 4,
    /** Minimum length of a contour. */
    var minContourLengthAllowed: Int = 100,
    /** Minimum distance between two contours. */
    var minInterContourDistance: Int = 100,
    /** Minimum distance between two checkers. */
    var minInterCheckerDistance: Int = 10000,
    /** Minimum size of the smaller image dimension. */
    var minImageSize: Int = 1000,
    /** Minimum number of chart squares that must be detected. */
    var minGroupSize: Int = 4,
)

/**
 * Parameters for the circles-grid detection algorithm
 * (`cv::CirclesGridFinderParameters`). Defaults match the C++ constructor.
 */
data class CirclesGridFinderParameters(
    /** Neighborhood size for keypoint density estimation. */
    var densityNeighborhoodSize: Size2f = Size2f(16f, 16f),
    /** Minimum keypoint density to accept a graph vertex. */
    var minDensity: Float = 10f,
    /** Attempts of the k-means keypoint clustering. */
    var kmeansAttempts: Int = 100,
    /** Minimum distance between the found grid and a new keypoint. */
    var minDistanceToAddKeypoint: Int = 20,
    /** Keypoint scale. */
    var keypointScale: Int = 1,
    /** Minimum graph confidence. */
    var minGraphConfidence: Float = 9f,
    /** Gain for adding a vertex. */
    var vertexGain: Float = 1f,
    /** Penalty for adding a vertex. */
    var vertexPenalty: Float = -0.6f,
    /** Gain for an existing vertex. */
    var existingVertexGain: Float = 10000f,
    /** Gain for adding an edge. */
    var edgeGain: Float = 1f,
    /** Penalty for adding an edge. */
    var edgePenalty: Float = -0.6f,
    /** Convex hull factor. */
    var convexHullFactor: Float = 1.1f,
    /** Minimum relative graph edge-switch distance. */
    var minRNGEdgeSwitchDist: Float = 5f,
    /** Grid type, [SYMMETRIC_GRID] or [ASYMMETRIC_GRID]. */
    var gridType: Int = SYMMETRIC_GRID,
    /** Distance between two adjacent points (used by CALIB_CB_CLUSTERING). */
    var squareSize: Float = 1f,
    /** Max deviation from prediction (used by CALIB_CB_CLUSTERING). */
    var maxRectifiedDistance: Float = 0.5f,
) {
    companion object {
        /** Symmetric grid of circles. */
        const val SYMMETRIC_GRID = 0
        /** Asymmetric grid of circles. */
        const val ASYMMETRIC_GRID = 1
    }
}

// =========================================================================
// wire-format decoders shared by the platform actuals
// =========================================================================

internal fun readU32Le(data: ByteArray, offset: Int): Int {
    var value = 0
    for (i in 0 until 4) {
        value = value or ((data[offset + i].toInt() and 0xFF) shl (8 * i))
    }
    return value
}

/**
 * Reads a packed string list starting at [offset]: [u32le count][per string:
 * u32le byteLen][utf8 bytes]. Returns the strings and the next offset.
 */
internal fun readPackedStringList(data: ByteArray, offset: Int): Pair<List<String>, Int> {
    val count = readU32Le(data, offset)
    var off = offset + 4
    val out = ArrayList<String>(count)
    repeat(count) {
        val len = readU32Le(data, off)
        off += 4
        out.add(data.decodeToString(off, off + len))
        off += len
    }
    return out to off
}

/** Decodes a multi-code result buffer: [u32le ok][packed decoded_info list]. */
internal fun decodeMultiBuffer(data: ByteArray): Pair<Boolean, List<String>> {
    val ok = readU32Le(data, 0) != 0
    val (info, _) = readPackedStringList(data, 4)
    return ok to info
}

/** Decodes a barcode with-type buffer: [u32le ok][info list][type list]. */
internal fun decodeWithTypeBuffer(data: ByteArray): BarcodeDecodeResult {
    val ok = readU32Le(data, 0) != 0
    val (info, next) = readPackedStringList(data, 4)
    val (type, _) = readPackedStringList(data, next)
    return BarcodeDecodeResult(ok, info, type)
}

/**
 * Decodes a CV_8UC1 wire Mat produced for vector<Mat> results: [u32le count]
 * then per Mat: rows, cols, type, dataLen, raw continuous bytes.
 */
internal fun decodeMatList(wire: Mat): List<Mat> {
    val bytes = wire.pixels
    var off = 0
    val count = readU32Le(bytes, off)
    off += 4
    val out = ArrayList<Mat>(count)
    repeat(count) {
        val rows = readU32Le(bytes, off)
        off += 4
        val cols = readU32Le(bytes, off)
        off += 4
        val type = readU32Le(bytes, off)
        off += 4
        val dataLen = readU32Le(bytes, off)
        off += 4
        val data = bytes.copyOfRange(off, off + dataLen)
        off += dataLen
        val m = mat(rows, cols, type)
        if (data.isNotEmpty()) {
            m.pixels = data
        }
        out.add(m)
    }
    return out
}

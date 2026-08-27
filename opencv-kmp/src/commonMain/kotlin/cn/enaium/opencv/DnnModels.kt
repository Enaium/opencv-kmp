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

import kotlin.math.max
import kotlin.math.min

// =========================================================================
// dnn data layout (cv::DataLayout, shared with Image2BlobParams)
// =========================================================================

/** Generic 2D layout (`cv::DATA_LAYOUT_ND`). */
const val DNN_LAYOUT_ND: Int = 1

/** NCHW 4D layout (`cv::DATA_LAYOUT_NCHW`). */
const val DNN_LAYOUT_NCHW: Int = 2

/** NCDHW 5D layout (`cv::DATA_LAYOUT_NCDHW`). */
const val DNN_LAYOUT_NCDHW: Int = 3

/** NHWC 4D layout (`cv::DATA_LAYOUT_NHWC`). */
const val DNN_LAYOUT_NHWC: Int = 4

/** NDHWC 5D layout (`cv::DATA_LAYOUT_NDHWC`). */
const val DNN_LAYOUT_NDHWC: Int = 5

/** Planar layout used at TF/TFLite parsing (`cv::DATA_LAYOUT_PLANAR`). */
const val DNN_LAYOUT_PLANAR: Int = 6

/** Block layout NC1HWC0 (`cv::DATA_LAYOUT_BLOCK`). */
const val DNN_LAYOUT_BLOCK: Int = 7

// =========================================================================
// Model (cv::dnn::Model)
// =========================================================================

/**
 * High-level API for neural networks (`cv::dnn::Model`).
 *
 * A [Model] loads a network from weights/config files (or wraps an existing
 * [Net]), stores the preprocessing parameters for its input frame and runs
 * the forward pass. All setters mutate the object in place and mirror the
 * fluent C++/Java `set*` methods.
 *
 * Model files are not bundled: the path-based factories return null when the
 * weights cannot be loaded (e.g. a missing file), matching [readNet]'s
 * contract. The [predict] forward pass additionally requires the input size
 * to be configured first via [setInputSize]/[setInputParams], otherwise
 * OpenCV throws ("Input size not specified").
 */
interface Model : AutoCloseable {

    /** Sets the input blob size; must be configured before [predict]. */
    fun setInputSize(width: Int, height: Int)

    /** Sets the input blob size from a [Size]. */
    fun setInputSize(size: Size)

    /** Mean values subtracted from each channel. */
    fun setInputMean(mean: Scalar)

    /** Multiplier applied to frame values. */
    fun setInputScale(scale: Scalar)

    /** Whether the image is cropped after resize. */
    fun setInputCrop(crop: Boolean)

    /** Whether the first and last channels are swapped. */
    fun setInputSwapRB(swapRB: Boolean)

    /** Names of the output layers returned by [predict]. */
    fun setOutputNames(outNames: List<String>)

    /**
     * Sets all preprocessing parameters at once:
     * `blob(n,c,y,x) = scale * resize(frame(y,x,c)) - mean(c)`.
     */
    fun setInputParams(
        scale: Double = 1.0,
        size: Size = Size(0, 0),
        mean: Scalar = Scalar(),
        swapRB: Boolean = false,
        crop: Boolean = false,
    )

    /**
     * Runs the network on [frame] and returns the output blobs. Every
     * returned Mat is owned by the caller and must be closed. Requires the
     * input size to have been set first.
     */
    fun predict(frame: Mat): List<Mat>

    /** Selects the inference backend (top-level `DNN_BACKEND_*` values). */
    fun setPreferableBackend(backendId: Int)

    /** Selects the inference target (top-level `DNN_TARGET_*` values). */
    fun setPreferableTarget(targetId: Int)

    /** Enables/disables Winograd-optimized convolutions. */
    fun enableWinograd(useWinograd: Boolean)

    /** Releases the native model; [close] is idempotent. */
    override fun close()
}

/**
 * Loads a [Model] from trained-weights [model] and optional [config] files;
 * null when the files cannot be loaded (no model files are bundled).
 */
expect fun model(model: String, config: String = ""): Model?

/** Wraps an already-loaded [net] in a [Model]. */
expect fun modelFromNet(net: Net): Model

// =========================================================================
// ClassificationModel
// =========================================================================

/**
 * High-level API for classification networks (`cv::dnn::ClassificationModel`).
 * Runs the forward pass and returns the top-1 prediction.
 */
interface ClassificationModel : Model {

    /**
     * When true, softmax is applied inside [classify] to normalize the
     * confidences into [0.0, 1.0]; defaults to false (the model is expected
     * to contain its own softmax layer).
     */
    var enableSoftmaxPostProcessing: Boolean

    /** Top-1 prediction for [frame]. */
    fun classify(frame: Mat): ClassificationResult

    override fun close()
}

/** Top-1 classification outcome. */
data class ClassificationResult(val classId: Int, val confidence: Float)

/**
 * Creates a [ClassificationModel] from [model]/[config] files; null when the
 * files cannot be loaded.
 */
expect fun classificationModel(model: String, config: String = ""): ClassificationModel?

/** Wraps an already-loaded [net] in a [ClassificationModel]. */
expect fun classificationModelFromNet(net: Net): ClassificationModel

// =========================================================================
// DetectionModel
// =========================================================================

/**
 * High-level API for object detection networks (`cv::dnn::DetectionModel`).
 * Supports SSD, Faster R-CNN and YOLO topologies.
 */
interface DetectionModel : Model {

    /**
     * When false (default) non-max suppression inside [detect] runs
     * per-class; when true it is applied across all classes.
     */
    var nmsAcrossClasses: Boolean

    /**
     * Runs detection on [frame]. [confThreshold] filters boxes by confidence
     * and [nmsThreshold] drives non-maximum suppression.
     */
    fun detect(
        frame: Mat,
        confThreshold: Float = 0.5f,
        nmsThreshold: Float = 0.0f,
    ): DetectionResult

    override fun close()
}

/** Bounding boxes plus the class id and confidence of each detection. */
data class DetectionResult(
    val classIds: List<Int>,
    val confidences: List<Float>,
    val boxes: List<Rect>,
)

/**
 * Creates a [DetectionModel] from [model]/[config] files; null when the
 * files cannot be loaded.
 */
expect fun detectionModel(model: String, config: String = ""): DetectionModel?

/** Wraps an already-loaded [net] in a [DetectionModel]. */
expect fun detectionModelFromNet(net: Net): DetectionModel

// =========================================================================
// KeypointsModel
// =========================================================================

/**
 * High-level API for keypoint models (`cv::dnn::KeypointsModel`); returns
 * the x/y coordinates of each detected keypoint.
 */
interface KeypointsModel : Model {

    /**
     * Runs the network on [frame]; [thresh] is the minimum confidence for a
     * keypoint to be kept.
     */
    fun estimate(frame: Mat, thresh: Float = 0.5f): List<Point2f>

    override fun close()
}

/**
 * Creates a [KeypointsModel] from [model]/[config] files; null when the
 * files cannot be loaded.
 */
expect fun keypointsModel(model: String, config: String = ""): KeypointsModel?

/** Wraps an already-loaded [net] in a [KeypointsModel]. */
expect fun keypointsModelFromNet(net: Net): KeypointsModel

// =========================================================================
// SegmentationModel
// =========================================================================

/**
 * High-level API for segmentation networks (`cv::dnn::SegmentationModel`);
 * returns the class prediction for every pixel.
 */
interface SegmentationModel : Model {

    /** Runs segmentation on [frame] and returns the class mask. */
    fun segment(frame: Mat): Mat

    override fun close()
}

/**
 * Creates a [SegmentationModel] from [model]/[config] files; null when the
 * files cannot be loaded.
 */
expect fun segmentationModel(model: String, config: String = ""): SegmentationModel?

/** Wraps an already-loaded [net] in a [SegmentationModel]. */
expect fun segmentationModelFromNet(net: Net): SegmentationModel

// =========================================================================
// TextDetectionModel (base; concrete factories are _DB / _EAST)
// =========================================================================

/**
 * Base class for text detection networks (`cv::dnn::TextDetectionModel`).
 * Cannot be constructed directly — use [textDetectionModelDb] or
 * [textDetectionModelEast].
 */
interface TextDetectionModel : Model {

    /**
     * Performs detection on [frame]. Each result is a quadrangle's 4 points
     * in this order: bottom-left, top-left, top-right, bottom-right.
     */
    fun detect(frame: Mat): TextDetections

    /**
     * Performs detection on [frame], returning rotated rectangles. Results
     * may be inaccurate under strong perspective transformations.
     */
    fun detectTextRectangles(frame: Mat): TextRectangles

    override fun close()
}

/** Text quadrangles (4 points each) with their confidences. */
data class TextDetections(
    val detections: List<List<Point>>,
    val confidences: List<Float>,
)

/** Rotated text rectangles with their confidences. */
data class TextRectangles(
    val detections: List<RotatedRect>,
    val confidences: List<Float>,
)

/**
 * Text detection compatible with the DB model (`cv::dnn::TextDetectionModel_DB`).
 */
interface TextDetectionModelDb : TextDetectionModel {

    /** Threshold of the binary map (usually 0.3). */
    var binaryThreshold: Float

    /** Threshold of the text polygons (default 0.5). */
    var polygonThreshold: Float

    /** Unclip ratio of the detected text region (usually 2.0). */
    var unclipRatio: Double

    /** Max number of output results (default 0 = unlimited). */
    var maxCandidates: Int

    override fun close()
}

/**
 * Creates a [TextDetectionModelDb] from [model]/[config] files; null when
 * the files cannot be loaded.
 */
expect fun textDetectionModelDb(model: String, config: String = ""): TextDetectionModelDb?

/** Wraps an already-loaded [net] in a [TextDetectionModelDb]. */
expect fun textDetectionModelDbFromNet(net: Net): TextDetectionModelDb

/**
 * Text detection compatible with the EAST model (`cv::dnn::TextDetectionModel_EAST`).
 */
interface TextDetectionModelEast : TextDetectionModel {

    /** Confidence threshold filtering boxes (default 0.5). */
    var confidenceThreshold: Float

    /** NMS threshold (default 0.0). */
    var nmsThreshold: Float

    override fun close()
}

/**
 * Creates a [TextDetectionModelEast] from [model]/[config] files; null when
 * the files cannot be loaded.
 */
expect fun textDetectionModelEast(model: String, config: String = ""): TextDetectionModelEast?

/** Wraps an already-loaded [net] in a [TextDetectionModelEast]. */
expect fun textDetectionModelEastFromNet(net: Net): TextDetectionModelEast

// =========================================================================
// TextRecognitionModel
// =========================================================================

/**
 * High-level API for text recognition networks (`cv::dnn::TextRecognitionModel`).
 * Supports CRNN-CTC topologies. [decodeType] and [vocabulary] must be set
 * before [recognize] is called.
 */
interface TextRecognitionModel : Model {

    /**
     * Decoding method: `"CTC-greedy"` or `"CTC-prefix-beam-search"`.
     * Defaults to an empty string until set.
     */
    var decodeType: String

    /** The vocabulary associated with the network. */
    var vocabulary: List<String>

    /** Options for the `"CTC-prefix-beam-search"` decode type. */
    fun setDecodeOptsCTCPrefixBeamSearch(beamSize: Int, vocPruneSize: Int = 0)

    /** Recognizes the text in [frame]. */
    fun recognize(frame: Mat): String

    /**
     * Recognizes each ROI of [frame] in turn. An empty [roiRects] list
     * recognizes the whole frame (mirroring the C++ overload).
     */
    fun recognize(frame: Mat, roiRects: List<Rect>): List<String>

    override fun close()
}

/**
 * Creates a [TextRecognitionModel] from [model]/[config] files; null when
 * the files cannot be loaded. Call [TextRecognitionModel.setDecodeType] and
 * [TextRecognitionModel.setVocabulary] after construction.
 */
expect fun textRecognitionModel(model: String, config: String = ""): TextRecognitionModel?

/** Wraps an already-loaded [net] in a [TextRecognitionModel]. */
expect fun textRecognitionModelFromNet(net: Net): TextRecognitionModel

// =========================================================================
// Image2BlobParams (pure Kotlin holder; no native object)
// =========================================================================

/**
 * Processing parameters for image-to-blob conversion
 * (`cv::dnn::Image2BlobParams`), in the same field order as the C++ struct.
 *
 * The preprocessing formula is `(input - mean) * scalefactor` with the
 * resize/crop behavior controlled by [size], [swapRB], [ddepth],
 * [datalayout] and [paddingmode]. Defaults match the C++ defaults:
 * scalefactor = 1, size = (0,0), mean = 0, swapRB = false, ddepth = CV_32F,
 * datalayout = [DNN_LAYOUT_NCHW], paddingmode = DNN_PMODE_NULL,
 * borderValue = 0.
 */
data class Image2BlobParams(
    var scalefactor: Scalar = Scalar(1.0, 1.0, 1.0, 1.0),
    var size: Size = Size(0, 0),
    var mean: Scalar = Scalar(),
    var swapRB: Boolean = false,
    var ddepth: Int = CV_32F,
    var datalayout: Int = DNN_LAYOUT_NCHW,
    var paddingmode: Int = DNN_PMODE_NULL,
    var borderValue: Scalar = Scalar(),
) {

    /**
     * Maps a rectangle in blob coordinates to original-image coordinates,
     * mirroring `cv::dnn::Image2BlobParams::blobRectToImageRect`.
     *
     * [imgSize] is the original input image size. The blob [size] must be
     * set to the actual blob size and the padding mode must be one of
     * [DNN_PMODE_NULL], [DNN_PMODE_CROP_CENTER] or [DNN_PMODE_LETTERBOX];
     * otherwise [IllegalArgumentException] is thrown.
     */
    fun blobRectToImageRect(rBlob: Rect, imgSize: Size): Rect =
        blobRectsToImageRects(listOf(rBlob), imgSize).single()

    /**
     * Maps blob-coordinate rectangles to original-image coordinates,
     * mirroring `cv::dnn::Image2BlobParams::blobRectsToImageRects`.
     * When [size] equals [imgSize] the rectangles are returned unchanged.
     */
    fun blobRectsToImageRects(rBlob: List<Rect>, imgSize: Size): List<Rect> {
        require(imgSize.width > 0 && imgSize.height > 0) { "imgSize must be non-empty" }
        if (size == imgSize) return rBlob
        require(size.width > 0 && size.height > 0) {
            "blob size must be set (size=$size) before mapping rectangles"
        }
        return when (paddingmode) {
            DNN_PMODE_CROP_CENTER -> {
                val resizeFactor = max(
                    size.width / imgSize.width.toFloat(),
                    size.height / imgSize.height.toFloat(),
                )
                rBlob.map { r ->
                    Rect(
                        ((r.x + 0.5f * (imgSize.width * resizeFactor - size.width)) / resizeFactor).toInt(),
                        ((r.y + 0.5f * (imgSize.height * resizeFactor - size.height)) / resizeFactor).toInt(),
                        (r.width / resizeFactor).toInt(),
                        (r.height / resizeFactor).toInt(),
                    )
                }
            }
            DNN_PMODE_LETTERBOX -> {
                val resizeFactor = min(
                    size.width / imgSize.width.toFloat(),
                    size.height / imgSize.height.toFloat(),
                )
                val rh = (imgSize.height * resizeFactor).toInt()
                val rw = (imgSize.width * resizeFactor).toInt()
                val top = (size.height - rh) / 2
                val left = (size.width - rw) / 2
                rBlob.map { r ->
                    Rect(
                        ((r.x - left) / resizeFactor).toInt(),
                        ((r.y - top) / resizeFactor).toInt(),
                        (r.width / resizeFactor).toInt(),
                        (r.height / resizeFactor).toInt(),
                    )
                }
            }
            DNN_PMODE_NULL -> rBlob.map { r ->
                Rect(
                    (r.x * imgSize.width.toFloat() / size.width).toInt(),
                    (r.y * imgSize.height.toFloat() / size.height).toInt(),
                    (r.width * imgSize.width.toFloat() / size.width).toInt(),
                    (r.height * imgSize.height.toFloat() / size.height).toInt(),
                )
            }
            else -> throw IllegalArgumentException("Unknown padding mode $paddingmode")
        }
    }
}

// =========================================================================
// wire decoders shared by the native and JVM backends
//
// The C shim returns variable-length results as malloc'd little-endian
// buffers; both backends materialize them as ByteArray and decode here.
// =========================================================================

/** Decodes a [u32 count][count x i32]… detection buffer. */
internal fun decodeDetectionBuffer(buffer: ByteArray): DetectionResult {
    var off = 0
    val count = buffer.readIntLE(off)
    off += 4
    val classIds = ArrayList<Int>(count)
    repeat(count) {
        classIds.add(buffer.readIntLE(off))
        off += 4
    }
    val confidences = ArrayList<Float>(count)
    repeat(count) {
        confidences.add(buffer.readFloatLE(off))
        off += 4
    }
    val boxes = ArrayList<Rect>(count)
    repeat(count) {
        val x = buffer.readIntLE(off)
        val y = buffer.readIntLE(off + 4)
        val w = buffer.readIntLE(off + 8)
        val h = buffer.readIntLE(off + 12)
        boxes.add(Rect(x, y, w, h))
        off += 16
    }
    return DetectionResult(classIds, confidences, boxes)
}

/** Decodes a [u32 count][count x 2 x f32] keypoint buffer. */
internal fun decodeKeypointsBuffer(buffer: ByteArray): List<Point2f> {
    var off = 0
    val count = buffer.readIntLE(off)
    off += 4
    val points = ArrayList<Point2f>(count)
    repeat(count) {
        points.add(Point2f(buffer.readFloatLE(off), buffer.readFloatLE(off + 4)))
        off += 8
    }
    return points
}

/** Decodes a [u32 count][per: u32 npts + npts x 2 x i32][count x f32] buffer. */
internal fun decodeTextDetectionsBuffer(buffer: ByteArray): TextDetections {
    var off = 0
    val count = buffer.readIntLE(off)
    off += 4
    val detections = ArrayList<List<Point>>(count)
    repeat(count) {
        val npts = buffer.readIntLE(off)
        off += 4
        val quad = ArrayList<Point>(npts)
        repeat(npts) {
            quad.add(Point(buffer.readIntLE(off), buffer.readIntLE(off + 4)))
            off += 8
        }
        detections.add(quad)
    }
    val confidences = ArrayList<Float>(count)
    repeat(count) {
        confidences.add(buffer.readFloatLE(off))
        off += 4
    }
    return TextDetections(detections, confidences)
}

/** Decodes a [u32 count][count x 5 x f32][count x f32] buffer. */
internal fun decodeTextRectanglesBuffer(buffer: ByteArray): TextRectangles {
    var off = 0
    val count = buffer.readIntLE(off)
    off += 4
    val detections = ArrayList<RotatedRect>(count)
    repeat(count) {
        detections.add(
            RotatedRect(
                centerX = buffer.readFloatLE(off).toDouble(),
                centerY = buffer.readFloatLE(off + 4).toDouble(),
                width = buffer.readFloatLE(off + 8).toDouble(),
                height = buffer.readFloatLE(off + 12).toDouble(),
                angle = buffer.readFloatLE(off + 16).toDouble(),
            ),
        )
        off += 20
    }
    val confidences = ArrayList<Float>(count)
    repeat(count) {
        confidences.add(buffer.readFloatLE(off))
        off += 4
    }
    return TextRectangles(detections, confidences)
}

/** Decodes a [u32 count][per: u32 len + len UTF-8 bytes] string list. */
internal fun decodeStringListBuffer(buffer: ByteArray): List<String> {
    var off = 0
    val count = buffer.readIntLE(off)
    off += 4
    val strings = ArrayList<String>(count)
    repeat(count) {
        val len = buffer.readIntLE(off)
        off += 4
        strings.add(buffer.decodeToString(off, off + len))
        off += len
    }
    return strings
}

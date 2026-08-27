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
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package cn.enaium.opencv

import cvk.cvk_classification_model_classify
import cvk.cvk_classification_model_create
import cvk.cvk_classification_model_create_from_net
import cvk.cvk_classification_model_get_enable_softmax_post_processing
import cvk.cvk_classification_model_set_enable_softmax_post_processing
import cvk.cvk_detection_model_create
import cvk.cvk_detection_model_create_from_net
import cvk.cvk_detection_model_detect
import cvk.cvk_detection_model_get_nms_across_classes
import cvk.cvk_detection_model_set_nms_across_classes
import cvk.cvk_free_buffer
import cvk.cvk_keypoints_model_create
import cvk.cvk_keypoints_model_create_from_net
import cvk.cvk_keypoints_model_estimate
import cvk.cvk_last_error
import cvk.cvk_model_create
import cvk.cvk_model_create_from_net
import cvk.cvk_model_enable_winograd
import cvk.cvk_model_predict
import cvk.cvk_model_release
import cvk.cvk_model_set_input_crop
import cvk.cvk_model_set_input_mean
import cvk.cvk_model_set_input_params
import cvk.cvk_model_set_input_scale
import cvk.cvk_model_set_input_size
import cvk.cvk_model_set_input_swap_rb
import cvk.cvk_model_set_output_names
import cvk.cvk_model_set_preferable_backend
import cvk.cvk_model_set_preferable_target
import cvk.cvk_model_t
import cvk.cvk_segmentation_model_create
import cvk.cvk_segmentation_model_create_from_net
import cvk.cvk_segmentation_model_segment
import cvk.cvk_text_detection_model_db_create
import cvk.cvk_text_detection_model_db_create_from_net
import cvk.cvk_text_detection_model_db_get_binary_threshold
import cvk.cvk_text_detection_model_db_get_max_candidates
import cvk.cvk_text_detection_model_db_get_polygon_threshold
import cvk.cvk_text_detection_model_db_get_unclip_ratio
import cvk.cvk_text_detection_model_db_set_binary_threshold
import cvk.cvk_text_detection_model_db_set_max_candidates
import cvk.cvk_text_detection_model_db_set_polygon_threshold
import cvk.cvk_text_detection_model_db_set_unclip_ratio
import cvk.cvk_text_detection_model_detect
import cvk.cvk_text_detection_model_detect_text_rectangles
import cvk.cvk_text_detection_model_east_create
import cvk.cvk_text_detection_model_east_create_from_net
import cvk.cvk_text_detection_model_east_get_confidence_threshold
import cvk.cvk_text_detection_model_east_get_nms_threshold
import cvk.cvk_text_detection_model_east_set_confidence_threshold
import cvk.cvk_text_detection_model_east_set_nms_threshold
import cvk.cvk_text_recognition_model_create
import cvk.cvk_text_recognition_model_create_from_net
import cvk.cvk_text_recognition_model_get_decode_type
import cvk.cvk_text_recognition_model_get_vocabulary
import cvk.cvk_text_recognition_model_recognize
import cvk.cvk_text_recognition_model_recognize_rois
import cvk.cvk_text_recognition_model_set_decode_opts_ctc_prefix_beam_search
import cvk.cvk_text_recognition_model_set_decode_type
import cvk.cvk_text_recognition_model_set_vocabulary
import kotlinx.cinterop.toKString
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.FloatVar
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.cstr
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import kotlin.concurrent.Volatile
import platform.posix.size_t
import platform.posix.size_tVar

// =========================================================================
// native (cinterop) Model family implementation
// =========================================================================

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

/**
 * Runs [acquire] to obtain a malloc'd cvk_ wire buffer, copies it into a
 * ByteArray, frees it and decodes with [decode].
 */
private inline fun <R> withModelBuffer(
    operation: String,
    acquire: (outLen: CPointer<size_tVar>) -> CPointer<UByteVar>?,
    decode: (ByteArray) -> R,
): R = memScoped {
    val outLen = alloc<size_tVar>()
    val buffer = acquire(outLen.ptr) ?: throw OpenCVException(operation, lastNativeError())
    try {
        decode(buffer.readBytes(outLen.value.toInt()))
    } finally {
        cvk_free_buffer(buffer)
    }
}

/**
 * Shared base-Model implementation. One raw `cvk_model_t` handle backs every
 * model class (the shim stores the concrete cv::dnn model inside it), so a
 * single release function frees any of them.
 */
internal open class NativeModelBase(
    @Volatile private var raw: CPointer<cvk_model_t>?,
) : Model {

    internal fun check(): CPointer<cvk_model_t> =
        raw ?: throw IllegalStateException("Model is closed")

    override fun setInputSize(width: Int, height: Int) {
        cvk_model_set_input_size(check(), width, height)
    }

    override fun setInputSize(size: Size) = setInputSize(size.width, size.height)

    override fun setInputMean(mean: Scalar) {
        cvk_model_set_input_mean(check(), mean.toCvk())
    }

    override fun setInputScale(scale: Scalar) {
        cvk_model_set_input_scale(check(), scale.toCvk())
    }

    override fun setInputCrop(crop: Boolean) {
        cvk_model_set_input_crop(check(), if (crop) 1 else 0)
    }

    override fun setInputSwapRB(swapRB: Boolean) {
        cvk_model_set_input_swap_rb(check(), if (swapRB) 1 else 0)
    }

    override fun setOutputNames(outNames: List<String>) {
        memScoped {
            val names = allocArray<CPointerVar<ByteVar>>(outNames.size)
            val cstrings = outNames.map { it.cstr }
            cstrings.forEachIndexed { i, c -> names[i] = c.ptr }
            cvk_model_set_output_names(check(), names, outNames.size)
        }
    }

    override fun setInputParams(
        scale: Double,
        size: Size,
        mean: Scalar,
        swapRB: Boolean,
        crop: Boolean,
    ) {
        cvk_model_set_input_params(
            check(), scale, size.width, size.height,
            mean.toCvk(), if (swapRB) 1 else 0, if (crop) 1 else 0,
        )
    }

    override fun predict(frame: Mat): List<Mat> = memScoped {
        val count = alloc<size_tVar>()
        val handles = cvk_model_predict(check(), frame.nativeHandle(), count.ptr)
            ?: throw OpenCVException("model.predict", lastNativeError())
        try {
            List(count.value.toInt()) { index -> NativeMat(handles[index]) }
        } finally {
            cvk_free_buffer(handles.reinterpret<UByteVar>())
        }
    }

    override fun setPreferableBackend(backendId: Int) {
        cvk_model_set_preferable_backend(check(), backendId)
    }

    override fun setPreferableTarget(targetId: Int) {
        cvk_model_set_preferable_target(check(), targetId)
    }

    override fun enableWinograd(useWinograd: Boolean) {
        cvk_model_enable_winograd(check(), if (useWinograd) 1 else 0)
    }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_model_release(handle)
    }
}

internal class NativeClassificationModel(private val base: NativeModelBase) :
    ClassificationModel, Model by base {

    override var enableSoftmaxPostProcessing: Boolean
        get() = cvk_classification_model_get_enable_softmax_post_processing(base.check()) != 0
        set(value) {
            cvk_classification_model_set_enable_softmax_post_processing(
                base.check(), if (value) 1 else 0,
            )
        }

    override fun classify(frame: Mat): ClassificationResult = memScoped {
        val classId = alloc<IntVar>()
        val confidence = alloc<FloatVar>()
        val ok = cvk_classification_model_classify(
            base.check(), frame.nativeHandle(), classId.ptr, confidence.ptr,
        )
        if (ok == 0) throw OpenCVException("classificationModel.classify", lastNativeError())
        ClassificationResult(classId.value, confidence.value)
    }
}

internal class NativeDetectionModel(private val base: NativeModelBase) :
    DetectionModel, Model by base {

    override var nmsAcrossClasses: Boolean
        get() = cvk_detection_model_get_nms_across_classes(base.check()) != 0
        set(value) {
            cvk_detection_model_set_nms_across_classes(base.check(), if (value) 1 else 0)
        }

    override fun detect(frame: Mat, confThreshold: Float, nmsThreshold: Float): DetectionResult =
        withModelBuffer(
            "detectionModel.detect",
            { outLen ->
                cvk_detection_model_detect(
                    base.check(), frame.nativeHandle(), confThreshold, nmsThreshold, outLen,
                )
            },
            ::decodeDetectionBuffer,
        )
}

internal class NativeKeypointsModel(private val base: NativeModelBase) :
    KeypointsModel, Model by base {

    override fun estimate(frame: Mat, thresh: Float): List<Point2f> = withModelBuffer(
        "keypointsModel.estimate",
        { outLen -> cvk_keypoints_model_estimate(base.check(), frame.nativeHandle(), thresh, outLen) },
        ::decodeKeypointsBuffer,
    )
}

internal class NativeSegmentationModel(private val base: NativeModelBase) :
    SegmentationModel, Model by base {

    override fun segment(frame: Mat): Mat =
        nativeMat(cvk_segmentation_model_segment(base.check(), frame.nativeHandle()), "segment")
}

internal class NativeTextDetectionModelDb(private val base: NativeModelBase) :
    TextDetectionModelDb, Model by base {

    override var binaryThreshold: Float
        get() = cvk_text_detection_model_db_get_binary_threshold(base.check())
        set(value) { cvk_text_detection_model_db_set_binary_threshold(base.check(), value) }

    override var polygonThreshold: Float
        get() = cvk_text_detection_model_db_get_polygon_threshold(base.check())
        set(value) { cvk_text_detection_model_db_set_polygon_threshold(base.check(), value) }

    override var unclipRatio: Double
        get() = cvk_text_detection_model_db_get_unclip_ratio(base.check())
        set(value) { cvk_text_detection_model_db_set_unclip_ratio(base.check(), value) }

    override var maxCandidates: Int
        get() = cvk_text_detection_model_db_get_max_candidates(base.check())
        set(value) { cvk_text_detection_model_db_set_max_candidates(base.check(), value) }

    override fun detect(frame: Mat): TextDetections = withModelBuffer(
        "textDetectionModel.detect",
        { outLen -> cvk_text_detection_model_detect(base.check(), frame.nativeHandle(), outLen) },
        ::decodeTextDetectionsBuffer,
    )

    override fun detectTextRectangles(frame: Mat): TextRectangles = withModelBuffer(
        "textDetectionModel.detectTextRectangles",
        { outLen ->
            cvk_text_detection_model_detect_text_rectangles(base.check(), frame.nativeHandle(), outLen)
        },
        ::decodeTextRectanglesBuffer,
    )
}

internal class NativeTextDetectionModelEast(private val base: NativeModelBase) :
    TextDetectionModelEast, Model by base {

    override var confidenceThreshold: Float
        get() = cvk_text_detection_model_east_get_confidence_threshold(base.check())
        set(value) { cvk_text_detection_model_east_set_confidence_threshold(base.check(), value) }

    override var nmsThreshold: Float
        get() = cvk_text_detection_model_east_get_nms_threshold(base.check())
        set(value) { cvk_text_detection_model_east_set_nms_threshold(base.check(), value) }

    override fun detect(frame: Mat): TextDetections = withModelBuffer(
        "textDetectionModel.detect",
        { outLen -> cvk_text_detection_model_detect(base.check(), frame.nativeHandle(), outLen) },
        ::decodeTextDetectionsBuffer,
    )

    override fun detectTextRectangles(frame: Mat): TextRectangles = withModelBuffer(
        "textDetectionModel.detectTextRectangles",
        { outLen ->
            cvk_text_detection_model_detect_text_rectangles(base.check(), frame.nativeHandle(), outLen)
        },
        ::decodeTextRectanglesBuffer,
    )
}

internal class NativeTextRecognitionModel(private val base: NativeModelBase) :
    TextRecognitionModel, Model by base {

    override var decodeType: String
        get() = cvk_text_recognition_model_get_decode_type(base.check())?.toKString()
            ?: throw OpenCVException("textRecognitionModel.decodeType", lastNativeError())
        set(value) { cvk_text_recognition_model_set_decode_type(base.check(), value) }

    override var vocabulary: List<String>
        get() = withModelBuffer(
            "textRecognitionModel.vocabulary",
            { outLen -> cvk_text_recognition_model_get_vocabulary(base.check(), outLen) },
            ::decodeStringListBuffer,
        )
        set(value) {
            memScoped {
                val entries = allocArray<CPointerVar<ByteVar>>(value.size)
                val cstrings = value.map { it.cstr }
                cstrings.forEachIndexed { i, c -> entries[i] = c.ptr }
                cvk_text_recognition_model_set_vocabulary(base.check(), entries, value.size)
            }
        }

    override fun setDecodeOptsCTCPrefixBeamSearch(beamSize: Int, vocPruneSize: Int) {
        cvk_text_recognition_model_set_decode_opts_ctc_prefix_beam_search(
            base.check(), beamSize, vocPruneSize,
        )
    }

    override fun recognize(frame: Mat): String =
        cvk_text_recognition_model_recognize(base.check(), frame.nativeHandle())?.toKString()
            ?: throw OpenCVException("textRecognitionModel.recognize", lastNativeError())

    override fun recognize(frame: Mat, roiRects: List<Rect>): List<String> {
        val flat = IntArray(roiRects.size * 4)
        roiRects.forEachIndexed { i, r ->
            flat[i * 4] = r.x
            flat[i * 4 + 1] = r.y
            flat[i * 4 + 2] = r.width
            flat[i * 4 + 3] = r.height
        }
        return withModelBuffer(
            "textRecognitionModel.recognize(roiRects)",
            { outLen ->
                flat.usePinned { pinned ->
                    cvk_text_recognition_model_recognize_rois(
                        base.check(), frame.nativeHandle(),
                        pinned.addressOf(0), roiRects.size, outLen,
                    )
                }
            },
            ::decodeStringListBuffer,
        )
    }
}

// =========================================================================
// actual factories
// =========================================================================

actual fun model(model: String, config: String): Model? =
    cvk_model_create(model, config)?.let { NativeModelBase(it) }

actual fun modelFromNet(net: Net): Model =
    NativeModelBase(
        cvk_model_create_from_net(net.nativeHandle())
            ?: throw OpenCVException("modelFromNet", lastNativeError()),
    )

actual fun classificationModel(model: String, config: String): ClassificationModel? =
    cvk_classification_model_create(model, config)?.let {
        NativeClassificationModel(NativeModelBase(it))
    }

actual fun classificationModelFromNet(net: Net): ClassificationModel =
    NativeClassificationModel(
        NativeModelBase(
            cvk_classification_model_create_from_net(net.nativeHandle())
                ?: throw OpenCVException("classificationModelFromNet", lastNativeError()),
        ),
    )

actual fun detectionModel(model: String, config: String): DetectionModel? =
    cvk_detection_model_create(model, config)?.let { NativeDetectionModel(NativeModelBase(it)) }

actual fun detectionModelFromNet(net: Net): DetectionModel =
    NativeDetectionModel(
        NativeModelBase(
            cvk_detection_model_create_from_net(net.nativeHandle())
                ?: throw OpenCVException("detectionModelFromNet", lastNativeError()),
        ),
    )

actual fun keypointsModel(model: String, config: String): KeypointsModel? =
    cvk_keypoints_model_create(model, config)?.let { NativeKeypointsModel(NativeModelBase(it)) }

actual fun keypointsModelFromNet(net: Net): KeypointsModel =
    NativeKeypointsModel(
        NativeModelBase(
            cvk_keypoints_model_create_from_net(net.nativeHandle())
                ?: throw OpenCVException("keypointsModelFromNet", lastNativeError()),
        ),
    )

actual fun segmentationModel(model: String, config: String): SegmentationModel? =
    cvk_segmentation_model_create(model, config)?.let { NativeSegmentationModel(NativeModelBase(it)) }

actual fun segmentationModelFromNet(net: Net): SegmentationModel =
    NativeSegmentationModel(
        NativeModelBase(
            cvk_segmentation_model_create_from_net(net.nativeHandle())
                ?: throw OpenCVException("segmentationModelFromNet", lastNativeError()),
        ),
    )

actual fun textDetectionModelDb(model: String, config: String): TextDetectionModelDb? =
    cvk_text_detection_model_db_create(model, config)?.let {
        NativeTextDetectionModelDb(NativeModelBase(it))
    }

actual fun textDetectionModelDbFromNet(net: Net): TextDetectionModelDb =
    NativeTextDetectionModelDb(
        NativeModelBase(
            cvk_text_detection_model_db_create_from_net(net.nativeHandle())
                ?: throw OpenCVException("textDetectionModelDbFromNet", lastNativeError()),
        ),
    )

actual fun textDetectionModelEast(model: String, config: String): TextDetectionModelEast? =
    cvk_text_detection_model_east_create(model, config)?.let {
        NativeTextDetectionModelEast(NativeModelBase(it))
    }

actual fun textDetectionModelEastFromNet(net: Net): TextDetectionModelEast =
    NativeTextDetectionModelEast(
        NativeModelBase(
            cvk_text_detection_model_east_create_from_net(net.nativeHandle())
                ?: throw OpenCVException("textDetectionModelEastFromNet", lastNativeError()),
        ),
    )

actual fun textRecognitionModel(model: String, config: String): TextRecognitionModel? =
    cvk_text_recognition_model_create(model, config)?.let {
        NativeTextRecognitionModel(NativeModelBase(it))
    }

actual fun textRecognitionModelFromNet(net: Net): TextRecognitionModel =
    NativeTextRecognitionModel(
        NativeModelBase(
            cvk_text_recognition_model_create_from_net(net.nativeHandle())
                ?: throw OpenCVException("textRecognitionModelFromNet", lastNativeError()),
        ),
    )

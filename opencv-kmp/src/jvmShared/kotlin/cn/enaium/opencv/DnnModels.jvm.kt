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

import kotlin.concurrent.Volatile

// =========================================================================
// JVM (JNI-backed) implementation of the dnn model family.
// Model handles are jlong pointers to cvk_model_t; the shim stores the
// concrete cv::dnn model inside that handle, so one release function and one
// base wrapper class serve every model type.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()

/** Wraps a raw handle; throws with the native error text when it is 0. */
private fun jvmModel(ptr: Long, operation: String): Model =
    if (ptr != 0L) JvmModelBase(ptr) else throw OpenCVException(operation, lastNativeError())

/** Requires a non-zero model handle, translating failure into an exception. */
private fun requireModel(ptr: Long, operation: String): Long =
    if (ptr != 0L) ptr else throw OpenCVException(operation, lastNativeError())

/**
 * Shared base-Model implementation. Every model class is a [JvmModelBase]
 * delegate plus its own typed members; [close] routes to the single
 * [JniDnn2.modelRelease].
 */
internal open class JvmModelBase(
    @Volatile private var handle: Long,
) : Model {

    internal fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Model is closed")

    override fun setInputSize(width: Int, height: Int) {
        JniDnn2.modelSetInputSize(check(), width, height)
    }

    override fun setInputSize(size: Size) = setInputSize(size.width, size.height)

    override fun setInputMean(mean: Scalar) {
        JniDnn2.modelSetInputMean(check(), mean.v0, mean.v1, mean.v2, mean.v3)
    }

    override fun setInputScale(scale: Scalar) {
        JniDnn2.modelSetInputScale(check(), scale.v0, scale.v1, scale.v2, scale.v3)
    }

    override fun setInputCrop(crop: Boolean) {
        JniDnn2.modelSetInputCrop(check(), crop)
    }

    override fun setInputSwapRB(swapRB: Boolean) {
        JniDnn2.modelSetInputSwapRB(check(), swapRB)
    }

    override fun setOutputNames(outNames: List<String>) {
        JniDnn2.modelSetOutputNames(check(), outNames.toTypedArray())
    }

    override fun setInputParams(
        scale: Double,
        size: Size,
        mean: Scalar,
        swapRB: Boolean,
        crop: Boolean,
    ) {
        JniDnn2.modelSetInputParams(
            check(), scale, size.width, size.height,
            mean.v0, mean.v1, mean.v2, mean.v3, swapRB, crop,
        )
    }

    override fun predict(frame: Mat): List<Mat> =
        (JniDnn2.modelPredict(check(), handleOf(frame))
            ?: throw OpenCVException("model.predict", lastNativeError())).map { JvmMat(it) }

    override fun setPreferableBackend(backendId: Int) {
        JniDnn2.modelSetPreferableBackend(check(), backendId)
    }

    override fun setPreferableTarget(targetId: Int) {
        JniDnn2.modelSetPreferableTarget(check(), targetId)
    }

    override fun enableWinograd(useWinograd: Boolean) {
        JniDnn2.modelEnableWinograd(check(), useWinograd)
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniDnn2.modelRelease(h)
        }
    }
}

internal class JvmClassificationModel(private val base: JvmModelBase) :
    ClassificationModel, Model by base {

    override var enableSoftmaxPostProcessing: Boolean
        get() = JniDnn2.classificationModelGetEnableSoftmaxPostProcessing(base.check())
        set(value) { JniDnn2.classificationModelSetEnableSoftmaxPostProcessing(base.check(), value) }

    override fun classify(frame: Mat): ClassificationResult {
        val out = JniDnn2.classificationModelClassify(base.check(), handleOf(frame))
            ?: throw OpenCVException("classificationModel.classify", lastNativeError())
        return ClassificationResult(out[0].toInt(), out[1].toFloat())
    }
}

internal class JvmDetectionModel(private val base: JvmModelBase) :
    DetectionModel, Model by base {

    override var nmsAcrossClasses: Boolean
        get() = JniDnn2.detectionModelGetNmsAcrossClasses(base.check())
        set(value) { JniDnn2.detectionModelSetNmsAcrossClasses(base.check(), value) }

    override fun detect(frame: Mat, confThreshold: Float, nmsThreshold: Float): DetectionResult =
        decodeDetectionBuffer(
            JniDnn2.detectionModelDetect(
                base.check(), handleOf(frame), confThreshold, nmsThreshold,
            ) ?: throw OpenCVException("detectionModel.detect", lastNativeError()),
        )
}

internal class JvmKeypointsModel(private val base: JvmModelBase) :
    KeypointsModel, Model by base {

    override fun estimate(frame: Mat, thresh: Float): List<Point2f> =
        decodeKeypointsBuffer(
            JniDnn2.keypointsModelEstimate(base.check(), handleOf(frame), thresh)
                ?: throw OpenCVException("keypointsModel.estimate", lastNativeError()),
        )
}

internal class JvmSegmentationModel(private val base: JvmModelBase) :
    SegmentationModel, Model by base {

    override fun segment(frame: Mat): Mat =
        jvmMat(JniDnn2.segmentationModelSegment(base.check(), handleOf(frame)), "segment")
}

internal class JvmTextDetectionModelDb(private val base: JvmModelBase) :
    TextDetectionModelDb, Model by base {

    override var binaryThreshold: Float
        get() = JniDnn2.textDetectionModelDbGetBinaryThreshold(base.check())
        set(value) { JniDnn2.textDetectionModelDbSetBinaryThreshold(base.check(), value) }

    override var polygonThreshold: Float
        get() = JniDnn2.textDetectionModelDbGetPolygonThreshold(base.check())
        set(value) { JniDnn2.textDetectionModelDbSetPolygonThreshold(base.check(), value) }

    override var unclipRatio: Double
        get() = JniDnn2.textDetectionModelDbGetUnclipRatio(base.check())
        set(value) { JniDnn2.textDetectionModelDbSetUnclipRatio(base.check(), value) }

    override var maxCandidates: Int
        get() = JniDnn2.textDetectionModelDbGetMaxCandidates(base.check())
        set(value) { JniDnn2.textDetectionModelDbSetMaxCandidates(base.check(), value) }

    override fun detect(frame: Mat): TextDetections =
        decodeTextDetectionsBuffer(
            JniDnn2.textDetectionModelDetect(base.check(), handleOf(frame))
                ?: throw OpenCVException("textDetectionModel.detect", lastNativeError()),
        )

    override fun detectTextRectangles(frame: Mat): TextRectangles =
        decodeTextRectanglesBuffer(
            JniDnn2.textDetectionModelDetectTextRectangles(base.check(), handleOf(frame))
                ?: throw OpenCVException("textDetectionModel.detectTextRectangles", lastNativeError()),
        )
}

internal class JvmTextDetectionModelEast(private val base: JvmModelBase) :
    TextDetectionModelEast, Model by base {

    override var confidenceThreshold: Float
        get() = JniDnn2.textDetectionModelEastGetConfidenceThreshold(base.check())
        set(value) { JniDnn2.textDetectionModelEastSetConfidenceThreshold(base.check(), value) }

    override var nmsThreshold: Float
        get() = JniDnn2.textDetectionModelEastGetNmsThreshold(base.check())
        set(value) { JniDnn2.textDetectionModelEastSetNmsThreshold(base.check(), value) }

    override fun detect(frame: Mat): TextDetections =
        decodeTextDetectionsBuffer(
            JniDnn2.textDetectionModelDetect(base.check(), handleOf(frame))
                ?: throw OpenCVException("textDetectionModel.detect", lastNativeError()),
        )

    override fun detectTextRectangles(frame: Mat): TextRectangles =
        decodeTextRectanglesBuffer(
            JniDnn2.textDetectionModelDetectTextRectangles(base.check(), handleOf(frame))
                ?: throw OpenCVException("textDetectionModel.detectTextRectangles", lastNativeError()),
        )
}

internal class JvmTextRecognitionModel(private val base: JvmModelBase) :
    TextRecognitionModel, Model by base {

    override var decodeType: String
        get() = JniDnn2.textRecognitionModelGetDecodeType(base.check())
            ?: throw OpenCVException("textRecognitionModel.decodeType", lastNativeError())
        set(value) { JniDnn2.textRecognitionModelSetDecodeType(base.check(), value) }

    override var vocabulary: List<String>
        get() = JniDnn2.textRecognitionModelGetVocabulary(base.check())
            ?.toList()
            ?: throw OpenCVException("textRecognitionModel.vocabulary", lastNativeError())
        set(value) { JniDnn2.textRecognitionModelSetVocabulary(base.check(), value.toTypedArray()) }

    override fun setDecodeOptsCTCPrefixBeamSearch(beamSize: Int, vocPruneSize: Int) {
        JniDnn2.textRecognitionModelSetDecodeOptsCtcPrefixBeamSearch(
            base.check(), beamSize, vocPruneSize,
        )
    }

    override fun recognize(frame: Mat): String =
        JniDnn2.textRecognitionModelRecognize(base.check(), handleOf(frame))
            ?: throw OpenCVException("textRecognitionModel.recognize", lastNativeError())

    override fun recognize(frame: Mat, roiRects: List<Rect>): List<String> {
        val flat = IntArray(roiRects.size * 4)
        roiRects.forEachIndexed { i, r ->
            flat[i * 4] = r.x
            flat[i * 4 + 1] = r.y
            flat[i * 4 + 2] = r.width
            flat[i * 4 + 3] = r.height
        }
        return decodeStringListBuffer(
            JniDnn2.textRecognitionModelRecognizeRois(base.check(), handleOf(frame), flat)
                ?: throw OpenCVException("textRecognitionModel.recognize(roiRects)", lastNativeError()),
        )
    }
}

// =========================================================================
// actual factories
// =========================================================================

actual fun model(model: String, config: String): Model? =
    JniDnn2.modelCreate(model, config).takeIf { it != 0L }?.let(::JvmModelBase)

actual fun modelFromNet(net: Net): Model =
    jvmModel(JniDnn2.modelCreateFromNet(netHandleOf(net)), "modelFromNet")

actual fun classificationModel(model: String, config: String): ClassificationModel? =
    JniDnn2.classificationModelCreate(model, config).takeIf { it != 0L }
        ?.let { JvmClassificationModel(JvmModelBase(it)) }

actual fun classificationModelFromNet(net: Net): ClassificationModel =
    JvmClassificationModel(
        JvmModelBase(
            requireModel(JniDnn2.classificationModelCreateFromNet(netHandleOf(net)), "classificationModelFromNet failed"),
        ),
    )

actual fun detectionModel(model: String, config: String): DetectionModel? =
    JniDnn2.detectionModelCreate(model, config).takeIf { it != 0L }
        ?.let { JvmDetectionModel(JvmModelBase(it)) }

actual fun detectionModelFromNet(net: Net): DetectionModel =
    JvmDetectionModel(
        JvmModelBase(
            requireModel(JniDnn2.detectionModelCreateFromNet(netHandleOf(net)), "detectionModelFromNet failed"),
        ),
    )

actual fun keypointsModel(model: String, config: String): KeypointsModel? =
    JniDnn2.keypointsModelCreate(model, config).takeIf { it != 0L }
        ?.let { JvmKeypointsModel(JvmModelBase(it)) }

actual fun keypointsModelFromNet(net: Net): KeypointsModel =
    JvmKeypointsModel(
        JvmModelBase(
            requireModel(JniDnn2.keypointsModelCreateFromNet(netHandleOf(net)), "keypointsModelFromNet failed"),
        ),
    )

actual fun segmentationModel(model: String, config: String): SegmentationModel? =
    JniDnn2.segmentationModelCreate(model, config).takeIf { it != 0L }
        ?.let { JvmSegmentationModel(JvmModelBase(it)) }

actual fun segmentationModelFromNet(net: Net): SegmentationModel =
    JvmSegmentationModel(
        JvmModelBase(
            requireModel(JniDnn2.segmentationModelCreateFromNet(netHandleOf(net)), "segmentationModelFromNet failed"),
        ),
    )

actual fun textDetectionModelDb(model: String, config: String): TextDetectionModelDb? =
    JniDnn2.textDetectionModelDbCreate(model, config).takeIf { it != 0L }
        ?.let { JvmTextDetectionModelDb(JvmModelBase(it)) }

actual fun textDetectionModelDbFromNet(net: Net): TextDetectionModelDb =
    JvmTextDetectionModelDb(
        JvmModelBase(
            requireModel(JniDnn2.textDetectionModelDbCreateFromNet(netHandleOf(net)), "textDetectionModelDbFromNet failed"),
        ),
    )

actual fun textDetectionModelEast(model: String, config: String): TextDetectionModelEast? =
    JniDnn2.textDetectionModelEastCreate(model, config).takeIf { it != 0L }
        ?.let { JvmTextDetectionModelEast(JvmModelBase(it)) }

actual fun textDetectionModelEastFromNet(net: Net): TextDetectionModelEast =
    JvmTextDetectionModelEast(
        JvmModelBase(
            requireModel(JniDnn2.textDetectionModelEastCreateFromNet(netHandleOf(net)), "textDetectionModelEastFromNet failed"),
        ),
    )

actual fun textRecognitionModel(model: String, config: String): TextRecognitionModel? =
    JniDnn2.textRecognitionModelCreate(model, config).takeIf { it != 0L }
        ?.let { JvmTextRecognitionModel(JvmModelBase(it)) }

actual fun textRecognitionModelFromNet(net: Net): TextRecognitionModel =
    JvmTextRecognitionModel(
        JvmModelBase(
            requireModel(JniDnn2.textRecognitionModelCreateFromNet(netHandleOf(net)), "textRecognitionModelFromNet failed"),
        ),
    )

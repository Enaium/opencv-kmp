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
 * JNI bridge for the high-level dnn model wrappers.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniDnn2_<name>`
 * function in jni/jni_dnn2.cpp. Model handles travel as jlong pointers to
 * cvk_model_t; Net handles (from the DnnCore slice) also travel as jlong.
 * Variable-length results come back as little-endian ByteArray wire buffers
 * or LongArray Mat-handle lists.
 */
internal object JniDnn2 {

    // Touching Jni here runs its init block, which loads the native library
    // before any of these external functions can be invoked.
    init {
        Jni.lastError()
    }

    // Model base

    external fun modelCreate(model: String, config: String): Long
    external fun modelCreateFromNet(net: Long): Long
    external fun modelSetInputSize(model: Long, width: Int, height: Int)
    external fun modelSetInputMean(model: Long, v0: Double, v1: Double, v2: Double, v3: Double)
    external fun modelSetInputScale(model: Long, v0: Double, v1: Double, v2: Double, v3: Double)
    external fun modelSetInputCrop(model: Long, crop: Boolean)
    external fun modelSetInputSwapRB(model: Long, swapRB: Boolean)
    external fun modelSetOutputNames(model: Long, names: Array<String>)
    external fun modelSetInputParams(
        model: Long,
        scale: Double,
        width: Int,
        height: Int,
        m0: Double, m1: Double, m2: Double, m3: Double,
        swapRB: Boolean,
        crop: Boolean,
    )
    external fun modelPredict(model: Long, frame: Long): LongArray
    external fun modelSetPreferableBackend(model: Long, backendId: Int)
    external fun modelSetPreferableTarget(model: Long, targetId: Int)
    external fun modelEnableWinograd(model: Long, useWinograd: Boolean)
    external fun modelRelease(model: Long)

    // ClassificationModel

    external fun classificationModelCreate(model: String, config: String): Long
    external fun classificationModelCreateFromNet(net: Long): Long
    external fun classificationModelSetEnableSoftmaxPostProcessing(model: Long, enable: Boolean)
    external fun classificationModelGetEnableSoftmaxPostProcessing(model: Long): Boolean
    external fun classificationModelClassify(model: Long, frame: Long): DoubleArray

    // DetectionModel

    external fun detectionModelCreate(model: String, config: String): Long
    external fun detectionModelCreateFromNet(net: Long): Long
    external fun detectionModelSetNmsAcrossClasses(model: Long, value: Boolean)
    external fun detectionModelGetNmsAcrossClasses(model: Long): Boolean
    external fun detectionModelDetect(
        model: Long,
        frame: Long,
        confThreshold: Float,
        nmsThreshold: Float,
    ): ByteArray

    // KeypointsModel

    external fun keypointsModelCreate(model: String, config: String): Long
    external fun keypointsModelCreateFromNet(net: Long): Long
    external fun keypointsModelEstimate(model: Long, frame: Long, thresh: Float): ByteArray

    // SegmentationModel

    external fun segmentationModelCreate(model: String, config: String): Long
    external fun segmentationModelCreateFromNet(net: Long): Long
    external fun segmentationModelSegment(model: Long, frame: Long): Long

    // TextDetectionModel base

    external fun textDetectionModelDetect(model: Long, frame: Long): ByteArray
    external fun textDetectionModelDetectTextRectangles(model: Long, frame: Long): ByteArray

    // TextDetectionModel_DB

    external fun textDetectionModelDbCreate(model: String, config: String): Long
    external fun textDetectionModelDbCreateFromNet(net: Long): Long
    external fun textDetectionModelDbSetBinaryThreshold(model: Long, value: Float)
    external fun textDetectionModelDbGetBinaryThreshold(model: Long): Float
    external fun textDetectionModelDbSetPolygonThreshold(model: Long, value: Float)
    external fun textDetectionModelDbGetPolygonThreshold(model: Long): Float
    external fun textDetectionModelDbSetUnclipRatio(model: Long, value: Double)
    external fun textDetectionModelDbGetUnclipRatio(model: Long): Double
    external fun textDetectionModelDbSetMaxCandidates(model: Long, value: Int)
    external fun textDetectionModelDbGetMaxCandidates(model: Long): Int

    // TextDetectionModel_EAST

    external fun textDetectionModelEastCreate(model: String, config: String): Long
    external fun textDetectionModelEastCreateFromNet(net: Long): Long
    external fun textDetectionModelEastSetConfidenceThreshold(model: Long, value: Float)
    external fun textDetectionModelEastGetConfidenceThreshold(model: Long): Float
    external fun textDetectionModelEastSetNmsThreshold(model: Long, value: Float)
    external fun textDetectionModelEastGetNmsThreshold(model: Long): Float

    // TextRecognitionModel

    external fun textRecognitionModelCreate(model: String, config: String): Long
    external fun textRecognitionModelCreateFromNet(net: Long): Long
    external fun textRecognitionModelSetDecodeType(model: Long, decodeType: String)
    external fun textRecognitionModelGetDecodeType(model: Long): String?
    external fun textRecognitionModelSetDecodeOptsCtcPrefixBeamSearch(
        model: Long,
        beamSize: Int,
        vocPruneSize: Int,
    )
    external fun textRecognitionModelSetVocabulary(model: Long, vocabulary: Array<String>)
    external fun textRecognitionModelGetVocabulary(model: Long): Array<String>?
    external fun textRecognitionModelRecognize(model: Long, frame: Long): String?
    external fun textRecognitionModelRecognizeRois(model: Long, frame: Long, rois: IntArray): ByteArray
}

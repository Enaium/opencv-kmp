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
 * JNI bridge for the objdetect QR / barcode / face / MCC slice.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniObjdetect2_<name>`
 * function in jni/jni_objdetect2.cpp. Mat handles travel as jlong pointers;
 * `String?` results are null on native failure; `ByteArray?` results carry
 * the packed string-list buffers documented in opencv_kmp_objdetect2.h.
 */
internal object JniObjdetect2 {

    // ---- cv::QRCodeDetector -------------------------------------------------

    external fun qrCodeDetectorCreate(): Long
    external fun qrCodeDetectorSetEpsX(h: Long, epsX: Double)
    external fun qrCodeDetectorSetEpsY(h: Long, epsY: Double)
    external fun qrCodeDetectorSetUseAlignmentMarkers(h: Long, use: Boolean)
    external fun qrCodeDetectorDecodeCurved(h: Long, img: Long, points: Long, straight: Long): String?
    external fun qrCodeDetectorDetectAndDecodeCurved(h: Long, img: Long, points: Long, straight: Long): String?
    external fun qrCodeDetectorGetEncoding(h: Long, codeIdx: Int): Int
    external fun qrCodeDetectorDetect(h: Long, img: Long, points: Long): Boolean
    external fun qrCodeDetectorDecode(h: Long, img: Long, points: Long, straight: Long): String?
    external fun qrCodeDetectorDetectAndDecode(h: Long, img: Long, points: Long, straight: Long): String?
    external fun qrCodeDetectorDetectMulti(h: Long, img: Long, points: Long): Boolean
    external fun qrCodeDetectorDecodeMulti(h: Long, img: Long, points: Long, straight: Long): ByteArray?
    external fun qrCodeDetectorDetectAndDecodeMulti(h: Long, img: Long, points: Long, straight: Long): ByteArray?
    external fun qrCodeDetectorRelease(h: Long)

    // ---- cv::QRCodeDetectorAruco ---------------------------------------------

    external fun qrCodeDetectorArucoCreate(): Long
    external fun qrCodeDetectorArucoCreateWithParams(
        minModuleSizeInPyramid: Float,
        maxRotation: Float,
        maxModuleSizeMismatch: Float,
        maxTimingPatternMismatch: Float,
        maxPenalties: Float,
        maxColorsMismatch: Float,
        scaleTimingPatternScore: Float,
    ): Long
    external fun qrCodeDetectorArucoGetDetectorParams(h: Long): FloatArray
    external fun qrCodeDetectorArucoSetDetectorParams(
        h: Long,
        minModuleSizeInPyramid: Float,
        maxRotation: Float,
        maxModuleSizeMismatch: Float,
        maxTimingPatternMismatch: Float,
        maxPenalties: Float,
        maxColorsMismatch: Float,
        scaleTimingPatternScore: Float,
    )
    external fun qrCodeDetectorArucoDetect(h: Long, img: Long, points: Long): Boolean
    external fun qrCodeDetectorArucoDecode(h: Long, img: Long, points: Long, straight: Long): String?
    external fun qrCodeDetectorArucoDetectAndDecode(h: Long, img: Long, points: Long, straight: Long): String?
    external fun qrCodeDetectorArucoDetectMulti(h: Long, img: Long, points: Long): Boolean
    external fun qrCodeDetectorArucoDecodeMulti(h: Long, img: Long, points: Long, straight: Long): ByteArray?
    external fun qrCodeDetectorArucoDetectAndDecodeMulti(h: Long, img: Long, points: Long, straight: Long): ByteArray?
    external fun qrCodeDetectorArucoRelease(h: Long)

    // ---- cv::QRCodeEncoder ----------------------------------------------------

    external fun qrCodeEncoderCreate(): Long
    external fun qrCodeEncoderCreateWithParams(
        version: Int,
        correctionLevel: Int,
        mode: Int,
        structureNumber: Int,
    ): Long
    external fun qrCodeEncoderEncode(h: Long, encodedInfo: String, qrcode: Long): Int
    external fun qrCodeEncoderEncodeBytes(h: Long, encodedInfo: ByteArray, qrcode: Long): Int
    external fun qrCodeEncoderEncodeStructuredAppend(h: Long, encodedInfo: String, qrcodes: Long): Int
    external fun qrCodeEncoderRelease(h: Long)

    // ---- cv::barcode::BarcodeDetector ------------------------------------------

    external fun barcodeDetectorCreate(): Long
    external fun barcodeDetectorCreateWithModel(modelPath: String): Long
    external fun barcodeDetectorDetect(h: Long, img: Long, points: Long): Boolean
    external fun barcodeDetectorDecode(h: Long, img: Long, points: Long, straight: Long): String?
    external fun barcodeDetectorDetectAndDecode(h: Long, img: Long, points: Long, straight: Long): String?
    external fun barcodeDetectorDetectMulti(h: Long, img: Long, points: Long): Boolean
    external fun barcodeDetectorDecodeMulti(h: Long, img: Long, points: Long, straight: Long): ByteArray?
    external fun barcodeDetectorDetectAndDecodeMulti(h: Long, img: Long, points: Long, straight: Long): ByteArray?
    external fun barcodeDetectorDecodeWithType(h: Long, img: Long, points: Long): ByteArray?
    external fun barcodeDetectorDetectAndDecodeWithType(h: Long, img: Long, points: Long): ByteArray?
    external fun barcodeDetectorGetDownsamplingThreshold(h: Long): Double
    external fun barcodeDetectorSetDownsamplingThreshold(h: Long, thresh: Double)
    external fun barcodeDetectorGetDetectorScales(h: Long, sizes: Long)
    external fun barcodeDetectorSetDetectorScales(h: Long, sizes: Long)
    external fun barcodeDetectorGetGradientThreshold(h: Long): Double
    external fun barcodeDetectorSetGradientThreshold(h: Long, thresh: Double)
    external fun barcodeDetectorRelease(h: Long)

    // ---- cv::FaceDetectorYN -----------------------------------------------------

    external fun faceDetectorYNCreate(
        model: String,
        config: String,
        inputWidth: Int,
        inputHeight: Int,
        scoreThreshold: Float,
        nmsThreshold: Float,
        topK: Int,
        backendId: Int,
        targetId: Int,
    ): Long
    external fun faceDetectorYNCreateFromBuffers(
        framework: String,
        bufferModel: Long,
        bufferConfig: Long,
        inputWidth: Int,
        inputHeight: Int,
        scoreThreshold: Float,
        nmsThreshold: Float,
        topK: Int,
        backendId: Int,
        targetId: Int,
    ): Long
    external fun faceDetectorYNSetInputSize(h: Long, width: Int, height: Int)
    external fun faceDetectorYNGetInputSize(h: Long): IntArray
    external fun faceDetectorYNSetScoreThreshold(h: Long, threshold: Float)
    external fun faceDetectorYNGetScoreThreshold(h: Long): Float
    external fun faceDetectorYNSetNmsThreshold(h: Long, threshold: Float)
    external fun faceDetectorYNGetNmsThreshold(h: Long): Float
    external fun faceDetectorYNSetTopK(h: Long, topK: Int)
    external fun faceDetectorYNGetTopK(h: Long): Int
    external fun faceDetectorYNDetect(h: Long, image: Long, faces: Long): Int
    external fun faceDetectorYNRelease(h: Long)

    // ---- cv::FaceRecognizerSF ----------------------------------------------------

    external fun faceRecognizerSFCreate(model: String, config: String, backendId: Int, targetId: Int): Long
    external fun faceRecognizerSFCreateFromBuffers(
        framework: String,
        bufferModel: Long,
        bufferConfig: Long,
        backendId: Int,
        targetId: Int,
    ): Long
    external fun faceRecognizerSFAlignCrop(h: Long, srcImg: Long, faceBox: Long, alignedImg: Long)
    external fun faceRecognizerSFFeature(h: Long, alignedImg: Long, faceFeature: Long)
    external fun faceRecognizerSFMatch(h: Long, feature1: Long, feature2: Long, disType: Int): Double
    external fun faceRecognizerSFRelease(h: Long)

    // ---- cv::mcc::CChecker --------------------------------------------------------

    external fun cCheckerCreate(): Long
    external fun cCheckerSetBox(h: Long, box: Long)
    external fun cCheckerSetChartsRgb(h: Long, charts: Long)
    external fun cCheckerSetChartsYCbCr(h: Long, charts: Long)
    external fun cCheckerSetCost(h: Long, cost: Float)
    external fun cCheckerSetCenter(h: Long, x: Double, y: Double)
    external fun cCheckerGetBox(h: Long): Long
    external fun cCheckerGetColorCharts(h: Long): Long
    external fun cCheckerGetChartsRgb(h: Long, getStats: Boolean): Long
    external fun cCheckerGetChartsYCbCr(h: Long): Long
    external fun cCheckerGetCost(h: Long): Float
    external fun cCheckerGetCenter(h: Long): DoubleArray
    external fun cCheckerClear(h: Long)
    external fun cCheckerEmpty(h: Long): Boolean
    external fun cCheckerSave(h: Long, filename: String)
    external fun cCheckerGetDefaultName(h: Long): String?
    external fun cCheckerRelease(h: Long)

    // ---- cv::mcc::CCheckerDetector --------------------------------------------------

    external fun cCheckerDetectorCreate(): Long
    external fun cCheckerDetectorProcessWithRoi(h: Long, image: Long, roi: Long, nc: Int): Boolean
    external fun cCheckerDetectorProcess(h: Long, image: Long, nc: Int): Boolean
    external fun cCheckerDetectorGetList(h: Long): LongArray?
    external fun cCheckerDetectorGetRefColors(h: Long): Long
    external fun cCheckerDetectorSetDetectionParams(
        h: Long,
        adaptiveThreshWinSizeMin: Int,
        adaptiveThreshWinSizeMax: Int,
        adaptiveThreshWinSizeStep: Int,
        adaptiveThreshConstant: Double,
        minContoursAreaRate: Double,
        minContoursArea: Double,
        confidenceThreshold: Double,
        minContourSolidity: Double,
        findCandidatesApproxPolyDPEpsMultiplier: Double,
        borderWidth: Int,
        b0factor: Float,
        maxError: Float,
        minContourPointsAllowed: Int,
        minContourLengthAllowed: Int,
        minInterContourDistance: Int,
        minInterCheckerDistance: Int,
        minImageSize: Int,
        minGroupSize: Int,
    )
    external fun cCheckerDetectorGetDetectionParams(h: Long): DoubleArray
    external fun cCheckerDetectorSetUseDnnModel(h: Long, useDnn: Boolean)
    external fun cCheckerDetectorGetUseDnnModel(h: Long): Boolean
    external fun cCheckerDetectorClear(h: Long)
    external fun cCheckerDetectorEmpty(h: Long): Boolean
    external fun cCheckerDetectorSave(h: Long, filename: String)
    external fun cCheckerDetectorGetDefaultName(h: Long): String?
    external fun cCheckerDetectorRelease(h: Long)
}

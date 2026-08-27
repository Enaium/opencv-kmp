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
// JVM (JNI-backed) implementation of the objdetect QR / barcode / face /
// MCC API. Handles are jlong pointers into the same cvk_ shim the native
// targets bind via cinterop.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()


/** Validates a raw handle factory result, throwing with the native error on 0. */
private fun jvmHandle(ptr: Long, operation: String): Long =
    if (ptr != 0L) ptr else throw OpenCVException(operation, lastNativeError())

// =========================================================================
// QRCodeDetector
// =========================================================================

internal class JvmQrCodeDetector(private var handle: Long) : QRCodeDetector {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("QRCodeDetector is closed")

    override fun setEpsX(epsX: Double): QRCodeDetector {
        JniObjdetect2.qrCodeDetectorSetEpsX(check(), epsX)
        return this
    }

    override fun setEpsY(epsY: Double): QRCodeDetector {
        JniObjdetect2.qrCodeDetectorSetEpsY(check(), epsY)
        return this
    }

    override fun setUseAlignmentMarkers(useAlignmentMarkers: Boolean): QRCodeDetector {
        JniObjdetect2.qrCodeDetectorSetUseAlignmentMarkers(check(), useAlignmentMarkers)
        return this
    }

    override fun decodeCurved(img: Mat, points: Mat, straightQrcode: Mat): String =
        JniObjdetect2.qrCodeDetectorDecodeCurved(
            check(), handleOf(img), handleOf(points), handleOf(straightQrcode),
        ) ?: throw OpenCVException("decodeCurved", lastNativeError())

    override fun detectAndDecodeCurved(img: Mat, points: Mat, straightQrcode: Mat): String =
        JniObjdetect2.qrCodeDetectorDetectAndDecodeCurved(
            check(), handleOf(img), handleOf(points), handleOf(straightQrcode),
        ) ?: throw OpenCVException("detectAndDecodeCurved", lastNativeError())

    override fun getEncoding(codeIdx: Int): Int =
        JniObjdetect2.qrCodeDetectorGetEncoding(check(), codeIdx)

    override fun detect(img: Mat, points: Mat): Boolean =
        JniObjdetect2.qrCodeDetectorDetect(check(), handleOf(img), handleOf(points))

    override fun decode(img: Mat, points: Mat, straightCode: Mat): String =
        JniObjdetect2.qrCodeDetectorDecode(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("decode", lastNativeError())

    override fun detectAndDecode(img: Mat, points: Mat, straightCode: Mat): String =
        JniObjdetect2.qrCodeDetectorDetectAndDecode(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("detectAndDecode", lastNativeError())

    override fun detectMulti(img: Mat, points: Mat): Boolean =
        JniObjdetect2.qrCodeDetectorDetectMulti(check(), handleOf(img), handleOf(points))

    override fun decodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult {
        val data = JniObjdetect2.qrCodeDetectorDecodeMulti(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("decodeMulti", lastNativeError())
        val (ok, info) = decodeMultiBuffer(data)
        return MultiDecodeResult(ok, info, decodeMatList(straightCode))
    }

    override fun detectAndDecodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult {
        val data = JniObjdetect2.qrCodeDetectorDetectAndDecodeMulti(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("detectAndDecodeMulti", lastNativeError())
        val (ok, info) = decodeMultiBuffer(data)
        return MultiDecodeResult(ok, info, decodeMatList(straightCode))
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.qrCodeDetectorRelease(h)
        }
    }
}

actual fun qrCodeDetectorCreate(): QRCodeDetector =
    JvmQrCodeDetector(jvmHandle(JniObjdetect2.qrCodeDetectorCreate(), "qrCodeDetectorCreate"))

// =========================================================================
// QRCodeDetectorAruco
// =========================================================================

internal class JvmQrCodeDetectorAruco(private var handle: Long) : QRCodeDetectorAruco {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("QRCodeDetectorAruco is closed")

    override fun getDetectorParameters(): QRCodeDetectorArucoParams {
        val out = JniObjdetect2.qrCodeDetectorArucoGetDetectorParams(check())
        return QRCodeDetectorArucoParams(out[0], out[1], out[2], out[3], out[4], out[5], out[6])
    }

    override fun setDetectorParameters(params: QRCodeDetectorArucoParams): QRCodeDetectorAruco {
        JniObjdetect2.qrCodeDetectorArucoSetDetectorParams(
            check(),
            params.minModuleSizeInPyramid,
            params.maxRotation,
            params.maxModuleSizeMismatch,
            params.maxTimingPatternMismatch,
            params.maxPenalties,
            params.maxColorsMismatch,
            params.scaleTimingPatternScore,
        )
        return this
    }

    override fun detect(img: Mat, points: Mat): Boolean =
        JniObjdetect2.qrCodeDetectorArucoDetect(check(), handleOf(img), handleOf(points))

    override fun decode(img: Mat, points: Mat, straightCode: Mat): String =
        JniObjdetect2.qrCodeDetectorArucoDecode(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("decode", lastNativeError())

    override fun detectAndDecode(img: Mat, points: Mat, straightCode: Mat): String =
        JniObjdetect2.qrCodeDetectorArucoDetectAndDecode(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("detectAndDecode", lastNativeError())

    override fun detectMulti(img: Mat, points: Mat): Boolean =
        JniObjdetect2.qrCodeDetectorArucoDetectMulti(check(), handleOf(img), handleOf(points))

    override fun decodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult {
        val data = JniObjdetect2.qrCodeDetectorArucoDecodeMulti(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("decodeMulti", lastNativeError())
        val (ok, info) = decodeMultiBuffer(data)
        return MultiDecodeResult(ok, info, decodeMatList(straightCode))
    }

    override fun detectAndDecodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult {
        val data = JniObjdetect2.qrCodeDetectorArucoDetectAndDecodeMulti(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("detectAndDecodeMulti", lastNativeError())
        val (ok, info) = decodeMultiBuffer(data)
        return MultiDecodeResult(ok, info, decodeMatList(straightCode))
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.qrCodeDetectorArucoRelease(h)
        }
    }
}

actual fun qrCodeDetectorArucoCreate(params: QRCodeDetectorArucoParams): QRCodeDetectorAruco =
    JvmQrCodeDetectorAruco(
        jvmHandle(
            JniObjdetect2.qrCodeDetectorArucoCreateWithParams(
                params.minModuleSizeInPyramid,
                params.maxRotation,
                params.maxModuleSizeMismatch,
                params.maxTimingPatternMismatch,
                params.maxPenalties,
                params.maxColorsMismatch,
                params.scaleTimingPatternScore,
            ),
            "qrCodeDetectorArucoCreate",
        ),
    )

// =========================================================================
// QRCodeEncoder
// =========================================================================

internal class JvmQrCodeEncoder(private var handle: Long) : QRCodeEncoder {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("QRCodeEncoder is closed")

    override fun encode(encodedInfo: String): Mat {
        val out = mat()
        if (JniObjdetect2.qrCodeEncoderEncode(check(), encodedInfo, handleOf(out)) == 0) {
            out.close()
            throw OpenCVException("encode", lastNativeError())
        }
        return out
    }

    override fun encode(encodedInfo: ByteArray): Mat {
        val out = mat()
        if (JniObjdetect2.qrCodeEncoderEncodeBytes(check(), encodedInfo, handleOf(out)) == 0) {
            out.close()
            throw OpenCVException("encode", lastNativeError())
        }
        return out
    }

    override fun encodeStructuredAppend(encodedInfo: String): List<Mat> {
        val wire = mat()
        try {
            if (JniObjdetect2.qrCodeEncoderEncodeStructuredAppend(check(), encodedInfo, handleOf(wire)) == 0) {
                throw OpenCVException("encodeStructuredAppend", lastNativeError())
            }
            return decodeMatList(wire)
        } finally {
            wire.close()
        }
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.qrCodeEncoderRelease(h)
        }
    }
}

actual fun qrCodeEncoderCreate(params: QRCodeEncoderParams): QRCodeEncoder =
    JvmQrCodeEncoder(
        jvmHandle(
            JniObjdetect2.qrCodeEncoderCreateWithParams(
                params.version, params.correctionLevel, params.mode, params.structureNumber,
            ),
            "qrCodeEncoderCreate",
        ),
    )

// =========================================================================
// BarcodeDetector
// =========================================================================

internal class JvmBarcodeDetector(private var handle: Long) : BarcodeDetector {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("BarcodeDetector is closed")

    override fun detect(img: Mat, points: Mat): Boolean =
        JniObjdetect2.barcodeDetectorDetect(check(), handleOf(img), handleOf(points))

    override fun decode(img: Mat, points: Mat, straightCode: Mat): String =
        JniObjdetect2.barcodeDetectorDecode(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("decode", lastNativeError())

    override fun detectAndDecode(img: Mat, points: Mat, straightCode: Mat): String =
        JniObjdetect2.barcodeDetectorDetectAndDecode(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("detectAndDecode", lastNativeError())

    override fun detectMulti(img: Mat, points: Mat): Boolean =
        JniObjdetect2.barcodeDetectorDetectMulti(check(), handleOf(img), handleOf(points))

    override fun decodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult {
        val data = JniObjdetect2.barcodeDetectorDecodeMulti(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("decodeMulti", lastNativeError())
        val (ok, info) = decodeMultiBuffer(data)
        return MultiDecodeResult(ok, info, decodeMatList(straightCode))
    }

    override fun detectAndDecodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult {
        val data = JniObjdetect2.barcodeDetectorDetectAndDecodeMulti(
            check(), handleOf(img), handleOf(points), handleOf(straightCode),
        ) ?: throw OpenCVException("detectAndDecodeMulti", lastNativeError())
        val (ok, info) = decodeMultiBuffer(data)
        return MultiDecodeResult(ok, info, decodeMatList(straightCode))
    }

    override fun decodeWithType(img: Mat, points: Mat): BarcodeDecodeResult {
        val data = JniObjdetect2.barcodeDetectorDecodeWithType(check(), handleOf(img), handleOf(points))
            ?: throw OpenCVException("decodeWithType", lastNativeError())
        return decodeWithTypeBuffer(data)
    }

    override fun detectAndDecodeWithType(img: Mat, points: Mat): BarcodeDecodeResult {
        val data = JniObjdetect2.barcodeDetectorDetectAndDecodeWithType(
            check(), handleOf(img), handleOf(points),
        ) ?: throw OpenCVException("detectAndDecodeWithType", lastNativeError())
        return decodeWithTypeBuffer(data)
    }

    override val downsamplingThreshold: Double
        get() = JniObjdetect2.barcodeDetectorGetDownsamplingThreshold(check())

    override fun setDownsamplingThreshold(thresh: Double): BarcodeDetector {
        JniObjdetect2.barcodeDetectorSetDownsamplingThreshold(check(), thresh)
        return this
    }

    override fun getDetectorScales(): MatOfFloat {
        val m = mat()
        JniObjdetect2.barcodeDetectorGetDetectorScales(check(), handleOf(m))
        return MatOfFloat(m)
    }

    override fun setDetectorScales(sizes: MatOfFloat): BarcodeDetector {
        JniObjdetect2.barcodeDetectorSetDetectorScales(check(), handleOf(sizes.mat))
        return this
    }

    override val gradientThreshold: Double
        get() = JniObjdetect2.barcodeDetectorGetGradientThreshold(check())

    override fun setGradientThreshold(thresh: Double): BarcodeDetector {
        JniObjdetect2.barcodeDetectorSetGradientThreshold(check(), thresh)
        return this
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.barcodeDetectorRelease(h)
        }
    }
}

actual fun barcodeDetectorCreate(): BarcodeDetector =
    JvmBarcodeDetector(jvmHandle(JniObjdetect2.barcodeDetectorCreate(), "barcodeDetectorCreate"))

actual fun barcodeDetectorCreate(modelPath: String): BarcodeDetector =
    JvmBarcodeDetector(
        jvmHandle(
            JniObjdetect2.barcodeDetectorCreateWithModel(modelPath),
            "barcodeDetectorCreate(modelPath)",
        ),
    )

// =========================================================================
// FaceDetectorYN
// =========================================================================

internal class JvmFaceDetectorYN(private var handle: Long) : FaceDetectorYN {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("FaceDetectorYN is closed")

    override var inputSize: Size
        get() {
            val out = JniObjdetect2.faceDetectorYNGetInputSize(check())
            return Size(out[0], out[1])
        }
        set(value) {
            JniObjdetect2.faceDetectorYNSetInputSize(check(), value.width, value.height)
        }

    override var scoreThreshold: Float
        get() = JniObjdetect2.faceDetectorYNGetScoreThreshold(check())
        set(value) {
            JniObjdetect2.faceDetectorYNSetScoreThreshold(check(), value)
        }

    override var nmsThreshold: Float
        get() = JniObjdetect2.faceDetectorYNGetNmsThreshold(check())
        set(value) {
            JniObjdetect2.faceDetectorYNSetNmsThreshold(check(), value)
        }

    override var topK: Int
        get() = JniObjdetect2.faceDetectorYNGetTopK(check())
        set(value) {
            JniObjdetect2.faceDetectorYNSetTopK(check(), value)
        }

    override fun detect(image: Mat, faces: Mat): Int =
        JniObjdetect2.faceDetectorYNDetect(check(), handleOf(image), handleOf(faces))

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.faceDetectorYNRelease(h)
        }
    }
}

actual fun faceDetectorYNCreate(
    model: String,
    config: String,
    inputSize: Size,
    scoreThreshold: Float,
    nmsThreshold: Float,
    topK: Int,
    backendId: Int,
    targetId: Int,
): FaceDetectorYN =
    JvmFaceDetectorYN(
        jvmHandle(
            JniObjdetect2.faceDetectorYNCreate(
                model, config, inputSize.width, inputSize.height,
                scoreThreshold, nmsThreshold, topK, backendId, targetId,
            ),
            "faceDetectorYNCreate",
        ),
    )

actual fun faceDetectorYNCreate(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte,
    inputSize: Size,
    scoreThreshold: Float,
    nmsThreshold: Float,
    topK: Int,
    backendId: Int,
    targetId: Int,
): FaceDetectorYN =
    JvmFaceDetectorYN(
        jvmHandle(
            JniObjdetect2.faceDetectorYNCreateFromBuffers(
                framework, handleOf(bufferModel.mat), handleOf(bufferConfig.mat),
                inputSize.width, inputSize.height, scoreThreshold, nmsThreshold, topK, backendId,
                targetId,
            ),
            "faceDetectorYNCreate(framework)",
        ),
    )

// =========================================================================
// FaceRecognizerSF
// =========================================================================

internal class JvmFaceRecognizerSF(private var handle: Long) : FaceRecognizerSF {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("FaceRecognizerSF is closed")

    override fun alignCrop(srcImg: Mat, faceBox: Mat): Mat {
        val out = mat()
        JniObjdetect2.faceRecognizerSFAlignCrop(check(), handleOf(srcImg), handleOf(faceBox), handleOf(out))
        return out
    }

    override fun feature(alignedImg: Mat): Mat {
        val out = mat()
        JniObjdetect2.faceRecognizerSFFeature(check(), handleOf(alignedImg), handleOf(out))
        return out
    }

    override fun match(faceFeature1: Mat, faceFeature2: Mat, disType: Int): Double =
        JniObjdetect2.faceRecognizerSFMatch(
            check(), handleOf(faceFeature1), handleOf(faceFeature2), disType,
        )

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.faceRecognizerSFRelease(h)
        }
    }
}

actual fun faceRecognizerSFCreate(
    model: String,
    config: String,
    backendId: Int,
    targetId: Int,
): FaceRecognizerSF =
    JvmFaceRecognizerSF(
        jvmHandle(
            JniObjdetect2.faceRecognizerSFCreate(model, config, backendId, targetId),
            "faceRecognizerSFCreate",
        ),
    )

actual fun faceRecognizerSFCreate(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte,
    backendId: Int,
    targetId: Int,
): FaceRecognizerSF =
    JvmFaceRecognizerSF(
        jvmHandle(
            JniObjdetect2.faceRecognizerSFCreateFromBuffers(
                framework, handleOf(bufferModel.mat), handleOf(bufferConfig.mat), backendId, targetId,
            ),
            "faceRecognizerSFCreate(framework)",
        ),
    )

// =========================================================================
// CChecker
// =========================================================================

internal class JvmCChecker(private var handle: Long) : CChecker {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("CChecker is closed")

    override fun setBox(box: MatOfPoint2f) {
        JniObjdetect2.cCheckerSetBox(check(), handleOf(box.mat))
    }

    override fun setChartsRGB(chartsRGB: Mat) {
        JniObjdetect2.cCheckerSetChartsRgb(check(), handleOf(chartsRGB))
    }

    override fun setChartsYCbCr(chartsYCbCr: Mat) {
        JniObjdetect2.cCheckerSetChartsYCbCr(check(), handleOf(chartsYCbCr))
    }

    override var cost: Float
        get() = JniObjdetect2.cCheckerGetCost(check())
        set(value) {
            JniObjdetect2.cCheckerSetCost(check(), value)
        }

    override var center: Point2f
        get() {
            val out = JniObjdetect2.cCheckerGetCenter(check())
            return Point2f(out[0].toFloat(), out[1].toFloat())
        }
        set(value) {
            JniObjdetect2.cCheckerSetCenter(check(), value.x.toDouble(), value.y.toDouble())
        }

    override fun getBox(): MatOfPoint2f =
        MatOfPoint2f(jvmMat(JniObjdetect2.cCheckerGetBox(check()), "getBox"))

    override fun getColorCharts(): MatOfPoint2f =
        MatOfPoint2f(jvmMat(JniObjdetect2.cCheckerGetColorCharts(check()), "getColorCharts"))

    override fun getChartsRGB(getStats: Boolean): Mat =
        jvmMat(JniObjdetect2.cCheckerGetChartsRgb(check(), getStats), "getChartsRGB")

    override fun getChartsYCbCr(): Mat =
        jvmMat(JniObjdetect2.cCheckerGetChartsYCbCr(check()), "getChartsYCbCr")

    override fun clear() {
        JniObjdetect2.cCheckerClear(check())
    }

    override fun empty(): Boolean = JniObjdetect2.cCheckerEmpty(check())

    override fun save(filename: String) {
        JniObjdetect2.cCheckerSave(check(), filename)
    }

    override fun getDefaultName(): String =
        JniObjdetect2.cCheckerGetDefaultName(check())
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.cCheckerRelease(h)
        }
    }
}

actual fun cCheckerCreate(): CChecker =
    JvmCChecker(jvmHandle(JniObjdetect2.cCheckerCreate(), "cCheckerCreate"))

// =========================================================================
// CCheckerDetector
// =========================================================================

internal class JvmCCheckerDetector(private var handle: Long) : CCheckerDetector {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("CCheckerDetector is closed")

    override fun processWithROI(image: Mat, regionsOfInterest: MatOfRect, nc: Int): Boolean =
        JniObjdetect2.cCheckerDetectorProcessWithRoi(
            check(), handleOf(image), handleOf(regionsOfInterest.mat), nc,
        )

    override fun process(image: Mat, nc: Int): Boolean =
        JniObjdetect2.cCheckerDetectorProcess(check(), handleOf(image), nc)

    override fun getListColorChecker(): List<CChecker> {
        val handles = JniObjdetect2.cCheckerDetectorGetList(check())
            ?: throw OpenCVException("getListColorChecker", lastNativeError())
        return List(handles.size) { index -> JvmCChecker(handles[index]) }
    }

    override fun getRefColors(): Mat =
        jvmMat(JniObjdetect2.cCheckerDetectorGetRefColors(check()), "getRefColors")

    override fun setDetectionParams(params: DetectorParametersMCC) {
        JniObjdetect2.cCheckerDetectorSetDetectionParams(
            check(),
            params.adaptiveThreshWinSizeMin,
            params.adaptiveThreshWinSizeMax,
            params.adaptiveThreshWinSizeStep,
            params.adaptiveThreshConstant,
            params.minContoursAreaRate,
            params.minContoursArea,
            params.confidenceThreshold,
            params.minContourSolidity,
            params.findCandidatesApproxPolyDPEpsMultiplier,
            params.borderWidth,
            params.b0factor,
            params.maxError,
            params.minContourPointsAllowed,
            params.minContourLengthAllowed,
            params.minInterContourDistance,
            params.minInterCheckerDistance,
            params.minImageSize,
            params.minGroupSize,
        )
    }

    override var useDnnModel: Boolean
        get() = JniObjdetect2.cCheckerDetectorGetUseDnnModel(check())
        set(value) {
            JniObjdetect2.cCheckerDetectorSetUseDnnModel(check(), value)
        }

    override fun getDetectionParams(): DetectorParametersMCC {
        val out = JniObjdetect2.cCheckerDetectorGetDetectionParams(check())
        return DetectorParametersMCC(
            adaptiveThreshWinSizeMin = out[0].toInt(),
            adaptiveThreshWinSizeMax = out[1].toInt(),
            adaptiveThreshWinSizeStep = out[2].toInt(),
            adaptiveThreshConstant = out[3],
            minContoursAreaRate = out[4],
            minContoursArea = out[5],
            confidenceThreshold = out[6],
            minContourSolidity = out[7],
            findCandidatesApproxPolyDPEpsMultiplier = out[8],
            borderWidth = out[9].toInt(),
            b0factor = out[10].toFloat(),
            maxError = out[11].toFloat(),
            minContourPointsAllowed = out[12].toInt(),
            minContourLengthAllowed = out[13].toInt(),
            minInterContourDistance = out[14].toInt(),
            minInterCheckerDistance = out[15].toInt(),
            minImageSize = out[16].toInt(),
            minGroupSize = out[17].toInt(),
        )
    }

    override fun clear() {
        JniObjdetect2.cCheckerDetectorClear(check())
    }

    override fun empty(): Boolean = JniObjdetect2.cCheckerDetectorEmpty(check())

    override fun save(filename: String) {
        JniObjdetect2.cCheckerDetectorSave(check(), filename)
    }

    override fun getDefaultName(): String =
        JniObjdetect2.cCheckerDetectorGetDefaultName(check())
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect2.cCheckerDetectorRelease(h)
        }
    }
}

actual fun cCheckerDetectorCreate(): CCheckerDetector =
    JvmCCheckerDetector(jvmHandle(JniObjdetect2.cCheckerDetectorCreate(), "cCheckerDetectorCreate"))

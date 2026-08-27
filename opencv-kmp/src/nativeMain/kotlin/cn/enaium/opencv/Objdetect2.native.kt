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

import kotlin.concurrent.Volatile

import cvk.*
import kotlinx.cinterop.*
import platform.posix.size_t
import platform.posix.size_tVar

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

private fun stringResult(ptr: CPointer<ByteVar>?, op: String): String {
    if (ptr == null) throw OpenCVException(op, lastNativeError())
    return ptr.toKString()
}

/** Copies a malloc'd byte buffer (released with cvk_free_buffer) into a ByteArray. */
private fun decodeBuffer(buf: CPointer<UByteVar>?, len: size_t, op: String): ByteArray {
    if (buf == null) throw OpenCVException(op, lastNativeError())
    return try {
        buf.readBytes(len.toInt())
    } finally {
        cvk_free_buffer(buf)
    }
}

// =========================================================================
// QRCodeDetector
// =========================================================================

internal class NativeQrCodeDetector(
    @Volatile private var raw: CPointer<cvk_qr_code_detector_t>?,
) : QRCodeDetector {

    private fun check(): CPointer<cvk_qr_code_detector_t> =
        raw ?: throw IllegalStateException("QRCodeDetector is closed")

    override fun setEpsX(epsX: Double): QRCodeDetector {
        cvk_qr_code_detector_set_eps_x(check(), epsX)
        return this
    }

    override fun setEpsY(epsY: Double): QRCodeDetector {
        cvk_qr_code_detector_set_eps_y(check(), epsY)
        return this
    }

    override fun setUseAlignmentMarkers(useAlignmentMarkers: Boolean): QRCodeDetector {
        cvk_qr_code_detector_set_use_alignment_markers(check(), if (useAlignmentMarkers) 1 else 0)
        return this
    }

    override fun decodeCurved(img: Mat, points: Mat, straightQrcode: Mat): String =
        stringResult(
            cvk_qr_code_detector_decode_curved(
                check(), img.nativeHandle(), points.nativeHandle(), straightQrcode.nativeHandle(),
            ),
            "decodeCurved",
        )

    override fun detectAndDecodeCurved(img: Mat, points: Mat, straightQrcode: Mat): String =
        stringResult(
            cvk_qr_code_detector_detect_and_decode_curved(
                check(), img.nativeHandle(), points.nativeHandle(), straightQrcode.nativeHandle(),
            ),
            "detectAndDecodeCurved",
        )

    override fun getEncoding(codeIdx: Int): Int =
        cvk_qr_code_detector_get_encoding(check(), codeIdx)

    override fun detect(img: Mat, points: Mat): Boolean =
        cvk_qr_code_detector_detect(check(), img.nativeHandle(), points.nativeHandle()) != 0

    override fun decode(img: Mat, points: Mat, straightCode: Mat): String =
        stringResult(
            cvk_qr_code_detector_decode(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
            ),
            "decode",
        )

    override fun detectAndDecode(img: Mat, points: Mat, straightCode: Mat): String =
        stringResult(
            cvk_qr_code_detector_detect_and_decode(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
            ),
            "detectAndDecode",
        )

    override fun detectMulti(img: Mat, points: Mat): Boolean =
        cvk_qr_code_detector_detect_multi(check(), img.nativeHandle(), points.nativeHandle()) != 0

    override fun decodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult =
        memScoped {
            val len = alloc<size_tVar>()
            val buf = cvk_qr_code_detector_decode_multi(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
                len.ptr,
            )
            val (ok, info) = decodeMultiBuffer(decodeBuffer(buf, len.value, "decodeMulti"))
            MultiDecodeResult(ok, info, decodeMatList(straightCode))
        }

    override fun detectAndDecodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult =
        memScoped {
            val len = alloc<size_tVar>()
            val buf = cvk_qr_code_detector_detect_and_decode_multi(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
                len.ptr,
            )
            val (ok, info) =
                decodeMultiBuffer(decodeBuffer(buf, len.value, "detectAndDecodeMulti"))
            MultiDecodeResult(ok, info, decodeMatList(straightCode))
        }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_qr_code_detector_release(handle)
    }
}

actual fun qrCodeDetectorCreate(): QRCodeDetector =
    NativeQrCodeDetector(cvk_qr_code_detector_create()
        ?: throw OpenCVException("qrCodeDetectorCreate", lastNativeError()))

// =========================================================================
// QRCodeDetectorAruco
// =========================================================================

internal class NativeQrCodeDetectorAruco(
    @Volatile private var raw: CPointer<cvk_qr_code_detector_aruco_t>?,
) : QRCodeDetectorAruco {

    private fun check(): CPointer<cvk_qr_code_detector_aruco_t> =
        raw ?: throw IllegalStateException("QRCodeDetectorAruco is closed")

    override fun getDetectorParameters(): QRCodeDetectorArucoParams = memScoped {
        val out = allocArray<FloatVar>(7)
        cvk_qr_code_detector_aruco_get_detector_params(check(), out)
        QRCodeDetectorArucoParams(out[0], out[1], out[2], out[3], out[4], out[5], out[6])
    }

    override fun setDetectorParameters(params: QRCodeDetectorArucoParams): QRCodeDetectorAruco {
        cvk_qr_code_detector_aruco_set_detector_params(
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
        cvk_qr_code_detector_aruco_detect(check(), img.nativeHandle(), points.nativeHandle()) != 0

    override fun decode(img: Mat, points: Mat, straightCode: Mat): String =
        stringResult(
            cvk_qr_code_detector_aruco_decode(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
            ),
            "decode",
        )

    override fun detectAndDecode(img: Mat, points: Mat, straightCode: Mat): String =
        stringResult(
            cvk_qr_code_detector_aruco_detect_and_decode(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
            ),
            "detectAndDecode",
        )

    override fun detectMulti(img: Mat, points: Mat): Boolean =
        cvk_qr_code_detector_aruco_detect_multi(check(), img.nativeHandle(), points.nativeHandle()) != 0

    override fun decodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult =
        memScoped {
            val len = alloc<size_tVar>()
            val buf = cvk_qr_code_detector_aruco_decode_multi(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
                len.ptr,
            )
            val (ok, info) = decodeMultiBuffer(decodeBuffer(buf, len.value, "decodeMulti"))
            MultiDecodeResult(ok, info, decodeMatList(straightCode))
        }

    override fun detectAndDecodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult =
        memScoped {
            val len = alloc<size_tVar>()
            val buf = cvk_qr_code_detector_aruco_detect_and_decode_multi(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
                len.ptr,
            )
            val (ok, info) =
                decodeMultiBuffer(decodeBuffer(buf, len.value, "detectAndDecodeMulti"))
            MultiDecodeResult(ok, info, decodeMatList(straightCode))
        }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_qr_code_detector_aruco_release(handle)
    }
}

actual fun qrCodeDetectorArucoCreate(params: QRCodeDetectorArucoParams): QRCodeDetectorAruco =
    NativeQrCodeDetectorAruco(
        cvk_qr_code_detector_aruco_create_with_params(
            params.minModuleSizeInPyramid,
            params.maxRotation,
            params.maxModuleSizeMismatch,
            params.maxTimingPatternMismatch,
            params.maxPenalties,
            params.maxColorsMismatch,
            params.scaleTimingPatternScore,
        ) ?: throw OpenCVException("qrCodeDetectorArucoCreate", lastNativeError()),
    )

// =========================================================================
// QRCodeEncoder
// =========================================================================

internal class NativeQrCodeEncoder(
    @Volatile private var raw: CPointer<cvk_qr_code_encoder_t>?,
) : QRCodeEncoder {

    private fun check(): CPointer<cvk_qr_code_encoder_t> =
        raw ?: throw IllegalStateException("QRCodeEncoder is closed")

    override fun encode(encodedInfo: String): Mat {
        val out = mat()
        if (cvk_qr_code_encoder_encode(check(), encodedInfo, out.nativeHandle()) == 0) {
            out.close()
            throw OpenCVException("encode", lastNativeError())
        }
        return out
    }

    override fun encode(encodedInfo: ByteArray): Mat {
        val out = mat()
        val ok = encodedInfo.asUByteArray().usePinned { pinned ->
            cvk_qr_code_encoder_encode_bytes(
                check(), pinned.addressOf(0), encodedInfo.size.convert<size_t>(),
                out.nativeHandle(),
            ) != 0
        }
        if (!ok) {
            out.close()
            throw OpenCVException("encode", lastNativeError())
        }
        return out
    }

    override fun encodeStructuredAppend(encodedInfo: String): List<Mat> {
        val wire = mat()
        try {
            if (cvk_qr_code_encoder_encode_structured_append(check(), encodedInfo, wire.nativeHandle()) == 0) {
                throw OpenCVException("encodeStructuredAppend", lastNativeError())
            }
            return decodeMatList(wire)
        } finally {
            wire.close()
        }
    }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_qr_code_encoder_release(handle)
    }
}

actual fun qrCodeEncoderCreate(params: QRCodeEncoderParams): QRCodeEncoder =
    NativeQrCodeEncoder(
        cvk_qr_code_encoder_create_with_params(
            params.version, params.correctionLevel, params.mode, params.structureNumber,
        ) ?: throw OpenCVException("qrCodeEncoderCreate", lastNativeError()),
    )

// =========================================================================
// BarcodeDetector
// =========================================================================

internal class NativeBarcodeDetector(
    @Volatile private var raw: CPointer<cvk_barcode_detector_t>?,
) : BarcodeDetector {

    private fun check(): CPointer<cvk_barcode_detector_t> =
        raw ?: throw IllegalStateException("BarcodeDetector is closed")

    override fun detect(img: Mat, points: Mat): Boolean =
        cvk_barcode_detector_detect(check(), img.nativeHandle(), points.nativeHandle()) != 0

    override fun decode(img: Mat, points: Mat, straightCode: Mat): String =
        stringResult(
            cvk_barcode_detector_decode(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
            ),
            "decode",
        )

    override fun detectAndDecode(img: Mat, points: Mat, straightCode: Mat): String =
        stringResult(
            cvk_barcode_detector_detect_and_decode(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
            ),
            "detectAndDecode",
        )

    override fun detectMulti(img: Mat, points: Mat): Boolean =
        cvk_barcode_detector_detect_multi(check(), img.nativeHandle(), points.nativeHandle()) != 0

    override fun decodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult =
        memScoped {
            val len = alloc<size_tVar>()
            val buf = cvk_barcode_detector_decode_multi(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
                len.ptr,
            )
            val (ok, info) = decodeMultiBuffer(decodeBuffer(buf, len.value, "decodeMulti"))
            MultiDecodeResult(ok, info, decodeMatList(straightCode))
        }

    override fun detectAndDecodeMulti(img: Mat, points: Mat, straightCode: Mat): MultiDecodeResult =
        memScoped {
            val len = alloc<size_tVar>()
            val buf = cvk_barcode_detector_detect_and_decode_multi(
                check(), img.nativeHandle(), points.nativeHandle(), straightCode.nativeHandle(),
                len.ptr,
            )
            val (ok, info) =
                decodeMultiBuffer(decodeBuffer(buf, len.value, "detectAndDecodeMulti"))
            MultiDecodeResult(ok, info, decodeMatList(straightCode))
        }

    override fun decodeWithType(img: Mat, points: Mat): BarcodeDecodeResult = memScoped {
        val len = alloc<size_tVar>()
        val buf = cvk_barcode_detector_decode_with_type(
            check(), img.nativeHandle(), points.nativeHandle(), len.ptr,
        )
        decodeWithTypeBuffer(decodeBuffer(buf, len.value, "decodeWithType"))
    }

    override fun detectAndDecodeWithType(img: Mat, points: Mat): BarcodeDecodeResult = memScoped {
        val len = alloc<size_tVar>()
        val buf = cvk_barcode_detector_detect_and_decode_with_type(
            check(), img.nativeHandle(), points.nativeHandle(), len.ptr,
        )
        decodeWithTypeBuffer(decodeBuffer(buf, len.value, "detectAndDecodeWithType"))
    }

    override val downsamplingThreshold: Double
        get() = cvk_barcode_detector_get_downsampling_threshold(check())

    override fun setDownsamplingThreshold(thresh: Double): BarcodeDetector {
        cvk_barcode_detector_set_downsampling_threshold(check(), thresh)
        return this
    }

    override fun getDetectorScales(): MatOfFloat {
        val m = mat()
        cvk_barcode_detector_get_detector_scales(check(), m.nativeHandle())
        return MatOfFloat(m)
    }

    override fun setDetectorScales(sizes: MatOfFloat): BarcodeDetector {
        cvk_barcode_detector_set_detector_scales(check(), sizes.mat.nativeHandle())
        return this
    }

    override val gradientThreshold: Double
        get() = cvk_barcode_detector_get_gradient_threshold(check())

    override fun setGradientThreshold(thresh: Double): BarcodeDetector {
        cvk_barcode_detector_set_gradient_threshold(check(), thresh)
        return this
    }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_barcode_detector_release(handle)
    }
}

actual fun barcodeDetectorCreate(): BarcodeDetector =
    NativeBarcodeDetector(cvk_barcode_detector_create()
        ?: throw OpenCVException("barcodeDetectorCreate", lastNativeError()))

actual fun barcodeDetectorCreate(modelPath: String): BarcodeDetector =
    NativeBarcodeDetector(
        cvk_barcode_detector_create_with_model(modelPath)
            ?: throw OpenCVException("barcodeDetectorCreate(modelPath)", lastNativeError()),
    )

// =========================================================================
// FaceDetectorYN
// =========================================================================

internal class NativeFaceDetectorYN(
    @Volatile private var raw: CPointer<cvk_face_detector_yn_t>?,
) : FaceDetectorYN {

    private fun check(): CPointer<cvk_face_detector_yn_t> =
        raw ?: throw IllegalStateException("FaceDetectorYN is closed")

    override var inputSize: Size
        get() = memScoped {
            val out = allocArray<IntVar>(2)
            cvk_face_detector_yn_get_input_size(check(), out)
            Size(out[0], out[1])
        }
        set(value) {
            cvk_face_detector_yn_set_input_size(check(), value.width, value.height)
        }

    override var scoreThreshold: Float
        get() = cvk_face_detector_yn_get_score_threshold(check())
        set(value) {
            cvk_face_detector_yn_set_score_threshold(check(), value)
        }

    override var nmsThreshold: Float
        get() = cvk_face_detector_yn_get_nms_threshold(check())
        set(value) {
            cvk_face_detector_yn_set_nms_threshold(check(), value)
        }

    override var topK: Int
        get() = cvk_face_detector_yn_get_top_k(check())
        set(value) {
            cvk_face_detector_yn_set_top_k(check(), value)
        }

    override fun detect(image: Mat, faces: Mat): Int =
        cvk_face_detector_yn_detect(check(), image.nativeHandle(), faces.nativeHandle())

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_face_detector_yn_release(handle)
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
    NativeFaceDetectorYN(
        cvk_face_detector_yn_create(
            model, config, inputSize.width, inputSize.height,
            scoreThreshold, nmsThreshold, topK, backendId, targetId,
        ) ?: throw OpenCVException("faceDetectorYNCreate", lastNativeError()),
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
    NativeFaceDetectorYN(
        cvk_face_detector_yn_create_from_buffers(
            framework, bufferModel.mat.nativeHandle(), bufferConfig.mat.nativeHandle(),
            inputSize.width, inputSize.height, scoreThreshold, nmsThreshold, topK, backendId,
            targetId,
        ) ?: throw OpenCVException("faceDetectorYNCreate(framework)", lastNativeError()),
    )

// =========================================================================
// FaceRecognizerSF
// =========================================================================

internal class NativeFaceRecognizerSF(
    @Volatile private var raw: CPointer<cvk_face_recognizer_sf_t>?,
) : FaceRecognizerSF {

    private fun check(): CPointer<cvk_face_recognizer_sf_t> =
        raw ?: throw IllegalStateException("FaceRecognizerSF is closed")

    override fun alignCrop(srcImg: Mat, faceBox: Mat): Mat {
        val out = mat()
        cvk_face_recognizer_sf_align_crop(
            check(), srcImg.nativeHandle(), faceBox.nativeHandle(), out.nativeHandle(),
        )
        return out
    }

    override fun feature(alignedImg: Mat): Mat {
        val out = mat()
        cvk_face_recognizer_sf_feature(check(), alignedImg.nativeHandle(), out.nativeHandle())
        return out
    }

    override fun match(faceFeature1: Mat, faceFeature2: Mat, disType: Int): Double =
        cvk_face_recognizer_sf_match(
            check(), faceFeature1.nativeHandle(), faceFeature2.nativeHandle(), disType,
        )

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_face_recognizer_sf_release(handle)
    }
}

actual fun faceRecognizerSFCreate(
    model: String,
    config: String,
    backendId: Int,
    targetId: Int,
): FaceRecognizerSF =
    NativeFaceRecognizerSF(
        cvk_face_recognizer_sf_create(model, config, backendId, targetId)
            ?: throw OpenCVException("faceRecognizerSFCreate", lastNativeError()),
    )

actual fun faceRecognizerSFCreate(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte,
    backendId: Int,
    targetId: Int,
): FaceRecognizerSF =
    NativeFaceRecognizerSF(
        cvk_face_recognizer_sf_create_from_buffers(
            framework, bufferModel.mat.nativeHandle(), bufferConfig.mat.nativeHandle(),
            backendId, targetId,
        ) ?: throw OpenCVException("faceRecognizerSFCreate(framework)", lastNativeError()),
    )

// =========================================================================
// CChecker
// =========================================================================

internal class NativeCChecker(
    @Volatile private var raw: CPointer<cvk_c_checker_t>?,
) : CChecker {

    private fun check(): CPointer<cvk_c_checker_t> =
        raw ?: throw IllegalStateException("CChecker is closed")

    override fun setBox(box: MatOfPoint2f) {
        cvk_c_checker_set_box(check(), box.mat.nativeHandle())
    }

    override fun setChartsRGB(chartsRGB: Mat) {
        cvk_c_checker_set_charts_rgb(check(), chartsRGB.nativeHandle())
    }

    override fun setChartsYCbCr(chartsYCbCr: Mat) {
        cvk_c_checker_set_charts_y_cb_cr(check(), chartsYCbCr.nativeHandle())
    }

    override var cost: Float
        get() = cvk_c_checker_get_cost(check())
        set(value) {
            cvk_c_checker_set_cost(check(), value)
        }

    override var center: Point2f
        get() = memScoped {
            val out = allocArray<DoubleVar>(2)
            cvk_c_checker_get_center(check(), out)
            Point2f(out[0].toFloat(), out[1].toFloat())
        }
        set(value) {
            cvk_c_checker_set_center(check(), value.x.toDouble(), value.y.toDouble())
        }

    override fun getBox(): MatOfPoint2f =
        MatOfPoint2f(nativeMat(cvk_c_checker_get_box(check()), "getBox"))

    override fun getColorCharts(): MatOfPoint2f =
        MatOfPoint2f(nativeMat(cvk_c_checker_get_color_charts(check()), "getColorCharts"))

    override fun getChartsRGB(getStats: Boolean): Mat =
        nativeMat(cvk_c_checker_get_charts_rgb(check(), if (getStats) 1 else 0), "getChartsRGB")

    override fun getChartsYCbCr(): Mat =
        nativeMat(cvk_c_checker_get_charts_y_cb_cr(check()), "getChartsYCbCr")

    override fun clear() {
        cvk_c_checker_clear(check())
    }

    override fun empty(): Boolean = cvk_c_checker_empty(check()) != 0

    override fun save(filename: String) {
        cvk_c_checker_save(check(), filename)
    }

    override fun getDefaultName(): String =
        stringResult(cvk_c_checker_get_default_name(check()), "getDefaultName")

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_c_checker_release(handle)
    }
}

actual fun cCheckerCreate(): CChecker =
    NativeCChecker(cvk_c_checker_create()
        ?: throw OpenCVException("cCheckerCreate", lastNativeError()))

// =========================================================================
// CCheckerDetector
// =========================================================================

internal class NativeCCheckerDetector(
    @Volatile private var raw: CPointer<cvk_c_checker_detector_t>?,
) : CCheckerDetector {

    private fun check(): CPointer<cvk_c_checker_detector_t> =
        raw ?: throw IllegalStateException("CCheckerDetector is closed")

    override fun processWithROI(image: Mat, regionsOfInterest: MatOfRect, nc: Int): Boolean =
        cvk_c_checker_detector_process_with_roi(
            check(), image.nativeHandle(), regionsOfInterest.mat.nativeHandle(), nc,
        ) != 0

    override fun process(image: Mat, nc: Int): Boolean =
        cvk_c_checker_detector_process(check(), image.nativeHandle(), nc) != 0

    override fun getListColorChecker(): List<CChecker> = memScoped {
        val count = cvk_c_checker_detector_get_list(check(), null, 0)
        if (count < 0) throw OpenCVException("getListColorChecker", lastNativeError())
        if (count == 0) return@memScoped emptyList()
        val handles = allocArray<CPointerVar<cvk_c_checker_t>>(count)
        val filled = cvk_c_checker_detector_get_list(check(), handles, count)
        if (filled < 0) throw OpenCVException("getListColorChecker", lastNativeError())
        List(filled) { index -> NativeCChecker(handles[index]) }
    }

    override fun getRefColors(): Mat =
        nativeMat(cvk_c_checker_detector_get_ref_colors(check()), "getRefColors")

    override fun setDetectionParams(params: DetectorParametersMCC) {
        cvk_c_checker_detector_set_detection_params(
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
        get() = cvk_c_checker_detector_get_use_dnn_model(check()) != 0
        set(value) {
            cvk_c_checker_detector_set_use_dnn_model(check(), if (value) 1 else 0)
        }

    override fun getDetectionParams(): DetectorParametersMCC = memScoped {
        val out = allocArray<DoubleVar>(18)
        cvk_c_checker_detector_get_detection_params(check(), out)
        DetectorParametersMCC(
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
        cvk_c_checker_detector_clear(check())
    }

    override fun empty(): Boolean = cvk_c_checker_detector_empty(check()) != 0

    override fun save(filename: String) {
        cvk_c_checker_detector_save(check(), filename)
    }

    override fun getDefaultName(): String =
        stringResult(cvk_c_checker_detector_get_default_name(check()), "getDefaultName")

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_c_checker_detector_release(handle)
    }
}

actual fun cCheckerDetectorCreate(): CCheckerDetector =
    NativeCCheckerDetector(cvk_c_checker_detector_create()
        ?: throw OpenCVException("cCheckerDetectorCreate", lastNativeError()))

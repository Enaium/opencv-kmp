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

import cvk.cvk_dnn_blob_from_image
import cvk.cvk_dnn_blob_from_images
import cvk.cvk_dnn_get_available_backends
import cvk.cvk_dnn_get_available_targets
import cvk.cvk_dnn_images_from_blob
import cvk.cvk_dnn_mat_dims
import cvk.cvk_dnn_mat_shape
import cvk.cvk_dnn_nms_boxes
import cvk.cvk_dnn_nms_boxes_batched
import cvk.cvk_dnn_nms_boxes_rotated
import cvk.cvk_dnn_read_net
import cvk.cvk_dnn_read_net_buffer
import cvk.cvk_dnn_read_net_from_model_optimizer
import cvk.cvk_dnn_read_net_from_model_optimizer_buffer
import cvk.cvk_dnn_read_net_from_onnx
import cvk.cvk_dnn_read_net_from_onnx_buffer
import cvk.cvk_dnn_read_net_from_tensorflow
import cvk.cvk_dnn_read_net_from_tensorflow_buffer
import cvk.cvk_dnn_read_net_from_tflite
import cvk.cvk_dnn_read_net_from_tflite_buffer
import cvk.cvk_dnn_read_tensor_from_onnx
import cvk.cvk_dnn_soft_nms_boxes
import cvk.cvk_dnn_write_text_graph
import cvk.cvk_free_buffer
import cvk.cvk_last_error
import cvk.cvk_layer_finalize
import cvk.cvk_layer_get_blobs
import cvk.cvk_layer_name
import cvk.cvk_layer_output_name_to_index
import cvk.cvk_layer_preferable_target
import cvk.cvk_layer_release
import cvk.cvk_layer_run
import cvk.cvk_layer_set_blobs
import cvk.cvk_layer_t
import cvk.cvk_layer_type
import cvk.cvk_mat_t
import cvk.cvk_net_connect
import cvk.cvk_net_create
import cvk.cvk_net_disable_kv_cache
import cvk.cvk_net_dump
import cvk.cvk_net_dump_to_file
import cvk.cvk_net_dump_to_pbtxt
import cvk.cvk_net_empty
import cvk.cvk_net_enable_fusion
import cvk.cvk_net_enable_kv_cache
import cvk.cvk_net_enable_winograd
import cvk.cvk_net_finalize
import cvk.cvk_net_forward
import cvk.cvk_net_forward_all
import cvk.cvk_net_forward_and_retrieve
import cvk.cvk_net_forward_layer
import cvk.cvk_net_forward_names
import cvk.cvk_net_get_flops
import cvk.cvk_net_get_layer
import cvk.cvk_net_get_layer_by_name
import cvk.cvk_net_get_layer_id
import cvk.cvk_net_get_layer_names
import cvk.cvk_net_get_layer_types
import cvk.cvk_net_get_layers_count
import cvk.cvk_net_get_memory_consumption
import cvk.cvk_net_get_param
import cvk.cvk_net_get_param_by_name
import cvk.cvk_net_get_perf_profile
import cvk.cvk_net_get_perf_profile_names
import cvk.cvk_net_get_unconnected_out_layers
import cvk.cvk_net_get_unconnected_out_layers_names
import cvk.cvk_net_print_perf_profile
import cvk.cvk_net_register_output
import cvk.cvk_net_release
import cvk.cvk_net_reset_kv_cache
import cvk.cvk_net_set_input
import cvk.cvk_net_set_input_shape
import cvk.cvk_net_set_inputs_names
import cvk.cvk_net_set_param
import cvk.cvk_net_set_param_by_name
import cvk.cvk_net_set_preferable_backend
import cvk.cvk_net_set_preferable_target
import cvk.cvk_net_t
import cvk.cvk_tokenizer_decode
import cvk.cvk_tokenizer_encode
import cvk.cvk_tokenizer_load
import cvk.cvk_tokenizer_release
import cvk.cvk_tokenizer_t
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.toCPointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.ULongVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.interpretCPointer
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.refTo
import platform.posix.size_t
import platform.posix.size_tVar
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value
import kotlin.concurrent.Volatile

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

// =========================================================================
// wire helpers
// =========================================================================

internal actual fun matAddress(mat: Mat): Long = mat.nativeHandle().rawValue.toLong()

internal actual fun matFromHandle(addr: Long): Mat =
    addr.toCPointer<cvk_mat_t>()
        ?.let { NativeMat(it) }
        ?: throw IllegalArgumentException("null Mat address $addr")

internal actual fun matDims(mat: Mat): Int = cvk_dnn_mat_dims(mat.nativeHandle())

internal actual fun matShape(mat: Mat): IntArray = memScoped {
    val dims = cvk_dnn_mat_dims(mat.nativeHandle())
    if (dims <= 0) return@memScoped IntArray(0)
    val shape = allocArray<IntVar>(dims)
    cvk_dnn_mat_shape(mat.nativeHandle(), shape, dims)
    IntArray(dims) { shape[it] }
}

private fun stringOf(ptr: CPointer<ByteVar>?, operation: String): String =
    ptr?.toKString() ?: throw OpenCVException(operation, lastNativeError())

private fun flatStrings(fn: (CPointer<size_tVar>?) -> CPointer<UByteVar>?, operation: String): List<String> =
    memScoped {
        val len = alloc<size_tVar>()
        val flat = fn(len.ptr) ?: throw OpenCVException(operation, lastNativeError())
        try {
            decodeStrings(flat.readBytes(len.value.toInt()))
        } finally {
            cvk_free_buffer(flat)
        }
    }

// =========================================================================
// Net
// =========================================================================

/** cinterop-backed [Net] wrapping a `cv::Ptr<cv::dnn::Net>` handle. */
/**
 * Decodes the getPerfProfile wire buffer produced by
 * cvk_net_get_perf_profile_names:
 * `[u32le count]` then per entry `[u32le name_len, bytes, u32le time_len,
 * bytes, u32le count_len, bytes]`.
 */
private fun decodePerfProfile(data: ByteArray): PerfProfile {
    val n = data.readIntLE(0)
    val names = ArrayList<String>(n)
    val timems = ArrayList<String>(n)
    val counts = ArrayList<String>(n)
    var off = 4
    repeat(n) {
        var len = data.readIntLE(off); off += 4
        names.add(data.decodeToString(off, off + len)); off += len
        len = data.readIntLE(off); off += 4
        timems.add(data.decodeToString(off, off + len)); off += len
        len = data.readIntLE(off); off += 4
        counts.add(data.decodeToString(off, off + len)); off += len
    }
    return PerfProfile(names, timems, counts)
}

internal class NativeNet(@Volatile private var raw: CPointer<cvk_net_t>?) : Net {

    private fun check(): CPointer<cvk_net_t> =
        raw ?: throw IllegalStateException("Net is closed")

    /** Raw handle for other slices (e.g. dnn2 Model wrappers). */
    internal fun netHandle(): CPointer<cvk_net_t> = check()

    override val empty: Boolean get() = cvk_net_empty(check()) != 0

    override fun dump(): String =
        stringOf(cvk_net_dump(check()), "dump")

    override fun dumpToFile(path: String) {
        cvk_net_dump_to_file(check(), path)
    }

    override fun dumpToPbtxt(path: String) {
        cvk_net_dump_to_pbtxt(check(), path)
    }

    override fun getLayerId(layer: String): Int = cvk_net_get_layer_id(check(), layer)

    override fun getLayerNames(): List<String> =
        flatStrings({ cvk_net_get_layer_names(check(), it) }, "getLayerNames")

    override fun getLayer(layerId: Int): Layer =
        NativeLayer(cvk_net_get_layer(check(), layerId)
            ?: throw OpenCVException("getLayer", lastNativeError()))

    override fun getLayer(layerName: String): Layer =
        NativeLayer(cvk_net_get_layer_by_name(check(), layerName)
            ?: throw OpenCVException("getLayer", lastNativeError()))

    override fun connect(outPin: String, inpPin: String) {
        cvk_net_connect(check(), outPin, inpPin)
    }

    override fun registerOutput(outputName: String, layerId: Int, outputPort: Int): Int =
        cvk_net_register_output(check(), outputName, layerId, outputPort)

    override fun setInputsNames(inputBlobNames: List<String>) {
        val flat = encodeStrings(inputBlobNames)
        cvk_net_set_inputs_names(check(), flat.asUByteArray().refTo(0), flat.size.convert<size_t>())
    }

    override fun setInputShape(inputName: String, shape: MatOfInt) {
        cvk_net_set_input_shape(check(), inputName, shape.mat.nativeHandle())
    }

    override fun forward(outputName: String): Mat =
        nativeMat(cvk_net_forward(check(), outputName), "forward")

    override fun forwardLayer(outputName: String): List<Mat> =
        decodeMatVector(nativeMat(cvk_net_forward_layer(check(), outputName), "forwardLayer"))

    override fun forward(outBlobNames: List<String>): List<Mat> {
        val flat = encodeStrings(outBlobNames)
        return decodeMatVector(
            nativeMat(cvk_net_forward_names(check(), flat.asUByteArray().refTo(0), flat.size.convert<size_t>()), "forward"),
        )
    }

    override fun forwardAndRetrieve(outBlobNames: List<String>): List<List<Mat>> {
        val flat = encodeStrings(outBlobNames)
        return decodeMatVector(
            nativeMat(cvk_net_forward_and_retrieve(check(), flat.asUByteArray().refTo(0), flat.size.convert<size_t>()), "forwardAndRetrieve"),
        ).map { decodeMatVector(it) }
    }

    override fun setPreferableBackend(backendId: Int) {
        cvk_net_set_preferable_backend(check(), backendId)
    }

    override fun setPreferableTarget(targetId: Int) {
        cvk_net_set_preferable_target(check(), targetId)
    }

    override fun finalizeNet() {
        cvk_net_finalize(check())
    }

    override fun setInput(blob: Mat, name: String, scalefactor: Double, mean: Scalar) {
        cvk_net_set_input(check(), blob.nativeHandle(), name, scalefactor, mean.toCvk())
    }

    override fun setParam(layer: Int, numParam: Int, blob: Mat) {
        cvk_net_set_param(check(), layer, numParam, blob.nativeHandle())
    }

    override fun setParam(layerName: String, numParam: Int, blob: Mat) {
        cvk_net_set_param_by_name(check(), layerName, numParam, blob.nativeHandle())
    }

    override fun getParam(layer: Int, numParam: Int): Mat =
        nativeMat(cvk_net_get_param(check(), layer, numParam), "getParam")

    override fun getParam(layerName: String, numParam: Int): Mat =
        nativeMat(cvk_net_get_param_by_name(check(), layerName, numParam), "getParam")

    override fun getUnconnectedOutLayers(): MatOfInt =
        MatOfInt(nativeMat(cvk_net_get_unconnected_out_layers(check()), "getUnconnectedOutLayers"))

    override fun getUnconnectedOutLayersNames(): List<String> =
        flatStrings({ cvk_net_get_unconnected_out_layers_names(check(), it) }, "getUnconnectedOutLayersNames")

    override fun getFLOPS(netInputShapes: List<MatOfInt>, netInputTypes: MatOfInt): Long {
        val shapes = encodeMatVector(netInputShapes.map { it.mat })
        return try {
            cvk_net_get_flops(check(), shapes.nativeHandle(), netInputTypes.mat.nativeHandle())
        } finally {
            shapes.close()
        }
    }

    override fun getMemoryConsumption(netInputShapes: List<MatOfInt>, netInputTypes: MatOfInt): MemoryConsumption = memScoped {
        val shapes = encodeMatVector(netInputShapes.map { it.mat })
        val out = allocArray<ULongVar>(2)
        try {
            cvk_net_get_memory_consumption(check(), shapes.nativeHandle(), netInputTypes.mat.nativeHandle(), out)
            MemoryConsumption(weights = out[0].toLong(), blobs = out[1].toLong())
        } finally {
            shapes.close()
        }
    }

    override fun getLayerTypes(): List<String> =
        flatStrings({ ptr ->
            memScoped {
                val flat = alloc<CPointerVar<UByteVar>>()
                cvk_net_get_layer_types(check(), flat.ptr, ptr)
                flat.value
            }
        }, "getLayerTypes")

    override fun getLayersCount(layerType: String): Int =
        cvk_net_get_layers_count(check(), layerType)

    override fun enableFusion(fusion: Boolean) {
        cvk_net_enable_fusion(check(), if (fusion) 1 else 0)
    }

    override fun enableWinograd(useWinograd: Boolean) {
        cvk_net_enable_winograd(check(), if (useWinograd) 1 else 0)
    }

    override fun getPerfProfile(timings: MatOfDouble): Long =
        cvk_net_get_perf_profile(check(), timings.mat.nativeHandle())

    override fun enableKVCache() {
        cvk_net_enable_kv_cache(check())
    }

    override fun disableKVCache() {
        cvk_net_disable_kv_cache(check())
    }

    override fun resetKVCache() {
        cvk_net_reset_kv_cache(check())
    }

    override fun getPerfProfile(): PerfProfile = memScoped {
        val len = alloc<size_tVar>()
        val flat = alloc<CPointerVar<UByteVar>>()
        cvk_net_get_perf_profile_names(check(), flat.ptr, len.ptr)
        val data = flat.value ?: throw OpenCVException("getPerfProfile", lastNativeError())
        try {
            decodePerfProfile(data.readBytes(len.value.toInt()))
        } finally {
            cvk_free_buffer(flat.value)
        }
    }

    override fun printPerfProfile() {
        cvk_net_print_perf_profile(check())
    }

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_net_release(h)
    }
}

/** Raw handle of any native-backed [Net] (used by sibling slices). */
internal fun Net.nativeHandle(): CPointer<cvk_net_t> =
    (this as? NativeNet)?.netHandle()
        ?: throw IllegalArgumentException("net belongs to another platform backend")

// =========================================================================
// Layer
// =========================================================================

/** cinterop-backed [Layer] wrapping a `cv::Ptr<cv::dnn::Layer>` handle. */
internal class NativeLayer(@Volatile private var raw: CPointer<cvk_layer_t>?) : Layer {

    private fun check(): CPointer<cvk_layer_t> =
        raw ?: throw IllegalStateException("Layer is closed")

    override var blobs: List<Mat>
        get() = decodeMatVector(nativeMat(cvk_layer_get_blobs(check()), "getBlobs"))
        set(value) {
            val encoded = encodeMatVector(value)
            try {
                cvk_layer_set_blobs(check(), encoded.nativeHandle())
            } finally {
                encoded.close()
            }
        }

    override val name: String
        get() = stringOf(cvk_layer_name(check()), "name")

    override val type: String
        get() = stringOf(cvk_layer_type(check()), "type")

    override val preferableTarget: Int
        get() = cvk_layer_preferable_target(check())

    override fun outputNameToIndex(outputName: String): Int =
        cvk_layer_output_name_to_index(check(), outputName)

    override fun finalize(inputs: List<Mat>): List<Mat> {
        val inVec = encodeMatVector(inputs)
        val outMat = mat(0, 0, cvMakeType(CV_32S, 2))
        try {
            cvk_layer_finalize(check(), inVec.nativeHandle(), outMat.nativeHandle())
            return decodeMatVector(outMat)
        } finally {
            inVec.close()
        }
    }

    override fun run(inputs: List<Mat>, internals: List<Mat>): Pair<List<Mat>, List<Mat>> {
        val inVec = encodeMatVector(inputs)
        val intVec = encodeMatVector(internals)
        val outMat = mat(0, 0, cvMakeType(CV_32S, 2))
        val intOutMat = mat(0, 0, cvMakeType(CV_32S, 2))
        try {
            cvk_layer_run(check(), inVec.nativeHandle(), outMat.nativeHandle(), intOutMat.nativeHandle())
            return decodeMatVector(outMat) to decodeMatVector(intOutMat)
        } finally {
            inVec.close()
            intVec.close()
        }
    }

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_layer_release(h)
    }
}

// =========================================================================
// Tokenizer
// =========================================================================

/** cinterop-backed [Tokenizer] wrapping a `cv::dnn::Tokenizer`. */
internal class NativeTokenizer(@Volatile private var raw: CPointer<cvk_tokenizer_t>?) : Tokenizer {

    private fun check(): CPointer<cvk_tokenizer_t> =
        raw ?: throw IllegalStateException("Tokenizer is closed")

    override fun encode(text: String): MatOfInt =
        MatOfInt(nativeMat(cvk_tokenizer_encode(check(), text), "encode"))

    override fun decode(tokens: MatOfInt): String =
        stringOf(cvk_tokenizer_decode(check(), tokens.mat.nativeHandle()), "decode")

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_tokenizer_release(h)
    }
}

// =========================================================================
// actual declarations
// =========================================================================

actual fun dnnReadNet(model: String, config: String, framework: String, engine: Int): Net? =
    cvk_dnn_read_net(model, config, framework, engine)?.let { NativeNet(it) }

actual fun dnnReadNetBuffer(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte?,
    engine: Int,
): Net? = cvk_dnn_read_net_buffer(
    framework,
    bufferModel.mat.nativeHandle(),
    bufferConfig?.mat?.nativeHandle(),
    engine,
)?.let { NativeNet(it) }

actual fun dnnReadNetFromOnnx(onnxFile: String, engine: Int): Net? =
    cvk_dnn_read_net_from_onnx(onnxFile, engine)?.let { NativeNet(it) }

actual fun dnnReadNetFromOnnxBuffer(buffer: MatOfByte, engine: Int): Net? =
    cvk_dnn_read_net_from_onnx_buffer(buffer.mat.nativeHandle(), engine)?.let { NativeNet(it) }

actual fun dnnReadNetFromTensorflow(model: String, config: String, engine: Int): Net? =
    cvk_dnn_read_net_from_tensorflow(model, config, engine)?.let { NativeNet(it) }

actual fun dnnReadNetFromTensorflowBuffer(
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte?,
    engine: Int,
): Net? = cvk_dnn_read_net_from_tensorflow_buffer(
    bufferModel.mat.nativeHandle(),
    bufferConfig?.mat?.nativeHandle(),
    engine,
)?.let { NativeNet(it) }

actual fun dnnReadNetFromTflite(model: String, engine: Int): Net? =
    cvk_dnn_read_net_from_tflite(model, engine)?.let { NativeNet(it) }

actual fun dnnReadNetFromTfliteBuffer(bufferModel: MatOfByte, engine: Int): Net? =
    cvk_dnn_read_net_from_tflite_buffer(bufferModel.mat.nativeHandle(), engine)?.let { NativeNet(it) }

actual fun dnnReadNetFromModelOptimizer(xml: String, bin: String): Net? =
    cvk_dnn_read_net_from_model_optimizer(xml, bin)?.let { NativeNet(it) }

actual fun dnnReadNetFromModelOptimizerBuffer(bufferModelConfig: MatOfByte, bufferWeights: MatOfByte): Net? =
    cvk_dnn_read_net_from_model_optimizer_buffer(
        bufferModelConfig.mat.nativeHandle(),
        bufferWeights.mat.nativeHandle(),
    )?.let { NativeNet(it) }

actual fun dnnReadTensorFromOnnx(path: String): Mat? =
    cvk_dnn_read_tensor_from_onnx(path)?.let { NativeMat(it) }

actual fun dnnBlobFromImage(
    image: Mat,
    scalefactor: Double,
    size: Size,
    mean: Scalar,
    swapRB: Boolean,
    crop: Boolean,
    ddepth: Int,
): Mat = nativeMat(
    cvk_dnn_blob_from_image(
        image.nativeHandle(), scalefactor, size.width, size.height, mean.toCvk(),
        if (swapRB) 1 else 0, if (crop) 1 else 0, ddepth,
    ),
    "blobFromImage",
)

actual fun dnnBlobFromImages(
    images: List<Mat>,
    scalefactor: Double,
    size: Size,
    mean: Scalar,
    swapRB: Boolean,
    crop: Boolean,
    ddepth: Int,
): Mat {
    val encoded = encodeMatVector(images)
    return try {
        nativeMat(
            cvk_dnn_blob_from_images(
                encoded.nativeHandle(), scalefactor, size.width, size.height, mean.toCvk(),
                if (swapRB) 1 else 0, if (crop) 1 else 0, ddepth,
            ),
            "blobFromImages",
        )
    } finally {
        encoded.close()
    }
}

actual fun dnnImagesFromBlob(blob: Mat): List<Mat> =
    decodeMatVector(nativeMat(cvk_dnn_images_from_blob(blob.nativeHandle()), "imagesFromBlob"))

actual fun dnnNmsBoxes(
    bboxes: MatOfRect2d,
    scores: MatOfFloat,
    scoreThreshold: Float,
    nmsThreshold: Float,
    eta: Float,
    topK: Int,
): MatOfInt {
    val indices = mat(0, 0, cvMakeType(CV_32S, 1))
    cvk_dnn_nms_boxes(
        bboxes.mat.nativeHandle(), scores.mat.nativeHandle(),
        scoreThreshold, nmsThreshold, indices.nativeHandle(), eta, topK,
    )
    return MatOfInt(indices)
}

actual fun dnnNmsBoxesRotated(
    bboxes: MatOfRotatedRect,
    scores: MatOfFloat,
    scoreThreshold: Float,
    nmsThreshold: Float,
    eta: Float,
    topK: Int,
): MatOfInt {
    val indices = mat(0, 0, cvMakeType(CV_32S, 1))
    cvk_dnn_nms_boxes_rotated(
        bboxes.mat.nativeHandle(), scores.mat.nativeHandle(),
        scoreThreshold, nmsThreshold, indices.nativeHandle(), eta, topK,
    )
    return MatOfInt(indices)
}

actual fun dnnNmsBoxesBatched(
    bboxes: MatOfRect2d,
    scores: MatOfFloat,
    classIds: MatOfInt,
    scoreThreshold: Float,
    nmsThreshold: Float,
    eta: Float,
    topK: Int,
): MatOfInt {
    val indices = mat(0, 0, cvMakeType(CV_32S, 1))
    cvk_dnn_nms_boxes_batched(
        bboxes.mat.nativeHandle(), scores.mat.nativeHandle(), classIds.mat.nativeHandle(),
        scoreThreshold, nmsThreshold, indices.nativeHandle(), eta, topK,
    )
    return MatOfInt(indices)
}

actual fun dnnSoftNmsBoxes(
    bboxes: MatOfRect,
    scores: MatOfFloat,
    scoreThreshold: Float,
    nmsThreshold: Float,
    topK: Long,
    sigma: Float,
    method: Int,
): SoftNmsResult {
    val updated = mat(0, 0, cvMakeType(CV_32F, 1))
    val indices = mat(0, 0, cvMakeType(CV_32S, 1))
    cvk_dnn_soft_nms_boxes(
        bboxes.mat.nativeHandle(), scores.mat.nativeHandle(), updated.nativeHandle(),
        scoreThreshold, nmsThreshold, indices.nativeHandle(), topK.toInt(), sigma, method,
    )
    return SoftNmsResult(MatOfFloat(updated), MatOfInt(indices))
}

actual fun dnnGetAvailableTargets(backend: Int): List<Int> =
    consumeIntArray(nativeMat(cvk_dnn_get_available_targets(backend), "getAvailableTargets")).toList()

actual fun dnnGetAvailableBackends(): List<Pair<Int, Int>> =
    consumeBackendPairs(nativeMat(cvk_dnn_get_available_backends(), "getAvailableBackends"))

actual fun dnnWriteTextGraph(model: String, output: String) {
    cvk_dnn_write_text_graph(model, output)
}

actual fun netCreate(): Net =
    NativeNet(cvk_net_create() ?: throw OpenCVException("netCreate", lastNativeError()))

actual fun tokenizerLoad(modelConfig: String): Tokenizer? =
    cvk_tokenizer_load(modelConfig)?.let { NativeTokenizer(it) }

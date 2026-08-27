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
// JVM (JNI-backed) implementation of the org.opencv.dnn surface.
// Net/Layer/Tokenizer handles are jlong pointers to the cvk_ C ABI structs;
// every call routes through [JniDnn], which forwards to the same shim the
// native targets bind via cinterop. Model-file readers return null (and
// record the native error) when the model cannot be read, exactly like the
// native actuals.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()

/** Requires a freshly-created non-zero handle or throws with the native error. */
private fun requireHandle(handle: Long, operation: String): Long =
    if (handle != 0L) handle else throw OpenCVException(operation, lastNativeError())

private fun stringOf(value: String?, operation: String): String =
    value ?: throw OpenCVException(operation, lastNativeError())

private fun flatStrings(data: ByteArray?, operation: String): List<String> =
    decodeStrings(data ?: throw OpenCVException(operation, lastNativeError()))

// =========================================================================
// wire helpers (actuals for the Dnn.kt internal expects)
// =========================================================================

internal actual fun matAddress(mat: Mat): Long = handleOf(mat)

internal actual fun matFromHandle(addr: Long): Mat = JvmMat(addr)

internal actual fun matDims(mat: Mat): Int = JniDnn.dnnMatDims(handleOf(mat))

internal actual fun matShape(mat: Mat): IntArray {
    val dims = matDims(mat)
    if (dims <= 0) return IntArray(0)
    return JniDnn.dnnMatShape(handleOf(mat)) ?: IntArray(0)
}

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

// =========================================================================
// Net
// =========================================================================

/** JNI-backed [Net] wrapping a `cv::Ptr<cv::dnn::Net>` handle. */
internal class JvmNet(@Volatile private var handle: Long) : Net {

    /** Raw handle for this slice's own methods and sibling slices. */
    internal fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Net is closed")

    override val empty: Boolean get() = JniDnn.netEmpty(check())

    override fun dump(): String = stringOf(JniDnn.netDump(check()), "dump")

    override fun dumpToFile(path: String) {
        JniDnn.netDumpToFile(check(), path)
    }

    override fun dumpToPbtxt(path: String) {
        JniDnn.netDumpToPbtxt(check(), path)
    }

    override fun getLayerId(layer: String): Int = JniDnn.netGetLayerId(check(), layer)

    override fun getLayerNames(): List<String> =
        flatStrings(JniDnn.netGetLayerNames(check()), "getLayerNames")

    override fun getLayer(layerId: Int): Layer =
        JvmLayer(requireHandle(JniDnn.netGetLayer(check(), layerId), "getLayer"))

    override fun getLayer(layerName: String): Layer =
        JvmLayer(requireHandle(JniDnn.netGetLayerByName(check(), layerName), "getLayer"))

    override fun connect(outPin: String, inpPin: String) {
        JniDnn.netConnect(check(), outPin, inpPin)
    }

    override fun registerOutput(outputName: String, layerId: Int, outputPort: Int): Int =
        JniDnn.netRegisterOutput(check(), outputName, layerId, outputPort)

    override fun setInputsNames(inputBlobNames: List<String>) {
        JniDnn.netSetInputsNames(check(), encodeStrings(inputBlobNames))
    }

    override fun setInputShape(inputName: String, shape: MatOfInt) {
        JniDnn.netSetInputShape(check(), inputName, handleOf(shape.mat))
    }

    override fun forward(outputName: String): Mat =
        jvmMat(JniDnn.netForward(check(), outputName), "forward")

    override fun forwardLayer(outputName: String): List<Mat> =
        decodeMatVector(jvmMat(JniDnn.netForwardLayer(check(), outputName), "forwardLayer"))

    override fun forward(outBlobNames: List<String>): List<Mat> =
        decodeMatVector(
            jvmMat(JniDnn.netForwardNames(check(), encodeStrings(outBlobNames)), "forward"),
        )

    override fun forwardAndRetrieve(outBlobNames: List<String>): List<List<Mat>> =
        decodeMatVector(
            jvmMat(
                JniDnn.netForwardAndRetrieve(check(), encodeStrings(outBlobNames)),
                "forwardAndRetrieve",
            ),
        ).map { decodeMatVector(it) }

    override fun setPreferableBackend(backendId: Int) {
        JniDnn.netSetPreferableBackend(check(), backendId)
    }

    override fun setPreferableTarget(targetId: Int) {
        JniDnn.netSetPreferableTarget(check(), targetId)
    }

    override fun finalizeNet() {
        JniDnn.netFinalize(check())
    }

    override fun setInput(blob: Mat, name: String, scalefactor: Double, mean: Scalar) {
        JniDnn.netSetInput(check(), handleOf(blob), name, scalefactor, mean.v0, mean.v1, mean.v2, mean.v3)
    }

    override fun setParam(layer: Int, numParam: Int, blob: Mat) {
        JniDnn.netSetParam(check(), layer, numParam, handleOf(blob))
    }

    override fun setParam(layerName: String, numParam: Int, blob: Mat) {
        JniDnn.netSetParamByName(check(), layerName, numParam, handleOf(blob))
    }

    override fun getParam(layer: Int, numParam: Int): Mat =
        jvmMat(JniDnn.netGetParam(check(), layer, numParam), "getParam")

    override fun getParam(layerName: String, numParam: Int): Mat =
        jvmMat(JniDnn.netGetParamByName(check(), layerName, numParam), "getParam")

    override fun getUnconnectedOutLayers(): MatOfInt =
        MatOfInt(jvmMat(JniDnn.netGetUnconnectedOutLayers(check()), "getUnconnectedOutLayers"))

    override fun getUnconnectedOutLayersNames(): List<String> =
        flatStrings(JniDnn.netGetUnconnectedOutLayersNames(check()), "getUnconnectedOutLayersNames")

    override fun getFLOPS(netInputShapes: List<MatOfInt>, netInputTypes: MatOfInt): Long {
        val shapes = encodeMatVector(netInputShapes.map { it.mat })
        return try {
            JniDnn.netGetFlops(check(), handleOf(shapes), handleOf(netInputTypes.mat))
        } finally {
            shapes.close()
        }
    }

    override fun getMemoryConsumption(
        netInputShapes: List<MatOfInt>,
        netInputTypes: MatOfInt,
    ): MemoryConsumption {
        val shapes = encodeMatVector(netInputShapes.map { it.mat })
        return try {
            val out = JniDnn.netGetMemoryConsumption(check(), handleOf(shapes), handleOf(netInputTypes.mat))
                ?: throw OpenCVException("getMemoryConsumption", lastNativeError())
            MemoryConsumption(weights = out[0], blobs = out[1])
        } finally {
            shapes.close()
        }
    }

    override fun getLayerTypes(): List<String> =
        flatStrings(JniDnn.netGetLayerTypes(check()), "getLayerTypes")

    override fun getLayersCount(layerType: String): Int =
        JniDnn.netGetLayersCount(check(), layerType)

    override fun enableFusion(fusion: Boolean) {
        JniDnn.netEnableFusion(check(), fusion)
    }

    override fun enableWinograd(useWinograd: Boolean) {
        JniDnn.netEnableWinograd(check(), useWinograd)
    }

    override fun getPerfProfile(timings: MatOfDouble): Long =
        JniDnn.netGetPerfProfile(check(), handleOf(timings.mat))

    override fun enableKVCache() {
        JniDnn.netEnableKvCache(check())
    }

    override fun disableKVCache() {
        JniDnn.netDisableKvCache(check())
    }

    override fun resetKVCache() {
        JniDnn.netResetKvCache(check())
    }

    override fun getPerfProfile(): PerfProfile =
        decodePerfProfile(
            JniDnn.netGetPerfProfileNames(check())
                ?: throw OpenCVException("getPerfProfile", lastNativeError()),
        )

    override fun printPerfProfile() {
        JniDnn.netPrintPerfProfile(check())
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniDnn.netRelease(h)
            // NOTE: the native handle is released via JniDnn.netRelease;
            // to reclaim at exit. Tracked as a known gap; once the JNI
            // function exists, call it here.
        }
    }
}

/** Raw handle of any JVM-backed [Net] (used by sibling slices). */
internal fun netHandleOf(net: Net): Long =
    (net as? JvmNet)?.check()
        ?: throw IllegalArgumentException("net belongs to another platform backend")

// =========================================================================
// Layer
// =========================================================================

/** JNI-backed [Layer] wrapping a `cv::Ptr<cv::dnn::Layer>` handle. */
internal class JvmLayer(@Volatile private var handle: Long) : Layer {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Layer is closed")

    override var blobs: List<Mat>
        get() = decodeMatVector(jvmMat(JniDnn.layerGetBlobs(check()), "getBlobs"))
        set(value) {
            val encoded = encodeMatVector(value)
            try {
                JniDnn.layerSetBlobs(check(), handleOf(encoded))
            } finally {
                encoded.close()
            }
        }

    override val name: String get() = stringOf(JniDnn.layerName(check()), "name")

    override val type: String get() = stringOf(JniDnn.layerType(check()), "type")

    override val preferableTarget: Int get() = JniDnn.layerPreferableTarget(check())

    override fun outputNameToIndex(outputName: String): Int =
        JniDnn.layerOutputNameToIndex(check(), outputName)

    override fun finalize(inputs: List<Mat>): List<Mat> {
        val inVec = encodeMatVector(inputs)
        val outMat = mat(0, 0, cvMakeType(CV_32S, 2))
        return try {
            JniDnn.layerFinalize(check(), handleOf(inVec), handleOf(outMat))
            decodeMatVector(outMat)
        } finally {
            inVec.close()
        }
    }

    override fun run(inputs: List<Mat>, internals: List<Mat>): Pair<List<Mat>, List<Mat>> {
        val inVec = encodeMatVector(inputs)
        val intVec = encodeMatVector(internals)
        val outMat = mat(0, 0, cvMakeType(CV_32S, 2))
        val intOutMat = mat(0, 0, cvMakeType(CV_32S, 2))
        return try {
            JniDnn.layerRun(check(), handleOf(inVec), handleOf(outMat), handleOf(intOutMat))
            decodeMatVector(outMat) to decodeMatVector(intOutMat)
        } finally {
            inVec.close()
            intVec.close()
        }
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniDnn.layerRelease(h)
        }
    }
}

// =========================================================================
// Tokenizer
// =========================================================================

/** JNI-backed [Tokenizer] wrapping a `cv::dnn::Tokenizer`. */
internal class JvmTokenizer(@Volatile private var handle: Long) : Tokenizer {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Tokenizer is closed")

    override fun encode(text: String): MatOfInt =
        MatOfInt(jvmMat(JniDnn.tokenizerEncode(check(), text), "encode"))

    override fun decode(tokens: MatOfInt): String =
        stringOf(JniDnn.tokenizerDecode(check(), handleOf(tokens.mat)), "decode")

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniDnn.tokenizerRelease(h)
        }
    }
}

// =========================================================================
// actual declarations
// =========================================================================

actual fun dnnReadNet(model: String, config: String, framework: String, engine: Int): Net? =
    JniDnn.dnnReadNet(model, config, framework, engine).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetBuffer(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte?,
    engine: Int,
): Net? = JniDnn.dnnReadNetBuffer(
    framework,
    handleOf(bufferModel.mat),
    bufferConfig?.let { handleOf(it.mat) } ?: 0L,
    engine,
).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromOnnx(onnxFile: String, engine: Int): Net? =
    JniDnn.dnnReadNetFromOnnx(onnxFile, engine).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromOnnxBuffer(buffer: MatOfByte, engine: Int): Net? =
    JniDnn.dnnReadNetFromOnnxBuffer(handleOf(buffer.mat), engine).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromTensorflow(model: String, config: String, engine: Int): Net? =
    JniDnn.dnnReadNetFromTensorflow(model, config, engine).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromTensorflowBuffer(
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte?,
    engine: Int,
): Net? = JniDnn.dnnReadNetFromTensorflowBuffer(
    handleOf(bufferModel.mat),
    bufferConfig?.let { handleOf(it.mat) } ?: 0L,
    engine,
).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromTflite(model: String, engine: Int): Net? =
    JniDnn.dnnReadNetFromTflite(model, engine).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromTfliteBuffer(bufferModel: MatOfByte, engine: Int): Net? =
    JniDnn.dnnReadNetFromTfliteBuffer(handleOf(bufferModel.mat), engine).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromModelOptimizer(xml: String, bin: String): Net? =
    JniDnn.dnnReadNetFromModelOptimizer(xml, bin).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadNetFromModelOptimizerBuffer(
    bufferModelConfig: MatOfByte,
    bufferWeights: MatOfByte,
): Net? = JniDnn.dnnReadNetFromModelOptimizerBuffer(
    handleOf(bufferModelConfig.mat),
    handleOf(bufferWeights.mat),
).takeIf { it != 0L }?.let(::JvmNet)

actual fun dnnReadTensorFromOnnx(path: String): Mat? =
    JniDnn.dnnReadTensorFromOnnx(path).takeIf { it != 0L }?.let(::JvmMat)

actual fun dnnBlobFromImage(
    image: Mat,
    scalefactor: Double,
    size: Size,
    mean: Scalar,
    swapRB: Boolean,
    crop: Boolean,
    ddepth: Int,
): Mat = jvmMat(
    JniDnn.dnnBlobFromImage(
        handleOf(image), scalefactor, size.width, size.height,
        mean.v0, mean.v1, mean.v2, mean.v3, swapRB, crop, ddepth,
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
        jvmMat(
            JniDnn.dnnBlobFromImages(
                handleOf(encoded), scalefactor, size.width, size.height,
                mean.v0, mean.v1, mean.v2, mean.v3, swapRB, crop, ddepth,
            ),
            "blobFromImages",
        )
    } finally {
        encoded.close()
    }
}

actual fun dnnImagesFromBlob(blob: Mat): List<Mat> =
    decodeMatVector(jvmMat(JniDnn.dnnImagesFromBlob(handleOf(blob)), "imagesFromBlob"))

actual fun dnnNmsBoxes(
    bboxes: MatOfRect2d,
    scores: MatOfFloat,
    scoreThreshold: Float,
    nmsThreshold: Float,
    eta: Float,
    topK: Int,
): MatOfInt {
    val indices = mat(0, 0, cvMakeType(CV_32S, 1))
    JniDnn.dnnNmsBoxes(
        handleOf(bboxes.mat), handleOf(scores.mat),
        scoreThreshold, nmsThreshold, handleOf(indices), eta, topK,
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
    JniDnn.dnnNmsBoxesRotated(
        handleOf(bboxes.mat), handleOf(scores.mat),
        scoreThreshold, nmsThreshold, handleOf(indices), eta, topK,
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
    JniDnn.dnnNmsBoxesBatched(
        handleOf(bboxes.mat), handleOf(scores.mat), handleOf(classIds.mat),
        scoreThreshold, nmsThreshold, handleOf(indices), eta, topK,
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
    JniDnn.dnnSoftNmsBoxes(
        handleOf(bboxes.mat), handleOf(scores.mat), handleOf(updated),
        scoreThreshold, nmsThreshold, handleOf(indices), topK, sigma, method,
    )
    return SoftNmsResult(MatOfFloat(updated), MatOfInt(indices))
}

actual fun dnnGetAvailableTargets(backend: Int): List<Int> =
    consumeIntArray(jvmMat(JniDnn.dnnGetAvailableTargets(backend), "getAvailableTargets")).toList()

actual fun dnnGetAvailableBackends(): List<Pair<Int, Int>> =
    consumeBackendPairs(jvmMat(JniDnn.dnnGetAvailableBackends(), "getAvailableBackends"))

actual fun dnnWriteTextGraph(model: String, output: String) {
    JniDnn.dnnWriteTextGraph(model, output)
}

actual fun netCreate(): Net =
    JvmNet(requireHandle(JniDnn.netCreate(), "netCreate"))

actual fun tokenizerLoad(modelConfig: String): Tokenizer? =
    JniDnn.tokenizerLoad(modelConfig).takeIf { it != 0L }?.let(::JvmTokenizer)

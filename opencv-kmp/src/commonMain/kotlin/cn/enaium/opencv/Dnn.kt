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
// org.opencv.dnn parity surface: Dnn statics, Net, Layer, DictValue,
// Tokenizer. Model / Image2BlobParams and friends live in the dnn2 slice.
//
// Model-file readers never crash on missing files: every dnnReadNet*
// factory returns null (and records the native error) instead of throwing,
// except the pure-image/blob helpers which throw OpenCVException on native
// failure like the rest of the binding.
// =========================================================================

// ---- ActivationType ------------------------------------------------------

const val ACTIV_NONE = 0
const val ACTIV_MISH = 1
const val ACTIV_SWISH = 2
const val ACTIV_SIGMOID = 3
const val ACTIV_TANH = 4
const val ACTIV_ELU = 5
const val ACTIV_HARDSWISH = 6
const val ACTIV_HARDSIGMOID = 7
const val ACTIV_GELU = 8
const val ACTIV_GELU_APPROX = 9
const val ACTIV_RELU = 10
const val ACTIV_CLIP = 11

// ---- ArgKind -------------------------------------------------------------

const val DNN_ARG_EMPTY = 0
const val DNN_ARG_CONST = 1
const val DNN_ARG_INPUT = 2
const val DNN_ARG_OUTPUT = 3
const val DNN_ARG_TEMP = 4
const val DNN_ARG_PATTERN = 5

// ---- AutoPadding ---------------------------------------------------------

const val AUTO_PAD_NONE = 0
const val AUTO_PAD_SAME_UPPER = 1
const val AUTO_PAD_SAME_LOWER = 2
const val AUTO_PAD_VALID = 3

// ---- Backend -------------------------------------------------------------

const val DNN_BACKEND_DEFAULT = 0
const val DNN_BACKEND_INFERENCE_ENGINE = 2
const val DNN_BACKEND_OPENCV = 3
const val DNN_BACKEND_VKCOM = 4
const val DNN_BACKEND_CUDA = 5
const val DNN_BACKEND_WEBNN = 6
const val DNN_BACKEND_TIMVX = 7
const val DNN_BACKEND_CANN = 8

/**
 * Engine selection for the model readers.
 *
 * NOTE: values follow this repository's `cv::dnn::EngineType` header
 * (ENGINE_AUTO=0, ENGINE_OPENCV=1, ENGINE_ORT=2), which is what the linked
 * native library accepts. The Android SDK 5.0.0 Java constants
 * (ENGINE_CLASSIC=1, ENGINE_NEW=2, ENGINE_AUTO=3, ENGINE_ORT=4) are stale
 * relative to this header and must NOT be used.
 */
const val ENGINE_AUTO = 0
const val ENGINE_OPENCV = 1
const val ENGINE_ORT = 2

// ---- ImagePaddingMode ----------------------------------------------------

const val DNN_PMODE_NULL = 0
const val DNN_PMODE_CROP_CENTER = 1
const val DNN_PMODE_LETTERBOX = 2

// ---- LossReduction -------------------------------------------------------

const val LOSS_REDUCTION_NONE = 0
const val LOSS_REDUCTION_MEAN = 1
const val LOSS_REDUCTION_SUM = 2

// ---- ModelFormat ---------------------------------------------------------

const val DNN_MODEL_GENERIC = 0
const val DNN_MODEL_ONNX = 1
const val DNN_MODEL_TF = 2
const val DNN_MODEL_TFLITE = 3

// ---- NaryEltwiseLayer.OPERATION ------------------------------------------

const val OPERATION_AND = 0
const val OPERATION_EQUAL = 1
const val OPERATION_GREATER = 2
const val OPERATION_GREATER_EQUAL = 3
const val OPERATION_LESS = 4
const val OPERATION_LESS_EQUAL = 5
const val OPERATION_OR = 6
const val OPERATION_POW = 7
const val OPERATION_XOR = 8
const val OPERATION_BITSHIFT = 9
const val OPERATION_MAX = 10
const val OPERATION_MEAN = 11
const val OPERATION_MIN = 12
const val OPERATION_MOD = 13
const val OPERATION_FMOD = 14
const val OPERATION_PROD = 15
const val OPERATION_SUB = 16
const val OPERATION_SUM = 17
const val OPERATION_ADD = 18
const val OPERATION_DIV = 19
const val OPERATION_WHERE = 20
const val OPERATION_BITWISE_AND = 21
const val OPERATION_BITWISE_OR = 22
const val OPERATION_BITWISE_XOR = 23

// ---- ProfilingMode -------------------------------------------------------

const val DNN_PROFILE_NONE = 0
const val DNN_PROFILE_SUMMARY = 1
const val DNN_PROFILE_DETAILED = 2

// ---- Reduce2Layer.ReduceType ---------------------------------------------

const val ReduceType_MAX = 0
const val ReduceType_MIN = 1
const val ReduceType_MEAN = 2
const val ReduceType_SUM = 3
const val ReduceType_L1 = 4
const val ReduceType_L2 = 5
const val ReduceType_PROD = 6
const val ReduceType_SUM_SQUARE = 7
const val ReduceType_LOG_SUM = 8
const val ReduceType_LOG_SUM_EXP = 9

// ---- SoftNMSMethod -------------------------------------------------------

const val SoftNMSMethod_SOFTNMS_LINEAR = 1
const val SoftNMSMethod_SOFTNMS_GAUSSIAN = 2

// ---- Target --------------------------------------------------------------

const val DNN_TARGET_CPU = 0
const val DNN_TARGET_OPENCL = 1
const val DNN_TARGET_OPENCL_FP16 = 2
const val DNN_TARGET_MYRIAD = 3
const val DNN_TARGET_VULKAN = 4
const val DNN_TARGET_FPGA = 5
const val DNN_TARGET_CUDA = 6
const val DNN_TARGET_CUDA_FP16 = 7
const val DNN_TARGET_HDDL = 8
const val DNN_TARGET_NPU = 9
const val DNN_TARGET_CPU_FP16 = 10

// ---- TracingMode ---------------------------------------------------------

const val DNN_TRACE_NONE = 0
const val DNN_TRACE_ALL = 1
const val DNN_TRACE_OP = 2

// =========================================================================
// value types
// =========================================================================

/**
 * Bytes required to store a loaded model's learned weights and its
 * intermediate blobs, reported by [Net.getMemoryConsumption].
 */
data class MemoryConsumption(val weights: Long, val blobs: Long)

/** Profiling rows from [Net.getPerfProfile]; entries sorted by time desc. */
data class PerfProfile(
    val names: List<String>,
    val timems: List<String>,
    val counts: List<String>,
)

/** Result of [dnnSoftNmsBoxes]: the updated confidences and kept indices. */
data class SoftNmsResult(val updatedScores: MatOfFloat, val indices: MatOfInt)

/**
 * Scalar value (or array) of one of the following types: int, double or
 * String. Mirrors `cv::dnn::DictValue`; pure Kotlin — no native handle.
 */
class DictValue internal constructor(internal val values: List<Any>) {

    /** Constructs an integer scalar. */
    constructor(i: Int) : this(listOf(i))

    /** Constructs a floating point scalar. */
    constructor(d: Double) : this(listOf(d))

    /** Constructs a string scalar. */
    constructor(s: String) : this(listOf(s))

    /** Whether the value is stored as an integer. */
    val isInt: Boolean get() = values.all { it is Int }

    /** Whether the value is stored as a floating point number. */
    val isReal: Boolean get() = values.all { it is Double }

    /** Whether the value is stored as a string. */
    val isString: Boolean get() = values.all { it is String }

    /** Element at [idx]; a negative index addresses the last element. */
    private fun at(idx: Int): Any {
        val i = if (idx < 0) values.lastIndex else idx
        require(i in values.indices) { "DictValue index $idx out of bounds (size=${values.size})" }
        return values[i]
    }

    /** The element at [idx] converted to an integer (truncating reals). */
    fun getIntValue(idx: Int = -1): Int = when (val v = at(idx)) {
        is Int -> v
        is Double -> v.toInt()
        else -> throw IllegalArgumentException("DictValue holds a String, not an int")
    }

    /** The element at [idx] converted to a double. */
    fun getRealValue(idx: Int = -1): Double = when (val v = at(idx)) {
        is Double -> v
        is Int -> v.toDouble()
        else -> throw IllegalArgumentException("DictValue holds a String, not a real")
    }

    /** The element at [idx] as a string. */
    fun getStringValue(idx: Int = -1): String = when (val v = at(idx)) {
        is String -> v
        else -> throw IllegalArgumentException("DictValue is not a string")
    }
}

// =========================================================================
// interfaces
// =========================================================================

/**
 * A deep neural network (`cv::dnn::Net`): a DAG of [Layer]s. Supports
 * reference counting of its instances; closing this handle releases one
 * reference. Construct via [dnnReadNet] family or [netCreate].
 */
interface Net : AutoCloseable {

    /** True when there are no layers in the network. */
    val empty: Boolean

    /** Dumps the net structure/hyperparameters to a string. */
    fun dump(): String

    /** Dumps the net structure to a `.dot` file. */
    fun dumpToFile(path: String)

    /** Dumps the net structure to a `.pbtxt` file. */
    fun dumpToPbtxt(path: String)

    /** Converts a layer name to its integer id, or -1 when not found. */
    fun getLayerId(layer: String): Int

    /** Names of every layer in the network. */
    fun getLayerNames(): List<String>

    /** Returns the layer with the given integer id (call [Layer.close]). */
    fun getLayer(layerId: Int): Layer

    /** Returns the layer with the given name (call [Layer.close]). */
    fun getLayer(layerName: String): Layer

    /** Connects output of the first layer to input of the second. */
    fun connect(outPin: String, inpPin: String)

    /** Registers a network output; returns the index of the bound layer. */
    fun registerOutput(outputName: String, layerId: Int, outputPort: Int): Int

    /** Sets the output names of the network input pseudo layer. */
    fun setInputsNames(inputBlobNames: List<String>)

    /** Specifies the shape of a network input. */
    fun setInputShape(inputName: String, shape: MatOfInt)

    /**
     * Runs a forward pass and returns the first output blob of the layer
     * named [outputName]; an empty name runs the whole network.
     */
    fun forward(outputName: String = ""): Mat

    /**
     * Runs a forward pass and returns every output blob of the layer named
     * [outputName] (all outputs when the name is empty).
     */
    fun forwardLayer(outputName: String = ""): List<Mat>

    /** Runs a forward pass for the layers in [outBlobNames]; first output each. */
    fun forward(outBlobNames: List<String>): List<Mat>

    /** Runs a forward pass returning ALL output blobs of each named layer. */
    fun forwardAndRetrieve(outBlobNames: List<String>): List<List<Mat>>

    /** Asks the network to use a specific computation backend. */
    fun setPreferableBackend(backendId: Int)

    /** Asks the network to compute on a specific target device. */
    fun setPreferableTarget(targetId: Int)

    /** Finalizes the network configuration and prepares it for inference. */
    fun finalizeNet()

    /**
     * Sets the new input value for the network; the final input blob is
     * `scalefactor * (blob - mean)`.
     */
    fun setInput(blob: Mat, name: String = "", scalefactor: Double = 1.0, mean: Scalar = Scalar())

    /** Sets the learned parameter blob of the layer with the given id. */
    fun setParam(layer: Int, numParam: Int, blob: Mat)

    /** Sets the learned parameter blob of the layer with the given name. */
    fun setParam(layerName: String, numParam: Int, blob: Mat)

    /** Returns the parameter blob of the layer with the given id. */
    fun getParam(layer: Int, numParam: Int = 0): Mat

    /** Returns the parameter blob of the layer with the given name. */
    fun getParam(layerName: String, numParam: Int = 0): Mat

    /** Indexes of layers with unconnected outputs. */
    fun getUnconnectedOutLayers(): MatOfInt

    /** Names of layers with unconnected outputs. */
    fun getUnconnectedOutLayersNames(): List<String>

    /** Computes the FLOPs of the loaded model for the given input shapes. */
    fun getFLOPS(netInputShapes: List<MatOfInt>, netInputTypes: MatOfInt): Long

    /** Bytes needed to store all weights and intermediate blobs. */
    fun getMemoryConsumption(netInputShapes: List<MatOfInt>, netInputTypes: MatOfInt): MemoryConsumption

    /** Types of the layers used in the model. */
    fun getLayerTypes(): List<String>

    /** Count of layers of the given type. */
    fun getLayersCount(layerType: String): Int

    /** Enables or disables layer fusion. */
    fun enableFusion(fusion: Boolean)

    /** Enables or disables the Winograd compute branch. */
    fun enableWinograd(useWinograd: Boolean)

    /**
     * Returns the overall inference time (ticks) and fills [timings] with
     * the per-layer tick timings.
     */
    fun getPerfProfile(timings: MatOfDouble): Long

    /** Enables KV-Cache for all AttentionOnnxI layers. */
    fun enableKVCache()

    /** Disables KV-Cache for all AttentionOnnxI layers. */
    fun disableKVCache()

    /** Resets KV-Cache for all AttentionOnnxI layers. */
    fun resetKVCache()

    /** Profiling data captured during the last forward pass. */
    fun getPerfProfile(): PerfProfile

    /** Prints the profile captured during the last forward pass. */
    fun printPerfProfile()

    override fun close()
}

/**
 * A network layer (`cv::dnn::Layer`) — building block of a [Net]. Returned
 * by [Net.getLayer]; close the handle when done.
 */
interface Layer : AutoCloseable {

    /** Learned parameter blobs of the layer. */
    var blobs: List<Mat>

    /** Layer name. */
    val name: String

    /** Layer type string (e.g. "Convolution", "ReLU"). */
    val type: String

    /** Preferred target for layer forwarding (a DNN_TARGET_* value). */
    val preferableTarget: Int

    /** Index of the output blob named [outputName], or -1. */
    fun outputNameToIndex(outputName: String): Int

    /** Computes internal parameters from [inputs]; returns the output blobs. */
    fun finalize(inputs: List<Mat>): List<Mat>

    /**
     * Allocates the layer and computes the output from [inputs]; returns
     * (outputs, internals) as freshly allocated blobs.
     */
    fun run(inputs: List<Mat>, internals: List<Mat> = emptyList()): Pair<List<Mat>, List<Mat>>

    override fun close()
}

/**
 * High-level tokenizer wrapper for DNN usage (`cv::dnn::Tokenizer`):
 * loads a BPE tokenizer from a model directory and encodes/decodes text.
 */
interface Tokenizer : AutoCloseable {

    /** Encodes UTF-8 [text] into token ids (special tokens disabled). */
    fun encode(text: String): MatOfInt

    /** Decodes [tokens] back into text. */
    fun decode(tokens: MatOfInt): String

    override fun close()
}

// =========================================================================
// expect factories
// =========================================================================

/**
 * Reads a network in one of the supported formats, auto-detecting the
 * framework (see [dnnReadNetFromOnnx], [dnnReadNetFromTensorflow]).
 * Returns null (recording the native error) when the model cannot be read.
 */
expect fun dnnReadNet(
    model: String,
    config: String = "",
    framework: String = "",
    engine: Int = ENGINE_AUTO,
): Net?

/** readNet from in-memory buffers; returns null when the model is invalid. */
expect fun dnnReadNetBuffer(
    framework: String,
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte? = null,
    engine: Int = ENGINE_AUTO,
): Net?

/** Reads an ONNX model from a file; null when the file is missing/invalid. */
expect fun dnnReadNetFromOnnx(onnxFile: String, engine: Int = ENGINE_AUTO): Net?

/** Reads an ONNX model from an in-memory buffer; null when invalid. */
expect fun dnnReadNetFromOnnxBuffer(buffer: MatOfByte, engine: Int = ENGINE_AUTO): Net?

/** Reads a TensorFlow model (.pb + optional .pbtxt); null when unreadable. */
expect fun dnnReadNetFromTensorflow(
    model: String,
    config: String = "",
    engine: Int = ENGINE_AUTO,
): Net?

/** Reads a TensorFlow model from in-memory buffers; null when invalid. */
expect fun dnnReadNetFromTensorflowBuffer(
    bufferModel: MatOfByte,
    bufferConfig: MatOfByte? = null,
    engine: Int = ENGINE_AUTO,
): Net?

/** Reads a TFLite model from a file; null when the file is missing/invalid. */
expect fun dnnReadNetFromTflite(model: String, engine: Int = ENGINE_AUTO): Net?

/** Reads a TFLite model from an in-memory buffer; null when invalid. */
expect fun dnnReadNetFromTfliteBuffer(bufferModel: MatOfByte, engine: Int = ENGINE_AUTO): Net?

/** Loads a network from Intel Model Optimizer IR (xml + bin); null on failure. */
expect fun dnnReadNetFromModelOptimizer(xml: String, bin: String = ""): Net?

/** Loads a network from Model Optimizer IR in-memory buffers; null on failure. */
expect fun dnnReadNetFromModelOptimizerBuffer(bufferModelConfig: MatOfByte, bufferWeights: MatOfByte): Net?

/** Creates a tensor blob from an ONNX `.pb` tensor file; null on failure. */
expect fun dnnReadTensorFromOnnx(path: String): Mat?

/**
 * Creates a 4-dimensional NCHW blob from an image, optionally resizing,
 * cropping, subtracting [mean] and swapping Red/Blue. The output is
 * `(input - mean) * scalefactor` with NCHW dimension order.
 */
expect fun dnnBlobFromImage(
    image: Mat,
    scalefactor: Double = 1.0,
    size: Size = Size(0, 0),
    mean: Scalar = Scalar(),
    swapRB: Boolean = false,
    crop: Boolean = false,
    ddepth: Int = CV_32F,
): Mat

/** [dnnBlobFromImage] over a batch of images. */
expect fun dnnBlobFromImages(
    images: List<Mat>,
    scalefactor: Double = 1.0,
    size: Size = Size(0, 0),
    mean: Scalar = Scalar(),
    swapRB: Boolean = false,
    crop: Boolean = false,
    ddepth: Int = CV_32F,
): Mat

/**
 * Parses a 4D blob and returns the images it contains as 2D CV_32F Mats
 * (batch size = first blob dimension); the inverse of [dnnBlobFromImage].
 */
expect fun dnnImagesFromBlob(blob: Mat): List<Mat>

/**
 * Non maximum suppression over axis-aligned boxes. Returns the kept box
 * indices ordered by descending score.
 */
expect fun dnnNmsBoxes(
    bboxes: MatOfRect2d,
    scores: MatOfFloat,
    scoreThreshold: Float,
    nmsThreshold: Float,
    eta: Float = 1f,
    topK: Int = 0,
): MatOfInt

/** [dnnNmsBoxes] over rotated rectangles. */
expect fun dnnNmsBoxesRotated(
    bboxes: MatOfRotatedRect,
    scores: MatOfFloat,
    scoreThreshold: Float,
    nmsThreshold: Float,
    eta: Float = 1f,
    topK: Int = 0,
): MatOfInt

/** Batched [dnnNmsBoxes] across classes (class ids must start at 0). */
expect fun dnnNmsBoxesBatched(
    bboxes: MatOfRect2d,
    scores: MatOfFloat,
    classIds: MatOfInt,
    scoreThreshold: Float,
    nmsThreshold: Float,
    eta: Float = 1f,
    topK: Int = 0,
): MatOfInt

/**
 * Soft non maximum suppression; returns the updated confidences and the
 * kept indices. [method] is a SoftNMSMethod_SOFTNMS_* value.
 */
expect fun dnnSoftNmsBoxes(
    bboxes: MatOfRect,
    scores: MatOfFloat,
    scoreThreshold: Float,
    nmsThreshold: Float,
    topK: Long = 0,
    sigma: Float = 0.5f,
    method: Int = SoftNMSMethod_SOFTNMS_GAUSSIAN,
): SoftNmsResult

/** Targets supported by the given backend (DNN_TARGET_* values). */
expect fun dnnGetAvailableTargets(backend: Int): List<Int>

/** (backend, target) pairs supported by the built library. */
expect fun dnnGetAvailableBackends(): List<Pair<Int, Int>>

/** Creates a text representation of a binary network (weights omitted). */
expect fun dnnWriteTextGraph(model: String, output: String)

/** Creates an empty network (cv::dnn::Net default constructor). */
expect fun netCreate(): Net

/**
 * Loads a tokenizer from a model directory (config.json + tokenizer.json).
 * Returns null (recording the native error) when the files are missing.
 */
expect fun tokenizerLoad(modelConfig: String): Tokenizer?

// =========================================================================
// internal wire helpers (shared by the platform actuals)
//
// vector<Mat> travels as a CV_32SC2 Mat whose rows are the 64-bit heap
// addresses of the member Mats, exactly like the official Java Converters
// layout. std::vector<std::string> travels as a flat buffer:
// [u32le count][per string: u32le byte_len + UTF-8].
// =========================================================================

/** Raw native address of a Mat handle (platform-specific). */
internal expect fun matAddress(mat: Mat): Long

/** Wraps a raw native Mat address into a platform Mat (caller-owned). */
internal expect fun matFromHandle(addr: Long): Mat

/** Number of dimensions of the Mat (2 for the common case). */
internal expect fun matDims(mat: Mat): Int

/** Dimensions of the Mat in native `size.p` order (W, H, C, N for NCHW). */
internal expect fun matShape(mat: Mat): IntArray

/** Encodes a list of Mats into a CV_32SC2 Mat-of-Mats. */
internal fun encodeMatVector(mats: List<Mat>): Mat {
    val m = mat(mats.size, 1, cvMakeType(CV_32S, 2))
    val bytes = ByteArray(mats.size * 8)
    for (i in mats.indices) {
        val addr = matAddress(mats[i])
        bytes.writeIntLE(i * 8, addr.toInt())
        bytes.writeIntLE(i * 8 + 4, (addr shr 32).toInt())
    }
    m.pixels = bytes
    return m
}

/**
 * Decodes a CV_32SC2 Mat-of-Mats into a list of caller-owned Mats and
 * closes the container Mat.
 */
internal fun decodeMatVector(addrMat: Mat): List<Mat> {
    val bytes = addrMat.pixels
    val n = bytes.size / 8
    val out = ArrayList<Mat>(n)
    for (i in 0 until n) {
        val lo = bytes.readIntLE(i * 8).toLong() and 0xFFFFFFFFL
        val hi = bytes.readIntLE(i * 8 + 4).toLong() and 0xFFFFFFFFL
        out.add(matFromHandle((hi shl 32) or lo))
    }
    addrMat.close()
    return out
}

/** Encodes a list of strings into the flat string-buffer wire format. */
internal fun encodeStrings(strings: List<String>): ByteArray {
    val encoded = strings.map { it.encodeToByteArray() }
    var size = 4
    for (e in encoded) size += 4 + e.size
    val out = ByteArray(size)
    out.writeIntLE(0, strings.size)
    var off = 4
    for (e in encoded) {
        out.writeIntLE(off, e.size)
        off += 4
        e.copyInto(out, off)
        off += e.size
    }
    return out
}

/** Decodes the flat string-buffer wire format into a list of strings. */
internal fun decodeStrings(data: ByteArray): List<String> {
    val n = data.readIntLE(0)
    val out = ArrayList<String>(n)
    var off = 4
    repeat(n) {
        val len = data.readIntLE(off)
        off += 4
        out.add(data.decodeToString(off, off + len))
        off += len
    }
    return out
}

/** Reads an Nx1 CV_32S Mat into an IntArray and closes the Mat. */
internal fun consumeIntArray(m: Mat): IntArray {
    val bytes = m.pixels
    val out = IntArray(bytes.size / 4)
    var i = 0
    var b = 0
    while (i < out.size) {
        out[i] = bytes.readIntLE(b)
        i++
        b += 4
    }
    m.close()
    return out
}

/** Reads an Nx2 CV_32SC2 Mat into (backend, target) pairs and closes it. */
internal fun consumeBackendPairs(m: Mat): List<Pair<Int, Int>> {
    val bytes = m.pixels
    val out = ArrayList<Pair<Int, Int>>(bytes.size / 8)
    var b = 0
    while (b + 8 <= bytes.size) {
        out.add(bytes.readIntLE(b) to bytes.readIntLE(b + 4))
        b += 8
    }
    m.close()
    return out
}

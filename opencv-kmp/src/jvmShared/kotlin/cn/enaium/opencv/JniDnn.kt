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
 * JNI bridge for the org.opencv.dnn surface (DnnCore slice).
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniDnn_<name>`
 * function in jni/jni_dnn.cpp. Net/Layer/Tokenizer/Mat handles travel as
 * jlong pointers; flat string buffers and the perf-profile buffer come back
 * as ByteArray; NMS outputs are filled in place through caller-owned Mat
 * handles. Scalar values are expanded into v0..v3 primitive arguments.
 */
internal object JniDnn {

    // Touching Jni here runs its init block, which loads the native library
    // before any of these external functions can be invoked.
    init {
        Jni.lastError()
    }

    // ---------------------------------------------------------------- dnn statics

    external fun dnnReadNet(model: String, config: String, framework: String, engine: Int): Long
    external fun dnnReadNetBuffer(
        framework: String,
        bufferModel: Long,
        bufferConfig: Long,
        engine: Int,
    ): Long
    external fun dnnReadNetFromOnnx(onnxFile: String, engine: Int): Long
    external fun dnnReadNetFromOnnxBuffer(buffer: Long, engine: Int): Long
    external fun dnnReadNetFromTensorflow(model: String, config: String, engine: Int): Long
    external fun dnnReadNetFromTensorflowBuffer(
        bufferModel: Long,
        bufferConfig: Long,
        engine: Int,
    ): Long
    external fun dnnReadNetFromTflite(model: String, engine: Int): Long
    external fun dnnReadNetFromTfliteBuffer(bufferModel: Long, engine: Int): Long
    external fun dnnReadNetFromModelOptimizer(xml: String, bin: String): Long
    external fun dnnReadNetFromModelOptimizerBuffer(bufferConfig: Long, bufferWeights: Long): Long
    external fun dnnReadTensorFromOnnx(path: String): Long
    external fun dnnBlobFromImage(
        image: Long,
        scalefactor: Double,
        sizeW: Int,
        sizeH: Int,
        v0: Double, v1: Double, v2: Double, v3: Double,
        swapRB: Boolean,
        crop: Boolean,
        ddepth: Int,
    ): Long
    external fun dnnBlobFromImages(
        images: Long,
        scalefactor: Double,
        sizeW: Int,
        sizeH: Int,
        v0: Double, v1: Double, v2: Double, v3: Double,
        swapRB: Boolean,
        crop: Boolean,
        ddepth: Int,
    ): Long
    external fun dnnImagesFromBlob(blob: Long): Long
    external fun dnnNmsBoxes(
        bboxes: Long,
        scores: Long,
        scoreThreshold: Float,
        nmsThreshold: Float,
        indices: Long,
        eta: Float,
        topK: Int,
    )
    external fun dnnNmsBoxesRotated(
        bboxes: Long,
        scores: Long,
        scoreThreshold: Float,
        nmsThreshold: Float,
        indices: Long,
        eta: Float,
        topK: Int,
    )
    external fun dnnNmsBoxesBatched(
        bboxes: Long,
        scores: Long,
        classIds: Long,
        scoreThreshold: Float,
        nmsThreshold: Float,
        indices: Long,
        eta: Float,
        topK: Int,
    )
    external fun dnnSoftNmsBoxes(
        bboxes: Long,
        scores: Long,
        updatedScores: Long,
        scoreThreshold: Float,
        nmsThreshold: Float,
        indices: Long,
        topK: Long,
        sigma: Float,
        method: Int,
    )
    external fun dnnGetAvailableTargets(backend: Int): Long
    external fun dnnGetAvailableBackends(): Long
    external fun dnnWriteTextGraph(model: String, output: String)

    // --------------------------------------------------------------------- Net

    external fun netCreate(): Long
    external fun netRelease(net: Long)
    external fun netEmpty(net: Long): Boolean
    external fun netDump(net: Long): String?
    external fun netDumpToFile(net: Long, path: String)
    external fun netDumpToPbtxt(net: Long, path: String)
    external fun netGetLayerId(net: Long, layer: String): Int
    external fun netGetLayerNames(net: Long): ByteArray?
    external fun netGetLayer(net: Long, layerId: Int): Long
    external fun netGetLayerByName(net: Long, layerName: String): Long
    external fun netConnect(net: Long, outPin: String, inpPin: String)
    external fun netRegisterOutput(net: Long, outputName: String, layerId: Int, outputPort: Int): Int
    external fun netSetInputsNames(net: Long, names: ByteArray)
    external fun netSetInputShape(net: Long, inputName: String, shape: Long)
    external fun netForward(net: Long, outputName: String): Long
    external fun netForwardLayer(net: Long, outputName: String): Long
    external fun netForwardNames(net: Long, names: ByteArray): Long
    external fun netForwardAndRetrieve(net: Long, names: ByteArray): Long
    external fun netSetPreferableBackend(net: Long, backendId: Int)
    external fun netSetPreferableTarget(net: Long, targetId: Int)
    external fun netFinalize(net: Long)
    external fun netSetInput(
        net: Long,
        blob: Long,
        name: String,
        scalefactor: Double,
        v0: Double, v1: Double, v2: Double, v3: Double,
    )
    external fun netSetParam(net: Long, layer: Int, numParam: Int, blob: Long)
    external fun netSetParamByName(net: Long, layerName: String, numParam: Int, blob: Long)
    external fun netGetParam(net: Long, layer: Int, numParam: Int): Long
    external fun netGetParamByName(net: Long, layerName: String, numParam: Int): Long
    external fun netGetUnconnectedOutLayers(net: Long): Long
    external fun netGetUnconnectedOutLayersNames(net: Long): ByteArray?
    external fun netGetFlops(net: Long, inputShapes: Long, inputTypes: Long): Long
    external fun netGetMemoryConsumption(net: Long, inputShapes: Long, inputTypes: Long): LongArray?
    external fun netGetLayerTypes(net: Long): ByteArray?
    external fun netGetLayersCount(net: Long, layerType: String): Int
    external fun netEnableFusion(net: Long, fusion: Boolean)
    external fun netEnableWinograd(net: Long, useWinograd: Boolean)
    external fun netGetPerfProfile(net: Long, timings: Long): Long
    external fun netEnableKvCache(net: Long)
    external fun netDisableKvCache(net: Long)
    external fun netResetKvCache(net: Long)
    external fun netGetPerfProfileNames(net: Long): ByteArray?
    external fun netPrintPerfProfile(net: Long)

    // ------------------------------------------------------------------- Layer

    external fun layerRelease(layer: Long)
    external fun layerName(layer: Long): String?
    external fun layerType(layer: Long): String?
    external fun layerPreferableTarget(layer: Long): Int
    external fun layerOutputNameToIndex(layer: Long, outputName: String): Int
    external fun layerGetBlobs(layer: Long): Long
    external fun layerSetBlobs(layer: Long, blobs: Long)
    external fun layerFinalize(layer: Long, inputs: Long, outputs: Long)
    external fun layerRun(layer: Long, inputs: Long, outputs: Long, internals: Long)

    // -------------------------------------------------------------- Tokenizer

    external fun tokenizerLoad(modelConfig: String): Long
    external fun tokenizerRelease(tokenizer: Long)
    external fun tokenizerEncode(tokenizer: Long, text: String): Long
    external fun tokenizerDecode(tokenizer: Long, tokens: Long): String?

    // ------------------------------------------- dnn Mat introspection helpers

    external fun dnnMatDims(mat: Long): Int
    external fun dnnMatShape(mat: Long): IntArray?
}

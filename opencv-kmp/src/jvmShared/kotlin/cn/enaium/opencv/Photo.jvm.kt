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
// JVM (JNI-backed) implementation of the photo module.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()


/** Unpacks a two-handle jlongArray output into a Mat pair. */
private fun jvmPair(handles: LongArray, operation: String): Pair<Mat, Mat> {
    if (handles.size != 2) throw OpenCVException(operation, lastNativeError())
    return jvmMat(handles[0], operation) to jvmMat(handles[1], operation)
}

/** Unpacks a nullable handle list; throws when the native call failed. */
private fun jvmMatList(handles: LongArray?, operation: String): List<Mat> {
    val hs = handles ?: throw OpenCVException(operation, lastNativeError())
    if (hs.isEmpty()) throw OpenCVException(operation, lastNativeError())
    return hs.map { jvmMat(it, operation) }
}

// =========================================================================
// Photo statics
// =========================================================================

actual fun photoInpaint(src: Mat, inpaintMask: Mat, inpaintRadius: Double, flags: Int): Mat =
    jvmMat(
        JniPhoto.inpaint(handleOf(src), handleOf(inpaintMask), inpaintRadius, flags),
        "photoInpaint",
    )

actual fun photoFastNlMeansDenoising(
    src: Mat,
    h: Float,
    templateWindowSize: Int,
    searchWindowSize: Int,
): Mat = jvmMat(
    JniPhoto.fastNlMeansDenoising(handleOf(src), h, templateWindowSize, searchWindowSize),
    "photoFastNlMeansDenoising",
)

actual fun photoFastNlMeansDenoising(
    src: Mat,
    h: MatOfFloat,
    templateWindowSize: Int,
    searchWindowSize: Int,
    normType: Int,
): Mat = jvmMat(
    JniPhoto.fastNlMeansDenoisingH(
        handleOf(src), handleOf(h.mat), templateWindowSize, searchWindowSize, normType,
    ),
    "photoFastNlMeansDenoising(h)",
)

actual fun photoFastNlMeansDenoisingColored(
    src: Mat,
    h: Float,
    hColor: Float,
    templateWindowSize: Int,
    searchWindowSize: Int,
): Mat = jvmMat(
    JniPhoto.fastNlMeansDenoisingColored(
        handleOf(src), h, hColor, templateWindowSize, searchWindowSize,
    ),
    "photoFastNlMeansDenoisingColored",
)

actual fun photoFastNlMeansDenoisingMulti(
    src: List<Mat>,
    imgToDenoiseIndex: Int,
    temporalWindowSize: Int,
    h: Float,
    templateWindowSize: Int,
    searchWindowSize: Int,
): Mat {
    require(src.isNotEmpty()) { "photoFastNlMeansDenoisingMulti needs at least one image" }
    return jvmMat(
        JniPhoto.fastNlMeansDenoisingMulti(
            src.map { handleOf(it) }.toLongArray(), imgToDenoiseIndex, temporalWindowSize,
            h, templateWindowSize, searchWindowSize,
        ),
        "photoFastNlMeansDenoisingMulti",
    )
}

actual fun photoFastNlMeansDenoisingMulti(
    src: List<Mat>,
    imgToDenoiseIndex: Int,
    temporalWindowSize: Int,
    h: MatOfFloat,
    templateWindowSize: Int,
    searchWindowSize: Int,
    normType: Int,
): Mat {
    require(src.isNotEmpty()) { "photoFastNlMeansDenoisingMulti needs at least one image" }
    return jvmMat(
        JniPhoto.fastNlMeansDenoisingMultiH(
            src.map { handleOf(it) }.toLongArray(), imgToDenoiseIndex, temporalWindowSize,
            handleOf(h.mat), templateWindowSize, searchWindowSize, normType,
        ),
        "photoFastNlMeansDenoisingMulti(h)",
    )
}

actual fun photoFastNlMeansDenoisingColoredMulti(
    src: List<Mat>,
    imgToDenoiseIndex: Int,
    temporalWindowSize: Int,
    h: Float,
    hColor: Float,
    templateWindowSize: Int,
    searchWindowSize: Int,
): Mat {
    require(src.isNotEmpty()) { "photoFastNlMeansDenoisingColoredMulti needs at least one image" }
    return jvmMat(
        JniPhoto.fastNlMeansDenoisingColoredMulti(
            src.map { handleOf(it) }.toLongArray(), imgToDenoiseIndex, temporalWindowSize,
            h, hColor, templateWindowSize, searchWindowSize,
        ),
        "photoFastNlMeansDenoisingColoredMulti",
    )
}

actual fun photoDenoiseTvl1(observations: List<Mat>, lambda: Double, niters: Int): Mat {
    require(observations.isNotEmpty()) { "photoDenoiseTvl1 needs at least one observation" }
    return jvmMat(
        JniPhoto.denoiseTvl1(observations.map { handleOf(it) }.toLongArray(), lambda, niters),
        "photoDenoiseTvl1",
    )
}

actual fun photoDecolor(src: Mat): Pair<Mat, Mat> =
    jvmPair(JniPhoto.decolor(handleOf(src)), "photoDecolor")

actual fun photoSeamlessClone(src: Mat, dst: Mat, mask: Mat, p: Point, flags: Int): Mat =
    jvmMat(
        JniPhoto.seamlessClone(handleOf(src), handleOf(dst), handleOf(mask), p.x, p.y, flags),
        "photoSeamlessClone",
    )

actual fun photoColorChange(
    src: Mat,
    mask: Mat,
    redMul: Float,
    greenMul: Float,
    blueMul: Float,
): Mat = jvmMat(
    JniPhoto.colorChange(handleOf(src), handleOf(mask), redMul, greenMul, blueMul),
    "photoColorChange",
)

actual fun photoIlluminationChange(src: Mat, mask: Mat, alpha: Float, beta: Float): Mat =
    jvmMat(
        JniPhoto.illuminationChange(handleOf(src), handleOf(mask), alpha, beta),
        "photoIlluminationChange",
    )

actual fun photoTextureFlattening(
    src: Mat,
    mask: Mat,
    lowThreshold: Float,
    highThreshold: Float,
    kernelSize: Int,
): Mat = jvmMat(
    JniPhoto.textureFlattening(
        handleOf(src), handleOf(mask), lowThreshold, highThreshold, kernelSize,
    ),
    "photoTextureFlattening",
)

actual fun photoEdgePreservingFilter(
    src: Mat,
    flags: Int,
    sigmaS: Float,
    sigmaR: Float,
): Mat = jvmMat(
    JniPhoto.edgePreservingFilter(handleOf(src), flags, sigmaS, sigmaR),
    "photoEdgePreservingFilter",
)

actual fun photoDetailEnhance(src: Mat, sigmaS: Float, sigmaR: Float): Mat =
    jvmMat(JniPhoto.detailEnhance(handleOf(src), sigmaS, sigmaR), "photoDetailEnhance")

actual fun photoPencilSketch(
    src: Mat,
    sigmaS: Float,
    sigmaR: Float,
    shadeFactor: Float,
): Pair<Mat, Mat> =
    jvmPair(JniPhoto.pencilSketch(handleOf(src), sigmaS, sigmaR, shadeFactor), "photoPencilSketch")

actual fun photoStylization(src: Mat, sigmaS: Float, sigmaR: Float): Mat =
    jvmMat(JniPhoto.stylization(handleOf(src), sigmaS, sigmaR), "photoStylization")

actual fun photoCorrectChromaticAberration(
    input: Mat,
    coefficients: Mat,
    imageSize: Size,
    calibDegree: Int,
    bayerPattern: Int,
): Mat = jvmMat(
    JniPhoto.correctChromaticAberration(
        handleOf(input), handleOf(coefficients), imageSize.width, imageSize.height,
        calibDegree, bayerPattern,
    ),
    "photoCorrectChromaticAberration",
)

actual fun photoGammaCorrection(src: Mat, gamma: Double): Mat =
    jvmMat(JniPhoto.gammaCorrection(handleOf(src), gamma), "photoGammaCorrection")

// =========================================================================
// Tonemap family
// =========================================================================

internal class JvmTonemap(private var handle: Long) : Tonemap {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("Tonemap is closed")

    override fun process(src: Mat): Mat =
        jvmMat(JniPhoto.tonemapProcess(check(), handleOf(src)), "tonemap.process")

    override var gamma: Float
        get() = JniPhoto.tonemapGetGamma(check())
        set(value) = JniPhoto.tonemapSetGamma(check(), value)

    override fun clear() {
        JniPhoto.tonemapClear(check())
    }

    override fun empty(): Boolean = JniPhoto.tonemapEmpty(check())

    override fun save(filename: String) {
        JniPhoto.tonemapSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.tonemapGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.tonemapRelease(current)
    }
}

actual fun createTonemap(gamma: Float): Tonemap =
    JvmTonemap(JniPhoto.tonemapCreate(gamma))

internal class JvmTonemapDrago(private var handle: Long) : TonemapDrago {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("TonemapDrago is closed")

    override fun process(src: Mat): Mat =
        jvmMat(JniPhoto.tonemapDragoProcess(check(), handleOf(src)), "tonemapDrago.process")

    override var gamma: Float
        get() = JniPhoto.tonemapDragoGetGamma(check())
        set(value) = JniPhoto.tonemapDragoSetGamma(check(), value)

    override var saturation: Float
        get() = JniPhoto.tonemapDragoGetSaturation(check())
        set(value) = JniPhoto.tonemapDragoSetSaturation(check(), value)

    override var bias: Float
        get() = JniPhoto.tonemapDragoGetBias(check())
        set(value) = JniPhoto.tonemapDragoSetBias(check(), value)

    override fun clear() {
        JniPhoto.tonemapDragoClear(check())
    }

    override fun empty(): Boolean = JniPhoto.tonemapDragoEmpty(check())

    override fun save(filename: String) {
        JniPhoto.tonemapDragoSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.tonemapDragoGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.tonemapDragoRelease(current)
    }
}

actual fun createTonemapDrago(gamma: Float, saturation: Float, bias: Float): TonemapDrago =
    JvmTonemapDrago(JniPhoto.tonemapDragoCreate(gamma, saturation, bias))

internal class JvmTonemapMantiuk(private var handle: Long) : TonemapMantiuk {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("TonemapMantiuk is closed")

    override fun process(src: Mat): Mat =
        jvmMat(JniPhoto.tonemapMantiukProcess(check(), handleOf(src)), "tonemapMantiuk.process")

    override var gamma: Float
        get() = JniPhoto.tonemapMantiukGetGamma(check())
        set(value) = JniPhoto.tonemapMantiukSetGamma(check(), value)

    override var scale: Float
        get() = JniPhoto.tonemapMantiukGetScale(check())
        set(value) = JniPhoto.tonemapMantiukSetScale(check(), value)

    override var saturation: Float
        get() = JniPhoto.tonemapMantiukGetSaturation(check())
        set(value) = JniPhoto.tonemapMantiukSetSaturation(check(), value)

    override fun clear() {
        JniPhoto.tonemapMantiukClear(check())
    }

    override fun empty(): Boolean = JniPhoto.tonemapMantiukEmpty(check())

    override fun save(filename: String) {
        JniPhoto.tonemapMantiukSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.tonemapMantiukGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.tonemapMantiukRelease(current)
    }
}

actual fun createTonemapMantiuk(gamma: Float, scale: Float, saturation: Float): TonemapMantiuk =
    JvmTonemapMantiuk(JniPhoto.tonemapMantiukCreate(gamma, scale, saturation))

internal class JvmTonemapReinhard(private var handle: Long) : TonemapReinhard {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("TonemapReinhard is closed")

    override fun process(src: Mat): Mat =
        jvmMat(JniPhoto.tonemapReinhardProcess(check(), handleOf(src)), "tonemapReinhard.process")

    override var gamma: Float
        get() = JniPhoto.tonemapReinhardGetGamma(check())
        set(value) = JniPhoto.tonemapReinhardSetGamma(check(), value)

    override var intensity: Float
        get() = JniPhoto.tonemapReinhardGetIntensity(check())
        set(value) = JniPhoto.tonemapReinhardSetIntensity(check(), value)

    override var lightAdaptation: Float
        get() = JniPhoto.tonemapReinhardGetLightAdaptation(check())
        set(value) = JniPhoto.tonemapReinhardSetLightAdaptation(check(), value)

    override var colorAdaptation: Float
        get() = JniPhoto.tonemapReinhardGetColorAdaptation(check())
        set(value) = JniPhoto.tonemapReinhardSetColorAdaptation(check(), value)

    override fun clear() {
        JniPhoto.tonemapReinhardClear(check())
    }

    override fun empty(): Boolean = JniPhoto.tonemapReinhardEmpty(check())

    override fun save(filename: String) {
        JniPhoto.tonemapReinhardSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.tonemapReinhardGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.tonemapReinhardRelease(current)
    }
}

actual fun createTonemapReinhard(
    gamma: Float,
    intensity: Float,
    lightAdaptation: Float,
    colorAdaptation: Float,
): TonemapReinhard =
    JvmTonemapReinhard(
        JniPhoto.tonemapReinhardCreate(gamma, intensity, lightAdaptation, colorAdaptation),
    )

// =========================================================================
// AlignMTB
// =========================================================================

internal class JvmAlignMtb(private var handle: Long) : AlignMTB {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("AlignMTB is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): List<Mat> {
        require(src.isNotEmpty()) { "alignMTB.process needs at least one image" }
        return jvmMatList(
            JniPhoto.alignMtbProcessTimes(
                check(), src.map { handleOf(it) }.toLongArray(),
                handleOf(times), handleOf(response),
            ),
            "alignMTB.process",
        )
    }

    override fun process(src: List<Mat>): List<Mat> {
        require(src.isNotEmpty()) { "alignMTB.process needs at least one image" }
        return jvmMatList(
            JniPhoto.alignMtbProcess(check(), src.map { handleOf(it) }.toLongArray()),
            "alignMTB.process",
        )
    }

    override fun calculateShift(img0: Mat, img1: Mat): Point {
        val shift = JniPhoto.alignMtbCalculateShift(check(), handleOf(img0), handleOf(img1))
        return Point(shift[0], shift[1])
    }

    override fun shiftMat(src: Mat, shift: Point): Mat =
        jvmMat(
            JniPhoto.alignMtbShiftMat(check(), handleOf(src), shift.x, shift.y),
            "alignMTB.shiftMat",
        )

    override fun computeBitmaps(img: Mat): Pair<Mat, Mat> =
        jvmPair(JniPhoto.alignMtbComputeBitmaps(check(), handleOf(img)), "alignMTB.computeBitmaps")

    override var maxBits: Int
        get() = JniPhoto.alignMtbGetMaxBits(check())
        set(value) = JniPhoto.alignMtbSetMaxBits(check(), value)

    override var excludeRange: Int
        get() = JniPhoto.alignMtbGetExcludeRange(check())
        set(value) = JniPhoto.alignMtbSetExcludeRange(check(), value)

    override var cut: Boolean
        get() = JniPhoto.alignMtbGetCut(check())
        set(value) = JniPhoto.alignMtbSetCut(check(), value)

    override fun clear() {
        JniPhoto.alignMtbClear(check())
    }

    override fun empty(): Boolean = JniPhoto.alignMtbEmpty(check())

    override fun save(filename: String) {
        JniPhoto.alignMtbSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.alignMtbGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.alignMtbRelease(current)
    }
}

actual fun createAlignMTB(maxBits: Int, excludeRange: Int, cut: Boolean): AlignMTB =
    JvmAlignMtb(JniPhoto.alignMtbCreate(maxBits, excludeRange, cut))

// =========================================================================
// CalibrateDebevec
// =========================================================================

internal class JvmCalibrateDebevec(private var handle: Long) : CalibrateDebevec {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("CalibrateDebevec is closed")

    override fun process(src: List<Mat>, times: Mat): Mat {
        require(src.isNotEmpty()) { "calibrateDebevec.process needs at least one image" }
        return jvmMat(
            JniPhoto.calibrateDebevecProcess(
                check(), src.map { handleOf(it) }.toLongArray(), handleOf(times),
            ),
            "calibrateDebevec.process",
        )
    }

    override var lambda: Float
        get() = JniPhoto.calibrateDebevecGetLambda(check())
        set(value) = JniPhoto.calibrateDebevecSetLambda(check(), value)

    override var samples: Int
        get() = JniPhoto.calibrateDebevecGetSamples(check())
        set(value) = JniPhoto.calibrateDebevecSetSamples(check(), value)

    override var random: Boolean
        get() = JniPhoto.calibrateDebevecGetRandom(check())
        set(value) = JniPhoto.calibrateDebevecSetRandom(check(), value)

    override fun clear() {
        JniPhoto.calibrateDebevecClear(check())
    }

    override fun empty(): Boolean = JniPhoto.calibrateDebevecEmpty(check())

    override fun save(filename: String) {
        JniPhoto.calibrateDebevecSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.calibrateDebevecGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.calibrateDebevecRelease(current)
    }
}

actual fun createCalibrateDebevec(samples: Int, lambda: Float, random: Boolean): CalibrateDebevec =
    JvmCalibrateDebevec(JniPhoto.calibrateDebevecCreate(samples, lambda, random))

// =========================================================================
// CalibrateRobertson
// =========================================================================

internal class JvmCalibrateRobertson(private var handle: Long) : CalibrateRobertson {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("CalibrateRobertson is closed")

    override fun process(src: List<Mat>, times: Mat): Mat {
        require(src.isNotEmpty()) { "calibrateRobertson.process needs at least one image" }
        return jvmMat(
            JniPhoto.calibrateRobertsonProcess(
                check(), src.map { handleOf(it) }.toLongArray(), handleOf(times),
            ),
            "calibrateRobertson.process",
        )
    }

    override var maxIter: Int
        get() = JniPhoto.calibrateRobertsonGetMaxIter(check())
        set(value) = JniPhoto.calibrateRobertsonSetMaxIter(check(), value)

    override var threshold: Float
        get() = JniPhoto.calibrateRobertsonGetThreshold(check())
        set(value) = JniPhoto.calibrateRobertsonSetThreshold(check(), value)

    override val radiance: Mat
        get() = jvmMat(JniPhoto.calibrateRobertsonGetRadiance(check()), "calibrateRobertson.radiance")

    override fun clear() {
        JniPhoto.calibrateRobertsonClear(check())
    }

    override fun empty(): Boolean = JniPhoto.calibrateRobertsonEmpty(check())

    override fun save(filename: String) {
        JniPhoto.calibrateRobertsonSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.calibrateRobertsonGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.calibrateRobertsonRelease(current)
    }
}

actual fun createCalibrateRobertson(maxIter: Int, threshold: Float): CalibrateRobertson =
    JvmCalibrateRobertson(JniPhoto.calibrateRobertsonCreate(maxIter, threshold))

// =========================================================================
// MergeDebevec
// =========================================================================

internal class JvmMergeDebevec(private var handle: Long) : MergeDebevec {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("MergeDebevec is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): Mat {
        require(src.isNotEmpty()) { "mergeDebevec.process needs at least one image" }
        return jvmMat(
            JniPhoto.mergeDebevecProcessResponse(
                check(), src.map { handleOf(it) }.toLongArray(), handleOf(times), handleOf(response),
            ),
            "mergeDebevec.process",
        )
    }

    override fun process(src: List<Mat>, times: Mat): Mat {
        require(src.isNotEmpty()) { "mergeDebevec.process needs at least one image" }
        return jvmMat(
            JniPhoto.mergeDebevecProcess(
                check(), src.map { handleOf(it) }.toLongArray(), handleOf(times),
            ),
            "mergeDebevec.process",
        )
    }

    override fun clear() {
        JniPhoto.mergeDebevecClear(check())
    }

    override fun empty(): Boolean = JniPhoto.mergeDebevecEmpty(check())

    override fun save(filename: String) {
        JniPhoto.mergeDebevecSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.mergeDebevecGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.mergeDebevecRelease(current)
    }
}

actual fun createMergeDebevec(): MergeDebevec =
    JvmMergeDebevec(JniPhoto.mergeDebevecCreate())

// =========================================================================
// MergeMertens
// =========================================================================

internal class JvmMergeMertens(private var handle: Long) : MergeMertens {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("MergeMertens is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): Mat {
        require(src.isNotEmpty()) { "mergeMertens.process needs at least one image" }
        return jvmMat(
            JniPhoto.mergeMertensProcessResponse(
                check(), src.map { handleOf(it) }.toLongArray(), handleOf(times), handleOf(response),
            ),
            "mergeMertens.process",
        )
    }

    override fun process(src: List<Mat>): Mat {
        require(src.isNotEmpty()) { "mergeMertens.process needs at least one image" }
        return jvmMat(
            JniPhoto.mergeMertensProcess(check(), src.map { handleOf(it) }.toLongArray()),
            "mergeMertens.process",
        )
    }

    override var contrastWeight: Float
        get() = JniPhoto.mergeMertensGetContrastWeight(check())
        set(value) = JniPhoto.mergeMertensSetContrastWeight(check(), value)

    override var saturationWeight: Float
        get() = JniPhoto.mergeMertensGetSaturationWeight(check())
        set(value) = JniPhoto.mergeMertensSetSaturationWeight(check(), value)

    override var exposureWeight: Float
        get() = JniPhoto.mergeMertensGetExposureWeight(check())
        set(value) = JniPhoto.mergeMertensSetExposureWeight(check(), value)

    override fun clear() {
        JniPhoto.mergeMertensClear(check())
    }

    override fun empty(): Boolean = JniPhoto.mergeMertensEmpty(check())

    override fun save(filename: String) {
        JniPhoto.mergeMertensSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.mergeMertensGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.mergeMertensRelease(current)
    }
}

actual fun createMergeMertens(
    contrastWeight: Float,
    saturationWeight: Float,
    exposureWeight: Float,
): MergeMertens =
    JvmMergeMertens(JniPhoto.mergeMertensCreate(contrastWeight, saturationWeight, exposureWeight))

// =========================================================================
// MergeRobertson
// =========================================================================

internal class JvmMergeRobertson(private var handle: Long) : MergeRobertson {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("MergeRobertson is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): Mat {
        require(src.isNotEmpty()) { "mergeRobertson.process needs at least one image" }
        return jvmMat(
            JniPhoto.mergeRobertsonProcessResponse(
                check(), src.map { handleOf(it) }.toLongArray(), handleOf(times), handleOf(response),
            ),
            "mergeRobertson.process",
        )
    }

    override fun process(src: List<Mat>, times: Mat): Mat {
        require(src.isNotEmpty()) { "mergeRobertson.process needs at least one image" }
        return jvmMat(
            JniPhoto.mergeRobertsonProcess(
                check(), src.map { handleOf(it) }.toLongArray(), handleOf(times),
            ),
            "mergeRobertson.process",
        )
    }

    override fun clear() {
        JniPhoto.mergeRobertsonClear(check())
    }

    override fun empty(): Boolean = JniPhoto.mergeRobertsonEmpty(check())

    override fun save(filename: String) {
        JniPhoto.mergeRobertsonSave(check(), filename)
    }

    override fun getDefaultName(): String = JniPhoto.mergeRobertsonGetDefaultName(check())

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.mergeRobertsonRelease(current)
    }
}

actual fun createMergeRobertson(): MergeRobertson =
    JvmMergeRobertson(JniPhoto.mergeRobertsonCreate())

// =========================================================================
// ColorCorrectionModel
// =========================================================================

internal class JvmColorCorrectionModel(private var handle: Long) : ColorCorrectionModel {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("ColorCorrectionModel is closed")

    override fun setLinearizationGamma(gamma: Double) {
        JniPhoto.colorCorrectionModelSetLinearizationGamma(check(), gamma)
    }

    override fun setLinearizationDegree(deg: Int) {
        JniPhoto.colorCorrectionModelSetLinearizationDegree(check(), deg)
    }

    override fun setSaturatedThreshold(lower: Double, upper: Double) {
        JniPhoto.colorCorrectionModelSetSaturatedThreshold(check(), lower, upper)
    }

    override fun setWeightsList(weights: Mat) {
        JniPhoto.colorCorrectionModelSetWeightsList(check(), handleOf(weights))
    }

    override fun setWeightCoeff(weightsCoeff: Double) {
        JniPhoto.colorCorrectionModelSetWeightCoeff(check(), weightsCoeff)
    }

    override fun setMaxCount(maxCount: Int) {
        JniPhoto.colorCorrectionModelSetMaxCount(check(), maxCount)
    }

    override fun setEpsilon(epsilon: Double) {
        JniPhoto.colorCorrectionModelSetEpsilon(check(), epsilon)
    }

    override fun setRGB(rgb: Boolean) {
        JniPhoto.colorCorrectionModelSetRgb(check(), rgb)
    }

    override fun compute(): Mat =
        jvmMat(JniPhoto.colorCorrectionModelCompute(check()), "colorCorrectionModel.compute")

    override val colorCorrectionMatrix: Mat
        get() = jvmMat(
            JniPhoto.colorCorrectionModelGetColorCorrectionMatrix(check()),
            "colorCorrectionModel.colorCorrectionMatrix",
        )

    override val loss: Double
        get() = JniPhoto.colorCorrectionModelGetLoss(check())

    override val srcLinearRGB: Mat
        get() = jvmMat(
            JniPhoto.colorCorrectionModelGetSrcLinearRgb(check()),
            "colorCorrectionModel.srcLinearRGB",
        )

    override val refLinearRGB: Mat
        get() = jvmMat(
            JniPhoto.colorCorrectionModelGetRefLinearRgb(check()),
            "colorCorrectionModel.refLinearRGB",
        )

    override val mask: Mat
        get() = jvmMat(JniPhoto.colorCorrectionModelGetMask(check()), "colorCorrectionModel.mask")

    override val weights: Mat
        get() = jvmMat(JniPhoto.colorCorrectionModelGetWeights(check()), "colorCorrectionModel.weights")

    override fun correctImage(src: Mat, islinear: Boolean): Mat =
        jvmMat(
            JniPhoto.colorCorrectionModelCorrectImage(check(), handleOf(src), islinear),
            "colorCorrectionModel.correctImage",
        )

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.colorCorrectionModelRelease(current)
    }
}

actual fun colorCorrectionModel(): ColorCorrectionModel =
    JvmColorCorrectionModel(JniPhoto.colorCorrectionModelCreateEmpty())

actual fun colorCorrectionModel(src: Mat, constColor: Int): ColorCorrectionModel =
    JvmColorCorrectionModel(JniPhoto.colorCorrectionModelCreate(handleOf(src), constColor))

// =========================================================================
// IntelligentScissorsMB
// =========================================================================

internal class JvmIntelligentScissorsMB(private var handle: Long) : IntelligentScissorsMB {

    private fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("IntelligentScissorsMB is closed")

    override fun setWeights(weightNonEdge: Float, weightGradientDirection: Float, weightGradientMagnitude: Float) {
        JniPhoto.scissorsSetWeights(check(), weightNonEdge, weightGradientDirection, weightGradientMagnitude)
    }

    override fun setGradientMagnitudeMaxLimit(gradientMagnitudeThresholdMax: Float) {
        JniPhoto.scissorsSetGradientMagnitudeMaxLimit(check(), gradientMagnitudeThresholdMax)
    }

    override fun setEdgeFeatureZeroCrossingParameters(gradientMagnitudeMinValue: Float) {
        JniPhoto.scissorsSetEdgeFeatureZeroCrossingParameters(check(), gradientMagnitudeMinValue)
    }

    override fun setEdgeFeatureCannyParameters(
        threshold1: Double,
        threshold2: Double,
        apertureSize: Int,
        l2gradient: Boolean,
    ) {
        JniPhoto.scissorsSetEdgeFeatureCannyParameters(check(), threshold1, threshold2, apertureSize, l2gradient)
    }

    override fun applyImage(image: Mat) {
        JniPhoto.scissorsApplyImage(check(), handleOf(image))
    }

    override fun applyImageFeatures(
        nonEdge: Mat,
        gradientDirection: Mat,
        gradientMagnitude: Mat,
        image: Mat?,
    ) {
        JniPhoto.scissorsApplyImageFeatures(
            check(), handleOf(nonEdge), handleOf(gradientDirection), handleOf(gradientMagnitude),
            image?.let { handleOf(it) } ?: 0L,
        )
    }

    override fun buildMap(sourcePt: Point) {
        JniPhoto.scissorsBuildMap(check(), sourcePt.x, sourcePt.y)
    }

    override fun getContour(targetPt: Point, backward: Boolean): Mat =
        jvmMat(
            JniPhoto.scissorsGetContour(check(), targetPt.x, targetPt.y, backward),
            "intelligentScissorsMB.getContour",
        )

    override fun close() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        JniPhoto.scissorsRelease(current)
    }
}

actual fun intelligentScissorsMB(): IntelligentScissorsMB =
    JvmIntelligentScissorsMB(JniPhoto.scissorsCreate())

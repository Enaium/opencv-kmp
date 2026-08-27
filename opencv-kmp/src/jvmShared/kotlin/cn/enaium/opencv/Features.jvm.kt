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
// JVM (JNI-backed) implementation of the features module.
// Detector handles are jlong pointers to the cvk_ handle structs; every
// operation forwards through [JniFeatures] to the same cvk_ shim the
// native targets bind via cinterop.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()


/** Unpacks a fixed-order two-handle jlongArray output into a Mat pair. */
private fun matPair(handles: LongArray, operation: String): Pair<Mat, Mat> =
    if (handles.size == 2 && handles[0] != 0L && handles[1] != 0L) {
        JvmMat(handles[0]) to JvmMat(handles[1])
    } else {
        throw OpenCVException(operation, lastNativeError())
    }

private fun blobParamsArgs(p: SimpleBlobDetector.Params): BlobParamsScalars = BlobParamsScalars(
    thresholdStep = p.thresholdStep,
    minThreshold = p.minThreshold,
    maxThreshold = p.maxThreshold,
    minRepeatability = p.minRepeatability,
    minDistBetweenBlobs = p.minDistBetweenBlobs,
    filterByColor = if (p.filterByColor) 1 else 0,
    blobColor = p.blobColor,
    filterByArea = if (p.filterByArea) 1 else 0,
    minArea = p.minArea,
    maxArea = p.maxArea,
    filterByCircularity = if (p.filterByCircularity) 1 else 0,
    minCircularity = p.minCircularity,
    maxCircularity = p.maxCircularity,
    filterByInertia = if (p.filterByInertia) 1 else 0,
    minInertiaRatio = p.minInertiaRatio,
    maxInertiaRatio = p.maxInertiaRatio,
    filterByConvexity = if (p.filterByConvexity) 1 else 0,
    minConvexity = p.minConvexity,
    maxConvexity = p.maxConvexity,
    collectContours = if (p.collectContours) 1 else 0,
)

/** Expanded scalar layout of [SimpleBlobDetector.Params] for the JNI bridge. */
private data class BlobParamsScalars(
    val thresholdStep: Float,
    val minThreshold: Float,
    val maxThreshold: Float,
    val minRepeatability: Long,
    val minDistBetweenBlobs: Float,
    val filterByColor: Int,
    val blobColor: Int,
    val filterByArea: Int,
    val minArea: Float,
    val maxArea: Float,
    val filterByCircularity: Int,
    val minCircularity: Float,
    val maxCircularity: Float,
    val filterByInertia: Int,
    val minInertiaRatio: Float,
    val maxInertiaRatio: Float,
    val filterByConvexity: Int,
    val minConvexity: Float,
    val maxConvexity: Float,
    val collectContours: Int,
)

private fun blobParamsOf(values: DoubleArray): SimpleBlobDetector.Params =
    SimpleBlobDetector.Params(
        thresholdStep = values[0].toFloat(),
        minThreshold = values[1].toFloat(),
        maxThreshold = values[2].toFloat(),
        minRepeatability = values[3].toLong(),
        minDistBetweenBlobs = values[4].toFloat(),
        filterByColor = values[5] != 0.0,
        blobColor = values[6].toInt(),
        filterByArea = values[7] != 0.0,
        minArea = values[8].toFloat(),
        maxArea = values[9].toFloat(),
        filterByCircularity = values[10] != 0.0,
        minCircularity = values[11].toFloat(),
        maxCircularity = values[12].toFloat(),
        filterByInertia = values[13] != 0.0,
        minInertiaRatio = values[14].toFloat(),
        maxInertiaRatio = values[15].toFloat(),
        filterByConvexity = values[16] != 0.0,
        minConvexity = values[17].toFloat(),
        maxConvexity = values[18].toFloat(),
        collectContours = values[19] != 0.0,
    )

/**
 * JNI-backed [Feature2D] base. Concrete detectors supply their per-class
 * [JniFeatures] functions; the common surface (detect/compute/
 * detectAndCompute, descriptors, Algorithm quartet) is implemented here.
 */
internal abstract class JvmFeature2D protected constructor(
    @Volatile protected var handle: Long,
    private val detectFn: (Long, Long, Long) -> Long,
    private val computeFn: (Long, Long, Long) -> LongArray,
    private val detectAndComputeFn: (Long, Long, Long) -> LongArray,
    private val descriptorSizeFn: (Long) -> Int,
    private val descriptorTypeFn: (Long) -> Int,
    private val defaultNormFn: (Long) -> Int,
    private val writeFn: (Long, String) -> Unit,
    private val readFn: (Long, String) -> Unit,
    private val clearFn: (Long) -> Unit,
    private val emptyFn: (Long) -> Boolean,
    private val saveFn: (Long, String) -> Unit,
    private val getDefaultNameFn: (Long) -> String?,
    private val releaseFn: (Long) -> Unit,
) : Feature2D {

    internal fun check(): Long =
        if (handle != 0L) handle else throw IllegalStateException("Feature2D is closed")

    /** Raw handle, usable as the generic Feature2D backend of [affineFeatureCreate]. */
    internal fun rawHandle(): Long = check()

    override fun detect(image: Mat, mask: Mat?): List<KeyPoint> {
        val out = detectFn(check(), handleOf(image), mask?.let { handleOf(it) } ?: 0L)
        return keypointsOf(jvmMat(out, "detect"))
    }

    override fun detectAndCompute(image: Mat, mask: Mat?): Pair<List<KeyPoint>, Mat> {
        val handles = detectAndComputeFn(check(), handleOf(image), mask?.let { handleOf(it) } ?: 0L)
        if (handles.size != 2) throw OpenCVException("detectAndCompute", lastNativeError())
        return keypointsOf(jvmMat(handles[0], "detectAndCompute")) to
            jvmMat(handles[1], "detectAndCompute")
    }

    override fun compute(image: Mat, keypoints: List<KeyPoint>): Pair<List<KeyPoint>, Mat> {
        val keypointsMat = keypointsMat(keypoints)
        val handles = try {
            computeFn(check(), handleOf(image), handleOf(keypointsMat))
        } finally {
            keypointsMat.close()
        }
        if (handles.size != 2) throw OpenCVException("compute", lastNativeError())
        return keypointsOf(jvmMat(handles[0], "compute")) to jvmMat(handles[1], "compute")
    }

    override val descriptorSize: Int get() = descriptorSizeFn(check())
    override val descriptorType: Int get() = descriptorTypeFn(check())
    override val defaultNorm: Int get() = defaultNormFn(check())

    override fun write(fileName: String) {
        writeFn(check(), fileName)
    }

    override fun read(fileName: String) {
        readFn(check(), fileName)
    }

    override fun clear() {
        clearFn(check())
    }

    override fun empty(): Boolean = emptyFn(check())

    override fun save(filename: String) {
        saveFn(check(), filename)
    }

    override fun getDefaultName(): String =
        getDefaultNameFn(check()) ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val current = handle
        if (current != 0L) {
            handle = 0L
            releaseFn(current)
        }
    }
}

// =========================================================================
// SIFT
// =========================================================================

internal class JvmSift(handle: Long) :
    JvmFeature2D(
        handle,
        detectFn = { h, img, mask -> JniFeatures.siftDetect(h, img, mask) },
        computeFn = { h, img, kp -> JniFeatures.siftCompute(h, img, kp) },
        detectAndComputeFn = { h, img, mask -> JniFeatures.siftDetectAndCompute(h, img, mask) },
        descriptorSizeFn = { JniFeatures.siftDescriptorSize(it) },
        descriptorTypeFn = { JniFeatures.siftDescriptorType(it) },
        defaultNormFn = { JniFeatures.siftDefaultNorm(it) },
        writeFn = { h, name -> JniFeatures.siftWrite(h, name) },
        readFn = { h, name -> JniFeatures.siftRead(h, name) },
        clearFn = { JniFeatures.siftClear(it) },
        emptyFn = { JniFeatures.siftEmpty(it) },
        saveFn = { h, name -> JniFeatures.siftSave(h, name) },
        getDefaultNameFn = { JniFeatures.siftGetDefaultName(it) },
        releaseFn = { JniFeatures.siftRelease(it) },
    ),
    SIFT {

    override var nFeatures: Int
        get() = JniFeatures.siftGetNFeatures(check())
        set(value) {
            JniFeatures.siftSetNFeatures(check(), value)
        }

    override var nOctaveLayers: Int
        get() = JniFeatures.siftGetNOctaveLayers(check())
        set(value) {
            JniFeatures.siftSetNOctaveLayers(check(), value)
        }

    override var contrastThreshold: Double
        get() = JniFeatures.siftGetContrastThreshold(check())
        set(value) {
            JniFeatures.siftSetContrastThreshold(check(), value)
        }

    override var edgeThreshold: Double
        get() = JniFeatures.siftGetEdgeThreshold(check())
        set(value) {
            JniFeatures.siftSetEdgeThreshold(check(), value)
        }

    override var sigma: Double
        get() = JniFeatures.siftGetSigma(check())
        set(value) {
            JniFeatures.siftSetSigma(check(), value)
        }
}

actual fun siftCreate(
    nfeatures: Int,
    nOctaveLayers: Int,
    contrastThreshold: Double,
    edgeThreshold: Double,
    sigma: Double,
    descriptorType: Int,
    enablePreciseUpscale: Boolean,
): SIFT = JvmSift(
    JniFeatures.siftCreate(
        nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma,
        descriptorType, enablePreciseUpscale,
    ),
)

// =========================================================================
// ORB
// =========================================================================

internal class JvmOrb(handle: Long) :
    JvmFeature2D(
        handle,
        detectFn = { h, img, mask -> JniFeatures.orbDetect(h, img, mask) },
        computeFn = { h, img, kp -> JniFeatures.orbCompute(h, img, kp) },
        detectAndComputeFn = { h, img, mask -> JniFeatures.orbDetectAndCompute(h, img, mask) },
        descriptorSizeFn = { JniFeatures.orbDescriptorSize(it) },
        descriptorTypeFn = { JniFeatures.orbDescriptorType(it) },
        defaultNormFn = { JniFeatures.orbDefaultNorm(it) },
        writeFn = { h, name -> JniFeatures.orbWrite(h, name) },
        readFn = { h, name -> JniFeatures.orbRead(h, name) },
        clearFn = { JniFeatures.orbClear(it) },
        emptyFn = { JniFeatures.orbEmpty(it) },
        saveFn = { h, name -> JniFeatures.orbSave(h, name) },
        getDefaultNameFn = { JniFeatures.orbGetDefaultName(it) },
        releaseFn = { JniFeatures.orbRelease(it) },
    ),
    ORB {

    override var maxFeatures: Int
        get() = JniFeatures.orbGetMaxFeatures(check())
        set(value) {
            JniFeatures.orbSetMaxFeatures(check(), value)
        }

    override var scaleFactor: Double
        get() = JniFeatures.orbGetScaleFactor(check())
        set(value) {
            JniFeatures.orbSetScaleFactor(check(), value)
        }

    override var nLevels: Int
        get() = JniFeatures.orbGetNLevels(check())
        set(value) {
            JniFeatures.orbSetNLevels(check(), value)
        }

    override var edgeThreshold: Int
        get() = JniFeatures.orbGetEdgeThreshold(check())
        set(value) {
            JniFeatures.orbSetEdgeThreshold(check(), value)
        }

    override var firstLevel: Int
        get() = JniFeatures.orbGetFirstLevel(check())
        set(value) {
            JniFeatures.orbSetFirstLevel(check(), value)
        }

    override var wtaK: Int
        get() = JniFeatures.orbGetWtaK(check())
        set(value) {
            JniFeatures.orbSetWtaK(check(), value)
        }

    override var scoreType: Int
        get() = JniFeatures.orbGetScoreType(check())
        set(value) {
            JniFeatures.orbSetScoreType(check(), value)
        }

    override var patchSize: Int
        get() = JniFeatures.orbGetPatchSize(check())
        set(value) {
            JniFeatures.orbSetPatchSize(check(), value)
        }

    override var fastThreshold: Int
        get() = JniFeatures.orbGetFastThreshold(check())
        set(value) {
            JniFeatures.orbSetFastThreshold(check(), value)
        }
}

actual fun orbCreate(
    nfeatures: Int,
    scaleFactor: Float,
    nlevels: Int,
    edgeThreshold: Int,
    firstLevel: Int,
    wtaK: Int,
    scoreType: Int,
    patchSize: Int,
    fastThreshold: Int,
): ORB = JvmOrb(
    JniFeatures.orbCreate(
        nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wtaK,
        scoreType, patchSize, fastThreshold,
    ),
)

// =========================================================================
// MSER
// =========================================================================

internal class JvmMser(handle: Long) :
    JvmFeature2D(
        handle,
        detectFn = { h, img, mask -> JniFeatures.mserDetect(h, img, mask) },
        computeFn = { h, img, kp -> JniFeatures.mserCompute(h, img, kp) },
        detectAndComputeFn = { h, img, mask -> JniFeatures.mserDetectAndCompute(h, img, mask) },
        descriptorSizeFn = { JniFeatures.mserDescriptorSize(it) },
        descriptorTypeFn = { JniFeatures.mserDescriptorType(it) },
        defaultNormFn = { JniFeatures.mserDefaultNorm(it) },
        writeFn = { h, name -> JniFeatures.mserWrite(h, name) },
        readFn = { h, name -> JniFeatures.mserRead(h, name) },
        clearFn = { JniFeatures.mserClear(it) },
        emptyFn = { JniFeatures.mserEmpty(it) },
        saveFn = { h, name -> JniFeatures.mserSave(h, name) },
        getDefaultNameFn = { JniFeatures.mserGetDefaultName(it) },
        releaseFn = { JniFeatures.mserRelease(it) },
    ),
    MSER {

    override var delta: Int
        get() = JniFeatures.mserGetDelta(check())
        set(value) {
            JniFeatures.mserSetDelta(check(), value)
        }

    override var minArea: Int
        get() = JniFeatures.mserGetMinArea(check())
        set(value) {
            JniFeatures.mserSetMinArea(check(), value)
        }

    override var maxArea: Int
        get() = JniFeatures.mserGetMaxArea(check())
        set(value) {
            JniFeatures.mserSetMaxArea(check(), value)
        }

    override var maxVariation: Double
        get() = JniFeatures.mserGetMaxVariation(check())
        set(value) {
            JniFeatures.mserSetMaxVariation(check(), value)
        }

    override var minDiversity: Double
        get() = JniFeatures.mserGetMinDiversity(check())
        set(value) {
            JniFeatures.mserSetMinDiversity(check(), value)
        }

    override var maxEvolution: Int
        get() = JniFeatures.mserGetMaxEvolution(check())
        set(value) {
            JniFeatures.mserSetMaxEvolution(check(), value)
        }

    override var areaThreshold: Double
        get() = JniFeatures.mserGetAreaThreshold(check())
        set(value) {
            JniFeatures.mserSetAreaThreshold(check(), value)
        }

    override var minMargin: Double
        get() = JniFeatures.mserGetMinMargin(check())
        set(value) {
            JniFeatures.mserSetMinMargin(check(), value)
        }

    override var edgeBlurSize: Int
        get() = JniFeatures.mserGetEdgeBlurSize(check())
        set(value) {
            JniFeatures.mserSetEdgeBlurSize(check(), value)
        }

    override var pass2Only: Boolean
        get() = JniFeatures.mserGetPass2Only(check())
        set(value) {
            JniFeatures.mserSetPass2Only(check(), value)
        }

    override fun detectRegions(image: Mat): Pair<List<List<Point>>, Mat> {
        val out = LongArray(1)
        val bytes = JniFeatures.mserDetectRegions(check(), handleOf(image), out)
        return ContourCodec.decode(bytes) to jvmMat(out[0], "detectRegions")
    }
}

actual fun mserCreate(
    delta: Int,
    minArea: Int,
    maxArea: Int,
    maxVariation: Double,
    minDiversity: Double,
    maxEvolution: Int,
    areaThreshold: Double,
    minMargin: Double,
    edgeBlurSize: Int,
): MSER = JvmMser(
    JniFeatures.mserCreate(
        delta, minArea, maxArea, maxVariation, minDiversity, maxEvolution,
        areaThreshold, minMargin, edgeBlurSize,
    ),
)

// =========================================================================
// FastFeatureDetector
// =========================================================================

internal class JvmFastFeatureDetector(handle: Long) :
    JvmFeature2D(
        handle,
        detectFn = { h, img, mask -> JniFeatures.fastDetect(h, img, mask) },
        computeFn = { h, img, kp -> JniFeatures.fastCompute(h, img, kp) },
        detectAndComputeFn = { h, img, mask -> JniFeatures.fastDetectAndCompute(h, img, mask) },
        descriptorSizeFn = { JniFeatures.fastDescriptorSize(it) },
        descriptorTypeFn = { JniFeatures.fastDescriptorType(it) },
        defaultNormFn = { JniFeatures.fastDefaultNorm(it) },
        writeFn = { h, name -> JniFeatures.fastWrite(h, name) },
        readFn = { h, name -> JniFeatures.fastRead(h, name) },
        clearFn = { JniFeatures.fastClear(it) },
        emptyFn = { JniFeatures.fastEmpty(it) },
        saveFn = { h, name -> JniFeatures.fastSave(h, name) },
        getDefaultNameFn = { JniFeatures.fastGetDefaultName(it) },
        releaseFn = { JniFeatures.fastRelease(it) },
    ),
    FastFeatureDetector {

    override var threshold: Int
        get() = JniFeatures.fastGetThreshold(check())
        set(value) {
            JniFeatures.fastSetThreshold(check(), value)
        }

    override var nonmaxSuppression: Boolean
        get() = JniFeatures.fastGetNonmaxSuppression(check())
        set(value) {
            JniFeatures.fastSetNonmaxSuppression(check(), value)
        }

    override var type: Int
        get() = JniFeatures.fastGetType(check())
        set(value) {
            JniFeatures.fastSetType(check(), value)
        }
}

actual fun fastCreate(threshold: Int, nonmaxSuppression: Boolean, type: Int): FastFeatureDetector =
    JvmFastFeatureDetector(JniFeatures.fastCreate(threshold, nonmaxSuppression, type))

// =========================================================================
// GFTTDetector
// =========================================================================

internal class JvmGfttDetector(handle: Long) :
    JvmFeature2D(
        handle,
        detectFn = { h, img, mask -> JniFeatures.gfttDetect(h, img, mask) },
        computeFn = { h, img, kp -> JniFeatures.gfttCompute(h, img, kp) },
        detectAndComputeFn = { h, img, mask -> JniFeatures.gfttDetectAndCompute(h, img, mask) },
        descriptorSizeFn = { JniFeatures.gfttDescriptorSize(it) },
        descriptorTypeFn = { JniFeatures.gfttDescriptorType(it) },
        defaultNormFn = { JniFeatures.gfttDefaultNorm(it) },
        writeFn = { h, name -> JniFeatures.gfttWrite(h, name) },
        readFn = { h, name -> JniFeatures.gfttRead(h, name) },
        clearFn = { JniFeatures.gfttClear(it) },
        emptyFn = { JniFeatures.gfttEmpty(it) },
        saveFn = { h, name -> JniFeatures.gfttSave(h, name) },
        getDefaultNameFn = { JniFeatures.gfttGetDefaultName(it) },
        releaseFn = { JniFeatures.gfttRelease(it) },
    ),
    GFTTDetector {

    override var maxFeatures: Int
        get() = JniFeatures.gfttGetMaxFeatures(check())
        set(value) {
            JniFeatures.gfttSetMaxFeatures(check(), value)
        }

    override var qualityLevel: Double
        get() = JniFeatures.gfttGetQualityLevel(check())
        set(value) {
            JniFeatures.gfttSetQualityLevel(check(), value)
        }

    override var minDistance: Double
        get() = JniFeatures.gfttGetMinDistance(check())
        set(value) {
            JniFeatures.gfttSetMinDistance(check(), value)
        }

    override var blockSize: Int
        get() = JniFeatures.gfttGetBlockSize(check())
        set(value) {
            JniFeatures.gfttSetBlockSize(check(), value)
        }

    override var gradientSize: Int
        get() = JniFeatures.gfttGetGradientSize(check())
        set(value) {
            JniFeatures.gfttSetGradientSize(check(), value)
        }

    override var harrisDetector: Boolean
        get() = JniFeatures.gfttGetHarrisDetector(check())
        set(value) {
            JniFeatures.gfttSetHarrisDetector(check(), value)
        }

    override var k: Double
        get() = JniFeatures.gfttGetK(check())
        set(value) {
            JniFeatures.gfttSetK(check(), value)
        }
}

actual fun gfttCreate(
    maxCorners: Int,
    qualityLevel: Double,
    minDistance: Double,
    blockSize: Int,
    gradientSize: Int,
    useHarrisDetector: Boolean,
    k: Double,
): GFTTDetector = JvmGfttDetector(
    JniFeatures.gfttCreate(
        maxCorners, qualityLevel, minDistance, blockSize, gradientSize,
        useHarrisDetector, k,
    ),
)

// =========================================================================
// SimpleBlobDetector
// =========================================================================

internal class JvmSimpleBlobDetector(handle: Long) :
    JvmFeature2D(
        handle,
        detectFn = { h, img, mask -> JniFeatures.simpleBlobDetectorDetect(h, img, mask) },
        computeFn = { h, img, kp -> JniFeatures.simpleBlobDetectorCompute(h, img, kp) },
        detectAndComputeFn = { h, img, mask -> JniFeatures.simpleBlobDetectorDetectAndCompute(h, img, mask) },
        descriptorSizeFn = { JniFeatures.simpleBlobDetectorDescriptorSize(it) },
        descriptorTypeFn = { JniFeatures.simpleBlobDetectorDescriptorType(it) },
        defaultNormFn = { JniFeatures.simpleBlobDetectorDefaultNorm(it) },
        writeFn = { h, name -> JniFeatures.simpleBlobDetectorWrite(h, name) },
        readFn = { h, name -> JniFeatures.simpleBlobDetectorRead(h, name) },
        clearFn = { JniFeatures.simpleBlobDetectorClear(it) },
        emptyFn = { JniFeatures.simpleBlobDetectorEmpty(it) },
        saveFn = { h, name -> JniFeatures.simpleBlobDetectorSave(h, name) },
        getDefaultNameFn = { JniFeatures.simpleBlobDetectorGetDefaultName(it) },
        releaseFn = { JniFeatures.simpleBlobDetectorRelease(it) },
    ),
    SimpleBlobDetector {

    override fun setParams(params: SimpleBlobDetector.Params) {
        val a = blobParamsArgs(params)
        JniFeatures.simpleBlobDetectorSetParams(
            check(), a.thresholdStep, a.minThreshold, a.maxThreshold, a.minRepeatability,
            a.minDistBetweenBlobs, a.filterByColor, a.blobColor, a.filterByArea,
            a.minArea, a.maxArea, a.filterByCircularity, a.minCircularity,
            a.maxCircularity, a.filterByInertia, a.minInertiaRatio, a.maxInertiaRatio,
            a.filterByConvexity, a.minConvexity, a.maxConvexity, a.collectContours,
        )
    }

    override fun getParams(): SimpleBlobDetector.Params =
        blobParamsOf(JniFeatures.simpleBlobDetectorGetParams(check()))

    override fun getBlobContours(): List<List<Point>> =
        ContourCodec.decode(JniFeatures.simpleBlobDetectorGetBlobContours(check()))
}

actual fun simpleBlobDetectorCreate(params: SimpleBlobDetector.Params): SimpleBlobDetector {
    val a = blobParamsArgs(params)
    return JvmSimpleBlobDetector(
        JniFeatures.simpleBlobDetectorCreate(
            a.thresholdStep, a.minThreshold, a.maxThreshold, a.minRepeatability,
            a.minDistBetweenBlobs, a.filterByColor, a.blobColor, a.filterByArea,
            a.minArea, a.maxArea, a.filterByCircularity, a.minCircularity,
            a.maxCircularity, a.filterByInertia, a.minInertiaRatio, a.maxInertiaRatio,
            a.filterByConvexity, a.minConvexity, a.maxConvexity, a.collectContours,
        ),
    )
}

// =========================================================================
// AffineFeature
// =========================================================================

internal class JvmAffine(handle: Long) :
    JvmFeature2D(
        handle,
        detectFn = { h, img, mask -> JniFeatures.affineDetect(h, img, mask) },
        computeFn = { h, img, kp -> JniFeatures.affineCompute(h, img, kp) },
        detectAndComputeFn = { h, img, mask -> JniFeatures.affineDetectAndCompute(h, img, mask) },
        descriptorSizeFn = { JniFeatures.affineDescriptorSize(it) },
        descriptorTypeFn = { JniFeatures.affineDescriptorType(it) },
        defaultNormFn = { JniFeatures.affineDefaultNorm(it) },
        writeFn = { h, name -> JniFeatures.affineWrite(h, name) },
        readFn = { h, name -> JniFeatures.affineRead(h, name) },
        clearFn = { JniFeatures.affineClear(it) },
        emptyFn = { JniFeatures.affineEmpty(it) },
        saveFn = { h, name -> JniFeatures.affineSave(h, name) },
        getDefaultNameFn = { JniFeatures.affineGetDefaultName(it) },
        releaseFn = { JniFeatures.affineRelease(it) },
    ),
    AffineFeature {

    override fun setViewParams(tilts: FloatArray, rolls: FloatArray) {
        val tiltsMat = mat(tilts.size, 1, cvMakeType(CV_32F, 1))
        val rollsMat = mat(rolls.size, 1, cvMakeType(CV_32F, 1))
        try {
            MatOfFloat(tiltsMat).fromArray(tilts)
            MatOfFloat(rollsMat).fromArray(rolls)
            JniFeatures.affineSetViewParams(check(), handleOf(tiltsMat), handleOf(rollsMat))
        } finally {
            tiltsMat.close()
            rollsMat.close()
        }
    }

    override fun getViewParams(): Pair<FloatArray, FloatArray> {
        val handles = JniFeatures.affineGetViewParams(check())
        if (handles.size != 2) throw OpenCVException("getViewParams", lastNativeError())
        return floatsOf(jvmMat(handles[0], "getViewParams")) to
            floatsOf(jvmMat(handles[1], "getViewParams"))
    }

    private fun floatsOf(mat: Mat): FloatArray =
        try {
            MatOfFloat(mat).toArray()
        } finally {
            mat.close()
        }
}

actual fun affineFeatureCreate(
    backend: Feature2D,
    maxTilt: Int,
    minTilt: Int,
    tiltStep: Float,
    rotateStepBase: Float,
): AffineFeature {
    val raw = (backend as? JvmFeature2D)?.rawHandle()
        ?: throw IllegalArgumentException("backend belongs to another platform backend")
    return JvmAffine(JniFeatures.affineCreate(raw, maxTilt, minTilt, tiltStep, rotateStepBase))
}

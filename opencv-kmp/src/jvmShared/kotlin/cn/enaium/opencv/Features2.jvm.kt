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
// JVM actuals for the features2 slice (JNI-backed).
// Mat handles are jlong pointers into the native heap; every operation goes
// through [JniFeatures2], which forwards to the cvk_ shim.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()

/** Requires a freshly-created non-zero handle or throws with the native error. */
private fun requireHandle(handle: Long, operation: String): Long =
    if (handle != 0L) handle else throw OpenCVException(operation, lastNativeError())

internal actual fun encodeMatList(mats: List<Mat>): Mat =
    packMatHandleWire(mats.map { handleOf(it) }.toLongArray())

internal actual fun decodeMatList(wire: Mat, operation: String): List<Mat> {
    if (wire.isEmpty) return emptyList()
    val bytes = wire.pixels
    return List(wire.rows) { index -> JvmMat(readMatHandleAddress(bytes, index * 8)) }
}

// =========================================================================
// DescriptorMatcher family
// =========================================================================

internal open class JvmDescriptorMatcher(private var handle: Long) : DescriptorMatcher {

    protected fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("DescriptorMatcher is closed")

    override fun add(descriptors: List<Mat>) {
        val wire = encodeMatList(descriptors)
        try {
            JniFeatures2.descriptorMatcherAdd(check(), handleOf(wire))
        } finally {
            wire.close()
        }
    }

    override fun getTrainDescriptors(): List<Mat> {
        val wire = jvmMat(JniFeatures2.descriptorMatcherGetTrainDescriptors(check()), "getTrainDescriptors")
        return try {
            decodeMatList(wire, "getTrainDescriptors")
        } finally {
            wire.close()
        }
    }

    override fun isMaskSupported(): Boolean = JniFeatures2.descriptorMatcherIsMaskSupported(check())

    override fun train() {
        JniFeatures2.descriptorMatcherTrain(check())
    }

    override fun write(fileName: String) {
        JniFeatures2.descriptorMatcherWrite(check(), fileName)
    }

    override fun read(fileName: String) {
        JniFeatures2.descriptorMatcherRead(check(), fileName)
    }

    override fun clear() {
        JniFeatures2.descriptorMatcherClear(check())
    }

    override fun empty(): Boolean = JniFeatures2.descriptorMatcherEmpty(check())

    override fun save(filename: String) {
        JniFeatures2.descriptorMatcherSave(check(), filename)
    }

    override fun getDefaultName(): String = JniFeatures2.descriptorMatcherGetDefaultName(check())

    override fun clone(emptyTrainData: Boolean): DescriptorMatcher =
        JvmDescriptorMatcher(JniFeatures2.descriptorMatcherClone(check(), emptyTrainData))

    override fun match(query: Mat, trainDescriptors: Mat, mask: Mat?): List<DMatch> {
        val m = jvmMat(
            JniFeatures2.descriptorMatcherMatchTrain(
                check(), handleOf(query), handleOf(trainDescriptors),
                mask?.let(::handleOf) ?: 0L,
            ),
            "match",
        )
        return try {
            matToVectorDMatch(m)
        } finally {
            m.close()
        }
    }

    override fun match(query: Mat, masks: List<Mat>): List<DMatch> {
        val wire = encodeMatList(masks)
        try {
            val m = jvmMat(
                JniFeatures2.descriptorMatcherMatch(check(), handleOf(query), handleOf(wire)),
                "match",
            )
            return try {
                matToVectorDMatch(m)
            } finally {
                m.close()
            }
        } finally {
            wire.close()
        }
    }

    override fun knnMatch(
        query: Mat,
        trainDescriptors: Mat,
        k: Int,
        mask: Mat?,
        compactResult: Boolean,
    ): List<List<DMatch>> {
        val wire = jvmMat(
            JniFeatures2.descriptorMatcherKnnMatchTrain(
                check(), handleOf(query), handleOf(trainDescriptors), k,
                mask?.let(::handleOf) ?: 0L, compactResult,
            ),
            "knnMatch",
        )
        return try {
            decodeDMatchGroups(wire, "knnMatch")
        } finally {
            wire.close()
        }
    }

    override fun knnMatch(
        query: Mat,
        k: Int,
        masks: List<Mat>,
        compactResult: Boolean,
    ): List<List<DMatch>> {
        val masksWire = encodeMatList(masks)
        try {
            val wire = jvmMat(
                JniFeatures2.descriptorMatcherKnnMatch(
                    check(), handleOf(query), k, handleOf(masksWire), compactResult,
                ),
                "knnMatch",
            )
            return try {
                decodeDMatchGroups(wire, "knnMatch")
            } finally {
                wire.close()
            }
        } finally {
            masksWire.close()
        }
    }

    override fun radiusMatch(
        query: Mat,
        trainDescriptors: Mat,
        maxDistance: Float,
        mask: Mat?,
        compactResult: Boolean,
    ): List<List<DMatch>> {
        val wire = jvmMat(
            JniFeatures2.descriptorMatcherRadiusMatchTrain(
                check(), handleOf(query), handleOf(trainDescriptors), maxDistance,
                mask?.let(::handleOf) ?: 0L, compactResult,
            ),
            "radiusMatch",
        )
        return try {
            decodeDMatchGroups(wire, "radiusMatch")
        } finally {
            wire.close()
        }
    }

    override fun radiusMatch(
        query: Mat,
        maxDistance: Float,
        masks: List<Mat>,
        compactResult: Boolean,
    ): List<List<DMatch>> {
        val masksWire = encodeMatList(masks)
        try {
            val wire = jvmMat(
                JniFeatures2.descriptorMatcherRadiusMatch(
                    check(), handleOf(query), maxDistance, handleOf(masksWire), compactResult,
                ),
                "radiusMatch",
            )
            return try {
                decodeDMatchGroups(wire, "radiusMatch")
            } finally {
                wire.close()
            }
        } finally {
            masksWire.close()
        }
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniFeatures2.descriptorMatcherRelease(h)
        }
    }
}

internal class JvmBFMatcher(handle: Long) : JvmDescriptorMatcher(handle), BFMatcher

internal class JvmFlannBasedMatcher(handle: Long) : JvmDescriptorMatcher(handle), FlannBasedMatcher

internal class JvmLightGlueMatcher(handle: Long) : JvmDescriptorMatcher(handle), LightGlueMatcher {

    override fun setPairInfo(
        queryKpts: Mat,
        trainKpts: Mat,
        queryImageSize: Size,
        trainImageSize: Size,
    ) {
        JniFeatures2.lightGlueMatcherSetPairInfo(
            check(), handleOf(queryKpts), handleOf(trainKpts),
            queryImageSize.width.toDouble(), queryImageSize.height.toDouble(),
            trainImageSize.width.toDouble(), trainImageSize.height.toDouble(),
        )
    }

    override fun clearPairInfo() {
        JniFeatures2.lightGlueMatcherClearPairInfo(check())
    }
}

// =========================================================================
// ANNIndex
// =========================================================================

internal class JvmAnnIndex(private var handle: Long) : ANNIndex {

    private fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("ANNIndex is closed")

    override fun addItems(features: Mat) {
        JniFeatures2.annIndexAddItems(check(), handleOf(features))
    }

    override fun build(trees: Int) {
        JniFeatures2.annIndexBuild(check(), trees)
    }

    override fun knnSearch(query: Mat, knn: Int, searchK: Int): Pair<Mat, Mat> {
        val handles = JniFeatures2.annIndexKnnSearch(check(), handleOf(query), knn, searchK)
        if (handles.size != 2) throw OpenCVException("knnSearch", lastNativeError())
        return JvmMat(handles[0]) to JvmMat(handles[1])
    }

    override fun save(filename: String, prefault: Boolean) {
        JniFeatures2.annIndexSave(check(), filename, prefault)
    }

    override fun load(filename: String, prefault: Boolean) {
        JniFeatures2.annIndexLoad(check(), filename, prefault)
    }

    override val treeNumber: Int get() = JniFeatures2.annIndexTreeNumber(check())

    override val itemNumber: Int get() = JniFeatures2.annIndexItemNumber(check())

    override fun setOnDiskBuild(filename: String): Boolean =
        JniFeatures2.annIndexSetOnDiskBuild(check(), filename)

    override fun setSeed(seed: Int) {
        JniFeatures2.annIndexSetSeed(check(), seed)
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniFeatures2.annIndexRelease(h)
        }
    }
}

// =========================================================================
// ALIKED / DISK (Feature2D)
// =========================================================================

private fun decodeKeypointsAndDescriptors(handles: LongArray, operation: String): Pair<List<KeyPoint>, Mat> {
    if (handles.size != 2) throw OpenCVException(operation, lastNativeError())
    val keypoints = JvmMat(handles[0])
    val descriptors = JvmMat(handles[1])
    return try {
        matToVectorKeyPoint(keypoints) to descriptors
    } finally {
        keypoints.close()
    }
}

internal class JvmAliked(private var handle: Long) : ALIKED {

    private fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("ALIKED is closed")

    override fun detect(image: Mat, mask: Mat?): List<KeyPoint> {
        val m = jvmMat(
            JniFeatures2.alikedDetect(check(), handleOf(image), mask?.let(::handleOf) ?: 0L),
            "detect",
        )
        return try {
            matToVectorKeyPoint(m)
        } finally {
            m.close()
        }
    }

    override fun compute(image: Mat, keypoints: List<KeyPoint>): Pair<List<KeyPoint>, Mat> {
        val kps = vectorKeyPointToMat(keypoints)
        try {
            val handles = JniFeatures2.alikedCompute(check(), handleOf(image), handleOf(kps))
            return decodeKeypointsAndDescriptors(handles, "compute")
        } finally {
            kps.close()
        }
    }

    override fun detectAndCompute(image: Mat, mask: Mat?): Pair<List<KeyPoint>, Mat> {
        val handles = JniFeatures2.alikedDetectAndCompute(
            check(), handleOf(image), mask?.let(::handleOf) ?: 0L, false,
        )
        return decodeKeypointsAndDescriptors(handles, "detectAndCompute")
    }

    override val descriptorSize: Int get() = JniFeatures2.alikedDescriptorSize(check())
    override val descriptorType: Int get() = JniFeatures2.alikedDescriptorType(check())
    override val defaultNorm: Int get() = JniFeatures2.alikedDefaultNorm(check())

    override fun write(fileName: String) {
        JniFeatures2.alikedWrite(check(), fileName)
    }

    override fun read(fileName: String) {
        JniFeatures2.alikedRead(check(), fileName)
    }

    override fun clear() {
        JniFeatures2.alikedClear(check())
    }

    override fun empty(): Boolean = JniFeatures2.alikedEmpty(check())

    override fun save(filename: String) {
        JniFeatures2.alikedSave(check(), filename)
    }

    override fun getDefaultName(): String = JniFeatures2.alikedGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniFeatures2.alikedRelease(h)
        }
    }
}

internal class JvmDisk(private var handle: Long) : DISK {

    private fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("DISK is closed")

    override fun detect(image: Mat, mask: Mat?): List<KeyPoint> {
        val m = jvmMat(
            JniFeatures2.diskDetect(check(), handleOf(image), mask?.let(::handleOf) ?: 0L),
            "detect",
        )
        return try {
            matToVectorKeyPoint(m)
        } finally {
            m.close()
        }
    }

    override fun compute(image: Mat, keypoints: List<KeyPoint>): Pair<List<KeyPoint>, Mat> {
        val kps = vectorKeyPointToMat(keypoints)
        try {
            val handles = JniFeatures2.diskCompute(check(), handleOf(image), handleOf(kps))
            return decodeKeypointsAndDescriptors(handles, "compute")
        } finally {
            kps.close()
        }
    }

    override fun detectAndCompute(image: Mat, mask: Mat?): Pair<List<KeyPoint>, Mat> {
        val handles = JniFeatures2.diskDetectAndCompute(
            check(), handleOf(image), mask?.let(::handleOf) ?: 0L, false,
        )
        return decodeKeypointsAndDescriptors(handles, "detectAndCompute")
    }

    override val descriptorSize: Int get() = JniFeatures2.diskDescriptorSize(check())
    override val descriptorType: Int get() = JniFeatures2.diskDescriptorType(check())
    override val defaultNorm: Int get() = JniFeatures2.diskDefaultNorm(check())

    override var maxKeypoints: Int
        get() = JniFeatures2.diskGetMaxKeypoints(check())
        set(value) {
            JniFeatures2.diskSetMaxKeypoints(check(), value)
        }

    override var scoreThreshold: Float
        get() = JniFeatures2.diskGetScoreThreshold(check())
        set(value) {
            JniFeatures2.diskSetScoreThreshold(check(), value)
        }

    override var imageSize: Size
        get() {
            val v = JniFeatures2.diskImageSize(check())
            return Size(v[0].toInt(), v[1].toInt())
        }
        set(value) {
            JniFeatures2.diskSetImageSize(check(), value.width.toDouble(), value.height.toDouble())
        }

    override fun write(fileName: String) {
        JniFeatures2.diskWrite(check(), fileName)
    }

    override fun read(fileName: String) {
        JniFeatures2.diskRead(check(), fileName)
    }

    override fun clear() {
        JniFeatures2.diskClear(check())
    }

    override fun empty(): Boolean = JniFeatures2.diskEmpty(check())

    override fun save(filename: String) {
        JniFeatures2.diskSave(check(), filename)
    }

    override fun getDefaultName(): String = JniFeatures2.diskGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniFeatures2.diskRelease(h)
        }
    }
}

// =========================================================================
// Factories
// =========================================================================

actual fun descriptorMatcherCreate(type: String): DescriptorMatcher =
    JvmDescriptorMatcher(requireHandle(JniFeatures2.descriptorMatcherCreate(type), "descriptorMatcherCreate"))

actual fun descriptorMatcherCreate(type: Int): DescriptorMatcher =
    JvmDescriptorMatcher(requireHandle(JniFeatures2.descriptorMatcherCreateType(type), "descriptorMatcherCreate"))

actual fun bfMatcherCreate(normType: Int, crossCheck: Boolean): BFMatcher =
    JvmBFMatcher(requireHandle(JniFeatures2.bfMatcherCreate(normType, crossCheck), "bfMatcherCreate"))

actual fun flannBasedMatcherCreate(indexParams: String): FlannBasedMatcher =
    JvmFlannBasedMatcher(requireHandle(JniFeatures2.flannMatcherCreate(indexParams), "flannBasedMatcherCreate"))

actual fun lightGlueMatcherCreate(modelPath: String, scoreThreshold: Float, backend: Int, target: Int): LightGlueMatcher =
    JvmLightGlueMatcher(requireHandle(JniFeatures2.lightGlueMatcherCreate(modelPath, scoreThreshold, backend, target), "lightGlueMatcherCreate"))

actual fun lightGlueMatcherCreateFromMemory(modelData: ByteArray, scoreThreshold: Float, backend: Int, target: Int): LightGlueMatcher =
    JvmLightGlueMatcher(requireHandle(JniFeatures2.lightGlueMatcherCreateFromMemory(modelData, scoreThreshold, backend, target), "lightGlueMatcherCreateFromMemory"))

actual fun annIndexCreate(dim: Int, distType: Int): ANNIndex =
    JvmAnnIndex(requireHandle(JniFeatures2.annIndexCreate(dim, distType), "annIndexCreate"))

actual fun alikedCreate(modelPath: String, params: AlikedParams): ALIKED =
    JvmAliked(requireHandle(
        JniFeatures2.alikedCreate(
            modelPath, params.inputSize.width, params.inputSize.height,
            params.normalizeDescriptors, params.engine, params.backend, params.target,
        ),
        "alikedCreate",
    ))

actual fun diskCreate(
    modelPath: String,
    maxKeypoints: Int,
    scoreThreshold: Float,
    imageSize: Size,
    backendId: Int,
    targetId: Int,
): DISK = JvmDisk(requireHandle(
    JniFeatures2.diskCreate(
        modelPath, maxKeypoints, scoreThreshold,
        imageSize.width.toDouble(), imageSize.height.toDouble(), backendId, targetId,
    ),
    "diskCreate",
))

actual fun diskCreateFromMemory(
    bufferModel: ByteArray,
    maxKeypoints: Int,
    scoreThreshold: Float,
    imageSize: Size,
    backendId: Int,
    targetId: Int,
): DISK = JvmDisk(requireHandle(
    JniFeatures2.diskCreateFromMemory(
        bufferModel, maxKeypoints, scoreThreshold,
        imageSize.width.toDouble(), imageSize.height.toDouble(), backendId, targetId,
    ),
    "diskCreateFromMemory",
))

// =========================================================================
// Features statics
// =========================================================================

actual fun goodFeaturesToTrack(
    image: Mat,
    maxCorners: Int,
    qualityLevel: Double,
    minDistance: Double,
    mask: Mat?,
    blockSize: Int,
    useHarrisDetector: Boolean,
    k: Double,
): List<Point> {
    val m = jvmMat(
        JniFeatures2.featuresGoodFeaturesToTrack(
            handleOf(image), maxCorners, qualityLevel, minDistance,
            mask?.let(::handleOf) ?: 0L, blockSize, useHarrisDetector, k,
        ),
        "goodFeaturesToTrack",
    )
    return try {
        matToVectorPoint(m)
    } finally {
        m.close()
    }
}

actual fun goodFeaturesToTrack(
    image: Mat,
    maxCorners: Int,
    qualityLevel: Double,
    minDistance: Double,
    mask: Mat,
    blockSize: Int,
    gradientSize: Int,
    useHarrisDetector: Boolean,
    k: Double,
): List<Point> {
    val m = jvmMat(
        JniFeatures2.featuresGoodFeaturesToTrackGradient(
            handleOf(image), maxCorners, qualityLevel, minDistance,
            handleOf(mask), blockSize, gradientSize, useHarrisDetector, k,
        ),
        "goodFeaturesToTrack",
    )
    return try {
        matToVectorPoint(m)
    } finally {
        m.close()
    }
}

actual fun goodFeaturesToTrackWithQuality(
    image: Mat,
    maxCorners: Int,
    qualityLevel: Double,
    minDistance: Double,
    mask: Mat?,
    blockSize: Int,
    gradientSize: Int,
    useHarrisDetector: Boolean,
    k: Double,
): GoodFeaturesResult {
    val handles = JniFeatures2.featuresGoodFeaturesToTrackQuality(
        handleOf(image), maxCorners, qualityLevel, minDistance,
        mask?.let(::handleOf) ?: 0L, blockSize, gradientSize, useHarrisDetector, k,
    )
    if (handles.size != 2) throw OpenCVException("goodFeaturesToTrackWithQuality", lastNativeError())
    val corners = JvmMat(handles[0])
    val quality = JvmMat(handles[1])
    return try {
        GoodFeaturesResult(matToVectorPoint(corners), matToVectorFloat(quality))
    } finally {
        corners.close()
        quality.close()
    }
}

actual fun drawKeypoints(image: Mat, keypoints: List<KeyPoint>, color: Scalar, flags: Int): Mat {
    val kps = vectorKeyPointToMat(keypoints)
    try {
        return jvmMat(
            JniFeatures2.drawKeypoints(
                handleOf(image), handleOf(kps), color.v0, color.v1, color.v2, color.v3, flags,
            ),
            "drawKeypoints",
        )
    } finally {
        kps.close()
    }
}

actual fun drawKeypoints(image: Mat, keypoints: List<KeyPoint>, outImage: Mat, color: Scalar, flags: Int) {
    val kps = vectorKeyPointToMat(keypoints)
    try {
        JniFeatures2.drawKeypointsOver(
            handleOf(image), handleOf(kps), handleOf(outImage),
            color.v0, color.v1, color.v2, color.v3, flags,
        )
    } finally {
        kps.close()
    }
}

actual fun drawMatches(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<DMatch>,
    matchColor: Scalar,
    singlePointColor: Scalar,
    matchesMask: List<Byte>,
    flags: Int,
): Mat {
    val kps1 = vectorKeyPointToMat(keypoints1)
    val kps2 = vectorKeyPointToMat(keypoints2)
    val mts = vectorDMatchToMat(matches)
    val mask = if (matchesMask.isEmpty()) null else vectorCharToMat(matchesMask)
    try {
        return jvmMat(
            JniFeatures2.drawMatches(
                handleOf(img1), handleOf(kps1), handleOf(img2), handleOf(kps2), handleOf(mts),
                matchColor.v0, matchColor.v1, matchColor.v2, matchColor.v3,
                singlePointColor.v0, singlePointColor.v1, singlePointColor.v2, singlePointColor.v3,
                mask?.let(::handleOf) ?: 0L, flags,
            ),
            "drawMatches",
        )
    } finally {
        kps1.close()
        kps2.close()
        mts.close()
        mask?.close()
    }
}

actual fun drawMatches(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<DMatch>,
    outImg: Mat,
    matchColor: Scalar,
    singlePointColor: Scalar,
    matchesMask: List<Byte>,
    flags: Int,
) {
    val kps1 = vectorKeyPointToMat(keypoints1)
    val kps2 = vectorKeyPointToMat(keypoints2)
    val mts = vectorDMatchToMat(matches)
    val mask = if (matchesMask.isEmpty()) null else vectorCharToMat(matchesMask)
    try {
        JniFeatures2.drawMatchesOver(
            handleOf(img1), handleOf(kps1), handleOf(img2), handleOf(kps2), handleOf(mts),
            handleOf(outImg),
            matchColor.v0, matchColor.v1, matchColor.v2, matchColor.v3,
            singlePointColor.v0, singlePointColor.v1, singlePointColor.v2, singlePointColor.v3,
            mask?.let(::handleOf) ?: 0L, flags,
        )
    } finally {
        kps1.close()
        kps2.close()
        mts.close()
        mask?.close()
    }
}

actual fun drawMatches(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<DMatch>,
    matchesThickness: Int,
    matchColor: Scalar,
    singlePointColor: Scalar,
    matchesMask: List<Byte>,
    flags: Int,
): Mat {
    val kps1 = vectorKeyPointToMat(keypoints1)
    val kps2 = vectorKeyPointToMat(keypoints2)
    val mts = vectorDMatchToMat(matches)
    val mask = if (matchesMask.isEmpty()) null else vectorCharToMat(matchesMask)
    try {
        return jvmMat(
            JniFeatures2.drawMatchesThickness(
                handleOf(img1), handleOf(kps1), handleOf(img2), handleOf(kps2), handleOf(mts),
                matchesThickness,
                matchColor.v0, matchColor.v1, matchColor.v2, matchColor.v3,
                singlePointColor.v0, singlePointColor.v1, singlePointColor.v2, singlePointColor.v3,
                mask?.let(::handleOf) ?: 0L, flags,
            ),
            "drawMatches",
        )
    } finally {
        kps1.close()
        kps2.close()
        mts.close()
        mask?.close()
    }
}

actual fun drawMatchesKnn(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<List<DMatch>>,
    matchColor: Scalar,
    singlePointColor: Scalar,
    matchesMask: List<List<Byte>>,
    flags: Int,
): Mat {
    val kps1 = vectorKeyPointToMat(keypoints1)
    val kps2 = vectorKeyPointToMat(keypoints2)
    val groupMats = matches.map { vectorDMatchToMat(it) }
    val maskMats = matchesMask.map { vectorCharToMat(it) }
    val matchesWire = encodeMatList(groupMats)
    val masksWire = if (maskMats.isEmpty()) null else encodeMatList(maskMats)
    try {
        return jvmMat(
            JniFeatures2.drawMatchesKnn(
                handleOf(img1), handleOf(kps1), handleOf(img2), handleOf(kps2), handleOf(matchesWire),
                matchColor.v0, matchColor.v1, matchColor.v2, matchColor.v3,
                singlePointColor.v0, singlePointColor.v1, singlePointColor.v2, singlePointColor.v3,
                masksWire?.let(::handleOf) ?: 0L, flags,
            ),
            "drawMatchesKnn",
        )
    } finally {
        kps1.close()
        kps2.close()
        matchesWire.close()
        masksWire?.close()
        groupMats.forEach { it.close() }
        maskMats.forEach { it.close() }
    }
}

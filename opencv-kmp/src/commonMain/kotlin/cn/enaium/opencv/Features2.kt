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
 * Feature matching and learned descriptors (`org.opencv.features2d` parity):
 * [DescriptorMatcher]/[BFMatcher]/[FlannBasedMatcher]/[LightGlueMatcher],
 * the Annoy-backed [ANNIndex], the DNN Feature2D extractors [ALIKED] and
 * [DISK], and the [goodFeaturesToTrack]/draw family statics.
 *
 * Match results travel through the MatOfDMatch wire format (Nx1 CV_32FC4
 * rows of queryIdx, trainIdx, imgIdx, distance); `List<Mat>` arguments and
 * knn/radius results use the SDK `vector_Mat` handle-packing wire (Nx1
 * CV_32SC2 rows carrying 64-bit Mat addresses).
 */

/** `cv::DescriptorMatcher::MatcherType` ids for [descriptorMatcherCreate]. */
object MatcherTypes {
    const val FLANNBASED: Int = 1
    const val BRUTEFORCE: Int = 2
    const val BRUTEFORCE_L1: Int = 3
    const val BRUTEFORCE_HAMMING: Int = 4
    const val BRUTEFORCE_HAMMINGLUT: Int = 5
    const val BRUTEFORCE_SL2: Int = 6
}

/**
 * Norm ids accepted by [bfMatcherCreate] (`cv::NormTypes` values as the
 * compiled OpenCV defines them: L1=2, L2=4, Hamming=6, Hamming2=7).
 */
object MatcherNorms {
    const val L1: Int = 2
    const val L2: Int = 4
    const val HAMMING: Int = 6
    const val HAMMING2: Int = 7
}

/** `cv::ANNIndex::Distance` metrics for [annIndexCreate]. */
object AnnIndexDistances {
    const val EUCLIDEAN: Int = 0
    const val MANHATTAN: Int = 1
    const val ANGULAR: Int = 2
    const val HAMMING: Int = 3
    const val DOTPRODUCT: Int = 4
}

/** `cv::DrawMatchesFlags` bit flags for the draw statics. */
object DrawMatchesFlags {
    /** Output image matrix is created; both images and matches are drawn. */
    const val DEFAULT: Int = 0

    /** Output image matrix is not created; matches draw over existing content. */
    const val DRAW_OVER_OUTIMG: Int = 1

    /** Single keypoints are not drawn. */
    const val NOT_DRAW_SINGLE_POINTS: Int = 2

    /** Keypoints draw as circles with size and orientation. */
    const val DRAW_RICH_KEYPOINTS: Int = 4
}

/**
 * `cv::ALIKED::Params`: network input size, descriptor normalization and
 * DNN engine/backend/target selectors. Defaults match the C++ class.
 */
data class AlikedParams(
    /** Input image size fed to the network (default 640x640). */
    var inputSize: Size = Size(640, 640),
    /** Whether to L2-normalize descriptors (default true). */
    var normalizeDescriptors: Boolean = true,
    /** DNN engine type (`cv::dnn::EngineType`); 0 = ENGINE_AUTO. */
    var engine: Int = 0,
    /** DNN backend; 0 = DNN_BACKEND_DEFAULT. */
    var backend: Int = 0,
    /** DNN target; 0 = DNN_TARGET_CPU. */
    var target: Int = 0,
)

/**
 * Abstract base class for matching keypoint descriptors
 * (`cv::DescriptorMatcher`).
 *
 * Match methods come in two groups: against an explicit train matrix, and
 * against the train collection stored via [add]. The stored-train variants
 * train implicitly before matching, like OpenCV does. Result DMatch lists
 * are decoded from the MatOfDMatch wire format; every [Mat] returned by
 * [getTrainDescriptors] must be closed by the caller.
 */
interface DescriptorMatcher : Algorithm {

    /** Adds [descriptors] (one set per train image) to the train collection. */
    fun add(descriptors: List<Mat>)

    /** Returns the train descriptor collection; the Mats are caller-owned. */
    fun getTrainDescriptors(): List<Mat>

    /** True if this matcher supports masking permissible matches. */
    fun isMaskSupported(): Boolean

    /** Trains the matcher (FlannBasedMatcher builds its index). */
    fun train()

    /** Best match per query descriptor against an explicit train matrix. */
    fun match(query: Mat, trainDescriptors: Mat, mask: Mat? = null): List<DMatch>

    /** Best match per query descriptor against the stored train collection. */
    fun match(query: Mat, masks: List<Mat> = emptyList()): List<DMatch>

    /** K best matches per query descriptor against an explicit train matrix. */
    fun knnMatch(
        query: Mat,
        trainDescriptors: Mat,
        k: Int,
        mask: Mat? = null,
        compactResult: Boolean = false,
    ): List<List<DMatch>>

    /** K best matches per query descriptor against the stored train collection. */
    fun knnMatch(
        query: Mat,
        k: Int,
        masks: List<Mat> = emptyList(),
        compactResult: Boolean = false,
    ): List<List<DMatch>>

    /** Matches within [maxDistance] against an explicit train matrix. */
    fun radiusMatch(
        query: Mat,
        trainDescriptors: Mat,
        maxDistance: Float,
        mask: Mat? = null,
        compactResult: Boolean = false,
    ): List<List<DMatch>>

    /** Matches within [maxDistance] against the stored train collection. */
    fun radiusMatch(
        query: Mat,
        maxDistance: Float,
        masks: List<Mat> = emptyList(),
        compactResult: Boolean = false,
    ): List<List<DMatch>>

    /** Persists the matcher configuration to [fileName]. */
    fun write(fileName: String)

    /** Loads the matcher configuration from [fileName]. */
    fun read(fileName: String)

    /**
     * Deep copy of the matcher. With [emptyTrainData] the copy keeps the
     * parameters but drops the train collection.
     */
    fun clone(emptyTrainData: Boolean = false): DescriptorMatcher

    override fun close()
}

/** Brute-force descriptor matcher (`cv::BFMatcher`). */
interface BFMatcher : DescriptorMatcher

/** FLANN-based descriptor matcher (`cv::FlannBasedMatcher`). */
interface FlannBasedMatcher : DescriptorMatcher

/**
 * CNN-based feature matcher (`cv::LightGlueMatcher`). Keypoint locations and
 * image sizes must be supplied via [setPairInfo] before matching unless the
 * context comes automatically from in-process ALIKED instances.
 */
interface LightGlueMatcher : DescriptorMatcher {

    /** Supplies the keypoint/image-size context for the next match call. */
    fun setPairInfo(
        queryKpts: Mat,
        trainKpts: Mat,
        queryImageSize: Size = Size(0, 0),
        trainImageSize: Size = Size(0, 0),
    )

    /** Clears stored pair context information. */
    fun clearPairInfo()
}

/**
 * Approximate Nearest Neighbors index over feature vectors
 * (`cv::ANNIndex`, Annoy-backed). Add items, [build], then [knnSearch].
 */
interface ANNIndex : AutoCloseable {

    /** Adds feature vectors (num_features x feature_dimension) to the index. */
    fun addItems(features: Mat)

    /** Builds the index; -1 picks the tree count automatically. */
    fun build(trees: Int = -1)

    /**
     * K-nearest-neighbor search. Returns a pair of fresh Mats: indices
     * (CV_32S, N x knn) and distances (CV_32F, N x knn); both are
     * caller-owned. [searchK] caps inspected nodes (-1 = trees x knn).
     */
    fun knnSearch(query: Mat, knn: Int, searchK: Int = -1): Pair<Mat, Mat>

    /** Saves the index to [filename] and loads it (mmap); no more adds. */
    fun save(filename: String, prefault: Boolean = false)

    /** Loads (mmaps) an index from [filename]. */
    fun load(filename: String, prefault: Boolean = false)

    /** Number of trees in the index. */
    val treeNumber: Int

    /** Number of feature vectors in the index. */
    val itemNumber: Int

    /** Prepares an on-disk build in [filename] (call before [addItems]). */
    fun setOnDiskBuild(filename: String): Boolean

    /** Seeds the index RNG (only meaningful before [build]). */
    fun setSeed(seed: Int)

    override fun close()
}

/**
 * ALIKED feature detector and descriptor extractor
 * (`cv::ALIKED`, CNN-based, 128-D float descriptors).
 */
interface ALIKED : Feature2D

/**
 * DISK feature detector and descriptor extractor
 * (`cv::DISK`, DNN-based, 128-D L2-normalized descriptors).
 */
interface DISK : Feature2D {
    /** Maximum keypoints returned per image (-1 keeps all detections). */
    var maxKeypoints: Int

    /** Discard keypoints with network score strictly below this value. */
    var scoreThreshold: Float

    /** Target input size fed to the network; Size(0,0) uses the default. */
    var imageSize: Size
}

/* =========================================================================
 * Factories
 * ========================================================================= */

/**
 * Creates a descriptor matcher of a given type with default parameters
 * (`cv::DescriptorMatcher::create(String)`), e.g. "BruteForce",
 * "BruteForce-L1", "BruteForce-Hamming", "BruteForce-Hamming(2)",
 * "FlannBased".
 */
expect fun descriptorMatcherCreate(type: String): DescriptorMatcher

/** Creates a matcher by [MatcherTypes] id. */
expect fun descriptorMatcherCreate(type: Int): DescriptorMatcher

/**
 * Brute-force matcher (`cv::BFMatcher::create`). [normType] is a
 * [MatcherNorms] value; with [crossCheck] only mutually best matches are
 * returned.
 */
expect fun bfMatcherCreate(
    normType: Int = MatcherNorms.L2,
    crossCheck: Boolean = false,
): BFMatcher

/**
 * FLANN-based matcher (`cv::FlannBasedMatcher`). [indexParams] selects the
 * flann index type: "kdtree" (default), "linear", "kmeans", "lsh",
 * "composite", "autotuned" or "hierarchical"; unknown values fall back to
 * the KD-tree index.
 */
expect fun flannBasedMatcherCreate(indexParams: String = "kdtree"): FlannBasedMatcher

/** LightGlue matcher from an ONNX model file. */
expect fun lightGlueMatcherCreate(
    modelPath: String,
    scoreThreshold: Float = 0.0f,
    backend: Int = 0,
    target: Int = 0,
): LightGlueMatcher

/** LightGlue matcher from in-memory ONNX model data. */
expect fun lightGlueMatcherCreateFromMemory(
    modelData: ByteArray,
    scoreThreshold: Float = 0.0f,
    backend: Int = 0,
    target: Int = 0,
): LightGlueMatcher

/**
 * Annoy index for approximate nearest neighbors
 * (`cv::ANNIndex::create`); [distType] is an [AnnIndexDistances] value.
 */
expect fun annIndexCreate(dim: Int, distType: Int = AnnIndexDistances.EUCLIDEAN): ANNIndex

/** ALIKED detector from an ONNX model file. */
expect fun alikedCreate(modelPath: String, params: AlikedParams = AlikedParams()): ALIKED

/** DISK detector from an ONNX model file. */
expect fun diskCreate(
    modelPath: String,
    maxKeypoints: Int = -1,
    scoreThreshold: Float = 0.0f,
    imageSize: Size = Size(0, 0),
    backendId: Int = 0,
    targetId: Int = 0,
): DISK

/** DISK detector from an in-memory ONNX model buffer. */
expect fun diskCreateFromMemory(
    bufferModel: ByteArray,
    maxKeypoints: Int = -1,
    scoreThreshold: Float = 0.0f,
    imageSize: Size = Size(0, 0),
    backendId: Int = 0,
    targetId: Int = 0,
): DISK

/* =========================================================================
 * Features statics (org.opencv.features.Features)
 * ========================================================================= */

/**
 * Determines strong corners on an image (`cv::goodFeaturesToTrack`).
 * Returns the strongest [maxCorners] corners; `maxCorners <= 0` returns all
 * detected corners.
 */
expect fun goodFeaturesToTrack(
    image: Mat,
    maxCorners: Int,
    qualityLevel: Double,
    minDistance: Double,
    mask: Mat? = null,
    blockSize: Int = 3,
    useHarrisDetector: Boolean = false,
    k: Double = 0.04,
): List<Point>

/** [goodFeaturesToTrack] with an explicit Sobel [gradientSize] aperture. */
expect fun goodFeaturesToTrack(
    image: Mat,
    maxCorners: Int,
    qualityLevel: Double,
    minDistance: Double,
    mask: Mat,
    blockSize: Int,
    gradientSize: Int,
    useHarrisDetector: Boolean = false,
    k: Double = 0.04,
): List<Point>

/** Detected corners together with their quality measures. */
data class GoodFeaturesResult(
    val corners: List<Point>,
    /** Per-corner quality measure (Nx1 CV_32FC1 decoded). */
    val quality: List<Float>,
)

/** [goodFeaturesToTrack] that also returns each corner's quality measure. */
expect fun goodFeaturesToTrackWithQuality(
    image: Mat,
    maxCorners: Int,
    qualityLevel: Double,
    minDistance: Double,
    mask: Mat? = null,
    blockSize: Int = 3,
    gradientSize: Int = 3,
    useHarrisDetector: Boolean = false,
    k: Double = 0.04,
): GoodFeaturesResult

/** Draws [keypoints] onto a fresh output image (`cv::drawKeypoints`). */
expect fun drawKeypoints(
    image: Mat,
    keypoints: List<KeyPoint>,
    color: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    flags: Int = DrawMatchesFlags.DEFAULT,
): Mat

/** Draws [keypoints] over the caller's [outImage] in place. */
expect fun drawKeypoints(
    image: Mat,
    keypoints: List<KeyPoint>,
    outImage: Mat,
    color: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    flags: Int = DrawMatchesFlags.DEFAULT,
)

/** Draws [matches] between two keypoint sets onto a fresh image. */
expect fun drawMatches(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<DMatch>,
    matchColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    singlePointColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    matchesMask: List<Byte> = emptyList(),
    flags: Int = DrawMatchesFlags.DEFAULT,
): Mat

/** Draws [matches] over the caller's [outImg] in place. */
expect fun drawMatches(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<DMatch>,
    outImg: Mat,
    matchColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    singlePointColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    matchesMask: List<Byte> = emptyList(),
    flags: Int = DrawMatchesFlags.DEFAULT,
)

/** Draws [matches] with an explicit line [matchesThickness]. */
expect fun drawMatches(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<DMatch>,
    matchesThickness: Int,
    matchColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    singlePointColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    matchesMask: List<Byte> = emptyList(),
    flags: Int = DrawMatchesFlags.DEFAULT,
): Mat

/** Draws grouped matches (one list per query descriptor) onto a fresh image. */
expect fun drawMatchesKnn(
    img1: Mat,
    keypoints1: List<KeyPoint>,
    img2: Mat,
    keypoints2: List<KeyPoint>,
    matches: List<List<DMatch>>,
    matchColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    singlePointColor: Scalar = Scalar(-1.0, -1.0, -1.0, -1.0),
    matchesMask: List<List<Byte>> = emptyList(),
    flags: Int = DrawMatchesFlags.DEFAULT,
): Mat

/* =========================================================================
 * Internal wire helpers shared with the platform implementations
 * ========================================================================= */

/** Writes a 64-bit Mat address into [bytes] as two LE int32 pairs. */
internal fun writeMatHandleAddress(bytes: ByteArray, offset: Int, address: Long) {
    val lo = (address and 0xFFFFFFFFL).toInt()
    val hi = (address ushr 32).toInt()
    bytes[offset] = (lo and 0xFF).toByte()
    bytes[offset + 1] = ((lo ushr 8) and 0xFF).toByte()
    bytes[offset + 2] = ((lo ushr 16) and 0xFF).toByte()
    bytes[offset + 3] = ((lo ushr 24) and 0xFF).toByte()
    bytes[offset + 4] = (hi and 0xFF).toByte()
    bytes[offset + 5] = ((hi ushr 8) and 0xFF).toByte()
    bytes[offset + 6] = ((hi ushr 16) and 0xFF).toByte()
    bytes[offset + 7] = ((hi ushr 24) and 0xFF).toByte()
}

/**
 * Reads a 64-bit Mat address from the SDK `vector_Mat` wire layout used by
 * the native shim (`mats_to_wire`): each CV_32SC2 row stores the address
 * high word first, low word second - [hi32][lo32].
 */
internal fun readMatHandleAddress(bytes: ByteArray, offset: Int): Long {
    val hi = (bytes[offset].toInt() and 0xFF) or
        ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
        ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
        ((bytes[offset + 3].toInt() and 0xFF) shl 24)
    val lo = (bytes[offset + 4].toInt() and 0xFF) or
        ((bytes[offset + 5].toInt() and 0xFF) shl 8) or
        ((bytes[offset + 6].toInt() and 0xFF) shl 16) or
        ((bytes[offset + 7].toInt() and 0xFF) shl 24)
    return (hi.toLong() shl 32) or (lo.toLong() and 0xFFFFFFFFL)
}

/**
 * Packs 64-bit Mat addresses into the SDK `vector_Mat` wire format: an Nx1
 * CV_32SC2 Mat, each row carrying (hi32, lo32). The returned Mat must be
 * closed by the caller.
 */
internal fun packMatHandleWire(addresses: LongArray): Mat {
    val wire = mat(addresses.size, 1, MatType.of(CV_32S, 2))
    if (addresses.isEmpty()) return wire
    val bytes = ByteArray(addresses.size * 8)
    addresses.forEachIndexed { index, address -> writeMatHandleAddress(bytes, index * 8, address) }
    wire.pixels = bytes
    return wire
}

/** Encodes [mats] into the `vector_Mat` wire format (platform-specific). */
internal expect fun encodeMatList(mats: List<Mat>): Mat

/**
 * Decodes a `vector_Mat` wire Mat into wrapped, caller-owned [Mat]s. The
 * caller must close the [wire] wrapper itself.
 */
internal expect fun decodeMatList(wire: Mat, operation: String): List<Mat>

/** Decodes a nested-DMatch wire Mat into per-query DMatch lists. */
internal fun decodeDMatchGroups(wire: Mat, operation: String): List<List<DMatch>> {
    val mats = decodeMatList(wire, operation)
    return mats.map { mat ->
        try {
            matToVectorDMatch(mat)
        } finally {
            mat.close()
        }
    }
}

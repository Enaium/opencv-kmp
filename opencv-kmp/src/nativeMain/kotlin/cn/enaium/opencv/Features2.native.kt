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

import cvk.cvk_aliked_clear
import cvk.cvk_aliked_compute
import cvk.cvk_aliked_create
import cvk.cvk_aliked_default_norm
import cvk.cvk_aliked_descriptor_size
import cvk.cvk_aliked_descriptor_type
import cvk.cvk_aliked_detect
import cvk.cvk_aliked_detect_and_compute
import cvk.cvk_aliked_empty
import cvk.cvk_aliked_get_default_name
import cvk.cvk_aliked_read
import cvk.cvk_aliked_release
import cvk.cvk_aliked_save
import cvk.cvk_aliked_t
import cvk.cvk_aliked_write
import cvk.cvk_ann_index_add_items
import cvk.cvk_ann_index_build
import cvk.cvk_ann_index_create
import cvk.cvk_ann_index_item_number
import cvk.cvk_ann_index_knn_search
import cvk.cvk_ann_index_load
import cvk.cvk_ann_index_release
import cvk.cvk_ann_index_save
import cvk.cvk_ann_index_set_on_disk_build
import cvk.cvk_ann_index_set_seed
import cvk.cvk_ann_index_t
import cvk.cvk_ann_index_tree_number
import cvk.cvk_bf_matcher_create
import cvk.cvk_descriptor_matcher_add
import cvk.cvk_descriptor_matcher_clear
import cvk.cvk_descriptor_matcher_clone
import cvk.cvk_descriptor_matcher_create
import cvk.cvk_descriptor_matcher_create_type
import cvk.cvk_descriptor_matcher_empty
import cvk.cvk_descriptor_matcher_get_default_name
import cvk.cvk_descriptor_matcher_get_train_descriptors
import cvk.cvk_descriptor_matcher_is_mask_supported
import cvk.cvk_descriptor_matcher_knn_match
import cvk.cvk_descriptor_matcher_knn_match_train
import cvk.cvk_descriptor_matcher_match
import cvk.cvk_descriptor_matcher_match_train
import cvk.cvk_descriptor_matcher_radius_match
import cvk.cvk_descriptor_matcher_radius_match_train
import cvk.cvk_descriptor_matcher_read
import cvk.cvk_descriptor_matcher_release
import cvk.cvk_descriptor_matcher_save
import cvk.cvk_descriptor_matcher_t
import cvk.cvk_descriptor_matcher_train
import cvk.cvk_descriptor_matcher_write
import cvk.cvk_disk_clear
import cvk.cvk_disk_compute
import cvk.cvk_disk_create
import cvk.cvk_disk_create_from_memory
import cvk.cvk_disk_default_norm
import cvk.cvk_disk_descriptor_size
import cvk.cvk_disk_descriptor_type
import cvk.cvk_disk_detect
import cvk.cvk_disk_detect_and_compute
import cvk.cvk_disk_empty
import cvk.cvk_disk_get_default_name
import cvk.cvk_disk_get_max_keypoints
import cvk.cvk_disk_get_score_threshold
import cvk.cvk_disk_image_size
import cvk.cvk_disk_read
import cvk.cvk_disk_release
import cvk.cvk_disk_save
import cvk.cvk_disk_set_image_size
import cvk.cvk_disk_set_max_keypoints
import cvk.cvk_disk_set_score_threshold
import cvk.cvk_disk_t
import cvk.cvk_disk_write
import cvk.cvk_draw_keypoints
import cvk.cvk_draw_keypoints_over
import cvk.cvk_draw_matches
import cvk.cvk_draw_matches_knn
import cvk.cvk_draw_matches_over
import cvk.cvk_draw_matches_thickness
import cvk.cvk_features_good_features_to_track
import cvk.cvk_features_good_features_to_track_gradient
import cvk.cvk_features_good_features_to_track_quality
import cvk.cvk_flann_matcher_create
import cvk.cvk_last_error
import cvk.cvk_lightglue_matcher_clear_pair_info
import cvk.cvk_lightglue_matcher_create
import cvk.cvk_lightglue_matcher_create_from_memory
import cvk.cvk_lightglue_matcher_set_pair_info
import cvk.cvk_mat_t
import kotlin.concurrent.Volatile
import kotlinx.cinterop.CPointed
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toCPointer
import kotlinx.cinterop.toKString
import kotlinx.cinterop.toLong
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

/** Requires a freshly-created non-null handle or throws with the native error. */
private fun <T : CPointed> requireHandle(ptr: CPointer<T>?, operation: String): CPointer<T> =
    ptr ?: throw OpenCVException(operation, lastNativeError())

internal actual fun encodeMatList(mats: List<Mat>): Mat =
    packMatHandleWire(mats.map { it.nativeHandle().toLong() }.toLongArray())

internal actual fun decodeMatList(wire: Mat, operation: String): List<Mat> {
    if (wire.isEmpty) return emptyList()
    val bytes = wire.pixels
    return List(wire.rows) { index ->
        val address = readMatHandleAddress(bytes, index * 8)
        address.toCPointer<cvk_mat_t>()?.let { NativeMat(it) }
            ?: throw OpenCVException(operation, lastNativeError())
    }
}

// =========================================================================
// DescriptorMatcher family
// =========================================================================

internal open class NativeDescriptorMatcher(
    @Volatile private var raw: CPointer<cvk_descriptor_matcher_t>?,
) : DescriptorMatcher {

    protected fun check(): CPointer<cvk_descriptor_matcher_t> =
        raw ?: throw IllegalStateException("DescriptorMatcher is closed")

    override fun add(descriptors: List<Mat>) {
        val wire = encodeMatList(descriptors)
        try {
            cvk_descriptor_matcher_add(check(), wire.nativeHandle())
        } finally {
            wire.close()
        }
    }

    override fun getTrainDescriptors(): List<Mat> {
        val wire = nativeMat(cvk_descriptor_matcher_get_train_descriptors(check()), "getTrainDescriptors")
        return try {
            decodeMatList(wire, "getTrainDescriptors")
        } finally {
            wire.close()
        }
    }

    override fun isMaskSupported(): Boolean = cvk_descriptor_matcher_is_mask_supported(check()) != 0

    override fun train() {
        cvk_descriptor_matcher_train(check())
    }

    override fun write(fileName: String) {
        cvk_descriptor_matcher_write(check(), fileName)
    }

    override fun read(fileName: String) {
        cvk_descriptor_matcher_read(check(), fileName)
    }

    override fun clear() {
        cvk_descriptor_matcher_clear(check())
    }

    override fun empty(): Boolean = cvk_descriptor_matcher_empty(check()) != 0

    override fun save(filename: String) {
        cvk_descriptor_matcher_save(check(), filename)
    }

    override fun getDefaultName(): String =
        cvk_descriptor_matcher_get_default_name(check())!!.toKString()

    override fun clone(emptyTrainData: Boolean): DescriptorMatcher =
        NativeDescriptorMatcher(
            requireHandle(cvk_descriptor_matcher_clone(check(), if (emptyTrainData) 1 else 0), "clone"),
        )

    override fun match(query: Mat, trainDescriptors: Mat, mask: Mat?): List<DMatch> {
        val m = nativeMat(
            cvk_descriptor_matcher_match_train(
                check(), query.nativeHandle(), trainDescriptors.nativeHandle(), mask?.nativeHandle(),
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
            val m = nativeMat(cvk_descriptor_matcher_match(check(), query.nativeHandle(), wire.nativeHandle()), "match")
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
        val wire = nativeMat(
            cvk_descriptor_matcher_knn_match_train(
                check(), query.nativeHandle(), trainDescriptors.nativeHandle(), k,
                mask?.nativeHandle(), if (compactResult) 1 else 0,
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
            val wire = nativeMat(
                cvk_descriptor_matcher_knn_match(
                    check(), query.nativeHandle(), k, masksWire.nativeHandle(),
                    if (compactResult) 1 else 0,
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
        val wire = nativeMat(
            cvk_descriptor_matcher_radius_match_train(
                check(), query.nativeHandle(), trainDescriptors.nativeHandle(), maxDistance,
                mask?.nativeHandle(), if (compactResult) 1 else 0,
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
            val wire = nativeMat(
                cvk_descriptor_matcher_radius_match(
                    check(), query.nativeHandle(), maxDistance, masksWire.nativeHandle(),
                    if (compactResult) 1 else 0,
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
        val h = raw ?: return
        raw = null
        cvk_descriptor_matcher_release(h)
    }
}

internal class NativeBFMatcher(handle: CPointer<cvk_descriptor_matcher_t>?) :
    NativeDescriptorMatcher(handle), BFMatcher

internal class NativeFlannBasedMatcher(handle: CPointer<cvk_descriptor_matcher_t>?) :
    NativeDescriptorMatcher(handle), FlannBasedMatcher

internal class NativeLightGlueMatcher(handle: CPointer<cvk_descriptor_matcher_t>?) :
    NativeDescriptorMatcher(handle), LightGlueMatcher {

    override fun setPairInfo(
        queryKpts: Mat,
        trainKpts: Mat,
        queryImageSize: Size,
        trainImageSize: Size,
    ) {
        cvk_lightglue_matcher_set_pair_info(
            check(), queryKpts.nativeHandle(), trainKpts.nativeHandle(),
            queryImageSize.width.toDouble(), queryImageSize.height.toDouble(),
            trainImageSize.width.toDouble(), trainImageSize.height.toDouble(),
        )
    }

    override fun clearPairInfo() {
        cvk_lightglue_matcher_clear_pair_info(check())
    }
}

// =========================================================================
// ANNIndex
// =========================================================================

internal class NativeAnnIndex(@Volatile private var raw: CPointer<cvk_ann_index_t>?) : ANNIndex {

    private fun check(): CPointer<cvk_ann_index_t> =
        raw ?: throw IllegalStateException("ANNIndex is closed")

    override fun addItems(features: Mat) {
        cvk_ann_index_add_items(check(), features.nativeHandle())
    }

    override fun build(trees: Int) {
        cvk_ann_index_build(check(), trees)
    }

    override fun knnSearch(query: Mat, knn: Int, searchK: Int): Pair<Mat, Mat> = memScoped {
        val indices = alloc<CPointerVar<cvk_mat_t>>()
        val dists = alloc<CPointerVar<cvk_mat_t>>()
        cvk_ann_index_knn_search(check(), query.nativeHandle(), knn, searchK, indices.ptr, dists.ptr)
        nativeMat(indices.value, "knnSearch") to nativeMat(dists.value, "knnSearch")
    }

    override fun save(filename: String, prefault: Boolean) {
        cvk_ann_index_save(check(), filename, if (prefault) 1 else 0)
    }

    override fun load(filename: String, prefault: Boolean) {
        cvk_ann_index_load(check(), filename, if (prefault) 1 else 0)
    }

    override val treeNumber: Int get() = cvk_ann_index_tree_number(check())

    override val itemNumber: Int get() = cvk_ann_index_item_number(check())

    override fun setOnDiskBuild(filename: String): Boolean =
        cvk_ann_index_set_on_disk_build(check(), filename) != 0

    override fun setSeed(seed: Int) {
        cvk_ann_index_set_seed(check(), seed)
    }

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_ann_index_release(h)
    }
}

// =========================================================================
// ALIKED / DISK (Feature2D)
// =========================================================================

private fun decodeKeypointsAndDescriptors(
    keypointsPtr: CPointer<cvk_mat_t>?,
    descriptorsPtr: CPointer<cvk_mat_t>?,
    operation: String,
): Pair<List<KeyPoint>, Mat> {
    val keypoints = nativeMat(keypointsPtr, operation)
    val descriptors = nativeMat(descriptorsPtr, operation)
    return try {
        matToVectorKeyPoint(keypoints) to descriptors
    } finally {
        keypoints.close()
    }
}

internal class NativeAliked(@Volatile private var raw: CPointer<cvk_aliked_t>?) : ALIKED {

    private fun check(): CPointer<cvk_aliked_t> =
        raw ?: throw IllegalStateException("ALIKED is closed")

    override fun detect(image: Mat, mask: Mat?): List<KeyPoint> {
        val m = nativeMat(cvk_aliked_detect(check(), image.nativeHandle(), mask?.nativeHandle()), "detect")
        return try {
            matToVectorKeyPoint(m)
        } finally {
            m.close()
        }
    }

    override fun compute(image: Mat, keypoints: List<KeyPoint>): Pair<List<KeyPoint>, Mat> {
        val kps = vectorKeyPointToMat(keypoints)
        try {
            return memScoped {
                val outKeypoints = alloc<CPointerVar<cvk_mat_t>>()
                val outDescriptors = alloc<CPointerVar<cvk_mat_t>>()
                cvk_aliked_compute(
                    check(), image.nativeHandle(), kps.nativeHandle(),
                    outKeypoints.ptr, outDescriptors.ptr,
                )
                decodeKeypointsAndDescriptors(outKeypoints.value, outDescriptors.value, "compute")
            }
        } finally {
            kps.close()
        }
    }

    override fun detectAndCompute(image: Mat, mask: Mat?): Pair<List<KeyPoint>, Mat> = memScoped {
        val outKeypoints = alloc<CPointerVar<cvk_mat_t>>()
        val outDescriptors = alloc<CPointerVar<cvk_mat_t>>()
        cvk_aliked_detect_and_compute(
            check(), image.nativeHandle(), mask?.nativeHandle(), 0,
            outKeypoints.ptr, outDescriptors.ptr,
        )
        decodeKeypointsAndDescriptors(outKeypoints.value, outDescriptors.value, "detectAndCompute")
    }

    override val descriptorSize: Int get() = cvk_aliked_descriptor_size(check())

    override val descriptorType: Int get() = cvk_aliked_descriptor_type(check())

    override val defaultNorm: Int get() = cvk_aliked_default_norm(check())

    override fun write(fileName: String) {
        cvk_aliked_write(check(), fileName)
    }

    override fun read(fileName: String) {
        cvk_aliked_read(check(), fileName)
    }

    override fun clear() {
        cvk_aliked_clear(check())
    }

    override fun empty(): Boolean = cvk_aliked_empty(check()) != 0

    override fun save(filename: String) {
        cvk_aliked_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_aliked_get_default_name(check())!!.toKString()

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_aliked_release(h)
    }
}

internal class NativeDisk(@Volatile private var raw: CPointer<cvk_disk_t>?) : DISK {

    private fun check(): CPointer<cvk_disk_t> =
        raw ?: throw IllegalStateException("DISK is closed")

    override fun detect(image: Mat, mask: Mat?): List<KeyPoint> {
        val m = nativeMat(cvk_disk_detect(check(), image.nativeHandle(), mask?.nativeHandle()), "detect")
        return try {
            matToVectorKeyPoint(m)
        } finally {
            m.close()
        }
    }

    override fun compute(image: Mat, keypoints: List<KeyPoint>): Pair<List<KeyPoint>, Mat> {
        val kps = vectorKeyPointToMat(keypoints)
        try {
            return memScoped {
                val outKeypoints = alloc<CPointerVar<cvk_mat_t>>()
                val outDescriptors = alloc<CPointerVar<cvk_mat_t>>()
                cvk_disk_compute(
                    check(), image.nativeHandle(), kps.nativeHandle(),
                    outKeypoints.ptr, outDescriptors.ptr,
                )
                decodeKeypointsAndDescriptors(outKeypoints.value, outDescriptors.value, "compute")
            }
        } finally {
            kps.close()
        }
    }

    override fun detectAndCompute(image: Mat, mask: Mat?): Pair<List<KeyPoint>, Mat> = memScoped {
        val outKeypoints = alloc<CPointerVar<cvk_mat_t>>()
        val outDescriptors = alloc<CPointerVar<cvk_mat_t>>()
        cvk_disk_detect_and_compute(
            check(), image.nativeHandle(), mask?.nativeHandle(), 0,
            outKeypoints.ptr, outDescriptors.ptr,
        )
        decodeKeypointsAndDescriptors(outKeypoints.value, outDescriptors.value, "detectAndCompute")
    }

    override val descriptorSize: Int get() = cvk_disk_descriptor_size(check())

    override val descriptorType: Int get() = cvk_disk_descriptor_type(check())

    override val defaultNorm: Int get() = cvk_disk_default_norm(check())

    override var maxKeypoints: Int
        get() = cvk_disk_get_max_keypoints(check())
        set(value) {
            cvk_disk_set_max_keypoints(check(), value)
        }

    override var scoreThreshold: Float
        get() = cvk_disk_get_score_threshold(check())
        set(value) {
            cvk_disk_set_score_threshold(check(), value)
        }

    override var imageSize: Size
        get() {
            val out = DoubleArray(2)
            out.usePinned { pinned -> cvk_disk_image_size(check(), pinned.addressOf(0)) }
            return Size(out[0].toInt(), out[1].toInt())
        }
        set(value) {
            cvk_disk_set_image_size(check(), value.width.toDouble(), value.height.toDouble())
        }

    override fun write(fileName: String) {
        cvk_disk_write(check(), fileName)
    }

    override fun read(fileName: String) {
        cvk_disk_read(check(), fileName)
    }

    override fun clear() {
        cvk_disk_clear(check())
    }

    override fun empty(): Boolean = cvk_disk_empty(check()) != 0

    override fun save(filename: String) {
        cvk_disk_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_disk_get_default_name(check())!!.toKString()

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_disk_release(h)
    }
}

// =========================================================================
// Factories
// =========================================================================

actual fun descriptorMatcherCreate(type: String): DescriptorMatcher =
    NativeDescriptorMatcher(requireHandle(cvk_descriptor_matcher_create(type), "descriptorMatcherCreate"))

actual fun descriptorMatcherCreate(type: Int): DescriptorMatcher =
    NativeDescriptorMatcher(requireHandle(cvk_descriptor_matcher_create_type(type), "descriptorMatcherCreate"))

actual fun bfMatcherCreate(normType: Int, crossCheck: Boolean): BFMatcher =
    NativeBFMatcher(requireHandle(cvk_bf_matcher_create(normType, if (crossCheck) 1 else 0), "bfMatcherCreate"))

actual fun flannBasedMatcherCreate(indexParams: String): FlannBasedMatcher =
    NativeFlannBasedMatcher(requireHandle(cvk_flann_matcher_create(indexParams), "flannBasedMatcherCreate"))

actual fun lightGlueMatcherCreate(modelPath: String, scoreThreshold: Float, backend: Int, target: Int): LightGlueMatcher =
    NativeLightGlueMatcher(requireHandle(
        cvk_lightglue_matcher_create(modelPath, scoreThreshold, backend, target),
        "lightGlueMatcherCreate",
    ))

actual fun lightGlueMatcherCreateFromMemory(modelData: ByteArray, scoreThreshold: Float, backend: Int, target: Int): LightGlueMatcher =
    NativeLightGlueMatcher(requireHandle(
        cvk_lightglue_matcher_create_from_memory(
            modelData.asUByteArray().usePinned { it.addressOf(0) },
            modelData.size.convert(),
            scoreThreshold, backend, target,
        ),
        "lightGlueMatcherCreateFromMemory",
    ))

actual fun annIndexCreate(dim: Int, distType: Int): ANNIndex =
    NativeAnnIndex(requireHandle(cvk_ann_index_create(dim, distType), "annIndexCreate"))

actual fun alikedCreate(modelPath: String, params: AlikedParams): ALIKED =
    NativeAliked(requireHandle(
        cvk_aliked_create(
            modelPath, params.inputSize.width, params.inputSize.height,
            if (params.normalizeDescriptors) 1 else 0,
            params.engine, params.backend, params.target,
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
): DISK = NativeDisk(requireHandle(
    cvk_disk_create(
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
): DISK = NativeDisk(requireHandle(
    cvk_disk_create_from_memory(
        bufferModel.asUByteArray().usePinned { it.addressOf(0) },
        bufferModel.size.convert(),
        maxKeypoints, scoreThreshold,
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
    val m = nativeMat(
        cvk_features_good_features_to_track(
            image.nativeHandle(), maxCorners, qualityLevel, minDistance,
            mask?.nativeHandle(), blockSize, if (useHarrisDetector) 1 else 0, k,
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
    val m = nativeMat(
        cvk_features_good_features_to_track_gradient(
            image.nativeHandle(), maxCorners, qualityLevel, minDistance,
            mask.nativeHandle(), blockSize, gradientSize, if (useHarrisDetector) 1 else 0, k,
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
): GoodFeaturesResult = memScoped {
    val corners = alloc<CPointerVar<cvk_mat_t>>()
    val quality = alloc<CPointerVar<cvk_mat_t>>()
    cvk_features_good_features_to_track_quality(
        image.nativeHandle(), maxCorners, qualityLevel, minDistance,
        mask?.nativeHandle(), blockSize, gradientSize, if (useHarrisDetector) 1 else 0, k,
        corners.ptr, quality.ptr,
    )
    val cornersMat = nativeMat(corners.value, "goodFeaturesToTrackWithQuality")
    val qualityMat = nativeMat(quality.value, "goodFeaturesToTrackWithQuality")
    return@memScoped try {
        GoodFeaturesResult(matToVectorPoint(cornersMat), matToVectorFloat(qualityMat))
    } finally {
        cornersMat.close()
        qualityMat.close()
    }
}

actual fun drawKeypoints(image: Mat, keypoints: List<KeyPoint>, color: Scalar, flags: Int): Mat {
    val kps = vectorKeyPointToMat(keypoints)
    try {
        return nativeMat(
            cvk_draw_keypoints(image.nativeHandle(), kps.nativeHandle(), color.toCvk(), flags),
            "drawKeypoints",
        )
    } finally {
        kps.close()
    }
}

actual fun drawKeypoints(image: Mat, keypoints: List<KeyPoint>, outImage: Mat, color: Scalar, flags: Int) {
    val kps = vectorKeyPointToMat(keypoints)
    try {
        cvk_draw_keypoints_over(image.nativeHandle(), kps.nativeHandle(), outImage.nativeHandle(), color.toCvk(), flags)
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
        return nativeMat(
            cvk_draw_matches(
                img1.nativeHandle(), kps1.nativeHandle(), img2.nativeHandle(), kps2.nativeHandle(),
                mts.nativeHandle(), matchColor.toCvk(), singlePointColor.toCvk(),
                mask?.nativeHandle(), flags,
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
        cvk_draw_matches_over(
            img1.nativeHandle(), kps1.nativeHandle(), img2.nativeHandle(), kps2.nativeHandle(),
            mts.nativeHandle(), outImg.nativeHandle(), matchColor.toCvk(), singlePointColor.toCvk(),
            mask?.nativeHandle(), flags,
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
        return nativeMat(
            cvk_draw_matches_thickness(
                img1.nativeHandle(), kps1.nativeHandle(), img2.nativeHandle(), kps2.nativeHandle(),
                mts.nativeHandle(), matchesThickness, matchColor.toCvk(), singlePointColor.toCvk(),
                mask?.nativeHandle(), flags,
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
        return nativeMat(
            cvk_draw_matches_knn(
                img1.nativeHandle(), kps1.nativeHandle(), img2.nativeHandle(), kps2.nativeHandle(),
                matchesWire.nativeHandle(), matchColor.toCvk(), singlePointColor.toCvk(),
                masksWire?.nativeHandle(), flags,
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

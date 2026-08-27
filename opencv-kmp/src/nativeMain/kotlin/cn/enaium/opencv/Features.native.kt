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

import cvk.cvk_affine_clear
import cvk.cvk_affine_compute
import cvk.cvk_affine_create
import cvk.cvk_affine_default_norm
import cvk.cvk_affine_descriptor_size
import cvk.cvk_affine_descriptor_type
import cvk.cvk_affine_detect
import cvk.cvk_affine_detect_and_compute
import cvk.cvk_affine_empty
import cvk.cvk_affine_get_default_name
import cvk.cvk_affine_get_view_params
import cvk.cvk_affine_read
import cvk.cvk_affine_release
import cvk.cvk_affine_save
import cvk.cvk_affine_set_view_params
import cvk.cvk_affine_t
import cvk.cvk_affine_write
import cvk.cvk_fast_feature_detector_clear
import cvk.cvk_fast_feature_detector_compute
import cvk.cvk_fast_feature_detector_create
import cvk.cvk_fast_feature_detector_default_norm
import cvk.cvk_fast_feature_detector_descriptor_size
import cvk.cvk_fast_feature_detector_descriptor_type
import cvk.cvk_fast_feature_detector_detect
import cvk.cvk_fast_feature_detector_detect_and_compute
import cvk.cvk_fast_feature_detector_empty
import cvk.cvk_fast_feature_detector_get_default_name
import cvk.cvk_fast_feature_detector_get_nonmax_suppression
import cvk.cvk_fast_feature_detector_get_threshold
import cvk.cvk_fast_feature_detector_get_type
import cvk.cvk_fast_feature_detector_read
import cvk.cvk_fast_feature_detector_release
import cvk.cvk_fast_feature_detector_save
import cvk.cvk_fast_feature_detector_set_nonmax_suppression
import cvk.cvk_fast_feature_detector_set_threshold
import cvk.cvk_fast_feature_detector_set_type
import cvk.cvk_fast_feature_detector_t
import cvk.cvk_fast_feature_detector_write
import cvk.cvk_feature2d_t
import cvk.cvk_free_buffer
import cvk.cvk_gftt_detector_clear
import cvk.cvk_gftt_detector_compute
import cvk.cvk_gftt_detector_create
import cvk.cvk_gftt_detector_default_norm
import cvk.cvk_gftt_detector_descriptor_size
import cvk.cvk_gftt_detector_descriptor_type
import cvk.cvk_gftt_detector_detect
import cvk.cvk_gftt_detector_detect_and_compute
import cvk.cvk_gftt_detector_empty
import cvk.cvk_gftt_detector_get_block_size
import cvk.cvk_gftt_detector_get_default_name
import cvk.cvk_gftt_detector_get_gradient_size
import cvk.cvk_gftt_detector_get_harris_detector
import cvk.cvk_gftt_detector_get_k
import cvk.cvk_gftt_detector_get_max_features
import cvk.cvk_gftt_detector_get_min_distance
import cvk.cvk_gftt_detector_get_quality_level
import cvk.cvk_gftt_detector_read
import cvk.cvk_gftt_detector_release
import cvk.cvk_gftt_detector_save
import cvk.cvk_gftt_detector_set_block_size
import cvk.cvk_gftt_detector_set_gradient_size
import cvk.cvk_gftt_detector_set_harris_detector
import cvk.cvk_gftt_detector_set_k
import cvk.cvk_gftt_detector_set_max_features
import cvk.cvk_gftt_detector_set_min_distance
import cvk.cvk_gftt_detector_set_quality_level
import cvk.cvk_gftt_detector_t
import cvk.cvk_gftt_detector_write
import cvk.cvk_last_error
import cvk.cvk_mat_t
import cvk.cvk_mser_clear
import cvk.cvk_mser_compute
import cvk.cvk_mser_create
import cvk.cvk_mser_default_norm
import cvk.cvk_mser_descriptor_size
import cvk.cvk_mser_descriptor_type
import cvk.cvk_mser_detect
import cvk.cvk_mser_detect_and_compute
import cvk.cvk_mser_detect_regions
import cvk.cvk_mser_empty
import cvk.cvk_mser_get_area_threshold
import cvk.cvk_mser_get_default_name
import cvk.cvk_mser_get_delta
import cvk.cvk_mser_get_edge_blur_size
import cvk.cvk_mser_get_max_area
import cvk.cvk_mser_get_max_evolution
import cvk.cvk_mser_get_max_variation
import cvk.cvk_mser_get_min_area
import cvk.cvk_mser_get_min_diversity
import cvk.cvk_mser_get_min_margin
import cvk.cvk_mser_get_pass2_only
import cvk.cvk_mser_read
import cvk.cvk_mser_release
import cvk.cvk_mser_save
import cvk.cvk_mser_set_area_threshold
import cvk.cvk_mser_set_delta
import cvk.cvk_mser_set_edge_blur_size
import cvk.cvk_mser_set_max_area
import cvk.cvk_mser_set_max_evolution
import cvk.cvk_mser_set_max_variation
import cvk.cvk_mser_set_min_area
import cvk.cvk_mser_set_min_diversity
import cvk.cvk_mser_set_min_margin
import cvk.cvk_mser_set_pass2_only
import cvk.cvk_mser_t
import cvk.cvk_mser_write
import cvk.cvk_orb_clear
import cvk.cvk_orb_compute
import cvk.cvk_orb_create
import cvk.cvk_orb_default_norm
import cvk.cvk_orb_descriptor_size
import cvk.cvk_orb_descriptor_type
import cvk.cvk_orb_detect
import cvk.cvk_orb_detect_and_compute
import cvk.cvk_orb_empty
import cvk.cvk_orb_get_default_name
import cvk.cvk_orb_get_edge_threshold
import cvk.cvk_orb_get_fast_threshold
import cvk.cvk_orb_get_first_level
import cvk.cvk_orb_get_max_features
import cvk.cvk_orb_get_n_levels
import cvk.cvk_orb_get_patch_size
import cvk.cvk_orb_get_scale_factor
import cvk.cvk_orb_get_score_type
import cvk.cvk_orb_get_wta_k
import cvk.cvk_orb_read
import cvk.cvk_orb_release
import cvk.cvk_orb_save
import cvk.cvk_orb_set_edge_threshold
import cvk.cvk_orb_set_fast_threshold
import cvk.cvk_orb_set_first_level
import cvk.cvk_orb_set_max_features
import cvk.cvk_orb_set_n_levels
import cvk.cvk_orb_set_patch_size
import cvk.cvk_orb_set_scale_factor
import cvk.cvk_orb_set_score_type
import cvk.cvk_orb_set_wta_k
import cvk.cvk_orb_t
import cvk.cvk_orb_write
import cvk.cvk_sift_clear
import cvk.cvk_sift_compute
import cvk.cvk_sift_create
import cvk.cvk_sift_default_norm
import cvk.cvk_sift_descriptor_size
import cvk.cvk_sift_descriptor_type
import cvk.cvk_sift_detect
import cvk.cvk_sift_detect_and_compute
import cvk.cvk_sift_empty
import cvk.cvk_sift_get_contrast_threshold
import cvk.cvk_sift_get_default_name
import cvk.cvk_sift_get_edge_threshold
import cvk.cvk_sift_get_n_features
import cvk.cvk_sift_get_n_octave_layers
import cvk.cvk_sift_get_sigma
import cvk.cvk_sift_read
import cvk.cvk_sift_release
import cvk.cvk_sift_save
import cvk.cvk_sift_set_contrast_threshold
import cvk.cvk_sift_set_edge_threshold
import cvk.cvk_sift_set_n_features
import cvk.cvk_sift_set_n_octave_layers
import cvk.cvk_sift_set_sigma
import cvk.cvk_sift_t
import cvk.cvk_sift_write
import cvk.cvk_simple_blob_detector_clear
import cvk.cvk_simple_blob_detector_compute
import cvk.cvk_simple_blob_detector_create
import cvk.cvk_simple_blob_detector_default_norm
import cvk.cvk_simple_blob_detector_descriptor_size
import cvk.cvk_simple_blob_detector_descriptor_type
import cvk.cvk_simple_blob_detector_detect
import cvk.cvk_simple_blob_detector_detect_and_compute
import cvk.cvk_simple_blob_detector_empty
import cvk.cvk_simple_blob_detector_get_blob_contours
import cvk.cvk_simple_blob_detector_get_default_name
import cvk.cvk_simple_blob_detector_get_params
import cvk.cvk_simple_blob_detector_read
import cvk.cvk_simple_blob_detector_release
import cvk.cvk_simple_blob_detector_save
import cvk.cvk_simple_blob_detector_set_params
import cvk.cvk_simple_blob_detector_t
import cvk.cvk_simple_blob_detector_write
import kotlinx.cinterop.CPointed
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CValuesRef
import kotlinx.cinterop.CPointerVarOf
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.reinterpret
import platform.posix.size_tVar
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value
import kotlin.concurrent.Volatile

// =========================================================================
// helpers
// =========================================================================

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

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

/** Expanded scalar layout of [SimpleBlobDetector.Params] for the C ABI. */
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

// =========================================================================
// Shared cinterop-backed Feature2D implementation
// =========================================================================

/**
 * cinterop-backed [Feature2D] base. Concrete detectors supply their typed
 * per-class C functions; the common surface (detect/compute/
 * detectAndCompute, descriptors, Algorithm quartet) is implemented here.
 */
internal abstract class NativeFeature2D<H : CPointed> protected constructor(
    @Volatile protected var raw: CPointer<H>?,
    private val detectFn: (CPointer<H>?, CPointer<cvk_mat_t>?, CPointer<cvk_mat_t>?) -> CPointer<cvk_mat_t>?,
    private val computeFn: (CPointer<H>?, CPointer<cvk_mat_t>?, CPointer<cvk_mat_t>?, CValuesRef<CPointerVarOf<CPointer<cvk_mat_t>>>?, CValuesRef<CPointerVarOf<CPointer<cvk_mat_t>>>?) -> Int,
    private val detectAndComputeFn: (CPointer<H>?, CPointer<cvk_mat_t>?, CPointer<cvk_mat_t>?, CValuesRef<CPointerVarOf<CPointer<cvk_mat_t>>>?) -> CPointer<cvk_mat_t>?,
    private val descriptorSizeFn: (CPointer<H>?) -> Int,
    private val descriptorTypeFn: (CPointer<H>?) -> Int,
    private val defaultNormFn: (CPointer<H>?) -> Int,
    private val writeFn: (CPointer<H>?, String) -> Unit,
    private val readFn: (CPointer<H>?, String) -> Unit,
    private val clearFn: (CPointer<H>?) -> Unit,
    private val emptyFn: (CPointer<H>?) -> Int,
    private val saveFn: (CPointer<H>?, String) -> Unit,
    private val getDefaultNameFn: (CPointer<H>?) -> String?,
    private val releaseFn: (CPointer<H>?) -> Unit,
) : Feature2D {

    protected fun check(): CPointer<H> =
        raw ?: throw IllegalStateException("Feature2D is closed")

    /** Raw handle viewed through the generic cvk_feature2d_t prefix. */
    internal fun feature2dHandle(): CPointer<cvk_feature2d_t> = check().reinterpret()

    override fun detect(image: Mat, mask: Mat?): List<KeyPoint> {
        val out = detectFn(check(), image.nativeHandle(), mask?.nativeHandle())
            ?: throw OpenCVException("detect", lastNativeError())
        return keypointsOf(nativeMat(out, "detect"))
    }

    override fun detectAndCompute(image: Mat, mask: Mat?): Pair<List<KeyPoint>, Mat> = memScoped {
        val descriptors = alloc<CPointerVar<cvk_mat_t>>()
        val keypoints = detectAndComputeFn(
            check(), image.nativeHandle(), mask?.nativeHandle(), descriptors.ptr,
        ) ?: throw OpenCVException("detectAndCompute", lastNativeError())
        keypointsOf(nativeMat(keypoints, "detectAndCompute")) to
            nativeMat(descriptors.value, "detectAndCompute")
    }

    override fun compute(image: Mat, keypoints: List<KeyPoint>): Pair<List<KeyPoint>, Mat> = memScoped {
        val keypointsOut = alloc<CPointerVar<cvk_mat_t>>()
        val descriptorsOut = alloc<CPointerVar<cvk_mat_t>>()
        val keypointsMat = keypointsMat(keypoints)
        val ok = try {
            computeFn(
                check(), image.nativeHandle(), keypointsMat.nativeHandle(),
                keypointsOut.ptr, descriptorsOut.ptr,
            )
        } finally {
            keypointsMat.close()
        }
        if (ok == 0) throw OpenCVException("compute", lastNativeError())
        keypointsOf(nativeMat(keypointsOut.value, "compute")) to
            nativeMat(descriptorsOut.value, "compute")
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

    override fun empty(): Boolean = emptyFn(check()) != 0

    override fun save(filename: String) {
        saveFn(check(), filename)
    }

    override fun getDefaultName(): String =
        getDefaultNameFn(check()) ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val handle = raw ?: return
        raw = null
        releaseFn(handle)
    }
}

// =========================================================================
// SIFT
// =========================================================================

internal class NativeSift(raw: CPointer<cvk_sift_t>?) :
    NativeFeature2D<cvk_sift_t>(
        raw,
        detectFn = { h, img, mask -> cvk_sift_detect(h, img, mask) },
        computeFn = { h, img, kp, kpOut, descOut -> cvk_sift_compute(h, img, kp, kpOut, descOut) },
        detectAndComputeFn = { h, img, mask, descOut -> cvk_sift_detect_and_compute(h, img, mask, descOut) },
        descriptorSizeFn = { cvk_sift_descriptor_size(it) },
        descriptorTypeFn = { cvk_sift_descriptor_type(it) },
        defaultNormFn = { cvk_sift_default_norm(it) },
        writeFn = { h, name -> cvk_sift_write(h, name) },
        readFn = { h, name -> cvk_sift_read(h, name) },
        clearFn = { cvk_sift_clear(it) },
        emptyFn = { cvk_sift_empty(it) },
        saveFn = { h, name -> cvk_sift_save(h, name) },
        getDefaultNameFn = { cvk_sift_get_default_name(it)?.toKString() },
        releaseFn = { cvk_sift_release(it) },
    ),
    SIFT {

    private fun checkSift(): CPointer<cvk_sift_t> = check()

    override var nFeatures: Int
        get() = cvk_sift_get_n_features(checkSift())
        set(value) {
            cvk_sift_set_n_features(checkSift(), value)
        }

    override var nOctaveLayers: Int
        get() = cvk_sift_get_n_octave_layers(checkSift())
        set(value) {
            cvk_sift_set_n_octave_layers(checkSift(), value)
        }

    override var contrastThreshold: Double
        get() = cvk_sift_get_contrast_threshold(checkSift())
        set(value) {
            cvk_sift_set_contrast_threshold(checkSift(), value)
        }

    override var edgeThreshold: Double
        get() = cvk_sift_get_edge_threshold(checkSift())
        set(value) {
            cvk_sift_set_edge_threshold(checkSift(), value)
        }

    override var sigma: Double
        get() = cvk_sift_get_sigma(checkSift())
        set(value) {
            cvk_sift_set_sigma(checkSift(), value)
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
): SIFT = NativeSift(
    cvk_sift_create(
        nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma,
        descriptorType, if (enablePreciseUpscale) 1 else 0,
    ) ?: throw OpenCVException("siftCreate", lastNativeError()),
)

// =========================================================================
// ORB
// =========================================================================

internal class NativeOrb(raw: CPointer<cvk_orb_t>?) :
    NativeFeature2D<cvk_orb_t>(
        raw,
        detectFn = { h, img, mask -> cvk_orb_detect(h, img, mask) },
        computeFn = { h, img, kp, kpOut, descOut -> cvk_orb_compute(h, img, kp, kpOut, descOut) },
        detectAndComputeFn = { h, img, mask, descOut -> cvk_orb_detect_and_compute(h, img, mask, descOut) },
        descriptorSizeFn = { cvk_orb_descriptor_size(it) },
        descriptorTypeFn = { cvk_orb_descriptor_type(it) },
        defaultNormFn = { cvk_orb_default_norm(it) },
        writeFn = { h, name -> cvk_orb_write(h, name) },
        readFn = { h, name -> cvk_orb_read(h, name) },
        clearFn = { cvk_orb_clear(it) },
        emptyFn = { cvk_orb_empty(it) },
        saveFn = { h, name -> cvk_orb_save(h, name) },
        getDefaultNameFn = { cvk_orb_get_default_name(it)?.toKString() },
        releaseFn = { cvk_orb_release(it) },
    ),
    ORB {

    private fun checkOrb(): CPointer<cvk_orb_t> = check()

    override var maxFeatures: Int
        get() = cvk_orb_get_max_features(checkOrb())
        set(value) {
            cvk_orb_set_max_features(checkOrb(), value)
        }

    override var scaleFactor: Double
        get() = cvk_orb_get_scale_factor(checkOrb())
        set(value) {
            cvk_orb_set_scale_factor(checkOrb(), value)
        }

    override var nLevels: Int
        get() = cvk_orb_get_n_levels(checkOrb())
        set(value) {
            cvk_orb_set_n_levels(checkOrb(), value)
        }

    override var edgeThreshold: Int
        get() = cvk_orb_get_edge_threshold(checkOrb())
        set(value) {
            cvk_orb_set_edge_threshold(checkOrb(), value)
        }

    override var firstLevel: Int
        get() = cvk_orb_get_first_level(checkOrb())
        set(value) {
            cvk_orb_set_first_level(checkOrb(), value)
        }

    override var wtaK: Int
        get() = cvk_orb_get_wta_k(checkOrb())
        set(value) {
            cvk_orb_set_wta_k(checkOrb(), value)
        }

    override var scoreType: Int
        get() = cvk_orb_get_score_type(checkOrb())
        set(value) {
            cvk_orb_set_score_type(checkOrb(), value)
        }

    override var patchSize: Int
        get() = cvk_orb_get_patch_size(checkOrb())
        set(value) {
            cvk_orb_set_patch_size(checkOrb(), value)
        }

    override var fastThreshold: Int
        get() = cvk_orb_get_fast_threshold(checkOrb())
        set(value) {
            cvk_orb_set_fast_threshold(checkOrb(), value)
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
): ORB = NativeOrb(
    cvk_orb_create(
        nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wtaK,
        scoreType, patchSize, fastThreshold,
    ) ?: throw OpenCVException("orbCreate", lastNativeError()),
)

// =========================================================================
// MSER
// =========================================================================

internal class NativeMser(raw: CPointer<cvk_mser_t>?) :
    NativeFeature2D<cvk_mser_t>(
        raw,
        detectFn = { h, img, mask -> cvk_mser_detect(h, img, mask) },
        computeFn = { h, img, kp, kpOut, descOut -> cvk_mser_compute(h, img, kp, kpOut, descOut) },
        detectAndComputeFn = { h, img, mask, descOut -> cvk_mser_detect_and_compute(h, img, mask, descOut) },
        descriptorSizeFn = { cvk_mser_descriptor_size(it) },
        descriptorTypeFn = { cvk_mser_descriptor_type(it) },
        defaultNormFn = { cvk_mser_default_norm(it) },
        writeFn = { h, name -> cvk_mser_write(h, name) },
        readFn = { h, name -> cvk_mser_read(h, name) },
        clearFn = { cvk_mser_clear(it) },
        emptyFn = { cvk_mser_empty(it) },
        saveFn = { h, name -> cvk_mser_save(h, name) },
        getDefaultNameFn = { cvk_mser_get_default_name(it)?.toKString() },
        releaseFn = { cvk_mser_release(it) },
    ),
    MSER {

    private fun checkMser(): CPointer<cvk_mser_t> = check()

    override var delta: Int
        get() = cvk_mser_get_delta(checkMser())
        set(value) {
            cvk_mser_set_delta(checkMser(), value)
        }

    override var minArea: Int
        get() = cvk_mser_get_min_area(checkMser())
        set(value) {
            cvk_mser_set_min_area(checkMser(), value)
        }

    override var maxArea: Int
        get() = cvk_mser_get_max_area(checkMser())
        set(value) {
            cvk_mser_set_max_area(checkMser(), value)
        }

    override var maxVariation: Double
        get() = cvk_mser_get_max_variation(checkMser())
        set(value) {
            cvk_mser_set_max_variation(checkMser(), value)
        }

    override var minDiversity: Double
        get() = cvk_mser_get_min_diversity(checkMser())
        set(value) {
            cvk_mser_set_min_diversity(checkMser(), value)
        }

    override var maxEvolution: Int
        get() = cvk_mser_get_max_evolution(checkMser())
        set(value) {
            cvk_mser_set_max_evolution(checkMser(), value)
        }

    override var areaThreshold: Double
        get() = cvk_mser_get_area_threshold(checkMser())
        set(value) {
            cvk_mser_set_area_threshold(checkMser(), value)
        }

    override var minMargin: Double
        get() = cvk_mser_get_min_margin(checkMser())
        set(value) {
            cvk_mser_set_min_margin(checkMser(), value)
        }

    override var edgeBlurSize: Int
        get() = cvk_mser_get_edge_blur_size(checkMser())
        set(value) {
            cvk_mser_set_edge_blur_size(checkMser(), value)
        }

    override var pass2Only: Boolean
        get() = cvk_mser_get_pass2_only(checkMser()) != 0
        set(value) {
            cvk_mser_set_pass2_only(checkMser(), if (value) 1 else 0)
        }

    override fun detectRegions(image: Mat): Pair<List<List<Point>>, Mat> = memScoped {
        val bboxes = alloc<CPointerVar<cvk_mat_t>>()
        val length = alloc<size_tVar>()
        val buffer = cvk_mser_detect_regions(checkMser(), image.nativeHandle(), bboxes.ptr, length.ptr)
            ?: throw OpenCVException("detectRegions", lastNativeError())
        val regions = try {
            ContourCodec.decode(buffer.readBytes(length.value.toInt()))
        } finally {
            cvk_free_buffer(buffer)
        }
        regions to nativeMat(bboxes.value, "detectRegions")
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
): MSER = NativeMser(
    cvk_mser_create(
        delta, minArea, maxArea, maxVariation, minDiversity, maxEvolution,
        areaThreshold, minMargin, edgeBlurSize,
    ) ?: throw OpenCVException("mserCreate", lastNativeError()),
)

// =========================================================================
// FastFeatureDetector
// =========================================================================

internal class NativeFastFeatureDetector(
    raw: CPointer<cvk_fast_feature_detector_t>?,
) : NativeFeature2D<cvk_fast_feature_detector_t>(
        raw,
        detectFn = { h, img, mask -> cvk_fast_feature_detector_detect(h, img, mask) },
        computeFn = { h, img, kp, kpOut, descOut ->
            cvk_fast_feature_detector_compute(h, img, kp, kpOut, descOut)
        },
        detectAndComputeFn = { h, img, mask, descOut ->
            cvk_fast_feature_detector_detect_and_compute(h, img, mask, descOut)
        },
        descriptorSizeFn = { cvk_fast_feature_detector_descriptor_size(it) },
        descriptorTypeFn = { cvk_fast_feature_detector_descriptor_type(it) },
        defaultNormFn = { cvk_fast_feature_detector_default_norm(it) },
        writeFn = { h, name -> cvk_fast_feature_detector_write(h, name) },
        readFn = { h, name -> cvk_fast_feature_detector_read(h, name) },
        clearFn = { cvk_fast_feature_detector_clear(it) },
        emptyFn = { cvk_fast_feature_detector_empty(it) },
        saveFn = { h, name -> cvk_fast_feature_detector_save(h, name) },
        getDefaultNameFn = { cvk_fast_feature_detector_get_default_name(it)?.toKString() },
        releaseFn = { cvk_fast_feature_detector_release(it) },
    ),
    FastFeatureDetector {

    private fun checkFast(): CPointer<cvk_fast_feature_detector_t> = check()

    override var threshold: Int
        get() = cvk_fast_feature_detector_get_threshold(checkFast())
        set(value) {
            cvk_fast_feature_detector_set_threshold(checkFast(), value)
        }

    override var nonmaxSuppression: Boolean
        get() = cvk_fast_feature_detector_get_nonmax_suppression(checkFast()) != 0
        set(value) {
            cvk_fast_feature_detector_set_nonmax_suppression(checkFast(), if (value) 1 else 0)
        }

    override var type: Int
        get() = cvk_fast_feature_detector_get_type(checkFast())
        set(value) {
            cvk_fast_feature_detector_set_type(checkFast(), value)
        }
}

actual fun fastCreate(
    threshold: Int,
    nonmaxSuppression: Boolean,
    type: Int,
): FastFeatureDetector = NativeFastFeatureDetector(
    cvk_fast_feature_detector_create(threshold, if (nonmaxSuppression) 1 else 0, type)
        ?: throw OpenCVException("fastCreate", lastNativeError()),
)

// =========================================================================
// GFTTDetector
// =========================================================================

internal class NativeGfttDetector(
    raw: CPointer<cvk_gftt_detector_t>?,
) : NativeFeature2D<cvk_gftt_detector_t>(
        raw,
        detectFn = { h, img, mask -> cvk_gftt_detector_detect(h, img, mask) },
        computeFn = { h, img, kp, kpOut, descOut ->
            cvk_gftt_detector_compute(h, img, kp, kpOut, descOut)
        },
        detectAndComputeFn = { h, img, mask, descOut ->
            cvk_gftt_detector_detect_and_compute(h, img, mask, descOut)
        },
        descriptorSizeFn = { cvk_gftt_detector_descriptor_size(it) },
        descriptorTypeFn = { cvk_gftt_detector_descriptor_type(it) },
        defaultNormFn = { cvk_gftt_detector_default_norm(it) },
        writeFn = { h, name -> cvk_gftt_detector_write(h, name) },
        readFn = { h, name -> cvk_gftt_detector_read(h, name) },
        clearFn = { cvk_gftt_detector_clear(it) },
        emptyFn = { cvk_gftt_detector_empty(it) },
        saveFn = { h, name -> cvk_gftt_detector_save(h, name) },
        getDefaultNameFn = { cvk_gftt_detector_get_default_name(it)?.toKString() },
        releaseFn = { cvk_gftt_detector_release(it) },
    ),
    GFTTDetector {

    private fun checkGftt(): CPointer<cvk_gftt_detector_t> = check()

    override var maxFeatures: Int
        get() = cvk_gftt_detector_get_max_features(checkGftt())
        set(value) {
            cvk_gftt_detector_set_max_features(checkGftt(), value)
        }

    override var qualityLevel: Double
        get() = cvk_gftt_detector_get_quality_level(checkGftt())
        set(value) {
            cvk_gftt_detector_set_quality_level(checkGftt(), value)
        }

    override var minDistance: Double
        get() = cvk_gftt_detector_get_min_distance(checkGftt())
        set(value) {
            cvk_gftt_detector_set_min_distance(checkGftt(), value)
        }

    override var blockSize: Int
        get() = cvk_gftt_detector_get_block_size(checkGftt())
        set(value) {
            cvk_gftt_detector_set_block_size(checkGftt(), value)
        }

    override var gradientSize: Int
        get() = cvk_gftt_detector_get_gradient_size(checkGftt())
        set(value) {
            cvk_gftt_detector_set_gradient_size(checkGftt(), value)
        }

    override var harrisDetector: Boolean
        get() = cvk_gftt_detector_get_harris_detector(checkGftt()) != 0
        set(value) {
            cvk_gftt_detector_set_harris_detector(checkGftt(), if (value) 1 else 0)
        }

    override var k: Double
        get() = cvk_gftt_detector_get_k(checkGftt())
        set(value) {
            cvk_gftt_detector_set_k(checkGftt(), value)
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
): GFTTDetector = NativeGfttDetector(
    cvk_gftt_detector_create(
        maxCorners, qualityLevel, minDistance, blockSize, gradientSize,
        if (useHarrisDetector) 1 else 0, k,
    ) ?: throw OpenCVException("gfttCreate", lastNativeError()),
)

// =========================================================================
// SimpleBlobDetector
// =========================================================================

internal class NativeSimpleBlobDetector(
    raw: CPointer<cvk_simple_blob_detector_t>?,
) : NativeFeature2D<cvk_simple_blob_detector_t>(
        raw,
        detectFn = { h, img, mask -> cvk_simple_blob_detector_detect(h, img, mask) },
        computeFn = { h, img, kp, kpOut, descOut ->
            cvk_simple_blob_detector_compute(h, img, kp, kpOut, descOut)
        },
        detectAndComputeFn = { h, img, mask, descOut ->
            cvk_simple_blob_detector_detect_and_compute(h, img, mask, descOut)
        },
        descriptorSizeFn = { cvk_simple_blob_detector_descriptor_size(it) },
        descriptorTypeFn = { cvk_simple_blob_detector_descriptor_type(it) },
        defaultNormFn = { cvk_simple_blob_detector_default_norm(it) },
        writeFn = { h, name -> cvk_simple_blob_detector_write(h, name) },
        readFn = { h, name -> cvk_simple_blob_detector_read(h, name) },
        clearFn = { cvk_simple_blob_detector_clear(it) },
        emptyFn = { cvk_simple_blob_detector_empty(it) },
        saveFn = { h, name -> cvk_simple_blob_detector_save(h, name) },
        getDefaultNameFn = { cvk_simple_blob_detector_get_default_name(it)?.toKString() },
        releaseFn = { cvk_simple_blob_detector_release(it) },
    ),
    SimpleBlobDetector {

    private fun checkBlob(): CPointer<cvk_simple_blob_detector_t> = check()

    override fun setParams(params: SimpleBlobDetector.Params) {
        val a = blobParamsArgs(params)
        cvk_simple_blob_detector_set_params(
            checkBlob(), a.thresholdStep, a.minThreshold, a.maxThreshold,
            a.minRepeatability, a.minDistBetweenBlobs, a.filterByColor, a.blobColor,
            a.filterByArea, a.minArea, a.maxArea, a.filterByCircularity,
            a.minCircularity, a.maxCircularity, a.filterByInertia, a.minInertiaRatio,
            a.maxInertiaRatio, a.filterByConvexity, a.minConvexity, a.maxConvexity,
            a.collectContours,
        )
    }

    override fun getParams(): SimpleBlobDetector.Params = memScoped {
        val out = allocArray<DoubleVar>(20)
        if (cvk_simple_blob_detector_get_params(checkBlob(), out) == 0) {
            throw OpenCVException("getParams", lastNativeError())
        }
        val values = DoubleArray(20) { out[it] }
        blobParamsOf(values)
    }

    override fun getBlobContours(): List<List<Point>> = memScoped {
        val length = alloc<size_tVar>()
        val buffer = cvk_simple_blob_detector_get_blob_contours(checkBlob(), length.ptr)
            ?: throw OpenCVException("getBlobContours", lastNativeError())
        try {
            ContourCodec.decode(buffer.readBytes(length.value.toInt()))
        } finally {
            cvk_free_buffer(buffer)
        }
    }
}

actual fun simpleBlobDetectorCreate(params: SimpleBlobDetector.Params): SimpleBlobDetector {
    val a = blobParamsArgs(params)
    return NativeSimpleBlobDetector(
        cvk_simple_blob_detector_create(
            a.thresholdStep, a.minThreshold, a.maxThreshold, a.minRepeatability,
            a.minDistBetweenBlobs, a.filterByColor, a.blobColor, a.filterByArea,
            a.minArea, a.maxArea, a.filterByCircularity, a.minCircularity,
            a.maxCircularity, a.filterByInertia, a.minInertiaRatio, a.maxInertiaRatio,
            a.filterByConvexity, a.minConvexity, a.maxConvexity, a.collectContours,
        ) ?: throw OpenCVException("simpleBlobDetectorCreate", lastNativeError()),
    )
}

// =========================================================================
// AffineFeature
// =========================================================================

internal class NativeAffine(raw: CPointer<cvk_affine_t>?) :
    NativeFeature2D<cvk_affine_t>(
        raw,
        detectFn = { h, img, mask -> cvk_affine_detect(h, img, mask) },
        computeFn = { h, img, kp, kpOut, descOut -> cvk_affine_compute(h, img, kp, kpOut, descOut) },
        detectAndComputeFn = { h, img, mask, descOut -> cvk_affine_detect_and_compute(h, img, mask, descOut) },
        descriptorSizeFn = { cvk_affine_descriptor_size(it) },
        descriptorTypeFn = { cvk_affine_descriptor_type(it) },
        defaultNormFn = { cvk_affine_default_norm(it) },
        writeFn = { h, name -> cvk_affine_write(h, name) },
        readFn = { h, name -> cvk_affine_read(h, name) },
        clearFn = { cvk_affine_clear(it) },
        emptyFn = { cvk_affine_empty(it) },
        saveFn = { h, name -> cvk_affine_save(h, name) },
        getDefaultNameFn = { cvk_affine_get_default_name(it)?.toKString() },
        releaseFn = { cvk_affine_release(it) },
    ),
    AffineFeature {

    private fun checkAffine(): CPointer<cvk_affine_t> = check()

    override fun setViewParams(tilts: FloatArray, rolls: FloatArray) {
        val tiltsMat = mat(tilts.size, 1, cvMakeType(CV_32F, 1))
        val rollsMat = mat(rolls.size, 1, cvMakeType(CV_32F, 1))
        try {
            MatOfFloat(tiltsMat).fromArray(tilts)
            MatOfFloat(rollsMat).fromArray(rolls)
            if (cvk_affine_set_view_params(checkAffine(), tiltsMat.nativeHandle(), rollsMat.nativeHandle()) == 0) {
                throw OpenCVException("setViewParams", lastNativeError())
            }
        } finally {
            tiltsMat.close()
            rollsMat.close()
        }
    }

    override fun getViewParams(): Pair<FloatArray, FloatArray> = memScoped {
        val tilts = alloc<CPointerVar<cvk_mat_t>>()
        val rolls = alloc<CPointerVar<cvk_mat_t>>()
        if (cvk_affine_get_view_params(checkAffine(), tilts.ptr, rolls.ptr) == 0) {
            throw OpenCVException("getViewParams", lastNativeError())
        }
        floatsOf(nativeMat(tilts.value, "getViewParams")) to
            floatsOf(nativeMat(rolls.value, "getViewParams"))
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
    val raw = (backend as? NativeFeature2D<*>)?.feature2dHandle()
        ?: throw IllegalArgumentException("backend belongs to another platform backend")
    return NativeAffine(
        cvk_affine_create(raw, maxTilt, minTilt, tiltStep, rotateStepBase)
            ?: throw OpenCVException("affineFeatureCreate", lastNativeError()),
    )
}

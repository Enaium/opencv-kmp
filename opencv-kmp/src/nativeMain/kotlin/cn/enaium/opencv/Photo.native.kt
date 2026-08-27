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

import cvk.cvk_align_mtb_calculate_shift
import cvk.cvk_mat_t
import cvk.cvk_align_mtb_clear
import cvk.cvk_align_mtb_compute_bitmaps
import cvk.cvk_align_mtb_create
import cvk.cvk_align_mtb_empty
import cvk.cvk_align_mtb_get_cut
import cvk.cvk_align_mtb_get_default_name
import cvk.cvk_align_mtb_get_exclude_range
import cvk.cvk_align_mtb_get_max_bits
import cvk.cvk_align_mtb_process
import cvk.cvk_align_mtb_process_times
import cvk.cvk_align_mtb_release
import cvk.cvk_align_mtb_save
import cvk.cvk_align_mtb_set_cut
import cvk.cvk_align_mtb_set_exclude_range
import cvk.cvk_align_mtb_set_max_bits
import cvk.cvk_align_mtb_shift_mat
import cvk.cvk_align_mtb_t
import cvk.cvk_calibrate_debevec_clear
import cvk.cvk_calibrate_debevec_create
import cvk.cvk_calibrate_debevec_empty
import cvk.cvk_calibrate_debevec_get_default_name
import cvk.cvk_calibrate_debevec_get_lambda
import cvk.cvk_calibrate_debevec_get_random
import cvk.cvk_calibrate_debevec_get_samples
import cvk.cvk_calibrate_debevec_process
import cvk.cvk_calibrate_debevec_release
import cvk.cvk_calibrate_debevec_save
import cvk.cvk_calibrate_debevec_set_lambda
import cvk.cvk_calibrate_debevec_set_random
import cvk.cvk_calibrate_debevec_set_samples
import cvk.cvk_calibrate_debevec_t
import cvk.cvk_calibrate_robertson_clear
import cvk.cvk_calibrate_robertson_create
import cvk.cvk_calibrate_robertson_empty
import cvk.cvk_calibrate_robertson_get_default_name
import cvk.cvk_calibrate_robertson_get_max_iter
import cvk.cvk_calibrate_robertson_get_radiance
import cvk.cvk_calibrate_robertson_get_threshold
import cvk.cvk_calibrate_robertson_process
import cvk.cvk_calibrate_robertson_release
import cvk.cvk_calibrate_robertson_save
import cvk.cvk_calibrate_robertson_set_max_iter
import cvk.cvk_calibrate_robertson_set_threshold
import cvk.cvk_calibrate_robertson_t
import cvk.cvk_color_change
import cvk.cvk_color_correction_model_compute
import cvk.cvk_color_correction_model_correct_image
import cvk.cvk_color_correction_model_create
import cvk.cvk_color_correction_model_create_empty
import cvk.cvk_color_correction_model_get_color_correction_matrix
import cvk.cvk_color_correction_model_get_loss
import cvk.cvk_color_correction_model_get_mask
import cvk.cvk_color_correction_model_get_ref_linear_rgb
import cvk.cvk_color_correction_model_get_src_linear_rgb
import cvk.cvk_color_correction_model_get_weights
import cvk.cvk_color_correction_model_release
import cvk.cvk_color_correction_model_set_epsilon
import cvk.cvk_color_correction_model_set_linearization_degree
import cvk.cvk_color_correction_model_set_linearization_gamma
import cvk.cvk_color_correction_model_set_max_count
import cvk.cvk_color_correction_model_set_rgb
import cvk.cvk_color_correction_model_set_saturated_threshold
import cvk.cvk_color_correction_model_set_weight_coeff
import cvk.cvk_color_correction_model_set_weights_list
import cvk.cvk_color_correction_model_t
import cvk.cvk_correct_chromatic_aberration
import cvk.cvk_decolor
import cvk.cvk_denoise_tvl1
import cvk.cvk_detail_enhance
import cvk.cvk_edge_preserving_filter
import cvk.cvk_fast_nl_means_denoising
import cvk.cvk_fast_nl_means_denoising_colored
import cvk.cvk_fast_nl_means_denoising_colored_multi
import cvk.cvk_fast_nl_means_denoising_h
import cvk.cvk_fast_nl_means_denoising_multi
import cvk.cvk_fast_nl_means_denoising_multi_h
import cvk.cvk_gamma_correction
import cvk.cvk_illumination_change
import cvk.cvk_inpaint
import cvk.cvk_intelligent_scissors_mb_apply_image
import cvk.cvk_intelligent_scissors_mb_apply_image_features
import cvk.cvk_intelligent_scissors_mb_build_map
import cvk.cvk_intelligent_scissors_mb_create
import cvk.cvk_intelligent_scissors_mb_get_contour
import cvk.cvk_intelligent_scissors_mb_release
import cvk.cvk_intelligent_scissors_mb_set_edge_feature_canny_parameters
import cvk.cvk_intelligent_scissors_mb_set_edge_feature_zero_crossing_parameters
import cvk.cvk_intelligent_scissors_mb_set_gradient_magnitude_max_limit
import cvk.cvk_intelligent_scissors_mb_set_weights
import cvk.cvk_intelligent_scissors_mb_t
import cvk.cvk_last_error
import cvk.cvk_merge_debevec_clear
import cvk.cvk_merge_debevec_create
import cvk.cvk_merge_debevec_empty
import cvk.cvk_merge_debevec_get_default_name
import cvk.cvk_merge_debevec_process
import cvk.cvk_merge_debevec_process_response
import cvk.cvk_merge_debevec_release
import cvk.cvk_merge_debevec_save
import cvk.cvk_merge_debevec_t
import cvk.cvk_merge_mertens_clear
import cvk.cvk_merge_mertens_create
import cvk.cvk_merge_mertens_empty
import cvk.cvk_merge_mertens_get_contrast_weight
import cvk.cvk_merge_mertens_get_default_name
import cvk.cvk_merge_mertens_get_exposure_weight
import cvk.cvk_merge_mertens_get_saturation_weight
import cvk.cvk_merge_mertens_process
import cvk.cvk_merge_mertens_process_response
import cvk.cvk_merge_mertens_release
import cvk.cvk_merge_mertens_save
import cvk.cvk_merge_mertens_set_contrast_weight
import cvk.cvk_merge_mertens_set_exposure_weight
import cvk.cvk_merge_mertens_set_saturation_weight
import cvk.cvk_merge_mertens_t
import cvk.cvk_merge_robertson_clear
import cvk.cvk_merge_robertson_create
import cvk.cvk_merge_robertson_empty
import cvk.cvk_merge_robertson_get_default_name
import cvk.cvk_merge_robertson_process
import cvk.cvk_merge_robertson_process_response
import cvk.cvk_merge_robertson_release
import cvk.cvk_merge_robertson_save
import cvk.cvk_merge_robertson_t
import cvk.cvk_pencil_sketch
import cvk.cvk_seamless_clone
import cvk.cvk_stylization
import cvk.cvk_texture_flattening
import cvk.cvk_tonemap_clear
import cvk.cvk_tonemap_create
import cvk.cvk_tonemap_drago_clear
import cvk.cvk_tonemap_drago_create
import cvk.cvk_tonemap_drago_empty
import cvk.cvk_tonemap_drago_get_bias
import cvk.cvk_tonemap_drago_get_default_name
import cvk.cvk_tonemap_drago_get_gamma
import cvk.cvk_tonemap_drago_get_saturation
import cvk.cvk_tonemap_drago_process
import cvk.cvk_tonemap_drago_release
import cvk.cvk_tonemap_drago_save
import cvk.cvk_tonemap_drago_set_bias
import cvk.cvk_tonemap_drago_set_gamma
import cvk.cvk_tonemap_drago_set_saturation
import cvk.cvk_tonemap_drago_t
import cvk.cvk_tonemap_empty
import cvk.cvk_tonemap_get_default_name
import cvk.cvk_tonemap_get_gamma
import cvk.cvk_tonemap_mantiuk_clear
import cvk.cvk_tonemap_mantiuk_create
import cvk.cvk_tonemap_mantiuk_empty
import cvk.cvk_tonemap_mantiuk_get_default_name
import cvk.cvk_tonemap_mantiuk_get_gamma
import cvk.cvk_tonemap_mantiuk_get_saturation
import cvk.cvk_tonemap_mantiuk_get_scale
import cvk.cvk_tonemap_mantiuk_process
import cvk.cvk_tonemap_mantiuk_release
import cvk.cvk_tonemap_mantiuk_save
import cvk.cvk_tonemap_mantiuk_set_gamma
import cvk.cvk_tonemap_mantiuk_set_saturation
import cvk.cvk_tonemap_mantiuk_set_scale
import cvk.cvk_tonemap_mantiuk_t
import cvk.cvk_tonemap_process
import cvk.cvk_tonemap_reinhard_clear
import cvk.cvk_tonemap_reinhard_create
import cvk.cvk_tonemap_reinhard_empty
import cvk.cvk_tonemap_reinhard_get_color_adaptation
import cvk.cvk_tonemap_reinhard_get_default_name
import cvk.cvk_tonemap_reinhard_get_gamma
import cvk.cvk_tonemap_reinhard_get_intensity
import cvk.cvk_tonemap_reinhard_get_light_adaptation
import cvk.cvk_tonemap_reinhard_process
import cvk.cvk_tonemap_reinhard_release
import cvk.cvk_tonemap_reinhard_save
import cvk.cvk_tonemap_reinhard_set_color_adaptation
import cvk.cvk_tonemap_reinhard_set_gamma
import cvk.cvk_tonemap_reinhard_set_intensity
import cvk.cvk_tonemap_reinhard_set_light_adaptation
import cvk.cvk_tonemap_reinhard_t
import cvk.cvk_tonemap_release
import cvk.cvk_tonemap_save
import cvk.cvk_tonemap_set_gamma
import cvk.cvk_tonemap_t
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value
import kotlin.concurrent.Volatile

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

// =========================================================================
// Photo statics
// =========================================================================

actual fun photoInpaint(src: Mat, inpaintMask: Mat, inpaintRadius: Double, flags: Int): Mat =
    nativeMat(
        cvk_inpaint(src.nativeHandle(), inpaintMask.nativeHandle(), inpaintRadius, flags),
        "photoInpaint",
    )

actual fun photoFastNlMeansDenoising(
    src: Mat,
    h: Float,
    templateWindowSize: Int,
    searchWindowSize: Int,
): Mat = nativeMat(
    cvk_fast_nl_means_denoising(src.nativeHandle(), h, templateWindowSize, searchWindowSize),
    "photoFastNlMeansDenoising",
)

actual fun photoFastNlMeansDenoising(
    src: Mat,
    h: MatOfFloat,
    templateWindowSize: Int,
    searchWindowSize: Int,
    normType: Int,
): Mat = nativeMat(
    cvk_fast_nl_means_denoising_h(
        src.nativeHandle(), h.mat.nativeHandle(), templateWindowSize, searchWindowSize, normType,
    ),
    "photoFastNlMeansDenoising(h)",
)

actual fun photoFastNlMeansDenoisingColored(
    src: Mat,
    h: Float,
    hColor: Float,
    templateWindowSize: Int,
    searchWindowSize: Int,
): Mat = nativeMat(
    cvk_fast_nl_means_denoising_colored(
        src.nativeHandle(), h, hColor, templateWindowSize, searchWindowSize,
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
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
        src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
        nativeMat(
            cvk_fast_nl_means_denoising_multi(
                handles, src.size, imgToDenoiseIndex, temporalWindowSize, h,
                templateWindowSize, searchWindowSize,
            ),
            "photoFastNlMeansDenoisingMulti",
        )
    }
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
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
        src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
        nativeMat(
            cvk_fast_nl_means_denoising_multi_h(
                handles, src.size, imgToDenoiseIndex, temporalWindowSize,
                h.mat.nativeHandle(), templateWindowSize, searchWindowSize, normType,
            ),
            "photoFastNlMeansDenoisingMulti(h)",
        )
    }
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
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
        src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
        nativeMat(
            cvk_fast_nl_means_denoising_colored_multi(
                handles, src.size, imgToDenoiseIndex, temporalWindowSize, h, hColor,
                templateWindowSize, searchWindowSize,
            ),
            "photoFastNlMeansDenoisingColoredMulti",
        )
    }
}

actual fun photoDenoiseTvl1(observations: List<Mat>, lambda: Double, niters: Int): Mat {
    require(observations.isNotEmpty()) { "photoDenoiseTvl1 needs at least one observation" }
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(observations.size)
        observations.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
        nativeMat(
            cvk_denoise_tvl1(handles, observations.size, lambda, niters),
            "photoDenoiseTvl1",
        )
    }
}

actual fun photoDecolor(src: Mat): Pair<Mat, Mat> = memScoped {
    val grayscale = alloc<CPointerVar<cvk_mat_t>>()
    val colorBoost = alloc<CPointerVar<cvk_mat_t>>()
    cvk_decolor(src.nativeHandle(), grayscale.ptr, colorBoost.ptr)
    nativeMat(grayscale.value, "photoDecolor") to nativeMat(colorBoost.value, "photoDecolor")
}

actual fun photoSeamlessClone(src: Mat, dst: Mat, mask: Mat, p: Point, flags: Int): Mat =
    nativeMat(
        cvk_seamless_clone(
            src.nativeHandle(), dst.nativeHandle(), mask.nativeHandle(), p.x, p.y, flags,
        ),
        "photoSeamlessClone",
    )

actual fun photoColorChange(
    src: Mat,
    mask: Mat,
    redMul: Float,
    greenMul: Float,
    blueMul: Float,
): Mat = nativeMat(
    cvk_color_change(src.nativeHandle(), mask.nativeHandle(), redMul, greenMul, blueMul),
    "photoColorChange",
)

actual fun photoIlluminationChange(src: Mat, mask: Mat, alpha: Float, beta: Float): Mat =
    nativeMat(
        cvk_illumination_change(src.nativeHandle(), mask.nativeHandle(), alpha, beta),
        "photoIlluminationChange",
    )

actual fun photoTextureFlattening(
    src: Mat,
    mask: Mat,
    lowThreshold: Float,
    highThreshold: Float,
    kernelSize: Int,
): Mat = nativeMat(
    cvk_texture_flattening(
        src.nativeHandle(), mask.nativeHandle(), lowThreshold, highThreshold, kernelSize,
    ),
    "photoTextureFlattening",
)

actual fun photoEdgePreservingFilter(
    src: Mat,
    flags: Int,
    sigmaS: Float,
    sigmaR: Float,
): Mat = nativeMat(
    cvk_edge_preserving_filter(src.nativeHandle(), flags, sigmaS, sigmaR),
    "photoEdgePreservingFilter",
)

actual fun photoDetailEnhance(src: Mat, sigmaS: Float, sigmaR: Float): Mat =
    nativeMat(cvk_detail_enhance(src.nativeHandle(), sigmaS, sigmaR), "photoDetailEnhance")

actual fun photoPencilSketch(
    src: Mat,
    sigmaS: Float,
    sigmaR: Float,
    shadeFactor: Float,
): Pair<Mat, Mat> = memScoped {
    val dst1 = alloc<CPointerVar<cvk_mat_t>>()
    val dst2 = alloc<CPointerVar<cvk_mat_t>>()
    cvk_pencil_sketch(src.nativeHandle(), dst1.ptr, dst2.ptr, sigmaS, sigmaR, shadeFactor)
    nativeMat(dst1.value, "photoPencilSketch") to nativeMat(dst2.value, "photoPencilSketch")
}

actual fun photoStylization(src: Mat, sigmaS: Float, sigmaR: Float): Mat =
    nativeMat(cvk_stylization(src.nativeHandle(), sigmaS, sigmaR), "photoStylization")

actual fun photoCorrectChromaticAberration(
    input: Mat,
    coefficients: Mat,
    imageSize: Size,
    calibDegree: Int,
    bayerPattern: Int,
): Mat = nativeMat(
    cvk_correct_chromatic_aberration(
        input.nativeHandle(), coefficients.nativeHandle(),
        imageSize.width, imageSize.height, calibDegree, bayerPattern,
    ),
    "photoCorrectChromaticAberration",
)

actual fun photoGammaCorrection(src: Mat, gamma: Double): Mat =
    nativeMat(cvk_gamma_correction(src.nativeHandle(), gamma), "photoGammaCorrection")

// =========================================================================
// Tonemap family
// =========================================================================

internal class NativeTonemap(
    @Volatile private var raw: CPointer<cvk_tonemap_t>?,
) : Tonemap {

    private fun check(): CPointer<cvk_tonemap_t> =
        raw ?: throw IllegalStateException("Tonemap is closed")

    override fun process(src: Mat): Mat =
        nativeMat(cvk_tonemap_process(check(), src.nativeHandle()), "tonemap.process")

    override var gamma: Float
        get() = cvk_tonemap_get_gamma(check())
        set(value) = cvk_tonemap_set_gamma(check(), value)

    override fun clear() {
        cvk_tonemap_clear(check())
    }

    override fun empty(): Boolean = cvk_tonemap_empty(check()) != 0

    override fun save(filename: String) {
        cvk_tonemap_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_tonemap_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_tonemap_release(handle)
    }
}

actual fun createTonemap(gamma: Float): Tonemap =
    NativeTonemap(cvk_tonemap_create(gamma) ?: throw OpenCVException("createTonemap", lastNativeError()))

internal class NativeTonemapDrago(
    @Volatile private var raw: CPointer<cvk_tonemap_drago_t>?,
) : TonemapDrago {

    private fun check(): CPointer<cvk_tonemap_drago_t> =
        raw ?: throw IllegalStateException("TonemapDrago is closed")

    override fun process(src: Mat): Mat =
        nativeMat(cvk_tonemap_drago_process(check(), src.nativeHandle()), "tonemapDrago.process")

    override var gamma: Float
        get() = cvk_tonemap_drago_get_gamma(check())
        set(value) = cvk_tonemap_drago_set_gamma(check(), value)

    override var saturation: Float
        get() = cvk_tonemap_drago_get_saturation(check())
        set(value) = cvk_tonemap_drago_set_saturation(check(), value)

    override var bias: Float
        get() = cvk_tonemap_drago_get_bias(check())
        set(value) = cvk_tonemap_drago_set_bias(check(), value)

    override fun clear() {
        cvk_tonemap_drago_clear(check())
    }

    override fun empty(): Boolean = cvk_tonemap_drago_empty(check()) != 0

    override fun save(filename: String) {
        cvk_tonemap_drago_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_tonemap_drago_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_tonemap_drago_release(handle)
    }
}

actual fun createTonemapDrago(gamma: Float, saturation: Float, bias: Float): TonemapDrago =
    NativeTonemapDrago(
        cvk_tonemap_drago_create(gamma, saturation, bias)
            ?: throw OpenCVException("createTonemapDrago", lastNativeError()),
    )

internal class NativeTonemapMantiuk(
    @Volatile private var raw: CPointer<cvk_tonemap_mantiuk_t>?,
) : TonemapMantiuk {

    private fun check(): CPointer<cvk_tonemap_mantiuk_t> =
        raw ?: throw IllegalStateException("TonemapMantiuk is closed")

    override fun process(src: Mat): Mat =
        nativeMat(cvk_tonemap_mantiuk_process(check(), src.nativeHandle()), "tonemapMantiuk.process")

    override var gamma: Float
        get() = cvk_tonemap_mantiuk_get_gamma(check())
        set(value) = cvk_tonemap_mantiuk_set_gamma(check(), value)

    override var scale: Float
        get() = cvk_tonemap_mantiuk_get_scale(check())
        set(value) = cvk_tonemap_mantiuk_set_scale(check(), value)

    override var saturation: Float
        get() = cvk_tonemap_mantiuk_get_saturation(check())
        set(value) = cvk_tonemap_mantiuk_set_saturation(check(), value)

    override fun clear() {
        cvk_tonemap_mantiuk_clear(check())
    }

    override fun empty(): Boolean = cvk_tonemap_mantiuk_empty(check()) != 0

    override fun save(filename: String) {
        cvk_tonemap_mantiuk_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_tonemap_mantiuk_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_tonemap_mantiuk_release(handle)
    }
}

actual fun createTonemapMantiuk(gamma: Float, scale: Float, saturation: Float): TonemapMantiuk =
    NativeTonemapMantiuk(
        cvk_tonemap_mantiuk_create(gamma, scale, saturation)
            ?: throw OpenCVException("createTonemapMantiuk", lastNativeError()),
    )

internal class NativeTonemapReinhard(
    @Volatile private var raw: CPointer<cvk_tonemap_reinhard_t>?,
) : TonemapReinhard {

    private fun check(): CPointer<cvk_tonemap_reinhard_t> =
        raw ?: throw IllegalStateException("TonemapReinhard is closed")

    override fun process(src: Mat): Mat =
        nativeMat(cvk_tonemap_reinhard_process(check(), src.nativeHandle()), "tonemapReinhard.process")

    override var gamma: Float
        get() = cvk_tonemap_reinhard_get_gamma(check())
        set(value) = cvk_tonemap_reinhard_set_gamma(check(), value)

    override var intensity: Float
        get() = cvk_tonemap_reinhard_get_intensity(check())
        set(value) = cvk_tonemap_reinhard_set_intensity(check(), value)

    override var lightAdaptation: Float
        get() = cvk_tonemap_reinhard_get_light_adaptation(check())
        set(value) = cvk_tonemap_reinhard_set_light_adaptation(check(), value)

    override var colorAdaptation: Float
        get() = cvk_tonemap_reinhard_get_color_adaptation(check())
        set(value) = cvk_tonemap_reinhard_set_color_adaptation(check(), value)

    override fun clear() {
        cvk_tonemap_reinhard_clear(check())
    }

    override fun empty(): Boolean = cvk_tonemap_reinhard_empty(check()) != 0

    override fun save(filename: String) {
        cvk_tonemap_reinhard_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_tonemap_reinhard_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_tonemap_reinhard_release(handle)
    }
}

actual fun createTonemapReinhard(
    gamma: Float,
    intensity: Float,
    lightAdaptation: Float,
    colorAdaptation: Float,
): TonemapReinhard =
    NativeTonemapReinhard(
        cvk_tonemap_reinhard_create(gamma, intensity, lightAdaptation, colorAdaptation)
            ?: throw OpenCVException("createTonemapReinhard", lastNativeError()),
    )

// =========================================================================
// AlignMTB
// =========================================================================

internal class NativeAlignMtb(
    @Volatile private var raw: CPointer<cvk_align_mtb_t>?,
) : AlignMTB {

    private fun check(): CPointer<cvk_align_mtb_t> =
        raw ?: throw IllegalStateException("AlignMTB is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): List<Mat> {
        require(src.isNotEmpty()) { "alignMTB.process needs at least one image" }
        return memScoped {
            val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
            src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
            val out = allocArray<CPointerVar<cvk_mat_t>>(src.size)
            val count = cvk_align_mtb_process_times(
                check(), handles, src.size, times.nativeHandle(), response.nativeHandle(),
                out, src.size,
            )
            if (count <= 0) throw OpenCVException("alignMTB.process", lastNativeError())
            (0 until count).map { index -> nativeMat(out[index], "alignMTB.process") }
        }
    }

    override fun process(src: List<Mat>): List<Mat> {
        require(src.isNotEmpty()) { "alignMTB.process needs at least one image" }
        return memScoped {
            val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
            src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
            val out = allocArray<CPointerVar<cvk_mat_t>>(src.size)
            val count = cvk_align_mtb_process(check(), handles, src.size, out, src.size)
            if (count <= 0) throw OpenCVException("alignMTB.process", lastNativeError())
            (0 until count).map { index -> nativeMat(out[index], "alignMTB.process") }
        }
    }

    override fun calculateShift(img0: Mat, img1: Mat): Point = memScoped {
        val out = allocArray<IntVar>(2)
        cvk_align_mtb_calculate_shift(check(), img0.nativeHandle(), img1.nativeHandle(), out)
        Point(out[0], out[1])
    }

    override fun shiftMat(src: Mat, shift: Point): Mat =
        nativeMat(
            cvk_align_mtb_shift_mat(check(), src.nativeHandle(), shift.x, shift.y),
            "alignMTB.shiftMat",
        )

    override fun computeBitmaps(img: Mat): Pair<Mat, Mat> = memScoped {
        val tb = alloc<CPointerVar<cvk_mat_t>>()
        val eb = alloc<CPointerVar<cvk_mat_t>>()
        cvk_align_mtb_compute_bitmaps(check(), img.nativeHandle(), tb.ptr, eb.ptr)
        nativeMat(tb.value, "alignMTB.computeBitmaps") to nativeMat(eb.value, "alignMTB.computeBitmaps")
    }

    override var maxBits: Int
        get() = cvk_align_mtb_get_max_bits(check())
        set(value) = cvk_align_mtb_set_max_bits(check(), value)

    override var excludeRange: Int
        get() = cvk_align_mtb_get_exclude_range(check())
        set(value) = cvk_align_mtb_set_exclude_range(check(), value)

    override var cut: Boolean
        get() = cvk_align_mtb_get_cut(check()) != 0
        set(value) = cvk_align_mtb_set_cut(check(), if (value) 1 else 0)

    override fun clear() {
        cvk_align_mtb_clear(check())
    }

    override fun empty(): Boolean = cvk_align_mtb_empty(check()) != 0

    override fun save(filename: String) {
        cvk_align_mtb_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_align_mtb_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_align_mtb_release(handle)
    }
}

actual fun createAlignMTB(maxBits: Int, excludeRange: Int, cut: Boolean): AlignMTB =
    NativeAlignMtb(
        cvk_align_mtb_create(maxBits, excludeRange, if (cut) 1 else 0)
            ?: throw OpenCVException("createAlignMTB", lastNativeError()),
    )

// =========================================================================
// CalibrateDebevec
// =========================================================================

internal class NativeCalibrateDebevec(
    @Volatile private var raw: CPointer<cvk_calibrate_debevec_t>?,
) : CalibrateDebevec {

    private fun check(): CPointer<cvk_calibrate_debevec_t> =
        raw ?: throw IllegalStateException("CalibrateDebevec is closed")

    override fun process(src: List<Mat>, times: Mat): Mat {
        require(src.isNotEmpty()) { "calibrateDebevec.process needs at least one image" }
        return memScoped {
            val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
            src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
            nativeMat(
                cvk_calibrate_debevec_process(
                    check(), handles, src.size, times.nativeHandle(),
                ),
                "calibrateDebevec.process",
            )
        }
    }

    override var lambda: Float
        get() = cvk_calibrate_debevec_get_lambda(check())
        set(value) = cvk_calibrate_debevec_set_lambda(check(), value)

    override var samples: Int
        get() = cvk_calibrate_debevec_get_samples(check())
        set(value) = cvk_calibrate_debevec_set_samples(check(), value)

    override var random: Boolean
        get() = cvk_calibrate_debevec_get_random(check()) != 0
        set(value) = cvk_calibrate_debevec_set_random(check(), if (value) 1 else 0)

    override fun clear() {
        cvk_calibrate_debevec_clear(check())
    }

    override fun empty(): Boolean = cvk_calibrate_debevec_empty(check()) != 0

    override fun save(filename: String) {
        cvk_calibrate_debevec_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_calibrate_debevec_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_calibrate_debevec_release(handle)
    }
}

actual fun createCalibrateDebevec(samples: Int, lambda: Float, random: Boolean): CalibrateDebevec =
    NativeCalibrateDebevec(
        cvk_calibrate_debevec_create(samples, lambda, if (random) 1 else 0)
            ?: throw OpenCVException("createCalibrateDebevec", lastNativeError()),
    )

// =========================================================================
// CalibrateRobertson
// =========================================================================

internal class NativeCalibrateRobertson(
    @Volatile private var raw: CPointer<cvk_calibrate_robertson_t>?,
) : CalibrateRobertson {

    private fun check(): CPointer<cvk_calibrate_robertson_t> =
        raw ?: throw IllegalStateException("CalibrateRobertson is closed")

    override fun process(src: List<Mat>, times: Mat): Mat {
        require(src.isNotEmpty()) { "calibrateRobertson.process needs at least one image" }
        return memScoped {
            val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
            src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
            nativeMat(
                cvk_calibrate_robertson_process(
                    check(), handles, src.size, times.nativeHandle(),
                ),
                "calibrateRobertson.process",
            )
        }
    }

    override var maxIter: Int
        get() = cvk_calibrate_robertson_get_max_iter(check())
        set(value) = cvk_calibrate_robertson_set_max_iter(check(), value)

    override var threshold: Float
        get() = cvk_calibrate_robertson_get_threshold(check())
        set(value) = cvk_calibrate_robertson_set_threshold(check(), value)

    override val radiance: Mat
        get() = nativeMat(cvk_calibrate_robertson_get_radiance(check()), "calibrateRobertson.radiance")

    override fun clear() {
        cvk_calibrate_robertson_clear(check())
    }

    override fun empty(): Boolean = cvk_calibrate_robertson_empty(check()) != 0

    override fun save(filename: String) {
        cvk_calibrate_robertson_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_calibrate_robertson_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_calibrate_robertson_release(handle)
    }
}

actual fun createCalibrateRobertson(maxIter: Int, threshold: Float): CalibrateRobertson =
    NativeCalibrateRobertson(
        cvk_calibrate_robertson_create(maxIter, threshold)
            ?: throw OpenCVException("createCalibrateRobertson", lastNativeError()),
    )

// =========================================================================
// MergeDebevec
// =========================================================================

internal class NativeMergeDebevec(
    @Volatile private var raw: CPointer<cvk_merge_debevec_t>?,
) : MergeDebevec {

    private fun check(): CPointer<cvk_merge_debevec_t> =
        raw ?: throw IllegalStateException("MergeDebevec is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): Mat =
        matList(src) { handles, count ->
            nativeMat(
                cvk_merge_debevec_process_response(
                    check(), handles, count, times.nativeHandle(), response.nativeHandle(),
                ),
                "mergeDebevec.process",
            )
        }

    override fun process(src: List<Mat>, times: Mat): Mat =
        matList(src) { handles, count ->
            nativeMat(
                cvk_merge_debevec_process(check(), handles, count, times.nativeHandle()),
                "mergeDebevec.process",
            )
        }

    override fun clear() {
        cvk_merge_debevec_clear(check())
    }

    override fun empty(): Boolean = cvk_merge_debevec_empty(check()) != 0

    override fun save(filename: String) {
        cvk_merge_debevec_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_merge_debevec_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_merge_debevec_release(handle)
    }
}

/** Runs [body] with a scoped handle array over [src]. */
private inline fun <R> matList(src: List<Mat>, body: (handles: CPointer<CPointerVar<cvk_mat_t>>, count: Int) -> R): R {
    require(src.isNotEmpty()) { "process needs at least one image" }
    return memScoped {
        val handles = allocArray<CPointerVar<cvk_mat_t>>(src.size)
        src.forEachIndexed { index, mat -> handles[index] = mat.nativeHandle() }
        body(handles, src.size)
    }
}

actual fun createMergeDebevec(): MergeDebevec =
    NativeMergeDebevec(
        cvk_merge_debevec_create() ?: throw OpenCVException("createMergeDebevec", lastNativeError()),
    )

// =========================================================================
// MergeMertens
// =========================================================================

internal class NativeMergeMertens(
    @Volatile private var raw: CPointer<cvk_merge_mertens_t>?,
) : MergeMertens {

    private fun check(): CPointer<cvk_merge_mertens_t> =
        raw ?: throw IllegalStateException("MergeMertens is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): Mat =
        matList(src) { handles, count ->
            nativeMat(
                cvk_merge_mertens_process_response(
                    check(), handles, count, times.nativeHandle(), response.nativeHandle(),
                ),
                "mergeMertens.process",
            )
        }

    override fun process(src: List<Mat>): Mat =
        matList(src) { handles, count ->
            nativeMat(
                cvk_merge_mertens_process(check(), handles, count),
                "mergeMertens.process",
            )
        }

    override var contrastWeight: Float
        get() = cvk_merge_mertens_get_contrast_weight(check())
        set(value) = cvk_merge_mertens_set_contrast_weight(check(), value)

    override var saturationWeight: Float
        get() = cvk_merge_mertens_get_saturation_weight(check())
        set(value) = cvk_merge_mertens_set_saturation_weight(check(), value)

    override var exposureWeight: Float
        get() = cvk_merge_mertens_get_exposure_weight(check())
        set(value) = cvk_merge_mertens_set_exposure_weight(check(), value)

    override fun clear() {
        cvk_merge_mertens_clear(check())
    }

    override fun empty(): Boolean = cvk_merge_mertens_empty(check()) != 0

    override fun save(filename: String) {
        cvk_merge_mertens_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_merge_mertens_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_merge_mertens_release(handle)
    }
}

actual fun createMergeMertens(
    contrastWeight: Float,
    saturationWeight: Float,
    exposureWeight: Float,
): MergeMertens =
    NativeMergeMertens(
        cvk_merge_mertens_create(contrastWeight, saturationWeight, exposureWeight)
            ?: throw OpenCVException("createMergeMertens", lastNativeError()),
    )

// =========================================================================
// MergeRobertson
// =========================================================================

internal class NativeMergeRobertson(
    @Volatile private var raw: CPointer<cvk_merge_robertson_t>?,
) : MergeRobertson {

    private fun check(): CPointer<cvk_merge_robertson_t> =
        raw ?: throw IllegalStateException("MergeRobertson is closed")

    override fun process(src: List<Mat>, times: Mat, response: Mat): Mat =
        matList(src) { handles, count ->
            nativeMat(
                cvk_merge_robertson_process_response(
                    check(), handles, count, times.nativeHandle(), response.nativeHandle(),
                ),
                "mergeRobertson.process",
            )
        }

    override fun process(src: List<Mat>, times: Mat): Mat =
        matList(src) { handles, count ->
            nativeMat(
                cvk_merge_robertson_process(check(), handles, count, times.nativeHandle()),
                "mergeRobertson.process",
            )
        }

    override fun clear() {
        cvk_merge_robertson_clear(check())
    }

    override fun empty(): Boolean = cvk_merge_robertson_empty(check()) != 0

    override fun save(filename: String) {
        cvk_merge_robertson_save(check(), filename)
    }

    override fun getDefaultName(): String = cvk_merge_robertson_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_merge_robertson_release(handle)
    }
}

actual fun createMergeRobertson(): MergeRobertson =
    NativeMergeRobertson(
        cvk_merge_robertson_create() ?: throw OpenCVException("createMergeRobertson", lastNativeError()),
    )

// =========================================================================
// ColorCorrectionModel
// =========================================================================

internal class NativeColorCorrectionModel(
    @Volatile private var raw: CPointer<cvk_color_correction_model_t>?,
) : ColorCorrectionModel {

    private fun check(): CPointer<cvk_color_correction_model_t> =
        raw ?: throw IllegalStateException("ColorCorrectionModel is closed")

    override fun setLinearizationGamma(gamma: Double) {
        cvk_color_correction_model_set_linearization_gamma(check(), gamma)
    }

    override fun setLinearizationDegree(deg: Int) {
        cvk_color_correction_model_set_linearization_degree(check(), deg)
    }

    override fun setSaturatedThreshold(lower: Double, upper: Double) {
        cvk_color_correction_model_set_saturated_threshold(check(), lower, upper)
    }

    override fun setWeightsList(weights: Mat) {
        cvk_color_correction_model_set_weights_list(check(), weights.nativeHandle())
    }

    override fun setWeightCoeff(weightsCoeff: Double) {
        cvk_color_correction_model_set_weight_coeff(check(), weightsCoeff)
    }

    override fun setMaxCount(maxCount: Int) {
        cvk_color_correction_model_set_max_count(check(), maxCount)
    }

    override fun setEpsilon(epsilon: Double) {
        cvk_color_correction_model_set_epsilon(check(), epsilon)
    }

    override fun setRGB(rgb: Boolean) {
        cvk_color_correction_model_set_rgb(check(), if (rgb) 1 else 0)
    }

    override fun compute(): Mat =
        nativeMat(cvk_color_correction_model_compute(check()), "colorCorrectionModel.compute")

    override val colorCorrectionMatrix: Mat
        get() = nativeMat(
            cvk_color_correction_model_get_color_correction_matrix(check()),
            "colorCorrectionModel.colorCorrectionMatrix",
        )

    override val loss: Double
        get() = cvk_color_correction_model_get_loss(check())

    override val srcLinearRGB: Mat
        get() = nativeMat(
            cvk_color_correction_model_get_src_linear_rgb(check()),
            "colorCorrectionModel.srcLinearRGB",
        )

    override val refLinearRGB: Mat
        get() = nativeMat(
            cvk_color_correction_model_get_ref_linear_rgb(check()),
            "colorCorrectionModel.refLinearRGB",
        )

    override val mask: Mat
        get() = nativeMat(cvk_color_correction_model_get_mask(check()), "colorCorrectionModel.mask")

    override val weights: Mat
        get() = nativeMat(cvk_color_correction_model_get_weights(check()), "colorCorrectionModel.weights")

    override fun correctImage(src: Mat, islinear: Boolean): Mat =
        nativeMat(
            cvk_color_correction_model_correct_image(check(), src.nativeHandle(), if (islinear) 1 else 0),
            "colorCorrectionModel.correctImage",
        )

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_color_correction_model_release(handle)
    }
}

actual fun colorCorrectionModel(): ColorCorrectionModel =
    NativeColorCorrectionModel(
        cvk_color_correction_model_create_empty()
            ?: throw OpenCVException("colorCorrectionModel", lastNativeError()),
    )

actual fun colorCorrectionModel(src: Mat, constColor: Int): ColorCorrectionModel =
    NativeColorCorrectionModel(
        cvk_color_correction_model_create(src.nativeHandle(), constColor)
            ?: throw OpenCVException("colorCorrectionModel", lastNativeError()),
    )

// =========================================================================
// IntelligentScissorsMB
// =========================================================================

internal class NativeIntelligentScissorsMB(
    @Volatile private var raw: CPointer<cvk_intelligent_scissors_mb_t>?,
) : IntelligentScissorsMB {

    private fun check(): CPointer<cvk_intelligent_scissors_mb_t> =
        raw ?: throw IllegalStateException("IntelligentScissorsMB is closed")

    override fun setWeights(weightNonEdge: Float, weightGradientDirection: Float, weightGradientMagnitude: Float) {
        cvk_intelligent_scissors_mb_set_weights(
            check(), weightNonEdge, weightGradientDirection, weightGradientMagnitude,
        )
    }

    override fun setGradientMagnitudeMaxLimit(gradientMagnitudeThresholdMax: Float) {
        cvk_intelligent_scissors_mb_set_gradient_magnitude_max_limit(check(), gradientMagnitudeThresholdMax)
    }

    override fun setEdgeFeatureZeroCrossingParameters(gradientMagnitudeMinValue: Float) {
        cvk_intelligent_scissors_mb_set_edge_feature_zero_crossing_parameters(
            check(), gradientMagnitudeMinValue,
        )
    }

    override fun setEdgeFeatureCannyParameters(
        threshold1: Double,
        threshold2: Double,
        apertureSize: Int,
        l2gradient: Boolean,
    ) {
        cvk_intelligent_scissors_mb_set_edge_feature_canny_parameters(
            check(), threshold1, threshold2, apertureSize, if (l2gradient) 1 else 0,
        )
    }

    override fun applyImage(image: Mat) {
        cvk_intelligent_scissors_mb_apply_image(check(), image.nativeHandle())
    }

    override fun applyImageFeatures(
        nonEdge: Mat,
        gradientDirection: Mat,
        gradientMagnitude: Mat,
        image: Mat?,
    ) {
        cvk_intelligent_scissors_mb_apply_image_features(
            check(), nonEdge.nativeHandle(), gradientDirection.nativeHandle(),
            gradientMagnitude.nativeHandle(), image?.nativeHandle(),
        )
    }

    override fun buildMap(sourcePt: Point) {
        cvk_intelligent_scissors_mb_build_map(check(), sourcePt.x, sourcePt.y)
    }

    override fun getContour(targetPt: Point, backward: Boolean): Mat =
        nativeMat(
            cvk_intelligent_scissors_mb_get_contour(
                check(), targetPt.x, targetPt.y, if (backward) 1 else 0,
            ),
            "intelligentScissorsMB.getContour",
        )

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_intelligent_scissors_mb_release(handle)
    }
}

actual fun intelligentScissorsMB(): IntelligentScissorsMB =
    NativeIntelligentScissorsMB(
        cvk_intelligent_scissors_mb_create()
            ?: throw OpenCVException("intelligentScissorsMB", lastNativeError()),
    )

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

import cvk.cvk_background_subtractor_knn_apply
import cvk.cvk_background_subtractor_knn_apply_mask
import cvk.cvk_background_subtractor_knn_clear
import cvk.cvk_background_subtractor_knn_create
import cvk.cvk_background_subtractor_knn_empty
import cvk.cvk_background_subtractor_knn_get_background_image
import cvk.cvk_background_subtractor_knn_get_default_name
import cvk.cvk_background_subtractor_knn_get_detect_shadows
import cvk.cvk_background_subtractor_knn_get_dist2_threshold
import cvk.cvk_background_subtractor_knn_get_history
import cvk.cvk_background_subtractor_knn_get_knn_samples
import cvk.cvk_background_subtractor_knn_get_n_samples
import cvk.cvk_background_subtractor_knn_get_shadow_threshold
import cvk.cvk_background_subtractor_knn_get_shadow_value
import cvk.cvk_background_subtractor_knn_release
import cvk.cvk_background_subtractor_knn_save
import cvk.cvk_background_subtractor_knn_set_detect_shadows
import cvk.cvk_background_subtractor_knn_set_dist2_threshold
import cvk.cvk_background_subtractor_knn_set_history
import cvk.cvk_background_subtractor_knn_set_knn_samples
import cvk.cvk_background_subtractor_knn_set_n_samples
import cvk.cvk_background_subtractor_knn_set_shadow_threshold
import cvk.cvk_background_subtractor_knn_set_shadow_value
import cvk.cvk_background_subtractor_knn_t
import cvk.cvk_background_subtractor_mog2_apply
import cvk.cvk_background_subtractor_mog2_apply_mask
import cvk.cvk_background_subtractor_mog2_clear
import cvk.cvk_background_subtractor_mog2_create
import cvk.cvk_background_subtractor_mog2_empty
import cvk.cvk_background_subtractor_mog2_get_background_image
import cvk.cvk_background_subtractor_mog2_get_background_ratio
import cvk.cvk_background_subtractor_mog2_get_complexity_reduction_threshold
import cvk.cvk_background_subtractor_mog2_get_default_name
import cvk.cvk_background_subtractor_mog2_get_detect_shadows
import cvk.cvk_background_subtractor_mog2_get_history
import cvk.cvk_background_subtractor_mog2_get_n_mixtures
import cvk.cvk_background_subtractor_mog2_get_shadow_threshold
import cvk.cvk_background_subtractor_mog2_get_shadow_value
import cvk.cvk_background_subtractor_mog2_get_var_init
import cvk.cvk_background_subtractor_mog2_get_var_max
import cvk.cvk_background_subtractor_mog2_get_var_min
import cvk.cvk_background_subtractor_mog2_get_var_threshold
import cvk.cvk_background_subtractor_mog2_get_var_threshold_gen
import cvk.cvk_background_subtractor_mog2_release
import cvk.cvk_background_subtractor_mog2_save
import cvk.cvk_background_subtractor_mog2_set_background_ratio
import cvk.cvk_background_subtractor_mog2_set_complexity_reduction_threshold
import cvk.cvk_background_subtractor_mog2_set_detect_shadows
import cvk.cvk_background_subtractor_mog2_set_history
import cvk.cvk_background_subtractor_mog2_set_n_mixtures
import cvk.cvk_background_subtractor_mog2_set_shadow_threshold
import cvk.cvk_background_subtractor_mog2_set_shadow_value
import cvk.cvk_background_subtractor_mog2_set_var_init
import cvk.cvk_background_subtractor_mog2_set_var_max
import cvk.cvk_background_subtractor_mog2_set_var_min
import cvk.cvk_background_subtractor_mog2_set_var_threshold
import cvk.cvk_background_subtractor_mog2_set_var_threshold_gen
import cvk.cvk_background_subtractor_mog2_t
import cvk.cvk_calc_optical_flow_farneback
import cvk.cvk_calc_optical_flow_pyr_lk
import cvk.cvk_compute_ecc
import cvk.cvk_last_error
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.value
import kotlinx.cinterop.toKString
import kotlin.concurrent.Volatile
import cvk.cvk_dis_optical_flow_calc
import cvk.cvk_dis_optical_flow_clear
import cvk.cvk_dis_optical_flow_collect_garbage
import cvk.cvk_dis_optical_flow_create
import cvk.cvk_dis_optical_flow_empty
import cvk.cvk_dis_optical_flow_get_coarsest_scale
import cvk.cvk_dis_optical_flow_get_default_name
import cvk.cvk_dis_optical_flow_get_finest_scale
import cvk.cvk_dis_optical_flow_get_gradient_descent_iterations
import cvk.cvk_dis_optical_flow_get_patch_size
import cvk.cvk_dis_optical_flow_get_patch_stride
import cvk.cvk_dis_optical_flow_get_use_mean_normalization
import cvk.cvk_dis_optical_flow_get_use_spatial_propagation
import cvk.cvk_dis_optical_flow_get_variational_refinement_alpha
import cvk.cvk_dis_optical_flow_get_variational_refinement_delta
import cvk.cvk_dis_optical_flow_get_variational_refinement_epsilon
import cvk.cvk_dis_optical_flow_get_variational_refinement_gamma
import cvk.cvk_dis_optical_flow_get_variational_refinement_iterations
import cvk.cvk_dis_optical_flow_release
import cvk.cvk_dis_optical_flow_save
import cvk.cvk_dis_optical_flow_set_coarsest_scale
import cvk.cvk_dis_optical_flow_set_finest_scale
import cvk.cvk_dis_optical_flow_set_gradient_descent_iterations
import cvk.cvk_dis_optical_flow_set_patch_size
import cvk.cvk_dis_optical_flow_set_patch_stride
import cvk.cvk_dis_optical_flow_set_use_mean_normalization
import cvk.cvk_dis_optical_flow_set_use_spatial_propagation
import cvk.cvk_dis_optical_flow_set_variational_refinement_alpha
import cvk.cvk_dis_optical_flow_set_variational_refinement_delta
import cvk.cvk_dis_optical_flow_set_variational_refinement_epsilon
import cvk.cvk_dis_optical_flow_set_variational_refinement_gamma
import cvk.cvk_dis_optical_flow_set_variational_refinement_iterations
import cvk.cvk_dis_optical_flow_t
import cvk.cvk_farneback_optical_flow_calc
import cvk.cvk_farneback_optical_flow_clear
import cvk.cvk_farneback_optical_flow_collect_garbage
import cvk.cvk_farneback_optical_flow_create
import cvk.cvk_farneback_optical_flow_empty
import cvk.cvk_farneback_optical_flow_get_default_name
import cvk.cvk_farneback_optical_flow_get_fast_pyramids
import cvk.cvk_farneback_optical_flow_get_flags
import cvk.cvk_farneback_optical_flow_get_num_iters
import cvk.cvk_farneback_optical_flow_get_num_levels
import cvk.cvk_farneback_optical_flow_get_poly_n
import cvk.cvk_farneback_optical_flow_get_poly_sigma
import cvk.cvk_farneback_optical_flow_get_pyr_scale
import cvk.cvk_farneback_optical_flow_get_win_size
import cvk.cvk_farneback_optical_flow_release
import cvk.cvk_farneback_optical_flow_save
import cvk.cvk_farneback_optical_flow_set_fast_pyramids
import cvk.cvk_farneback_optical_flow_set_flags
import cvk.cvk_farneback_optical_flow_set_num_iters
import cvk.cvk_farneback_optical_flow_set_num_levels
import cvk.cvk_farneback_optical_flow_set_poly_n
import cvk.cvk_farneback_optical_flow_set_poly_sigma
import cvk.cvk_farneback_optical_flow_set_pyr_scale
import cvk.cvk_farneback_optical_flow_set_win_size
import cvk.cvk_farneback_optical_flow_t
import cvk.cvk_find_transform_ecc
import cvk.cvk_find_transform_ecc_multi_scale
import cvk.cvk_find_transform_ecc_with_mask
import cvk.cvk_kalman_filter_correct
import cvk.cvk_kalman_filter_create
import cvk.cvk_kalman_filter_get_control_matrix
import cvk.cvk_kalman_filter_get_error_cov_post
import cvk.cvk_kalman_filter_get_error_cov_pre
import cvk.cvk_kalman_filter_get_gain
import cvk.cvk_kalman_filter_get_measurement_matrix
import cvk.cvk_kalman_filter_get_measurement_noise_cov
import cvk.cvk_kalman_filter_get_process_noise_cov
import cvk.cvk_kalman_filter_get_state_post
import cvk.cvk_kalman_filter_get_state_pre
import cvk.cvk_kalman_filter_get_transition_matrix
import cvk.cvk_kalman_filter_predict
import cvk.cvk_kalman_filter_release
import cvk.cvk_kalman_filter_set_control_matrix
import cvk.cvk_kalman_filter_set_error_cov_post
import cvk.cvk_kalman_filter_set_error_cov_pre
import cvk.cvk_kalman_filter_set_gain
import cvk.cvk_kalman_filter_set_measurement_matrix
import cvk.cvk_kalman_filter_set_measurement_noise_cov
import cvk.cvk_kalman_filter_set_process_noise_cov
import cvk.cvk_kalman_filter_set_state_post
import cvk.cvk_kalman_filter_set_state_pre
import cvk.cvk_kalman_filter_set_transition_matrix
import cvk.cvk_kalman_filter_t
import cvk.cvk_sparse_pyr_lk_optical_flow_calc
import cvk.cvk_sparse_pyr_lk_optical_flow_clear
import cvk.cvk_sparse_pyr_lk_optical_flow_create
import cvk.cvk_sparse_pyr_lk_optical_flow_empty
import cvk.cvk_sparse_pyr_lk_optical_flow_get_default_name
import cvk.cvk_sparse_pyr_lk_optical_flow_get_flags
import cvk.cvk_sparse_pyr_lk_optical_flow_get_max_level
import cvk.cvk_sparse_pyr_lk_optical_flow_get_min_eig_threshold
import cvk.cvk_sparse_pyr_lk_optical_flow_get_term_criteria
import cvk.cvk_sparse_pyr_lk_optical_flow_get_win_h
import cvk.cvk_sparse_pyr_lk_optical_flow_get_win_w
import cvk.cvk_sparse_pyr_lk_optical_flow_release
import cvk.cvk_sparse_pyr_lk_optical_flow_save
import cvk.cvk_sparse_pyr_lk_optical_flow_set_flags
import cvk.cvk_sparse_pyr_lk_optical_flow_set_max_level
import cvk.cvk_sparse_pyr_lk_optical_flow_set_min_eig_threshold
import cvk.cvk_sparse_pyr_lk_optical_flow_set_term_criteria
import cvk.cvk_sparse_pyr_lk_optical_flow_set_win_size
import cvk.cvk_sparse_pyr_lk_optical_flow_t
import cvk.cvk_variational_refinement_calc
import cvk.cvk_variational_refinement_calc_uv
import cvk.cvk_variational_refinement_clear
import cvk.cvk_variational_refinement_collect_garbage
import cvk.cvk_variational_refinement_create
import cvk.cvk_variational_refinement_empty
import cvk.cvk_variational_refinement_get_alpha
import cvk.cvk_variational_refinement_get_default_name
import cvk.cvk_variational_refinement_get_delta
import cvk.cvk_variational_refinement_get_epsilon
import cvk.cvk_variational_refinement_get_fixed_point_iterations
import cvk.cvk_variational_refinement_get_gamma
import cvk.cvk_variational_refinement_get_omega
import cvk.cvk_variational_refinement_get_sor_iterations
import cvk.cvk_variational_refinement_release
import cvk.cvk_variational_refinement_save
import cvk.cvk_variational_refinement_set_alpha
import cvk.cvk_variational_refinement_set_delta
import cvk.cvk_variational_refinement_set_epsilon
import cvk.cvk_variational_refinement_set_fixed_point_iterations
import cvk.cvk_variational_refinement_set_gamma
import cvk.cvk_variational_refinement_set_omega
import cvk.cvk_variational_refinement_set_sor_iterations
import cvk.cvk_variational_refinement_t

/** Message of the last native OpenCV error on this thread, or null. */
private fun lastNativeError(): String? = cvk_last_error()?.toKString()

// =========================================================================
// video statics
// =========================================================================

actual fun calcOpticalFlowFarneback(
    prev: Mat,
    next: Mat,
    flow: Mat,
    pyrScale: Double,
    levels: Int,
    winSize: Int,
    iterations: Int,
    polyN: Int,
    polySigma: Double,
    flags: Int,
) = cvk_calc_optical_flow_farneback(
    prev.nativeHandle(), next.nativeHandle(), flow.nativeHandle(),
    pyrScale, levels, winSize, iterations, polyN, polySigma, flags,
)

actual fun calcOpticalFlowPyrLK(
    prevImg: Mat,
    nextImg: Mat,
    prevPts: MatOfPoint2f,
    nextPts: MatOfPoint2f,
    status: MatOfByte,
    err: MatOfFloat,
    winSize: Size,
    maxLevel: Int,
    criteria: TermCriteria,
    flags: Int,
    minEigThreshold: Double,
) = cvk_calc_optical_flow_pyr_lk(
    prevImg.nativeHandle(), nextImg.nativeHandle(), prevPts.mat.nativeHandle(),
    nextPts.mat.nativeHandle(), status.mat.nativeHandle(), err.mat.nativeHandle(),
    winSize.width, winSize.height, maxLevel, criteria.type, criteria.maxCount,
    criteria.epsilon, flags, minEigThreshold,
)

actual fun computeECC(templateImage: Mat, inputImage: Mat, inputMask: Mat?): Double =
    cvk_compute_ecc(templateImage.nativeHandle(), inputImage.nativeHandle(), inputMask?.nativeHandle())

actual fun findTransformECC(
    templateImage: Mat,
    inputImage: Mat,
    warpMatrix: Mat,
    motionType: Int,
    criteria: TermCriteria,
    inputMask: Mat?,
    gaussFiltSize: Int,
): Double = cvk_find_transform_ecc(
    templateImage.nativeHandle(), inputImage.nativeHandle(), warpMatrix.nativeHandle(),
    motionType, criteria.type, criteria.maxCount, criteria.epsilon,
    inputMask?.nativeHandle(), gaussFiltSize,
)

actual fun findTransformECCWithMask(
    templateImage: Mat,
    inputImage: Mat,
    templateMask: Mat,
    inputMask: Mat,
    warpMatrix: Mat,
    motionType: Int,
    criteria: TermCriteria,
    gaussFiltSize: Int,
): Double = cvk_find_transform_ecc_with_mask(
    templateImage.nativeHandle(), inputImage.nativeHandle(), templateMask.nativeHandle(),
    inputMask.nativeHandle(), warpMatrix.nativeHandle(), motionType, criteria.type,
    criteria.maxCount, criteria.epsilon, gaussFiltSize,
)

actual fun findTransformECCMultiScale(
    reference: Mat,
    sample: Mat,
    warpMatrix: Mat,
    eccParams: ECCParameters,
    referenceMask: Mat?,
    sampleMask: Mat?,
): Double = cvk_find_transform_ecc_multi_scale(
    reference.nativeHandle(), sample.nativeHandle(), warpMatrix.nativeHandle(),
    eccParams.motionType, eccParams.criteria.type, eccParams.criteria.maxCount,
    eccParams.criteria.epsilon, eccParams.itersPerLevel?.mat?.nativeHandle(),
    eccParams.gaussFiltSize, eccParams.nlevels, eccParams.interpolation,
    referenceMask?.nativeHandle(), sampleMask?.nativeHandle(),
)

// =========================================================================
// background subtraction
// =========================================================================

internal class NativeBackgroundSubtractorKNN(
    @Volatile private var raw: CPointer<cvk_background_subtractor_knn_t>?,
) : BackgroundSubtractorKNN {

    private fun check(): CPointer<cvk_background_subtractor_knn_t> =
        raw ?: throw IllegalStateException("BackgroundSubtractorKNN is closed")

    override fun apply(image: Mat, fgmask: Mat, learningRate: Double) {
        cvk_background_subtractor_knn_apply(check(), image.nativeHandle(), fgmask.nativeHandle(), learningRate)
    }

    override fun apply(image: Mat, knownForegroundMask: Mat, fgmask: Mat, learningRate: Double) {
        cvk_background_subtractor_knn_apply_mask(
            check(), image.nativeHandle(), knownForegroundMask.nativeHandle(),
            fgmask.nativeHandle(), learningRate,
        )
    }

    override fun getBackgroundImage(): Mat =
        nativeMat(cvk_background_subtractor_knn_get_background_image(check()), "getBackgroundImage")

    override var history: Int
        get() = cvk_background_subtractor_knn_get_history(check())
        set(value) = cvk_background_subtractor_knn_set_history(check(), value)

    override var nSamples: Int
        get() = cvk_background_subtractor_knn_get_n_samples(check())
        set(value) = cvk_background_subtractor_knn_set_n_samples(check(), value)

    override var dist2Threshold: Double
        get() = cvk_background_subtractor_knn_get_dist2_threshold(check())
        set(value) = cvk_background_subtractor_knn_set_dist2_threshold(check(), value)

    override var kNNsamples: Int
        get() = cvk_background_subtractor_knn_get_knn_samples(check())
        set(value) = cvk_background_subtractor_knn_set_knn_samples(check(), value)

    override var detectShadows: Boolean
        get() = cvk_background_subtractor_knn_get_detect_shadows(check()) != 0
        set(value) = cvk_background_subtractor_knn_set_detect_shadows(check(), if (value) 1 else 0)

    override var shadowValue: Int
        get() = cvk_background_subtractor_knn_get_shadow_value(check())
        set(value) = cvk_background_subtractor_knn_set_shadow_value(check(), value)

    override var shadowThreshold: Double
        get() = cvk_background_subtractor_knn_get_shadow_threshold(check())
        set(value) = cvk_background_subtractor_knn_set_shadow_threshold(check(), value)

    override fun clear() = cvk_background_subtractor_knn_clear(check())

    override fun empty(): Boolean = cvk_background_subtractor_knn_empty(check()) != 0

    override fun save(filename: String) = cvk_background_subtractor_knn_save(check(), filename)

    override fun getDefaultName(): String =
        cvk_background_subtractor_knn_get_default_name(check())?.toKString()
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_background_subtractor_knn_release(h)
    }
}

internal class NativeBackgroundSubtractorMOG2(
    @Volatile private var raw: CPointer<cvk_background_subtractor_mog2_t>?,
) : BackgroundSubtractorMOG2 {

    private fun check(): CPointer<cvk_background_subtractor_mog2_t> =
        raw ?: throw IllegalStateException("BackgroundSubtractorMOG2 is closed")

    override fun apply(image: Mat, fgmask: Mat, learningRate: Double) {
        cvk_background_subtractor_mog2_apply(check(), image.nativeHandle(), fgmask.nativeHandle(), learningRate)
    }

    override fun apply(image: Mat, knownForegroundMask: Mat, fgmask: Mat, learningRate: Double) {
        cvk_background_subtractor_mog2_apply_mask(
            check(), image.nativeHandle(), knownForegroundMask.nativeHandle(),
            fgmask.nativeHandle(), learningRate,
        )
    }

    override fun getBackgroundImage(): Mat =
        nativeMat(cvk_background_subtractor_mog2_get_background_image(check()), "getBackgroundImage")

    override var history: Int
        get() = cvk_background_subtractor_mog2_get_history(check())
        set(value) = cvk_background_subtractor_mog2_set_history(check(), value)

    override var nMixtures: Int
        get() = cvk_background_subtractor_mog2_get_n_mixtures(check())
        set(value) = cvk_background_subtractor_mog2_set_n_mixtures(check(), value)

    override var backgroundRatio: Double
        get() = cvk_background_subtractor_mog2_get_background_ratio(check())
        set(value) = cvk_background_subtractor_mog2_set_background_ratio(check(), value)

    override var varThreshold: Double
        get() = cvk_background_subtractor_mog2_get_var_threshold(check())
        set(value) = cvk_background_subtractor_mog2_set_var_threshold(check(), value)

    override var varThresholdGen: Double
        get() = cvk_background_subtractor_mog2_get_var_threshold_gen(check())
        set(value) = cvk_background_subtractor_mog2_set_var_threshold_gen(check(), value)

    override var varInit: Double
        get() = cvk_background_subtractor_mog2_get_var_init(check())
        set(value) = cvk_background_subtractor_mog2_set_var_init(check(), value)

    override var varMin: Double
        get() = cvk_background_subtractor_mog2_get_var_min(check())
        set(value) = cvk_background_subtractor_mog2_set_var_min(check(), value)

    override var varMax: Double
        get() = cvk_background_subtractor_mog2_get_var_max(check())
        set(value) = cvk_background_subtractor_mog2_set_var_max(check(), value)

    override var complexityReductionThreshold: Double
        get() = cvk_background_subtractor_mog2_get_complexity_reduction_threshold(check())
        set(value) = cvk_background_subtractor_mog2_set_complexity_reduction_threshold(check(), value)

    override var detectShadows: Boolean
        get() = cvk_background_subtractor_mog2_get_detect_shadows(check()) != 0
        set(value) = cvk_background_subtractor_mog2_set_detect_shadows(check(), if (value) 1 else 0)

    override var shadowValue: Int
        get() = cvk_background_subtractor_mog2_get_shadow_value(check())
        set(value) = cvk_background_subtractor_mog2_set_shadow_value(check(), value)

    override var shadowThreshold: Double
        get() = cvk_background_subtractor_mog2_get_shadow_threshold(check())
        set(value) = cvk_background_subtractor_mog2_set_shadow_threshold(check(), value)

    override fun clear() = cvk_background_subtractor_mog2_clear(check())

    override fun empty(): Boolean = cvk_background_subtractor_mog2_empty(check()) != 0

    override fun save(filename: String) = cvk_background_subtractor_mog2_save(check(), filename)

    override fun getDefaultName(): String =
        cvk_background_subtractor_mog2_get_default_name(check())?.toKString()
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_background_subtractor_mog2_release(h)
    }
}

actual fun createBackgroundSubtractorKNN(
    history: Int,
    dist2Threshold: Double,
    detectShadows: Boolean,
): BackgroundSubtractorKNN =
    NativeBackgroundSubtractorKNN(
        cvk_background_subtractor_knn_create(history, dist2Threshold, if (detectShadows) 1 else 0)
            ?: throw OpenCVException("createBackgroundSubtractorKNN", lastNativeError()),
    )

actual fun createBackgroundSubtractorMOG2(
    history: Int,
    varThreshold: Double,
    detectShadows: Boolean,
): BackgroundSubtractorMOG2 =
    NativeBackgroundSubtractorMOG2(
        cvk_background_subtractor_mog2_create(history, varThreshold, if (detectShadows) 1 else 0)
            ?: throw OpenCVException("createBackgroundSubtractorMOG2", lastNativeError()),
    )

// =========================================================================
// optical flow
// =========================================================================

internal class NativeFarnebackOpticalFlow(
    @Volatile private var raw: CPointer<cvk_farneback_optical_flow_t>?,
) : FarnebackOpticalFlow {

    private fun check(): CPointer<cvk_farneback_optical_flow_t> =
        raw ?: throw IllegalStateException("FarnebackOpticalFlow is closed")

    override fun calc(i0: Mat, i1: Mat, flow: Mat) {
        cvk_farneback_optical_flow_calc(check(), i0.nativeHandle(), i1.nativeHandle(), flow.nativeHandle())
    }

    override fun collectGarbage() = cvk_farneback_optical_flow_collect_garbage(check())

    override var numLevels: Int
        get() = cvk_farneback_optical_flow_get_num_levels(check())
        set(value) = cvk_farneback_optical_flow_set_num_levels(check(), value)

    override var pyrScale: Double
        get() = cvk_farneback_optical_flow_get_pyr_scale(check())
        set(value) = cvk_farneback_optical_flow_set_pyr_scale(check(), value)

    override var fastPyramids: Boolean
        get() = cvk_farneback_optical_flow_get_fast_pyramids(check()) != 0
        set(value) = cvk_farneback_optical_flow_set_fast_pyramids(check(), if (value) 1 else 0)

    override var winSize: Int
        get() = cvk_farneback_optical_flow_get_win_size(check())
        set(value) = cvk_farneback_optical_flow_set_win_size(check(), value)

    override var numIters: Int
        get() = cvk_farneback_optical_flow_get_num_iters(check())
        set(value) = cvk_farneback_optical_flow_set_num_iters(check(), value)

    override var polyN: Int
        get() = cvk_farneback_optical_flow_get_poly_n(check())
        set(value) = cvk_farneback_optical_flow_set_poly_n(check(), value)

    override var polySigma: Double
        get() = cvk_farneback_optical_flow_get_poly_sigma(check())
        set(value) = cvk_farneback_optical_flow_set_poly_sigma(check(), value)

    override var flags: Int
        get() = cvk_farneback_optical_flow_get_flags(check())
        set(value) = cvk_farneback_optical_flow_set_flags(check(), value)

    override fun clear() = cvk_farneback_optical_flow_clear(check())

    override fun empty(): Boolean = cvk_farneback_optical_flow_empty(check()) != 0

    override fun save(filename: String) = cvk_farneback_optical_flow_save(check(), filename)

    override fun getDefaultName(): String =
        cvk_farneback_optical_flow_get_default_name(check())?.toKString()
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_farneback_optical_flow_release(h)
    }
}

internal class NativeDisOpticalFlow(
    @Volatile private var raw: CPointer<cvk_dis_optical_flow_t>?,
) : DISOpticalFlow {

    private fun check(): CPointer<cvk_dis_optical_flow_t> =
        raw ?: throw IllegalStateException("DISOpticalFlow is closed")

    override fun calc(i0: Mat, i1: Mat, flow: Mat) {
        cvk_dis_optical_flow_calc(check(), i0.nativeHandle(), i1.nativeHandle(), flow.nativeHandle())
    }

    override fun collectGarbage() = cvk_dis_optical_flow_collect_garbage(check())

    override var finestScale: Int
        get() = cvk_dis_optical_flow_get_finest_scale(check())
        set(value) = cvk_dis_optical_flow_set_finest_scale(check(), value)

    override var coarsestScale: Int
        get() = cvk_dis_optical_flow_get_coarsest_scale(check())
        set(value) = cvk_dis_optical_flow_set_coarsest_scale(check(), value)

    override var patchSize: Int
        get() = cvk_dis_optical_flow_get_patch_size(check())
        set(value) = cvk_dis_optical_flow_set_patch_size(check(), value)

    override var patchStride: Int
        get() = cvk_dis_optical_flow_get_patch_stride(check())
        set(value) = cvk_dis_optical_flow_set_patch_stride(check(), value)

    override var gradientDescentIterations: Int
        get() = cvk_dis_optical_flow_get_gradient_descent_iterations(check())
        set(value) = cvk_dis_optical_flow_set_gradient_descent_iterations(check(), value)

    override var variationalRefinementIterations: Int
        get() = cvk_dis_optical_flow_get_variational_refinement_iterations(check())
        set(value) = cvk_dis_optical_flow_set_variational_refinement_iterations(check(), value)

    override var variationalRefinementAlpha: Float
        get() = cvk_dis_optical_flow_get_variational_refinement_alpha(check())
        set(value) = cvk_dis_optical_flow_set_variational_refinement_alpha(check(), value)

    override var variationalRefinementDelta: Float
        get() = cvk_dis_optical_flow_get_variational_refinement_delta(check())
        set(value) = cvk_dis_optical_flow_set_variational_refinement_delta(check(), value)

    override var variationalRefinementGamma: Float
        get() = cvk_dis_optical_flow_get_variational_refinement_gamma(check())
        set(value) = cvk_dis_optical_flow_set_variational_refinement_gamma(check(), value)

    override var variationalRefinementEpsilon: Float
        get() = cvk_dis_optical_flow_get_variational_refinement_epsilon(check())
        set(value) = cvk_dis_optical_flow_set_variational_refinement_epsilon(check(), value)

    override var useMeanNormalization: Boolean
        get() = cvk_dis_optical_flow_get_use_mean_normalization(check()) != 0
        set(value) = cvk_dis_optical_flow_set_use_mean_normalization(check(), if (value) 1 else 0)

    override var useSpatialPropagation: Boolean
        get() = cvk_dis_optical_flow_get_use_spatial_propagation(check()) != 0
        set(value) = cvk_dis_optical_flow_set_use_spatial_propagation(check(), if (value) 1 else 0)

    override fun clear() = cvk_dis_optical_flow_clear(check())

    override fun empty(): Boolean = cvk_dis_optical_flow_empty(check()) != 0

    override fun save(filename: String) = cvk_dis_optical_flow_save(check(), filename)

    override fun getDefaultName(): String =
        cvk_dis_optical_flow_get_default_name(check())?.toKString()
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_dis_optical_flow_release(h)
    }
}

internal class NativeSparsePyrLKOpticalFlow(
    @Volatile private var raw: CPointer<cvk_sparse_pyr_lk_optical_flow_t>?,
) : SparsePyrLKOpticalFlow {

    private fun check(): CPointer<cvk_sparse_pyr_lk_optical_flow_t> =
        raw ?: throw IllegalStateException("SparsePyrLKOpticalFlow is closed")

    override fun calc(prevImg: Mat, nextImg: Mat, prevPts: Mat, nextPts: Mat, status: Mat, err: Mat?) {
        cvk_sparse_pyr_lk_optical_flow_calc(
            check(), prevImg.nativeHandle(), nextImg.nativeHandle(), prevPts.nativeHandle(),
            nextPts.nativeHandle(), status.nativeHandle(), err?.nativeHandle(),
        )
    }

    override var winSize: Size
        get() = Size(
            cvk_sparse_pyr_lk_optical_flow_get_win_w(check()),
            cvk_sparse_pyr_lk_optical_flow_get_win_h(check()),
        )
        set(value) = cvk_sparse_pyr_lk_optical_flow_set_win_size(check(), value.width, value.height)

    override var maxLevel: Int
        get() = cvk_sparse_pyr_lk_optical_flow_get_max_level(check())
        set(value) = cvk_sparse_pyr_lk_optical_flow_set_max_level(check(), value)

    override var termCriteria: TermCriteria
        get() = memScoped {
            val type = alloc<IntVar>()
            val maxCount = alloc<IntVar>()
            val epsilon = alloc<DoubleVar>()
            cvk_sparse_pyr_lk_optical_flow_get_term_criteria(check(), type.ptr, maxCount.ptr, epsilon.ptr)
            TermCriteria(type.value, maxCount.value, epsilon.value)
        }
        set(value) = cvk_sparse_pyr_lk_optical_flow_set_term_criteria(
            check(), value.type, value.maxCount, value.epsilon,
        )

    override var flags: Int
        get() = cvk_sparse_pyr_lk_optical_flow_get_flags(check())
        set(value) = cvk_sparse_pyr_lk_optical_flow_set_flags(check(), value)

    override var minEigThreshold: Double
        get() = cvk_sparse_pyr_lk_optical_flow_get_min_eig_threshold(check())
        set(value) = cvk_sparse_pyr_lk_optical_flow_set_min_eig_threshold(check(), value)

    override fun clear() = cvk_sparse_pyr_lk_optical_flow_clear(check())

    override fun empty(): Boolean = cvk_sparse_pyr_lk_optical_flow_empty(check()) != 0

    override fun save(filename: String) = cvk_sparse_pyr_lk_optical_flow_save(check(), filename)

    override fun getDefaultName(): String =
        cvk_sparse_pyr_lk_optical_flow_get_default_name(check())?.toKString()
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_sparse_pyr_lk_optical_flow_release(h)
    }
}

internal class NativeVariationalRefinement(
    @Volatile private var raw: CPointer<cvk_variational_refinement_t>?,
) : VariationalRefinement {

    private fun check(): CPointer<cvk_variational_refinement_t> =
        raw ?: throw IllegalStateException("VariationalRefinement is closed")

    override fun calc(i0: Mat, i1: Mat, flow: Mat) {
        cvk_variational_refinement_calc(check(), i0.nativeHandle(), i1.nativeHandle(), flow.nativeHandle())
    }

    override fun calcUV(i0: Mat, i1: Mat, flowU: Mat, flowV: Mat) {
        cvk_variational_refinement_calc_uv(
            check(), i0.nativeHandle(), i1.nativeHandle(), flowU.nativeHandle(), flowV.nativeHandle(),
        )
    }

    override fun collectGarbage() = cvk_variational_refinement_collect_garbage(check())

    override var fixedPointIterations: Int
        get() = cvk_variational_refinement_get_fixed_point_iterations(check())
        set(value) = cvk_variational_refinement_set_fixed_point_iterations(check(), value)

    override var sorIterations: Int
        get() = cvk_variational_refinement_get_sor_iterations(check())
        set(value) = cvk_variational_refinement_set_sor_iterations(check(), value)

    override var omega: Float
        get() = cvk_variational_refinement_get_omega(check())
        set(value) = cvk_variational_refinement_set_omega(check(), value)

    override var alpha: Float
        get() = cvk_variational_refinement_get_alpha(check())
        set(value) = cvk_variational_refinement_set_alpha(check(), value)

    override var delta: Float
        get() = cvk_variational_refinement_get_delta(check())
        set(value) = cvk_variational_refinement_set_delta(check(), value)

    override var gamma: Float
        get() = cvk_variational_refinement_get_gamma(check())
        set(value) = cvk_variational_refinement_set_gamma(check(), value)

    override var epsilon: Float
        get() = cvk_variational_refinement_get_epsilon(check())
        set(value) = cvk_variational_refinement_set_epsilon(check(), value)

    override fun clear() = cvk_variational_refinement_clear(check())

    override fun empty(): Boolean = cvk_variational_refinement_empty(check()) != 0

    override fun save(filename: String) = cvk_variational_refinement_save(check(), filename)

    override fun getDefaultName(): String =
        cvk_variational_refinement_get_default_name(check())?.toKString()
            ?: throw OpenCVException("getDefaultName", lastNativeError())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_variational_refinement_release(h)
    }
}

actual fun createFarnebackOpticalFlow(
    numLevels: Int,
    pyrScale: Double,
    fastPyramids: Boolean,
    winSize: Int,
    numIters: Int,
    polyN: Int,
    polySigma: Double,
    flags: Int,
): FarnebackOpticalFlow = NativeFarnebackOpticalFlow(
    cvk_farneback_optical_flow_create(
        numLevels, pyrScale, if (fastPyramids) 1 else 0, winSize, numIters, polyN, polySigma, flags,
    ) ?: throw OpenCVException("createFarnebackOpticalFlow", lastNativeError()),
)

actual fun createDisOpticalFlow(preset: Int): DISOpticalFlow =
    NativeDisOpticalFlow(
        cvk_dis_optical_flow_create(preset)
            ?: throw OpenCVException("createDisOpticalFlow", lastNativeError()),
    )

actual fun createVariationalRefinement(): VariationalRefinement =
    NativeVariationalRefinement(
        cvk_variational_refinement_create()
            ?: throw OpenCVException("createVariationalRefinement", lastNativeError()),
    )

actual fun createSparsePyrLKOpticalFlow(
    winSize: Size,
    maxLevel: Int,
    criteria: TermCriteria,
    flags: Int,
    minEigThreshold: Double,
): SparsePyrLKOpticalFlow = NativeSparsePyrLKOpticalFlow(
    cvk_sparse_pyr_lk_optical_flow_create(
        winSize.width, winSize.height, maxLevel, criteria.type, criteria.maxCount,
        criteria.epsilon, flags, minEigThreshold,
    ) ?: throw OpenCVException("createSparsePyrLKOpticalFlow", lastNativeError()),
)

// =========================================================================
// Kalman filter
// =========================================================================

internal class NativeKalmanFilter(
    @Volatile private var raw: CPointer<cvk_kalman_filter_t>?,
) : KalmanFilter {

    private fun check(): CPointer<cvk_kalman_filter_t> =
        raw ?: throw IllegalStateException("KalmanFilter is closed")

    override fun predict(control: Mat?): Mat =
        nativeMat(cvk_kalman_filter_predict(check(), control?.nativeHandle()), "predict")

    override fun correct(measurement: Mat): Mat =
        nativeMat(cvk_kalman_filter_correct(check(), measurement.nativeHandle()), "correct")

    override var statePre: Mat
        get() = nativeMat(cvk_kalman_filter_get_state_pre(check()), "statePre")
        set(value) = cvk_kalman_filter_set_state_pre(check(), value.nativeHandle())

    override var statePost: Mat
        get() = nativeMat(cvk_kalman_filter_get_state_post(check()), "statePost")
        set(value) = cvk_kalman_filter_set_state_post(check(), value.nativeHandle())

    override var transitionMatrix: Mat
        get() = nativeMat(cvk_kalman_filter_get_transition_matrix(check()), "transitionMatrix")
        set(value) = cvk_kalman_filter_set_transition_matrix(check(), value.nativeHandle())

    override var controlMatrix: Mat
        get() = nativeMat(cvk_kalman_filter_get_control_matrix(check()), "controlMatrix")
        set(value) = cvk_kalman_filter_set_control_matrix(check(), value.nativeHandle())

    override var measurementMatrix: Mat
        get() = nativeMat(cvk_kalman_filter_get_measurement_matrix(check()), "measurementMatrix")
        set(value) = cvk_kalman_filter_set_measurement_matrix(check(), value.nativeHandle())

    override var processNoiseCov: Mat
        get() = nativeMat(cvk_kalman_filter_get_process_noise_cov(check()), "processNoiseCov")
        set(value) = cvk_kalman_filter_set_process_noise_cov(check(), value.nativeHandle())

    override var measurementNoiseCov: Mat
        get() = nativeMat(cvk_kalman_filter_get_measurement_noise_cov(check()), "measurementNoiseCov")
        set(value) = cvk_kalman_filter_set_measurement_noise_cov(check(), value.nativeHandle())

    override var errorCovPre: Mat
        get() = nativeMat(cvk_kalman_filter_get_error_cov_pre(check()), "errorCovPre")
        set(value) = cvk_kalman_filter_set_error_cov_pre(check(), value.nativeHandle())

    override var gain: Mat
        get() = nativeMat(cvk_kalman_filter_get_gain(check()), "gain")
        set(value) = cvk_kalman_filter_set_gain(check(), value.nativeHandle())

    override var errorCovPost: Mat
        get() = nativeMat(cvk_kalman_filter_get_error_cov_post(check()), "errorCovPost")
        set(value) = cvk_kalman_filter_set_error_cov_post(check(), value.nativeHandle())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_kalman_filter_release(h)
    }
}

actual fun kalmanFilter(
    dynamParams: Int,
    measureParams: Int,
    controlParams: Int,
    type: Int,
): KalmanFilter = NativeKalmanFilter(
    cvk_kalman_filter_create(dynamParams, measureParams, controlParams, type)
        ?: throw OpenCVException("kalmanFilter", lastNativeError()),
)
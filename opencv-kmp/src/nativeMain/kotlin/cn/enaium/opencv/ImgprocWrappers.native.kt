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

import kotlin.concurrent.Volatile
import cvk.cvk_gh_ballard_clear
import cvk.cvk_gh_ballard_create
import cvk.cvk_gh_ballard_detect
import cvk.cvk_gh_ballard_detect_edges
import cvk.cvk_gh_ballard_empty
import cvk.cvk_gh_ballard_get_canny_high_thresh
import cvk.cvk_gh_ballard_get_canny_low_thresh
import cvk.cvk_gh_ballard_get_default_name
import cvk.cvk_gh_ballard_get_dp
import cvk.cvk_gh_ballard_get_levels
import cvk.cvk_gh_ballard_get_max_buffer_size
import cvk.cvk_gh_ballard_get_min_dist
import cvk.cvk_gh_ballard_get_votes_threshold
import cvk.cvk_gh_ballard_release
import cvk.cvk_gh_ballard_save
import cvk.cvk_gh_ballard_set_canny_high_thresh
import cvk.cvk_gh_ballard_set_canny_low_thresh
import cvk.cvk_gh_ballard_set_dp
import cvk.cvk_gh_ballard_set_levels
import cvk.cvk_gh_ballard_set_max_buffer_size
import cvk.cvk_gh_ballard_set_min_dist
import cvk.cvk_gh_ballard_set_template
import cvk.cvk_gh_ballard_set_template_edges
import cvk.cvk_gh_ballard_set_votes_threshold
import cvk.cvk_gh_ballard_t
import cvk.cvk_gh_guil_clear
import cvk.cvk_gh_guil_create
import cvk.cvk_gh_guil_detect
import cvk.cvk_gh_guil_detect_edges
import cvk.cvk_gh_guil_empty
import cvk.cvk_gh_guil_get_angle_epsilon
import cvk.cvk_gh_guil_get_angle_step
import cvk.cvk_gh_guil_get_angle_thresh
import cvk.cvk_gh_guil_get_canny_high_thresh
import cvk.cvk_gh_guil_get_canny_low_thresh
import cvk.cvk_gh_guil_get_default_name
import cvk.cvk_gh_guil_get_dp
import cvk.cvk_gh_guil_get_levels
import cvk.cvk_gh_guil_get_max_angle
import cvk.cvk_gh_guil_get_max_buffer_size
import cvk.cvk_gh_guil_get_max_scale
import cvk.cvk_gh_guil_get_min_angle
import cvk.cvk_gh_guil_get_min_dist
import cvk.cvk_gh_guil_get_min_scale
import cvk.cvk_gh_guil_get_pos_thresh
import cvk.cvk_gh_guil_get_scale_step
import cvk.cvk_gh_guil_get_scale_thresh
import cvk.cvk_gh_guil_get_xi
import cvk.cvk_gh_guil_release
import cvk.cvk_gh_guil_save
import cvk.cvk_gh_guil_set_angle_epsilon
import cvk.cvk_gh_guil_set_angle_step
import cvk.cvk_gh_guil_set_angle_thresh
import cvk.cvk_gh_guil_set_canny_high_thresh
import cvk.cvk_gh_guil_set_canny_low_thresh
import cvk.cvk_gh_guil_set_dp
import cvk.cvk_gh_guil_set_levels
import cvk.cvk_gh_guil_set_max_angle
import cvk.cvk_gh_guil_set_max_buffer_size
import cvk.cvk_gh_guil_set_max_scale
import cvk.cvk_gh_guil_set_min_angle
import cvk.cvk_gh_guil_set_min_dist
import cvk.cvk_gh_guil_set_min_scale
import cvk.cvk_gh_guil_set_pos_thresh
import cvk.cvk_gh_guil_set_scale_step
import cvk.cvk_gh_guil_set_scale_thresh
import cvk.cvk_gh_guil_set_template
import cvk.cvk_gh_guil_set_template_edges
import cvk.cvk_gh_guil_set_xi
import cvk.cvk_gh_guil_t
import cvk.cvk_last_error
import cvk.cvk_line_segment_detector_clear
import cvk.cvk_line_segment_detector_compare_segments
import cvk.cvk_line_segment_detector_create
import cvk.cvk_line_segment_detector_detect
import cvk.cvk_line_segment_detector_draw_segments
import cvk.cvk_line_segment_detector_empty
import cvk.cvk_line_segment_detector_get_default_name
import cvk.cvk_line_segment_detector_release
import cvk.cvk_line_segment_detector_save
import cvk.cvk_line_segment_detector_t
import cvk.cvk_mat_t
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

// =========================================================================
// LineSegmentDetector
// =========================================================================

/** Native (cinterop) [LineSegmentDetector] over a `cv::Ptr<cv::LineSegmentDetector>`. */
internal class NativeLineSegmentDetector(
    @Volatile private var raw: CPointer<cvk_line_segment_detector_t>?,
) : LineSegmentDetector {

    private fun check(): CPointer<cvk_line_segment_detector_t> =
        raw ?: throw IllegalStateException("LineSegmentDetector is closed")

    override fun detect(image: Mat): LineSegments = memScoped {
        val width = alloc<CPointerVar<cvk_mat_t>>()
        val prec = alloc<CPointerVar<cvk_mat_t>>()
        val nfa = alloc<CPointerVar<cvk_mat_t>>()
        val lines = cvk_line_segment_detector_detect(
            check(), image.nativeHandle(), width.ptr, prec.ptr, nfa.ptr,
        )
        LineSegments(
            lines = nativeMat(lines, "LineSegmentDetector.detect"),
            width = nativeMat(width.value, "LineSegmentDetector.detect"),
            prec = nativeMat(prec.value, "LineSegmentDetector.detect"),
            nfa = nativeMat(nfa.value, "LineSegmentDetector.detect"),
        )
    }

    override fun drawSegments(image: Mat, segments: Mat) {
        cvk_line_segment_detector_draw_segments(check(), image.nativeHandle(), segments.nativeHandle())
    }

    override fun compareSegments(size: Size, lines1: Mat, lines2: Mat, image: Mat?): Int =
        cvk_line_segment_detector_compare_segments(
            check(), size.width, size.height,
            lines1.nativeHandle(), lines2.nativeHandle(), image?.nativeHandle(),
        )

    override fun clear() {
        cvk_line_segment_detector_clear(check())
    }

    override fun empty(): Boolean = cvk_line_segment_detector_empty(check()) != 0

    override fun save(filename: String) {
        cvk_line_segment_detector_save(check(), filename)
    }

    override fun getDefaultName(): String =
        cvk_line_segment_detector_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_line_segment_detector_release(handle)
    }
}

actual fun createLineSegmentDetector(
    refine: Int,
    scale: Double,
    sigmaScale: Double,
    quant: Double,
    angTh: Double,
    logEps: Double,
    densityTh: Double,
    nBins: Int,
): LineSegmentDetector =
    NativeLineSegmentDetector(
        cvk_line_segment_detector_create(
            refine, scale, sigmaScale, quant, angTh, logEps, densityTh, nBins,
        ) ?: throw OpenCVException("createLineSegmentDetector", lastNativeError()),
    )

// =========================================================================
// GeneralizedHoughBallard
// =========================================================================

/** Native (cinterop) [GeneralizedHoughBallard] over a `cv::Ptr<cv::GeneralizedHoughBallard>`. */
internal class NativeGeneralizedHoughBallard(
    @Volatile private var raw: CPointer<cvk_gh_ballard_t>?,
) : GeneralizedHoughBallard {

    private fun check(): CPointer<cvk_gh_ballard_t> =
        raw ?: throw IllegalStateException("GeneralizedHoughBallard is closed")

    override fun setTemplate(templ: Mat, templCenter: Point) {
        cvk_gh_ballard_set_template(
            check(), templ.nativeHandle(), templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun setTemplate(edges: Mat, dx: Mat, dy: Mat, templCenter: Point) {
        cvk_gh_ballard_set_template_edges(
            check(), edges.nativeHandle(), dx.nativeHandle(), dy.nativeHandle(),
            templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun detect(image: Mat): GeneralizedHoughResult = memScoped {
        val votes = alloc<CPointerVar<cvk_mat_t>>()
        val positions = cvk_gh_ballard_detect(check(), image.nativeHandle(), votes.ptr)
        GeneralizedHoughResult(
            positions = nativeMat(positions, "GeneralizedHough.detect"),
            votes = nativeMat(votes.value, "GeneralizedHough.detect"),
        )
    }

    override fun detect(edges: Mat, dx: Mat, dy: Mat): GeneralizedHoughResult = memScoped {
        val votes = alloc<CPointerVar<cvk_mat_t>>()
        val positions = cvk_gh_ballard_detect_edges(
            check(), edges.nativeHandle(), dx.nativeHandle(), dy.nativeHandle(), votes.ptr,
        )
        GeneralizedHoughResult(
            positions = nativeMat(positions, "GeneralizedHough.detect"),
            votes = nativeMat(votes.value, "GeneralizedHough.detect"),
        )
    }

    override var cannyLowThresh: Int
        get() = cvk_gh_ballard_get_canny_low_thresh(check())
        set(value) { cvk_gh_ballard_set_canny_low_thresh(check(), value) }

    override var cannyHighThresh: Int
        get() = cvk_gh_ballard_get_canny_high_thresh(check())
        set(value) { cvk_gh_ballard_set_canny_high_thresh(check(), value) }

    override var minDist: Double
        get() = cvk_gh_ballard_get_min_dist(check())
        set(value) { cvk_gh_ballard_set_min_dist(check(), value) }

    override var dp: Double
        get() = cvk_gh_ballard_get_dp(check())
        set(value) { cvk_gh_ballard_set_dp(check(), value) }

    override var maxBufferSize: Int
        get() = cvk_gh_ballard_get_max_buffer_size(check())
        set(value) { cvk_gh_ballard_set_max_buffer_size(check(), value) }

    override var levels: Int
        get() = cvk_gh_ballard_get_levels(check())
        set(value) { cvk_gh_ballard_set_levels(check(), value) }

    override var votesThreshold: Int
        get() = cvk_gh_ballard_get_votes_threshold(check())
        set(value) { cvk_gh_ballard_set_votes_threshold(check(), value) }

    override fun clear() {
        cvk_gh_ballard_clear(check())
    }

    override fun empty(): Boolean = cvk_gh_ballard_empty(check()) != 0

    override fun save(filename: String) {
        cvk_gh_ballard_save(check(), filename)
    }

    override fun getDefaultName(): String =
        cvk_gh_ballard_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_gh_ballard_release(handle)
    }
}

actual fun createGeneralizedHoughBallard(): GeneralizedHoughBallard =
    NativeGeneralizedHoughBallard(
        cvk_gh_ballard_create() ?: throw OpenCVException("createGeneralizedHoughBallard", lastNativeError()),
    )

// =========================================================================
// GeneralizedHoughGuil
// =========================================================================

/** Native (cinterop) [GeneralizedHoughGuil] over a `cv::Ptr<cv::GeneralizedHoughGuil>`. */
internal class NativeGeneralizedHoughGuil(
    @Volatile private var raw: CPointer<cvk_gh_guil_t>?,
) : GeneralizedHoughGuil {

    private fun check(): CPointer<cvk_gh_guil_t> =
        raw ?: throw IllegalStateException("GeneralizedHoughGuil is closed")

    override fun setTemplate(templ: Mat, templCenter: Point) {
        cvk_gh_guil_set_template(
            check(), templ.nativeHandle(), templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun setTemplate(edges: Mat, dx: Mat, dy: Mat, templCenter: Point) {
        cvk_gh_guil_set_template_edges(
            check(), edges.nativeHandle(), dx.nativeHandle(), dy.nativeHandle(),
            templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun detect(image: Mat): GeneralizedHoughResult = memScoped {
        val votes = alloc<CPointerVar<cvk_mat_t>>()
        val positions = cvk_gh_guil_detect(check(), image.nativeHandle(), votes.ptr)
        GeneralizedHoughResult(
            positions = nativeMat(positions, "GeneralizedHough.detect"),
            votes = nativeMat(votes.value, "GeneralizedHough.detect"),
        )
    }

    override fun detect(edges: Mat, dx: Mat, dy: Mat): GeneralizedHoughResult = memScoped {
        val votes = alloc<CPointerVar<cvk_mat_t>>()
        val positions = cvk_gh_guil_detect_edges(
            check(), edges.nativeHandle(), dx.nativeHandle(), dy.nativeHandle(), votes.ptr,
        )
        GeneralizedHoughResult(
            positions = nativeMat(positions, "GeneralizedHough.detect"),
            votes = nativeMat(votes.value, "GeneralizedHough.detect"),
        )
    }

    override var cannyLowThresh: Int
        get() = cvk_gh_guil_get_canny_low_thresh(check())
        set(value) { cvk_gh_guil_set_canny_low_thresh(check(), value) }

    override var cannyHighThresh: Int
        get() = cvk_gh_guil_get_canny_high_thresh(check())
        set(value) { cvk_gh_guil_set_canny_high_thresh(check(), value) }

    override var minDist: Double
        get() = cvk_gh_guil_get_min_dist(check())
        set(value) { cvk_gh_guil_set_min_dist(check(), value) }

    override var dp: Double
        get() = cvk_gh_guil_get_dp(check())
        set(value) { cvk_gh_guil_set_dp(check(), value) }

    override var maxBufferSize: Int
        get() = cvk_gh_guil_get_max_buffer_size(check())
        set(value) { cvk_gh_guil_set_max_buffer_size(check(), value) }

    override var xi: Double
        get() = cvk_gh_guil_get_xi(check())
        set(value) { cvk_gh_guil_set_xi(check(), value) }

    override var levels: Int
        get() = cvk_gh_guil_get_levels(check())
        set(value) { cvk_gh_guil_set_levels(check(), value) }

    override var angleEpsilon: Double
        get() = cvk_gh_guil_get_angle_epsilon(check())
        set(value) { cvk_gh_guil_set_angle_epsilon(check(), value) }

    override var minAngle: Double
        get() = cvk_gh_guil_get_min_angle(check())
        set(value) { cvk_gh_guil_set_min_angle(check(), value) }

    override var maxAngle: Double
        get() = cvk_gh_guil_get_max_angle(check())
        set(value) { cvk_gh_guil_set_max_angle(check(), value) }

    override var angleStep: Double
        get() = cvk_gh_guil_get_angle_step(check())
        set(value) { cvk_gh_guil_set_angle_step(check(), value) }

    override var angleThresh: Int
        get() = cvk_gh_guil_get_angle_thresh(check())
        set(value) { cvk_gh_guil_set_angle_thresh(check(), value) }

    override var minScale: Double
        get() = cvk_gh_guil_get_min_scale(check())
        set(value) { cvk_gh_guil_set_min_scale(check(), value) }

    override var maxScale: Double
        get() = cvk_gh_guil_get_max_scale(check())
        set(value) { cvk_gh_guil_set_max_scale(check(), value) }

    override var scaleStep: Double
        get() = cvk_gh_guil_get_scale_step(check())
        set(value) { cvk_gh_guil_set_scale_step(check(), value) }

    override var scaleThresh: Int
        get() = cvk_gh_guil_get_scale_thresh(check())
        set(value) { cvk_gh_guil_set_scale_thresh(check(), value) }

    override var posThresh: Int
        get() = cvk_gh_guil_get_pos_thresh(check())
        set(value) { cvk_gh_guil_set_pos_thresh(check(), value) }

    override fun clear() {
        cvk_gh_guil_clear(check())
    }

    override fun empty(): Boolean = cvk_gh_guil_empty(check()) != 0

    override fun save(filename: String) {
        cvk_gh_guil_save(check(), filename)
    }

    override fun getDefaultName(): String =
        cvk_gh_guil_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_gh_guil_release(handle)
    }
}

actual fun createGeneralizedHoughGuil(): GeneralizedHoughGuil =
    NativeGeneralizedHoughGuil(
        cvk_gh_guil_create() ?: throw OpenCVException("createGeneralizedHoughGuil", lastNativeError()),
    )

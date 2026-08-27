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

import cvk.cvk_last_error
import cvk.cvk_rect
import cvk.cvk_tracker_dasiamrpn_create
import cvk.cvk_tracker_get_tracking_score
import cvk.cvk_tracker_init
import cvk.cvk_tracker_mil_create
import cvk.cvk_tracker_nano_create
import cvk.cvk_tracker_release
import cvk.cvk_tracker_t
import cvk.cvk_tracker_update
import cvk.cvk_tracker_vit_create
import kotlin.concurrent.Volatile
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CValue
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.cValue
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.useContents
import kotlinx.cinterop.toKString

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

private fun Rect.toCvkRect(): CValue<cvk_rect> = cValue<cvk_rect> {
    x = this@toCvkRect.x
    y = this@toCvkRect.y
    width = this@toCvkRect.width
    height = this@toCvkRect.height
}

// =========================================================================
// Native (cinterop) Tracker implementation
// =========================================================================

internal class NativeTracker(
    @Volatile private var raw: CPointer<cvk_tracker_t>?,
) : TrackerMIL, TrackerDaSiamRPN, TrackerNano, TrackerVit {

    private fun check(): CPointer<cvk_tracker_t> =
        raw ?: throw IllegalStateException("Tracker is closed")

    override fun init(image: Mat, boundingBox: Rect) {
        cvk_tracker_init(check(), image.nativeHandle(), boundingBox.toCvkRect())
    }

    override fun update(image: Mat): TrackerUpdate = memScoped {
        val out = alloc<cvk_rect>()
        val ok = cvk_tracker_update(check(), image.nativeHandle(), out.ptr)
        TrackerUpdate(ok != 0, Rect(out.x, out.y, out.width, out.height))
    }

    override val trackingScore: Float
        get() = cvk_tracker_get_tracking_score(check())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_tracker_release(h)
    }
}

// =========================================================================
// actual factories
// =========================================================================

actual fun createTrackerMIL(parameters: TrackerMILParams): TrackerMIL =
    NativeTracker(
        cvk_tracker_mil_create(
            parameters.samplerInitInRadius,
            parameters.samplerInitMaxNegNum,
            parameters.samplerSearchWinSize,
            parameters.samplerTrackInRadius,
            parameters.samplerTrackMaxPosNum,
            parameters.samplerTrackMaxNegNum,
            parameters.featureSetNumFeatures,
        ) ?: throw OpenCVException("createTrackerMIL", lastNativeError()),
    )

actual fun createTrackerDaSiamRPN(parameters: TrackerDaSiamRPNParams): TrackerDaSiamRPN =
    NativeTracker(
        cvk_tracker_dasiamrpn_create(
            parameters.model,
            parameters.kernelCls1,
            parameters.kernelR1,
            parameters.backend,
            parameters.target,
        ) ?: throw OpenCVException("createTrackerDaSiamRPN", lastNativeError()),
    )

actual fun createTrackerNano(parameters: TrackerNanoParams): TrackerNano =
    NativeTracker(
        cvk_tracker_nano_create(
            parameters.backbone,
            parameters.neckhead,
            parameters.backend,
            parameters.target,
        ) ?: throw OpenCVException("createTrackerNano", lastNativeError()),
    )

actual fun createTrackerVit(parameters: TrackerVitParams): TrackerVit =
    NativeTracker(
        cvk_tracker_vit_create(
            parameters.net,
            parameters.backend,
            parameters.target,
            parameters.meanvalue.toCvk(),
            parameters.stdvalue.toCvk(),
            parameters.trackingScoreThreshold,
        ) ?: throw OpenCVException("createTrackerVit", lastNativeError()),
    )

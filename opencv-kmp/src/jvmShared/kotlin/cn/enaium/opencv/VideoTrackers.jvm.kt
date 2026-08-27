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

private fun lastNativeError(): String? = Jni.lastError()

// =========================================================================
// JVM (JNI-backed) Tracker implementation
// =========================================================================

internal class JvmTracker(private var handle: Long) : TrackerMIL, TrackerDaSiamRPN, TrackerNano, TrackerVit {

    private fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("Tracker is closed")

    override fun init(image: Mat, boundingBox: Rect) {
        JniVideo2.trackerInit(
            check(),
            handleOf(image),
            boundingBox.x,
            boundingBox.y,
            boundingBox.width,
            boundingBox.height,
        )
    }

    override fun update(image: Mat): TrackerUpdate {
        val out = IntArray(4)
        val ok = JniVideo2.trackerUpdate(check(), handleOf(image), out)
        return TrackerUpdate(ok, Rect(out[0], out[1], out[2], out[3]))
    }

    override val trackingScore: Float
        get() = JniVideo2.trackerGetTrackingScore(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniVideo2.trackerRelease(h)
        }
    }
}

// =========================================================================
// actual factories
// =========================================================================

actual fun createTrackerMIL(parameters: TrackerMILParams): TrackerMIL =
    JvmTracker(
        JniVideo2.trackerMILCreate(
            parameters.samplerInitInRadius,
            parameters.samplerInitMaxNegNum,
            parameters.samplerSearchWinSize,
            parameters.samplerTrackInRadius,
            parameters.samplerTrackMaxPosNum,
            parameters.samplerTrackMaxNegNum,
            parameters.featureSetNumFeatures,
        ).takeIf { it != 0L } ?: throw OpenCVException("createTrackerMIL", lastNativeError()),
    )

actual fun createTrackerDaSiamRPN(parameters: TrackerDaSiamRPNParams): TrackerDaSiamRPN =
    JvmTracker(
        JniVideo2.trackerDaSiamRPNCreate(
            parameters.model,
            parameters.kernelCls1,
            parameters.kernelR1,
            parameters.backend,
            parameters.target,
        ).takeIf { it != 0L } ?: throw OpenCVException("createTrackerDaSiamRPN", lastNativeError()),
    )

actual fun createTrackerNano(parameters: TrackerNanoParams): TrackerNano =
    JvmTracker(
        JniVideo2.trackerNanoCreate(
            parameters.backbone,
            parameters.neckhead,
            parameters.backend,
            parameters.target,
        ).takeIf { it != 0L } ?: throw OpenCVException("createTrackerNano", lastNativeError()),
    )

actual fun createTrackerVit(parameters: TrackerVitParams): TrackerVit =
    JvmTracker(
        JniVideo2.trackerVitCreate(
            parameters.net,
            parameters.backend,
            parameters.target,
            parameters.meanvalue.v0,
            parameters.meanvalue.v1,
            parameters.meanvalue.v2,
            parameters.meanvalue.v3,
            parameters.stdvalue.v0,
            parameters.stdvalue.v1,
            parameters.stdvalue.v2,
            parameters.stdvalue.v3,
            parameters.trackingScoreThreshold,
        ).takeIf { it != 0L } ?: throw OpenCVException("createTrackerVit", lastNativeError()),
    )

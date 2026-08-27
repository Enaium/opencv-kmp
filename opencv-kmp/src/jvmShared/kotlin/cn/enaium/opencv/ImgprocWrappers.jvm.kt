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
// JVM (JNI-backed) implementation of the imgproc2 object wrappers.
// Handles are jlong pointers to the cv::Ptr<> objects owned by the native
// side; every call forwards through [JniImgproc2] to the same cvk_ shim the
// native targets bind via cinterop.
// =========================================================================

private fun lastNativeError(): String? = Jni.lastError()


// =========================================================================
// LineSegmentDetector
// =========================================================================

/** JNI-backed [LineSegmentDetector] wrapping a `cv::Ptr<cv::LineSegmentDetector>` handle. */
internal class JvmLineSegmentDetector(private var handle: Long) : LineSegmentDetector {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("LineSegmentDetector is closed")

    override fun detect(image: Mat): LineSegments {
        val out = JniImgproc2.lsdDetect(check(), handleOf(image))
        return LineSegments(
            lines = jvmMat(out[0], "LineSegmentDetector.detect"),
            width = jvmMat(out[1], "LineSegmentDetector.detect"),
            prec = jvmMat(out[2], "LineSegmentDetector.detect"),
            nfa = jvmMat(out[3], "LineSegmentDetector.detect"),
        )
    }

    override fun drawSegments(image: Mat, segments: Mat) {
        JniImgproc2.lsdDrawSegments(check(), handleOf(image), handleOf(segments))
    }

    override fun compareSegments(size: Size, lines1: Mat, lines2: Mat, image: Mat?): Int =
        JniImgproc2.lsdCompareSegments(
            check(), size.width, size.height,
            handleOf(lines1), handleOf(lines2), image?.let(::handleOf) ?: 0L,
        )

    override fun clear() {
        JniImgproc2.lsdClear(check())
    }

    override fun empty(): Boolean = JniImgproc2.lsdEmpty(check())

    override fun save(filename: String) {
        JniImgproc2.lsdSave(check(), filename)
    }

    override fun getDefaultName(): String = JniImgproc2.lsdGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniImgproc2.lsdRelease(h)
        }
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
    JvmLineSegmentDetector(
        JniImgproc2.lsdCreate(refine, scale, sigmaScale, quant, angTh, logEps, densityTh, nBins),
    )

// =========================================================================
// GeneralizedHoughBallard
// =========================================================================

/** JNI-backed [GeneralizedHoughBallard] wrapping a `cv::Ptr<cv::GeneralizedHoughBallard>` handle. */
internal class JvmGeneralizedHoughBallard(private var handle: Long) : GeneralizedHoughBallard {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("GeneralizedHoughBallard is closed")

    override fun setTemplate(templ: Mat, templCenter: Point) {
        JniImgproc2.ghBallardSetTemplate(
            check(), handleOf(templ), templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun setTemplate(edges: Mat, dx: Mat, dy: Mat, templCenter: Point) {
        JniImgproc2.ghBallardSetTemplateEdges(
            check(), handleOf(edges), handleOf(dx), handleOf(dy),
            templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun detect(image: Mat): GeneralizedHoughResult {
        val out = JniImgproc2.ghBallardDetect(check(), handleOf(image))
        return GeneralizedHoughResult(
            positions = jvmMat(out[0], "GeneralizedHough.detect"),
            votes = jvmMat(out[1], "GeneralizedHough.detect"),
        )
    }

    override fun detect(edges: Mat, dx: Mat, dy: Mat): GeneralizedHoughResult {
        val out = JniImgproc2.ghBallardDetectEdges(check(), handleOf(edges), handleOf(dx), handleOf(dy))
        return GeneralizedHoughResult(
            positions = jvmMat(out[0], "GeneralizedHough.detect"),
            votes = jvmMat(out[1], "GeneralizedHough.detect"),
        )
    }

    override var cannyLowThresh: Int
        get() = JniImgproc2.ghBallardGetCannyLowThresh(check())
        set(value) { JniImgproc2.ghBallardSetCannyLowThresh(check(), value) }

    override var cannyHighThresh: Int
        get() = JniImgproc2.ghBallardGetCannyHighThresh(check())
        set(value) { JniImgproc2.ghBallardSetCannyHighThresh(check(), value) }

    override var minDist: Double
        get() = JniImgproc2.ghBallardGetMinDist(check())
        set(value) { JniImgproc2.ghBallardSetMinDist(check(), value) }

    override var dp: Double
        get() = JniImgproc2.ghBallardGetDp(check())
        set(value) { JniImgproc2.ghBallardSetDp(check(), value) }

    override var maxBufferSize: Int
        get() = JniImgproc2.ghBallardGetMaxBufferSize(check())
        set(value) { JniImgproc2.ghBallardSetMaxBufferSize(check(), value) }

    override var levels: Int
        get() = JniImgproc2.ghBallardGetLevels(check())
        set(value) { JniImgproc2.ghBallardSetLevels(check(), value) }

    override var votesThreshold: Int
        get() = JniImgproc2.ghBallardGetVotesThreshold(check())
        set(value) { JniImgproc2.ghBallardSetVotesThreshold(check(), value) }

    override fun clear() {
        JniImgproc2.ghBallardClear(check())
    }

    override fun empty(): Boolean = JniImgproc2.ghBallardEmpty(check())

    override fun save(filename: String) {
        JniImgproc2.ghBallardSave(check(), filename)
    }

    override fun getDefaultName(): String = JniImgproc2.ghBallardGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniImgproc2.ghBallardRelease(h)
        }
    }
}

actual fun createGeneralizedHoughBallard(): GeneralizedHoughBallard =
    JvmGeneralizedHoughBallard(JniImgproc2.ghBallardCreate())

// =========================================================================
// GeneralizedHoughGuil
// =========================================================================

/** JNI-backed [GeneralizedHoughGuil] wrapping a `cv::Ptr<cv::GeneralizedHoughGuil>` handle. */
internal class JvmGeneralizedHoughGuil(private var handle: Long) : GeneralizedHoughGuil {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("GeneralizedHoughGuil is closed")

    override fun setTemplate(templ: Mat, templCenter: Point) {
        JniImgproc2.ghGuilSetTemplate(
            check(), handleOf(templ), templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun setTemplate(edges: Mat, dx: Mat, dy: Mat, templCenter: Point) {
        JniImgproc2.ghGuilSetTemplateEdges(
            check(), handleOf(edges), handleOf(dx), handleOf(dy),
            templCenter.x.toDouble(), templCenter.y.toDouble(),
        )
    }

    override fun detect(image: Mat): GeneralizedHoughResult {
        val out = JniImgproc2.ghGuilDetect(check(), handleOf(image))
        return GeneralizedHoughResult(
            positions = jvmMat(out[0], "GeneralizedHough.detect"),
            votes = jvmMat(out[1], "GeneralizedHough.detect"),
        )
    }

    override fun detect(edges: Mat, dx: Mat, dy: Mat): GeneralizedHoughResult {
        val out = JniImgproc2.ghGuilDetectEdges(check(), handleOf(edges), handleOf(dx), handleOf(dy))
        return GeneralizedHoughResult(
            positions = jvmMat(out[0], "GeneralizedHough.detect"),
            votes = jvmMat(out[1], "GeneralizedHough.detect"),
        )
    }

    override var cannyLowThresh: Int
        get() = JniImgproc2.ghGuilGetCannyLowThresh(check())
        set(value) { JniImgproc2.ghGuilSetCannyLowThresh(check(), value) }

    override var cannyHighThresh: Int
        get() = JniImgproc2.ghGuilGetCannyHighThresh(check())
        set(value) { JniImgproc2.ghGuilSetCannyHighThresh(check(), value) }

    override var minDist: Double
        get() = JniImgproc2.ghGuilGetMinDist(check())
        set(value) { JniImgproc2.ghGuilSetMinDist(check(), value) }

    override var dp: Double
        get() = JniImgproc2.ghGuilGetDp(check())
        set(value) { JniImgproc2.ghGuilSetDp(check(), value) }

    override var maxBufferSize: Int
        get() = JniImgproc2.ghGuilGetMaxBufferSize(check())
        set(value) { JniImgproc2.ghGuilSetMaxBufferSize(check(), value) }

    override var xi: Double
        get() = JniImgproc2.ghGuilGetXi(check())
        set(value) { JniImgproc2.ghGuilSetXi(check(), value) }

    override var levels: Int
        get() = JniImgproc2.ghGuilGetLevels(check())
        set(value) { JniImgproc2.ghGuilSetLevels(check(), value) }

    override var angleEpsilon: Double
        get() = JniImgproc2.ghGuilGetAngleEpsilon(check())
        set(value) { JniImgproc2.ghGuilSetAngleEpsilon(check(), value) }

    override var minAngle: Double
        get() = JniImgproc2.ghGuilGetMinAngle(check())
        set(value) { JniImgproc2.ghGuilSetMinAngle(check(), value) }

    override var maxAngle: Double
        get() = JniImgproc2.ghGuilGetMaxAngle(check())
        set(value) { JniImgproc2.ghGuilSetMaxAngle(check(), value) }

    override var angleStep: Double
        get() = JniImgproc2.ghGuilGetAngleStep(check())
        set(value) { JniImgproc2.ghGuilSetAngleStep(check(), value) }

    override var angleThresh: Int
        get() = JniImgproc2.ghGuilGetAngleThresh(check())
        set(value) { JniImgproc2.ghGuilSetAngleThresh(check(), value) }

    override var minScale: Double
        get() = JniImgproc2.ghGuilGetMinScale(check())
        set(value) { JniImgproc2.ghGuilSetMinScale(check(), value) }

    override var maxScale: Double
        get() = JniImgproc2.ghGuilGetMaxScale(check())
        set(value) { JniImgproc2.ghGuilSetMaxScale(check(), value) }

    override var scaleStep: Double
        get() = JniImgproc2.ghGuilGetScaleStep(check())
        set(value) { JniImgproc2.ghGuilSetScaleStep(check(), value) }

    override var scaleThresh: Int
        get() = JniImgproc2.ghGuilGetScaleThresh(check())
        set(value) { JniImgproc2.ghGuilSetScaleThresh(check(), value) }

    override var posThresh: Int
        get() = JniImgproc2.ghGuilGetPosThresh(check())
        set(value) { JniImgproc2.ghGuilSetPosThresh(check(), value) }

    override fun clear() {
        JniImgproc2.ghGuilClear(check())
    }

    override fun empty(): Boolean = JniImgproc2.ghGuilEmpty(check())

    override fun save(filename: String) {
        JniImgproc2.ghGuilSave(check(), filename)
    }

    override fun getDefaultName(): String = JniImgproc2.ghGuilGetDefaultName(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniImgproc2.ghGuilRelease(h)
        }
    }
}

actual fun createGeneralizedHoughGuil(): GeneralizedHoughGuil =
    JvmGeneralizedHoughGuil(JniImgproc2.ghGuilCreate())

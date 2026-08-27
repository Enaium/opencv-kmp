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
 * Result of [Tracker.update].
 *
 * [boundingBox] is always populated with the tracker's latest estimate
 * (the native side writes it on every call, even when the target was not
 * located); [success] tells whether that estimate is trustworthy. The SDK
 * mirrors this with a boolean return plus a `Rect` out-parameter that is
 * only defined when `true` is returned.
 */
data class TrackerUpdate(
    val success: Boolean,
    val boundingBox: Rect,
)

/**
 * Base abstract class for the long-term tracker (`cv::Tracker`).
 *
 * A tracker must be initialized once with [init] before [update] is called.
 */
interface Tracker : AutoCloseable {

    /**
     * Initializes the tracker with a known bounding box that surrounded the
     * target.
     *
     * @param image The initial frame
     * @param boundingBox The initial bounding box
     */
    fun init(image: Mat, boundingBox: Rect)

    /**
     * Updates the tracker, finding the new most likely bounding box for the
     * target.
     *
     * @param image The current frame
     * @return [TrackerUpdate.success] is `true` when the target was located and
     *   `false` when the tracker cannot locate the target in the current frame.
     *   Note that the latter *does not* imply the tracker has failed — the
     *   target may indeed be missing from the frame (say, out of sight).
     */
    fun update(image: Mat): TrackerUpdate

    /** Return tracking score (may be `-1` when the tracker provides none). */
    val trackingScore: Float

    /** Releases the native tracker handle; the tracker must not be used after. */
    override fun close()
}

/**
 * The MIL algorithm trains a classifier in an online manner to separate the
 * object from the background (`cv::TrackerMIL`).
 *
 * Multiple Instance Learning avoids the drift problem for robust tracking.
 * MIL is not DNN-based: it needs no model files and [createTrackerMIL] always
 * succeeds.
 */
interface TrackerMIL : Tracker

/** Parameters for [createTrackerMIL]; defaults match `cv::TrackerMIL::Params`. */
data class TrackerMILParams(
    /** Radius for gathering positive instances during init. */
    var samplerInitInRadius: Float = 3.0f,
    /** # negative samples to use during init. */
    var samplerInitMaxNegNum: Int = 65,
    /** Size of search window. */
    var samplerSearchWinSize: Float = 25.0f,
    /** Radius for gathering positive instances during tracking. */
    var samplerTrackInRadius: Float = 4.0f,
    /** # positive samples to use during tracking. */
    var samplerTrackMaxPosNum: Int = 100000,
    /** # negative samples to use during tracking. */
    var samplerTrackMaxNegNum: Int = 65,
    /** # features. */
    var featureSetNumFeatures: Int = 250,
)

/**
 * DaSiamRPN tracker — a DNN-based tracker (`cv::TrackerDaSiamRPN`).
 *
 * The [createTrackerDaSiamRPN] factory loads the three ONNX models given in
 * [TrackerDaSiamRPNParams]; it throws [OpenCVException] when the model files
 * cannot be found or read.
 */
interface TrackerDaSiamRPN : Tracker

/**
 * Parameters for [createTrackerDaSiamRPN]; defaults match
 * `cv::TrackerDaSiamRPN::Params`.
 *
 * The model files are resolved relative to the working directory at creation
 * time; the defaults are the file names shipped with OpenCV. Provide absolute
 * paths to the downloaded models
 * (https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_dasiamrpn).
 */
data class TrackerDaSiamRPNParams(
    /** Path to the SiamRPN model (`dasiamrpn_model.onnx`). */
    var model: String = "dasiamrpn_model.onnx",
    /** Path to the CLS kernel model (`dasiamrpn_kernel_cls1.onnx`). */
    var kernelCls1: String = "dasiamrpn_kernel_cls1.onnx",
    /** Path to the R1 kernel model (`dasiamrpn_kernel_r1.onnx`). */
    var kernelR1: String = "dasiamrpn_kernel_r1.onnx",
    /** dnn backend (`dnn::DNN_BACKEND_DEFAULT` = 0). */
    var backend: Int = 0,
    /** dnn target (`dnn::DNN_TARGET_CPU` = 0). */
    var target: Int = 0,
)

/**
 * The Nano tracker — a super lightweight DNN-based general object tracking
 * (`cv::TrackerNano`).
 *
 * Nano tracker is much faster and extremely lightweight due to its special
 * model structure; the whole model size is about 1.9 MB. It needs two models:
 * one for feature extraction (backbone) and another for localization
 * (neckhead). [createTrackerNano] throws [OpenCVException] when the model
 * files cannot be found or read.
 */
interface TrackerNano : Tracker

/**
 * Parameters for [createTrackerNano]; defaults match `cv::TrackerNano::Params`.
 *
 * Model download link:
 * https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
 */
data class TrackerNanoParams(
    /** Path to the backbone model (`backbone.onnx`). */
    var backbone: String = "backbone.onnx",
    /** Path to the neckhead model (`neckhead.onnx`). */
    var neckhead: String = "neckhead.onnx",
    /** dnn backend (`dnn::DNN_BACKEND_DEFAULT` = 0). */
    var backend: Int = 0,
    /** dnn target (`dnn::DNN_TARGET_CPU` = 0). */
    var target: Int = 0,
)

/**
 * The VIT tracker — a super lightweight DNN-based general object tracking
 * (`cv::TrackerVit`).
 *
 * VIT tracker is much faster and extremely lightweight due to its special
 * model structure; the model file is about 767 KB. [createTrackerVit] throws
 * [OpenCVException] when the model file cannot be found or read.
 */
interface TrackerVit : Tracker

/**
 * Parameters for [createTrackerVit]; defaults match `cv::TrackerVit::Params`.
 *
 * Model download link:
 * https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
 */
data class TrackerVitParams(
    /** Path to the VIT model (`vitTracker.onnx`). */
    var net: String = "vitTracker.onnx",
    /** dnn backend (`dnn::DNN_BACKEND_DEFAULT` = 0). */
    var backend: Int = 0,
    /** dnn target (`dnn::DNN_TARGET_CPU` = 0). */
    var target: Int = 0,
    /** Mean value for image preprocessing. */
    var meanvalue: Scalar = Scalar(0.485, 0.456, 0.406),
    /** Std value for image preprocessing. */
    var stdvalue: Scalar = Scalar(0.229, 0.224, 0.225),
    /** Threshold for the tracking score (`0.20`). */
    var trackingScoreThreshold: Float = 0.20f,
)

/**
 * Creates a MIL tracker instance (`cv::TrackerMIL::create`).
 *
 * @param parameters MIL parameters [TrackerMILParams]
 */
expect fun createTrackerMIL(parameters: TrackerMILParams = TrackerMILParams()): TrackerMIL

/**
 * Creates a DaSiamRPN tracker instance (`cv::TrackerDaSiamRPN::create`).
 *
 * Throws [OpenCVException] when the models configured in [parameters] cannot
 * be loaded.
 *
 * @param parameters DaSiamRPN parameters [TrackerDaSiamRPNParams]
 */
expect fun createTrackerDaSiamRPN(parameters: TrackerDaSiamRPNParams = TrackerDaSiamRPNParams()): TrackerDaSiamRPN

/**
 * Creates a Nano tracker instance (`cv::TrackerNano::create`).
 *
 * Throws [OpenCVException] when the models configured in [parameters] cannot
 * be loaded.
 *
 * @param parameters NanoTrack parameters [TrackerNanoParams]
 */
expect fun createTrackerNano(parameters: TrackerNanoParams = TrackerNanoParams()): TrackerNano

/**
 * Creates a VIT tracker instance (`cv::TrackerVit::create`).
 *
 * Throws [OpenCVException] when the model configured in [parameters] cannot
 * be loaded.
 *
 * @param parameters VIT tracker parameters [TrackerVitParams]
 */
expect fun createTrackerVit(parameters: TrackerVitParams = TrackerVitParams()): TrackerVit

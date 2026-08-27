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
 * JNI bridge for the features module.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniFeatures_<name>`
 * function in jni/jni_features.cpp. Detector handles and Mat handles travel
 * as jlong pointers; two-Mat results (compute/detectAndCompute/view params)
 * travel as jlong[2]; the MSER region flat buffer travels as jbyteArray with
 * the bounding-box Mat written into a caller-provided jlong[1].
 */
internal object JniFeatures {

    // SIFT

    external fun siftCreate(
        nfeatures: Int,
        nOctaveLayers: Int,
        contrastThreshold: Double,
        edgeThreshold: Double,
        sigma: Double,
        descriptorType: Int,
        enablePreciseUpscale: Boolean,
    ): Long

    external fun siftDetect(h: Long, image: Long, mask: Long): Long
    external fun siftCompute(h: Long, image: Long, keypoints: Long): LongArray
    external fun siftDetectAndCompute(h: Long, image: Long, mask: Long): LongArray
    external fun siftDescriptorSize(h: Long): Int
    external fun siftDescriptorType(h: Long): Int
    external fun siftDefaultNorm(h: Long): Int
    external fun siftWrite(h: Long, fileName: String)
    external fun siftRead(h: Long, fileName: String)
    external fun siftClear(h: Long)
    external fun siftEmpty(h: Long): Boolean
    external fun siftSave(h: Long, fileName: String)
    external fun siftGetDefaultName(h: Long): String?
    external fun siftRelease(h: Long)

    external fun siftSetNFeatures(h: Long, maxFeatures: Int)
    external fun siftGetNFeatures(h: Long): Int
    external fun siftSetNOctaveLayers(h: Long, nOctaveLayers: Int)
    external fun siftGetNOctaveLayers(h: Long): Int
    external fun siftSetContrastThreshold(h: Long, contrastThreshold: Double)
    external fun siftGetContrastThreshold(h: Long): Double
    external fun siftSetEdgeThreshold(h: Long, edgeThreshold: Double)
    external fun siftGetEdgeThreshold(h: Long): Double
    external fun siftSetSigma(h: Long, sigma: Double)
    external fun siftGetSigma(h: Long): Double

    // ORB

    external fun orbCreate(
        nfeatures: Int,
        scaleFactor: Float,
        nlevels: Int,
        edgeThreshold: Int,
        firstLevel: Int,
        wtaK: Int,
        scoreType: Int,
        patchSize: Int,
        fastThreshold: Int,
    ): Long

    external fun orbDetect(h: Long, image: Long, mask: Long): Long
    external fun orbCompute(h: Long, image: Long, keypoints: Long): LongArray
    external fun orbDetectAndCompute(h: Long, image: Long, mask: Long): LongArray
    external fun orbDescriptorSize(h: Long): Int
    external fun orbDescriptorType(h: Long): Int
    external fun orbDefaultNorm(h: Long): Int
    external fun orbWrite(h: Long, fileName: String)
    external fun orbRead(h: Long, fileName: String)
    external fun orbClear(h: Long)
    external fun orbEmpty(h: Long): Boolean
    external fun orbSave(h: Long, fileName: String)
    external fun orbGetDefaultName(h: Long): String?
    external fun orbRelease(h: Long)

    external fun orbSetMaxFeatures(h: Long, maxFeatures: Int)
    external fun orbGetMaxFeatures(h: Long): Int
    external fun orbSetScaleFactor(h: Long, scaleFactor: Double)
    external fun orbGetScaleFactor(h: Long): Double
    external fun orbSetNLevels(h: Long, nlevels: Int)
    external fun orbGetNLevels(h: Long): Int
    external fun orbSetEdgeThreshold(h: Long, edgeThreshold: Int)
    external fun orbGetEdgeThreshold(h: Long): Int
    external fun orbSetFirstLevel(h: Long, firstLevel: Int)
    external fun orbGetFirstLevel(h: Long): Int
    external fun orbSetWtaK(h: Long, wtaK: Int)
    external fun orbGetWtaK(h: Long): Int
    external fun orbSetScoreType(h: Long, scoreType: Int)
    external fun orbGetScoreType(h: Long): Int
    external fun orbSetPatchSize(h: Long, patchSize: Int)
    external fun orbGetPatchSize(h: Long): Int
    external fun orbSetFastThreshold(h: Long, fastThreshold: Int)
    external fun orbGetFastThreshold(h: Long): Int

    // MSER

    external fun mserCreate(
        delta: Int,
        minArea: Int,
        maxArea: Int,
        maxVariation: Double,
        minDiversity: Double,
        maxEvolution: Int,
        areaThreshold: Double,
        minMargin: Double,
        edgeBlurSize: Int,
    ): Long

    external fun mserDetect(h: Long, image: Long, mask: Long): Long
    external fun mserCompute(h: Long, image: Long, keypoints: Long): LongArray
    external fun mserDetectAndCompute(h: Long, image: Long, mask: Long): LongArray
    external fun mserDescriptorSize(h: Long): Int
    external fun mserDescriptorType(h: Long): Int
    external fun mserDefaultNorm(h: Long): Int
    external fun mserWrite(h: Long, fileName: String)
    external fun mserRead(h: Long, fileName: String)
    external fun mserClear(h: Long)
    external fun mserEmpty(h: Long): Boolean
    external fun mserSave(h: Long, fileName: String)
    external fun mserGetDefaultName(h: Long): String?
    external fun mserRelease(h: Long)

    external fun mserDetectRegions(h: Long, image: Long, out: LongArray): ByteArray

    external fun mserSetDelta(h: Long, delta: Int)
    external fun mserGetDelta(h: Long): Int
    external fun mserSetMinArea(h: Long, minArea: Int)
    external fun mserGetMinArea(h: Long): Int
    external fun mserSetMaxArea(h: Long, maxArea: Int)
    external fun mserGetMaxArea(h: Long): Int
    external fun mserSetMaxVariation(h: Long, maxVariation: Double)
    external fun mserGetMaxVariation(h: Long): Double
    external fun mserSetMinDiversity(h: Long, minDiversity: Double)
    external fun mserGetMinDiversity(h: Long): Double
    external fun mserSetMaxEvolution(h: Long, maxEvolution: Int)
    external fun mserGetMaxEvolution(h: Long): Int
    external fun mserSetAreaThreshold(h: Long, areaThreshold: Double)
    external fun mserGetAreaThreshold(h: Long): Double
    external fun mserSetMinMargin(h: Long, minMargin: Double)
    external fun mserGetMinMargin(h: Long): Double
    external fun mserSetEdgeBlurSize(h: Long, edgeBlurSize: Int)
    external fun mserGetEdgeBlurSize(h: Long): Int
    external fun mserSetPass2Only(h: Long, f: Boolean)
    external fun mserGetPass2Only(h: Long): Boolean

    // FastFeatureDetector

    external fun fastCreate(threshold: Int, nonmaxSuppression: Boolean, type: Int): Long

    external fun fastDetect(h: Long, image: Long, mask: Long): Long
    external fun fastCompute(h: Long, image: Long, keypoints: Long): LongArray
    external fun fastDetectAndCompute(h: Long, image: Long, mask: Long): LongArray
    external fun fastDescriptorSize(h: Long): Int
    external fun fastDescriptorType(h: Long): Int
    external fun fastDefaultNorm(h: Long): Int
    external fun fastWrite(h: Long, fileName: String)
    external fun fastRead(h: Long, fileName: String)
    external fun fastClear(h: Long)
    external fun fastEmpty(h: Long): Boolean
    external fun fastSave(h: Long, fileName: String)
    external fun fastGetDefaultName(h: Long): String?
    external fun fastRelease(h: Long)

    external fun fastSetThreshold(h: Long, threshold: Int)
    external fun fastGetThreshold(h: Long): Int
    external fun fastSetNonmaxSuppression(h: Long, f: Boolean)
    external fun fastGetNonmaxSuppression(h: Long): Boolean
    external fun fastSetType(h: Long, type: Int)
    external fun fastGetType(h: Long): Int

    // GFTTDetector

    external fun gfttCreate(
        maxCorners: Int,
        qualityLevel: Double,
        minDistance: Double,
        blockSize: Int,
        gradientSize: Int,
        useHarrisDetector: Boolean,
        k: Double,
    ): Long

    external fun gfttDetect(h: Long, image: Long, mask: Long): Long
    external fun gfttCompute(h: Long, image: Long, keypoints: Long): LongArray
    external fun gfttDetectAndCompute(h: Long, image: Long, mask: Long): LongArray
    external fun gfttDescriptorSize(h: Long): Int
    external fun gfttDescriptorType(h: Long): Int
    external fun gfttDefaultNorm(h: Long): Int
    external fun gfttWrite(h: Long, fileName: String)
    external fun gfttRead(h: Long, fileName: String)
    external fun gfttClear(h: Long)
    external fun gfttEmpty(h: Long): Boolean
    external fun gfttSave(h: Long, fileName: String)
    external fun gfttGetDefaultName(h: Long): String?
    external fun gfttRelease(h: Long)

    external fun gfttSetMaxFeatures(h: Long, maxFeatures: Int)
    external fun gfttGetMaxFeatures(h: Long): Int
    external fun gfttSetQualityLevel(h: Long, qlevel: Double)
    external fun gfttGetQualityLevel(h: Long): Double
    external fun gfttSetMinDistance(h: Long, minDistance: Double)
    external fun gfttGetMinDistance(h: Long): Double
    external fun gfttSetBlockSize(h: Long, blockSize: Int)
    external fun gfttGetBlockSize(h: Long): Int
    external fun gfttSetGradientSize(h: Long, gradientSize: Int)
    external fun gfttGetGradientSize(h: Long): Int
    external fun gfttSetHarrisDetector(h: Long, harrisDetector: Boolean)
    external fun gfttGetHarrisDetector(h: Long): Boolean
    external fun gfttSetK(h: Long, k: Double)
    external fun gfttGetK(h: Long): Double

    // SimpleBlobDetector

    external fun simpleBlobDetectorCreate(
        thresholdStep: Float,
        minThreshold: Float,
        maxThreshold: Float,
        minRepeatability: Long,
        minDistBetweenBlobs: Float,
        filterByColor: Int,
        blobColor: Int,
        filterByArea: Int,
        minArea: Float,
        maxArea: Float,
        filterByCircularity: Int,
        minCircularity: Float,
        maxCircularity: Float,
        filterByInertia: Int,
        minInertiaRatio: Float,
        maxInertiaRatio: Float,
        filterByConvexity: Int,
        minConvexity: Float,
        maxConvexity: Float,
        collectContours: Int,
    ): Long

    external fun simpleBlobDetectorDetect(h: Long, image: Long, mask: Long): Long
    external fun simpleBlobDetectorCompute(h: Long, image: Long, keypoints: Long): LongArray
    external fun simpleBlobDetectorDetectAndCompute(h: Long, image: Long, mask: Long): LongArray
    external fun simpleBlobDetectorDescriptorSize(h: Long): Int
    external fun simpleBlobDetectorDescriptorType(h: Long): Int
    external fun simpleBlobDetectorDefaultNorm(h: Long): Int
    external fun simpleBlobDetectorWrite(h: Long, fileName: String)
    external fun simpleBlobDetectorRead(h: Long, fileName: String)
    external fun simpleBlobDetectorClear(h: Long)
    external fun simpleBlobDetectorEmpty(h: Long): Boolean
    external fun simpleBlobDetectorSave(h: Long, fileName: String)
    external fun simpleBlobDetectorGetDefaultName(h: Long): String?
    external fun simpleBlobDetectorRelease(h: Long)

    external fun simpleBlobDetectorSetParams(
        h: Long,
        thresholdStep: Float,
        minThreshold: Float,
        maxThreshold: Float,
        minRepeatability: Long,
        minDistBetweenBlobs: Float,
        filterByColor: Int,
        blobColor: Int,
        filterByArea: Int,
        minArea: Float,
        maxArea: Float,
        filterByCircularity: Int,
        minCircularity: Float,
        maxCircularity: Float,
        filterByInertia: Int,
        minInertiaRatio: Float,
        maxInertiaRatio: Float,
        filterByConvexity: Int,
        minConvexity: Float,
        maxConvexity: Float,
        collectContours: Int,
    )

    external fun simpleBlobDetectorGetParams(h: Long): DoubleArray
    external fun simpleBlobDetectorGetBlobContours(h: Long): ByteArray

    // AffineFeature

    external fun affineCreate(backend: Long, maxTilt: Int, minTilt: Int, tiltStep: Float, rotateStepBase: Float): Long

    external fun affineDetect(h: Long, image: Long, mask: Long): Long
    external fun affineCompute(h: Long, image: Long, keypoints: Long): LongArray
    external fun affineDetectAndCompute(h: Long, image: Long, mask: Long): LongArray
    external fun affineDescriptorSize(h: Long): Int
    external fun affineDescriptorType(h: Long): Int
    external fun affineDefaultNorm(h: Long): Int
    external fun affineWrite(h: Long, fileName: String)
    external fun affineRead(h: Long, fileName: String)
    external fun affineClear(h: Long)
    external fun affineEmpty(h: Long): Boolean
    external fun affineSave(h: Long, fileName: String)
    external fun affineGetDefaultName(h: Long): String?
    external fun affineRelease(h: Long)

    external fun affineSetViewParams(h: Long, tilts: Long, rolls: Long)
    external fun affineGetViewParams(h: Long): LongArray
}

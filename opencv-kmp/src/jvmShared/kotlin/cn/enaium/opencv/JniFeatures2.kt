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
 * JNI bridge for the "features2" slice (descriptor matchers, ANNIndex,
 * ALIKED/DISK, Features statics).
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniFeatures2_<name>`
 * function in jni/jni_features2.cpp. Handles travel as jlong pointers; the
 * scalar/rect structs are expanded into primitive arguments.
 */
internal object JniFeatures2 {

    // ---- factories ------------------------------------------------------

    external fun descriptorMatcherCreate(type: String): Long

    external fun descriptorMatcherCreateType(type: Int): Long

    external fun bfMatcherCreate(normType: Int, crossCheck: Boolean): Long

    external fun flannMatcherCreate(indexParams: String): Long

    external fun lightGlueMatcherCreate(
        modelPath: String,
        scoreThreshold: Float,
        backend: Int,
        target: Int,
    ): Long

    external fun lightGlueMatcherCreateFromMemory(
        modelData: ByteArray,
        scoreThreshold: Float,
        backend: Int,
        target: Int,
    ): Long

    external fun descriptorMatcherClone(h: Long, emptyTrainData: Boolean): Long

    // ---- train collection ----------------------------------------------

    external fun descriptorMatcherAdd(h: Long, descriptorsWire: Long)

    external fun descriptorMatcherGetTrainDescriptors(h: Long): Long

    external fun descriptorMatcherIsMaskSupported(h: Long): Boolean

    external fun descriptorMatcherTrain(h: Long)

    external fun descriptorMatcherWrite(h: Long, fileName: String)

    external fun descriptorMatcherRead(h: Long, fileName: String)

    // ---- matching ------------------------------------------------------

    external fun descriptorMatcherMatchTrain(h: Long, query: Long, train: Long, mask: Long): Long

    external fun descriptorMatcherMatch(h: Long, query: Long, masksWire: Long): Long

    external fun descriptorMatcherKnnMatchTrain(
        h: Long,
        query: Long,
        train: Long,
        k: Int,
        mask: Long,
        compactResult: Boolean,
    ): Long

    external fun descriptorMatcherKnnMatch(
        h: Long,
        query: Long,
        k: Int,
        masksWire: Long,
        compactResult: Boolean,
    ): Long

    external fun descriptorMatcherRadiusMatchTrain(
        h: Long,
        query: Long,
        train: Long,
        maxDistance: Float,
        mask: Long,
        compactResult: Boolean,
    ): Long

    external fun descriptorMatcherRadiusMatch(
        h: Long,
        query: Long,
        maxDistance: Float,
        masksWire: Long,
        compactResult: Boolean,
    ): Long

    // ---- Algorithm surface ---------------------------------------------

    external fun descriptorMatcherClear(h: Long)

    external fun descriptorMatcherEmpty(h: Long): Boolean

    external fun descriptorMatcherSave(h: Long, fileName: String)

    external fun descriptorMatcherGetDefaultName(h: Long): String

    external fun descriptorMatcherRelease(h: Long)

    // ---- LightGlueMatcher extras ---------------------------------------

    external fun lightGlueMatcherSetPairInfo(
        h: Long,
        queryKpts: Long,
        trainKpts: Long,
        queryWidth: Double,
        queryHeight: Double,
        trainWidth: Double,
        trainHeight: Double,
    )

    external fun lightGlueMatcherClearPairInfo(h: Long)

    // ---- ANNIndex ------------------------------------------------------

    external fun annIndexCreate(dim: Int, distType: Int): Long

    external fun annIndexAddItems(h: Long, features: Long)

    external fun annIndexBuild(h: Long, trees: Int)

    external fun annIndexKnnSearch(h: Long, query: Long, knn: Int, searchK: Int): LongArray

    external fun annIndexSave(h: Long, filename: String, prefault: Boolean)

    external fun annIndexLoad(h: Long, filename: String, prefault: Boolean)

    external fun annIndexTreeNumber(h: Long): Int

    external fun annIndexItemNumber(h: Long): Int

    external fun annIndexSetOnDiskBuild(h: Long, filename: String): Boolean

    external fun annIndexSetSeed(h: Long, seed: Int)

    external fun annIndexRelease(h: Long)

    // ---- ALIKED --------------------------------------------------------

    external fun alikedCreate(
        modelPath: String,
        inputWidth: Int,
        inputHeight: Int,
        normalizeDescriptors: Boolean,
        engine: Int,
        backend: Int,
        target: Int,
    ): Long

    external fun alikedDetect(h: Long, image: Long, mask: Long): Long

    external fun alikedCompute(h: Long, image: Long, keypoints: Long): LongArray

    external fun alikedDetectAndCompute(
        h: Long,
        image: Long,
        mask: Long,
        useProvidedKeypoints: Boolean,
    ): LongArray

    external fun alikedDescriptorSize(h: Long): Int

    external fun alikedDescriptorType(h: Long): Int

    external fun alikedDefaultNorm(h: Long): Int

    external fun alikedWrite(h: Long, fileName: String)

    external fun alikedRead(h: Long, fileName: String)

    external fun alikedClear(h: Long)

    external fun alikedEmpty(h: Long): Boolean

    external fun alikedSave(h: Long, fileName: String)

    external fun alikedGetDefaultName(h: Long): String

    external fun alikedRelease(h: Long)

    // ---- DISK ----------------------------------------------------------

    external fun diskCreate(
        modelPath: String,
        maxKeypoints: Int,
        scoreThreshold: Float,
        imageWidth: Double,
        imageHeight: Double,
        backendId: Int,
        targetId: Int,
    ): Long

    external fun diskCreateFromMemory(
        modelData: ByteArray,
        maxKeypoints: Int,
        scoreThreshold: Float,
        imageWidth: Double,
        imageHeight: Double,
        backendId: Int,
        targetId: Int,
    ): Long

    external fun diskDetect(h: Long, image: Long, mask: Long): Long

    external fun diskCompute(h: Long, image: Long, keypoints: Long): LongArray

    external fun diskDetectAndCompute(
        h: Long,
        image: Long,
        mask: Long,
        useProvidedKeypoints: Boolean,
    ): LongArray

    external fun diskDescriptorSize(h: Long): Int

    external fun diskDescriptorType(h: Long): Int

    external fun diskDefaultNorm(h: Long): Int

    external fun diskWrite(h: Long, fileName: String)

    external fun diskRead(h: Long, fileName: String)

    external fun diskSetMaxKeypoints(h: Long, maxKeypoints: Int)

    external fun diskGetMaxKeypoints(h: Long): Int

    external fun diskSetScoreThreshold(h: Long, threshold: Float)

    external fun diskGetScoreThreshold(h: Long): Float

    external fun diskSetImageSize(h: Long, width: Double, height: Double)

    external fun diskImageSize(h: Long): DoubleArray

    external fun diskClear(h: Long)

    external fun diskEmpty(h: Long): Boolean

    external fun diskSave(h: Long, fileName: String)

    external fun diskGetDefaultName(h: Long): String

    external fun diskRelease(h: Long)

    // ---- Features statics ----------------------------------------------

    external fun featuresGoodFeaturesToTrack(
        image: Long,
        maxCorners: Int,
        qualityLevel: Double,
        minDistance: Double,
        mask: Long,
        blockSize: Int,
        useHarrisDetector: Boolean,
        k: Double,
    ): Long

    external fun featuresGoodFeaturesToTrackGradient(
        image: Long,
        maxCorners: Int,
        qualityLevel: Double,
        minDistance: Double,
        mask: Long,
        blockSize: Int,
        gradientSize: Int,
        useHarrisDetector: Boolean,
        k: Double,
    ): Long

    external fun featuresGoodFeaturesToTrackQuality(
        image: Long,
        maxCorners: Int,
        qualityLevel: Double,
        minDistance: Double,
        mask: Long,
        blockSize: Int,
        gradientSize: Int,
        useHarrisDetector: Boolean,
        k: Double,
    ): LongArray

    external fun drawKeypoints(
        image: Long,
        keypoints: Long,
        c0: Double,
        c1: Double,
        c2: Double,
        c3: Double,
        flags: Int,
    ): Long

    external fun drawKeypointsOver(
        image: Long,
        keypoints: Long,
        outImage: Long,
        c0: Double,
        c1: Double,
        c2: Double,
        c3: Double,
        flags: Int,
    )

    external fun drawMatches(
        img1: Long,
        keypoints1: Long,
        img2: Long,
        keypoints2: Long,
        matches: Long,
        mc0: Double,
        mc1: Double,
        mc2: Double,
        mc3: Double,
        sc0: Double,
        sc1: Double,
        sc2: Double,
        sc3: Double,
        matchesMask: Long,
        flags: Int,
    ): Long

    external fun drawMatchesOver(
        img1: Long,
        keypoints1: Long,
        img2: Long,
        keypoints2: Long,
        matches: Long,
        outImg: Long,
        mc0: Double,
        mc1: Double,
        mc2: Double,
        mc3: Double,
        sc0: Double,
        sc1: Double,
        sc2: Double,
        sc3: Double,
        matchesMask: Long,
        flags: Int,
    )

    external fun drawMatchesThickness(
        img1: Long,
        keypoints1: Long,
        img2: Long,
        keypoints2: Long,
        matches: Long,
        matchesThickness: Int,
        mc0: Double,
        mc1: Double,
        mc2: Double,
        mc3: Double,
        sc0: Double,
        sc1: Double,
        sc2: Double,
        sc3: Double,
        matchesMask: Long,
        flags: Int,
    ): Long

    external fun drawMatchesKnn(
        img1: Long,
        keypoints1: Long,
        img2: Long,
        keypoints2: Long,
        matchesWire: Long,
        mc0: Double,
        mc1: Double,
        mc2: Double,
        mc3: Double,
        sc0: Double,
        sc1: Double,
        sc2: Double,
        sc3: Double,
        masksWire: Long,
        flags: Int,
    ): Long
}

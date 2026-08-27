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
 * JNI bridge for the OpenCV photo module.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniPhoto_<name>`
 * function in jni_photo.cpp. All members are public (no `internal`
 * modifier) so their JVM names are not mangled by the Kotlin compiler.
 *
 * Mat handles travel as jlong pointers; Mat lists as jlongArray; the
 * two-output filters (decolor/pencilSketch/computeBitmaps) and the
 * alignMTB.process vector output come back as jlongArray too.
 */
internal object JniPhoto {

    // ---- statics ---------------------------------------------------------

    external fun inpaint(src: Long, mask: Long, radius: Double, flags: Int): Long
    external fun fastNlMeansDenoising(src: Long, h: Float, templateWindowSize: Int, searchWindowSize: Int): Long
    external fun fastNlMeansDenoisingH(src: Long, h: Long, templateWindowSize: Int, searchWindowSize: Int, normType: Int): Long
    external fun fastNlMeansDenoisingMulti(src: LongArray, imgIndex: Int, temporalWindowSize: Int, h: Float, templateWindowSize: Int, searchWindowSize: Int): Long
    external fun fastNlMeansDenoisingMultiH(src: LongArray, imgIndex: Int, temporalWindowSize: Int, h: Long, templateWindowSize: Int, searchWindowSize: Int, normType: Int): Long
    external fun fastNlMeansDenoisingColored(src: Long, h: Float, hColor: Float, templateWindowSize: Int, searchWindowSize: Int): Long
    external fun fastNlMeansDenoisingColoredMulti(src: LongArray, imgIndex: Int, temporalWindowSize: Int, h: Float, hColor: Float, templateWindowSize: Int, searchWindowSize: Int): Long
    external fun denoiseTvl1(src: LongArray, lambda: Double, niters: Int): Long
    external fun decolor(src: Long): LongArray
    external fun seamlessClone(src: Long, dst: Long, mask: Long, pX: Int, pY: Int, flags: Int): Long
    external fun colorChange(src: Long, mask: Long, redMul: Float, greenMul: Float, blueMul: Float): Long
    external fun illuminationChange(src: Long, mask: Long, alpha: Float, beta: Float): Long
    external fun textureFlattening(src: Long, mask: Long, lowThreshold: Float, highThreshold: Float, kernelSize: Int): Long
    external fun edgePreservingFilter(src: Long, flags: Int, sigmaS: Float, sigmaR: Float): Long
    external fun detailEnhance(src: Long, sigmaS: Float, sigmaR: Float): Long
    external fun pencilSketch(src: Long, sigmaS: Float, sigmaR: Float, shadeFactor: Float): LongArray
    external fun stylization(src: Long, sigmaS: Float, sigmaR: Float): Long
    external fun correctChromaticAberration(input: Long, coefficients: Long, width: Int, height: Int, calibDegree: Int, bayerPattern: Int): Long
    external fun gammaCorrection(src: Long, gamma: Double): Long

    // ---- Tonemap ---------------------------------------------------------

    external fun tonemapCreate(gamma: Float): Long
    external fun tonemapProcess(handle: Long, src: Long): Long
    external fun tonemapGetGamma(handle: Long): Float
    external fun tonemapSetGamma(handle: Long, gamma: Float)
    external fun tonemapClear(handle: Long)
    external fun tonemapEmpty(handle: Long): Boolean
    external fun tonemapSave(handle: Long, filename: String)
    external fun tonemapGetDefaultName(handle: Long): String
    external fun tonemapRelease(handle: Long)

    // ---- TonemapDrago ----------------------------------------------------

    external fun tonemapDragoCreate(gamma: Float, saturation: Float, bias: Float): Long
    external fun tonemapDragoProcess(handle: Long, src: Long): Long
    external fun tonemapDragoGetGamma(handle: Long): Float
    external fun tonemapDragoSetGamma(handle: Long, gamma: Float)
    external fun tonemapDragoGetSaturation(handle: Long): Float
    external fun tonemapDragoSetSaturation(handle: Long, saturation: Float)
    external fun tonemapDragoGetBias(handle: Long): Float
    external fun tonemapDragoSetBias(handle: Long, bias: Float)
    external fun tonemapDragoClear(handle: Long)
    external fun tonemapDragoEmpty(handle: Long): Boolean
    external fun tonemapDragoSave(handle: Long, filename: String)
    external fun tonemapDragoGetDefaultName(handle: Long): String
    external fun tonemapDragoRelease(handle: Long)

    // ---- TonemapMantiuk --------------------------------------------------

    external fun tonemapMantiukCreate(gamma: Float, scale: Float, saturation: Float): Long
    external fun tonemapMantiukProcess(handle: Long, src: Long): Long
    external fun tonemapMantiukGetGamma(handle: Long): Float
    external fun tonemapMantiukSetGamma(handle: Long, gamma: Float)
    external fun tonemapMantiukGetScale(handle: Long): Float
    external fun tonemapMantiukSetScale(handle: Long, scale: Float)
    external fun tonemapMantiukGetSaturation(handle: Long): Float
    external fun tonemapMantiukSetSaturation(handle: Long, saturation: Float)
    external fun tonemapMantiukClear(handle: Long)
    external fun tonemapMantiukEmpty(handle: Long): Boolean
    external fun tonemapMantiukSave(handle: Long, filename: String)
    external fun tonemapMantiukGetDefaultName(handle: Long): String
    external fun tonemapMantiukRelease(handle: Long)

    // ---- TonemapReinhard -------------------------------------------------

    external fun tonemapReinhardCreate(gamma: Float, intensity: Float, lightAdaptation: Float, colorAdaptation: Float): Long
    external fun tonemapReinhardProcess(handle: Long, src: Long): Long
    external fun tonemapReinhardGetGamma(handle: Long): Float
    external fun tonemapReinhardSetGamma(handle: Long, gamma: Float)
    external fun tonemapReinhardGetIntensity(handle: Long): Float
    external fun tonemapReinhardSetIntensity(handle: Long, intensity: Float)
    external fun tonemapReinhardGetLightAdaptation(handle: Long): Float
    external fun tonemapReinhardSetLightAdaptation(handle: Long, lightAdaptation: Float)
    external fun tonemapReinhardGetColorAdaptation(handle: Long): Float
    external fun tonemapReinhardSetColorAdaptation(handle: Long, colorAdaptation: Float)
    external fun tonemapReinhardClear(handle: Long)
    external fun tonemapReinhardEmpty(handle: Long): Boolean
    external fun tonemapReinhardSave(handle: Long, filename: String)
    external fun tonemapReinhardGetDefaultName(handle: Long): String
    external fun tonemapReinhardRelease(handle: Long)

    // ---- AlignMTB --------------------------------------------------------

    external fun alignMtbCreate(maxBits: Int, excludeRange: Int, cut: Boolean): Long
    external fun alignMtbProcess(handle: Long, src: LongArray): LongArray?
    external fun alignMtbProcessTimes(handle: Long, src: LongArray, times: Long, response: Long): LongArray?
    external fun alignMtbCalculateShift(handle: Long, img0: Long, img1: Long): IntArray
    external fun alignMtbShiftMat(handle: Long, src: Long, shiftX: Int, shiftY: Int): Long
    external fun alignMtbComputeBitmaps(handle: Long, img: Long): LongArray
    external fun alignMtbGetMaxBits(handle: Long): Int
    external fun alignMtbSetMaxBits(handle: Long, value: Int)
    external fun alignMtbGetExcludeRange(handle: Long): Int
    external fun alignMtbSetExcludeRange(handle: Long, value: Int)
    external fun alignMtbGetCut(handle: Long): Boolean
    external fun alignMtbSetCut(handle: Long, value: Boolean)
    external fun alignMtbClear(handle: Long)
    external fun alignMtbEmpty(handle: Long): Boolean
    external fun alignMtbSave(handle: Long, filename: String)
    external fun alignMtbGetDefaultName(handle: Long): String
    external fun alignMtbRelease(handle: Long)

    // ---- CalibrateDebevec ------------------------------------------------

    external fun calibrateDebevecCreate(samples: Int, lambda: Float, random: Boolean): Long
    external fun calibrateDebevecProcess(handle: Long, src: LongArray, times: Long): Long
    external fun calibrateDebevecGetLambda(handle: Long): Float
    external fun calibrateDebevecSetLambda(handle: Long, lambda: Float)
    external fun calibrateDebevecGetSamples(handle: Long): Int
    external fun calibrateDebevecSetSamples(handle: Long, samples: Int)
    external fun calibrateDebevecGetRandom(handle: Long): Boolean
    external fun calibrateDebevecSetRandom(handle: Long, random: Boolean)
    external fun calibrateDebevecClear(handle: Long)
    external fun calibrateDebevecEmpty(handle: Long): Boolean
    external fun calibrateDebevecSave(handle: Long, filename: String)
    external fun calibrateDebevecGetDefaultName(handle: Long): String
    external fun calibrateDebevecRelease(handle: Long)

    // ---- CalibrateRobertson ----------------------------------------------

    external fun calibrateRobertsonCreate(maxIter: Int, threshold: Float): Long
    external fun calibrateRobertsonProcess(handle: Long, src: LongArray, times: Long): Long
    external fun calibrateRobertsonGetRadiance(handle: Long): Long
    external fun calibrateRobertsonGetMaxIter(handle: Long): Int
    external fun calibrateRobertsonSetMaxIter(handle: Long, maxIter: Int)
    external fun calibrateRobertsonGetThreshold(handle: Long): Float
    external fun calibrateRobertsonSetThreshold(handle: Long, threshold: Float)
    external fun calibrateRobertsonClear(handle: Long)
    external fun calibrateRobertsonEmpty(handle: Long): Boolean
    external fun calibrateRobertsonSave(handle: Long, filename: String)
    external fun calibrateRobertsonGetDefaultName(handle: Long): String
    external fun calibrateRobertsonRelease(handle: Long)

    // ---- MergeDebevec ----------------------------------------------------

    external fun mergeDebevecCreate(): Long
    external fun mergeDebevecProcess(handle: Long, src: LongArray, times: Long): Long
    external fun mergeDebevecProcessResponse(handle: Long, src: LongArray, times: Long, response: Long): Long
    external fun mergeDebevecClear(handle: Long)
    external fun mergeDebevecEmpty(handle: Long): Boolean
    external fun mergeDebevecSave(handle: Long, filename: String)
    external fun mergeDebevecGetDefaultName(handle: Long): String
    external fun mergeDebevecRelease(handle: Long)

    // ---- MergeMertens ----------------------------------------------------

    external fun mergeMertensCreate(contrastWeight: Float, saturationWeight: Float, exposureWeight: Float): Long
    external fun mergeMertensProcess(handle: Long, src: LongArray): Long
    external fun mergeMertensProcessResponse(handle: Long, src: LongArray, times: Long, response: Long): Long
    external fun mergeMertensGetContrastWeight(handle: Long): Float
    external fun mergeMertensSetContrastWeight(handle: Long, value: Float)
    external fun mergeMertensGetSaturationWeight(handle: Long): Float
    external fun mergeMertensSetSaturationWeight(handle: Long, value: Float)
    external fun mergeMertensGetExposureWeight(handle: Long): Float
    external fun mergeMertensSetExposureWeight(handle: Long, value: Float)
    external fun mergeMertensClear(handle: Long)
    external fun mergeMertensEmpty(handle: Long): Boolean
    external fun mergeMertensSave(handle: Long, filename: String)
    external fun mergeMertensGetDefaultName(handle: Long): String
    external fun mergeMertensRelease(handle: Long)

    // ---- MergeRobertson --------------------------------------------------

    external fun mergeRobertsonCreate(): Long
    external fun mergeRobertsonProcess(handle: Long, src: LongArray, times: Long): Long
    external fun mergeRobertsonProcessResponse(handle: Long, src: LongArray, times: Long, response: Long): Long
    external fun mergeRobertsonClear(handle: Long)
    external fun mergeRobertsonEmpty(handle: Long): Boolean
    external fun mergeRobertsonSave(handle: Long, filename: String)
    external fun mergeRobertsonGetDefaultName(handle: Long): String
    external fun mergeRobertsonRelease(handle: Long)

    // ---- ColorCorrectionModel --------------------------------------------

    external fun colorCorrectionModelCreate(src: Long, constColor: Int): Long
    external fun colorCorrectionModelCreateEmpty(): Long
    external fun colorCorrectionModelSetLinearizationGamma(handle: Long, gamma: Double)
    external fun colorCorrectionModelSetLinearizationDegree(handle: Long, deg: Int)
    external fun colorCorrectionModelSetSaturatedThreshold(handle: Long, lower: Double, upper: Double)
    external fun colorCorrectionModelSetWeightsList(handle: Long, weights: Long)
    external fun colorCorrectionModelSetWeightCoeff(handle: Long, coeff: Double)
    external fun colorCorrectionModelSetMaxCount(handle: Long, maxCount: Int)
    external fun colorCorrectionModelSetEpsilon(handle: Long, epsilon: Double)
    external fun colorCorrectionModelSetRgb(handle: Long, rgb: Boolean)
    external fun colorCorrectionModelCompute(handle: Long): Long
    external fun colorCorrectionModelGetColorCorrectionMatrix(handle: Long): Long
    external fun colorCorrectionModelGetLoss(handle: Long): Double
    external fun colorCorrectionModelGetSrcLinearRgb(handle: Long): Long
    external fun colorCorrectionModelGetRefLinearRgb(handle: Long): Long
    external fun colorCorrectionModelGetMask(handle: Long): Long
    external fun colorCorrectionModelGetWeights(handle: Long): Long
    external fun colorCorrectionModelCorrectImage(handle: Long, src: Long, islinear: Boolean): Long
    external fun colorCorrectionModelRelease(handle: Long)

    // ---- IntelligentScissorsMB -------------------------------------------

    external fun scissorsCreate(): Long
    external fun scissorsSetWeights(handle: Long, weightNonEdge: Float, weightGradientDirection: Float, weightGradientMagnitude: Float)
    external fun scissorsSetGradientMagnitudeMaxLimit(handle: Long, limit: Float)
    external fun scissorsSetEdgeFeatureZeroCrossingParameters(handle: Long, minValue: Float)
    external fun scissorsSetEdgeFeatureCannyParameters(handle: Long, threshold1: Double, threshold2: Double, apertureSize: Int, l2gradient: Boolean)
    external fun scissorsApplyImage(handle: Long, image: Long)
    external fun scissorsApplyImageFeatures(handle: Long, nonEdge: Long, gradientDirection: Long, gradientMagnitude: Long, image: Long)
    external fun scissorsBuildMap(handle: Long, x: Int, y: Int)
    external fun scissorsGetContour(handle: Long, x: Int, y: Int, backward: Boolean): Long
    external fun scissorsRelease(handle: Long)
}

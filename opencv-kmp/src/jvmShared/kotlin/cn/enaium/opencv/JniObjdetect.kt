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
 * JNI bridge for the objdetect slice.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniObjdetect_<name>`
 * function in jni/jni_objdetect.cpp. Mat handles travel as jlong pointers;
 * DetectorParameters/RefineParameters travel as DoubleArray in the fixed
 * field order documented in [DetectorParameters] (which matches the C struct
 * `cvk_detector_params_t` and the JNI conversion). Multi-output operations
 * return a jlongArray of Mat handles; null means the native call failed.
 */
internal object JniObjdetect {

    // ---- Dictionary ------------------------------------------------------

    external fun dictionaryCreate(bytesList: Long, markerSize: Int, maxCorrectionBits: Int): Long
    external fun dictionaryRelease(handle: Long)
    external fun dictionaryGetBytesList(handle: Long): Long
    external fun dictionarySetBytesList(handle: Long, bytesList: Long)
    external fun dictionaryGetMarkerSize(handle: Long): Int
    external fun dictionarySetMarkerSize(handle: Long, markerSize: Int)
    external fun dictionaryGetMaxCorrectionBits(handle: Long): Int
    external fun dictionarySetMaxCorrectionBits(handle: Long, maxCorrectionBits: Int)
    /** [found, idx, rotation]. */
    external fun dictionaryIdentify(handle: Long, onlyBits: Long, maxCorrectionRate: Double): IntArray
    /** [found, idx, rotation]. */
    external fun dictionaryIdentifyPixelRatio(handle: Long, onlyCellPixelRatio: Long, maxCorrectionRate: Double, validBitIdThreshold: Float): IntArray
    external fun dictionaryGetDistanceToId(handle: Long, bits: Long, id: Int, allRotations: Boolean): Int
    external fun dictionaryGenerateImageMarker(handle: Long, id: Int, sidePixels: Int, borderBits: Int): Long
    external fun dictionaryGetMarkerBits(handle: Long, markerId: Int, rotationId: Int): Long
    external fun dictionaryGetByteListFromBits(bits: Long): Long
    external fun dictionaryGetBitsFromByteList(byteList: Long, markerSize: Int, rotationId: Int): Long
    external fun getPredefinedDictionary(dict: Int): Long
    external fun extendDictionary(nMarkers: Int, markerSize: Int, baseDictionary: Long, randomSeed: Int): Long

    // ---- Board -----------------------------------------------------------

    external fun boardCreate(objPoints: Long, dictionary: Long, ids: Long): Long
    external fun boardRelease(handle: Long)
    external fun boardGetDictionary(handle: Long): Long
    external fun boardGetObjPoints(handle: Long): Long
    external fun boardGetIds(handle: Long): Long
    external fun boardGetRightBottomCorner(handle: Long): DoubleArray
    /** [objPoints, imgPoints] handles. */
    external fun boardMatchImagePoints(handle: Long, detectedCorners: Long, detectedIds: Long): LongArray?
    external fun boardGenerateImage(handle: Long, outSizeWidth: Double, outSizeHeight: Double, marginSize: Int, borderBits: Int): Long

    // ---- GridBoard -------------------------------------------------------

    external fun gridBoardCreate(sizeWidth: Double, sizeHeight: Double, markerLength: Float, markerSeparation: Float, dictionary: Long, ids: Long): Long
    external fun gridBoardRelease(handle: Long)
    /** [width, height]. */
    external fun gridBoardGetGridSize(handle: Long): IntArray
    external fun gridBoardGetMarkerLength(handle: Long): Float
    external fun gridBoardGetMarkerSeparation(handle: Long): Float

    // ---- CharucoBoard ----------------------------------------------------

    external fun charucoBoardCreate(sizeWidth: Double, sizeHeight: Double, squareLength: Float, markerLength: Float, dictionary: Long, ids: Long): Long
    external fun charucoBoardRelease(handle: Long)
    external fun charucoBoardSetLegacyPattern(handle: Long, legacyPattern: Boolean)
    external fun charucoBoardGetLegacyPattern(handle: Long): Boolean
    /** [width, height]. */
    external fun charucoBoardGetChessboardSize(handle: Long): IntArray
    external fun charucoBoardGetSquareLength(handle: Long): Float
    external fun charucoBoardGetMarkerLength(handle: Long): Float
    external fun charucoBoardGetChessboardCorners(handle: Long): Long
    external fun charucoBoardCheckCharucoCornersCollinear(handle: Long, charucoIds: Long): Boolean

    // ---- ArucoDetector ---------------------------------------------------

    external fun arucoDetectorCreate(dictionary: Long, detectorParams: DoubleArray, refineParams: DoubleArray): Long
    external fun arucoDetectorRelease(handle: Long)
    /** [corners, ids, rejected] handles. */
    external fun arucoDetectorDetectMarkers(handle: Long, image: Long): LongArray?
    /** [corners, ids, confidence, rejected] handles. */
    external fun arucoDetectorDetectMarkersWithConfidence(handle: Long, image: Long): LongArray?
    /** [corners, ids, rejected, recoveredIdxs] handles. */
    external fun arucoDetectorRefineDetectedMarkers(handle: Long, image: Long, board: Long, detectedCorners: Long, detectedIds: Long, rejectedCorners: Long, cameraMatrix: Long, distCoeffs: Long): LongArray?
    /** [corners, ids, rejected, dictIndices] handles. */
    external fun arucoDetectorDetectMarkersMultiDict(handle: Long, image: Long): LongArray?
    external fun arucoDetectorGetDictionary(handle: Long): Long
    external fun arucoDetectorSetDictionary(handle: Long, dictionary: Long)
    external fun arucoDetectorGetDetectorParams(handle: Long): DoubleArray
    external fun arucoDetectorSetDetectorParams(handle: Long, detectorParams: DoubleArray)
    external fun arucoDetectorGetRefineParams(handle: Long): DoubleArray
    external fun arucoDetectorSetRefineParams(handle: Long, refineParams: DoubleArray)
    external fun arucoDetectorClear(handle: Long)
    external fun arucoDetectorEmpty(handle: Long): Boolean
    external fun arucoDetectorSave(handle: Long, filename: String)
    external fun arucoDetectorGetDefaultName(handle: Long): String?

    // ---- CharucoDetector -------------------------------------------------

    external fun charucoDetectorCreate(board: Long, cameraMatrix: Long, distCoeffs: Long, minMarkers: Int, tryRefineMarkers: Boolean, checkMarkers: Boolean, detectorParams: DoubleArray, refineParams: DoubleArray): Long
    external fun charucoDetectorRelease(handle: Long)
    external fun charucoDetectorGetBoard(handle: Long): Long
    external fun charucoDetectorSetBoard(handle: Long, board: Long)
    /** [cameraMatrix, distCoeffs, minMarkers, tryRefineMarkers, checkMarkers]. */
    external fun charucoDetectorGetCharucoParams(handle: Long): LongArray
    external fun charucoDetectorSetCharucoParams(handle: Long, cameraMatrix: Long, distCoeffs: Long, minMarkers: Int, tryRefineMarkers: Boolean, checkMarkers: Boolean)
    external fun charucoDetectorGetDetectorParams(handle: Long): DoubleArray
    external fun charucoDetectorSetDetectorParams(handle: Long, detectorParams: DoubleArray)
    external fun charucoDetectorGetRefineParams(handle: Long): DoubleArray
    external fun charucoDetectorSetRefineParams(handle: Long, refineParams: DoubleArray)
    /** [charucoCorners, charucoIds, markerCorners, markerIds] handles. */
    external fun charucoDetectorDetectBoard(handle: Long, image: Long, markerCorners: Long, markerIds: Long): LongArray?
    /** [diamondCorners, diamondIds, markerCorners, markerIds] handles. */
    external fun charucoDetectorDetectDiamonds(handle: Long, image: Long, markerCorners: Long, markerIds: Long): LongArray?
    external fun charucoDetectorClear(handle: Long)
    external fun charucoDetectorEmpty(handle: Long): Boolean
    external fun charucoDetectorSave(handle: Long, filename: String)
    external fun charucoDetectorGetDefaultName(handle: Long): String?

    // ---- Objdetect statics ----------------------------------------------

    /** Returns the corners Mat handle; writes `found` into foundOut[0]. */
    external fun findChessboardCorners(image: Long, patternWidth: Double, patternHeight: Double, flags: Int, foundOut: BooleanArray): Long
    external fun checkChessboard(image: Long, sizeWidth: Double, sizeHeight: Double): Boolean
    /** Returns the corners Mat handle; writes `found` into foundOut[0]. */
    external fun findChessboardCornersSB(image: Long, patternWidth: Double, patternHeight: Double, flags: Int, foundOut: BooleanArray): Long
    /** Returns [corners, meta] handles; writes `found` into foundOut[0]. */
    external fun findChessboardCornersSBWithMeta(image: Long, patternWidth: Double, patternHeight: Double, flags: Int, foundOut: BooleanArray): LongArray?
    /** Returns the sharpness Mat handle; writes the Scalar into scalarOut[0..3]. */
    external fun estimateChessboardSharpness(image: Long, patternWidth: Double, patternHeight: Double, corners: Long, riseDistance: Float, vertical: Boolean, scalarOut: DoubleArray): Long
    external fun find4QuadCornerSubpix(image: Long, corners: Long, regionWidth: Double, regionHeight: Double): Boolean
    external fun drawChessboardCorners(image: Long, patternWidth: Double, patternHeight: Double, corners: Long, patternWasFound: Boolean)
    /** Returns the centers Mat handle; writes `found` into foundOut[0]. */
    external fun findCirclesGrid(image: Long, patternWidth: Double, patternHeight: Double, flags: Int, foundOut: BooleanArray): Long
    external fun drawDetectedMarkers(image: Long, corners: Long, ids: Long, v0: Double, v1: Double, v2: Double, v3: Double)
    external fun generateImageMarker(dictionary: Long, id: Int, sidePixels: Int, borderBits: Int): Long
    external fun drawDetectedCornersCharuco(image: Long, charucoCorners: Long, charucoIds: Long, v0: Double, v1: Double, v2: Double, v3: Double)
    external fun drawDetectedDiamonds(image: Long, diamondCorners: Long, diamondIds: Long, v0: Double, v1: Double, v2: Double, v3: Double)
}

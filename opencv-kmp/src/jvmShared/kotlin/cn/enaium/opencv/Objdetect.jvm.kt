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

/** JVM actuals for the objdetect slice (see [JniObjdetect]). */
private fun lastNativeError(): String? = Jni.lastError()


// ---- vector<Mat> <-> CV_32SC2 address container --------------------------

private fun writeLeInt(data: ByteArray, offset: Int, value: Int) {
    data[offset] = (value and 0xFF).toByte()
    data[offset + 1] = ((value ushr 8) and 0xFF).toByte()
    data[offset + 2] = ((value ushr 16) and 0xFF).toByte()
    data[offset + 3] = ((value ushr 24) and 0xFF).toByte()
}

private fun readLeInt(data: ByteArray, offset: Int): Int =
    (data[offset].toInt() and 0xFF) or
        ((data[offset + 1].toInt() and 0xFF) shl 8) or
        ((data[offset + 2].toInt() and 0xFF) shl 16) or
        ((data[offset + 3].toInt() and 0xFF) shl 24)

/** Encodes platform Mat handles into the CV_32SC2 vector-of-Mat container. */
private fun List<Mat>.toVectorMatContainer(): Mat {
    if (isEmpty()) return mat()
    val container = mat(size, 1, vectorMatContainerType)
    val data = ByteArray(size * 8)
    forEachIndexed { index, m ->
        val addr = handleOf(m)
        writeLeInt(data, index * 8, (addr ushr 32).toInt())
        writeLeInt(data, index * 8 + 4, addr.toInt())
    }
    container.pixels = data
    return container
}

/**
 * Decodes a container into platform Mat handles owned by the caller; the
 * container Mat itself must be closed by the caller separately.
 */
private fun Mat.toMatList(): List<Mat> {
    if (isEmpty) return emptyList()
    val data = pixels
    val count = total
    val result = ArrayList<Mat>(count)
    var offset = 0
    repeat(count) {
        val hi = readLeInt(data, offset).toLong()
        val lo = readLeInt(data, offset + 4).toLong() and 0xFFFFFFFFL
        val addr = (hi shl 32) or lo
        result.add(JvmMat(addr))
        offset += 8
    }
    return result
}

/** Decodes a fresh container Mat into caller-owned Mats and closes the container. */
private fun Mat.toMatListClosing(): List<Mat> = use { it.toMatList() }

private fun Long.ifNull(operation: String): Long =
    if (this != 0L) this else throw OpenCVException(operation, lastNativeError())

private fun LongArray?.ifNull(operation: String): LongArray? =
    if (this != null && isNotEmpty()) this else throw OpenCVException(operation, lastNativeError())

// =========================================================================
// Dictionary
// =========================================================================

internal class JvmDictionary(@Volatile private var handle: Long) : Dictionary {
    internal fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("Dictionary is closed")

    override val bytesList: Mat
        get() = jvmMat(JniObjdetect.dictionaryGetBytesList(check()), "dictionary.bytesList")
    override val markerSize: Int get() = JniObjdetect.dictionaryGetMarkerSize(check())
    override val maxCorrectionBits: Int get() = JniObjdetect.dictionaryGetMaxCorrectionBits(check())
    override fun setBytesList(bytesList: Mat) { JniObjdetect.dictionarySetBytesList(check(), handleOf(bytesList)) }
    override fun setMarkerSize(markerSize: Int) { JniObjdetect.dictionarySetMarkerSize(check(), markerSize) }
    override fun setMaxCorrectionBits(maxCorrectionBits: Int) { JniObjdetect.dictionarySetMaxCorrectionBits(check(), maxCorrectionBits) }

    override fun identify(onlyBits: Mat, maxCorrectionRate: Double): IdentifyResult {
        val result = JniObjdetect.dictionaryIdentify(check(), handleOf(onlyBits), maxCorrectionRate)
        return IdentifyResult(result[0] != 0, result[1], result[2])
    }

    override fun identify(onlyCellPixelRatio: Mat, maxCorrectionRate: Double, validBitIdThreshold: Float): IdentifyResult {
        val result = JniObjdetect.dictionaryIdentifyPixelRatio(
            check(), handleOf(onlyCellPixelRatio), maxCorrectionRate, validBitIdThreshold,
        )
        return IdentifyResult(result[0] != 0, result[1], result[2])
    }

    override fun getDistanceToId(bits: Mat, id: Int, allRotations: Boolean): Int =
        JniObjdetect.dictionaryGetDistanceToId(check(), handleOf(bits), id, allRotations)

    override fun generateImageMarker(id: Int, sidePixels: Int, borderBits: Int): Mat =
        jvmMat(JniObjdetect.dictionaryGenerateImageMarker(check(), id, sidePixels, borderBits), "dictionary.generateImageMarker")

    override fun getMarkerBits(markerId: Int, rotationId: Int): Mat =
        jvmMat(JniObjdetect.dictionaryGetMarkerBits(check(), markerId, rotationId), "dictionary.getMarkerBits")

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect.dictionaryRelease(h)
        }
    }
}

// =========================================================================
// Board
// =========================================================================

internal class JvmBoard(@Volatile private var handle: Long) : Board {
    internal fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("Board is closed")

    override val dictionary: Dictionary
        get() = JvmDictionary(JniObjdetect.boardGetDictionary(check()))
    override val objPoints: List<Mat>
        get() = jvmMat(JniObjdetect.boardGetObjPoints(check()), "board.objPoints").toMatListClosing()
    override val ids: Mat
        get() = jvmMat(JniObjdetect.boardGetIds(check()), "board.ids")
    override val rightBottomCorner: Point3
        get() {
            val out = JniObjdetect.boardGetRightBottomCorner(check())
            return Point3(out[0], out[1], out[2])
        }

    override fun matchImagePoints(detectedCorners: List<Mat>, detectedIds: Mat): BoardMatchPoints {
        detectedCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = JniObjdetect.boardMatchImagePoints(
                check(), handleOf(cornersContainer), handleOf(detectedIds),
            ) ?: throw OpenCVException("board.matchImagePoints", lastNativeError())
            return BoardMatchPoints(
                jvmMat(handles[0], "board.matchImagePoints"),
                jvmMat(handles[1], "board.matchImagePoints"),
            )
        }
    }

    override fun generateImage(outSize: Size, marginSize: Int, borderBits: Int): Mat =
        jvmMat(
            JniObjdetect.boardGenerateImage(check(), outSize.width.toDouble(), outSize.height.toDouble(), marginSize, borderBits),
            "board.generateImage",
        )

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect.boardRelease(h)
        }
    }
}

internal class JvmGridBoard(@Volatile private var handle: Long) : GridBoard {
    internal fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("GridBoard is closed")

    // Board-level JNI functions take the same handle (the C++ Board base
    // subobject sits at offset 0 of the GridBoard).

    override val dictionary: Dictionary
        get() = JvmDictionary(JniObjdetect.boardGetDictionary(check()))
    override val objPoints: List<Mat>
        get() = jvmMat(JniObjdetect.boardGetObjPoints(check()), "board.objPoints").toMatListClosing()
    override val ids: Mat
        get() = jvmMat(JniObjdetect.boardGetIds(check()), "board.ids")
    override val rightBottomCorner: Point3
        get() {
            val out = JniObjdetect.boardGetRightBottomCorner(check())
            return Point3(out[0], out[1], out[2])
        }

    override fun matchImagePoints(detectedCorners: List<Mat>, detectedIds: Mat): BoardMatchPoints {
        detectedCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = JniObjdetect.boardMatchImagePoints(
                check(), handleOf(cornersContainer), handleOf(detectedIds),
            ) ?: throw OpenCVException("board.matchImagePoints", lastNativeError())
            return BoardMatchPoints(
                jvmMat(handles[0], "board.matchImagePoints"),
                jvmMat(handles[1], "board.matchImagePoints"),
            )
        }
    }

    override fun generateImage(outSize: Size, marginSize: Int, borderBits: Int): Mat =
        jvmMat(
            JniObjdetect.boardGenerateImage(check(), outSize.width.toDouble(), outSize.height.toDouble(), marginSize, borderBits),
            "board.generateImage",
        )

    override val gridSize: Size
        get() {
            val out = JniObjdetect.gridBoardGetGridSize(check())
            return Size(out[0], out[1])
        }
    override val markerLength: Float get() = JniObjdetect.gridBoardGetMarkerLength(check())
    override val markerSeparation: Float get() = JniObjdetect.gridBoardGetMarkerSeparation(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect.gridBoardRelease(h)
        }
    }
}

internal class JvmCharucoBoard(@Volatile private var handle: Long) : CharucoBoard {
    internal fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("CharucoBoard is closed")

    override val dictionary: Dictionary
        get() = JvmDictionary(JniObjdetect.boardGetDictionary(check()))
    override val objPoints: List<Mat>
        get() = jvmMat(JniObjdetect.boardGetObjPoints(check()), "board.objPoints").toMatListClosing()
    override val ids: Mat
        get() = jvmMat(JniObjdetect.boardGetIds(check()), "board.ids")
    override val rightBottomCorner: Point3
        get() {
            val out = JniObjdetect.boardGetRightBottomCorner(check())
            return Point3(out[0], out[1], out[2])
        }

    override fun matchImagePoints(detectedCorners: List<Mat>, detectedIds: Mat): BoardMatchPoints {
        detectedCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = JniObjdetect.boardMatchImagePoints(
                check(), handleOf(cornersContainer), handleOf(detectedIds),
            ) ?: throw OpenCVException("board.matchImagePoints", lastNativeError())
            return BoardMatchPoints(
                jvmMat(handles[0], "board.matchImagePoints"),
                jvmMat(handles[1], "board.matchImagePoints"),
            )
        }
    }

    override fun generateImage(outSize: Size, marginSize: Int, borderBits: Int): Mat =
        jvmMat(
            JniObjdetect.boardGenerateImage(check(), outSize.width.toDouble(), outSize.height.toDouble(), marginSize, borderBits),
            "board.generateImage",
        )

    override var legacyPattern: Boolean
        get() = JniObjdetect.charucoBoardGetLegacyPattern(check())
        set(value) { JniObjdetect.charucoBoardSetLegacyPattern(check(), value) }
    override val chessboardSize: Size
        get() {
            val out = JniObjdetect.charucoBoardGetChessboardSize(check())
            return Size(out[0], out[1])
        }
    override val squareLength: Float get() = JniObjdetect.charucoBoardGetSquareLength(check())
    override val markerLength: Float get() = JniObjdetect.charucoBoardGetMarkerLength(check())
    override val chessboardCorners: Mat
        get() = jvmMat(JniObjdetect.charucoBoardGetChessboardCorners(check()), "charucoBoard.chessboardCorners")

    override fun checkCharucoCornersCollinear(charucoIds: Mat): Boolean =
        JniObjdetect.charucoBoardCheckCharucoCornersCollinear(check(), handleOf(charucoIds))

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect.charucoBoardRelease(h)
        }
    }
}

// =========================================================================
// ArucoDetector
// =========================================================================

internal class JvmArucoDetector(@Volatile private var handle: Long) : ArucoDetector {
    internal fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("ArucoDetector is closed")

    override var dictionary: Dictionary
        get() = JvmDictionary(JniObjdetect.arucoDetectorGetDictionary(check()))
        set(value) {
            val native = value as? JvmDictionary
                ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
            JniObjdetect.arucoDetectorSetDictionary(check(), native.check())
        }

    override var detectorParameters: DetectorParameters
        get() = JniObjdetect.arucoDetectorGetDetectorParams(check()).toDetectorParameters()
        set(value) { JniObjdetect.arucoDetectorSetDetectorParams(check(), value.toParamArray()) }

    override var refineParameters: RefineParameters
        get() = JniObjdetect.arucoDetectorGetRefineParams(check()).toRefineParameters()
        set(value) { JniObjdetect.arucoDetectorSetRefineParams(check(), value.toParamArray()) }

    override fun detectMarkers(image: Mat): MarkerDetection {
        val handles = JniObjdetect.arucoDetectorDetectMarkers(check(), handleOf(image))
            ?: throw OpenCVException("arucoDetector.detectMarkers", lastNativeError())
        val cornersMat = jvmMat(handles[0], "arucoDetector.detectMarkers")
        val idsMat = jvmMat(handles[1], "arucoDetector.detectMarkers")
        val rejectedMat = jvmMat(handles[2], "arucoDetector.detectMarkers")
        return MarkerDetection(
            corners = cornersMat.toMatListClosing(),
            ids = idsMat,
            rejectedImgPoints = rejectedMat.toMatListClosing(),
        )
    }

    override fun detectMarkersWithConfidence(image: Mat): MarkerDetectionWithConfidence {
        val handles = JniObjdetect.arucoDetectorDetectMarkersWithConfidence(check(), handleOf(image))
            ?: throw OpenCVException("arucoDetector.detectMarkersWithConfidence", lastNativeError())
        val cornersMat = jvmMat(handles[0], "arucoDetector.detectMarkersWithConfidence")
        val idsMat = jvmMat(handles[1], "arucoDetector.detectMarkersWithConfidence")
        val confidenceMat = jvmMat(handles[2], "arucoDetector.detectMarkersWithConfidence")
        val rejectedMat = jvmMat(handles[3], "arucoDetector.detectMarkersWithConfidence")
        return MarkerDetectionWithConfidence(
            corners = cornersMat.toMatListClosing(),
            ids = idsMat,
            markersConfidence = confidenceMat,
            rejectedImgPoints = rejectedMat.toMatListClosing(),
        )
    }

    override fun refineDetectedMarkers(
        image: Mat,
        board: Board,
        detectedCorners: List<Mat>,
        detectedIds: Mat,
        rejectedCorners: List<Mat>,
        cameraMatrix: Mat?,
        distCoeffs: Mat?,
    ): RefinedMarkers {
        detectedCorners.toVectorMatContainer().use { cornersContainer ->
            rejectedCorners.toVectorMatContainer().use { rejectedContainer ->
                val boardHandle = when (board) {
                    is JvmBoard -> board.check()
                    is JvmGridBoard -> board.check()
                    is JvmCharucoBoard -> board.check()
                    else -> throw IllegalArgumentException("board belongs to another platform backend")
                }
                val handles = JniObjdetect.arucoDetectorRefineDetectedMarkers(
                    check(), handleOf(image), boardHandle,
                    handleOf(cornersContainer), handleOf(detectedIds),
                    handleOf(rejectedContainer),
                    cameraMatrix?.let { handleOf(it) } ?: 0L,
                    distCoeffs?.let { handleOf(it) } ?: 0L,
                ) ?: throw OpenCVException("arucoDetector.refineDetectedMarkers", lastNativeError())
                val cornersMat = jvmMat(handles[0], "arucoDetector.refineDetectedMarkers")
                val idsMat = jvmMat(handles[1], "arucoDetector.refineDetectedMarkers")
                val rejectedMat = jvmMat(handles[2], "arucoDetector.refineDetectedMarkers")
                val recoveredMat = jvmMat(handles[3], "arucoDetector.refineDetectedMarkers")
                return RefinedMarkers(
                    corners = cornersMat.toMatListClosing(),
                    ids = idsMat,
                    rejectedCorners = rejectedMat.toMatListClosing(),
                    recoveredIdxs = recoveredMat,
                )
            }
        }
    }

    override fun detectMarkersMultiDict(image: Mat): MultiDictDetection {
        val handles = JniObjdetect.arucoDetectorDetectMarkersMultiDict(check(), handleOf(image))
            ?: throw OpenCVException("arucoDetector.detectMarkersMultiDict", lastNativeError())
        val cornersMat = jvmMat(handles[0], "arucoDetector.detectMarkersMultiDict")
        val idsMat = jvmMat(handles[1], "arucoDetector.detectMarkersMultiDict")
        val rejectedMat = jvmMat(handles[2], "arucoDetector.detectMarkersMultiDict")
        val dictIndicesMat = jvmMat(handles[3], "arucoDetector.detectMarkersMultiDict")
        return MultiDictDetection(
            corners = cornersMat.toMatListClosing(),
            ids = idsMat,
            rejectedImgPoints = rejectedMat.toMatListClosing(),
            dictIndices = dictIndicesMat,
        )
    }

    override fun clear() { JniObjdetect.arucoDetectorClear(check()) }
    override fun empty(): Boolean = JniObjdetect.arucoDetectorEmpty(check())
    override fun save(filename: String) { JniObjdetect.arucoDetectorSave(check(), filename) }
    override fun getDefaultName(): String = JniObjdetect.arucoDetectorGetDefaultName(check()) ?: ""

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect.arucoDetectorRelease(h)
        }
    }
}

// =========================================================================
// CharucoDetector
// =========================================================================

internal class JvmCharucoDetector(@Volatile private var handle: Long) : CharucoDetector {
    internal fun check(): Long = handle.takeIf { it != 0L }
        ?: throw IllegalStateException("CharucoDetector is closed")

    override var board: CharucoBoard
        get() = JvmCharucoBoard(JniObjdetect.charucoDetectorGetBoard(check()))
        set(value) {
            val native = value as? JvmCharucoBoard
                ?: throw IllegalArgumentException("board belongs to another platform backend")
            JniObjdetect.charucoDetectorSetBoard(check(), native.check())
        }

    override var charucoParameters: CharucoParameters
        get() {
            val out = JniObjdetect.charucoDetectorGetCharucoParams(check())
            return CharucoParameters(
                cameraMatrix = out[0].takeIf { it != 0L }?.let { jvmMat(it, "charucoDetector.getCharucoParameters") },
                distCoeffs = out[1].takeIf { it != 0L }?.let { jvmMat(it, "charucoDetector.getCharucoParameters") },
                minMarkers = out[2].toInt(),
                tryRefineMarkers = out[3] != 0L,
                checkMarkers = out[4] != 0L,
            )
        }
        set(value) {
            JniObjdetect.charucoDetectorSetCharucoParams(
                check(),
                value.cameraMatrix?.let { handleOf(it) } ?: 0L,
                value.distCoeffs?.let { handleOf(it) } ?: 0L,
                value.minMarkers,
                value.tryRefineMarkers,
                value.checkMarkers,
            )
        }

    override var detectorParameters: DetectorParameters
        get() = JniObjdetect.charucoDetectorGetDetectorParams(check()).toDetectorParameters()
        set(value) { JniObjdetect.charucoDetectorSetDetectorParams(check(), value.toParamArray()) }

    override var refineParameters: RefineParameters
        get() = JniObjdetect.charucoDetectorGetRefineParams(check()).toRefineParameters()
        set(value) { JniObjdetect.charucoDetectorSetRefineParams(check(), value.toParamArray()) }

    override fun detectBoard(
        image: Mat,
        markerCorners: List<Mat>,
        markerIds: Mat?,
    ): CharucoBoardDetection {
        markerCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = JniObjdetect.charucoDetectorDetectBoard(
                check(), handleOf(image), handleOf(cornersContainer),
                markerIds?.let { handleOf(it) } ?: 0L,
            ) ?: throw OpenCVException("charucoDetector.detectBoard", lastNativeError())
            val cc = jvmMat(handles[0], "charucoDetector.detectBoard")
            val ci = jvmMat(handles[1], "charucoDetector.detectBoard")
            val mc = jvmMat(handles[2], "charucoDetector.detectBoard")
            val mi = jvmMat(handles[3], "charucoDetector.detectBoard")
            return CharucoBoardDetection(
                charucoCorners = cc,
                charucoIds = ci,
                markerCorners = mc.toMatListClosing(),
                markerIds = mi,
            )
        }
    }

    override fun detectDiamonds(
        image: Mat,
        markerCorners: List<Mat>,
        markerIds: Mat?,
    ): DiamondDetection {
        markerCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = JniObjdetect.charucoDetectorDetectDiamonds(
                check(), handleOf(image), handleOf(cornersContainer),
                markerIds?.let { handleOf(it) } ?: 0L,
            ) ?: throw OpenCVException("charucoDetector.detectDiamonds", lastNativeError())
            val dc = jvmMat(handles[0], "charucoDetector.detectDiamonds")
            val di = jvmMat(handles[1], "charucoDetector.detectDiamonds")
            val mc = jvmMat(handles[2], "charucoDetector.detectDiamonds")
            val mi = jvmMat(handles[3], "charucoDetector.detectDiamonds")
            return DiamondDetection(
                diamondCorners = dc.toMatListClosing(),
                diamondIds = di,
                markerCorners = mc.toMatListClosing(),
                markerIds = mi,
            )
        }
    }

    override fun clear() { JniObjdetect.charucoDetectorClear(check()) }
    override fun empty(): Boolean = JniObjdetect.charucoDetectorEmpty(check())
    override fun save(filename: String) { JniObjdetect.charucoDetectorSave(check(), filename) }
    override fun getDefaultName(): String = JniObjdetect.charucoDetectorGetDefaultName(check()) ?: ""

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniObjdetect.charucoDetectorRelease(h)
        }
    }
}

// =========================================================================
// factories
// =========================================================================

actual fun dictionary(bytesList: Mat, markerSize: Int, maxCorrectionBits: Int): Dictionary =
    JvmDictionary(JniObjdetect.dictionaryCreate(handleOf(bytesList), markerSize, maxCorrectionBits).ifNull("dictionary"))

actual fun getPredefinedDictionary(dict: Int): Dictionary =
    JvmDictionary(JniObjdetect.getPredefinedDictionary(dict).ifNull("getPredefinedDictionary"))

actual fun extendDictionary(nMarkers: Int, markerSize: Int, baseDictionary: Dictionary?, randomSeed: Int): Dictionary {
    val base = (baseDictionary as? JvmDictionary)?.check() ?: 0L
    return JvmDictionary(JniObjdetect.extendDictionary(nMarkers, markerSize, base, randomSeed).ifNull("extendDictionary"))
}

actual fun board(objPoints: List<Mat>, dictionary: Dictionary, ids: Mat): Board {
    val dict = dictionary as? JvmDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return objPoints.toVectorMatContainer().use { container ->
        JvmBoard(JniObjdetect.boardCreate(handleOf(container), dict.check(), handleOf(ids)).ifNull("board"))
    }
}

actual fun gridBoard(
    size: Size,
    markerLength: Float,
    markerSeparation: Float,
    dictionary: Dictionary,
    ids: Mat?,
): GridBoard {
    val dict = dictionary as? JvmDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return JvmGridBoard(JniObjdetect.gridBoardCreate(
        size.width.toDouble(), size.height.toDouble(), markerLength, markerSeparation,
        dict.check(), ids?.let { handleOf(it) } ?: 0L,
    ).ifNull("gridBoard"))
}

actual fun charucoBoard(
    size: Size,
    squareLength: Float,
    markerLength: Float,
    dictionary: Dictionary,
    ids: Mat?,
): CharucoBoard {
    val dict = dictionary as? JvmDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return JvmCharucoBoard(JniObjdetect.charucoBoardCreate(
        size.width.toDouble(), size.height.toDouble(), squareLength, markerLength,
        dict.check(), ids?.let { handleOf(it) } ?: 0L,
    ).ifNull("charucoBoard"))
}

actual fun arucoDetector(
    dictionary: Dictionary,
    detectorParams: DetectorParameters,
    refineParams: RefineParameters,
): ArucoDetector {
    val dict = dictionary as? JvmDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return JvmArucoDetector(JniObjdetect.arucoDetectorCreate(
        dict.check(), detectorParams.toParamArray(), refineParams.toParamArray(),
    ).ifNull("arucoDetector"))
}

actual fun charucoDetector(
    board: CharucoBoard,
    charucoParams: CharucoParameters,
    detectorParams: DetectorParameters,
    refineParams: RefineParameters,
): CharucoDetector {
    val nativeBoard = board as? JvmCharucoBoard
        ?: throw IllegalArgumentException("board belongs to another platform backend")
    return JvmCharucoDetector(JniObjdetect.charucoDetectorCreate(
        nativeBoard.check(),
        charucoParams.cameraMatrix?.let { handleOf(it) } ?: 0L,
        charucoParams.distCoeffs?.let { handleOf(it) } ?: 0L,
        charucoParams.minMarkers,
        charucoParams.tryRefineMarkers,
        charucoParams.checkMarkers,
        detectorParams.toParamArray(),
        refineParams.toParamArray(),
    ).ifNull("charucoDetector"))
}

// =========================================================================
// Objdetect statics
// =========================================================================

actual fun findChessboardCorners(
    image: Mat,
    patternSize: Size,
    flags: Int,
): ChessboardCornersResult {
    val found = BooleanArray(1)
    val corners = JniObjdetect.findChessboardCorners(
        handleOf(image), patternSize.width.toDouble(), patternSize.height.toDouble(), flags, found,
    ).ifNull("findChessboardCorners")
    return ChessboardCornersResult(found[0], jvmMat(corners, "findChessboardCorners"))
}

actual fun checkChessboard(img: Mat, size: Size): Boolean =
    JniObjdetect.checkChessboard(handleOf(img), size.width.toDouble(), size.height.toDouble())

actual fun findChessboardCornersSB(
    image: Mat,
    patternSize: Size,
    flags: Int,
): ChessboardCornersResult {
    val found = BooleanArray(1)
    val corners = JniObjdetect.findChessboardCornersSB(
        handleOf(image), patternSize.width.toDouble(), patternSize.height.toDouble(), flags, found,
    ).ifNull("findChessboardCornersSB")
    return ChessboardCornersResult(found[0], jvmMat(corners, "findChessboardCornersSB"))
}

actual fun findChessboardCornersSBWithMeta(
    image: Mat,
    patternSize: Size,
    flags: Int,
): ChessboardSbMetaResult {
    val found = BooleanArray(1)
    val handles = JniObjdetect.findChessboardCornersSBWithMeta(
        handleOf(image), patternSize.width.toDouble(), patternSize.height.toDouble(), flags, found,
    ) ?: throw OpenCVException("findChessboardCornersSBWithMeta", lastNativeError())
    return ChessboardSbMetaResult(
        found[0],
        jvmMat(handles[0], "findChessboardCornersSBWithMeta"),
        jvmMat(handles[1], "findChessboardCornersSBWithMeta"),
    )
}

actual fun estimateChessboardSharpness(
    image: Mat,
    patternSize: Size,
    corners: Mat,
    riseDistance: Float,
    vertical: Boolean,
): ChessboardSharpnessResult {
    val scalar = DoubleArray(4)
    val sharpness = JniObjdetect.estimateChessboardSharpness(
        handleOf(image), patternSize.width.toDouble(), patternSize.height.toDouble(),
        handleOf(corners), riseDistance, vertical, scalar,
    ).ifNull("estimateChessboardSharpness")
    return ChessboardSharpnessResult(
        Scalar(scalar[0], scalar[1], scalar[2], scalar[3]),
        jvmMat(sharpness, "estimateChessboardSharpness"),
    )
}

actual fun find4QuadCornerSubpix(img: Mat, corners: Mat, regionSize: Size): Boolean =
    JniObjdetect.find4QuadCornerSubpix(
        handleOf(img), handleOf(corners), regionSize.width.toDouble(), regionSize.height.toDouble(),
    )

actual fun drawChessboardCorners(
    image: Mat,
    patternSize: Size,
    corners: Mat,
    patternWasFound: Boolean,
) {
    JniObjdetect.drawChessboardCorners(
        handleOf(image), patternSize.width.toDouble(), patternSize.height.toDouble(),
        handleOf(corners), patternWasFound,
    )
}

actual fun findCirclesGrid(
    image: Mat,
    patternSize: Size,
    flags: Int,
): CirclesGridResult {
    val found = BooleanArray(1)
    val centers = JniObjdetect.findCirclesGrid(
        handleOf(image), patternSize.width.toDouble(), patternSize.height.toDouble(), flags, found,
    ).ifNull("findCirclesGrid")
    return CirclesGridResult(found[0], jvmMat(centers, "findCirclesGrid"))
}

actual fun drawDetectedMarkers(
    image: Mat,
    corners: List<Mat>,
    ids: Mat?,
    borderColor: Scalar,
) {
    corners.toVectorMatContainer().use { container ->
        JniObjdetect.drawDetectedMarkers(
            handleOf(image), handleOf(container), ids?.let { handleOf(it) } ?: 0L,
            borderColor.v0, borderColor.v1, borderColor.v2, borderColor.v3,
        )
    }
}

actual fun generateImageMarker(
    dictionary: Dictionary,
    id: Int,
    sidePixels: Int,
    borderBits: Int,
): Mat {
    val dict = dictionary as? JvmDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return jvmMat(JniObjdetect.generateImageMarker(dict.check(), id, sidePixels, borderBits), "generateImageMarker")
}

actual fun drawDetectedCornersCharuco(
    image: Mat,
    charucoCorners: Mat,
    charucoIds: Mat?,
    cornerColor: Scalar,
) {
    JniObjdetect.drawDetectedCornersCharuco(
        handleOf(image), handleOf(charucoCorners), charucoIds?.let { handleOf(it) } ?: 0L,
        cornerColor.v0, cornerColor.v1, cornerColor.v2, cornerColor.v3,
    )
}

actual fun drawDetectedDiamonds(
    image: Mat,
    diamondCorners: List<Mat>,
    diamondIds: Mat?,
    borderColor: Scalar,
) {
    diamondCorners.toVectorMatContainer().use { container ->
        JniObjdetect.drawDetectedDiamonds(
            handleOf(image), handleOf(container), diamondIds?.let { handleOf(it) } ?: 0L,
            borderColor.v0, borderColor.v1, borderColor.v2, borderColor.v3,
        )
    }
}

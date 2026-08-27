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

import cvk.cvk_aruco_detector_clear
import cvk.cvk_aruco_detector_create
import cvk.cvk_aruco_detector_detect_markers
import cvk.cvk_aruco_detector_detect_markers_multi_dict
import cvk.cvk_aruco_detector_detect_markers_with_confidence
import cvk.cvk_aruco_detector_empty
import cvk.cvk_aruco_detector_get_default_name
import cvk.cvk_aruco_detector_get_detector_params
import cvk.cvk_aruco_detector_get_dictionary
import cvk.cvk_aruco_detector_get_refine_params
import cvk.cvk_aruco_detector_refine_detected_markers
import cvk.cvk_aruco_detector_release
import cvk.cvk_aruco_detector_save
import cvk.cvk_aruco_detector_set_detector_params
import cvk.cvk_aruco_detector_set_dictionary
import cvk.cvk_aruco_detector_set_refine_params
import cvk.cvk_aruco_detector_t
import cvk.cvk_board_create
import cvk.cvk_board_generate_image
import cvk.cvk_board_get_dictionary
import cvk.cvk_board_get_ids
import cvk.cvk_board_get_obj_points
import cvk.cvk_board_get_right_bottom_corner
import cvk.cvk_board_match_image_points
import cvk.cvk_board_release
import cvk.cvk_board_t
import cvk.cvk_charuco_board_check_charuco_corners_collinear
import cvk.cvk_charuco_board_create
import cvk.cvk_charuco_board_get_chessboard_corners
import cvk.cvk_charuco_board_get_chessboard_size
import cvk.cvk_charuco_board_get_legacy_pattern
import cvk.cvk_charuco_board_get_marker_length
import cvk.cvk_charuco_board_get_square_length
import cvk.cvk_charuco_board_release
import cvk.cvk_charuco_board_set_legacy_pattern
import cvk.cvk_charuco_board_t
import cvk.cvk_charuco_detector_clear
import cvk.cvk_charuco_detector_create
import cvk.cvk_charuco_detector_detect_board
import cvk.cvk_charuco_detector_detect_diamonds
import cvk.cvk_charuco_detector_empty
import cvk.cvk_charuco_detector_get_board
import cvk.cvk_charuco_detector_get_charuco_params
import cvk.cvk_charuco_detector_get_default_name
import cvk.cvk_charuco_detector_get_detector_params
import cvk.cvk_charuco_detector_get_refine_params
import cvk.cvk_charuco_detector_release
import cvk.cvk_charuco_detector_save
import cvk.cvk_charuco_detector_set_board
import cvk.cvk_charuco_detector_set_charuco_params
import cvk.cvk_charuco_detector_set_detector_params
import cvk.cvk_charuco_detector_set_refine_params
import cvk.cvk_charuco_detector_t
import cvk.cvk_charuco_params_t
import cvk.cvk_check_chessboard
import cvk.cvk_refine_params_t
import kotlin.concurrent.Volatile
import cvk.cvk_detector_params_t
import cvk.cvk_dictionary_create
import cvk.cvk_dictionary_generate_image_marker
import cvk.cvk_dictionary_get_bits_from_byte_list
import cvk.cvk_dictionary_get_byte_list_from_bits
import cvk.cvk_dictionary_get_bytes_list
import cvk.cvk_dictionary_get_distance_to_id
import cvk.cvk_dictionary_get_marker_bits
import cvk.cvk_dictionary_get_marker_size
import cvk.cvk_dictionary_get_max_correction_bits
import cvk.cvk_dictionary_identify
import cvk.cvk_dictionary_identify_pixel_ratio
import cvk.cvk_dictionary_release
import cvk.cvk_dictionary_set_bytes_list
import cvk.cvk_dictionary_set_marker_size
import cvk.cvk_dictionary_set_max_correction_bits
import cvk.cvk_dictionary_t
import cvk.cvk_draw_chessboard_corners
import cvk.cvk_draw_detected_corners_charuco
import cvk.cvk_draw_detected_diamonds
import cvk.cvk_draw_detected_markers
import cvk.cvk_estimate_chessboard_sharpness
import cvk.cvk_extend_dictionary
import cvk.cvk_find4_quad_corner_subpix
import cvk.cvk_find_chessboard_corners
import cvk.cvk_find_chessboard_corners_sb
import cvk.cvk_find_chessboard_corners_sb_with_meta
import cvk.cvk_find_circles_grid
import cvk.cvk_generate_image_marker
import cvk.cvk_get_predefined_dictionary
import cvk.cvk_grid_board_create
import cvk.cvk_grid_board_get_grid_size
import cvk.cvk_grid_board_get_marker_length
import cvk.cvk_grid_board_get_marker_separation
import cvk.cvk_grid_board_release
import cvk.cvk_grid_board_t
import cvk.cvk_last_error
import cvk.cvk_mat_t
import cvk.cvk_scalar_t
import kotlinx.cinterop.CArrayPointer
import kotlinx.cinterop.CValue
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.cValue
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.toCPointer
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.toKString
import kotlinx.cinterop.useContents
import kotlinx.cinterop.value

private fun lastNativeError(): String? {
    val message = cvk_last_error() ?: return null
    return message.toKString()
}

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
        val addr = m.nativeHandle().rawValue.toLong()
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
        result.add(NativeMat(addr.toCPointer<cvk_mat_t>() ?: throw OpenCVException("decodeMatList", lastNativeError())))
        offset += 8
    }
    return result
}

/** Decodes a fresh container Mat into caller-owned Mats and closes the container. */
private fun Mat.toMatListClosing(): List<Mat> = use { it.toMatList() }

// ---- params struct <-> Kotlin data classes --------------------------------

private fun cvk_detector_params_t.fromCvk(): DetectorParameters = DetectorParameters(
    adaptive_thresh_win_size_min,
    adaptive_thresh_win_size_max,
    adaptive_thresh_win_size_step,
    adaptive_thresh_constant,
    min_marker_perimeter_rate,
    max_marker_perimeter_rate,
    polygonal_approx_accuracy_rate,
    min_corner_distance_rate,
    min_distance_to_border,
    min_marker_distance_rate,
    min_group_distance,
    corner_refinement_method,
    corner_refinement_win_size,
    relative_corner_refinment_win_size,
    corner_refinement_max_iterations,
    corner_refinement_min_accuracy,
    marker_border_bits,
    perspective_remove_pixel_per_cell,
    perspective_remove_ignored_margin_per_cell,
    max_erroneous_bits_in_border_rate,
    min_otsu_std_dev,
    error_correction_rate,
    april_tag_quad_decimate,
    april_tag_quad_sigma,
    april_tag_min_cluster_pixels,
    april_tag_max_nmaxima,
    april_tag_critical_rad,
    april_tag_max_line_fit_mse,
    april_tag_min_white_black_diff,
    april_tag_deglitch,
    detect_inverted_marker != 0,
    use_aruco != 0,
    min_side_length_canonical_img,
    min_marker_length_ratio_original_img,
    valid_bit_id_threshold,
)

private fun DetectorParameters.toCvk(): CValue<cvk_detector_params_t> = cValue<cvk_detector_params_t> {
    adaptive_thresh_win_size_min = this@toCvk.adaptiveThreshWinSizeMin
    adaptive_thresh_win_size_max = this@toCvk.adaptiveThreshWinSizeMax
    adaptive_thresh_win_size_step = this@toCvk.adaptiveThreshWinSizeStep
    adaptive_thresh_constant = this@toCvk.adaptiveThreshConstant
    min_marker_perimeter_rate = this@toCvk.minMarkerPerimeterRate
    max_marker_perimeter_rate = this@toCvk.maxMarkerPerimeterRate
    polygonal_approx_accuracy_rate = this@toCvk.polygonalApproxAccuracyRate
    min_corner_distance_rate = this@toCvk.minCornerDistanceRate
    min_distance_to_border = this@toCvk.minDistanceToBorder
    min_marker_distance_rate = this@toCvk.minMarkerDistanceRate
    min_group_distance = this@toCvk.minGroupDistance
    corner_refinement_method = this@toCvk.cornerRefinementMethod
    corner_refinement_win_size = this@toCvk.cornerRefinementWinSize
    relative_corner_refinment_win_size = this@toCvk.relativeCornerRefinmentWinSize
    corner_refinement_max_iterations = this@toCvk.cornerRefinementMaxIterations
    corner_refinement_min_accuracy = this@toCvk.cornerRefinementMinAccuracy
    marker_border_bits = this@toCvk.markerBorderBits
    perspective_remove_pixel_per_cell = this@toCvk.perspectiveRemovePixelPerCell
    perspective_remove_ignored_margin_per_cell = this@toCvk.perspectiveRemoveIgnoredMarginPerCell
    max_erroneous_bits_in_border_rate = this@toCvk.maxErroneousBitsInBorderRate
    min_otsu_std_dev = this@toCvk.minOtsuStdDev
    error_correction_rate = this@toCvk.errorCorrectionRate
    april_tag_quad_decimate = this@toCvk.aprilTagQuadDecimate
    april_tag_quad_sigma = this@toCvk.aprilTagQuadSigma
    april_tag_min_cluster_pixels = this@toCvk.aprilTagMinClusterPixels
    april_tag_max_nmaxima = this@toCvk.aprilTagMaxNmaxima
    april_tag_critical_rad = this@toCvk.aprilTagCriticalRad
    april_tag_max_line_fit_mse = this@toCvk.aprilTagMaxLineFitMse
    april_tag_min_white_black_diff = this@toCvk.aprilTagMinWhiteBlackDiff
    april_tag_deglitch = this@toCvk.aprilTagDeglitch
    detect_inverted_marker = if (this@toCvk.detectInvertedMarker) 1 else 0
    use_aruco = if (this@toCvk.useAruco) 1 else 0
    min_side_length_canonical_img = this@toCvk.minSideLengthCanonicalImg
    min_marker_length_ratio_original_img = this@toCvk.minMarkerLengthRatioOriginalImg
    valid_bit_id_threshold = this@toCvk.validBitIdThreshold
}

private fun cvk_refine_params_t.fromCvk(): RefineParameters = RefineParameters(
    min_rep_distance,
    error_correction_rate,
    check_all_orders != 0,
)

private fun RefineParameters.toCvk(): CValue<cvk_refine_params_t> = cValue<cvk_refine_params_t> {
    min_rep_distance = this@toCvk.minRepDistance
    error_correction_rate = this@toCvk.errorCorrectionRate
    check_all_orders = if (this@toCvk.checkAllOrders) 1 else 0
}

private fun cvk_charuco_params_t.fromCvk(): CharucoParameters = CharucoParameters(
    cameraMatrix = camera_matrix?.let { nativeMat(it, "charucoParams.cameraMatrix") },
    distCoeffs = dist_coeffs?.let { nativeMat(it, "charucoParams.distCoeffs") },
    minMarkers = min_markers,
    tryRefineMarkers = try_refine_markers != 0,
    checkMarkers = check_markers != 0,
)

private fun CharucoParameters.toCvk(): CValue<cvk_charuco_params_t> = cValue<cvk_charuco_params_t> {
    camera_matrix = cameraMatrix?.nativeHandle()
    dist_coeffs = distCoeffs?.nativeHandle()
    min_markers = this@toCvk.minMarkers
    try_refine_markers = if (this@toCvk.tryRefineMarkers) 1 else 0
    check_markers = if (this@toCvk.checkMarkers) 1 else 0
}

// =========================================================================
// Dictionary
// =========================================================================

internal class NativeDictionary(@Volatile private var raw: CPointer<cvk_dictionary_t>?) : Dictionary {
    internal fun check(): CPointer<cvk_dictionary_t> =
        raw ?: throw IllegalStateException("Dictionary is closed")

    override val bytesList: Mat
        get() = nativeMat(cvk_dictionary_get_bytes_list(check()), "dictionary.bytesList")
    override val markerSize: Int get() = cvk_dictionary_get_marker_size(check())
    override val maxCorrectionBits: Int get() = cvk_dictionary_get_max_correction_bits(check())
    override fun setBytesList(bytesList: Mat) { cvk_dictionary_set_bytes_list(check(), bytesList.nativeHandle()) }
    override fun setMarkerSize(markerSize: Int) { cvk_dictionary_set_marker_size(check(), markerSize) }
    override fun setMaxCorrectionBits(maxCorrectionBits: Int) { cvk_dictionary_set_max_correction_bits(check(), maxCorrectionBits) }

    override fun identify(onlyBits: Mat, maxCorrectionRate: Double): IdentifyResult = memScoped {
        val idx = alloc<IntVar>()
        val rotation = alloc<IntVar>()
        val found = cvk_dictionary_identify(
            check(), onlyBits.nativeHandle(), maxCorrectionRate, idx.ptr, rotation.ptr,
        ) != 0
        IdentifyResult(found, idx.value, rotation.value)
    }

    override fun identify(onlyCellPixelRatio: Mat, maxCorrectionRate: Double, validBitIdThreshold: Float): IdentifyResult =
        memScoped {
            val idx = alloc<IntVar>()
            val rotation = alloc<IntVar>()
            val found = cvk_dictionary_identify_pixel_ratio(
                check(), onlyCellPixelRatio.nativeHandle(), maxCorrectionRate,
                validBitIdThreshold, idx.ptr, rotation.ptr,
            ) != 0
            IdentifyResult(found, idx.value, rotation.value)
        }

    override fun getDistanceToId(bits: Mat, id: Int, allRotations: Boolean): Int =
        cvk_dictionary_get_distance_to_id(check(), bits.nativeHandle(), id, if (allRotations) 1 else 0)

    override fun generateImageMarker(id: Int, sidePixels: Int, borderBits: Int): Mat =
        nativeMat(cvk_dictionary_generate_image_marker(check(), id, sidePixels, borderBits), "dictionary.generateImageMarker")

    override fun getMarkerBits(markerId: Int, rotationId: Int): Mat =
        nativeMat(cvk_dictionary_get_marker_bits(check(), markerId, rotationId), "dictionary.getMarkerBits")

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_dictionary_release(h)
    }
}

// =========================================================================
// Board
// =========================================================================

internal class NativeBoard(@Volatile private var raw: CPointer<cvk_board_t>?) : Board {
    internal fun check(): CPointer<cvk_board_t> =
        raw ?: throw IllegalStateException("Board is closed")

    override val dictionary: Dictionary
        get() = NativeDictionary(cvk_board_get_dictionary(check()))
    override val objPoints: List<Mat>
        get() = nativeMat(cvk_board_get_obj_points(check()), "board.objPoints").toMatListClosing()
    override val ids: Mat
        get() = nativeMat(cvk_board_get_ids(check()), "board.ids")
    override val rightBottomCorner: Point3
        get() = memScoped {
            val out = allocArray<DoubleVar>(3)
            cvk_board_get_right_bottom_corner(check(), out)
            Point3(out[0], out[1], out[2])
        }

    override fun matchImagePoints(detectedCorners: List<Mat>, detectedIds: Mat): BoardMatchPoints {
        detectedCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = memScoped {
                val obj = alloc<CPointerVar<cvk_mat_t>>()
                val img = alloc<CPointerVar<cvk_mat_t>>()
                val ok = cvk_board_match_image_points(
                    check(), cornersContainer.nativeHandle(), detectedIds.nativeHandle(),
                    obj.ptr, img.ptr,
                )
                if (ok == 0) throw OpenCVException("board.matchImagePoints", lastNativeError())
                obj.value to img.value
            }
            return BoardMatchPoints(
                nativeMat(handles.first, "board.matchImagePoints"),
                nativeMat(handles.second, "board.matchImagePoints"),
            )
        }
    }

    override fun generateImage(outSize: Size, marginSize: Int, borderBits: Int): Mat =
        nativeMat(
            cvk_board_generate_image(check(), outSize.width.toDouble(), outSize.height.toDouble(), marginSize, borderBits),
            "board.generateImage",
        )

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_board_release(h)
    }
}

internal class NativeGridBoard(@Volatile private var raw: CPointer<cvk_grid_board_t>?) : GridBoard {
    internal fun check(): CPointer<cvk_grid_board_t> =
        raw ?: throw IllegalStateException("GridBoard is closed")

    /** The Board base subobject sits at offset 0 of the GridBoard handle. */
    internal fun boardHandle(): CPointer<cvk_board_t> = check().reinterpret()

    override val dictionary: Dictionary
        get() = NativeDictionary(cvk_board_get_dictionary(boardHandle()))
    override val objPoints: List<Mat>
        get() = nativeMat(cvk_board_get_obj_points(boardHandle()), "board.objPoints").toMatListClosing()
    override val ids: Mat
        get() = nativeMat(cvk_board_get_ids(boardHandle()), "board.ids")
    override val rightBottomCorner: Point3
        get() = memScoped {
            val out = allocArray<DoubleVar>(3)
            cvk_board_get_right_bottom_corner(boardHandle(), out)
            Point3(out[0], out[1], out[2])
        }

    override fun matchImagePoints(detectedCorners: List<Mat>, detectedIds: Mat): BoardMatchPoints {
        detectedCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = memScoped {
                val obj = alloc<CPointerVar<cvk_mat_t>>()
                val img = alloc<CPointerVar<cvk_mat_t>>()
                val ok = cvk_board_match_image_points(
                    boardHandle(), cornersContainer.nativeHandle(), detectedIds.nativeHandle(),
                    obj.ptr, img.ptr,
                )
                if (ok == 0) throw OpenCVException("board.matchImagePoints", lastNativeError())
                obj.value to img.value
            }
            return BoardMatchPoints(
                nativeMat(handles.first, "board.matchImagePoints"),
                nativeMat(handles.second, "board.matchImagePoints"),
            )
        }
    }

    override fun generateImage(outSize: Size, marginSize: Int, borderBits: Int): Mat =
        nativeMat(
            cvk_board_generate_image(boardHandle(), outSize.width.toDouble(), outSize.height.toDouble(), marginSize, borderBits),
            "board.generateImage",
        )

    override val gridSize: Size
        get() = memScoped {
            val out = allocArray<IntVar>(2)
            cvk_grid_board_get_grid_size(check(), out)
            Size(out[0], out[1])
        }
    override val markerLength: Float get() = cvk_grid_board_get_marker_length(check())
    override val markerSeparation: Float get() = cvk_grid_board_get_marker_separation(check())

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_grid_board_release(h)
    }
}

internal class NativeCharucoBoard(@Volatile private var raw: CPointer<cvk_charuco_board_t>?) : CharucoBoard {
    internal fun check(): CPointer<cvk_charuco_board_t> =
        raw ?: throw IllegalStateException("CharucoBoard is closed")

    /** The Board base subobject sits at offset 0 of the CharucoBoard handle. */
    internal fun boardHandle(): CPointer<cvk_board_t> = check().reinterpret()

    override val dictionary: Dictionary
        get() = NativeDictionary(cvk_board_get_dictionary(boardHandle()))
    override val objPoints: List<Mat>
        get() = nativeMat(cvk_board_get_obj_points(boardHandle()), "board.objPoints").toMatListClosing()
    override val ids: Mat
        get() = nativeMat(cvk_board_get_ids(boardHandle()), "board.ids")
    override val rightBottomCorner: Point3
        get() = memScoped {
            val out = allocArray<DoubleVar>(3)
            cvk_board_get_right_bottom_corner(boardHandle(), out)
            Point3(out[0], out[1], out[2])
        }

    override fun matchImagePoints(detectedCorners: List<Mat>, detectedIds: Mat): BoardMatchPoints {
        detectedCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = memScoped {
                val obj = alloc<CPointerVar<cvk_mat_t>>()
                val img = alloc<CPointerVar<cvk_mat_t>>()
                val ok = cvk_board_match_image_points(
                    boardHandle(), cornersContainer.nativeHandle(), detectedIds.nativeHandle(),
                    obj.ptr, img.ptr,
                )
                if (ok == 0) throw OpenCVException("board.matchImagePoints", lastNativeError())
                obj.value to img.value
            }
            return BoardMatchPoints(
                nativeMat(handles.first, "board.matchImagePoints"),
                nativeMat(handles.second, "board.matchImagePoints"),
            )
        }
    }

    override fun generateImage(outSize: Size, marginSize: Int, borderBits: Int): Mat =
        nativeMat(
            cvk_board_generate_image(boardHandle(), outSize.width.toDouble(), outSize.height.toDouble(), marginSize, borderBits),
            "board.generateImage",
        )

    override var legacyPattern: Boolean
        get() = cvk_charuco_board_get_legacy_pattern(check()) != 0
        set(value) { cvk_charuco_board_set_legacy_pattern(check(), if (value) 1 else 0) }
    override val chessboardSize: Size
        get() = memScoped {
            val out = allocArray<IntVar>(2)
            cvk_charuco_board_get_chessboard_size(check(), out)
            Size(out[0], out[1])
        }
    override val squareLength: Float get() = cvk_charuco_board_get_square_length(check())
    override val markerLength: Float get() = cvk_charuco_board_get_marker_length(check())
    override val chessboardCorners: Mat
        get() = nativeMat(cvk_charuco_board_get_chessboard_corners(check()), "charucoBoard.chessboardCorners")

    override fun checkCharucoCornersCollinear(charucoIds: Mat): Boolean =
        cvk_charuco_board_check_charuco_corners_collinear(check(), charucoIds.nativeHandle()) != 0

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_charuco_board_release(h)
    }
}

// =========================================================================
// ArucoDetector
// =========================================================================

internal class NativeArucoDetector(@Volatile private var raw: CPointer<cvk_aruco_detector_t>?) : ArucoDetector {
    private fun check(): CPointer<cvk_aruco_detector_t> =
        raw ?: throw IllegalStateException("ArucoDetector is closed")

    override var dictionary: Dictionary
        get() = NativeDictionary(cvk_aruco_detector_get_dictionary(check()))
        set(value) {
            val native = value as? NativeDictionary
                ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
            cvk_aruco_detector_set_dictionary(check(), native.check())
        }

    override var detectorParameters: DetectorParameters
        get() = memScoped {
            val params = alloc<cvk_detector_params_t>()
            cvk_aruco_detector_get_detector_params(check(), params.ptr)
            params.fromCvk()
        }
        set(value) { value.toCvk().useContents { cvk_aruco_detector_set_detector_params(check(), ptr) } }

    override var refineParameters: RefineParameters
        get() = memScoped {
            val params = alloc<cvk_refine_params_t>()
            cvk_aruco_detector_get_refine_params(check(), params.ptr)
            params.fromCvk()
        }
        set(value) { value.toCvk().useContents { cvk_aruco_detector_set_refine_params(check(), ptr) } }

    override fun detectMarkers(image: Mat): MarkerDetection = memScoped {
        val corners = alloc<CPointerVar<cvk_mat_t>>()
        val ids = alloc<CPointerVar<cvk_mat_t>>()
        val rejected = alloc<CPointerVar<cvk_mat_t>>()
        val ok = cvk_aruco_detector_detect_markers(
            check(), image.nativeHandle(), corners.ptr, ids.ptr, rejected.ptr,
        )
        if (ok == 0) throw OpenCVException("arucoDetector.detectMarkers", lastNativeError())
        val cornersMat = nativeMat(corners.value, "arucoDetector.detectMarkers")
        val idsMat = nativeMat(ids.value, "arucoDetector.detectMarkers")
        val rejectedMat = nativeMat(rejected.value, "arucoDetector.detectMarkers")
        MarkerDetection(
            corners = cornersMat.toMatListClosing(),
            ids = idsMat,
            rejectedImgPoints = rejectedMat.toMatListClosing(),
        )
    }

    override fun detectMarkersWithConfidence(image: Mat): MarkerDetectionWithConfidence = memScoped {
        val corners = alloc<CPointerVar<cvk_mat_t>>()
        val ids = alloc<CPointerVar<cvk_mat_t>>()
        val confidence = alloc<CPointerVar<cvk_mat_t>>()
        val rejected = alloc<CPointerVar<cvk_mat_t>>()
        val ok = cvk_aruco_detector_detect_markers_with_confidence(
            check(), image.nativeHandle(), corners.ptr, ids.ptr, confidence.ptr, rejected.ptr,
        )
        if (ok == 0) throw OpenCVException("arucoDetector.detectMarkersWithConfidence", lastNativeError())
        val cornersMat = nativeMat(corners.value, "arucoDetector.detectMarkersWithConfidence")
        val idsMat = nativeMat(ids.value, "arucoDetector.detectMarkersWithConfidence")
        val confidenceMat = nativeMat(confidence.value, "arucoDetector.detectMarkersWithConfidence")
        val rejectedMat = nativeMat(rejected.value, "arucoDetector.detectMarkersWithConfidence")
        MarkerDetectionWithConfidence(
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
                val handles = memScoped {
                    val corners = alloc<CPointerVar<cvk_mat_t>>()
                    val ids = alloc<CPointerVar<cvk_mat_t>>()
                    val rejected = alloc<CPointerVar<cvk_mat_t>>()
                    val recovered = alloc<CPointerVar<cvk_mat_t>>()
                    val boardHandle = (board as? NativeBoard)?.check()
                        ?: (board as? NativeGridBoard)?.boardHandle()
                        ?: (board as? NativeCharucoBoard)?.boardHandle()
                        ?: throw IllegalArgumentException("board belongs to another platform backend")
                    val ok = cvk_aruco_detector_refine_detected_markers(
                        check(), image.nativeHandle(), boardHandle,
                        cornersContainer.nativeHandle(), detectedIds.nativeHandle(),
                        rejectedContainer.nativeHandle(),
                        cameraMatrix?.nativeHandle(), distCoeffs?.nativeHandle(),
                        corners.ptr, ids.ptr, rejected.ptr, recovered.ptr,
                    )
                    if (ok == 0) throw OpenCVException("arucoDetector.refineDetectedMarkers", lastNativeError())
                    val cornersMat = nativeMat(corners.value, "arucoDetector.refineDetectedMarkers")
                    val idsMat = nativeMat(ids.value, "arucoDetector.refineDetectedMarkers")
                    val rejectedMat = nativeMat(rejected.value, "arucoDetector.refineDetectedMarkers")
                    val recoveredMat = nativeMat(recovered.value, "arucoDetector.refineDetectedMarkers")
                    RefinedMarkers(
                        corners = cornersMat.toMatListClosing(),
                        ids = idsMat,
                        rejectedCorners = rejectedMat.toMatListClosing(),
                        recoveredIdxs = recoveredMat,
                    )
                }
                return handles
            }
        }
    }

    override fun detectMarkersMultiDict(image: Mat): MultiDictDetection = memScoped {
        val corners = alloc<CPointerVar<cvk_mat_t>>()
        val ids = alloc<CPointerVar<cvk_mat_t>>()
        val rejected = alloc<CPointerVar<cvk_mat_t>>()
        val dictIndices = alloc<CPointerVar<cvk_mat_t>>()
        val ok = cvk_aruco_detector_detect_markers_multi_dict(
            check(), image.nativeHandle(), corners.ptr, ids.ptr, rejected.ptr, dictIndices.ptr,
        )
        if (ok == 0) throw OpenCVException("arucoDetector.detectMarkersMultiDict", lastNativeError())
        val cornersMat = nativeMat(corners.value, "arucoDetector.detectMarkersMultiDict")
        val idsMat = nativeMat(ids.value, "arucoDetector.detectMarkersMultiDict")
        val rejectedMat = nativeMat(rejected.value, "arucoDetector.detectMarkersMultiDict")
        val dictIndicesMat = nativeMat(dictIndices.value, "arucoDetector.detectMarkersMultiDict")
        MultiDictDetection(
            corners = cornersMat.toMatListClosing(),
            ids = idsMat,
            rejectedImgPoints = rejectedMat.toMatListClosing(),
            dictIndices = dictIndicesMat,
        )
    }

    override fun clear() { cvk_aruco_detector_clear(check()) }
    override fun empty(): Boolean = cvk_aruco_detector_empty(check()) != 0
    override fun save(filename: String) { cvk_aruco_detector_save(check(), filename) }
    override fun getDefaultName(): String = cvk_aruco_detector_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_aruco_detector_release(h)
    }
}

// =========================================================================
// CharucoDetector
// =========================================================================

internal class NativeCharucoDetector(@Volatile private var raw: CPointer<cvk_charuco_detector_t>?) : CharucoDetector {
    private fun check(): CPointer<cvk_charuco_detector_t> =
        raw ?: throw IllegalStateException("CharucoDetector is closed")

    override var board: CharucoBoard
        get() = NativeCharucoBoard(cvk_charuco_detector_get_board(check()))
        set(value) {
            val native = value as? NativeCharucoBoard
                ?: throw IllegalArgumentException("board belongs to another platform backend")
            cvk_charuco_detector_set_board(check(), native.check())
        }

    override var charucoParameters: CharucoParameters
        get() = memScoped {
            val params = alloc<cvk_charuco_params_t>()
            cvk_charuco_detector_get_charuco_params(check(), params.ptr)
            params.fromCvk()
        }
        set(value) { value.toCvk().useContents { cvk_charuco_detector_set_charuco_params(check(), ptr) } }

    override var detectorParameters: DetectorParameters
        get() = memScoped {
            val params = alloc<cvk_detector_params_t>()
            cvk_charuco_detector_get_detector_params(check(), params.ptr)
            params.fromCvk()
        }
        set(value) { value.toCvk().useContents { cvk_charuco_detector_set_detector_params(check(), ptr) } }

    override var refineParameters: RefineParameters
        get() = memScoped {
            val params = alloc<cvk_refine_params_t>()
            cvk_charuco_detector_get_refine_params(check(), params.ptr)
            params.fromCvk()
        }
        set(value) { value.toCvk().useContents { cvk_charuco_detector_set_refine_params(check(), ptr) } }

    override fun detectBoard(
        image: Mat,
        markerCorners: List<Mat>,
        markerIds: Mat?,
    ): CharucoBoardDetection {
        markerCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = memScoped {
                val charucoCorners = alloc<CPointerVar<cvk_mat_t>>()
                val charucoIds = alloc<CPointerVar<cvk_mat_t>>()
                val cornersOut = alloc<CPointerVar<cvk_mat_t>>()
                val idsOut = alloc<CPointerVar<cvk_mat_t>>()
                val ok = cvk_charuco_detector_detect_board(
                    check(), image.nativeHandle(), cornersContainer.nativeHandle(),
                    markerIds?.nativeHandle(), charucoCorners.ptr, charucoIds.ptr,
                    cornersOut.ptr, idsOut.ptr,
                )
                if (ok == 0) throw OpenCVException("charucoDetector.detectBoard", lastNativeError())
                val cc = nativeMat(charucoCorners.value, "charucoDetector.detectBoard")
                val ci = nativeMat(charucoIds.value, "charucoDetector.detectBoard")
                val mc = nativeMat(cornersOut.value, "charucoDetector.detectBoard")
                val mi = nativeMat(idsOut.value, "charucoDetector.detectBoard")
                CharucoBoardDetection(
                    charucoCorners = cc,
                    charucoIds = ci,
                    markerCorners = mc.toMatListClosing(),
                    markerIds = mi,
                )
            }
            return handles
        }
    }

    override fun detectDiamonds(
        image: Mat,
        markerCorners: List<Mat>,
        markerIds: Mat?,
    ): DiamondDetection {
        markerCorners.toVectorMatContainer().use { cornersContainer ->
            val handles = memScoped {
                val diamondCorners = alloc<CPointerVar<cvk_mat_t>>()
                val diamondIds = alloc<CPointerVar<cvk_mat_t>>()
                val cornersOut = alloc<CPointerVar<cvk_mat_t>>()
                val idsOut = alloc<CPointerVar<cvk_mat_t>>()
                val ok = cvk_charuco_detector_detect_diamonds(
                    check(), image.nativeHandle(), cornersContainer.nativeHandle(),
                    markerIds?.nativeHandle(), diamondCorners.ptr, diamondIds.ptr,
                    cornersOut.ptr, idsOut.ptr,
                )
                if (ok == 0) throw OpenCVException("charucoDetector.detectDiamonds", lastNativeError())
                val dc = nativeMat(diamondCorners.value, "charucoDetector.detectDiamonds")
                val di = nativeMat(diamondIds.value, "charucoDetector.detectDiamonds")
                val mc = nativeMat(cornersOut.value, "charucoDetector.detectDiamonds")
                val mi = nativeMat(idsOut.value, "charucoDetector.detectDiamonds")
                DiamondDetection(
                    diamondCorners = dc.toMatListClosing(),
                    diamondIds = di,
                    markerCorners = mc.toMatListClosing(),
                    markerIds = mi,
                )
            }
            return handles
        }
    }

    override fun clear() { cvk_charuco_detector_clear(check()) }
    override fun empty(): Boolean = cvk_charuco_detector_empty(check()) != 0
    override fun save(filename: String) { cvk_charuco_detector_save(check(), filename) }
    override fun getDefaultName(): String = cvk_charuco_detector_get_default_name(check())?.toKString() ?: ""

    override fun close() {
        val h = raw ?: return
        raw = null
        cvk_charuco_detector_release(h)
    }
}

// =========================================================================
// factories
// =========================================================================

actual fun dictionary(bytesList: Mat, markerSize: Int, maxCorrectionBits: Int): Dictionary =
    NativeDictionary(cvk_dictionary_create(bytesList.nativeHandle(), markerSize, maxCorrectionBits)
        ?: throw OpenCVException("dictionary", lastNativeError()))

actual fun getPredefinedDictionary(dict: Int): Dictionary =
    NativeDictionary(cvk_get_predefined_dictionary(dict)
        ?: throw OpenCVException("getPredefinedDictionary", lastNativeError()))

actual fun extendDictionary(nMarkers: Int, markerSize: Int, baseDictionary: Dictionary?, randomSeed: Int): Dictionary {
    val base = (baseDictionary as? NativeDictionary)?.check()
    return NativeDictionary(cvk_extend_dictionary(nMarkers, markerSize, base, randomSeed)
        ?: throw OpenCVException("extendDictionary", lastNativeError()))
}

actual fun board(objPoints: List<Mat>, dictionary: Dictionary, ids: Mat): Board {
    val dict = dictionary as? NativeDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return objPoints.toVectorMatContainer().use { container ->
        NativeBoard(cvk_board_create(container.nativeHandle(), dict.check(), ids.nativeHandle())
            ?: throw OpenCVException("board", lastNativeError()))
    }
}

actual fun gridBoard(
    size: Size,
    markerLength: Float,
    markerSeparation: Float,
    dictionary: Dictionary,
    ids: Mat?,
): GridBoard {
    val dict = dictionary as? NativeDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return NativeGridBoard(cvk_grid_board_create(
        size.width.toDouble(), size.height.toDouble(), markerLength, markerSeparation,
        dict.check(), ids?.nativeHandle(),
    ) ?: throw OpenCVException("gridBoard", lastNativeError()))
}

actual fun charucoBoard(
    size: Size,
    squareLength: Float,
    markerLength: Float,
    dictionary: Dictionary,
    ids: Mat?,
): CharucoBoard {
    val dict = dictionary as? NativeDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return NativeCharucoBoard(cvk_charuco_board_create(
        size.width.toDouble(), size.height.toDouble(), squareLength, markerLength,
        dict.check(), ids?.nativeHandle(),
    ) ?: throw OpenCVException("charucoBoard", lastNativeError()))
}

actual fun arucoDetector(
    dictionary: Dictionary,
    detectorParams: DetectorParameters,
    refineParams: RefineParameters,
): ArucoDetector {
    val dict = dictionary as? NativeDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    val handle = memScoped {
        val dp = detectorParams.toCvk()
        val rp = refineParams.toCvk()
        cvk_aruco_detector_create(dict.check(), dp.ptr, rp.ptr)
    } ?: throw OpenCVException("arucoDetector", lastNativeError())
    return NativeArucoDetector(handle)
}

actual fun charucoDetector(
    board: CharucoBoard,
    charucoParams: CharucoParameters,
    detectorParams: DetectorParameters,
    refineParams: RefineParameters,
): CharucoDetector {
    val nativeBoard = board as? NativeCharucoBoard
        ?: throw IllegalArgumentException("board belongs to another platform backend")
    val handle = memScoped {
        val cp = charucoParams.toCvk()
        val dp = detectorParams.toCvk()
        val rp = refineParams.toCvk()
        cvk_charuco_detector_create(nativeBoard.check(), cp.ptr, dp.ptr, rp.ptr)
    } ?: throw OpenCVException("charucoDetector", lastNativeError())
    return NativeCharucoDetector(handle)
}

// =========================================================================
// Objdetect statics
// =========================================================================

actual fun findChessboardCorners(
    image: Mat,
    patternSize: Size,
    flags: Int,
): ChessboardCornersResult = memScoped {
    val cornersOut = alloc<CPointerVar<cvk_mat_t>>()
    val found = cvk_find_chessboard_corners(
        image.nativeHandle(), patternSize.width.toDouble(), patternSize.height.toDouble(),
        flags, cornersOut.ptr,
    ) != 0
    ChessboardCornersResult(found, nativeMat(cornersOut.value, "findChessboardCorners"))
}

actual fun checkChessboard(img: Mat, size: Size): Boolean =
    cvk_check_chessboard(img.nativeHandle(), size.width.toDouble(), size.height.toDouble()) != 0

actual fun findChessboardCornersSB(
    image: Mat,
    patternSize: Size,
    flags: Int,
): ChessboardCornersResult = memScoped {
    val cornersOut = alloc<CPointerVar<cvk_mat_t>>()
    val found = cvk_find_chessboard_corners_sb(
        image.nativeHandle(), patternSize.width.toDouble(), patternSize.height.toDouble(),
        flags, cornersOut.ptr,
    ) != 0
    ChessboardCornersResult(found, nativeMat(cornersOut.value, "findChessboardCornersSB"))
}

actual fun findChessboardCornersSBWithMeta(
    image: Mat,
    patternSize: Size,
    flags: Int,
): ChessboardSbMetaResult = memScoped {
    val cornersOut = alloc<CPointerVar<cvk_mat_t>>()
    val metaOut = alloc<CPointerVar<cvk_mat_t>>()
    val found = cvk_find_chessboard_corners_sb_with_meta(
        image.nativeHandle(), patternSize.width.toDouble(), patternSize.height.toDouble(),
        flags, cornersOut.ptr, metaOut.ptr,
    ) != 0
    ChessboardSbMetaResult(
        found,
        nativeMat(cornersOut.value, "findChessboardCornersSBWithMeta"),
        nativeMat(metaOut.value, "findChessboardCornersSBWithMeta"),
    )
}

actual fun estimateChessboardSharpness(
    image: Mat,
    patternSize: Size,
    corners: Mat,
    riseDistance: Float,
    vertical: Boolean,
): ChessboardSharpnessResult = memScoped {
    val scalar = alloc<cvk_scalar_t>()
    val sharpnessOut = alloc<CPointerVar<cvk_mat_t>>()
    val ok = cvk_estimate_chessboard_sharpness(
        image.nativeHandle(), patternSize.width.toDouble(), patternSize.height.toDouble(),
        corners.nativeHandle(), riseDistance, if (vertical) 1 else 0,
        scalar.ptr, sharpnessOut.ptr,
    )
    if (ok == 0) throw OpenCVException("estimateChessboardSharpness", lastNativeError())
    ChessboardSharpnessResult(
        Scalar(scalar.v0, scalar.v1, scalar.v2, scalar.v3),
        nativeMat(sharpnessOut.value, "estimateChessboardSharpness"),
    )
}

actual fun find4QuadCornerSubpix(img: Mat, corners: Mat, regionSize: Size): Boolean =
    cvk_find4_quad_corner_subpix(
        img.nativeHandle(), corners.nativeHandle(),
        regionSize.width.toDouble(), regionSize.height.toDouble(),
    ) != 0

actual fun drawChessboardCorners(
    image: Mat,
    patternSize: Size,
    corners: Mat,
    patternWasFound: Boolean,
) {
    cvk_draw_chessboard_corners(
        image.nativeHandle(), patternSize.width.toDouble(), patternSize.height.toDouble(),
        corners.nativeHandle(), if (patternWasFound) 1 else 0,
    )
}

actual fun findCirclesGrid(
    image: Mat,
    patternSize: Size,
    flags: Int,
): CirclesGridResult = memScoped {
    val centersOut = alloc<CPointerVar<cvk_mat_t>>()
    val found = cvk_find_circles_grid(
        image.nativeHandle(), patternSize.width.toDouble(), patternSize.height.toDouble(),
        flags, centersOut.ptr,
    ) != 0
    CirclesGridResult(found, nativeMat(centersOut.value, "findCirclesGrid"))
}

actual fun drawDetectedMarkers(
    image: Mat,
    corners: List<Mat>,
    ids: Mat?,
    borderColor: Scalar,
) {
    corners.toVectorMatContainer().use { container ->
        cvk_draw_detected_markers(
            image.nativeHandle(), container.nativeHandle(), ids?.nativeHandle(),
            borderColor.toCvk(),
        )
    }
}

actual fun generateImageMarker(
    dictionary: Dictionary,
    id: Int,
    sidePixels: Int,
    borderBits: Int,
): Mat {
    val dict = dictionary as? NativeDictionary
        ?: throw IllegalArgumentException("dictionary belongs to another platform backend")
    return nativeMat(
        cvk_generate_image_marker(dict.check(), id, sidePixels, borderBits),
        "generateImageMarker",
    )
}

actual fun drawDetectedCornersCharuco(
    image: Mat,
    charucoCorners: Mat,
    charucoIds: Mat?,
    cornerColor: Scalar,
) {
    cvk_draw_detected_corners_charuco(
        image.nativeHandle(), charucoCorners.nativeHandle(), charucoIds?.nativeHandle(),
        cornerColor.toCvk(),
    )
}

actual fun drawDetectedDiamonds(
    image: Mat,
    diamondCorners: List<Mat>,
    diamondIds: Mat?,
    borderColor: Scalar,
) {
    diamondCorners.toVectorMatContainer().use { container ->
        cvk_draw_detected_diamonds(
            image.nativeHandle(), container.nativeHandle(), diamondIds?.nativeHandle(),
            borderColor.toCvk(),
        )
    }
}

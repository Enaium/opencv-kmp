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

import cvk.cvk_free_mat_array
import cvk.cvk_last_error
import kotlin.concurrent.Volatile
import cvk.cvk_mat_t
import cvk.cvk_octree_clear
import cvk.cvk_octree_create_with_depth
import cvk.cvk_octree_create_with_depth_cloud
import cvk.cvk_octree_create_with_resolution
import cvk.cvk_octree_create_with_resolution_cloud
import cvk.cvk_octree_delete_point
import cvk.cvk_octree_empty
import cvk.cvk_octree_get_point_cloud
import cvk.cvk_octree_get_point_cloud_color
import cvk.cvk_octree_insert_point
import cvk.cvk_octree_insert_point_color
import cvk.cvk_octree_is_point_in_bound
import cvk.cvk_octree_knn_search
import cvk.cvk_octree_knn_search_color
import cvk.cvk_octree_radius_nn_search
import cvk.cvk_octree_radius_nn_search_color
import cvk.cvk_octree_release
import cvk.cvk_octree_t
import cvk.cvk_odometry_compute_depth
import cvk.cvk_odometry_compute_frames
import cvk.cvk_odometry_compute_rgbd
import cvk.cvk_odometry_create
import cvk.cvk_odometry_create_settings
import cvk.cvk_odometry_create_type
import cvk.cvk_odometry_frame_create
import cvk.cvk_odometry_frame_get_depth
import cvk.cvk_odometry_frame_get_gray_image
import cvk.cvk_odometry_frame_get_image
import cvk.cvk_odometry_frame_get_mask
import cvk.cvk_odometry_frame_get_normals
import cvk.cvk_odometry_frame_get_processed_depth
import cvk.cvk_odometry_frame_get_pyramid_at
import cvk.cvk_odometry_frame_get_pyramid_levels
import cvk.cvk_odometry_frame_release
import cvk.cvk_odometry_frame_t
import cvk.cvk_odometry_get_normals_computer
import cvk.cvk_odometry_prepare_frame
import cvk.cvk_odometry_prepare_frames
import cvk.cvk_odometry_release
import cvk.cvk_odometry_t
import cvk.cvk_pose_graph_add_edge
import cvk.cvk_pose_graph_add_node
import cvk.cvk_pose_graph_create
import cvk.cvk_pose_graph_get_pose
import cvk.cvk_pose_graph_optimize
import cvk.cvk_pose_graph_release
import cvk.cvk_pose_graph_t
import cvk.cvk_ptcloud_depth_to_3d
import cvk.cvk_ptcloud_depth_to_3d_sparse
import cvk.cvk_ptcloud_find_planes
import cvk.cvk_ptcloud_load_mesh
import cvk.cvk_ptcloud_load_point_cloud
import cvk.cvk_ptcloud_register_depth
import cvk.cvk_ptcloud_rescale_depth
import cvk.cvk_ptcloud_save_mesh
import cvk.cvk_ptcloud_save_point_cloud
import cvk.cvk_ptcloud_triangle_rasterize
import cvk.cvk_ptcloud_triangle_rasterize_color
import cvk.cvk_ptcloud_triangle_rasterize_depth
import cvk.cvk_ptcloud_warp_frame
import cvk.cvk_rgbd_normals_apply
import cvk.cvk_rgbd_normals_cache
import cvk.cvk_rgbd_normals_create
import cvk.cvk_rgbd_normals_get_cols
import cvk.cvk_rgbd_normals_get_depth
import cvk.cvk_rgbd_normals_get_k
import cvk.cvk_rgbd_normals_get_method
import cvk.cvk_rgbd_normals_get_rows
import cvk.cvk_rgbd_normals_get_window_size
import cvk.cvk_rgbd_normals_release
import cvk.cvk_rgbd_normals_set_cols
import cvk.cvk_rgbd_normals_set_k
import cvk.cvk_rgbd_normals_set_rows
import cvk.cvk_rgbd_normals_set_window_size
import cvk.cvk_rgbd_normals_t
import cvk.cvk_volume_create
import cvk.cvk_volume_create_settings
import cvk.cvk_volume_fetch_normals
import cvk.cvk_volume_fetch_points_normals
import cvk.cvk_volume_fetch_points_normals_colors
import cvk.cvk_volume_get_bounding_box
import cvk.cvk_volume_get_enable_growth
import cvk.cvk_volume_get_total_volume_units
import cvk.cvk_volume_get_visible_blocks
import cvk.cvk_volume_integrate
import cvk.cvk_volume_integrate_color
import cvk.cvk_volume_integrate_frame
import cvk.cvk_volume_raycast
import cvk.cvk_volume_raycast_color
import cvk.cvk_volume_raycast_ex
import cvk.cvk_volume_raycast_ex_color
import cvk.cvk_volume_release
import cvk.cvk_volume_reset
import cvk.cvk_volume_set_enable_growth
import cvk.cvk_volume_t
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ptr
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.get
import kotlinx.cinterop.set
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.refTo
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value

private fun lastNativeError(): String? = cvk_last_error()?.toKString()

private fun Mat?.handleOrNull(): CPointer<cvk_mat_t>? = this?.nativeHandle()

// =========================================================================
// Octree
// =========================================================================

internal class NativeOctree(@Volatile private var raw: CPointer<cvk_octree_t>?) : Octree {

    private fun check(): CPointer<cvk_octree_t> =
        raw ?: throw IllegalStateException("Octree is closed")

    override fun insertPoint(point: Point3, color: Point3?): Boolean =
        if (color != null) {
            cvk_octree_insert_point_color(
                check(), point.x, point.y, point.z, color.x, color.y, color.z,
            ) != 0
        } else {
            cvk_octree_insert_point(check(), point.x, point.y, point.z) != 0
        }

    override fun isPointInBound(point: Point3): Boolean =
        cvk_octree_is_point_in_bound(check(), point.x, point.y, point.z) != 0

    override val empty: Boolean get() = cvk_octree_empty(check()) != 0

    override fun clear() {
        cvk_octree_clear(check())
    }

    override fun deletePoint(point: Point3): Boolean =
        cvk_octree_delete_point(check(), point.x, point.y, point.z) != 0

    override fun getPointCloudByOctree(): Mat =
        nativeMat(cvk_octree_get_point_cloud(check()), "Octree.getPointCloudByOctree")

    override fun getPointCloudByOctreeWithColors(): Pair<Mat, Mat> = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val colors = alloc<CPointerVar<cvk_mat_t>>()
        cvk_octree_get_point_cloud_color(check(), points.ptr, colors.ptr)
        Pair(
            nativeMat(points.value, "Octree.getPointCloudByOctree.points"),
            nativeMat(colors.value, "Octree.getPointCloudByOctree.colors"),
        )
    }

    override fun radiusNNSearch(query: Point3, radius: Float): OctreeRadiusNNSearchResult = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val dists = alloc<CPointerVar<cvk_mat_t>>()
        val count = cvk_octree_radius_nn_search(
            check(), query.x, query.y, query.z, radius, points.ptr, dists.ptr,
        )
        OctreeRadiusNNSearchResult(
            count = count,
            points = nativeMat(points.value, "Octree.radiusNNSearch.points"),
            squareDists = nativeMat(dists.value, "Octree.radiusNNSearch.squareDists"),
        )
    }

    override fun radiusNNSearch(query: Point3, radius: Float, includeColors: Boolean): OctreeRadiusNNSearchColorResult =
        memScoped {
            val points = alloc<CPointerVar<cvk_mat_t>>()
            val colors = alloc<CPointerVar<cvk_mat_t>>()
            val dists = alloc<CPointerVar<cvk_mat_t>>()
            val count = cvk_octree_radius_nn_search_color(
                check(), query.x, query.y, query.z, radius, points.ptr, colors.ptr, dists.ptr,
            )
            OctreeRadiusNNSearchColorResult(
                count = count,
                points = nativeMat(points.value, "Octree.radiusNNSearch.points"),
                colors = nativeMat(colors.value, "Octree.radiusNNSearch.colors"),
                squareDists = nativeMat(dists.value, "Octree.radiusNNSearch.squareDists"),
            )
        }

    override fun KNNSearch(query: Point3, k: Int): OctreeKNNSearchResult = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val dists = alloc<CPointerVar<cvk_mat_t>>()
        cvk_octree_knn_search(check(), query.x, query.y, query.z, k, points.ptr, dists.ptr)
        OctreeKNNSearchResult(
            points = nativeMat(points.value, "Octree.KNNSearch.points"),
            squareDists = nativeMat(dists.value, "Octree.KNNSearch.squareDists"),
        )
    }

    override fun KNNSearch(query: Point3, k: Int, includeColors: Boolean): OctreeKNNSearchColorResult =
        memScoped {
            val points = alloc<CPointerVar<cvk_mat_t>>()
            val colors = alloc<CPointerVar<cvk_mat_t>>()
            val dists = alloc<CPointerVar<cvk_mat_t>>()
            cvk_octree_knn_search_color(check(), query.x, query.y, query.z, k, points.ptr, colors.ptr, dists.ptr)
            OctreeKNNSearchColorResult(
                points = nativeMat(points.value, "Octree.KNNSearch.points"),
                colors = nativeMat(colors.value, "Octree.KNNSearch.colors"),
                squareDists = nativeMat(dists.value, "Octree.KNNSearch.squareDists"),
            )
        }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_octree_release(handle)
    }
}

actual fun octreeCreateWithDepth(maxDepth: Int, size: Double, origin: Point3, withColors: Boolean): Octree =
    NativeOctree(cvk_octree_create_with_depth(
        maxDepth, size, origin.x, origin.y, origin.z, if (withColors) 1 else 0,
    ) ?: throw OpenCVException("octreeCreateWithDepth", lastNativeError()))

actual fun octreeCreateWithDepth(maxDepth: Int, pointCloud: Mat, colors: Mat?): Octree =
    NativeOctree(cvk_octree_create_with_depth_cloud(
        maxDepth, pointCloud.nativeHandle(), colors.handleOrNull(),
    ) ?: throw OpenCVException("octreeCreateWithDepth", lastNativeError()))

actual fun octreeCreateWithResolution(resolution: Double, size: Double, origin: Point3, withColors: Boolean): Octree =
    NativeOctree(cvk_octree_create_with_resolution(
        resolution, size, origin.x, origin.y, origin.z, if (withColors) 1 else 0,
    ) ?: throw OpenCVException("octreeCreateWithResolution", lastNativeError()))

actual fun octreeCreateWithResolution(resolution: Double, pointCloud: Mat, colors: Mat?): Octree =
    NativeOctree(cvk_octree_create_with_resolution_cloud(
        resolution, pointCloud.nativeHandle(), colors.handleOrNull(),
    ) ?: throw OpenCVException("octreeCreateWithResolution", lastNativeError()))

// =========================================================================
// Odometry
// =========================================================================

internal class NativeOdometry(@Volatile private var raw: CPointer<cvk_odometry_t>?) : Odometry {

    private fun check(): CPointer<cvk_odometry_t> =
        raw ?: throw IllegalStateException("Odometry is closed")

    override fun prepareFrame(frame: OdometryFrame) {
        cvk_odometry_prepare_frame(check(), (frame as NativeOdometryFrame).check())
    }

    override fun prepareFrames(srcFrame: OdometryFrame, dstFrame: OdometryFrame) {
        cvk_odometry_prepare_frames(
            check(), (srcFrame as NativeOdometryFrame).check(), (dstFrame as NativeOdometryFrame).check(),
        )
    }

    override fun compute(srcFrame: OdometryFrame, dstFrame: OdometryFrame): OdometryComputeResult = memScoped {
        val rt = alloc<CPointerVar<cvk_mat_t>>()
        val ok = cvk_odometry_compute_frames(
            check(), (srcFrame as NativeOdometryFrame).check(), (dstFrame as NativeOdometryFrame).check(), rt.ptr,
        )
        OdometryComputeResult(
            success = ok != 0,
            rt = nativeMat(rt.value, "Odometry.compute.rt"),
        )
    }

    override fun compute(srcDepth: Mat, dstDepth: Mat): OdometryComputeResult = memScoped {
        val rt = alloc<CPointerVar<cvk_mat_t>>()
        val ok = cvk_odometry_compute_depth(
            check(), srcDepth.nativeHandle(), dstDepth.nativeHandle(), rt.ptr,
        )
        OdometryComputeResult(
            success = ok != 0,
            rt = nativeMat(rt.value, "Odometry.compute.rt"),
        )
    }

    override fun compute(srcDepth: Mat, srcRgb: Mat, dstDepth: Mat, dstRgb: Mat): OdometryComputeResult = memScoped {
        val rt = alloc<CPointerVar<cvk_mat_t>>()
        val ok = cvk_odometry_compute_rgbd(
            check(), srcDepth.nativeHandle(), srcRgb.nativeHandle(), dstDepth.nativeHandle(),
            dstRgb.nativeHandle(), rt.ptr,
        )
        OdometryComputeResult(
            success = ok != 0,
            rt = nativeMat(rt.value, "Odometry.compute.rt"),
        )
    }

    override fun getNormalsComputer(): RgbdNormals? =
        cvk_odometry_get_normals_computer(check())?.let { NativeRgbdNormals(it) }

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_odometry_release(handle)
    }
}

actual fun odometryCreate(otype: Int, settings: OdometrySettings?, algType: Int): Odometry {
    val raw = if (settings == null) {
        cvk_odometry_create_type(otype)
    } else {
        cvk_odometry_create_settings(
            otype,
            settings.cameraMatrix.handleOrNull(),
            settings.iterCounts.handleOrNull(),
            settings.minDepth, settings.maxDepth, settings.maxDepthDiff, settings.maxPointsPart,
            settings.sobelSize, settings.sobelScale, settings.normalWinSize,
            settings.normalDiffThreshold, settings.normalMethod, settings.angleThreshold,
            settings.maxTranslation, settings.maxRotation, settings.minGradientMagnitude,
            settings.minGradientMagnitudes.handleOrNull(),
            algType,
        )
    }
    return NativeOdometry(raw ?: throw OpenCVException("odometryCreate", lastNativeError()))
}

// =========================================================================
// OdometryFrame
// =========================================================================

internal class NativeOdometryFrame(@Volatile private var raw: CPointer<cvk_odometry_frame_t>?) : OdometryFrame {

    internal fun check(): CPointer<cvk_odometry_frame_t> =
        raw ?: throw IllegalStateException("OdometryFrame is closed")

    override fun getImage(): Mat =
        nativeMat(cvk_odometry_frame_get_image(check()), "OdometryFrame.getImage")

    override fun getGrayImage(): Mat =
        nativeMat(cvk_odometry_frame_get_gray_image(check()), "OdometryFrame.getGrayImage")

    override fun getDepth(): Mat =
        nativeMat(cvk_odometry_frame_get_depth(check()), "OdometryFrame.getDepth")

    override fun getProcessedDepth(): Mat =
        nativeMat(cvk_odometry_frame_get_processed_depth(check()), "OdometryFrame.getProcessedDepth")

    override fun getMask(): Mat =
        nativeMat(cvk_odometry_frame_get_mask(check()), "OdometryFrame.getMask")

    override fun getNormals(): Mat =
        nativeMat(cvk_odometry_frame_get_normals(check()), "OdometryFrame.getNormals")

    override val pyramidLevels: Int
        get() = cvk_odometry_frame_get_pyramid_levels(check())

    override fun getPyramidAt(pyrType: Int, level: Long): Mat =
        nativeMat(cvk_odometry_frame_get_pyramid_at(check(), pyrType, level.toULong()), "OdometryFrame.getPyramidAt")

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_odometry_frame_release(handle)
    }
}

actual fun odometryFrameCreate(depth: Mat?, image: Mat?, mask: Mat?, normals: Mat?): OdometryFrame =
    NativeOdometryFrame(cvk_odometry_frame_create(
        depth.handleOrNull(), image.handleOrNull(), mask.handleOrNull(), normals.handleOrNull(),
    ) ?: throw OpenCVException("odometryFrameCreate", lastNativeError()))

// =========================================================================
// RgbdNormals
// =========================================================================

internal class NativeRgbdNormals(@Volatile private var raw: CPointer<cvk_rgbd_normals_t>?) : RgbdNormals {

    internal fun check(): CPointer<cvk_rgbd_normals_t> =
        raw ?: throw IllegalStateException("RgbdNormals is closed")

    override fun apply(points: Mat): Mat =
        nativeMat(cvk_rgbd_normals_apply(check(), points.nativeHandle()), "RgbdNormals.apply")

    override fun cache() {
        cvk_rgbd_normals_cache(check())
    }

    override var rows: Int
        get() = cvk_rgbd_normals_get_rows(check())
        set(value) = cvk_rgbd_normals_set_rows(check(), value)

    override var cols: Int
        get() = cvk_rgbd_normals_get_cols(check())
        set(value) = cvk_rgbd_normals_set_cols(check(), value)

    override var windowSize: Int
        get() = cvk_rgbd_normals_get_window_size(check())
        set(value) = cvk_rgbd_normals_set_window_size(check(), value)

    override val depth: Int get() = cvk_rgbd_normals_get_depth(check())

    override var k: Mat
        get() = nativeMat(cvk_rgbd_normals_get_k(check()), "RgbdNormals.k")
        set(value) = cvk_rgbd_normals_set_k(check(), value.nativeHandle())

    override val method: Int get() = cvk_rgbd_normals_get_method(check())

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_rgbd_normals_release(handle)
    }
}

actual fun rgbdNormalsCreate(
    rows: Int,
    cols: Int,
    depth: Int,
    k: Mat?,
    windowSize: Int,
    diffThreshold: Float,
    method: Int,
): RgbdNormals = NativeRgbdNormals(cvk_rgbd_normals_create(
    rows, cols, depth, k.handleOrNull(), windowSize, diffThreshold, method,
) ?: throw OpenCVException("rgbdNormalsCreate", lastNativeError()))

// =========================================================================
// Volume
// =========================================================================

internal class NativeVolume(@Volatile private var raw: CPointer<cvk_volume_t>?) : Volume {

    private fun check(): CPointer<cvk_volume_t> =
        raw ?: throw IllegalStateException("Volume is closed")

    override fun integrateFrame(frame: OdometryFrame, pose: Mat) {
        cvk_volume_integrate_frame(check(), (frame as NativeOdometryFrame).check(), pose.nativeHandle())
    }

    override fun integrate(depth: Mat, pose: Mat) {
        cvk_volume_integrate(check(), depth.nativeHandle(), pose.nativeHandle())
    }

    override fun integrateColor(depth: Mat, image: Mat, pose: Mat) {
        cvk_volume_integrate_color(check(), depth.nativeHandle(), image.nativeHandle(), pose.nativeHandle())
    }

    override fun raycast(cameraPose: Mat): VolumeRaycastResult = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val normals = alloc<CPointerVar<cvk_mat_t>>()
        cvk_volume_raycast(check(), cameraPose.nativeHandle(), points.ptr, normals.ptr)
        VolumeRaycastResult(
            points = nativeMat(points.value, "Volume.raycast.points"),
            normals = nativeMat(normals.value, "Volume.raycast.normals"),
        )
    }

    override fun raycastColor(cameraPose: Mat): VolumeRaycastColorResult = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val normals = alloc<CPointerVar<cvk_mat_t>>()
        val colors = alloc<CPointerVar<cvk_mat_t>>()
        cvk_volume_raycast_color(check(), cameraPose.nativeHandle(), points.ptr, normals.ptr, colors.ptr)
        VolumeRaycastColorResult(
            points = nativeMat(points.value, "Volume.raycastColor.points"),
            normals = nativeMat(normals.value, "Volume.raycastColor.normals"),
            colors = nativeMat(colors.value, "Volume.raycastColor.colors"),
        )
    }

    override fun raycastEx(cameraPose: Mat, height: Int, width: Int, k: Mat): VolumeRaycastResult = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val normals = alloc<CPointerVar<cvk_mat_t>>()
        cvk_volume_raycast_ex(check(), cameraPose.nativeHandle(), height, width, k.nativeHandle(), points.ptr, normals.ptr)
        VolumeRaycastResult(
            points = nativeMat(points.value, "Volume.raycastEx.points"),
            normals = nativeMat(normals.value, "Volume.raycastEx.normals"),
        )
    }

    override fun raycastExColor(cameraPose: Mat, height: Int, width: Int, k: Mat): VolumeRaycastColorResult =
        memScoped {
            val points = alloc<CPointerVar<cvk_mat_t>>()
            val normals = alloc<CPointerVar<cvk_mat_t>>()
            val colors = alloc<CPointerVar<cvk_mat_t>>()
            cvk_volume_raycast_ex_color(
                check(), cameraPose.nativeHandle(), height, width, k.nativeHandle(),
                points.ptr, normals.ptr, colors.ptr,
            )
            VolumeRaycastColorResult(
                points = nativeMat(points.value, "Volume.raycastExColor.points"),
                normals = nativeMat(normals.value, "Volume.raycastExColor.normals"),
                colors = nativeMat(colors.value, "Volume.raycastExColor.colors"),
            )
        }

    override fun fetchNormals(points: Mat): Mat =
        nativeMat(cvk_volume_fetch_normals(check(), points.nativeHandle()), "Volume.fetchNormals")

    override fun fetchPointsNormals(): VolumePointsNormalsResult = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val normals = alloc<CPointerVar<cvk_mat_t>>()
        cvk_volume_fetch_points_normals(check(), points.ptr, normals.ptr)
        VolumePointsNormalsResult(
            points = nativeMat(points.value, "Volume.fetchPointsNormals.points"),
            normals = nativeMat(normals.value, "Volume.fetchPointsNormals.normals"),
        )
    }

    override fun fetchPointsNormalsColors(): VolumePointsNormalsColorsResult = memScoped {
        val points = alloc<CPointerVar<cvk_mat_t>>()
        val normals = alloc<CPointerVar<cvk_mat_t>>()
        val colors = alloc<CPointerVar<cvk_mat_t>>()
        cvk_volume_fetch_points_normals_colors(check(), points.ptr, normals.ptr, colors.ptr)
        VolumePointsNormalsColorsResult(
            points = nativeMat(points.value, "Volume.fetchPointsNormalsColors.points"),
            normals = nativeMat(normals.value, "Volume.fetchPointsNormalsColors.normals"),
            colors = nativeMat(colors.value, "Volume.fetchPointsNormalsColors.colors"),
        )
    }

    override fun reset() {
        cvk_volume_reset(check())
    }

    override val visibleBlocks: Int get() = cvk_volume_get_visible_blocks(check())

    override val totalVolumeUnits: Long get() = cvk_volume_get_total_volume_units(check()).toLong()

    override fun getBoundingBox(precision: Int): Mat =
        nativeMat(cvk_volume_get_bounding_box(check(), precision), "Volume.getBoundingBox")

    override var enableGrowth: Boolean
        get() = cvk_volume_get_enable_growth(check()) != 0
        set(value) = cvk_volume_set_enable_growth(check(), if (value) 1 else 0)

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_volume_release(handle)
    }
}

actual fun volumeCreate(vtype: Int, settings: VolumeSettings?): Volume {
    val raw = if (settings == null) {
        cvk_volume_create(vtype)
    } else {
        cvk_volume_create_settings(
            vtype, settings.integrateWidth, settings.integrateHeight, settings.raycastWidth,
            settings.raycastHeight, settings.depthFactor, settings.voxelSize,
            settings.tsdfTruncateDistance, settings.maxDepth, settings.maxWeight,
            settings.raycastStepFactor, settings.volumePose.handleOrNull(),
            settings.volumeResolution.handleOrNull(),
            settings.cameraIntegrateIntrinsics.handleOrNull(),
            settings.cameraRaycastIntrinsics.handleOrNull(),
        )
    }
    return NativeVolume(raw ?: throw OpenCVException("volumeCreate", lastNativeError()))
}

// =========================================================================
// PoseGraph
// =========================================================================

internal class NativePoseGraph(@Volatile private var raw: CPointer<cvk_pose_graph_t>?) : PoseGraph {

    private fun check(): CPointer<cvk_pose_graph_t> =
        raw ?: throw IllegalStateException("PoseGraph is closed")

    override fun addNode(nodeId: Long, pose: Mat, fixed: Boolean) {
        cvk_pose_graph_add_node(check(), nodeId.toULong(), pose.nativeHandle(), if (fixed) 1 else 0)
    }

    override fun addEdge(source: Long, target: Long, transformation: Mat, information: Mat?) {
        cvk_pose_graph_add_edge(
            check(), source.toULong(), target.toULong(), transformation.nativeHandle(), information.handleOrNull(),
        )
    }

    override fun optimize(): Int = cvk_pose_graph_optimize(check())

    override fun getPose(nodeId: Long): Mat =
        nativeMat(cvk_pose_graph_get_pose(check(), nodeId.toULong()), "PoseGraph.getPose")

    override fun close() {
        val handle = raw ?: return
        raw = null
        cvk_pose_graph_release(handle)
    }
}

actual fun poseGraphCreate(): PoseGraph =
    NativePoseGraph(cvk_pose_graph_create() ?: throw OpenCVException("poseGraphCreate", lastNativeError()))

// =========================================================================
// Ptcloud statics
// =========================================================================

actual fun loadPointCloud(filename: String): PointCloudData = memScoped {
    val vertices = alloc<CPointerVar<cvk_mat_t>>()
    val normals = alloc<CPointerVar<cvk_mat_t>>()
    val rgb = alloc<CPointerVar<cvk_mat_t>>()
    cvk_ptcloud_load_point_cloud(filename, vertices.ptr, normals.ptr, rgb.ptr)
    PointCloudData(
        vertices = nativeMat(vertices.value, "loadPointCloud.vertices"),
        normals = nativeMat(normals.value, "loadPointCloud.normals"),
        rgb = nativeMat(rgb.value, "loadPointCloud.rgb"),
    )
}

actual fun savePointCloud(filename: String, vertices: Mat, normals: Mat?, rgb: Mat?) {
    cvk_ptcloud_save_point_cloud(filename, vertices.nativeHandle(), normals.handleOrNull(), rgb.handleOrNull())
}

actual fun loadMesh(filename: String): MeshData = memScoped {
    val vertices = alloc<CPointerVar<cvk_mat_t>>()
    val normals = alloc<CPointerVar<cvk_mat_t>>()
    val colors = alloc<CPointerVar<cvk_mat_t>>()
    val texCoords = alloc<CPointerVar<cvk_mat_t>>()
    val indices = alloc<CPointerVar<CPointerVar<cvk_mat_t>>>()
    val count = alloc<IntVar>()
    cvk_ptcloud_load_mesh(filename, vertices.ptr, indices.ptr, count.ptr, normals.ptr, colors.ptr, texCoords.ptr)
    val faceHandles = indices.value ?: throw OpenCVException("loadMesh", lastNativeError())
    val faces = (0 until count.value).map { i ->
        nativeMat(faceHandles[i], "loadMesh.indices")
    }
    cvk_free_mat_array(faceHandles)
    MeshData(
        vertices = nativeMat(vertices.value, "loadMesh.vertices"),
        indices = faces,
        normals = nativeMat(normals.value, "loadMesh.normals"),
        colors = nativeMat(colors.value, "loadMesh.colors"),
        texCoords = nativeMat(texCoords.value, "loadMesh.texCoords"),
    )
}

actual fun saveMesh(
    filename: String,
    vertices: Mat,
    indices: List<Mat>,
    normals: Mat?,
    colors: Mat?,
    texCoords: Mat?,
) = memScoped {
    val faceArray = allocArray<CPointerVar<cvk_mat_t>>(if (indices.isEmpty()) 1 else indices.size)
    indices.forEachIndexed { i, m -> faceArray[i] = m.nativeHandle() }
    cvk_ptcloud_save_mesh(
        filename, vertices.nativeHandle(),
        if (indices.isEmpty()) null else faceArray, indices.size,
        normals.handleOrNull(), colors.handleOrNull(), texCoords.handleOrNull(),
    )
}

actual fun triangleRasterize(
    vertices: Mat,
    indices: Mat,
    colors: Mat,
    colorBuf: Mat,
    depthBuf: Mat,
    world2cam: Mat,
    fovY: Double,
    zNear: Double,
    zFar: Double,
    settings: TriangleRasterizeSettings,
) {
    cvk_ptcloud_triangle_rasterize(
        vertices.nativeHandle(), indices.nativeHandle(), colors.nativeHandle(),
        colorBuf.nativeHandle(), depthBuf.nativeHandle(), world2cam.nativeHandle(), fovY,
        zNear, zFar, settings.shadingType, settings.cullingMode, settings.glCompatibleMode,
    )
}

actual fun triangleRasterizeDepth(
    vertices: Mat,
    indices: Mat,
    depthBuf: Mat,
    world2cam: Mat,
    fovY: Double,
    zNear: Double,
    zFar: Double,
    settings: TriangleRasterizeSettings,
) {
    cvk_ptcloud_triangle_rasterize_depth(
        vertices.nativeHandle(), indices.nativeHandle(), depthBuf.nativeHandle(),
        world2cam.nativeHandle(), fovY, zNear, zFar, settings.shadingType,
        settings.cullingMode, settings.glCompatibleMode,
    )
}

actual fun triangleRasterizeColor(
    vertices: Mat,
    indices: Mat,
    colors: Mat,
    colorBuf: Mat,
    world2cam: Mat,
    fovY: Double,
    zNear: Double,
    zFar: Double,
    settings: TriangleRasterizeSettings,
) {
    cvk_ptcloud_triangle_rasterize_color(
        vertices.nativeHandle(), indices.nativeHandle(), colors.nativeHandle(),
        colorBuf.nativeHandle(), world2cam.nativeHandle(), fovY, zNear, zFar,
        settings.shadingType, settings.cullingMode, settings.glCompatibleMode,
    )
}

actual fun registerDepth(
    unregisteredCameraMatrix: Mat,
    registeredCameraMatrix: Mat,
    registeredDistCoeffs: Mat,
    rt: Mat,
    unregisteredDepth: Mat,
    outputImagePlaneSize: Size,
    depthDilation: Boolean,
): Mat = memScoped {
    val registered = alloc<CPointerVar<cvk_mat_t>>()
    val ok = cvk_ptcloud_register_depth(
        unregisteredCameraMatrix.nativeHandle(), registeredCameraMatrix.nativeHandle(),
        registeredDistCoeffs.nativeHandle(), rt.nativeHandle(), unregisteredDepth.nativeHandle(),
        outputImagePlaneSize.width, outputImagePlaneSize.height, if (depthDilation) 1 else 0,
        registered.ptr,
    )
    if (ok == 0) throw OpenCVException("registerDepth", lastNativeError())
    nativeMat(registered.value, "registerDepth")
}

actual fun depthTo3dSparse(depth: Mat, inK: Mat, inPoints: Mat): Mat =
    nativeMat(
        cvk_ptcloud_depth_to_3d_sparse(depth.nativeHandle(), inK.nativeHandle(), inPoints.nativeHandle()),
        "depthTo3dSparse",
    )

actual fun depthTo3d(depth: Mat, k: Mat, mask: Mat?): Mat =
    nativeMat(
        cvk_ptcloud_depth_to_3d(depth.nativeHandle(), k.nativeHandle(), mask.handleOrNull()),
        "depthTo3d",
    )

actual fun rescaleDepth(input: Mat, type: Int, depthFactor: Double): Mat =
    nativeMat(cvk_ptcloud_rescale_depth(input.nativeHandle(), type, depthFactor), "rescaleDepth")

actual fun warpFrame(
    depth: Mat,
    image: Mat?,
    mask: Mat?,
    rt: Mat,
    cameraMatrix: Mat,
): WarpFrameResult = memScoped {
    val warpedDepth = alloc<CPointerVar<cvk_mat_t>>()
    val warpedImage = alloc<CPointerVar<cvk_mat_t>>()
    val warpedMask = alloc<CPointerVar<cvk_mat_t>>()
    cvk_ptcloud_warp_frame(
        depth.nativeHandle(), image.handleOrNull(), mask.handleOrNull(), rt.nativeHandle(),
        cameraMatrix.nativeHandle(), warpedDepth.ptr, warpedImage.ptr, warpedMask.ptr,
    )
    WarpFrameResult(
        warpedDepth = nativeMat(warpedDepth.value, "warpFrame.warpedDepth"),
        warpedImage = nativeMat(warpedImage.value, "warpFrame.warpedImage"),
        warpedMask = nativeMat(warpedMask.value, "warpFrame.warpedMask"),
    )
}

actual fun findPlanes(
    points3d: Mat,
    normals: Mat?,
    blockSize: Int,
    minSize: Int,
    threshold: Double,
    sensorErrorA: Double,
    sensorErrorB: Double,
    sensorErrorC: Double,
    method: Int,
): PlanesResult = memScoped {
    val mask = alloc<CPointerVar<cvk_mat_t>>()
    val coefficients = alloc<CPointerVar<cvk_mat_t>>()
    cvk_ptcloud_find_planes(
        points3d.nativeHandle(), normals.handleOrNull(), mask.ptr, coefficients.ptr, blockSize,
        minSize, threshold, sensorErrorA, sensorErrorB, sensorErrorC, method,
    )
    PlanesResult(
        mask = nativeMat(mask.value, "findPlanes.mask"),
        planeCoefficients = nativeMat(coefficients.value, "findPlanes.planeCoefficients"),
    )
}

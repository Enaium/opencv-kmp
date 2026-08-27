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

import kotlin.concurrent.Volatile

private fun lastNativeError(): String? = Jni.lastError()

/** Validates a fresh factory handle; throws with the native error text when it is 0. */
private fun Long.checkHandle(operation: String): Long =
    takeIf { it != 0L } ?: throw OpenCVException(operation, lastNativeError())

// =========================================================================
// Octree
// =========================================================================

internal class JvmOctree(@Volatile private var handle: Long) : Octree {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Octree is closed")

    override fun insertPoint(point: Point3, color: Point3?): Boolean =
        if (color != null) {
            JniPtcloud.octreeInsertPointColor(
                check(), point.x, point.y, point.z, color.x, color.y, color.z,
            )
        } else {
            JniPtcloud.octreeInsertPoint(check(), point.x, point.y, point.z)
        }

    override fun isPointInBound(point: Point3): Boolean =
        JniPtcloud.octreeIsPointInBound(check(), point.x, point.y, point.z)

    override val empty: Boolean get() = JniPtcloud.octreeEmpty(check())

    override fun clear() {
        JniPtcloud.octreeClear(check())
    }

    override fun deletePoint(point: Point3): Boolean =
        JniPtcloud.octreeDeletePoint(check(), point.x, point.y, point.z)

    override fun getPointCloudByOctree(): Mat =
        jvmMat(JniPtcloud.octreeGetPointCloud(check()), "Octree.getPointCloudByOctree")

    override fun getPointCloudByOctreeWithColors(): Pair<Mat, Mat> {
        val handles = JniPtcloud.octreeGetPointCloudColor(check())
        return jvmMat(handles[0], "Octree.getPointCloudByOctree.points") to
            jvmMat(handles[1], "Octree.getPointCloudByOctree.colors")
    }

    override fun radiusNNSearch(query: Point3, radius: Float): OctreeRadiusNNSearchResult {
        val handles = JniPtcloud.octreeRadiusNNSearch(check(), query.x, query.y, query.z, radius)
        return OctreeRadiusNNSearchResult(
            count = handles[0].toInt(),
            points = jvmMat(handles[1], "Octree.radiusNNSearch.points"),
            squareDists = jvmMat(handles[2], "Octree.radiusNNSearch.squareDists"),
        )
    }

    override fun radiusNNSearch(query: Point3, radius: Float, includeColors: Boolean): OctreeRadiusNNSearchColorResult {
        val handles = JniPtcloud.octreeRadiusNNSearchColor(
            check(), query.x, query.y, query.z, radius,
        )
        return OctreeRadiusNNSearchColorResult(
            count = handles[0].toInt(),
            points = jvmMat(handles[1], "Octree.radiusNNSearch.points"),
            colors = jvmMat(handles[2], "Octree.radiusNNSearch.colors"),
            squareDists = jvmMat(handles[3], "Octree.radiusNNSearch.squareDists"),
        )
    }

    override fun KNNSearch(query: Point3, k: Int): OctreeKNNSearchResult {
        val handles = JniPtcloud.octreeKNNSearch(check(), query.x, query.y, query.z, k)
        return OctreeKNNSearchResult(
            points = jvmMat(handles[0], "Octree.KNNSearch.points"),
            squareDists = jvmMat(handles[1], "Octree.KNNSearch.squareDists"),
        )
    }

    override fun KNNSearch(query: Point3, k: Int, includeColors: Boolean): OctreeKNNSearchColorResult {
        val handles = JniPtcloud.octreeKNNSearchColor(check(), query.x, query.y, query.z, k)
        return OctreeKNNSearchColorResult(
            points = jvmMat(handles[0], "Octree.KNNSearch.points"),
            colors = jvmMat(handles[1], "Octree.KNNSearch.colors"),
            squareDists = jvmMat(handles[2], "Octree.KNNSearch.squareDists"),
        )
    }

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniPtcloud.octreeRelease(h)
        }
    }
}

actual fun octreeCreateWithDepth(maxDepth: Int, size: Double, origin: Point3, withColors: Boolean): Octree =
    JvmOctree(JniPtcloud.octreeCreateWithDepth(
        maxDepth, size, origin.x, origin.y, origin.z, withColors,
    ).checkHandle("octreeCreateWithDepth"))

actual fun octreeCreateWithDepth(maxDepth: Int, pointCloud: Mat, colors: Mat?): Octree =
    JvmOctree(JniPtcloud.octreeCreateWithDepthCloud(
        maxDepth, handleOf(pointCloud), colors?.let(::handleOf) ?: 0L,
    ).checkHandle("octreeCreateWithDepth"))

actual fun octreeCreateWithResolution(resolution: Double, size: Double, origin: Point3, withColors: Boolean): Octree =
    JvmOctree(JniPtcloud.octreeCreateWithResolution(
        resolution, size, origin.x, origin.y, origin.z, withColors,
    ).checkHandle("octreeCreateWithResolution"))

actual fun octreeCreateWithResolution(resolution: Double, pointCloud: Mat, colors: Mat?): Octree =
    JvmOctree(JniPtcloud.octreeCreateWithResolutionCloud(
        resolution, handleOf(pointCloud), colors?.let(::handleOf) ?: 0L,
    ).checkHandle("octreeCreateWithResolution"))

// =========================================================================
// Odometry
// =========================================================================

internal class JvmOdometry(@Volatile private var handle: Long) : Odometry {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Odometry is closed")

    override fun prepareFrame(frame: OdometryFrame) {
        JniPtcloud.odometryPrepareFrame(check(), (frame as JvmOdometryFrame).check())
    }

    override fun prepareFrames(srcFrame: OdometryFrame, dstFrame: OdometryFrame) {
        JniPtcloud.odometryPrepareFrames(
            check(), (srcFrame as JvmOdometryFrame).check(), (dstFrame as JvmOdometryFrame).check(),
        )
    }

    override fun compute(srcFrame: OdometryFrame, dstFrame: OdometryFrame): OdometryComputeResult {
        val out = JniPtcloud.odometryComputeFrames(
            check(), (srcFrame as JvmOdometryFrame).check(), (dstFrame as JvmOdometryFrame).check(),
        )
        return OdometryComputeResult(
            success = out[0] != 0L,
            rt = jvmMat(out[1], "Odometry.compute.rt"),
        )
    }

    override fun compute(srcDepth: Mat, dstDepth: Mat): OdometryComputeResult {
        val out = JniPtcloud.odometryComputeDepth(check(), handleOf(srcDepth), handleOf(dstDepth))
        return OdometryComputeResult(
            success = out[0] != 0L,
            rt = jvmMat(out[1], "Odometry.compute.rt"),
        )
    }

    override fun compute(srcDepth: Mat, srcRgb: Mat, dstDepth: Mat, dstRgb: Mat): OdometryComputeResult {
        val out = JniPtcloud.odometryComputeRgbd(
            check(), handleOf(srcDepth), handleOf(srcRgb), handleOf(dstDepth), handleOf(dstRgb),
        )
        return OdometryComputeResult(
            success = out[0] != 0L,
            rt = jvmMat(out[1], "Odometry.compute.rt"),
        )
    }

    override fun getNormalsComputer(): RgbdNormals? =
        JniPtcloud.odometryGetNormalsComputer(check())
            .takeIf { it != 0L }
            ?.let(::JvmRgbdNormals)

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniPtcloud.odometryRelease(h)
        }
    }
}

actual fun odometryCreate(otype: Int, settings: OdometrySettings?, algType: Int): Odometry {
    val raw = if (settings == null) {
        JniPtcloud.odometryCreateType(otype)
    } else {
        JniPtcloud.odometryCreateSettings(
            otype,
            settings.cameraMatrix?.let(::handleOf) ?: 0L,
            settings.iterCounts?.let(::handleOf) ?: 0L,
            settings.minDepth, settings.maxDepth, settings.maxDepthDiff, settings.maxPointsPart,
            settings.sobelSize, settings.sobelScale, settings.normalWinSize,
            settings.normalDiffThreshold, settings.normalMethod, settings.angleThreshold,
            settings.maxTranslation, settings.maxRotation, settings.minGradientMagnitude,
            settings.minGradientMagnitudes?.let(::handleOf) ?: 0L,
            algType,
        )
    }
    return JvmOdometry(raw.checkHandle("odometryCreate"))
}

// =========================================================================
// OdometryFrame
// =========================================================================

internal class JvmOdometryFrame(@Volatile private var handle: Long) : OdometryFrame {

    internal fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("OdometryFrame is closed")

    override fun getImage(): Mat =
        jvmMat(JniPtcloud.odometryFrameGetImage(check()), "OdometryFrame.getImage")

    override fun getGrayImage(): Mat =
        jvmMat(JniPtcloud.odometryFrameGetGrayImage(check()), "OdometryFrame.getGrayImage")

    override fun getDepth(): Mat =
        jvmMat(JniPtcloud.odometryFrameGetDepth(check()), "OdometryFrame.getDepth")

    override fun getProcessedDepth(): Mat =
        jvmMat(JniPtcloud.odometryFrameGetProcessedDepth(check()), "OdometryFrame.getProcessedDepth")

    override fun getMask(): Mat =
        jvmMat(JniPtcloud.odometryFrameGetMask(check()), "OdometryFrame.getMask")

    override fun getNormals(): Mat =
        jvmMat(JniPtcloud.odometryFrameGetNormals(check()), "OdometryFrame.getNormals")

    override val pyramidLevels: Int
        get() = JniPtcloud.odometryFrameGetPyramidLevels(check())

    override fun getPyramidAt(pyrType: Int, level: Long): Mat =
        jvmMat(JniPtcloud.odometryFrameGetPyramidAt(check(), pyrType, level), "OdometryFrame.getPyramidAt")

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniPtcloud.odometryFrameRelease(h)
        }
    }
}

actual fun odometryFrameCreate(depth: Mat?, image: Mat?, mask: Mat?, normals: Mat?): OdometryFrame =
    JvmOdometryFrame(JniPtcloud.odometryFrameCreate(
        depth?.let(::handleOf) ?: 0L,
        image?.let(::handleOf) ?: 0L,
        mask?.let(::handleOf) ?: 0L,
        normals?.let(::handleOf) ?: 0L,
    ).checkHandle("odometryFrameCreate"))

// =========================================================================
// RgbdNormals
// =========================================================================

internal class JvmRgbdNormals(@Volatile private var handle: Long) : RgbdNormals {

    internal fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("RgbdNormals is closed")

    override fun apply(points: Mat): Mat =
        jvmMat(JniPtcloud.rgbdNormalsApply(check(), handleOf(points)), "RgbdNormals.apply")

    override fun cache() {
        JniPtcloud.rgbdNormalsCache(check())
    }

    override var rows: Int
        get() = JniPtcloud.rgbdNormalsGetRows(check())
        set(value) = JniPtcloud.rgbdNormalsSetRows(check(), value)

    override var cols: Int
        get() = JniPtcloud.rgbdNormalsGetCols(check())
        set(value) = JniPtcloud.rgbdNormalsSetCols(check(), value)

    override var windowSize: Int
        get() = JniPtcloud.rgbdNormalsGetWindowSize(check())
        set(value) = JniPtcloud.rgbdNormalsSetWindowSize(check(), value)

    override val depth: Int get() = JniPtcloud.rgbdNormalsGetDepth(check())

    override var k: Mat
        get() = jvmMat(JniPtcloud.rgbdNormalsGetK(check()), "RgbdNormals.k")
        set(value) = JniPtcloud.rgbdNormalsSetK(check(), handleOf(value))

    override val method: Int get() = JniPtcloud.rgbdNormalsGetMethod(check())

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniPtcloud.rgbdNormalsRelease(h)
        }
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
): RgbdNormals = JvmRgbdNormals(JniPtcloud.rgbdNormalsCreate(
    rows, cols, depth, k?.let(::handleOf) ?: 0L, windowSize, diffThreshold, method,
).checkHandle("rgbdNormalsCreate"))

// =========================================================================
// Volume
// =========================================================================

internal class JvmVolume(@Volatile private var handle: Long) : Volume {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("Volume is closed")

    override fun integrateFrame(frame: OdometryFrame, pose: Mat) {
        JniPtcloud.volumeIntegrateFrame(check(), (frame as JvmOdometryFrame).check(), handleOf(pose))
    }

    override fun integrate(depth: Mat, pose: Mat) {
        JniPtcloud.volumeIntegrate(check(), handleOf(depth), handleOf(pose))
    }

    override fun integrateColor(depth: Mat, image: Mat, pose: Mat) {
        JniPtcloud.volumeIntegrateColor(check(), handleOf(depth), handleOf(image), handleOf(pose))
    }

    override fun raycast(cameraPose: Mat): VolumeRaycastResult {
        val handles = JniPtcloud.volumeRaycast(check(), handleOf(cameraPose))
        return VolumeRaycastResult(
            points = jvmMat(handles[0], "Volume.raycast.points"),
            normals = jvmMat(handles[1], "Volume.raycast.normals"),
        )
    }

    override fun raycastColor(cameraPose: Mat): VolumeRaycastColorResult {
        val handles = JniPtcloud.volumeRaycastColor(check(), handleOf(cameraPose))
        return VolumeRaycastColorResult(
            points = jvmMat(handles[0], "Volume.raycastColor.points"),
            normals = jvmMat(handles[1], "Volume.raycastColor.normals"),
            colors = jvmMat(handles[2], "Volume.raycastColor.colors"),
        )
    }

    override fun raycastEx(cameraPose: Mat, height: Int, width: Int, k: Mat): VolumeRaycastResult {
        val handles = JniPtcloud.volumeRaycastEx(
            check(), handleOf(cameraPose), height, width, handleOf(k),
        )
        return VolumeRaycastResult(
            points = jvmMat(handles[0], "Volume.raycastEx.points"),
            normals = jvmMat(handles[1], "Volume.raycastEx.normals"),
        )
    }

    override fun raycastExColor(cameraPose: Mat, height: Int, width: Int, k: Mat): VolumeRaycastColorResult {
        val handles = JniPtcloud.volumeRaycastExColor(
            check(), handleOf(cameraPose), height, width, handleOf(k),
        )
        return VolumeRaycastColorResult(
            points = jvmMat(handles[0], "Volume.raycastExColor.points"),
            normals = jvmMat(handles[1], "Volume.raycastExColor.normals"),
            colors = jvmMat(handles[2], "Volume.raycastExColor.colors"),
        )
    }

    override fun fetchNormals(points: Mat): Mat =
        jvmMat(JniPtcloud.volumeFetchNormals(check(), handleOf(points)), "Volume.fetchNormals")

    override fun fetchPointsNormals(): VolumePointsNormalsResult {
        val handles = JniPtcloud.volumeFetchPointsNormals(check())
        return VolumePointsNormalsResult(
            points = jvmMat(handles[0], "Volume.fetchPointsNormals.points"),
            normals = jvmMat(handles[1], "Volume.fetchPointsNormals.normals"),
        )
    }

    override fun fetchPointsNormalsColors(): VolumePointsNormalsColorsResult {
        val handles = JniPtcloud.volumeFetchPointsNormalsColors(check())
        return VolumePointsNormalsColorsResult(
            points = jvmMat(handles[0], "Volume.fetchPointsNormalsColors.points"),
            normals = jvmMat(handles[1], "Volume.fetchPointsNormalsColors.normals"),
            colors = jvmMat(handles[2], "Volume.fetchPointsNormalsColors.colors"),
        )
    }

    override fun reset() {
        JniPtcloud.volumeReset(check())
    }

    override val visibleBlocks: Int get() = JniPtcloud.volumeGetVisibleBlocks(check())

    override val totalVolumeUnits: Long get() = JniPtcloud.volumeGetTotalVolumeUnits(check())

    override fun getBoundingBox(precision: Int): Mat =
        jvmMat(JniPtcloud.volumeGetBoundingBox(check(), precision), "Volume.getBoundingBox")

    override var enableGrowth: Boolean
        get() = JniPtcloud.volumeGetEnableGrowth(check())
        set(value) = JniPtcloud.volumeSetEnableGrowth(check(), value)

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniPtcloud.volumeRelease(h)
        }
    }
}

actual fun volumeCreate(vtype: Int, settings: VolumeSettings?): Volume {
    val raw = if (settings == null) {
        JniPtcloud.volumeCreate(vtype)
    } else {
        JniPtcloud.volumeCreateSettings(
            vtype, settings.integrateWidth, settings.integrateHeight, settings.raycastWidth,
            settings.raycastHeight, settings.depthFactor, settings.voxelSize,
            settings.tsdfTruncateDistance, settings.maxDepth, settings.maxWeight,
            settings.raycastStepFactor,
            settings.volumePose?.let(::handleOf) ?: 0L,
            settings.volumeResolution?.let(::handleOf) ?: 0L,
            settings.cameraIntegrateIntrinsics?.let(::handleOf) ?: 0L,
            settings.cameraRaycastIntrinsics?.let(::handleOf) ?: 0L,
        )
    }
    return JvmVolume(raw.checkHandle("volumeCreate"))
}

// =========================================================================
// PoseGraph
// =========================================================================

internal class JvmPoseGraph(@Volatile private var handle: Long) : PoseGraph {

    private fun check(): Long =
        handle.takeIf { it != 0L } ?: throw IllegalStateException("PoseGraph is closed")

    override fun addNode(nodeId: Long, pose: Mat, fixed: Boolean) {
        JniPtcloud.poseGraphAddNode(check(), nodeId, handleOf(pose), fixed)
    }

    override fun addEdge(source: Long, target: Long, transformation: Mat, information: Mat?) {
        JniPtcloud.poseGraphAddEdge(
            check(), source, target, handleOf(transformation),
            information?.let(::handleOf) ?: 0L,
        )
    }

    override fun optimize(): Int = JniPtcloud.poseGraphOptimize(check())

    override fun getPose(nodeId: Long): Mat =
        jvmMat(JniPtcloud.poseGraphGetPose(check(), nodeId), "PoseGraph.getPose")

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            JniPtcloud.poseGraphRelease(h)
        }
    }
}

actual fun poseGraphCreate(): PoseGraph =
    JvmPoseGraph(JniPtcloud.poseGraphCreate().checkHandle("poseGraphCreate"))

// =========================================================================
// Ptcloud statics
// =========================================================================

actual fun loadPointCloud(filename: String): PointCloudData {
    val handles = JniPtcloud.ptcloudLoadPointCloud(filename)
    return PointCloudData(
        vertices = jvmMat(handles[0], "loadPointCloud.vertices"),
        normals = jvmMat(handles[1], "loadPointCloud.normals"),
        rgb = jvmMat(handles[2], "loadPointCloud.rgb"),
    )
}

actual fun savePointCloud(filename: String, vertices: Mat, normals: Mat?, rgb: Mat?) {
    JniPtcloud.ptcloudSavePointCloud(
        filename, handleOf(vertices), normals?.let(::handleOf) ?: 0L, rgb?.let(::handleOf) ?: 0L,
    )
}

actual fun loadMesh(filename: String): MeshData {
    val handles = JniPtcloud.ptcloudLoadMesh(filename)
    if (handles.size < 4 || handles[0] == 0L) throw OpenCVException("loadMesh", lastNativeError())
    return MeshData(
        vertices = jvmMat(handles[0], "loadMesh.vertices"),
        indices = (4 until handles.size).map { jvmMat(handles[it], "loadMesh.indices") },
        normals = jvmMat(handles[1], "loadMesh.normals"),
        colors = jvmMat(handles[2], "loadMesh.colors"),
        texCoords = jvmMat(handles[3], "loadMesh.texCoords"),
    )
}

actual fun saveMesh(
    filename: String,
    vertices: Mat,
    indices: List<Mat>,
    normals: Mat?,
    colors: Mat?,
    texCoords: Mat?,
) {
    JniPtcloud.ptcloudSaveMesh(
        filename,
        handleOf(vertices),
        LongArray(indices.size) { handleOf(indices[it]) },
        normals?.let(::handleOf) ?: 0L,
        colors?.let(::handleOf) ?: 0L,
        texCoords?.let(::handleOf) ?: 0L,
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
    JniPtcloud.ptcloudTriangleRasterize(
        handleOf(vertices), handleOf(indices), handleOf(colors),
        handleOf(colorBuf), handleOf(depthBuf), handleOf(world2cam), fovY, zNear, zFar,
        settings.shadingType, settings.cullingMode, settings.glCompatibleMode,
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
    JniPtcloud.ptcloudTriangleRasterizeDepth(
        handleOf(vertices), handleOf(indices), handleOf(depthBuf), handleOf(world2cam),
        fovY, zNear, zFar, settings.shadingType, settings.cullingMode, settings.glCompatibleMode,
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
    JniPtcloud.ptcloudTriangleRasterizeColor(
        handleOf(vertices), handleOf(indices), handleOf(colors), handleOf(colorBuf),
        handleOf(world2cam), fovY, zNear, zFar,
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
): Mat {
    val out = JniPtcloud.ptcloudRegisterDepth(
        handleOf(unregisteredCameraMatrix), handleOf(registeredCameraMatrix),
        handleOf(registeredDistCoeffs), handleOf(rt), handleOf(unregisteredDepth),
        outputImagePlaneSize.width, outputImagePlaneSize.height, depthDilation,
    )
    if (out[0] == 0L) throw OpenCVException("registerDepth", lastNativeError())
    return jvmMat(out[1], "registerDepth")
}

actual fun depthTo3dSparse(depth: Mat, inK: Mat, inPoints: Mat): Mat =
    jvmMat(
        JniPtcloud.ptcloudDepthTo3dSparse(handleOf(depth), handleOf(inK), handleOf(inPoints)),
        "depthTo3dSparse",
    )

actual fun depthTo3d(depth: Mat, k: Mat, mask: Mat?): Mat =
    jvmMat(
        JniPtcloud.ptcloudDepthTo3d(handleOf(depth), handleOf(k), mask?.let(::handleOf) ?: 0L),
        "depthTo3d",
    )

actual fun rescaleDepth(input: Mat, type: Int, depthFactor: Double): Mat =
    jvmMat(JniPtcloud.ptcloudRescaleDepth(handleOf(input), type, depthFactor), "rescaleDepth")

actual fun warpFrame(
    depth: Mat,
    image: Mat?,
    mask: Mat?,
    rt: Mat,
    cameraMatrix: Mat,
): WarpFrameResult {
    val handles = JniPtcloud.ptcloudWarpFrame(
        handleOf(depth), image?.let(::handleOf) ?: 0L, mask?.let(::handleOf) ?: 0L,
        handleOf(rt), handleOf(cameraMatrix),
    )
    return WarpFrameResult(
        warpedDepth = jvmMat(handles[0], "warpFrame.warpedDepth"),
        warpedImage = jvmMat(handles[1], "warpFrame.warpedImage"),
        warpedMask = jvmMat(handles[2], "warpFrame.warpedMask"),
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
): PlanesResult {
    val handles = JniPtcloud.ptcloudFindPlanes(
        handleOf(points3d), normals?.let(::handleOf) ?: 0L, blockSize, minSize, threshold,
        sensorErrorA, sensorErrorB, sensorErrorC, method,
    )
    return PlanesResult(
        mask = jvmMat(handles[0], "findPlanes.mask"),
        planeCoefficients = jvmMat(handles[1], "findPlanes.planeCoefficients"),
    )
}

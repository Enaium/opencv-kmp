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
// Ptcloud enums (org.opencv.ptcloud.Ptcloud parity)
// =========================================================================

/** Odometry algorithm implementation flavor. */
object OdometryAlgoType {
    /** Generic implementation (the default). */
    const val COMMON = 0

    /** Faster implementation. */
    const val FAST = 1
}

/** Pyramid kinds stored inside an [OdometryFrame]. */
object OdometryFramePyramidType {
    const val PYR_IMAGE = 0
    const val PYR_DEPTH = 1
    const val PYR_MASK = 2
    const val PYR_CLOUD = 3
    const val PYR_DIX = 4
    const val PYR_DIY = 5
    const val PYR_TEXMASK = 6
    const val PYR_NORM = 7
    const val PYR_NORMMASK = 8
    const val N_PYRAMIDS = 9
}

/** Data modality used by an [Odometry] instance. */
object OdometryType {
    /** Depth-only odometry (ICP). */
    const val DEPTH = 0

    /** RGB-only odometry. */
    const val RGB = 1

    /** Combined RGB + depth odometry. */
    const val RGB_DEPTH = 2
}

/** Plane detection method for [findPlanes]. */
object RgbdPlaneMethod {
    const val DEFAULT = 0
}

/** Face culling behavior for triangle rasterization. */
object TriangleCullingMode {
    const val RASTERIZE_CULLING_NONE = 0
    const val RASTERIZE_CULLING_CW = 1
    const val RASTERIZE_CULLING_CCW = 2
}

/** GL-compatibility mode for triangle rasterization. */
object TriangleGlCompatibleMode {
    const val RASTERIZE_COMPAT_DISABLED = 0
    const val RASTERIZE_COMPAT_INVDEPTH = 1
}

/** Shading used when rasterizing triangles. */
object TriangleShadingType {
    const val RASTERIZE_SHADING_WHITE = 0
    const val RASTERIZE_SHADING_FLAT = 1
    const val RASTERIZE_SHADING_SHADED = 2
}

/** Volumetric integration storage types. */
object VolumeType {
    const val TSDF = 0
    const val HASH_TSDF = 1
    const val COLOR_TSDF = 2
}

// =========================================================================
// Settings value classes (pure Kotlin; defaults mirror the C++ defaults)
// =========================================================================

/**
 * Odometry configuration (`cv::OdometrySettings`).
 *
 * Mat-valued fields default to `null`, meaning "keep the C++ default" (the
 * default camera matrix is a 640x480, fx=fy=525 pinhole; the default
 * iteration counts are {7,7,7,10}).
 */
data class OdometrySettings(
    /** 3x3 camera intrinsics; null keeps the default 525px pinhole. */
    var cameraMatrix: Mat? = null,
    /** Iteration count per pyramid level (CV_32S); null keeps {7,7,7,10}. */
    var iterCounts: Mat? = null,
    /** Minimum depth in meters. */
    var minDepth: Float = 0.0f,
    /** Maximum depth in meters. */
    var maxDepth: Float = 4.0f,
    /** Maximum allowed depth difference between corresponding pixels. */
    var maxDepthDiff: Float = 0.07f,
    /** Maximum fraction of points used for the ICP calculation. */
    var maxPointsPart: Float = 0.07f,
    /** Sobel filter window size (3 or 5). */
    var sobelSize: Int = 3,
    /** Sobel filter scale. */
    var sobelScale: Double = 1.0 / 8.0,
    /** Normal estimation window size (1, 3, 5 or 7). */
    var normalWinSize: Int = 5,
    /** Depth difference threshold used by the LINEMOD normal method. */
    var normalDiffThreshold: Float = 50.0f,
    /** Normal estimation method ([RgbdNormals.RGBD_NORMALS_METHOD_FALS]..). */
    var normalMethod: Int = RgbdNormals.RGBD_NORMALS_METHOD_FALS,
    /** Maximum allowed angle between corresponding surface normals. */
    var angleThreshold: Float = (30.0 * kotlin.math.PI / 180.0).toFloat(),
    /** Maximum allowed translation between frames (meters). */
    var maxTranslation: Float = 0.15f,
    /** Maximum allowed rotation between frames (degrees). */
    var maxRotation: Float = 15.0f,
    /** Minimum gradient magnitude of pixels used for RGB odometry. */
    var minGradientMagnitude: Float = 10.0f,
    /** Per-level minimum gradient magnitudes (CV_32F); null keeps the default. */
    var minGradientMagnitudes: Mat? = null,
)

/**
 * Volume configuration (`cv::VolumeSettings`).
 *
 * The scalar defaults match the C++ TSDF volume defaults (640x480 images,
 * 525px intrinsics, 128^3 voxels over a 3 m cube). When creating a
 * [VolumeType.HASH_TSDF] or [VolumeType.COLOR_TSDF] volume the defaults for
 * the fields below should be adjusted: HashTSDF uses voxelSize 3/512,
 * tsdfTruncateDistance 7*voxelSize, maxDepth 4 m, raycastStepFactor 0.25 and
 * a 16^3 resolution.
 */
data class VolumeSettings(
    /** Width of the integration image. */
    var integrateWidth: Int = 640,
    /** Height of the integration image. */
    var integrateHeight: Int = 480,
    /** Width of the raycasted image (used when not passed to raycast). */
    var raycastWidth: Int = 640,
    /** Height of the raycasted image. */
    var raycastHeight: Int = 480,
    /** Depth scaling factor (5000 for 16-bit PNG depth, 1 for float depth). */
    var depthFactor: Float = 5000.0f,
    /** Size of one voxel in meters (3/128 for TSDF). */
    var voxelSize: Float = 3.0f / 128.0f,
    /** TSDF truncation distance in meters (2 * voxelSize for TSDF). */
    var tsdfTruncateDistance: Float = 2.0f * (3.0f / 128.0f),
    /** Depth truncation threshold in meters; larger depth is dropped. */
    var maxDepth: Float = 0.0f,
    /** Max number of frames integrated per voxel. */
    var maxWeight: Int = 64,
    /** Length of a single raycast step, as a fraction of a voxel. */
    var raycastStepFactor: Float = 0.75f,
    /** 4x4 volume pose (null keeps the default centered pose). */
    var volumePose: Mat? = null,
    /** 1x3/3x1 voxel resolution (null keeps 128^3 for TSDF). */
    var volumeResolution: Mat? = null,
    /** 3x3 intrinsics of the integration camera (null keeps 525px pinhole). */
    var cameraIntegrateIntrinsics: Mat? = null,
    /** 3x3 intrinsics of the raycast camera (null keeps 525px pinhole). */
    var cameraRaycastIntrinsics: Mat? = null,
)

/**
 * Rasterization settings (`cv::TriangleRasterizeSettings`); defaults are
 * smooth shading with CW culling and GL compatibility disabled.
 */
data class TriangleRasterizeSettings(
    /** Shading mode ([TriangleShadingType]). */
    var shadingType: Int = TriangleShadingType.RASTERIZE_SHADING_SHADED,
    /** Face culling mode ([TriangleCullingMode]). */
    var cullingMode: Int = TriangleCullingMode.RASTERIZE_CULLING_CW,
    /** GL compatibility mode ([TriangleGlCompatibleMode]). */
    var glCompatibleMode: Int = TriangleGlCompatibleMode.RASTERIZE_COMPAT_DISABLED,
)

// =========================================================================
// Octree
// =========================================================================

/** Result of [Octree.radiusNNSearch]; Mats must be closed by the caller. */
data class OctreeRadiusNNSearchResult(
    /** Number of points found within the radius. */
    val count: Int,
    /** Found points, 3-float format, unordered. */
    val points: Mat,
    /** Squared distances to the query point, unordered. */
    val squareDists: Mat,
)

/** Result of [Octree.radiusNNSearch] with colors; Mats must be closed. */
data class OctreeRadiusNNSearchColorResult(
    /** Number of points found within the radius. */
    val count: Int,
    /** Found points, 3-float format, unordered. */
    val points: Mat,
    /** Colors corresponding to the found points, unordered. */
    val colors: Mat,
    /** Squared distances to the query point, unordered. */
    val squareDists: Mat,
)

/** Result of [Octree.KNNSearch]; Mats must be closed by the caller. */
data class OctreeKNNSearchResult(
    /** K points in 3-float format, ordered near to far. */
    val points: Mat,
    /** K squared distances, ordered near to far. */
    val squareDists: Mat,
)

/** Result of [Octree.KNNSearch] with colors; Mats must be closed. */
data class OctreeKNNSearchColorResult(
    /** K points in 3-float format, ordered near to far. */
    val points: Mat,
    /** Colors corresponding to the found points, ordered near to far. */
    val colors: Mat,
    /** K squared distances, ordered near to far. */
    val squareDists: Mat,
)

/**
 * Octree data structure for point cloud processing (`cv::Octree`).
 *
 * Each Octree has a fixed depth (or leaf resolution); points inside the same
 * leaf collapse to the leaf center when restored via
 * [getPointCloudByOctree].
 */
interface Octree : AutoCloseable {

    /**
     * Inserts a point (optionally with its color) into the Octree; returns
     * whether the insertion succeeded.
     */
    fun insertPoint(point: Point3, color: Point3? = null): Boolean

    /** Whether [point] lies inside this Octree's bounding box. */
    fun isPointInBound(point: Point3): Boolean

    /** Whether the Octree contains no points. */
    val empty: Boolean

    /** Removes every point and resets the Octree parameters. */
    fun clear()

    /** Deletes a point (epsilon-based comparison); true when deleted. */
    fun deletePoint(point: Point3): Boolean

    /** Restores the point cloud (leaf centers); returns a new Mat. */
    fun getPointCloudByOctree(): Mat

    /** Restores the point cloud and its colors; Mats must be closed. */
    fun getPointCloudByOctreeWithColors(): Pair<Mat, Mat>

    /** Radius nearest neighbor search around [query]. */
    fun radiusNNSearch(query: Point3, radius: Float): OctreeRadiusNNSearchResult

    /** Radius nearest neighbor search also returning point colors. */
    fun radiusNNSearch(query: Point3, radius: Float, includeColors: Boolean): OctreeRadiusNNSearchColorResult

    /** K nearest neighbor search around [query]. */
    fun KNNSearch(query: Point3, k: Int): OctreeKNNSearchResult

    /** K nearest neighbor search also returning point colors. */
    fun KNNSearch(query: Point3, k: Int, includeColors: Boolean): OctreeKNNSearchColorResult

    override fun close()
}

/** Creates an empty Octree with a given maximum depth (`cv::Octree::createWithDepth`). */
expect fun octreeCreateWithDepth(
    maxDepth: Int,
    size: Double,
    origin: Point3 = Point3(0.0, 0.0, 0.0),
    withColors: Boolean = false,
): Octree

/** Creates an Octree from point cloud data with a given maximum depth. */
expect fun octreeCreateWithDepth(maxDepth: Int, pointCloud: Mat, colors: Mat? = null): Octree

/** Creates an empty Octree whose leaf nodes have the given size. */
expect fun octreeCreateWithResolution(
    resolution: Double,
    size: Double,
    origin: Point3 = Point3(0.0, 0.0, 0.0),
    withColors: Boolean = false,
): Octree

/** Creates an Octree from point cloud data with the given leaf resolution. */
expect fun octreeCreateWithResolution(resolution: Double, pointCloud: Mat, colors: Mat? = null): Octree

// =========================================================================
// Odometry
// =========================================================================

/**
 * Odometry over RGB-D frames (`cv::Odometry`).
 *
 * Frames passed to [compute] should first be prepared with [prepareFrame] /
 * [prepareFrames] so per-frame pyramids are reused between calls.
 */
interface Odometry : AutoCloseable {

    /** Prepares [frame] as both the source and the destination frame. */
    fun prepareFrame(frame: OdometryFrame)

    /** Prepares [srcFrame] and [dstFrame] for odometry calculation. */
    fun prepareFrames(srcFrame: OdometryFrame, dstFrame: OdometryFrame)

    /** Computes Rt with Rt * src = dst over prepared frames. */
    fun compute(srcFrame: OdometryFrame, dstFrame: OdometryFrame): OdometryComputeResult

    /** Computes Rt from two depth images. */
    fun compute(srcDepth: Mat, dstDepth: Mat): OdometryComputeResult

    /** Computes Rt from two RGB-D pairs. */
    fun compute(srcDepth: Mat, srcRgb: Mat, dstDepth: Mat, dstRgb: Mat): OdometryComputeResult

    /**
     * The normals computer used by the ICP algorithm (created on demand);
     * null when the current settings require no normals.
     */
    fun getNormalsComputer(): RgbdNormals?

    override fun close()
}

/** Result of [Odometry.compute]; [rt] must be closed by the caller. */
data class OdometryComputeResult(
    /** Whether a transformation was found. */
    val success: Boolean,
    /** 4x4 rigid transformation such that rt * src = dst. */
    val rt: Mat,
)

/** Creates an [Odometry] instance (`cv::Odometry` constructors). */
expect fun odometryCreate(
    otype: Int = OdometryType.DEPTH,
    settings: OdometrySettings? = null,
    algType: Int = OdometryAlgoType.COMMON,
): Odometry

/**
 * Per-frame data holder for odometry algorithms (`cv::OdometryFrame`).
 *
 * When non-empty it contains a depth image, a mask of valid pixels and the
 * pyramids generated from that data; a BGR/gray image and normals are
 * optional.
 */
interface OdometryFrame : AutoCloseable {

    /** Original user-provided BGR/gray image; returns a new Mat. */
    fun getImage(): Mat

    /** Gray image derived from the BGR/gray image; returns a new Mat. */
    fun getGrayImage(): Mat

    /** Original user-provided depth image; returns a new Mat. */
    fun getDepth(): Mat

    /** Depth image rescaled/filtered for the ICP algorithm; new Mat. */
    fun getProcessedDepth(): Mat

    /** Valid-pixels mask used for ICP; returns a new Mat. */
    fun getMask(): Mat

    /** Normals image (generated or user-provided); returns a new Mat. */
    fun getNormals(): Mat

    /** Number of pyramid levels, or 0 if no pyramids were prepared. */
    val pyramidLevels: Int

    /** Image from one of the pyramids; returns a new Mat (empty if absent). */
    fun getPyramidAt(pyrType: Int, level: Long): Mat

    override fun close()
}

/** Creates an [OdometryFrame] from depth/image/mask/normals Mats. */
expect fun odometryFrameCreate(
    depth: Mat? = null,
    image: Mat? = null,
    mask: Mat? = null,
    normals: Mat? = null,
): OdometryFrame

// =========================================================================
// RgbdNormals
// =========================================================================

/**
 * Computes surface normals in a range image (`cv::RgbdNormals`); caches
 * data internally for speed.
 */
interface RgbdNormals : AutoCloseable {

    companion object {
        const val RGBD_NORMALS_METHOD_FALS = 0
        const val RGBD_NORMALS_METHOD_LINEMOD = 1
        const val RGBD_NORMALS_METHOD_SRI = 2
        const val RGBD_NORMALS_METHOD_CROSS_PRODUCT = 3
    }

    /**
     * Computes the normals at each point of a rows x cols depth/points
     * image; returns a new rows x cols x 3 normals Mat.
     */
    fun apply(points: Mat): Mat

    /** Prepares the cached data (called automatically at first [apply]). */
    fun cache()

    /** Number of rows of the depth image. */
    var rows: Int

    /** Number of cols of the depth image. */
    var cols: Int

    /** Window size used to compute the normals (1, 3, 5 or 7). */
    var windowSize: Int

    /** Depth of the normals (only CV_32F or CV_64F). */
    val depth: Int

    /** Calibration matrix. */
    var k: Mat

    /** Normals computation method ([RGBD_NORMALS_METHOD_FALS]..). */
    val method: Int

    override fun close()
}

/** Creates an [RgbdNormals] computer (`cv::RgbdNormals::create`). */
expect fun rgbdNormalsCreate(
    rows: Int = 0,
    cols: Int = 0,
    depth: Int = 0,
    k: Mat? = null,
    windowSize: Int = 5,
    diffThreshold: Float = 50.0f,
    method: Int = RgbdNormals.RGBD_NORMALS_METHOD_FALS,
): RgbdNormals

// =========================================================================
// Volume
// =========================================================================

/** Result of [Volume.raycast]; Mats must be closed by the caller. */
data class VolumeRaycastResult(
    /** Rendered points in camera coordinates. */
    val points: Mat,
    /** Normals corresponding to [points]. */
    val normals: Mat,
)

/** Result of [Volume.raycastColor]; Mats must be closed by the caller. */
data class VolumeRaycastColorResult(
    /** Rendered points in camera coordinates. */
    val points: Mat,
    /** Normals corresponding to [points]. */
    val normals: Mat,
    /** Colors corresponding to [points] (ColorTSDF only). */
    val colors: Mat,
)

/** Result of [Volume.fetchPointsNormals]; Mats must be closed. */
data class VolumePointsNormalsResult(
    /** All points stored in the volume. */
    val points: Mat,
    /** Normals corresponding to [points]. */
    val normals: Mat,
)

/** Result of [Volume.fetchPointsNormalsColors]; Mats must be closed. */
data class VolumePointsNormalsColorsResult(
    /** All points stored in the volume. */
    val points: Mat,
    /** Normals corresponding to [points]. */
    val normals: Mat,
    /** Colors corresponding to [points] (ColorTSDF only). */
    val colors: Mat,
)

/**
 * Volumetric TSDF/HashTSDF/ColorTSDF reconstruction (`cv::Volume`).
 *
 * Camera intrinsics are taken from the [VolumeSettings] used at creation;
 * depth values are divided by `depthFactor` during integration.
 */
interface Volume : AutoCloseable {

    companion object {
        /** Bounding box precision: up to volume unit. */
        const val VOLUME_UNIT = 0

        /** Bounding box precision: up to voxel (currently unsupported). */
        const val VOXEL = 1
    }

    /** Integrates an [OdometryFrame] into the volume. */
    fun integrateFrame(frame: OdometryFrame, pose: Mat)

    /** Integrates a depth image into the volume. */
    fun integrate(depth: Mat, pose: Mat)

    /** Integrates depth + color into a ColorTSDF volume. */
    fun integrateColor(depth: Mat, image: Mat, pose: Mat)

    /** Renders the volume into a points/normals image pair. */
    fun raycast(cameraPose: Mat): VolumeRaycastResult

    /** Renders the volume into a points/normals/colors image triple. */
    fun raycastColor(cameraPose: Mat): VolumeRaycastColorResult

    /** Renders with an explicit image size and intrinsics. */
    fun raycastEx(cameraPose: Mat, height: Int, width: Int, k: Mat): VolumeRaycastResult

    /** Renders with explicit size/intrinsics, including colors. */
    fun raycastExColor(cameraPose: Mat, height: Int, width: Int, k: Mat): VolumeRaycastColorResult

    /** Extracts the normals for existing [points] from the volume. */
    fun fetchNormals(points: Mat): Mat

    /** Extracts all points and normals from the volume. */
    fun fetchPointsNormals(): VolumePointsNormalsResult

    /** Extracts all points, normals and colors from the volume. */
    fun fetchPointsNormalsColors(): VolumePointsNormalsColorsResult

    /** Clears all data in the volume. */
    fun reset()

    /** Number of visible blocks in the volume. */
    val visibleBlocks: Int

    /** Number of volume units in the volume. */
    val totalVolumeUnits: Long

    /** Bounding box as a 6-float Mat (min_x, min_y, min_z, max_x, max_y, max_z). */
    fun getBoundingBox(precision: Int = VOLUME_UNIT): Mat

    /** Whether new volume units are allocated during integration (HashTSDF). */
    var enableGrowth: Boolean

    override fun close()
}

/** Creates a [Volume] (`cv::Volume` constructors). */
expect fun volumeCreate(
    vtype: Int = VolumeType.TSDF,
    settings: VolumeSettings? = null,
): Volume

// =========================================================================
// PoseGraph
// =========================================================================

/**
 * Pose graph optimizer used in SLAM back-ends (`cv::detail::PoseGraph`);
 * solves the Ceres-style 3D pose graph optimization problem.
 */
interface PoseGraph : AutoCloseable {

    /** Adds a node (any id >= 0) with its 4x4 pose; [fixed] nodes stay fixed. */
    fun addNode(nodeId: Long, pose: Mat, fixed: Boolean = false)

    /** Adds an edge between two nodes with a 4x4 transformation. */
    fun addEdge(source: Long, target: Long, transformation: Mat, information: Mat? = null)

    /**
     * Runs the optimization; returns the number of iterations elapsed, or
     * -1 when the graph is invalid or the optimization failed.
     */
    fun optimize(): Int

    /** Returns the optimized 4x4 pose of [nodeId] as a new Mat. */
    fun getPose(nodeId: Long): Mat

    override fun close()
}

/** Creates an empty [PoseGraph]. */
expect fun poseGraphCreate(): PoseGraph

// =========================================================================
// Ptcloud free functions (org.opencv.ptcloud.Ptcloud parity)
// =========================================================================

/** Point cloud loaded by [loadPointCloud]; Mats must be closed by the caller. */
data class PointCloudData(
    /** Vertex coordinates, 3 floats each. */
    val vertices: Mat,
    /** Per-vertex normals, 3 floats each. */
    val normals: Mat,
    /** Per-vertex colors, 3 floats each. */
    val rgb: Mat,
)

/** Mesh loaded by [loadMesh]; every Mat must be closed by the caller. */
data class MeshData(
    /** Vertex coordinates, 3 floats each. */
    val vertices: Mat,
    /** Per-face vertex index lists (one Mat of ints per face). */
    val indices: List<Mat>,
    /** Per-vertex normals, 3 floats each. */
    val normals: Mat,
    /** Per-vertex colors, 3 floats each. */
    val colors: Mat,
    /** Per-vertex texture coordinates, 2 or 3 floats each. */
    val texCoords: Mat,
)

/** Result of [warpFrame]; Mats must be closed by the caller. */
data class WarpFrameResult(
    /** The warped depth data. */
    val warpedDepth: Mat,
    /** The warped RGB image. */
    val warpedImage: Mat,
    /** The mask of valid pixels in the warped image. */
    val warpedMask: Mat,
)

/** Result of [findPlanes]; Mats must be closed by the caller. */
data class PlanesResult(
    /** Per-pixel plane label; 255 = no plane. */
    val mask: Mat,
    /** Plane coefficients (a,b,c,d) per plane, ax+by+cz+d=0, |(a,b,c)|=1. */
    val planeCoefficients: Mat,
)

/**
 * Loads a point cloud from an OBJ/PLY file (`cv::loadPointCloud`).
 * Vertex coordinates, normals and colors are returned as stored in the file.
 */
expect fun loadPointCloud(filename: String): PointCloudData

/** Saves a point cloud to an OBJ/PLY file chosen by the extension. */
expect fun savePointCloud(
    filename: String,
    vertices: Mat,
    normals: Mat? = null,
    rgb: Mat? = null,
)

/** Loads a triangulated mesh from an OBJ/PLY file (`cv::loadMesh`). */
expect fun loadMesh(filename: String): MeshData

/** Saves a mesh to an OBJ/PLY file chosen by the extension. */
expect fun saveMesh(
    filename: String,
    vertices: Mat,
    indices: List<Mat>,
    normals: Mat? = null,
    colors: Mat? = null,
    texCoords: Mat? = null,
)

/**
 * Renders a set of triangles into a depth and color image
 * (`cv::triangleRasterize`). Both [colorBuf] and [depthBuf] are reused, not
 * cleared, before rendering.
 */
expect fun triangleRasterize(
    vertices: Mat,
    indices: Mat,
    colors: Mat,
    colorBuf: Mat,
    depthBuf: Mat,
    world2cam: Mat,
    fovY: Double,
    zNear: Double,
    zFar: Double,
    settings: TriangleRasterizeSettings = TriangleRasterizeSettings(),
)

/** Depth-only triangle rendering (`cv::triangleRasterizeDepth`). */
expect fun triangleRasterizeDepth(
    vertices: Mat,
    indices: Mat,
    depthBuf: Mat,
    world2cam: Mat,
    fovY: Double,
    zNear: Double,
    zFar: Double,
    settings: TriangleRasterizeSettings = TriangleRasterizeSettings(),
)

/** Color-only triangle rendering (`cv::triangleRasterizeColor`). */
expect fun triangleRasterizeColor(
    vertices: Mat,
    indices: Mat,
    colors: Mat,
    colorBuf: Mat,
    world2cam: Mat,
    fovY: Double,
    zNear: Double,
    zFar: Double,
    settings: TriangleRasterizeSettings = TriangleRasterizeSettings(),
)

/**
 * Registers depth data to an external camera (`cv::registerDepth`);
 * returns a new registered depth Mat.
 */
expect fun registerDepth(
    unregisteredCameraMatrix: Mat,
    registeredCameraMatrix: Mat,
    registeredDistCoeffs: Mat,
    rt: Mat,
    unregisteredDepth: Mat,
    outputImagePlaneSize: Size,
    depthDilation: Boolean = false,
): Mat

/** Converts sparse depth points to 3D (`cv::depthTo3dSparse`); new Mat. */
expect fun depthTo3dSparse(depth: Mat, inK: Mat, inPoints: Mat): Mat

/** Converts a depth image to 3D points (`cv::depthTo3d`); new Mat. */
expect fun depthTo3d(depth: Mat, k: Mat, mask: Mat? = null): Mat

/**
 * Rescales a depth image to meters (`cv::rescaleDepth`); CV_16U input is
 * divided by [depthFactor] and zeros become NaN. Returns a new Mat.
 */
expect fun rescaleDepth(input: Mat, type: Int, depthFactor: Double = 1000.0): Mat

/**
 * Warps depth/RGB-D data by reprojection through [rt]
 * (`cv::warpFrame`); useful to visualize odometry results.
 */
expect fun warpFrame(
    depth: Mat,
    image: Mat? = null,
    mask: Mat? = null,
    rt: Mat,
    cameraMatrix: Mat,
): WarpFrameResult

/**
 * Finds planes in a depth image (`cv::findPlanes`); the returned mask labels
 * each pixel with its plane id (255 = no plane) and [PlanesResult.planeCoefficients]
 * holds one (a,b,c,d) row per plane.
 */
expect fun findPlanes(
    points3d: Mat,
    normals: Mat? = null,
    blockSize: Int = 40,
    minSize: Int = 40 * 40,
    threshold: Double = 0.01,
    sensorErrorA: Double = 0.0,
    sensorErrorB: Double = 0.0,
    sensorErrorC: Double = 0.0,
    method: Int = RgbdPlaneMethod.DEFAULT,
): PlanesResult

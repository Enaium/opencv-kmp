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
 * JNI bridge for the ptcloud slice.
 *
 * Every `external fun` maps 1:1 to a `Java_cn_enaium_opencv_JniPtcloud_<name>`
 * function in jni/jni_ptcloud.cpp. All members are public (no `internal`
 * modifier) so their JVM names are not mangled by the Kotlin compiler.
 *
 * Handles travel as jlong pointers; multi-output calls return a jlongArray
 * whose layout is documented per function (counts/ok flags are encoded as
 * 0/1 longs).
 */
internal object JniPtcloud {

    init {
        // Trigger the canonical native-library load via Jni.
        Jni.lastError()
    }

    // Octree

    external fun octreeCreateWithDepth(maxDepth: Int, size: Double, ox: Double, oy: Double, oz: Double, withColors: Boolean): Long
    external fun octreeCreateWithDepthCloud(maxDepth: Int, pointCloud: Long, colors: Long): Long
    external fun octreeCreateWithResolution(resolution: Double, size: Double, ox: Double, oy: Double, oz: Double, withColors: Boolean): Long
    external fun octreeCreateWithResolutionCloud(resolution: Double, pointCloud: Long, colors: Long): Long
    external fun octreeInsertPoint(h: Long, x: Double, y: Double, z: Double): Boolean
    external fun octreeInsertPointColor(h: Long, x: Double, y: Double, z: Double, cx: Double, cy: Double, cz: Double): Boolean
    external fun octreeIsPointInBound(h: Long, x: Double, y: Double, z: Double): Boolean
    external fun octreeEmpty(h: Long): Boolean
    external fun octreeClear(h: Long)
    external fun octreeDeletePoint(h: Long, x: Double, y: Double, z: Double): Boolean
    external fun octreeGetPointCloud(h: Long): Long

    /** 2 handles [points, colors]. */
    external fun octreeGetPointCloudColor(h: Long): LongArray

    /** 3 longs [count, points, dists]. */
    external fun octreeRadiusNNSearch(h: Long, qx: Double, qy: Double, qz: Double, radius: Float): LongArray

    /** 4 longs [count, points, colors, dists]. */
    external fun octreeRadiusNNSearchColor(h: Long, qx: Double, qy: Double, qz: Double, radius: Float): LongArray

    /** 2 handles [points, dists]. */
    external fun octreeKNNSearch(h: Long, qx: Double, qy: Double, qz: Double, k: Int): LongArray

    /** 3 handles [points, colors, dists]. */
    external fun octreeKNNSearchColor(h: Long, qx: Double, qy: Double, qz: Double, k: Int): LongArray

    external fun octreeRelease(h: Long)

    // Odometry

    external fun odometryCreate(): Long
    external fun odometryCreateType(otype: Int): Long
    external fun odometryCreateSettings(
        otype: Int,
        cameraMatrix: Long,
        iterCounts: Long,
        minDepth: Float,
        maxDepth: Float,
        maxDepthDiff: Float,
        maxPointsPart: Float,
        sobelSize: Int,
        sobelScale: Double,
        normalWinSize: Int,
        normalDiffThreshold: Float,
        normalMethod: Int,
        angleThreshold: Float,
        maxTranslation: Float,
        maxRotation: Float,
        minGradientMagnitude: Float,
        minGradientMagnitudes: Long,
        algtype: Int,
    ): Long

    external fun odometryPrepareFrame(h: Long, frame: Long)
    external fun odometryPrepareFrames(h: Long, src: Long, dst: Long)

    /** 2 longs [ok, rt]. */
    external fun odometryComputeFrames(h: Long, src: Long, dst: Long): LongArray

    /** 2 longs [ok, rt]. */
    external fun odometryComputeDepth(h: Long, srcDepth: Long, dstDepth: Long): LongArray

    /** 2 longs [ok, rt]. */
    external fun odometryComputeRgbd(h: Long, srcDepth: Long, srcRgb: Long, dstDepth: Long, dstRgb: Long): LongArray

    external fun odometryGetNormalsComputer(h: Long): Long
    external fun odometryRelease(h: Long)

    // OdometryFrame

    external fun odometryFrameCreate(depth: Long, image: Long, mask: Long, normals: Long): Long
    external fun odometryFrameGetImage(h: Long): Long
    external fun odometryFrameGetGrayImage(h: Long): Long
    external fun odometryFrameGetDepth(h: Long): Long
    external fun odometryFrameGetProcessedDepth(h: Long): Long
    external fun odometryFrameGetMask(h: Long): Long
    external fun odometryFrameGetNormals(h: Long): Long
    external fun odometryFrameGetPyramidLevels(h: Long): Int
    external fun odometryFrameGetPyramidAt(h: Long, pyrType: Int, level: Long): Long
    external fun odometryFrameRelease(h: Long)

    // RgbdNormals

    external fun rgbdNormalsCreate(rows: Int, cols: Int, depth: Int, k: Long, windowSize: Int, diffThreshold: Float, method: Int): Long
    external fun rgbdNormalsApply(h: Long, points: Long): Long
    external fun rgbdNormalsCache(h: Long)
    external fun rgbdNormalsGetRows(h: Long): Int
    external fun rgbdNormalsSetRows(h: Long, v: Int)
    external fun rgbdNormalsGetCols(h: Long): Int
    external fun rgbdNormalsSetCols(h: Long, v: Int)
    external fun rgbdNormalsGetWindowSize(h: Long): Int
    external fun rgbdNormalsSetWindowSize(h: Long, v: Int)
    external fun rgbdNormalsGetDepth(h: Long): Int
    external fun rgbdNormalsGetK(h: Long): Long
    external fun rgbdNormalsSetK(h: Long, k: Long)
    external fun rgbdNormalsGetMethod(h: Long): Int
    external fun rgbdNormalsRelease(h: Long)

    // Volume

    external fun volumeCreate(vtype: Int): Long
    external fun volumeCreateSettings(
        vtype: Int,
        integrateWidth: Int,
        integrateHeight: Int,
        raycastWidth: Int,
        raycastHeight: Int,
        depthFactor: Float,
        voxelSize: Float,
        tsdfTruncateDistance: Float,
        maxDepth: Float,
        maxWeight: Int,
        raycastStepFactor: Float,
        volumePose: Long,
        volumeResolution: Long,
        cameraIntegrateIntrinsics: Long,
        cameraRaycastIntrinsics: Long,
    ): Long

    external fun volumeIntegrateFrame(h: Long, frame: Long, pose: Long)
    external fun volumeIntegrate(h: Long, depth: Long, pose: Long)
    external fun volumeIntegrateColor(h: Long, depth: Long, image: Long, pose: Long)

    /** 2 handles [points, normals]. */
    external fun volumeRaycast(h: Long, cameraPose: Long): LongArray

    /** 3 handles [points, normals, colors]. */
    external fun volumeRaycastColor(h: Long, cameraPose: Long): LongArray

    /** 2 handles [points, normals]. */
    external fun volumeRaycastEx(h: Long, cameraPose: Long, height: Int, width: Int, k: Long): LongArray

    /** 3 handles [points, normals, colors]. */
    external fun volumeRaycastExColor(h: Long, cameraPose: Long, height: Int, width: Int, k: Long): LongArray

    external fun volumeFetchNormals(h: Long, points: Long): Long

    /** 2 handles [points, normals]. */
    external fun volumeFetchPointsNormals(h: Long): LongArray

    /** 3 handles [points, normals, colors]. */
    external fun volumeFetchPointsNormalsColors(h: Long): LongArray

    external fun volumeReset(h: Long)
    external fun volumeGetVisibleBlocks(h: Long): Int
    external fun volumeGetTotalVolumeUnits(h: Long): Long
    external fun volumeGetBoundingBox(h: Long, precision: Int): Long
    external fun volumeSetEnableGrowth(h: Long, v: Boolean)
    external fun volumeGetEnableGrowth(h: Long): Boolean
    external fun volumeRelease(h: Long)

    // PoseGraph

    external fun poseGraphCreate(): Long
    external fun poseGraphAddNode(h: Long, nodeId: Long, pose: Long, fixed: Boolean)
    external fun poseGraphAddEdge(h: Long, source: Long, target: Long, transformation: Long, information: Long)
    external fun poseGraphOptimize(h: Long): Int
    external fun poseGraphGetPose(h: Long, nodeId: Long): Long
    external fun poseGraphRelease(h: Long)

    // Ptcloud free functions

    /** 3 handles [vertices, normals, rgb]. */
    external fun ptcloudLoadPointCloud(filename: String): LongArray

    external fun ptcloudSavePointCloud(filename: String, vertices: Long, normals: Long, rgb: Long)

    /** 4 head handles [vertices, normals, colors, texCoords] + one handle per face. */
    external fun ptcloudLoadMesh(filename: String): LongArray

    external fun ptcloudSaveMesh(filename: String, vertices: Long, indices: LongArray, normals: Long, colors: Long, texCoords: Long)
    external fun ptcloudTriangleRasterize(
        vertices: Long,
        indices: Long,
        colors: Long,
        colorBuf: Long,
        depthBuf: Long,
        world2cam: Long,
        fovY: Double,
        zNear: Double,
        zFar: Double,
        shadingType: Int,
        cullingMode: Int,
        glCompatibleMode: Int,
    )

    external fun ptcloudTriangleRasterizeDepth(
        vertices: Long,
        indices: Long,
        depthBuf: Long,
        world2cam: Long,
        fovY: Double,
        zNear: Double,
        zFar: Double,
        shadingType: Int,
        cullingMode: Int,
        glCompatibleMode: Int,
    )

    external fun ptcloudTriangleRasterizeColor(
        vertices: Long,
        indices: Long,
        colors: Long,
        colorBuf: Long,
        world2cam: Long,
        fovY: Double,
        zNear: Double,
        zFar: Double,
        shadingType: Int,
        cullingMode: Int,
        glCompatibleMode: Int,
    )

    /** 2 longs [ok, registered]. */
    external fun ptcloudRegisterDepth(
        unregisteredCameraMatrix: Long,
        registeredCameraMatrix: Long,
        registeredDistCoeffs: Long,
        rt: Long,
        unregisteredDepth: Long,
        outputWidth: Int,
        outputHeight: Int,
        depthDilation: Boolean,
    ): LongArray

    external fun ptcloudDepthTo3dSparse(depth: Long, inK: Long, inPoints: Long): Long
    external fun ptcloudDepthTo3d(depth: Long, k: Long, mask: Long): Long
    external fun ptcloudRescaleDepth(input: Long, type: Int, depthFactor: Double): Long

    /** 3 handles [warpedDepth, warpedImage, warpedMask]. */
    external fun ptcloudWarpFrame(depth: Long, image: Long, mask: Long, rt: Long, cameraMatrix: Long): LongArray

    /** 2 handles [mask, coefficients]. */
    external fun ptcloudFindPlanes(
        points3d: Long,
        normals: Long,
        blockSize: Int,
        minSize: Int,
        threshold: Double,
        sensorErrorA: Double,
        sensorErrorB: Double,
        sensorErrorC: Double,
        method: Int,
    ): LongArray
}

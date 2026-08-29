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
package cn.enaium.opencv.tutorial

import cn.enaium.opencv.Mat
import cn.enaium.opencv.MatType
import cn.enaium.opencv.Size
import cn.enaium.opencv.calibrateCamera
import cn.enaium.opencv.findChessboardCorners
import cn.enaium.opencv.mat
import cn.enaium.opencv.zeros
import cn.enaium.opencv.vectorPoint2fToMat
import cn.enaium.opencv.vectorPoint3fToMat

/**
 * Ports of `tutorial_code/calib3d/camera_calibration/`:
 * draw a synthetic 9x6 chessboard, locate it with [findChessboardCorners],
 * then run [calibrateCamera] over several synthetic views and report the
 * recovered intrinsic matrix.
 */
fun runCalib3dTutorials(): String = buildString {
    appendLine("-- chessboard detection --")
    chessboardDemo().also { append(it) }

    appendLine("-- camera calibration --")
    calibrationDemo().also { append(it) }
}

private const val BOARD_COLS = 9
private const val BOARD_ROWS = 6
private const val SQUARE = 21 // pixels per square

/** Draws a checkerboard with [cols]x[rows] internal corners. */
private fun drawBoard(cols: Int, rows: Int): Mat {
    val m = zeros(rows * SQUARE, cols * SQUARE, MatType.CV_8UC1)
    for (r in 0 until rows) {
        for (c in 0 until cols) {
            if ((r + c) % 2 == 0) continue
            val y0 = r * SQUARE
            val x0 = c * SQUARE
            for (y in y0 until y0 + SQUARE) {
                for (x in x0 until x0 + SQUARE) m[y, x] = 255.0
            }
        }
    }
    return m
}

private fun chessboardDemo(): String = buildString {
    drawBoard(BOARD_COLS, BOARD_ROWS).use { board ->
        line("board size", "${board.rows}x${board.cols}")
        val patternSize = Size(BOARD_COLS - 1, BOARD_ROWS - 1)
        val res = findChessboardCorners(board, patternSize)
        line("pattern found", "${res.found}")
        line("corner rows", "${res.corners.rows}")
    }
}

private fun calibrationDemo(): String = buildString {
    val patternSize = Size(BOARD_COLS - 1, BOARD_ROWS - 1)
    val views = 4

    val objectPoints = List(views) { boardObjectPoints() }
    val imagePoints = List(views) { view -> projectBoard(view) }

    val cameraMatrix = mat(3, 3, MatType.CV_64FC1)
    val distCoeffs = mat(1, 5, MatType.CV_64FC1)
    try {
        val result = calibrateCamera(objectPoints, imagePoints, Size(640, 480), cameraMatrix, distCoeffs)
        line("rms", "${result.rms}")
        line("cameraMatrix", "${cameraMatrix.describeMatrix()}")
        line("distCoeffs", "${distCoeffs.describeMatrix()}")
        line("views", "${result.rvecs.size}")
    } finally {
        objectPoints.forEach { it.close() }
        imagePoints.forEach { it.close() }
        cameraMatrix.close()
        distCoeffs.close()
    }
}

/** World-frame plane corners for the internal (cols-1)x(rows-1) grid. */
private fun boardObjectPoints(): Mat {
    val pts = ArrayList<cn.enaium.opencv.Point3>(BOARD_COLS * BOARD_ROWS)
    for (r in 0 until BOARD_ROWS - 1) {
        for (c in 0 until BOARD_COLS - 1) {
            pts.add(cn.enaium.opencv.Point3(c * SQUARE.toDouble(), r * SQUARE.toDouble(), 0.0))
        }
    }
    return vectorPoint3fToMat(pts)
}

/** Projects the same corners into a 640x480 view with a plausible pose. */
private fun projectBoard(view: Int): Mat {
    val fx = 400.0
    val fy = 400.0
    val cx = 320.0
    val cy = 240.0
    val z = 900.0
    val cols = BOARD_COLS - 1
    val rows = BOARD_ROWS - 1
    val pts = ArrayList<cn.enaium.opencv.Point2f>(rows * cols)
    for (r in 0 until rows) {
        for (c in 0 until cols) {
            val x = (c - (cols - 1) / 2.0) * SQUARE
            val y = (r - (rows - 1) / 2.0) * SQUARE
            val u = fx * x / z + cx + view * 5.0
            val v = fy * y / z + cy + view * 3.0
            pts.add(cn.enaium.opencv.Point2f(u.toFloat(), v.toFloat()))
        }
    }
    return vectorPoint2fToMat(pts)
}

private fun Mat.describeMatrix(): String {
    val sb = StringBuilder("[")
    for (r in 0 until rows) {
        if (r > 0) sb.append("; ")
        for (c in 0 until cols) {
            if (c > 0) sb.append(", ")
            sb.append(fmt2(at(r, c)))
        }
    }
    return sb.append("]").toString()
}

/** Two-decimal formatting without java.util (K/N compatible). */
private fun fmt2(v: Double): String {
    val scaled = if (v >= 0) (v * 100 + 0.5).toLong() else (v * 100 - 0.5).toLong()
    return "${scaled / 100}.${(if (scaled < 0) -scaled else scaled) % 100}"
}
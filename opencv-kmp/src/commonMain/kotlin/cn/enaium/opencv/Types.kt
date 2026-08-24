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

/** Four-component color value backing OpenCV's `cv::Scalar`. */
data class Scalar(
    val v0: Double = 0.0,
    val v1: Double = 0.0,
    val v2: Double = 0.0,
    val v3: Double = 0.0,
) {
    /** The i-th component; [index] must be within 0..3. */
    operator fun get(index: Int): Double = when (index) {
        0 -> v0
        1 -> v1
        2 -> v2
        else -> v3
    }

    companion object {
        /** Same value in every component, e.g. [all] for gray fills. */
        fun all(value: Double): Scalar = Scalar(value, value, value, value)

        /** Gray color for single-channel images. */
        fun gray(value: Double): Scalar = all(value)

        /** BGR color for three-channel images (blue first, like OpenCV). */
        fun bgr(blue: Double, green: Double, red: Double): Scalar = Scalar(blue, green, red)
    }
}

/** Integer size used by resize-style operations (`cv::Size`). */
data class Size(val width: Int, val height: Int)

/** Integer rectangle backing ROI operations (`cv::Rect`). */
data class Rect(val x: Int, val y: Int, val width: Int, val height: Int) {
    val left: Int get() = x
    val top: Int get() = y
    val right: Int get() = x + width
    val bottom: Int get() = y + height

    companion object {
        /** Rectangle covering the whole [rows]x[cols] image area. */
        fun ofSize(rows: Int, cols: Int): Rect = Rect(0, 0, cols, rows)
    }
}

/** Result of [Mat.minMaxLoc]. Points are (col,row) pairs like OpenCV's. */
data class MinMaxLoc(
    val minVal: Double,
    val maxVal: Double,
    val minX: Int,
    val minY: Int,
    val maxX: Int,
    val maxY: Int,
) {
    val minPoint: Point get() = Point(minX, minY)
    val maxPoint: Point get() = Point(maxX, maxY)
}

/** Integer pixel coordinate (`cv::Point`). */
data class Point(val x: Int, val y: Int)

/** Bounding rotated rectangle from [minAreaRect]; angle in degrees. */
data class RotatedRect(
    val centerX: Double,
    val centerY: Double,
    val width: Double,
    val height: Double,
    val angle: Double,
)

/** Enclosing circle from [minEnclosingCircle]. */
data class Circle(val centerX: Double, val centerY: Double, val radius: Double)

/**
 * Raw spatial moments of a shape ([Mat.moments]); central (`mu`) and
 * normalized (`nu`) moments are derivable from these ten values.
 */
data class Moments(
    val m00: Double,
    val m10: Double,
    val m01: Double,
    val m20: Double,
    val m11: Double,
    val m02: Double,
    val m30: Double,
    val m21: Double,
    val m12: Double,
    val m03: Double,
) {
    /** Central moment mu(i,j); requires m00 != 0. */
    fun mu(i: Int, j: Int): Double {
        val xBar = m10 / m00
        val yBar = m01 / m00
        return when (i * 10 + j) {
            20 -> m20 - xBar * xBar * m00
            11 -> m11 - xBar * yBar * m00
            2 -> m02 - yBar * yBar * m00
            30 -> m30 - 3 * xBar * m20 + 2 * xBar * xBar * xBar * m00
            21 -> m21 - 2 * xBar * m11 - yBar * m20 + 2 * xBar * xBar * yBar * m00
            12 -> m12 - 2 * yBar * m11 - xBar * m02 + 2 * xBar * yBar * yBar * m00
            else -> m03 - 3 * yBar * m02 + 2 * yBar * yBar * yBar * m00
        }
    }
}

/** Result of [Mat.connectedComponentsWithStats]; every Mat must be closed. */
data class Components(
    val count: Int,
    val labels: Mat,
    val stats: Mat,

    /** Nx2 CV_64F centroids. */
    val centroids: Mat,
)

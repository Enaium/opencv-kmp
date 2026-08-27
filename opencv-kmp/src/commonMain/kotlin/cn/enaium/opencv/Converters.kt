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
 * Pure-Kotlin port of `org.opencv.utils.Converters`: List <-> Mat
 * marshalling for every typed vector the SDK moves between Java and native
 * code. There is no native layer — each conversion is built on the [Mat]
 * pixel API (`put`/`get`/`pixels`) and the [MatOfPoint]-style view classes.
 *
 * SDK method names map to camelCase (`vector_Point_to_Mat` ->
 * [vectorPointToMat], `Mat_to_vector_Point` -> [matToVectorPoint]); the Java
 * out-parameter `List` becomes the return value.
 *
 * ### Wire types
 *
 * Produced Mats are Nx1 with the depth/channel layout of the corresponding
 * MatOf* view (frozen cross-slice wire format):
 * `Point` -> CV_32SC2, `Point2f` -> CV_32FC2, `Point2d` -> CV_64FC2,
 * `Point3i` -> CV_32SC3, `Point3f` -> CV_32FC3, `Point3d` -> CV_64FC3,
 * `float` -> CV_32FC1, `uchar` -> CV_8UC1, `char` -> CV_8SC1,
 * `int` -> CV_32SC1, `double` -> CV_64FC1, `Rect` -> CV_32SC4,
 * `Rect2d` -> CV_64FC4, `RotatedRect` -> CV_32FC5, `KeyPoint` -> CV_32FC7,
 * `DMatch` -> CV_32FC4.
 *
 * ### Deviations from the Android SDK
 *
 * 1. `KeyPoint`/`DMatch` encode as CV_32FC7 / CV_32FC4 — the MatOf* wire
 *    types — not the Java `Converters` CV_64FC7 / CV_64FC4 choices. The Java
 *    `Converters` is internally inconsistent: `MatOfKeyPoint`/`MatOfDMatch`
 *    themselves are CV_32FC7 / CV_32FC4, so this port follows the MatOf*
 *    classes that consume and produce these Mats.
 * 2. An empty input list produces an empty Mat (like Java `new Mat()`), and
 *    an empty Mat decodes to an empty list. Java throws when converting the
 *    default-typed empty Mat it produced, making its own empty round trip
 *    impossible; here the empty round trip is total.
 * 3. The SDK's single double-based `Point` maps to two Kotlin types:
 *    [Point] (Int coordinates, CV_32SC2) and [Point2f] (Float coordinates,
 *    CV_32FC2). The CV_32F path therefore takes [Point2f]
 *    ([vectorPoint2fToMat]) and decodes through [matToVectorPoint2f];
 *    [matToVectorPoint] keeps the SDK's three-depth behavior and truncates
 *    float/double coordinates to the Int-based [Point], matching the
 *    behavior of [MatOfPoint2f]'s own `toList()`.
 * 4. The `Mat_to_vector_Point2f`/`Point2d` and `Point3i`/`Point3f`/`Point3d`
 *    SDK aliases collapse into [matToVectorPoint] / [matToVectorPoint3] —
 *    each decodes all three wire depths.
 *
 * ### Not ported (documented skip)
 *
 * Every method whose SDK semantic is *pointer encoding* is omitted:
 * `vector_Mat_to_Mat`, `Mat_to_vector_Mat`, `vector_vector_Mat_to_Mat`,
 * `Mat_to_vector_vector_Mat`, `vector_vector_Point_to_Mat`,
 * `Mat_to_vector_vector_Point`, `vector_vector_Point2f_to_Mat`,
 * `Mat_to_vector_vector_Point2f`, `vector_vector_Point3f_to_Mat`,
 * `Mat_to_vector_vector_Point3f`, `vector_vector_KeyPoint_to_Mat`,
 * `Mat_to_vector_vector_KeyPoint`, `vector_vector_DMatch_to_Mat`,
 * `Mat_to_vector_vector_DMatch`, `vector_vector_char_to_Mat`,
 * `Mat_to_vector_vector_char`, `vector_MatShape_to_Mat`,
 * `Mat_to_vector_MatShape`, `vector_vector_MatShape_to_Mat`,
 * `Mat_to_vector_vector_MatShape`. These pack native object addresses
 * (jlongs) as CV_32SC2 pairs — pure JNI marshalling with no meaning outside
 * the JNI layer. The Kotlin surface already covers the same data flow
 * directly: [Mat.split] / [merge] for `List<Mat>`, the MatOf* views for
 * typed lists, and `List<List<Point>>` contour flows in
 * [Mat.findContours] / [Mat.drawContours].
 */

// =========================================================================
// Points
// =========================================================================

/**
 * `vector_Point_to_Mat`: encodes [pts] as an Nx1 CV_32SC2 Mat (int pairs).
 * The returned Mat must be closed by the caller.
 */
fun vectorPointToMat(pts: List<Point>): Mat = vectorPointToMat(pts, CV_32S)

/**
 * `vector_Point2f_to_Mat`: encodes [pts] as an Nx1 CV_32FC2 Mat (float
 * pairs). The returned Mat must be closed by the caller.
 */
fun vectorPoint2fToMat(pts: List<Point2f>): Mat {
    if (pts.isEmpty()) return mat()
    val values = DoubleArray(pts.size * 2)
    for (i in pts.indices) {
        values[i * 2] = pts[i].x.toDouble()
        values[i * 2 + 1] = pts[i].y.toDouble()
    }
    return packedMat(values, cvMakeType(CV_32F, 2))
}

/**
 * `vector_Point2d_to_Mat`: encodes [pts] as an Nx1 CV_64FC2 Mat (double
 * pairs). [Point] has Int coordinates in this binding, so this is the
 * closest mapping to the SDK's double `Point` input; no precision is lost
 * on encode. The returned Mat must be closed by the caller.
 */
fun vectorPoint2dToMat(pts: List<Point>): Mat = vectorPointToMat(pts, CV_64F)

/**
 * `vector_Point_to_Mat(List, int typeDepth)`: encodes [pts] as an Nx1 Mat of
 * two-channel depth [typeDepth], which must be [CV_32S], [CV_32F] or
 * [CV_64F]; anything else throws [IllegalArgumentException]. An empty list
 * yields an empty Mat. The returned Mat must be closed by the caller.
 *
 * [Point] is Int-based, so the CV_32F/CV_64F branches store integer
 * coordinates; for genuine fractional coordinates use
 * [vectorPoint2fToMat] with [Point2f].
 */
fun vectorPointToMat(pts: List<Point>, typeDepth: Int): Mat {
    if (pts.isEmpty()) return mat()
    val values = DoubleArray(pts.size * 2)
    for (i in pts.indices) {
        values[i * 2] = pts[i].x.toDouble()
        values[i * 2 + 1] = pts[i].y.toDouble()
    }
    return when (typeDepth) {
        CV_32S -> packedMat(values, cvMakeType(CV_32S, 2))
        CV_32F -> packedMat(values, cvMakeType(CV_32F, 2))
        CV_64F -> packedMat(values, cvMakeType(CV_64F, 2))
        else -> throw IllegalArgumentException("'typeDepth' can be CV_32S, CV_32F or CV_64F")
    }
}

/**
 * `Mat_to_vector_Point` (and the `Point2f`/`Point2d` aliases): decodes an
 * Nx1 CV_32SC2, CV_32FC2 or CV_64FC2 Mat back into [Point]s. Fractional
 * coordinates truncate because [Point] is Int-based — use
 * [matToVectorPoint2f] for exact float points. An empty Mat decodes to an
 * empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column or is of
 *   another type.
 */
fun matToVectorPoint(m: Mat): List<Point> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    return when (m.type) {
        cvMakeType(CV_32S, 2) -> MatOfPoint(m).toList()
        cvMakeType(CV_32F, 2) -> MatOfPoint2f(m).toList()
        cvMakeType(CV_64F, 2) -> {
            val values = DoubleArray(m.rows * 2)
            m.get(0, 0, values)
            List(m.rows) { i -> Point(values[i * 2].toInt(), values[i * 2 + 1].toInt()) }
        }
        else -> throw IllegalArgumentException("Input Mat should be of CV_32SC2, CV_32FC2 or CV_64FC2 type")
    }
}

/**
 * Exact float-point decode: reads an Nx1 CV_32FC2 Mat back into [Point2f]s.
 * This is the lossless counterpart of [vectorPoint2fToMat]; the SDK's
 * `Mat_to_vector_Point2f` is otherwise approximated by [matToVectorPoint]
 * (which truncates to the Int-based [Point]). An empty Mat decodes to an
 * empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_32FC2.
 */
fun matToVectorPoint2f(m: Mat): List<Point2f> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_32F, 2)) throw IllegalArgumentException("Input Mat should be of CV_32FC2 type")
    val values = DoubleArray(m.rows * 2)
    m.get(0, 0, values)
    return List(m.rows) { i -> Point2f(values[i * 2].toFloat(), values[i * 2 + 1].toFloat()) }
}

// =========================================================================
// 3D points
// =========================================================================

/**
 * `vector_Point3i_to_Mat`: encodes [pts] as an Nx1 CV_32SC3 Mat, truncating
 * fractional coordinates like the SDK's `(int)` cast. The returned Mat must
 * be closed by the caller.
 */
fun vectorPoint3iToMat(pts: List<Point3>): Mat = vectorPoint3ToMat(pts, CV_32S)

/**
 * `vector_Point3f_to_Mat`: encodes [pts] as an Nx1 CV_32FC3 Mat (float
 * triples). The returned Mat must be closed by the caller.
 */
fun vectorPoint3fToMat(pts: List<Point3>): Mat = vectorPoint3ToMat(pts, CV_32F)

/**
 * `vector_Point3d_to_Mat`: encodes [pts] as an Nx1 CV_64FC3 Mat (double
 * triples). The returned Mat must be closed by the caller.
 */
fun vectorPoint3dToMat(pts: List<Point3>): Mat = vectorPoint3ToMat(pts, CV_64F)

/**
 * `vector_Point3_to_Mat(List, int typeDepth)`: encodes [pts] as an Nx1 Mat
 * of three-channel depth [typeDepth], which must be [CV_32S], [CV_32F] or
 * [CV_64F]; anything else throws [IllegalArgumentException]. An empty list
 * yields an empty Mat. The returned Mat must be closed by the caller.
 */
fun vectorPoint3ToMat(pts: List<Point3>, typeDepth: Int): Mat {
    if (pts.isEmpty()) return mat()
    val values = DoubleArray(pts.size * 3)
    for (i in pts.indices) {
        // The SDK's vector_Point3i_to_Mat casts with (int), which TRUNCATES
        // toward zero (saturate_cast in the native put rounds instead).
        val x = if (typeDepth == CV_32S) pts[i].x.toInt().toDouble() else pts[i].x
        val y = if (typeDepth == CV_32S) pts[i].y.toInt().toDouble() else pts[i].y
        val z = if (typeDepth == CV_32S) pts[i].z.toInt().toDouble() else pts[i].z
        values[i * 3] = x
        values[i * 3 + 1] = y
        values[i * 3 + 2] = z
    }
    return when (typeDepth) {
        CV_32S -> packedMat(values, cvMakeType(CV_32S, 3))
        CV_32F -> packedMat(values, cvMakeType(CV_32F, 3))
        CV_64F -> packedMat(values, cvMakeType(CV_64F, 3))
        else -> throw IllegalArgumentException("'typeDepth' can be CV_32S, CV_32F or CV_64F")
    }
}

/**
 * `Mat_to_vector_Point3` (and the `Point3i`/`Point3f`/`Point3d` aliases):
 * decodes an Nx1 CV_32SC3, CV_32FC3 or CV_64FC3 Mat back into [Point3]s.
 * An empty Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column or is of
 *   another type.
 */
fun matToVectorPoint3(m: Mat): List<Point3> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    return when (m.type) {
        cvMakeType(CV_32S, 3) -> MatOfPoint3(m).toList()
        cvMakeType(CV_32F, 3) -> MatOfPoint3f(m).toList()
        cvMakeType(CV_64F, 3) -> {
            val values = DoubleArray(m.rows * 3)
            m.get(0, 0, values)
            List(m.rows) { i -> Point3(values[i * 3], values[i * 3 + 1], values[i * 3 + 2]) }
        }
        else -> throw IllegalArgumentException("Input Mat should be of CV_32SC3, CV_32FC3 or CV_64FC3 type")
    }
}

// =========================================================================
// Numeric scalars
// =========================================================================

/**
 * `vector_float_to_Mat`: encodes [fs] as an Nx1 CV_32FC1 Mat. The returned
 * Mat must be closed by the caller.
 */
fun vectorFloatToMat(fs: List<Float>): Mat {
    if (fs.isEmpty()) return mat()
    val values = DoubleArray(fs.size) { fs[it].toDouble() }
    return packedMat(values, cvMakeType(CV_32F, 1))
}

/**
 * `Mat_to_vector_float`: decodes an Nx1 CV_32FC1 Mat into [Float]s. An empty
 * Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_32FC1.
 */
fun matToVectorFloat(m: Mat): List<Float> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_32F, 1)) throw IllegalArgumentException("CvType.CV_32FC1 != m.type() || m.cols() != 1")
    return MatOfFloat(m).toList()
}

/**
 * `vector_uchar_to_Mat`: encodes [bs] as an Nx1 CV_8UC1 Mat. The returned
 * Mat must be closed by the caller.
 */
fun vectorUcharToMat(bs: List<Byte>): Mat {
    if (bs.isEmpty()) return mat()
    return byteMat(bs.toByteArray(), cvMakeType(CV_8U, 1))
}

/**
 * `Mat_to_vector_uchar`: decodes an Nx1 CV_8UC1 Mat into signed [Byte]s
 * (bit patterns preserved). An empty Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_8UC1.
 */
fun matToVectorUchar(m: Mat): List<Byte> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_8U, 1)) throw IllegalArgumentException("CvType.CV_8UC1 != m.type() || m.cols() != 1")
    return MatOfByte(m).toList()
}

/**
 * `vector_char_to_Mat`: encodes [bs] as an Nx1 CV_8SC1 Mat. The returned Mat
 * must be closed by the caller.
 */
fun vectorCharToMat(bs: List<Byte>): Mat {
    if (bs.isEmpty()) return mat()
    return byteMat(bs.toByteArray(), cvMakeType(CV_8S, 1))
}

/**
 * `Mat_to_vector_char`: decodes an Nx1 CV_8SC1 Mat into [Byte]s. An empty
 * Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_8SC1.
 */
fun matToVectorChar(m: Mat): List<Byte> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_8S, 1)) throw IllegalArgumentException("CvType.CV_8SC1 != m.type() || m.cols() != 1")
    return m.pixels.toList()
}

/**
 * `vector_int_to_Mat`: encodes [is] as an Nx1 CV_32SC1 Mat. The returned Mat
 * must be closed by the caller.
 */
fun vectorIntToMat(ints: List<Int>): Mat {
    if (ints.isEmpty()) return mat()
    val values = DoubleArray(ints.size) { ints[it].toDouble() }
    return packedMat(values, cvMakeType(CV_32S, 1))
}

/**
 * `Mat_to_vector_int`: decodes an Nx1 CV_32SC1 Mat into [Int]s. An empty Mat
 * decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_32SC1.
 */
fun matToVectorInt(m: Mat): List<Int> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_32S, 1)) throw IllegalArgumentException("CvType.CV_32SC1 != m.type() || m.cols() != 1")
    return MatOfInt(m).toList()
}

/**
 * `vector_double_to_Mat`: encodes [ds] as an Nx1 CV_64FC1 Mat. The returned
 * Mat must be closed by the caller.
 */
fun vectorDoubleToMat(ds: List<Double>): Mat {
    if (ds.isEmpty()) return mat()
    return packedMat(ds.toDoubleArray(), cvMakeType(CV_64F, 1))
}

/**
 * `Mat_to_vector_double`: decodes an Nx1 CV_64FC1 Mat into [Double]s. An
 * empty Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_64FC1.
 */
fun matToVectorDouble(m: Mat): List<Double> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_64F, 1)) throw IllegalArgumentException("CvType.CV_64FC1 != m.type() || m.cols() != 1")
    return MatOfDouble(m).toList()
}

// =========================================================================
// Rectangles
// =========================================================================

/**
 * `vector_Rect_to_Mat`: encodes [rs] as an Nx1 CV_32SC4 Mat
 * (x, y, width, height). The returned Mat must be closed by the caller.
 */
fun vectorRectToMat(rs: List<Rect>): Mat {
    if (rs.isEmpty()) return mat()
    val values = DoubleArray(rs.size * 4)
    for (i in rs.indices) {
        values[i * 4] = rs[i].x.toDouble()
        values[i * 4 + 1] = rs[i].y.toDouble()
        values[i * 4 + 2] = rs[i].width.toDouble()
        values[i * 4 + 3] = rs[i].height.toDouble()
    }
    return packedMat(values, cvMakeType(CV_32S, 4))
}

/**
 * `Mat_to_vector_Rect`: decodes an Nx1 CV_32SC4 Mat into [Rect]s. An empty
 * Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_32SC4.
 */
fun matToVectorRect(m: Mat): List<Rect> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_32S, 4)) throw IllegalArgumentException("CvType.CV_32SC4 != m.type() || m.cols() != 1")
    return MatOfRect(m).toList()
}

/**
 * `vector_Rect2d_to_Mat`: encodes [rs] as an Nx1 CV_64FC4 Mat
 * (x, y, width, height). The returned Mat must be closed by the caller.
 */
fun vectorRect2dToMat(rs: List<Rect2d>): Mat {
    if (rs.isEmpty()) return mat()
    val values = DoubleArray(rs.size * 4)
    for (i in rs.indices) {
        values[i * 4] = rs[i].x
        values[i * 4 + 1] = rs[i].y
        values[i * 4 + 2] = rs[i].width
        values[i * 4 + 3] = rs[i].height
    }
    return packedMat(values, cvMakeType(CV_64F, 4))
}

/**
 * `Mat_to_vector_Rect2d`: decodes an Nx1 CV_64FC4 Mat into [Rect2d]s. An
 * empty Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_64FC4.
 */
fun matToVectorRect2d(m: Mat): List<Rect2d> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_64F, 4)) throw IllegalArgumentException("CvType.CV_64FC4 != m.type() || m.cols() != 1")
    return MatOfRect2d(m).toList()
}

/**
 * `vector_RotatedRect_to_Mat`: encodes [rs] as an Nx1 CV_32FC5 Mat
 * (centerX, centerY, width, height, angle). The returned Mat must be closed
 * by the caller.
 */
fun vectorRotatedRectToMat(rs: List<RotatedRect>): Mat {
    if (rs.isEmpty()) return mat()
    val values = DoubleArray(rs.size * 5)
    for (i in rs.indices) {
        values[i * 5] = rs[i].centerX
        values[i * 5 + 1] = rs[i].centerY
        values[i * 5 + 2] = rs[i].width
        values[i * 5 + 3] = rs[i].height
        values[i * 5 + 4] = rs[i].angle
    }
    return packedMat(values, cvMakeType(CV_32F, 5))
}

/**
 * `Mat_to_vector_RotatedRect`: decodes an Nx1 CV_32FC5 Mat into
 * [RotatedRect]s. An empty Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_32FC5.
 */
fun matToVectorRotatedRect(m: Mat): List<RotatedRect> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_32F, 5)) throw IllegalArgumentException("CvType.CV_32FC5 != m.type() || m.cols() != 1")
    return MatOfRotatedRect(m).toList()
}

// =========================================================================
// Feature types
// =========================================================================

/**
 * `vector_KeyPoint_to_Mat`: encodes [kps] as an Nx1 CV_32FC7 Mat
 * (x, y, size, angle, response, octave, classId) — the [MatOfKeyPoint] wire
 * type. The returned Mat must be closed by the caller.
 */
fun vectorKeyPointToMat(kps: List<KeyPoint>): Mat {
    if (kps.isEmpty()) return mat()
    val values = DoubleArray(kps.size * 7)
    for (i in kps.indices) {
        val kp = kps[i]
        values[i * 7] = kp.x.toDouble()
        values[i * 7 + 1] = kp.y.toDouble()
        values[i * 7 + 2] = kp.size.toDouble()
        values[i * 7 + 3] = kp.angle.toDouble()
        values[i * 7 + 4] = kp.response.toDouble()
        values[i * 7 + 5] = kp.octave.toDouble()
        values[i * 7 + 6] = kp.classId.toDouble()
    }
    return packedMat(values, cvMakeType(CV_32F, 7))
}

/**
 * `Mat_to_vector_KeyPoint`: decodes an Nx1 CV_32FC7 Mat into [KeyPoint]s.
 * An empty Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_32FC7.
 */
fun matToVectorKeyPoint(m: Mat): List<KeyPoint> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_32F, 7)) throw IllegalArgumentException("CvType.CV_32FC7 != m.type() || m.cols() != 1")
    return MatOfKeyPoint(m).toList()
}

/**
 * `vector_DMatch_to_Mat`: encodes [matches] as an Nx1 CV_32FC4 Mat
 * (queryIdx, trainIdx, imgIdx, distance) — the [MatOfDMatch] wire type. The
 * returned Mat must be closed by the caller.
 */
fun vectorDMatchToMat(matches: List<DMatch>): Mat {
    if (matches.isEmpty()) return mat()
    val values = DoubleArray(matches.size * 4)
    for (i in matches.indices) {
        val d = matches[i]
        values[i * 4] = d.queryIdx.toDouble()
        values[i * 4 + 1] = d.trainIdx.toDouble()
        values[i * 4 + 2] = d.imgIdx.toDouble()
        values[i * 4 + 3] = d.distance.toDouble()
    }
    return packedMat(values, cvMakeType(CV_32F, 4))
}

/**
 * `Mat_to_vector_DMatch`: decodes an Nx1 CV_32FC4 Mat into [DMatch]s. An
 * empty Mat decodes to an empty list.
 *
 * @throws IllegalArgumentException if the Mat is not single-column CV_32FC4.
 */
fun matToVectorDMatch(m: Mat): List<DMatch> {
    if (m.isEmpty) return emptyList()
    if (m.cols != 1) throw IllegalArgumentException("Input Mat should have one column")
    if (m.type != cvMakeType(CV_32F, 4)) throw IllegalArgumentException("CvType.CV_32FC4 != m.type() || m.cols() != 1")
    return MatOfDMatch(m).toList()
}

// =========================================================================
// Internal helpers
// =========================================================================

/**
 * Wraps [values] (row-major, channel-interleaved) into an Nx1 Mat of [type];
 * the element count is derived from [cvChannelsOf].
 */
private fun packedMat(values: DoubleArray, type: Int): Mat {
    val res = mat(values.size / cvChannelsOf(type), 1, type)
    res.put(0, 0, values)
    return res
}

/** Wraps [bytes] into an Nx1 Mat of a single-byte [type] (CV_8UC1/CV_8SC1). */
private fun byteMat(bytes: ByteArray, type: Int): Mat {
    val res = mat(bytes.size, 1, type)
    res.pixels = bytes
    return res
}

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
@file:Suppress("MemberVisibilityCanBePrivate", "PropertyName")

package cn.enaium.opencv

// =========================================================================
// org.opencv.core.Core constants (values match the OpenCV 5.0 Java SDK).
// Constants already defined in Const.kt (NormTypes, DecompTypes, DftFlags,
// RotateFlags, BorderTypes, CompareOps, SortFlags, ReduceTypes, KmeansFlags,
// TermCriteriaTypes, LineTypes.FILLED) are not repeated here.
// =========================================================================

/** OpenCV's `cv::SVD::Flags`. */
const val SVD_MODIFY_A: Int = 1
const val SVD_NO_UV: Int = 2
const val SVD_FULL_UV: Int = 4

/** Fill value used by drawing functions (`cv::FILLED`). */
const val FILLED: Int = -1

/** OpenCV's `cv::CovarFlags`. */
const val COVAR_SCRAMBLED: Int = 0
const val COVAR_NORMAL: Int = 1
const val COVAR_USE_AVG: Int = 2
const val COVAR_SCALE: Int = 4
const val COVAR_ROWS: Int = 8
const val COVAR_COLS: Int = 16

/** OpenCV's `cv::PCA::Flags`. */
const val PCA_DATA_AS_ROW: Int = 0
const val PCA_DATA_AS_COL: Int = 1
const val PCA_USE_AVG: Int = 2

/** OpenCV's `cv::RNG` distribution types. */
const val RNG_UNIFORM: Int = 0
const val RNG_NORMAL: Int = 1

/** OpenCV's `cv::AlgorithmHint`. */
const val ALGO_HINT_DEFAULT: Int = 0
const val ALGO_HINT_ACCURATE: Int = 1
const val ALGO_HINT_APPROX: Int = 2

/** OpenCV's `cv::GemmFlags`. */
const val GEMM_1_T: Int = 1
const val GEMM_2_T: Int = 2
const val GEMM_3_T: Int = 4

/** OpenCV's `cv::ReduceTypes` extension (sum of squares). */
const val REDUCE_SUM2: Int = 4

/** OpenCV's `cv::Param` type ids used by Algorithm parameter maps. */
const val Param_INT: Int = 0
const val Param_BOOLEAN: Int = 1
const val Param_REAL: Int = 2
const val Param_STRING: Int = 3
const val Param_MAT: Int = 4
const val Param_MAT_VECTOR: Int = 5
const val Param_ALGORITHM: Int = 6
const val Param_FLOAT: Int = 7
const val Param_UNSIGNED_INT: Int = 8
const val Param_UINT64: Int = 9
const val Param_UCHAR: Int = 11
const val Param_SCALAR: Int = 12

/** OpenCV's `cv::dnn::DataLayout`. */
const val DATA_LAYOUT_UNKNOWN: Int = 0
const val DATA_LAYOUT_ND: Int = 1
const val DATA_LAYOUT_NCHW: Int = 2
const val DATA_LAYOUT_NCDHW: Int = 3
const val DATA_LAYOUT_NHWC: Int = 4
const val DATA_LAYOUT_NDHWC: Int = 5
const val DATA_LAYOUT_PLANAR: Int = 6
const val DATA_LAYOUT_BLOCK: Int = 7

/** OpenCV's `cv::Formatter::FormatType`. */
const val Formatter_FMT_DEFAULT: Int = 0
const val Formatter_FMT_MATLAB: Int = 1
const val Formatter_FMT_CSV: Int = 2
const val Formatter_FMT_PYTHON: Int = 3
const val Formatter_FMT_NUMPY: Int = 4
const val Formatter_FMT_C: Int = 5

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

/*
 * Plain C ABI over the C++ OpenCV API.
 *
 * OpenCV 4/5 has no usable pure-C interface, which cinterop cannot bind, so
 * opencv-kmp defines its own thin C layer ("cvk_" prefix). The same functions
 * back both Kotlin/Native (static library embedded in the klib via cinterop)
 * and the JVM (JNI shared library built from jni_bridge.cpp on top of these).
 *
 * Every function is noexcept: cv::Exception is caught and reported through
 * cvk_last_error(); calls that return a new Mat return NULL instead. Mat
 * handles returned by cvk_mat_* are owned by the caller and must be freed
 * with exactly one cvk_mat_release().
 */
#ifndef OPENCV_KMP_H
#define OPENCV_KMP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvk_mat cvk_mat_t;

/** Four-component color value backing cv::Scalar. */
typedef struct cvk_scalar {
    double v0;
    double v1;
    double v2;
    double v3;
} cvk_scalar_t;

/** Integer rectangle backing cv::Rect (x/y is a pixel corner). */
typedef struct cvk_rect {
    int x;
    int y;
    int width;
    int height;
} cvk_rect_t;

/* =========================================================================
 * Info / errors
 * ========================================================================= */

/** OpenCV version string, e.g. "5.0.0"; never NULL. */
const char *cvk_version(void);

/**
 * Message of the last caught OpenCV exception on this thread, or NULL when
 * no exception was seen yet. The string stays valid until the next call
 * into this library from the same thread.
 */
const char *cvk_last_error(void);

/** Clears the thread-local last-error state. */
void cvk_clear_error(void);

/* =========================================================================
 * Mat lifecycle
 * ========================================================================= */

/** Creates rows x cols matrix of the given type; rows==0 means empty. */
cvk_mat_t *cvk_mat_create(int rows, int cols, int type);

/** Like cvk_mat_create but fills every element with `value`. */
cvk_mat_t *cvk_mat_create_filled(int rows, int cols, int type, cvk_scalar_t value);

/** cv::Mat::zeros(rows, cols, type). */
cvk_mat_t *cvk_mat_zeros(int rows, int cols, int type);

/** cv::Mat::ones(rows, cols, type). */
cvk_mat_t *cvk_mat_ones(int rows, int cols, int type);

/** Identity matrix; for multi-channel types only channel 0 is set to 1. */
cvk_mat_t *cvk_mat_eye(int rows, int cols, int type);

/** Deep copy. */
cvk_mat_t *cvk_mat_clone(const cvk_mat_t *mat);

/**
 * Region of interest sharing the source pixels (no copy), like mat(rect).
 * Releasing the returned handle decrements the shared reference count only.
 */
cvk_mat_t *cvk_mat_roi(const cvk_mat_t *mat, cvk_rect_t rect);

/** Decrements the reference count and frees the handle when it reaches 0. */
void cvk_mat_release(cvk_mat_t *mat);

/* =========================================================================
 * Mat properties
 * ========================================================================= */

int cvk_mat_rows(const cvk_mat_t *mat);
int cvk_mat_cols(const cvk_mat_t *mat);
int cvk_mat_type(const cvk_mat_t *mat);
int cvk_mat_channels(const cvk_mat_t *mat);
/** Bytes per element across all channels. */
size_t cvk_mat_elem_size(const cvk_mat_t *mat);
/** Number of elements (rows*cols, or more in multi-dim cases). */
size_t cvk_mat_total(const cvk_mat_t *mat);
int cvk_mat_is_empty(const cvk_mat_t *mat);

/** Raw pointer to the first byte of pixel data. */
unsigned char *cvk_mat_data(cvk_mat_t *mat);

/** Reads one element as double (bounds-checked by cv::at). */
double cvk_mat_get(const cvk_mat_t *mat, int row, int col, int channel);
/** Writes one element (converted to the Mat's depth like cv::at assignment). */
void cvk_mat_set(cvk_mat_t *mat, int row, int col, int channel, double value);

/* =========================================================================
 * Conversions / arithmetic (every function returns a new Mat or NULL)
 * ========================================================================= */

/** dst = src.convertTo(rtype, alpha, beta). rtype < 0 keeps the depth. */
cvk_mat_t *cvk_mat_convert_to(const cvk_mat_t *mat, int rtype, double alpha, double beta);

cvk_mat_t *cvk_mat_add(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_mat_subtract(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_mat_multiply(const cvk_mat_t *a, const cvk_mat_t *b, double scale);
cvk_mat_t *cvk_mat_divide(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_mat_scale_add(const cvk_mat_t *mat, double alpha, double beta);
cvk_mat_t *cvk_mat_absdiff(const cvk_mat_t *a, const cvk_mat_t *b);

cvk_mat_t *cvk_mat_bitwise_and(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_mat_bitwise_or(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_mat_bitwise_xor(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_mat_bitwise_not(const cvk_mat_t *a);
cvk_mat_t *cvk_mat_min(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_mat_max(const cvk_mat_t *a, const cvk_mat_t *b);

/** dst(i) = lower(i) <= src(i) <= upper(i) ? 255 : 0. */
cvk_mat_t *cvk_mat_in_range(const cvk_mat_t *mat, cvk_scalar_t lower, cvk_scalar_t upper);

cvk_mat_t *cvk_mat_transpose(const cvk_mat_t *mat);
/** flipCode: 0 x-axis, >0 y-axis, <0 both. */
cvk_mat_t *cvk_mat_flip(const cvk_mat_t *mat, int flip_code);

/* Reductions / statistics. */

cvk_scalar_t cvk_mat_mean(const cvk_mat_t *mat);
cvk_scalar_t cvk_mat_sum(const cvk_mat_t *mat);
/** Writes mean then stddev into out[0..7]. */
void cvk_mat_mean_stddev(const cvk_mat_t *mat, double *out8);
/** Writes minVal,maxVal,minX,minY,maxX,maxY into out6. */
void cvk_mat_min_max_loc(const cvk_mat_t *mat, double *out6);
int cvk_mat_count_non_zero(const cvk_mat_t *mat);

/* =========================================================================
 * imgproc
 * ========================================================================= */

cvk_mat_t *cvk_cvt_color(const cvk_mat_t *src, int code);
cvk_mat_t *cvk_resize(const cvk_mat_t *src, int width, int height, int interpolation);
cvk_mat_t *cvk_gaussian_blur(const cvk_mat_t *src, int kernel_width, int kernel_height,
                             double sigma_x, double sigma_y);
cvk_mat_t *cvk_median_blur(const cvk_mat_t *src, int kernel_size);
cvk_mat_t *cvk_threshold(const cvk_mat_t *src, double thresh, double maxval, int type);
cvk_mat_t *cvk_adaptive_threshold(const cvk_mat_t *src, double max_value, int method,
                                  int type, int block_size, double c);
cvk_mat_t *cvk_canny(const cvk_mat_t *src, double threshold1, double threshold2,
                     int aperture_size, int l2_gradient);
cvk_mat_t *cvk_sobel(const cvk_mat_t *src, int dx, int dy, int kernel_size);
cvk_mat_t *cvk_laplacian(const cvk_mat_t *src, int kernel_size);

/* Drawing (in-place, thickness < 0 fills the shape). */

void cvk_rectangle(cvk_mat_t *mat, int x1, int y1, int x2, int y2,
                   cvk_scalar_t color, int thickness);
void cvk_circle(cvk_mat_t *mat, int center_x, int center_y, int radius,
                cvk_scalar_t color, int thickness);
void cvk_line(cvk_mat_t *mat, int x1, int y1, int x2, int y2,
              cvk_scalar_t color, int thickness);

/* =========================================================================
 * imgcodecs
 * ========================================================================= */

cvk_mat_t *cvk_imread(const char *filename, int flags);
int cvk_imwrite(const char *filename, const cvk_mat_t *mat);

/**
 * Encodes the Mat into `ext` format ("png", "jpg", ...); stores the byte
 * count into *out_len and returns a buffer released by cvk_free_buffer.
 * Returns NULL on failure.
 */
unsigned char *cvk_imencode(const char *ext, const cvk_mat_t *mat, size_t *out_len);

/** Decodes an encoded image from memory; NULL on failure. */
cvk_mat_t *cvk_imdecode(const unsigned char *data, size_t len, int flags);

/** Releases a buffer produced by cvk_imencode. */
void cvk_free_buffer(unsigned char *buffer);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_H */

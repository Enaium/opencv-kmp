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

/** Opaque handle to a cv::Ptr<cv::CLAHE>. */
typedef struct cvk_clahe cvk_clahe_t;

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
/** Centralized per-channel Scalar arithmetic (cv::add/subtract/multiply/divide). */
cvk_mat_t *cvk_mat_add_scalar(const cvk_mat_t *a, cvk_scalar_t s);
cvk_mat_t *cvk_mat_subtract_scalar(const cvk_mat_t *a, cvk_scalar_t s);
cvk_mat_t *cvk_mat_multiply_scalar(const cvk_mat_t *a, cvk_scalar_t s);
cvk_mat_t *cvk_mat_divide_scalar(const cvk_mat_t *a, cvk_scalar_t s);
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

/* =========================================================================
 * Mat members (core)
 * ========================================================================= */

/** mat.reshape(channels, rows); rows==0 keeps the row count. */
cvk_mat_t *cvk_mat_reshape(const cvk_mat_t *mat, int channels, int rows);

/** Row range sharing pixels, like mat.rowRange(start,end) (end exclusive). */
cvk_mat_t *cvk_mat_row_range(const cvk_mat_t *mat, int start, int end);

/** Column range sharing pixels (end exclusive). */
cvk_mat_t *cvk_mat_col_range(const cvk_mat_t *mat, int start, int end);

/** mat.diag(d): d==0 main diagonal, d>0 below, d<0 above. */
cvk_mat_t *cvk_mat_diag(const cvk_mat_t *mat, int d);

/** Scales the diagonal by `scale` in place. */
void cvk_mat_set_identity(cvk_mat_t *mat, double scale);

/** Dot product of two same-shaped single-channel matrices. */
double cvk_mat_dot(const cvk_mat_t *a, const cvk_mat_t *b);

/** Matrix inverse; method is a DecompTypes value. NULL on singular input. */
cvk_mat_t *cvk_mat_inv(const cvk_mat_t *mat, int method);

/** Determinant of a square single-channel matrix; NaN on failure. */
double cvk_mat_determinant(const cvk_mat_t *mat);

/** Sum of diagonal elements. */
cvk_scalar_t cvk_mat_trace(const cvk_mat_t *mat);

/* =========================================================================
 * core: array operations
 * ========================================================================= */

/**
 * Splits channels into up to `max_count` matrices; writes their handles
 * into out[] and returns the channel count. Handles are caller-owned.
 */
int cvk_split(const cvk_mat_t *src, cvk_mat_t **out, int max_count);

/** Merges `count` single-channel matrices into one multi-channel result. */
cvk_mat_t *cvk_merge(const cvk_mat_t **mv, int count);

cvk_mat_t *cvk_hconcat(const cvk_mat_t *a, const cvk_mat_t *b);
cvk_mat_t *cvk_vconcat(const cvk_mat_t *a, const cvk_mat_t *b);

/** Norm of a matrix (NormTypes value). */
double cvk_norm(const cvk_mat_t *src, int norm_type);

/** Absolute-difference norm between two matrices. */
double cvk_norm_diff(const cvk_mat_t *a, const cvk_mat_t *b, int norm_type);

/** Normalizes to alpha..beta with a NormTypes norm; dtype<0 keeps depth. */
cvk_mat_t *cvk_normalize(const cvk_mat_t *src, double alpha, double beta,
                         int norm_type, int dtype);

/** Lookup-table remap for 8-bit sources. */
cvk_mat_t *cvk_lut(const cvk_mat_t *src, const cvk_mat_t *lut);

/** Rotates by 90/180/270 degrees (RotateFlags). */
cvk_mat_t *cvk_rotate(const cvk_mat_t *src, int code);

/** Adds a border of `border_type` with the constant `value`. */
cvk_mat_t *cvk_copy_make_border(const cvk_mat_t *src, int top, int bottom,
                                int left, int right, int border_type,
                                cvk_scalar_t value);

/** dst = alpha*a + beta*b + gamma. */
cvk_mat_t *cvk_add_weighted(const cvk_mat_t *a, double alpha,
                            const cvk_mat_t *b, double beta, double gamma);

/** convertScaleAbs then absolute values into CV_8U. */
cvk_mat_t *cvk_convert_scale_abs(const cvk_mat_t *src, double alpha, double beta);

/** Element-wise comparison (CompareTypes) producing an 8-bit 0/255 mask. */
cvk_mat_t *cvk_compare(const cvk_mat_t *a, const cvk_mat_t *b, int cmp_op);

/** Solves a*x = b; returns x or NULL when no solution (DecompTypes flags). */
cvk_mat_t *cvk_solve(const cvk_mat_t *a, const cvk_mat_t *b, int flags);

/** Repeats the matrix nx times horizontally and ny times vertically. */
cvk_mat_t *cvk_repeat(const cvk_mat_t *src, int nx, int ny);

/** Per-element transform through a matrix (channels map to m rows). */
cvk_mat_t *cvk_transform(const cvk_mat_t *src, const cvk_mat_t *m);

/** Point transform by a 3x3 (2D points) or 4x4 (3D points) matrix. */
cvk_mat_t *cvk_perspective_transform(const cvk_mat_t *src, const cvk_mat_t *m);

cvk_mat_t *cvk_pow(const cvk_mat_t *src, double power);
cvk_mat_t *cvk_sqrt(const cvk_mat_t *src);
cvk_mat_t *cvk_exp(const cvk_mat_t *src);
cvk_mat_t *cvk_log(const cvk_mat_t *src);

/** Magnitude/phase helpers over complex pairs expressed as two matrices. */
cvk_mat_t *cvk_magnitude(const cvk_mat_t *x, const cvk_mat_t *y);
cvk_mat_t *cvk_phase(const cvk_mat_t *x, const cvk_mat_t *y, int angle_in_degrees);
void cvk_cart_to_polar(const cvk_mat_t *x, const cvk_mat_t *y,
                       int angle_in_degrees, cvk_mat_t **magnitude,
                       cvk_mat_t **angle);
void cvk_polar_to_cart(const cvk_mat_t *magnitude, const cvk_mat_t *angle,
                       int angle_in_degrees, cvk_mat_t **x, cvk_mat_t **y);

/** Replaces NaNs with `value` in place. */
void cvk_patch_nans(cvk_mat_t *mat, double value);

/** Non-zero coordinates as an Nx1 CV_32SC2 matrix. */
cvk_mat_t *cvk_find_non_zero(const cvk_mat_t *src);

int cvk_has_non_zero(const cvk_mat_t *src);

/** Sorts every row/column (SortFlags). */
cvk_mat_t *cvk_sort(const cvk_mat_t *src, int flags);
cvk_mat_t *cvk_sort_idx(const cvk_mat_t *src, int flags);

/** Reduces rows (dim==0) / cols (dim==1) with a ReduceTypes op; dtype<0 keeps depth. */
cvk_mat_t *cvk_reduce(const cvk_mat_t *src, int dim, int rtype, int dtype);

/** Arg-max/min along dim; result is CV_32S indices. */
cvk_mat_t *cvk_reduce_arg_max(const cvk_mat_t *src, int dim);
cvk_mat_t *cvk_reduce_arg_min(const cvk_mat_t *src, int dim);

/** Copies channel [coi] of src into its own matrix / back from one. */
cvk_mat_t *cvk_extract_channel(const cvk_mat_t *src, int coi);
void cvk_insert_channel(const cvk_mat_t *src, cvk_mat_t *dst, int coi);

/** Uniformly distributed random fill (Scalar bounds). */
void cvk_randu(cvk_mat_t *dst, cvk_scalar_t low, cvk_scalar_t high);

/** Normally distributed random fill. */
void cvk_randn(cvk_mat_t *dst, cvk_scalar_t mean, cvk_scalar_t stddev);

/** Seeds the global RNG (theRNG). */
void cvk_set_rng_seed(unsigned long long seed);

/** PSNR in decibels between two images. */
double cvk_psnr(const cvk_mat_t *a, const cvk_mat_t *b, double r);

/** Discrete Fourier/Cosine transforms (DftFlags). */
cvk_mat_t *cvk_dft(const cvk_mat_t *src, int flags);
cvk_mat_t *cvk_idft(const cvk_mat_t *src, int flags);
cvk_mat_t *cvk_dct(const cvk_mat_t *src, int flags);
cvk_mat_t *cvk_idct(const cvk_mat_t *src, int flags);
int cvk_get_optimal_dft_size(int rowsize);

/** Element-wise complex spectrum multiply/divide (CV_32FC2/CV_64FC2). */
cvk_mat_t *cvk_mul_spectrums(const cvk_mat_t *a, const cvk_mat_t *b,
                             int conj_flag, int dft_rows);
cvk_mat_t *cvk_div_spectrums(const cvk_mat_t *a, const cvk_mat_t *b,
                             int conj_flag);

/** Generalized matrix product dst = alpha*a*b + gamma*c (c may be NULL). */
cvk_mat_t *cvk_gemm(const cvk_mat_t *a, const cvk_mat_t *b, double alpha,
                    const cvk_mat_t *c, double gamma);

/** Symmetric eigen decomposition; writes eigenvalues/eigenvectors handles. */
void cvk_eigen(const cvk_mat_t *src, cvk_mat_t **eigenvalues,
               cvk_mat_t **eigenvectors);

/** Number of threads OpenCV will use / caps it (0 restores default). */
int cvk_num_threads(void);
void cvk_set_num_threads(int count);

/** Verbose build information string; never NULL. */
const char *cvk_build_information(void);

/* =========================================================================
 * imgproc: filters
 * ========================================================================= */

/** Simple box blur with a centered kernel. */
cvk_mat_t *cvk_blur(const cvk_mat_t *src, int kernel_width, int kernel_height);

/** Normalized (or not) box filter; ddepth may be -1. */
cvk_mat_t *cvk_box_filter(const cvk_mat_t *src, int ddepth, int kernel_width,
                          int kernel_height, int normalize);

/** Squared box filter; ddepth should be CV_32F/CV_64F/-1. */
cvk_mat_t *cvk_sqr_box_filter(const cvk_mat_t *src, int ddepth,
                              int kernel_width, int kernel_height);

/** Edge-preserving smoothing. */
cvk_mat_t *cvk_bilateral_filter(const cvk_mat_t *src, int d,
                                double sigma_color, double sigma_space);

/** Exact sliding-window median (stackBlur). */
cvk_mat_t *cvk_stack_blur(const cvk_mat_t *src, int kernel_size);

/** Erodes/dilates/morphs with `kernel`; iterations >= 1. */
cvk_mat_t *cvk_erode(const cvk_mat_t *src, const cvk_mat_t *kernel, int iterations);
cvk_mat_t *cvk_dilate(const cvk_mat_t *src, const cvk_mat_t *kernel, int iterations);
cvk_mat_t *cvk_morphology_ex(const cvk_mat_t *src, int op,
                             const cvk_mat_t *kernel, int iterations);

/** Kernel factory (MorphShapes). */
cvk_mat_t *cvk_get_structuring_element(int shape, int width, int height);

/** Gaussian coefficient row kernel. */
cvk_mat_t *cvk_get_gaussian_kernel(int ksize, double sigma);

/** Convolves with `kernel` keeping depth (ddepth=-1) plus offset delta. */
cvk_mat_t *cvk_filter_2d(const cvk_mat_t *src, const cvk_mat_t *kernel,
                         int ddepth, double delta);

cvk_mat_t *cvk_pyr_down(const cvk_mat_t *src);
cvk_mat_t *cvk_pyr_up(const cvk_mat_t *src);

/* =========================================================================
 * imgproc: geometry / warps
 * ========================================================================= */

/** Warps with a 2x3 affine matrix into width x height (InterpolationFlags). */
cvk_mat_t *cvk_warp_affine(const cvk_mat_t *src, const cvk_mat_t *m,
                           int width, int height, int flags);

/** Warps with a 3x3 perspective matrix. */
cvk_mat_t *cvk_warp_perspective(const cvk_mat_t *src, const cvk_mat_t *m,
                                int width, int height, int flags);

/** Applies generic coordinate maps produced by warp helpers. */
cvk_mat_t *cvk_remap(const cvk_mat_t *src, const cvk_mat_t *map1,
                     const cvk_mat_t *map2, int interpolation);

/** Polar/log-polar remap around center with maxRadius (WarpPolarMode flags). */
cvk_mat_t *cvk_warp_polar(const cvk_mat_t *src, int radius,
                          double center_x, double center_y,
                          double max_radius, int flags);

/** Affine transform mapping three source points onto three destinations. */
cvk_mat_t *cvk_get_affine_transform(double sx0, double sy0, double sx1,
                                    double sy1, double sx2, double sy2,
                                    double dx0, double dy0, double dx1,
                                    double dy1, double dx2, double dy2);

/** Inverts a 2x3 affine transform. */
cvk_mat_t *cvk_invert_affine_transform(const cvk_mat_t *m);

/** Perspective transform mapping four source points onto four destinations. */
cvk_mat_t *cvk_get_perspective_transform(double sx0, double sy0, double sx1,
                                         double sy1, double sx2, double sy2,
                                         double sx3, double sy3, double dx0,
                                         double dy0, double dx1, double dy1,
                                         double dx2, double dy2, double dx3,
                                         double dy3);

/** Rotation matrix around (cx,cy) by `angle` degrees with uniform scale. */
cvk_mat_t *cvk_get_rotation_matrix_2d(double cx, double cy, double angle,
                                      double scale);

/** Extracts a sub-pixel patch centered at (cx,cy). */
cvk_mat_t *cvk_get_rect_sub_pix(const cvk_mat_t *src, int width, int height,
                                double cx, double cy);

/** Lens undistortion using camera intrinsics and distortion coefficients. */
cvk_mat_t *cvk_undistort(const cvk_mat_t *src, const cvk_mat_t *camera_matrix,
                         const cvk_mat_t *dist_coeffs);

/* =========================================================================
 * imgproc: color / histogram
 * ========================================================================= */

/** Converts Bayer patterns to color (ColorConversionCodes demosaic codes). */
cvk_mat_t *cvk_demosaicing(const cvk_mat_t *src, int code);

/** Pseudo-coloring with a built-in palette (ColormapTypes). */
cvk_mat_t *cvk_apply_colormap(const cvk_mat_t *src, int colormap);

/** Pseudo-coloring with a user-supplied lookup table (CV_8UC1 or CV_8UC3). */
cvk_mat_t *cvk_apply_colormap_user(const cvk_mat_t *src, const cvk_mat_t *user_color);

/**
 * One-dimensional histogram of [channel] over [hist_size] bins spanning
 * [min_value, max_value]; returns a hist_size x 1 CV_32F matrix.
 */
cvk_mat_t *cvk_calc_hist(const cvk_mat_t *src, int channel, int hist_size,
                         float min_value, float max_value);

/** Back-projects a histogram computed by cvk_calc_hist onto the image. */
cvk_mat_t *cvk_calc_back_project(const cvk_mat_t *src, int channel,
                                 const cvk_mat_t *hist, float min_value,
                                 float max_value);

/** Histogram similarity (HistCompMethods). */
double cvk_compare_hist(const cvk_mat_t *h1, const cvk_mat_t *h2, int method);

/** Histogram equalization of an 8-bit grayscale image. */
cvk_mat_t *cvk_equalize_hist(const cvk_mat_t *src);

/* =========================================================================
 * imgproc: segmentation / contours / features
 * ========================================================================= */

/** Fills the connected component at (seed_x, seed_y); returns filled area. */
int cvk_flood_fill(cvk_mat_t *image, int seed_x, int seed_y,
                   cvk_scalar_t new_value, cvk_scalar_t lo_diff,
                   cvk_scalar_t up_diff, int flags);

/** Segments background/foreground via graph cuts on marker labels (in place). */
void cvk_watershed(cvk_mat_t *image, cvk_mat_t *markers);

/**
 * Contours travel as one flat little-endian byte buffer:
 *
 *   uint32 count
 *   per contour: uint32 point-count followed by int32 x,y pairs
 *
 * Buffers are allocated with malloc and released by cvk_free_buffer.
 */

/** Finds contours of a binary image (RetrievalModes, ContourApproximationModes). */
unsigned char *cvk_find_contours(const cvk_mat_t *src, int mode, int method,
                                 size_t *out_len);

/** Draws contours (-1 draws all); flat is a cvk_find_contours buffer. */
void cvk_draw_contours(cvk_mat_t *image, const unsigned char *flat, size_t len,
                       int contour_index, cvk_scalar_t color, int thickness);

/** Area enclosed by exactly one contour. */
double cvk_contour_area(const unsigned char *flat, size_t len);

/** Perimeter of exactly one contour. */
double cvk_arc_length(const unsigned char *flat, size_t len, int closed);

/** Axis-aligned bounding box of exactly one contour; writes x,y,w,h. */
void cvk_bounding_rect(const unsigned char *flat, size_t len, int out[4]);

/** Approximates a contour with fewer points (Douglas-Peucker). */
unsigned char *cvk_approx_poly_dp(const unsigned char *flat, size_t len,
                                  double epsilon, int closed, size_t *out_len);

/** Bounding rotated rectangle: cx,cy,width,height,angle. */
void cvk_min_area_rect(const unsigned char *flat, size_t len, double out[5]);

/** Enclosing circle: cx,cy,radius. */
void cvk_min_enclosing_circle(const unsigned char *flat, size_t len, double out[3]);

/** Image moments of a rasterized shape or contour matrix. */
void cvk_moments(const cvk_mat_t *arr, int binary_image, double out[10]);

/** Shape-context-free Hu moment matching (ContourApproximationModes? no: HistCompMethods-like 1..3). */
double cvk_match_shapes(const cvk_mat_t *a, const cvk_mat_t *b, int method);

/** Line segment detection via probabilistic Hough; Nx1 CV_32SC4 lines. */
cvk_mat_t *cvk_hough_lines_p(const cvk_mat_t *src, double rho, double theta,
                             int threshold, double min_line_length,
                             double max_line_gap);

/** Standard Hough lines; Nx1 CV_32FC2 (rho, theta). */
cvk_mat_t *cvk_hough_lines(const cvk_mat_t *src, double rho, double theta,
                           int threshold, double srn, double stn);

/** Circle Hough; Nx1 CV_32FC3 (x, y, radius). */
cvk_mat_t *cvk_hough_circles(const cvk_mat_t *src, int method, double dp,
                             double min_dist, double param1, double param2,
                             int min_radius, int max_radius);

/** Harris corner response map. */
cvk_mat_t *cvk_corner_harris(const cvk_mat_t *src, int block_size, int ksize,
                             double k);

/** Minimum eigenvalue corner map (Shi-Tomasi). */
cvk_mat_t *cvk_corner_min_eigen_val(const cvk_mat_t *src, int block_size,
                                    int ksize);

/** Strong corners; Nx1 CV_32FC2 points. */
cvk_mat_t *cvk_good_features_to_track(const cvk_mat_t *src, int max_corners,
                                      double quality_level, double min_distance,
                                      int block_size, int use_harris_detector,
                                      double k);

/** Slides `templ` over the image (TemplateMatchModes). */
cvk_mat_t *cvk_match_template(const cvk_mat_t *image, const cvk_mat_t *templ,
                              int method);

/** Distance transform of an edge map (DistanceTypes, mask size 3/5/FILLED). */
cvk_mat_t *cvk_distance_transform(const cvk_mat_t *src, int distance_type,
                                  int mask_size);

/** Running 32-bit integral image (sum only). */
cvk_mat_t *cvk_integral(const cvk_mat_t *src, int sdepth);

/** Labels connected components; returns label count. */
int cvk_connected_components(const cvk_mat_t *src, cvk_mat_t **labels,
                             int connectivity, int ltype);

/** Like cvk_connected_components plus stats (Nx5 CV_32S) and centroids (Nx2 CV_64F). */
int cvk_connected_components_with_stats(const cvk_mat_t *src, cvk_mat_t **labels,
                                        cvk_mat_t **stats, cvk_mat_t **centroids,
                                        int connectivity, int ltype);

/** Mean-shift segmentation. */
cvk_mat_t *cvk_pyr_mean_shift_filtering(const cvk_mat_t *src, double sp,
                                        double sr, int max_level);

/** Thresholds with a mask; returns the computed threshold (Otsu/Triangle). */
double cvk_threshold_with_mask(const cvk_mat_t *src, const cvk_mat_t *mask,
                               cvk_mat_t *dst, double thresh, double maxval,
                               int type);

/** Window function (createHanningWindow) of given size and depth. */
cvk_mat_t *cvk_create_hanning_window(int width, int height, int type);

/** Frame accumulators updating `this` in place. */
void cvk_accumulate(const cvk_mat_t *src, cvk_mat_t *dst);
void cvk_accumulate_square(const cvk_mat_t *src, cvk_mat_t *dst);
void cvk_accumulate_product(const cvk_mat_t *a, const cvk_mat_t *b, cvk_mat_t *dst);
void cvk_accumulate_weighted(const cvk_mat_t *src, cvk_mat_t *dst, double alpha);

/* =========================================================================
 * imgproc: drawing
 * ========================================================================= */

void cvk_arrowed_line(cvk_mat_t *mat, int x1, int y1, int x2, int y2,
                      cvk_scalar_t color, int thickness);

/** Marker glyph (MarkerTypes) centered at (x,y). */
void cvk_draw_marker(cvk_mat_t *mat, int x, int y, int marker_type, int size,
                     cvk_scalar_t color, int thickness);

/** Elliptical arc (angles in degrees, thickness<0 fills). */
void cvk_ellipse(cvk_mat_t *mat, int cx, int cy, int axes_x, int axes_y,
                 double angle, double start_angle, double end_angle,
                 cvk_scalar_t color, int thickness);

/**
 * Fills/strokes polygons. Points come as a cvk_find_contours flat buffer;
 * thickness < 0 fills.
 */
void cvk_fill_poly(cvk_mat_t *mat, const unsigned char *flat, size_t len,
                   cvk_scalar_t color, int thickness);

/** Strokes polylines from a contours flat buffer. */
void cvk_polylines(cvk_mat_t *mat, const unsigned char *flat, size_t len,
                   int closed, cvk_scalar_t color, int thickness);

/* =========================================================================
 * imgproc: CLAHE
 * ========================================================================= */

/** Creates a CLAHE algorithm; release with cvk_clahe_release. */
cvk_clahe_t *cvk_clahe_create(double clip_limit, int tile_width, int tile_height);

/** Applies contrast limited adaptive histogram equalization. */
cvk_mat_t *cvk_clahe_apply(cvk_clahe_t *clahe, const cvk_mat_t *src);

/** Updates the clip limit. */
void cvk_clahe_set_clip_limit(cvk_clahe_t *clahe, double clip_limit);

/** Releases a cvk_clahe_create handle (NULL tolerated). */
void cvk_clahe_release(cvk_clahe_t *clahe);

/* =========================================================================
 * imgcodecs additions
 * ========================================================================= */

/** Number of frames inside an image file (gif/tiff pages), else 1. */
int cvk_imcount(const char *filename);

/** Whether the image file at this path can be decoded / re-encoded. */
int cvk_have_image_reader(const char *filename);
int cvk_have_image_writer(const char *filename);

/** imwrite with codec parameters (IMWRITE_JPEG_QUALITY, ...). */
int cvk_imwrite_params(const char *filename, const cvk_mat_t *mat,
                       const int *params, size_t params_len);

/** cvk_imencode with codec parameters. */
unsigned char *cvk_imencode_params(const char *ext, const cvk_mat_t *mat,
                                   const int *params, size_t params_len,
                                   size_t *out_len);
/* =========================================================================
 * Official Java/Python SDK parity surface
 * =========================================================================
 */

/* ---- Mat members (org.opencv.core.Mat parity) ---- */

/** Debug dump of the matrix content (Java Mat.dump). Static buffer. */
const char *cvk_mat_dump(const cvk_mat_t *mat);

int cvk_mat_is_continuous(const cvk_mat_t *mat);

/** Whether the matrix is a ROI view of a larger one. */
int cvk_mat_is_submatrix(const cvk_mat_t *mat);

/** Java Mat.adjustROI; returns the adjusted sharing view. */
cvk_mat_t *cvk_mat_adjust_roi(const cvk_mat_t *mat, int dtop, int dbottom,
                              int dleft, int dright);

/** Java Mat.locateROI; writes x,y (offset) then width,height. */
void cvk_mat_locate_roi(const cvk_mat_t *mat, int out[4]);

/** 3-element cross product. */
cvk_mat_t *cvk_mat_cross(const cvk_mat_t *a, const cvk_mat_t *b);

/**
 * Java Mat.put(row, col, double[]): writes count elements starting at
 * (row, col) running along the row; returns elements written.
 */
size_t cvk_mat_put_values(cvk_mat_t *mat, int row, int col,
                          const double *values, size_t count);

/** Java Mat.get(row, col, double[]); returns elements read into out. */
size_t cvk_mat_get_values(const cvk_mat_t *mat, int row, int col,
                          double *out, size_t count);

/* ---- core: clustering / decomposition (Core parity) ---- */

/**
 * cv::kmeans. bestLabels is an Nx1 CV_32S in/out array (zeros for fresh);
 * criteria = (type, maxCount, epsilon). Writes centers handle and the
 * compactness into out_compactness.
 */
void cvk_kmeans(const cvk_mat_t *data, int k, cvk_mat_t *best_labels,
                int crit_type, int crit_max_count, double crit_epsilon,
                int attempts, int flags, cvk_mat_t **centers,
                double *out_compactness);

void cvk_svd_decomp(const cvk_mat_t *src, cvk_mat_t **w, cvk_mat_t **u,
                    cvk_mat_t **vt, int flags);

/** dst = w*u'*vt*b solved back-substituting the decomposition. */
void cvk_svd_backsubst(const cvk_mat_t *w, const cvk_mat_t *u,
                       const cvk_mat_t *vt, const cvk_mat_t *b,
                       cvk_mat_t **dst);

/** PCA over rows of data; mean is Nx? overwritten output. */
void cvk_pca_compute(const cvk_mat_t *data, cvk_mat_t **mean,
                     cvk_mat_t **vectors, int max_components);

/** PCA keeping the given fraction of variance. */
void cvk_pca_compute_variance(const cvk_mat_t *data, cvk_mat_t **mean,
                              cvk_mat_t **vectors, double retained_variance);

void cvk_pca_project(const cvk_mat_t *data, const cvk_mat_t *mean,
                     const cvk_mat_t *vectors, cvk_mat_t **result);

void cvk_pca_backproject(const cvk_mat_t *data, const cvk_mat_t *mean,
                         const cvk_mat_t *vectors, cvk_mat_t **result);

double cvk_mahalanobis(const cvk_mat_t *v1, const cvk_mat_t *v2,
                       const cvk_mat_t *icovar);

/* ---- imgproc (Imgproc parity) ---- */

/**
 * Refines corner coordinates to sub-pixel accuracy. corners travel as a
 * single-contour wire buffer (Nx1 CV_32FC2 semantics); refined points come
 * back as float x,y pairs in the same flat layout.
 */
unsigned char *cvk_corner_sub_pix(const cvk_mat_t *image,
                                  const unsigned char *flat, size_t len,
                                  int win_w, int win_h,
                                  int zero_w, int zero_h,
                                  int crit_type, int crit_max_count,
                                  double crit_epsilon, size_t *out_len);

/** Earth Mover's Distance between two signatures (DistTypes or user+cost). */
float cvk_emd(const cvk_mat_t *signature1, const cvk_mat_t *signature2,
              int dist_type);

/**
 * cv::grabCut. mask is CV_8UC1 in/out; rect used when mode==0 (WITH_RECT);
 * models are 1x13 CV_64F buffers created empty by the caller.
 */
void cvk_grab_cut(const cvk_mat_t *img, cvk_mat_t *mask,
                  int rx, int ry, int rw, int rh,
                  cvk_mat_t *bgd_model, cvk_mat_t *fgd_model,
                  int iterations, int mode);

/* =========================================================================
 * highgui (desktop only; the Android shim build provides no-ops)
 * ========================================================================= */

/** cv::namedWindow. flags: WindowFlags (WINDOW_AUTOSIZE etc.). */
void cvk_named_window(const char *winname, int flags);

/** cv::resizeWindow. */
void cvk_resize_window(const char *winname, int width, int height);

/** cv::imshow. */
void cvk_imshow(const char *winname, const cvk_mat_t *mat);

/** cv::waitKey; returns the pressed key code or -1. */
int cvk_wait_key(int delay_ms);

/** cv::destroyWindow. */
void cvk_destroy_window(const char *winname);

/** cv::destroyAllWindows. */
void cvk_destroy_all_windows(void);

/* =========================================================================
 * Per-module declarations (opencv_kmp_<module>.h)
 *
 * Newly added API surface lives in its own module header, included here
 * after the shared types. The cinterop def and the JNI bridge compile the
 * whole set through this umbrella header.
 * ========================================================================= */
#include "opencv_kmp_coretypes.h"
#include "opencv_kmp_core.h"
#include "opencv_kmp_imgproc.h"
#include "opencv_kmp_imgproc2.h"
#include "opencv_kmp_video2.h"
#include "opencv_kmp_objdetect2.h"
#include "opencv_kmp_dnn2.h"
#include "opencv_kmp_imgcodecs.h"
#include "opencv_kmp_features.h"
#include "opencv_kmp_features2.h"
#include "opencv_kmp_geometry.h"
#include "opencv_kmp_calib.h"
#include "opencv_kmp_photo.h"
#include "opencv_kmp_video.h"
#include "opencv_kmp_videoio.h"
#include "opencv_kmp_objdetect.h"
#include "opencv_kmp_stereo.h"
#include "opencv_kmp_ptcloud.h"
#include "opencv_kmp_dnn.h"

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_H */

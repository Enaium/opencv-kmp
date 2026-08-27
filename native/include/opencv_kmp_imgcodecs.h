/*
 * cvk_ C ABI declarations for the OpenCV "imgcodecs" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 */
#ifndef OPENCV_KMP_IMGCODECS_H
#define OPENCV_KMP_IMGCODECS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Animation handle; the struct definition (cv::Animation value) lives
 * in shim_imgcodecs.cpp. */
typedef struct cvk_animation cvk_animation_t;

/**
 * Array of Mat handles produced by the multi-image readers and
 * cvk_animation_get_frames. `items` owns `count` pointers to Mat objects
 * created with `new`; each Mat is owned by the caller (wrap it in a
 * platform Mat handle). The array itself is freed with
 * cvk_mat_list_release — that does NOT free the Mats.
 */
typedef struct cvk_mat_list {
    cvk_mat_t **items;
    size_t count;
} cvk_mat_list_t;

/** Frees the pointer array and the list struct; the Mats stay owned by the caller. */
void cvk_mat_list_release(cvk_mat_list_t *list);

/* =========================================================================
 * multi-page / multi-image codecs
 * ========================================================================= */

/**
 * cv::imreadmulti: loads every page of a multi-page image (TIFF/GIF/...).
 * Returns a Mat list, or NULL on failure (see cvk_last_error). For
 * single-image formats the list holds exactly one Mat.
 */
cvk_mat_list_t *cvk_imreadmulti(const char *filename, int flags);

/** cv::imreadmulti(filename, start, count, flags): loads a page range. */
cvk_mat_list_t *cvk_imreadmulti_range(const char *filename, int start, int count, int flags);

/**
 * cv::imdecodemulti: decodes a multi-page image from an in-memory buffer
 * (a Mat wrapping the encoded bytes), reading pages [start, start+count).
 */
cvk_mat_list_t *cvk_imdecodemulti(const cvk_mat_t *buf, int flags, int start, int count);

/** cv::imwritemulti: writes `count` images into a multi-image file. */
int cvk_imwritemulti(const char *filename, const cvk_mat_t *const *mats, size_t count,
                     const int *params, size_t params_len);

/** cv::imencodemulti: encodes `count` images into a memory buffer. */
unsigned char *cvk_imencodemulti(const char *ext, const cvk_mat_t *const *mats, size_t count,
                                 const int *params, size_t params_len, size_t *out_len);

/* =========================================================================
 * Animation (cv::Animation struct wrapper)
 * ========================================================================= */

/** cv::Animation(loopCount, bgColor); loop_count is clamped to 0..0xffff. */
cvk_animation_t *cvk_animation_create(int loop_count, double bg0, double bg1, double bg2, double bg3);

/** Frees the Animation handle (exactly once). */
void cvk_animation_release(cvk_animation_t *h);

int cvk_animation_get_loop_count(const cvk_animation_t *h);
void cvk_animation_set_loop_count(cvk_animation_t *h, int loop_count);

cvk_scalar_t cvk_animation_get_bgcolor(const cvk_animation_t *h);
void cvk_animation_set_bgcolor(cvk_animation_t *h, cvk_scalar_t bg);

/** Frame durations in milliseconds as a CV_32SC1 Mat (MatOfInt wire). */
cvk_mat_t *cvk_animation_get_durations(const cvk_animation_t *h);
int cvk_animation_set_durations(cvk_animation_t *h, const cvk_mat_t *durations);

/** Frames of the animation; caller owns each returned Mat. */
cvk_mat_list_t *cvk_animation_get_frames(const cvk_animation_t *h);
int cvk_animation_set_frames(cvk_animation_t *h, const cvk_mat_t *const *mats, size_t count);

/** Still image (PNG-style fallback), or NULL when empty. */
cvk_mat_t *cvk_animation_get_still_image(const cvk_animation_t *h);
int cvk_animation_set_still_image(cvk_animation_t *h, const cvk_mat_t *img);

/* =========================================================================
 * animation codecs
 * ========================================================================= */

/** cv::imreadanimation: loads frames (start..start+count) into `h`. */
int cvk_imreadanimation(const char *filename, cvk_animation_t *h, int start, int count);

/** cv::imdecodeanimation: decodes frames from an in-memory buffer. */
int cvk_imdecodeanimation(const cvk_mat_t *buf, cvk_animation_t *h, int start, int count);

/** cv::imwriteanimation: writes the animation to a file (GIF/APNG/...). */
int cvk_imwriteanimation(const char *filename, const cvk_animation_t *h,
                         const int *params, size_t params_len);

/** cv::imencodeanimation: encodes the animation into a memory buffer. */
unsigned char *cvk_imencodeanimation(const char *ext, const cvk_animation_t *h,
                                     const int *params, size_t params_len, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_IMGCODECS_H */

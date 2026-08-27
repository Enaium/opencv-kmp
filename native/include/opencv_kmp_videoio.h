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
 * cvk_ C ABI declarations for the OpenCV "videoio" module.
 *
 * Included by opencv_kmp.h after the shared types; never include this file
 * directly. Declarations here are added by the module's implementation and
 * back both Kotlin/Native (cinterop) and the JVM (JNI).
 *
 * VideoCapture / VideoWriter are thin value wrappers over cv::VideoCapture /
 * cv::VideoWriter. Every function is noexcept: failures return NULL / 0 /
 * false and are reported through cvk_last_error().
 *
 * "delete" functions free the C++ object (called from Kotlin close());
 * "release" functions only close the stream, leaving the object reusable
 * for a later open(), mirroring the Java API.
 */
#ifndef OPENCV_KMP_VIDEOIO_H
#define OPENCV_KMP_VIDEOIO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvk_video_capture cvk_video_capture_t;
typedef struct cvk_video_writer cvk_video_writer_t;

/* =========================================================================
 * VideoCapture (cv::VideoCapture)
 * ========================================================================= */

/** Default constructor; the capture is not opened. */
cvk_video_capture_t *cvk_video_capture_create(void);

/** Creates and opens camera `index` (apiPreference is a VideoCaptureAPIs value). */
cvk_video_capture_t *cvk_video_capture_create_index(int index, int api_preference);

/** Creates and opens `filename` (file, image sequence, or URL). */
cvk_video_capture_t *cvk_video_capture_create_file(const char *filename, int api_preference);

/** Camera variant with extra `params` (CV_32SC1 Mat of propId/value pairs). */
cvk_video_capture_t *cvk_video_capture_create_index_params(int index, int api_preference,
                                                           const cvk_mat_t *params);

/** Filename variant with extra `params` (CV_32SC1 Mat of propId/value pairs). */
cvk_video_capture_t *cvk_video_capture_create_file_params(const char *filename,
                                                          int api_preference,
                                                          const cvk_mat_t *params);

/** Opens camera `index` on an existing capture; returns 0/1. */
int cvk_video_capture_open_index(cvk_video_capture_t *h, int index, int api_preference);

/** Opens `filename` on an existing capture; returns 0/1. */
int cvk_video_capture_open_file(cvk_video_capture_t *h, const char *filename,
                                int api_preference);

/** Camera variant with extra `params`; returns 0/1. */
int cvk_video_capture_open_index_params(cvk_video_capture_t *h, int index,
                                        int api_preference, const cvk_mat_t *params);

/** Filename variant with extra `params`; returns 0/1. */
int cvk_video_capture_open_file_params(cvk_video_capture_t *h, const char *filename,
                                       int api_preference, const cvk_mat_t *params);

/** Returns 1 when video capturing has been initialized. */
int cvk_video_capture_is_opened(const cvk_video_capture_t *h);

/** Closes the stream; the handle stays valid for a later open(). */
void cvk_video_capture_release(cvk_video_capture_t *h);

/** Frees the C++ object (Kotlin close()); exactly once. */
void cvk_video_capture_delete(cvk_video_capture_t *h);

/** Grabs the next frame; returns 0/1. */
int cvk_video_capture_grab(cvk_video_capture_t *h);

/** Decodes the grabbed frame; returns a new Mat (caller-owned) or NULL. */
cvk_mat_t *cvk_video_capture_retrieve(cvk_video_capture_t *h, int flag);

/**
 * grab + retrieve in one call. On success *out receives a new Mat
 * (caller-owned) and 1 is returned; on failure (end of stream, camera
 * disconnected) *out is NULL and 0 is returned.
 */
int cvk_video_capture_read(cvk_video_capture_t *h, cvk_mat_t **out);

/** Sets property `prop_id` to `value`; returns 0/1. */
int cvk_video_capture_set(cvk_video_capture_t *h, int prop_id, double value);

/** Returns the property value (CAP_PROP_UNKNOWN when unsupported/closed). */
double cvk_video_capture_get(const cvk_video_capture_t *h, int prop_id);

/** Used backend API name; NULL when the stream is not opened. */
const char *cvk_video_capture_get_backend_name(const cvk_video_capture_t *h);

/** Switches exceptions mode (methods raise instead of returning codes). */
void cvk_video_capture_set_exception_mode(cvk_video_capture_t *h, int enable);

/** Returns the exceptions mode. */
int cvk_video_capture_get_exception_mode(const cvk_video_capture_t *h);

/* =========================================================================
 * VideoWriter (cv::VideoWriter)
 * ========================================================================= */

/** Default constructor; the writer is not opened. */
cvk_video_writer_t *cvk_video_writer_create(void);

/** Creates and opens `filename`; fourcc is the 4-char code int. */
cvk_video_writer_t *cvk_video_writer_create_file(const char *filename, int fourcc, double fps,
                                                 int width, int height, int is_color);

/** Variant with an explicit backend (apiPreference). */
cvk_video_writer_t *cvk_video_writer_create_file_api(const char *filename, int api_preference,
                                                     int fourcc, double fps, int width,
                                                     int height, int is_color);

/** Variant with extra encoder `params` (CV_32SC1 Mat of propId/value pairs). */
cvk_video_writer_t *cvk_video_writer_create_file_params(const char *filename, int fourcc,
                                                        double fps, int width, int height,
                                                        const cvk_mat_t *params);

/** Variant with backend and extra `params`. */
cvk_video_writer_t *cvk_video_writer_create_file_api_params(const char *filename,
                                                            int api_preference, int fourcc,
                                                            double fps, int width, int height,
                                                            const cvk_mat_t *params);

/** Opens `filename` on an existing writer; returns 0/1. */
int cvk_video_writer_open_file(cvk_video_writer_t *h, const char *filename, int fourcc,
                               double fps, int width, int height, int is_color);

/** Backend variant; returns 0/1. */
int cvk_video_writer_open_file_api(cvk_video_writer_t *h, const char *filename,
                                   int api_preference, int fourcc, double fps, int width,
                                   int height, int is_color);

/** Params variant; returns 0/1. */
int cvk_video_writer_open_file_params(cvk_video_writer_t *h, const char *filename, int fourcc,
                                      double fps, int width, int height,
                                      const cvk_mat_t *params);

/** Backend + params variant; returns 0/1. */
int cvk_video_writer_open_file_api_params(cvk_video_writer_t *h, const char *filename,
                                          int api_preference, int fourcc, double fps,
                                          int width, int height, const cvk_mat_t *params);

/** Returns 1 when the writer has been successfully initialized. */
int cvk_video_writer_is_opened(const cvk_video_writer_t *h);

/** Closes the output file; the handle stays valid for a later open(). */
void cvk_video_writer_release(cvk_video_writer_t *h);

/** Frees the C++ object (Kotlin close()); exactly once. */
void cvk_video_writer_delete(cvk_video_writer_t *h);

/** Writes the next frame; returns 0/1. */
int cvk_video_writer_write(cvk_video_writer_t *h, const cvk_mat_t *image);

/** Sets property `prop_id`; returns 0/1. */
int cvk_video_writer_set(cvk_video_writer_t *h, int prop_id, double value);

/** Returns the property value (VIDEOWRITER_PROP_UNKNOWN when unsupported). */
double cvk_video_writer_get(const cvk_video_writer_t *h, int prop_id);

/** Used backend API name; NULL when the writer is not opened. */
const char *cvk_video_writer_get_backend_name(const cvk_video_writer_t *h);

/* =========================================================================
 * videoio_registry statics (cv::videoio_registry)
 * ========================================================================= */

/** Backend API name or "UnknownVideoAPI(xxx)". */
const char *cvk_videoio_get_backend_name(int api);

/* The backend lists are returned as 1xN CV_32SC1 Mats (MatOfInt wire format). */

cvk_mat_t *cvk_videoio_get_backends(void);
cvk_mat_t *cvk_videoio_get_camera_backends(void);
cvk_mat_t *cvk_videoio_get_stream_backends(void);
cvk_mat_t *cvk_videoio_get_stream_buffered_backends(void);
cvk_mat_t *cvk_videoio_get_writer_backends(void);

/** Returns 1 if backend `api` is available. */
int cvk_videoio_has_backend(int api);

/** Returns 1 if backend `api` is built in (0 = plugin). */
int cvk_videoio_is_backend_built_in(int api);

/* Plugin version queries: fill *version_abi / *version_api and return the
 * description string; NULL on failure (e.g. built-in or unknown backend). */

const char *cvk_get_camera_plugin_version(int api, int *version_abi, int *version_api);
const char *cvk_get_stream_plugin_version(int api, int *version_abi, int *version_api);
const char *cvk_get_stream_buffered_plugin_version(int api, int *version_abi, int *version_api);
const char *cvk_get_writer_plugin_version(int api, int *version_abi, int *version_api);

#ifdef __cplusplus
}
#endif

#endif /* OPENCV_KMP_VIDEOIO_H */

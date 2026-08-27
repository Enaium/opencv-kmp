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
 * C shim for the OpenCV ptcloud module (cvk_octree_*, cvk_odometry_*,
 * cvk_rgbd_normals_*, cvk_volume_*, cvk_pose_graph_*, cvk_ptcloud_*).
 *
 * Settings structs (OdometrySettings / VolumeSettings / TriangleRasterizeSettings)
 * are pure Kotlin value classes, so every settings-consuming factory takes the
 * fields expanded as scalar arguments; Mat-valued fields travel as cvk_mat_t
 * handles (NULL keeps the C++ default).
 */
#include "opencv_kmp.h"
#include "opencv_kmp_ptcloud.h"

#include <opencv2/ptcloud.hpp>
#include <opencv2/ptcloud/detail/pose_graph.hpp>

#include <string>

struct cvk_octree {
    cv::Ptr<cv::Octree> ptr;
};
struct cvk_odometry {
    cv::Odometry *ptr;
};
struct cvk_odometry_frame {
    cv::OdometryFrame *ptr;
};
struct cvk_rgbd_normals {
    cv::Ptr<cv::RgbdNormals> ptr;
};
struct cvk_volume {
    cv::Volume *ptr;
};
struct cvk_pose_graph {
    cv::Ptr<cv::detail::PoseGraph> ptr;
};

namespace {

thread_local std::string g_ptcloud_str;

void record_error(const char *m) { g_ptcloud_str = m ? m : "unknown error"; }

template <typename F>
auto guarded(F &&body) -> decltype(body()) {
    try {
        return body();
    } catch (const cv::Exception &e) {
        record_error(e.what());
    } catch (const std::exception &e) {
        record_error(e.what());
    } catch (...) {
        record_error("unknown native error");
    }
    return decltype(body())();
}

cv::Mat *require(cvk_mat_t *m) {
    if (!m) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<cv::Mat *>(m);
}

const cv::Mat *require_const(const cvk_mat_t *m) {
    if (!m) {
        record_error("null Mat handle");
        return nullptr;
    }
    return reinterpret_cast<const cv::Mat *>(m);
}

cv::Matx44d mat_to_matx44d(const cv::Mat &m) {
    cv::Mat md;
    m.convertTo(md, CV_64F);
    cv::Matx44d out;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out(i, j) = md.at<double>(i, j);
        }
    }
    return out;
}

cv::Matx44f mat_to_matx44f(const cv::Mat &m) {
    cv::Mat mf;
    m.convertTo(mf, CV_32F);
    cv::Matx44f out;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out(i, j) = mf.at<float>(i, j);
        }
    }
    return out;
}

}  // namespace

extern "C" {

/* =========================================================================
 * Octree
 * ========================================================================= */

cvk_octree_t *cvk_octree_create_with_depth(int max_depth, double size, double origin_x,
                                           double origin_y, double origin_z, int with_colors) {
    return guarded([&]() -> cvk_octree_t * {
        auto *h = new cvk_octree;
        h->ptr = cv::Octree::createWithDepth(
            max_depth, size,
            cv::Point3f(static_cast<float>(origin_x), static_cast<float>(origin_y),
                        static_cast<float>(origin_z)),
            with_colors != 0);
        return reinterpret_cast<cvk_octree_t *>(h);
    });
}

cvk_octree_t *cvk_octree_create_with_depth_cloud(int max_depth, const cvk_mat_t *point_cloud,
                                                 const cvk_mat_t *colors) {
    const cv::Mat *pc = require_const(point_cloud);
    if (!pc) return nullptr;
    return guarded([&]() -> cvk_octree_t * {
        auto *h = new cvk_octree;
        if (colors != nullptr) {
            const cv::Mat *c = require_const(colors);
            if (!c) {
                delete h;
                return nullptr;
            }
            h->ptr = cv::Octree::createWithDepth(max_depth, *pc, *c);
        } else {
            h->ptr = cv::Octree::createWithDepth(max_depth, *pc);
        }
        return reinterpret_cast<cvk_octree_t *>(h);
    });
}

cvk_octree_t *cvk_octree_create_with_resolution(double resolution, double size,
                                                double origin_x, double origin_y,
                                                double origin_z, int with_colors) {
    return guarded([&]() -> cvk_octree_t * {
        auto *h = new cvk_octree;
        h->ptr = cv::Octree::createWithResolution(
            resolution, size,
            cv::Point3f(static_cast<float>(origin_x), static_cast<float>(origin_y),
                        static_cast<float>(origin_z)),
            with_colors != 0);
        return reinterpret_cast<cvk_octree_t *>(h);
    });
}

cvk_octree_t *cvk_octree_create_with_resolution_cloud(double resolution,
                                                      const cvk_mat_t *point_cloud,
                                                      const cvk_mat_t *colors) {
    const cv::Mat *pc = require_const(point_cloud);
    if (!pc) return nullptr;
    return guarded([&]() -> cvk_octree_t * {
        auto *h = new cvk_octree;
        if (colors != nullptr) {
            const cv::Mat *c = require_const(colors);
            if (!c) {
                delete h;
                return nullptr;
            }
            h->ptr = cv::Octree::createWithResolution(resolution, *pc, *c);
        } else {
            h->ptr = cv::Octree::createWithResolution(resolution, *pc);
        }
        return reinterpret_cast<cvk_octree_t *>(h);
    });
}

int cvk_octree_insert_point(cvk_octree_t *h, double x, double y, double z) {
    auto *p = reinterpret_cast<cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return 0;
    }
    return guarded([&]() -> int {
        return p->ptr->insertPoint(cv::Point3f(static_cast<float>(x), static_cast<float>(y),
                                               static_cast<float>(z)))
                   ? 1
                   : 0;
    });
}

int cvk_octree_insert_point_color(cvk_octree_t *h, double x, double y, double z, double cx,
                                  double cy, double cz) {
    auto *p = reinterpret_cast<cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return 0;
    }
    return guarded([&]() -> int {
        return p->ptr->insertPoint(cv::Point3f(static_cast<float>(x), static_cast<float>(y),
                                               static_cast<float>(z)),
                                   cv::Point3f(static_cast<float>(cx), static_cast<float>(cy),
                                               static_cast<float>(cz)))
                   ? 1
                   : 0;
    });
}

int cvk_octree_is_point_in_bound(const cvk_octree_t *h, double x, double y, double z) {
    auto *p = reinterpret_cast<const cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return 0;
    }
    return guarded([&]() -> int {
        return p->ptr->isPointInBound(cv::Point3f(static_cast<float>(x), static_cast<float>(y),
                                                  static_cast<float>(z)))
                   ? 1
                   : 0;
    });
}

int cvk_octree_empty(const cvk_octree_t *h) {
    auto *p = reinterpret_cast<const cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return 1;
    }
    return guarded([&]() -> int { return p->ptr->empty() ? 1 : 0; });
}

void cvk_octree_clear(cvk_octree_t *h) {
    auto *p = reinterpret_cast<cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return;
    }
    guarded([&]() -> int {
        p->ptr->clear();
        return 0;
    });
}

int cvk_octree_delete_point(cvk_octree_t *h, double x, double y, double z) {
    auto *p = reinterpret_cast<cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return 0;
    }
    return guarded([&]() -> int {
        return p->ptr->deletePoint(cv::Point3f(static_cast<float>(x), static_cast<float>(y),
                                               static_cast<float>(z)))
                   ? 1
                   : 0;
    });
}

cvk_mat_t *cvk_octree_get_point_cloud(cvk_octree_t *h) {
    auto *p = reinterpret_cast<cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat points;
        p->ptr->getPointCloudByOctree(points);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(points));
    });
}

void cvk_octree_get_point_cloud_color(cvk_octree_t *h, cvk_mat_t **points, cvk_mat_t **colors) {
    auto *p = reinterpret_cast<cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return;
    }
    if (points) *points = nullptr;
    if (colors) *colors = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, cols;
        p->ptr->getPointCloudByOctree(pts, cols);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (colors) *colors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(cols));
        return 0;
    });
}

int cvk_octree_radius_nn_search(const cvk_octree_t *h, double qx, double qy, double qz,
                                float radius, cvk_mat_t **points, cvk_mat_t **square_dists) {
    auto *p = reinterpret_cast<const cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return 0;
    }
    if (points) *points = nullptr;
    if (square_dists) *square_dists = nullptr;
    return guarded([&]() -> int {
        cv::Mat pts, dists;
        int count = p->ptr->radiusNNSearch(
            cv::Point3f(static_cast<float>(qx), static_cast<float>(qy), static_cast<float>(qz)),
            radius, pts, dists);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (square_dists) *square_dists = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dists));
        return count;
    });
}

int cvk_octree_radius_nn_search_color(const cvk_octree_t *h, double qx, double qy, double qz,
                                      float radius, cvk_mat_t **points, cvk_mat_t **colors,
                                      cvk_mat_t **square_dists) {
    auto *p = reinterpret_cast<const cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return 0;
    }
    if (points) *points = nullptr;
    if (colors) *colors = nullptr;
    if (square_dists) *square_dists = nullptr;
    return guarded([&]() -> int {
        cv::Mat pts, cols, dists;
        int count = p->ptr->radiusNNSearch(
            cv::Point3f(static_cast<float>(qx), static_cast<float>(qy), static_cast<float>(qz)),
            radius, pts, cols, dists);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (colors) *colors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(cols));
        if (square_dists) *square_dists = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dists));
        return count;
    });
}

void cvk_octree_knn_search(const cvk_octree_t *h, double qx, double qy, double qz, int k,
                           cvk_mat_t **points, cvk_mat_t **square_dists) {
    auto *p = reinterpret_cast<const cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return;
    }
    if (points) *points = nullptr;
    if (square_dists) *square_dists = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, dists;
        p->ptr->KNNSearch(cv::Point3f(static_cast<float>(qx), static_cast<float>(qy),
                                      static_cast<float>(qz)),
                          k, pts, dists);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (square_dists) *square_dists = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dists));
        return 0;
    });
}

void cvk_octree_knn_search_color(const cvk_octree_t *h, double qx, double qy, double qz, int k,
                                 cvk_mat_t **points, cvk_mat_t **colors,
                                 cvk_mat_t **square_dists) {
    auto *p = reinterpret_cast<const cvk_octree *>(h);
    if (!p) {
        record_error("null Octree handle");
        return;
    }
    if (points) *points = nullptr;
    if (colors) *colors = nullptr;
    if (square_dists) *square_dists = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, cols, dists;
        p->ptr->KNNSearch(cv::Point3f(static_cast<float>(qx), static_cast<float>(qy),
                                      static_cast<float>(qz)),
                          k, pts, cols, dists);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (colors) *colors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(cols));
        if (square_dists) *square_dists = reinterpret_cast<cvk_mat_t *>(new cv::Mat(dists));
        return 0;
    });
}

void cvk_octree_release(cvk_octree_t *h) { delete reinterpret_cast<cvk_octree *>(h); }

/* =========================================================================
 * Odometry / OdometryFrame
 * ========================================================================= */

cvk_odometry_t *cvk_odometry_create(void) {
    return guarded([&]() -> cvk_odometry_t * {
        auto *h = new cvk_odometry;
        h->ptr = new cv::Odometry();
        return reinterpret_cast<cvk_odometry_t *>(h);
    });
}

cvk_odometry_t *cvk_odometry_create_type(int otype) {
    return guarded([&]() -> cvk_odometry_t * {
        auto *h = new cvk_odometry;
        h->ptr = new cv::Odometry(static_cast<cv::OdometryType>(otype));
        return reinterpret_cast<cvk_odometry_t *>(h);
    });
}

cvk_odometry_t *cvk_odometry_create_settings(
    int otype, const cvk_mat_t *camera_matrix, const cvk_mat_t *iter_counts, float min_depth,
    float max_depth, float max_depth_diff, float max_points_part, int sobel_size,
    double sobel_scale, int normal_win_size, float normal_diff_threshold, int normal_method,
    float angle_threshold, float max_translation, float max_rotation,
    float min_gradient_magnitude, const cvk_mat_t *min_gradient_magnitudes, int algtype) {
    return guarded([&]() -> cvk_odometry_t * {
        cv::OdometrySettings s;
        if (camera_matrix != nullptr) {
            const cv::Mat *m = require_const(camera_matrix);
            if (!m) return nullptr;
            s.setCameraMatrix(*m);
        }
        if (iter_counts != nullptr) {
            const cv::Mat *m = require_const(iter_counts);
            if (!m) return nullptr;
            s.setIterCounts(*m);
        }
        s.setMinDepth(min_depth);
        s.setMaxDepth(max_depth);
        s.setMaxDepthDiff(max_depth_diff);
        s.setMaxPointsPart(max_points_part);
        s.setSobelSize(sobel_size);
        s.setSobelScale(sobel_scale);
        s.setNormalWinSize(normal_win_size);
        s.setNormalDiffThreshold(normal_diff_threshold);
        s.setNormalMethod(static_cast<cv::RgbdNormals::RgbdNormalsMethod>(normal_method));
        s.setAngleThreshold(angle_threshold);
        s.setMaxTranslation(max_translation);
        s.setMaxRotation(max_rotation);
        s.setMinGradientMagnitude(min_gradient_magnitude);
        if (min_gradient_magnitudes != nullptr) {
            const cv::Mat *m = require_const(min_gradient_magnitudes);
            if (!m) return nullptr;
            s.setMinGradientMagnitudes(*m);
        }
        auto *h = new cvk_odometry;
        h->ptr = new cv::Odometry(static_cast<cv::OdometryType>(otype), s,
                                  static_cast<cv::OdometryAlgoType>(algtype));
        return reinterpret_cast<cvk_odometry_t *>(h);
    });
}

void cvk_odometry_prepare_frame(const cvk_odometry_t *h, cvk_odometry_frame_t *frame) {
    auto *p = reinterpret_cast<const cvk_odometry *>(h);
    auto *f = reinterpret_cast<cvk_odometry_frame *>(frame);
    if (!p || !f) {
        record_error("null Odometry/OdometryFrame handle");
        return;
    }
    guarded([&]() -> int {
        p->ptr->prepareFrame(*f->ptr);
        return 0;
    });
}

void cvk_odometry_prepare_frames(const cvk_odometry_t *h, cvk_odometry_frame_t *src_frame,
                                 cvk_odometry_frame_t *dst_frame) {
    auto *p = reinterpret_cast<const cvk_odometry *>(h);
    auto *s = reinterpret_cast<cvk_odometry_frame *>(src_frame);
    auto *d = reinterpret_cast<cvk_odometry_frame *>(dst_frame);
    if (!p || !s || !d) {
        record_error("null Odometry/OdometryFrame handle");
        return;
    }
    guarded([&]() -> int {
        p->ptr->prepareFrames(*s->ptr, *d->ptr);
        return 0;
    });
}

int cvk_odometry_compute_frames(const cvk_odometry_t *h, const cvk_odometry_frame_t *src_frame,
                                const cvk_odometry_frame_t *dst_frame, cvk_mat_t **rt) {
    auto *p = reinterpret_cast<const cvk_odometry *>(h);
    auto *s = reinterpret_cast<const cvk_odometry_frame *>(src_frame);
    auto *d = reinterpret_cast<const cvk_odometry_frame *>(dst_frame);
    if (!p || !s || !d) {
        record_error("null Odometry/OdometryFrame handle");
        return 0;
    }
    if (rt) *rt = nullptr;
    return guarded([&]() -> int {
        cv::Mat out;
        bool ok = p->ptr->compute(*s->ptr, *d->ptr, out);
        if (rt) *rt = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
        return ok ? 1 : 0;
    });
}

int cvk_odometry_compute_depth(const cvk_odometry_t *h, const cvk_mat_t *src_depth,
                               const cvk_mat_t *dst_depth, cvk_mat_t **rt) {
    auto *p = reinterpret_cast<const cvk_odometry *>(h);
    const cv::Mat *sd = require_const(src_depth);
    const cv::Mat *dd = require_const(dst_depth);
    if (!p || !sd || !dd) return 0;
    if (rt) *rt = nullptr;
    return guarded([&]() -> int {
        cv::Mat out;
        bool ok = p->ptr->compute(*sd, *dd, out);
        if (rt) *rt = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
        return ok ? 1 : 0;
    });
}

int cvk_odometry_compute_rgbd(const cvk_odometry_t *h, const cvk_mat_t *src_depth,
                              const cvk_mat_t *src_rgb, const cvk_mat_t *dst_depth,
                              const cvk_mat_t *dst_rgb, cvk_mat_t **rt) {
    auto *p = reinterpret_cast<const cvk_odometry *>(h);
    const cv::Mat *sd = require_const(src_depth);
    const cv::Mat *sr = require_const(src_rgb);
    const cv::Mat *dd = require_const(dst_depth);
    const cv::Mat *dr = require_const(dst_rgb);
    if (!p || !sd || !sr || !dd || !dr) return 0;
    if (rt) *rt = nullptr;
    return guarded([&]() -> int {
        cv::Mat out;
        bool ok = p->ptr->compute(*sd, *sr, *dd, *dr, out);
        if (rt) *rt = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
        return ok ? 1 : 0;
    });
}

cvk_rgbd_normals_t *cvk_odometry_get_normals_computer(const cvk_odometry_t *h) {
    auto *p = reinterpret_cast<const cvk_odometry *>(h);
    if (!p) {
        record_error("null Odometry handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_rgbd_normals_t * {
        cv::Ptr<cv::RgbdNormals> normals = p->ptr->getNormalsComputer();
        if (!normals) return nullptr;
        auto *hh = new cvk_rgbd_normals;
        hh->ptr = normals;
        return reinterpret_cast<cvk_rgbd_normals_t *>(hh);
    });
}

void cvk_odometry_release(cvk_odometry_t *h) {
    auto *p = reinterpret_cast<cvk_odometry *>(h);
    delete p->ptr;
    delete p;
}

cvk_odometry_frame_t *cvk_odometry_frame_create(const cvk_mat_t *depth, const cvk_mat_t *image,
                                                const cvk_mat_t *mask, const cvk_mat_t *normals) {
    const cv::Mat *d = depth != nullptr ? require_const(depth) : nullptr;
    const cv::Mat *i = image != nullptr ? require_const(image) : nullptr;
    const cv::Mat *m = mask != nullptr ? require_const(mask) : nullptr;
    const cv::Mat *n = normals != nullptr ? require_const(normals) : nullptr;
    if ((depth != nullptr && !d) || (image != nullptr && !i) || (mask != nullptr && !m) ||
        (normals != nullptr && !n)) {
        return nullptr;
    }
    return guarded([&]() -> cvk_odometry_frame_t * {
        auto *h = new cvk_odometry_frame;
        h->ptr = new cv::OdometryFrame(d ? *d : cv::Mat(), i ? *i : cv::Mat(), m ? *m : cv::Mat(),
                                       n ? *n : cv::Mat());
        return reinterpret_cast<cvk_odometry_frame_t *>(h);
    });
}

#define CVK_ODOM_FRAME_GETTER(name, cpp_get)                                   \
    cvk_mat_t *cvk_odometry_frame_get_##name(const cvk_odometry_frame_t *h) {  \
        auto *p = reinterpret_cast<const cvk_odometry_frame *>(h);             \
        if (!p) {                                                              \
            record_error("null OdometryFrame handle");                         \
            return nullptr;                                                    \
        }                                                                      \
        return guarded([&]() -> cvk_mat_t * {                                  \
            cv::Mat out;                                                       \
            p->ptr->cpp_get(out);                                              \
            return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));            \
        });                                                                    \
    }

CVK_ODOM_FRAME_GETTER(image, getImage)
CVK_ODOM_FRAME_GETTER(gray_image, getGrayImage)
CVK_ODOM_FRAME_GETTER(depth, getDepth)
CVK_ODOM_FRAME_GETTER(processed_depth, getProcessedDepth)
CVK_ODOM_FRAME_GETTER(mask, getMask)
CVK_ODOM_FRAME_GETTER(normals, getNormals)

#undef CVK_ODOM_FRAME_GETTER

int cvk_odometry_frame_get_pyramid_levels(const cvk_odometry_frame_t *h) {
    auto *p = reinterpret_cast<const cvk_odometry_frame *>(h);
    if (!p) {
        record_error("null OdometryFrame handle");
        return 0;
    }
    return guarded([&]() -> int { return p->ptr->getPyramidLevels(); });
}

cvk_mat_t *cvk_odometry_frame_get_pyramid_at(const cvk_odometry_frame_t *h, int pyr_type,
                                             unsigned long long level) {
    auto *p = reinterpret_cast<const cvk_odometry_frame *>(h);
    if (!p) {
        record_error("null OdometryFrame handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out;
        p->ptr->getPyramidAt(out, static_cast<cv::OdometryFramePyramidType>(pyr_type),
                             static_cast<size_t>(level));
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

void cvk_odometry_frame_release(cvk_odometry_frame_t *h) {
    auto *p = reinterpret_cast<cvk_odometry_frame *>(h);
    delete p->ptr;
    delete p;
}

/* =========================================================================
 * RgbdNormals
 * ========================================================================= */

cvk_rgbd_normals_t *cvk_rgbd_normals_create(int rows, int cols, int depth,
                                            const cvk_mat_t *k, int window_size,
                                            float diff_threshold, int method) {
    const cv::Mat *kk = k != nullptr ? require_const(k) : nullptr;
    if (k != nullptr && !kk) return nullptr;
    return guarded([&]() -> cvk_rgbd_normals_t * {
        auto *h = new cvk_rgbd_normals;
        h->ptr = cv::RgbdNormals::create(
            rows, cols, depth, kk ? *kk : cv::Mat(), window_size, diff_threshold,
            static_cast<cv::RgbdNormals::RgbdNormalsMethod>(method));
        return reinterpret_cast<cvk_rgbd_normals_t *>(h);
    });
}

cvk_mat_t *cvk_rgbd_normals_apply(const cvk_rgbd_normals_t *h, const cvk_mat_t *points) {
    auto *p = reinterpret_cast<const cvk_rgbd_normals *>(h);
    const cv::Mat *pts = require_const(points);
    if (!p || !pts) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat normals;
        p->ptr->apply(*pts, normals);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(normals));
    });
}

void cvk_rgbd_normals_cache(const cvk_rgbd_normals_t *h) {
    auto *p = reinterpret_cast<const cvk_rgbd_normals *>(h);
    if (!p) {
        record_error("null RgbdNormals handle");
        return;
    }
    guarded([&]() -> int {
        p->ptr->cache();
        return 0;
    });
}

#define CVK_RGBD_GET_SET(name, cpp_get, cpp_set)                              \
    int cvk_rgbd_normals_get_##name(const cvk_rgbd_normals_t *h) {            \
        auto *p = reinterpret_cast<const cvk_rgbd_normals *>(h);              \
        if (!p) {                                                             \
            record_error("null RgbdNormals handle");                          \
            return 0;                                                         \
        }                                                                     \
        return guarded([&]() -> int { return p->ptr->cpp_get(); });           \
    }                                                                         \
    void cvk_rgbd_normals_set_##name(cvk_rgbd_normals_t *h, int v) {          \
        auto *p = reinterpret_cast<cvk_rgbd_normals *>(h);                    \
        if (!p) {                                                             \
            record_error("null RgbdNormals handle");                          \
            return;                                                           \
        }                                                                     \
        guarded([&]() -> int { p->ptr->cpp_set(v); return 0; });              \
    }

CVK_RGBD_GET_SET(rows, getRows, setRows)
CVK_RGBD_GET_SET(cols, getCols, setCols)
CVK_RGBD_GET_SET(window_size, getWindowSize, setWindowSize)

#undef CVK_RGBD_GET_SET

int cvk_rgbd_normals_get_depth(const cvk_rgbd_normals_t *h) {
    auto *p = reinterpret_cast<const cvk_rgbd_normals *>(h);
    if (!p) {
        record_error("null RgbdNormals handle");
        return 0;
    }
    return guarded([&]() -> int { return p->ptr->getDepth(); });
}

cvk_mat_t *cvk_rgbd_normals_get_k(const cvk_rgbd_normals_t *h) {
    auto *p = reinterpret_cast<const cvk_rgbd_normals *>(h);
    if (!p) {
        record_error("null RgbdNormals handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out;
        p->ptr->getK(out);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

void cvk_rgbd_normals_set_k(cvk_rgbd_normals_t *h, const cvk_mat_t *k) {
    auto *p = reinterpret_cast<cvk_rgbd_normals *>(h);
    const cv::Mat *kk = require_const(k);
    if (!p || !kk) return;
    guarded([&]() -> int {
        p->ptr->setK(*kk);
        return 0;
    });
}

int cvk_rgbd_normals_get_method(const cvk_rgbd_normals_t *h) {
    auto *p = reinterpret_cast<const cvk_rgbd_normals *>(h);
    if (!p) {
        record_error("null RgbdNormals handle");
        return 0;
    }
    return guarded([&]() -> int { return static_cast<int>(p->ptr->getMethod()); });
}

void cvk_rgbd_normals_release(cvk_rgbd_normals_t *h) {
    delete reinterpret_cast<cvk_rgbd_normals *>(h);
}

/* =========================================================================
 * Volume
 * ========================================================================= */

cvk_volume_t *cvk_volume_create(int vtype) {
    return guarded([&]() -> cvk_volume_t * {
        auto *h = new cvk_volume;
        h->ptr = new cv::Volume(static_cast<cv::VolumeType>(vtype));
        return reinterpret_cast<cvk_volume_t *>(h);
    });
}

cvk_volume_t *cvk_volume_create_settings(
    int vtype, int integrate_width, int integrate_height, int raycast_width,
    int raycast_height, float depth_factor, float voxel_size, float tsdf_truncate_distance,
    float max_depth, int max_weight, float raycast_step_factor, const cvk_mat_t *volume_pose,
    const cvk_mat_t *volume_resolution, const cvk_mat_t *camera_integrate_intrinsics,
    const cvk_mat_t *camera_raycast_intrinsics) {
    return guarded([&]() -> cvk_volume_t * {
        cv::VolumeSettings s(static_cast<cv::VolumeType>(vtype));
        s.setIntegrateWidth(integrate_width);
        s.setIntegrateHeight(integrate_height);
        s.setRaycastWidth(raycast_width);
        s.setRaycastHeight(raycast_height);
        s.setDepthFactor(depth_factor);
        s.setVoxelSize(voxel_size);
        s.setTsdfTruncateDistance(tsdf_truncate_distance);
        s.setMaxDepth(max_depth);
        s.setMaxWeight(max_weight);
        s.setRaycastStepFactor(raycast_step_factor);
        if (volume_pose != nullptr) {
            const cv::Mat *m = require_const(volume_pose);
            if (!m) return nullptr;
            s.setVolumePose(*m);
        }
        if (volume_resolution != nullptr) {
            const cv::Mat *m = require_const(volume_resolution);
            if (!m) return nullptr;
            s.setVolumeResolution(*m);
        }
        if (camera_integrate_intrinsics != nullptr) {
            const cv::Mat *m = require_const(camera_integrate_intrinsics);
            if (!m) return nullptr;
            s.setCameraIntegrateIntrinsics(*m);
        }
        if (camera_raycast_intrinsics != nullptr) {
            const cv::Mat *m = require_const(camera_raycast_intrinsics);
            if (!m) return nullptr;
            s.setCameraRaycastIntrinsics(*m);
        }
        auto *h = new cvk_volume;
        h->ptr = new cv::Volume(static_cast<cv::VolumeType>(vtype), s);
        return reinterpret_cast<cvk_volume_t *>(h);
    });
}

void cvk_volume_integrate_frame(cvk_volume_t *h, const cvk_odometry_frame_t *frame,
                                const cvk_mat_t *pose) {
    auto *p = reinterpret_cast<cvk_volume *>(h);
    auto *f = reinterpret_cast<const cvk_odometry_frame *>(frame);
    const cv::Mat *pp = require_const(pose);
    if (!p || !f || !pp) return;
    guarded([&]() -> int {
        p->ptr->integrate(*f->ptr, *pp);
        return 0;
    });
}

void cvk_volume_integrate(cvk_volume_t *h, const cvk_mat_t *depth, const cvk_mat_t *pose) {
    auto *p = reinterpret_cast<cvk_volume *>(h);
    const cv::Mat *d = require_const(depth);
    const cv::Mat *pp = require_const(pose);
    if (!p || !d || !pp) return;
    guarded([&]() -> int {
        p->ptr->integrate(*d, *pp);
        return 0;
    });
}

void cvk_volume_integrate_color(cvk_volume_t *h, const cvk_mat_t *depth,
                                const cvk_mat_t *image, const cvk_mat_t *pose) {
    auto *p = reinterpret_cast<cvk_volume *>(h);
    const cv::Mat *d = require_const(depth);
    const cv::Mat *i = require_const(image);
    const cv::Mat *pp = require_const(pose);
    if (!p || !d || !i || !pp) return;
    guarded([&]() -> int {
        p->ptr->integrate(*d, *i, *pp);
        return 0;
    });
}

void cvk_volume_raycast(const cvk_volume_t *h, const cvk_mat_t *camera_pose,
                        cvk_mat_t **points, cvk_mat_t **normals) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    const cv::Mat *cp = require_const(camera_pose);
    if (!p || !cp) return;
    if (points) *points = nullptr;
    if (normals) *normals = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, nrm;
        p->ptr->raycast(*cp, pts, nrm);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(nrm));
        return 0;
    });
}

void cvk_volume_raycast_color(const cvk_volume_t *h, const cvk_mat_t *camera_pose,
                              cvk_mat_t **points, cvk_mat_t **normals, cvk_mat_t **colors) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    const cv::Mat *cp = require_const(camera_pose);
    if (!p || !cp) return;
    if (points) *points = nullptr;
    if (normals) *normals = nullptr;
    if (colors) *colors = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, nrm, cols;
        p->ptr->raycast(*cp, pts, nrm, cols);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(nrm));
        if (colors) *colors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(cols));
        return 0;
    });
}

void cvk_volume_raycast_ex(const cvk_volume_t *h, const cvk_mat_t *camera_pose, int height,
                           int width, const cvk_mat_t *k, cvk_mat_t **points,
                           cvk_mat_t **normals) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    const cv::Mat *cp = require_const(camera_pose);
    const cv::Mat *kk = require_const(k);
    if (!p || !cp || !kk) return;
    if (points) *points = nullptr;
    if (normals) *normals = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, nrm;
        p->ptr->raycast(*cp, height, width, *kk, pts, nrm);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(nrm));
        return 0;
    });
}

void cvk_volume_raycast_ex_color(const cvk_volume_t *h, const cvk_mat_t *camera_pose,
                                 int height, int width, const cvk_mat_t *k, cvk_mat_t **points,
                                 cvk_mat_t **normals, cvk_mat_t **colors) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    const cv::Mat *cp = require_const(camera_pose);
    const cv::Mat *kk = require_const(k);
    if (!p || !cp || !kk) return;
    if (points) *points = nullptr;
    if (normals) *normals = nullptr;
    if (colors) *colors = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, nrm, cols;
        p->ptr->raycast(*cp, height, width, *kk, pts, nrm, cols);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(nrm));
        if (colors) *colors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(cols));
        return 0;
    });
}

cvk_mat_t *cvk_volume_fetch_normals(const cvk_volume_t *h, const cvk_mat_t *points) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    const cv::Mat *pts = require_const(points);
    if (!p || !pts) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat nrm;
        p->ptr->fetchNormals(*pts, nrm);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(nrm));
    });
}

void cvk_volume_fetch_points_normals(const cvk_volume_t *h, cvk_mat_t **points,
                                     cvk_mat_t **normals) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return;
    }
    if (points) *points = nullptr;
    if (normals) *normals = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, nrm;
        p->ptr->fetchPointsNormals(pts, nrm);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(nrm));
        return 0;
    });
}

void cvk_volume_fetch_points_normals_colors(const cvk_volume_t *h, cvk_mat_t **points,
                                            cvk_mat_t **normals, cvk_mat_t **colors) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return;
    }
    if (points) *points = nullptr;
    if (normals) *normals = nullptr;
    if (colors) *colors = nullptr;
    guarded([&]() -> int {
        cv::Mat pts, nrm, cols;
        p->ptr->fetchPointsNormalsColors(pts, nrm, cols);
        if (points) *points = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pts));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(nrm));
        if (colors) *colors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(cols));
        return 0;
    });
}

void cvk_volume_reset(cvk_volume_t *h) {
    auto *p = reinterpret_cast<cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return;
    }
    guarded([&]() -> int {
        p->ptr->reset();
        return 0;
    });
}

int cvk_volume_get_visible_blocks(const cvk_volume_t *h) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return 0;
    }
    return guarded([&]() -> int { return p->ptr->getVisibleBlocks(); });
}

unsigned long long cvk_volume_get_total_volume_units(const cvk_volume_t *h) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return 0;
    }
    return guarded([&]() -> unsigned long long {
        return static_cast<unsigned long long>(p->ptr->getTotalVolumeUnits());
    });
}

cvk_mat_t *cvk_volume_get_bounding_box(const cvk_volume_t *h, int precision) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat bb;
        p->ptr->getBoundingBox(bb, precision);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(bb));
    });
}

void cvk_volume_set_enable_growth(cvk_volume_t *h, int v) {
    auto *p = reinterpret_cast<cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return;
    }
    guarded([&]() -> int {
        p->ptr->setEnableGrowth(v != 0);
        return 0;
    });
}

int cvk_volume_get_enable_growth(const cvk_volume_t *h) {
    auto *p = reinterpret_cast<const cvk_volume *>(h);
    if (!p) {
        record_error("null Volume handle");
        return 0;
    }
    return guarded([&]() -> int { return p->ptr->getEnableGrowth() ? 1 : 0; });
}

void cvk_volume_release(cvk_volume_t *h) {
    auto *p = reinterpret_cast<cvk_volume *>(h);
    delete p->ptr;
    delete p;
}

/* =========================================================================
 * PoseGraph
 * ========================================================================= */

cvk_pose_graph_t *cvk_pose_graph_create(void) {
    return guarded([&]() -> cvk_pose_graph_t * {
        auto *h = new cvk_pose_graph;
        h->ptr = cv::detail::PoseGraph::create();
        return reinterpret_cast<cvk_pose_graph_t *>(h);
    });
}

void cvk_pose_graph_add_node(cvk_pose_graph_t *h, unsigned long long node_id,
                             const cvk_mat_t *pose, int fixed) {
    auto *p = reinterpret_cast<cvk_pose_graph *>(h);
    const cv::Mat *pp = require_const(pose);
    if (!p || !pp) return;
    guarded([&]() -> int {
        p->ptr->addNode(static_cast<size_t>(node_id), cv::Affine3d(mat_to_matx44d(*pp)),
                        fixed != 0);
        return 0;
    });
}

void cvk_pose_graph_add_edge(cvk_pose_graph_t *h, unsigned long long source,
                             unsigned long long target, const cvk_mat_t *transformation,
                             const cvk_mat_t *information) {
    auto *p = reinterpret_cast<cvk_pose_graph *>(h);
    const cv::Mat *t = require_const(transformation);
    if (!p || !t) return;
    guarded([&]() -> int {
        cv::Matx66f info = cv::Matx66f::eye();
        if (information != nullptr) {
            const cv::Mat *i = require_const(information);
            if (!i) return 0;
            cv::Mat mi;
            i->convertTo(mi, CV_32F);
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 6; ++c) {
                    info(r, c) = mi.at<float>(r, c);
                }
            }
        }
        p->ptr->addEdge(static_cast<size_t>(source), static_cast<size_t>(target),
                        cv::Affine3f(mat_to_matx44f(*t)), info);
        return 0;
    });
}

int cvk_pose_graph_optimize(cvk_pose_graph_t *h) {
    auto *p = reinterpret_cast<cvk_pose_graph *>(h);
    if (!p) {
        record_error("null PoseGraph handle");
        return -1;
    }
    return guarded([&]() -> int {
        cv::LevMarq::Report report = p->ptr->optimize();
        return report.found ? report.iters : -1;
    });
}

cvk_mat_t *cvk_pose_graph_get_pose(const cvk_pose_graph_t *h, unsigned long long node_id) {
    auto *p = reinterpret_cast<const cvk_pose_graph *>(h);
    if (!p) {
        record_error("null PoseGraph handle");
        return nullptr;
    }
    return guarded([&]() -> cvk_mat_t * {
        cv::Affine3d pose = p->ptr->getNodePose(static_cast<size_t>(node_id));
        cv::Matx44d m = pose.matrix;
        auto *out = new cv::Mat(4, 4, CV_64F);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out->at<double>(i, j) = m(i, j);
            }
        }
        return reinterpret_cast<cvk_mat_t *>(out);
    });
}

void cvk_pose_graph_release(cvk_pose_graph_t *h) {
    delete reinterpret_cast<cvk_pose_graph *>(h);
}

/* =========================================================================
 * Ptcloud free functions
 * ========================================================================= */

void cvk_ptcloud_load_point_cloud(const char *filename, cvk_mat_t **vertices,
                                  cvk_mat_t **normals, cvk_mat_t **rgb) {
    if (!filename) {
        record_error("null filename");
        return;
    }
    if (vertices) *vertices = nullptr;
    if (normals) *normals = nullptr;
    if (rgb) *rgb = nullptr;
    guarded([&]() -> int {
        cv::Mat v, n, c;
        cv::loadPointCloud(filename, v, n, c);
        if (vertices) *vertices = reinterpret_cast<cvk_mat_t *>(new cv::Mat(v));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(n));
        if (rgb) *rgb = reinterpret_cast<cvk_mat_t *>(new cv::Mat(c));
        return 0;
    });
}

void cvk_ptcloud_save_point_cloud(const char *filename, const cvk_mat_t *vertices,
                                  const cvk_mat_t *normals, const cvk_mat_t *rgb) {
    if (!filename) {
        record_error("null filename");
        return;
    }
    const cv::Mat *v = require_const(vertices);
    if (!v) return;
    const cv::Mat *n = normals != nullptr ? require_const(normals) : nullptr;
    const cv::Mat *c = rgb != nullptr ? require_const(rgb) : nullptr;
    if ((normals != nullptr && !n) || (rgb != nullptr && !c)) return;
    guarded([&]() -> int {
        cv::savePointCloud(filename, *v, n ? *n : cv::Mat(), c ? *c : cv::Mat());
        return 0;
    });
}

void cvk_ptcloud_load_mesh(const char *filename, cvk_mat_t **vertices, cvk_mat_t ***indices,
                           int *indices_count, cvk_mat_t **normals, cvk_mat_t **colors,
                           cvk_mat_t **tex_coords) {
    if (!filename) {
        record_error("null filename");
        return;
    }
    if (vertices) *vertices = nullptr;
    if (indices) *indices = nullptr;
    if (indices_count) *indices_count = 0;
    if (normals) *normals = nullptr;
    if (colors) *colors = nullptr;
    if (tex_coords) *tex_coords = nullptr;
    guarded([&]() -> int {
        std::vector<cv::Mat> faces;
        cv::Mat v, n, c, tc;
        cv::loadMesh(filename, v, faces, n, c, tc);
        if (vertices) *vertices = reinterpret_cast<cvk_mat_t *>(new cv::Mat(v));
        if (normals) *normals = reinterpret_cast<cvk_mat_t *>(new cv::Mat(n));
        if (colors) *colors = reinterpret_cast<cvk_mat_t *>(new cv::Mat(c));
        if (tex_coords) *tex_coords = reinterpret_cast<cvk_mat_t *>(new cv::Mat(tc));
        if (indices) {
            auto **arr = new cvk_mat_t *[faces.size()];
            for (size_t i = 0; i < faces.size(); ++i) {
                arr[i] = reinterpret_cast<cvk_mat_t *>(new cv::Mat(faces[i]));
            }
            *indices = arr;
        }
        if (indices_count) *indices_count = static_cast<int>(faces.size());
        return 0;
    });
}

void cvk_ptcloud_save_mesh(const char *filename, const cvk_mat_t *vertices,
                           const cvk_mat_t *const *indices, int indices_count,
                           const cvk_mat_t *normals, const cvk_mat_t *colors,
                           const cvk_mat_t *tex_coords) {
    if (!filename) {
        record_error("null filename");
        return;
    }
    const cv::Mat *v = require_const(vertices);
    if (!v) return;
    const cv::Mat *n = normals != nullptr ? require_const(normals) : nullptr;
    const cv::Mat *c = colors != nullptr ? require_const(colors) : nullptr;
    const cv::Mat *tc = tex_coords != nullptr ? require_const(tex_coords) : nullptr;
    if ((normals != nullptr && !n) || (colors != nullptr && !c) ||
        (tex_coords != nullptr && !tc)) {
        return;
    }
    guarded([&]() -> int {
        std::vector<cv::Mat> faces;
        if (indices != nullptr) {
            faces.reserve(static_cast<size_t>(indices_count));
            for (int i = 0; i < indices_count; ++i) {
                const cv::Mat *f = require_const(indices[i]);
                if (!f) return 0;
                faces.push_back(*f);
            }
        }
        cv::saveMesh(filename, *v, faces, n ? *n : cv::Mat(), c ? *c : cv::Mat(),
                     tc ? *tc : cv::Mat());
        return 0;
    });
}

void cvk_free_mat_array(cvk_mat_t **arr) { delete[] arr; }

void cvk_ptcloud_triangle_rasterize(const cvk_mat_t *vertices, const cvk_mat_t *indices,
                                    const cvk_mat_t *colors, cvk_mat_t *color_buf,
                                    cvk_mat_t *depth_buf, const cvk_mat_t *world2cam,
                                    double fov_y, double z_near, double z_far, int shading_type,
                                    int culling_mode, int gl_compatible_mode) {
    const cv::Mat *v = require_const(vertices);
    const cv::Mat *i = require_const(indices);
    const cv::Mat *c = require_const(colors);
    cv::Mat *cb = require(color_buf);
    cv::Mat *db = require(depth_buf);
    const cv::Mat *w = require_const(world2cam);
    if (!v || !i || !c || !cb || !db || !w) return;
    guarded([&]() -> int {
        cv::TriangleRasterizeSettings settings;
        settings.shadingType = static_cast<cv::TriangleShadingType>(shading_type);
        settings.cullingMode = static_cast<cv::TriangleCullingMode>(culling_mode);
        settings.glCompatibleMode =
            static_cast<cv::TriangleGlCompatibleMode>(gl_compatible_mode);
        cv::triangleRasterize(*v, *i, *c, *cb, *db, *w, fov_y, z_near, z_far, settings);
        return 0;
    });
}

void cvk_ptcloud_triangle_rasterize_depth(const cvk_mat_t *vertices, const cvk_mat_t *indices,
                                          cvk_mat_t *depth_buf, const cvk_mat_t *world2cam,
                                          double fov_y, double z_near, double z_far,
                                          int shading_type, int culling_mode,
                                          int gl_compatible_mode) {
    const cv::Mat *v = require_const(vertices);
    const cv::Mat *i = require_const(indices);
    cv::Mat *db = require(depth_buf);
    const cv::Mat *w = require_const(world2cam);
    if (!v || !i || !db || !w) return;
    guarded([&]() -> int {
        cv::TriangleRasterizeSettings settings;
        settings.shadingType = static_cast<cv::TriangleShadingType>(shading_type);
        settings.cullingMode = static_cast<cv::TriangleCullingMode>(culling_mode);
        settings.glCompatibleMode =
            static_cast<cv::TriangleGlCompatibleMode>(gl_compatible_mode);
        cv::triangleRasterizeDepth(*v, *i, *db, *w, fov_y, z_near, z_far, settings);
        return 0;
    });
}

void cvk_ptcloud_triangle_rasterize_color(const cvk_mat_t *vertices, const cvk_mat_t *indices,
                                          const cvk_mat_t *colors, cvk_mat_t *color_buf,
                                          const cvk_mat_t *world2cam, double fov_y,
                                          double z_near, double z_far, int shading_type,
                                          int culling_mode, int gl_compatible_mode) {
    const cv::Mat *v = require_const(vertices);
    const cv::Mat *i = require_const(indices);
    const cv::Mat *c = require_const(colors);
    cv::Mat *cb = require(color_buf);
    const cv::Mat *w = require_const(world2cam);
    if (!v || !i || !c || !cb || !w) return;
    guarded([&]() -> int {
        cv::TriangleRasterizeSettings settings;
        settings.shadingType = static_cast<cv::TriangleShadingType>(shading_type);
        settings.cullingMode = static_cast<cv::TriangleCullingMode>(culling_mode);
        settings.glCompatibleMode =
            static_cast<cv::TriangleGlCompatibleMode>(gl_compatible_mode);
        cv::triangleRasterizeColor(*v, *i, *c, *cb, *w, fov_y, z_near, z_far, settings);
        return 0;
    });
}

int cvk_ptcloud_register_depth(const cvk_mat_t *unregistered_camera_matrix,
                               const cvk_mat_t *registered_camera_matrix,
                               const cvk_mat_t *registered_dist_coeffs, const cvk_mat_t *rt,
                               const cvk_mat_t *unregistered_depth, int output_width,
                               int output_height, int depth_dilation, cvk_mat_t **registered_depth) {
    const cv::Mat *ucm = require_const(unregistered_camera_matrix);
    const cv::Mat *rcm = require_const(registered_camera_matrix);
    const cv::Mat *rdc = require_const(registered_dist_coeffs);
    const cv::Mat *rr = require_const(rt);
    const cv::Mat *ud = require_const(unregistered_depth);
    if (!ucm || !rcm || !rdc || !rr || !ud) return 0;
    if (registered_depth) *registered_depth = nullptr;
    return guarded([&]() -> int {
        cv::Mat out;
        cv::registerDepth(*ucm, *rcm, *rdc, *rr, *ud, cv::Size(output_width, output_height),
                          out, depth_dilation != 0);
        if (registered_depth) *registered_depth = reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
        return 1;
    });
}

cvk_mat_t *cvk_ptcloud_depth_to_3d_sparse(const cvk_mat_t *depth, const cvk_mat_t *in_k,
                                          const cvk_mat_t *in_points) {
    const cv::Mat *d = require_const(depth);
    const cv::Mat *k = require_const(in_k);
    const cv::Mat *p = require_const(in_points);
    if (!d || !k || !p) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out;
        cv::depthTo3dSparse(*d, *k, *p, out);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

cvk_mat_t *cvk_ptcloud_depth_to_3d(const cvk_mat_t *depth, const cvk_mat_t *k,
                                   const cvk_mat_t *mask) {
    const cv::Mat *d = require_const(depth);
    const cv::Mat *kk = require_const(k);
    if (!d || !kk) return nullptr;
    const cv::Mat *m = mask != nullptr ? require_const(mask) : nullptr;
    if (mask != nullptr && !m) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out;
        cv::depthTo3d(*d, *kk, out, m ? *m : cv::Mat());
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

cvk_mat_t *cvk_ptcloud_rescale_depth(const cvk_mat_t *in, int type, double depth_factor) {
    const cv::Mat *m = require_const(in);
    if (!m) return nullptr;
    return guarded([&]() -> cvk_mat_t * {
        cv::Mat out;
        cv::rescaleDepth(*m, type, out, depth_factor);
        return reinterpret_cast<cvk_mat_t *>(new cv::Mat(out));
    });
}

void cvk_ptcloud_warp_frame(const cvk_mat_t *depth, const cvk_mat_t *image,
                            const cvk_mat_t *mask, const cvk_mat_t *rt,
                            const cvk_mat_t *camera_matrix, cvk_mat_t **warped_depth,
                            cvk_mat_t **warped_image, cvk_mat_t **warped_mask) {
    const cv::Mat *d = require_const(depth);
    const cv::Mat *i = image != nullptr ? require_const(image) : nullptr;
    const cv::Mat *m = mask != nullptr ? require_const(mask) : nullptr;
    const cv::Mat *rr = require_const(rt);
    const cv::Mat *cm = require_const(camera_matrix);
    if (!d || !rr || !cm) return;
    if ((image != nullptr && !i) || (mask != nullptr && !m)) return;
    if (warped_depth) *warped_depth = nullptr;
    if (warped_image) *warped_image = nullptr;
    if (warped_mask) *warped_mask = nullptr;
    guarded([&]() -> int {
        cv::Mat wd, wi, wm;
        cv::warpFrame(*d, i ? *i : cv::Mat(), m ? *m : cv::Mat(), *rr, *cm, wd, wi, wm);
        if (warped_depth) *warped_depth = reinterpret_cast<cvk_mat_t *>(new cv::Mat(wd));
        if (warped_image) *warped_image = reinterpret_cast<cvk_mat_t *>(new cv::Mat(wi));
        if (warped_mask) *warped_mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(wm));
        return 0;
    });
}

void cvk_ptcloud_find_planes(const cvk_mat_t *points3d, const cvk_mat_t *normals,
                             cvk_mat_t **mask, cvk_mat_t **plane_coefficients, int block_size,
                             int min_size, double threshold, double sensor_error_a,
                             double sensor_error_b, double sensor_error_c, int method) {
    const cv::Mat *p3 = require_const(points3d);
    const cv::Mat *n = normals != nullptr ? require_const(normals) : nullptr;
    if (!p3) return;
    if (normals != nullptr && !n) return;
    if (mask) *mask = nullptr;
    if (plane_coefficients) *plane_coefficients = nullptr;
    guarded([&]() -> int {
        cv::Mat m, pc;
        cv::findPlanes(*p3, n ? *n : cv::Mat(), m, pc, block_size, min_size, threshold,
                       sensor_error_a, sensor_error_b, sensor_error_c,
                       static_cast<cv::RgbdPlaneMethod>(method));
        if (mask) *mask = reinterpret_cast<cvk_mat_t *>(new cv::Mat(m));
        if (plane_coefficients)
            *plane_coefficients = reinterpret_cast<cvk_mat_t *>(new cv::Mat(pc));
        return 0;
    });
}

} /* extern "C" */

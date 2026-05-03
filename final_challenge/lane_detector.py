#!/usr/bin/env python3
"""Hough-line lane detector + lookahead-point publisher.

Pipeline (single image callback):

    BGR frame
        │
        ▼  HSV mask of "white-ish" pixels
        │  (broad V floor, narrow S ceiling)
        ▼
    Polygon ROI (drop sky and the car's hood)
        │
        ▼  Canny edges
        ▼  Probabilistic Hough segments
        ▼  Cluster co-linear segments, fit one line per cluster
        ▼
    Project each fit to ground plane (homography)
        │
        ▼  Reject implausible angles
        │  Pick innermost left + innermost right boundary
        ▼
    Intersect the two boundary lines in image space
        │
        ▼  Walk the angle bisector from the intersection
        │  toward the car a fixed pixel distance
        ▼
    Project that pixel back to ground plane → publish as Point32
    on /lookahead_point

Topics
------
Subscribes:
    /zed/zed_node/rgb/image_rect_color/compressed   (sensor_msgs/CompressedImage)
Publishes:
    /lookahead_point                                 (geometry_msgs/Point32)
    /lane_debug_img                                  (sensor_msgs/Image)
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Point32
from sensor_msgs.msg import CompressedImage, Image

from final_challenge.homography_transformer import (
    build_homography, transform_uv_to_xy,
)


# ───────────────────────────── tunables ──────────────────────────────────
HSV_LOW         = np.array([0,   0, 200], dtype=np.uint8)
HSV_HIGH        = np.array([180, 50, 255], dtype=np.uint8)

ROI_TOP_FRAC    = 0.40   # crop everything above this fraction of image height

CANNY_LOW       = 50
CANNY_HIGH      = 150

HOUGH_RHO       = 1
HOUGH_THETA     = math.pi / 180.0
HOUGH_THRESHOLD = 50
HOUGH_MIN_LEN   = 100
HOUGH_MAX_GAP   = 10

CLUSTER_DIST_PX = 100.0
CLUSTER_ANG_DEG = 10.0

GROUND_ANGLE_FLOOR_DEG = -15.0
GROUND_ANGLE_CEIL_DEG  =  60.0

LOOKAHEAD_BISECTOR_PX  = 50.0  # walk this many pixels from intersection
LOOKAHEAD_FALLBACK_X   = 1.5   # m forward, used when the geometry breaks


# ───────────────────────────── geometry helpers ──────────────────────────
def _segment_to_segment_min_distance(seg_a: np.ndarray,
                                     seg_b: np.ndarray) -> float:
    """Return min Euclidean distance between two 2-D segments
    seg = [x1, y1, x2, y2]."""
    p1 = seg_a[:2]; p2 = seg_a[2:]
    p3 = seg_b[:2]; p4 = seg_b[2:]
    return min(
        _point_to_segment(p1, p3, p4),
        _point_to_segment(p2, p3, p4),
        _point_to_segment(p3, p1, p2),
        _point_to_segment(p4, p1, p2),
    )


def _point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 <= 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab)) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return float(np.linalg.norm(p - (a + t * ab)))


def _line_intersection_2d(seg_a: Sequence[float],
                          seg_b: Sequence[float]
                          ) -> Optional[Tuple[float, float]]:
    """Intersect two infinite lines defined by their endpoints.  None when
    parallel/coincident."""
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b
    A1 = y2 - y1; B1 = x1 - x2; C1 = x2 * y1 - x1 * y2
    A2 = y4 - y3; B2 = x3 - x4; C2 = x4 * y3 - x3 * y4
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-9:
        return None
    return ((B1 * C2 - B2 * C1) / det,
            (C1 * A2 - C2 * A1) / det)


def _angle_bisector_step(apex: Tuple[float, float],
                         arm_a: Tuple[float, float],
                         arm_b: Tuple[float, float],
                         step_px: float) -> Tuple[float, float]:
    """From apex, take step_px along the angle bisector of the two arms."""
    ap = np.asarray(apex, dtype=float)
    va = np.asarray(arm_a, dtype=float) - ap
    vb = np.asarray(arm_b, dtype=float) - ap
    na = float(np.linalg.norm(va)); nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return float(ap[0]), float(ap[1])
    bisect = va / na + vb / nb
    norm = float(np.linalg.norm(bisect))
    if norm < 1e-9:
        return float(ap[0]), float(ap[1])
    bisect /= norm
    end = ap + step_px * bisect
    return float(end[0]), float(end[1])


# ───────────────────────────── line clustering ───────────────────────────
def _cluster_and_regress(segments: np.ndarray,
                         dist_thresh_px: float,
                         angle_thresh_deg: float
                         ) -> List[List[int]]:
    """Greedy clustering of co-linear-ish Hough segments.  Returns a list
    of clusters, each a list of indices into `segments`."""
    n = segments.shape[0]
    if n == 0:
        return []
    angles = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1],
                                   segments[:, 2] - segments[:, 0]))
    angles = np.mod(angles + 180.0, 180.0)

    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        bucket = [i]
        for j in range(i + 1, n):
            if visited[j]:
                continue
            d_ang = abs(angles[i] - angles[j])
            d_ang = min(d_ang, 180.0 - d_ang)
            if d_ang > angle_thresh_deg:
                continue
            if _segment_to_segment_min_distance(segments[i], segments[j]) > dist_thresh_px:
                continue
            visited[j] = True
            bucket.append(j)
        clusters.append(bucket)
    return clusters


def _regress_cluster(segments: np.ndarray, idxs: List[int]) -> List[int]:
    """Linear regression y = m·x + c through all endpoints in the cluster.
    Returns [x_min, y(x_min), x_max, y(x_max)] as ints."""
    members = segments[idxs]
    pts = np.vstack([members[:, [0, 1]], members[:, [2, 3]]]).astype(float)
    xs, ys = pts[:, 0], pts[:, 1]
    if xs.shape[0] < 2 or float(np.ptp(xs)) < 1e-3:
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.mean()), float(ys.mean())
    else:
        m, c = np.polyfit(xs, ys, 1)
        x_min = float(xs.min()); x_max = float(xs.max())
        y_min = m * x_min + c;   y_max = m * x_max + c
    return [int(round(x_min)), int(round(y_min)),
            int(round(x_max)), int(round(y_max))]


# ───────────────────────────── boundary picking ──────────────────────────
def _pick_left_right(regression_segments: List[List[int]],
                     project_uv_to_xy
                     ) -> Tuple[Optional[List[int]],
                                Optional[List[int]],
                                Optional[Tuple[Tuple[float, float],
                                               Tuple[float, float]]],
                                Optional[Tuple[Tuple[float, float],
                                               Tuple[float, float]]]]:
    """For each regression segment, project both endpoints to the ground
    plane.  Filter out lines whose ground-plane heading falls outside the
    plausible band, then pick:
      - left  boundary = ground line with the *smallest* |y|, y > 0
      - right boundary = ground line with the *smallest* |y|, y < 0
    Returns (left_pixel_segment, right_pixel_segment,
             left_ground_segment, right_ground_segment)."""
    if not regression_segments:
        return None, None, None, None

    annotated = []  # (ground_segment, pixel_segment, near_y)
    for pix in regression_segments:
        x1, y1, x2, y2 = pix
        g1 = project_uv_to_xy(x1, y1)
        g2 = project_uv_to_xy(x2, y2)
        # Order by ground-plane x (near → far) so "near_y" is well defined.
        if g2[0] < g1[0]:
            g1, g2 = g2, g1
            pix_ordered = [x2, y2, x1, y1]
        else:
            pix_ordered = pix
        heading_deg = math.degrees(math.atan2(g2[1] - g1[1], g2[0] - g1[0]))
        if not (GROUND_ANGLE_FLOOR_DEG < heading_deg < GROUND_ANGLE_CEIL_DEG):
            continue
        annotated.append(((g1, g2), pix_ordered, g1[1]))

    if not annotated:
        return None, None, None, None

    left_pool  = [a for a in annotated if a[2] > 0]
    right_pool = [a for a in annotated if a[2] <= 0]

    left  = min(left_pool,  key=lambda a: abs(a[2])) if left_pool  else None
    right = min(right_pool, key=lambda a: abs(a[2])) if right_pool else None

    return (left[1]  if left  else None,
            right[1] if right else None,
            left[0]  if left  else None,
            right[0] if right else None)


# ───────────────────────────── main detector node ────────────────────────
class WhiteLineHunter(Node):
    """Subscribe to the ZED stream, run the Hough pipeline, and publish a
    single car-frame lookahead point per camera frame."""

    def __init__(self) -> None:
        super().__init__("white_line_hunter")

        self._H = build_homography()
        self._H_inv = np.linalg.inv(self._H)
        self._bridge = CvBridge()

        self.declare_parameter("camera_topic",
                               "/zed/zed_node/rgb/image_rect_color/compressed")
        self.declare_parameter("lookahead_topic",  "/lookahead_point")
        self.declare_parameter("debug_image_topic", "/lane_debug_img")

        cam_topic   = str(self.get_parameter("camera_topic").value)
        look_topic  = str(self.get_parameter("lookahead_topic").value)
        debug_topic = str(self.get_parameter("debug_image_topic").value)

        latest_only = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.image_sub = self.create_subscription(
            CompressedImage, cam_topic, self._on_image, latest_only,
        )
        self.lookahead_pub = self.create_publisher(Point32, look_topic, 10)
        self.debug_pub     = self.create_publisher(Image,   debug_topic, 10)

        # Per-frame snapshots so external visualisers can pull the most
        # recent geometry without re-running the pipeline.
        self.last_left_pixel:  Optional[List[int]] = None
        self.last_right_pixel: Optional[List[int]] = None
        self.last_left_ground:  Optional[Tuple[Tuple[float, float],
                                              Tuple[float, float]]] = None
        self.last_right_ground: Optional[Tuple[Tuple[float, float],
                                               Tuple[float, float]]] = None
        self.last_lookahead_px: Optional[Tuple[float, float]] = None
        self.last_lookahead_xy: Optional[Tuple[float, float]] = None
        self.last_clusters: List[List[int]] = []

        self.get_logger().info("WhiteLineHunter up.")

    # ── pipeline ─────────────────────────────────────────────────────────
    def _on_image(self, msg: CompressedImage) -> None:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn(
                "imdecode returned None — dropping frame",
                throttle_duration_sec=2.0,
            )
            return

        regression_segments = self._extract_regression_segments(frame)
        self.last_clusters = list(regression_segments)

        left_px, right_px, left_g, right_g = _pick_left_right(
            regression_segments,
            lambda u, v: transform_uv_to_xy(self._H, u, v),
        )
        self.last_left_pixel = left_px
        self.last_right_pixel = right_px
        self.last_left_ground = left_g
        self.last_right_ground = right_g

        target_x, target_y, target_uv = self._derive_lookahead(
            left_px, right_px,
        )
        self.last_lookahead_px = target_uv
        self.last_lookahead_xy = (target_x, target_y)

        self._publish_lookahead(target_x, target_y)
        self._publish_debug(frame, regression_segments,
                            left_px, right_px, target_uv)

    # ── stage 1: HSV mask + ROI + Canny + Hough → clustered regressions ──
    def _extract_regression_segments(self, frame: np.ndarray) -> List[List[int]]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

        # Trapezoid ROI: clip sky + a small band off the bottom corners so
        # the car hood reflection doesn't leak in as bright pixels.
        roi_mask = np.zeros_like(white_mask)
        top = int(h * ROI_TOP_FRAC)
        poly = np.array([[
            (0,                     h),
            (w,                     h),
            (int(w * 0.95), top),
            (int(w * 0.05), top),
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, poly, 255)
        masked = cv2.bitwise_and(white_mask, roi_mask)

        edges = cv2.Canny(masked, CANNY_LOW, CANNY_HIGH)
        raw = cv2.HoughLinesP(
            edges,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LEN,
            maxLineGap=HOUGH_MAX_GAP,
        )
        if raw is None or len(raw) == 0:
            return []

        segs = raw[:, 0, :].astype(np.int32)
        clusters = _cluster_and_regress(segs, CLUSTER_DIST_PX, CLUSTER_ANG_DEG)
        return [_regress_cluster(segs, c) for c in clusters]

    # ── stage 2: lookahead point from the two boundary lines ────────────
    def _derive_lookahead(self, left_pix, right_pix
                          ) -> Tuple[float, float, Optional[Tuple[float, float]]]:
        """Compute (x_car, y_car, uv) of the lookahead.  Falls through to
        a forward-axis fallback when geometry is unavailable."""
        if left_pix is None or right_pix is None:
            return 0.0, 0.0, None

        apex = _line_intersection_2d(left_pix, right_pix)
        if apex is None:
            return 0.0, 0.0, None

        # Walk the bisector toward the car (away from the apex) using the
        # *near* endpoints of the two boundary lines as the angle arms.
        u_target, v_target = _angle_bisector_step(
            apex,
            (right_pix[0], right_pix[1]),
            (left_pix[0],  left_pix[1]),
            LOOKAHEAD_BISECTOR_PX,
        )
        x, y = transform_uv_to_xy(self._H, u_target, v_target)
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0, 0.0, None
        return float(x), float(y), (float(u_target), float(v_target))

    # ── publishers ───────────────────────────────────────────────────────
    def _publish_lookahead(self, x_car: float, y_car: float) -> None:
        m = Point32()
        m.x = float(x_car)
        m.y = float(y_car)
        m.z = 0.0
        self.lookahead_pub.publish(m)

    def _publish_debug(self, frame, regression_segments, left_pix, right_pix,
                       target_uv) -> None:
        if self.debug_pub is None:
            return
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        cv2.line(canvas, (0, int(h * ROI_TOP_FRAC)),
                 (w, int(h * ROI_TOP_FRAC)), (110, 110, 110), 1)

        for seg in regression_segments:
            x1, y1, x2, y2 = seg
            cv2.line(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)

        if left_pix is not None:
            cv2.line(canvas,
                     (left_pix[0], left_pix[1]),
                     (left_pix[2], left_pix[3]),
                     (255, 120, 0), 3, cv2.LINE_AA)
        if right_pix is not None:
            cv2.line(canvas,
                     (right_pix[0], right_pix[1]),
                     (right_pix[2], right_pix[3]),
                     (40, 40, 255), 3, cv2.LINE_AA)

        if target_uv is not None:
            u, v = int(round(target_uv[0])), int(round(target_uv[1]))
            cv2.drawMarker(canvas, (u, v), (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 3)
            cv2.circle(canvas, (u, v), 9, (0, 255, 255), 2)

        self.debug_pub.publish(self._bridge.cv2_to_imgmsg(canvas, "bgr8"))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WhiteLineHunter()
    pool = MultiThreadedExecutor()
    pool.add_node(node)
    try:
        pool.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

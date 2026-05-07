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
    Intersect each ground-frame boundary with x = lookahead_ground_x_m
        │
        ▼  Average the two intersection y's → ground midpoint
        ▼
    Publish (lookahead_ground_x_m, y_mid) as Point32 on /lookahead_point

The `lookahead_ground_x_m` ROS parameter is owned by the launch file's
`slow` arg (see launch/lane_follow_deploy.launch.xml): 1.0 m on the
fast / non-slow tuning path, 0.6 m on the frozen slow path that mirrors
commit cff690f's known-working configuration.

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
from typing import List, Optional, Tuple

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

# ── lookahead aim-point distance ─────────────────────────────────────────
# Forward distance (metres, car frame) at which we sample the midpoint
# between the left and right ground-frame boundary lines. Exposed as a
# ROS parameter (`lookahead_ground_x_m`) so the launch file can flip it
# between the two known-good operating points without code edits:
#   • non-slow / fast path (default, current tuning):   1.0 m
#   • slow path           (frozen cff690f reference):   0.6 m
# Larger values look further ahead → smoother but laggier steering;
# smaller values are more reactive but noisier. The default below is the
# value the node uses when no parameter is supplied (e.g. when launched
# standalone outside lane_follow_deploy.launch.xml).
LOOKAHEAD_GROUND_X_M_DEFAULT = 1.0


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

        # Forward aim-point distance (m). Owned by the launch file's `slow`
        # toggle: 1.0 m on the non-slow / fast path, 0.6 m on the frozen
        # slow path that mirrors commit cff690f. See LOOKAHEAD_GROUND_X_M_DEFAULT
        # above for the rationale on each value.
        self.declare_parameter("lookahead_ground_x_m",
                               LOOKAHEAD_GROUND_X_M_DEFAULT)

        cam_topic   = str(self.get_parameter("camera_topic").value)
        look_topic  = str(self.get_parameter("lookahead_topic").value)
        debug_topic = str(self.get_parameter("debug_image_topic").value)
        # Cached at startup — the parameter is treated as static; flipping
        # `slow` requires relaunching the stack, which is intentional since
        # the slow path is a fall-back configuration, not a runtime mode.
        self._lookahead_ground_x_m = float(
            self.get_parameter("lookahead_ground_x_m").value
        )

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
        """Compute (x_car, y_car, uv) of the lookahead as the ground-frame
        midpoint of the two boundaries at x = self._lookahead_ground_x_m.

        Each boundary segment is projected to the ground plane, then we
        intersect each ground line with x = L_x and average the two y's.
        This is invariant to perspective foreshortening and avoids the
        vanishing-point sensitivity of the angle-bisector approach.
        """
        if left_pix is None or right_pix is None:
            return 0.0, 0.0, None

        # Pull from the cached parameter (set by the launch file's `slow`
        # arg) instead of a module constant, so swapping operating points
        # is purely a launch-file concern.
        L_x = self._lookahead_ground_x_m

        def _y_at_Lx(seg):
            x1, y1, x2, y2 = seg
            (gx1, gy1) = transform_uv_to_xy(self._H, x1, y1)
            (gx2, gy2) = transform_uv_to_xy(self._H, x2, y2)
            dx = gx2 - gx1
            if abs(dx) < 1e-6:
                return None
            t = (L_x - gx1) / dx
            return gy1 + t * (gy2 - gy1)

        y_left  = _y_at_Lx(left_pix)
        y_right = _y_at_Lx(right_pix)
        if y_left is None or y_right is None:
            return 0.0, 0.0, None
        if not (math.isfinite(y_left) and math.isfinite(y_right)):
            return 0.0, 0.0, None

        y_mid = 0.5 * (y_left + y_right)
        if not math.isfinite(y_mid):
            return 0.0, 0.0, None

        # Map (L_x, y_mid) back to image space for debug viz.
        target_uv = self._ground_to_uv(L_x, y_mid)
        return float(L_x), float(y_mid), target_uv

    def _ground_to_uv(self, x_m: float, y_m: float
                      ) -> Optional[Tuple[float, float]]:
        try:
            p = self._H_inv @ np.array([[x_m], [y_m], [1.0]])
            w = float(p[2, 0])
            if abs(w) < 1e-9:
                return None
            return float(p[0, 0] / w), float(p[1, 0] / w)
        except Exception:
            return None

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

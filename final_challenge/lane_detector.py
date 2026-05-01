#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from final_challenge.white_line_detection import (
    detect_lane_lines_hough,
    detect_white_lines,
    ROI_TOP_FRAC,
)
from final_challenge.homography_transformer import build_homography, transform_uv_to_xy

# Tunable
N_SAMPLES = 15   # points sampled per detected Hough line
MIN_VOTES = 5    # discard Hough clusters with fewer accumulator votes (filters noisy near-zero artifacts)
MAX_ABS_Y = 1.5  # discard candidate lines whose median car-frame |y| exceeds this (outliers from
                 # near-horizon homography projection)
MIN_FORWARD_SPAN = 0.3  # path must span >= this many metres forward in car frame
MAX_X_CAR        = 4.0  # cap forward distance per sample. The homography horizon is at
                        # v ≈ 138 px and ROI top is at v ≈ 150 px, so samples near the
                        # ROI top project to wildly large x_car. 4 m is well past the
                        # 1.2 m lookahead and bounds away from the projective horizon.
MAX_DY_OVER_DX = 0.4    # path must be roughly forward-aligned. Real boundaries on
                        # straights have |slope| < 0.15; mild curves up to ~0.35;
                        # transverse markers register at |slope| ≥ 0.48. 0.4 keeps
                        # gentle curves and rejects transverse markers cleanly.


def _path_is_plausible_boundary(pts):
    """A car-frame path is a plausible lane boundary iff it spans enough
    forward distance and is roughly forward-aligned (small lateral slope).
    Filters out cross-track markers and short fragments. Uses a regression
    slope rather than first-last delta so blob spines (jittery per-row means)
    don't get rejected on noise."""
    if len(pts) < 2:
        return False
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if xs[-1] - xs[0] < MIN_FORWARD_SPAN:
        return False
    if np.ptp(xs) < 1e-6:
        return False
    slope = np.polyfit(xs, ys, 1)[0]
    if abs(slope) > MAX_DY_OVER_DX:
        return False
    return True


def _make_path(points, stamp, frame_id="base_link"):
    """Build a nav_msgs/Path from a list of (x, y) tuples."""
    msg = Path()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    for x, y in points:
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


class LaneDetector(Node):
    """
    Detects left and right lane boundaries and publishes them as
    nav_msgs/Path in base_link frame for BoundaryPurePursuit.

    Detection strategy
    ------------------
    Primary: detect_lane_lines_hough() — stable on straight sections.
    Fallback: detect_white_lines() blob detector — handles curves where
    Hough produces fewer than 2 valid lines.

    Left/right classification is done in the car frame (after homography),
    not by image-space slope sign, so it remains correct on curves.

    Subscriptions
    -------------
    /zed/zed_node/rgb/image_rect_color  (sensor_msgs/Image)

    Publications
    ------------
    /left_lane_line   (nav_msgs/Path)  — left boundary in base_link frame
    /right_lane_line  (nav_msgs/Path)  — right boundary in base_link frame
    /lane_debug_img   (sensor_msgs/Image) — annotated frame for rqt_image_view
    """

    def __init__(self):
        super().__init__("lane_detector")

        self.H = build_homography()
        self.bridge = CvBridge()

        self.left_pub  = self.create_publisher(Path,  "/left_lane_line",  10)
        self.right_pub = self.create_publisher(Path,  "/right_lane_line", 10)
        self.debug_pub = self.create_publisher(Image, "/lane_debug_img",  10)

        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback,
            5,
        )

        self.get_logger().info("LaneDetector initialised.")

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w  = frame.shape[:2]
        roi_top = int(ROI_TOP_FRAC * h)
        stamp = msg.header.stamp

        left_pts, right_pts, debug_lines = self._detect_boundaries(frame, h, w, roi_top)

        self.left_pub.publish(_make_path(left_pts,  stamp))
        self.right_pub.publish(_make_path(right_pts, stamp))
        self._publish_debug(frame, debug_lines, roi_top)

    # ------------------------------------------------------------------
    # Boundary detection — Hough primary, blob fallback
    # ------------------------------------------------------------------
    def _detect_boundaries(self, frame, h, w, roi_top):
        """
        Return (left_pts, right_pts, debug_lines).

        left_pts / right_pts : list of (x_car, y_car) sorted near-to-far.
        debug_lines          : list of Hough line dicts for the debug image
                               (may be empty when using blob fallback).
        """
        # ── Primary: Hough detector ───────────────────────────────────
        hough_lines = detect_lane_lines_hough(frame)

        # Convert every sufficiently-voted Hough line to car-frame points
        # and record its median lateral position.
        car_lines = []
        for ll in hough_lines:
            if ll['votes'] < MIN_VOTES:
                continue
            pts = self._hough_line_to_car_pts(ll, h, w, roi_top)
            if not _path_is_plausible_boundary(pts):
                continue
            median_y = float(np.median([p[1] for p in pts]))
            if abs(median_y) > MAX_ABS_Y:
                continue
            car_lines.append((median_y, ll, pts))

        if len(car_lines) >= 2:
            left_pts, right_pts = self._split_left_right(car_lines)
            debug_lines = [ll for _, ll, _ in car_lines]
            return left_pts, right_pts, debug_lines

        # ── Fallback: blob detector (better on curves) ────────────────
        # Blob fallback is a normal mode of operation — many frames yield <2
        # Hough lines after plausibility filtering. Only warn if BOTH fail.
        blob_spines = detect_white_lines(frame)

        car_lines = []
        for spine in blob_spines:
            pts = self._spine_to_car_pts(spine)
            if not _path_is_plausible_boundary(pts):
                continue
            median_y = float(np.median([p[1] for p in pts]))
            if abs(median_y) > MAX_ABS_Y:
                continue
            car_lines.append((median_y, None, pts))

        if len(car_lines) >= 2:
            left_pts, right_pts = self._split_left_right(car_lines)
            return left_pts, right_pts, []

        # Both detectors found <2 plausible lines — this is a real degraded frame.
        self.get_logger().warn(
            "Both Hough and blob produced <2 valid lines.",
            throttle_duration_sec=1.0,
        )

        # Only one or zero lines — classify the single line by the sign of its
        # median y so it lands on its true side. Blindly assigning to "left"
        # would steer the car toward the wall when the detection is on the right.
        left_pts, right_pts = [], []
        if car_lines:
            median_y, _, pts = car_lines[0]
            if median_y >= 0:
                left_pts = pts
            else:
                right_pts = pts
        return left_pts, right_pts, []

    # ------------------------------------------------------------------
    # Left / right classification in car frame
    # ------------------------------------------------------------------
    @staticmethod
    def _split_left_right(car_lines):
        """
        Given a list of (median_y, ll, pts), find the pair of lane boundaries
        that bracket y=0 (the car centreline).

        Left boundary  = line with y > 0 closest to y=0.
        Right boundary = line with y < 0 closest to y=0 (least negative).

        If all detected lines are on one side, only that side is returned;
        the other is left empty. The controller handles an empty path safely
        via stale-path memory + stop_if_no_path. Labelling a same-side line
        as the opposite boundary would feed the wrong path to the controller
        and steer the car toward the wall.
        """
        sorted_lines = sorted(car_lines, key=lambda t: t[0], reverse=True)

        left_candidates  = [(m, pts) for m, _, pts in sorted_lines if m >= 0]
        right_candidates = [(m, pts) for m, _, pts in sorted_lines if m <  0]

        if left_candidates and right_candidates:
            left_pts  = left_candidates[-1][1]   # smallest positive y
            right_pts = right_candidates[0][1]   # least-negative y
        elif left_candidates:
            left_pts  = left_candidates[-1][1]
            right_pts = []
        else:
            right_pts = right_candidates[0][1]
            left_pts  = []

        return left_pts, right_pts

    # ------------------------------------------------------------------
    # Pixel → car-frame conversion helpers
    # ------------------------------------------------------------------
    def _hough_line_to_car_pts(self, ll, h, w, roi_top):
        """Sample a Hough line model into car-frame (x, y) points, near-to-far.

        Skip samples where the line exits the image (clipping x_px to [0, w-1]
        would pile points at the edge and the homography turns that L-kink into
        fake curvature). Skip samples that project past MAX_X_CAR (the projective
        horizon makes far samples meaningless).
        """
        m, b = ll['coeffs']
        pts  = []
        for y_px in np.linspace(h - 1, roi_top, N_SAMPLES):
            x_px = m * y_px + b
            if x_px < 0 or x_px > w - 1:
                continue
            x_car, y_car = transform_uv_to_xy(self.H, float(x_px), float(y_px))
            if 0.0 < x_car < MAX_X_CAR:
                pts.append((x_car, y_car))
        pts.sort(key=lambda p: p[0])
        return pts

    def _spine_to_car_pts(self, spine):
        """Convert a blob spine to car-frame points, near-to-far.

        Cap x_car at MAX_X_CAR for the same reason as Hough: rows near the
        ROI top project past the homography horizon and produce noise.
        """
        pts = []
        for u, v in spine:
            x_car, y_car = transform_uv_to_xy(self.H, float(u), float(v))
            if 0.0 < x_car < MAX_X_CAR:
                pts.append((x_car, y_car))
        pts.sort(key=lambda p: p[0])
        return pts

    # ------------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------------
    def _publish_debug(self, frame, debug_lines, roi_top):
        dbg = frame.copy()
        h, w = dbg.shape[:2]

        cv2.line(dbg, (0, roi_top), (w, roi_top), (128, 128, 128), 1)

        colors = [(0, 255, 0), (0, 255, 255), (255, 165, 0), (255, 0, 255)]
        for i, ll in enumerate(debug_lines):
            if ll is None:
                continue
            color = colors[i % len(colors)]
            p1, p2 = ll['segment']
            cv2.line(dbg, p1, p2, color, 2)
            cv2.circle(dbg, p1, 4, color, -1)
            cv2.circle(dbg, p2, 4, color, -1)

        cv2.putText(dbg, "lines=%d" % len(debug_lines),
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

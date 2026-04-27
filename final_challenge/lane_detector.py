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
MIN_VOTES = 3    # discard Hough clusters with fewer accumulator votes


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
            if len(pts) >= 2:
                median_y = float(np.median([p[1] for p in pts]))
                car_lines.append((median_y, ll, pts))

        if len(car_lines) >= 2:
            left_pts, right_pts = self._split_left_right(car_lines)
            debug_lines = [ll for _, ll, _ in car_lines]
            return left_pts, right_pts, debug_lines

        # ── Fallback: blob detector (better on curves) ────────────────
        self.get_logger().warn(
            "Hough produced <2 valid lines — falling back to blob detector.",
            throttle_duration_sec=1.0,
        )
        blob_spines = detect_white_lines(frame)

        car_lines = []
        for spine in blob_spines:
            pts = self._spine_to_car_pts(spine)
            if len(pts) >= 2:
                median_y = float(np.median([p[1] for p in pts]))
                car_lines.append((median_y, None, pts))

        if len(car_lines) >= 2:
            left_pts, right_pts = self._split_left_right(car_lines)
            return left_pts, right_pts, []

        # Only one or zero lines — return whatever we have
        all_pts = [pts for _, _, pts in car_lines]
        left_pts  = all_pts[0] if len(all_pts) > 0 else []
        right_pts = all_pts[1] if len(all_pts) > 1 else []
        return left_pts, right_pts, []

    # ------------------------------------------------------------------
    # Left / right classification in car frame
    # ------------------------------------------------------------------
    @staticmethod
    def _split_left_right(car_lines):
        """
        Given a list of (median_y, ll, pts) sorted arbitrarily, find the pair
        of lane boundaries that bracket y=0 (the car centreline).

        Left boundary  = line with y > 0 closest to y=0.
        Right boundary = line with y < 0 closest to y=0 (least negative).

        If all detected lines are on one side (e.g. car near a boundary),
        the two closest to y=0 are returned as left and right respectively.
        """
        # Sort descending by median y: leftmost first
        sorted_lines = sorted(car_lines, key=lambda t: t[0], reverse=True)

        left_candidates  = [(m, pts) for m, _, pts in sorted_lines if m >= 0]
        right_candidates = [(m, pts) for m, _, pts in sorted_lines if m <  0]

        if left_candidates and right_candidates:
            # Ideal case: lines on both sides.  Pick those closest to y=0.
            left_pts  = left_candidates[-1][1]   # smallest positive y
            right_pts = right_candidates[0][1]   # least-negative y
        elif left_candidates:
            # All lines to the left; take the two closest to the car
            left_pts  = left_candidates[-1][1]
            right_pts = left_candidates[-2][1] if len(left_candidates) >= 2 else []
        else:
            # All lines to the right; take the two closest to the car
            right_pts = right_candidates[0][1]
            left_pts  = right_candidates[1][1] if len(right_candidates) >= 2 else []

        return left_pts, right_pts

    # ------------------------------------------------------------------
    # Pixel → car-frame conversion helpers
    # ------------------------------------------------------------------
    def _hough_line_to_car_pts(self, ll, h, w, roi_top):
        """Sample a Hough line model into car-frame (x, y) points, near-to-far."""
        m, b = ll['coeffs']
        pts  = []
        for y_px in np.linspace(h - 1, roi_top, N_SAMPLES):
            x_px = float(np.clip(m * y_px + b, 0, w - 1))
            x_car, y_car = transform_uv_to_xy(self.H, x_px, y_px)
            if x_car > 0.0:
                pts.append((x_car, y_car))
        pts.sort(key=lambda p: p[0])
        return pts

    def _spine_to_car_pts(self, spine):
        """
        Convert a blob spine (list of image-space (x, y) points, bottom-to-top)
        into car-frame (x, y) points, near-to-far.
        """
        pts = []
        for u, v in spine:
            x_car, y_car = transform_uv_to_xy(self.H, float(u), float(v))
            if x_car > 0.0:
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

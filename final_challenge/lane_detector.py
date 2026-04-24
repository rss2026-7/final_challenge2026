#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from visual_servoing.computer_vision.white_line_detection import (
    detect_lane_lines_hough,
    ROI_TOP_FRAC,
)

# ── Homography calibration ────────────────────────────────────────────────────
# Map image pixels (u, v) → car-frame ground plane (x, y) in metres.
# Convention: +x = forward (direction the car faces), +y = left of the car.
#
# TODO: fill these in before running on the real robot.
# How to calibrate:
#   1. Place a marker at a known position (x_m, y_m) metres from the car.
#   2. Find its pixel location (u, v) in a live ZED frame.
#   3. Add both to the lists below.
#   Spread points across the full lane-visible range: 0.5–3 m ahead,
#   left and right of centre.  At least 4 points required.
#
# Example (replace with real measurements):
PTS_IMAGE_PLANE = [
    # [u, v],   # description
    [320, 400],  # placeholder — 1.0 m ahead, centre
    [200, 400],  # placeholder — 1.0 m ahead, left
    [440, 400],  # placeholder — 1.0 m ahead, right
    [320, 300],  # placeholder — 2.0 m ahead, centre
]
PTS_GROUND_PLANE = [
    # [x_m,  y_m],   # description
    [1.00,  0.00],   # placeholder — 1.0 m ahead, centre
    [1.00,  0.30],   # placeholder — 1.0 m ahead, left
    [1.00, -0.30],   # placeholder — 1.0 m ahead, right
    [2.00,  0.00],   # placeholder — 2.0 m ahead, centre
]
# ─────────────────────────────────────────────────────────────────────────────

METERS_PER_UNIT = 1.0  # PTS_GROUND_PLANE already in metres

# Tunable
N_SAMPLES = 15   # points sampled per detected lane line
MIN_VOTES = 3    # discard clusters with fewer accumulator votes


def _build_homography():
    np_img    = np.float32(PTS_IMAGE_PLANE)[:, np.newaxis, :]
    np_ground = np.float32(PTS_GROUND_PLANE)[:, np.newaxis, :]
    H, _ = cv2.findHomography(np_img, np_ground)
    return H


def _px_to_car(u, v, H):
    """Transform a single image pixel to car-frame (x_fwd, y_left) in metres."""
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


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
    Detects left and right lane boundaries with Hough transforms and publishes
    them as nav_msgs/Path messages consumed by BoundaryPurePursuit.

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

        self.H = _build_homography()
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

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w  = frame.shape[:2]
        roi_top = int(ROI_TOP_FRAC * h)
        stamp = msg.header.stamp

        lane_lines = detect_lane_lines_hough(frame)

        # Separate detections by sign of slope; keep best (highest votes) per side.
        best_left  = None
        best_right = None
        for ll in lane_lines:
            if ll['votes'] < MIN_VOTES:
                continue
            m, _ = ll['coeffs']
            if m < 0:
                if best_left is None or ll['votes'] > best_left['votes']:
                    best_left = ll
            else:
                if best_right is None or ll['votes'] > best_right['votes']:
                    best_right = ll

        left_pts  = self._line_to_car_pts(best_left,  h, w, roi_top)
        right_pts = self._line_to_car_pts(best_right, h, w, roi_top)

        self.left_pub.publish(_make_path(left_pts,  stamp))
        self.right_pub.publish(_make_path(right_pts, stamp))

        self._publish_debug(frame, best_left, best_right, left_pts, right_pts, roi_top)

    def _line_to_car_pts(self, ll, h, w, roi_top):
        """Sample a detected line into car-frame (x, y) points, near-to-far."""
        if ll is None:
            return []

        m, b = ll['coeffs']
        pts  = []
        for y_px in np.linspace(h - 1, roi_top, N_SAMPLES):
            x_px = float(np.clip(m * y_px + b, 0, w - 1))
            x_car, y_car = _px_to_car(x_px, y_px, self.H)
            if x_car > 0.0:
                pts.append((x_car, y_car))

        pts.sort(key=lambda p: p[0])
        return pts

    def _publish_debug(self, frame, best_left, best_right, left_pts, right_pts, roi_top):
        dbg = frame.copy()
        h, w = dbg.shape[:2]

        # ROI boundary
        cv2.line(dbg, (0, roi_top), (w, roi_top), (128, 128, 128), 1)

        # Draw detected line segments
        for ll, color in [(best_left, (0, 255, 0)), (best_right, (0, 255, 255))]:
            if ll is None:
                continue
            p1, p2 = ll['segment']
            cv2.line(dbg, p1, p2, color, 2)
            cv2.circle(dbg, p1, 4, color, -1)
            cv2.circle(dbg, p2, 4, color, -1)

        # Indicate number of lines
        n_left  = 1 if best_left  else 0
        n_right = 1 if best_right else 0
        cv2.putText(dbg, "L=%d R=%d" % (n_left, n_right),
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

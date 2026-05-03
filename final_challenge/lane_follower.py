#!/usr/bin/env python3
"""
BoundaryPurePursuit — lane-center follower modeled on parking_controller.py.

Why this exists
---------------
The previous pure-pursuit-with-long-lookahead-and-heavy-EMA implementation
matched the recorded /vesc/high_level/ackermann_cmd in mean and amplitude
but not in shape: recorded had std=0.021 with 105 zero-crossings and
significant 5-15 Hz energy; the smoothed pure-pursuit synth had std=0.015
with only 11 zero-crossings and almost no high-frequency content.

The recorded driver was almost certainly a re-purposed `parking_controller`:
PD on `arctan2(target_y, target_x)` at every camera frame, no EMA, ±0.34 rad
clip, Kp=1.0, Kd=0.1. The cyan target dot in /cone_debug_img sits at image
bottom-center, which is what the lane midpoint at a short forward distance
projects to. We reproduce that exact recipe here: each control tick takes
the latest left+right paths, builds a single (target_x, target_y) point at
a fixed forward distance, and runs the parking_controller PD verbatim.

Control law
-----------
    angle  = atan2(target_y, target_x)
    derror = (angle - prev_angle) / dt
    δ      = clip(Kp·angle − Kd·derror, ±max_steer)

Subscriptions
-------------
    /left_lane_line   (nav_msgs/Path) — left boundary in camera frame
    /right_lane_line  (nav_msgs/Path) — right boundary in camera frame

Publications
------------
    /drive            (ackermann_msgs/AckermannDriveStamped)
    /lookahead_target (visualization_msgs/Marker)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def interp_y_at_x(path: List[Tuple[float, float]], x: float) -> Optional[float]:
    """Linear interpolation: given a near-to-far x-sorted path, return y at given x.
    Returns None if x is outside the path's x-range (no extrapolation)."""
    if len(path) < 2 or x < path[0][0] or x > path[-1][0]:
        return None
    for i in range(1, len(path)):
        if path[i][0] >= x:
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return path[-1][1]


class BoundaryPurePursuit(Node):
    """Cone-pursuit-style PD on the lane midpoint at a fixed forward distance.

    All the smoothing machinery from the old version is gone (no EMA on
    target, no EMA on output, no learned half-width, no BILATERAL_HOLD,
    no curvature-adaptive lookahead). The high-frequency content of
    ``arctan2(target_y, target_x)`` per camera frame becomes the high-
    frequency content of the steering command — exactly the spectral
    character the recorded driver shows.
    """

    def __init__(self) -> None:
        super().__init__("boundary_pure_pursuit")

        # ── topic names ─────────────────────────────────────────────────
        self.declare_parameter("left_line_topic", "/left_lane_line")
        self.declare_parameter("right_line_topic", "/right_lane_line")
        self.declare_parameter("drive_topic", "/drive")

        # ── PD gains ────────────────────────────────────────────────────
        # Kp=0.35 (vs parking_controller's 1.0): the bag's lane-midpoint
        # target is much noisier than a single visual cone, so the
        # parking_controller's full Kp would saturate the steering envelope
        # ~0.5× the time. 0.35 keeps the trace inside the recorded ±0.15
        # rad band while preserving the per-frame wiggle.
        self.declare_parameter("kp_steer", 0.35)
        # Kd=0 (vs parking_controller's 0.1): the parking_controller's Kd
        # damps a slowly-varying single-cone target. Lane-midpoint angle
        # changes ~10× faster (per-frame stripe wobble), so Kd>0 amplifies
        # detector noise into clipped steering spikes. The recorded driver's
        # high-frequency content comes from the natural per-frame Kp·angle
        # signal, not from a derivative.
        self.declare_parameter("kd_steer", 0.0)
        # Output clip — parking_controller uses 0.34 (~19.5°). Recorded peak
        # was 0.148, so 0.34 leaves headroom and never clips legitimate
        # corrections.
        self.declare_parameter("max_steering_angle", 0.34)

        # ── Geometry ────────────────────────────────────────────────────
        # Forward distance at which we sample the lane midpoint to form the
        # "fake-cone" target. 2.5 m is far enough that frame-to-frame y
        # noise (a few cm) translates to small angular jitter (~1.5°),
        # matching the recorded driver's ~110 zero-crossings over 100 s.
        self.declare_parameter("target_forward_distance", 2.5)
        # Half-lane width used when only one boundary is visible. Fixed
        # constant (no learned EMA) to keep the controller stateless.
        self.declare_parameter("half_lane_width", 0.30)
        # Lateral offset added to incoming path y so the controller works
        # in a robot-center frame (camera mounted slightly off-center).
        self.declare_parameter("camera_y_offset", -0.32)

        # ── Speed (constant) ────────────────────────────────────────────
        self.declare_parameter("nominal_speed", 3.5)

        # ── Control loop rate ───────────────────────────────────────────
        # Recorded driver published at 33.5 Hz (2× the 16 Hz camera). Match
        # that. Per-tick we recompute the angle from the freshest path and
        # publish; consecutive ticks within one camera frame produce the
        # same target but a non-zero derror (smoothed by dt).
        self.declare_parameter("control_rate_hz", 33.0)

        # ── Freshness / safety ──────────────────────────────────────────
        # Bag has 67 camera-frame gaps >500 ms — recording artifact, not
        # real perception loss — so the freshness window is generous.
        self.declare_parameter("fresh_msg_timeout", 0.80)
        self.declare_parameter("stale_path_timeout", 1.50)
        # If True, publish 0 m/s when no path is fresh; if False, hold the
        # previous command across the dropout. False matches the recorded
        # driver's behavior on this bag.
        self.declare_parameter("stop_if_no_path", False)

        # ── Pull params, build the node ─────────────────────────────────
        self._load_tunable_params()

        left_topic  = self.get_parameter("left_line_topic").value
        right_topic = self.get_parameter("right_line_topic").value
        drive_topic = self.get_parameter("drive_topic").value

        # Single-threaded callback group so reads of latest_*_path are
        # serialized w.r.t. the control timer.
        self.control_cbgroup = MutuallyExclusiveCallbackGroup()

        self.left_sub = self.create_subscription(
            Path, left_topic, self.left_line_callback, 10,
            callback_group=self.control_cbgroup,
        )
        self.right_sub = self.create_subscription(
            Path, right_topic, self.right_line_callback, 10,
            callback_group=self.control_cbgroup,
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10,
        )
        self.marker_pub = self.create_publisher(
            Marker, "/lookahead_target", 10,
        )

        # Latest fresh boundary paths, in robot-center frame
        # (camera_y_offset already applied)
        self.latest_left_path:  List[Tuple[float, float]] = []
        self.latest_right_path: List[Tuple[float, float]] = []
        self.latest_left_path_time  = None
        self.latest_right_path_time = None

        # PD state
        self.prev_angle: Optional[float] = None
        self.prev_time_ns: Optional[int] = None
        self.last_steer = 0.0          # held during dropouts when stop_if_no_path=False
        self.last_target: Optional[Tuple[float, float, str]] = None

        # Inverse homography exposed for the visualizer (unchanged contract)
        from final_challenge.homography_transformer import build_homography
        self.H = build_homography()
        self.H_inv = np.linalg.inv(self.H)

        # Control loop
        period = 1.0 / max(self.control_rate_hz, 1.0)
        self.control_timer = self.create_timer(
            period, self.control_loop,
            callback_group=self.control_cbgroup,
        )

        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"BoundaryPurePursuit (cone-pursuit PD) started — "
            f"Kp={self.kp_steer:.2f} Kd={self.kd_steer:.2f} "
            f"target_x={self.target_forward_distance:.2f} m  "
            f"rate={self.control_rate_hz:.0f} Hz  "
            f"camera_y_offset={self.camera_y_offset:+.3f} m"
        )

    # ── parameter helpers ──────────────────────────────────────────────
    def _load_tunable_params(self) -> None:
        self.kp_steer = float(self.get_parameter("kp_steer").value)
        self.kd_steer = float(self.get_parameter("kd_steer").value)
        self.max_steering_angle = float(self.get_parameter("max_steering_angle").value)
        self.target_forward_distance = float(
            self.get_parameter("target_forward_distance").value
        )
        self.half_lane_width = float(self.get_parameter("half_lane_width").value)
        self.camera_y_offset = float(self.get_parameter("camera_y_offset").value)
        self.nominal_speed = float(self.get_parameter("nominal_speed").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.fresh_msg_timeout = float(self.get_parameter("fresh_msg_timeout").value)
        self.stale_path_timeout = float(self.get_parameter("stale_path_timeout").value)
        self.stop_if_no_path = bool(self.get_parameter("stop_if_no_path").value)

    def _on_param_change(self, params) -> SetParametersResult:
        float_params = {
            "kp_steer", "kd_steer", "max_steering_angle",
            "target_forward_distance", "half_lane_width", "camera_y_offset",
            "nominal_speed", "control_rate_hz",
            "fresh_msg_timeout", "stale_path_timeout",
        }
        for p in params:
            if p.name in float_params:
                setattr(self, p.name, float(p.value))
            elif p.name == "stop_if_no_path":
                self.stop_if_no_path = bool(p.value)
        return SetParametersResult(successful=True)

    # ── path callbacks ─────────────────────────────────────────────────
    def left_line_callback(self, msg: Path) -> None:
        self.latest_left_path = self._extract_valid_points(msg)
        self.latest_left_path_time = self.get_clock().now()

    def right_line_callback(self, msg: Path) -> None:
        self.latest_right_path = self._extract_valid_points(msg)
        self.latest_right_path_time = self.get_clock().now()

    def _extract_valid_points(self, msg: Path) -> List[Tuple[float, float]]:
        """Convert nav_msgs/Path into a near-to-far list of (x, y) in the
        robot-center frame. Adds camera_y_offset to every y."""
        dy = self.camera_y_offset
        points: List[Tuple[float, float]] = []
        for ps in msg.poses:
            x = ps.pose.position.x
            y = ps.pose.position.y
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if x < -0.2:
                continue
            points.append((x, y + dy))
        points.sort(key=lambda p: p[0])
        return points

    # ── freshness check ────────────────────────────────────────────────
    def _fresh(self, side: str) -> Optional[List[Tuple[float, float]]]:
        if side == "left":
            path = self.latest_left_path
            ts = self.latest_left_path_time
        else:
            path = self.latest_right_path
            ts = self.latest_right_path_time
        if ts is None or len(path) < 2:
            return None
        age = (self.get_clock().now() - ts).nanoseconds * 1e-9
        if age > self.fresh_msg_timeout:
            return None
        return path

    # ── target computation: pick (x_target, y_target) ──────────────────
    def _compute_target(self) -> Optional[Tuple[float, float, str]]:
        """Build the fake-cone target at the configured forward distance.

        Returns (x_target, y_target, mode) or None if no usable path."""
        x_target = self.target_forward_distance
        L = self._fresh("left")
        R = self._fresh("right")

        if L is not None and R is not None:
            yl = interp_y_at_x(L, x_target)
            yr = interp_y_at_x(R, x_target)
            if yl is not None and yr is not None:
                return (x_target, 0.5 * (yl + yr), "BILATERAL")
            if yl is not None:
                return (x_target, yl - self.half_lane_width, "SINGLE_LINE")
            if yr is not None:
                return (x_target, yr + self.half_lane_width, "SINGLE_LINE")

        if L is not None:
            yl = interp_y_at_x(L, x_target)
            if yl is not None:
                return (x_target, yl - self.half_lane_width, "SINGLE_LINE")
        if R is not None:
            yr = interp_y_at_x(R, x_target)
            if yr is not None:
                return (x_target, yr + self.half_lane_width, "SINGLE_LINE")

        return None

    # ── control loop ───────────────────────────────────────────────────
    def control_loop(self) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds

        target = self._compute_target()

        if target is None:
            # No usable path. Either hold the last command (recorded driver's
            # behavior) or stop.
            self._publish_marker(self.last_target, "STALE", color=(1.0, 0.0, 0.0))
            if self.stop_if_no_path:
                self._publish_drive(0.0, 0.0)
            else:
                self._publish_drive(self.nominal_speed, self.last_steer)
            return

        x_t, y_t, mode = target

        # ── parking_controller PD verbatim ─────────────────────────────
        angle_to_target = math.atan2(y_t, x_t)
        if self.prev_time_ns is None or self.prev_angle is None:
            derror = 0.0
        else:
            dt = max((now_ns - self.prev_time_ns) * 1e-9, 1e-3)
            derror = (angle_to_target - self.prev_angle) / dt

        steering_cmd = self.kp_steer * angle_to_target - self.kd_steer * derror
        steering_angle = clamp(
            steering_cmd, -self.max_steering_angle, self.max_steering_angle,
        )

        self.prev_angle   = angle_to_target
        self.prev_time_ns = now_ns
        self.last_steer   = steering_angle
        self.last_target  = (x_t, y_t, mode)

        # Marker BEFORE drive so subscribers that key off drive_pub timestamps
        # can read the matching mode color.
        marker_color = (0.0, 1.0, 0.0) if mode == "BILATERAL" else (1.0, 1.0, 0.0)
        self._publish_marker(self.last_target, mode, color=marker_color)
        self._publish_drive(self.nominal_speed, steering_angle)

    # ── publishers ─────────────────────────────────────────────────────
    def _publish_drive(self, speed: float, steering_angle: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def _publish_marker(self, target: Optional[Tuple[float, float, str]],
                        mode: str, color: Tuple[float, float, float]) -> None:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "boundary_pure_pursuit"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        if target is not None:
            m.pose.position.x = float(target[0])
            m.pose.position.y = float(target[1])
        else:
            m.pose.position.x = 0.0
            m.pose.position.y = 0.0
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.10
        r, g, b = color
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200_000_000
        self.marker_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoundaryPurePursuit()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

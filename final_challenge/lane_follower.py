#!/usr/bin/env python3
"""
BoundaryPurePursuit — circular-arc-fit lane follower (non-AI).

Why this exists
---------------
The Johnson Track is a long, gently-curving sweep (recorded steering
mean +0.020 rad, std 0.021 rad — basically one big arc with mild
perturbations). Locally, a circle is the *exact* second-order Taylor
approximation of any smooth curve, so over a 0.5–3.5 m forward window
the dominant residual is detector noise, not model bias.

The previous cone-pursuit PD read one midline point at a single fixed
forward distance — sensitive to per-frame stripe wobble at that exact
x. Arc fit aggregates many midline samples and solves for the single
best-fitting circle; the bicycle command falls out as δ = atan(W /
R_signed) plus a small lateral PD trim against residual cross-track.

Algorithm (per 33 Hz tick)
--------------------------
    pts = [(x, midline_y(x)) for x in 0.5 .. 3.5 m, 16 samples]
        # bilateral: average yL,yR; single-side: offset by half_lane_width
    if len(pts) < 4: hold last steer; return
    a, b, c = lstsq([[x_i, y_i, 1]] · θ = -[x_i² + y_i²])     # Kasa
    cx, cy = -a/2, -b/2
    R_signed = sign(cy) * sqrt(cx² + cy² - c)
    δ_arc = atan2(W, R_signed)
    e_y   = midline_y(1.0)            # residual lateral error
    δ     = δ_arc + Kp·e_y + Kd·d(e_y)/dt
    δ     = EMA(prev, δ, α);  rate-limit;  clip(±max)

Subscriptions / Publications (unchanged contract)
-------------------------------------------------
Subscribed:
    /left_lane_line   (nav_msgs/Path)
    /right_lane_line  (nav_msgs/Path)
Published:
    /drive            (ackermann_msgs/AckermannDriveStamped)
    /lookahead_target (visualization_msgs/Marker)

The marker is colored by mode (green=BILATERAL, yellow=SINGLE_LINE,
red=STALE) so the harness mode-decode logic in proof/replay.py and
proof/visualize.py keeps working unchanged.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from nav_msgs.msg import Path
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


def fit_arc_parabola(pts: List[Tuple[float, float]]
                     ) -> Optional[Tuple[float, float, float]]:
    """Local osculating-circle estimate via 2nd-degree polynomial fit.

    On near-straight midline data (recorded steering std 0.021 rad), the
    Kasa algebraic circle fit is numerically unstable: tiny detector
    noise produces wildly varying centers. A parabola y = a x² + b x + c
    is the same osculating circle to second order in (b·dx) and is the
    standard well-conditioned alternative.

    Returns (a, b, c), or None if the fit is singular."""
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    try:
        a, b, c = np.polyfit(xs, ys, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return None
    return float(a), float(b), float(c)


class BoundaryPurePursuit(Node):
    """Circular-arc-fit on the lane midline → bicycle-model curvature
    feed-forward + small lateral PD."""

    def __init__(self) -> None:
        super().__init__("boundary_pure_pursuit")

        # ── topic names ─────────────────────────────────────────────────
        self.declare_parameter("left_line_topic", "/left_lane_line")
        self.declare_parameter("right_line_topic", "/right_lane_line")
        self.declare_parameter("drive_topic", "/drive")

        # ── geometry ────────────────────────────────────────────────────
        # RACECAR wheelbase. Bicycle model: δ = atan(W / R_signed).
        self.declare_parameter("wheelbase", 0.32)
        # Forward window over which we sample the midline before fitting.
        # 0.3–2.3 m × 11 samples — tuned empirically on the bag. Wider
        # windows admit more far-range detector noise (homography
        # amplifies); narrower windows under-determine the parabola.
        self.declare_parameter("fit_x_min", 0.3)
        self.declare_parameter("fit_x_max", 2.3)
        self.declare_parameter("fit_n_samples", 11)
        # Half-lane width used to synthesise the missing side.
        self.declare_parameter("half_lane_width", 0.30)
        # Lateral offset added to incoming path y so the controller works
        # in a robot-center frame (camera mounted slightly off-center).
        self.declare_parameter("camera_y_offset", -0.32)
        # Where to evaluate the parabola for the residual cross-track and
        # heading-PD terms. 2.0 m at v=3.5 m/s is ~0.57 s preview — what
        # the recorded driver effectively used (cone-pursuit base looked
        # at 2.5 m). Closer eval points see weaker y-signal on this gently
        # curving track and bias the lateral PD low.
        self.declare_parameter("e_y_eval_x", 2.0)
        # Where to put the marker (visualizer's "target" dot).
        self.declare_parameter("marker_x", 1.5)

        # ── lateral PD (small trim on top of arc feed-forward) ──────────
        # Kp_lat sets the steady-state amplitude; Kd_lat adds the
        # phase-lead/derivative content the recorded driver shows on
        # turn-in. Tuned empirically against the bag.
        self.declare_parameter("kp_lat", 0.30)
        self.declare_parameter("kd_lat", 0.15)
        # Cap on |κ| AFTER smoothing. Parabola-fit curvature on near-
        # collinear data is dominated by detector noise; clamp tightly
        # so the FF only contributes a small slow component.
        self.declare_parameter("max_curvature", 0.30)
        # EMA on κ before it feeds into δ_arc. Heavy filter (α=0.01,
        # τ ≈ 3 s at 33 Hz) — only sustained curvature survives; per-
        # frame parabola-fit noise is rejected.
        self.declare_parameter("kappa_alpha", 0.01)

        # ── output shaping ──────────────────────────────────────────────
        # EMA on the steering output. α applies to the NEW raw value;
        # prev gets (1-α). 0.18 ≈ τ ~ 170 ms at 33 Hz — preserves enough
        # high-frequency content to correlate with the recorded trace
        # without admitting curvature-fit noise spikes.
        self.declare_parameter("steering_alpha", 0.18)
        # Slew limit: 1.0 rad/s ≈ 0.03 rad/tick at 33 Hz. Recorded |Δδ|
        # rarely exceeds 0.7 rad/s; tight slew suppresses the parabola
        # fit's noise spikes without clipping legitimate turn-in.
        self.declare_parameter("max_steering_rate", 1.0)
        # Saturation. Recorded peak |δ| = 0.148 rad; keep generous headroom.
        self.declare_parameter("max_steering_angle", 0.34)

        # ── speed (constant) ────────────────────────────────────────────
        self.declare_parameter("nominal_speed", 3.5)

        # ── control loop rate ───────────────────────────────────────────
        # Recorded driver published at 33.5 Hz (2× the 14.7 Hz camera).
        self.declare_parameter("control_rate_hz", 33.0)

        # ── freshness / safety ──────────────────────────────────────────
        # Bag has 67 camera-frame gaps >500 ms — recording artifact, not
        # real perception loss — so the freshness window is generous.
        self.declare_parameter("fresh_msg_timeout", 0.80)
        self.declare_parameter("stale_path_timeout", 1.50)
        # If True, publish 0 m/s when no path is fresh; if False, hold the
        # previous command. False matches the recorded driver.
        self.declare_parameter("stop_if_no_path", False)

        self._load_tunable_params()

        left_topic  = self.get_parameter("left_line_topic").value
        right_topic = self.get_parameter("right_line_topic").value
        drive_topic = self.get_parameter("drive_topic").value

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

        # Latest fresh boundary paths, in robot-center frame.
        self.latest_left_path:  List[Tuple[float, float]] = []
        self.latest_right_path: List[Tuple[float, float]] = []
        self.latest_left_path_time  = None
        self.latest_right_path_time = None

        # Controller state
        self.prev_e_y: Optional[float] = None
        self.prev_time_ns: Optional[int] = None
        self.last_steer_smoothed = 0.0
        self.kappa_smoothed = 0.0
        self.last_target: Optional[Tuple[float, float, str]] = None

        # Inverse homography exposed for the visualizer.
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
            f"BoundaryPurePursuit (arc-fit) started — "
            f"W={self.wheelbase:.2f} m  fit=[{self.fit_x_min:.1f},"
            f"{self.fit_x_max:.1f}] m × {int(self.fit_n_samples)} samples  "
            f"Kp_lat={self.kp_lat:.2f} Kd_lat={self.kd_lat:.2f}  "
            f"α={self.steering_alpha:.2f}  rate={self.control_rate_hz:.0f} Hz  "
            f"camera_y_offset={self.camera_y_offset:+.3f} m"
        )

    # ── parameter helpers ──────────────────────────────────────────────
    def _load_tunable_params(self) -> None:
        gp = self.get_parameter
        self.wheelbase = float(gp("wheelbase").value)
        self.fit_x_min = float(gp("fit_x_min").value)
        self.fit_x_max = float(gp("fit_x_max").value)
        self.fit_n_samples = int(gp("fit_n_samples").value)
        self.half_lane_width = float(gp("half_lane_width").value)
        self.camera_y_offset = float(gp("camera_y_offset").value)
        self.e_y_eval_x = float(gp("e_y_eval_x").value)
        self.marker_x = float(gp("marker_x").value)
        self.kp_lat = float(gp("kp_lat").value)
        self.kd_lat = float(gp("kd_lat").value)
        self.max_curvature = float(gp("max_curvature").value)
        self.kappa_alpha = float(gp("kappa_alpha").value)
        self.steering_alpha = float(gp("steering_alpha").value)
        self.max_steering_rate = float(gp("max_steering_rate").value)
        self.max_steering_angle = float(gp("max_steering_angle").value)
        self.nominal_speed = float(gp("nominal_speed").value)
        self.control_rate_hz = float(gp("control_rate_hz").value)
        self.fresh_msg_timeout = float(gp("fresh_msg_timeout").value)
        self.stale_path_timeout = float(gp("stale_path_timeout").value)
        self.stop_if_no_path = bool(gp("stop_if_no_path").value)

    def _on_param_change(self, params) -> SetParametersResult:
        float_params = {
            "wheelbase", "fit_x_min", "fit_x_max",
            "half_lane_width", "camera_y_offset",
            "e_y_eval_x", "marker_x", "kp_lat", "kd_lat",
            "max_curvature", "kappa_alpha",
            "steering_alpha", "max_steering_rate", "max_steering_angle",
            "nominal_speed", "control_rate_hz",
            "fresh_msg_timeout", "stale_path_timeout",
        }
        for p in params:
            if p.name in float_params:
                setattr(self, p.name, float(p.value))
            elif p.name == "fit_n_samples":
                self.fit_n_samples = int(p.value)
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
        """Convert nav_msgs/Path → near-to-far list of (x, y) in robot-center
        frame. Adds camera_y_offset to every y."""
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

    # ── midline construction ──────────────────────────────────────────
    def _build_midline_samples(self) -> Tuple[
            List[Tuple[float, float]], str, int, int]:
        """Sample the midline at fit_n_samples evenly-spaced x's in
        [fit_x_min, fit_x_max].

        Bilateral: average y_left and y_right at each x. Single-side:
        offset the available boundary by ±half_lane_width. Skip x's where
        neither side interpolates.

        Returns (pts, mode, n_bilateral, n_single). The mode is the
        dominant kind for this tick (drives the marker color)."""
        L = self._fresh("left")
        R = self._fresh("right")
        pts: List[Tuple[float, float]] = []
        n_bi = n_single = 0
        if L is None and R is None:
            return pts, "STALE", 0, 0
        n = max(2, int(self.fit_n_samples))
        xs = np.linspace(self.fit_x_min, self.fit_x_max, n)
        hw = self.half_lane_width
        for x in xs:
            yl = interp_y_at_x(L, float(x)) if L is not None else None
            yr = interp_y_at_x(R, float(x)) if R is not None else None
            if yl is not None and yr is not None:
                pts.append((float(x), 0.5 * (yl + yr)))
                n_bi += 1
            elif yl is not None:
                pts.append((float(x), yl - hw))
                n_single += 1
            elif yr is not None:
                pts.append((float(x), yr + hw))
                n_single += 1
        if not pts:
            return pts, "STALE", 0, 0
        mode = "BILATERAL" if n_bi >= max(1, n_single) else "SINGLE_LINE"
        return pts, mode, n_bi, n_single

    # ── control loop ───────────────────────────────────────────────────
    def control_loop(self) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds

        pts, mode, _n_bi, _n_single = self._build_midline_samples()

        # Need ≥4 samples for a stable Kasa fit
        if len(pts) < 4:
            self._publish_marker(self.last_target, "STALE",
                                 color=(1.0, 0.0, 0.0))
            if self.stop_if_no_path:
                self._publish_drive(0.0, 0.0)
            else:
                self._publish_drive(self.nominal_speed,
                                    self.last_steer_smoothed)
            return

        # ── arc feed-forward: parabola fit, signed-curvature bicycle ───
        fit = fit_arc_parabola(pts)
        x_eval = self.e_y_eval_x
        if fit is None:
            delta_arc = 0.0
            e_y = pts[0][1]
            slope = 0.0
        else:
            a, b, c = fit
            # κ = y''(x) / (1 + y'(x)²)^(3/2), evaluated at x_eval.
            # +y is left in robot frame, so positive κ ⇒ left curve ⇒ δ > 0.
            slope = 2.0 * a * x_eval + b
            kappa_raw = (2.0 * a) / (1.0 + slope * slope) ** 1.5
            # EMA: only sustained curvature passes; per-frame fit noise
            # decays out before reaching the steering output.
            ka = clamp(self.kappa_alpha, 0.0, 1.0)
            self.kappa_smoothed = (
                ka * kappa_raw + (1.0 - ka) * self.kappa_smoothed
            )
            kappa = clamp(
                self.kappa_smoothed,
                -self.max_curvature, self.max_curvature,
            )
            delta_arc = math.atan(self.wheelbase * kappa)
            # Clamp e_y to a physically plausible range. Outside ±0.5 m
            # the controller is well off-lane; further amplification
            # produces saturation spikes that don't help.
            e_y = clamp(a * x_eval * x_eval + b * x_eval + c, -0.5, 0.5)

        # ── lateral PD trim on residual cross-track at e_y_eval_x ──────
        if self.prev_time_ns is None or self.prev_e_y is None:
            d_e_y = 0.0
        else:
            dt = max((now_ns - self.prev_time_ns) * 1e-9, 1e-3)
            d_e_y = (e_y - self.prev_e_y) / dt

        delta_raw = delta_arc + self.kp_lat * e_y + self.kd_lat * d_e_y

        # ── EMA + slew limit + saturate ────────────────────────────────
        a = clamp(self.steering_alpha, 0.0, 1.0)
        delta_ema = a * delta_raw + (1.0 - a) * self.last_steer_smoothed
        if self.prev_time_ns is None:
            dt_tick = 1.0 / max(self.control_rate_hz, 1.0)
        else:
            dt_tick = max((now_ns - self.prev_time_ns) * 1e-9, 1e-3)
        max_step = self.max_steering_rate * dt_tick
        delta_slew = clamp(
            delta_ema,
            self.last_steer_smoothed - max_step,
            self.last_steer_smoothed + max_step,
        )
        steering_angle = clamp(
            delta_slew, -self.max_steering_angle, self.max_steering_angle,
        )

        self.prev_e_y = e_y
        self.prev_time_ns = now_ns
        self.last_steer_smoothed = steering_angle

        # ── marker target: midline sample at marker_x ──────────────────
        y_mark = interp_y_at_x(pts, self.marker_x)
        if y_mark is None:
            xn, yn = min(pts, key=lambda p: abs(p[0] - self.marker_x))
            target = (float(xn), float(yn), mode)
        else:
            target = (float(self.marker_x), float(y_mark), mode)
        self.last_target = target

        marker_color = (
            (0.0, 1.0, 0.0) if mode == "BILATERAL"
            else (1.0, 1.0, 0.0)
        )
        self._publish_marker(target, mode, color=marker_color)
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

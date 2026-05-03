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

import csv
import math
import os
import time
from collections import deque
from datetime import datetime
from time import monotonic
from typing import List, Optional, Tuple

import numpy as np
import rclpy


def _resolve_lane_tune_dir() -> str:
    """Resolve the per-run tuning directory shared with lane_detector.

    Coordination strategy: env var LANE_TUNE_DIR wins. Otherwise, look
    in ~/lane_tune for a run-* directory created in the last 10 s — if
    found, reuse it (the other node started slightly before us); if
    not, mint a new timestamped one. Two nodes launched together end
    up in the same folder without explicit IPC."""
    env = os.environ.get("LANE_TUNE_DIR")
    if env:
        d = os.path.expanduser(env)
        os.makedirs(d, exist_ok=True)
        return d
    base = os.path.expanduser("~/lane_tune")
    os.makedirs(base, exist_ok=True)
    now = time.time()
    recent = []
    try:
        for name in os.listdir(base):
            if not name.startswith("run-"):
                continue
            full = os.path.join(base, name)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            if os.path.isdir(full) and now - mtime < 10.0:
                recent.append((mtime, full))
    except OSError:
        pass
    if recent:
        recent.sort(reverse=True)
        return recent[0][1]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(base, f"run-{ts}")
    os.makedirs(d, exist_ok=True)
    return d
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
        # Track lane is 85 cm wide → half = 0.425 m (measured).
        self.declare_parameter("half_lane_width", 0.425)
        # Width-consistency gate. A BILATERAL pair is only accepted if
        # |y_L − y_R| is within `width_tol` of `2·half_lane_width`. On
        # the failing run-20260503-060123, BI pairs frequently bracketed
        # *different lanes* (implied L–R separation up to 2.25 m vs the
        # 0.85 m measured). On reject, the controller falls back to
        # single-side synthesis using whichever side has more raw
        # detected points at that x (proxied by left_n vs right_n at
        # the tick level). 0.20 m tolerance ≈ ±23 % of true width.
        self.declare_parameter("width_tol", 0.20)
        # Lateral offset added to incoming path y so the controller works
        # in a robot-center frame. Camera is 6.5 cm to the left of the
        # rear axle (measured) → -0.065.
        self.declare_parameter("camera_y_offset", -0.065)
        # Where to evaluate the parabola for the residual cross-track and
        # heading-PD terms. CSV showed left_xmax mean 1.90 m, right_xmax
        # mean 1.95 m — eval at 2.0 m was extrapolating past the data and
        # producing |e_y_raw| up to 1.85 m on the noisiest ticks. 1.5 m
        # sits well inside the fit window so e_y is interpolation, not
        # extrapolation.
        self.declare_parameter("e_y_eval_x", 1.5)
        # Where to put the marker (visualizer's "target" dot).
        self.declare_parameter("marker_x", 1.5)

        # ── lateral PD (small trim on top of arc feed-forward) ──────────
        # Kp_lat sets the steady-state amplitude; Kd_lat adds the
        # phase-lead/derivative content the recorded driver shows on
        # turn-in. Tuned empirically against the bag.
        self.declare_parameter("kp_lat", 0.30)
        # Kd_lat=0: tuning trace run-20260503-060123 showed the D-term
        # contributing 91.8 % of σ²(δ_raw) by amplifying perception
        # jitter (|d_e_y| up to 50 m/s — physically impossible at v=3.5).
        # Σ|δ_D|/Σ|δ_P| = 2.47×; replay with Kd=0 cuts σ(δ_final) by
        # 17 %, slew-clip rate from 22 % → 3.5 %. Re-enable only after
        # smoothing the e_y signal it differentiates.
        self.declare_parameter("kd_lat", 0.0)
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
        # 2.0 m/s: at 3.5 m/s the controller's catch-up distance was
        # 4.8× the lane half-width (structural under-damping). Halving
        # v doubles the preview budget per meter and quarters lateral
        # acceleration at the same δ. Restore to 3.5 once oscillation
        # is shown to be tame at this speed.
        self.declare_parameter("nominal_speed", 2.0)

        # ── control loop rate ───────────────────────────────────────────
        # Recorded driver published at 33.5 Hz (2× the 14.7 Hz camera).
        self.declare_parameter("control_rate_hz", 33.0)

        # ── freshness / safety ──────────────────────────────────────────
        # Bag has 67 camera-frame gaps >500 ms — recording artifact, not
        # real perception loss — so the freshness window is generous.
        self.declare_parameter("fresh_msg_timeout", 0.80)
        self.declare_parameter("stale_path_timeout", 1.50)
        # If True, publish 0 m/s when no path is fresh; if False, hold
        # the previous command. True prevents the start-of-run blind-
        # drive that put the car one lane over before perception came
        # online (see the run-20260503-050721 tuning trace).
        self.declare_parameter("stop_if_no_path", True)

        # ── tuning instrumentation ──────────────────────────────────────
        # Per-tick CSV at <run_dir>/data.csv + a throttled console line
        # every debug_print_every ticks. Run dir is shared with the
        # lane_detector frame dump (see _resolve_lane_tune_dir above).
        # Override the path with debug_log_path if you want the CSV in
        # a specific spot. Disable by setting debug_log:=false once tuned.
        self.declare_parameter("debug_log", True)
        self.declare_parameter("debug_log_path", "")
        self.declare_parameter("debug_print_every", 6)

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
        # Camera-message stamps (ns) from each side's path header. Logged
        # per CSV row so an entry maps to frames/<stamp_ns>.jpg directly.
        self.latest_left_stamp_ns: int = 0
        self.latest_right_stamp_ns: int = 0

        # Controller state
        self.prev_e_y: Optional[float] = None
        self.prev_time_ns: Optional[int] = None
        self.last_steer_smoothed = 0.0
        self.kappa_smoothed = 0.0
        self.last_target: Optional[Tuple[float, float, str]] = None
        # Latch: stay in a non-driving "WAIT_BI" state until perception
        # has produced a BILATERAL fix at least once. SINGLE_LINE is
        # blocked before then because synthesising midline by offsetting
        # one boundary by half_lane_width can land in the wrong lane
        # (this is the start-of-run lane-skip mechanism the prior trace
        # exposed). After the first BILATERAL, SINGLE_LINE is allowed
        # as a normal fallback.
        self._seen_bilateral: bool = False

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

        # ── tuning instrumentation state ───────────────────────────────
        self._tick_count = 0
        self._t0 = monotonic()
        self._prev_mode: Optional[str] = None
        # 1 s rolling window at 33 Hz for σ(e_y), σ(δ), sign-flip count.
        roll_n = max(8, int(round(self.control_rate_hz)))
        self._roll_e_y: deque = deque(maxlen=roll_n)
        self._roll_delta: deque = deque(maxlen=roll_n)
        self._csv_file = None
        self._csv_writer = None
        self._csv_header = [
            "t", "tick", "dt",
            "left_stamp_ns", "right_stamp_ns",
            "mode", "mode_changed", "n_pts", "n_bi", "n_single",
            "n_width_rejected",
            "left_n", "left_xmax", "left_age",
            "right_n", "right_xmax", "right_age",
            "fit_a", "fit_b", "fit_c", "slope",
            "kappa_raw", "kappa_smoothed", "kappa_clamped", "kappa_clipped",
            "e_y_raw", "e_y", "d_e_y",
            "delta_arc", "delta_p", "delta_d", "delta_raw",
            "delta_ema", "delta_slew", "delta_final",
            "slew_clipped", "sat_clipped",
            "speed",
        ]
        if self.debug_log:
            try:
                override = self.debug_log_path.strip()
                if override:
                    path = os.path.expanduser(override)
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                else:
                    tune_dir = _resolve_lane_tune_dir()
                    path = os.path.join(tune_dir, "data.csv")
                self._csv_file = open(path, "w", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(self._csv_header)
                self._csv_file.flush()
                self.get_logger().info(f"[TUNE] writing CSV to {path}")
            except OSError as e:
                self.get_logger().warn(
                    f"[TUNE] failed to open CSV ({e}) — continuing without it"
                )
                self._csv_file = None
                self._csv_writer = None

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
        self.width_tol = float(gp("width_tol").value)
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
        self.debug_log = bool(gp("debug_log").value)
        self.debug_log_path = str(gp("debug_log_path").value)
        self.debug_print_every = int(gp("debug_print_every").value)

    def _on_param_change(self, params) -> SetParametersResult:
        float_params = {
            "wheelbase", "fit_x_min", "fit_x_max",
            "half_lane_width", "width_tol", "camera_y_offset",
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
    @staticmethod
    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def left_line_callback(self, msg: Path) -> None:
        self.latest_left_path = self._extract_valid_points(msg)
        self.latest_left_path_time = self.get_clock().now()
        self.latest_left_stamp_ns = self._stamp_ns(msg.header.stamp)

    def right_line_callback(self, msg: Path) -> None:
        self.latest_right_path = self._extract_valid_points(msg)
        self.latest_right_path_time = self.get_clock().now()
        self.latest_right_stamp_ns = self._stamp_ns(msg.header.stamp)

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
            List[Tuple[float, float]], str, int, int, int]:
        """Sample the midline at fit_n_samples evenly-spaced x's in
        [fit_x_min, fit_x_max].

        Bilateral: average y_left and y_right at each x — but ONLY if
        |y_left − y_right| is within `width_tol` of `2·half_lane_width`.
        On the failing run the perception was bracketing different
        physical lanes (implied separation up to 2.25 m vs measured
        0.85 m); polyfit absorbed the geometric inconsistency as fake
        curvature. The width gate refuses such pairs and falls back to
        single-side synthesis on whichever side has more raw points.

        Single-side: offset the available boundary by ±half_lane_width.
        Skip x's where neither side interpolates.

        Returns (pts, mode, n_bilateral, n_single, n_width_rejected)."""
        L = self._fresh("left")
        R = self._fresh("right")
        pts: List[Tuple[float, float]] = []
        n_bi = n_single = n_width_reject = 0
        if L is None and R is None:
            return pts, "STALE", 0, 0, 0
        n = max(2, int(self.fit_n_samples))
        xs = np.linspace(self.fit_x_min, self.fit_x_max, n)
        hw = self.half_lane_width
        full_width = 2.0 * hw
        tol = self.width_tol
        # When we have to fall back to single-side on a width-rejected
        # pair, prefer whichever side has more raw detected points
        # globally for this tick. (Per-x density is unavailable.)
        prefer_left = (
            L is not None and R is not None and len(L) >= len(R)
        ) or (L is not None and R is None)
        for x in xs:
            yl = interp_y_at_x(L, float(x)) if L is not None else None
            yr = interp_y_at_x(R, float(x)) if R is not None else None
            if yl is not None and yr is not None:
                if abs((yl - yr) - full_width) <= tol:
                    pts.append((float(x), 0.5 * (yl + yr)))
                    n_bi += 1
                else:
                    # width-inconsistent pair → fall back to single-side
                    n_width_reject += 1
                    if prefer_left:
                        pts.append((float(x), yl - hw))
                    else:
                        pts.append((float(x), yr + hw))
                    n_single += 1
            elif yl is not None:
                pts.append((float(x), yl - hw))
                n_single += 1
            elif yr is not None:
                pts.append((float(x), yr + hw))
                n_single += 1
        if not pts:
            return pts, "STALE", 0, 0, n_width_reject
        mode = "BILATERAL" if n_bi >= max(1, n_single) else "SINGLE_LINE"
        return pts, mode, n_bi, n_single, n_width_reject

    # ── control loop ───────────────────────────────────────────────────
    def control_loop(self) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        self._tick_count += 1

        if self.prev_time_ns is None:
            dt_tick = 1.0 / max(self.control_rate_hz, 1.0)
        else:
            dt_tick = max((now_ns - self.prev_time_ns) * 1e-9, 1e-3)

        pts, mode, n_bi, n_single, n_width_reject = self._build_midline_samples()
        if mode == "BILATERAL":
            self._seen_bilateral = True
        gate_wait_bi = (mode == "SINGLE_LINE" and not self._seen_bilateral)
        if gate_wait_bi:
            mode = "WAIT_BI"
        mode_changed = (
            self._prev_mode is not None and self._prev_mode != mode
        )
        self._prev_mode = mode

        L = self.latest_left_path
        R = self.latest_right_path
        left_xmax = L[-1][0] if L else float("nan")
        right_xmax = R[-1][0] if R else float("nan")
        left_age = (
            (now - self.latest_left_path_time).nanoseconds * 1e-9
            if self.latest_left_path_time is not None else float("nan")
        )
        right_age = (
            (now - self.latest_right_path_time).nanoseconds * 1e-9
            if self.latest_right_path_time is not None else float("nan")
        )

        dbg = {
            "dt": dt_tick,
            "left_stamp_ns": self.latest_left_stamp_ns,
            "right_stamp_ns": self.latest_right_stamp_ns,
            "mode": mode, "mode_changed": mode_changed,
            "n_pts": len(pts), "n_bi": n_bi, "n_single": n_single,
            "n_width_rejected": n_width_reject,
            "left_n": len(L), "left_xmax": left_xmax, "left_age": left_age,
            "right_n": len(R), "right_xmax": right_xmax, "right_age": right_age,
            "fit_a": float("nan"), "fit_b": float("nan"), "fit_c": float("nan"),
            "slope": 0.0,
            "kappa_raw": float("nan"),
            "kappa_smoothed": self.kappa_smoothed,
            "kappa_clamped": float("nan"), "kappa_clipped": False,
            "e_y_raw": float("nan"), "e_y": float("nan"), "d_e_y": 0.0,
            "delta_arc": 0.0, "delta_p": 0.0, "delta_d": 0.0,
            "delta_raw": 0.0,
            "delta_ema": self.last_steer_smoothed,
            "delta_slew": self.last_steer_smoothed,
            "delta_final": self.last_steer_smoothed,
            "slew_clipped": False, "sat_clipped": False,
            "speed": float("nan"),
        }

        # Need ≥4 samples for a stable parabola fit, AND we must have
        # seen a BILATERAL fix at least once (else any SINGLE_LINE
        # detection is treated as not-yet-tracking).
        if len(pts) < 4 or gate_wait_bi:
            marker_label = "WAIT_BI" if gate_wait_bi else "STALE"
            self._publish_marker(self.last_target, marker_label,
                                 color=(1.0, 0.0, 0.0))
            if self.stop_if_no_path:
                self._publish_drive(0.0, 0.0)
                dbg["delta_final"] = 0.0
                dbg["speed"] = 0.0
            else:
                self._publish_drive(self.nominal_speed,
                                    self.last_steer_smoothed)
                dbg["speed"] = self.nominal_speed
            self._log_tick(dbg)
            return

        # ── arc feed-forward: parabola fit, signed-curvature bicycle ───
        # Gate the curvature FF on FULL bilateral coverage. Mixing
        # BILATERAL and SINGLE_LINE-synthesized samples in one polyfit
        # produces phantom curvature whose sign and magnitude depend on
        # the L/R coverage geometry, not the road. When gated:
        #   - δ_arc forced to 0 (don't fight the lateral P term)
        #   - kappa_smoothed multiplicatively decayed so the τ ≈ 3 s EMA
        #     can't latch onto the saturated state for many seconds
        # See run-20260503-052900: with the unguarded FF the controller
        # sat at δ ≈ +0.013 rad for 17 s while e_y was +0.36 m — the FF
        # cancelled 72 % of the P term every tick.
        fit = fit_arc_parabola(pts)
        x_eval = self.e_y_eval_x
        ff_gated = (fit is None) or (n_bi < len(pts))
        if ff_gated:
            delta_arc = 0.0
            self.kappa_smoothed *= 0.90
            if fit is None:
                e_y_raw = pts[0][1]
                slope = 0.0
            else:
                fa, fb, fc = fit
                dbg["fit_a"], dbg["fit_b"], dbg["fit_c"] = fa, fb, fc
                slope = 2.0 * fa * x_eval + fb
                e_y_raw = fa * x_eval * x_eval + fb * x_eval + fc
            e_y = clamp(e_y_raw, -0.5, 0.5)
            dbg["kappa_smoothed"] = self.kappa_smoothed
        else:
            fa, fb, fc = fit
            dbg["fit_a"], dbg["fit_b"], dbg["fit_c"] = fa, fb, fc
            # κ = y''(x) / (1 + y'(x)²)^(3/2), evaluated at x_eval.
            # +y is left in robot frame, so positive κ ⇒ left curve ⇒ δ > 0.
            slope = 2.0 * fa * x_eval + fb
            kappa_raw = (2.0 * fa) / (1.0 + slope * slope) ** 1.5
            ka = clamp(self.kappa_alpha, 0.0, 1.0)
            self.kappa_smoothed = (
                ka * kappa_raw + (1.0 - ka) * self.kappa_smoothed
            )
            kappa_clamped = clamp(
                self.kappa_smoothed,
                -self.max_curvature, self.max_curvature,
            )
            dbg["kappa_raw"] = kappa_raw
            dbg["kappa_smoothed"] = self.kappa_smoothed
            dbg["kappa_clamped"] = kappa_clamped
            dbg["kappa_clipped"] = (kappa_clamped != self.kappa_smoothed)
            delta_arc = math.atan(self.wheelbase * kappa_clamped)
            # Clamp e_y to a physically plausible range. Outside ±0.5 m
            # the controller is well off-lane; further amplification
            # produces saturation spikes that don't help.
            e_y_raw = fa * x_eval * x_eval + fb * x_eval + fc
            e_y = clamp(e_y_raw, -0.5, 0.5)
        dbg["slope"] = slope
        dbg["e_y_raw"] = e_y_raw
        dbg["e_y"] = e_y
        dbg["delta_arc"] = delta_arc

        # ── lateral PD trim on residual cross-track at e_y_eval_x ──────
        if self.prev_time_ns is None or self.prev_e_y is None:
            d_e_y = 0.0
        else:
            d_e_y = (e_y - self.prev_e_y) / dt_tick
        delta_p = self.kp_lat * e_y
        delta_d = self.kd_lat * d_e_y
        delta_raw = delta_arc + delta_p + delta_d
        dbg["d_e_y"] = d_e_y
        dbg["delta_p"] = delta_p
        dbg["delta_d"] = delta_d
        dbg["delta_raw"] = delta_raw

        # ── EMA + slew limit + saturate ────────────────────────────────
        alpha = clamp(self.steering_alpha, 0.0, 1.0)
        delta_ema = alpha * delta_raw + (1.0 - alpha) * self.last_steer_smoothed
        max_step = self.max_steering_rate * dt_tick
        delta_slew = clamp(
            delta_ema,
            self.last_steer_smoothed - max_step,
            self.last_steer_smoothed + max_step,
        )
        steering_angle = clamp(
            delta_slew, -self.max_steering_angle, self.max_steering_angle,
        )
        dbg["delta_ema"] = delta_ema
        dbg["delta_slew"] = delta_slew
        dbg["delta_final"] = steering_angle
        dbg["slew_clipped"] = (delta_slew != delta_ema)
        dbg["sat_clipped"] = (steering_angle != delta_slew)
        dbg["speed"] = self.nominal_speed

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

        self._log_tick(dbg)

    # ── tuning instrumentation ─────────────────────────────────────────
    def _log_tick(self, dbg: dict) -> None:
        t = monotonic() - self._t0
        if self.debug_log and self._csv_writer is not None:
            row = [
                f"{t:.4f}", self._tick_count, f"{dbg['dt']:.4f}",
                dbg["left_stamp_ns"], dbg["right_stamp_ns"],
                dbg["mode"], int(bool(dbg["mode_changed"])),
                dbg["n_pts"], dbg["n_bi"], dbg["n_single"],
                dbg["n_width_rejected"],
                dbg["left_n"], dbg["left_xmax"], dbg["left_age"],
                dbg["right_n"], dbg["right_xmax"], dbg["right_age"],
                dbg["fit_a"], dbg["fit_b"], dbg["fit_c"], dbg["slope"],
                dbg["kappa_raw"], dbg["kappa_smoothed"],
                dbg["kappa_clamped"], int(bool(dbg["kappa_clipped"])),
                dbg["e_y_raw"], dbg["e_y"], dbg["d_e_y"],
                dbg["delta_arc"], dbg["delta_p"], dbg["delta_d"],
                dbg["delta_raw"], dbg["delta_ema"],
                dbg["delta_slew"], dbg["delta_final"],
                int(bool(dbg["slew_clipped"])), int(bool(dbg["sat_clipped"])),
                dbg["speed"],
            ]
            try:
                self._csv_writer.writerow(row)
                self._csv_file.flush()
            except (OSError, ValueError) as e:
                self.get_logger().warn(f"[TUNE] CSV write failed: {e}")

        if not math.isnan(dbg["e_y"]):
            self._roll_e_y.append(dbg["e_y"])
        if not math.isnan(dbg["delta_final"]):
            self._roll_delta.append(dbg["delta_final"])

        if (self.debug_print_every > 0
                and self._tick_count % self.debug_print_every == 0):
            self._print_tune_line(t, dbg)

    def _print_tune_line(self, t: float, dbg: dict) -> None:
        e_arr = list(self._roll_e_y)
        d_arr = list(self._roll_delta)
        e_std = float(np.std(e_arr)) if e_arr else 0.0
        d_std = float(np.std(d_arr)) if d_arr else 0.0
        flips = sum(
            1 for i in range(1, len(d_arr))
            if d_arr[i - 1] * d_arr[i] < 0.0
        )
        flag = ""
        if dbg["slew_clipped"]:  flag += "S"
        if dbg["sat_clipped"]:   flag += "T"
        if dbg["kappa_clipped"]: flag += "K"
        if dbg["mode_changed"]:  flag += "M"
        if not flag:
            flag = "-"
        e_y = dbg["e_y"]
        df = dbg["delta_final"]
        e_y_str = f"{e_y:+.3f}" if not math.isnan(e_y) else "  nan"
        df_str = f"{df:+.3f}" if not math.isnan(df) else "  nan"
        self.get_logger().info(
            f"[TUNE t={t:6.2f}s] {dbg['mode']:<11} "
            f"n={dbg['n_pts']:2d}(bi={dbg['n_bi']} sg={dbg['n_single']}) "
            f"e_y={e_y_str}(σ={e_std:.3f}) "
            f"δ={df_str}(σ={d_std:.3f} flips={flips:2d}) "
            f"ff={dbg['delta_arc']:+.3f} "
            f"P={dbg['delta_p']:+.3f} "
            f"D={dbg['delta_d']:+.3f} "
            f"flag={flag}"
        )

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
        if getattr(node, "_csv_file", None) is not None:
            try:
                node._csv_file.close()
            except OSError:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

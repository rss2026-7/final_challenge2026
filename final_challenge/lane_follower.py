#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def path_arc_length(points: List[Tuple[float, float]]) -> float:
    """Total arc length of a polyline."""
    length = 0.0
    for i in range(1, len(points)):
        length += point_dist(points[i - 1], points[i])
    return length


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


def build_midpoint_path(
    left: List[Tuple[float, float]],
    right: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Build a midpoint polyline between left and right boundary paths.
    Sample at common x-range of both, step ~0.2m."""
    if len(left) < 2 or len(right) < 2:
        return []
    x_start = max(left[0][0], right[0][0])
    x_end = min(left[-1][0], right[-1][0])
    if x_end - x_start < 0.2:
        return []

    midpts = []
    x = x_start
    step = 0.2
    while x <= x_end + 1e-6:
        yl = interp_y_at_x(left, x)
        yr = interp_y_at_x(right, x)
        if yl is not None and yr is not None:
            midpts.append((x, 0.5 * (yl + yr)))
        x += step
    return midpts


def offset_path_y(
    path: List[Tuple[float, float]],
    dy: float,
) -> List[Tuple[float, float]]:
    """Shift each point of `path` by dy in the y direction.

    Plausibility filter accepts only forward-aligned paths (|slope| < 0.4),
    so a pure y-shift is within ~6% of the true normal-shift. Per-point local
    normals computed from blob spines are jittery; y-shift is deterministic
    and produces a stable target.
    """
    if len(path) < 1 or abs(dy) < 1e-6:
        return list(path)
    return [(x, y + dy) for (x, y) in path]


def fit_target_at_forward(
    path: List[Tuple[float, float]],
    forward_distance: float,
) -> Optional[Tuple[float, float]]:
    """Fit a line to the path's (x, y) and evaluate at x = forward_distance.

    Used instead of walking-along-arc so that noisy 200-point blob spines
    produce the same target as 14-point Hough samples for the same physical
    line. Removes the dominant source of frame-to-frame target jitter.
    """
    if len(path) < 2:
        return None
    import numpy as np  # local import to keep module dep tidy
    xs = np.array([p[0] for p in path])
    ys = np.array([p[1] for p in path])
    if float(np.ptp(xs)) < 0.05:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    return (float(forward_distance),
            float(slope * forward_distance + intercept))


class BoundaryPurePursuit(Node):
    """
    Pure pursuit controller that drives the lane center.

    Strategy
    --------
    BILATERAL — when both /left_lane_line and /right_lane_line are fresh:
        target path = midpoint of the two boundaries (per-x average).
        No offset, no track-side preference. The geometric center IS the goal.
        Also updates a learned half-lane-width estimate.

    LEFT_ONLY / RIGHT_ONLY — when only one boundary is fresh:
        target path = visible boundary shifted inward by the learned half-width
        (along local normal). Sign is determined by which side is missing.

    STALE — when neither side is fresh:
        follow the last good midpoint path for up to stale_path_timeout seconds.
        After that, stop.

    Subscriptions
    -------------
    /left_lane_line  (nav_msgs/Path) — left boundary in base_link
    /right_lane_line (nav_msgs/Path) — right boundary in base_link

    Publications
    ------------
    /drive            (ackermann_msgs/AckermannDriveStamped)
    /lookahead_target (visualization_msgs/Marker)
    """

    def __init__(self) -> None:
        super().__init__("boundary_pure_pursuit")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("left_line_topic", "/left_lane_line")
        self.declare_parameter("right_line_topic", "/right_lane_line")
        self.declare_parameter("drive_topic", "/drive")

        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("lookahead_distance", 1.2)
        self.declare_parameter("lost_line_lookahead_distance", 0.9)
        self.declare_parameter("min_lookahead_distance", 0.5)

        self.declare_parameter("nominal_speed", 2.5)
        self.declare_parameter("lost_line_speed", 1.0)
        self.declare_parameter("min_speed", 0.6)
        self.declare_parameter("max_speed", 3.5)

        self.declare_parameter("max_steering_angle", 0.40)
        self.declare_parameter("curvature_speed_gain", 1.2)
        self.declare_parameter("curvature_lookahead_gain", 2.0)

        # Initial half-lane-width before any bilateral observation. EMA-updated at runtime.
        self.declare_parameter("half_width_init", 0.5)
        # EMA learning rate for half-width
        self.declare_parameter("half_width_alpha", 0.1)
        # Plausibility window for the lane half-width. BILATERAL is accepted only when
        # the measured width sits in [2*half_width_min, 2*half_width_max].
        self.declare_parameter("half_width_min", 0.35)
        self.declare_parameter("half_width_max", 0.65)

        # How long we trust old midpoint after losing fresh detections
        self.declare_parameter("stale_path_timeout", 0.75)
        # How old a "latest" message can be before we consider it stale (seconds)
        self.declare_parameter("fresh_msg_timeout", 0.2)
        # Minimum arc length of a single boundary path to be useful (meters)
        self.declare_parameter("min_path_arc_length", 0.3)
        # If no valid path available, stop
        self.declare_parameter("stop_if_no_path", True)
        # Steering smoothing factor (0 = no smoothing, 1 = instant)
        self.declare_parameter("steering_alpha", 0.15)
        # Target low-pass filter factor. Smooths mode-switch jumps and
        # frame-to-frame jitter on the lookahead point. Smaller = more damping.
        self.declare_parameter("target_alpha", 0.20)

        left_line_topic = self.get_parameter("left_line_topic").value
        right_line_topic = self.get_parameter("right_line_topic").value
        drive_topic = self.get_parameter("drive_topic").value

        self._load_tunable_params()

        # Subscriptions
        self.left_sub = self.create_subscription(
            Path, left_line_topic, self.left_line_callback, 10
        )
        self.right_sub = self.create_subscription(
            Path, right_line_topic, self.right_line_callback, 10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )
        self.marker_pub = self.create_publisher(
            Marker, "/lookahead_target", 10
        )

        # Latest fresh boundary paths
        self.latest_left_path: List[Tuple[float, float]] = []
        self.latest_right_path: List[Tuple[float, float]] = []
        self.latest_left_path_time = None
        self.latest_right_path_time = None

        # Last good midpoint path (used during STALE)
        self.last_good_midpoint: List[Tuple[float, float]] = []
        self.last_good_midpoint_time = None

        # Learned half-width (EMA)
        self.half_width = self.half_width_init

        # Steering low-pass filter state
        self.prev_steering = 0.0
        # Target low-pass filter state (smooths mode-switch jumps:
        # BILATERAL→LEFT_ONLY can shift target ~half-width laterally)
        self.prev_target: Optional[Tuple[float, float]] = None

        # 20 Hz control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            "BoundaryPurePursuit (bilateral midpoint mode) started. "
            f"half_width_init={self.half_width_init:.2f}"
        )

    # -------------------------------------------------
    # Parameter helpers
    # -------------------------------------------------
    def _load_tunable_params(self) -> None:
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.lost_line_lookahead_distance = float(
            self.get_parameter("lost_line_lookahead_distance").value
        )
        self.min_lookahead_distance = float(
            self.get_parameter("min_lookahead_distance").value
        )

        self.nominal_speed = float(self.get_parameter("nominal_speed").value)
        self.lost_line_speed = float(self.get_parameter("lost_line_speed").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)

        self.max_steering_angle = float(self.get_parameter("max_steering_angle").value)
        self.curvature_speed_gain = float(self.get_parameter("curvature_speed_gain").value)
        self.curvature_lookahead_gain = float(
            self.get_parameter("curvature_lookahead_gain").value
        )

        self.half_width_init = float(self.get_parameter("half_width_init").value)
        self.half_width_alpha = float(self.get_parameter("half_width_alpha").value)
        self.half_width_min = float(self.get_parameter("half_width_min").value)
        self.half_width_max = float(self.get_parameter("half_width_max").value)

        self.stale_path_timeout = float(self.get_parameter("stale_path_timeout").value)
        self.fresh_msg_timeout = float(self.get_parameter("fresh_msg_timeout").value)
        self.min_path_arc_length = float(self.get_parameter("min_path_arc_length").value)
        self.stop_if_no_path = bool(self.get_parameter("stop_if_no_path").value)
        self.steering_alpha = float(self.get_parameter("steering_alpha").value)
        self.target_alpha = float(self.get_parameter("target_alpha").value)

    def _on_param_change(self, params) -> SetParametersResult:
        float_params = {
            "wheelbase", "lookahead_distance", "lost_line_lookahead_distance",
            "min_lookahead_distance", "nominal_speed", "lost_line_speed",
            "min_speed", "max_speed", "max_steering_angle",
            "curvature_speed_gain", "curvature_lookahead_gain",
            "half_width_init", "half_width_alpha",
            "half_width_min", "half_width_max",
            "stale_path_timeout", "fresh_msg_timeout",
            "min_path_arc_length", "steering_alpha", "target_alpha",
        }
        for p in params:
            if p.name in float_params:
                setattr(self, p.name, float(p.value))
                self.get_logger().info(f"Parameter {p.name} updated to {p.value}")
            elif p.name == "stop_if_no_path":
                self.stop_if_no_path = bool(p.value)
                self.get_logger().info(f"Parameter stop_if_no_path updated to {p.value}")
        return SetParametersResult(successful=True)

    # -------------------------------------------------
    # Callbacks — only store data and timestamp
    # -------------------------------------------------
    def left_line_callback(self, msg: Path) -> None:
        self.latest_left_path = self.extract_valid_points(msg)
        self.latest_left_path_time = self.get_clock().now()

    def right_line_callback(self, msg: Path) -> None:
        self.latest_right_path = self.extract_valid_points(msg)
        self.latest_right_path_time = self.get_clock().now()

    def extract_valid_points(self, msg: Path) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if x < -0.2:
                continue
            points.append((x, y))
        points.sort(key=lambda p: p[0])
        return points

    # -------------------------------------------------
    # Freshness check on a single side
    # -------------------------------------------------
    def _fresh_path(self, side: str) -> Optional[List[Tuple[float, float]]]:
        if side == "left":
            path = self.latest_left_path
            ts = self.latest_left_path_time
        else:
            path = self.latest_right_path
            ts = self.latest_right_path_time

        if ts is None:
            return None
        age = (self.get_clock().now() - ts).nanoseconds * 1e-9
        if age > self.fresh_msg_timeout:
            return None
        if len(path) < 2 or path_arc_length(path) < self.min_path_arc_length:
            return None
        return path

    # -------------------------------------------------
    # Main control loop
    # -------------------------------------------------
    def control_loop(self) -> None:
        L = self._fresh_path("left")
        R = self._fresh_path("right")

        path_to_follow: Optional[List[Tuple[float, float]]] = None
        mode = "NONE"
        is_stale = False

        # ── Priority 1: BILATERAL with plausible lane width ─────────────
        if L is not None and R is not None:
            mid = build_midpoint_path(L, R)
            width = self._mean_width(L, R)
            if (len(mid) >= 2
                    and 2.0 * self.half_width_min <= width
                    <= 2.0 * self.half_width_max):
                path_to_follow = mid
                mode = "BILATERAL"
                self._update_half_width(L, R)
                self.last_good_midpoint = mid
                self.last_good_midpoint_time = self.get_clock().now()
            else:
                # Width implausible — drop the side whose median |y| is larger.
                # The remaining side is usually the real adjacent boundary;
                # the dropped side is most likely a far stripe or an artifact.
                # Median (not min) is robust to sloped lines where one endpoint
                # passes through y=0.
                import numpy as np
                l_med = abs(float(np.median([p[1] for p in L])))
                r_med = abs(float(np.median([p[1] for p in R])))
                if l_med <= r_med:
                    R = None
                else:
                    L = None

        # Hysteresis: if a fresh bilateral midpoint exists in the very
        # recent past, keep using it instead of dropping to a single-side
        # offset path. This eats the BILATERAL ↔ LEFT_ONLY oscillation that
        # happens when a frame loses one boundary momentarily.
        bilateral_hold_window = 0.3  # seconds
        if (path_to_follow is None
                and len(self.last_good_midpoint) >= 2
                and self.last_good_midpoint_time is not None):
            age = (self.get_clock().now()
                   - self.last_good_midpoint_time).nanoseconds * 1e-9
            if age <= bilateral_hold_window and (L is not None or R is not None):
                path_to_follow = self.last_good_midpoint
                mode = "BILATERAL_HOLD"

        if path_to_follow is None and L is not None:
            offset = offset_path_y(L, -self.half_width)
            if len(offset) >= 2:
                path_to_follow = offset
                mode = "LEFT_ONLY"
                self.last_good_midpoint = offset
                self.last_good_midpoint_time = self.get_clock().now()

        if path_to_follow is None and R is not None:
            offset = offset_path_y(R, +self.half_width)
            if len(offset) >= 2:
                path_to_follow = offset
                mode = "RIGHT_ONLY"
                self.last_good_midpoint = offset
                self.last_good_midpoint_time = self.get_clock().now()

        if path_to_follow is None:
            if (len(self.last_good_midpoint) >= 2
                    and self.last_good_midpoint_time is not None):
                age = (self.get_clock().now()
                       - self.last_good_midpoint_time).nanoseconds * 1e-9
                if age <= self.stale_path_timeout:
                    path_to_follow = self.last_good_midpoint
                    mode = "STALE"
                    is_stale = True

        if path_to_follow is None:
            if self.stop_if_no_path:
                self.publish_stop()
            self.get_logger().info("No usable path — stopped.",
                                   throttle_duration_sec=1.0)
            return

        # Curvature-adaptive lookahead is only trustworthy on the bilateral
        # midpoint path. Single-side offsets are noisy enough that the 3-point
        # Menger curvature estimate spikes (we've seen κ=3.3 on a real frame),
        # which then floors the lookahead and doubles the steering gain.
        if mode == "BILATERAL" and not is_stale:
            lookahead = self._compute_adaptive_lookahead(path_to_follow, False)
        elif is_stale:
            lookahead = self.lost_line_lookahead_distance
        else:
            lookahead = self.lookahead_distance

        # Polyfit-based target. Replaces walk-along-arc which is sensitive to
        # detector switching (Hough vs. blob) — same physical line gives the
        # same fitted target whether the path has 14 sample points or 200.
        target = fit_target_at_forward(path_to_follow, lookahead)
        if target is None:
            target = self.find_lookahead_target(path_to_follow, lookahead)
        if target is None:
            if self.stop_if_no_path:
                self.publish_stop()
            return

        # Low-pass the target so mode switches and per-frame jitter don't
        # whip the steering. Reset the filter on stale recoveries to avoid
        # carrying old state across long blackouts.
        if self.prev_target is not None and not is_stale:
            a = self.target_alpha
            target = (
                a * target[0] + (1.0 - a) * self.prev_target[0],
                a * target[1] + (1.0 - a) * self.prev_target[1],
            )
        self.prev_target = target

        self._publish_target_marker(target, mode)

        steering_angle, curvature = self.compute_pure_pursuit_command(target)

        # Steering low-pass filter
        steering_angle = (
            self.steering_alpha * steering_angle
            + (1.0 - self.steering_alpha) * self.prev_steering
        )
        self.prev_steering = steering_angle

        if is_stale:
            speed = self.lost_line_speed
        else:
            speed = self.compute_speed_from_curvature(curvature, steering_angle)

        self.publish_drive(speed, steering_angle)

        self.get_logger().info(
            f"mode={mode} steer={steering_angle:.3f} speed={speed:.2f} "
            f"hw={self.half_width:.2f} pts={len(path_to_follow)} "
            f"la={lookahead:.2f} curv={curvature:.3f}",
            throttle_duration_sec=0.5,
        )

    # -------------------------------------------------
    # Half-width learning
    # -------------------------------------------------
    @staticmethod
    def _mean_width(
        left: List[Tuple[float, float]],
        right: List[Tuple[float, float]],
    ) -> float:
        """|median_y_left - median_y_right|. Robust summary of the lateral gap.

        Using medians rather than per-x interpolation keeps the width estimate
        stable when the two paths span different forward ranges or when one
        path has a few outlier points near a clipped edge.
        """
        if len(left) < 2 or len(right) < 2:
            return 0.0
        import numpy as np  # local — avoid module-level dep
        yl = float(np.median([p[1] for p in left]))
        yr = float(np.median([p[1] for p in right]))
        return abs(yl - yr)

    def _update_half_width(
        self,
        left: List[Tuple[float, float]],
        right: List[Tuple[float, float]],
    ) -> None:
        w = self._mean_width(left, right)
        if w <= 0.0:
            return
        half = 0.5 * w
        # half_width_min/max are also the BILATERAL acceptance bounds, so any
        # accepted bilateral pair already lies in this range.
        a = self.half_width_alpha
        self.half_width = (1.0 - a) * self.half_width + a * half

    # -------------------------------------------------
    # Adaptive lookahead
    # -------------------------------------------------
    def _compute_adaptive_lookahead(
        self,
        points: List[Tuple[float, float]],
        is_stale: bool,
    ) -> float:
        if is_stale:
            return self.lost_line_lookahead_distance

        curvature_est = self._estimate_path_curvature(points)
        adaptive = self.lookahead_distance / (
            1.0 + self.curvature_lookahead_gain * curvature_est
        )
        return clamp(adaptive, self.min_lookahead_distance, self.lookahead_distance)

    @staticmethod
    def _estimate_path_curvature(points: List[Tuple[float, float]]) -> float:
        """Mean unsigned Menger curvature over consecutive triplets.

        Each triplet's κ is clamped to 1.0 (1/m, R≥1m) before averaging so
        that a single noisy point can't drive the average to bogus values
        like 3.3 (R=0.3m), which would floor the adaptive lookahead.
        """
        if len(points) < 3:
            return 0.0

        total_curvature = 0.0
        count = 0
        step = max(1, (len(points) - 2) // 10)
        for i in range(0, len(points) - 2, step):
            p0 = points[i]
            p1 = points[i + 1]
            p2 = points[i + 2]

            ax = p1[0] - p0[0]
            ay = p1[1] - p0[1]
            bx = p2[0] - p1[0]
            by = p2[1] - p1[1]

            cross = abs(ax * by - ay * bx)
            la = math.hypot(ax, ay)
            lb = math.hypot(bx, by)
            chord = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
            denom = la * lb * chord
            if denom < 1e-9:
                continue
            kappa = 2.0 * cross / denom
            total_curvature += min(kappa, 1.0)
            count += 1

        return total_curvature / count if count > 0 else 0.0

    # -------------------------------------------------
    # Lookahead target — with extrapolation
    # -------------------------------------------------
    def find_closest_point_index(self, points: List[Tuple[float, float]]) -> int:
        min_idx = 0
        min_dist = float("inf")
        for i, (x, y) in enumerate(points):
            d = math.hypot(x, y)
            if d < min_dist:
                min_dist = d
                min_idx = i
        return min_idx

    def find_lookahead_target(
        self,
        points: List[Tuple[float, float]],
        lookahead_distance: float
    ) -> Optional[Tuple[float, float]]:
        if len(points) < 2:
            return None

        closest_idx = self.find_closest_point_index(points)

        if closest_idx >= len(points) - 1:
            return self._extrapolate_from_end(points, lookahead_distance)

        accumulated = 0.0
        p_prev = points[closest_idx]

        for i in range(closest_idx + 1, len(points)):
            p_curr = points[i]
            seg_len = point_dist(p_prev, p_curr)

            if accumulated + seg_len >= lookahead_distance:
                remaining = lookahead_distance - accumulated
                if seg_len < 1e-6:
                    return p_curr
                t = remaining / seg_len
                x = p_prev[0] + t * (p_curr[0] - p_prev[0])
                y = p_prev[1] + t * (p_curr[1] - p_prev[1])
                return (x, y)

            accumulated += seg_len
            p_prev = p_curr

        remaining = lookahead_distance - accumulated
        return self._extrapolate_from_end(points, remaining)

    @staticmethod
    def _extrapolate_from_end(
        points: List[Tuple[float, float]],
        distance: float,
    ) -> Tuple[float, float]:
        if len(points) < 2:
            return points[-1]
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return points[-1]
        ux = dx / seg_len
        uy = dy / seg_len
        return (points[-1][0] + distance * ux, points[-1][1] + distance * uy)

    # -------------------------------------------------
    # Pure pursuit math
    # -------------------------------------------------
    def compute_pure_pursuit_command(
        self,
        target: Tuple[float, float]
    ) -> Tuple[float, float]:
        tx, ty = target
        ld = math.hypot(tx, ty)
        ld = max(ld, 1e-3)

        alpha = math.atan2(ty, tx)
        curvature = 2.0 * math.sin(alpha) / ld

        steering_angle = math.atan(self.wheelbase * curvature)
        steering_angle = clamp(
            steering_angle, -self.max_steering_angle, self.max_steering_angle
        )
        return steering_angle, abs(curvature)

    def compute_speed_from_curvature(
        self, curvature: float, steering_angle: float
    ) -> float:
        speed = self.nominal_speed / (1.0 + self.curvature_speed_gain * curvature)
        steer_ratio = abs(steering_angle) / self.max_steering_angle
        speed *= (1.0 - 0.3 * steer_ratio)
        return clamp(speed, self.min_speed, self.max_speed)

    # -------------------------------------------------
    # Publishing
    # -------------------------------------------------
    def publish_drive(self, speed: float, steering_angle: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def publish_stop(self) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)

    def _publish_target_marker(self, target: Tuple[float, float], mode: str) -> None:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "boundary_pure_pursuit"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = target[0]
        m.pose.position.y = target[1]
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0

        m.scale.x = 0.10
        m.scale.y = 0.10
        m.scale.z = 0.10

        # green=BILATERAL, yellow=single-side, red=stale
        if mode == "BILATERAL":
            m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
        elif mode == "STALE":
            m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
        else:
            m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
        m.color.a = 1.0

        m.lifetime.sec = 0
        m.lifetime.nanosec = 200_000_000

        self.marker_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoundaryPurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

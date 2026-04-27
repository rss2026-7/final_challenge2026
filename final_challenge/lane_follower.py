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


class BoundaryPurePursuit(Node):
    """
    Pure pursuit controller that follows either the left or right lane boundary
    with an inward offset.

    Features:
    - choose left or right boundary
    - follow last known valid path when line is temporarily lost
    - slow down when using stale path
    - stop if stale for too long
    - curvature-adaptive lookahead
    - steering low-pass filter
    - RViz lookahead marker for debugging
    - dynamic parameter reconfiguration

    Assumptions:
    - Upstream publishes:
        /left_lane_line  (nav_msgs/Path)
        /right_lane_line (nav_msgs/Path)
    - Points are in base_link frame
    - x forward, y left
    - path is ordered from near to far
    """

    def __init__(self) -> None:
        super().__init__("boundary_pure_pursuit")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("left_line_topic", "/left_lane_line")
        self.declare_parameter("right_line_topic", "/right_lane_line")
        self.declare_parameter("drive_topic", "/drive")

        self.declare_parameter("track_side", "left")  # "left" or "right"

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

        # Distance inward from chosen boundary.
        # Example: 0.20 means drive 20 cm inside the boundary.
        self.declare_parameter("inward_offset", 0.20)

        # How long we trust old path after losing fresh detections
        self.declare_parameter("stale_path_timeout", 0.75)

        # How old a "latest" message can be before we consider it stale (seconds)
        self.declare_parameter("fresh_msg_timeout", 0.2)

        # Minimum arc length of a path to be considered useful (meters)
        self.declare_parameter("min_path_arc_length", 0.3)

        # If no valid line/path available, stop
        self.declare_parameter("stop_if_no_path", True)

        # Steering smoothing factor (0 = no smoothing, 1 = instant)
        self.declare_parameter("steering_alpha", 0.35)

        left_line_topic = self.get_parameter("left_line_topic").value
        right_line_topic = self.get_parameter("right_line_topic").value
        drive_topic = self.get_parameter("drive_topic").value

        self._load_tunable_params()

        if self.track_side not in ("left", "right"):
            raise ValueError("track_side must be 'left' or 'right'")

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

        # RViz marker for the lookahead target
        self.marker_pub = self.create_publisher(
            Marker, "/lookahead_target", 10
        )

        # Store latest fresh paths separately, with timestamps
        self.latest_left_path: List[Tuple[float, float]] = []
        self.latest_right_path: List[Tuple[float, float]] = []
        self.latest_left_path_time = None
        self.latest_right_path_time = None

        # Last good path for the chosen side
        self.last_good_path: List[Tuple[float, float]] = []
        self.last_good_path_time = None

        # Steering low-pass filter state
        self.prev_steering = 0.0

        # Timer to run control continuously even if detections momentarily stop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # Dynamic parameter reconfiguration
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"BoundaryPurePursuit started. Tracking {self.track_side} boundary."
        )

    # -------------------------------------------------
    # Parameter helpers
    # -------------------------------------------------
    def _load_tunable_params(self) -> None:
        """Read all tunable parameters from the parameter server."""
        self.track_side = str(self.get_parameter("track_side").value).strip().lower()

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

        self.inward_offset = float(self.get_parameter("inward_offset").value)
        self.stale_path_timeout = float(self.get_parameter("stale_path_timeout").value)
        self.fresh_msg_timeout = float(self.get_parameter("fresh_msg_timeout").value)
        self.min_path_arc_length = float(self.get_parameter("min_path_arc_length").value)
        self.stop_if_no_path = bool(self.get_parameter("stop_if_no_path").value)
        self.steering_alpha = float(self.get_parameter("steering_alpha").value)

    def _on_param_change(self, params) -> SetParametersResult:
        """Handle live parameter changes via `ros2 param set`."""
        # Map of parameter names to attributes
        float_params = {
            "wheelbase", "lookahead_distance", "lost_line_lookahead_distance",
            "min_lookahead_distance", "nominal_speed", "lost_line_speed",
            "min_speed", "max_speed", "max_steering_angle",
            "curvature_speed_gain", "curvature_lookahead_gain",
            "inward_offset", "stale_path_timeout", "fresh_msg_timeout",
            "min_path_arc_length", "steering_alpha",
        }
        for p in params:
            if p.name in float_params:
                setattr(self, p.name, float(p.value))
                self.get_logger().info(f"Parameter {p.name} updated to {p.value}")
            elif p.name == "track_side":
                val = str(p.value).strip().lower()
                if val not in ("left", "right"):
                    return SetParametersResult(
                        successful=False, reason="track_side must be 'left' or 'right'"
                    )
                self.track_side = val
                self.get_logger().info(f"Parameter track_side updated to {val}")
            elif p.name == "stop_if_no_path":
                self.stop_if_no_path = bool(p.value)
                self.get_logger().info(f"Parameter stop_if_no_path updated to {p.value}")
        return SetParametersResult(successful=True)

    # -------------------------------------------------
    # Callbacks — only store data and timestamp
    # -------------------------------------------------
    def left_line_callback(self, msg: Path) -> None:
        points = self.extract_valid_points(msg)
        self.latest_left_path = points
        self.latest_left_path_time = self.get_clock().now()

    def right_line_callback(self, msg: Path) -> None:
        points = self.extract_valid_points(msg)
        self.latest_right_path = points
        self.latest_right_path_time = self.get_clock().now()

    # -------------------------------------------------
    # Main control loop
    # -------------------------------------------------
    def control_loop(self) -> None:
        fresh_path = self._get_selected_fresh_path()

        using_stale_path = False
        path_to_follow: Optional[List[Tuple[float, float]]] = None

        if fresh_path is not None and len(fresh_path) >= 2:
            path_to_follow = fresh_path
            # Update last good path (single canonical location)
            self.last_good_path = fresh_path
            self.last_good_path_time = self.get_clock().now()
        else:
            # Fall back to last good path if still recent enough
            if len(self.last_good_path) >= 2 and self.last_good_path_time is not None:
                age = (self.get_clock().now() - self.last_good_path_time).nanoseconds * 1e-9
                if age <= self.stale_path_timeout:
                    path_to_follow = self.last_good_path
                    using_stale_path = True

        if path_to_follow is None:
            if self.stop_if_no_path:
                self.publish_stop()
            self.get_logger().info("No usable path — stopped.", throttle_duration_sec=1.0)
            return

        # Use previous curvature estimate for adaptive lookahead
        lookahead = self._compute_adaptive_lookahead(
            path_to_follow, is_stale=using_stale_path
        )

        target = self.find_lookahead_target(path_to_follow, lookahead)
        if target is None:
            if self.stop_if_no_path:
                self.publish_stop()
            return

        target = self.apply_inward_offset(path_to_follow, target)

        # Publish RViz marker for debugging
        self._publish_target_marker(target)

        steering_angle, curvature = self.compute_pure_pursuit_command(target)

        # Steering low-pass filter
        steering_angle = (
            self.steering_alpha * steering_angle
            + (1.0 - self.steering_alpha) * self.prev_steering
        )
        self.prev_steering = steering_angle

        if using_stale_path:
            speed = self.lost_line_speed
        else:
            speed = self.compute_speed_from_curvature(curvature, steering_angle)

        self.publish_drive(speed, steering_angle)

        # Diagnostic logging (throttled)
        self.get_logger().info(
            f"steer={steering_angle:.3f}  speed={speed:.2f}  "
            f"stale={using_stale_path}  pts={len(path_to_follow)}  "
            f"la={lookahead:.2f}  curv={curvature:.3f}",
            throttle_duration_sec=0.5,
        )

    # -------------------------------------------------
    # Path selection — with freshness checking
    # -------------------------------------------------
    def _get_selected_fresh_path(self) -> Optional[List[Tuple[float, float]]]:
        """Return the selected side's path only if the message is fresh enough
        and the path has sufficient arc length."""
        now = self.get_clock().now()

        if self.track_side == "left":
            path = self.latest_left_path
            ts = self.latest_left_path_time
        else:
            path = self.latest_right_path
            ts = self.latest_right_path_time

        # No message received yet
        if ts is None:
            return None

        age = (now - ts).nanoseconds * 1e-9
        if age > self.fresh_msg_timeout:
            return None

        # Minimum arc length check
        if len(path) < 2 or path_arc_length(path) < self.min_path_arc_length:
            return None

        return path

    def extract_valid_points(self, msg: Path) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []

        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            if not math.isfinite(x) or not math.isfinite(y):
                continue

            # Mostly ignore points significantly behind the vehicle
            if x < -0.2:
                continue

            points.append((x, y))

        # Ensure near-to-far ordering
        points.sort(key=lambda p: p[0])
        return points

    # -------------------------------------------------
    # Adaptive lookahead
    # -------------------------------------------------
    def _compute_adaptive_lookahead(
        self,
        points: List[Tuple[float, float]],
        is_stale: bool,
    ) -> float:
        """Scale lookahead inversely with path curvature.
        On tight turns use a shorter lookahead; on straights use the full value."""
        if is_stale:
            return self.lost_line_lookahead_distance

        # Estimate curvature from the first few segments of the path
        curvature_est = self._estimate_path_curvature(points)

        adaptive = self.lookahead_distance / (
            1.0 + self.curvature_lookahead_gain * curvature_est
        )
        return clamp(adaptive, self.min_lookahead_distance, self.lookahead_distance)

    @staticmethod
    def _estimate_path_curvature(points: List[Tuple[float, float]]) -> float:
        """Estimate average unsigned curvature from consecutive triplets."""
        if len(points) < 3:
            return 0.0

        total_curvature = 0.0
        count = 0
        # Sample up to 10 evenly spaced triplets
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
            denom = la * lb
            if denom < 1e-9:
                continue

            # Approximate curvature ≈ |cross| / (avg_segment_length * segment_length)
            avg_seg = (la + lb) / 2.0
            if avg_seg < 1e-9:
                continue
            total_curvature += cross / (denom * avg_seg) * 2.0
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
            # Closest point is the last point; extrapolate along the final segment
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

        # Path was too short — extrapolate beyond the last point
        remaining = lookahead_distance - accumulated
        return self._extrapolate_from_end(points, remaining)

    @staticmethod
    def _extrapolate_from_end(
        points: List[Tuple[float, float]],
        distance: float,
    ) -> Tuple[float, float]:
        """Extrapolate beyond the last path point along the final segment direction."""
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
    # Offset logic
    # -------------------------------------------------
    def apply_inward_offset(
        self,
        points: List[Tuple[float, float]],
        target: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Shift the target inward from the selected boundary.

        base_link convention:
        - x forward
        - y left

        Inward offset direction:
        - tracking left boundary  -> shift right  -> negative normal direction
        - tracking right boundary -> shift left   -> positive normal direction
        """
        if len(points) < 2 or self.inward_offset <= 1e-6:
            return target

        nearest_idx = 0
        nearest_dist = float("inf")
        for i, p in enumerate(points):
            d = point_dist(p, target)
            if d < nearest_dist:
                nearest_dist = d
                nearest_idx = i

        i0 = max(0, nearest_idx - 1)
        i1 = min(len(points) - 1, nearest_idx + 1)

        p0 = points[i0]
        p1 = points[i1]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return target

        # Unit normal pointing left of tangent
        nx = -dy / norm
        ny = dx / norm

        # Choose sign so offset is inward
        if self.track_side == "left":
            signed_offset = -self.inward_offset
        else:  # right boundary
            signed_offset = +self.inward_offset

        return (
            target[0] + signed_offset * nx,
            target[1] + signed_offset * ny
        )

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
            steering_angle,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        return steering_angle, abs(curvature)

    def compute_speed_from_curvature(
        self, curvature: float, steering_angle: float
    ) -> float:
        """Speed that accounts for both curvature and steering angle magnitude."""
        # Primary: slow down with curvature
        speed = self.nominal_speed / (1.0 + self.curvature_speed_gain * curvature)

        # Secondary: further reduce if steering angle is large relative to max
        steer_ratio = abs(steering_angle) / self.max_steering_angle  # 0..1
        speed *= (1.0 - 0.3 * steer_ratio)  # up to 30 % additional reduction

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

    def _publish_target_marker(self, target: Tuple[float, float]) -> None:
        """Publish a sphere marker at the offset lookahead target for RViz."""
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

        # Bright green when fresh, yellow when stale (marker color set in caller
        # would add complexity; keep it simple green for now)
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        m.lifetime.sec = 0
        m.lifetime.nanosec = 200_000_000  # 200 ms auto-expire

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
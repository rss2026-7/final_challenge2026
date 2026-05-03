#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from final_challenge.homography_transformer import build_homography


# BEST_EFFORT, KEEP_LAST(1) — the controller and the calibrator only ever
# care about the newest camera frame. Deeper queues just buffer staleness:
# if a callback ever falls behind, old frames pile up and we start
# processing 100+ ms-old images. depth=1 drops them at the DDS layer
# instead. BEST_EFFORT also matches what most camera publishers (incl.
# zed_wrapper's image_transport) use, so the QoS handshake actually matches.
LATEST_IMAGE_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
)


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

    SINGLE_LINE — when only one boundary is fresh:
        target path = visible boundary shifted toward the lane center by the
        learned half-width. Offset direction is chosen from the line's actual
        body-frame median y (and a recent-midpoint anchor when available),
        not from the topic label, so a line straddling y=0 still produces a
        correct center-seeking target.

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
        self.declare_parameter("lookahead_distance", 1.68)
        self.declare_parameter("lost_line_lookahead_distance", 0.9)
        self.declare_parameter("min_lookahead_distance", 0.5)

        # Speed pinned at the recorded constant 3.5 m/s (min == max ==
        # nominal disables the curvature-driven slowdown loop).
        self.declare_parameter("nominal_speed", 3.5)
        self.declare_parameter("lost_line_speed", 3.5)
        self.declare_parameter("min_speed", 3.5)
        self.declare_parameter("max_speed", 3.5)

        # Wider steering envelope so the controller can reproduce the
        # recorded -0.148 rad excursion without clipping.
        self.declare_parameter("max_steering_angle", 0.20)
        # Curvature speed-down off (recorded driver did not slow on curves).
        self.declare_parameter("curvature_speed_gain", 0.0)
        self.declare_parameter("curvature_lookahead_gain", 2.0)

        # Initial half-lane-width matches the bag's apparent ~0.6 m gap
        # between the two visible white stripes (was 0.5 generic).
        self.declare_parameter("half_width_init", 0.30)
        # EMA learning rate for half-width
        self.declare_parameter("half_width_alpha", 0.1)
        # Plausibility window for the lane half-width. The bag's apparent
        # half-width sits at 0.30 m, so the acceptance window starts at
        # 0.20 m to keep BILATERAL active on this geometry.
        self.declare_parameter("half_width_min", 0.20)
        self.declare_parameter("half_width_max", 0.65)

        # Wider freshness/staleness windows than the previous 0.2/0.75 s
        # so a 200–700 ms camera dropout does not knock the controller
        # into STALE on every gap. Pair with stop_if_no_path=False below
        # so we never emit zero-speed during one of those gaps.
        self.declare_parameter("stale_path_timeout", 1.50)
        self.declare_parameter("fresh_msg_timeout", 0.80)
        # How long to keep using the last bilateral midpoint after one side
        # momentarily drops out (BILATERAL_HOLD coverage). 1.5 s covers
        # virtually every short single-side blackout in the Johnson Track
        # bag and lifts effective bilateral coverage from 80% to 98.85%.
        self.declare_parameter("bilateral_hold_window", 1.5)
        # Control loop rate. 50 Hz beats the old 20 Hz hardcoded value
        # on smoothness and the recorded-vs-synth correlation; per-tick
        # math is well under 1 ms.
        self.declare_parameter("control_rate_hz", 50.0)
        # Minimum arc length of a single boundary path to be useful (meters).
        # Set to 0 so the controller accepts whatever the calibration-GUI-
        # equivalent detection pipeline publishes — even a 2-point spine
        # that projects to a tiny forward arc still represents a real line.
        self.declare_parameter("min_path_arc_length", 0.0)
        # If no valid path available, stop. Default False so a single bad
        # frame in the middle of an otherwise-good run doesn't slam the
        # brakes; flip to True if you want a hard safety-stop on perception
        # loss.
        self.declare_parameter("stop_if_no_path", False)
        # Steering smoothing factor (0 = no smoothing, 1 = instant). 0.20
        # damps detector jitter without lagging real corrections.
        self.declare_parameter("steering_alpha", 0.20)
        # Target low-pass filter factor. Smooths mode-switch jumps and
        # frame-to-frame jitter on the lookahead point. Smaller = more damping.
        self.declare_parameter("target_alpha", 0.20)
        # Cross-track-error feedback gain (rad/m). Default 0.0 — with a
        # well-calibrated camera_y_offset the BILATERAL midpoint already
        # sits at y≈0 so CTE has nothing to do; non-zero gain just amplifies
        # detector noise into per-frame steering wobble. Bump to 0.5–1.0
        # for live driving if active recovery from large lateral pushes
        # is needed.
        # Pure pursuit at lookahead
        # ≈1.2 m has a large turn radius for a forward-aligned offset, so a
        # 0.3 m lateral error only yields ~7° of steering — the robot drifts
        # back to center very slowly. This term adds a direct P-feedback on
        # the path's lateral position at the car (intercept-y), which makes
        # the controller actively pull toward center even when heading is
        # already aligned with the lane.
        self.declare_parameter("cte_gain", 0.0)
        # Lateral offset (m) added to incoming path-y so the controller
        # works in a robot-center frame. +y = LEFT in REP-103. The Johnson
        # Track empirical sweep landed at -0.22 m, which combines the
        # camera's mechanical mount offset (~-0.06 m) with a residual
        # detector-+homography asymmetry on this bag. Recalibrate per
        # camera mount and per track by running bag_replay/sweep_offset.py
        # against a fresh recording and reading the BEST y_off from its
        # output.
        self.declare_parameter("camera_y_offset", -0.22)

        # Visualization — gated off by default. The /TEST_FEED overlay runs
        # detect_white_lines a second time per control tick (20 Hz on top of
        # lane_detector's per-frame run) and JPEG-encodes the annotated
        # frame; both are noticeable on the Jetson and are the prime suspects
        # for control-loop lag. Enable by setting `enable_visualization:=true`.
        self.declare_parameter("enable_visualization", False)
        self.declare_parameter(
            "image_topic", "/zed/zed_node/rgb/image_rect_color/compressed"
        )
        self.declare_parameter("test_feed_topic", "/TEST_FEED")
        # JPEG quality used to publish /TEST_FEED/compressed.
        self.declare_parameter("test_feed_jpeg_quality", 40)

        left_line_topic = self.get_parameter("left_line_topic").value
        right_line_topic = self.get_parameter("right_line_topic").value
        drive_topic = self.get_parameter("drive_topic").value
        image_topic = self.get_parameter("image_topic").value
        test_feed_topic = self.get_parameter("test_feed_topic").value

        self._load_tunable_params()

        # Callback groups — control timer + path subs share one
        # MutuallyExclusive group so they serialize w.r.t. each other (the
        # control loop reads `latest_*_path` written by the subs, and the
        # original single-threaded executor was the only thing keeping that
        # safe). The image callback lives in its own Reentrant group so a
        # slow JPEG decode never blocks the 20 Hz control loop.
        self.control_cbgroup = MutuallyExclusiveCallbackGroup()
        self.image_cbgroup = ReentrantCallbackGroup()

        # Subscriptions
        self.left_sub = self.create_subscription(
            Path, left_line_topic, self.left_line_callback, 10,
            callback_group=self.control_cbgroup,
        )
        self.right_sub = self.create_subscription(
            Path, right_line_topic, self.right_line_callback, 10,
            callback_group=self.control_cbgroup,
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )
        self.marker_pub = self.create_publisher(
            Marker, "/lookahead_target", 10
        )

        # Visualization: only wire up the camera subscription, projection,
        # and /TEST_FEED publisher when explicitly enabled. With it off we
        # skip the per-tick blob run, draw, and JPEG encode entirely.
        self.bridge = CvBridge()
        self.H = build_homography()
        self.H_inv = np.linalg.inv(self.H)
        self.latest_image: Optional[np.ndarray] = None
        self.enable_visualization = bool(
            self.get_parameter("enable_visualization").value
        )
        self.test_feed_jpeg_quality = int(
            self.get_parameter("test_feed_jpeg_quality").value
        )
        if self.enable_visualization:
            self.image_sub = self.create_subscription(
                CompressedImage, image_topic, self.image_callback,
                LATEST_IMAGE_QOS,
                callback_group=self.image_cbgroup,
            )
            self.test_feed_pub = self.create_publisher(
                CompressedImage, f"{test_feed_topic}/compressed", 10
            )
            self.get_logger().info(
                f"Visualization ON: subscribed to {image_topic}, "
                f"publishing to {test_feed_topic}/compressed."
            )
        else:
            self.image_sub = None
            self.test_feed_pub = None
            self.get_logger().info(
                "Visualization OFF (set enable_visualization:=true to "
                "publish /TEST_FEED/compressed)."
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
        # BILATERAL→SINGLE_LINE can shift target ~half-width laterally)
        self.prev_target: Optional[Tuple[float, float]] = None

        # Control loop — rate from parameter (default 20 Hz).
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        if control_rate_hz <= 0:
            control_rate_hz = 20.0
        self.control_period_sec = 1.0 / control_rate_hz
        self.control_timer = self.create_timer(
            self.control_period_sec, self.control_loop,
            callback_group=self.control_cbgroup,
        )

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
        self.bilateral_hold_window = float(
            self.get_parameter("bilateral_hold_window").value
        )
        self.min_path_arc_length = float(self.get_parameter("min_path_arc_length").value)
        self.stop_if_no_path = bool(self.get_parameter("stop_if_no_path").value)
        self.steering_alpha = float(self.get_parameter("steering_alpha").value)
        self.target_alpha = float(self.get_parameter("target_alpha").value)
        self.cte_gain = float(self.get_parameter("cte_gain").value)
        self.camera_y_offset = float(self.get_parameter("camera_y_offset").value)

    def _on_param_change(self, params) -> SetParametersResult:
        float_params = {
            "wheelbase", "lookahead_distance", "lost_line_lookahead_distance",
            "min_lookahead_distance", "nominal_speed", "lost_line_speed",
            "min_speed", "max_speed", "max_steering_angle",
            "curvature_speed_gain", "curvature_lookahead_gain",
            "half_width_init", "half_width_alpha",
            "half_width_min", "half_width_max",
            "stale_path_timeout", "fresh_msg_timeout",
            "bilateral_hold_window",
            "min_path_arc_length", "steering_alpha", "target_alpha",
            "cte_gain", "camera_y_offset",
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

    def image_callback(self, msg: CompressedImage) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("cv2.imdecode returned None")
            self.latest_image = decoded
        except Exception as e:
            self.get_logger().warn(f"image_callback decode failed: {e}",
                                   throttle_duration_sec=2.0)

    def extract_valid_points(self, msg: Path) -> List[Tuple[float, float]]:
        # Shift each point from the camera frame (origin at the ZED's LEFT
        # lens) into a robot-center frame. Without this, the controller aims
        # for the camera centerline — which sits a few cm left of the rig's
        # actual center — and the resulting bias makes near-zero detections
        # asymmetric (the left lane line crosses y=0 first when the robot is
        # offset, the right one stays clearly negative).
        dy = self.camera_y_offset
        points: List[Tuple[float, float]] = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if x < -0.2:
                continue
            points.append((x, y + dy))
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
    # Side resolution — trust geometry, not topic labels
    # -------------------------------------------------
    def _recent_midpoint_y(self) -> Optional[float]:
        """Median body-frame y of the last good midpoint, if it's still recent.
        Used as an anchor to disambiguate a single line whose median y has
        crossed zero (which inverts what lane_detector publishes)."""
        import numpy as np
        if (len(self.last_good_midpoint) < 2
                or self.last_good_midpoint_time is None):
            return None
        age = (self.get_clock().now()
               - self.last_good_midpoint_time).nanoseconds * 1e-9
        if age > self.stale_path_timeout:
            return None
        return float(np.median([p[1] for p in self.last_good_midpoint]))

    def _sort_by_median_y(
        self,
        L: Optional[List[Tuple[float, float]]],
        R: Optional[List[Tuple[float, float]]],
    ) -> Tuple[Optional[List[Tuple[float, float]]],
               Optional[List[Tuple[float, float]]]]:
        """Reassign the two paths so the higher-median-y path is in slot L.

        Why: lane_detector classifies lines by `median_y >= 0` with no
        dead-band. A line that straddles y=0 — which happens when the car is
        right next to a lane line — flips between /left_lane_line and
        /right_lane_line frame to frame. Sorting by actual median y here makes
        the BILATERAL midpoint and the implausible-width drop independent of
        whichever topic the line came in on.
        """
        if L is None or R is None:
            return L, R
        import numpy as np
        med_l = float(np.median([p[1] for p in L]))
        med_r = float(np.median([p[1] for p in R]))
        if med_l < med_r:
            return R, L
        return L, R

    def _signed_offset_to_center(
        self,
        pts: List[Tuple[float, float]],
        half_width: float,
    ) -> List[Tuple[float, float]]:
        """Shift a single-boundary path by half_width along ±y so the result
        sits at the lane center. Direction is chosen from the path's actual
        body-frame y — not from the topic name — so a misclassified line
        (median y straddling zero) still produces a center-seeking target.

        Priority: if a recent midpoint exists, offset toward that anchor;
        otherwise offset toward y=0 (correct for the in-lane case)."""
        import numpy as np
        if len(pts) < 1 or abs(half_width) < 1e-6:
            return list(pts)
        m = float(np.median([p[1] for p in pts]))

        anchor = self._recent_midpoint_y()
        if anchor is not None:
            sign = 1.0 if anchor > m else -1.0
        else:
            sign = -1.0 if m >= 0 else 1.0
        return [(x, y + sign * half_width) for (x, y) in pts]

    # -------------------------------------------------
    # Main control loop
    # -------------------------------------------------
    def control_loop(self) -> None:
        L = self._fresh_path("left")
        R = self._fresh_path("right")

        # Keep the original (pre-sort) detections around so the /TEST_FEED
        # overlay shows what each topic actually published this frame, not
        # the geometry-sorted reassignment used internally.
        raw_left = L
        raw_right = R

        # Reclassify by actual body-frame y. This neutralises the topic-label
        # flip caused by lane_detector's no-dead-band sign cutoff at y=0,
        # which would otherwise invert downstream offsets when the car is
        # hugging a boundary.
        L, R = self._sort_by_median_y(L, R)

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
                # Width implausible — drop one side. Prefer the path whose
                # median y matches where we expect a boundary to be (anchor
                # ± half_width); otherwise fall back to "keep the side
                # closer to y=0", which assumes that side is the real
                # adjacent boundary.
                import numpy as np
                med_l = float(np.median([p[1] for p in L]))
                med_r = float(np.median([p[1] for p in R]))
                anchor = self._recent_midpoint_y()
                if anchor is not None:
                    expected_l = anchor + self.half_width
                    expected_r = anchor - self.half_width
                    err_l = abs(med_l - expected_l)
                    err_r = abs(med_r - expected_r)
                    if err_l <= err_r:
                        R = None
                    else:
                        L = None
                else:
                    if abs(med_l) <= abs(med_r):
                        R = None
                    else:
                        L = None

        # Hysteresis: if a fresh bilateral midpoint exists in the very
        # recent past, keep using it instead of dropping to a single-side
        # offset path. This eats the BILATERAL ↔ single-side oscillation
        # that happens when a frame loses one boundary momentarily.
        if (path_to_follow is None
                and len(self.last_good_midpoint) >= 2
                and self.last_good_midpoint_time is not None):
            age = (self.get_clock().now()
                   - self.last_good_midpoint_time).nanoseconds * 1e-9
            if age <= self.bilateral_hold_window and (L is not None or R is not None):
                path_to_follow = self.last_good_midpoint
                mode = "BILATERAL_HOLD"

        # ── Priority 3: SINGLE_LINE — only one boundary visible ─────────
        # Offset direction comes from the line's actual body-frame y (via
        # _signed_offset_to_center), not from which slot it's in. A line
        # whose median y is on the "wrong" side after _sort_by_median_y
        # still produces a center-seeking target.
        if path_to_follow is None:
            solo = L if L is not None else R
            if solo is not None:
                offset = self._signed_offset_to_center(solo, self.half_width)
                if len(offset) >= 2:
                    path_to_follow = offset
                    mode = "SINGLE_LINE"
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
            self._publish_visualization(
                raw_left, raw_right, None, None, "NONE",
                0.0, 0.0, 0.0, 0.0,
            )
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
            self._publish_visualization(
                raw_left, raw_right, path_to_follow, None, mode,
                0.0, 0.0, 0.0, lookahead,
            )
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

        # Cross-track error: the path's lateral position at the car (closest
        # path point's y). Positive = path is to the left of the car, so to
        # close the error the car must steer left → positive added steering.
        cte = self._compute_cte(path_to_follow)

        steering_angle, curvature = self.compute_pure_pursuit_command(target, cte)

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
            f"la={lookahead:.2f} cte={cte:+.3f} curv={curvature:.3f}",
            throttle_duration_sec=0.5,
        )

        self._publish_visualization(
            raw_left, raw_right, path_to_follow, target, mode,
            steering_angle, speed, cte, lookahead,
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
    # Cross-track error
    # -------------------------------------------------
    def _compute_cte(self, path: List[Tuple[float, float]]) -> float:
        """Cross-track error: the path's lateral y at the car. Used as a
        direct P-feedback into steering so the controller actively pulls
        toward the lane center even when the heading is already aligned —
        pure pursuit alone has too large a turn radius for that case.

        Uses polyfit intercept (path's y at x=0) when the path has enough
        x-spread; otherwise falls back to the closest-point's y."""
        if len(path) < 1:
            return 0.0
        if len(path) >= 2:
            import numpy as np
            xs = np.array([p[0] for p in path])
            if float(np.ptp(xs)) >= 0.05:
                ys = np.array([p[1] for p in path])
                _, intercept = np.polyfit(xs, ys, 1)
                return float(intercept)
        return float(path[self.find_closest_point_index(path)][1])

    # -------------------------------------------------
    # Pure pursuit math
    # -------------------------------------------------
    def compute_pure_pursuit_command(
        self,
        target: Tuple[float, float],
        cte: float = 0.0,
    ) -> Tuple[float, float]:
        tx, ty = target
        ld = math.hypot(tx, ty)
        ld = max(ld, 1e-3)

        alpha = math.atan2(ty, tx)
        curvature = 2.0 * math.sin(alpha) / ld

        pp_steer = math.atan(self.wheelbase * curvature)

        # Direct cross-track-error feedback. Pure pursuit at the configured
        # 1.2 m lookahead has a 2.5 m turn radius for a 0.3 m forward-aligned
        # offset (≈7° steering), so the robot drifts back to center over
        # several seconds. The CTE term adds an angle proportional to the
        # path's lateral position at the car, so the controller commits to a
        # firm correction immediately on detecting an offset.
        cte_steer = self.cte_gain * cte

        steering_angle = clamp(
            pp_steer + cte_steer,
            -self.max_steering_angle, self.max_steering_angle,
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

    # -------------------------------------------------
    # Visualization on /TEST_FEED
    # -------------------------------------------------
    def _project_xy_to_uv(
        self, x: float, y: float
    ) -> Optional[Tuple[int, int]]:
        """Project a body-frame (x, y) — robot-centered — to image pixel (u, v).

        Body frame is robot-centered: y has had camera_y_offset added to it
        in extract_valid_points(). The homography was calibrated in the
        camera's own ground frame, so we subtract that offset back out
        before applying H_inv.

        Returns None for points behind the camera (x ≤ 0), points whose
        homogeneous denominator vanishes, or anything past a sane far range
        (which projects very near the horizon and isn't useful to draw)."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if x <= 0.05:
            return None
        y_cam = y - self.camera_y_offset
        p = self.H_inv @ np.array([x, y_cam, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u = p[0] / p[2]
        v = p[1] / p[2]
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return (int(round(u)), int(round(v)))

    def _draw_path_on_image(
        self,
        img: np.ndarray,
        pts: Optional[List[Tuple[float, float]]],
        color: Tuple[int, int, int],
        thickness: int = 2,
        radius: int = 3,
    ) -> None:
        if pts is None or len(pts) < 1:
            return
        h, w = img.shape[:2]
        uvs: List[Tuple[int, int]] = []
        for (x, y) in pts:
            uv = self._project_xy_to_uv(x, y)
            if uv is None:
                continue
            u, v = uv
            if -50 <= u <= w + 50 and -50 <= v <= h + 50:
                uvs.append((u, v))
        for (u, v) in uvs:
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (u, v), radius, color, -1)
        if len(uvs) >= 2:
            arr = np.array(uvs, dtype=np.int32)
            cv2.polylines(img, [arr], False, color, thickness, lineType=cv2.LINE_AA)

    def _publish_visualization(
        self,
        raw_left: Optional[List[Tuple[float, float]]],
        raw_right: Optional[List[Tuple[float, float]]],
        path: Optional[List[Tuple[float, float]]],
        target: Optional[Tuple[float, float]],
        mode: str,
        steering: float,
        speed: float,
        cte: float,
        lookahead: float,
    ) -> None:
        if not self.enable_visualization:
            return
        if self.latest_image is None:
            return

        img = self.latest_image.copy()
        h, w = img.shape[:2]

        # BGR colors
        GRAY     = (170, 170, 170)  # raw blob spines (no filters at all)
        DIM_BLUE = (140, 70, 0)     # latest /left_lane_line, regardless of freshness
        DIM_RED  = (0, 50, 140)     # latest /right_lane_line, regardless of freshness
        BLUE     = (255, 120, 0)    # left path the controller is using right now
        RED      = (0, 80, 255)     # right path the controller is using right now
        GREEN    = (0, 255, 0)      # chosen midpoint / followed path
        CYAN     = (255, 255, 0)
        YELLOW   = (0, 255, 255)
        WHITE    = (255, 255, 255)
        BLACK    = (0, 0, 0)

        # ── Layer A: raw blob spines, drawn from the live image with the
        # same call the calibration GUI uses. No homography, no filters.
        # If you see gray dots and no path on top, the line is being
        # killed somewhere in lane_detector or _fresh_path.
        raw_spines: List[List[Tuple[int, int]]] = []
        try:
            from final_challenge import white_line_detection as wld
            raw_spines = wld.detect_white_lines(self.latest_image)
        except Exception as e:
            self.get_logger().warn(
                f"blob debug overlay failed: {e}",
                throttle_duration_sec=5.0)
        for spine in raw_spines:
            for (sx, sy) in spine:
                if 0 <= sx < w and 0 <= sy < h:
                    cv2.circle(img, (sx, sy), 1, GRAY, -1)

        # ── Layer B: latest published Path messages, drawn regardless of
        # freshness or arc-length filtering. Faded so we can compare them
        # to layer C (what the controller is willing to use). If layer B
        # shows the line but layer C doesn't, _fresh_path dropped it.
        self._draw_path_on_image(img, self.latest_left_path,
                                 DIM_BLUE, thickness=1, radius=2)
        self._draw_path_on_image(img, self.latest_right_path,
                                 DIM_RED, thickness=1, radius=2)

        # ── Layer C: paths the controller is using this tick.
        self._draw_path_on_image(img, raw_left,  BLUE, thickness=2, radius=3)
        self._draw_path_on_image(img, raw_right, RED,  thickness=2, radius=3)

        # Path the controller is actually following.
        path_color = GREEN
        if mode == "STALE":
            path_color = (0, 0, 255)  # bright red when following stale memory
        elif mode == "SINGLE_LINE":
            path_color = CYAN
        self._draw_path_on_image(img, path, path_color, thickness=3, radius=4)

        # Lookahead target — yellow ring + spoke from bottom-center.
        if target is not None:
            uv = self._project_xy_to_uv(target[0], target[1])
            if uv is not None:
                u, v = uv
                if 0 <= u < w and 0 <= v < h:
                    cv2.line(img, (w // 2, h - 1), (u, v),
                             YELLOW, 1, lineType=cv2.LINE_AA)
                    cv2.circle(img, (u, v), 14, YELLOW, 2, lineType=cv2.LINE_AA)
                    cv2.circle(img, (u, v), 4, YELLOW, -1)

        # Mode-colored header.
        if mode == "BILATERAL":
            mode_color = GREEN
        elif mode == "BILATERAL_HOLD":
            mode_color = CYAN
        elif mode == "SINGLE_LINE":
            mode_color = YELLOW
        elif mode == "STALE":
            mode_color = (0, 0, 255)
        else:
            mode_color = (200, 200, 200)

        target_str = (f"({target[0]:.2f}, {target[1]:+.2f}) m"
                      if target is not None else "---")

        # Counts at each stage so the gap is obvious in text too.
        spine_pt_total = sum(len(s) for s in raw_spines)
        text_lines = [
            (f"MODE: {mode}", mode_color),
            (f"steer={steering:+.3f} rad   speed={speed:.2f} m/s", WHITE),
            (f"hw={self.half_width:.2f} m   la={lookahead:.2f} m   cte={cte:+.3f} m",
             WHITE),
            (f"target={target_str}", WHITE),
            (f"blob: {len(raw_spines)} spines / {spine_pt_total} pts   "
             f"published L/R: {len(self.latest_left_path)}/{len(self.latest_right_path)} pts   "
             f"controller L/R: {len(raw_left or [])}/{len(raw_right or [])} pts",
             (210, 210, 210)),
            ("gray=raw blob   faded=lane_detector publish   "
             "bright=controller in use", (180, 180, 180)),
        ]
        y0 = 22
        for i, (line, col) in enumerate(text_lines):
            y = y0 + i * 22
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, BLACK, 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)

        try:
            ok, buf = cv2.imencode(
                ".jpg", img,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.test_feed_jpeg_quality],
            )
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            out = CompressedImage()
            out.header.stamp = self.get_clock().now().to_msg()
            out.format = "jpeg"
            out.data = buf.tobytes()
            self.test_feed_pub.publish(out)
        except Exception as e:
            self.get_logger().warn(f"TEST_FEED publish failed: {e}",
                                   throttle_duration_sec=2.0)

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

        # green=BILATERAL, cyan=BILATERAL_HOLD, yellow=SINGLE_LINE, red=STALE
        if mode == "BILATERAL":
            m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
        elif mode == "BILATERAL_HOLD":
            m.color.r, m.color.g, m.color.b = 0.0, 1.0, 1.0
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
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


# ── Calibration GUI ──────────────────────────────────────────────────────────
# White-only HSV calibrator that runs against the live ZED feed.
#
# How to use
# ----------
#     # ROS must be sourced first; run as a module from the package root so
#     # the `from final_challenge.x import y` imports resolve.
#     cd <path>/final_challenge2026
#     python3 -m final_challenge.lane_follower
#
#     # Or, if you've colcon-built and sourced the workspace, run from anywhere:
#     source install/setup.bash
#     python3 -m final_challenge.lane_follower
#
#     The window has a button bar across the top, a side-by-side image below
#     (left = original + lane-detection overlay, right = white filter only),
#     and HSV trackbars + a Min_Area trackbar.
#
#     Fastest path:
#       1. Point the camera at a white lane stripe.
#       2. LEFT-click on a lit white pixel.  A 15x15 patch around the click
#          tightens the saturation upper bound and the value lower bound so
#          the band encloses those pixels.  Hue stays at the full [0, 179]
#          range — saturation/value are what define "white."
#       3. Tune the Min_Area threshold.  The RIGHT panel boxes every contour
#          the filter let through — thick when the area clears Min_Area,
#          thin when it doesn't.
#       4. Click [Print Vals]; paste the printed lines over the constants
#          at the top of final_challenge/white_line_detection.py.
#
# This file's __main__ block launches the GUI.  Running the ROS2 node is
# done via `ros2 run final_challenge boundary_pure_pursuit` (which calls
# main() above through the entry point in setup.py).

ZED_IMAGE_TOPIC = "/zed/zed_node/rgb/image_rect_color/compressed"

CALIB_TOOLBAR_HEIGHT = 44
CALIB_BUTTON_HEIGHT  = 30
CALIB_BUTTON_Y       = (CALIB_TOOLBAR_HEIGHT - CALIB_BUTTON_HEIGHT) // 2

CALIB_BUTTONS = [
    {"id": "white", "label": "White",      "x": 10,  "w": 70},
    {"id": "print", "label": "Print Vals", "x": 85,  "w": 100},
    {"id": "quit",  "label": "Quit",       "x": 190, "w": 60},
]

CALIB_AUTOCAL_PATCH_HALF = 7
CALIB_AUTOCAL_SAT_PAD    = 40
CALIB_AUTOCAL_VAL_PAD    = 50

CALIB_MIN_AREA_TRACKBAR_MAX      = 5000
CALIB_MIN_LONG_SIDE_TRACKBAR_MAX = 500

# Defaults mirror the constants at the top of white_line_detection.py.
CALIB_DEFAULT_LOW       = [0,   0,   160]
CALIB_DEFAULT_HIGH      = [179, 60,  255]
CALIB_DEFAULT_AREA      = 500
CALIB_DEFAULT_LONG_SIDE = 40

CALIB_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


class _ZedImageGrabber(Node):
    """Minimal node — subscribe to the ZED rectified RGB feed and keep the
    most recent BGR frame for the GUI thread to consume."""

    def __init__(self, topic: str = ZED_IMAGE_TOPIC) -> None:
        super().__init__("lane_follower_calibrator")
        self.bridge = CvBridge()
        self.latest: Optional[np.ndarray] = None
        self.create_subscription(
            CompressedImage, topic, self._cb, LATEST_IMAGE_QOS
        )
        self.get_logger().info(f"Calibrator subscribed to {topic}")

    def _cb(self, msg: CompressedImage) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("cv2.imdecode returned None")
            self.latest = decoded
        except Exception as e:
            self.get_logger().warn(
                f"image decode failed: {e}", throttle_duration_sec=2.0)


def _calib_clean(mask: np.ndarray) -> np.ndarray:
    """Open then close — matches _morphological_cleanup in white_line_detection."""
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  CALIB_MORPH_KERNEL, iterations=1)
    out = cv2.morphologyEx(out,  cv2.MORPH_CLOSE, CALIB_MORPH_KERNEL, iterations=2)
    return out


def _calib_segment_white(bgr: np.ndarray, low, high) -> np.ndarray:
    """HSV inRange + morphology for the white filter on a BGR frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return _calib_clean(cv2.inRange(
        hsv,
        np.asarray(low,  dtype=np.uint8),
        np.asarray(high, dtype=np.uint8),
    ))


def _calib_print_values(low, high, min_area, min_long_side) -> None:
    print("\n# Paste these over the constants in white_line_detection.py:")
    print(f"HSV_LOW         = np.array([{low[0]:>3}, {low[1]:>3}, {low[2]:>3}])")
    print(f"HSV_HIGH        = np.array([{high[0]:>3}, {high[1]:>3}, {high[2]:>3}])")
    print(f"MIN_AREA        = {min_area}")
    print(f"MIN_LONG_SIDE   = {min_long_side}\n")


def _calib_autocal_white(state, img, x, y, half=CALIB_AUTOCAL_PATCH_HALF) -> None:
    """Update the white HSV bounds from a patch around (x, y).

    For white the hue is unstable — low saturation makes H jitter across the
    full range — so we leave H at [0, 179] and only constrain S (upper) and
    V (lower) from the patch.  That matches how white_line_detection.py
    defines its default band: any hue, low sat, bright."""
    h, w = img.shape[:2]
    x0 = max(0, x - half); x1 = min(w, x + half + 1)
    y0 = max(0, y - half); y1 = min(h, y + half + 1)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return

    hsv_patch = cv2.cvtColor(cv2.GaussianBlur(patch, (3, 3), 0),
                             cv2.COLOR_BGR2HSV)
    S = hsv_patch[..., 1].ravel()
    V = hsv_patch[..., 2].ravel()
    s_high = min(255, int(np.percentile(S, 90)) + CALIB_AUTOCAL_SAT_PAD)
    v_low  = max(0,   int(np.percentile(V, 10)) - CALIB_AUTOCAL_VAL_PAD)
    state["low"]  = [0,   0,      v_low]
    state["high"] = [179, s_high, 255]
    print(f"  auto-cal white at ({x},{y}): S<={s_high}, V>={v_low}")


def _calib_draw_button(toolbar, btn, active=False) -> None:
    x, y = btn["x"], CALIB_BUTTON_Y
    w, h = btn["w"], CALIB_BUTTON_HEIGHT
    bg = (60, 110, 60) if active else (55, 55, 55)
    cv2.rectangle(toolbar, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(toolbar, (x, y), (x + w, y + h), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(btn["label"],
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(toolbar, btn["label"], (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)


def _calib_find_button(x, y):
    if not (CALIB_BUTTON_Y <= y < CALIB_BUTTON_Y + CALIB_BUTTON_HEIGHT):
        return None
    for b in CALIB_BUTTONS:
        if b["x"] <= x < b["x"] + b["w"]:
            return b["id"]
    return None


def calibrate_with_zed() -> None:
    """Live-feed white-line HSV calibration GUI using the ZED camera."""
    import threading
    from rclpy.executors import SingleThreadedExecutor
    from final_challenge import white_line_detection as wld

    rclpy.init()
    node = _ZedImageGrabber()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    state = {
        "low":           list(CALIB_DEFAULT_LOW),
        "high":          list(CALIB_DEFAULT_HIGH),
        "min_area":      CALIB_DEFAULT_AREA,
        "min_long_side": CALIB_DEFAULT_LONG_SIDE,
        "quit":          False,
    }

    win = "lane follower calibrator (white)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H_low",    win, state["low"][0],   179, lambda _: None)
    cv2.createTrackbar("S_low",    win, state["low"][1],   255, lambda _: None)
    cv2.createTrackbar("V_low",    win, state["low"][2],   255, lambda _: None)
    cv2.createTrackbar("H_high",   win, state["high"][0],  179, lambda _: None)
    cv2.createTrackbar("S_high",   win, state["high"][1],  255, lambda _: None)
    cv2.createTrackbar("V_high",   win, state["high"][2],  255, lambda _: None)
    cv2.createTrackbar("Min_Area", win, state["min_area"],
                       CALIB_MIN_AREA_TRACKBAR_MAX, lambda _: None)
    cv2.createTrackbar("Min_Long", win, state["min_long_side"],
                       CALIB_MIN_LONG_SIDE_TRACKBAR_MAX, lambda _: None)

    def push_to_trackbars():
        cv2.setTrackbarPos("H_low",  win, int(state["low"][0]))
        cv2.setTrackbarPos("S_low",  win, int(state["low"][1]))
        cv2.setTrackbarPos("V_low",  win, int(state["low"][2]))
        cv2.setTrackbarPos("H_high", win, int(state["high"][0]))
        cv2.setTrackbarPos("S_high", win, int(state["high"][1]))
        cv2.setTrackbarPos("V_high", win, int(state["high"][2]))
        cv2.setTrackbarPos("Min_Area", win, int(state["min_area"]))
        cv2.setTrackbarPos("Min_Long", win, int(state["min_long_side"]))

    def pull_from_trackbars():
        state["low"] = [
            cv2.getTrackbarPos("H_low", win),
            cv2.getTrackbarPos("S_low", win),
            cv2.getTrackbarPos("V_low", win),
        ]
        state["high"] = [
            cv2.getTrackbarPos("H_high", win),
            cv2.getTrackbarPos("S_high", win),
            cv2.getTrackbarPos("V_high", win),
        ]
        state["min_area"]      = cv2.getTrackbarPos("Min_Area", win)
        state["min_long_side"] = cv2.getTrackbarPos("Min_Long", win)

    def handle_button(btn_id):
        if btn_id == "white":
            return  # white is the only filter — button is just a label
        if btn_id == "print":
            _calib_print_values(state["low"], state["high"],
                                state["min_area"], state["min_long_side"])
        elif btn_id == "quit":
            state["quit"] = True

    def on_mouse(event, x, y, flags, _):
        if y < CALIB_TOOLBAR_HEIGHT:
            if event == cv2.EVENT_LBUTTONDOWN:
                btn = _calib_find_button(x, y)
                if btn is not None:
                    handle_button(btn)
            return

        # Below the toolbar: left-click on the LEFT panel auto-cals from a
        # patch around the click.  Anything else (incl. clicks on the right
        # filtered panel) is ignored.
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        img = node.latest
        if img is None:
            return
        h_img, w_img = img.shape[:2]
        if x >= w_img:
            return
        x_l = max(0, min(x, w_img - 1))
        y_l = max(0, min(y - CALIB_TOOLBAR_HEIGHT, h_img - 1))
        _calib_autocal_white(state, img, x_l, y_l)
        push_to_trackbars()

    cv2.setMouseCallback(win, on_mouse)
    push_to_trackbars()

    print(
        "Calibrator ready.  LEFT-click on a white lane stripe to auto-cal\n"
        "the white filter.  Buttons in the top bar drive everything else.\n"
        f"Subscribed to: {ZED_IMAGE_TOPIC}"
    )

    waiting_msg_shown = False

    try:
        while not state["quit"]:
            img = node.latest
            if img is None:
                if not waiting_msg_shown:
                    print(f"Waiting for first frame on {ZED_IMAGE_TOPIC}...")
                    waiting_msg_shown = True
                placeholder = np.zeros(
                    (CALIB_TOOLBAR_HEIGHT + 480, 640 * 2, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, "Waiting for ZED image...",
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
                cv2.imshow(win, placeholder)
                cv2.waitKey(50)
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                continue

            h, w = img.shape[:2]
            pull_from_trackbars()

            mask = _calib_segment_white(img, state["low"], state["high"])
            filtered = cv2.bitwise_and(img, img, mask=mask)

            # Match the real detector's filter chain (white_line_detection.
            # detect_white_lines): connected-components → MIN_AREA → minAreaRect
            # → MIN_LONG_SIDE → MIN_ELONGATION. Only components that pass ALL
            # of these are drawn — that way the right panel shows exactly
            # what the program "sees", not every random contour.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            total_components = max(0, num_labels - 1)
            after_area = 0
            accepted: list = []
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < state["min_area"]:
                    continue
                after_area += 1
                component_mask = np.uint8(labels == i)
                contours, _ = cv2.findContours(
                    component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                pts = np.vstack([c.reshape(-1, 2) for c in contours])
                if len(pts) < 5:
                    continue
                rect = cv2.minAreaRect(pts)
                (rw, rh) = rect[1]
                long_side  = max(rw, rh)
                short_side = min(rw, rh) + 1e-6
                if long_side < state["min_long_side"]:
                    continue
                if long_side / short_side < wld.MIN_ELONGATION:
                    continue
                accepted.append((area, rect, long_side))
            accepted.sort(key=lambda c: c[0], reverse=True)

            passes = len(accepted) > 0
            largest_area = accepted[0][0] if accepted else 0

            # ── Right panel: only the rotated min-area boxes the program
            # actually accepts as lane-line candidates.
            for area, rect, _ in accepted:
                box = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(filtered, [box], 0, (255, 255, 255), 3)
            if accepted:
                area, rect, long_side = accepted[0]
                bx, by = int(rect[0][0]), int(rect[0][1])
                cv2.putText(filtered,
                            f"{area} px  long={long_side:.0f}",
                            (max(0, bx - 60), max(12, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            status_text  = "DETECTED WHITE" if passes else "no candidates"
            status_color = (255, 255, 255) if passes else (190, 190, 190)
            cv2.putText(filtered, status_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(filtered,
                        f"largest accepted: {largest_area} px    "
                        f"area>={state['min_area']}    "
                        f"long>={state['min_long_side']}",
                        (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(filtered,
                        f"components: {total_components}  "
                        f"after area: {after_area}  "
                        f"accepted: {len(accepted)}",
                        (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # ── Left panel: original + lane-detection overlay
            # Override module globals so detect_white_lines /
            # detect_lane_lines_hough use the trackbar values.
            wld.HSV_LOW       = np.asarray(state["low"],  dtype=np.uint8)
            wld.HSV_HIGH      = np.asarray(state["high"], dtype=np.uint8)
            wld.MIN_AREA      = int(state["min_area"])
            wld.MIN_LONG_SIDE = int(state["min_long_side"])

            try:
                spines = wld.detect_white_lines(img)
            except Exception:
                spines = []
            try:
                hough_lines = wld.detect_lane_lines_hough(img)
            except Exception:
                hough_lines = []

            left = img.copy()
            roi_top = int(wld.ROI_TOP_FRAC * h)
            cv2.line(left, (0, roi_top), (w, roi_top), (128, 128, 128), 1)

            for ll in hough_lines:
                p1, p2 = ll['segment']
                cv2.line(left, p1, p2, (0, 200, 255), 2)
                cv2.circle(left, p1, 4, (0, 200, 255), -1)
                cv2.circle(left, p2, 4, (0, 200, 255), -1)

            for spine in spines:
                for (sx, sy) in spine:
                    cv2.circle(left, (sx, sy), 2, (0, 255, 0), -1)

            cv2.putText(left, "editing: white",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(left,
                        f"hough={len(hough_lines)}  blob={len(spines)}",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(left, ZED_IMAGE_TOPIC,
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            panel = np.hstack([left, filtered])
            toolbar = np.full((CALIB_TOOLBAR_HEIGHT, panel.shape[1], 3), 30,
                              dtype=np.uint8)
            for b in CALIB_BUTTONS:
                _calib_draw_button(toolbar, b, active=(b["id"] == "white"))
            hint = ("left-click on white = auto-cal   "
                    "trackbars below   boxes = passes all filters")
            cv2.putText(
                toolbar, hint,
                (max(b["x"] + b["w"] for b in CALIB_BUTTONS) + 16,
                 CALIB_BUTTON_Y + CALIB_BUTTON_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

            display = np.vstack([toolbar, panel])
            cv2.imshow(win, display)
            cv2.waitKey(20)
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    calibrate_with_zed()

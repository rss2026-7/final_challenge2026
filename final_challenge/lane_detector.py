#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage
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
MIN_VOTES = 5    # discard Hough clusters with fewer accumulator votes
MAX_ABS_Y = 1.5  # discard candidate lines whose median car-frame |y| exceeds this (outliers from
                 # near-horizon homography projection)
MIN_FORWARD_SPAN = 0.3  # forward-span gate applied to HOUGH detections only
                        # (Hough lines are extrapolated and a short fit is
                        # often noise). Blob spines skip this gate so the
                        # controller sees the same short lines the calibration
                        # GUI sees.
MAX_X_CAR        = 4.0  # cap forward distance per sample. The homography horizon is at
                        # v ≈ 138 px and ROI top is at v ≈ 150 px, so samples near the
                        # ROI top project to wildly large x_car. 4 m is well past the
                        # 1.2 m lookahead and bounds away from the projective horizon.
MAX_DY_OVER_DX = 0.4    # path must be roughly forward-aligned. Real boundaries on
                        # straights have |slope| < 0.15; mild curves up to ~0.35;
                        # transverse markers register at |slope| ≥ 0.48. 0.4 keeps
                        # gentle curves and rejects transverse markers cleanly.


def _path_is_plausible_boundary(pts):
    """Strict plausibility check used for HOUGH detections only: forward-span
    + slope cap + non-degenerate x-spread. See _spine_path_is_sane for the
    blob equivalent (no forward-span gate, since the GUI's blob overlay is
    the single source of truth)."""
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


def _spine_path_is_sane(pts):
    """Blob-spine acceptance, intentionally lenient — matches the calibration
    GUI's behavior (which displays raw blob spines with no car-frame filter).
    Only the slope cap survives, so cross-track markers that slipped through
    blob's internal _tangent_ok check still get rejected here. Short forward
    spans are accepted: a partially-occluded boundary should not be dropped
    just because the visible portion projects to <0.3 m of forward extent."""
    if len(pts) < 2:
        return False
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if np.ptp(xs) < 1e-6:
        return False
    slope = np.polyfit(xs, ys, 1)[0]
    if abs(slope) > MAX_DY_OVER_DX:
        return False
    return True


def _image_space_slope(spine):
    """du/dv slope of an image-space spine (u = horizontal pixel, v = vertical
    pixel; v increases downward).

    Sign convention used for single-line classification:
      du/dv > 0  → line slants down-and-to-the-right in the image. The bottom
                   end of the line sits to the RIGHT of its top end. This is
                   what a RIGHT lane boundary looks like in standard image
                   coords.
      du/dv < 0  → line slants down-and-to-the-left → LEFT boundary.
      du/dv ≈ 0  → near-vertical line; angle is ambiguous and we should fall
                   back to lateral-position classification.
    """
    if len(spine) < 2:
        return 0.0
    us = np.array([p[0] for p in spine], dtype=float)
    vs = np.array([p[1] for p in spine], dtype=float)
    if np.ptp(vs) < 2.0:
        return 0.0
    return float(np.polyfit(vs, us, 1)[0])


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
    1:1 with the calibration GUI: detect_white_lines() (blob) is the only
    detector. Whatever spines the GUI overlays as green dots are the spines
    we project through the homography and publish — no slope cap, no
    forward-span gate, no median-y cut. The blob detector's own internal
    filters (MIN_AREA → MIN_LONG_SIDE → MIN_ELONGATION → _tangent_ok) are
    the single source of truth for what counts as a lane line. Hough is
    intentionally not used here.

    Single-line case: classified by image-space tilt (du/dv). Falling back
    to lateral position only when the line is too close to vertical for the
    angle to be unambiguous.

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

        # BEST_EFFORT, KEEP_LAST(1): always work on the newest frame; if
        # we ever fall behind, drop the backlog at the DDS layer rather
        # than queueing stale images that produce stale lane detections.
        latest_image_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/zed/zed_node/rgb/image_rect_color/compressed",
            self.image_callback,
            latest_image_qos,
        )

        self.get_logger().info("LaneDetector initialised.")

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn(
                "lane_detector: cv2.imdecode returned None",
                throttle_duration_sec=2.0,
            )
            return
        h, w  = frame.shape[:2]
        roi_top = int(ROI_TOP_FRAC * h)
        stamp = msg.header.stamp

        (left_pts, right_pts,
         debug_lines, blob_spines) = self._detect_boundaries(
            frame, h, w, roi_top
        )

        self.left_pub.publish(_make_path(left_pts,  stamp))
        self.right_pub.publish(_make_path(right_pts, stamp))
        self._publish_debug(frame, debug_lines, blob_spines, roi_top)

    # ------------------------------------------------------------------
    # Boundary detection — blob only, 1:1 with the calibration GUI
    # ------------------------------------------------------------------
    def _detect_boundaries(self, frame, h, w, roi_top):
        """
        Return (left_pts, right_pts, debug_lines, blob_spines).

        Whatever the calibration GUI shows as a green spine, this method
        publishes. The only post-detection processing is the homography
        projection itself, plus dropping points that don't project into
        the car-frame strip 0 < x_car < MAX_X_CAR (the homography horizon
        — points beyond it are numerically meaningless, not a content
        filter).
        """
        blob_spines    = detect_white_lines(frame)
        blob_car_lines = self._spines_to_car_lines(blob_spines)

        if not blob_car_lines:
            self.get_logger().info(
                f"No blob spines projected to a usable path "
                f"(raw spines: {len(blob_spines)}).",
                throttle_duration_sec=1.0,
            )
            return [], [], [], blob_spines

        left_pts, right_pts = self._classify_car_lines(blob_car_lines)
        return left_pts, right_pts, [], blob_spines

    def _spines_to_car_lines(self, spines):
        """1:1 with the calibration GUI: trust whatever detect_white_lines
        returns. Project to car-frame, drop nothing on shape grounds. The
        only criterion for skipping a spine is that fewer than 2 of its
        points survived the projection's MAX_X_CAR clip — at which point
        there's no path to publish.

        Each entry carries the line's image-space slope so the single-line
        classifier can decide left/right by tilt angle, not lateral
        position alone."""
        car_lines = []
        for spine in spines:
            pts = self._spine_to_car_pts(spine)
            if len(pts) < 2:
                continue
            median_y = float(np.median([p[1] for p in pts]))
            image_slope = _image_space_slope(spine)
            car_lines.append((median_y, None, pts, image_slope))
        return car_lines

    def _hough_lines_to_car_lines(self, hough_lines, h, w, roi_top):
        """Strict plausibility for the Hough fallback path: forward-span +
        slope cap + accumulator-vote gate. Hough fits are noisier than blob
        spines, so they need the tighter filter."""
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
            # Hough returns x = m·y + b in image coords, so m IS du/dv.
            image_slope = float(ll['coeffs'][0])
            car_lines.append((median_y, ll, pts, image_slope))
        return car_lines

    # Slope (du/dv) magnitudes below this are treated as "near-vertical"
    # and fall through to lateral-position classification.
    SINGLE_LINE_SLOPE_DEADBAND = 0.05

    def _classify_car_lines(self, car_lines):
        """Two or more plausible lines: split innermost-pair as left/right
        using their lateral position.

        One line: classify by image-space tilt angle. A line slanting
        down-and-to-the-right (du/dv > 0) is a RIGHT boundary; down-and-to-
        the-left (du/dv < 0) is a LEFT boundary. Tilt is more stable than
        lateral position when the car is hugging or straddling a boundary
        (median_y can flip sign frame-to-frame in that case; the line's
        tilt cannot). Falls back to median-y sign only when the line is
        too close to vertical for the angle to be unambiguous."""
        # _split_left_right expects 3-tuples; strip the slope.
        if len(car_lines) >= 2:
            return self._split_left_right([(c[0], c[1], c[2]) for c in car_lines])

        median_y, _, pts, image_slope = car_lines[0]
        if image_slope > self.SINGLE_LINE_SLOPE_DEADBAND:
            return [], pts          # tilts down-right → RIGHT lane line
        if image_slope < -self.SINGLE_LINE_SLOPE_DEADBAND:
            return pts, []          # tilts down-left  → LEFT lane line
        # Near-vertical: angle is unreliable, fall back to lateral position.
        if median_y >= 0:
            return pts, []
        return [], pts

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
    def _publish_debug(self, frame, debug_lines, blob_spines, roi_top):
        dbg = frame.copy()
        h, w = dbg.shape[:2]

        cv2.line(dbg, (0, roi_top), (w, roi_top), (128, 128, 128), 1)

        # Blob spines as green dots (matches the calibration GUI overlay).
        for spine in blob_spines:
            for (sx, sy) in spine:
                cv2.circle(dbg, (sx, sy), 2, (0, 255, 0), -1)

        # Hough lines that survived the plausibility / dedup pass — orange.
        for ll in debug_lines:
            p1, p2 = ll['segment']
            cv2.line(dbg, p1, p2, (0, 200, 255), 2)
            cv2.circle(dbg, p1, 4, (0, 200, 255), -1)
            cv2.circle(dbg, p2, 4, (0, 200, 255), -1)

        cv2.putText(dbg,
                    f"blob={len(blob_spines)}  hough_kept={len(debug_lines)}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

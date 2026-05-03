#!/usr/bin/env python3

import math
import os
import queue
import threading
import time
from datetime import datetime

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
from ackermann_msgs.msg import AckermannDriveStamped

from final_challenge.white_line_detection import (
    detect_lane_lines_hough,
    detect_white_lines,
    ROI_TOP_FRAC,
)
from final_challenge.homography_transformer import build_homography, transform_uv_to_xy


def _resolve_lane_tune_dir() -> str:
    """Resolve the per-run tuning directory shared with lane_follower.

    Same logic as in lane_follower: LANE_TUNE_DIR wins; else reuse the
    most recent ~/lane_tune/run-* younger than 10 s; else mint a new
    timestamped one. The two nodes meet in the same folder without
    needing explicit IPC, as long as they're launched together."""
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


class _AsyncFrameWriter:
    """Drop-on-overflow async JPEG writer.

    submit() is called from the perception callback and must NOT block —
    it does a nonblocking put on a bounded queue and returns immediately.
    A daemon worker thread pulls frames, encodes JPEG, writes to disk.
    If the queue saturates (slow disk), frames are dropped rather than
    stalling the perception loop. Drops are counted and logged."""

    def __init__(self, out_dir: str, every: int = 1, jpeg_q: int = 75,
                 max_queue: int = 30, logger=None):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.every = max(1, int(every))
        self.jpeg_q = int(jpeg_q)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.written = 0
        self._submit_count = 0
        self._stopped = False
        self._logger = logger
        self._thread = threading.Thread(
            target=self._run, name="lane_tune_frame_writer", daemon=True,
        )
        self._thread.start()

    def submit(self, bgr_image: np.ndarray, stamp_ns: int) -> None:
        """stamp_ns is the camera-message timestamp; it doubles as the
        filename so CSV rows can name their corresponding frame directly."""
        if self._stopped:
            return
        self._submit_count += 1
        if self._submit_count % self.every != 0:
            return
        try:
            self._q.put_nowait((int(stamp_ns), bgr_image))
        except queue.Full:
            self.dropped += 1
            if self._logger and self.dropped % 30 == 0:
                self._logger(
                    f"[TUNE] frame writer dropped {self.dropped} frames "
                    f"(disk slower than camera; consider raising "
                    f"debug_frame_every)"
                )

    def _run(self) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q]
        while True:
            item = self._q.get()
            if item is None:
                return
            stamp_ns, img = item
            try:
                ok, buf = cv2.imencode(".jpg", img, params)
                if not ok:
                    continue
                path = os.path.join(self.out_dir, f"{stamp_ns}.jpg")
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
                self.written += 1
            except (cv2.error, OSError) as e:
                if self._logger:
                    self._logger(f"[TUNE] frame write failed: {e}")

    def close(self) -> None:
        self._stopped = True
        try:
            self._q.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)

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
        self.H_inv = np.linalg.inv(self.H)
        self.bridge = CvBridge()

        # Mirror lane_follower's midline-fit + eval params so the overlay
        # matches what the controller actually sees. Defaults must track
        # lane_follower.py defaults exactly.
        self.declare_parameter("e_y_eval_x", 1.5)
        self.declare_parameter("fit_x_min", 0.3)
        self.declare_parameter("fit_x_max", 2.3)
        self.declare_parameter("fit_n_samples", 11)
        self.declare_parameter("half_lane_width", 0.425)
        self.declare_parameter("camera_y_offset", -0.065)
        self.declare_parameter("wheelbase", 0.32)
        self.e_y_eval_x = float(self.get_parameter("e_y_eval_x").value)
        self.fit_x_min = float(self.get_parameter("fit_x_min").value)
        self.fit_x_max = float(self.get_parameter("fit_x_max").value)
        self.fit_n_samples = int(self.get_parameter("fit_n_samples").value)
        self.half_lane_width = float(self.get_parameter("half_lane_width").value)
        self.camera_y_offset = float(self.get_parameter("camera_y_offset").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)

        self.last_drive_steering = 0.0
        self.last_drive_speed = 0.0
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped, "/drive", self._drive_callback, 10,
        )

        # ── annotated-frame dump (off by default; opt in for tuning) ────
        self.declare_parameter("debug_save_frames", True)
        self.declare_parameter("debug_frame_every", 1)
        self.declare_parameter("debug_jpeg_quality", 75)
        save_frames = bool(self.get_parameter("debug_save_frames").value)
        frame_every = int(self.get_parameter("debug_frame_every").value)
        jpeg_q = int(self.get_parameter("debug_jpeg_quality").value)
        self._frame_writer = None
        if save_frames:
            try:
                tune_dir = _resolve_lane_tune_dir()
                frames_dir = os.path.join(tune_dir, "frames")
                self._frame_writer = _AsyncFrameWriter(
                    frames_dir,
                    every=frame_every,
                    jpeg_q=jpeg_q,
                    logger=lambda m: self.get_logger().warn(m),
                )
                self.get_logger().info(
                    f"[TUNE] dumping annotated frames to {frames_dir} "
                    f"(every={frame_every}, q={jpeg_q})"
                )
            except OSError as e:
                self.get_logger().warn(
                    f"[TUNE] could not start frame writer ({e}) — "
                    f"continuing without frames"
                )

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
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        self._publish_debug(frame, debug_lines, blob_spines, roi_top,
                            left_pts, right_pts, stamp_ns)

    def _drive_callback(self, msg: AckermannDriveStamped) -> None:
        self.last_drive_steering = float(msg.drive.steering_angle)
        self.last_drive_speed = float(msg.drive.speed)

    # ------------------------------------------------------------------
    # Car-frame ↔ image projection helpers (debug only)
    # ------------------------------------------------------------------
    def _xy_to_uv(self, x: float, y: float):
        """Inverse-project a car-frame ground-plane point back to image
        pixels. Returns (u, v) ints, or None if the point lies on/behind
        the homography horizon (proj[2] ≈ 0)."""
        p = self.H_inv @ np.array([x, y, 1.0], dtype=float)
        if abs(p[2]) < 1e-9:
            return None
        u = p[0] / p[2]
        v = p[1] / p[2]
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    @staticmethod
    def _interp_y_at_x(path, x: float):
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

    def _build_midline(self, left_pts, right_pts):
        """Mirror BoundaryPurePursuit._build_midline_samples so the
        overlay shows the same target the controller is steering on.
        left_pts/right_pts come in raw car frame; apply camera_y_offset
        here, since the published Path topics don't carry it."""
        dy = self.camera_y_offset
        L = [(x, y + dy) for x, y in left_pts]
        R = [(x, y + dy) for x, y in right_pts]
        n = max(2, int(self.fit_n_samples))
        xs = np.linspace(self.fit_x_min, self.fit_x_max, n)
        midline = []
        n_bi = n_single = 0
        hw = self.half_lane_width
        for x in xs:
            yl = self._interp_y_at_x(L, float(x)) if len(L) >= 2 else None
            yr = self._interp_y_at_x(R, float(x)) if len(R) >= 2 else None
            if yl is not None and yr is not None:
                midline.append((float(x), 0.5 * (yl + yr)))
                n_bi += 1
            elif yl is not None:
                midline.append((float(x), yl - hw))
                n_single += 1
            elif yr is not None:
                midline.append((float(x), yr + hw))
                n_single += 1
        if not midline:
            return [], "STALE", 0, 0
        mode = "BILATERAL" if n_bi >= max(1, n_single) else "SINGLE_LINE"
        return midline, mode, n_bi, n_single

    @staticmethod
    def _fit_parabola(midline):
        if len(midline) < 3:
            return None
        xs = np.array([p[0] for p in midline], dtype=float)
        ys = np.array([p[1] for p in midline], dtype=float)
        try:
            a, b, c = np.polyfit(xs, ys, 2)
        except (np.linalg.LinAlgError, ValueError):
            return None
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
            return None
        return float(a), float(b), float(c)

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
        ys = np.linspace(h - 1, roi_top, N_SAMPLES)
        xs = m * ys + b
        keep = (xs >= 0) & (xs <= w - 1)
        if not keep.any():
            return []
        return self._project_uv_array(xs[keep], ys[keep])

    def _spine_to_car_pts(self, spine):
        """Convert a blob spine to car-frame points, near-to-far.

        Cap x_car at MAX_X_CAR for the same reason as Hough: rows near the
        ROI top project past the homography horizon and produce noise.
        """
        if not spine:
            return []
        arr = np.asarray(spine, dtype=np.float64)
        return self._project_uv_array(arr[:, 0], arr[:, 1])

    def _project_uv_array(self, us, vs):
        """Batch homography projection for a vector of (u, v) pixel coords.
        Returns the projected points as a near-to-far list of (x, y) tuples,
        filtered to the strip 0 < x_car < MAX_X_CAR.
        """
        n = us.shape[0]
        if n == 0:
            return []
        homo = np.empty((3, n), dtype=np.float64)
        homo[0] = us
        homo[1] = vs
        homo[2] = 1.0
        proj = self.H @ homo
        w = proj[2]
        # Avoid divide-by-zero on the horizon line (proj[2] == 0).
        valid_w = np.abs(w) > 1e-12
        x_car = np.where(valid_w, proj[0] / np.where(valid_w, w, 1.0),
                         np.inf)
        y_car = np.where(valid_w, proj[1] / np.where(valid_w, w, 1.0),
                         np.inf)
        mask = valid_w & (x_car > 0.0) & (x_car < MAX_X_CAR)
        if not mask.any():
            return []
        x_car = x_car[mask]
        y_car = y_car[mask]
        order = np.argsort(x_car)
        return list(zip(x_car[order].tolist(), y_car[order].tolist()))

    # ------------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------------
    def _publish_debug(self, frame, debug_lines, blob_spines, roi_top,
                       left_pts, right_pts, stamp_ns):
        dbg = frame.copy()
        h, w = dbg.shape[:2]

        cv2.line(dbg, (0, roi_top), (w, roi_top), (128, 128, 128), 1)

        # Blob spines as small green dots (matches the calibration GUI).
        for spine in blob_spines:
            for (sx, sy) in spine:
                cv2.circle(dbg, (sx, sy), 2, (0, 255, 0), -1)

        # Hough lines that survived the plausibility / dedup pass — orange.
        for ll in debug_lines:
            p1, p2 = ll['segment']
            cv2.line(dbg, p1, p2, (0, 200, 255), 2)
            cv2.circle(dbg, p1, 4, (0, 200, 255), -1)
            cv2.circle(dbg, p2, 4, (0, 200, 255), -1)

        # Faint car-center reference line (y = 0) projected from x = 0.3
        # to 3.0 m. Lets you see lateral offset of the parabola at a glance.
        center_pts = []
        for x in np.linspace(0.3, 3.0, 14):
            uv = self._xy_to_uv(float(x), 0.0)
            if uv is not None:
                center_pts.append(uv)
        for i in range(1, len(center_pts)):
            cv2.line(dbg, center_pts[i - 1], center_pts[i],
                     (160, 160, 160), 1, cv2.LINE_AA)

        # Eval-distance lateral line at x = e_y_eval_x: from y = -1 to +1.
        eval_l = self._xy_to_uv(self.e_y_eval_x, -1.0)
        eval_r = self._xy_to_uv(self.e_y_eval_x,  1.0)
        if eval_l is not None and eval_r is not None:
            cv2.line(dbg, eval_l, eval_r, (255, 200, 100), 1, cv2.LINE_AA)

        # Classified lane-line paths (raw car-frame, no camera_y_offset).
        # Bright blue = LEFT boundary, red = RIGHT boundary. Lets you see
        # left/right misclassification and per-side density imbalance.
        for (x, y) in left_pts:
            uv = self._xy_to_uv(x, y)
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(dbg, (u, v), 4, (255, 100, 0), -1)
        for (x, y) in right_pts:
            uv = self._xy_to_uv(x, y)
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(dbg, (u, v), 4, (40, 40, 255), -1)

        # Build midline + parabola exactly as the controller does.
        midline, mode, n_bi, n_single = self._build_midline(
            left_pts, right_pts,
        )
        # Midline samples (yellow, larger).
        for (x, y) in midline:
            uv = self._xy_to_uv(x, y)
            if uv is None:
                continue
            u, v = uv
            cv2.circle(dbg, (u, v), 5, (0, 255, 255), -1)

        fit = self._fit_parabola(midline)
        e_y = float("nan")
        slope_eval = float("nan")
        kappa = float("nan")
        delta_arc = float("nan")
        if fit is not None:
            a, b, c = fit
            x_lo = self.fit_x_min
            x_hi = max(self.fit_x_max, self.e_y_eval_x + 0.1)
            curve_pts = []
            for x in np.linspace(x_lo, x_hi, 60):
                uv = self._xy_to_uv(float(x), float(a * x * x + b * x + c))
                if uv is not None:
                    curve_pts.append(uv)
            for i in range(1, len(curve_pts)):
                cv2.line(dbg, curve_pts[i - 1], curve_pts[i],
                         (255, 0, 255), 2, cv2.LINE_AA)

            x_e = self.e_y_eval_x
            y_e = a * x_e * x_e + b * x_e + c
            e_y = float(y_e)
            slope_eval = float(2.0 * a * x_e + b)
            denom = (1.0 + slope_eval * slope_eval) ** 1.5
            kappa = float(2.0 * a / denom)
            delta_arc = math.atan(self.wheelbase * kappa)

            # Eval-point target (cyan crosshair) and arrow from the
            # car-center reference at the same x. The arrow length IS
            # the lateral error e_y, in real ground-plane units.
            uv_t = self._xy_to_uv(x_e, y_e)
            uv_c = self._xy_to_uv(x_e, 0.0)
            if uv_t is not None:
                cv2.drawMarker(dbg, uv_t, (255, 255, 0),
                               cv2.MARKER_CROSS, 24, 3)
                cv2.circle(dbg, uv_t, 9, (255, 255, 0), 2)
            if uv_c is not None and uv_t is not None:
                cv2.arrowedLine(dbg, uv_c, uv_t, (255, 255, 0),
                                2, tipLength=0.18)

        # Steering arrow at bottom-center, scaled ×3 for visibility on
        # the typical 0.05–0.20 rad command range.
        delta_cmd = self.last_drive_steering
        scale = 3.0
        origin = (w // 2, h - 30)
        L_arrow = 90
        end = (
            int(origin[0] - L_arrow * math.sin(delta_cmd * scale)),
            int(origin[1] - L_arrow * math.cos(delta_cmd * scale)),
        )
        cv2.line(dbg, (origin[0], origin[1] - L_arrow),
                 (origin[0], origin[1]), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.arrowedLine(dbg, origin, end, (0, 255, 0), 3, tipLength=0.22)

        # Text overlay.
        mode_color = ((0, 255, 0) if mode == "BILATERAL"
                      else (0, 200, 255) if mode == "SINGLE_LINE"
                      else (0, 0, 255))
        text_lines = [
            (f"mode={mode}  bi={n_bi} sg={n_single}  "
             f"L={len(left_pts)} R={len(right_pts)}  blob={len(blob_spines)}",
             mode_color),
            (f"e_y={e_y:+.3f} m   slope={slope_eval:+.3f}   "
             f"kappa={kappa:+.3f}",
             (220, 220, 220)),
            (f"delta_arc={delta_arc:+.3f} rad   "
             f"drive_delta={delta_cmd:+.3f}   "
             f"speed={self.last_drive_speed:.2f}",
             (220, 220, 220)),
            (f"eval_x={self.e_y_eval_x:.2f} m   "
             f"fit=[{self.fit_x_min:.1f},{self.fit_x_max:.1f}] m",
             (180, 180, 180)),
        ]
        y0 = 18
        for i, (line, color) in enumerate(text_lines):
            cv2.putText(dbg, line, (4, y0 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Camera stamp (same value the frame file is named by + the value
        # the CSV's left/right_stamp_ns columns will record). Bottom-right.
        stamp_str = f"stamp_ns={stamp_ns}"
        (tw, th), _ = cv2.getTextSize(
            stamp_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1,
        )
        cv2.putText(dbg, stamp_str, (w - tw - 6, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                    cv2.LINE_AA)

        # Tiny legend bottom-left.
        legend = [
            ("L lane",  (255, 100, 0)),
            ("R lane",  (40, 40, 255)),
            ("midline", (0, 255, 255)),
            ("parab.",  (255, 0, 255)),
            ("target",  (255, 255, 0)),
            ("delta",   (0, 255, 0)),
        ]
        for i, (text, color) in enumerate(legend):
            yy = h - 12 - i * 14
            cv2.putText(dbg, text, (6, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

        # Hand the annotated frame to the async writer (non-blocking;
        # drops on overflow rather than stalling perception). The frame
        # is named by the camera message stamp_ns so lane_follower's
        # CSV (which logs the same stamp) maps row → file directly.
        if self._frame_writer is not None:
            self._frame_writer.submit(dbg, stamp_ns)


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
        if getattr(node, "_frame_writer", None) is not None:
            try:
                node._frame_writer.close()
                node.get_logger().info(
                    f"[TUNE] frame writer: wrote={node._frame_writer.written} "
                    f"dropped={node._frame_writer.dropped}"
                )
            except Exception:
                pass
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

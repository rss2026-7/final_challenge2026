#!/usr/bin/env python3
"""
Stoplight color segmentation — red & green light detection.

Three pieces in this file:
  1. Pure helpers     — segment_lights / find_largest_blob / detect_color
  2. ROS2 node        — StoplightDetector  (publishes "red" / "green" / "none")
  3. Calibration GUI  — fully mouse-driven HSV tuner (buttons + trackbars + ROI)

Detection model
---------------
A pixel passes the filter if it falls inside the per-color HSV bounds.
A *detection* requires more than scattered passing pixels: the largest
contiguous contour across both color masks must clear MIN_AREA.  This stops
specular highlights, stray red car paint, etc. from triggering "red".

How to calibrate (no keyboard shortcuts — everything is in the GUI)
-------------------------------------------------------------------
    python3 stoplight_detection.py testing_images/traffic_light/
    python3 stoplight_detection.py testing_images/traffic_light/3.jpeg

    The window has a button bar across the top, a side-by-side image below
    (left = original + diagnostics, right = active filter only), and HSV
    trackbars + a Min_Area trackbar.

    Fastest path:
      1. Click [Red 1].  Right-click on a lit red bulb.
         A 15×15 patch around the click sets red sub-range 1 to enclose
         those pixels.  Right-click ONLY ever updates the currently active
         filter — it never switches the active filter on you.
      2. Click [Red 2] only if your red bulbs straddle the hue seam (rare).
      3. Click [Green], right-click on a lit green bulb.
      4. Tune the Min_Area threshold.  The RIGHT panel draws a box around
         every contour the active filter let through — thick when the area
         clears Min_Area, thin when it doesn't.  Drag the Min_Area
         trackbar so real bulbs become thick boxes and noise stays thin.
      5. Click [Print Values]; paste the printed lines over the constants
         at the top of this file.

    To verify on other images: re-run the script with a different path
    (e.g. `... testing_images/traffic_light/3.jpeg`).  No navigation buttons in the GUI.

    Manual tuning:
      - Click [Red 1] / [Red 2] / [Green] to choose which interval the
        H_low/S_low/V_low/H_high/S_high/V_high trackbars edit.
      - Right panel updates live to show only the active interval's hits.
      - Left-click + drag on the LEFT panel to crop to the stoplight's
        bounding box; the filter applies only inside.  [Clear ROI] undoes.

Run modes
---------
ROS2 node (no positional args):    python3 stoplight_detection.py
Calibration GUI / demo:            python3 stoplight_detection.py <image_or_dir>
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, RegionOfInterest
    from std_msgs.msg import String
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False
    Node = object

try:
    import torch
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── Calibration values ────────────────────────────────────────────────────────
# Edit these after running the calibration GUI and clicking [Print Values].
# OpenCV uses H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255].
# Red wraps the hue seam at 0/180, so it needs two intervals.
DEFAULT_RED_LOW_1   = [0, 0, 204]
DEFAULT_RED_HIGH_1  = [86, 255, 255]
DEFAULT_RED_LOW_2   = [170, 120, 90]
DEFAULT_RED_HIGH_2  = [179, 255, 255]
DEFAULT_GREEN_LOW   = [40, 90, 90]
DEFAULT_GREEN_HIGH  = [85, 255, 255]
MIN_AREA            = 50



# ──────────────────────────────────────────────────────────────────────────────

BLUR_KSIZE   = (5, 5)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ── YOLO sign overlay (mirrors final_challenge/sign_detector.py) ─────────────
# Used to annotate the calibration GUI's LEFT panel with YOLO bounding boxes
# for the same objects the SignDetectorNode publishes.  Strictly diagnostic —
# the StoplightDetector ROS node never runs YOLO.
YOLO_MODEL_NAME     = "yolo11n.pt"
YOLO_CONF_THRESHOLD = 0.5
YOLO_TARGET_CLASSES = {"parking meter", "fire hydrant", "traffic light", "person"}
YOLO_CLASS_COLORS = {
    "parking meter": (0, 255, 0),
    "fire hydrant":  (0, 0, 255),
    "traffic light": (255, 255, 0),
    "person":        (255, 0, 255),
}


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def default_ranges():
    """Snapshot of the module-level defaults as a (mutable) ranges dict."""
    return {
        "red": [
            list(DEFAULT_RED_LOW_1),  list(DEFAULT_RED_HIGH_1),
            list(DEFAULT_RED_LOW_2),  list(DEFAULT_RED_HIGH_2),
        ],
        "green": [
            list(DEFAULT_GREEN_LOW),  list(DEFAULT_GREEN_HIGH),
        ],
    }


def _inrange(hsv, low, high):
    return cv2.inRange(hsv,
                       np.asarray(low,  dtype=np.uint8),
                       np.asarray(high, dtype=np.uint8))


def _clean(mask):
    """Open + close so speckle is removed and the lit bulb fills in."""
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL, iterations=1)
    out = cv2.morphologyEx(out,  cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    return out


def segment_lights(bgr, ranges=None):
    """
    Build per-color masks + a composite image keeping only red + green pixels.

    ranges schema:
      "red":   [low1, high1, low2, high2]   — two HSV intervals (hue wraps)
      "green": [low, high]                  — one HSV interval

    Returns ({"red": uint8 mask, "green": uint8 mask}, composite_bgr).
    """
    if ranges is None:
        ranges = default_ranges()

    blurred = cv2.GaussianBlur(bgr, BLUR_KSIZE, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    rl1, rh1, rl2, rh2 = ranges["red"]
    gl,  gh = ranges["green"]

    red_mask   = _clean(cv2.bitwise_or(_inrange(hsv, rl1, rh1),
                                       _inrange(hsv, rl2, rh2)))
    green_mask = _clean(_inrange(hsv, gl, gh))

    masks = {"red": red_mask, "green": green_mask}
    union = cv2.bitwise_or(red_mask, green_mask)
    composite = cv2.bitwise_and(bgr, bgr, mask=union)
    return masks, composite


def find_largest_blob(masks):
    """
    Return (color, area, bbox) for the single largest contiguous contour
    across both color masks.

    color : "red" | "green" | None    (None ⇔ no contours at all)
    area  : int — px², 0 if nothing found
    bbox  : (x, y, w, h) or None

    Contour area (not pixel count) is used so a halo of scattered red noise
    doesn't sum up to a fake "patch."
    """
    best_color = None
    best_area = 0
    best_bbox = None
    for color in ("red", "green"):
        contours, _ = cv2.findContours(masks[color],
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(cnt))
        if area > best_area:
            best_color = color
            best_area  = area
            best_bbox  = cv2.boundingRect(cnt)
    return best_color, best_area, best_bbox


def detect_color(masks, min_area=MIN_AREA):
    """
    Return 'red', 'green', or None — color of the largest contour across both
    masks, gated by min_area.  This is what the ROS node publishes.
    """
    color, area, _ = find_largest_blob(masks)
    if area < min_area:
        return None
    return color


def _yolo_results_to_detections(model, result):
    """Convert ultralytics Results.boxes into YoloDetection rows."""
    detections = []
    if result.boxes is None:
        return detections

    xyxy = result.boxes.xyxy
    conf = result.boxes.conf
    cls  = result.boxes.cls
    xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
    conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
    cls_np  = cls.detach().cpu().numpy()  if hasattr(cls,  "detach") else np.asarray(cls)

    for box, conf_val, cls_val in zip(xyxy_np, conf_np, cls_np):
        detections.append(YoloDetection(
            class_id=int(cls_val),
            class_name=model.names[int(cls_val)],
            confidence=float(conf_val),
            x1=int(box[0]),
            y1=int(box[1]),
            x2=int(box[2]),
            y2=int(box[3]),
        ))
    return detections


def _draw_yolo_detections(bgr, detections):
    """Draw every YoloDetection in `detections` onto `bgr` in place."""
    for det in detections:
        color = YOLO_CLASS_COLORS.get(det.class_name, (255, 255, 255))
        cv2.rectangle(bgr, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(bgr, label,
                    (det.x1, max(12, det.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return bgr


# ── ROS2 node ────────────────────────────────────────────────────────────────
class StoplightDetector(Node):
    """
    Assumes yolo_node.py is running and publishing the traffic-light ROI.
    Each frame is cropped to the latest traffic-light bbox before HSV
    segmentation; if no valid bbox has arrived within `roi_timeout_sec`,
    the node skips analysis and publishes "none".

    Subscriptions
    -------------
    /zed/zed_node/rgb/image_rect_color (sensor_msgs/Image) — overridable via
        the `image_topic` parameter.
    /yolo/traffic_light/roi (sensor_msgs/RegionOfInterest) — overridable via
        the `traffic_light_roi_topic` parameter.

    Publications
    ------------
    /stoplight/result      (std_msgs/String)    "red" | "green" | "none"
    /stoplight/segmented   (sensor_msgs/Image)  debug — red+green pixels only,
        zeroed outside the active ROI.

    HSV bounds and the min_area threshold are exposed as parameters so they
    can be tuned at launch time without editing the source.
    """

    def __init__(self):
        super().__init__("stoplight_detector")

        self.image_topic = (
            self.declare_parameter("image_topic",
                                   "/zed/zed_node/rgb/image_rect_color")
            .get_parameter_value().string_value
        )
        self.roi_topic = (
            self.declare_parameter("traffic_light_roi_topic",
                                   "/yolo/traffic_light/roi")
            .get_parameter_value().string_value
        )
        self.roi_timeout_sec = (
            self.declare_parameter("roi_timeout_sec", 2.0)
            .get_parameter_value().double_value
        )

        self.ranges = {
            "red": [
                self._declare_hsv("red_low_1",  DEFAULT_RED_LOW_1),
                self._declare_hsv("red_high_1", DEFAULT_RED_HIGH_1),
                self._declare_hsv("red_low_2",  DEFAULT_RED_LOW_2),
                self._declare_hsv("red_high_2", DEFAULT_RED_HIGH_2),
            ],
            "green": [
                self._declare_hsv("green_low",  DEFAULT_GREEN_LOW),
                self._declare_hsv("green_high", DEFAULT_GREEN_HIGH),
            ],
        }
        self.min_area = (
            self.declare_parameter("min_area", MIN_AREA)
            .get_parameter_value().integer_value
        )

        self.bridge = CvBridge()

        self._latest_roi = None        # (x, y, w, h) — most recent valid bbox
        self._latest_roi_stamp = None  # rclpy.time.Time when it arrived

        self.create_subscription(Image, self.image_topic, self._image_cb, 5)
        self.create_subscription(
            RegionOfInterest, self.roi_topic, self._roi_cb, 10)
        self.result_pub = self.create_publisher(String, "/stoplight/result",    10)
        self.seg_pub    = self.create_publisher(Image,  "/stoplight/segmented", 10)

        self.get_logger().info(
            f"StoplightDetector ready — image={self.image_topic}, "
            f"roi={self.roi_topic}, roi_timeout={self.roi_timeout_sec}s, "
            f"min_area={self.min_area}")

    def _declare_hsv(self, name, default):
        val = list(
            self.declare_parameter(name, default)
            .get_parameter_value().integer_array_value
        )
        if len(val) != 3:
            self.get_logger().warn(
                f"Parameter '{name}' has {len(val)} values, expected 3 — "
                f"falling back to default {default}")
            return list(default)
        return val

    def _roi_cb(self, msg: RegionOfInterest):
        # yolo_node publishes width=0/height=0 for "no detection this frame";
        # only treat non-empty boxes as a fresh bbox and reset the stale clock.
        if msg.width == 0 or msg.height == 0:
            return
        self._latest_roi = (
            int(msg.x_offset), int(msg.y_offset),
            int(msg.width),    int(msg.height),
        )
        self._latest_roi_stamp = self.get_clock().now()
        self.get_logger().info(
            f"traffic_light bbox: x={msg.x_offset} y={msg.y_offset} "
            f"w={msg.width} h={msg.height}")

    def _current_roi(self):
        """Return the latest bbox if it arrived within the timeout, else None."""
        if self._latest_roi is None or self._latest_roi_stamp is None:
            return None
        elapsed = (self.get_clock().now() - self._latest_roi_stamp).nanoseconds / 1e9
        if elapsed > self.roi_timeout_sec:
            return None
        return self._latest_roi

    def _image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        roi = self._current_roi()
        result_msg = String()
        composite = np.zeros_like(bgr)

        if roi is not None:
            h_img, w_img = bgr.shape[:2]
            x, y, w, h = roi
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            crop = bgr[y:y + h, x:x + w]

            masks, crop_composite = segment_lights(crop, self.ranges)
            composite[y:y + h, x:x + w] = crop_composite
            result_msg.data = detect_color(masks, self.min_area) or "none"
        else:
            result_msg.data = "none"

        self.result_pub.publish(result_msg)
        seg_msg = self.bridge.cv2_to_imgmsg(composite, encoding="bgr8")
        seg_msg.header = msg.header
        self.seg_pub.publish(seg_msg)


def main(args=None):
    if not _ROS_AVAILABLE:
        print("ROS2 (rclpy / cv_bridge) is not available in this Python "
              "environment — cannot start the StoplightDetector node.\n"
              "Tip: pass an image or directory as the first argument to run "
              "the standalone calibration GUI instead.")
        sys.exit(1)
    rclpy.init(args=args)
    node = StoplightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── Calibration GUI ──────────────────────────────────────────────────────────
TOOLBAR_HEIGHT = 44
BUTTON_HEIGHT  = 30
BUTTON_Y       = (TOOLBAR_HEIGHT - BUTTON_HEIGHT) // 2

BUTTONS = [
    {"id": "red1",  "label": "Red 1",      "x": 10,  "w": 70},
    {"id": "red2",  "label": "Red 2",      "x": 85,  "w": 70},
    {"id": "green", "label": "Green",      "x": 160, "w": 70},
    {"id": "clear", "label": "Clear ROI",  "x": 245, "w": 100},
    {"id": "print", "label": "Print Vals", "x": 350, "w": 100},
    {"id": "quit",  "label": "Quit",       "x": 460, "w": 60},
]

# Minimum contour area (px²) drawn as a candidate bbox on the filtered panel.
# Anything below this is treated as visual noise and not boxed.
CANDIDATE_MIN_AREA = 5

AUTOCAL_PATCH_HALF = 7
AUTOCAL_HUE_PAD    = 8
AUTOCAL_SAT_PAD    = 40
AUTOCAL_VAL_PAD    = 50

MIN_AREA_TRACKBAR_MAX = 5000  # upper bound of the Min_Area slider


def _collect_paths(root):
    if os.path.isdir(root):
        return sorted(
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(IMG_EXTS)
        )
    if os.path.isfile(root):
        return [root]
    return []


def _print_values(ranges, min_area):
    print("\n# Paste these over the constants at the top of the file:")
    print(f"DEFAULT_RED_LOW_1   = {ranges['red'][0]}")
    print(f"DEFAULT_RED_HIGH_1  = {ranges['red'][1]}")
    print(f"DEFAULT_RED_LOW_2   = {ranges['red'][2]}")
    print(f"DEFAULT_RED_HIGH_2  = {ranges['red'][3]}")
    print(f"DEFAULT_GREEN_LOW   = {ranges['green'][0]}")
    print(f"DEFAULT_GREEN_HIGH  = {ranges['green'][1]}")
    print(f"MIN_AREA            = {min_area}\n")


def _auto_calibrate_from_click(state, img, x, y, half=AUTOCAL_PATCH_HALF):
    """
    Update ONLY the active filter's HSV interval based on a patch around
    (x, y).  No auto-detection, no auto-switching.

    For red sub-ranges, the patch's hues are restricted to the relevant side
    of the seam (sub-1 → H<90, sub-2 → H>=90) so a click that straddles the
    seam can't blow the band up.  If no relevant pixels exist in the patch,
    nothing changes and a hint is printed.
    """
    h, w = img.shape[:2]
    x0 = max(0, x - half); x1 = min(w, x + half + 1)
    y0 = max(0, y - half); y1 = min(h, y + half + 1)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return

    hsv_patch = cv2.cvtColor(cv2.GaussianBlur(patch, (3, 3), 0),
                             cv2.COLOR_BGR2HSV)
    H_all = hsv_patch[..., 0].ravel()
    S = hsv_patch[..., 1].ravel()
    V = hsv_patch[..., 2].ravel()

    active = state["active"]
    if active == "red1":
        H = H_all[H_all < 90]
        clamp_lo, clamp_hi = 0, 89
        color_key, slot_lo, slot_hi = "red", 0, 1
    elif active == "red2":
        H = H_all[H_all >= 90]
        clamp_lo, clamp_hi = 90, 179
        color_key, slot_lo, slot_hi = "red", 2, 3
    else:  # green
        H = H_all
        clamp_lo, clamp_hi = 0, 179
        color_key, slot_lo, slot_hi = "green", 0, 1

    if len(H) == 0:
        other = "Red 2" if active == "red1" else "Red 1"
        print(f"  auto-cal {active}: patch has no pixels on this side of "
              f"the hue seam — try [{other}] instead, or pick a redder pixel.")
        return

    sat_low = max(0, int(np.percentile(S, 10)) - AUTOCAL_SAT_PAD)
    val_low = max(0, int(np.percentile(V, 10)) - AUTOCAL_VAL_PAD)
    h_lo = max(clamp_lo, int(H.min()) - AUTOCAL_HUE_PAD)
    h_hi = min(clamp_hi, int(H.max()) + AUTOCAL_HUE_PAD)

    state["ranges"][color_key][slot_lo] = [h_lo, sat_low, val_low]
    state["ranges"][color_key][slot_hi] = [h_hi, 255, 255]

    print(f"  auto-cal {active} at ({x},{y}): "
          f"H {h_lo}-{h_hi}, S>={sat_low}, V>={val_low}")


def _segment_active(bgr, low, high):
    """Apply blur + HSV inRange + morphology for a single [low, high] band."""
    blurred = cv2.GaussianBlur(bgr, BLUR_KSIZE, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return _clean(_inrange(hsv, low, high))


def _draw_button(toolbar, btn, active=False):
    x, y = btn["x"], BUTTON_Y
    w, h = btn["w"], BUTTON_HEIGHT
    bg = (60, 110, 60) if active else (55, 55, 55)
    cv2.rectangle(toolbar, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(toolbar, (x, y), (x + w, y + h), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(toolbar, btn["label"], (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)


def _find_button(x, y):
    if not (BUTTON_Y <= y < BUTTON_Y + BUTTON_HEIGHT):
        return None
    for b in BUTTONS:
        if b["x"] <= x < b["x"] + b["w"]:
            return b["id"]
    return None


def calibrate(paths):
    """Mouse-driven HSV calibration GUI for red + green stoplight detection."""
    images = [(p, cv2.imread(p)) for p in paths]
    images = [(p, img) for p, img in images if img is not None]
    if not images:
        print("No loadable images.")
        return

    yolo_model = None
    yolo_allowed_cls = []
    yolo_cache = {}
    if _YOLO_AVAILABLE:
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            yolo_model = YOLO(YOLO_MODEL_NAME)
            yolo_model.to(device)
            yolo_allowed_cls = [
                i for i, name in yolo_model.names.items()
                if name in YOLO_TARGET_CLASSES
            ]
            print(f"YOLO loaded on {device}. "
                  f"Annotating: {sorted(YOLO_TARGET_CLASSES)}")
        except Exception as e:
            print(f"  YOLO failed to load: {e} — left panel skips YOLO overlay.")
            yolo_model = None
    else:
        print("  ultralytics/torch not installed — "
              "left panel skips YOLO overlay.")

    def yolo_dets_for(path, img):
        if yolo_model is None:
            return []
        if path in yolo_cache:
            return yolo_cache[path]
        try:
            results = yolo_model(img, classes=yolo_allowed_cls,
                                 conf=YOLO_CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"  YOLO inference failed on {path}: {e}")
            yolo_cache[path] = []
            return []
        dets = (_yolo_results_to_detections(yolo_model, results[0])
                if results else [])
        yolo_cache[path] = dets
        return dets

    state = {
        "active":   "red1",
        "ranges":   default_ranges(),
        "min_area": MIN_AREA,
        "roi":      None,
        "drawing":  False,
        "drag_a":   None,
        "drag_b":   None,
        "img_idx":  0,
        "quit":     False,
    }

    win = "stoplight calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H_low",    win, 0,        179,                  lambda _: None)
    cv2.createTrackbar("S_low",    win, 0,        255,                  lambda _: None)
    cv2.createTrackbar("V_low",    win, 0,        255,                  lambda _: None)
    cv2.createTrackbar("H_high",   win, 179,      179,                  lambda _: None)
    cv2.createTrackbar("S_high",   win, 255,      255,                  lambda _: None)
    cv2.createTrackbar("V_high",   win, 255,      255,                  lambda _: None)
    cv2.createTrackbar("Min_Area", win, MIN_AREA, MIN_AREA_TRACKBAR_MAX, lambda _: None)

    def active_indices():
        if state["active"] == "red1":
            return "red", 0, 1
        if state["active"] == "red2":
            return "red", 2, 3
        return "green", 0, 1

    def push_to_trackbars():
        color, lo_i, hi_i = active_indices()
        low  = state["ranges"][color][lo_i]
        high = state["ranges"][color][hi_i]
        cv2.setTrackbarPos("H_low",  win, int(low[0]))
        cv2.setTrackbarPos("S_low",  win, int(low[1]))
        cv2.setTrackbarPos("V_low",  win, int(low[2]))
        cv2.setTrackbarPos("H_high", win, int(high[0]))
        cv2.setTrackbarPos("S_high", win, int(high[1]))
        cv2.setTrackbarPos("V_high", win, int(high[2]))

    def pull_from_trackbars():
        color, lo_i, hi_i = active_indices()
        state["ranges"][color][lo_i] = [
            cv2.getTrackbarPos("H_low",  win),
            cv2.getTrackbarPos("S_low",  win),
            cv2.getTrackbarPos("V_low",  win),
        ]
        state["ranges"][color][hi_i] = [
            cv2.getTrackbarPos("H_high", win),
            cv2.getTrackbarPos("S_high", win),
            cv2.getTrackbarPos("V_high", win),
        ]
        state["min_area"] = cv2.getTrackbarPos("Min_Area", win)

    def handle_button(btn_id):
        if btn_id in ("red1", "red2", "green"):
            state["active"] = btn_id
            push_to_trackbars()
        elif btn_id == "clear":
            state["roi"] = None
        elif btn_id == "print":
            _print_values(state["ranges"], state["min_area"])
        elif btn_id == "quit":
            state["quit"] = True

    def on_mouse(event, x, y, flags, _):
        # Toolbar area: button clicks.
        if y < TOOLBAR_HEIGHT:
            if event == cv2.EVENT_LBUTTONDOWN:
                btn = _find_button(x, y)
                if btn is not None:
                    handle_button(btn)
            return

        y_img = y - TOOLBAR_HEIGHT
        _, img = images[state["img_idx"]]
        h_img, w_img = img.shape[:2]
        if x >= w_img:
            return  # click on the right (filtered) panel — ignore
        x_l = max(0, min(x, w_img - 1))
        y_l = max(0, min(y_img, h_img - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["drag_a"]  = (x_l, y_l)
            state["drag_b"]  = (x_l, y_l)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["drag_b"] = (x_l, y_l)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            state["drag_b"]  = (x_l, y_l)
            x0, y0 = state["drag_a"]
            x1, y1 = state["drag_b"]
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            if xb - xa > 5 and yb - ya > 5:
                state["roi"] = (xa, ya, xb - xa, yb - ya)
            state["drag_a"] = None
            state["drag_b"] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            _auto_calibrate_from_click(state, img, x_l, y_l)
            push_to_trackbars()

    cv2.setMouseCallback(win, on_mouse)

    push_to_trackbars()
    print(
        "Calibrator ready.  Right-click on a lit bulb to auto-calibrate the\n"
        "active filter.  Buttons in the top bar drive everything else."
    )

    while not state["quit"]:
        path, img = images[state["img_idx"]]
        h, w = img.shape[:2]

        pull_from_trackbars()

        # Pick the active interval the trackbars are editing.
        if state["active"] == "red1":
            active_low  = state["ranges"]["red"][0]
            active_high = state["ranges"]["red"][1]
        elif state["active"] == "red2":
            active_low  = state["ranges"]["red"][2]
            active_high = state["ranges"]["red"][3]
        else:
            active_low  = state["ranges"]["green"][0]
            active_high = state["ranges"]["green"][1]

        # Build the active filter's mask + filtered display, optionally inside
        # the ROI.  Right panel = active filter only, so what the trackbars
        # control is what you see, isolated.
        if state["roi"] is not None:
            x_r, y_r, w_r, h_r = state["roi"]
            x_r  = max(0, min(x_r, w - 1))
            y_r  = max(0, min(y_r, h - 1))
            w_r  = max(1, min(w_r, w - x_r))
            h_r  = max(1, min(h_r, h - y_r))
            roi  = img[y_r:y_r+h_r, x_r:x_r+w_r]
            active_mask = _segment_active(roi, active_low, active_high)
            roi_filtered = cv2.bitwise_and(roi, roi, mask=active_mask)
            filtered = np.zeros_like(img)
            filtered[y_r:y_r+h_r, x_r:x_r+w_r] = roi_filtered
            mask_offset = (x_r, y_r)
        else:
            active_mask = _segment_active(img, active_low, active_high)
            filtered = cv2.bitwise_and(img, img, mask=active_mask)
            mask_offset = (0, 0)

        # Every contour that survived the filter is a candidate; the threshold
        # slider determines which "really count" as a detection.
        contours, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < CANDIDATE_MIN_AREA:
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            candidates.append((
                area,
                (cx + mask_offset[0], cy + mask_offset[1], cw, ch),
            ))
        candidates.sort(key=lambda c: c[0], reverse=True)

        largest_area = candidates[0][0] if candidates else 0
        passes_count = sum(1 for area, _ in candidates
                           if area >= state["min_area"])
        passes = largest_area >= state["min_area"]

        active_color_bgr = (
            (0, 0, 255) if state["active"].startswith("red") else (0, 200, 0)
        )
        active_color_name = "red" if state["active"].startswith("red") else "green"

        # ── Annotate RIGHT panel (filtered): every candidate gets a bbox.
        # Thick = above threshold, thin = below.  Largest gets its area
        # labelled so you can read the number against the slider.
        for area, (bx, by, bw, bh) in candidates:
            t = 3 if area >= state["min_area"] else 1
            cv2.rectangle(filtered, (bx, by), (bx + bw, by + bh),
                          active_color_bgr, t)
        if candidates:
            area, (bx, by, bw, bh) = candidates[0]
            cv2.putText(filtered, f"{area} px",
                        (bx, max(12, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color_bgr, 2)

        # Detection status overlay on the right panel.
        status_text  = ("DETECTED " + active_color_name.upper()) if passes \
            else "below threshold"
        status_color = active_color_bgr if passes else (190, 190, 190)
        cv2.putText(filtered, status_text,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(filtered,
                    f"largest: {largest_area} px    thresh: {state['min_area']}",
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(filtered,
                    f"candidates: {len(candidates)}  "
                    f"({passes_count} above threshold)",
                    (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ── Decorate the LEFT panel: original + YOLO overlay + ROI + label.
        left = img.copy()
        _draw_yolo_detections(left, yolo_dets_for(path, img))
        if state["roi"] is not None:
            x_r, y_r, w_r, h_r = state["roi"]
            cv2.rectangle(left, (x_r, y_r), (x_r + w_r, y_r + h_r),
                          (255, 255, 255), 2)
        if state["drawing"] and state["drag_a"] and state["drag_b"]:
            cv2.rectangle(left, state["drag_a"], state["drag_b"],
                          (200, 200, 200), 1)

        active_label = {"red1": "red sub-1",
                        "red2": "red sub-2",
                        "green": "green"}[state["active"]]
        cv2.putText(left, f"editing: {active_label}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(left, os.path.basename(path),
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── Compose toolbar + side-by-side image ─────────────────────────────
        panel = np.hstack([left, filtered])
        toolbar = np.full((TOOLBAR_HEIGHT, panel.shape[1], 3), 30,
                          dtype=np.uint8)
        for b in BUTTONS:
            _draw_button(toolbar, b, active=(b["id"] == state["active"]))
        hint = ("right-click on a bulb = auto-cal active filter   "
                "left-drag = ROI   trackbars below   thick boxes = above threshold")
        cv2.putText(toolbar, hint,
                    (max(b["x"] + b["w"] for b in BUTTONS) + 16,
                     BUTTON_Y + BUTTON_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

        display = np.vstack([toolbar, panel])
        cv2.imshow(win, display)

        cv2.waitKey(20)
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    positional = []
    for a in sys.argv[1:]:
        if a.startswith("-"):
            break
        positional.append(a)

    if positional:
        paths = []
        for arg in positional:
            found = _collect_paths(arg)
            if not found:
                print(f"  not a file or directory of images: {arg}")
            paths.extend(found)
        if paths:
            calibrate(paths)
        else:
            print("Nothing to load.")
            sys.exit(1)
    else:
        main()

#!/usr/bin/env python3
"""
sign_detector.py — pure consumer of yolo_node outputs.

Watches yolo_node's per-class ROI topics for parking_meter and
fire_hydrant.  Runs no YOLO inference of its own and never reads the
raw camera image — every pixel it touches comes from yolo_node, so
there is no race between bbox and source frame.

Behaviors
---------
1. Continuous (whenever `/sign_detection/trigger` is True): every time
   `/yolo/parking_meter/roi` arrives non-empty, republish the bbox's
   bottom-center as ConeLocationPixel on `/relative_cone_px` for the
   parking controller's visual servo.

2. One-shot (whenever `/sign_detection/trigger` is True, until the next
   `False` resets it): on the first frame where either parking_meter or
   fire_hydrant is detected, pick the larger-area bbox, publish the
   class name on `/sign_detection/result`, and save the most recent
   `/yolo/annotated_image` to disk under `save_dir`.

3. Live debug feed: while triggered, the latest `/yolo/annotated_image`
   is rebroadcast on `/sign_detection/live_feed` so existing RViz/foxglove
   layouts keep working.

The trigger Bool gates ALL publishing; toggling it back to False also
re-arms the one-shot for the next leg.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Bool, String
from vs_msgs.msg import ConeLocationPixel


# Class names follow yolo_node's TARGET_CLASSES (Ultralytics COCO labels).
# Topic suffixes replace spaces with underscores, matching yolo_node's
# `_topic_safe()`.
WATCHED_CLASSES = ("parking meter", "fire hydrant")


def _topic_safe(name: str) -> str:
    return name.replace(" ", "_")


@dataclass
class RoiSnapshot:
    x_offset: int
    y_offset: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return int(self.width) * int(self.height)

    @property
    def u_center(self) -> float:
        return float(self.x_offset) + float(self.width) / 2.0

    @property
    def v_bottom(self) -> float:
        return float(self.y_offset) + float(self.height)


class SignDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("sign_detector")

        self.save_dir = (
            self.declare_parameter(
                "save_dir",
                "/root/racecar_ws/src/final_challenge2026/final_challenge/sign_detections",
            )
            .get_parameter_value().string_value
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.active = False
        self.result_published = False

        # Latest non-empty bbox per watched class. None when the class is
        # not currently detected; set on every fresh non-empty ROI message.
        self._latest_roi: dict[str, Optional[RoiSnapshot]] = {
            name: None for name in WATCHED_CLASSES
        }
        # Latest annotated frame from yolo_node — used for the saved-on-trigger
        # snapshot and the live-feed passthrough.
        self._latest_annotated: Optional[Image] = None

        for name in WATCHED_CLASSES:
            topic = f"/yolo/{_topic_safe(name)}/roi"
            self.create_subscription(
                RegionOfInterest, topic,
                lambda msg, n=name: self._roi_cb(msg, n), 10,
            )

        self.create_subscription(
            Image, "/yolo/annotated_image", self._annotated_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool, "/sign_detection/trigger", self._trigger_cb, 10,
        )

        self.result_pub = self.create_publisher(String, "/sign_detection/result", 10)
        self.annotated_pub = self.create_publisher(
            Image, "/sign_detection/annotated_image", 10)
        self.live_feed_pub = self.create_publisher(
            Image, "/sign_detection/live_feed", qos_profile_sensor_data)
        self.cone_px_pub = self.create_publisher(
            ConeLocationPixel, "/relative_cone_px", 10)

        self.get_logger().info(
            f"SignDetector ready — listening to yolo_node only. "
            f"watched={list(WATCHED_CLASSES)}, save_dir={self.save_dir}"
        )

    def _trigger_cb(self, msg: Bool) -> None:
        if msg.data:
            self.active = True
            self.get_logger().info("Detection triggered.")
        else:
            self.active = False
            # Re-arm the one-shot for the next leg.
            self.result_published = False

    def _annotated_cb(self, msg: Image) -> None:
        self._latest_annotated = msg
        if self.active:
            self.live_feed_pub.publish(msg)

    def _roi_cb(self, msg: RegionOfInterest, class_name: str) -> None:
        # yolo_node publishes width=0/height=0 every frame the class is
        # not detected, so absence-of-detection naturally clears the slot.
        if msg.width == 0 or msg.height == 0:
            self._latest_roi[class_name] = None
            return

        snap = RoiSnapshot(
            x_offset=int(msg.x_offset),
            y_offset=int(msg.y_offset),
            width=int(msg.width),
            height=int(msg.height),
        )
        self._latest_roi[class_name] = snap

        if not self.active:
            return

        # Continuous: feed the parking controller a pixel target every
        # frame parking_meter is visible. Bottom-center matches the
        # homography calibration used by homography_transformer.
        if class_name == "parking meter":
            cone_px = ConeLocationPixel()
            cone_px.u = snap.u_center
            cone_px.v = snap.v_bottom
            self.cone_px_pub.publish(cone_px)

        # One-shot: announce the class and save proof, once per trigger.
        if self.result_published:
            return
        self._maybe_publish_one_shot()

    def _maybe_publish_one_shot(self) -> None:
        """Pick the larger-area watched bbox currently visible and emit
        the one-shot result + saved annotated frame. RegionOfInterest
        messages don't carry confidence, so larger-area is the proxy for
        "more confident / closer" when both classes are detected."""
        candidates = [
            (name, snap)
            for name, snap in self._latest_roi.items()
            if snap is not None
        ]
        if not candidates:
            return
        candidates.sort(key=lambda nc: nc[1].area, reverse=True)
        best_name, best_snap = candidates[0]

        if self._latest_annotated is None:
            # Without an annotated frame we can still publish the result,
            # but we can't save proof to disk. That's a soft failure: log
            # and emit the class anyway so the state machine isn't blocked.
            self.get_logger().warn(
                f"Detected '{best_name}' but no /yolo/annotated_image "
                f"received yet — publishing result without saving."
            )
            self.result_pub.publish(String(data=best_name))
            self.result_published = True
            return

        try:
            annotated_bgr = self.bridge.imgmsg_to_cv2(
                self._latest_annotated, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed on annotated: {e}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(
            self.save_dir, f"{_topic_safe(best_name)}_{timestamp}.jpg")
        cv2.imwrite(save_path, annotated_bgr)

        self.result_pub.publish(String(data=best_name))
        self.annotated_pub.publish(self._latest_annotated)
        self.result_published = True

        self.get_logger().info(
            f"Detected '{best_name}' (bbox={best_snap.width}x{best_snap.height}"
            f"={best_snap.area}px), saved to {save_path}"
        )


def main() -> None:
    rclpy.init()
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

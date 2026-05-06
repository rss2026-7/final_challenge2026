#!/usr/bin/env python3
"""
yolo_node.py — central YOLO inference for the perception stack.

Runs YOLO once per incoming camera frame and publishes per-class bounding
boxes, per-class cropped images, and an annotated debug image.  Every
downstream perception node (stoplight_detection, sign_detector, the
safety controller's person check) consumes ONLY this node's outputs —
nobody else runs YOLO and nobody re-crops the source image, which would
race the bbox against a newer frame.

Topology
--------
Subscribes:
  <image_topic>  sensor_msgs/Image
      Default /zed/zed_node/rgb/image_rect_color (override via the
      `image_topic` parameter).  Subscription QoS is sensor_data
      (BEST_EFFORT, depth 1) to match the ZED publisher and stop frames
      from queueing behind a slow YOLO inference under load.

Publishes:
  /yolo/<class>/roi         sensor_msgs/RegionOfInterest
      One topic per name in TARGET_CLASSES (spaces → underscores).
      Carries the highest-confidence bbox of <class> in the latest frame.
      width == 0 or height == 0 means "not detected this frame" — published
      every frame so consumers can distinguish "no detection now" from
      "stale, no message yet".
  /yolo/<class>/image       sensor_msgs/Image
      The current source frame cropped to <class>'s highest-confidence
      bbox.  Published ONLY on frames where the class is detected — its
      header.stamp matches the source frame, so downstream HSV /
      classification work is guaranteed to run on the same pixels YOLO
      saw the object in.  Consumers should treat "no message arrived
      within K seconds" as "not detected" via a watchdog timer.
  /yolo/annotated_image     sensor_msgs/Image
      The input frame with every surviving box drawn on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, RegionOfInterest
from ultralytics import YOLO


# Names must match Ultralytics COCO labels exactly (with spaces). The
# /yolo/<class>/roi topics replace spaces with underscores via _topic_safe().
TARGET_CLASSES = {"parking meter", "fire hydrant", "traffic light", "person"}

CLASS_COLORS = {
    "parking meter": (0, 255, 0),
    "fire hydrant":  (0, 0, 255),
    "traffic light": (255, 255, 0),
    "person":        (255, 0, 255),
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def _topic_safe(name: str) -> str:
    return name.replace(" ", "_")


class YoloNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.image_topic = (
            self.declare_parameter("image_topic",
                                   "/zed/zed_node/rgb/image_rect_color")
            .get_parameter_value().string_value
        )
        self.model_name = (
            self.declare_parameter("model", "yolo11n.pt")
            .get_parameter_value().string_value
        )
        self.conf_threshold = (
            self.declare_parameter("conf_threshold", 0.25)
            .get_parameter_value().double_value
        )
        self.imgsz = (
            self.declare_parameter("imgsz", 960)
            .get_parameter_value().integer_value
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.allowed_cls = [
            i for i, name in self.model.names.items()
            if name in TARGET_CLASSES
        ]

        self.bridge = CvBridge()

        self.create_subscription(
            Image, self.image_topic, self._image_cb, qos_profile_sensor_data)
        self.roi_pubs: Dict[str, rclpy.publisher.Publisher] = {
            name: self.create_publisher(
                RegionOfInterest, f"/yolo/{_topic_safe(name)}/roi", 10)
            for name in TARGET_CLASSES
        }
        self.crop_pubs: Dict[str, rclpy.publisher.Publisher] = {
            name: self.create_publisher(
                Image, f"/yolo/{_topic_safe(name)}/image", qos_profile_sensor_data)
            for name in TARGET_CLASSES
        }
        self.annotated_pub = self.create_publisher(
            Image, "/yolo/annotated_image", qos_profile_sensor_data)

        self.get_logger().info(
            f"YoloNode ready — image={self.image_topic}, "
            f"model={self.model_name}, device={self.device}, "
            f"conf>={self.conf_threshold}, imgsz={self.imgsz}"
        )
        self.get_logger().info(
            f"Per-class ROI topics: "
            f"{sorted(p.topic_name for p in self.roi_pubs.values())}"
        )
        self.get_logger().info(
            f"Per-class crop topics: "
            f"{sorted(p.topic_name for p in self.crop_pubs.values())}"
        )

    def _image_cb(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        try:
            results = self.model(
                bgr,
                classes=self.allowed_cls,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        dets = self._results_to_detections(results[0]) if results else []

        best_per_class: Dict[str, Detection] = {}
        for det in dets:
            cur = best_per_class.get(det.class_name)
            if cur is None or det.confidence > cur.confidence:
                best_per_class[det.class_name] = det

        h, w = bgr.shape[:2]
        for name, roi_pub in self.roi_pubs.items():
            roi = RegionOfInterest()
            det = best_per_class.get(name)
            if det is not None:
                x1 = max(0, min(det.x1, w - 1))
                y1 = max(0, min(det.y1, h - 1))
                x2 = max(0, min(det.x2, w))
                y2 = max(0, min(det.y2, h))
                roi.x_offset = x1
                roi.y_offset = y1
                roi.width    = max(0, x2 - x1)
                roi.height   = max(0, y2 - y1)
                if roi.width > 0 and roi.height > 0:
                    crop = bgr[y1:y2, x1:x2]
                    crop_msg = self.bridge.cv2_to_imgmsg(crop, encoding="bgr8")
                    crop_msg.header = msg.header
                    self.crop_pubs[name].publish(crop_msg)
            roi_pub.publish(roi)

        annotated = self._draw_detections(bgr, dets)
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.annotated_pub.publish(out_msg)

    def _results_to_detections(self, result) -> List[Detection]:
        detections: List[Detection] = []
        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls  = result.boxes.cls

        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        cls_np  = cls.detach().cpu().numpy()  if hasattr(cls,  "detach") else np.asarray(cls)

        for box, conf_val, cls_val in zip(xyxy_np, conf_np, cls_np):
            detections.append(Detection(
                class_id=int(cls_val),
                class_name=self.model.names[int(cls_val)],
                confidence=float(conf_val),
                x1=int(box[0]),
                y1=int(box[1]),
                x2=int(box[2]),
                y2=int(box[3]),
            ))
        return detections

    def _draw_detections(self, bgr: np.ndarray,
                         dets: List[Detection]) -> np.ndarray:
        out = bgr.copy()
        for det in dets:
            color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(out, label,
                        (det.x1, max(12, det.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out


def main() -> None:
    rclpy.init()
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

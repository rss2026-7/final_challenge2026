#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rclpy
import torch

from dataclasses import dataclass
from datetime import datetime
from typing import List

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from ultralytics import YOLO
from vs_msgs.msg import ConeLocationPixel


TARGET_CLASSES = {"parking_meter", "fire_hydrant", "bird", "traffic light"}

CLASS_COLORS = {
    "parking_meter": (0, 255, 0),
    "fire_hydrant":  (0, 0, 255),
    "bird":          (255, 165, 0),
    "traffic light": (255, 255, 0),
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


class SignDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("sign_detector")

        self.model_name = (
            self.declare_parameter("model", "yolo11n.pt")
            .get_parameter_value().string_value
        )
        self.conf_threshold = (
            self.declare_parameter("conf_threshold", 0.5)
            .get_parameter_value().double_value
        )
        self.save_dir = (
            self.declare_parameter("save_dir", "/home/racecar/sign_detections")
            .get_parameter_value().string_value
        )

        os.makedirs(self.save_dir, exist_ok=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.allowed_cls = [
            i for i, name in self.model.names.items()
            if name in TARGET_CLASSES
        ]

        self.bridge = CvBridge()
        self.active = False
        self.result_published = False

        self.img_sub = self.create_subscription(
            Image, "/zed/zed_node/rgb/image_rect_color", self._image_cb, 10)
        self.trigger_sub = self.create_subscription(
            Bool, "/sign_detection/trigger", self._trigger_cb, 10)

        self.result_pub = self.create_publisher(String, "/sign_detection/result", 10)
        self.annotated_pub = self.create_publisher(Image, "/sign_detection/annotated_image", 10)
        self.live_feed_pub = self.create_publisher(Image, "/sign_detection/live_feed", 10)
        self.cone_px_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)

        self.get_logger().info(
            f"SignDetector ready — model={self.model_name}, device={self.device}, "
            f"conf={self.conf_threshold}, save_dir={self.save_dir}"
        )
        self.get_logger().info(f"Watching class IDs: {self.allowed_cls}")

    def _trigger_cb(self, msg: Bool) -> None:
        if msg.data:
            self.active = True
            self.get_logger().info("Detection triggered.")
        else:
            self.active = False

    def _image_cb(self, msg: Image) -> None:
        if not self.active:
            return

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
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results:
            return

        dets = self._results_to_detections(results[0])

        # Always publish live feed with all detections drawn
        live = self._draw_all_detections(bgr, dets)
        live_msg = self.bridge.cv2_to_imgmsg(live, encoding="bgr8")
        live_msg.header = msg.header
        self.live_feed_pub.publish(live_msg)

        if not dets:
            return

        best = max(dets, key=lambda d: d.confidence)

        # Continuous: publish pixel location every frame for parking controller servo
        if best.class_name == "parking_meter":
            cone_px = ConeLocationPixel()
            cone_px.u = float((best.x1 + best.x2) / 2)
            cone_px.v = float(best.y2)  # bottom-center, matches homography calibration
            self.cone_px_pub.publish(cone_px)

        # One-shot: publish class result and save annotated image
        if self.result_published:
            return

        annotated = self._draw_detection(bgr, best)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(self.save_dir, f"{best.class_name}_{timestamp}.jpg")
        cv2.imwrite(save_path, annotated)

        result_msg = String()
        result_msg.data = best.class_name
        self.result_pub.publish(result_msg)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.annotated_pub.publish(out_msg)

        self.result_published = True
        self.get_logger().info(
            f"Detected '{best.class_name}' (conf={best.confidence:.2f}), saved to {save_path}"
        )

    def _results_to_detections(self, result) -> List[Detection]:
        detections = []
        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

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

    def _draw_detection(self, bgr: np.ndarray, det: Detection) -> np.ndarray:
        out = bgr.copy()
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(out, label, (det.x1, det.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out

    def _draw_all_detections(self, bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
        out = bgr.copy()
        for det in dets:
            color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(out, label, (det.x1, det.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out


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

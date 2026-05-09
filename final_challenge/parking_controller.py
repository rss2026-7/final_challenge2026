#!/usr/bin/env python3
"""
parking_controller.py  —  Part B: visual-servo parking controller

Ported directly from visual_servoing_pkg/visual_servoing/parking_controller.py
and extended with state-machine integration (trigger / done topics).

Pipeline
--------
  Weiming YOLO node
    → publishes /relative_cone_px  (vs_msgs/ConeLocationPixel)

  HomographyTransformer  (homography_transformer.py)
    → subscribes /relative_cone_px
    → publishes   /relative_cone   (vs_msgs/ConeLocation, metres, car frame)

  ParkingController  (this node)
    → subscribes /relative_cone
    → PD servo toward meter, stops within parking_distance metres
    → publishes /drive and /parking/done

State-machine interface
-----------------------
  /parking/trigger  (std_msgs/Bool)  — True to activate, False to deactivate
  /parking/done     (std_msgs/Bool)  — publishes True once within jitter zone
"""

import os
import csv
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped


class ParkingController(Node):
    """
    Visual-servo controller that parks in front of a detected object.

    Subscribes to /relative_cone (ConeLocation in metres, car frame)
    and drives toward it using:
      - PD controller on angle-to-target for steering
      - P  controller on distance error for speed (can reverse if overshot)
      - Jitter deadband: stops when close AND aligned
    """

    def __init__(self):
        super().__init__("parking_controller")

        # ── Parameters ────────────────────────────────────────────────────
        # Publish to the high-level nav input so the safety_controller
        # (subscribed there) can interpose stops / speed limits before the
        # cmd hits the VESC. /vesc/low_level/input/navigation bypasses
        # safety entirely — only use it when explicitly testing without
        # the safety stack.
        self.declare_parameter("drive_topic",      "/vesc/high_level/input/nav_0")
        self.declare_parameter("parking_distance", 0.75)  # metres — target stop distance
        self.declare_parameter(
            "save_dir",
            "/root/racecar_ws/src/final_challenge2026/final_challenge/sign_detections",
        )

        drive_topic = self.get_parameter("drive_topic").value
        self.parking_distance = float(self.get_parameter("parking_distance").value)
        self.save_dir = self.get_parameter("save_dir").value
        os.makedirs(self.save_dir, exist_ok=True)

        # ── Internal state ────────────────────────────────────────────────
        # active: set True by /parking/trigger; controller does nothing when False
        self.active = False
        # done: True once jitter condition was first satisfied this activation
        self.done   = False

        self.relative_x = 0.0
        self.relative_y = 0.0
        self.prev_angle_to_cone = None
        self.prev_time_sec      = None

        # Image saving — cache the latest annotated frame from yolo_node.
        # Saved to disk the moment the jitter deadband is first satisfied
        # (car stationary, close to sign) so the bounding box is large and blur-free.
        self.bridge             = CvBridge()
        self._latest_annotated  = None  # most recent sensor_msgs/Image from /yolo/annotated_image

        # ── Publications ──────────────────────────────────────────────────
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)
        self.done_pub  = self.create_publisher(Bool, "/parking/done", 10)

        # ── Subscriptions ─────────────────────────────────────────────────
        # Activated / deactivated by the state machine
        self.create_subscription(Bool, "/parking/trigger", self._trigger_cb, 10)

        # Relative cone position in car frame metres — fed by HomographyTransformer
        self.create_subscription(ConeLocation, "/relative_cone",
                                 self._relative_cone_cb, 1)

        # Annotated image from yolo_node — cached and saved when parked.
        # Uses sensor_data QoS (BEST_EFFORT, depth 1) to match yolo_node's publisher.
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Image, "/yolo/annotated_image",
                                 self._annotated_cb, qos_profile_sensor_data)

        # ── CSV logging (same as visual_servoing_pkg) ─────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join(os.getcwd(), f"parking_run_{timestamp}.csv")
        self._csv_file   = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "time_sec", "relative_x", "relative_y",
            "distance_to_cone", "distance_error", "angle_to_cone",
            "derror", "steering_angle", "speed", "Kp_steer", "Kd_steer",
        ])
        self.get_logger().info(f"Logging to {csv_path}")
        self.get_logger().info(
            f"ParkingController ready. parking_distance={self.parking_distance} m"
        )

    # ======================================================================
    #  State-machine integration
    # ======================================================================

    def _annotated_cb(self, msg: Image):
        """Cache the latest annotated frame from yolo_node for saving when parked."""
        self._latest_annotated = msg

    def _trigger_cb(self, msg: Bool):
        """
        Sent by the state machine when entering / leaving PARKING state.
        True  → activate; reset per-run state so a second attempt is clean.
        False → deactivate; stop driving immediately.
        """
        self.get_logger().info(
            f"[DEBUG] /parking/trigger received: data={msg.data}, "
            f"current active={self.active}, done={self.done}"
        )
        if msg.data and not self.active:
            self.get_logger().info("[DEBUG] Parking controller ACTIVATED.")
            self.active             = True
            self.done               = False
            self.prev_angle_to_cone = None
            self.prev_time_sec      = None
            self.get_logger().info(
                f"[DEBUG] State reset: active={self.active}, done={self.done}, "
                f"target_distance={self.parking_distance} m, "
                f"last_relative=({self.relative_x:.3f}, {self.relative_y:.3f})"
            )
        elif not msg.data and self.active:
            self.get_logger().info("[DEBUG] Parking controller DEACTIVATED.")
            self.active = False
            self._publish_stop()
            self.get_logger().info("[DEBUG] Stop command published on deactivation.")
        else:
            self.get_logger().info(
                f"[DEBUG] Trigger ignored — already in requested state "
                f"(msg.data={msg.data}, active={self.active})."
            )

    # ======================================================================
    #  Main control callback
    # ======================================================================

    def _relative_cone_cb(self, msg: ConeLocation):
        """
        Called every time HomographyTransformer publishes a /relative_cone.
        Does nothing when inactive so the lane-following drive topic is
        not disturbed.
        """
        # Store latest for error_publisher regardless of active state
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos

        self.get_logger().info(
            f"[DEBUG] /relative_cone received: x={msg.x_pos:.3f} y={msg.y_pos:.3f} "
            f"active={self.active} done={self.done}",
            throttle_duration_sec=0.5,
        )

        if not self.active:
            self.get_logger().info(
                "[DEBUG] Inactive — skipping control loop, not publishing drive.",
                throttle_duration_sec=2.0,
            )
            return

        drive_cmd = AckermannDriveStamped()

        # ── Geometry ──────────────────────────────────────────────────────
        angle_to_cone    = np.arctan2(self.relative_y, self.relative_x)
        distance_to_cone = np.hypot(self.relative_x, self.relative_y)
        distance_error   = distance_to_cone - self.parking_distance

        self.get_logger().info(
            f"[DEBUG] Geometry: dist={distance_to_cone:.3f} m, "
            f"angle={np.degrees(angle_to_cone):.2f} deg, "
            f"dist_error={distance_error:+.3f} m (target={self.parking_distance} m)",
            throttle_duration_sec=0.25,
        )

        # ── PD steering — derivative of angle error damps oscillation ─────
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time_sec is None or self.prev_angle_to_cone is None:
            derror = 0.0
            self.get_logger().info("[DEBUG] First control tick — derror forced to 0.")
        else:
            dt     = max(now_sec - self.prev_time_sec, 1e-3)
            derror = (angle_to_cone - self.prev_angle_to_cone) / dt

        # k_p=1.0 (radians/radian) saturates the steering to ±0.34 for any
        # angle >19.5° off-axis, producing a hard "full-lock" lurch on the
        # first control tick when the meter is well to the side. 0.5 keeps
        # response gentle until ~39° before saturating.
        k_p_steer = 0.5
        k_d_steer = 0.1
        steering_cmd   = k_p_steer * angle_to_cone - k_d_steer * derror
        steering_angle = float(np.clip(steering_cmd, -0.34, 0.34))

        if abs(steering_cmd) > 0.34:
            self.get_logger().info(
                f"[DEBUG] Steering saturated: raw={steering_cmd:+.3f} → "
                f"clipped={steering_angle:+.3f}",
                throttle_duration_sec=0.5,
            )

        self.prev_angle_to_cone = angle_to_cone
        self.prev_time_sec      = now_sec

        # ── Jitter deadband ───────────────────────────────────────────────
        # Only declare done when BOTH close enough AND facing the meter.
        # This prevents stopping while still pointing sideways.
        jitter_distance = 0.3   # metres
        jitter_angle    = 0.05  # radians (~3°)
        angle_error     = abs(angle_to_cone)

        if abs(distance_error) < jitter_distance and angle_error < jitter_angle:
            drive_cmd.drive.speed          = 0.0
            drive_cmd.drive.steering_angle = 0.0

            self.get_logger().info(
                f"[DEBUG] DECISION=STOP — within jitter deadband "
                f"(|dist_err|={abs(distance_error):.3f} < {jitter_distance}, "
                f"|angle|={angle_error:.3f} < {jitter_angle})"
            )

            # Publish /parking/done once per activation and save proof image.
            if not self.done:
                self.get_logger().info(
                    f"[DEBUG] Parked at {distance_to_cone:.2f} m — "
                    f"publishing /parking/done=True."
                )
                self.done   = True
                self.active = False
                done_msg      = Bool()
                done_msg.data = True
                self.done_pub.publish(done_msg)
                self.get_logger().info("[DEBUG] /parking/done published, controller now inactive.")

                # ── Save annotated image ──────────────────────────────────
                # Car is stationary and within parking_distance of the sign,
                # so the bounding box is large and motion-blur free.
                # Mirrors the format used previously in sign_detector.py.
                if self._latest_annotated is not None:
                    try:
                        annotated_bgr = self.bridge.imgmsg_to_cv2(
                            self._latest_annotated, desired_encoding="bgr8")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        save_path = os.path.join(
                            self.save_dir, f"parking_meter_{timestamp}.jpg")
                        cv2.imwrite(save_path, annotated_bgr)
                        self.get_logger().info(
                            f"[DEBUG] Parked image saved to {save_path}"
                        )
                    except Exception as e:
                        self.get_logger().error(
                            f"[DEBUG] Failed to save parked image: {e}"
                        )
                else:
                    self.get_logger().warn(
                        "[DEBUG] No annotated image available at park time — "
                        "image not saved. Check /yolo/annotated_image is publishing."
                    )
            else:
                self.get_logger().info(
                    "[DEBUG] Already done — not republishing /parking/done.",
                    throttle_duration_sec=1.0,
                )

        elif distance_error > 0:
            # Too far — drive forward, proportional speed scaled by angle.
            # cos(angle) gives full speed straight ahead, 0.5× at 60°, 0 at
            # 90°. Floor at 0.15 so we keep some progress rather than freezing
            # when the meter is hard to the side.
            angle_speed_factor = float(max(np.cos(angle_to_cone), 0.15))
            speed = float(np.clip(0.5 * distance_error, 0.2, 1.0)) * angle_speed_factor
            drive_cmd.drive.speed          = speed
            drive_cmd.drive.steering_angle = steering_angle
            self.get_logger().info(
                f"[DEBUG] DECISION=FORWARD — speed={speed:.3f} steer={steering_angle:+.3f} "
                f"(dist_err=+{distance_error:.3f}, angle_factor={angle_speed_factor:.2f})",
                throttle_duration_sec=0.25,
            )
        else:
            # Overshot — reverse slowly, scaled by angle factor as well.
            angle_speed_factor = float(max(np.cos(angle_to_cone), 0.15))
            speed = float(np.clip(0.5 * distance_error, -1.0, -0.2)) * angle_speed_factor
            drive_cmd.drive.speed          = speed
            drive_cmd.drive.steering_angle = -steering_angle
            self.get_logger().info(
                f"[DEBUG] DECISION=REVERSE — speed={speed:.3f} steer={-steering_angle:+.3f} "
                f"(overshot, dist_err={distance_error:.3f}, angle_factor={angle_speed_factor:.2f})",
                throttle_duration_sec=0.25,
            )

        # ── Log + publish ─────────────────────────────────────────────────
        self._csv_writer.writerow([
            f"{now_sec:.4f}", f"{self.relative_x:.4f}", f"{self.relative_y:.4f}",
            f"{distance_to_cone:.4f}", f"{distance_error:.4f}", f"{angle_to_cone:.4f}",
            f"{derror:.4f}", f"{drive_cmd.drive.steering_angle:.4f}",
            f"{drive_cmd.drive.speed:.4f}", f"{k_p_steer}", f"{k_d_steer}",
        ])
        self._csv_file.flush()

        self.drive_pub.publish(drive_cmd)
        self.get_logger().info(
            f"[DEBUG] Drive published: speed={drive_cmd.drive.speed:+.3f} "
            f"steer={drive_cmd.drive.steering_angle:+.3f}",
            throttle_duration_sec=0.25,
        )
        self._publish_error()

    # ======================================================================
    #  Error publisher (rqt_plot)
    # ======================================================================

    def _publish_error(self):
        error_msg = ParkingError()
        error_msg.x_error        = float(self.relative_x)
        error_msg.y_error        = float(self.relative_y)
        error_msg.distance_error = float(np.hypot(self.relative_x, self.relative_y))
        self.error_pub.publish(error_msg)

    # ======================================================================
    #  Helpers
    # ======================================================================

    def _publish_stop(self):
        msg = AckermannDriveStamped()
        msg.drive.speed          = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)
        self.get_logger().info("[DEBUG] _publish_stop() — zero-velocity drive sent.")


def main(args=None):
    rclpy.init(args=args)
    node = ParkingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

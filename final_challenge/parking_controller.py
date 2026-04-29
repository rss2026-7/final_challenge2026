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

import rclpy
from rclpy.node import Node
import numpy as np

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
        self.declare_parameter("drive_topic",      "/vesc/low_level/input/navigation")
        self.declare_parameter("parking_distance", 0.75)  # metres — target stop distance

        drive_topic = self.get_parameter("drive_topic").value
        self.parking_distance = float(self.get_parameter("parking_distance").value)

        # ── Internal state ────────────────────────────────────────────────
        # active: set True by /parking/trigger; controller does nothing when False
        self.active = False
        # done: True once jitter condition was first satisfied this activation
        self.done   = False

        self.relative_x = 0.0
        self.relative_y = 0.0
        self.prev_angle_to_cone = None
        self.prev_time_sec      = None

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

    def _trigger_cb(self, msg: Bool):
        """
        Sent by the state machine when entering / leaving PARKING state.
        True  → activate; reset per-run state so a second attempt is clean.
        False → deactivate; stop driving immediately.
        """
        if msg.data and not self.active:
            self.get_logger().info("Parking controller ACTIVATED.")
            self.active             = True
            self.done               = False
            self.prev_angle_to_cone = None
            self.prev_time_sec      = None
        elif not msg.data and self.active:
            self.get_logger().info("Parking controller DEACTIVATED.")
            self.active = False
            self._publish_stop()

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

        if not self.active:
            return

        drive_cmd = AckermannDriveStamped()

        # ── Geometry ──────────────────────────────────────────────────────
        angle_to_cone    = np.arctan2(self.relative_y, self.relative_x)
        distance_to_cone = np.hypot(self.relative_x, self.relative_y)
        distance_error   = distance_to_cone - self.parking_distance

        # ── PD steering — derivative of angle error damps oscillation ─────
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time_sec is None or self.prev_angle_to_cone is None:
            derror = 0.0
        else:
            dt     = max(now_sec - self.prev_time_sec, 1e-3)
            derror = (angle_to_cone - self.prev_angle_to_cone) / dt

        k_p_steer = 1.0
        k_d_steer = 0.1
        steering_cmd   = k_p_steer * angle_to_cone - k_d_steer * derror
        steering_angle = float(np.clip(steering_cmd, -0.34, 0.34))

        self.prev_angle_to_cone = angle_to_cone
        self.prev_time_sec      = now_sec

        # ── Jitter deadband ───────────────────────────────────────────────
        # Only declare done when BOTH close enough AND facing the meter.
        # This prevents stopping while still pointing sideways.
        jitter_distance = 0.1   # metres
        jitter_angle    = 0.05  # radians (~3°)
        angle_error     = abs(angle_to_cone)

        if abs(distance_error) < jitter_distance and angle_error < jitter_angle:
            drive_cmd.drive.speed          = 0.0
            drive_cmd.drive.steering_angle = 0.0

            # Publish /parking/done once per activation
            if not self.done:
                self.get_logger().info(
                    f"Parked at {distance_to_cone:.2f} m — signalling done."
                )
                self.done   = True
                self.active = False
                done_msg      = Bool()
                done_msg.data = True
                self.done_pub.publish(done_msg)

        elif distance_error > 0:
            # Too far — drive forward, proportional speed
            speed = float(np.clip(0.5 * distance_error, 0.2, 1.0))
            drive_cmd.drive.speed          = speed
            drive_cmd.drive.steering_angle = steering_angle
        else:
            # Overshot — reverse slowly
            speed = float(np.clip(0.5 * distance_error, -1.0, -0.2))
            drive_cmd.drive.speed          = speed
            drive_cmd.drive.steering_angle = -steering_angle

        # ── Log + publish ─────────────────────────────────────────────────
        self._csv_writer.writerow([
            f"{now_sec:.4f}", f"{self.relative_x:.4f}", f"{self.relative_y:.4f}",
            f"{distance_to_cone:.4f}", f"{distance_error:.4f}", f"{angle_to_cone:.4f}",
            f"{derror:.4f}", f"{drive_cmd.drive.steering_angle:.4f}",
            f"{drive_cmd.drive.speed:.4f}", f"{k_p_steer}", f"{k_d_steer}",
        ])
        self._csv_file.flush()

        self.drive_pub.publish(drive_cmd)
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

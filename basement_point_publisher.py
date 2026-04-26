#!/usr/bin/env python3
"""
basement_point_publisher.py
===========================
Testing stand-in for the TA's basement_point_publisher node.

Usage
-----
1. Launch this node alongside your RViz session.
2. In RViz, select the "Publish Point" tool (toolbar or press 'G').
3. Click your first goal location on the map  → logged as Goal 1.
4. Click your second goal location            → logged as Goal 2.
5. The node latches and publishes both as a PoseArray on /basement_goals.
6. The state machine picks it up and begins the mission.

To reset and pick new points, publish anything to /basement_goals/reset
(std_msgs/Bool) or just restart the node.

Topics
------
  Subscribes:  /clicked_point   (geometry_msgs/PointStamped)  — RViz Publish Point
  Publishes:   /basement_goals  (geometry_msgs/PoseArray)      — latched, QoS depth 1
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PointStamped, PoseArray, Pose
from std_msgs.msg import Bool


class BasementPointPublisher(Node):

    def __init__(self):
        super().__init__("basement_point_publisher")

        self._goals: list[Pose] = []
        self._published = False

        # Latched publisher so the state machine receives the goals even if it
        # starts after this node has already published.
        latch_qos = QoSProfile(depth=1,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._pub = self.create_publisher(PoseArray, "/basement_goals", latch_qos)

        # RViz "Publish Point" tool publishes here
        self._click_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self._click_cb,
            10,
        )

        # Optional reset topic
        self._reset_sub = self.create_subscription(
            Bool,
            "/basement_goals/reset",
            self._reset_cb,
            10,
        )

        self.get_logger().info(
            "BasementPointPublisher ready.\n"
            "  → In RViz, select 'Publish Point' and click two locations on the map."
        )

    # ------------------------------------------------------------------

    def _click_cb(self, msg: PointStamped):
        if self._published:
            self.get_logger().warn(
                "Goals already published. Publish to /basement_goals/reset to pick again."
            )
            return

        pose = Pose()
        pose.position.x = msg.point.x
        pose.position.y = msg.point.y
        pose.position.z = 0.0
        pose.orientation.w = 1.0  # heading doesn't matter for navigation goals

        self._goals.append(pose)
        n = len(self._goals)
        self.get_logger().info(
            f"Goal {n} set: ({msg.point.x:.3f}, {msg.point.y:.3f})"
        )

        if n == 2:
            self._publish()

    def _publish(self):
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = self._goals

        self._pub.publish(msg)
        self._published = True

        self.get_logger().info(
            f"Published /basement_goals with 2 locations:\n"
            f"  Goal 1: ({self._goals[0].position.x:.3f}, {self._goals[0].position.y:.3f})\n"
            f"  Goal 2: ({self._goals[1].position.x:.3f}, {self._goals[1].position.y:.3f})\n"
            f"State machine should now begin."
        )

    def _reset_cb(self, msg: Bool):
        if msg.data:
            self._goals = []
            self._published = False
            self.get_logger().info("Reset. Click two new points in RViz.")


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = BasementPointPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

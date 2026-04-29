#!/usr/bin/env python3
"""
state_machine.py  —  Mrs. Puff's Boating School (Part B)
Author: Nathaniel (Path Finding + State Machine)

This node drives the high-level logic for Part B of the RSS Final Challenge.
It re-uses the Lab 6 trajectory planner (A*/RRT) and pure-pursuit follower by
publishing goals to /goal_pose and letting the existing nodes do the rest.

Placeholders for teammates are clearly marked:
  # [WEIMING] — object detection (YOLO): parking meter, traffic light
  # [KEVIN]   — parking controller: precise stop within 1 m for 5 s

Safety controller (Diego) is FULLY EXTERNAL — his node publishes zero-velocity
drive commands directly to the drive topic at VESC level. The state machine
never enters a STOPPED state for traffic laws; the car pauses and resumes
naturally without any state machine involvement.

Goal locations are provided by basement_point_publisher.py (our testing node)
which listens for two RViz "Publish Point" clicks and publishes them as a
latched PoseArray on /basement_goals. On race day, swap in the TA's node.

Run alongside:
  ros2 launch path_planning_pkg real.launch.xml      (Lab 6 planner + follower)
  ros2 run final_challenge basement_point_publisher  (click 2 goals in RViz)
  ros2 run final_challenge state_machine
"""

import math
import rclpy

from enum import Enum, auto
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String


# ---------------------------------------------------------------------------
#  States
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE                = auto()  # Waiting for goal list from basement_point_publisher
    PLANNING            = auto()  # Goal sent to planner; waiting for trajectory
    NAVIGATING          = auto()  # Pure pursuit following path to current goal
    ARRIVED             = auto()  # Within arrival threshold; trigger sign detection
    DETECTING_SIGN      = auto()  # Waiting for YOLO to identify the parking meter
    PARKING             = auto()  # Kevin's parking controller executing maneuver
    PARKED              = auto()  # Holding 5-second stop in front of correct meter
    RECOVERING          = auto()  # Backing out of parking spot before replanning
    RETURNING_TO_START  = auto()  # Optional: navigating back to start for bonus pts
    DONE                = auto()  # All tasks complete


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class FinalChallengeStateMachine(Node):

    def __init__(self):
        super().__init__("final_challenge_state_machine")

        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        self.declare_parameter("odom_topic",             "/pf/pose/odom")
        self.declare_parameter("drive_topic",            "/vesc/ackermann_cmd")
        self.declare_parameter("arrival_threshold",      0.5)   # metres
        self.declare_parameter("park_duration",          5.0)   # seconds
        self.declare_parameter("return_to_start",        True)
        self.declare_parameter("planning_wait_secs",     1.0)   # wait after sending goal

        self.odom_topic        = self.get_parameter("odom_topic").value
        self.drive_topic       = self.get_parameter("drive_topic").value
        self.arrival_threshold = self.get_parameter("arrival_threshold").value
        self.park_duration     = self.get_parameter("park_duration").value
        self.return_to_start   = self.get_parameter("return_to_start").value
        self.planning_wait     = self.get_parameter("planning_wait_secs").value

        # ------------------------------------------------------------------ #
        #  Internal state                                                      #
        # ------------------------------------------------------------------ #
        self.state              = State.IDLE
        self.goal_locations     = []    # list of (x, y) from basement_point_publisher
        self.current_goal_idx   = 0
        self.current_pose       = None  # (x, y, yaw) — updated continuously from odom
        self.start_pose         = None  # saved once on first odom message
        self.park_start_time    = None
        self.recover_start_time = None  # time RECOVERING state was entered
        self.plan_sent_time     = None  # time goal was sent to planner

        # Signals set by teammate callbacks (None/False until integrated)
        self.detected_sign      = None  # str e.g. "parking_meter" — set by [WEIMING]
        self.parking_done       = False # set by [KEVIN]

        # ------------------------------------------------------------------ #
        #  Subscriptions                                                       #
        # ------------------------------------------------------------------ #

        # Goal locations from basement_point_publisher.py (our testing node) or
        # the TA's node on race day. Uses TRANSIENT_LOCAL (latched) QoS so we
        # receive the goals even if this node starts after the publisher already fired.
        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goals_sub = self.create_subscription(
            PoseArray,
            "/basement_goals",
            self._goals_cb,
            latch_qos,
        )

        # Odometry / particle-filter pose — tracked continuously, no explicit
        # "get pose" step needed before re-planning.
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            10,
        )

        # [WEIMING] YOLO sign detection result
        self.sign_sub = self.create_subscription(
            String,
            "/sign_detection/result",
            self._sign_cb,
            10,
        )

        # ------------------------------------------------------------------
        # [KEVIN] Parking controller done signal
        # Kevin's parking controller publishes True on /parking/done once the
        # car has stopped within 1 m of the correct meter.
        # ------------------------------------------------------------------
        # self.parking_done_sub = self.create_subscription(
        #     Bool,
        #     "/parking/done",
        #     self._parking_done_cb,
        #     10,
        # )

        # ------------------------------------------------------------------ #
        #  Publishers                                                          #
        # ------------------------------------------------------------------ #

        # Triggers the Lab 6 path planner (trajectory_planner.py)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Emergency stop / direct drive command
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        # [WEIMING] Trigger sign detection
        self.trigger_detection_pub = self.create_publisher(
            Bool, "/sign_detection/trigger", 10
        )

        # ------------------------------------------------------------------
        # [KEVIN] Parking trigger
        # Publish True to start Kevin's visual-servo parking controller.
        # His node steers toward the YOLO bounding box centroid and stops
        # when homography distance <= 1 m. No world-frame pose needed.
        # ------------------------------------------------------------------
        # self.parking_trigger_pub = self.create_publisher(
        #     Bool, "/parking/trigger", 10
        # )

        # ------------------------------------------------------------------ #
        #  State machine timer — runs at 10 Hz                                #
        # ------------------------------------------------------------------ #
        self.timer = self.create_timer(0.1, self._step)

        self.get_logger().info("FinalChallengeStateMachine ready — waiting for /basement_goals.")

    # ======================================================================
    #  Subscriber callbacks
    # ======================================================================

    def _goals_cb(self, msg: PoseArray):
        """Receive the 2 target locations from basement_point_publisher."""
        if self.state != State.IDLE:
            return
        self.goal_locations = [(p.position.x, p.position.y) for p in msg.poses]
        self.get_logger().info(
            f"Received {len(self.goal_locations)} goals: {self.goal_locations}"
        )

    def _odom_cb(self, msg: Odometry):
        """Track current pose from particle filter / odometry."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.current_pose = (p.x, p.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose
            self.get_logger().info(
                f"Start pose saved: ({p.x:.2f}, {p.y:.2f}, yaw={math.degrees(yaw):.1f} deg)"
            )

    # [WEIMING]
    def _sign_cb(self, msg: String):
        self.detected_sign = msg.data
        self.get_logger().info(f"Sign detected: {msg.data}")

    # ------------------------------------------------------------------
    # [KEVIN] Uncomment when parking controller is ready
    # ------------------------------------------------------------------
    # def _parking_done_cb(self, msg: Bool):
    #     if msg.data:
    #         self.parking_done = True

    # ======================================================================
    #  Helpers
    # ======================================================================

    def _stop(self):
        cmd = AckermannDriveStamped()
        cmd.drive.speed = 0.0
        cmd.drive.steering_angle = 0.0
        self.drive_pub.publish(cmd)

    def _start_recovery(self):
        """Transition into RECOVERING so the robot backs up before every replan."""
        self.recover_start_time = self.get_clock().now()
        self._to(State.RECOVERING)

    def _dist_to(self, xy):
        """Euclidean distance from current pose to (x, y)."""
        if self.current_pose is None:
            return float("inf")
        return math.hypot(self.current_pose[0] - xy[0], self.current_pose[1] - xy[1])

    def _send_goal(self, x, y, yaw=0.0):
        """
        Publish a PoseStamped to /goal_pose.
        The Lab 6 trajectory_planner.py will pick this up, run A*, and publish
        the resulting path to /trajectory/current for the follower to track.
        """
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_pub.publish(msg)
        self.plan_sent_time = self.get_clock().now()
        self.get_logger().info(f"Goal sent to planner: ({x:.2f}, {y:.2f})")

    def _to(self, new_state: State):
        self.get_logger().info(f"[STATE] {self.state.name} → {new_state.name}")
        self.state = new_state

    # ======================================================================
    #  Main state machine tick (10 Hz)
    # ======================================================================

    def _step(self):
        # Safety controller (Diego) handles traffic law stops externally at
        # VESC level — no interrupt logic needed here.
        {
            State.IDLE:               self._idle,
            State.PLANNING:           self._planning,
            State.NAVIGATING:         self._navigating,
            State.ARRIVED:            self._arrived,
            State.DETECTING_SIGN:     self._detecting_sign,
            State.PARKING:            self._parking,
            State.PARKED:             self._parked,
            State.RECOVERING:         self._recovering,
            State.RETURNING_TO_START: self._returning_to_start,
            State.DONE:               self._done,
        }[self.state]()

    # ======================================================================
    #  Per-state handlers
    # ======================================================================

    def _idle(self):
        """Wait until we have received both goal locations."""
        if len(self.goal_locations) >= 2:
            self.current_goal_idx = 0
            goal = self.goal_locations[0]
            self._send_goal(goal[0], goal[1])
            self._to(State.PLANNING)

    def _planning(self):
        """
        Wait a brief moment after sending the goal to give the planner time to
        compute and publish the trajectory. The trajectory_follower will pick up
        /trajectory/current automatically.

        A more robust approach (TODO): subscribe to /trajectory/current and only
        transition when a new message arrives after plan_sent_time.
        """
        if self.plan_sent_time is None:
            return
        elapsed = (self.get_clock().now() - self.plan_sent_time).nanoseconds / 1e9
        if elapsed >= self.planning_wait:
            goal = self.goal_locations[self.current_goal_idx]
            self.get_logger().info(
                f"Navigating to location {self.current_goal_idx + 1}: "
                f"({goal[0]:.2f}, {goal[1]:.2f})"
            )
            self._to(State.NAVIGATING)

    def _navigating(self):
        """
        Pure pursuit (trajectory_follower.py from Lab 6) is driving the car.
        We just monitor distance to the goal and transition when close enough.
        Traffic-law interrupts are handled in _step() above.
        """
        goal = self.goal_locations[self.current_goal_idx]
        if self._dist_to(goal) < self.arrival_threshold:
            self._stop()
            self._to(State.ARRIVED)

    def _arrived(self):
        """
        Reached the goal zone. Reset detection state and trigger YOLO.
        """
        self.detected_sign = None
        trigger = Bool()
        trigger.data = True
        self.trigger_detection_pub.publish(trigger)
        self.get_logger().info(
            f"Arrived at location {self.current_goal_idx + 1}. Triggering sign detection."
        )
        self._to(State.DETECTING_SIGN)

    def _detecting_sign(self):
        """
        Wait for YOLO to identify which of the three objects is at this location.
        self.detected_sign is set by _sign_cb() once the detector publishes a result.
        """
        if self.detected_sign is None:
        # Check timeout
            if not hasattr(self, '_detect_start_time') or self._detect_start_time is None:
                self._detect_start_time = self.get_clock().now()
            elapsed = (self.get_clock().now() - self._detect_start_time).nanoseconds / 1e9
            if elapsed > 10.0:  # 10-second timeout
                self.get_logger().warn("Sign detection timed out — skipping location.")
                self._detect_start_time = None
                self._advance_to_next_goal()
            return

        self._detect_start_time = None  # reset for next time
     
        if self.detected_sign == "parking_meter":
            self.get_logger().info("Parking meter confirmed — handing off to parking controller.")
            self.parking_done = False

            # ------------------------------------------------------------------
            # [KEVIN] Send parking target to Kevin's controller.
            # ------------------------------------------------------------------
            # target_pose = PoseStamped()
            # target_pose.header.frame_id = "map"
            # target_pose.header.stamp = self.get_clock().now().to_msg()
            # target_pose.pose.position.x = <meter_world_x>
            # target_pose.pose.position.y = <meter_world_y>
            # self.parking_target_pub.publish(target_pose)

            self._to(State.PARKING)
        else:
            self.get_logger().info(
                f"Detected '{self.detected_sign}' — not a parking meter. Skipping location."
            )
            self._start_recovery()

    def _advance_to_next_goal(self):
        self.current_goal_idx += 1
        if self.current_goal_idx < len(self.goal_locations):
            goal = self.goal_locations[self.current_goal_idx]
            self._send_goal(goal[0], goal[1])
            self._to(State.PLANNING)
        elif self.return_to_start and self.start_pose is not None:
            self.get_logger().info("All locations visited. Returning to start for +2 bonus.")
            self._send_goal(self.start_pose[0], self.start_pose[1])
            self._to(State.RETURNING_TO_START)
        else:
            self._to(State.DONE)

    def _parking(self):
        """
        Kevin's visual-servo parking controller steers toward the detected
        meter's bounding box centroid and stops when homography distance <= 1 m.
        No localization or world-frame pose required.
        self.parking_done is set by _parking_done_cb() once Kevin's node signals done.

        PLACEHOLDER: auto-advance for testing until Kevin's controller is live.
        Remove the else-branch once integrated.
        """
        if self.parking_done:
            self.get_logger().info("Parking complete. Starting 5-second hold.")
            self._stop()
            self.park_start_time = self.get_clock().now()
            self._to(State.PARKED)
        else:
            # PLACEHOLDER — remove once Kevin's visual-servo controller is integrated
            self.get_logger().warn(
                "PLACEHOLDER: parking controller not yet integrated. "
                "Auto-advancing to PARKED for testing.",
                throttle_duration_sec=2.0,
            )
            self._stop()
            self.park_start_time = self.get_clock().now()
            self._to(State.PARKED)

    def _parked(self):
        """
        Hold stop for the required 5 seconds (spec requirement).
        Then navigate to the next location, or return to start for bonus.
        """
        self._stop()
        elapsed = (self.get_clock().now() - self.park_start_time).nanoseconds / 1e9
        if elapsed < self.park_duration:
            return

        self.get_logger().info(
            f"Held for {self.park_duration:.0f}s at location {self.current_goal_idx + 1}. "
            "Backing out before replanning."
        )
        self._start_recovery()

    def _recovering(self):
        """
        Back slowly out of the parking spot for 1.5 s so the robot is clear of
        the meter and back in open corridor space before replanning.  No pose
        knowledge required — open-loop reverse gives the particle filter time to
        reconverge on familiar geometry before the next goal is issued.
        """
        elapsed = (self.get_clock().now() - self.recover_start_time).nanoseconds / 1e9
        if elapsed < 1.5:
            cmd = AckermannDriveStamped()
            cmd.drive.speed = -0.5          # slow reverse
            cmd.drive.steering_angle = 0.0  # straight back
            self.drive_pub.publish(cmd)
        else:
            self._stop()
            self.get_logger().info("Recovery complete — resuming navigation.")
            self._advance_to_next_goal()

    def _returning_to_start(self):
        """
        Optional return trip to earn +2 bonus points.
        Diego's safety controller handles any traffic law stops on this leg too.
        """
        if self.start_pose is None:
            self._to(State.DONE)
            return

        if self._dist_to((self.start_pose[0], self.start_pose[1])) < self.arrival_threshold:
            self._stop()
            self.get_logger().info("Returned to start! +2 bonus points.")
            self._to(State.DONE)

    def _done(self):
        """Challenge complete. Sit still and log."""
        self._stop()
        self.get_logger().info("Challenge complete!", throttle_duration_sec=5.0)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = FinalChallengeStateMachine()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

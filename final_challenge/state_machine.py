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

Safety controller (Diego) is FULLY EXTERNAL — it subscribes to
/vesc/high_level/input/nav_0 to learn nav intent and publishes its
own commands on /vesc/low_level/input/safety, which has higher mux
priority and therefore overrides nav. The state machine never enters
a STOPPED state for traffic laws; the car pauses and resumes naturally
without any state machine involvement. For this to work, every node
that emits drive commands (including this one) must publish to
/vesc/high_level/input/nav_0 — NOT directly to /vesc/ackermann_cmd
or /vesc/low_level/input/navigation, both of which bypass safety.

Goal locations are provided by basement_point_publisher.py (our testing node)
which listens for two RViz "Publish Point" clicks and publishes them as a
latched PoseArray on /basement_goals. On race day, swap in the TA's node.

Run alongside:
  ros2 launch path_planning_pkg real.launch.xml      (Lab 6 planner + follower)
  ros2 run final_challenge basement_point_publisher  (click 2 goals in RViz)
  ros2 run final_challenge state_machine

─── HEADHUNTER ───────────────────────────────────────────────────────────────
Problem: if the TA sets a goal point on top of or past the sign, the path
planner drives the car right there — safety controller fires before parking
controller ever gets a chance, and we stall in NAVIGATING forever.

Fix: while navigating and within HEADHUNTER_RADIUS metres of the goal, poll
YOLO's /yolo/parking_meter/roi. If the sign is visible for
HEADHUNTER_DEBOUNCE_TICKS consecutive state-machine ticks, immediately kill
path planning and jump straight to PARKING — skipping ARRIVED and
DETECTING_SIGN entirely.

Killing path planning means publishing a zero-pose PoseArray to
/trajectory/current so the trajectory_follower silences itself (len(pts) < 2
in pose_callback → returns without publishing). Both the HEADHUNTER and the
normal ARRIVED transition do this.
──────────────────────────────────────────────────────────────────────────────
"""

import math
import rclpy

from enum import Enum, auto
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import Bool, String


# ─────────────────────────────────────────────────────────────────────────────
#  HEADHUNTER tuning — adjust these without touching logic
# ─────────────────────────────────────────────────────────────────────────────

# Distance from the TA goal at which HEADHUNTER starts watching YOLO (metres).
# If the sign is visible while the car is within this bubble, skip straight to
# PARKING.  Increase if the sign isn't in frame early enough; decrease if you
# get false triggers far from the goal.
HEADHUNTER_RADIUS = 2.0   # metres — tune here

# Number of consecutive 10 Hz state-machine ticks with a valid parking-meter
# ROI before HEADHUNTER commits.  At 10 Hz, 3 ticks = ~300 ms debounce.
# Increase to reduce false positives; decrease to react faster.
HEADHUNTER_DEBOUNCE_TICKS = 3


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
        self.declare_parameter("drive_topic",            "/vesc/high_level/input/nav_0")
        self.declare_parameter("arrival_threshold",      .1)  # metres
        self.declare_parameter("park_duration",          5.0)   # seconds
        self.declare_parameter("return_to_start",        True)
        self.declare_parameter("planning_wait_secs",     1.0)
        self.declare_parameter("recovery_duration",      3.0)   # seconds to reverse

        self.odom_topic        = self.get_parameter("odom_topic").value
        self.drive_topic       = self.get_parameter("drive_topic").value
        self.arrival_threshold = self.get_parameter("arrival_threshold").value
        self.park_duration     = self.get_parameter("park_duration").value
        self.return_to_start   = self.get_parameter("return_to_start").value
        self.planning_wait     = self.get_parameter("planning_wait_secs").value
        self.recovery_duration = self.get_parameter("recovery_duration").value

        # ------------------------------------------------------------------ #
        #  Internal state                                                      #
        # ------------------------------------------------------------------ #
        self.state              = State.IDLE
        self.goal_locations     = []
        self.current_goal_idx   = 0
        self.current_pose       = None   # (x, y, yaw)
        self.start_pose         = None
        self.park_start_time    = None
        self.recover_start_time = None
        self.plan_sent_time     = None

        # Teammate signals
        self.detected_sign  = None   # str e.g. "parking meter" — set by [WEIMING]
        self.parking_done   = False  # set by [KEVIN]

        # HEADHUNTER state
        self.headhunter_roi        = None   # latest RegionOfInterest from YOLO
        self.headhunter_hit_count  = 0      # consecutive ticks with valid ROI

        # DETECTING_SIGN timeout
        self._detect_start_time = None

        # ------------------------------------------------------------------ #
        #  Subscriptions                                                       #
        # ------------------------------------------------------------------ #
        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goals_sub = self.create_subscription(
            PoseArray,
            "/basement_goals",
            self._goals_cb,
            latch_qos,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            10,
        )

        # Watch what pure pursuit is publishing on the drive topic so we can
        # see the steering/speed it commands while NAVIGATING. Logged only in
        # that state to keep the rest of the run quiet.
        self.drive_monitor_sub = self.create_subscription(
            AckermannDriveStamped,
            self.drive_topic,
            self._drive_monitor_cb,
            10,
        )

        # [WEIMING] YOLO sign detection result
        self.sign_sub = self.create_subscription(
            String,
            "/sign_detection/result",
            self._sign_cb,
            10,
        )

        # [KEVIN] Parking controller done signal
        self.parking_done_sub = self.create_subscription(
            Bool,
            "/parking/done",
            self._parking_done_cb,
            10,
        )

        # HEADHUNTER — watch YOLO's parking-meter ROI at all times.
        # yolo_node publishes every frame: width=0/height=0 means not detected.
        self.headhunter_roi_sub = self.create_subscription(
            RegionOfInterest,
            "/yolo/parking_meter/roi",
            self._headhunter_roi_cb,
            10,
        )
        self.get_logger().info(
            "[HEADHUNTER] Subscribed to /yolo/parking_meter/roi — "
            f"activation radius={HEADHUNTER_RADIUS} m, "
            f"debounce={HEADHUNTER_DEBOUNCE_TICKS} ticks"
        )

        # ------------------------------------------------------------------ #
        #  Publishers                                                          #
        # ------------------------------------------------------------------ #

        # Lab 6 path planner goal
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Emergency stop / direct drive command
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        # [WEIMING] Trigger sign detection
        self.trigger_detection_pub = self.create_publisher(
            Bool, "/sign_detection/trigger", 10
        )

        # [KEVIN] Parking trigger
        self.parking_trigger_pub = self.create_publisher(
            Bool, "/parking/trigger", 10
        )

        # Trajectory override — publishing a zero-pose PoseArray here causes
        # trajectory_follower.py to call trajectory_callback (setting
        # initialized_traj=True) but then pose_callback sees len(pts)<2 and
        # returns without publishing any drive command.  This is the correct
        # way to silence the follower while parking controller takes over.
        # Without this, both nodes publish to /vesc/high_level/input/nav_0 and
        # race each other — the follower wins because it runs at odom rate.
        self.trajectory_override_pub = self.create_publisher(
            PoseArray, "/trajectory/current", 10
        )

        # ------------------------------------------------------------------ #
        #  State machine timer — 10 Hz                                        #
        # ------------------------------------------------------------------ #
        self.timer = self.create_timer(0.1, self._step)

        self.get_logger().info("FinalChallengeStateMachine ready — waiting for /basement_goals.")

    # ======================================================================
    #  Subscriber callbacks
    # ======================================================================

    def _goals_cb(self, msg: PoseArray):
        if self.state != State.IDLE:
            return
        self.goal_locations = [(p.position.x, p.position.y) for p in msg.poses]
        self.get_logger().info(
            f"Received {len(self.goal_locations)} goals: {self.goal_locations}"
        )

    def _odom_cb(self, msg: Odometry):
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

    def _parking_done_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info("[DEBUG] /parking/done received True — parking complete.")
            self.parking_done = True

    def _drive_monitor_cb(self, msg: AckermannDriveStamped):
        """Log drive commands hitting the nav drive topic — but only while
        NAVIGATING, since pure pursuit owns the topic in that state. Outside
        NAVIGATING, this topic is also written by state_machine itself
        (_stop, _recovering) and parking_controller, so logging there would
        just be noise.
        """
        if self.state != State.NAVIGATING:
            return
        self.get_logger().info(
            f"[NAVIGATING][PP→drive] speed={msg.drive.speed:+.2f} m/s  "
            f"steer={math.degrees(msg.drive.steering_angle):+.1f} deg "
            f"({msg.drive.steering_angle:+.3f} rad)",
            throttle_duration_sec=0.5,
        )

    def _headhunter_roi_cb(self, msg: RegionOfInterest):
        """Cache the latest parking-meter ROI from yolo_node.
        Called at camera/YOLO rate — just store, don't act.
        Action happens in _navigating() at the state-machine tick rate.
        """
        self.headhunter_roi = msg

    # ======================================================================
    #  Helpers
    # ======================================================================

    def _stop(self):
        cmd = AckermannDriveStamped()
        cmd.drive.speed = 0.0
        cmd.drive.steering_angle = 0.0
        # Active brake — VESC honors negative acceleration as a deceleration
        # target. Without this, speed=0 is interpreted as "duty cycle 0 /
        # coast" and the car drifts on momentum. Matches safety_controller's
        # brake_accel (8.0 m/s²).
        cmd.drive.acceleration = -8.0
        self.drive_pub.publish(cmd)

    def _kill_trajectory_follower(self):
        """
        Publish a zero-pose PoseArray to /trajectory/current.

        trajectory_follower.trajectory_callback detects len(poses)<2, clears
        initialized_traj, and publishes its own zero AckermannDriveStamped so
        the LAST command on the drive topic is an explicit stop.  Without this
        contract, the VESC would hold whatever steering angle pure pursuit had
        last computed at the goal and the car would swerve during handoff.

        Called both from _arrived() (normal path) and _headhunter_activate()
        so parking_controller can own the drive topic without fighting pure
        pursuit.
        """
        empty_traj = PoseArray()
        empty_traj.header.frame_id = "map"
        empty_traj.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_override_pub.publish(empty_traj)
        self.get_logger().info(
            "[DEBUG] Empty PoseArray published to /trajectory/current — "
            "trajectory_follower silenced."
        )

    def _start_recovery(self):
        self.recover_start_time = self.get_clock().now()
        self._to(State.RECOVERING)

    def _dist_to(self, xy):
        if self.current_pose is None:
            return float("inf")
        return math.hypot(self.current_pose[0] - xy[0], self.current_pose[1] - xy[1])

    def _send_goal(self, x, y, yaw=0.0):
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

    def _arm_parking(self, source: str):
        """
        Shared activation sequence used by both normal DETECTING_SIGN path and
        HEADHUNTER path.  Fires sign detection trigger (arms /relative_cone_px
        feed) then fires parking trigger.

        source: a short string for debug logging, e.g. "DETECTING_SIGN" or
                "HEADHUNTER" so logs clearly show which path was taken.
        """
        self.get_logger().info(
            f"[DEBUG] [{source}] Arming sign detector — "
            "publishing /sign_detection/trigger=True"
        )
        self.trigger_detection_pub.publish(Bool(data=True))

        self.get_logger().info(
            f"[DEBUG] [{source}] Activating parking controller — "
            "publishing /parking/trigger=True"
        )
        trigger = Bool()
        trigger.data = True
        self.parking_trigger_pub.publish(trigger)

    def _headhunter_activate(self):
        """
        Called when HEADHUNTER conditions are met in _navigating().
        Resets all flags that would normally be set by ARRIVED/DETECTING_SIGN,
        kills the trajectory follower, arms the perception stack, and jumps
        straight to PARKING.
        """
        self.get_logger().warn(
            f"[HEADHUNTER] *** TRIGGERED *** — sign visible within "
            f"{HEADHUNTER_RADIUS} m of goal, skipping ARRIVED+DETECTING_SIGN. "
            f"ROI: x_offset={self.headhunter_roi.x_offset} "
            f"y_offset={self.headhunter_roi.y_offset} "
            f"w={self.headhunter_roi.width} h={self.headhunter_roi.height}"
        )

        # ── Reset state flags ────────────────────────────────────────────
        # parking_done MUST be False — on goal 2 it could still be True
        # from goal 1 and would cause an immediate fall-through to PARKED.
        if self.parking_done:
            self.get_logger().warn(
                "[HEADHUNTER] parking_done was True — resetting to False. "
                "This would have caused an instant PARKED skip."
            )
        self.parking_done = False

        # detected_sign is normally set by _sign_cb in DETECTING_SIGN.
        # We know it's a parking meter because YOLO told us, so set it now.
        self.detected_sign = "parking meter"
        self.get_logger().info(
            "[HEADHUNTER] detected_sign set to 'parking meter' directly."
        )

        # Reset the DETECTING_SIGN timeout in case it was ever set.
        self._detect_start_time = None

        # Reset HEADHUNTER debounce counter for next goal.
        self.headhunter_hit_count = 0

        # ── Kill trajectory follower ─────────────────────────────────────
        # Without this, pure pursuit and parking_controller both publish to
        # /vesc/high_level/input/nav_0 at the same time and fight.
        self._stop()
        self._kill_trajectory_follower()

        # ── Arm perception stack and parking controller ──────────────────
        self._arm_parking("HEADHUNTER")

        # ── Transition ───────────────────────────────────────────────────
        self._to(State.PARKING)

    # ======================================================================
    #  Main state machine tick (10 Hz)
    # ======================================================================

    def _step(self):
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
        if len(self.goal_locations) >= 2:
            self.current_goal_idx = 0
            goal = self.goal_locations[0]
            self._send_goal(goal[0], goal[1])
            self._to(State.PLANNING)

    def _planning(self):
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
        Pure pursuit is driving the car.  Two exit conditions:

        1. Normal arrival: Euclidean distance to goal drops below
           arrival_threshold (0.75 m).  → ARRIVED

        2. HEADHUNTER: while within HEADHUNTER_RADIUS of the goal AND the
           parking-meter ROI from yolo_node is non-zero for
           HEADHUNTER_DEBOUNCE_TICKS consecutive ticks → skip ARRIVED and
           DETECTING_SIGN, jump straight to PARKING.
        """
        if self.current_pose is None:
            return

        goal = self.goal_locations[self.current_goal_idx]
        dist = self._dist_to(goal)

        self.get_logger().info(
            f"[NAVIGATING] goal_idx={self.current_goal_idx} "
            f"goal=({goal[0]:.2f}, {goal[1]:.2f}) "
            f"pose=({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}) "
            f"dist={dist:.3f} m  "
            f"arrival_threshold={self.arrival_threshold} m  "
            f"headhunter_radius={HEADHUNTER_RADIUS} m",
            throttle_duration_sec=1.0,
        )

        # ── Normal arrival exit ──────────────────────────────────────────
        if dist < self.arrival_threshold:
            self._stop()
            self._to(State.ARRIVED)
            return

        # ── HEADHUNTER exit ──────────────────────────────────────────────
        if dist < HEADHUNTER_RADIUS:
            roi = self.headhunter_roi
            roi_valid = (
                roi is not None
                and roi.width > 0
                and roi.height > 0
            )

            if roi_valid:
                self.headhunter_hit_count += 1
                self.get_logger().info(
                    f"[HEADHUNTER] Active — dist={dist:.3f} m < radius={HEADHUNTER_RADIUS} m, "
                    f"ROI w={roi.width} h={roi.height}, "
                    f"hit_count={self.headhunter_hit_count}/{HEADHUNTER_DEBOUNCE_TICKS}"
                )
                if self.headhunter_hit_count >= HEADHUNTER_DEBOUNCE_TICKS:
                    self._headhunter_activate()
            else:
                # Inside radius but no valid ROI — reset debounce counter.
                if self.headhunter_hit_count > 0:
                    self.get_logger().info(
                        f"[HEADHUNTER] Inside radius but ROI invalid/missing — "
                        f"resetting hit_count {self.headhunter_hit_count} → 0"
                    )
                self.headhunter_hit_count = 0
        else:
            # Outside HEADHUNTER_RADIUS — ensure debounce counter is clear.
            if self.headhunter_hit_count > 0:
                self.headhunter_hit_count = 0

    def _arrived(self):
        """
        Reached the goal zone via normal distance threshold.
        Kill the trajectory follower before handing off to parking stack.
        Without _kill_trajectory_follower(), pure pursuit keeps publishing
        drive commands at odom rate and fights parking_controller on the
        same /vesc/high_level/input/nav_0 topic.
        """
        # Flood a zero on the drive topic before we touch the follower. The
        # NAVIGATING→ARRIVED transition publishes one stop ~100 ms ago, but
        # pure pursuit's pose_callback (50 Hz) keeps firing forward+steer
        # commands until trajectory_callback processes the empty PoseArray.
        # That window is what produces the "swerve at the goal" — adding this
        # second zero now narrows it.
        self._stop()
        self.get_logger().info(
            f"[ARRIVED] Within {self.arrival_threshold} m of goal "
            f"{self.current_goal_idx + 1}. "
            "Silencing trajectory follower, triggering sign detection."
        )
        self._kill_trajectory_follower()

        self.detected_sign = None
        self._detect_start_time = None

        # Reset HEADHUNTER counter for this leg (may have accumulated non-zero).
        self.headhunter_hit_count = 0

        # Fire the sign detection trigger False then True to re-arm the one-shot.
        self.trigger_detection_pub.publish(Bool(data=False))
        self.trigger_detection_pub.publish(Bool(data=True))
        self.get_logger().info(
            "[DEBUG] [ARRIVED] /sign_detection/trigger pulsed False→True"
        )
        self._to(State.DETECTING_SIGN)

    def _detecting_sign(self):
        """
        Wait for YOLO to identify which of the three objects is a parking meter.
        self.detected_sign is set by _sign_cb() once the detector publishes.
        """
        if self.detected_sign is None:
            if self._detect_start_time is None:
                self._detect_start_time = self.get_clock().now()
            elapsed = (self.get_clock().now() - self._detect_start_time).nanoseconds / 1e9
            if elapsed > 10.0:
                self.get_logger().warn("Sign detection timed out — skipping location.")
                self._detect_start_time = None
                # Disarm sign detector before recovering.
                self.trigger_detection_pub.publish(Bool(data=False))
                self.get_logger().info(
                    "[DEBUG] [DETECTING_SIGN] timeout — "
                    "/sign_detection/trigger=False published"
                )
                self._start_recovery()
            return

        self._detect_start_time = None

        if self.detected_sign == "parking meter":
            self.get_logger().info(
                "Parking meter confirmed — handing off to parking controller."
            )
            self.parking_done = False
            self._arm_parking("DETECTING_SIGN")
            self._to(State.PARKING)
        else:
            self.get_logger().info(
                f"Detected '{self.detected_sign}' — not a parking meter. "
                "Disarming sign detector and skipping location."
            )
            self.trigger_detection_pub.publish(Bool(data=False))
            self.get_logger().info(
                "[DEBUG] [DETECTING_SIGN] wrong sign — "
                "/sign_detection/trigger=False published"
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
        Wait for parking_controller.py to finish its visual-servo approach.
        The parking controller owns the drive topic while we're here.
        The trajectory follower was already silenced in _arrived() or
        _headhunter_activate() before we got here, so there is no conflict.
        """
        if self.parking_done:
            self.get_logger().info(
                "[DEBUG] [PARKING] parking_done=True received — "
                "stopping car and entering PARKED hold."
            )
            self._stop()
            self.park_start_time = self.get_clock().now()
            self._to(State.PARKED)

    def _parked(self):
        """
        Hold stop for the required 5 seconds.
        Image save happens here (not in DETECTING_SIGN) — the car is stationary
        and close to the sign so the bounding box is large and blur-free.
        The save is triggered by the state machine's entry into this state
        (parking_controller publishes /parking/done → _parking() fires _stop()
        and transitions here); sign_detector.py saves on the next valid frame
        it receives while /sign_detection/trigger is still True.
        """
        self._stop()
        elapsed = (self.get_clock().now() - self.park_start_time).nanoseconds / 1e9
        if elapsed < self.park_duration:
            return

        self.get_logger().info(
            f"Held for {self.park_duration:.0f} s at location "
            f"{self.current_goal_idx + 1}. Disarming sign detector, backing out."
        )
        # Disarm sign detector so it re-arms cleanly for the next location.
        self.trigger_detection_pub.publish(Bool(data=False))
        self.get_logger().info(
            "[DEBUG] [PARKED] /sign_detection/trigger=False published"
        )
        self._start_recovery()

    def _recovering(self):
        """
        Back slowly out of the parking spot for recovery_duration seconds so
        the robot is clear of the sign before replanning.
        """
        elapsed = (self.get_clock().now() - self.recover_start_time).nanoseconds / 1e9
        if elapsed < self.recovery_duration:
            cmd = AckermannDriveStamped()
            cmd.drive.speed = -1.0
            cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(cmd)
            self.get_logger().info(
                f"[RECOVERING] elapsed={elapsed:.2f}/{self.recovery_duration:.1f}s "
                f"publishing speed={cmd.drive.speed:+.2f} steer={cmd.drive.steering_angle:+.2f}",
                throttle_duration_sec=0.3,
            )
        else:
            self._stop()
            self.get_logger().info("Recovery complete — resuming navigation.")
            self._advance_to_next_goal()

    def _returning_to_start(self):
        if self.start_pose is None:
            self._to(State.DONE)
            return
        if self._dist_to((self.start_pose[0], self.start_pose[1])) < self.arrival_threshold:
            self._stop()
            self.get_logger().info("Returned to start! +2 bonus points.")
            self._to(State.DONE)

    def _done(self):
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

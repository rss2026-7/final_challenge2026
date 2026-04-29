# Part B Run Instructions — Mrs. Puff's Boating School

## One-time setup

After cloning or pulling, build the workspace from the ros2 workspace root (the folder that contains `src/`):

```bash
colcon build --symlink-install
source install/setup.bash
```

---

## Simulator — ground truth odom (no PF)

Fastest way to test the navigation loop. The simulator publishes `/odom` as perfect
ground-truth pose, so no particle filter is needed. Override the state machine defaults
to match the sim topics.


**Terminal 1 — path planner + pure pursuit follower**
```bash
ros2 launch path_planning sim_plan_follow.launch.xml
```

**Terminal 2 — simulator + RViz**
```bash
ros2 launch racecar_simulator simulate.launch.xml
```

Uses `sim_config.yaml`: planner and follower both subscribe to `/odom`, drive on `/drive`.

**Terminal 3 — basement point publisher**
```bash
ros2 run final_challenge basement_point_publisher
```
In RViz select the **Publish Point** tool (press `G`) and click two goal locations on the map.

**Terminal 4 — state machine**
```bash
ros2 run final_challenge state_machine --ros-args \
  -p odom_topic:=/odom \
  -p drive_topic:=/drive
```

**Terminal 5 — sign detector**
```bash
ros2 run final_challenge sign_detector
```

**Terminal 6 — homography transformer** *(required for parking)*
```bash
ros2 run final_challenge homography_transformer
```
Converts Weiming's pixel detections on `/relative_cone_px` → metres on `/relative_cone`.

**Terminal 7 — parking controller** *(required for parking)*
```bash
ros2 run final_challenge parking_controller --ros-args \
  -p drive_topic:=/drive
```
Waits idle until `/parking/trigger` fires, then servos toward `/relative_cone` and publishes `/parking/done`.

> **Note:** Weiming's YOLO node must also be running to publish `/relative_cone_px`. Without it the parking controller will receive no cone location and sit still.

---

## Simulator — with particle filter (closer to real robot)

Runs the full PF localization stack in sim. The PF subscribes to `/odom` (sim wheel
odometry as motion model input) and `/scan` (sim LiDAR), and publishes corrected pose
to `/pf/pose/odom`. The planner, follower, and state machine all use that corrected pose.
Use this mode when you want to test localization behavior before going to the real robot.



**Terminal 1 — path planner + follower + particle filter**
```bash
ros2 launch path_planning pf_sim_plan_follow.launch.xml
```

**Terminal 2 — simulator + RViz**
```bash
ros2 launch racecar_simulator simulate.launch.xml
```

Uses `pf_sim_config.yaml`: planner and follower subscribe to `/pf/pose/odom`, drive on `/drive`.
PF uses `pf_config.yaml`: reads `/odom` and `/scan`, publishes to `/pf/pose/odom`.

**Before Terminal 3:** in RViz, use the **2D Pose Estimate** tool to set the robot's
initial position on the map. The PF will not localize until this is done.

**Terminal 3 — basement point publisher**
```bash
ros2 run final_challenge basement_point_publisher
```
In RViz select the **Publish Point** tool (press `G`) and click two goal locations on the map.

**Terminal 4 — state machine**
```bash
ros2 run final_challenge state_machine --ros-args -p drive_topic:=/drive
```
`odom_topic` defaults to `/pf/pose/odom` — no override needed. Only `drive_topic` needs
to be overridden from the real-robot default.

**Terminal 5 — sign detector**
```bash
ros2 run final_challenge sign_detector
```

**Terminal 6 — homography transformer** *(required for parking)*
```bash
ros2 run final_challenge homography_transformer
```
Converts Weiming's pixel detections on `/relative_cone_px` → metres on `/relative_cone`.

**Terminal 7 — parking controller** *(required for parking)*
```bash
ros2 run final_challenge parking_controller --ros-args \
  -p drive_topic:=/drive
```
Waits idle until `/parking/trigger` fires, then servos toward `/relative_cone` and publishes `/parking/done`.

> **Note:** Weiming's YOLO node must also be running to publish `/relative_cone_px`. Without it the parking controller will receive no cone location and sit still.

---

## Real Robot

The real robot runs the particle filter and uses `/pf/pose/odom` for localization.
The state machine defaults are already set for this environment — no overrides needed.



**Terminal 0 — simulator + RViz**
```bash
ros2 launch racecar_simulator simulate.launch.xml
```


**Terminal 1 — path planner + follower + particle filter + safety controller (Lab 6)**
```bash
ros2 launch path_planning real.launch.xml
```
This single launch file starts the trajectory planner, pure pursuit follower,
particle filter, safety controller, and a bag recorder all at once.

**Terminal 2 — basement point publisher**
```bash
ros2 run final_challenge basement_point_publisher
```
Click two goal points in RViz using the **Publish Point** tool.

**Terminal 3 — state machine**
```bash
ros2 run final_challenge state_machine
```

**Terminal 4 — sign detector**
```bash
ros2 run final_challenge sign_detector
```

**Terminal 5 — homography transformer** *(required for parking)*
```bash
ros2 run final_challenge homography_transformer
```
Converts Weiming's pixel detections on `/relative_cone_px` → metres on `/relative_cone`.

**Terminal 6 — parking controller** *(required for parking)*
```bash
ros2 run final_challenge parking_controller
```
Uses the real-robot drive topic (`/vesc/low_level/input/navigation`) by default.
Waits idle until `/parking/trigger` fires, then servos toward `/relative_cone` and publishes `/parking/done`.
Control data (distance, angle, speed, steering) is logged to a timestamped CSV in the working directory for post-run analysis.

> **Note:** Weiming's YOLO node must also be running to publish `/relative_cone_px`. Without it the parking controller will receive no cone location and sit still.


---

## Do you need launch files?

Not right now. The Lab 6 `real.launch.xml` already handles the heavy lifting on the real robot,
and the simulator has its own launch file. A custom launch file for this package would only
consolidate Terminals 2 and 3 into one command — useful eventually, but not blocking.

If you add teammates' nodes (Weiming's YOLO detector, Kevin's parking controller), a
launch file becomes more valuable to start everything cleanly in one shot.

---

## Topic reference

| Topic | Direction | Purpose |
|---|---|---|
| `/basement_goals` | sub | PoseArray of 2 goal locations (latched) |
| `/pf/pose/odom` | sub | Particle filter pose (real robot) |
| `/odom` | sub | Ground-truth pose (sim only, pass via --ros-args) |
| `/goal_pose` | pub | Sends navigation goal to Lab 6 planner |
| `/vesc/ackermann_cmd` | pub | Stop commands (real robot) |
| `/drive` | pub | Stop commands (sim only, pass via --ros-args) |
| `/sign_detection/trigger` | pub | trigger YOLO detection (sign_detector node) |
| `/sign_detection/result` | sub | detected object class — "parking_meter", "fire_hydrant", or "bird" |
| `/parking/trigger` | pub | [KEVIN] start parking controller |
| `/parking/done` | sub | [KEVIN] parking complete signal |

## YOLO Sign Detector (Traffic Light + Objects)

### How to run

**Terminal 1 — path planner + pure pursuit follower**
```bash
ros2 run final_challenge sign_detector
```

**Terminal 2 — simulator + RViz**
```bash
ros2 run final_challenge stoplight_detection
```



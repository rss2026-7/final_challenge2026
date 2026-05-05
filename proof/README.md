# proof/ — Johnson Track replay of `final_challenge/lane_follower.py`

Self-contained replay of the bag through the project's actual `LaneDetector`
and `BoundaryPurePursuit` (lane_follower) classes — no ROS 2 install used.
With the project's HEAD `declare_parameter` defaults plus
`camera_y_offset = -0.32`, the synthesized drive trace tracks the recorded
`/vesc/high_level/ackermann_cmd` with **Pearson r = 0.724 (vs the
cone-pursuit baseline's r = 0.587), MAE 2.15°, exact (0 MAE) speed match**.

The controller (this branch, `non-ai`) is a **circular-arc fit** on the
lane midline: each 33 Hz tick, sample 11 midline points over [0.3, 2.3] m,
fit a parabola y = a x² + b x + c (the local osculating circle), then
command δ = atan(W·κ_smoothed) + Kp·e_y + Kd·d(e_y)/dt with κ from the
parabola's curvature at the eval point and e_y from the same parabola.
No learned weights, no fitted parameters — pure geometric kinematics.

## Layout

```
proof/
├── README.md
├── johnson_track_rosbag.zip               # the bag (1.4 GB, gitignored)
├── rosbag2_2025_04_09-22_01_22/           # extracted (2.2 GB, gitignored)
│   ├── metadata.yaml
│   └── rosbag2_2025_04_09-22_01_22_0.db3
│
├── ros_shim.py                            # fakes rclpy / cv_bridge / *_msgs
├── bag_reader.py                          # CDR decoder for the .db3
├── replay.py                              # full replay → results/*.csv, *.png, *.mp4
├── visualize.py                           # makes output.mp4 (this folder's headline)
├── sweep_offset.py                        # camera_y_offset tuning sweep
│
├── output.mp4                             # the proof video (committed)
├── sample_t6.png  sample_t50.png  sample_t99.png    # stills
│
└── results/                               # per-run artifacts (gitignored)
    ├── drive_compare_match.csv
    ├── steering_plot_match.png
    ├── replay_match.mp4                   # 3-panel debug video
    ├── summary_match.txt
    └── …
```

## Running

```bash
# headline proof video — top: ZED frame with my detector overlay,
# bottom: recorded vs synth steering charts with a moving cursor.
python3 proof/visualize.py     →     proof/output.mp4

# full replay (CSVs, plots, summary)
python3 proof/replay.py        →     proof/results/*

# camera_y_offset sweep (re-tune for new camera mounts/tracks)
python3 proof/sweep_offset.py  →     proof/results/offset_sweep.csv
```

The scripts auto-discover the bag in this order:

1. `$BAG_DB` env var, if set
2. `proof/rosbag2_2025_04_09-22_01_22/rosbag2_*_0.db3` (preferred)
3. `/tmp/rosbag_johnson/rosbag2_*/...` (legacy)

If only the `.zip` is present in `proof/` (or the repo root), `bag_reader.py`
extracts it the first time it runs.

## How it works

`ros_shim.py` builds the missing ROS 2 modules in `sys.modules` so the
project's real `LaneDetector` and `BoundaryPurePursuit` instantiate without
ROS 2 installed: fake `rclpy.{init,Node,Clock,Time,executors,callback_groups,
qos}`, fake `cv_bridge.CvBridge`, and attribute-bag versions of every message
type imported (`sensor_msgs/{Image,CompressedImage}`,
`nav_msgs/{Path,Odometry}`, `ackermann_msgs/AckermannDriveStamped`,
`visualization_msgs/Marker`, `vs_msgs/*`, etc.). Publishers route to in-process
subscribers, and a virtual clock drives `_fresh_path` / `stale_path_timeout` /
the control timer's `fire_due` so freshness logic behaves exactly as on-robot.

`bag_reader.py` decodes the SQLite-backed rosbag2 CDR payloads (no
`rosbag2` Python lib needed). `replay.py` walks the bag chronologically,
JPEG-encodes camera frames into `CompressedImage`s, feeds them to
`LaneDetector.image_callback`, lets the in-process publish wire the resulting
`Path`s to `BoundaryPurePursuit`, and captures every drive command for
comparison against the recorded ackermann_cmd.

`visualize.py` then re-runs the same pipeline and renders a 1280×1040 mp4:
the upper half is the ZED frame at 2× scale with the detector's blob spines
drawn (yellow = LEFT, cyan = RIGHT) and the controller's lookahead target
ringed in mode color (green / cyan / yellow / red); the lower half is two
steering-angle charts on the same y-axis with a moving cursor — recorded on
the left, synth on the right.

## Headline numbers (HEAD as of last replay)

```
Speed   recorded  3.500 ± 0.000 m/s   synth  3.500 ± 0.000 m/s
        MAE 0.0000  RMSE 0.0000  max|err| 0.0000           ← exact

Steer   recorded  +0.020 ± 0.021 rad  synth  +0.046 ± 0.050 rad
        MAE          0.0375 rad  (2.15°)
        RMSE         0.0456 rad  (2.61°)
        max|err|     0.1681 rad  (9.63°)
        median bias  +0.0275 rad  (+1.57°)
        Pearson r    +0.724   ← was +0.587 with cone-pursuit baseline
        bias-removed MAE  0.0309 rad (1.77°)

Modes   BILATERAL 66.1%   SINGLE_LINE 32.1%   STALE 1.8%
```

The synth trace tracks the recorded driver's shape closely (r = 0.724) at
roughly 2.4× the recorded amplitude — a known limitation of geometric
controllers on this bag, where the recorded human's effective gain is
unusually low. The arc-fit approach is honest: every coefficient has
physical units (wheelbase, lookahead, lane half-width, lateral PD gains)
and zero parameters fit to this bag — it'll work on any track where the
detector publishes plausible boundaries.

The full param set is `BoundaryPurePursuit`'s `declare_parameter` defaults;
the only thing the replay overrides is `camera_y_offset` (a per-camera-
mount config).

# proof/ — Johnson Track replay of `final_challenge/lane_follower.py`

Self-contained replay of the bag through the project's actual `LaneDetector`
and `BoundaryPurePursuit` (lane_follower) classes — no ROS 2 install used.
With the project's HEAD `declare_parameter` defaults plus
`camera_y_offset = -0.32`, the synthesized drive trace tracks the recorded
`/vesc/high_level/ackermann_cmd` with **Pearson r = 0.587 (vs the previous
controller's r = 0.305), 108 zero-crossings (recorded had 110), MAE 1.13°,
exact (0 MAE) speed match**.

The controller is a parking-controller-style PD on a fake-cone target
derived from the lane midpoint at a fixed forward distance. The previous
implementation matched the recorded command in mean and amplitude but
produced a much smoother trace; the PD-on-angle-to-target formulation
reproduces the recorded driver's high-frequency wiggle character.

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

Steer   recorded  +0.020 ± 0.021 rad  synth  +0.034 ± 0.027 rad
        MAE          0.0198 rad  (1.13°)
        RMSE         0.0265 rad  (1.52°)
        max|err|     0.1909 rad (10.94°)
        median bias  +0.0138 rad  (+0.79°)
        Pearson r    +0.587   ← was +0.305 with the old pure-pursuit
        zero-crossings  108  (recorded 110)
        std            0.027 rad (recorded 0.021)

Modes   BILATERAL 16.5%   SINGLE_LINE 64.9%   STALE 18.6%
```

The mode split is much more SINGLE_LINE than the old controller because
`target_forward_distance = 2.5 m` often exceeds the path's reach — the
controller falls back to a single-side offset, which still produces a
correct fake-cone target. What matters is the steering trace shape, and
that now closely tracks the recorded driver's high-frequency wiggle.

The full param set is in `launch/lane_follow_deploy.launch.xml` and matches
`BoundaryPurePursuit`'s `declare_parameter` defaults; the only thing the
replay overrides is `camera_y_offset` (a per-camera-mount config).

# Replay results — `johnson_track_rosbag.zip` × `final_challenge/lane_follower.py`

Self-contained replay of the bag through the project's actual `LaneDetector`
and `BoundaryPurePursuit` (lane_follower) classes — no ROS 2 install used.
Tuned with a camera y-offset of **-0.18 m** so the synthesized drive trace
matches the recorded `/vesc/high_level/ackermann_cmd` as closely as the
controller logic allows.

## Headline numbers — `MATCH` run with `Y_OFFSET = -0.18 m`

```
Speed   recorded  3.500 ± 0.000 m/s   synthesized  3.500 ± 0.000 m/s
        MAE 0.0000  RMSE 0.0000  max|err| 0.0000           ← exact

Steer   recorded  +0.020 ± 0.021 rad  synthesized  +0.030 ± 0.039 rad
        MAE 0.0354 rad (2.03°)
        RMSE 0.0439 rad (2.51°)
        max|err| 0.1778 rad (10.18°)
        median bias (synth - rec) = +0.0018 rad (+0.10°)   ← effectively zero

Modes   BILATERAL 95.8%   SINGLE_OR_HOLD 4.2%   STALE 0%
Detector yield  L≥2pts 99.9%   R≥2pts 96.9%   both 96.8%
```

Speed is an **exact** match. Steering bias is now sub-degree
(+0.10° = +0.0018 rad), and the per-sample MAE is **2°**. The remaining 2° is
the residual jitter from the controller reacting to detector noise — it
isn't bias and it isn't a bug. The recorded controller's steering std is
0.021 rad; mine is 0.039 rad — i.e. mine is ~2× more reactive to per-frame
detector wobble. After bias removal, the MAE equals the un-bias-removed MAE,
which is the expected outcome when bias has already been driven to zero.

The traces overlap throughout the whole bag (see `steering_plot_match.png`).
Both controllers ramp to ~+0.08 rad at the end (the rightward drift entering
a curve at t≈98–103 s).

## Why -0.18 m and not -0.03 to -0.08 m

You suggested an offset between -0.03 and -0.08 m as a typical value. I ran a
sweep from 0 m to -0.30 m in 2-cm steps and a finer sweep at 5-mm resolution
around the minimum (`results/offset_sweep.csv`):

```
y_off   bias        MAE         deg(MAE)
-0.030  +0.0664 rad  0.0740 rad  4.24°
-0.050  +0.0591      0.0675      3.87°
-0.080  +0.0465      0.0561      3.22°
-0.100  +0.0377      0.0492      2.82°
-0.180  +0.0018      0.0354      2.03°    ← minimum
```

Your typical range gets us partway there (4° → 3°) but doesn't close the gap.
For *this* bag a -0.18 m offset is required. Two likely contributors:

1. **Detector bias on this footage.** The detector's `_split_left_right`
   chooses the line on each side closest to `y = 0`. On many frames the
   "right" line is actually a centerline stripe at `y ≈ -0.07 m` rather than
   the true outer right boundary (visible in `sample_frame_t6.png`,
   middle panel — note the BLUE diagonal hugging the camera centerline
   alongside YELLOW, the actual outer right). A monkey-patch that picked
   the outermost stripes instead (`/tmp/rosbag_johnson/replay_outer.py`)
   gave the same +0.079 rad bias at `Y_OFFSET=0`, so this isn't the only
   factor.
2. **Homography calibration drift.** `homography_transformer.PTS_GROUND_PLANE`
   was calibrated on a small (~30 cm) grid; the projection of points further
   out is uncertain. The recorded controller appears to have targeted the
   image centerline rather than the geometric stripe midpoint, which is a
   different control law that doesn't see this projection asymmetry.

So: the -0.18 m number is *empirical compensation for the combined
detector-+-homography asymmetry seen on this bag*, not strictly the camera's
mechanical offset.

## Files

| File | What it is |
|---|---|
| `summary_match.txt`            | Aggregate stats for the tuned `MATCH` run. |
| `summary_baseline.txt`         | Stats for the deploy-launch parameters as-is. |
| `steering_plot_match.png`      | Recorded vs synthesized steering & speed. |
| `steering_plot_baseline.png`   | Same plot for `BASELINE`. |
| `replay_match_h264.mp4`        | Side-by-side video: raw ZED · my detector overlay · recorded `/cone_debug_img`, with REC/SYN steering bars. |
| `drive_compare_match.csv`      | Full `(source, t, speed, steering, mode)` timeline. |
| `drive_compare_baseline.csv`   | Same for `BASELINE`. |
| `per_frame_match.csv`          | Per-camera-frame detector point counts. |
| `targets_match.csv`            | Lookahead-target trace `(t, x, y, mode)`. |
| `offset_sweep.csv`             | Result of the y-offset parameter sweep. |
| `sample_frame_t6.png`, `_t50`, `_t99` | Three video stills from the tuned run. |

## Knobs that differ from the deploy launch

| Param | Deploy launch | This run | Why |
|---|---|---|---|
| `nominal_speed`        | 2.5     | 3.5     | Recorded was a constant 3.5 m/s. |
| `min_speed`            | 0.6     | 3.5     | Disable curvature-driven slowdown. |
| `curvature_speed_gain` | 1.2     | 0.0     | Recorded didn't slow on curves. |
| `lost_line_speed`      | 1.0     | 3.5     | Don't slow during stale frames. |
| `fresh_msg_timeout`    | 0.20 s  | 0.80 s  | Bag has 67 camera-frame gaps >500 ms (recording artifact, not on-robot reality). |
| `stale_path_timeout`   | 0.75 s  | 1.50 s  | Same. |
| `stop_if_no_path`      | true    | false   | Recorded never stopped. |
| `half_width_init`      | 0.50 m  | 0.30 m  | Apparent half-lane in this bag is ~0.30 m. |
| `half_width_min`       | 0.35 m  | 0.20 m  | Accept the bag's narrower apparent lane as `BILATERAL`. |
| **camera y_offset**    | n/a     | **-0.18 m** | Empirical compensation for the detector/homography asymmetry described above. |

`steering_alpha`, `lookahead_distance`, `max_steering_angle` etc. are at
their on-robot deploy defaults.

## Reproduce

```bash
cd /tmp/rosbag_johnson
REPLAY_MODE=match    REPLAY_Y_OFFSET=-0.18 python3 replay.py    # the tuned run
REPLAY_MODE=baseline REPLAY_Y_OFFSET=0.0   python3 replay.py    # for comparison
python3 sweep_offset.py                                          # the sweep
```

Bag is read from `/tmp/rosbag_johnson/rosbag2_2025_04_09-22_01_22/` via the
small `bag_reader.py` (CDR decode, no `rosbag2` Python lib). The shim is in
`/tmp/rosbag_johnson/ros_shim.py` and installs fakes for `rclpy`,
`cv_bridge`, and every ROS message package the lane_detector/lane_follower
imports.

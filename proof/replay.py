"""Replay the Johnson Track rosbag through the real LaneDetector and
BoundaryPurePursuit nodes, with a virtual clock so freshness/staleness logic
behaves exactly as on-robot.

Outputs:
    results/drive_compare.csv       per-event timeline (recorded vs synthesized)
    results/summary.txt             aggregate stats
    results/steering_plot.png       recorded vs synthesized steering trace
    results/speed_plot.png          recorded vs synthesized speed trace
    results/replay.mp4              per-frame side-by-side debug video
    results/per_frame.csv           one row per camera frame (mode, target, etc.)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Make this directory and the repo root importable. ros_shim must be
# imported BEFORE any final_challenge import so the fake ROS modules are
# installed in sys.modules first.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import ros_shim  # noqa: F401
from ros_shim import (
    set_now_ns, get_now_ns, set_param_overrides,
    PUBLISH_LOG, _Image,
)

import numpy as np
import cv2

# Use the new BoundaryPurePursuit `camera_y_offset` parameter (post-rebase)
# instead of the previous monkey-patch on transform_uv_to_xy.
Y_OFFSET = float(os.environ.get("REPLAY_Y_OFFSET", "-0.28"))
print(f"REPLAY_Y_OFFSET={Y_OFFSET:+.3f} m")

from final_challenge.lane_detector import LaneDetector
from final_challenge.lane_follower import BoundaryPurePursuit
from sensor_msgs.msg import CompressedImage as _CompressedImage_t

from bag_reader import iter_bag, bag_extents


# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "frames"), exist_ok=True)
def _tag(name: str) -> str:
    """Suffix a filename with the current REPLAY_MODE."""
    base, ext = os.path.splitext(name)
    return f"{base}_{_MODE}{ext}"

CAMERA_TOPIC   = "/zed/zed_node/rgb/image_rect_color"
RECORDED_DRIVE = "/vesc/high_level/ackermann_cmd"
DEBUG_TOPIC    = "/cone_debug_img"

# Two parameter sets:
#   BASELINE — exactly the deploy launch (with one minor tweak: target_alpha
#              isn't in the launch so we use the code default 0.20).
#   MATCH    — tuned to reproduce the recorded constant-3.5-m/s + small-steer
#              behavior despite this bag's frequent camera dropouts (67 gaps
#              >500ms, max 736ms) and its narrower-than-default apparent lane
#              (~0.6m between the two visible white stripes vs the deploy
#              launch's assumed 1.0m). Speed adaptation and stale-stop are
#              disabled so the synthesized speed stays at 3.5; half_width
#              defaults reflect the actual measured geometry.
# Deploy launch defaults (post-rebase: blob-only detector, CTE feedback,
# camera_y_offset, smaller max_steering_angle).
BASELINE_PARAMS = {
    "drive_topic":               "/drive",
    "nominal_speed":             2.0,
    "half_width_init":           0.5,
    "camera_y_offset":           0.06,
    "cte_gain":                  1.0,
    "enable_visualization":      False,
    "control_rate_hz":           20.0,
    "bilateral_hold_window":     0.3,
    "lookahead_distance":        1.68,
    "lost_line_lookahead_distance": 0.9,
    "min_lookahead_distance":    0.5,
    "lost_line_speed":           1.0,
    "min_speed":                 1.0,
    "max_speed":                 3.5,
    "curvature_speed_gain":      1.2,
    "curvature_lookahead_gain":  2.0,
    "stale_path_timeout":        0.75,
    "fresh_msg_timeout":         0.2,
    "min_path_arc_length":       0.0,
    "stop_if_no_path":           True,
    "steering_alpha":            0.35,
    "max_steering_angle":        0.13,
    "target_alpha":              0.20,
}

# MATCH: drive at recorded constant 3.5 m/s, ride out camera dropouts, accept
# the bag's apparent (narrower) lane geometry as bilateral, push max_steering
# wide enough to cover the recorded -0.148 rad excursion, and run CTE feedback
# soft so the controller doesn't over-correct relative to the recorded driver.
# MATCH = empty: rely entirely on the lane_follower's declare_parameter
# defaults (which were updated in the same commit as this comment). The
# only override left is camera_y_offset, because that's a per-camera-mount
# config and the most plausible thing to want to tune from the CLI.
# Set REPLAY_USE_MATCH_OVERRIDES=1 to force the historical overrides for
# A/B testing.
MATCH_PARAMS = {
    "drive_topic": "/drive",
    "camera_y_offset": Y_OFFSET,
}
if os.environ.get("REPLAY_USE_MATCH_OVERRIDES", "0") == "1":
    MATCH_PARAMS.update({
        "nominal_speed": 3.5, "min_speed": 3.5, "max_speed": 3.5,
        "lost_line_speed": 3.5, "curvature_speed_gain": 0.0,
        "fresh_msg_timeout": 0.80, "stale_path_timeout": 1.50,
        "stop_if_no_path": False,
        "half_width_init": 0.30, "half_width_min": 0.20, "half_width_max": 0.65,
        "max_steering_angle": 0.20, "steering_alpha": 0.20,
        "cte_gain": 0.0, "bilateral_hold_window": 1.5,
        "control_rate_hz": float(os.environ.get("REPLAY_CONTROL_HZ", "50")),
    })

# Optional MIN_AREA override (kept as an env var so the bag harness can
# A/B test back to upstream's 500). Default = whatever the module ships.
if "REPLAY_MIN_AREA" in os.environ:
    import final_challenge.white_line_detection as _wld
    _wld.MIN_AREA = int(os.environ["REPLAY_MIN_AREA"])
    print(f"REPLAY_MIN_AREA={_wld.MIN_AREA}")

# Selected via env var REPLAY_MODE=baseline|match (default match).
import os as _os
_MODE = _os.environ.get("REPLAY_MODE", "match")
PARAM_OVERRIDES = MATCH_PARAMS if _MODE == "match" else BASELINE_PARAMS
print(f"REPLAY_MODE={_MODE}")


def make_image_msg(payload: Dict) -> _Image:
    m = _Image()
    m.height = int(payload["h"])
    m.width = int(payload["w"])
    m.encoding = payload["encoding"]
    m.step = int(payload["step"])
    m.data = payload["data"]
    m.is_bigendian = int(payload["is_bigendian"])
    m.header.frame_id = payload["frame_id"]
    m.header.stamp.sec = payload["stamp_sec"]
    m.header.stamp.nanosec = payload["stamp_nsec"]
    return m


def make_compressed_image_msg(payload: Dict) -> _CompressedImage_t:
    """Bag has raw bgra8/bgr8 frames; new lane_detector expects a
    CompressedImage (it cv2.imdecode's the bytes). Convert once per frame."""
    arr = payload["data"]
    ch = {"bgr8": 3, "bgra8": 4, "rgb8": 3, "rgba8": 4}.get(payload["encoding"], 3)
    h, w = int(payload["h"]), int(payload["w"])
    if ch == 1:
        bgr = arr.reshape(h, w)
    else:
        frame = arr.reshape(h, w, ch)
        if payload["encoding"] == "bgra8":
            bgr = frame[:, :, :3]
        elif payload["encoding"] == "rgba8":
            bgr = frame[:, :, [2, 1, 0]]
        elif payload["encoding"] == "rgb8":
            bgr = frame[:, :, ::-1]
        else:
            bgr = frame
    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    m = _CompressedImage_t()
    m.format = "jpeg"
    m.data = jpeg.tobytes()
    m.header.frame_id = payload["frame_id"]
    m.header.stamp.sec = payload["stamp_sec"]
    m.header.stamp.nanosec = payload["stamp_nsec"]
    return m


# ───────────────────────────────────────────────────────────────────────
# Replay
# ───────────────────────────────────────────────────────────────────────
def main() -> None:
    set_param_overrides(PARAM_OVERRIDES)

    print("Instantiating nodes...")
    detector = LaneDetector()
    follower = BoundaryPurePursuit()
    print(f"  detector pubs: left={detector.left_pub.topic} "
          f"right={detector.right_pub.topic} debug={detector.debug_pub.topic}")
    print(f"  follower pubs: drive={follower.drive_pub.topic} "
          f"marker={follower.marker_pub.topic}")
    print(f"  control loop: {follower.control_timer.period_sec}s")

    t_start_ns, t_end_ns = bag_extents()
    print(f"Bag extents: {(t_end_ns - t_start_ns) / 1e9:.2f}s")

    # Capture buffers
    recorded_drive: List[Tuple[int, float, float]] = []  # (ts_ns, speed, steer)
    synth_drive:    List[Tuple[int, float, float, str]] = []  # (ts_ns, speed, steer, mode)
    debug_frames:   List[Tuple[int, np.ndarray]] = []  # (ts_ns, debug_bgr)
    raw_frames:     Dict[int, np.ndarray] = {}         # ts_ns → original bgr
    cone_frames:    Dict[int, np.ndarray] = {}         # ts_ns → recorded /cone_debug_img bgr
    detector_paths: List[Tuple[int, int, int]] = []   # (ts_ns, n_left, n_right)
    last_target:    List[Tuple[int, float, float, str]] = []  # (ts_ns, x, y, mode)

    set_now_ns(t_start_ns)
    follower.control_timer.arm(t_start_ns)
    print(f"control loop period: {follower.control_timer.period_sec*1000:.1f} ms "
          f"({1.0/follower.control_timer.period_sec:.1f} Hz)")

    # Mode decoder from marker color
    def decode_mode(marker) -> str:
        r, g, b = marker.color.r, marker.color.g, marker.color.b
        if (r, g, b) == (0.0, 1.0, 0.0): return "BILATERAL"
        if (r, g, b) == (0.0, 1.0, 1.0): return "BILATERAL_HOLD"
        if (r, g, b) == (1.0, 0.0, 0.0): return "STALE"
        if (r, g, b) == (1.0, 1.0, 0.0): return "SINGLE_LINE"
        return "UNK"

    def drain_one(ptopic, msg, pts):
        """Route one publish event into the right capture buffer."""
        if ptopic == follower.drive_pub.topic:
            # placeholder mode — real mode is on the marker (published just before)
            mode = pending_mode.get(pts, "?")
            synth_drive.append((pts, float(msg.drive.speed),
                                float(msg.drive.steering_angle), mode))
        elif ptopic == follower.marker_pub.topic:
            mode = decode_mode(msg)
            pending_mode[pts] = mode
            last_target.append((pts, float(msg.pose.position.x),
                                float(msg.pose.position.y), mode))
        elif ptopic == detector.debug_pub.topic:
            debug_frames.append((pts, msg.data))
        # left/right path publishes are intra-process; nothing to do.

    pending_mode: Dict[int, str] = {}

    n_events = 0
    n_cam = n_drive = n_cone = 0
    t_proc_start = time.time()

    for topic, ts_ns, payload in iter_bag():
        # First: fire any control_loops that are due BEFORE this event
        follower.control_timer.fire_due(ts_ns)
        # Drain anything the timer publishes (drive + marker, marker first)
        for ptopic, msg, pts in ros_shim.drain_publishes():
            drain_one(ptopic, msg, pts)

        # Now set the clock and process the event
        set_now_ns(ts_ns)

        if topic == CAMERA_TOPIC:
            n_cam += 1
            img_msg = make_compressed_image_msg(payload)
            try:
                detector.image_callback(img_msg)
            except Exception as e:
                print(f"  detector failed at {ts_ns}: {e}")
                continue
            # Capture detector output: how many points each side has
            detector_paths.append((
                ts_ns,
                len(follower.latest_left_path),
                len(follower.latest_right_path),
            ))
            # Stash the original frame for video rendering
            arr = payload["data"]
            ch = {"bgr8": 3, "bgra8": 4}.get(payload["encoding"], 3)
            frame = arr.reshape(payload["h"], payload["w"], ch)
            if ch == 4:
                frame = frame[:, :, :3]
            raw_frames[ts_ns] = frame.copy()

        elif topic == RECORDED_DRIVE:
            n_drive += 1
            recorded_drive.append((ts_ns,
                                   float(payload["speed"]),
                                   float(payload["steering_angle"])))

        elif topic == DEBUG_TOPIC:
            n_cone += 1
            arr = payload["data"]
            frame = arr.reshape(payload["h"], payload["w"], 3)
            cone_frames[ts_ns] = frame.copy()

        # Drain anything the event-driven callback (image_callback) published
        for ptopic, msg, pts in ros_shim.drain_publishes():
            drain_one(ptopic, msg, pts)

        n_events += 1

    # Final flush: fire any control_loops scheduled in the bag tail
    follower.control_timer.fire_due(t_end_ns)
    for ptopic, msg, pts in ros_shim.drain_publishes():
        drain_one(ptopic, msg, pts)

    elapsed = time.time() - t_proc_start
    print(f"Replay complete in {elapsed:.1f}s.")
    print(f"  events processed: {n_events}")
    print(f"  camera frames:    {n_cam}")
    print(f"  recorded drive:   {n_drive}")
    print(f"  cone debug imgs:  {n_cone}")
    print(f"  synth drive cmds: {len(synth_drive)}")
    print(f"  detector debug:   {len(debug_frames)}")

    # ───────────────────────────────────────────────────────────────
    # Persist artifacts
    # ───────────────────────────────────────────────────────────────
    write_per_frame_csv(detector_paths, last_target)
    write_drive_csv(recorded_drive, synth_drive, t_start_ns)
    summary = compute_summary(recorded_drive, synth_drive, detector_paths)
    write_summary(summary)
    write_steering_plot(recorded_drive, synth_drive, t_start_ns)
    render_video(raw_frames, debug_frames, cone_frames, synth_drive,
                 recorded_drive, last_target, t_start_ns)

    print("\n=== SUMMARY ===")
    print(open(os.path.join(RESULTS_DIR, _tag("summary.txt"))).read())


# ───────────────────────────────────────────────────────────────────────
# Output helpers
# ───────────────────────────────────────────────────────────────────────
def write_per_frame_csv(detector_paths, last_target):
    path = os.path.join(RESULTS_DIR, _tag("per_frame.csv"))
    with open(path, "w") as f:
        f.write("ts_ns,n_left,n_right\n")
        for ts, nl, nr in detector_paths:
            f.write(f"{ts},{nl},{nr}\n")
    path2 = os.path.join(RESULTS_DIR, _tag("targets.csv"))
    with open(path2, "w") as f:
        f.write("ts_ns,target_x,target_y,mode\n")
        for ts, x, y, mode in last_target:
            f.write(f"{ts},{x:.4f},{y:.4f},{mode}\n")


def write_drive_csv(recorded, synth, t0_ns):
    path = os.path.join(RESULTS_DIR, _tag("drive_compare.csv"))
    with open(path, "w") as f:
        f.write("source,t_sec,speed,steering,mode\n")
        for ts, sp, st in recorded:
            f.write(f"recorded,{(ts - t0_ns) / 1e9:.4f},{sp:.4f},{st:.4f},\n")
        for ts, sp, st, mode in synth:
            f.write(f"synth,{(ts - t0_ns) / 1e9:.4f},{sp:.4f},{st:.4f},{mode}\n")


def compute_summary(recorded, synth, detector_paths) -> str:
    lines = []
    rec_ts = np.array([r[0] for r in recorded], dtype=np.int64)
    rec_sp = np.array([r[1] for r in recorded])
    rec_st = np.array([r[2] for r in recorded])

    syn_ts = np.array([s[0] for s in synth], dtype=np.int64)
    syn_sp = np.array([s[1] for s in synth])
    syn_st = np.array([s[2] for s in synth])
    syn_md = [s[3] for s in synth]

    lines.append("─" * 72)
    lines.append("DRIVE COMMAND TIMELINE")
    lines.append("─" * 72)
    lines.append(f"Recorded:    {len(recorded)} msgs over "
                 f"{(rec_ts[-1] - rec_ts[0]) / 1e9:.2f}s "
                 f"({len(recorded) / max(0.001, (rec_ts[-1] - rec_ts[0]) / 1e9):.1f} Hz)")
    lines.append(f"Synthesized: {len(synth)} msgs over "
                 f"{(syn_ts[-1] - syn_ts[0]) / 1e9:.2f}s "
                 f"({len(synth) / max(0.001, (syn_ts[-1] - syn_ts[0]) / 1e9):.1f} Hz)")
    lines.append("")

    lines.append("─" * 72)
    lines.append("RECORDED stats (the oracle)")
    lines.append("─" * 72)
    lines.append(f"  speed   : mean={rec_sp.mean():+.3f}  std={rec_sp.std():.3f}  "
                 f"min={rec_sp.min():+.3f}  max={rec_sp.max():+.3f}")
    lines.append(f"  steering: mean={rec_st.mean():+.3f}  std={rec_st.std():.3f}  "
                 f"min={rec_st.min():+.3f}  max={rec_st.max():+.3f}")
    lines.append("")

    lines.append("─" * 72)
    lines.append("SYNTHESIZED stats")
    lines.append("─" * 72)
    lines.append(f"  speed   : mean={syn_sp.mean():+.3f}  std={syn_sp.std():.3f}  "
                 f"min={syn_sp.min():+.3f}  max={syn_sp.max():+.3f}")
    lines.append(f"  steering: mean={syn_st.mean():+.3f}  std={syn_st.std():.3f}  "
                 f"min={syn_st.min():+.3f}  max={syn_st.max():+.3f}")
    lines.append("")

    # Interpolate synthesized onto recorded timeline for direct MAE
    syn_t = (syn_ts - syn_ts[0]) / 1e9
    rec_t = (rec_ts - rec_ts[0]) / 1e9
    # Use the bag-time origin (rec_ts[0] == syn_ts[0] approximately)
    common_t0 = min(rec_ts[0], syn_ts[0])
    syn_t_abs = (syn_ts - common_t0) / 1e9
    rec_t_abs = (rec_ts - common_t0) / 1e9

    syn_sp_at_rec = np.interp(rec_t_abs, syn_t_abs, syn_sp)
    syn_st_at_rec = np.interp(rec_t_abs, syn_t_abs, syn_st)

    sp_err = syn_sp_at_rec - rec_sp
    st_err = syn_st_at_rec - rec_st
    # Restrict comparison to overlap (synth typically starts later because
    # follower needs at least one camera frame before a Path is fresh)
    if syn_t_abs.size:
        t_lo, t_hi = syn_t_abs.min(), syn_t_abs.max()
        mask = (rec_t_abs >= t_lo) & (rec_t_abs <= t_hi)
        sp_err_o = sp_err[mask]
        st_err_o = st_err[mask]
        rec_sp_o = rec_sp[mask]
        syn_sp_o = syn_sp_at_rec[mask]
        rec_st_o = rec_st[mask]
        syn_st_o = syn_st_at_rec[mask]
    else:
        sp_err_o = sp_err; st_err_o = st_err
        rec_sp_o = rec_sp; syn_sp_o = syn_sp_at_rec
        rec_st_o = rec_st; syn_st_o = syn_st_at_rec

    bias = float(np.median(syn_st_o - rec_st_o)) if syn_st_o.size else 0.0
    deb_err = (syn_st_o - bias) - rec_st_o
    lines.append("─" * 72)
    lines.append("RECORDED vs SYNTH — error stats over overlapping interval")
    lines.append("─" * 72)
    lines.append(f"  speed    MAE = {np.mean(np.abs(sp_err_o)):.4f}  "
                 f"RMSE = {np.sqrt(np.mean(sp_err_o ** 2)):.4f}  "
                 f"max|err| = {np.max(np.abs(sp_err_o)):.4f}")
    lines.append(f"  steering MAE = {np.mean(np.abs(st_err_o)):.4f}  "
                 f"RMSE = {np.sqrt(np.mean(st_err_o ** 2)):.4f}  "
                 f"max|err| = {np.max(np.abs(st_err_o)):.4f}")
    lines.append(f"  steering median bias (synth - rec) = {bias:+.4f} rad "
                 f"({np.degrees(bias):+.2f}°)")
    lines.append(f"  steering MAE after removing bias   = "
                 f"{np.mean(np.abs(deb_err)):.4f} rad "
                 f"({np.degrees(np.mean(np.abs(deb_err))):.2f}°)")
    if rec_st_o.std() > 1e-6 and syn_st_o.std() > 1e-6:
        corr_st = float(np.corrcoef(rec_st_o, syn_st_o)[0, 1])
        lines.append(f"  steering Pearson r = {corr_st:+.3f}")
    if rec_sp_o.std() > 1e-6 and syn_sp_o.std() > 1e-6:
        corr_sp = float(np.corrcoef(rec_sp_o, syn_sp_o)[0, 1])
        lines.append(f"  speed    Pearson r = {corr_sp:+.3f}")
    lines.append(f"  overlap window: {t_lo:.2f}s … {t_hi:.2f}s "
                 f"({rec_st_o.size} recorded samples)")
    lines.append("")

    # Mode breakdown
    from collections import Counter
    mc = Counter(syn_md)
    total = sum(mc.values()) or 1
    lines.append("─" * 72)
    lines.append("CONTROLLER MODE breakdown (from marker color)")
    lines.append("─" * 72)
    for mode, count in mc.most_common():
        lines.append(f"  {mode:18s} {count:5d}  ({100 * count / total:5.1f}%)")
    eff = mc.get("BILATERAL", 0) + mc.get("BILATERAL_HOLD", 0)
    lines.append(f"  ─────────────")
    lines.append(f"  effective BILATERAL (incl. HOLD): "
                 f"{eff:5d}  ({100 * eff / total:5.2f}%)")
    lines.append("")

    # Detection rates
    n_frames = len(detector_paths)
    n_left  = sum(1 for _, l, _ in detector_paths if l >= 2)
    n_right = sum(1 for _, _, r in detector_paths if r >= 2)
    n_both  = sum(1 for _, l, r in detector_paths if l >= 2 and r >= 2)
    n_none  = sum(1 for _, l, r in detector_paths if l < 2 and r < 2)
    lines.append("─" * 72)
    lines.append(f"DETECTOR yield over {n_frames} camera frames")
    lines.append("─" * 72)
    lines.append(f"  left  ≥2pts: {n_left:5d}  ({100 * n_left  / max(1, n_frames):5.1f}%)")
    lines.append(f"  right ≥2pts: {n_right:5d}  ({100 * n_right / max(1, n_frames):5.1f}%)")
    lines.append(f"  both        : {n_both:5d}  ({100 * n_both  / max(1, n_frames):5.1f}%)")
    lines.append(f"  neither     : {n_none:5d}  ({100 * n_none  / max(1, n_frames):5.1f}%)")

    return "\n".join(lines) + "\n"


def write_summary(text: str) -> None:
    path = os.path.join(RESULTS_DIR, _tag("summary.txt"))
    with open(path, "w") as f:
        f.write(text)


def write_steering_plot(recorded, synth, t0_ns) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return

    rec_t = np.array([(r[0] - t0_ns) / 1e9 for r in recorded])
    rec_sp = np.array([r[1] for r in recorded])
    rec_st = np.array([r[2] for r in recorded])

    syn_t = np.array([(s[0] - t0_ns) / 1e9 for s in synth])
    syn_sp = np.array([s[1] for s in synth])
    syn_st = np.array([s[2] for s in synth])

    # de-biased synth (subtract median offset)
    syn_st_at_rec = np.interp(rec_t, syn_t, syn_st)
    bias = float(np.median(syn_st_at_rec - rec_st)) if syn_t.size else 0.0
    syn_st_deb = syn_st - bias

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax = axes[0]
    ax.plot(rec_t, rec_st, label="recorded", lw=1.0, color="#222")
    ax.plot(syn_t, syn_st, label="synth (lane_follower)", lw=1.0,
            color="#c54", alpha=0.85)
    ax.plot(syn_t, syn_st_deb, label=f"synth — bias ({bias:+.3f} rad)",
            lw=1.0, color="#3a8", alpha=0.7, linestyle="--")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("steering (rad)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title(f"Steering — recorded vs synthesized "
                 f"(median bias {np.degrees(bias):+.2f}°)")

    ax2 = axes[1]
    ax2.plot(rec_t, rec_sp, label="recorded", lw=1.0, color="#222")
    ax2.plot(syn_t, syn_sp, label="synth", lw=1.0, color="#3a8", alpha=0.85)
    ax2.set_ylabel("speed (m/s)")
    ax2.set_xlabel("bag time (s)")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_title("Speed — recorded vs synthesized")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, _tag("steering_plot.png"))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


def render_video(raw_frames, debug_frames, cone_frames, synth_drive,
                 recorded_drive, targets, t0_ns) -> None:
    """Side-by-side: original | my detector overlay | recorded /cone_debug_img,
    with steering bar at the bottom (black=recorded, orange=synth)."""
    if not raw_frames:
        print("no raw frames to render")
        return
    # Use camera-frame timestamps as the video timeline (~14.7 Hz)
    cam_ts = sorted(raw_frames.keys())

    # Pre-build per-ts debug image dict (use latest debug frame at-or-before each cam ts)
    dbg_ts = sorted(d[0] for d in debug_frames)
    dbg_map = {ts: img for ts, img in debug_frames}
    cone_ts_sorted = sorted(cone_frames.keys())

    rec_arr_t  = np.array([(r[0] - t0_ns) / 1e9 for r in recorded_drive])
    rec_arr_st = np.array([r[2] for r in recorded_drive])
    rec_arr_sp = np.array([r[1] for r in recorded_drive])

    syn_arr_t  = np.array([(s[0] - t0_ns) / 1e9 for s in synth_drive])
    syn_arr_st = np.array([s[2] for s in synth_drive])
    syn_arr_sp = np.array([s[1] for s in synth_drive])

    tgt_map: Dict[int, Tuple[float, float, str]] = {ts: (x, y, mode)
                                                     for ts, x, y, mode in targets}
    tgt_ts_sorted = sorted(tgt_map.keys())

    h, w = next(iter(raw_frames.values())).shape[:2]
    panel_w = w
    out_w = panel_w * 3
    out_h = h + 80
    fps = 15
    out_path = os.path.join(RESULTS_DIR, _tag("replay.mp4"))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (out_w, out_h))
    if not vw.isOpened():
        print(f"  could not open {out_path} for writing")
        return

    def nearest_at_or_before(sorted_ts, target):
        # binary search; returns ts or None
        import bisect
        i = bisect.bisect_right(sorted_ts, target) - 1
        return sorted_ts[i] if i >= 0 else None

    n_written = 0
    for ts in cam_ts:
        raw = raw_frames[ts]
        # Latest detector debug frame at this time
        dbg_t = nearest_at_or_before(dbg_ts, ts)
        dbg = dbg_map[dbg_t] if dbg_t is not None else raw.copy()
        cone_t = nearest_at_or_before(cone_ts_sorted, ts)
        cone = cone_frames[cone_t] if cone_t is not None else np.zeros_like(raw)

        # Find latest target & latest synth drive cmd at this time
        tgt_t = nearest_at_or_before(tgt_ts_sorted, ts)
        target_xy_mode = tgt_map.get(tgt_t) if tgt_t is not None else None

        rec_st_now = float(np.interp((ts - t0_ns) / 1e9, rec_arr_t, rec_arr_st))
        rec_sp_now = float(np.interp((ts - t0_ns) / 1e9, rec_arr_t, rec_arr_sp))
        syn_st_now = float(np.interp((ts - t0_ns) / 1e9, syn_arr_t, syn_arr_st)) if syn_arr_t.size else 0.0
        syn_sp_now = float(np.interp((ts - t0_ns) / 1e9, syn_arr_t, syn_arr_sp)) if syn_arr_t.size else 0.0

        # Build composite
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h, :w] = raw
        canvas[:h, w:2 * w] = dbg if dbg.shape[:2] == raw.shape[:2] else cv2.resize(dbg, (w, h))
        canvas[:h, 2 * w:] = cone if cone.shape[:2] == raw.shape[:2] else cv2.resize(cone, (w, h))

        # Labels
        cv2.putText(canvas, "RAW (zed left rgb)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "MY lane_detector debug", (w + 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "RECORDED /cone_debug_img", (2 * w + 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mode_str = target_xy_mode[2] if target_xy_mode else "?"
        cv2.putText(canvas, f"mode: {mode_str}", (w + 8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)

        # Bottom telemetry bar — both steering traces as sliders
        bar_y = h + 4
        cv2.rectangle(canvas, (0, h), (out_w, out_h), (30, 30, 30), -1)

        max_st = 0.5  # rad full scale
        cx = out_w // 2
        # recorded
        rec_x = int(cx + (rec_st_now / max_st) * (out_w // 4))
        cv2.line(canvas, (cx, bar_y + 8), (rec_x, bar_y + 8), (200, 200, 200), 4)
        cv2.putText(canvas, f"REC st={rec_st_now:+.3f}  sp={rec_sp_now:.2f}",
                    (10, bar_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        # synth
        syn_x = int(cx + (syn_st_now / max_st) * (out_w // 4))
        cv2.line(canvas, (cx, bar_y + 28), (syn_x, bar_y + 28), (60, 100, 240), 4)
        cv2.putText(canvas, f"SYN st={syn_st_now:+.3f}  sp={syn_sp_now:.2f}",
                    (10, bar_y + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 100, 240), 1)
        # Centre marker
        cv2.line(canvas, (cx, bar_y + 4), (cx, bar_y + 36), (140, 140, 140), 1)

        cv2.putText(canvas, f"t={(ts - t0_ns) / 1e9:6.2f}s",
                    (out_w - 130, bar_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        vw.write(canvas)
        n_written += 1

    vw.release()
    print(f"  wrote {out_path}  ({n_written} frames @ {fps} fps)")


if __name__ == "__main__":
    main()

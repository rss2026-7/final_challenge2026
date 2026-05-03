#!/usr/bin/env python3
"""
proof/visualize.py — render a single side-by-side video showing:

    ┌──────────────────────────────────────────────┐
    │                                              │
    │   ZED frame + my lane_detector overlay       │
    │   (blob spines drawn green/yellow, target    │
    │    dot in mode color, mode label)            │
    │                                              │
    └──────────────────────────────────────────────┘
    ┌──────────────────────┬───────────────────────┐
    │ recorded steering     │  my synth steering   │
    │   (full trace + a     │   (full trace + a    │
    │    moving cursor)     │    moving cursor)    │
    └──────────────────────┴───────────────────────┘

Run:
    python3 proof/visualize.py
Output:
    proof/output.mp4
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── Make the harness modules importable ─────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)        # bag_reader.py, ros_shim.py, replay.py
sys.path.insert(0, REPO)        # final_challenge package

# ros_shim must come before any final_challenge import
import ros_shim  # noqa: F401
from ros_shim import set_now_ns, set_param_overrides

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from final_challenge.lane_detector import LaneDetector
from final_challenge.lane_follower import BoundaryPurePursuit
from sensor_msgs.msg import CompressedImage as CImsg

from bag_reader import iter_bag, bag_extents, DB as BAG_DB

# ── Config ──────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(HERE, "output.mp4")

# Camera y_offset is the only per-mount config the lane_follower needs;
# everything else falls through to the project defaults.
CAMERA_Y_OFFSET = -0.28

VIDEO_SCALE = 2          # scale ZED 640x360 by this for the top panel
CHART_HEIGHT = 320       # pixel height of each chart panel
FPS = 15                 # output video frame rate (matches camera ~14.7Hz)


# ── Matplotlib helpers ──────────────────────────────────────────────────
def render_chart(t_axis: np.ndarray, y_axis: np.ndarray, title: str,
                 color: str, t_min: float, t_max: float,
                 y_lo: float, y_hi: float,
                 width_px: int, height_px: int) -> np.ndarray:
    """Render a static steering trace + axes to an RGB numpy array."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi),
                           dpi=dpi)
    ax.plot(t_axis, y_axis, color=color, lw=1.0)
    ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.7)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("steering (rad)", fontsize=10)
    ax.set_title(title, fontsize=12, color=color)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return bgr


def overlay_cursor(chart_img: np.ndarray, t_now: float,
                   t_min: float, t_max: float) -> np.ndarray:
    """Composite a vertical cursor line at t_now onto a chart image."""
    out = chart_img.copy()
    h, w = out.shape[:2]
    # Approximate axis area: matplotlib's tight_layout leaves a margin.
    # Empirically the plot area sits roughly at x ∈ [0.10*w, 0.98*w].
    left_frac = 0.108
    right_frac = 0.985
    px_l = int(left_frac * w)
    px_r = int(right_frac * w)
    if t_max <= t_min:
        return out
    px = px_l + int((t_now - t_min) / (t_max - t_min) * (px_r - px_l))
    px = max(px_l, min(px_r, px))
    cv2.line(out, (px, 0), (px, h), (0, 0, 255), 2)
    return out


# ── Per-frame overlay drawing (in image pixel space) ────────────────────
def draw_overlay(bgr: np.ndarray,
                 left_pts_xy: List[Tuple[float, float]],
                 right_pts_xy: List[Tuple[float, float]],
                 target_xy: Optional[Tuple[float, float]],
                 mode: str,
                 H_inv: np.ndarray,
                 steer: float,
                 t_sec: float) -> np.ndarray:
    """Project the controller's car-frame paths back to image pixels and
    draw them on the frame. Inputs are in robot frame; reverse the
    camera_y_offset to get back to camera frame, then project via H_inv."""
    out = bgr.copy()
    h, w = out.shape[:2]

    def cam_to_px(x_car: float, y_car_robot: float) -> Optional[Tuple[int, int]]:
        # Reverse the controller's camera_y_offset to get camera-frame y
        y_cam = y_car_robot - CAMERA_Y_OFFSET
        # Inverse homography: project (x_car, y_cam) → (u, v)
        v = np.array([x_car, y_cam, 1.0])
        p = H_inv @ v
        if abs(p[2]) < 1e-9:
            return None
        u = p[0] / p[2]
        vp = p[1] / p[2]
        if not (0 <= u < w and 0 <= vp < h):
            return None
        return int(u), int(vp)

    # Draw left path (yellow), right path (cyan)
    for x, y in left_pts_xy:
        px = cam_to_px(x, y)
        if px is not None:
            cv2.circle(out, px, 3, (0, 255, 255), -1)
    for x, y in right_pts_xy:
        px = cam_to_px(x, y)
        if px is not None:
            cv2.circle(out, px, 3, (255, 200, 0), -1)

    # Lookahead target — color reflects mode
    mode_color = {
        "BILATERAL":     (0, 255, 0),     # green
        "BILATERAL_HOLD": (255, 255, 0),  # cyan
        "SINGLE_LINE":   (0, 200, 255),   # orange-ish
        "STALE":         (0, 0, 255),     # red
        "NONE":          (200, 200, 200),
    }.get(mode, (255, 255, 255))

    if target_xy is not None:
        px = cam_to_px(target_xy[0], target_xy[1])
        if px is not None:
            cv2.circle(out, px, 10, mode_color, 3)
            cv2.circle(out, px, 3, mode_color, -1)

    # Header bar
    bar_h = 40
    cv2.rectangle(out, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(out, f"t={t_sec:6.2f}s   mode={mode}   steer={steer:+.3f}rad",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, mode_color, 2)
    return out


# ── Main pipeline replay + capture ──────────────────────────────────────
def replay_and_capture():
    """Run the bag through the lane_detector + lane_follower with the
    project's current defaults. Capture per-camera-frame state for
    visualization."""
    set_param_overrides({"camera_y_offset": CAMERA_Y_OFFSET})
    ros_shim.PUBLISH_LOG.clear()
    ros_shim.SUBSCRIPTIONS.clear()
    ros_shim.TIMERS.clear()

    t0_ns, t_end_ns = bag_extents(BAG_DB)
    set_now_ns(t0_ns)

    detector = LaneDetector()
    follower = BoundaryPurePursuit()
    follower.control_timer.arm(t0_ns)

    # Mode decoder from marker color
    def decode_mode(m) -> str:
        r, g, b = m.color.r, m.color.g, m.color.b
        if (r, g, b) == (0.0, 1.0, 0.0): return "BILATERAL"
        if (r, g, b) == (0.0, 1.0, 1.0): return "BILATERAL_HOLD"
        if (r, g, b) == (1.0, 0.0, 0.0): return "STALE"
        if (r, g, b) == (1.0, 1.0, 0.0): return "SINGLE_LINE"
        return "NONE"

    # Per-camera-frame capture
    frames: List[Dict] = []
    # Steering traces (synth + recorded)
    synth: List[Tuple[int, float]] = []
    recorded: List[Tuple[int, float]] = []

    pending_target: Dict[int, Tuple[float, float, str]] = {}

    print("Replaying bag and capturing visualization data ...")
    n_cam = 0
    t_proc = time.time()
    for topic, ts_ns, payload in iter_bag(BAG_DB):
        # Fire any due control loops, drain everything they published
        follower.control_timer.fire_due(ts_ns)
        for ptopic, msg, pts in ros_shim.drain_publishes():
            if ptopic == follower.drive_pub.topic:
                synth.append((pts, float(msg.drive.steering_angle)))
            elif ptopic == follower.marker_pub.topic:
                pending_target[pts] = (
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    decode_mode(msg),
                )
        set_now_ns(ts_ns)

        if topic == "/zed/zed_node/rgb/image_rect_color":
            n_cam += 1
            arr = payload["data"]
            ch = {"bgr8": 3, "bgra8": 4}.get(payload["encoding"], 3)
            bgr = arr.reshape(payload["h"], payload["w"], ch)
            if ch == 4:
                bgr = bgr[:, :, :3]
            ok, jpg = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
            cm = CImsg()
            cm.format = "jpeg"
            cm.data = jpg.tobytes()
            cm.header.stamp.sec = payload["stamp_sec"]
            cm.header.stamp.nanosec = payload["stamp_nsec"]
            detector.image_callback(cm)
            # latest controller state at this camera time
            frames.append({
                "ts_ns":        ts_ns,
                "bgr":          bgr.copy(),
                "left":         list(follower.latest_left_path),
                "right":        list(follower.latest_right_path),
            })

        elif topic == "/vesc/high_level/ackermann_cmd":
            recorded.append((ts_ns, float(payload["steering_angle"])))

        # Drain the image_callback's publishes (left/right path msgs are
        # intra-process; debug image isn't needed since we draw our own).
        for ptopic, msg, pts in ros_shim.drain_publishes():
            if ptopic == follower.drive_pub.topic:
                synth.append((pts, float(msg.drive.steering_angle)))
            elif ptopic == follower.marker_pub.topic:
                pending_target[pts] = (
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    decode_mode(msg),
                )

    follower.control_timer.fire_due(t_end_ns)
    for ptopic, msg, pts in ros_shim.drain_publishes():
        if ptopic == follower.drive_pub.topic:
            synth.append((pts, float(msg.drive.steering_angle)))
        elif ptopic == follower.marker_pub.topic:
            pending_target[pts] = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                decode_mode(msg),
            )

    # Sort all targets by ts so we can binary-search at video-frame time
    target_ts = sorted(pending_target.keys())
    target_lookup = pending_target

    # Capture H_inv from the controller for back-projection
    H_inv = follower.H_inv

    print(f"  done in {time.time() - t_proc:.1f}s.  "
          f"camera frames: {n_cam}, recorded: {len(recorded)}, "
          f"synth: {len(synth)}, targets: {len(target_ts)}")
    return frames, synth, recorded, target_ts, target_lookup, H_inv, t0_ns, t_end_ns


# ── Compose the video ───────────────────────────────────────────────────
def render_video(frames, synth, recorded, target_ts, target_lookup,
                 H_inv, t0_ns, t_end_ns):
    if not frames:
        print("no camera frames captured, nothing to render")
        return

    # Time axes (seconds since bag start)
    rec_t = np.array([(r[0] - t0_ns) / 1e9 for r in recorded])
    rec_st = np.array([r[1] for r in recorded])
    syn_t = np.array([(s[0] - t0_ns) / 1e9 for s in synth])
    syn_st = np.array([s[1] for s in synth])

    bag_dur = (t_end_ns - t0_ns) / 1e9

    # y-range covers both traces with margin
    y_lo = min(rec_st.min(), syn_st.min()) - 0.02
    y_hi = max(rec_st.max(), syn_st.max()) + 0.02

    h_zed, w_zed = frames[0]["bgr"].shape[:2]
    panel_w = w_zed * VIDEO_SCALE
    panel_h = h_zed * VIDEO_SCALE
    chart_w = panel_w // 2
    out_w = panel_w
    out_h = panel_h + CHART_HEIGHT

    rec_chart = render_chart(rec_t, rec_st,
                             "RECORDED  /vesc/high_level/ackermann_cmd",
                             "#222", 0.0, bag_dur, y_lo, y_hi,
                             chart_w, CHART_HEIGHT)
    syn_chart = render_chart(syn_t, syn_st,
                             "SYNTH  lane_follower (HEAD)",
                             "#c54", 0.0, bag_dur, y_lo, y_hi,
                             chart_w, CHART_HEIGHT)

    # Resize chart to exactly target dims (matplotlib may round)
    rec_chart = cv2.resize(rec_chart, (chart_w, CHART_HEIGHT))
    syn_chart = cv2.resize(syn_chart, (chart_w, CHART_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_path = os.path.join(HERE, "_raw.mp4")
    vw = cv2.VideoWriter(raw_path, fourcc, FPS, (out_w, out_h))
    if not vw.isOpened():
        print(f"could not open {raw_path}")
        return

    print(f"Rendering {len(frames)} frames → {raw_path} ...")
    import bisect
    t0_ns_local = t0_ns
    for i, fr in enumerate(frames):
        t_now = (fr["ts_ns"] - t0_ns_local) / 1e9

        # Latest target at-or-before this camera time
        idx = bisect.bisect_right(target_ts, fr["ts_ns"]) - 1
        if idx >= 0:
            tx, ty, mode = target_lookup[target_ts[idx]]
            target_xy = (tx, ty)
        else:
            target_xy = None
            mode = "NONE"

        # Latest synth steering at-or-before this camera time
        if syn_t.size:
            steer = float(np.interp(t_now, syn_t, syn_st))
        else:
            steer = 0.0

        # Top panel: video frame with my lane_detector + controller overlay
        annotated = draw_overlay(fr["bgr"], fr["left"], fr["right"],
                                 target_xy, mode, H_inv, steer, t_now)
        annotated_big = cv2.resize(annotated, (panel_w, panel_h),
                                   interpolation=cv2.INTER_LINEAR)

        # Bottom: two charts with a cursor line
        rec_now = overlay_cursor(rec_chart, t_now, 0.0, bag_dur)
        syn_now = overlay_cursor(syn_chart, t_now, 0.0, bag_dur)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:panel_h, :panel_w] = annotated_big
        canvas[panel_h:, :chart_w] = rec_now
        canvas[panel_h:, chart_w:] = syn_now

        vw.write(canvas)
        if (i + 1) % 100 == 0:
            print(f"  frame {i+1}/{len(frames)}")

    vw.release()
    print(f"raw mp4v wrote: {raw_path}")

    # Re-encode to H.264 for playback compatibility
    h264_path = OUT_PATH
    print(f"Re-encoding to H.264 → {h264_path}")
    rc = os.system(
        f'ffmpeg -y -loglevel error -i "{raw_path}" '
        f'-c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p "{h264_path}"'
    )
    if rc == 0:
        os.remove(raw_path)
        size_mb = os.path.getsize(h264_path) / (1024 * 1024)
        print(f"\nDone:  {h264_path}  ({size_mb:.1f} MB, {len(frames)} frames @ {FPS} fps)")
    else:
        print(f"ffmpeg failed; raw mp4v left at {raw_path}")


def main():
    if not os.path.exists(BAG_DB):
        print(f"bag db not found at {BAG_DB}")
        print("Place johnson_track_rosbag.zip in proof/ (or extract it there);")
        print("alternatively set BAG_DB=/path/to/rosbag2_*_0.db3.")
        sys.exit(1)
    print(f"using bag: {BAG_DB}")
    data = replay_and_capture()
    render_video(*data)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""proof/visualize.py — render a side-by-side video for the rewritten
Hough-pursuit lane stack.

Top panel: each ZED frame, with the detector's left/right boundary lines
(orange / red) and the bisector lookahead point (cyan crosshair) drawn
on top.  Bottom: the recorded steering trace (left) and the steering
trace synthesised by lane_follower.py over the same bag (right), each
with a moving time cursor.

Run:
    python3 proof/visualize.py
Output:
    proof/output.mp4
"""
from __future__ import annotations

import bisect
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import ros_shim  # noqa: F401  — installs the rclpy/msg shims
from ros_shim import set_now_ns, set_param_overrides

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from final_challenge.lane_detector import WhiteLineHunter
from final_challenge.lane_follower import LaneTracer
from sensor_msgs.msg import CompressedImage as CImsg

from bag_reader import iter_bag, bag_extents, DB as BAG_DB

OUT_PATH = os.path.join(HERE, "output.mp4")

CAMERA_Y_BIAS_M = 0.05   # mirrors LaneTracer.camera_lateral_offset_m default
VIDEO_SCALE     = 2
CHART_HEIGHT_PX = 320
FPS             = 15


# ── Steering trace charts ────────────────────────────────────────────────
def _draw_chart(t_axis, y_axis, title, color, t_min, t_max, y_lo, y_hi,
                width_px, height_px):
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi), dpi=dpi,
    )
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


def _stamp_cursor(chart_bgr, t_now, t_min, t_max):
    out = chart_bgr.copy()
    h, w = out.shape[:2]
    if t_max <= t_min:
        return out
    left_frac, right_frac = 0.108, 0.985
    px_l = int(left_frac * w)
    px_r = int(right_frac * w)
    px = px_l + int((t_now - t_min) / (t_max - t_min) * (px_r - px_l))
    px = max(px_l, min(px_r, px))
    cv2.line(out, (px, 0), (px, h), (0, 0, 255), 2)
    return out


# ── Per-frame overlay (everything is in pixel space already) ────────────
def _annotate_frame(bgr, snap, steer, t_sec, state):
    out = bgr.copy()
    h, w = out.shape[:2]

    color_for_state = {
        "FRESH_POINT": (0, 255,   0),
        "HELD_POINT":  (0, 220, 255),
        "BLIND":       (0,   0, 255),
    }.get(state, (200, 200, 200))

    left_px  = snap.get("left_px")
    right_px = snap.get("right_px")
    apex_uv  = snap.get("lookahead_uv")

    if left_px is not None:
        cv2.line(out,
                 (int(left_px[0]),  int(left_px[1])),
                 (int(left_px[2]),  int(left_px[3])),
                 (255, 120, 0), 3, cv2.LINE_AA)
    if right_px is not None:
        cv2.line(out,
                 (int(right_px[0]), int(right_px[1])),
                 (int(right_px[2]), int(right_px[3])),
                 (40, 40, 255), 3, cv2.LINE_AA)

    if apex_uv is not None:
        u, v = int(round(apex_uv[0])), int(round(apex_uv[1]))
        cv2.drawMarker(out, (u, v), (0, 255, 255),
                       cv2.MARKER_CROSS, 28, 3)
        cv2.circle(out, (u, v), 11, (0, 255, 255), 2)

    bar_h = 40
    cv2.rectangle(out, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(
        out,
        f"t={t_sec:6.2f}s   state={state}   steer={steer:+.3f} rad",
        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_for_state, 2,
    )
    return out


# ── Replay the bag through the detector + follower ──────────────────────
def replay_and_capture():
    set_param_overrides({
        "camera_lateral_offset_m": CAMERA_Y_BIAS_M,
    })
    ros_shim.PUBLISH_LOG.clear()
    ros_shim.SUBSCRIPTIONS.clear()
    ros_shim.TIMERS.clear()

    t0_ns, t_end_ns = bag_extents(BAG_DB)
    set_now_ns(t0_ns)

    detector = WhiteLineHunter()
    follower = LaneTracer()
    follower.control_timer.arm(t0_ns)

    def decode_state(marker) -> str:
        rgb = (marker.color.r, marker.color.g, marker.color.b)
        if rgb == (0.0, 1.0, 0.0): return "FRESH_POINT"
        if rgb == (1.0, 1.0, 0.0): return "HELD_POINT"
        if rgb == (1.0, 0.0, 0.0): return "BLIND"
        return "?"

    frames: List[Dict] = []
    synth_drives: List[Tuple[int, float]] = []
    recorded_drives: List[Tuple[int, float]] = []
    state_history: Dict[int, str] = {}

    print("Replaying bag through WhiteLineHunter + LaneTracer ...")
    t_start = time.time()
    n_cam = 0
    drive_topic  = follower.drive_pub.topic
    marker_topic = follower.marker_pub.topic

    def drain():
        for ptopic, msg, when_ns in ros_shim.drain_publishes():
            if ptopic == drive_topic:
                synth_drives.append((when_ns, float(msg.drive.steering_angle)))
            elif ptopic == marker_topic:
                state_history[when_ns] = decode_state(msg)

    for topic, ts_ns, payload in iter_bag(BAG_DB):
        follower.control_timer.fire_due(ts_ns)
        drain()
        set_now_ns(ts_ns)

        if topic == "/zed/zed_node/rgb/image_rect_color":
            n_cam += 1
            arr = payload["data"]
            ch = {"bgr8": 3, "bgra8": 4}.get(payload["encoding"], 3)
            bgr = arr.reshape(payload["h"], payload["w"], ch)
            if ch == 4:
                bgr = bgr[:, :, :3]
            ok, jpg = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            cm = CImsg()
            cm.format = "jpeg"
            cm.data = jpg.tobytes()
            cm.header.stamp.sec = payload["stamp_sec"]
            cm.header.stamp.nanosec = payload["stamp_nsec"]
            detector._on_image(cm)
            drain()
            frames.append({
                "ts_ns":        ts_ns,
                "bgr":          bgr.copy(),
                "left_px":      list(detector.last_left_pixel)
                                if detector.last_left_pixel  is not None else None,
                "right_px":     list(detector.last_right_pixel)
                                if detector.last_right_pixel is not None else None,
                "lookahead_uv": tuple(detector.last_lookahead_px)
                                if detector.last_lookahead_px is not None else None,
            })
        elif topic == "/vesc/high_level/ackermann_cmd":
            recorded_drives.append((ts_ns, float(payload["steering_angle"])))

    follower.control_timer.fire_due(t_end_ns)
    drain()

    state_ts = sorted(state_history.keys())

    print(
        f"  done in {time.time() - t_start:.1f}s. "
        f"camera frames: {n_cam}, recorded steers: {len(recorded_drives)}, "
        f"synth steers: {len(synth_drives)}, marker updates: {len(state_ts)}"
    )

    return (frames, synth_drives, recorded_drives, state_ts, state_history,
            t0_ns, t_end_ns)


# ── Compose the side-by-side video ──────────────────────────────────────
def render_video(frames, synth, recorded, state_ts, state_lookup,
                 t0_ns, t_end_ns):
    if not frames:
        print("no camera frames captured, nothing to render")
        return

    rec_t  = np.array([(r[0] - t0_ns) / 1e9 for r in recorded])
    rec_st = np.array([r[1] for r in recorded])
    syn_t  = np.array([(s[0] - t0_ns) / 1e9 for s in synth])
    syn_st = np.array([s[1] for s in synth])

    bag_dur = (t_end_ns - t0_ns) / 1e9

    if rec_st.size and syn_st.size:
        y_lo = float(min(rec_st.min(), syn_st.min()) - 0.02)
        y_hi = float(max(rec_st.max(), syn_st.max()) + 0.02)
    elif rec_st.size:
        y_lo, y_hi = float(rec_st.min() - 0.02), float(rec_st.max() + 0.02)
    elif syn_st.size:
        y_lo, y_hi = float(syn_st.min() - 0.02), float(syn_st.max() + 0.02)
    else:
        y_lo, y_hi = -0.4, 0.4

    h_zed, w_zed = frames[0]["bgr"].shape[:2]
    panel_w = w_zed * VIDEO_SCALE
    panel_h = h_zed * VIDEO_SCALE
    chart_w = panel_w // 2
    out_w = panel_w
    out_h = panel_h + CHART_HEIGHT_PX

    rec_chart = _draw_chart(
        rec_t, rec_st,
        "RECORDED  /vesc/high_level/ackermann_cmd",
        "#222", 0.0, bag_dur, y_lo, y_hi, chart_w, CHART_HEIGHT_PX,
    )
    syn_chart = _draw_chart(
        syn_t, syn_st,
        "SYNTH  lane_follower (HEAD)",
        "#c54", 0.0, bag_dur, y_lo, y_hi, chart_w, CHART_HEIGHT_PX,
    )
    rec_chart = cv2.resize(rec_chart, (chart_w, CHART_HEIGHT_PX))
    syn_chart = cv2.resize(syn_chart, (chart_w, CHART_HEIGHT_PX))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_path = os.path.join(HERE, "_raw.mp4")
    vw = cv2.VideoWriter(raw_path, fourcc, FPS, (out_w, out_h))
    if not vw.isOpened():
        print(f"could not open {raw_path}")
        return

    print(f"Rendering {len(frames)} frames → {raw_path} ...")
    for i, fr in enumerate(frames):
        t_now = (fr["ts_ns"] - t0_ns) / 1e9

        idx = bisect.bisect_right(state_ts, fr["ts_ns"]) - 1
        state = state_lookup[state_ts[idx]] if idx >= 0 else "BLIND"

        steer = float(np.interp(t_now, syn_t, syn_st)) if syn_t.size else 0.0

        annotated = _annotate_frame(fr["bgr"], fr, steer, t_now, state)
        annotated_big = cv2.resize(
            annotated, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR,
        )

        rec_now = _stamp_cursor(rec_chart, t_now, 0.0, bag_dur)
        syn_now = _stamp_cursor(syn_chart, t_now, 0.0, bag_dur)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:panel_h, :panel_w] = annotated_big
        canvas[panel_h:, :chart_w] = rec_now
        canvas[panel_h:, chart_w:] = syn_now

        vw.write(canvas)
        if (i + 1) % 100 == 0:
            print(f"  frame {i + 1}/{len(frames)}")

    vw.release()
    print(f"raw mp4v wrote: {raw_path}")

    h264_path = OUT_PATH
    print(f"Re-encoding to H.264 → {h264_path}")
    rc = os.system(
        f'ffmpeg -y -loglevel error -i "{raw_path}" '
        f'-c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p "{h264_path}"'
    )
    if rc == 0:
        os.remove(raw_path)
        size_mb = os.path.getsize(h264_path) / (1024 * 1024)
        print(
            f"\nDone:  {h264_path}  ({size_mb:.1f} MB, "
            f"{len(frames)} frames @ {FPS} fps)"
        )
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

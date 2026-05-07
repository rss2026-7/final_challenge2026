"""Ablation harness: replay the Johnson Track bag through the live
WhiteLineHunter + LaneTracer with each candidate fix applied, and report
how the controller behaves at the deploy cruise speed (4.1 m/s default).

Cases tested:
    base       — current launch defaults (mirrors visualize.py)
    idle_blind — idle_when_blind:=true  (candidate a is "false"; default already false)
    bias_neg5  — camera_lateral_offset_m:=-0.05 (candidate b)
    look_x12   — LOOKAHEAD_GROUND_X_M = 1.2  (candidate c, larger lookahead)
    look_x09   — LOOKAHEAD_GROUND_X_M = 0.9  (candidate c, intermediate)
    tick60     — tick_rate_hz:=60.0 (candidate d, faster tick)

For each case the script reports:
    n_fresh / n_held / n_blind ticks
    steering: mean, std, |max|, jerk (mean |delta| between consecutive ticks)
    drives published / sec
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import ros_shim  # noqa
from ros_shim import set_now_ns, set_param_overrides

import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage as CImsg

from bag_reader import iter_bag, bag_extents, DB as BAG_DB


def _state(marker) -> str:
    rgb = (marker.color.r, marker.color.g, marker.color.b)
    if rgb == (0.0, 1.0, 0.0): return "FRESH"
    if rgb == (1.0, 1.0, 0.0): return "HELD"
    if rgb == (1.0, 0.0, 0.0): return "BLIND"
    return "?"


def _to_jpeg(p):
    arr = p["data"]
    ch = {"bgr8": 3, "bgra8": 4}.get(p["encoding"], 3)
    bgr = arr.reshape(p["h"], p["w"], ch)
    if ch == 4:
        bgr = bgr[:, :, :3]
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    m = CImsg()
    m.format = "jpeg"
    m.data = jpg.tobytes()
    m.header.stamp.sec = p["stamp_sec"]
    m.header.stamp.nanosec = p["stamp_nsec"]
    return m


def run(case: str, overrides: dict, lookahead_m: float | None = None) -> dict:
    # Reset fake-ROS singletons
    ros_shim.PUBLISH_LOG.clear()
    ros_shim.SUBSCRIPTIONS.clear()
    ros_shim.TIMERS.clear()
    set_param_overrides(overrides)

    # Re-import targets (module-level state survives, but instances are fresh)
    if "final_challenge.lane_detector" in sys.modules:
        del sys.modules["final_challenge.lane_detector"]
    if "final_challenge.lane_follower" in sys.modules:
        del sys.modules["final_challenge.lane_follower"]
    import final_challenge.lane_detector as ld_mod
    import final_challenge.lane_follower as lf_mod

    if lookahead_m is not None:
        ld_mod.LOOKAHEAD_GROUND_X_M = float(lookahead_m)

    t0, t_end = bag_extents(BAG_DB)
    set_now_ns(t0)
    det = ld_mod.WhiteLineHunter()
    flw = lf_mod.LaneTracer()
    flw.control_timer.arm(t0)

    drive_topic = flw.drive_pub.topic
    marker_topic = flw.marker_pub.topic

    drives: List[Tuple[int, float, float]] = []  # (ts, speed, steer)
    states: List[str] = []
    last_marker_ts = -1
    last_marker_state = None
    drive_states: List[str] = []  # state at the time of each drive

    def drain():
        nonlocal last_marker_state, last_marker_ts
        for ptopic, msg, when_ns in ros_shim.drain_publishes():
            if ptopic == marker_topic:
                last_marker_state = _state(msg)
                last_marker_ts = when_ns
                states.append(last_marker_state)
            elif ptopic == drive_topic:
                drives.append((when_ns,
                               float(msg.drive.speed),
                               float(msg.drive.steering_angle)))
                drive_states.append(last_marker_state or "?")

    n_cam = 0
    for topic, ts_ns, payload in iter_bag(BAG_DB):
        flw.control_timer.fire_due(ts_ns)
        drain()
        set_now_ns(ts_ns)
        if topic == "/zed/zed_node/rgb/image_rect_color":
            n_cam += 1
            try:
                det._on_image(_to_jpeg(payload))
            except Exception:
                pass
            drain()

    flw.control_timer.fire_due(t_end)
    drain()

    if not drives:
        return {"case": case, "n_drives": 0}

    ts = np.array([d[0] for d in drives])
    sp = np.array([d[1] for d in drives])
    st = np.array([d[2] for d in drives])
    dt = (ts[-1] - ts[0]) / 1e9

    # Steering jerk: mean |delta| between consecutive ticks
    if st.size >= 2:
        jerk = float(np.mean(np.abs(np.diff(st))))
        max_jerk = float(np.max(np.abs(np.diff(st))))
    else:
        jerk = max_jerk = 0.0

    from collections import Counter
    sc = Counter(drive_states)
    total = sum(sc.values()) or 1

    return {
        "case": case,
        "n_cam": n_cam,
        "n_drives": len(drives),
        "rate_hz": len(drives) / max(dt, 1e-3),
        "speed_mean": float(sp.mean()),
        "steer_mean": float(st.mean()),
        "steer_std": float(st.std()),
        "steer_absmax": float(np.max(np.abs(st))),
        "steer_jerk_mean": jerk,
        "steer_jerk_max": max_jerk,
        "frac_fresh": sc.get("FRESH", 0) / total,
        "frac_held":  sc.get("HELD",  0) / total,
        "frac_blind": sc.get("BLIND", 0) / total,
    }


def main():
    cases = [
        ("base",       {}, None),
        ("idle_blind", {"idle_when_blind": True}, None),
        ("bias_neg5",  {"camera_lateral_offset_m": -0.05}, None),
        ("look_x09",   {}, 0.9),
        ("look_x12",   {}, 1.2),
        ("tick60",     {"tick_rate_hz": 60.0}, None),
    ]
    rows = []
    for name, overrides, look in cases:
        t0 = time.time()
        r = run(name, overrides, look)
        r["wallclock_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"[{name}] done in {r['wallclock_s']}s — {r}")

    print()
    print("─" * 110)
    hdr = ("case", "rate_hz", "fresh%", "held%", "blind%",
           "st_mean", "st_std", "|st|max", "jerk_mean", "jerk_max")
    print("{:<12s} {:>8s} {:>7s} {:>7s} {:>7s} {:>9s} {:>8s} {:>9s} {:>10s} {:>9s}".format(*hdr))
    print("─" * 110)
    for r in rows:
        if r.get("n_drives", 0) == 0:
            print(f"{r['case']:<12s}  (no drives)"); continue
        print("{:<12s} {:>8.1f} {:>6.1f}% {:>6.1f}% {:>6.1f}% "
              "{:>+9.4f} {:>8.4f} {:>9.4f} {:>10.4f} {:>9.4f}".format(
                  r["case"], r["rate_hz"],
                  100*r["frac_fresh"], 100*r["frac_held"], 100*r["frac_blind"],
                  r["steer_mean"], r["steer_std"], r["steer_absmax"],
                  r["steer_jerk_mean"], r["steer_jerk_max"]))


if __name__ == "__main__":
    main()

"""Sweep camera-y offsets and report MAE-vs-recorded for each.

Patches `transform_uv_to_xy` in lane_detector's namespace (where it was
already pulled in via `from … import`) so both Hough and blob spine
projections get the same offset applied uniformly.

Use the MATCH parameter set otherwise. Speed is constant 3.5 — the only
moving variable is steering, which is what we're trying to align.
"""
from __future__ import annotations
import sys, os, importlib
sys.path.insert(0, "/tmp/rosbag_johnson")
sys.path.insert(0, "/home/adhoc/Desktop/final_challenge2026")
import ros_shim  # noqa
from ros_shim import set_now_ns, set_param_overrides, _Image
import numpy as np

# Import the modules we'll patch
import final_challenge.homography_transformer as ht
import final_challenge.lane_detector as ld_mod
from final_challenge.lane_follower import BoundaryPurePursuit

from bag_reader import iter_bag, bag_extents

ORIG_TRANSFORM = ht.transform_uv_to_xy
Y_OFFSET = 0.0  # patched per run

def shifted_uvxy(H, u, v):
    x, y = ORIG_TRANSFORM(H, u, v)
    return x, y + Y_OFFSET

# Patch in BOTH places: the source module AND the binding lane_detector
# already pulled in via `from … import transform_uv_to_xy`.
ht.transform_uv_to_xy = shifted_uvxy
ld_mod.transform_uv_to_xy = shifted_uvxy

MATCH_PARAMS = {
    "drive_topic": "/drive",
    "nominal_speed": 3.5, "min_speed": 3.5, "max_speed": 3.5,
    "curvature_speed_gain": 0.0, "lost_line_speed": 3.5,
    "fresh_msg_timeout": 0.80, "stale_path_timeout": 1.50,
    "stop_if_no_path": False,
    "half_width_init": 0.30, "half_width_min": 0.20, "half_width_max": 0.65,
    "lookahead_distance": 1.2, "lost_line_lookahead_distance": 0.9,
    "min_lookahead_distance": 0.5, "curvature_lookahead_gain": 2.0,
    "min_path_arc_length": 0.3,
    "steering_alpha": 0.35, "max_steering_angle": 0.40,
    "target_alpha": 0.20, "half_width_alpha": 0.1,
}


def run_one(y_off: float):
    global Y_OFFSET
    Y_OFFSET = y_off

    set_param_overrides(MATCH_PARAMS)
    # Wipe state from prior runs by clearing shim's pub/sub registries
    ros_shim.PUBLISH_LOG.clear()
    ros_shim.SUBSCRIPTIONS.clear()
    ros_shim.TIMERS.clear()

    t0, t_end = bag_extents()
    set_now_ns(t0)
    det = ld_mod.LaneDetector()
    ppc = BoundaryPurePursuit()
    ppc.control_timer.arm(t0)

    rec = []; syn = []
    for topic, ts, p in iter_bag():
        ppc.control_timer.fire_due(ts)
        for ptopic, msg, pts in ros_shim.drain_publishes():
            if ptopic == ppc.drive_pub.topic:
                syn.append((pts, float(msg.drive.steering_angle)))
        set_now_ns(ts)
        if topic.endswith("rgb/image_rect_color"):
            m = _Image()
            m.height = p["h"]; m.width = p["w"]; m.encoding = p["encoding"]
            m.step = p["step"]; m.data = p["data"]
            m.header.stamp.sec = p["stamp_sec"]; m.header.stamp.nanosec = p["stamp_nsec"]
            det.image_callback(m)
        elif topic.endswith("ackermann_cmd"):
            rec.append((ts, float(p["steering_angle"])))
        for ptopic, msg, pts in ros_shim.drain_publishes():
            if ptopic == ppc.drive_pub.topic:
                syn.append((pts, float(msg.drive.steering_angle)))
    ppc.control_timer.fire_due(t_end)
    for ptopic, msg, pts in ros_shim.drain_publishes():
        if ptopic == ppc.drive_pub.topic:
            syn.append((pts, float(msg.drive.steering_angle)))

    rec_t = np.array([(r[0] - t0) / 1e9 for r in rec])
    rec_st = np.array([r[1] for r in rec])
    syn_t = np.array([(s[0] - t0) / 1e9 for s in syn])
    syn_st = np.array([s[1] for s in syn])
    syn_at = np.interp(rec_t, syn_t, syn_st)
    bias = float(np.median(syn_at - rec_st))
    mae = float(np.mean(np.abs(syn_at - rec_st)))
    return {
        "y_off": y_off,
        "bias": bias,
        "mae": mae,
        "syn_mean": float(syn_st.mean()),
        "syn_std": float(syn_st.std()),
        "syn_min": float(syn_st.min()),
        "syn_max": float(syn_st.max()),
    }


def main():
    offsets = [round(v, 3) for v in np.arange(-0.30, 0.05, 0.02)]
    print(f"sweeping {len(offsets)} offsets …")
    print(f"{'y_off':>8s}  {'bias':>9s}  {'MAE':>8s}  "
          f"{'syn_mean':>9s}  {'syn_std':>8s}  {'min':>8s}  {'max':>8s}")
    rows = []
    for y in offsets:
        r = run_one(y)
        rows.append(r)
        print(f"{r['y_off']:+8.3f}  {r['bias']:+9.4f}  {r['mae']:8.4f}  "
              f"{r['syn_mean']:+9.4f}  {r['syn_std']:8.4f}  "
              f"{r['syn_min']:+8.4f}  {r['syn_max']:+8.4f}")
    best = min(rows, key=lambda r: r["mae"])
    print()
    print(f"BEST y_off = {best['y_off']:+.3f}  (MAE {best['mae']:.4f}, "
          f"bias {best['bias']:+.4f})")
    return rows, best


if __name__ == "__main__":
    rows, best = main()
    # Save CSV
    import csv
    with open("/tmp/rosbag_johnson/results/offset_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote results/offset_sweep.csv")

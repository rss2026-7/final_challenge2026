"""Sweep camera_y_offset values (post-rebase: this is now a real lane_follower
parameter, no homography monkey-patching needed) and report MAE-vs-recorded
for each. Speed is held constant 3.5 m/s — the only moving variable is
steering, which is what we're aligning."""
from __future__ import annotations
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
import ros_shim  # noqa
from ros_shim import set_now_ns, set_param_overrides
import numpy as np
import cv2

from final_challenge.lane_detector import LaneDetector
from final_challenge.lane_follower import BoundaryPurePursuit
from sensor_msgs.msg import CompressedImage as _CImsg

from bag_reader import iter_bag, bag_extents

Y_OFFSET = 0.0  # set per run

BASE_PARAMS = {
    "drive_topic": "/drive",
    "nominal_speed": 3.5, "min_speed": 3.5, "max_speed": 3.5,
    "curvature_speed_gain": 0.0, "lost_line_speed": 3.5,
    "fresh_msg_timeout": 0.80, "stale_path_timeout": 1.50,
    "stop_if_no_path": False,
    "half_width_init": 0.30, "half_width_min": 0.20, "half_width_max": 0.65,
    "lookahead_distance": 1.68, "lost_line_lookahead_distance": 0.9,
    "min_lookahead_distance": 0.5, "curvature_lookahead_gain": 2.0,
    "min_path_arc_length": 0.0,
    "steering_alpha": 0.35, "max_steering_angle": 0.20,
    "target_alpha": 0.20, "half_width_alpha": 0.1,
    "cte_gain": 0.5,
    "enable_visualization": False,
}


def _to_jpeg(payload):
    arr = payload["data"]
    ch = {"bgr8": 3, "bgra8": 4}.get(payload["encoding"], 3)
    h, w = payload["h"], payload["w"]
    frame = arr.reshape(h, w, ch)
    if ch == 4: frame = frame[:, :, :3]
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    m = _CImsg()
    m.format = "jpeg"; m.data = jpeg.tobytes()
    m.header.stamp.sec = payload["stamp_sec"]; m.header.stamp.nanosec = payload["stamp_nsec"]
    return m


def run_one(y_off: float):
    global Y_OFFSET
    Y_OFFSET = y_off

    params = dict(BASE_PARAMS); params["camera_y_offset"] = y_off
    set_param_overrides(params)
    ros_shim.PUBLISH_LOG.clear()
    ros_shim.SUBSCRIPTIONS.clear()
    ros_shim.TIMERS.clear()

    t0, t_end = bag_extents()
    set_now_ns(t0)
    det = LaneDetector()
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
            det.image_callback(_to_jpeg(p))
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
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "offset_sweep.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_path}")

#!/usr/bin/env python3
"""Live YOLO preview on a video file.

Detects only: fire hydrant, parking meter, backpack, bottle, traffic light, person.
Fire hydrants, parking meters, backpacks, and bottles are highlighted in red.
(COCO has a single "bottle" class — water bottles fall under it.)

Usage: python yolo_preview.py path/to/video [--model yolov8s.pt] [--conf 0.25]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class IDs we keep. Targets are highlighted in red; the others
# (traffic light, person) draw in green for context. Anything else is hidden.
HIGHLIGHT_CLASSES = {10: "fire hydrant", 12: "parking meter",
                     24: "backpack", 39: "bottle"}
ALLOWED_CLASSES = {0: "person", 9: "traffic light",
                   10: "fire hydrant", 12: "parking meter",
                   24: "backpack", 39: "bottle"}

# OpenCV ignores QuickTime rotation metadata and gives us the raw landscape
# buffer. iPhone clips are filmed portrait with rotation=-90 — rotate 90° CW to
# put them right-side-up.
ROTATE_MODES = {
    "none": None,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", help="Path to the input video")
    p.add_argument("--model", default="yolov8s.pt",
                   help="Ultralytics model weights (default: yolov8s.pt — small but more stable than n)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size (default: 640)")
    p.add_argument("--only-targets", action="store_true",
                   help="Show only fire hydrant + parking meter detections")
    p.add_argument("--rotate", choices=ROTATE_MODES.keys(), default="cw",
                   help="Rotate frames before inference (default: cw — portrait for iPhone clips)")
    p.add_argument("--no-track", action="store_true",
                   help="Disable ByteTrack smoothing (use raw per-frame detections)")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame; intermediate frames reuse last boxes (default: 1)")
    p.add_argument("--playback-fps", type=float, default=0.0,
                   help="Cap display FPS (e.g. 15 to slow down). 0 = run as fast as possible.")
    return p.parse_args()


def extract_dets(result, only_targets):
    """Pull detections out of a ultralytics result into plain tuples."""
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    names = result.names
    for box in result.boxes:
        cls = int(box.cls[0])
        if cls not in ALLOWED_CLASSES:
            continue
        if only_targets and cls not in HIGHLIGHT_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        tid = int(box.id[0]) if getattr(box, "id", None) is not None else None
        out.append((x1, y1, x2, y2, cls, conf, tid, names.get(cls, str(cls))))
    return out


def draw_dets(frame, dets):
    for x1, y1, x2, y2, cls, conf, tid, name in dets:
        is_target = cls in HIGHLIGHT_CLASSES
        color = (0, 0, 255) if is_target else (0, 255, 0)
        thickness = 3 if is_target else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{name} {conf:.2f}"
        if tid is not None:
            label = f"#{tid} " + label
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"error: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotate_code = ROTATE_MODES[args.rotate]
    win = "YOLO live preview — q/ESC to quit, space to pause"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    paused = False
    frame_idx = 0
    t_prev = time.time()
    fps_smooth = 0.0
    last_frame = None
    last_dets = []
    target_frame_dt = 1.0 / args.playback_fps if args.playback_fps > 0 else 0.0
    next_show_t = time.time()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("end of video.")
                break
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            last_frame = frame
            frame_idx += 1
        else:
            frame = last_frame.copy() if last_frame is not None else None
            if frame is None:
                break

        run_inference = (frame_idx % max(1, args.stride) == 0) or paused
        if run_inference:
            allowed_ids = list(ALLOWED_CLASSES.keys())
            if args.no_track:
                result = model.predict(
                    frame, conf=args.conf, imgsz=args.imgsz,
                    classes=allowed_ids, verbose=False
                )[0]
            else:
                # ByteTrack keeps IDs across frames and stabilizes boxes.
                result = model.track(
                    frame, conf=args.conf, imgsz=args.imgsz,
                    classes=allowed_ids, persist=True,
                    tracker="bytetrack.yaml", verbose=False
                )[0]
            last_dets = extract_dets(result, args.only_targets)

        annotated = draw_dets(frame.copy(), last_dets)

        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

        targets_seen = sorted({HIGHLIGHT_CLASSES[d[4]] for d in last_dets
                               if d[4] in HIGHLIGHT_CLASSES})

        hud_lines = [
            f"frame {frame_idx}  fps {fps_smooth:5.1f} (src {src_fps:.1f})",
            f"detections drawn: {len(last_dets)}"
            + ("  [tracked]" if not args.no_track else "  [raw]"),
        ]
        if targets_seen:
            hud_lines.append("TARGETS: " + ", ".join(targets_seen))
        if paused:
            hud_lines.append("[PAUSED]")
        for i, line in enumerate(hud_lines):
            y = 28 + i * 26
            cv2.putText(annotated, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(annotated, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if target_frame_dt > 0:
            sleep_for = next_show_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_show_t += target_frame_dt
            if next_show_t < time.time():  # we fell behind, resync
                next_show_t = time.time() + target_frame_dt

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

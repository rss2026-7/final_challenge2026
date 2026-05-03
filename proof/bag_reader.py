"""Iterate the Johnson Track rosbag chronologically, decoding each message
into Python objects suitable for feeding the ROS shim.

Yields tuples (topic_name, bag_time_ns, payload) where payload is one of:
    - Image dict  : {h, w, encoding, step, data (np.ndarray), stamp_sec, stamp_nsec, frame_id}
    - AckermannDriveStamped dict : {speed, steering_angle, ..., stamp_sec, stamp_nsec, frame_id}
"""
from __future__ import annotations

import os
import sqlite3
import struct
from typing import Dict, Iterator, Tuple

import numpy as np

# Default bag location: extracted alongside this file. Override with $BAG_DB.
# Common alternates also probed automatically below.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_db() -> str:
    if "BAG_DB" in os.environ:
        return os.environ["BAG_DB"]
    candidates = [
        os.path.join(_HERE, "rosbag2_2025_04_09-22_01_22",
                     "rosbag2_2025_04_09-22_01_22_0.db3"),
        "/tmp/rosbag_johnson/rosbag2_2025_04_09-22_01_22/"
        "rosbag2_2025_04_09-22_01_22_0.db3",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Not extracted yet — try to unzip the bag from a sibling .zip.
    zip_candidates = [
        os.path.join(_HERE, "johnson_track_rosbag.zip"),
        os.path.join(_HERE, "..", "johnson_track_rosbag.zip"),
    ]
    for z in zip_candidates:
        if os.path.exists(z):
            import zipfile
            print(f"bag_reader: extracting {z} into {_HERE} (one-time, ~2 GB) …")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(_HERE)
            for c in candidates:
                if os.path.exists(c):
                    return c
            break

    # Fall back to the proof-relative path; error message will point users
    # to drop the .zip in proof/ if neither extracted nor zipped.
    return candidates[0]


DB = _resolve_db()


# ───────────────────────────────────────────────────────────────────────
# Minimal CDR (DDS XCDR1) reader — only what these three message types need
# ───────────────────────────────────────────────────────────────────────
class CDR:
    __slots__ = ("buf", "pos", "origin", "endian", "le")

    def __init__(self, blob: bytes):
        if len(blob) < 4:
            raise ValueError("CDR blob too short")
        self.le = (blob[1] & 0x01) == 0x01
        self.endian = "<" if self.le else ">"
        self.buf = blob
        self.pos = 4
        self.origin = 4

    def _align(self, n: int) -> None:
        rel = (self.pos - self.origin) % n
        if rel:
            self.pos += n - rel

    def u8(self) -> int:
        v = self.buf[self.pos]; self.pos += 1; return v

    def i32(self) -> int:
        self._align(4)
        v = struct.unpack_from(self.endian + "i", self.buf, self.pos)[0]
        self.pos += 4; return v

    def u32(self) -> int:
        self._align(4)
        v = struct.unpack_from(self.endian + "I", self.buf, self.pos)[0]
        self.pos += 4; return v

    def f32(self) -> float:
        self._align(4)
        v = struct.unpack_from(self.endian + "f", self.buf, self.pos)[0]
        self.pos += 4; return v

    def string(self) -> str:
        n = self.u32()
        s = self.buf[self.pos:self.pos + n - 1].decode("utf-8", errors="replace")
        self.pos += n
        return s

    def bytes(self, n: int) -> bytes:
        self._align(1)
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v


def parse_image(blob: bytes) -> Dict:
    c = CDR(blob)
    sec = c.i32(); nsec = c.u32(); frame = c.string()
    h = c.u32(); w = c.u32()
    enc = c.string()
    is_be = c.u8()
    step = c.u32()
    n = c.u32()
    raw = c.buf[c.pos:c.pos + n]
    arr = np.frombuffer(raw, dtype=np.uint8)  # zero-copy view of the bag blob
    return {
        "stamp_sec": sec, "stamp_nsec": nsec, "frame_id": frame,
        "h": h, "w": w, "encoding": enc, "is_bigendian": is_be, "step": step,
        "data": arr,
    }


def parse_ackermann(blob: bytes) -> Dict:
    c = CDR(blob)
    sec = c.i32(); nsec = c.u32(); frame = c.string()
    return {
        "stamp_sec": sec, "stamp_nsec": nsec, "frame_id": frame,
        "steering_angle":         c.f32(),
        "steering_angle_velocity": c.f32(),
        "speed":                  c.f32(),
        "acceleration":           c.f32(),
        "jerk":                   c.f32(),
    }


# ───────────────────────────────────────────────────────────────────────
# Iterate
# ───────────────────────────────────────────────────────────────────────
def iter_bag(db_path: str = DB) -> Iterator[Tuple[str, int, Dict]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    topics = {tid: name for tid, name in cur.execute("SELECT id,name FROM topics")}
    parsers = {
        "/zed/zed_node/rgb/image_rect_color": parse_image,
        "/cone_debug_img":                    parse_image,
        "/vesc/high_level/ackermann_cmd":     parse_ackermann,
    }
    for tid, ts, data in cur.execute(
        "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp ASC"
    ):
        name = topics[tid]
        parser = parsers.get(name)
        if parser is None:
            continue
        yield name, int(ts), parser(bytes(data))
    con.close()


def bag_extents(db_path: str = DB) -> Tuple[int, int]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    lo, hi = cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages").fetchone()
    con.close()
    return int(lo), int(hi)


if __name__ == "__main__":
    counts = {}
    for topic, ts, _ in iter_bag():
        counts[topic] = counts.get(topic, 0) + 1
    print(counts)

"""Minimal sys.modules monkey-patch that fakes the ROS 2 packages used by
final_challenge.lane_detector and final_challenge.lane_follower so the
real classes can be instantiated and exercised without ROS 2 installed.

Public knobs (set BEFORE constructing any Node):
    set_now_ns(ns)          — virtual clock value used by Clock.now()
    set_param_overrides(d)  — dict of name->value to override declare_parameter defaults
    drain_publishes()       — returns and clears the global publish log
    drain_timers()          — returns and clears the global timer registry

Each `publish()` from a fake Publisher logs (topic, msg, now_ns) into PUBLISH_LOG
*and* invokes any subscriber callbacks that were registered for the same topic
(intra-process delivery — that's how lane_detector → lane_follower is wired here).
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ───────────────────────────────────────────────────────────────────────
# Globals
# ───────────────────────────────────────────────────────────────────────
_NOW_NS: int = 0
_PARAM_OVERRIDES: Dict[str, Any] = {}
PUBLISH_LOG: List[Tuple[str, Any, int]] = []
TIMERS: List["Timer"] = []
SUBSCRIPTIONS: Dict[str, List[Callable[[Any], None]]] = {}
LOG_LEVEL_VERBOSE = False

def set_now_ns(ns: int) -> None:
    global _NOW_NS
    _NOW_NS = int(ns)

def get_now_ns() -> int:
    return _NOW_NS

def set_param_overrides(overrides: Dict[str, Any]) -> None:
    _PARAM_OVERRIDES.clear()
    _PARAM_OVERRIDES.update(overrides)

def drain_publishes() -> List[Tuple[str, Any, int]]:
    out = list(PUBLISH_LOG)
    PUBLISH_LOG.clear()
    return out

def set_verbose(v: bool) -> None:
    global LOG_LEVEL_VERBOSE
    LOG_LEVEL_VERBOSE = bool(v)


# ───────────────────────────────────────────────────────────────────────
# Time / Clock / Duration
# ───────────────────────────────────────────────────────────────────────
class _TimeMsg:
    def __init__(self, sec: int = 0, nanosec: int = 0):
        self.sec = int(sec)
        self.nanosec = int(nanosec)


class Time:
    __slots__ = ("nanoseconds",)
    def __init__(self, nanoseconds: int):
        self.nanoseconds = int(nanoseconds)

    def __sub__(self, other: "Time") -> "Duration":
        return Duration(self.nanoseconds - other.nanoseconds)

    def to_msg(self) -> _TimeMsg:
        return _TimeMsg(self.nanoseconds // 1_000_000_000,
                        self.nanoseconds % 1_000_000_000)


class Duration:
    __slots__ = ("nanoseconds",)
    def __init__(self, nanoseconds: int):
        self.nanoseconds = int(nanoseconds)


class Clock:
    def now(self) -> Time:
        return Time(_NOW_NS)


# ───────────────────────────────────────────────────────────────────────
# Logger (handles throttle_duration_sec kwarg)
# ───────────────────────────────────────────────────────────────────────
class _Logger:
    def __init__(self, name: str):
        self.name = name
        self._last: Dict[str, int] = {}

    def _emit(self, level: str, msg: str, throttle_duration_sec: float = 0.0,
              **_: Any) -> None:
        if throttle_duration_sec > 0:
            key = f"{level}:{msg[:40]}"
            now = _NOW_NS
            last = self._last.get(key, -10**18)
            if (now - last) < int(throttle_duration_sec * 1e9):
                return
            self._last[key] = now
        if LOG_LEVEL_VERBOSE:
            print(f"[{level} {self.name}] {msg}")

    def info(self, msg: str, **kw: Any) -> None:  self._emit("INFO",  msg, **kw)
    def warn(self, msg: str, **kw: Any) -> None:  self._emit("WARN",  msg, **kw)
    def warning(self, msg: str, **kw: Any) -> None: self._emit("WARN", msg, **kw)
    def error(self, msg: str, **kw: Any) -> None: self._emit("ERROR", msg, **kw)
    def debug(self, msg: str, **kw: Any) -> None: self._emit("DEBUG", msg, **kw)


# ───────────────────────────────────────────────────────────────────────
# Parameters
# ───────────────────────────────────────────────────────────────────────
class _Param:
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value
        self.type_ = type(value).__name__


# ───────────────────────────────────────────────────────────────────────
# Pub/Sub/Timer
# ───────────────────────────────────────────────────────────────────────
class Publisher:
    def __init__(self, msg_type: type, topic: str, qos: Any):
        self.msg_type = msg_type
        self.topic = topic

    def publish(self, msg: Any) -> None:
        PUBLISH_LOG.append((self.topic, msg, _NOW_NS))
        # Intra-process delivery: forward to any subscribers on the same topic
        for cb in SUBSCRIPTIONS.get(self.topic, ()):
            cb(msg)


class Subscription:
    def __init__(self, msg_type: type, topic: str, callback: Callable[[Any], None],
                 qos: Any):
        self.msg_type = msg_type
        self.topic = topic
        self.callback = callback
        SUBSCRIPTIONS.setdefault(topic, []).append(callback)


class Timer:
    def __init__(self, period_sec: float, callback: Callable[[], None]):
        self.period_sec = float(period_sec)
        self.period_ns = int(period_sec * 1e9)
        self.callback = callback
        self.next_fire_ns: Optional[int] = None  # set by harness via .arm(start_ns)
        TIMERS.append(self)

    def arm(self, start_ns: int) -> None:
        self.next_fire_ns = start_ns + self.period_ns

    def fire_due(self, now_ns: int) -> int:
        """Fire the callback once for each period elapsed up to now_ns. Returns count."""
        n = 0
        if self.next_fire_ns is None:
            self.next_fire_ns = now_ns + self.period_ns
            return 0
        while self.next_fire_ns <= now_ns:
            set_now_ns(self.next_fire_ns)
            self.callback()
            self.next_fire_ns += self.period_ns
            n += 1
        return n


# ───────────────────────────────────────────────────────────────────────
# Node base
# ───────────────────────────────────────────────────────────────────────
class Node:
    def __init__(self, name: str):
        self._name = name
        self._params: Dict[str, _Param] = {}
        self._on_param_cb: Optional[Callable] = None
        self._logger = _Logger(name)

    def get_name(self) -> str:
        return self._name

    def get_logger(self) -> _Logger:
        return self._logger

    def get_clock(self) -> Clock:
        return Clock()

    def declare_parameter(self, name: str, default: Any) -> _Param:
        if name in _PARAM_OVERRIDES:
            value = _PARAM_OVERRIDES[name]
        else:
            value = default
        # cast to default's type when sensible
        if default is not None and value is not None:
            try:
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int) and not isinstance(default, bool):
                    value = int(value)
                elif isinstance(default, float):
                    value = float(value)
            except (TypeError, ValueError):
                pass
        p = _Param(name, value)
        self._params[name] = p
        return p

    def get_parameter(self, name: str) -> _Param:
        if name not in self._params:
            raise KeyError(f"parameter {name} not declared")
        return self._params[name]

    def set_parameters(self, params: List[_Param]) -> None:
        for p in params:
            self._params[p.name] = p
        if self._on_param_cb is not None:
            self._on_param_cb(params)

    def add_on_set_parameters_callback(self, cb: Callable) -> None:
        self._on_param_cb = cb

    def create_subscription(self, msg_type: type, topic: str,
                            callback: Callable[[Any], None], qos: Any,
                            callback_group: Any = None) -> Subscription:
        return Subscription(msg_type, topic, callback, qos)

    def create_publisher(self, msg_type: type, topic: str, qos: Any,
                         callback_group: Any = None) -> Publisher:
        return Publisher(msg_type, topic, qos)

    def create_timer(self, period_sec: float, callback: Callable[[], None],
                     callback_group: Any = None) -> Timer:
        return Timer(period_sec, callback)

    def destroy_node(self) -> None:
        pass


# ───────────────────────────────────────────────────────────────────────
# Generic message holder
# ───────────────────────────────────────────────────────────────────────
class _Msg:
    """Attribute-bag message; nested attrs auto-instantiated by subclasses."""
    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.__dict__}>"


def _msg(**fields: Any) -> Callable[[], _Msg]:
    """Factory for a message class with given default-value factories."""
    class _M(_Msg):
        def __init__(self) -> None:
            for k, factory in fields.items():
                setattr(self, k, factory() if callable(factory) else factory)
    return _M


# ── std_msgs ────────────────────────────────────────────────────────────
class _Header(_Msg):
    def __init__(self) -> None:
        self.stamp = _TimeMsg()
        self.frame_id = ""

class _String(_Msg):
    def __init__(self) -> None:
        self.data = ""

class _Bool(_Msg):
    def __init__(self) -> None:
        self.data = False


# ── geometry_msgs ───────────────────────────────────────────────────────
class _Point(_Msg):
    def __init__(self) -> None:
        self.x = 0.0; self.y = 0.0; self.z = 0.0

class _Point32(_Msg):
    def __init__(self) -> None:
        self.x = 0.0; self.y = 0.0; self.z = 0.0

class _Quat(_Msg):
    def __init__(self) -> None:
        self.x = 0.0; self.y = 0.0; self.z = 0.0; self.w = 1.0

class _Pose(_Msg):
    def __init__(self) -> None:
        self.position = _Point()
        self.orientation = _Quat()

class _PoseStamped(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.pose = _Pose()

class _PointStamped(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.point = _Point()

class _PoseArray(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.poses: List[_Pose] = []


# ── nav_msgs ────────────────────────────────────────────────────────────
class _Path(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.poses: List[_PoseStamped] = []

class _Odom(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.child_frame_id = ""
        self.pose = _Pose()
        self.twist = _Pose()  # close enough; not used by lane stack


# ── ackermann_msgs ──────────────────────────────────────────────────────
class _AckermannDrive(_Msg):
    def __init__(self) -> None:
        self.steering_angle = 0.0
        self.steering_angle_velocity = 0.0
        self.speed = 0.0
        self.acceleration = 0.0
        self.jerk = 0.0

class _AckermannDriveStamped(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.drive = _AckermannDrive()


# ── visualization_msgs ──────────────────────────────────────────────────
class _Color(_Msg):
    def __init__(self) -> None:
        self.r = 0.0; self.g = 0.0; self.b = 0.0; self.a = 1.0

class _Vec3(_Msg):
    def __init__(self) -> None:
        self.x = 0.0; self.y = 0.0; self.z = 0.0

class _DurMsg(_Msg):
    def __init__(self) -> None:
        self.sec = 0; self.nanosec = 0

class _Marker(_Msg):
    # type constants
    ARROW = 0; CUBE = 1; SPHERE = 2; CYLINDER = 3
    LINE_STRIP = 4; LINE_LIST = 5; CUBE_LIST = 6; SPHERE_LIST = 7
    POINTS = 8; TEXT_VIEW_FACING = 9; MESH_RESOURCE = 10; TRIANGLE_LIST = 11
    # action constants
    ADD = 0; MODIFY = 0; DELETE = 2; DELETEALL = 3

    def __init__(self) -> None:
        self.header = _Header()
        self.ns = ""
        self.id = 0
        self.type = 0
        self.action = 0
        self.pose = _Pose()
        self.scale = _Vec3()
        self.color = _Color()
        self.lifetime = _DurMsg()


# ── sensor_msgs ─────────────────────────────────────────────────────────
class _Image(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.step = 0
        self.data = b""   # bytes OR np.ndarray

class _CompressedImage(_Msg):
    def __init__(self) -> None:
        self.header = _Header()
        self.format = ""   # e.g. "jpeg"
        self.data = b""    # JPEG/PNG-encoded bytes

class _RegionOfInterest(_Msg):
    def __init__(self) -> None:
        self.x_offset = 0; self.y_offset = 0
        self.height = 0; self.width = 0
        self.do_rectify = False


# ── vs_msgs ─────────────────────────────────────────────────────────────
class _ConeLocation(_Msg):
    def __init__(self) -> None:
        self.x_pos = 0.0; self.y_pos = 0.0

class _ConeLocationPixel(_Msg):
    def __init__(self) -> None:
        self.u = 0.0; self.v = 0.0

class _ParkingError(_Msg):
    def __init__(self) -> None:
        self.x_error = 0.0; self.y_error = 0.0; self.distance_error = 0.0


# ── rcl_interfaces ──────────────────────────────────────────────────────
class _SetParametersResult(_Msg):
    def __init__(self, successful: bool = True, reason: str = "") -> None:
        self.successful = bool(successful)
        self.reason = reason


# ── QoS (just enums; unused but imported) ───────────────────────────────
class _DurabilityPolicy:
    VOLATILE = 1; TRANSIENT_LOCAL = 2

class _ReliabilityPolicy:
    SYSTEM_DEFAULT = 0; RELIABLE = 1; BEST_EFFORT = 2; UNKNOWN = 3

class _HistoryPolicy:
    SYSTEM_DEFAULT = 0; KEEP_LAST = 1; KEEP_ALL = 2; UNKNOWN = 3

class _QoSProfile:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


# ── Executors / Callback groups (no-op for our single-threaded harness) ──
class _MultiThreadedExecutor:
    def __init__(self, *a: Any, **kw: Any) -> None: pass
    def add_node(self, n: Any) -> None: pass
    def spin(self) -> None: pass
    def spin_once(self, timeout_sec: float = 0.0) -> None: pass
    def shutdown(self) -> None: pass

class _SingleThreadedExecutor(_MultiThreadedExecutor): pass

class _MutuallyExclusiveCallbackGroup: pass
class _ReentrantCallbackGroup: pass


# ───────────────────────────────────────────────────────────────────────
# CvBridge
# ───────────────────────────────────────────────────────────────────────
class CvBridge:
    def imgmsg_to_cv2(self, msg: _Image, desired_encoding: str = "passthrough") -> np.ndarray:
        h, w = int(msg.height), int(msg.width)
        enc = msg.encoding
        data = msg.data
        if isinstance(data, np.ndarray):
            arr = data
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 1:
                channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4,
                            "mono8": 1}.get(enc, 3)
                arr = arr.reshape(h, w, channels) if channels > 1 else arr.reshape(h, w)
        else:
            buf = bytes(data)
            channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4,
                        "mono8": 1}.get(enc, 3)
            arr = np.frombuffer(buf, dtype=np.uint8)
            if channels > 1:
                arr = arr.reshape(h, w, channels)
            else:
                arr = arr.reshape(h, w)

        # convert to desired
        if desired_encoding in ("passthrough", enc):
            return arr
        if enc == "bgra8" and desired_encoding == "bgr8":
            return arr[:, :, :3]
        if enc == "rgba8" and desired_encoding == "bgr8":
            return arr[:, :, [2, 1, 0]]
        if enc == "rgb8" and desired_encoding == "bgr8":
            return arr[:, :, ::-1]
        if enc == "bgr8" and desired_encoding == "rgb8":
            return arr[:, :, ::-1]
        return arr  # best-effort

    def cv2_to_imgmsg(self, img: np.ndarray, encoding: str = "passthrough") -> _Image:
        m = _Image()
        if img.ndim == 2:
            h, w = img.shape; ch = 1
        else:
            h, w, ch = img.shape
        m.height = int(h); m.width = int(w)
        m.encoding = encoding
        m.step = int(w * ch)
        m.data = np.ascontiguousarray(img)
        return m


class CvBridgeError(Exception):
    pass


# ───────────────────────────────────────────────────────────────────────
# rclpy module skeleton
# ───────────────────────────────────────────────────────────────────────
def init(args: Any = None) -> None: pass
def shutdown() -> None: pass
def spin(node: Node) -> None: pass
def spin_once(node: Node, timeout_sec: float = 0.0) -> None: pass


# ───────────────────────────────────────────────────────────────────────
# Install into sys.modules
# ───────────────────────────────────────────────────────────────────────
def _mk(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def install() -> None:
    rclpy = _mk("rclpy")
    rclpy.init = init
    rclpy.shutdown = shutdown
    rclpy.spin = spin
    rclpy.spin_once = spin_once

    rclpy_node = _mk("rclpy.node")
    rclpy_node.Node = Node
    rclpy.node = rclpy_node

    rclpy_qos = _mk("rclpy.qos")
    rclpy_qos.QoSProfile = _QoSProfile
    rclpy_qos.DurabilityPolicy = _DurabilityPolicy
    rclpy_qos.QoSReliabilityPolicy = _ReliabilityPolicy
    rclpy_qos.ReliabilityPolicy   = _ReliabilityPolicy
    rclpy_qos.QoSHistoryPolicy    = _HistoryPolicy
    rclpy_qos.HistoryPolicy       = _HistoryPolicy
    rclpy.qos = rclpy_qos

    rclpy_exec = _mk("rclpy.executors")
    rclpy_exec.MultiThreadedExecutor  = _MultiThreadedExecutor
    rclpy_exec.SingleThreadedExecutor = _SingleThreadedExecutor
    rclpy.executors = rclpy_exec

    rclpy_cbg = _mk("rclpy.callback_groups")
    rclpy_cbg.MutuallyExclusiveCallbackGroup = _MutuallyExclusiveCallbackGroup
    rclpy_cbg.ReentrantCallbackGroup         = _ReentrantCallbackGroup
    rclpy.callback_groups = rclpy_cbg

    rclpy_time = _mk("rclpy.time")
    rclpy_time.Time = Time
    rclpy_time.Duration = Duration
    rclpy.time = rclpy_time

    rclpy_duration = _mk("rclpy.duration")
    rclpy_duration.Duration = Duration
    rclpy.duration = rclpy_duration

    cv_bridge = _mk("cv_bridge")
    cv_bridge.CvBridge = CvBridge
    cv_bridge.CvBridgeError = CvBridgeError

    # std_msgs
    std = _mk("std_msgs")
    std_msg = _mk("std_msgs.msg")
    std_msg.Header = _Header
    std_msg.String = _String
    std_msg.Bool   = _Bool
    std.msg = std_msg

    # geometry_msgs
    geom = _mk("geometry_msgs")
    geom_msg = _mk("geometry_msgs.msg")
    geom_msg.Point = _Point
    geom_msg.Point32 = _Point32
    geom_msg.Pose = _Pose
    geom_msg.PoseStamped = _PoseStamped
    geom_msg.PointStamped = _PointStamped
    geom_msg.PoseArray = _PoseArray
    geom_msg.Quaternion = _Quat
    geom.msg = geom_msg

    # nav_msgs
    nav = _mk("nav_msgs")
    nav_msg = _mk("nav_msgs.msg")
    nav_msg.Path = _Path
    nav_msg.Odometry = _Odom
    nav.msg = nav_msg

    # ackermann_msgs
    ack = _mk("ackermann_msgs")
    ack_msg = _mk("ackermann_msgs.msg")
    ack_msg.AckermannDriveStamped = _AckermannDriveStamped
    ack_msg.AckermannDrive = _AckermannDrive
    ack.msg = ack_msg

    # visualization_msgs
    viz = _mk("visualization_msgs")
    viz_msg = _mk("visualization_msgs.msg")
    viz_msg.Marker = _Marker
    viz.msg = viz_msg

    # sensor_msgs
    sens = _mk("sensor_msgs")
    sens_msg = _mk("sensor_msgs.msg")
    sens_msg.Image = _Image
    sens_msg.CompressedImage = _CompressedImage
    sens_msg.RegionOfInterest = _RegionOfInterest
    sens.msg = sens_msg

    # vs_msgs (custom — not actually used by detector/follower at runtime,
    # only at module-import time inside homography_transformer.py)
    vs = _mk("vs_msgs")
    vs_msg = _mk("vs_msgs.msg")
    vs_msg.ConeLocation = _ConeLocation
    vs_msg.ConeLocationPixel = _ConeLocationPixel
    vs_msg.ParkingError = _ParkingError
    vs.msg = vs_msg

    # rcl_interfaces
    rcl_i = _mk("rcl_interfaces")
    rcl_i_msg = _mk("rcl_interfaces.msg")
    rcl_i_msg.SetParametersResult = _SetParametersResult
    rcl_i.msg = rcl_i_msg

    # builtin_interfaces (not directly imported by our targets, but harmless)
    bi = _mk("builtin_interfaces")
    bi_msg = _mk("builtin_interfaces.msg")
    bi_msg.Time = _TimeMsg
    bi_msg.Duration = _DurMsg
    bi.msg = bi_msg


# Auto-install on import — harness just imports this module first.
install()

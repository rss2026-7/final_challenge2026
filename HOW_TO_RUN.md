# How to run

Both files live in `final_challenge/` and run as ROS2 nodes. `stoplight_detection.py` also doubles as a standalone HSV calibration GUI.

---

## `yolo_node.py`

Generic YOLO inference. Runs once per camera frame, publishes a `RegionOfInterest` per target class plus an annotated debug image.

```bash
python3 final_challenge/yolo_node.py
# override the camera topic / model / threshold:
python3 final_challenge/yolo_node.py --ros-args \
    -p image_topic:=/my/cam/image_raw \
    -p model:=yolo11n.pt \
    -p conf_threshold:=0.5
```

Target classes are hard-coded: `parking_meter`, `fire_hydrant`, `bird`, `traffic light`.

### Inputs

| Name | Kind | Type | Default | Notes |
|---|---|---|---|---|
| `image_topic` | param | string | `/zed/zed_node/rgb/image_rect_color` | Camera topic to subscribe to |
| `model` | param | string | `yolo11n.pt` | Ultralytics YOLO weights |
| `conf_threshold` | param | double | `0.5` | Min detection confidence |
| *(image_topic)* | sub | `sensor_msgs/Image` | — | Live camera frames |

### Outputs

| Topic | Type | Notes |
|---|---|---|
| `/yolo/parking_meter/roi` | `sensor_msgs/RegionOfInterest` | Highest-conf bbox; `width==0 && height==0` ⇒ not detected this frame |
| `/yolo/fire_hydrant/roi` | `sensor_msgs/RegionOfInterest` | same |
| `/yolo/bird/roi` | `sensor_msgs/RegionOfInterest` | same |
| `/yolo/traffic_light/roi` | `sensor_msgs/RegionOfInterest` | same (note `_` not space) |
| `/yolo/annotated_image` | `sensor_msgs/Image` | Input frame with all surviving boxes drawn |

---

## `stoplight_detection.py`

Red/green stoplight color segmentation. Two modes:

```bash
# ROS2 node (no positional args)
python3 final_challenge/stoplight_detection.py

# Calibration GUI (positional path = image or directory of images)
python3 final_challenge/stoplight_detection.py testing_images/traffic_light/
python3 final_challenge/stoplight_detection.py testing_images/traffic_light/3.jpeg
```

Calibration workflow: click `[Red 1]` → right-click a lit red bulb → click `[Green]` → right-click a lit green bulb → tune `Min_Area` → click `[Print Vals]` and paste the printed lines over the constants at the top of the file.

### Inputs

| Name | Kind | Type | Default | Notes |
|---|---|---|---|---|
| `image_topic` | param | string | `/zed/zed_node/rgb/image_rect_color` | Camera topic |
| `red_low_1`, `red_high_1` | param | int[3] | `[0,120,90]`, `[10,255,255]` | HSV lower band of red |
| `red_low_2`, `red_high_2` | param | int[3] | `[170,120,90]`, `[179,255,255]` | HSV upper band (red wraps the hue seam) |
| `green_low`, `green_high` | param | int[3] | `[40,90,90]`, `[85,255,255]` | HSV band for green |
| `min_area` | param | int | `50` | px² gate — largest contour must exceed this to count |
| *(image_topic)* | sub | `sensor_msgs/Image` | — | Live camera frames |
| *(positional CLI arg)* | arg | path | — | GUI mode only: image file or directory |

### Outputs

| Topic | Type | Notes |
|---|---|---|
| `/stoplight/result` | `std_msgs/String` | `"red"`, `"green"`, or `"none"` per frame |
| `/stoplight/segmented` | `sensor_msgs/Image` | Debug — input with only red+green pixels kept |

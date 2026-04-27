#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation, ConeLocationPixel

# Homography calibration: each PTS_IMAGE_PLANE[i] must correspond to PTS_GROUND_PLANE[i].
# Place an object on the ground at a known position, note its bottom-center pixel [u, v]
# in the camera image AND measure its physical [x, y] from the car.
# Spread points across near/far and left/right for a good fit.

# [u, v] pixel coordinates in the camera image.
# Origin is top-left of image. u increases rightward, v increases downward.
######################################################
PTS_IMAGE_PLANE = [[305, 222],
                   [419, 223],
                   [516, 292],
                   [162, 288]]
######################################################

# [x, y] real-world position in centimeters, relative to the car.
# +x = forward (direction the car faces), +y = left of the car.
# Negative y = to the right of the car.
######################################################
PTS_GROUND_PLANE = [[88.00, 14.00],
                    [84.00, -12.70],
                    [44.00, -19.00],
                    [41.00, 30.00]]
######################################################

METERS_PER_CM = 0.01

# used for lane_detector
def build_homography():
    """Build and return the 3x3 homography matrix from the calibration points above."""
    np_pts_image  = np.float32(np.array(PTS_IMAGE_PLANE)[:, np.newaxis, :])
    np_pts_ground = np.float32((np.array(PTS_GROUND_PLANE) * METERS_PER_CM)[:, np.newaxis, :])
    H, _ = cv2.findHomography(np_pts_image, np_pts_ground)
    return H


def transform_uv_to_xy(H, u, v):
    """
    Transform an image pixel (u, v) to car-frame (x, y) in metres using H.

    u, v : pixel coordinates — origin top-left, u right, v down.
    x    : metres forward from the car.
    y    : metres left of the car.
    """
    p = np.dot(H, np.array([[u], [v], [1.0]]))
    scale = 1.0 / p[2, 0]
    return float(p[0, 0] * scale), float(p[1, 0] * scale)


class HomographyTransformer(Node):
    def __init__(self):
        super().__init__("homography_transformer")

        self.cone_pub = self.create_publisher(ConeLocation, "/relative_cone", 10)
        self.marker_pub = self.create_publisher(Marker, "/cone_marker", 1)
        self.cone_px_sub = self.create_subscription(ConeLocationPixel, "/relative_cone_px", self.cone_detection_callback, 1)

        if not len(PTS_GROUND_PLANE) == len(PTS_IMAGE_PLANE):
            rclpy.logerr("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")

        self.h = build_homography()

        self.get_logger().info("Homography Transformer Initialized")

    def cone_detection_callback(self, msg):
        # Extract information from message
        u = msg.u
        v = msg.v

        # Call to main function
        x, y = self.transformUvToXy(u, v)

        # Publish relative xy position of object in real world
        relative_xy_msg = ConeLocation()
        relative_xy_msg.x_pos = x
        relative_xy_msg.y_pos = y

        self.cone_pub.publish(relative_xy_msg)

    def transformUvToXy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns (x, y) displacement in metres from the camera to the point on
        the ground plane.  x is forward, y is left of the car.
        """
        return transform_uv_to_xy(self.h, u, v)

    def draw_marker(self, cone_x, cone_y, message_frame):
        """
        Publish a marker to represent the cone in rviz.
        (Call this function if you want)
        """
        marker = Marker()
        marker.header.frame_id = message_frame
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = .5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = cone_x
        marker.pose.position.y = cone_y
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    homography_transformer = HomographyTransformer()
    rclpy.spin(homography_transformer)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

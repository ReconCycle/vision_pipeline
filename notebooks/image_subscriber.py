from pathlib import Path
import cv2
from PIL import Image as PILImage
import numpy as np
from tqdm import tqdm
from rich import print

# ros packages
from context_action_framework.types import Detection, Label, Module, Camera, detections_to_ros, detections_to_py
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

from cv_bridge import CvBridge
import rospy


rospy.init_node("test_node")

rate = rospy.Rate(10) # 10hz


def im_handler(img_msg):
    cv2_img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
    print("received response", cv2_img.shape)

# rospy.Subscriber("/basler/image_rect_color", Image, im_handler, queue_size=1)
rospy.Subscriber("/vision/basler/colour", Image, im_handler, queue_size=1)


def im_compressed_handler(img_msg):
    np_arr = np.frombuffer(img_msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # cv2_img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
    print("received compressed basler", image_np.shape)

# rospy.Subscriber("/basler/image_rect_color/compressed", CompressedImage, im_compressed_handler, queue_size=1)


def im_compressed_handler2(img_msg):
    np_arr = np.frombuffer(img_msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # cv2_img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
    print("received compressed realsense", image_np.shape)

# rospy.Subscriber("/realsense/color/image_raw/compressed", CompressedImage, im_compressed_handler2, queue_size=1)


# rospy.spin()
while not rospy.is_shutdown():
    pass
    rate.sleep()

print("end")
import sys
import os
sys.path.append(os.path.expanduser("~/vision-pipeline/"))
import cv2
import numpy as np
import json
import argparse
import time
import atexit
from rich import print
import commentjson
import asyncio

import rospy

from config import load_config
from helpers import str2bool, path

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest
from context_action_framework.types import Camera
from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.msg import VisionDetails
from context_action_framework.types import detections_to_ros, gaps_to_ros, detections_to_py
from sensor_msgs.msg import Image, CameraInfo, CompressedImage


class ROSSubscriberVision():
    def __init__(self) -> None:
        # load config
        
        rospy.init_node("basler_subscribe_test")
        
        self.rate = rospy.Rate(100)
        
        # self.basler_service = rospy.ServiceProxy(path(self.config.node_name, self.config.basler.topic, "enable"), SetBool)
        
        # enable basler
        # self.basler_service(True)
        
        def exit_handler():
            # self.basler_service(False)
            print("stopping basler subscriber and exiting...")
        
        atexit.register(exit_handler)

        img_topic = "vision/basler/colour"
        img_topic2 = "vision/basler/colour"
        print(f"img_topic {img_topic}")

        rospy.Subscriber(img_topic, Image, self.img_colour_callback, queue_size=1)

        rospy.Subscriber(img_topic2, Image, self.img_colour_callback, queue_size=1)
        
        # running the pipeline is blocking
        print("running loop...")
        self.run_loop()

    def run_loop(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
    
    def img_colour_callback(self, msg):
        # detections = detections_to_py(ros_detections.detections)
        # print("[green]detections:")
        # for detection in detections:
        #     print(f"det: {detection.label.name}, {detection.tracking_id}")
        #     # print("tf", detection.tf)


if __name__ == '__main__':
    subscriber =  ROSSubscriberVision()

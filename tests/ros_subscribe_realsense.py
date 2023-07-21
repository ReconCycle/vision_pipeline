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


class ROSRealsenseTest():
    def __init__(self) -> None:
        # load config
        self.config = load_config(filepath=os.path.expanduser("~/vision-pipeline/config.yaml"))
        print("config", self.config)
        
        rospy.init_node("realsense_subscribe_test")
        
        self.rate = rospy.Rate(100)
        
        # self.realsense_service = rospy.ServiceProxy(path(self.config.node_name, self.config.realsense.topic, "enable"), SetBool)
        
        # enable realsense
        # self.realsense_service(True)
        
        def exit_handler():
            # self.realsense_service(False)
            print("stopping vision realsense and exiting...")
        
        atexit.register(exit_handler)

        detections_topic = path(self.config.node_name, self.config.realsense.topic, "detections")
        self.detections_sub = rospy.Subscriber(detections_topic, ROSDetections, self.detections_callback, queue_size=1)
        
        # running the pipeline is blocking
        print("running loop...")
        self.run_loop()

    def run_loop(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
    
    def detections_callback(self, ros_detections):
        detections = detections_to_py(ros_detections.detections)
        print("[green]detections:")
        for detection in detections:
            print(f"det: {detection.label.name}, {detection.tracking_id}")
            print("tf", detection.tf)


if __name__ == '__main__':
    ros_pipeline =  ROSRealsenseTest()

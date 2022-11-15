import sys
import os
sys.path.append(os.path.dirname("/root/vision-pipeline/"))
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
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from ros_vision_pipeline.msg import ColourDepth
from cv_bridge import CvBridge
import message_filters
from camera_control_msgs.srv import SetSleeping

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from config import load_config
from pipeline_basler import BaslerPipeline
from pipeline_realsense import RealsensePipeline
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
        self.config = load_config(filepath="../config.yaml")
        print("config", self.config)
        
        rospy.init_node("realsense_subscribe_test")
        
        self.rate = rospy.Rate(100)
        
        self.realsense_service = rospy.ServiceProxy(path(self.config.node_name, self.config.realsense.topic, "enable"), SetBool)
        
        # enable realsense
        self.realsense_service(True)
        
        def exit_handler():
            self.realsense_service(False)
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
        for detection in detections:
            print("ros_detection label", detection.label)
            print("ros_detection tf", detection.tf)


if __name__ == '__main__':
    ros_pipeline =  ROSRealsenseTest()

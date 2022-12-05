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
import signal

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
# from gap_detection.nn_gap_detector import NNGapDetector
from helpers import str2bool, path

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest
from context_action_framework.types import Camera
from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.msg import VisionDetails
from context_action_framework.types import detections_to_ros, gaps_to_ros



class ROSPipeline():
    def __init__(self) -> None:
        # load config
        self.config = load_config()
        print("config", self.config)
        
        rospy.init_node(self.config.node_name)
        
        self.rate = rospy.Rate(100)
        
        self.camera_acquisition_stamp = None
        self.img_msg = None
        self.img_id = 0
        
        self.t_camera_service_called = -1 # time camera_service was last called
        
        self.camera_service = rospy.ServiceProxy(path(self.config.realsense.camera_node, "enable"), SetBool)
        
        # set enable to True
        self.enable_camera(True)
        
        # handle ctrl + c
        def sigint_handler(sig, frame):
            print("called end application...")
            self.enable_camera(False)
            sys.exit()

        signal.signal(signal.SIGINT, sigint_handler)
        
        
        img_topic = path(self.config.realsense.camera_node, self.config.realsense.image_topic)
        print("img_topic", img_topic)
        
        self.img_sub = rospy.Subscriber(img_topic, Image, self.img_from_camera_callback, queue_size=1)
        
        # running the pipeline is blocking
        print("running loop...")
        self.run_loop()

    def enable_camera(self, state):
        # only allow enabling/disabling camera every 2 seconds. To avoid camera breaking.
        rate_limit = 2
        t_now = time.time()
        if t_now - self.t_camera_service_called > rate_limit:
            self.t_camera_service_called = t_now
            try:
                res = self.camera_service(state)
                if state:
                    print("realsense: enabled camera:", res.success)
                else:
                    print("realsense: disabled camera:", res.success)
            except rospy.ServiceException as e:
                print("[red]realsense: Service call failed (state " + str(state) + "):[/red]", e)
        else:
            print("[red]realsense: Camera enable/disable rate limited to " + str(rate_limit) + " seconds.")

    def run_loop(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            # self.pipeline_realsense.run()
    
    def img_from_camera_callback(self, img_msg):
        new_camera_acquisition_stamp = img_msg.header.stamp
        
        if self.camera_acquisition_stamp:
            diff = new_camera_acquisition_stamp.to_sec() - self.camera_acquisition_stamp.to_sec()
            if diff > 0.0000000001:
                fps = 1.0/diff
                print("fps", fps)
        
        self.camera_acquisition_stamp = img_msg.header.stamp
        self.img_msg = img_msg
        self.img_id += 1
        
        t = rospy.get_rostime()
        time_delay = t.to_sec() - self.camera_acquisition_stamp.to_sec()
        
        print("realsense delay:", time_delay)
        


if __name__ == '__main__':
    ros_pipeline =  ROSPipeline()

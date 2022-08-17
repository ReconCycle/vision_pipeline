import sys
import os
import cv2
import numpy as np
import json
import argparse
import time
import atexit
from rich import print

import rospy
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from ros_vision_pipeline.msg import ColourDepth
from cv_bridge import CvBridge
import message_filters

from pipeline_basler import BaslerPipeline
from pipeline_realsense import RealsensePipeline
from gap_detection.nn_gap_detector import NNGapDetector
import helpers


class ROSPipeline():
    def __init__(self, args) -> None:
        args = self.arg_parser(args)

        rospy.init_node(args.node_name)

        if args.camera_type == "basler":
            self.pipeline = BaslerPipeline(args.camera_topic, args.node_name)
        elif args.camera_type == "realsense":
            self.pipeline = RealsensePipeline(args.camera_topic, args.node_name)
        # elif args.camera_type == "realsense_nn":
        #     self.pipeline = NNGapDetector()
        
        rospy.Service("/" + args.node_name + "/enable", SetBool, self.enable_vision_callback)
        
        if args.auto_start:
            self.pipeline.enable(True)

        def exit_handler():
            self.pipeline.enable(False)
            print("stopping pipeline and exiting...")
        
        atexit.register(exit_handler)
        
        # running the pipeline is blocking
        self.pipeline.run()

    def enable_vision_callback(self, req):
        state = req.data
        if state:
            print("starting pipeline...")
            self.pipeline.enable(True)
            msg = args.node_name + " started."
        else:
            print("stopping pipeline...")
            self.pipeline.enable(False)
            msg = args.node_name + " stopped."
        
        return True, msg

    @staticmethod
    def arg_parser(args):
        # set the camera_topic to realsense as well, if not set manually
        if args.camera_type.startswith("realsense") and args.camera_topic == "basler":
            args.camera_topic = "realsense"

        if args.camera_type.startswith("realsense") and args.node_name == "vision_basler":
            args.node_name = "vision_realsense"

        print("\ncamera_type:", args.camera_type)
        print("camera_topic:", args.camera_topic)
        print("node_name:", args.node_name)
        print("auto_start:", args.auto_start, "\n")

        return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_type", help="Which camera: basler/realsense", nargs='?', type=str, default="basler")
    parser.add_argument("--auto_start", help="Publish continuously otherwise create service.", nargs='?', type=helpers.str2bool, default=False)
    parser.add_argument("--camera_topic", help="The name of the camera topic to subscribe to", nargs='?', type=str, default="basler")
    parser.add_argument("--node_name", help="The name of the node", nargs='?', type=str, default="vision_basler")
    args = parser.parse_args()

    ros_pipeline =  ROSPipeline(args)

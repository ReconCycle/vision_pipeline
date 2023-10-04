import sys
import os
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
from geometry_msgs.msg import PoseStamped, PointStamped, Pose, TransformStamped
from ros_vision_pipeline.msg import ColourDepth
from cv_bridge import CvBridge
import message_filters
import tf2_ros
import tf

from object_detection_model import ObjectDetectionModel
from object_reid_sift import ObjectReIdSift
from object_reid_superglue import ObjectReIdSuperGlue
from config import load_config
from pipeline_basler import PipelineBasler
from pipeline_realsense import PipelineRealsense
# from gap_detection.nn_gap_detector import NNGapDetector
from helpers import str2bool, path
from static_transform_manager import StaticTransformManager

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

        rospy.init_node(self.config.node_name)
        
        self.static_tf_manager = StaticTransformManager()
        
        # load object detection model
        model = ObjectDetectionModel(self.config.obj_detection)
        
        # load object reid
        object_reid = None
        if self.config.reid:
            object_reid = ObjectReIdSuperGlue(self.config, model)

        self.pipeline_basler = PipelineBasler(model, object_reid, self.config, self.static_tf_manager)
        self.pipeline_realsense = PipelineRealsense(model, object_reid, self.config, self.static_tf_manager)
        
        def exit_handler():
            time.sleep(2) # sleep such that on restart, the cameras are not immediately re-enabled
            print("stopping pipeline and exiting...")
        
        atexit.register(exit_handler)
        
        print("running camera pipelines...\n")
        self.run_loop()
        

    def run_loop(self):
        while not rospy.is_shutdown():
            
            self.pipeline_basler.run()
            self.pipeline_realsense.run()
            
            # sleep is done within these two separate pipelines.


if __name__ == '__main__':
    ros_pipeline =  ROSPipeline()

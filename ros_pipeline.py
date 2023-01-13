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
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from ros_vision_pipeline.msg import ColourDepth
from cv_bridge import CvBridge
import message_filters

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from object_reid import ObjectReId
from config import load_config
from pipeline_basler import PipelineBasler
from pipeline_realsense import PipelineRealsense
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
        
        # load yolact
        yolact, dataset = self.load_yolact(self.config.obj_detection)
        
        # load object reid
        object_reid = None
        if self.config.reid:
            object_reid = ObjectReId()

        self.pipeline_basler = PipelineBasler(yolact, dataset, object_reid, self.config)
        self.pipeline_realsense = PipelineRealsense(yolact, dataset, object_reid, self.config)

        # ! FIX THIS AGAIN
        # if self.config.realsense.run_continuous:
        #     self.pipeline_realsense.enable(True) # TODO
        # if self.config.basler.run_continuous:
        #     self.pipeline_basler.enable_continuous(True)
        
        def exit_handler():
            print("stopping pipeline and exiting...")
        
        atexit.register(exit_handler)
        
        print("running loop...")
        self.run_loop()
        
    
    def load_yolact(self, yolact_config):
        yolact_dataset = None
        
        if os.path.isfile(yolact_config.yolact_dataset_file):
            print("loading", yolact_config.yolact_dataset_file)
            with open(yolact_config.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
        else:
            raise Exception("config.yolact_dataset_file is incorrect: " +  str(yolact_config.yolact_dataset_file))
                
        dataset = Config(yolact_dataset)
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': dataset,
            'num_classes': len(dataset.class_names) + 1,

            # Image Size
            'max_size': 1100,

            # These are in BGR and are for ImageNet
            'MEANS': (103.94, 116.78, 123.68),
            'STD': (57.38, 57.12, 58.40),
            
            # the save path should contain resnet101_reducedfc.pth
            'save_path': './data_limited/yolact/',
            'score_threshold': yolact_config.yolact_score_threshold,
            'top_k': len(dataset.class_names)
        }
        
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(yolact_config.yolact_dataset_file), yolact_dataset["model"])
            
        print("model_path", model_path)
        
        yolact = Yolact(config_override)
        yolact.cfg.print()
        yolact.eval()
        yolact.load_weights(model_path)
        
        return yolact, dataset

    def run_loop(self):
        while not rospy.is_shutdown():
            
            self.pipeline_basler.run()
            # self.pipeline_realsense.run() #! disabled while testing
            
            # Now the sleeping is done within these two separate pipelines.

if __name__ == '__main__':
    ros_pipeline =  ROSPipeline()

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

import rospy
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from ros_vision_pipeline.msg import ColourDepth
from cv_bridge import CvBridge
import message_filters

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from config import load_config
from pipeline_basler import BaslerPipeline
from pipeline_realsense import RealsensePipeline
# from gap_detection.nn_gap_detector import NNGapDetector

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest
from context_action_framework.types import Camera
from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.msg import VisionDetails
from context_action_framework.types import detections_to_ros, gaps_to_ros


class ROSPipeline():
    def __init__(self, args) -> None:
        args = self.arg_parser(args)
        self.node_name = args.node_name

        rospy.init_node(args.node_name)
        self.rate = rospy.Rate(1)
        
        # load config
        self.config = load_config()
        print("config", self.config)
        
        # load yolact
        yolact, dataset = self.load_yolact(self.config.obj_detection)

        self.pipeline_basler = BaslerPipeline(yolact, dataset, "basler", args.node_name + "/basler")
        self.pipeline_realsense = RealsensePipeline(yolact, dataset, "realsense", args.node_name + "/realsense")
        
        rospy.Service("/" + args.node_name + "/basler/enable", SetBool, self.enable_basler_callback)
        rospy.Service("/" + args.node_name + "/realsense/enable", SetBool, self.enable_realsense_callback)
        
        if "realsense" in args.auto_start:
            self.pipeline_realsense.enable(True)
        if "basler" in args.auto_start:
            self.pipeline_basler.enable(True)

        def exit_handler():
            self.pipeline_realsense.enable(False)
            self.pipeline_basler.enable(False)
            print("stopping pipeline and exiting...")
        
        atexit.register(exit_handler)
        
        # register vision server
        print("creating vision service server...")
        self.vision_service_server()
        
        # running the pipeline is blocking
        print("running loop...")
        self.run_loop()
        
    def load_yolact(self, config):
        yolact_dataset = None
        
        if config is None:
            config = load_config().obj_detection
        
        if os.path.isfile(config.yolact_dataset_file):
            print("loading", config.yolact_dataset_file)
            with open(config.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
        else:
            raise Exception("config.yolact_dataset_file is incorrect: " +  str(config.yolact_dataset_file))
                
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
            'score_threshold': config.yolact_score_threshold,
            'top_k': len(dataset.class_names)
        }
        
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(config.yolact_dataset_file), yolact_dataset["model"])
            
        print("model_path", model_path)
        
        yolact = Yolact(config_override)
        yolact.cfg.print()
        yolact.eval()
        yolact.load_weights(model_path)
        
        return yolact, dataset

    def run_loop(self):
        while not rospy.is_shutdown():
            
            self.pipeline_basler.run()
            self.pipeline_realsense.run()
            
            self.rate.sleep()

    def enable_basler_callback(self, req):
        state = req.data
        if state:
            print("starting pipeline...")
            self.pipeline_basler.enable(True)
            msg = args.node_name + " started."
        else:
            print("stopping pipeline...")
            self.pipeline_basler.enable(False)
            msg = args.node_name + " stopped."
        
        return True, msg
    
    def enable_realsense_callback(self, req):
        state = req.data
        if state:
            print("starting pipeline...")
            self.pipeline_realsense.enable(True)
            msg = args.node_name + " started."
        else:
            print("stopping pipeline...")
            self.pipeline_realsense.enable(False)
            msg = args.node_name + " stopped."
        
        return True, msg
    

    def vision_det_callback(self, req):
        print("vision_gap_det_callback", req.camera, )
        if req.camera == Camera.basler:
            
            print("enabling basler...")
            self.pipeline_basler.enable(True)
            
            print("getting basler detection")
            labelled_img, detections = self.pipeline_basler.get_stable_detection()
            
            # todo: get detections
            # todo: wait until movement/results are stable
            
            print("disabling basler...")
            self.pipeline_basler.enable(False)
            
            print("returning detection")
            
            if detections is not None:
                vision_details = VisionDetails(False, detections_to_ros(detections), [])
                return VisionDetectionResponse(True, vision_details)
            else:
                return VisionDetectionResponse(False, VisionDetails())
            
            
        elif req.camera == Camera.realsense:
            
            print("enabling realsense...")
            self.pipeline_realsense.enable(True)
            
            # todo: get detections
            # todo: wait until movement/results are stable
            print("getting realsense detection")
            labelled_img, detections, gaps = self.pipeline_realsense.get_stable_detection(gap_detection=req.gap_detection)
            
            print("disabling realsense...")
            self.pipeline_realsense.enable(False)

            if detections is not None:
                vision_details = VisionDetails(req.gap_detection, detections_to_ros(detections), gaps_to_ros(gaps))
                return VisionDetectionResponse(True, vision_details)
            else:
                return VisionDetectionResponse(False, VisionDetails())

    def vision_service_server(self):
        rospy.Service("/" + self.node_name + "/vision_get_detection", VisionDetection, self.vision_det_callback)

    @staticmethod
    def arg_parser(args):
        print("node_name:", args.node_name)
        print("auto_start:", args.auto_start, "\n")

        return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_start", help="Which camera: basler/realsense", nargs='?', type=str, default="")
    parser.add_argument("--node_name", help="The name of the node", nargs='?', type=str, default="vision")
    args = parser.parse_args()

    ros_pipeline =  ROSPipeline(args)

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

        self.pipeline_basler = BaslerPipeline(yolact, dataset, self.config)
        self.pipeline_realsense = RealsensePipeline(yolact, dataset, self.config)
        
        basler_topic_enable = path(self.config.node_name, self.config.basler.topic, "enable")
        realsense_topic_enable = path(self.config.node_name, self.config.realsense.topic, "enable")
        
        rospy.Service(basler_topic_enable, SetBool, self.enable_basler_callback)
        rospy.Service(realsense_topic_enable, SetBool, self.enable_realsense_callback)
        
        if self.config.realsense.run_continuous:
            self.pipeline_realsense.enable(True)
        if self.config.basler.run_continuous:
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
            self.pipeline_realsense.run()
            
            # Now the sleeping is done within these two separate pipelines. We might want, for example, a higher FPS from realsense.

    def enable_basler_callback(self, req):
        state = req.data
        if self.config.basler.run_continuous:
            msg = "basler: won't start/stop. Running in continuous mode."
            print(msg)
            return True, msg
        
        if state:
            print("basler: starting pipeline...")
            self.pipeline_basler.enable(True)
            msg = self.config.node_name + " started."
        else:
            print("basler: stopping pipeline...")
            self.pipeline_basler.enable(False)
            msg = self.config.node_name + " stopped."
        
        return True, msg
    
    def enable_realsense_callback(self, req):
        state = req.data
        if self.config.realsense.run_continuous:
            msg = "basler: won't start/stop. Running in continuous mode."
            print(msg)
            return True, msg
        
        if state:
            print("realsense: starting pipeline...")
            self.pipeline_realsense.enable(True)
            msg = self.config.node_name + " started."
        else:
            print("realsense: stopping pipeline...")
            self.pipeline_realsense.enable(False)
            msg = self.config.node_name + " stopped."
        
        return True, msg
    

    def vision_det_callback(self, req):
        print("vision_gap_det_callback", Camera(req.camera))
        if req.camera == Camera.basler:
            
            print("basler: enabling...")
            self.pipeline_basler.enable(True)
            
            print("basler: getting detection")
            camera_acq_stamp, img, detections, img_id = self.pipeline_basler.get_stable_detection()
            print("[blue]basler:: received stable detection, img_id:" + str(img_id) + "[/blue]")
            
            # todo: get detections
            # todo: wait until movement/results are stable
            
            print("basler: disabling...")
            self.pipeline_basler.enable(False)
            
            print("basler: returning detection")
            
            if detections is not None:
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                vision_details = VisionDetails(header, camera_acq_stamp, False, detections_to_ros(detections), [])
                return VisionDetectionResponse(True, vision_details, CvBridge().cv2_to_imgmsg(img))
            else:
                print("basler: returning empty response!")
                return VisionDetectionResponse(False, VisionDetails(), CvBridge().cv2_to_imgmsg(img))
            
            
        elif req.camera == Camera.realsense:
            
            print("realsense: enabling...")
            self.pipeline_realsense.enable(True)
            
            # todo: get detections
            # todo: wait until movement/results are stable
            print("realsense: getting detection")
            camera_acq_stamp, img, detections, gaps, img_id = self.pipeline_realsense.get_stable_detection(gap_detection=req.gap_detection)
            print("[blue]realsense: received stable detection, img_id:" + str(img_id) + "[/blue]")
            
            print("realsense: disabling...")
            self.pipeline_realsense.enable(False)

            if detections is not None:
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                vision_details = VisionDetails(header, camera_acq_stamp, req.gap_detection, detections_to_ros(detections), gaps_to_ros(gaps))
                return VisionDetectionResponse(True, vision_details, CvBridge().cv2_to_imgmsg(img))
            else:
                print("realsense: returning empty response!")
                return VisionDetectionResponse(False, VisionDetails(), CvBridge().cv2_to_imgmsg(img))

    def vision_service_server(self):
        rospy.Service(path(self.config.node_name, "vision_get_detection"), VisionDetection, self.vision_det_callback)


if __name__ == '__main__':
    ros_pipeline =  ROSPipeline()

from argparse import Action
import os
import numpy as np
import time
import cv2
from rich import print
import json
from io import BytesIO, StringIO

import rospy
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from std_msgs.msg import String
import message_filters

from rospy_message_converter import message_converter

from context_action_framework.types import ros_to_str, str_to_ros
from context_action_framework.types import Action, Detection, Gap, Label, Module, Robot, EndEffector, Camera
from context_action_framework.srv import NextAction, NextActionRequest, VisionDetection
from context_action_framework.msg import CutBlock, LeverBlock, MoveBlock, PushBlock, TurnOverBlock, ViceBlock, VisionBlock, \
    CutDetails, LeverDetails, MoveDetails, PushDetails, TurnOverDetails, ViceDetails, VisionDetails


class ROSGetServiceTest():
    def __init__(self) -> None:

        print("starting ROS Test for Vision Pipeline")
        
        rospy.init_node('vision_pipeliine_test')
        self.rate = rospy.Rate(1)
        
        self.vision_topic = "vision/basler"
        
        self.create_service_clients()
        self.run_test()
        
            
    def create_service_clients(self):
        print("waiting for vision_get_detection ...")
        rospy.wait_for_service("/" + self.vision_topic + "/vision_get_detection")
        print("vision module online")
        self.detection_service = rospy.ServiceProxy("/" + self.vision_topic + "/vision_get_detection", VisionDetection)
    
    def run_test(self):
        
        if not rospy.is_shutdown():
            # get detection
            det_response = self.detection_service(Camera.realsense, True)
            # det_response = self.detection_service(Camera.basler, False)
            
            # provide the details for next request
            success = det_response.success
            vision_details = det_response.vision_details # type VisionDetails
            
            print("vision_details", vision_details)
            
            print("vision_details.header.stamp", vision_details.header.stamp)
            print("camera_acquisition_stamp", vision_details.camera_acquisition_stamp)
        
            print("vision delay: ", (vision_details.header.stamp - vision_details.camera_acquisition_stamp))
        
        #     self.rate.sleep()

if __name__ == '__main__':
    ros_test = ROSGetServiceTest()

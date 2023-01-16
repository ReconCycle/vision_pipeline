import sys
import numpy as np
import time
from rich import print
import json
import rospy
import tf2_ros
import tf
import copy
import asyncio
from threading import Event

from pipeline_camera import PipelineCamera
from helpers import path, rotate_img
from object_detection import ObjectDetection
from work_surface_detection_opencv import WorkSurfaceDetection
from aruco_detection import ArucoDetection

from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from std_msgs.msg import String
from camera_control_msgs.srv import SetSleeping
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray, TransformStamped

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest
from context_action_framework.msg import VisionDetails
from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.types import detections_to_ros
from context_action_framework.types import Label, Camera

from obb import obb_px_to_quat


class PipelineBasler(PipelineCamera):
    def __init__(self, yolact, dataset, object_reid, config):
        config.basler.enable_topic = "set_sleeping" # basler camera specific
        config.basler.enable_camera_invert = True # enable = True, but the topic is called set_sleeping, so the inverse
        config.use_worksurface_detection = True
        
        super().__init__(yolact, dataset, object_reid, config, config.basler, Camera.basler)
        
    def create_service_client(self):
        super().create_service_client()
        self.camera_service = rospy.ServiceProxy(path(self.camera_config.camera_node, self.camera_config.enable_topic), SetSleeping)

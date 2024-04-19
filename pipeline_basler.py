import sys
import numpy as np
import time
from rich import print
import rospy

from pipeline_camera import PipelineCamera
from helpers import path
from camera_control_msgs.srv import SetSleeping
from context_action_framework.types import Camera, Module


class PipelineBasler(PipelineCamera):
    def __init__(self, model, object_reid, config, static_broadcaster):
        self.camera_config = config.basler
        
        self.camera_config.enable_topic = "set_sleeping" # basler camera specific
        self.camera_config.enable_camera_invert = True # enable = True, but the topic is called set_sleeping, so the inverse
        self.camera_config.use_worksurface_detection = True

        table_name = "vision"
        
        super().__init__(model, object_reid, config, self.camera_config, Camera.basler, static_broadcaster, table_name)
        
    def create_service_client(self):
        super().create_service_client()
        self.camera_service = rospy.ServiceProxy(path(self.camera_config.camera_node, self.camera_config.enable_topic), SetSleeping)

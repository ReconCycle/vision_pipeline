import sys
import numpy as np
import time
from rich import print
import json

from object_detection import ObjectDetection
from work_surface_detection_opencv import WorkSurfaceDetection
from config import load_config

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from std_msgs.msg import String
from camera_control_msgs.srv import SetSleeping

from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.types import detections_to_ros

class BaslerPipeline:
    def __init__(self, yolact, dataset, camera_topic="basler", node_name="vision_basler"):     
        self.rate = rospy.Rate(1) # fps

        # don't automatically start
        self.pipeline_enabled = False

        self.camera_topic = camera_topic
        self.node_name = node_name

        self.img_sub = None

        self.colour_img = None
        self.img_id = 0
        
        self.processed_img_id = -1  # don't keep processing the same image
        self.t_now = None
        
        self.labelled_img = None
        self.detections = None

        print("creating camera subscribers...")
        self.create_camera_subscribers()
        print("creating publishers...")
        self.create_publishers()
        print("creating service client...")
        self.create_service_client()
        print("creating basler pipeline...")
        self.init_basler_pipeline(yolact, dataset)

        print("waiting for pipeline to be enabled...")

    def init_basler_pipeline(self, yolact, dataset):
        self.object_detection = ObjectDetection(yolact, dataset)
        
        self.worksurface_detection = None

    def img_from_camera_callback(self, img):
        colour_img = CvBridge().imgmsg_to_cv2(img)
        self.colour_img = np.array(colour_img)
        self.img_id += 1

    def create_camera_subscribers(self):
        img_topic = "/" + self.camera_topic + "/image_rect_color"
        self.img_sub = rospy.Subscriber(img_topic, Image, self.img_from_camera_callback)

    def create_service_client(self):
        try:
            rospy.wait_for_service("/" + self.camera_topic + "/set_sleeping", 2) # 2 seconds
        except rospy.ROSException as e:
            print("[red]Couldn't find to service![/red]")
    
        self.camera_service = rospy.ServiceProxy("/" + self.camera_topic + "/set_sleeping", SetSleeping)

    def create_publishers(self):
        self.br = CvBridge()
        self.labelled_img_publisher = rospy.Publisher("/" + self.node_name + "/colour", Image, queue_size=20)
        self.detections_publisher = rospy.Publisher("/" + self.node_name + "/detections", ROSDetections, queue_size=20)

    def publish(self, img, detections):       
        print("publishing...")
        ros_detections = ROSDetections(detections_to_ros(detections))
        
        self.labelled_img_publisher.publish(self.br.cv2_to_imgmsg(img))
        self.detections_publisher.publish(ros_detections)

    def enable_camera(self, state):
        # enable = True, but the topic is called set_sleeping, so the inverse
        state = not state
        try:
            res = self.camera_service(state)
            if state:
                print("enabled camera:", res)
            else:
                print("disabled camera:", res)
        except rospy.ServiceException as e:
            print("Service call failed (state " + str(state) + "): ", e)

    def enable(self, state):
        self.enable_camera(state)
        self.pipeline_enabled = state
        if state == False:
            self.labelled_img = None
            self.detections = None

    def get_stable_detection(self):
        # todo: logic to get stable detection
        
        # todo: wait until we get at least one detection
        while self.detections is None:
            print("waiting for detection...")
            time.sleep(1) #! debug
            
        if self.detections is not None:            

            return self.labelled_img, self.detections        

        else:
            print("stable detection failed!")
            return None, None
        
    def run(self):
        if self.pipeline_enabled:
            if self.colour_img is not None and self.processed_img_id < self.img_id:
                print("\n[green]running pipeline basler frame...[/green]")
                self.processed_img_id = self.img_id
                t_prev = self.t_now
                self.t_now = time.time()
                fps = None
                if t_prev is not None and self.t_now - t_prev > 0:
                    fps = "fps_total: " + str(round(1 / (self.t_now - t_prev), 1)) + ", "

                labelled_img, detections = self.process_img(self.colour_img, fps)

                self.publish(labelled_img, detections)
                
                self.labelled_img = labelled_img
                self.detections = detections

                if self.img_id == sys.maxsize:
                    self.img_id = 0
                    self.processed_img_id = -1

            else:
                print("Waiting to receive image (basler).")
                time.sleep(0.1)

    def process_img(self, img, fps=None):
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img)
            # self.worksurface_detection = WorkSurfaceDetection(img, self.config.dlc)
        
        labelled_img, detections = self.object_detection.get_prediction(img, worksurface_detection=self.worksurface_detection, extra_text=fps)

        # json_detections = json.dumps(detections, cls=EnhancedJSONEncoder)
        
        return labelled_img, detections

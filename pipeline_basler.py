import sys
import numpy as np
import time
from rich import print
import json

from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection
from helpers import EnhancedJSONEncoder
from config import load_config

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from std_msgs.msg import String
from camera_control_msgs.srv import SetSleeping


class BaslerPipeline:
    def __init__(self, camera_topic="basler", node_name="vision_basler"):
        self.rate = rospy.Rate(1) # fps

        # don't automatically start
        self.pipeline_enabled = False

        self.camera_topic = camera_topic
        self.node_name = node_name

        self.img_sub = None

        self.colour_img = None
        self.img_id = 0

        print("creating camera subscribers...")
        self.create_camera_subscribers()
        print("creating publishers...")
        self.create_publishers()
        print("creating service client...")
        self.create_service_client()
        print("creating basler pipeline...")
        self.init_basler_pipeline()

        print("waiting for pipeline to be enabled...")


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
        self.detections_publisher = rospy.Publisher("/" + self.node_name + "/detections", String, queue_size=20)

    def publish(self, img, detections):
        self.labelled_img_publisher.publish(self.br.cv2_to_imgmsg(img))
        self.detections_publisher.publish(String(detections))

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
            print("Service call failed: ", e)

    def enable(self, state):
        self.enable_camera(state)
        self.pipeline_enabled = state

    def init_basler_pipeline(self):

        self.config = load_config()
        print("config", self.config)
        
        # 1. work surface coordinates, will be initialised on first received image
        self.worksurface_detection = None

        # 2. object detection
        self.object_detection = ObjectDetection(self.config.obj_detection)
        self.labels = self.object_detection.labels

    def run(self):
        processed_img_id = -1  # don't keep processing the same image
        t_now = None
        t_prev = None
        fps = None
        while not rospy.is_shutdown():
            if self.pipeline_enabled:
                if self.colour_img is not None and processed_img_id < self.img_id:
                    processed_img_id = self.img_id
                    t_prev = t_now
                    t_now = time.time()
                    if t_prev is not None and t_now - t_prev > 0:
                        fps = "fps_total: " + str(round(1 / (t_now - t_prev), 1)) + ", "

                    labelled_img, detections, json_detections = self.process_img(self.colour_img, fps)
                    print("json_detections", json_detections)

                    self.publish(labelled_img, json_detections)

                else:
                    print("Waiting to receive image.")
                    time.sleep(0.1)

                if self.img_id == sys.maxsize:
                    self.img_id = 0
                    processed_img_id = -1
            
            self.rate.sleep()

    def process_img(self, img, fps=None):
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img)
            # self.worksurface_detection = WorkSurfaceDetection(img, self.config.dlc)
        
        labelled_img, detections = self.object_detection.get_prediction(img, self.worksurface_detection, extra_text=fps)

        json_detections = json.dumps(detections, cls=EnhancedJSONEncoder)
        
        return labelled_img, detections, json_detections

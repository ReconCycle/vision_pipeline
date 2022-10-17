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
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray, TransformStamped
import tf2_ros
import tf

from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.types import detections_to_ros

class BaslerPipeline:
    def __init__(self, yolact, dataset, camera_topic="basler", node_name="vision_basler", image_topic="image_rect_color", wait_for_services=True):     
        self.rate = rospy.Rate(3) # fps

        # don't automatically start
        self.pipeline_enabled = False

        self.camera_topic = camera_topic
        self.node_name = node_name
        self.image_topic = image_topic # will subscribe to camera_topic / image_topic
        self.wait_for_services = wait_for_services

        self.img_sub = None

        self.colour_img = None
        self.img_id = 0
        
        self.processed_img_id = -1  # don't keep processing the same image
        self.t_now = None
        
        self.labelled_img = None
        self.detections = None
        self.markers = None
        self.poses = None
        
        self.frame_id = "vision_module"
        
        self.create_static_tf(self.frame_id)

        print("creating camera subscribers...")
        self.create_camera_subscribers()
        print("creating publishers...")
        self.create_publishers()
        print("creating service client...")
        self.create_service_client()
        print("creating basler pipeline...")
        self.init_basler_pipeline(yolact, dataset)

        print("waiting for pipeline to be enabled...")

        # Checking the rosparam server for whether to publish labeled imgs etc.
        self.publish_labeled_img = False
        self.publish_labeled_rosparamname = '/vision/basler/publish_labeled_img'
        self.last_rosparam_check_time = time.time() # Keeping track of when we last polled the rosparam server
        self.rosparam_check_dt_seconds = 1 # Check rosparam server every 1 second for changes.
        try:
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
        except:
            rospy.set_param(self.publish_labeled_rosparamname, False)
    
    def init_basler_pipeline(self, yolact, dataset):
        self.object_detection = ObjectDetection(yolact, dataset, self.frame_id)
        
        self.worksurface_detection = None

    def check_rosparam_server(self):
        """ Check the rosparam server for whether we want to publish labeled imgs, IF enough time has elapsed between now and last check. """
        cur_t = time.time()
        if cur_t - self.last_rosparam_check_time > self.rosparam_check_dt_seconds:
            self.last_rosparam_check_time = cur_t
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
    
    def img_from_camera_callback(self, img):
        colour_img = CvBridge().imgmsg_to_cv2(img)
        self.colour_img = np.array(colour_img)
        self.img_id += 1

    def create_camera_subscribers(self):
        img_topic = "/" + self.camera_topic + "/" + self.image_topic
        self.img_sub = rospy.Subscriber(img_topic, Image, self.img_from_camera_callback)

    def create_service_client(self):
        timeout = 2 # 2 second timeout
        if self.wait_for_services:
            timeout = None
        try:
            print("waiting for service: /" + self.camera_topic + "/set_sleeping ...")
            rospy.wait_for_service("/" + self.camera_topic + "/set_sleeping", timeout)
        except rospy.ROSException as e:
            print("[red]Couldn't find to service! /" + self.camera_topic + "/set_sleeping[/red]")
    
        self.camera_service = rospy.ServiceProxy("/" + self.camera_topic + "/set_sleeping", SetSleeping)

    def create_publishers(self):
        self.br = CvBridge()
        self.labelled_img_pub = rospy.Publisher("/" + self.node_name + "/colour", Image, queue_size=2)
        self.detections_pub = rospy.Publisher("/" + self.node_name + "/detections", ROSDetections, queue_size=2)
        self.markers_pub = rospy.Publisher("/" + self.node_name + "/markers", MarkerArray, queue_size=2)
        self.poses_pub = rospy.Publisher("/" + self.node_name + "/poses", PoseArray, queue_size=2)

    def publish(self, img, detections, markers, poses):       
        print("publishing...")
        
        #self.labelled_img_pub.publish(self.br.cv2_to_imgmsg(img))
        timestamp = rospy.Time.now()
        header = rospy.Header()
        header.stamp = timestamp
        ros_detections = ROSDetections(header, detections_to_ros(detections))
        
        img_msg = self.br.cv2_to_imgmsg(img)
        img_msg.header.stamp = timestamp
        if self.publish_labeled_img:
            self.labelled_img_pub.publish(img_msg)
            # self.labelled_img_pub.publish(self.br.cv2_to_imgmsg(img))
            
        self.detections_pub.publish(ros_detections)
        
        for marker in markers.markers:
            marker.header.stamp = timestamp    
        self.markers_pub.publish(markers)
        
        poses.header.stamp = timestamp
        self.poses_pub.publish(poses)



    def enable_camera(self, state):
        # enable = True, but the topic is called set_sleeping, so the inverse
        state = not state
        try:
            res = self.camera_service(state)
            if state:
                print("enabled basler camera:", res.success)
            else:
                print("disabled basler camera:", res.success)
        except rospy.ServiceException as e:
            print("[red]Service call failed (state " + str(state) + "):[/red]", e)

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
            print("waiting for detection (basler)...")
            time.sleep(1) #! debug
        
        if self.detections is not None:
            return self.colour_img, self.detections        

        else:
            print("stable detection failed!")
            return None, None
        
    def run(self):
        if self.pipeline_enabled:
            if self.colour_img is not None and self.processed_img_id < self.img_id:
                self.check_rosparam_server() # Check rosparam server for whether to publish labeled imgs
               
                print("\n[green]running pipeline basler frame...[/green]")
                self.processed_img_id = self.img_id
                t_prev = self.t_now
                self.t_now = time.time()
                fps = None
                if t_prev is not None and self.t_now - t_prev > 0:
                    fps = "fps_total: " + str(round(1 / (self.t_now - t_prev), 1)) + ", "

                labelled_img, detections, markers, poses = self.process_img(self.colour_img, fps)

                # recheck if pipeline is enabled
                if self.pipeline_enabled:
                    self.publish(labelled_img, detections, markers, poses)
                    
                    self.labelled_img = labelled_img
                    self.detections = detections
                    self.markers = markers
                    self.poses = poses

                    if self.img_id == sys.maxsize:
                        self.img_id = 0
                        self.processed_img_id = -1

            else:
                print("Waiting to receive image (basler).")


    def process_img(self, img, fps=None):
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img)
            # self.worksurface_detection = WorkSurfaceDetection(img, self.config.dlc)
        
        labelled_img, detections, markers, poses = self.object_detection.get_prediction(img, worksurface_detection=self.worksurface_detection, extra_text=fps)

        # json_detections = json.dumps(detections, cls=EnhancedJSONEncoder)
        
        return labelled_img, detections, markers, poses

    def create_static_tf(self, frame_id):
        pass
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = frame_id

        static_transformStamped.transform.translation.x = float(0)
        static_transformStamped.transform.translation.y = float(0)
        static_transformStamped.transform.translation.z = float(0)

        quat = tf.transformations.quaternion_from_euler(float(0),float(0),float(0))
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        broadcaster.sendTransform(static_transformStamped)

import sys
import numpy as np
import time
from rich import print
import json

from gap_detection.gap_detector_clustering import GapDetectorClustering
from object_detection import ObjectDetection
from config import load_config
from helpers import EnhancedJSONEncoder

import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from std_msgs.msg import String
from camera_control_msgs.srv import SetSleeping
import message_filters


class RealsensePipeline:
    def __init__(self, camera_topic="realsense", node_name="vision_realsense"):
        self.rate = rospy.Rate(1)

        # don't automatically start
        self.pipeline_enabled = False

        self.camera_topic = camera_topic
        self.node_name = node_name

        # realsense data
        self.colour_img = None
        self.depth_img = None
        self.camera_info = None
        self.img_id = 0

        # aruco data
        self.aruco_pose = None
        self.aruco_point = None

        print("creating camera subscribers...")
        self.create_camera_subscribers()
        print("creating publishers...")
        self.create_publishers()
        print("creating service client...")
        self.create_service_client()
        print("creating realsense pipeline...")
        self.init_realsense_pipeline()

        print("waiting for pipeline to be enabled...")

    
    def img_from_camera_callback(self, camera_info, img_msg, depth_msg):
        colour_img = CvBridge().imgmsg_to_cv2(img_msg)
        self.colour_img = np.array(colour_img)
        self.depth_img = CvBridge().imgmsg_to_cv2(depth_msg)
        self.camera_info = camera_info
        self.img_id += 1

    def aruco_callback(self, aruco_pose_ros, aruco_point_ros):
        self.aruco_pose = aruco_pose_ros.pose
        self.aruco_point = aruco_point_ros

    def create_camera_subscribers(self):
        camera_info_topic = "/" + self.camera_topic + "/color/camera_info"
        img_topic = "/" + self.camera_topic + "/color/image_raw"
        depth_topic = "/" + self.camera_topic + "/aligned_depth_to_color/image_raw"
        camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
        img_sub = message_filters.Subscriber(img_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)

        ts = message_filters.ApproximateTimeSynchronizer([camera_info_sub, img_sub, depth_sub], 10, slop=0.01, allow_headerless=False)
        ts.registerCallback(self.img_from_camera_callback)

        # subscribe to aruco
        aruco_sub = message_filters.Subscriber("/realsense_aruco/pose", PoseStamped)
        aruco_pixel_sub = message_filters.Subscriber("/realsense_aruco/pixel", PointStamped)

        # we might not always see the aruco markers, so subscribe to them separately
        ts2 = message_filters.ApproximateTimeSynchronizer([aruco_sub, aruco_pixel_sub], 10, slop=0.01, allow_headerless=False)
        ts2.registerCallback(self.aruco_callback)

    def create_service_client(self):
        try:
            rospy.wait_for_service("/" + self.camera_topic + "/enable", 2) # 2 seconds
        except rospy.ROSException as e:
            print("Couldn't find to service!")
        self.camera_service = rospy.ServiceProxy("/" + self.camera_topic + "/enable", SetBool)

    def create_publishers(self):
        self.br = CvBridge()
        self.labelled_img_publisher = rospy.Publisher("/" + self.node_name + "/colour", Image, queue_size=20)
        self.detections_publisher = rospy.Publisher("/" + self.node_name + "/detections", String, queue_size=20)

        self.clustered_img_pub = rospy.Publisher("/" + self.node_name + "/cluster", Image, queue_size=20)
        self.mask_img_pub = rospy.Publisher("/" + self.node_name + "/mask", Image, queue_size=20)
        self.depth_img_pub = rospy.Publisher("/" + self.node_name + "/depth", Image, queue_size=20)
        self.lever_pose_pub = rospy.Publisher("/" + self.node_name + "/lever", PoseStamped, queue_size=1)
        self.lever_actions_pub = rospy.Publisher("/" + self.node_name + "/lever_actions", String, queue_size=1)


    def publish(self, img, json_detections, lever_actions, cluster_img, depth_scaled, device_mask):
        self.labelled_img_publisher.publish(self.br.cv2_to_imgmsg(img))
        self.detections_publisher.publish(String(json_detections))

        # todo: publish all with the same timestamp
        if cluster_img is not None:
            self.clustered_img_pub.publish(self.br.cv2_to_imgmsg(cluster_img))
        if device_mask is not None:
            self.mask_img_pub.publish(self.br.cv2_to_imgmsg(device_mask))
        if depth_scaled is not None:
            self.depth_img_pub.publish(self.br.cv2_to_imgmsg(depth_scaled))

        # publish only the most probable lever action, for now
        # we could use pose_stamped array instead to publish all the lever possibilities
        if lever_actions is not None:
            self.lever_pose_pub.publish(lever_actions[0].pose_stamped)
            self.lever_actions_pub.publish(lever_actions)

    def enable_camera(self, state):
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

    def init_realsense_pipeline(self):
        self.config = load_config()
        print("config", self.config)

        self.object_detection = ObjectDetection(self.config.obj_detection)
        self.labels = self.object_detection.labels
        self.gap_detector = GapDetectorClustering()

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

                    labelled_img, detections, lever_actions, cluster_img, depth_scaled, device_mask \
                        = self.process_img(fps=fps)

                    json_detections = json.dumps(detections, cls=EnhancedJSONEncoder)

                    self.publish(labelled_img, json_detections, lever_actions, cluster_img, depth_scaled, device_mask)

                else:
                    print("Waiting to receive image.")
                    time.sleep(0.1)

                if self.img_id == sys.maxsize:
                    self.img_id = 0
                    processed_img_id = -1
            
            self.rate.sleep()

    def process_img(self, fps=None):
        print("running pipeline realsense frame...")

        # 2. apply yolact to image and get hca_back
        labelled_img, detections = self.object_detection.get_prediction(self.colour_img, extra_text=fps)

        # 3. apply mask to depth image and convert to pointcloud
        lever_actions, cluster_img, depth_scaled, device_mask \
            = self.gap_detector.lever_detector(
                self.depth_img, 
                detections, 
                self.labels, 
                self.camera_info, 
                aruco_pose=self.aruco_pose, 
                aruco_point=self.aruco_point
            )

        return labelled_img, detections, lever_actions, cluster_img, depth_scaled, device_mask


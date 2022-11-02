import sys
import numpy as np
import time
from rich import print
import json
import cv2

from gap_detection.gap_detector_clustering import GapDetectorClustering
from object_detection import ObjectDetection
from config import load_config

import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from std_msgs.msg import String
from camera_control_msgs.srv import SetSleeping
import message_filters
from visualization_msgs.msg import MarkerArray

from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.msg import Gaps as ROSGaps
from context_action_framework.types import detections_to_ros, gaps_to_ros

class RealsensePipeline:
    def __init__(self, yolact, dataset, camera_topic="realsense", node_name="vision_realsense", wait_for_services=True):
        self.rate = rospy.Rate(10)

        # don't automatically start
        self.pipeline_enabled = False

        self.camera_topic = camera_topic
        self.node_name = node_name
        self.wait_for_services = wait_for_services

        # realsense data
        self.colour_img = None
        self.depth_img = None
        self.camera_info = None
        self.img_id = 0
        
        # scale depth from mm to meters
        self.depth_rescaling_factor = 1/1000

        # aruco data
        self.aruco_pose = None
        self.aruco_point = None
        
        self.processed_img_id = -1  # don't keep processing the same image
        self.t_now = None
        
        self.labelled_img = None
        self.detections = None
        self.gaps = None
        self.markers = None
        self.poses = None
        
        self.frame_id = "realsense_link"

        print("creating camera subscribers...")
        self.create_camera_subscribers()
        print("creating publishers...")
        self.create_publishers()
        print("creating service client...")
        self.create_service_client()
        print("creating realsense pipeline...")
        self.init_realsense_pipeline(yolact, dataset)

        print("waiting for pipeline to be enabled...")
        
        # Checking the rosparam server for whether to publish labeled imgs etc.
        self.publish_labeled_img = False
        self.publish_depth_img = False
        self.publish_cluster_img = False
 
        self.publish_labeled_rosparamname = '/vision/realsense/publish_labeled_img'
        self.publish_depth_rosparamname = '/vision/realsense/publish_depth_img'
        self.publish_cluster_rosparamname = '/vision/realsense/publish_cluster_img'

        self.last_rosparam_check_time = time.time() # Keeping track of when we last polled the rosparam server
        self.rosparam_check_dt_seconds = 1 # Check rosparam server every 1 second for changes.
        try:
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
            self.publish_depth_img = rospy.get_param(self.publish_depth_rosparamname)
            self.publish_cluster_img = rospy.get_param(self.publish_cluster_rosparamname)
        except:
            rospy.set_param(self.publish_labeled_rosparamname, False)
            rospy.set_param(self.publish_depth_rosparamname, False)
            rospy.set_param(self.publish_cluster_rosparamname, False)

    def check_rosparam_server(self):
        """ Check the rosparam server for whether we want to publish labeled imgs, IF enough time has elapsed between now and last check. """
        cur_t = time.time()
        if cur_t - self.last_rosparam_check_time > self.rosparam_check_dt_seconds:
            self.last_rosparam_check_time = cur_t
  
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
            self.publish_depth_img = rospy.get_param(self.publish_depth_rosparamname)
            self.publish_cluster_img = rospy.get_param(self.publish_cluster_rosparamname)

    def init_realsense_pipeline(self, yolact, dataset):
        self.object_detection = ObjectDetection(yolact, dataset, self.frame_id)
        self.gap_detector = GapDetectorClustering()
    
    def img_from_camera_callback(self, camera_info, img_msg, depth_msg):
        colour_img = CvBridge().imgmsg_to_cv2(img_msg)
        colour_img = cv2.cvtColor(colour_img, cv2.COLOR_BGR2RGB)
        self.colour_img = np.array(colour_img)
        self.depth_img = CvBridge().imgmsg_to_cv2(depth_msg) * self.depth_rescaling_factor
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

        ts = message_filters.ApproximateTimeSynchronizer([camera_info_sub, img_sub, depth_sub], 10, slop=0.05, allow_headerless=False)
        ts.registerCallback(self.img_from_camera_callback)

        # subscribe to aruco
        aruco_sub = message_filters.Subscriber("/realsense_aruco/pose", PoseStamped)
        aruco_pixel_sub = message_filters.Subscriber("/realsense_aruco/pixel", PointStamped)

        # we might not always see the aruco markers, so subscribe to them separately
        ts2 = message_filters.ApproximateTimeSynchronizer([aruco_sub, aruco_pixel_sub], 10, slop=0.05, allow_headerless=False)
        ts2.registerCallback(self.aruco_callback)

    def create_service_client(self):
        timeout = 2 # 2 second timeout
        if self.wait_for_services:
            timeout = None
        try:
            print("waiting for service: /" + self.camera_topic + "/enable ...")
            rospy.wait_for_service("/" + self.camera_topic + "/enable", timeout) # 2 seconds
        except rospy.ROSException as e:
            print("[red]Couldn't find to service! /" + self.camera_topic + "/enable [/red]")
        self.camera_service = rospy.ServiceProxy("/" + self.camera_topic + "/enable", SetBool)

    def create_publishers(self):
        self.br = CvBridge()
        self.labelled_img_pub = rospy.Publisher("/" + self.node_name + "/colour", Image, queue_size=1)
        self.detections_pub = rospy.Publisher("/" + self.node_name + "/detections", ROSDetections, queue_size=1)
        self.markers_pub = rospy.Publisher("/" + self.node_name + "/markers", MarkerArray, queue_size=1)
        self.poses_pub = rospy.Publisher("/" + self.node_name + "/poses", PoseArray, queue_size=1)
        
        self.gaps_pub = rospy.Publisher("/" + self.node_name + "/gaps", ROSGaps, queue_size=1)
        
        self.clustered_img_pub = rospy.Publisher("/" + self.node_name + "/cluster", Image, queue_size=1)
        self.mask_img_pub = rospy.Publisher("/" + self.node_name + "/mask", Image, queue_size=1)
        self.depth_img_pub = rospy.Publisher("/" + self.node_name + "/depth", Image, queue_size=1)
        self.lever_pose_pub = rospy.Publisher("/" + self.node_name + "/lever", PoseStamped, queue_size=1)

    def publish(self, img, detections, markers, poses, gaps, cluster_img, depth_scaled, device_mask):
        
        # all messages are published with the same timestamp
        timestamp = rospy.Time.now()
        header = rospy.Header()
        header.stamp = timestamp
        ros_detections = ROSDetections(header, detections_to_ros(detections))
        
        img_msg = self.br.cv2_to_imgmsg(img)
        img_msg.header.stamp = timestamp
        self.labelled_img_pub.publish(img_msg)
        
        if self.publish_labeled_img:
            self.labelled_img_pub.publish(self.br.cv2_to_imgmsg(img))
        self.detections_pub.publish(ros_detections)
        for marker in markers.markers:
            marker.header.stamp = timestamp
        self.markers_pub.publish(markers)
        poses.header.stamp = timestamp
        self.poses_pub.publish(poses)
        
        if cluster_img is not None:
            cluster_img_msg = self.br.cv2_to_imgmsg(cluster_img)
            cluster_img_msg.header.stamp = timestamp
            self.clustered_img_pub.publish(cluster_img_msg)
            if self.publish_cluster_img:
                self.clustered_img_pub.publish(self.br.cv2_to_imgmsg(cluster_img))
        if device_mask is not None:
            device_mask_msg = self.br.cv2_to_imgmsg(device_mask)
            device_mask_msg.header.stamp = timestamp
            self.mask_img_pub.publish(device_mask_msg)
        if depth_scaled is not None:
            depth_scaled_msg = self.br.cv2_to_imgmsg(depth_scaled)
            depth_scaled_msg.header.stamp = timestamp
            self.depth_img_pub.publish(depth_scaled_msg)
            if self.publish_depth_img:
                self.depth_img_pub.publish(self.br.cv2_to_imgmsg(depth_scaled))

        # publish only the most probable lever action, for now
        # we could use pose_stamped array instead to publish all the lever possibilities
        if gaps is not None and len(gaps) > 0:
            gaps_msg = ROSGaps(header, gaps_to_ros(gaps))
            lever_pose_msg = gaps[0].pose_stamped
            lever_pose_msg.header.stamp = timestamp
            
            self.lever_pose_pub.publish(lever_pose_msg)
            self.gaps_pub.publish(gaps_msg)

    def enable_camera(self, state):
        try:
            res = self.camera_service(state)
            if state:
                print("enabled realsense camera:", res.success)
            else:
                print("disabled realsense camera:", res.success)
        except rospy.ServiceException as e:
            print("[red]Service call failed (state " + str(state) + "):[/red]", e)

    def enable(self, state):
        self.enable_camera(state)
        self.pipeline_enabled = state

    def get_stable_detection(self, gap_detection: bool=True):
        # todo: logic to get stable detection
        # todo: wait until we get at least one detection
        
        if gap_detection:
            while self.gaps is None or self.detections is None:
                print("waiting for detection (realsense)...")
                time.sleep(1) #! debug
        else:
            while self.detections is None:
                print("waiting for detection (realsense)...")
                time.sleep(1) #! debug
            
        if self.detections is not None:
            if gap_detection:
                return self.colour_img, self.detections, self.gaps
            else:
                return self.colour_img, self.detections, None

        else:
            print("stable detection failed!")
            return None, None, None

    def run(self):
        if self.pipeline_enabled:
            if self.colour_img is not None and self.processed_img_id < self.img_id:
                
                self.check_rosparam_server() # Periodically check rosparam server for whether we wish to publish labeled, depth and cluster imgs
                
                print("\n[green]running pipeline realsense frame...[/green]")
                self.processed_img_id = self.img_id
                t_prev = self.t_now
                self.t_now = time.time()
                fps = None
                if t_prev is not None and self.t_now - t_prev > 0:
                    fps = "fps_total: " + str(round(1 / (self.t_now - t_prev), 1)) + ", "

                labelled_img, detections, markers, poses, gaps, cluster_img, depth_scaled, device_mask \
                    = self.process_img(fps=fps)

                self.publish(labelled_img, detections, markers, poses, gaps, cluster_img, depth_scaled, device_mask)
                
                # recheck if pipeline is enabled
                if self.pipeline_enabled:
                    self.labelled_img = labelled_img
                    self.detections = detections
                    self.markers = markers
                    self.poses = poses
                    self.gaps = gaps
                    
                    if self.img_id == sys.maxsize:
                        self.img_id = 0
                        self.processed_img_id = -1
                
            else:
                print("Waiting to receive image (realsense).")
                #! shouldn't need to do this but the realsense camera sometimes stops working
                if self.pipeline_enabled:
                    print("resending rosservice call /realsense/enable True")
                    self.enable_camera(True)


    def process_img(self, fps=None):
        # 2. apply yolact to image and get hca_back
        labelled_img, detections, markers, poses = self.object_detection.get_prediction(self.colour_img, depth_img=self.depth_img, extra_text=fps, camera_info=self.camera_info)

        # 3. apply mask to depth image and convert to pointcloud
        gaps, cluster_img, depth_scaled, device_mask \
            = self.gap_detector.lever_detector(
                self.depth_img, 
                detections,
                self.camera_info, 
                aruco_pose=self.aruco_pose, 
                aruco_point=self.aruco_point
            )

        return labelled_img, detections, markers, poses, gaps, cluster_img, depth_scaled, device_mask


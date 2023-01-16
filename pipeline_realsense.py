import sys
import numpy as np
import time
from rich import print
import json
import rospy
import cv2
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
from gap_detection.gap_detector_clustering import GapDetectorClustering

from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from std_msgs.msg import String
from camera_control_msgs.srv import SetSleeping
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray, TransformStamped
import message_filters

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest
from context_action_framework.msg import VisionDetails
from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.types import detections_to_ros, gaps_to_ros, Label, Camera
from context_action_framework.types import Label, Camera
from context_action_framework.msg import Gaps as ROSGaps

from obb import obb_px_to_quat


class PipelineRealsense(PipelineCamera):    
    def __init__(self, yolact, dataset, object_reid, config):
        self.camera_config = config.realsense
        self.camera_config.enable_topic = "enable" # realsense specific
        
        self.camera_info = None
        self.depth_msg = None
        self.depth_img = None
        
        self.aruco_pose = None
        self.aruco_point = None
        
        # scale depth from mm to meters
        self.depth_rescaling_factor = 1/1000
        
        super().__init__(yolact, dataset, object_reid, config, self.camera_config, Camera.realsense)
    
    def init_pipeline(self, yolact, dataset, object_reid):
        super().init_pipeline(yolact, dataset, object_reid)
        self.gap_detector = GapDetectorClustering(self.config)
        
    
    def create_subscribers(self):
        camera_info_topic = path(self.camera_config.camera_node, "color/camera_info")
        img_topic = path(self.camera_config.camera_node, "color/image_raw")
        depth_topic = path(self.camera_config.camera_node, "aligned_depth_to_color/image_raw")
        camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
        img_sub = message_filters.Subscriber(img_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)

        ts = message_filters.ApproximateTimeSynchronizer([camera_info_sub, img_sub, depth_sub], 10, slop=0.05, allow_headerless=False)
        ts.registerCallback(self.img_from_camera_cb)

        # subscribe to aruco
        aruco_sub = message_filters.Subscriber("/realsense_aruco/pose", PoseStamped)
        aruco_pixel_sub = message_filters.Subscriber("/realsense_aruco/pixel", PointStamped)

        # we might not always see the aruco markers, so subscribe to them separately
        ts2 = message_filters.ApproximateTimeSynchronizer([aruco_sub, aruco_pixel_sub], 10, slop=0.05, allow_headerless=False)
        ts2.registerCallback(self.aruco_cb)
    
    
    def create_publishers(self):
        super().create_publishers()

        self.gaps_pub = rospy.Publisher(path(self.camera_topic, "gaps"), ROSGaps, queue_size=1)
        self.clustered_img_pub = rospy.Publisher(path(self.camera_topic, "cluster"), Image, queue_size=1)
        self.mask_img_pub = rospy.Publisher(path(self.camera_topic, "mask"), Image, queue_size=1)
        self.depth_img_pub = rospy.Publisher(path(self.camera_topic, "depth"), Image, queue_size=1)
        self.lever_pose_pub = rospy.Publisher(path(self.camera_topic, "lever"), PoseStamped, queue_size=1)

    
    def create_services(self):
        super().create_services()
        
        depth_img_enable = path(self.config.node_name, self.config.realsense.topic, "depth_img", "enable")
        cluster_img_enable = path(self.config.node_name, self.config.realsense.topic, "cluster_img", "enable")
        
        rospy.Service(depth_img_enable, SetBool, self.depth_img_enable_cb)
        rospy.Service(cluster_img_enable, SetBool, self.cluster_img_enable_cb)
     

    def img_from_camera_cb(self, camera_info, img_msg, depth_msg):
        self.camera_info = camera_info
        self.depth_msg = depth_msg
        super().img_from_camera_cb(img_msg)
        
    
    def depth_img_enable_cb(self, req):
        state = req.data
        self.camera_config.publish_depth_img = state
        msg = "publish depth_img: " + ("enabled" if state else "disabled")
        return True, msg
    
    def cluster_img_enable_cb(self, req):
        state = req.data
        self.camera_config.publish_cluster_img = state
        msg = "publish cluster_img: " + ("enabled" if state else "disabled")
        return True, msg
    
    def aruco_cb(self, aruco_pose_ros, aruco_point_ros):
        self.aruco_pose = aruco_pose_ros.pose
        self.aruco_point = aruco_point_ros
        
        
    def create_service_client(self):
        super().create_service_client()
        self.camera_service = rospy.ServiceProxy(path(self.camera_config.camera_node, self.camera_config.enable_topic), SetBool)
        
    def process_img(self, fps=None):
        depth_img = CvBridge().imgmsg_to_cv2(self.depth_msg) * self.depth_rescaling_factor
        self.depth_img = rotate_img(depth_img, self.camera_config.rotate_img)
        
        labelled_img, detections, markers, poses, graph_img, graph_relations = super().process_img(fps)

        # apply mask to depth image and convert to pointcloud
        gaps, cluster_img, depth_scaled, device_mask \
            = self.gap_detector.lever_detector(
                self.colour_img,
                self.depth_img,
                detections,
                graph_relations,
                self.camera_info,
                aruco_pose=self.aruco_pose,
                aruco_point=self.aruco_point
            )

        return labelled_img, detections, markers, poses, graph_img, gaps, cluster_img, depth_scaled, device_mask
    

    def publish(self, img, detections, markers, poses, graph_img, gaps, cluster_img, depth_scaled, device_mask):
        header, timestamp = super().publish(img, detections, markers, poses, graph_img)
        
        #! can we make these markers generic?
        for marker in markers.markers:
            marker.header.stamp = timestamp
            marker.header.frame_id = self.config.realsense.parent_frame
            marker.ns = self.config.realsense.topic
            marker.lifetime = rospy.Duration(1)
            # Hack to change coordinates. Z should point away from the camera
            x = marker.pose.position.x
            y = marker.pose.position.y
            z = marker.pose.position.z

            marker.pose.position.x = -y
            marker.pose.position.y = -z
            marker.pose.position.z = x
            # Modifying the quaternion
            x = marker.pose.orientation.x
            y = marker.pose.orientation.y
            z = marker.pose.orientation.z
            w = marker.pose.orientation.w
            
            q_diff = tf.transformations.quaternion_from_euler(1.5708, 0 ,0)
            q_old = [x,y,z,w]
            q_new = tf.transformations.quaternion_multiply(q_diff, q_old)
            # Rotate quaternion by 90 degs
            
            marker.pose.orientation.w = q_new[0]
            marker.pose.orientation.x = q_new[1]
            marker.pose.orientation.y = q_new[2]
            marker.pose.orientation.z = q_new[3]


        self.markers_pub.publish(markers)
        
        #! can we make these poses generic?
        poses.header.stamp = timestamp
        self.poses_pub.publish(poses)
        
        try:
            self.publish_transforms(detections, timestamp)
        except AttributeError as e:
            rospy.loginfo("realsense: ttribute error: {}".format(e))
        if cluster_img is not None:
            cluster_img_msg = self.br.cv2_to_imgmsg(cluster_img, encoding="bgr8")
            cluster_img_msg.header.stamp = timestamp
            if self.camera_config.publish_cluster_img:
                self.clustered_img_pub.publish(cluster_img_msg)
        if device_mask is not None:
            device_mask_msg = self.br.cv2_to_imgmsg(device_mask, encoding="8UC1")
            device_mask_msg.header.stamp = timestamp
            self.mask_img_pub.publish(device_mask_msg)
        if depth_scaled is not None:
            depth_scaled_msg = self.br.cv2_to_imgmsg(depth_scaled)
            depth_scaled_msg.header.stamp = timestamp
            if self.camera_config.publish_depth_img:
                self.depth_img_pub.publish(depth_scaled_msg)

        # publish only the most probable lever action, for now
        # we could use pose_stamped array instead to publish all the lever possibilities
        if gaps is not None and len(gaps) > 0:
            gaps_msg = ROSGaps(header, gaps_to_ros(gaps))
            lever_pose_msg = gaps[0].pose_stamped
            lever_pose_msg.header.stamp = timestamp
            
            self.lever_pose_pub.publish(lever_pose_msg)
            self.gaps_pub.publish(gaps_msg)
            
        if graph_img is not None and self.config.realsense.publish_graph_img:
            graph_img_msg = self.br.cv2_to_imgmsg(graph_img, encoding="8UC4")
            graph_img_msg.header.stamp = timestamp
            self.graph_img_pub.publish(graph_img_msg)
    
        # ! gaps, should be in def run_frame
        self.gaps = gaps

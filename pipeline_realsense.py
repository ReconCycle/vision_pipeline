import sys
import numpy as np
import time
from rich import print
import json
import cv2
import rospy
import tf
import copy
import yaml
import asyncio

from helpers import path, rotate_img
from gap_detection.gap_detector_clustering import GapDetectorClustering
from object_detection import ObjectDetection

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
from context_action_framework.types import detections_to_ros, gaps_to_ros, Label

from obb import obb_px_to_quat

class RealsensePipeline:
    def __init__(self, yolact, dataset, object_reid, config):
        self.config = config

        self.realsense_topic = path(self.config.node_name, self.config.realsense.topic) # /vision/realsense
        
        self.tf_name_suffix = 'realsense' # We will make the detections' TF be called for example 'battery_0_realsense'

        self.camera_height = self.config.realsense.camera_height
        self.camera_height_rosparamname = '/vision/realsense_height' # Get height from camera to table/surface
        init_p, D, K, self.P, w, h = self.parse_calib_yaml(self.config.realsense.calibration_file)

        self.target_fps = 4
        self.min_dt = 1 / self.target_fps # Minimal time between subsequent pipeline runs
        self.last_run_time = time.time()
        self.max_allowed_delay_in_seconds = 0.4 # If cur_t - img_t is larger than this, ignore the detections.

        self.tf_broadcaster = tf.TransformBroadcaster()

        # don't automatically start
        self.pipeline_enabled = False

        # latest realsense data
        self.camera_acquisition_stamp = None
        self.colour_img = None
        self.depth_img = None
        self.camera_info = None
        self.img_id = 0
        
        # scale depth from mm to meters
        self.depth_rescaling_factor = 1/1000

        # aruco data
        self.aruco_pose = None
        self.aruco_point = None
        
        self.t_now = None
        
        # processed image data
        self.processed_img_id = -1  # don't keep processing the same image
        self.processed_colour_img = None
        self.processed_depth_img = None
        self.labelled_img = None
        self.detections = None
        self.gaps = None
        self.markers = None
        self.poses = None
        
        self.frame_id = "realsense_link"

        self.t_camera_service_called = -1 # time camera_service was last called

        print("creating camera subscribers...")
        self.create_camera_subscribers()
        print("creating publishers...")
        self.create_publishers()
        print("creating service client...")
        self.create_service_client()
        print("creating realsense pipeline...")
        self.init_realsense_pipeline(yolact, dataset, object_reid)

        print("waiting for pipeline to be enabled...")
        
        # Checking the rosparam server for whether to publish labeled imgs etc.
        self.publish_labeled_img = self.config.realsense.publish_labelled_img
        self.publish_depth_img = self.config.realsense.publish_depth_img
        self.publish_cluster_img = self.config.realsense.publish_cluster_img
 
        self.publish_labeled_rosparamname = path(self.realsense_topic, "publish_labeled_img")
        self.publish_depth_rosparamname = path(self.realsense_topic, "publish_depth_img")
        self.publish_cluster_rosparamname = path(self.realsense_topic, "publish_cluster_img")

        self.last_rosparam_check_time = time.time() # Keeping track of when we last polled the rosparam server
        self.rosparam_check_dt_seconds = 1 # Check rosparam server every 1 second for changes.
        try:
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
            self.publish_depth_img = rospy.get_param(self.publish_depth_rosparamname)
            self.publish_cluster_img = rospy.get_param(self.publish_cluster_rosparamname)
        except:
            rospy.set_param(self.publish_labeled_rosparamname, self.publish_labeled_img)
            rospy.set_param(self.publish_depth_rosparamname, self.publish_depth_img)
            rospy.set_param(self.publish_cluster_rosparamname, self.publish_cluster_img)

    def check_rosparam_server(self):
        """ Check the rosparam server for whether we want to publish labeled imgs, IF enough time has elapsed between now and last check. """
        cur_t = time.time()
        if cur_t - self.last_rosparam_check_time > self.rosparam_check_dt_seconds:
            self.last_rosparam_check_time = cur_t
  
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
            self.publish_depth_img = rospy.get_param(self.publish_depth_rosparamname)
            self.publish_cluster_img = rospy.get_param(self.publish_cluster_rosparamname)

    def init_realsense_pipeline(self, yolact, dataset, object_reid):
        self.object_detection = ObjectDetection(yolact, dataset, object_reid, self.frame_id)
        self.gap_detector = GapDetectorClustering()
    
    def img_from_camera_callback(self, camera_info, img_msg, depth_msg):
        self.camera_acquisition_stamp = img_msg.header.stamp
        colour_img = CvBridge().imgmsg_to_cv2(img_msg)
        colour_img = np.array(cv2.cvtColor(colour_img, cv2.COLOR_BGR2RGB))
        self.colour_img = rotate_img(colour_img, self.config.realsense.rotate_img)
        depth_img = CvBridge().imgmsg_to_cv2(depth_msg) * self.depth_rescaling_factor
        self.depth_img = rotate_img(depth_img, self.config.realsense.rotate_img)
        
        self.camera_info = camera_info
        self.img_id += 1
        # print("realsense: received realsense image! id:", self.img_id)

    def aruco_callback(self, aruco_pose_ros, aruco_point_ros):
        self.aruco_pose = aruco_pose_ros.pose
        self.aruco_point = aruco_point_ros

    def create_camera_subscribers(self):
        camera_info_topic = path(self.config.realsense.camera_node, "color/camera_info")
        img_topic = path(self.config.realsense.camera_node, "color/image_raw")
        depth_topic = path(self.config.realsense.camera_node, "aligned_depth_to_color/image_raw")
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
        if self.config.realsense.wait_for_services:
            timeout = None
        try:
            print("waiting for service:" + path(self.config.realsense.camera_node, "enable") + " ...")
            rospy.wait_for_service(path(self.config.realsense.camera_node, "enable"), timeout) # 2 seconds
        except rospy.ROSException as e:
            print("[red]Couldn't find to service! " + path(self.config.realsense.camera_node, "enable") + "[/red]")
        self.camera_service = rospy.ServiceProxy(path(self.config.realsense.camera_node, "enable"), SetBool)

    def create_publishers(self):
        self.br = CvBridge()
        self.labelled_img_pub = rospy.Publisher(path(self.realsense_topic, "colour"), Image, queue_size=1)
        self.detections_pub = rospy.Publisher(path(self.realsense_topic, "detections"), ROSDetections, queue_size=1)
        self.markers_pub = rospy.Publisher(path(self.realsense_topic, "markers"), MarkerArray, queue_size=1)
        self.poses_pub = rospy.Publisher(path(self.realsense_topic, "poses"), PoseArray, queue_size=1)
        
        self.gaps_pub = rospy.Publisher(path(self.realsense_topic, "gaps"), ROSGaps, queue_size=1)
        
        self.clustered_img_pub = rospy.Publisher(path(self.realsense_topic, "cluster"), Image, queue_size=1)
        self.mask_img_pub = rospy.Publisher(path(self.realsense_topic, "mask"), Image, queue_size=1)
        self.depth_img_pub = rospy.Publisher(path(self.realsense_topic, "depth"), Image, queue_size=1)
        self.lever_pose_pub = rospy.Publisher(path(self.realsense_topic, "lever"), PoseStamped, queue_size=1)

    def publish(self, img, detections, markers, poses, gaps, cluster_img, depth_scaled, device_mask):

        cur_t = rospy.Time.now()
        delay = (self.camera_acquisition_stamp.nsecs - cur_t.nsecs)*0.000000001 # Ah yes, the secs. I do the secs all the time.

        #rospy.loginfo("RS Delay: {}".format(delay))
        if np.abs(delay) > self.max_allowed_delay_in_seconds:
            rospy.loginfo("RS ignoring img, delay: {}".format(round(delay,2)))
            return 0


        # all messages are published with the same timestamp
        timestamp = cur_t
        header = rospy.Header()
        header.stamp = timestamp
        ros_detections = ROSDetections(header, self.camera_acquisition_stamp, detections_to_ros(detections))
        
        img_msg = self.br.cv2_to_imgmsg(img)
        img_msg.header.stamp = timestamp
        if self.publish_labeled_img:
            print("publishing img_msg")
            self.labelled_img_pub.publish(img_msg)
        
        self.detections_pub.publish(ros_detections)
        for marker in markers.markers:
            marker.header.stamp = timestamp
        self.markers_pub.publish(markers)
        poses.header.stamp = timestamp
        self.poses_pub.publish(poses)

        self.publish_transforms(detections, timestamp)
        
        if cluster_img is not None:
            cluster_img_msg = self.br.cv2_to_imgmsg(cluster_img)
            cluster_img_msg.header.stamp = timestamp
            if self.publish_cluster_img:
                self.clustered_img_pub.publish(cluster_img_msg)
        if device_mask is not None:
            device_mask_msg = self.br.cv2_to_imgmsg(device_mask)
            device_mask_msg.header.stamp = timestamp
            self.mask_img_pub.publish(device_mask_msg)
        if depth_scaled is not None:
            depth_scaled_msg = self.br.cv2_to_imgmsg(depth_scaled)
            depth_scaled_msg.header.stamp = timestamp
            if self.publish_depth_img:
                self.depth_img_pub.publish(depth_scaled_msg)

        # publish only the most probable lever action, for now
        # we could use pose_stamped array instead to publish all the lever possibilities
        if gaps is not None and len(gaps) > 0:
            gaps_msg = ROSGaps(header, gaps_to_ros(gaps))
            lever_pose_msg = gaps[0].pose_stamped
            lever_pose_msg.header.stamp = timestamp
            
            self.lever_pose_pub.publish(lever_pose_msg)
            self.gaps_pub.publish(gaps_msg)

    def publish_transforms(self, detections, timestamp):
        for detection in detections:
            X,Y,Z = self.uv_to_XY(u = detection.center_px[0], v = detection.center_px[1], Z = None, get_z_from_rosparam = True)

            translation = copy.deepcopy(detection.tf.translation)
            translation.x = X; translation.y = Y; translation.z = Z
            rotation = obb_px_to_quat(detection.obb_px)
            #print(str(Label(detection.label).name))
            #print(str(detection.id))
            #print(self.config.realsense.parent_frame)
            tr = (translation.x, translation.y, translation.z)

            child_frame = '%s_%s_%s'%(Label(detection.label).name, detection.id, self.tf_name_suffix)
            #rospy.loginfo("Child frame: {}".format(child_frame))
            self.tf_broadcaster.sendTransform(tr, rotation, timestamp, child_frame, self.config.realsense.parent_frame)
    
    def uv_to_XY(self, u,v, Z=0.35, get_z_from_rosparam=True):
        """Convert pixel coordinated (u,v) from realsense camera into real world coordinates X,Y,Z """
	    
        assert self.P.shape == (3,4)
	    
        if get_z_from_rosparam:
            try:
                self.camera_height = rospy.get_param(self.camera_height_rosparamname)
            except Exception as e:
                rospy.loginfo("realsense: Exception, couldn't get param: " + self.camera_height_rosparamname)
        
        Z = self.camera_height
	    
        fx = self.P[0,0]
        fy = self.P[1,1]

        x = (u - (self.P[0,2])) / fx
        y = (v - (self.P[1,2])) / fy

        X = (Z * x)
        Y = (Z * y)
        Z = Z
        return X, Y, Z

    def enable_camera(self, state):
        # only allow enabling/disabling camera every 2 seconds. To avoid camera breaking.
        rate_limit = 2
        t_now = time.time()
        if t_now - self.t_camera_service_called > rate_limit:
            self.t_camera_service_called = t_now
            try:
                res = self.camera_service(state)
                if state:
                    print("realsense: enabled camera:", res.success)
                else:
                    print("realsense: disabled camera:", res.success)
            except rospy.ServiceException as e:
                print("[red]realsense: Service call failed (state " + str(state) + "):[/red]", e)
        else:
            print("[red]realsense: Camera enable/disable rate limited to " + str(rate_limit) + " seconds.")

    def enable(self, state):
        self.enable_camera(state)
        self.pipeline_enabled = state

    async def get_stable_detection(self, gap_detection: bool=True):
        # todo: logic to get stable detection
        
        #  wait until we get at least one detection
        if gap_detection:
            while self.gaps is None or self.detections is None:
                await asyncio.sleep(0.01)
        else:
            while self.detections is None:
                await asyncio.sleep(0.01)
            
        if self.detections is not None:
            if gap_detection:
                return self.camera_acquisition_stamp, self.processed_colour_img, self.detections, self.gaps, self.processed_img_id
            else:
                return self.camera_acquisition_stamp, self.processed_colour_img, self.detections, None, self.processed_img_id

        else:
            print("realsense: stable detection failed!")
            return None, None, None, None, None

    def run(self):
        t = time.time()
        if self.pipeline_enabled and ((t - self.last_run_time) > self.min_dt) :

            if self.colour_img is not None and self.processed_img_id < self.img_id:
                self.last_run_time = t # reset timer
                
                self.check_rosparam_server() # Periodically check rosparam server for whether we wish to publish labeled, depth and cluster imgs
                processing_img_id = self.img_id
                processing_depth_img = np.copy(self.depth_img)
                processing_colour_img = np.copy(self.colour_img)
                print("\n[green]realsense: running pipeline on img: "+ str(processing_img_id) +"...[/green]")
                
                t_prev = self.t_now
                self.t_now = t
                fps = None
                if t_prev is not None and self.t_now - t_prev > 0:
                    fps = "fps_total: " + str(round(1 / (self.t_now - t_prev), 1)) + ", "

                labelled_img, detections, markers, poses, gaps, cluster_img, depth_scaled, device_mask \
                    = self.process_img(fps=fps)
                
                # recheck if pipeline is enabled
                if self.pipeline_enabled:
                    self.publish(labelled_img, detections, markers, poses, gaps, cluster_img, depth_scaled, device_mask)
                    
                    self.processed_img_id = processing_img_id
                    self.processed_depth_img = processing_depth_img
                    self.processed_colour_img = processing_colour_img
                    self.labelled_img = labelled_img
                    self.detections = detections
                    self.markers = markers
                    self.poses = poses
                    self.gaps = gaps
                    
                    if self.img_id == sys.maxsize:
                        self.img_id = 0
                        self.processed_img_id = -1

                    print("[green]realsense: published img: "+ str(processing_img_id) +", num. dets: " + str(len(detections)) + "[/green]")
                else:
                    print("[green]realsense: aborted publishing on img: " + str(processing_img_id) + " bc pipeline disabled[/green]")
                
            # else:
                # print("realsense: Waiting to receive image.", self.processed_img_id, self.img_id)
                # time.sleep(1) # debug
                #! shouldn't need to do this but the realsense camera sometimes stops working
                # if self.pipeline_enabled:
                    # print("resending rosservice call /realsense/enable True")
                    # self.enable_camera(True) # ! JSI, BUT THIS REALLY BREAKS THINGS FOR ME


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

    def parse_calib_yaml(self, fn):
        """Parse camera calibration file (which is hand-made using ros camera_calibration) """

        with open(fn, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        data = data['Realsense']
        init_p = data['init_robot_pos']
        #print(data)
        w  = data['coeffs'][0]['width']
        h = data['coeffs'][0]['height']
    
        D = np.array(data['coeffs'][0]['D'])
        K = np.array(data['coeffs'][0]['K']).reshape(3,3)
        P = np.array(data['coeffs'][0]['P']).reshape(3,4)
    
        return init_p, D, K, P, w, h


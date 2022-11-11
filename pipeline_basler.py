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


from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.types import detections_to_ros
from context_action_framework.types import Label

from obb import obb_px_to_quat


class BaslerPipeline:
    def __init__(self, yolact, dataset, config):
        self.config = config
        
        self.target_fps = 1
        self.min_run_pipeline_dt = 1 / self.target_fps # Minimal time between subsequent pipeline runs
        self.last_run_time = time.time()
        #self.rate = rospy.Rate(1) # fps.

        self.tf_broadcaster = tf.TransformBroadcaster()

        # don't automatically start
        self.pipeline_enabled = False
        
        self.basler_topic = path(self.config.node_name, self.config.basler.topic) # /vision/basler
        self.img_sub = None

        # latest basler data
        self.camera_acquisition_stamp = rospy.Time.now() # Dont crash on first run
        self.colour_img = None
        self.img_msg = None
        self.img_id = 0
        self.last_stale_id = 0 # Keep track of the ID for last stale img so we dont print several errors for same img

        
        self.t_now = None

        self.max_allowed_delay_in_seconds = 1
        
        # processed image data
        self.processed_img_id = -1  # don't keep processing the same image
        self.processed_colour_img = None
        self.labelled_img = None
        self.detections = None
        self.markers = None
        self.poses = None
        
        self.frame_id = self.config.basler.parent_frame
        
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
        self.publish_labeled_img = self.config.basler.publish_labelled_img
        self.publish_labeled_rosparamname = path(self.basler_topic, "publish_labeled_img")
        self.last_rosparam_check_time = time.time() # Keeping track of when we last polled the rosparam server
        self.rosparam_check_dt_seconds = 1 # Check rosparam server every 1 second for changes.
        try:
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
        except:
            rospy.set_param(self.publish_labeled_rosparamname, False)

        #rospy.loginfo("Pipeline_basler: x = 0.6 -x, hack for table rotation. FIX")
    
    def init_basler_pipeline(self, yolact, dataset):
        self.object_detection = ObjectDetection(yolact, dataset, self.frame_id)
        
        self.worksurface_detection = None

    def check_rosparam_server(self):
        """ Check the rosparam server for whether we want to publish labeled imgs, IF enough time has elapsed between now and last check. """
        cur_t = time.time()
        if cur_t - self.last_rosparam_check_time > self.rosparam_check_dt_seconds:
            self.last_rosparam_check_time = cur_t
            self.publish_labeled_img = rospy.get_param(self.publish_labeled_rosparamname)
    
    def img_from_camera_callback(self, img_msg):
        self.camera_acquisition_stamp = img_msg.header.stamp
        self.img_msg = img_msg 
        self.img_id += 1

    def create_camera_subscribers(self):
        img_topic = path(self.config.basler.camera_node, self.config.basler.image_topic)
        self.img_sub = rospy.Subscriber(img_topic, Image, self.img_from_camera_callback)

    def create_service_client(self):
        timeout = 2 # 2 second timeout
        if self.config.basler.wait_for_services:
            timeout = None
        try:
            print("waiting for service: " + path(self.config.basler.camera_node, "set_sleeping") + " ...")
            rospy.wait_for_service(path(self.config.basler.camera_node, "set_sleeping"), timeout)
        except rospy.ROSException as e:
            print("[red]Couldn't find to service! " + path(self.config.basler.camera_node, "set_sleeping") + "[/red]")
    
        self.camera_service = rospy.ServiceProxy(path(self.config.basler.camera_node, "set_sleeping"), SetSleeping)

    def create_publishers(self):
        self.br = CvBridge()
        self.labelled_img_pub = rospy.Publisher(path(self.basler_topic, "colour"), Image, queue_size=1)
        self.detections_pub = rospy.Publisher(path(self.basler_topic, "detections"), ROSDetections, queue_size=1)
        self.markers_pub = rospy.Publisher(path(self.basler_topic, "markers"), MarkerArray, queue_size=1)
        self.poses_pub = rospy.Publisher(path(self.basler_topic, "poses"), PoseArray, queue_size=1)
        
    def publish(self, img, detections, markers, poses):
        
        cur_t = rospy.Time.now()
        
        timestamp = cur_t
        header = rospy.Header()
        header.stamp = timestamp
        ros_detections = ROSDetections(header, self.camera_acquisition_stamp, detections_to_ros(detections))
        
        img_msg = self.br.cv2_to_imgmsg(img)
        img_msg.header.stamp = timestamp
        if self.publish_labeled_img:
            self.labelled_img_pub.publish(img_msg)
            # self.labelled_img_pub.publish(self.br.cv2_to_imgmsg(img))
        
        #rospy.loginfo("DET: {}".format(ros_detections))
        self.detections_pub.publish(ros_detections)
        
        for marker in markers.markers:
            marker.header.stamp = timestamp
            marker.ns = self.config.basler.parent_frame 
            marker.lifetime = rospy.Duration(1) # Each marker is valid for 1 second max.

        self.markers_pub.publish(markers)
        
        poses.header.stamp = timestamp
        self.poses_pub.publish(poses)

        # Publish the TFs
        self.publish_transforms(detections, timestamp)

    def publish_transforms(self, detections, timestamp):
        for detection in detections:
            translation = copy.deepcopy(detection.tf.translation)   # Table should be rotated in config but that is a lot more work
            translation.z = 0
            rotation = detection.tf.rotation
            rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
            #rotation = obb_px_to_quat(detection.obb_px)
            
            tr = (translation.x, translation.y, translation.z)

            child_frame = '%s_%s_%s'%(Label(detection.label).name, detection.id,   self.config.basler.parent_frame)

            self.tf_broadcaster.sendTransform(tr, rotation, rospy.Time.now(), child_frame, self.config.basler.parent_frame)
            

    def enable_camera(self, state):
        # enable = True, but the topic is called set_sleeping, so the inverse
        state = not state
        try:
            res = self.camera_service(state)
            if state:
                print("basler: enabled camera:", res.success)
            else:
                print("basler: disabled camera:", res.success)
        except rospy.ServiceException as e:
            print("[red]basler: Service call failed (state " + str(state) + "):[/red]", e)

    def enable(self, state):
        self.enable_camera(state)
        self.pipeline_enabled = state
        if state == False:
            self.labelled_img = None
            self.detections = None

    async def get_stable_detection(self):
        # todo: logic to get stable detection
        
        # wait until we get at least one detection
        while self.detections is None:
            await asyncio.sleep(0.01)
        
        if self.detections is not None:
            return self.camera_acquisition_stamp, self.colour_img, self.detections, self.processed_img_id

        else:
            print("basler: stable detection failed!")
            return None, None, None, None
        
    def run(self):

        if not self.pipeline_enabled: 
            #print("[green]basler: aborted bc pipeline disabled[/green]")
            return 0

        if self.img_msg is None: 
            #print("basler: Waiting to receive image.")
            return 0

        # Pipeline is enabled and we have an image
        t = time.time()
 
        # Check if the image is stale
        cam_img_delay = t - self.camera_acquisition_stamp.secs
        if (cam_img_delay > self.max_allowed_delay_in_seconds): 
            # So we dont print several times for the same image
            if self.img_id != self.last_stale_id:
                print("Basler STALE img ID %d, not processing. Delay: %.2f"% (self.img_id, cam_img_delay))
                self.last_stale_id = self.img_id
            return 0 

        # Check that more than minimal time has elapsed since last running the pipeline
        dt = np.abs(t - self.last_run_time)       
        if (dt < self.min_run_pipeline_dt):
            #print("Basler not running due to minimal dt")
            return 0
        self.last_run_time = t


        # All the checks passes, run the pipeline
        colour_img = np.array(CvBridge().imgmsg_to_cv2(self.img_msg))
        self.colour_img = rotate_img(colour_img, self.config.basler.rotate_img)

                            
        self.check_rosparam_server() # Check rosparam server for whether to publish labeled imgs
        
        processing_img_id = self.img_id
        processing_colour_img = np.copy(self.colour_img)
        
        #print("\n[green]basler: running pipeline on img: "+ str(processing_img_id) +"...[/green]")
        #print("\n[green]basler: delay is %.2f[/green]"%cam_img_delay)
        t_prev = self.t_now
        self.t_now = t
        fps = None
        if t_prev is not None and self.t_now - t_prev > 0:
            fps = "fps_total: " + str(round(1 / (self.t_now - t_prev), 1)) + ", "

        labelled_img, detections, markers, poses = self.process_img(self.colour_img, fps)

        self.publish(labelled_img, detections, markers, poses)
        
        self.processed_img_id = processing_img_id
        self.processed_colour_img = processing_colour_img
        self.labelled_img = labelled_img
        self.detections = detections
        self.markers = markers
        self.poses = poses

        if self.img_id == sys.maxsize:
            self.img_id = 0
            self.processed_img_id = -1
        
        print("[green]basler: published img: "+ str(processing_img_id) +", num. dets: " + str(len(detections)) + "[/green]")
            
    def process_img(self, img, fps=None):
        if self.worksurface_detection is None:
            print("basler: detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img, self.config.basler.work_surface_ignore_border_width)
        
        labelled_img, detections, markers, poses = self.object_detection.get_prediction(img, worksurface_detection=self.worksurface_detection, extra_text=fps)

        # debug
        if self.config.basler.show_work_surface_detection:
            self.worksurface_detection.draw_corners_and_circles(labelled_img)

        if self.config.basler.detect_arucos:
            self.aruco_detection = ArucoDetection()
            labelled_img = self.aruco_detection.run(labelled_img, worksurface_detection=self.worksurface_detection)
        
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

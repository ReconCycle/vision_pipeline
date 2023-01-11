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

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest
from context_action_framework.msg import VisionDetails
from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.types import detections_to_ros
from context_action_framework.types import Label, Camera

from obb import obb_px_to_quat


class BaslerPipeline:
    def __init__(self, yolact, dataset, object_reid, config):
        self.config = config
        
        # time stuff
        self.rate_limit_continuous = rospy.Rate(self.config.basler.target_fps)
        self.rate_limit_single = rospy.Rate(100)
        self.max_allowed_acquisition_delay = self.config.basler.max_allowed_acquisition_delay
        self.last_run_time = rospy.get_rostime().to_sec()

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.frame_id = self.config.basler.parent_frame

        # don't automatically start
        self.continuous_mode = False
        self.single_mode = False
        
        # time of single_mode call
        self.single_mode_time = None
        
        self.basler_topic = path(self.config.node_name, self.config.basler.topic) # /vision/basler
        self.img_sub = None

        # latest basler data
        self.acquisition_stamp = rospy.Time.now() # Dont crash on first run
        self.colour_img = None
        self.img_msg = None
        self.img_id = 0
        self.last_stale_id = 0 # Keep track of the ID for last stale img so we dont print several errors for same img
        
        # processed image data
        self.processed_delay = None
        self.processed_acquisition_stamp = None
        self.processed_img_id = -1  # don't keep processing the same image
        self.processed_colour_img = None
        self.labelled_img = None
        self.detections = None
        self.markers = None
        self.poses = None
        self.graph_img = None
        
        self.create_static_tf(self.frame_id)

        print("creating realsense services...")
        self.create_services()
        print("creating subscribers...")
        self.create_camera_subscribers()
        print("creating publishers...")
        self.create_publishers()
        print("creating service client...")
        self.create_service_client()
        print("creating basler pipeline...")
        self.init_basler_pipeline(yolact, dataset, object_reid)

        print("waiting for pipeline to be enabled...")

    
    def init_basler_pipeline(self, yolact, dataset, object_reid):
        self.object_detection = ObjectDetection(self.config, yolact, dataset, object_reid, Camera.basler, self.frame_id)
        
        self.worksurface_detection = None

    
    def img_from_camera_callback(self, img_msg):
        self.acquisition_stamp = img_msg.header.stamp
        self.img_msg = img_msg
        self.img_id += 1
        
        if self.single_mode:
            t = rospy.get_rostime().to_sec()
 
            # time since single mode called
            img_age = t - self.acquisition_stamp.to_sec()
            print("[blue]basler (single_mode): received img from camera, img_id:" + str(self.img_id) + ", age: "+ str(img_age) +"[/blue]")

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
        self.graph_img_pub = rospy.Publisher(path(self.basler_topic, "graph"), Image, queue_size=1)
        
        
    def create_services(self):
        basler_enable = path(self.config.node_name, self.config.basler.topic, "enable")
        labelled_img_enable = path(self.config.node_name, self.config.basler.topic, "labelled_img", "enable")
        graph_img_enable = path(self.config.node_name, self.config.basler.topic, "graph_img", "enable")
        debug_enable = path(self.config.node_name, self.config.basler.topic, "debug", "enable")
        vision_get_detection = path(self.config.node_name, self.config.basler.topic, "get_detection")
        
        rospy.Service(basler_enable, SetBool, self.enable_basler_cb)
        rospy.Service(labelled_img_enable, SetBool, self.labelled_img_enable_cb)
        rospy.Service(graph_img_enable, SetBool, self.graph_img_enable_cb)
        rospy.Service(debug_enable, SetBool, self.debug_enable_cb)
        rospy.Service(vision_get_detection, VisionDetection, self.vision_single_det_cb)
      
      
    def enable_basler_cb(self, req):
        state = req.data
        
        if state:
            print("basler: starting pipeline...")
            self.enable_continuous(True)
            msg = self.config.node_name + " started."
        else:
            print("basler: stopping pipeline...")
            self.enable_continuous(False)
            msg = self.config.node_name + " stopped."
        
        return True, msg
    
    
    def labelled_img_enable_cb(self, req):
        state = req.data
        self.config.basler.publish_labelled_img = state
        msg = "publish labelled_img: " + ("enabled" if state else "disabled")
        return True, msg
    
    
    def graph_img_enable_cb(self, req):
        state = req.data
        self.config.basler.publish_graph_img = state
        msg = "publish graph_img: " + ("enabled" if state else "disabled")
        return True, msg

    def debug_enable_cb(self, req):
        state = req.data
        self.config.basler.debug_work_surface_detection = state
        msg = "debug: " + ("enabled" if state else "disabled")
        return True, msg
    
    def vision_single_det_cb(self, req):            
        print("basler: enabling...")
        self.enable_camera(True)
        
        print("basler: getting detection")
        self.single_mode = True
        self.single_mode_time = rospy.get_rostime().to_sec()
        #! JSI: replace this with with syncronous method
        
        async def wait_for_single_img_processing(self):
            while self.single_mode:
                await asyncio.sleep(0.001)
            return

        asyncio.run(wait_for_single_img_processing(self))
        
        t = rospy.get_rostime().to_sec()
 
        # Check if the image is stale
        img_age = t - self.processed_acquisition_stamp.to_sec()
        
        # camera_acq_stamp, img_age, img, detections, img_id = asyncio.run(self.get_stable_detection())
        print("[blue]basler (single_mode): returning single detection, img_id:" + str(self.processed_img_id) + ", age: "+ str(img_age) +"[/blue]")
        
        if self.detections is not None:
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            vision_details = VisionDetails(header, self.processed_acquisition_stamp, Camera.basler, False, detections_to_ros(self.detections), [])
            return VisionDetectionResponse(True, vision_details, CvBridge().cv2_to_imgmsg(self.labelled_img))
        else:
            print("basler: returning empty response!")
            return VisionDetectionResponse(False, VisionDetails(), CvBridge().cv2_to_imgmsg(self.labelled_img))
    
        
    def publish(self, img, detections, markers, poses, graph_img):
        
        cur_t = rospy.Time.now()
        
        timestamp = cur_t
        header = rospy.Header()
        header.stamp = timestamp
        ros_detections = ROSDetections(header, self.acquisition_stamp, detections_to_ros(detections))
        
        img_msg = self.br.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = timestamp
        if self.config.basler.publish_labelled_img:
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
        
        if graph_img is not None and self.config.basler.publish_graph_img:
            graph_img_msg = self.br.cv2_to_imgmsg(graph_img, encoding="8UC4")
            graph_img_msg.header.stamp = timestamp
            self.graph_img_pub.publish(graph_img_msg)

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
                print("basler: disabled camera:", res.success)
            else:
                print("basler: enabled camera:", res.success)
        except rospy.ServiceException as e:
            print("[red]basler: Service call failed (state " + str(state) + "):[/red]", e)

    def enable_continuous(self, state):
        #! we shouldn't disable the camera often. Only disable when explicitely called via a service.
        self.enable_camera(state)
        self.continuous_mode = state
        if state == False:
            self.labelled_img = None
            self.detections = None

    def run(self):
        if self.continuous_mode:

            self.run_frame()
            self.rate_limit_continuous.sleep()
        
        elif self.single_mode:
            # wait for next image and process immediately
            if self.single_mode_time is not None and self.acquisition_stamp.to_sec() > self.single_mode_time:
                
                t = rospy.get_rostime().to_sec()
    
                # time since single mode called
                img_age = t - self.acquisition_stamp.to_sec()
                print("[blue]basler (single_mode): about to process, img_id:" + str(self.img_id) + ",age: "+ str(img_age) +"[/blue]")
                
                # process camera image
                success = self.run_frame()
                
                if success:
                    # disable single_mode
                    self.single_mode = False
                    self.single_mode_time = None

            # runs much faster, but only processes one frame
            self.rate_limit_single.sleep()
            
    # ! This has to be run on the main thread :( very annoying
    def run_frame(self):

        if self.img_msg is None:
            #print("basler: Waiting to receive image.")
            return False
        
        # check we haven't processed this frame already
        if self.processed_img_id >= self.img_id:
            return False

        # Pipeline is enabled and we have an image
        t = rospy.get_rostime().to_sec()
 
        # Check if the image is stale
        cam_img_delay = t - self.acquisition_stamp.to_sec()
        if (cam_img_delay > self.max_allowed_acquisition_delay):
            # So we dont print several times for the same image
            if self.img_id != self.last_stale_id:
                print("[red]basler: STALE img ID %d, not processing. Delay: %.2f [/red]"% (self.img_id, cam_img_delay))
                self.last_stale_id = self.img_id
            
            return False
        
        t_prev = self.last_run_time
        self.last_run_time = t

        # All the checks passes, run the pipeline
        #! we should now lock these variables from the callback
        colour_img = np.array(CvBridge().imgmsg_to_cv2(self.img_msg))
        self.colour_img = rotate_img(colour_img, self.config.basler.rotate_img)
                            
        # self.check_rosparam_server() # Check rosparam server for whether to publish labeled imgs
        
        processing_img_id = self.img_id
        processing_colour_img = np.copy(self.colour_img)
        processing_acquisition_stamp = self.acquisition_stamp
        
        #print("\n[green]basler: running pipeline on img: "+ str(processing_img_id) +"...[/green]")
        #print("\n[green]basler: delay is %.2f[/green]"%cam_img_delay)
        fps = None
        if t_prev is not None and self.last_run_time - t_prev > 0:
            fps = "fps_total: " + str(round(1 / (self.last_run_time - t_prev), 1)) + ", "

        labelled_img, detections, markers, poses, graph_img = self.process_img(self.colour_img, fps)

        self.publish(labelled_img, detections, markers, poses, graph_img)
        
        self.processed_delay = rospy.get_rostime().to_sec() - t
        self.processed_acquisition_stamp = processing_acquisition_stamp
        self.processed_img_id = processing_img_id
        self.processed_colour_img = processing_colour_img
        self.labelled_img = labelled_img
        self.detections = detections
        self.markers = markers
        self.poses = poses
        self.graph_img = graph_img

        if self.img_id == sys.maxsize:
            self.img_id = 0
            self.processed_img_id = -1
        
        print("[green]basler: published img: "+ str(processing_img_id) +", num. dets: " + str(len(detections)) + "[/green]")
        
        return True
            
    def process_img(self, img, fps=None):
        if self.worksurface_detection is None:
            print("basler: detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img, self.config.basler.work_surface_ignore_border_width, debug=self.config.basler.debug_work_surface_detection)
        
        labelled_img, detections, markers, poses, graph_img, graph_relations = self.object_detection.get_prediction(img, worksurface_detection=self.worksurface_detection, extra_text=fps)

        # debug
        if self.config.basler.show_work_surface_detection:
            self.worksurface_detection.draw_corners_and_circles(labelled_img)

        if self.config.basler.detect_arucos:
            self.aruco_detection = ArucoDetection()
            labelled_img = self.aruco_detection.run(labelled_img, worksurface_detection=self.worksurface_detection)
        
        return labelled_img, detections, markers, poses, graph_img

    #! do we still need this?
    def create_static_tf(self, frame_id):
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

import sys
import numpy as np
import time
from rich import print
from rich.markup import escape
import json
import rospy
import tf2_ros
import tf
import copy
import asyncio
from threading import Event
from scipy.spatial.transform import Rotation


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
from geometry_msgs.msg import Transform, Vector3, Quaternion, PoseArray, TransformStamped

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest
from context_action_framework.msg import VisionDetails
from context_action_framework.msg import Detection as ROSDetection
from context_action_framework.msg import Detections as ROSDetections
from context_action_framework.types import detections_to_ros, gaps_to_ros, Label, Camera

from obb import obb_px_to_quat


class PipelineCamera:
    def __init__(self, yolact, dataset, object_reid, config, camera_config, camera_type, static_tf_manager):
        self.config = config
        self.camera_config = camera_config
        
        self.camera_type = camera_type
        self.camera_name = camera_type.name
        
        self.camera_enabled = False
        
        # time stuff
        self.rate_limit_continuous = rospy.Rate(self.camera_config.target_fps)
        self.rate_limit_single = rospy.Rate(1000)
        self.max_allowed_acquisition_delay = self.camera_config.max_allowed_acquisition_delay
        self.last_run_time = rospy.get_rostime().to_sec()

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.parent_frame = self.camera_config.parent_frame

        # process first frame
        self.is_first_frame = True

        # don't automatically start
        self.continuous_mode = False
        self.single_mode = False
        
        # time of single_mode call
        self.single_mode_time = None
        
        self.processed_single_event = Event()
        
        self.camera_topic = path(self.config.node_name, self.camera_config.topic) # /vision/camera
        self.img_sub = None

        # latest camera data
        self.acquisition_stamp = rospy.Time.now() # Dont crash on first run
        self.colour_img = None
        self.img_msg = None
        self.img_id = 0
        self.last_stale_id = 0 # Keep track of the ID for last stale img so we dont print several errors for same img
        
        # data staged for proessing
        self.processing_acquisition_stamp = None
        
        # processed image data
        self.processed_delay = None
        self.processed_acquisition_stamp = None
        self.processed_img_id = -1  # don't keep processing the same image
        # self.processed_colour_img = None # ? unused
        self.labelled_img = None
        self.detections = None
        self.markers = None
        # self.poses = None # ? unused
        self.graph_img = None
        self.gaps = None # ? unused
        
        if self.camera_config.create_parent_frame:
            transform = None
            # GOE only
            if self.parent_frame == "panda_2/realsense":
                rot_quat = Rotation.from_euler('xyz', [0, 180, 180], degrees=True).as_quat()
                transform = Transform(Vector3(1.0, 0.4, 0.2), Quaternion(*rot_quat))
            
            static_tf_manager.create_tf(self.parent_frame, parent_frame="world", transform=transform)

        print(self.camera_name +": creating services...")
        self.create_services()
        print(self.camera_name +": creating subscribers...")
        self.create_subscribers()
        print(self.camera_name +": creating publishers...")
        self.create_publishers()
        print(self.camera_name +": creating service client...")
        self.create_service_client()
        print(self.camera_name +": creating pipeline...")
        self.init_pipeline(yolact, dataset, object_reid)

        print(self.camera_name +": enabling camera ...")
        self.enable_camera(True)
        
        # register what to do on shutdown
        rospy.on_shutdown(self.exit)
    
    def init_pipeline(self, yolact, dataset, object_reid):
        self.object_detection = ObjectDetection(self.config, yolact, dataset, object_reid, self.camera_type, self.parent_frame)
        self.worksurface_detection = None

    def create_subscribers(self):
        img_topic = path(self.camera_config.camera_node, self.camera_config.image_topic)
        self.img_sub = rospy.Subscriber(img_topic, Image, self.img_from_camera_cb)

    def create_service_client(self):
        timeout = 2 # 2 second timeout
        if self.camera_config.wait_for_services:
            timeout = None
        try:
            print(self.camera_name +": waiting for service: " + path(self.camera_config.camera_node, self.camera_config.enable_topic) + " ...")
            rospy.wait_for_service(path(self.camera_config.camera_node, self.camera_config.enable_topic), timeout)
        except rospy.ROSException as e:
            print("[red]" + self.camera_name +": Couldn't find to service! " + path(self.camera_config.camera_node, self.camera_config.enable_topic) + "[/red]")


    def create_publishers(self):
        self.br = CvBridge()
        self.labelled_img_pub = rospy.Publisher(path(self.camera_topic, "colour"), Image, queue_size=1)
        self.detections_pub = rospy.Publisher(path(self.camera_topic, "detections"), ROSDetections, queue_size=1)
        self.markers_pub = rospy.Publisher(path(self.camera_topic, "markers"), MarkerArray, queue_size=1)
        self.poses_pub = rospy.Publisher(path(self.camera_topic, "poses"), PoseArray, queue_size=1)
        self.graph_img_pub = rospy.Publisher(path(self.camera_topic, "graph"), Image, queue_size=1)
        

    def create_services(self):
        camera_enable = path(self.config.node_name, self.camera_config.topic, "enable")
        labelled_img_enable = path(self.config.node_name, self.camera_config.topic, "labelled_img", "enable")
        graph_img_enable = path(self.config.node_name, self.camera_config.topic, "graph_img", "enable")
        debug_enable = path(self.config.node_name, self.camera_config.topic, "debug", "enable")
        vision_get_detection = path(self.config.node_name, self.camera_config.topic, "get_detection")
        continuous_enable = path(self.config.node_name, self.camera_config.topic, "continuous")
        
        rospy.Service(camera_enable, SetBool, self.enable_camera_cb)
        rospy.Service(labelled_img_enable, SetBool, self.labelled_img_enable_cb)
        rospy.Service(graph_img_enable, SetBool, self.graph_img_enable_cb)
        rospy.Service(debug_enable, SetBool, self.debug_enable_cb)
        rospy.Service(vision_get_detection, VisionDetection, self.vision_single_det_cb)
        rospy.Service(continuous_enable, SetBool, self.enable_continuous_cb)
        
      
    def img_from_camera_cb(self, img_msg):
        self.acquisition_stamp = img_msg.header.stamp
        self.img_msg = img_msg
        self.img_id += 1
        
        if self.single_mode:
            if self.processing_acquisition_stamp is None:
                t = rospy.get_rostime().to_sec()
                # time since single mode called
                img_age = np.round(t - self.acquisition_stamp.to_sec(), 2)
                single_mode_age = str(np.round(t - self.single_mode_time, 2))
                img_age_since_single_mode = str(np.round(self.acquisition_stamp.to_sec() - self.single_mode_time, 2))
                print("[blue]"+ self.camera_name +" (single_mode): received img from camera, img_id:" + str(self.img_id) + ", img_age: "+ str(img_age) +", single_mode_age: "+ single_mode_age+", img_age_since_single_mode: "+img_age_since_single_mode+"[/blue]")
      
    def enable_camera_cb(self, req):
        state = req.data
        success, msg = self.enable_camera(state)
        return success, msg
    
    def labelled_img_enable_cb(self, req):
        state = req.data
        self.camera_config.publish_labelled_img = state
        msg = "publish labelled_img: " + ("enabled" if state else "disabled")
        return True, msg
    
    def graph_img_enable_cb(self, req):
        state = req.data
        self.camera_config.publish_graph_img = state
        msg = "publish graph_img: " + ("enabled" if state else "disabled")
        return True, msg

    def debug_enable_cb(self, req):
        state = req.data
        self.camera_config.debug_work_surface_detection = state
        msg = "debug: " + ("enabled" if state else "disabled")
        return True, msg
    
    # TODO: return false if the cameras aren't running... otherwise the service call will hang forever
    def vision_single_det_cb(self, req):      
        request_gap_detection = req.gap_detection
              
        print(self.camera_name +": getting detection")
        self.single_mode = True
        self.single_mode_time = rospy.get_rostime().to_sec()
        single_mode_time = self.single_mode_time

        self.processed_single_event.wait()
        self.processed_single_event.clear() # immediately reset again for next time
        
        t = rospy.get_rostime().to_sec()
 
        img_age = np.round(t - self.processed_acquisition_stamp.to_sec(), 2)
        single_mode_age = str(np.round(t - single_mode_time, 2))
        print("[blue]"+self.camera_name +" (single_mode): returning single detection, img_id:" + str(self.processed_img_id) + ", img_age: "+ str(img_age) +", single_mode_age: "+ single_mode_age+"[/blue]")
        
        # TODO: also only process gap stuff if it is requested... otherwise it is time-wasting
        ros_gaps = []
        will_return_gaps = False
        if self.gaps is not None and request_gap_detection:
            ros_gaps = gaps_to_ros(self.gaps)
            will_return_gaps = True
        
        if self.detections is not None:
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            vision_details = VisionDetails(header, self.processed_acquisition_stamp, self.camera_type, will_return_gaps, detections_to_ros(self.detections), ros_gaps)
            return VisionDetectionResponse(True, vision_details, CvBridge().cv2_to_imgmsg(self.labelled_img))
        else:
            print(self.camera_name +": returning empty response!")
            return VisionDetectionResponse(False, VisionDetails(), CvBridge().cv2_to_imgmsg(self.labelled_img))
    
    def enable_continuous_cb(self, req):
        state = req.data
        if state and not self.camera_enabled:
            return False, "camera is disabled"
        
        self.enable_continuous(state)
        if state:
            return True, "Enabled continuous mode."
        else:
            return True, "Disabled continuous mode."
        
    def enable_continuous(self, state):
        self.continuous_mode = state
        if state == False:
            self.labelled_img = None
            self.detections = None
    
            
    def enable_camera(self, state):
        success = False
        msg = self.camera_name + ": "
        try:
            # inverse required for basler camera
            if hasattr(self.camera_config, 'enable_camera_invert') and self.camera_config.enable_camera_invert:
                res = self.camera_service(not state)
            else:
                res = self.camera_service(state)
            
            success = res.success
            if success:
                if state:
                    msg += "enabled camera"
                else:
                    msg += "disabled camera"
            else:
                if state:
                    msg += "FAILED to enable camera"
                else:
                    msg += "FAILED to disable camera"
            
        except rospy.ServiceException as e:
            if "UVC device is streaming" in str(e):
                msg += "enabled camera (already streaming)"
                success = True
            else:
                if state:
                    msg += "FAILED to enable camera"
                else:
                    msg += "FAILED to disable camera"
                
                msg += ", service call failed: " + escape(str(e))
        
        if success: 
            print("[green]" + msg)
            # update internal state
            self.camera_enabled = state
        else:
            print("[red]" + msg)
        
        return success, msg
    
            
    def exit(self):
        # disable the camera
        print(self.camera_name + ": stopping...")
        self.enable_continuous(False)
        self.enable_camera(False)

    def run(self):
        single_mode_frame_accepted = False
        if self.single_mode:
            t = rospy.get_rostime().to_sec()
            img_age = np.round(t - self.acquisition_stamp.to_sec(), 2)
            single_mode_age = str(np.round(t - self.single_mode_time, 2))
            img_age_since_single_mode = str(np.round(self.acquisition_stamp.to_sec() - self.single_mode_time, 2))
            # print("[blue]"+self.camera_name + " (single_mode): could process, img_id: " + str(self.img_id) + ", img_age: "+ str(img_age) +", single_mode_age: "+ single_mode_age+", img_age_since_single_mode: "+img_age_since_single_mode+"[/blue]")
            
            if self.single_mode_time < self.acquisition_stamp.to_sec():
                print("[blue]"+self.camera_name +" (single_mode): about to process, img_id:" + str(self.img_id) + ", img_age: "+ str(img_age) +", single_mode_age: "+ single_mode_age+", img_age_since_single_mode: "+img_age_since_single_mode+"[/blue]")
                single_mode_frame_accepted = True
            # else:
            #     print("[red]"+self.camera_name +" (single_mode): invalid, img_id:" + str(self.img_id) + ", img_age: "+ str(img_age) +", single_mode_age: "+ single_mode_age+", img_age_since_single_mode: "+img_age_since_single_mode+"[/red]")
        
        # process frame if in continuous mode or if single mode frame is accepted
        if self.continuous_mode or single_mode_frame_accepted or self.is_first_frame:
            
            success = self.run_frame()

            # compute the first frame immediately
            if success:
                self.is_first_frame = False
        
            if single_mode_frame_accepted:
                
                if success:
                    # disable single_mode
                    self.single_mode = False
                    self.single_mode_time = None
                    
                    self.processed_single_event.set()

        # if in continuous mode, run at slower speed
        if self.continuous_mode:
            self.rate_limit_continuous.sleep()
        else:
            # runs much faster, but only processes one frame
            self.rate_limit_single.sleep()
            
    # this has to be run on the main thread
    def run_frame(self):

        if self.img_msg is None:
            #print(self.camera_name +": Waiting to receive image.")
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
                print("[red]"+self.camera_name +": STALE img ID %d, not processing. Delay: %.2f [/red]"% (self.img_id, cam_img_delay))
                self.last_stale_id = self.img_id
            
            return False
        
        t_prev = self.last_run_time
        self.last_run_time = t

        # All the checks passes, run the pipeline
        #! we should now lock these variables from the callback

        
        processing_img_id = self.img_id
        # processing_colour_img = np.copy(self.colour_img) #? unused
        self.processing_acquisition_stamp = self.acquisition_stamp
        
        fps = None
        if t_prev is not None and self.last_run_time - t_prev > 0:
            fps = "fps_total: " + str(round(1 / (self.last_run_time - t_prev), 1)) + ", "

        # TODO: don't know if remaining_args works
        labelled_img, detections, markers, poses, graph_img, *remaining_args = self.process_img(fps)

        self.publish(labelled_img, detections, markers, poses, graph_img, *remaining_args)
        
        self.processed_delay = rospy.get_rostime().to_sec() - t
        self.processed_acquisition_stamp = self.processing_acquisition_stamp
        self.processed_img_id = processing_img_id
        # self.processed_colour_img = processing_colour_img # ? unused
        self.labelled_img = labelled_img
        self.detections = detections
        # self.markers = markers # ? unused
        # self.poses = poses # ? unused
        # self.graph_img = graph_img # ? unused
        
        # set processing timestamp to None again
        self.processing_acquisition_stamp = None

        if self.img_id == sys.maxsize:
            self.img_id = 0
            self.processed_img_id = -1
        
        print("[green]"+self.camera_name +": published img: "+ str(processing_img_id) +", num. dets: " + str(len(detections)) + "[/green]")
        
        return True
    

    def process_img(self, fps=None, camera_info=None, depth_img=None):
        colour_img = np.array(CvBridge().imgmsg_to_cv2(self.img_msg, "bgr8"))
        self.colour_img = rotate_img(colour_img, self.camera_config.rotate_img)
        
        if hasattr(self.camera_config, "use_worksurface_detection") and self.camera_config.use_worksurface_detection:
            if self.worksurface_detection is None:
                print(self.camera_name +": detecting work surface...")
                self.worksurface_detection = WorkSurfaceDetection(self.colour_img, self.camera_config.work_surface_ignore_border_width, debug=self.camera_config.debug_work_surface_detection)
        
        labelled_img, detections, markers, poses, graph_img, graph_relations = self.object_detection.get_prediction(self.colour_img, depth_img=depth_img, worksurface_detection=self.worksurface_detection, extra_text=fps, camera_info=camera_info)

        # debug
        if hasattr(self.camera_config, "use_worksurface_detection") and self.camera_config.use_worksurface_detection:
            if self.camera_config.show_work_surface_detection:
                self.worksurface_detection.draw_corners_and_circles(labelled_img)

        if self.camera_config.detect_arucos:
            self.aruco_detection = ArucoDetection()
            labelled_img = self.aruco_detection.run(labelled_img, worksurface_detection=self.worksurface_detection)
        
        return labelled_img, detections, markers, poses, graph_img, graph_relations

    def publish(self, img, detections, markers, poses, graph_img, *args):
        
        # publish everything with the same timestamp
        timestamp = rospy.Time.now()
        
        # publish images
        if img is not None and self.camera_config.publish_labelled_img:
            img_msg = self.br.cv2_to_imgmsg(img, encoding="bgr8")
            img_msg.header.stamp = timestamp
            self.labelled_img_pub.publish(img_msg)
        
        if graph_img is not None and self.camera_config.publish_graph_img:
            graph_img_msg = self.br.cv2_to_imgmsg(graph_img, encoding="8UC4")
            graph_img_msg.header.stamp = timestamp
            self.graph_img_pub.publish(graph_img_msg)
        
        # publish detections
        header = rospy.Header()
        header.stamp = timestamp
        ros_detections = ROSDetections(header, self.acquisition_stamp, detections_to_ros(detections))
        self.detections_pub.publish(ros_detections)
        
        # publish markers
        for marker in markers.markers:
            marker.header.stamp = timestamp
            if hasattr(self.camera_config, "marker_lifetime"): 
                marker.lifetime = rospy.Duration(self.camera_config.marker_lifetime)
            else:
                marker.lifetime = rospy.Duration(1) # Each marker is valid for 1 second max.

        self.markers_pub.publish(markers)
        
        # publish poses
        poses.header.stamp = timestamp
        self.poses_pub.publish(poses)
    
        # Publish the TFs
        #! TFs seem incorrect for Realsense camera
        self.publish_transforms(detections, timestamp)
        
        return header, timestamp


    def publish_transforms(self, detections, timestamp):
        for detection in detections:
            if detection.tf is not None:
                t = TransformStamped()
                t.header.stamp = timestamp
                t.header.frame_id = self.parent_frame
                # t.child_frame_id = "obj_"+ str(detection.id)
                t.child_frame_id = '%s_%s_%s'%(Label(detection.label).name, detection.id, self.camera_config.parent_frame)
                t.transform = detection.tf
                
                
                # translation = copy.deepcopy(detection.tf.translation)   # Table should be rotated in config but that is a lot more work
                # translation.z = 0 #? why not at 2.5cm or whatever for basler? What about for realsense?
                # tr = (translation.x, translation.y, translation.z)
                
                # rotation = detection.tf.rotation
                # rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
                
                
                # self.tf_broadcaster.sendTransform(tr, rotation, timestamp, child_frame, self.camera_config.parent_frame)
                self.tf_broadcaster.sendTransform(t)
                

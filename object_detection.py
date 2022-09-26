import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
import cv2
from rich import print
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.validation import explain_validity
from scipy.spatial.transform import Rotation

from yolact_pkg.data.config import Config, COLORS
from yolact_pkg.yolact import Yolact

from tracker.byte_tracker import BYTETracker

import obb
import graphics
from helpers import Struct, make_valid_poly, img_to_camera_coords
from context_action_framework.types import Detection, Label

from geometry_msgs.msg import Transform, Vector3, Quaternion, Pose, PoseArray, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospy
import tf
import tf2_ros


class ObjectDetection:
    def __init__(self, yolact, dataset, frame_id=""):

        self.yolact = yolact
        self.dataset = dataset
        self.frame_id = frame_id
        self.object_depth = 0.025 # in meters, the depth of the objects

        # parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
        # parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        # parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
        # parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
        # parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

        self.tracker_args = SimpleNamespace()
        self.tracker_args.track_thresh = 0.1
        self.tracker_args.track_buffer = 10 # num of frames to remember lost tracks
        self.tracker_args.match_thresh = 2.5 # default: 0.9 # higher number is more lenient
        self.tracker_args.min_box_area = 10
        self.tracker_args.mot20 = False
        
        self.tracker = BYTETracker(self.tracker_args)
        self.fps_graphics = -1.
        self.fps_objdet = -1.

    def get_prediction(self, img_path, depth_img=None, worksurface_detection=None, extra_text=None, camera_info=None):
        t_start = time.time()
        
        if depth_img is not None:
            if img_path.shape[:2] != depth_img.shape[:2]:
                raise ValueError("[red]image and depth image shapes do not match! [/red]")
        
        frame, classes, scores, boxes, masks = self.yolact.infer(img_path)
        fps_nn = 1.0 / (time.time() - t_start)

        detections = []
        for i in np.arange(len(classes)):
                
            detection = Detection()
            detection.id = int(i)
            
            detection.label = Label(classes[i]) # self.dataset.class_names[classes[i]]
            
            detection.score = float(scores[i])
            
            box_px = boxes[i].reshape((-1,2)) # convert tlbr
            detection.box_px = obb.clip_box_to_img_shape(box_px, img_path.shape) 
            detection.mask = masks[i]
            
            # compute contour. Required for obb and graph_relations
            mask = masks[i].cpu().numpy().astype("uint8")
            # print("mask.shape", mask.shape)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                # get the contour with the largest area. Assume this is the one containing our object
                cnt = max(cnts, key = cv2.contourArea)
                detection.mask_contour = np.squeeze(cnt)

                poly = None
                if len(detection.mask_contour) > 2:
                    poly = Polygon(detection.mask_contour)
                    poly = make_valid_poly(poly)

                detection.polygon_px = poly
            
            detections.append(detection)
                
        tracker_start = time.time()
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        online_targets = self.tracker.update(boxes, scores) #? does the tracker not benefit from the predicted classes?

        for t in online_targets:
            detections[t.input_id].tracking_id = int(t.track_id)
            detections[t.input_id].tracking_box = t.tlbr
            detections[t.input_id].score = float(t.score)

        fps_tracker = 1.0 / (time.time() - tracker_start)
        
        
        obb_start = time.time()
        
        markers = MarkerArray()
        markers.markers = []
        markers.markers.append(self.delete_all_markers())
        
        poses = PoseArray()
        poses.header.frame_id = self.frame_id
        poses.header.stamp = rospy.Time.now()
        
        # calculate the oriented bounding boxes
        for detection in detections:                
            corners_px, center_px, angle = obb.get_obb_from_contour(detection.mask_contour, img_path)
            detection.obb_px = corners_px
            detection.center_px = center_px
            detection.angle_px = angle
            
            if worksurface_detection is not None:
                # basler, w.r.t. vision_module
                rotation_axis='z'
                rot_quat = Rotation.from_euler(rotation_axis, angle, degrees=True).inv().as_quat()
            else:
                # realsense module, w.r.t. realsense_link
                rotation_axis='x'
                rot_quat = Rotation.from_euler(rotation_axis, angle, degrees=True).as_quat()
            
            # todo: obb_3d
            detection.tf_px = Transform(Vector3(*center_px, 0), Quaternion(*rot_quat))
            
            if worksurface_detection is not None and corners_px is not None:
                center = worksurface_detection.pixels_to_meters(center_px)
                corners = worksurface_detection.pixels_to_meters(corners_px)
                
                detection.center = center
                detection.tf = Transform(Vector3(*center, 0), Quaternion(*rot_quat))
                detection.box = worksurface_detection.pixels_to_meters(detection.box_px)
                detection.obb = corners
                detection.polygon = worksurface_detection.pixels_to_meters(detection.polygon_px, depth=self.object_depth)
                
                # obb_3d calculations
                # first convert to x, y, z coords by adding 0 to each coord for z
                # original obb + obb raised 2.5cm
                corners_padded = np.pad(corners, [(0, 0), (0, 1)], mode='constant')
                corners_padded_high = np.pad(corners, [(0, 0), (0, 1)], mode='constant', constant_values=self.object_depth)
                corners_3d = np.concatenate((corners_padded, corners_padded_high))
                
                detection.obb_3d = corners_3d
                
            elif camera_info is not None and depth_img is not None and center_px is not None and corners_px is not None:
                # mask depth image
                depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=detection.mask.cpu().detach().numpy().astype(np.uint8))
                depth_masked_np = np.ma.masked_equal(depth_masked, 0.0, copy=False)
                depth_mean = depth_masked_np.mean()
                
                if isinstance(depth_mean, np.float):
                
                    detection.center = img_to_camera_coords(center_px, depth_mean, camera_info)
                    detection.tf = Transform(Vector3(*detection.center), Quaternion(*rot_quat))
                    detection.box = img_to_camera_coords(detection.box_px, depth_mean, camera_info)
                    detection.polygon = img_to_camera_coords(detection.polygon_px, depth_mean, camera_info)
                    
                    corners = img_to_camera_coords(corners_px, depth_mean, camera_info)
                    # the lower corners are 2.5cm further away from the camera
                    corners_low = img_to_camera_coords(corners_px, depth_mean - self.object_depth, camera_info)
                    detection.obb = corners
                    detection.obb_3d = np.concatenate((corners, corners_low))
                
            else:
                detection.obb = None
                detection.tf = None
            
            # draw the cuboids
            if detection.tf is not None and detection.obb is not None:
                # todo: x and y could be the wrong way around! they should be chosen depending on the rot_quat
                changing_height = np.linalg.norm(detection.obb[0]-detection.obb[1])
                changing_width = np.linalg.norm(detection.obb[1]-detection.obb[2])
                
                height = changing_height
                width = changing_width
                if changing_height < changing_width:
                    height = changing_width
                    width = changing_height
                
                if worksurface_detection is not None:
                    # basler, w.r.t. worksurface_module
                    marker = self.make_marker(detection.tf, height, width, self.object_depth, detection.id, detection.label)
                else:
                    # realsense, w.r.t. realsense_link
                    marker = self.make_marker(detection.tf, self.object_depth, height, width, detection.id, detection.label)
                markers.markers.append(marker)
            
            # draw the poses and tfs
            if detection.tf is not None:
                # change the angle of the pose so that it looks good for visualisation
                if worksurface_detection is not None:
                    # basler, w.r.t. worksurface_module
                    rot_multiplier = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
                else:
                    # realsense, w.r.t. realsense_link
                    rot_multiplier = Rotation.from_euler('xyz', [90, 0, 90], degrees=True).as_quat()
                pretty_rot = obb.quaternion_multiply(rot_multiplier, rot_quat)
                
                pose = Pose()
                pose.position = detection.tf.translation
                pose.orientation = Quaternion(*pretty_rot)
                poses.poses.append(pose)
                
                #! move this to the pipeline.py files
                br = tf2_ros.TransformBroadcaster()
                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.frame_id
                t.child_frame_id = "obj_"+ str(detection.id)
                t.transform = detection.tf

                br.sendTransform(t)
                
        fps_obb = -1
        if time.time() - obb_start > 0:
            fps_obb = 1.0 / (time.time() - obb_start)
                
        graphics_start = time.time()
        if extra_text is not None:
            extra_text + ", "
        else:
            extra_text = ""
        fps_str = extra_text + "objdet: " + str(round(self.fps_objdet, 1)) + ", nn: " + str(round(fps_nn, 1)) + ", tracker: " + str(np.int(round(fps_tracker, 0))) + ", obb: " + str(np.int(round(fps_obb, 0))) + ", graphics: " + str(np.int(round(self.fps_graphics, 0)))
        labelled_img = graphics.get_labelled_img(frame, masks, detections, fps=fps_str, worksurface_detection=worksurface_detection)
        
        self.fps_graphics = 1.0 / (time.time() - graphics_start)
        self.fps_objdet = 1.0 / (time.time() - t_start)
        
        return labelled_img, detections, markers, poses

    def make_marker(self, tf, x, y, z, id, label):
        # make a visualization marker array for the occupancy grid
        marker = Marker()
        marker.action = Marker.ADD
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'detection_%d' % Marker.CUBE
        marker.id = id
        marker.type = Marker.CUBE
        
        marker.pose.position = tf.translation
        marker.pose.orientation = tf.rotation

        marker.scale.x = x
        marker.scale.y = y
        marker.scale.z = z
        
        color = COLORS[(label.value * 5) % len(COLORS)]
        
        marker.color.r = color[0] / 255
        marker.color.g = color[1] / 255
        marker.color.b = color[2] / 255
        marker.color.a = 0.75
        
        return marker
    
    def delete_all_markers(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'detection_%d' % Marker.CUBE
        marker.type = Marker.CUBE
        
        return marker
        
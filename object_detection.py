import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
import cv2
import torch
from rich import print
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.validation import explain_validity
from scipy.spatial.transform import Rotation

from yolact_pkg.data.config import Config, COLORS
from yolact_pkg.yolact import Yolact

from tracker.byte_tracker import BYTETracker
from graph_relations import GraphRelations, exists_detection, compute_iou
import obb
import graphics
from helpers import Struct, make_valid_poly, img_to_camera_coords
from context_action_framework.types import Detection, Label, Camera
from object_detector_opencv import SimpleDetector

from geometry_msgs.msg import Transform, Vector3, Quaternion, Pose, PoseArray, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospy
import tf
import tf2_ros
from tf.transformations import quaternion_from_euler, quaternion_multiply

import ros_numpy


class ObjectDetection:
    def __init__(self, config=None, camera_config=None, yolact=None, dataset=None, object_reid=None, camera=None, frame_id="", use_ros=True):

        self.config = config
        self.camera_config = camera_config
        
        self.yolact = yolact
        self.dataset = dataset
        self.camera = camera
        self.frame_id = frame_id
        self.object_depth = 0.025 # in meters, the depth of the objects
        
        self.simple_detector = None
        if yolact is None:
            self.simple_detector = SimpleDetector()
        
        self.object_reid = object_reid

        self.tracker_args = SimpleNamespace()
        self.tracker_args.track_thresh = 0.1
        self.tracker_args.track_buffer = 10 # num of frames to remember lost tracks
        self.tracker_args.match_thresh = 2.5 # default: 0.9 # higher number is more lenient
        self.tracker_args.min_box_area = 10
        self.tracker_args.mot20 = False
        
        self.tracker = BYTETracker(self.tracker_args)
        self.fps_graphics = -1.
        self.fps_objdet = -1.
        
        self.use_ros = use_ros

    def get_prediction(self, colour_img, depth_img=None, worksurface_detection=None, extra_text=None, camera_info=None, use_tracker=True):
        t_start = time.time()
        
        if depth_img is not None:
            if colour_img.shape[:2] != depth_img.shape[:2]:
                raise ValueError("[red]image and depth image shapes do not match! [/red]")
        
        
        if self.simple_detector is not None:
            # simple detector opencv
            cnts, boxes = self.simple_detector.run(colour_img)
            frame = torch.from_numpy(colour_img)
            fps_nn = 1.0 / (time.time() - t_start)
            masks = None
            # After this, mask is of size [num_dets, h, w, 1]
            # masks = []
            # for i in np.arange(len(cnts)):
            #     mask = np.zeros((colour_img.shape[0], colour_img.shape[1], 1), np.uint8)
            #     print("mask.shape", mask.shape)
            #     # cv2.drawContours(mask, [cnts[i]], -1, (0,255,0), 1)
            #     cv2.drawContours(mask, [cnts[i]], -1, 255, -1)
            #     masks.append(mask)
            
            # masks = np.array(masks)
            # masks = torch.from_numpy(masks)
            
            # print("masks.shape", masks.shape)
            
            detections = []
            for i in np.arange(len(cnts)):
                detection = Detection()
                detection.id = int(i)
                
                # hardcode all detections as hca_back
                detection.label = Label.hca_back
                detection.score = float(1.0)
                detection.box_px = boxes[i].reshape((-1,2))
                detection.mask_contour = np.squeeze(cnts[i])
                
                detections.append(detection)
                
        else:
        
            frame, classes, scores, boxes, masks = self.yolact.infer(colour_img)
            fps_nn = 1.0 / (time.time() - t_start)

            detections = []
            for i in np.arange(len(classes)):
                    
                detection = Detection()
                detection.id = int(i)

                # the self.dataset.class_names dict may not correlate with Label Enum,
                # therefore we have to convert:
                detection.label = Label[self.dataset.class_names[classes[i]]]
                detection.score = float(scores[i])
                
                box_px = boxes[i].reshape((-1,2)) # convert tlbr
                detection.box_px = obb.clip_box_to_img_shape(box_px, colour_img.shape)
                
                
                # compute contour. Required for obb and graph_relations
                mask = masks[i].cpu().numpy().astype("uint8")
                # detection.mask = masks[i]
                detection.mask = mask # TODO: does this break stuff changing this?
                
                # print("mask.shape", mask.shape)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) > 0:
                    # get the contour with the largest area. Assume this is the one containing our object
                    cnt = max(cnts, key = cv2.contourArea)
                    detection.mask_contour = np.squeeze(cnt)
                else:
                    detection.mask_contour = None
                
                detections.append(detection)
        
        if use_tracker:
            tracker_start = time.time()
            # apply tracker
            # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
            
            # todo: add classes to tracker
            online_targets = self.tracker.update(boxes, scores)

            for t in online_targets:
                detections[t.input_id].tracking_id = int(t.track_id)
                detections[t.input_id].tracking_box = t.tlbr
                detections[t.input_id].score = float(t.score)

            fps_tracker = 1.0 / (time.time() - tracker_start)
        else:
            fps_tracker = 0
        
        # TODO: GET WORKING!
        detections, markers, poses, graph_img, graph_relations, fps_obb = self.get_detections(detections, colour_img, depth_img, worksurface_detection, camera_info)


        graphics_start = time.time()
        if extra_text is not None:
            extra_text + ", "
        else:
            extra_text = ""
        fps_str = extra_text + "objdet: " + str(round(self.fps_objdet, 1)) + ", nn: " + str(round(fps_nn, 1)) + ", tracker: " + str(np.int(round(fps_tracker, 0))) + ", obb: " + str(np.int(round(fps_obb, 0))) + ", graphics: " + str(np.int(round(self.fps_graphics, 0)))
        labelled_img = graphics.get_labelled_img(frame, masks, detections, fps=fps_str, worksurface_detection=worksurface_detection)
        
        self.fps_graphics = 1.0 / (time.time() - graphics_start)
        self.fps_objdet = 1.0 / (time.time() - t_start)


        return labelled_img, detections, markers, poses, graph_img, graph_relations


    def get_detections(self, detections, colour_img=None, depth_img=None, worksurface_detection=None, camera_info=None):
        
        obb_start = time.time()
        
        markers = None
        poses = None
        if self.use_ros:
            markers = MarkerArray()
            markers.markers = []
            markers.markers.append(self.delete_all_markers())
            
            poses = PoseArray()
            poses.header.frame_id = self.frame_id
            poses.header.stamp = rospy.Time.now()

        # calculate the oriented bounding boxes
        for detection in detections:
            if colour_img is not None:
                corners_px, center_px, angle = obb.get_obb_from_contour(detection.mask_contour, colour_img.shape)
            else:
                corners_px, center_px, angle = obb.get_obb_from_contour(detection.mask_contour, None)
            detection.obb_px = corners_px
            detection.center_px = center_px
            detection.angle_px = angle

            poly = None
            if detection.mask_contour is not None and len(detection.mask_contour) > 2:
                poly = Polygon(detection.mask_contour)
                poly = make_valid_poly(poly)

            detection.polygon_px = poly
            
            if angle is not None:
                if worksurface_detection is not None:
                    # detections from basler, w.r.t. vision module table
                    
                    rot_quat = Rotation.from_euler('xyz', [0, 0, angle], degrees=True).inv().as_quat() # ? why inverse?
                else:
                    # detections from realsense, w.r.t. realsense camera
                    # rotate 180 degrees because camera is pointing down
                    rot_quat = Rotation.from_euler('xyz', [180, 0, angle], degrees=True).as_quat()
                
                # todo: obb_3d
                detection.tf_px = Transform(Vector3(*center_px, 0), Quaternion(*rot_quat))
                
                if worksurface_detection is not None and corners_px is not None:
                    center = worksurface_detection.pixels_to_meters(center_px)
                    corners = worksurface_detection.pixels_to_meters(corners_px)
                    
                    detection.center = np.array([*center, 0])
                    detection.tf = Transform(Vector3(*detection.center), Quaternion(*rot_quat))
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
                    # depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=detection.mask.cpu().detach().numpy().astype(np.uint8))

                    depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=detection.mask)
                    depth_masked_np = np.ma.masked_equal(depth_masked, 0.0, copy=False)
                    depth_mean = depth_masked_np.mean()
                    depth_median = np.ma.median(depth_masked_np)
                    
                    if not np.count_nonzero(depth_img):
                        print("[red]detection: depth image is all 0")
                    
                    if isinstance(depth_mean, np.float):
                        
                        # tf translation comes from img_to_camera_coords(...)
                        if self.config.obj_detection.debug:
                            print(f"depth mean {round(depth_mean, 5)}, median {round(depth_median, 5)}")
                        
                        detection.center = img_to_camera_coords(center_px, depth_mean, camera_info)
                        detection.tf = Transform(Vector3(*detection.center), Quaternion(*rot_quat))
                        detection.box = img_to_camera_coords(detection.box_px, depth_mean, camera_info)
                        detection.polygon = img_to_camera_coords(detection.polygon_px, depth_mean, camera_info)
                        
                        corners = img_to_camera_coords(corners_px, depth_mean, camera_info)
                        # the lower corners are 2.5cm further away from the camera
                        corners_low = img_to_camera_coords(corners_px, depth_mean - self.object_depth, camera_info)
                        detection.obb = corners
                        detection.obb_3d = np.concatenate((corners, corners_low))
                        
                        
                        # print("detection: detection.obb", detection.obb)
                    
                else:
                    print("[red]detection: detection real-world info couldn't be determined!")
                    detection.obb = None
                    detection.tf = None
        
            
        # remove objects that don't fit certain constraints, eg. too small, too thin, too big
        def is_valid_detection(detection):
            if detection.angle_px is None:
                print(f"[red]detection: {detection.label.name} angle is None![/red]")
                return False


            
            if detection.obb is None:
                print(f"[red]detection: {detection.label.name} obb is None![/red]")
                return False
            
            # check ratio of obb sides
            edge_small = np.linalg.norm(detection.obb[0] - detection.obb[1])
            edge_large = np.linalg.norm(detection.obb[1] - detection.obb[2])
            
            if edge_large < edge_small:
                edge_small, edge_large = edge_large, edge_small
                
            ratio = edge_large / edge_small
            
            # edge_small should be longer than 0.1cm
            if edge_small < 0.001:
                print(f"[red]detection: {detection.label.name} invalid: edge too small: {round(edge_small, 3)}m")
                return False
            
            # edge_large should be shorter than 25cm
            if edge_large > 0.25:
                print(f"[red]detection: {detection.label.name} invalid: edge too large: {round(edge_large, 3)}m")
                return False
            
            if self.config.obj_detection.debug:
                print(f"{detection.label.name}, edge large: {round(edge_large, 3)}m, edge small: {round(edge_small, 3)}m")
            
            # ratio of longest HCA: 3.4
            # ratio of Kalo 1.5 battery: 1.7
            # a really big ratio corresponds to a really long device. Ratio of 5 is probably false positive.
            if ratio > 5:
                print("[red]detection: invalid: obb ratio of sides too large[/red]")
                return False

            # area should be larger than 1cm^2 = 0.0001 m^2
            if detection.polygon is not None and detection.polygon.area < 0.0001:
                print(f"[red]detection: {detection.label.name} invalid: polygon area too small "+ str(detection.polygon.area) +"[/red]")
                return False
            
            return True
                    
        # TODO: we could show these in a different colour for debugging
        # TODO: filter out invalid detections
        for detection in detections:
            is_valid = is_valid_detection(detection)
            detection.valid = is_valid
        
        # graph relations only uses valid detections
        graph_relations = GraphRelations(detections)    
        
        # TODO: filter out duplicate detections in a group
        for group in graph_relations.groups:
            for detection in group:
                pass
                    
        
        # TODO: track groups
        # based on groups, orientate HCA to always point the same way, dependent on battery position in the device.
        for group in graph_relations.groups:
            hca_back = graph_relations.get_first(group, Label.hca_back)
            battery = graph_relations.get_first(group, Label.battery)
            # TODO: Also fix orientation of other internals like pcb, pcb_uncovered
            # pcb = graph_relations.get_first(group, Label.pcb)
            # pcb_covered = graph_relations.get_first(group, Label.pcb_covered)
            
            if hca_back is not None and battery is not None:
                if graph_relations.is_inside(battery, hca_back):
                    
                    # compute relative TF of battery w.r.t. hca_back
                    hca_back_np_tf = ros_numpy.numpify(hca_back.tf)
                    battery_np_tf = ros_numpy.numpify(battery.tf)
                    
                    inv_hca = np.linalg.inv(hca_back_np_tf)
                    prod_np_tf = np.dot(inv_hca, battery_np_tf)
                    battery_rel_tf = ros_numpy.msgify(Transform, prod_np_tf)
                    
                    # now we check if battery_rel_tf.translation.x > 0, in which case the battery is in the positive direction of the center of the HCA.
                    if battery_rel_tf.translation.x < 0:
                        quat = ros_numpy.numpify(hca_back.tf.rotation)
                        quat_180 = quaternion_from_euler(0, 0, np.pi)
                        quat_new = quaternion_multiply(quat_180, quat)
                        
                        # write new tf
                        hca_back.angle_px = (hca_back.angle_px + 180) % 360
                        hca_back.tf = Transform(Vector3(*hca_back.center), Quaternion(*quat_new))
                        hca_back.tf_px = Transform(Vector3(*center_px, 0), Quaternion(*quat_new))
                        
            
        # if device contains battery:
            # possibly flip orientation
        
        if self.config is not None and self.config.reid and colour_img is not None:
            # object re-id
            self.object_reid.process_detection(colour_img, detections, graph_relations, visualise=True)
        
        graph_img = None
        if self.camera_config.publish_graph_img:
            graph_img = graph_relations.draw_network_x()
        
        # drawing stuff
        for detection in detections:
            # draw the cuboids (makers)
            if detection.valid and detection.tf is not None and detection.obb is not None:
                
                # TODO: x and y could be the wrong way around! they should be chosen depending on the rot_quat
                changing_height = np.linalg.norm(detection.obb[0]-detection.obb[1])
                changing_width = np.linalg.norm(detection.obb[1]-detection.obb[2])
                
                height = changing_height
                width = changing_width
                if changing_height < changing_width:
                    height = changing_width
                    width = changing_height
                
                #! JSI added this implementation instead: Which one is correct? Why do we even need it?
                # o = detection.obb
                # changing_height = np.linalg.norm((o[0]-o[2], o[1] - o[3]))
                # changing_width = np.linalg.norm((o[2] - o[4], o[3] - o[5]))
                # height = np.min((changing_height, changing_width))
                # width = np.max((changing_height, changing_width))
                if self.use_ros:
                    marker = self.make_marker(detection.tf, height, width, self.object_depth, detection.id, detection.label)
                    markers.markers.append(marker)
            
            # draw the poses
            if self.use_ros and detection.valid and detection.tf is not None:
                pose = Pose()
                pose.position = detection.tf.translation
                pose.orientation = detection.tf.rotation
                poses.poses.append(pose)
                
        fps_obb = -1
        if time.time() - obb_start > 0:
            fps_obb = 1.0 / (time.time() - obb_start)
        
        return detections, markers, poses, graph_img, graph_relations, fps_obb

    def make_marker(self, tf, x, y, z, id, label):
        # make a visualization marker array for the occupancy grid
        marker = Marker()
        marker.action = Marker.ADD
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        # marker.ns = 'detection_%d' % Marker.CUBE
        marker.ns = self.frame_id
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
        # marker.ns = 'detection_%d' % Marker.CUBE
        marker.ns = self.frame_id
        marker.type = Marker.CUBE
        
        return marker
        
import os
from types import SimpleNamespace
import numpy as np
import time
from timeit import default_timer as timer
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
from context_action_framework.graph_relations import GraphRelations, exists_detection, compute_iou
import obb
import graphics
from helpers import Struct, make_valid_poly, img_to_camera_coords, add_angles, circular_median
from context_action_framework.types import Detection, Label, LabelFace, Camera, lookup_label_precise_name
from object_detector_opencv import SimpleDetector
from object_reid import ObjectReId

from geometry_msgs.msg import Transform, Vector3, Quaternion, Pose, PoseArray, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospy
import tf
import tf2_ros
from tf.transformations import quaternion_from_euler, quaternion_multiply, euler_from_quaternion, quaternion_inverse
import ros_numpy


class ObjectDetection:
    def __init__(self, config=None, camera_config=None, model=None, object_reid=None, camera=None, frame_id="", use_ros=True):

        self.config = config
        self.camera_config = camera_config
        
        self.model = model
        self.camera = camera
        self.frame_id = frame_id
        self.object_depth = 0.025 # in meters, the depth of the objects
        
        self.simple_detector = None
        if model is None:
            self.simple_detector = SimpleDetector()
        
        self.object_reid = object_reid

        self.tracker_args = SimpleNamespace()
        self.tracker_args.track_thresh = 0.1
        self.tracker_args.track_buffer = 2 * 10 # num of frames to remember lost tracks. 2fps, then 10 seconds
        self.tracker_args.match_thresh = 2.5 # default: 0.9 # higher number is more lenient
        self.tracker_args.min_box_area = 10
        self.tracker_args.mot20 = False

        self.tracker_dict = {}

        # for each class, create a tracker
        for idx, a_class in enumerate(self.model.dataset.class_names):
            #! frame_rate is actually unused inside BYTETracker
            self.tracker_dict[a_class] = BYTETracker(self.tracker_args, cls_id=idx) 

        self.fps_graphics = -1.
        self.fps_objdet = -1.
        
        self.use_ros = use_ros

        self.angle_hist_dict = {}

    @staticmethod
    def fix_labels(model_label):
        label = model_label
        label_face = None
        label_mapping = {
            "hca_front": ["hca", LabelFace.front],
            "hca_back": ["hca", LabelFace.back],
            "hca_side1": ["hca", LabelFace.side1],
            "hca_side2": ["hca", LabelFace.side2],
            "hca_back_empty": ["hca_empty", LabelFace.back],
            "firealarm_front": ["smoke_detector", LabelFace.front],
            "firealarm_back": ["smoke_detector", LabelFace.back],
            "firealarm_insides": ["smoke_detector_insides", None],
            "firealarm_insides_empty": ["smoke_detector_insides_empty", None],

        }
        if model_label in label_mapping:
            label = label_mapping[model_label][0]
            label_face = label_mapping[model_label][1]
        
        return Label[label], label_face

    def get_prediction(self, colour_img, depth_img=None, worksurface_detection=None, extra_text=None, camera_info=None, use_tracker=True, use_classify=True, parent_frame=None, table_name=None):
        t_start = time.time()
        
        if depth_img is not None:
            if colour_img.shape[:2] != depth_img.shape[:2]:
                raise ValueError("[red]image and depth image shapes do not match! [/red]")
        
        # apply median blur to reduce noise
        colour_img = cv2.medianBlur(colour_img, 5)
        
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
                detection.label = Label.hca
                detection.label_face = LabelFace.back
                detection.label_precise = None
                detection.score = float(1.0)
                detection.box_px = boxes[i].reshape((-1,2))
                detection.mask_contour = np.squeeze(cnts[i])
                
                detections.append(detection)
                
        else:
        
            frame, classes, scores, boxes, masks = self.model.infer(colour_img)
            fps_nn = 1.0 / (time.time() - t_start)

            detections = []
            for i in np.arange(len(classes)):
                    
                detection = Detection()
                detection.id = int(i)

                detection.parent_frame = parent_frame
                detection.table_name = table_name

                # the self.dataset.class_names dict may not correlate with Label Enum,
                # therefore we have to convert:
                model_label = self.model.dataset.class_names[classes[i]]

                detection._model_label = model_label # internal use only
                detection._model_box = boxes[i] # internal only
                
                label, label_face = self.fix_labels(model_label)

                detection.label = label
                detection.label_face = label_face
                detection.label_precise = None
                
                detection.score = float(scores[i])
                
                box_px = boxes[i].reshape((-1,2)) # convert tlbr
                detection.box_px = obb.clip_box_to_img_shape(box_px, colour_img.shape)
                
                # compute contour. Required for obb and graph_relations
                mask = masks[i].cpu().numpy().astype("uint8")
                detection.mask = mask
                
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) == 1:
                    cnt = np.squeeze(cnts[0]) # (n, 2)
                    detection.mask_contour = cnt
                elif len(cnts) > 1:
                    # getting only the biggest contour is not the best:
                    # cnt = np.squeeze(max(cnts, key = cv2.contourArea))

                    # we should merge the contours
                    flattened_cnts = np.vstack(cnts).squeeze() # (n, 2)
                    convex_hull = np.squeeze(cv2.convexHull(flattened_cnts, False)) # (n, 2)
                    detection.mask_contour = convex_hull
                else:
                    detection.mask_contour = None
                
                detections.append(detection)
        
        if use_tracker:
            tracker_start = time.time()
            # apply tracker
            # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
            # https://github.com/mnuske/fairmot-exploration-room/blob/multi-class/src/lib/tracker/multitracker.py

            # for each class, update the tracker
            for a_class in self.model.dataset.class_names:
                # lookup all detections with this class
                class_detections = [detection for detection in detections if detection._model_label == a_class]

                boxes = np.array([detection._model_box for detection in class_detections])
                scores = np.array([detection.score for detection in class_detections])

                online_targets = self.tracker_dict[a_class].update(boxes, scores)

                for t in online_targets:
                    class_detections[t.input_id].tracking_id = int(t.track_id)
                    class_detections[t.input_id].tracking_box = t.tlbr
                    class_detections[t.input_id].score = float(t.score) #! why are we setting the score again??

            fps_tracker = 1.0 / (time.time() - tracker_start)
        else:
            fps_tracker = 0

        # add tf_name
        for detection in detections:
            tf_name = detection.label.name
            if detection.label_face is not None:
                tf_name += f"_{detection.label_face.name}"
            if use_tracker:
                if detection.tracking_id is not None:
                    tf_name += f"_{detection.tracking_id}"
                else:
                    tf_name += f"_no_tracking_{detection.id}"
            else:
                tf_name += f"_{detection.id}"

            if parent_frame is not None:
                tf_name = parent_frame + "/" + tf_name

            detection.tf_name = tf_name

        detections, markers, poses, graph_img, graph_relations, fps_obb, batch_crop_imgs = self.get_detections(detections, colour_img, depth_img, worksurface_detection, camera_info, use_classify=use_classify)

        if use_tracker:
            if self.config.obj_detection.rotation_median_filter:
                # apply filter to angle
                for detection in detections:
                    # filter by object
                    if detection.label == Label.smoke_detector and detection.tracking_id is not None:

                        if detection.tf_name not in self.angle_hist_dict:
                            # print("[blue]created history dict")
                            self.angle_hist_dict[detection.tf_name] = []

                        # print(f"[blue]added angle to history_dict track: {detection.tf_name, detection.angle_px}deg, list len: {len(self.angle_hist_dict[detection.tf_name])}")
                        # add angle_px to history
                        self.angle_hist_dict[detection.tf_name].append(detection.angle_px)
                        
                        # prune list
                        n = 10
                        if len(self.angle_hist_dict[detection.tf_name]) > n+4:
                            # keep last 4 elements in list, when pruning
                            del self.angle_hist_dict[detection.tf_name][:n]

                        # last 4 elements:
                        if len(self.angle_hist_dict[detection.tf_name]) >= 4:
                            last_angles = self.angle_hist_dict[detection.tf_name][-4:]

                            median_idx = circular_median(last_angles, degrees=True)
                            median_angle = last_angles[median_idx]

                            # print(f"[green] {detection.label.name}, median_angle {median_angle}, {last_angles}")

                            # set the angle and the median angle
                            self.set_rotation(detection, median_angle, worksurface_detection)

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
    
    def angle_to_quat(self, worksurface_detection, angle_px):
        if worksurface_detection is not None:
            # detections from basler, w.r.t. vision module table
            rot_quat = Rotation.from_euler('xyz', [0, 0, angle_px], degrees=True).as_quat() 
        else:
            # detections from realsense, w.r.t. realsense camera
            # rotate 180 degrees because camera is pointing down, and invert object angle
            inv_angle_px = np.rad2deg(add_angles(0, np.deg2rad(-angle_px)))
            rot_quat = Rotation.from_euler('xyz', [180, 0, inv_angle_px], degrees=True).as_quat()

        return rot_quat
    
    def set_rotation(self, detection, angle_px, worksurface_detection):
        # angle in degrees
        detection.angle_px = angle_px
        
        rot_quat = self.angle_to_quat(worksurface_detection, angle_px)
        
        detection.tf = Transform(detection.tf.translation, Quaternion(*rot_quat))
        detection.tf_px = Transform(detection.tf_px.translation, Quaternion(*rot_quat))

    def get_detections(self, detections, colour_img=None, depth_img=None, worksurface_detection=None, camera_info=None, use_classify=True):
        
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

        # calculate the oriented bounding boxes in pixels
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
                
                for _ in np.arange(3):
                    if len(poly.exterior.coords) > 20:
                        poly = poly.simplify(tolerance=5.) # tolerance in pixels
                    else:
                        break
                        
            detection.polygon_px = poly

        batch_crop_imgs = None
        if use_classify:
            time0 = timer()

            # use classifier for precise label
            # estimate angle using superglue model
            batch_crop_imgs = []
            batch_imgs_detection_mapping = [None] * len(detections)
            # batch the classifier
            for idx, detection in enumerate(detections):
                if detection.label in [Label.smoke_detector, Label.hca]:
                    
                    sample_crop, _ = ObjectReId.crop_det(colour_img, detection, size=400)
                    batch_imgs_detection_mapping[idx] = len(batch_crop_imgs)
                    batch_crop_imgs.append(sample_crop)
            
            if len(batch_crop_imgs) > 0:
                batch_crop_imgs_arr = np.array(batch_crop_imgs)
                classify_labels, confs = self.model.infer_classify(batch_crop_imgs_arr)

                for idx, detection in enumerate(detections):
                    if detection.label in [Label.smoke_detector, Label.hca]:
                        
                        classify_label = classify_labels[batch_imgs_detection_mapping[idx]]
                        conf = confs[batch_imgs_detection_mapping[idx]]
                        crop_img = batch_crop_imgs_arr[batch_imgs_detection_mapping[idx]]

                        if conf > self.config.obj_detection.classifier_threshold:
                            
                            # ! I trust the classifier more than the segmentation. 
                            # print("classify_label", classify_label, "conf", conf)
                            label_split = classify_label.rsplit('_')
                            classify_type = label_split[0]
                            classify_face = label_split[1]
                            classify_num = label_split[2]

                            # print("label_precise", label_precise)
                            # print("classify_face", classify_face)
                            # print("classify_type", classify_type)

                            # if classify_face == "inside":
                            #     print(f"[red]classifier says {classify_face}, label {classify_label}. Skipping...")
                            #     break

                            # fix for old naming convention
                            if classify_type == "firealarm":
                                classify_type = Label.smoke_detector
                            elif classify_type == "hca":
                                classify_type = Label.hca

                            if classify_type.name != detection.label.name:
                                print(f"[red]classifier says type: {classify_type} (face: {classify_face}), but yolo says type: {detection.label.name}")
                            
                            if classify_face != detection.label_face.name:
                                print(f"[red]classifier says face:  {classify_face} (type: {classify_type}), but yolo says face: {detection.label_face.name}")
                            
                            
                            label_precise_name = lookup_label_precise_name(classify_type, classify_num)
                            detection.label_precise = classify_num
                            detection.label_precise_name = label_precise_name

                            print(f"classify: {label_precise_name}, {classify_num}, conf: {conf}")

                            if detection.label == Label.smoke_detector:
                                angle_rad, *_ = self.model.superglue_rot_estimation(crop_img, classify_label)

                                if angle_rad is not None:
                                    # update angle
                                    detection.angle_px = np.rad2deg(angle_rad)
                        else:
                            print(f"[red]classify conf too low: {conf}, {classify_label}")

                elapsed_time_classify_and_rot = timer() - time0
                print("elapsed_time_classify_and_rot", elapsed_time_classify_and_rot)


        # calculate real world information
        for detection in detections:
            if detection.angle_px is not None:
                rot_quat = self.angle_to_quat(worksurface_detection, detection.angle_px)
                
                # todo: obb_3d
                #! this tf_px coordinate system is horrible, because the y-axis is wrong.
                #! so something is wrong here
                detection.tf_px = Transform(Vector3(*detection.center_px, 0), Quaternion(*rot_quat))
                
                if worksurface_detection is not None and detection.obb_px is not None:
                    center = worksurface_detection.pixels_to_meters(detection.center_px)
                    corners = worksurface_detection.pixels_to_meters(detection.obb_px)
                    
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
                    
                elif camera_info is not None and \
                        depth_img is not None and \
                        detection.center_px is not None and \
                        detection.obb_px is not None:
                    # mask depth image
                    # depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=detection.mask.cpu().detach().numpy().astype(np.uint8))

                    depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=detection.mask)
                    depth_masked_np = np.ma.masked_equal(depth_masked, 0.0, copy=False)
                    depth_mean = depth_masked_np.mean()
                    depth_median = np.ma.median(depth_masked_np)
                    
                    if not np.count_nonzero(depth_img):
                        if self.config.obj_detection.debug:
                            print("[red]detection: depth image is all 0")
                    
                    if isinstance(depth_mean, np.float):
                        
                        # tf translation comes from img_to_camera_coords(...)
                        if self.config.obj_detection.debug:
                            print(f"depth mean {round(depth_mean, 5)}, median {round(depth_median, 5)}")
                        
                        detection.center = img_to_camera_coords(detection.center_px, depth_mean, camera_info)
                        detection.tf = Transform(Vector3(*detection.center), Quaternion(*rot_quat))
                        detection.box = img_to_camera_coords(detection.box_px, depth_mean, camera_info)
                        detection.polygon = img_to_camera_coords(detection.polygon_px, depth_mean, camera_info)
                        
                        corners = img_to_camera_coords(detection.obb_px, depth_mean, camera_info)
                        # the lower corners are 2.5cm further away from the camera
                        corners_low = img_to_camera_coords(detection.obb_px, depth_mean - self.object_depth, camera_info)
                        detection.obb = corners
                        detection.obb_3d = np.concatenate((corners, corners_low))
                        
                        
                        # print("detection: detection.obb", detection.obb)
                    
                else:
                    if self.config.obj_detection.debug:
                        print("[red]detection: detection real-world info couldn't be determined!")
                    detection.obb = None
                    detection.tf = None
        
            
        # remove objects that don't fit certain constraints, eg. too small, too thin, too big
        def is_valid_detection(detection):
            if detection.angle_px is None:
                if self.config.obj_detection.debug:
                    print(f"[red]detection: {detection.label.name} angle is None![/red]")
                return False
            
            if detection.obb is None:
                if self.config.obj_detection.debug:
                    print(f"[red]detection: {detection.label.name} obb is None![/red]")
                return False
            
            # check ratio of obb sides
            edge_small = np.linalg.norm(detection.obb[0] - detection.obb[1])
            edge_large = np.linalg.norm(detection.obb[1] - detection.obb[2])
            
            if edge_large < edge_small:
                edge_small, edge_large = edge_large, edge_small
                
            ratio = edge_large / edge_small

            #!
            #! These area constraints don't make sense with close up camera!
            #!

            # edge_small should be longer than 0.1cm
            if edge_small < 0.001:
                if self.config.obj_detection.debug:
                    print(f"[red]detection: {detection.label.name} invalid: edge too small: {round(edge_small, 3)}m")
                return False
            
            # edge_large should be shorter than 25cm
            if edge_large > 0.25:
                if self.config.obj_detection.debug:
                    print(f"[red]detection: {detection.label.name} invalid: edge too large: {round(edge_large, 3)}m")
                return False
            
            # if self.config.obj_detection.debug:
            #     print(f"{detection.label.name}, edge large: {round(edge_large, 3)}m, edge small: {round(edge_small, 3)}m")
            
            # ratio of longest HCA: 3.4
            # ratio of Kalo 1.5 battery: 1.7
            # a really big ratio corresponds to a really long device. Ratio of 5 is probably false positive.
            if ratio > 5:
                if self.config.obj_detection.debug:
                    print("[red]detection: invalid: obb ratio of sides too large[/red]")
                return False

            if detection.label == Label.screw:
                # screws cannot be bigger than 
                if detection.polygon.area > 0.001:
                    return False
            else:
                # area should be larger than 1cm^2 = 0.0001 m^2
                if detection.polygon is not None and detection.polygon.area < 0.0001:
                    if self.config.obj_detection.debug:
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

        # TODO: for a group with multiple objects, add an enclosing object
        # for group in graph_relations.groups:
        #     print("group:", len(group))
        #     if len(group) > 1:
        #         # TODO: add enclosing object
        #         for detection in group:
        #             print(f"detection: {detection.label.name}")

        #         # TODO: create enclosing detection:
        #         detection = Detection()
        #         detection.id = ????
                
        #         # hardcode all detections as hca_back
        #         detection.label = ????
        #         detection.score = float(1.0)
        #         detection.box_px = ????
        #         detection.mask_contour = ????
                
        #         detections.append(detection)
                    
        
        # TODO: track groups
        # based on groups, orientate HCA to always point the same way, dependent on battery position in the device.
        for group in graph_relations.groups:
            hca_back = graph_relations.get_first(group, Label.hca, LabelFace.back)
            battery = graph_relations.get_first(group, Label.battery)
            # TODO: Also fix orientation of other internals like pcb, pcb_uncovered
            # pcb = graph_relations.get_first(group, Label.pcb)
            # pcb_covered = graph_relations.get_first(group, Label.pcb_covered)
            
            if hca_back is not None and battery is not None and hca_back.tf is not None and battery.tf is not None:
                if graph_relations.is_inside(battery, hca_back):
                    # compute relative TF of battery w.r.t. hca_back
                    hca_back_np_tf = ros_numpy.numpify(hca_back.tf)
                    battery_np_tf = ros_numpy.numpify(battery.tf)
                    
                    inv_hca = np.linalg.inv(hca_back_np_tf)
                    battery_rel_np_tf = np.dot(inv_hca, battery_np_tf)
                    battery_rel_tf = ros_numpy.msgify(Transform, battery_rel_np_tf)
                    
                    # now we check if battery_rel_tf.translation.x > 0, in which case the battery is in the positive direction of the center of the HCA.
                    if battery_rel_tf.translation.x < 0:
                        angle_px = add_angles(hca_back.angle_px, 180, degrees=True)
                        self.set_rotation(hca_back, angle_px, worksurface_detection)


        # based on groups, orientate firealarm_back to always orientate based on battery_covered (if it exists).
        for group in graph_relations.groups:
            firealarm_back = graph_relations.get_first(group, Label.smoke_detector, LabelFace.back)
            battery_covered = graph_relations.get_first(group, Label.battery_covered)

            if firealarm_back is not None and battery_covered is not None and firealarm_back.tf is not None and battery_covered.tf is not None:
                if graph_relations.is_inside(battery_covered, firealarm_back): 

                    # set the rotation of firealarm_back equal to battery_covered
                    firealarm_back.angle_px = battery_covered.angle_px

                    firealarm_back.tf = Transform(firealarm_back.tf.translation, battery_covered.tf.rotation)
                    firealarm_back.tf_px = Transform(firealarm_back.tf_px.translation, battery_covered.tf_px.rotation)

                    # compute relative TF of battery_covered w.r.t. firealarm_back
                    firealarm_back_np_tf = ros_numpy.numpify(firealarm_back.tf)
                    battery_covered_np_tf = ros_numpy.numpify(battery_covered.tf)

                    # vector from firealarm_back center to battery_covered center
                    inv_firealarm = np.linalg.inv(firealarm_back_np_tf)
                    battery_covered_rel_np_tf = np.dot(inv_firealarm, battery_covered_np_tf)
                    battery_covered_rel_tf = ros_numpy.msgify(Transform, battery_covered_rel_np_tf)

                    # rotate by 180 degrees based on relative positions
                    if battery_covered_rel_tf.translation.x < 0:
                        angle_px = add_angles(battery_covered.angle_px, 180, degrees=True)
                        self.set_rotation(firealarm_back, angle_px, worksurface_detection)
                        self.set_rotation(battery_covered, angle_px, worksurface_detection)


                    # --> ANOTHER METHOD:

                    # # vector from firealarm_back center to battery_covered center
                    # rel_px = - firealarm_back.center_px + battery_covered.center_px 
                    # rel_angle_px = np.rad2deg(np.arctan2(-rel_px[1], rel_px[0])) # negative y because opencv coords has y inverted
                    # # this angle is from firealarm_back.center_px to battery_covered.center_px
                    # print("rel_angle_px", rel_angle_px)

                    # # the angle from firealarm_back.center_px to battery_covered.center_px should be about the same 
                    # # as the angle of the battery_covered. If not (eg. > 90 degrees), then we are 180 degrees out, 
                    # # so rotate by 180 degrees
                    # if np.abs(battery_covered.angle_px - rel_angle_px) > 90:
                    #     angle_px = add_angles(battery_covered.angle_px, 180, degrees=True)

                    # <-- END, ANOTHER METHOD

        # if device contains battery:
            # possibly flip orientation

        # TODO: classify and estimate rotation
        # for group in graph_relations.groups:
            
        #     pass

        # ! remove reid
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
                # TODO: changing_height, changing_width is also specified somewhere else....
                changing_height = np.linalg.norm(detection.obb[0]-detection.obb[1])
                changing_width = np.linalg.norm(detection.obb[1]-detection.obb[2])
                
                height = changing_height
                width = changing_width
                if changing_height < changing_width:
                    height = changing_width
                    width = changing_height
                
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
        
        return detections, markers, poses, graph_img, graph_relations, fps_obb, batch_crop_imgs


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
        
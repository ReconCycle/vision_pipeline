from __future__ import annotations
import os
import sys
import numpy as np
import time
import cv2
import copy
from rich import print
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, Point
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from types import SimpleNamespace
from object_detection import ObjectDetection

from action_predictor.graph_relations import GraphRelations, exists_detection, compute_iou
from work_surface_detection_opencv import WorkSurfaceDetection

from context_action_framework.types import Action, Detection, Gap, Label, Camera
from tqdm import tqdm
from helpers import scale_img


class ObjectReId:
    def __init__(self, config, det_model) -> None:
        self.det_model = det_model
        self.config = config

        self.reid_dataset = {}

        # ! we can use opencv detector if we set det_model = None

        # pretend to use Basler camera
        self.camera_type = Camera.basler
        self.camera_name = self.camera_type.name
        
        self.camera_config = self.config.basler
        
        self.camera_config.enable_topic = "set_sleeping" # basler camera specific
        self.camera_config.enable_camera_invert = True # enable = True, but the topic is called set_sleeping, so the inverse
        self.camera_config.use_worksurface_detection = True
        
        self.parent_frame = None

        self.worksurface_detection = None

        self.object_detection = ObjectDetection(self.config, self.camera_config, self.det_model, None, self.camera_type, self.parent_frame, use_ros=False)

        self.root_dir = "./datasets_reid/2023-02-20_hca_backs"
        self.processed_dir = "./datasets_reid/2023-02-20_hca_backs_processed"

        self.process_dataset()
        self.load_dataset()


    def process_dataset(self):
        print("[blue]processing dataset for reid...")

        # make the processed directory if it doesn't already exist
        if not os.path.isdir(self.processed_dir):
            os.makedirs(self.processed_dir)

        # iterate over folders and process images
        for subdir, dirs, files in tqdm(list(os.walk(self.root_dir))):
            for file in files:
                filepath = os.path.join(subdir, file)
                subfolder = os.path.basename(subdir)

                if filepath.endswith(".jpg") or filepath.endswith(".png"):
                    processed_filepath = os.path.join(self.processed_dir, subfolder, file)

                    # check if file exists in processed_dir
                    if not os.path.isfile(processed_filepath):
                        self.process_img(filepath, self.processed_dir, subfolder, file)


    def process_img(self, filepath, processed_dir, subfolder, file):
        processed_filepath = os.path.join(processed_dir, subfolder, file)
        processed_dirpath = os.path.dirname(processed_filepath)

        print(f"process {processed_filepath}")

        # make the processed directory if it doesn't already exist
        if not os.path.isdir(processed_dirpath):
            os.makedirs(processed_dirpath)
        
        img = cv2.imread(filepath)

        # TODO: do worksurface_detection for each image do it for each image
        if self.worksurface_detection is None:
                print(self.camera_name +": detecting work surface...")
                self.worksurface_detection = WorkSurfaceDetection(img, self.camera_config.work_surface_ignore_border_width, debug=self.camera_config.debug_work_surface_detection)

        labelled_img, detections, markers, poses, graph_img, graph_relations = self.object_detection.get_prediction(img, depth_img=None, worksurface_detection=self.worksurface_detection, extra_text=None, camera_info=None)

        img0_cropped, obb_poly1 = self.find_and_crop_det(img, graph_relations)

        # write image to file
        cv2.imwrite(processed_filepath, img0_cropped)

        # remove properties we don't need to save
        for detection in detections:
            detection.mask = None
            detection.mask_contour = None
        
        # pickle detections
        obj_templates_json_str = jsonpickle.encode(detections, keys=True, warn=True, indent=2)

        filename = os.path.splitext(file)[0]

        # write detections to file
        with open(os.path.join(processed_dirpath, filename + ".json"), 'w', encoding='utf-8') as f:
            f.write(obj_templates_json_str)


    def load_dataset(self):
        print("[blue]loading dataset...")
        
        # iterate over folders load images and detections
        for subdir, dirs, files in tqdm(list(os.walk(self.processed_dir))):
            for file in files:
                filepath = os.path.join(subdir, file)
                subfolder = os.path.basename(subdir)
                filename = os.path.splitext(file)[0]

                filepath_json = os.path.join(subdir, filename + ".json")

                if filepath.endswith(".jpg") or filepath.endswith(".png"):

                    # print("filepath", filepath)
                    # print("filepath_json", filepath_json)

                    # check if file json exists
                    if os.path.isfile(filepath_json):
                        # load image and json
                        img = cv2.imread(filepath)
                        try:
                            with open(filepath_json, 'r') as json_file:
                                detections = jsonpickle.decode(json_file.read(), keys=True)
                                
                        except ValueError as e:
                            print("couldn't read json file properly: ", e)

                        # TODO: put img, detections in list or something

                        if subfolder not in self.reid_dataset:
                            self.reid_dataset[subfolder] = []

                        reid_item = SimpleNamespace()
                        reid_item.img = img
                        reid_item.detections = detections
                        reid_item.name = subfolder

                        self.reid_dataset[subfolder].append(reid_item)
        
        print("[green]reid dataset loaded.")



    @staticmethod
    def calculate_matching_error(pts1_matches, pts2_matches):
        
        # todo: compute keypoint locations in world coordinates. Then our error will also be in meters.
        
        #! BUG: sometimes there are 4 matches, but in the visualiser, there are only 3. What is going on?
        
        # compute affine transform
        # https://en.wikipedia.org/wiki/Affine_transformation
        # we want to find A and b to solve: y = Ax + b
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        X = pad(pts1_matches)
        # print("X.shape", X.shape)
        Y = pad(pts2_matches)
        # print("Y.shape", X.shape)
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        affine_transform = lambda x: unpad(np.dot(pad(x), A))
        
        # print("affine_transform(pts1_matches).shape", affine_transform(pts1_matches).shape)
        
        # find mean error
        abs_diff = np.abs(pts2_matches - affine_transform(pts1_matches))
        mean_error = np.mean(abs_diff)
        max_error = np.max(abs_diff)
        median_error = np.median(abs_diff)
        
        # print("mean_error", mean_error)
        # print("median_error", median_error)
        # print("max_error", max_error)
        
        return mean_error, median_error, max_error


    @classmethod
    def find_and_crop_det(cls, img, graph, rotate_180=False):
        # some kind of derivative of: process_detection
        detections_hca_back = graph.exists(Label.hca_back)
        # print("dets1, num. of hca_back: " + str(len(detections_hca_back1)))
        if len(detections_hca_back) < 1:
            print("dets1, hca_back not found")
            return None, None
        
        det_hca_back = detections_hca_back[0]
        
        img_cropped, center_cropped = cls.get_det_img(img, det_hca_back)
        
        # unrotated obb:
        # obb = hca_back.obb_px - hca_back.center_px + center_cropped
        # obb_arr = np.array(obb).astype(int)
        # cv2.drawContours(img_cropped, [obb_arr], 0, (0, 255, 255), 2)
        
        # rotated obb:
        obb2 = cls.rotated_and_centered_obb(det_hca_back.obb_px, det_hca_back.center_px, det_hca_back.tf.rotation, center_cropped)
        obb2_arr = np.array(obb2).astype(int)
        # obb2_list = list(obb2_arr)

        # print("obb2_arr", obb2_arr.shape)

        #! somehow go to list and then back to polygon
        # obb2_poly = Polygon(obb2_arr)
        
        # rotated polygon
        # poly = self.rotated_and_centered_obb(hca_back.polygon_px, hca_back.center_px, hca_back.tf.rotation, center_cropped)
        # poly_arr = np.array(poly.exterior.coords).astype(int)
        # print("poly_arr", poly_arr)
        
        return img_cropped, obb2_arr

    @classmethod
    def rotated_and_centered_obb(cls, obb_or_poly, center, quat, new_center=None, world_coords=False):
        if isinstance(obb_or_poly, Polygon):
            points = np.array(obb_or_poly.exterior.coords)
            points = points[:, :2] # ignore the z value
        else:
            # it is already an array
            points = obb_or_poly
        
        # print("points", points.shape)
        # print("center", center.shape)

        # move obb to (0, 0)
        points_centered = points - center

        # sometimes the angle is in the z axis (basler) and for realsense it is different.
        # this is a hack for that
        angle = cls.ros_quat_to_rad(quat)

        # correction to make longer side along y-axis
        angle = ((0.5 * np.pi) - angle) % np.pi

        # for pixel coords we use the same rotation matrix as in getRotationMatrix2D
        # that we use to rotate the image
        # I don't know why OPENCV uses a different rotation matrix in getRotationMatrix2D
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle),  np.cos(angle)]])

        # for world coordinates, use standard rotation matrix
        if world_coords:
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])

        points_rotated = np.dot(points_centered, rot_mat.T)

        # print("points_rotated", points_rotated.shape)
        if new_center is not None:
            points_rotated += new_center
        
        if isinstance(obb_or_poly, Polygon):
            return Polygon(points_rotated)
        else:
            return points_rotated

    @staticmethod
    def rotate_180(obb_or_poly):
        if isinstance(obb_or_poly, Polygon):
            points = np.array(obb_or_poly.exterior.coords)
        else:
            # it is already an array
            points = obb_or_poly
            
        angle = np.pi
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle),  np.cos(angle)]])
        
        points_rotated = np.dot(points, rot_mat.T)
        
        if isinstance(obb_or_poly, Polygon):
            return Polygon(points_rotated)
        else:
            return points_rotated


    @classmethod
    def get_det_img(cls, img, det):
        center = det.center_px
        center = (int(center[0]), int(center[1]))
        
        height, width = img.shape[:2]
        
        # rotate image around center
        angle_rad = cls.ros_quat_to_rad(det.tf.rotation)
        angle_rad = ((0.5 * np.pi) - angle_rad) % np.pi
        
        # note: getRotationMatrix2D rotation matrix is different from standard rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, np.rad2deg(angle_rad), 1.0)
        img_rot = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        # crop image around center point
        size = 200
        x1 = np.clip(int(det.center_px[0]) - size, 0, width)
        x2 = np.clip(int(det.center_px[0]) + size, 0, width)
        
        y1 = np.clip(int(det.center_px[1]) - size, 0, height)
        y2 = np.clip(int(det.center_px[1]) + size, 0, height)
        
        
        img_cropped = img_rot[y1:y2, x1:x2]
        
        # new center at:
        center_cropped = det.center_px[0] - x1, det.center_px[1] - y1
        
        # debug
        # cv2.circle(img_cropped, (int(center_cropped[0]), int(center_cropped[1])), 6, (0, 0, 255), -1)
        
        return img_cropped, center_cropped
    
    @staticmethod
    def m_to_px(x):
        if isinstance(x, np.ndarray):
            # x is in meters, to show in pixels, scale up...
            #! THIS IS A MASSIVE HACK. WE SHOULD KNOW PIXELS TO METERS
            x = x * 1500
            x = x.astype(int)
            return x
        elif isinstance(x, Polygon):
            x_arr = np.array(x.exterior.coords)
            x_arr = x_arr * 1500
            x_arr = x_arr.astype(int)
            return x_arr

    @staticmethod
    def ros_quat_to_rad(quat):
        quat_to_np = lambda quat : np.array([quat.x, quat.y, quat.z, quat.w])
        rot_mat = Rotation.from_quat(quat_to_np(quat)).as_euler('xyz')

        # print("rot_mat", rot_mat)

        # rotation is only in one dimenseion
        angle = 0.0
        for angle_in_dim in rot_mat:
            if angle_in_dim != 0.0:
                angle = angle_in_dim
        
        return angle
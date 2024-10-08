from __future__ import annotations
import os
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

from context_action_framework.graph_relations import GraphRelations, exists_detection, compute_iou

from context_action_framework.types import Action, Detection, Gap, Label

from helpers import scale_img


class ObjectReId:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_affine_matching_error(pts1_matches, pts2_matches):
        
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
        # print("Y.shape", Y.shape)
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        affine_transform = lambda x: unpad(np.dot(pad(x), A))
        
        # print("affine_transform(pts1_matches).shape", affine_transform(pts1_matches).shape)
        
        # find mean error
        diff = np.linalg.norm(pts2_matches - affine_transform(pts1_matches), axis=0)
        # print("diff", diff)
        mean_error = np.mean(diff)
        max_error = np.max(diff)
        median_error = np.median(diff)
        
        # print("mean_error", mean_error)
        # print("median_error", median_error)
        # print("max_error", max_error)
        
        return mean_error, median_error, max_error, A


    @classmethod
    def find_and_crop_det(cls, img, graph, rotate_180=False, rotate=False, labels=[Label.hca], size=400):
        # some kind of derivative of: process_detection
        chosen_label = None
        for label in labels:
            detections = graph.exists(label)
            if len(detections) >= 1:
                # print("[green]chosen label", label)
                chosen_label = label
                break

        if chosen_label is None:
            print(f"[red]label from list {labels} not found!")
            return None, None
        
        if rotate:
            return cls.crop_and_rotate_det(img, detections[0], size)
        
        return cls.crop_det(img, detections[0], size)


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
        if isinstance(quat, float):
            angle = np.deg2rad(quat)
            print("[blue]debug rotated_and_centered_obb angle", angle)
        else:
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
    def crop_and_rotate_det(cls, img, det, size=400):
        center = det.center_px
        center = (int(center[0]), int(center[1]))
        
        # rotate image around center
        if det.tf is not None:
            angle_rad = cls.ros_quat_to_rad(det.tf.rotation)
        else:
            angle_rad = np.deg2rad(det.angle_px)
            print("[blue]debug angle object_reid.py: angle_px", det.angle_px)
        
        angle_rad = ((0.5 * np.pi) - angle_rad) % np.pi
        
        # note: getRotationMatrix2D rotation matrix is different from standard rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, np.rad2deg(angle_rad), 1.0)
        img_rot = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
        img_cropped, center_cropped = cls.crop_det(img_rot, det, size=size)

        # unrotated obb:
        # obb = hca_back.obb_px - hca_back.center_px + center_cropped
        # obb_arr = np.array(obb).astype(int)
        # cv2.drawContours(img_cropped, [obb_arr], 0, (0, 255, 255), 2)
        
        # rotated obb:
        if det.tf is not None:
            obb2 = cls.rotated_and_centered_obb(det.obb_px, det.center_px, det.tf.rotation, center_cropped)
        else:
            obb2 = cls.rotated_and_centered_obb(det.obb_px, det.center_px, det.angle_px, center_cropped)

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
    def crop_det(cls, img, det, size=400):

        height, width = img.shape[:2]

        # crop image around center point
        # half_size = int(size/2)
        # x1 = np.clip(int(det.center_px[0]) - half_size, 0, width)
        # x2 = np.clip(int(det.center_px[0]) + half_size, 0, width)
        
        # y1 = np.clip(int(det.center_px[1]) - half_size, 0, height)
        # y2 = np.clip(int(det.center_px[1]) + half_size, 0, height)
        # img_cropped = img[y1:y2, x1:x2]
        
        # # new center at:
        # center_cropped = det.center_px[0] - x1, det.center_px[1] - y1
        
        # # debug
        # # cv2.circle(img_cropped, (int(center_cropped[0]), int(center_cropped[1])), 6, (0, 0, 255), -1)
        
        # return img_cropped, center_cropped
    
        half_size = size // 2  # Using integer division directly

        # Original crop coordinates
        x_center, y_center = int(det.center_px[0]), int(det.center_px[1])
        x1 = x_center - half_size
        x2 = x_center + half_size
        y1 = y_center - half_size
        y2 = y_center + half_size

        # Calculate padding requirements
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - width)
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - height)

        # Pad image if necessary
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
            # Update coordinates to match the padding
            x1 += pad_left
            x2 += pad_left
            y1 += pad_top
            y2 += pad_top
        else:
            img_padded = img

        img_cropped = img_padded[y1:y2, x1:x2]
        center_cropped = x_center - x1, y_center - y1

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
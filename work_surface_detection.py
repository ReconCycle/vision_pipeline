import sys, os
os.environ["DLClight"] = "True" # no gui

from dlc.config import *
import numpy as np
from dlc.infer import Inference
from config_default import *
import cv2
import itertools


class WorkSurfaceDetection:
    def __init__(self, img):

        # get corners of work surface
        inference = Inference(cfg.dlc_config_file)

        if isinstance(img, str):
            img = np.array(cv2.imread(img))
        else:
            img = np.array(img)
        
        corners_x, corners_y, corners_likelihood, corner_labels, bpts2connect = inference.get_pose(img)
        self.corners_likelihood = corners_likelihood
        self.corner_labels = corner_labels

        # these will be populated below...
        self.coord_transform = None
        self.corners_in_pixels = None #! depracate this
        self.corners_in_meters = None #! depracate this

        self.mounts_in_pixels = None #! depracate this

        self.points_dict = {}

        if len(corners_x) >= 4 and len(corners_y) >= 4 and len(corner_labels) >= 4:
            self.points_dict = {
                'corner1': None,
                'corner2': None,
                'corner3': None,
                'corner4': None,
                'calibrationmount1': None,
                'calibrationmount2': None,
            }

            # populate the true_corner_labels dictionary with the [x, y] values
            for true_corner_label in self.points_dict.keys():
                if true_corner_label in corner_labels:
                    index = corner_labels.index(true_corner_label)
                    print(true_corner_label, index)
                    self.points_dict[true_corner_label] = np.array([corners_x[index, 0], corners_y[index, 0]])

            # count how many corners are detected
            num_corners_detected = 0
            corner_keys = ['corner1', 'corner2', 'corner3', 'corner4']
            for dict_key in corner_keys:
                if self.points_dict[dict_key] is not None:
                    num_corners_detected += 1

            for key_1, key_2 in itertools.combinations(corner_keys, 2):
                print(key_1, key_2)
                # check distance in pixels between corner points.
                # If two of the corners are the same corner then the distance between the two corners will be less than 30 pixels.
                if np.linalg.norm(self.points_dict[key_1]-self.points_dict[key_2]) < 30:
                    print("Corners on top of each other!", key_1, key_2)
            
            print("corners detected:", num_corners_detected)

            

            self.corners_in_pixels = np.array([self.points_dict["corner1"],
                                                self.points_dict["corner2"],
                                                self.points_dict["corner3"],
                                                self.points_dict["corner4"]])

            self.mounts_in_pixels = np.array([self.points_dict["calibrationmount1"],
                                                self.points_dict["calibrationmount2"]])

            self.corners_in_meters = np.array([[0, 0], [0.6, 0], [0.6, 0.6], [0, 0.6]])

            self.pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
            self.unpad = lambda x: x[:, :-1]

            X = self.pad(self.corners_in_pixels)
            Y = self.pad(self.corners_in_meters)

            # Solve the least squares problem X * A = Y
            # to find our transformation matrix A
            A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
            self.coord_transform = lambda x: self.unpad(np.dot(self.pad(x), A))
            print("Target:", self.corners_in_meters)
            print("Result:", self.coord_transform(self.corners_in_pixels))
            print("Max error:", np.abs(self.corners_in_meters - self.coord_transform(self.corners_in_pixels)).max())

            first_corner_pixels = np.array([self.corners_in_pixels[0]])
            print("first_corner_meters", self.coord_transform(first_corner_pixels))

            # now convert pixels to ints
            self.corners_in_pixels = np.around(self.corners_in_pixels).astype(int)
            self.mounts_in_pixels = np.around(self.mounts_in_pixels).astype(int)
        else:
            raise ValueError("too few corners detected. Number of corners detected:", len(corners_x), len(corners_y), len(corner_labels))

    def pixels_to_meters(self, coords):
        if isinstance(coords, tuple) or len(coords.shape) == 1:
            # single coordinate pair.
            return self.coord_transform(np.array([coords]))[0]
        else:
            # assume array of coordinate pairs. 
            # Each row contains a coordinate (x, y) pair
            return self.coord_transform(coords)
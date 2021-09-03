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
        self.corner_labels = corner_labels # ['corner1', 'corner2', 'corner3', 'corner4', 'calibrationmount1', 'calibrationmount2']

        # these will be populated below...
        self.coord_transform = None

        print("self.corners_likelihood", self.corners_likelihood, type(self.corners_likelihood))
        print("self.corner_labels", self.corner_labels, type(self.corner_labels))

        corners = ['corner1', 'corner2', 'corner3', 'corner4']
        calibrationmounts = ['calibrationmount1', 'calibrationmount2']

        self.points_px_dict = {
            'corner1': None,
            'corner2': None,
            'corner3': None,
            'corner4': None,
            'calibrationmount1': None,
            'calibrationmount2': None,
        }
        
        self.points_m_dict = {
            'corner1': [0, 0],
            'corner2': [0.6, 0],
            'corner3': [0.6, 0.6],
            'corner4': [0, 0.6]
        }

        if len(corners_x) >= 3 and len(corners_y) >= 3 and len(corner_labels) >= 3:

            # populate the true_corner_labels dictionary with the [x, y] values
            for true_corner_label in self.points_px_dict.keys():
                if true_corner_label in corner_labels:
                    index = corner_labels.index(true_corner_label)
                    self.points_px_dict[true_corner_label] = np.array([corners_x[index, 0], corners_y[index, 0]])

            print("self.points_px_dict", self.points_px_dict)

            # check distance in pixels between corner points.
            # If two of the corners are the same corner (e.g. less than 30px apart) remove one of the corners
            for key_1, key_2 in itertools.combinations(self.points_px_dict.keys(), 2):
                if (self.points_px_dict[key_1] is not None and 
                    self.points_px_dict[key_2] is not None and 
                    np.linalg.norm(self.points_px_dict[key_1]-self.points_px_dict[key_2]) < 30):
                    print("Corners or calibration mount on top of each other!", key_1, key_2)

                    # remove one of the detected points
                    if (key_1 in corners and key_2 in corners) or (key_1 in calibrationmounts and key_2 in calibrationmounts):
                        self.points_px_dict[key_2] = None
                    elif key_1 in corners and key_2 in calibrationmounts:
                        # something is quite wrong here.
                        # assume that the corners are more likely to be detected correctly.
                        self.points_px_dict[key_2] = None
                    elif key_2 in corners and key_1 in calibrationmounts:
                        # something is quite wrong here.
                        # assume that the corners are more likely to be detected correctly.
                        self.points_px_dict[key_1] = None    

            print("new self.points_px_dict", self.points_px_dict)

            # create arrays for affine transform
            corners_in_pixels = []
            corners_in_meters = []
            for key, value in self.points_px_dict.items():
                if value is not None and key in self.points_m_dict and self.points_m_dict[key] is not None:
                    corners_in_pixels.append(value)
                    corners_in_meters.append(self.points_m_dict[key])

            corners_in_pixels = np.array(corners_in_pixels)
            corners_in_meters = np.array(corners_in_meters)

            # count how many corners are detected
            num_corners_detected = len(corners_in_pixels)
            print("corners detected:", num_corners_detected)

            self.pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
            self.unpad = lambda x: x[:, :-1]

            X = self.pad(corners_in_pixels)
            Y = self.pad(corners_in_meters)

            # Solve the least squares problem X * A = Y
            # to find our transformation matrix A
            A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
            self.coord_transform = lambda x: self.unpad(np.dot(self.pad(x), A))
            print("Target:", corners_in_meters)
            print("Result:", self.coord_transform(corners_in_pixels))
            print("Max error:", np.abs(corners_in_meters - self.coord_transform(corners_in_pixels)).max())

            first_corner_pixels = np.array([corners_in_pixels[0]])
            print("first_corner_meters", self.coord_transform(first_corner_pixels))

            # now convert pixels to ints
            # self.corners_in_pixels = np.around(self.corners_in_pixels).astype(int)
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
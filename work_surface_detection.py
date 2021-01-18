import sys, os
# having trouble importing the yolact directory. Doing this as a workaround:
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deeplabcut'))
os.environ["DLClight"] = "True" # no gui

from config import *
import numpy as np
from infer import Inference


class WorkSurfaceDetection:
    def __init__(self, model_config_path, img):

        # get corners of work surface
        inference = Inference(model_config_path)
        corners_x, corners_y, corners_likelihood, corner_labels, bpts2connect = inference.infer_from_img(img)
        self.corners_likelihood = corners_likelihood
        self.corner_labels = corner_labels

        # these will be populated below...
        self.coord_transform = None
        self.corners_in_pixels = None
        self.corners_in_meters = None

        if len(corners_x) == 4 and len(corners_y) == 4 and len(corner_labels) == 4:
            # all 4 corners have been detected
            self.corners_in_pixels = np.hstack((corners_x, corners_y))
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
        else:
            raise ValueError("too many/few corners detected.")

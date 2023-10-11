import cv2
import numpy as np
from shapely.geometry import Polygon, Point


class ObjectSift():
    def __init__(self) -> None:
        self.sift = cv2.SIFT_create()

    def calculate_sift(self, img_cropped, obb=None, visualise=False, vis_id=1):

        keypoints, descriptors = self.sift.detectAndCompute(img_cropped, None)

        if obb is not None:
            keypoints_in_poly = []
            descriptors_in_poly = []

            if keypoints is None:
                print("keypoints is None!")
                return [], []
            
            if descriptors is None:
                print("descriptors is None!")
                return [], []
            
            obb_poly = Polygon(obb)

            # only include keypoints that are inside the obb
            for keypoint, descriptor in zip(keypoints, descriptors):
                if obb_poly.contains(Point(*keypoint.pt)):
                    keypoints_in_poly.append(keypoint)
                    descriptors_in_poly.append(descriptor)
        else:
            keypoints_in_poly = keypoints
            descriptors_in_poly = descriptors
        
        # descriptors is an array, keypoints is a list
        descriptors_in_poly = np.array(descriptors_in_poly)

        return keypoints_in_poly, descriptors_in_poly

import sys, os
# os.environ["DLClight"] = "True" # no gui
# from dlc.config import *
# from dlc.infer import Inference
import numpy as np
from config_default import *
import cv2
import itertools


def scale_img(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def order_points(pts):
    """order points in: top-left, top-right, bottom-right, and bottom-left
    
    Source: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    Args:
        pts (arr[*, 3]): array of points, where each point is [x, y, r]
        or:
        pts (arr[*, 2]): array of points, where each point is [x, y]

    Returns:
        arr[4,2]: corners
    """
    points_dim = pts.shape[1]
    
    print("pts.shape", pts.shape)
    
    corners = np.zeros((4, points_dim), dtype = "float32")
    
    # determine if pts are [x, y, r] or [x, y]
    summing_pts = pts
    if points_dim == 3:
        summing_pts = pts[:, :2]
        
    s = np.sum(summing_pts, axis = 1)
    
    print("sum", s)
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    corners[0] = pts[np.argmin(s)]
    corners[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(summing_pts, axis = 1)
    corners[1] = pts[np.argmin(diff)]
    corners[3] = pts[np.argmax(diff)]
    
    return corners


class WorkSurfaceDetection2:
    def __init__(self, img, debug=False):

        self.bolt_distance = 53

        if isinstance(img, str):
            img = np.array(cv2.imread(img))
        else:
            img = np.array(img)

        self.coord_transform = None

        self.points_px_dict = {
            'corner0': None,
            'corner1': None,
            'corner2': None,
            'corner3': None,
            'bolt0': None,
            'bolt1': None,
            'bolt2': None,
            'bolt3': None,
            'calibrationmount0': None,
            'calibrationmount1': None,
            'calibrationmount2': None,
            'calibrationmount3': None,
        }
                
        self.points_m_dict = {
            'corner0': [0, 0],                       # top-left
            'corner1': [0.6, 0],                     # top-right
            'corner2': [0.6, 0.6],                   # bottom-right
            'corner3': [0, 0.6],                     # bottom-left
            'bolt0': [0.035, 0.035],                 # top-left
            'bolt1': [0.6 - 0.035, 0.035],           # top-right
            'bolt2': [0.6 - 0.035, 0.6 - 0.035],     # bottom-right
            'bolt3': [0.035, 0.6 - 0.035],           # bottom-left            
            'calibrationmount0': [0.3, 0.03],        # top-center
            'calibrationmount1': [0.6 - 0.03, 0.3],  # left-center
            'calibrationmount2': [0.3, 0.6 - 0.03],  # bottom-center
            'calibrationmount3': [0.03, 0.3],        # right-center
        }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Finds circles in a grayscale image using the Hough transform
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                    param1=200,param2=10,minRadius=10,maxRadius=20)

        bolts = None

        if circles is not None:
            circles = np.array([[x, y, r] for (x, y, r) in circles[0]])

            bolts = order_points(circles)
            
            # write bolts to dictionary
            for i, bolt in enumerate(bolts):
                self.points_px_dict["bolt"+ str(i)] = bolt

            # compute which circle corresponds to calibration mount, if any
            # and write calibration mount(s) to dictionary
            midpoints = []
            midpoints.append((bolts[0] + bolts[1])/2)
            midpoints.append((bolts[1] + bolts[2])/2)
            midpoints.append((bolts[2] + bolts[3])/2)
            midpoints.append((bolts[3] + bolts[0])/2)

            epsilon = 20
            for [x, y, r] in circles:
                for i, midpoint in enumerate(midpoints):
                    if np.linalg.norm(np.array([x, y])-midpoint[:2]) < epsilon:
                        self.points_px_dict["calibrationmount" + str(i)] = [x, y, r]
                        

        print("circles", circles)
        print("bolts", bolts)
        print("midpoints", midpoints)
        
        # create arrays for affine transform
        points_px_arr = []
        points_m_arr = []
        for key, value in self.points_px_dict.items():
            if value is not None and key in self.points_m_dict and self.points_m_dict[key] is not None:
                points_px_arr.append(value[:2]) # only [x, y] not radius
                points_m_arr.append(self.points_m_dict[key])

        points_px_arr = np.array(points_px_arr)
        points_m_arr = np.array(points_m_arr)

        print("points_px_arr", points_px_arr)
        print("points_m_arr", points_m_arr)

        self.pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        self.unpad = lambda x: x[:, :-1]

        X = self.pad(points_px_arr)
        Y = self.pad(points_m_arr)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        self.coord_transform = lambda x: self.unpad(np.dot(self.pad(x), A))
        
        print("A.shape", A.shape)
        A_inv = np.linalg.solve(A.T.dot(A), A.T)
        
        print("A_inv.shape", A_inv.shape)
        self.coord_transform_inv = lambda x: self.unpad(np.dot(self.pad(x), A_inv))
        
        
        print("Target:", points_m_arr)
        print("Result:", self.pixels_to_meters(points_px_arr))
        print("Max error:", np.abs(points_m_arr - self.pixels_to_meters(points_px_arr)).max())

        print("self.meters_to_pixels", self.meters_to_pixels(np.array([0.0, 0.0])))
        
        for key, value in self.points_px_dict.items():
            if value is None and key.startswith("corner"):
                print("self.points_px_dict[key]", self.points_px_dict[key])
                self.points_px_dict[key] = self.meters_to_pixels(np.array(self.points_m_dict[key]))
                
        print("self.points_px_dict", self.points_px_dict)

        # draw stuff on image
        if circles is not None and debug:
            # draw all detections in green
            for (x, y, r) in circles:
                # Draw the circle in the output image
                cv2.circle(img, (int(x), int(y)), int(r), (0,255,0), 3)
                # Draw a rectangle(center) in the output image
                cv2.rectangle(img, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0,255,0), -1)
        
            # for bolts and calibration mounts
            for key in self.points_px_dict:
                if self.points_px_dict[key] is not None:
                    if len(self.points_px_dict[key]) == 3:
                        x, y, r = self.points_px_dict[key]
                    else:
                        x, y = self.points_px_dict[key]
                        r = 5
                    cv2.circle(img, (int(x), int(y)), int(r), (0,0,255), 3)
                    # Draw a rectangle(center) in the output image
                    cv2.rectangle(img, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0,0,255), -1)


            # cv2.imshow("0", scale_img(blur))
            cv2.imshow("1", scale_img(img))
            cv2.waitKey(0)

    def pixels_to_meters(self, coords):
        if isinstance(coords, tuple) or len(coords.shape) == 1:
            # single coordinate pair.
            return self.coord_transform(np.array([coords]))[0]
        else:
            # assume array of coordinate pairs. 
            # Each row contains a coordinate (x, y) pair
            return self.coord_transform(coords)
        
    def meters_to_pixels(self, coords):
        if isinstance(coords, tuple) or len(coords.shape) == 1:
            # single coordinate pair.
            return self.coord_transform_inv(np.array([coords]))[0]
        else:
            # assume array of coordinate pairs. 
            # Each row contains a coordinate (x, y) pair
            return self.coord_transform_inv(coords)
        
if __name__ == '__main__':
    img = cv2.imread("data_full/dlc/dlc_work_surface_jsi_05-07-2021/labeled-data/raw_work_surface_jsi_08-07-2021/img000.png")
    work_surface_det2 = WorkSurfaceDetection2(img, debug=True)
    
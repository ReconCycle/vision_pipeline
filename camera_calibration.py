'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

'''

import numpy as np
import cv2
import rospy
import glob
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json
from helpers import NumpyEncoder

class CameraCalibration(object):
    def __init__(self, node_name):
        self.node_name = node_name

        # rospy.init_node(node_name)

        # rospy.loginfo("Starting node " + str(node_name))

        # rospy.on_shutdown(self.cleanup)
        
        # Initialize output folder path to current terminal directory
        self.img_dirs = [
            # Path("~/vision_pipeline/saves/2024-07-30_09:10:37_realsense_calibration").expanduser(),
            Path("~/vision_pipeline/saves/2024-07-30_09:47:16_realsense_calibration").expanduser(),
        ] 
        

        # Initialize findChessboardCorners and drawChessboardCorners parameters
        self.patternSize_columns = 10
        self.patternSize_rows = 7
        self.square_size = 0.0201
        
        # Initialize cornerSubPix parameters
        self.winSize_columns = 11
        self.winSize_rows = 11
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

        img_names = []
        for img_dir in self.img_dirs:
            if not img_dir.exists():
                raise ValueError(f"path doesn't exist! {img_dir}")
            img_names_for_dir = img_dir.glob("*.jpg")
            img_names.extend([f for f in img_names_for_dir if "depth" not in f.stem])
    
        pattern_size = (self.patternSize_columns, self.patternSize_rows)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= self.square_size
    
        obj_points = []
        img_points = []
        i = 0

        img_names_pbar = tqdm(img_names)
        for fname in img_names_pbar:
            img_names_pbar.set_description(f"processing {fname.name}")
            img = cv2.imread(str(fname))
            
            if img is None:
                print("Failed to load", fname)
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (self.winSize_columns, self.winSize_rows), (-1, -1), self.criteria)

                # cv2.drawChessboardCorners(img, (self.patternSize_columns,self.patternSize_rows), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)
    
            if not ret:
                print('chessboard not found')
                continue
    
            img_points.append(corners2)
            obj_points.append(pattern_points)
    
        # calculate camera distortion
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        
        tot_error = 0
        
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
            error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
            tot_error += error
        
        mean_error = tot_error/len(obj_points)

        print(f"\nRMS: {rms}\n")
        print(f"camera matrix:\n{camera_matrix}\n")
        print(f"distortion coefficients: {dist_coefs.ravel()}\n")
        print(f"mean error: {mean_error}\n")

        output_file_path = self.img_dirs[-1] / Path("camera_calibration.json")
        # output_file = open(output_file_path, "w+")
        # output_file.write(f"\nRMS: {rms}\n")
        # output_file.write(f"camera matrix:\n{camera_matrix}\n")
        # output_file.write(f"distortion coefficients: {dist_coefs.ravel()}\n")
        # output_file.write(f"mean error: {mean_error}\n")
        print("camera_matrix.shape", camera_matrix.shape)
        print("dist_coefs.shape", dist_coefs.shape)
        data_dict = {
            "rms": rms,
            "mtx": camera_matrix,
            "dist": dist_coefs,
            "mean_error": mean_error
        }
        with open(output_file_path, 'w') as f:
            json.dump(data_dict, f, indent=4, sort_keys=True,
              separators=(', ', ': '), ensure_ascii=False,
              cls=NumpyEncoder)

        print("saved output to:", output_file_path)

    def cleanup(self):
        print ("Shutting down vision node.")
        cv2.destroyAllWindows()

def main(args):
    try:
        node_name = "CameraCalibration"
        CameraCalibration(node_name)
        
    except KeyboardInterrupt:
        print ("Shutting down CameraCalibration node.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
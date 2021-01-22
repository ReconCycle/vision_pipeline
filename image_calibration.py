#!/usr/bin/env python
import numpy as np
import cv2
import os
import argparse
import yaml
from glob import glob
import time
import regex as re
from config import *


class ImageCalibration:
    def __init__(self, camera_calibration_file=None):
        self.calibration = {}
    
        if camera_calibration_file is None:
            camera_calibration_file = cfg.camera_calibration_file
    
        self.load_calibration_file(camera_calibration_file)

        self.undistort = cfg.camera_parameters.undistort

        self.resize = cfg.camera_parameters.resize
        self.resized_resolution = cfg.camera_parameters.resized_resolution

        self.add_borders = cfg.camera_parameters.add_borders
        self.camera_new_resolution = cfg.camera_parameters.camera_new_resolution
        self.camera_offsets = cfg.camera_parameters.camera_offsets

        self.crop = cfg.camera_parameters.crop
        self.crop_margins = cfg.camera_parameters.crop_margins

    def load_calibration_file(self, calibration_file):
        if calibration_file is not None:
            with open(calibration_file) as fr:
                self.calibration = yaml.load(fr)


    def get_images(self, input_dir):
        images = [img for img in os.listdir(input_dir) if img.endswith(".png")]
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        images = [os.path.join(input_dir, image) for image in images]
        return images

    def undistort_imgs(self, input, output=None):

        def undistort_img(img):
            print("input img.shape", img.shape)

            t0 = time.time()

            if self.resize:
                # img is now at self.camera_native_resolution resolution
                img = cv2.resize(img, self.resized_resolution, interpolation=cv2.INTER_AREA)
                print("resized:", img.shape)
                # img is now at self.resized_resolution resolution

            if self.undistort:
                k_undistort = np.array(self.calibration['camera_matrix'])
                img = cv2.undistort(img, np.array(self.calibration['camera_matrix']), np.array(self.calibration['dist_coefs']),
                                        newCameraMatrix=k_undistort)
                print("undistorted img.shape", img.shape)

            if self.add_borders:
                # add camera offsets to image as black
                # add black to image to make image have native camera resolution
                x_offset, y_offset = self.camera_offsets
                # if self.camera_offsets is not (0, 0):
                #     print("Applying camera offsets")

                if self.camera_new_resolution is not None:
                    # print("camera_native_resolution", self.camera_native_resolution, img.shape)
                    x_native, y_native = self.camera_new_resolution
                else:
                    x_native = img.shape[1]
                    y_native = img.shape[0]

                # top, bottom, left, right
                border_top = y_offset
                border_bottom = y_native - img.shape[0] - y_offset
                border_left = x_offset
                border_right = x_native - img.shape[1] - x_offset
                black = [0, 0, 0]
                if border_top > 0 or border_bottom > 0 or border_left > 0 or border_right > 0:
                    print("border_top", border_top, "border_bottom", border_bottom, "border_left", border_left, "border_right", border_right)
                    img = cv2.copyMakeBorder(img, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=black)
                    print("applied borders:", img.shape)


            if self.crop:
                # crop image
                margin_top, margin_right, margin_bottom, margin_left = self.crop_margins
                img = img[margin_top:-1 - margin_bottom, margin_left:-1 - margin_right]
                # cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
                # cv2.imshow("cropped", img)
                # cv2.waitKey(0)
                # return  # debug

            t1 = time.time()
            print("time taken for undistort:", t1 - t0)

            return img

        if isinstance(input, str):
            if os.path.isdir(input):
                input = self.get_images(input)
            else:
                input = [input]

        if isinstance(input, list):
            for input_img in input:
                img = cv2.imread(input_img)
                if img is None:
                    print("Failed to load " + input_img)
                    continue

                resized_img = undistort_img(img)

                if output is not None:
                    name, ext = os.path.splitext(os.path.basename(input_img))
                    cv2.imwrite(os.path.join(output, name + ext), resized_img)

                if len(input) == 1:
                    return resized_img
        elif isinstance(input, np.ndarray):
            return undistort_img(input)


    def calibrate(self, input, output, pattern_size=(8, 6), debug_dir=None, framestep=20):

        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        # pattern_points *= square_size

        obj_points = []
        img_points = []
        h, w = 0, 0
        i = -1

        if isinstance(input, list):
            # assume folder of png files.
            source = input
        elif isinstance(input, str):
            if os.path.isdir(input):
                source = self.get_images(input)
        else:
            source = cv2.VideoCapture(input)

        while True:
            i += 1
            if isinstance(source, list):
                # glob
                if i == len(source):
                    break
                img = cv2.imread(source[i])
            else:
                # cv2.VideoCapture
                retval, img = source.read()
                if not retval:
                    break
                if i % framestep != 0:
                    continue

            print('Searching for chessboard in frame ' + str(i) + '...'),
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            found, corners = cv2.findChessboardCorners(img, pattern_size, flags=cv2.CALIB_CB_FILTER_QUADS)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            if debug_dir:
                img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
                cv2.imwrite(os.path.join(debug_dir, '%04d.png' % i), img_chess)
            if not found:
                print('not found')
                continue
            img_points.append(corners.reshape(1, -1, 2))
            obj_points.append(pattern_points.reshape(1, -1, 3))

            print('ok')

        print('\nPerforming calibration...')
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
        print("RMS:", rms)
        print("camera matrix:\n", camera_matrix)
        print("distortion coefficients: ", dist_coefs.ravel())

        calibration = {'rms': rms, 'camera_matrix': camera_matrix.tolist(), 'dist_coefs': dist_coefs.tolist()}
        output_file = output
        if os.path.isdir(output):
            output_file = os.path.join(output, "calibration.yaml")
        with open(output_file, 'w') as fw:
            yaml.dump(calibration, fw)


if __name__ == '__main__':
    # image_calibration = ImageCalibration(calibration_file="data/kalo_v2_calibration/calibration_2k.yaml",
    #                           basler_config_file="basler_config.yaml")

    # # step 1: resize to self.resized_resolution
    # # step 2: undistort resized images
    # # step 3: crop using self.crop_margins
    # image_calibration.resize(calibration.get_images("data/kalo_v2_test_imgs_distorted"),
    #                    output="data/kalo_v2_test_imgs_undistorted")

    # To redo the calibration, run:
    image_calibration = ImageCalibration(calibration_file="data/kalo_v2_calibration/calibration_1450x1450.yaml",
                              basler_config_file="basler_config.yaml")
    # image_calibration.undistort_imgs('data/calibration_2900x2900', output='data/calibration_undistorted_19-11-2020')
    image_calibration.calibrate('data/calibration_undistorted_19-11-2020', 'data/kalo_v2_calibration/calibration_1450x1450_undistorted.yaml')

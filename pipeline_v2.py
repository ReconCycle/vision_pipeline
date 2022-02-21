import os
import numpy as np
import time
import cv2
# import commentjson

from image_calibration import ImageCalibration
from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection

from helpers import *


class Pipeline:
    def __init__(self):
        # 1. load camera calibration files
        self.calibration = ImageCalibration()

        # 2. work surface coordinates, will be initialised on first received image
        self.worksurface_detection = None

        # 3. object detection
        self.object_detection = ObjectDetection()
        # self.class_names = self.object_detection.dataset.class_names


    def process_img(self, img):
        if isinstance(img, str):
            print("img path:", img)
            img = cv2.imread(img)
        
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img)

        # print("img.shape", img.shape)
        
        labelled_img, detections = self.object_detection.get_prediction(img, self.worksurface_detection)

        return labelled_img, detections


if __name__ == '__main__':

    pipeline = Pipeline()

    # pipeline
    show_imgs = True

    # Iterate over images and run:
    # we can use a directory here or a single image /163.png
    # img_path = "data_full/dlc/dlc_work_surface_jsi_05-07-2021/labeled-data/raw_work_surface_jsi_08-07-2021"
    img_path = "/home/sruiz/datasets/labelme/2022-02-17_kalo_tracking_2"
    save_path = None # set to None to not save
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs = get_images(img_path)

    for img_p in imgs:
        labeled_img, detections = pipeline.process_img(img_p)
        
        t_start = time.time()
        if show_imgs:
            cv2.imshow('labeled_img', scale_img(labeled_img))

        waitkey = cv2.waitKey(1)

        # break
        if waitkey == 27:
            break  # esc to quit
        elif waitkey == ord('p'): # pause
            cv2.waitKey(-1)  # wait until any key is pressed

        if save_path is not None:
            save_file_path = os.path.join(save_path, os.path.basename(img_p))
            print("saving!", save_file_path)
            cv2.imwrite(save_file_path, labeled_img)
            
        fps_imshow = 1.0 / (time.time() - t_start)
        print("fps_imshow", fps_imshow)

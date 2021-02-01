import sys, os
os.environ["DLClight"] = "True" # no gui
os.environ['KMP_WARNINGS'] = 'off' # some info about KMP_AFFINITY
from dlc.config import *
# from infer import Inference
from image_calibration import ImageCalibration
from work_surface_detection import WorkSurfaceDetection
import numpy as np
from object_detection import ObjectDetection
from helpers import *
import graphics
import cv2
import torch
import time

from yolact.data import cfg


import json

class Pipeline:
    def __init__(self):
        # 1. load camera calibration files
        self.calibration = ImageCalibration()

        # 2. get work surface coordinates

        #! Todo, accept image as well as image path for work surface detection
        # self.worksurface_detection = WorkSurfaceDetection("/home/sruiz/datasets/deeplabcut/kalo_v2_imgs_20-11-2020/0.png")

        # 3. object detection
        self.object_detection = ObjectDetection()

    def process_img(self, img):
        with torch.no_grad():
            if isinstance(img, str):
                print("img path:", img)
                img = cv2.imread(img)
            frame = torch.from_numpy(img).cuda().float()
        
        print("frame.shape", frame.shape)

        preds = self.object_detection.get_prediction(frame)
        classes, scores, boxes, masks, obb_corners, obb_centers, obb_rot_quarts, num_dets_to_consider = self.object_detection.post_process(preds)

        # todo: write a function that converts from px coordinates to meters using worksurface_detection

        # labelled_img = graphics.get_labelled_img(frame, classes, scores, boxes, masks, obb_corners, obb_centers, num_dets_to_consider, worksurface_detection=self.worksurface_detection)
        labelled_img = graphics.get_labelled_img(frame, classes, scores, boxes, masks, obb_corners, obb_centers, num_dets_to_consider)
        print("labelled_img.shape", labelled_img.shape)

        # todo: the graphics part should accept a list of detections like this below instead of what it is doing now
        detections = []
        for i in np.arange(len(classes)):
            detection = {}
            detection["class_name"] = cfg.dataset.class_names[classes[i]]
            detection["score"] = float(scores[i])
            detection["obb_corners"] = obb_corners[i].tolist()
            detection["obb_center"] = obb_centers[i].tolist()
            detection["obb_rot_quart"] = obb_rot_quarts[i].tolist()
            detections.append(detection)

        return labelled_img, detections



if __name__ == '__main__':

    pipeline = Pipeline()

    # pipeline
    show_imgs = False

    # Iterate over images and run:
    img_path = "/home/sruiz/datasets/deeplabcut/kalo_v2_imgs_20-11-2020/163.png"  # we can use a directory here or a single image /163.png
    save_path = 'output' # set to None to not save
    imgs = get_images(img_path)
    if show_imgs:
        cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
    frame_count = 0
    for img_p in imgs:
        labeled_img, detections = pipeline.process_img(img_p)
        
        if show_imgs:
            cv2.imshow('labeled_img', labeled_img)

        waitkey = cv2.waitKey(1)

        # break
        if waitkey == 27:
            break  # esc to quit
        elif waitkey == ord('p'): # pause
            cv2.waitKey(-1)  # wait until any key is pressed

        t_now = time.time()
        # only start calculating avg fps after the first frame
        if frame_count == 0:
            t_start = time.time()
        else:
            print("avg. FPS:", frame_count / (t_now - t_start))
        frame_count += 1

        if save_path is not None:
            save_file_path = os.path.join(save_path, os.path.basename(img_p))
            print("saving!", save_file_path)
            cv2.imwrite(save_file_path, labeled_img)




    


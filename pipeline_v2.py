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

        # 2. work surface coordinates, will be initialised on first received image
        self.worksurface_detection = None

        # 3. object detection
        self.object_detection = ObjectDetection()


    def process_img(self, img, fps=None):
        with torch.no_grad():
            if isinstance(img, str):
                print("img path:", img)
                img = cv2.imread(img)
            
            if torch.cuda.is_available():
                frame = torch.from_numpy(img).cuda().float()
            else:
                frame = torch.from_numpy(img).float()
        
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img)

        print("frame.shape", frame.shape)

        preds = self.object_detection.get_prediction(frame)
        classes, scores, boxes, masks, obb_corners, obb_centers, obb_rot_quats, num_dets_to_consider = self.object_detection.post_process(preds)

        labelled_img = graphics.get_labelled_img(frame, classes, scores, boxes, masks, obb_corners, obb_centers, num_dets_to_consider, fps=fps, worksurface_detection=self.worksurface_detection)

        # todo: the graphics part should accept a list of detections like this below instead of what it is doing now
        detections = []
        for i in np.arange(len(classes)):
            if obb_corners[i] is not None:
                detection = {}
                detection["class_name"] = cfg.dataset.class_names[classes[i]]
                detection["score"] = float(scores[i])
                detection["obb_corners"] = self.worksurface_detection.pixels_to_meters(obb_corners[i]).tolist()
                detection["obb_center"] = self.worksurface_detection.pixels_to_meters(obb_centers[i]).tolist()
                detection["obb_rot_quat"] = obb_rot_quats[i].tolist()
                detections.append(detection)

        return labelled_img, detections


if __name__ == '__main__':

    pipeline = Pipeline()

    # pipeline
    show_imgs = False

    # Iterate over images and run:
    img_path = "./camera_images_3"  # we can use a directory here or a single image /163.png
    save_path = './labelled_images_3' # set to None to not save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs = get_images(img_path)
    if show_imgs:
        cv2.namedWindow("labeled_img", cv2.WINDOW_NORMAL)
    frame_count = 0
    fps = None
    t_prev = None
    for img_p in imgs:
        labeled_img, detections = pipeline.process_img(img_p, fps)
        
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
            # if t_prev is not None and t_now - t_prev > 0:
                # fps = 1 / (t_now - t_prev)
            print("avg. FPS:", frame_count / (t_now - t_start))
        frame_count += 1
        t_prev = t_now

        if save_path is not None:
            save_file_path = os.path.join(save_path, os.path.basename(img_p))
            print("saving!", save_file_path)
            cv2.imwrite(save_file_path, labeled_img)




    


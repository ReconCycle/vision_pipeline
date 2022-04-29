import os
import numpy as np
import time
import cv2
from rich import print
import json

from image_calibration import ImageCalibration
from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection
from graph_relations import GraphRelations

from helpers import scale_img, get_images, EnhancedJSONEncoder
from config import load_config


class Pipeline:
    def __init__(self):
        self.config = load_config()
        print("config", self.config)
        
        # 1. load camera calibration files
        self.calibration = ImageCalibration(self.config.camera)

        # 2. work surface coordinates, will be initialised on first received image
        self.worksurface_detection = None

        # 3. object detection
        self.object_detection = ObjectDetection(self.config.obj_detection)
        self.labels = self.object_detection.labels


    def process_img(self, img, fps=None):
        if isinstance(img, str):
            print("\nimg path:", img)
            img = cv2.imread(img)
        
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img)
            # self.worksurface_detection = WorkSurfaceDetection(img, self.config.dlc)
        
        labelled_img, detections = self.object_detection.get_prediction(img, self.worksurface_detection, extra_text=fps)

        ########################
        graph_relations = GraphRelations(self.labels, detections)
        
        graph_img, action = graph_relations.using_network_x()
        
        joined_img_size = [labelled_img.shape[0], labelled_img.shape[1] + graph_img.shape[1], labelled_img.shape[2]]
        
        joined_img = np.zeros(joined_img_size, dtype=np.uint8)
        joined_img.fill(200)
        joined_img[:labelled_img.shape[0], :labelled_img.shape[1]] = labelled_img
        
        joined_img[:graph_img.shape[0], labelled_img.shape[1]:labelled_img.shape[1]+graph_img.shape[1]] = graph_img
        
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        font_thickness = 1
        text_color = (int(0), int(0), int(0))
        text_pt = (labelled_img.shape[1] + 30, graph_img.shape[0] + 50)
        cv2.putText(joined_img, action[3], text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        print(action[3])
        ########################

        json_detections = json.dumps(detections, cls=EnhancedJSONEncoder)
        json_action = json.dumps(action, cls=EnhancedJSONEncoder)
        
        return joined_img, detections, json_detections, action, json_action
    

if __name__ == '__main__':

    pipeline = Pipeline()

    # pipeline
    show_imgs = False

    # Iterate over images and run:
    # we can use a directory here or a single image /163.png
    # img_path = "data_full/dlc/dlc_work_surface_jsi_05-07-2021/labeled-data/raw_work_surface_jsi_08-07-2021"
    # img_path = "/home/sruiz/datasets/reconcycle/2022-02-17_kalo_tracking_2"
    # img_path = "/home/sruiz/datasets/reconcycle/2022-02-17_kalo_tracking/"
    img_path = "/home/sruiz/datasets/reconcycle/2022-04-04_qundis_disassembly/"
    save_path = "./save_images" # set to None to not save
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs = get_images(img_path)

    for img_p in imgs:
        labeled_img, *_ = pipeline.process_img(img_p)
        
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
        print("fps_imshow", str(np.int(round(fps_imshow, 0))))

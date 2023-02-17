import sys
import os
import cv2
import numpy as np
import json
import argparse
import time
import atexit
from rich import print
import commentjson
import asyncio
from PIL import Image
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# do as if we are in the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection

from object_reid import ObjectReId
from config import load_config

from context_action_framework.types import Camera

from data_loader_even_pairwise import DataLoaderEvenPairwise
from data_loader import DataLoader

class Main():
    def __init__(self) -> None:
        # load config
        self.config = load_config()
        
        # load yolact
        yolact, dataset = self.load_yolact(self.config.obj_detection)
        
        # load object reid
        object_reid = None
        if self.config.reid:
            object_reid = ObjectReId()
        
        # pretend to use Basler camera
        self.camera_type = Camera.basler
        self.camera_name = self.camera_type.name
        
        self.camera_config = self.config.basler
        
        self.camera_config.enable_topic = "set_sleeping" # basler camera specific
        self.camera_config.enable_camera_invert = True # enable = True, but the topic is called set_sleeping, so the inverse
        self.camera_config.use_worksurface_detection = True
        
        self.parent_frame = None
        self.worksurface_detection = None
        
        self.object_detection = ObjectDetection(self.config, self.camera_config, yolact, dataset, object_reid, self.camera_type, self.parent_frame, use_ros=False)
        
        
        self.run()
    
    def load_yolact(self, yolact_config):
        yolact_dataset = None
        
        if os.path.isfile(yolact_config.yolact_dataset_file):
            print("loading", yolact_config.yolact_dataset_file)
            with open(yolact_config.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
        else:
            raise Exception("config.yolact_dataset_file is incorrect: " +  str(yolact_config.yolact_dataset_file))
                
        dataset = Config(yolact_dataset)
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': dataset,
            'num_classes': len(dataset.class_names) + 1,

            # Image Size
            'max_size': 1100,

            # These are in BGR and are for ImageNet
            'MEANS': (103.94, 116.78, 123.68),
            'STD': (57.38, 57.12, 58.40),
            
            # the save path should contain resnet101_reducedfc.pth
            'save_path': './data_limited/yolact/',
            'score_threshold': yolact_config.yolact_score_threshold,
            'top_k': len(dataset.class_names)
        }
        
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(yolact_config.yolact_dataset_file), yolact_dataset["model"])
            
        print("model_path", model_path)
        
        yolact = Yolact(config_override)
        yolact.cfg.print()
        yolact.eval()
        yolact.load_weights(model_path)
        
        return yolact, dataset

    def run_yolact(self, img):
        # self.colour_img = rotate_img(colour_img, self.camera_config.rotate_img)
        
        # if hasattr(self.camera_config, "use_worksurface_detection") and self.camera_config.use_worksurface_detection:
        if self.worksurface_detection is None:
            print("detecting work surface...")
            self.worksurface_detection = WorkSurfaceDetection(img, self.camera_config.work_surface_ignore_border_width, debug=self.camera_config.debug_work_surface_detection)
        
        # on random images, don't use tracker
        labelled_img, detections, markers, poses, graph_img, graph_relations = self.object_detection.get_prediction(img, depth_img=None, worksurface_detection=self.worksurface_detection, extra_text=None, camera_info=None, use_tracker=False)

        # debug
        # if hasattr(self.camera_config, "use_worksurface_detection") and self.camera_config.use_worksurface_detection:
        #     if self.camera_config.show_work_surface_detection:
        #         self.worksurface_detection.draw_corners_and_circles(labelled_img)
        
        return labelled_img, detections, markers, poses, graph_img, graph_relations

    def run(self):
        img_dir = "experiments/datasets/hca_simon/sorted_in_folders"
        preprocessing_dir = "experiments/datasets/preprocessing"
        
        dl = DataLoader(img_dir, shuffle=False, shuffle_train_val_split=False)
        
        # TODO: iterate over all images in dataset
        # TODO: for each image, run Yolact and determine OBBs.
        # TODO: save OBBs to file
        # TODO: load OBBs from file
        
        # make preprocessing directory for each class
        for dirs in [dl.seen_dirs, dl.unseen_dirs]:
            for class_dir in dirs:
                class_path = os.path.join(preprocessing_dir, class_dir)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
        
        for dl_name in ["seen_train", "seen_val", "unseen_val"]:
        
            for i, (inputs, labels, path) in enumerate(dl.dataloaders[dl_name]):
                
                inputs = inputs.cpu().detach().numpy()
                for j in np.arange(len(inputs)):
                    # run Yolact on inputs
                    labelled_img, detections, *_ = self.run_yolact(inputs[j])
                    filename = os.path.basename(path[j])
                    dirname = os.path.basename(os.path.dirname(path[j]))
                    
                    print("path", path[j])
                    print("filename", filename)
                    print("dirname", dirname)
                    print("label", labels[j])
                    
                    # save labelled img
                    im = Image.fromarray(labelled_img)
                    im.save(os.path.join(preprocessing_dir, dirname, filename))
                    
                
                    # remove properties we don't need to save
                    for detection in detections:
                        detection.mask = None
                        detection.mask_contour = None
                    
                    # save detections
                    obj_templates_json_str = jsonpickle.encode(detections, keys=True, warn=True, indent=2)
                    with open(os.path.join(preprocessing_dir, dirname, filename + ".json"), 'w', encoding='utf-8') as f:
                        f.write(obj_templates_json_str)


if __name__ == '__main__':
    main = Main()

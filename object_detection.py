import os
import numpy as np
import commentjson
from rich import print

import config_default

# from yolact_pkg.data.config import cfg
from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact
from yolact_pkg.eval import infer, annotate_img

import obb


class ObjectDetection:
    def __init__(self):
        yolact_dataset = None
        
        if os.path.isfile(config_default.cfg.yolact_dataset_file):
            print("loading", config_default.cfg.yolact_dataset_file)
            with open(config_default.cfg.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
                
        self.dataset = Config(yolact_dataset)    
        # dataset = Config({
        #     'name': 'Base Dataset',

        #     # Training images and annotations
        #     'train_images': './data/coco/train_images/',
        #     'train_info':   './data/coco/_train.json',

        #     # Validation images and annotations.
        #     'valid_images': './data/coco/test_images/',
        #     'valid_info':   './data/coco/_test.json',

        #     # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
        #     'has_gt': True,

        #     # A list of names for each of you classes.
        #     'class_names': ('person', 'bicycle', 'car', 'motorcycle'),

        #     # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
        #     # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
        #     # If not specified, this just assumes category ids start at 1 and increase sequentially.
        #     'label_map': None
        # })
        
        # ! for testing only
        self.dataset = Config({
            'name': 'Base Dataset',
            "model": "real_857_36000.pth",
            "class_names": ("background", "battery", "hca_back", "hca_front", "hca_side1", "hca_side2", "internals_back", "internals_front", "pcb", "internals"),

            "label_map":
            {
                "0": 1,
                "1": 4,
                "2": 3,
                "3": 5,
                "4": 6,
                "5": 2,
                "6": 9,
                "7": 7,
                "8": 8,
                "9": 10
            },
            'has_gt': True,
            "train_images": "/home/sruiz/datasets/labelme/kalo_jsi_goe_combined_coco-07-07-2021",
            "train_info": "/home/sruiz/datasets/labelme/kalo_jsi_goe_combined_coco-07-07-2021/train.json",

            "valid_images": "/home/sruiz/datasets/labelme/kalo_jsi_goe_combined_coco-07-07-2021",
            "valid_info": "/home/sruiz/datasets/labelme/kalo_jsi_goe_combined_coco-07-07-2021/test.json"

        })
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': self.dataset,
            'num_classes': len(self.dataset.class_names) + 1,

            # Image Size
            'max_size': 550,
            
            'save_path': './data_limited/yolact/',
        }
        
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(config_default.cfg.yolact_dataset_file), yolact_dataset["model"])
            
        print("model_path", model_path)
        
        self.yolact = Yolact(config_override)
        self.yolact.eval()
        # self.yolact.load_weights("./yolact_weights/training_2021-11-05-13꞉09/yolact_base_36_2200.pth")
        self.yolact.load_weights(model_path)


    def get_prediction(self, img_path):
        
        frame, classes, scores, boxes, masks = infer(self.yolact, img_path)
        
        # calculate the oriented bounding boxes
        obb_corners = []
        obb_centers = []
        obb_rot_quats = []
        for i in np.arange(len(masks)):
            obb_mask = masks[i].cpu().numpy()[:,:, 0] == 1
            corners, center, rot_quat = obb.get_obb_from_mask(obb_mask)
            obb_corners.append(corners)
            obb_centers.append(center)
            obb_rot_quats.append(rot_quat)
        
        # return classes, scores, boxes, masks
        return frame, classes, scores, boxes, masks, obb_corners, obb_centers, obb_rot_quats
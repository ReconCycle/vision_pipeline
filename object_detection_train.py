import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
from rich import print
import cv2

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact
from yolact_pkg.eval import infer, annotate_img
from yolact_pkg.train import train
import torch

from tracker.byte_tracker import BYTETracker
from graph_relations import GraphRelations

import obb
import graphics
from config import load_config
from helpers import Struct


if __name__ == '__main__':
    
    print("dir()", dir())
    
    
    dataset = Config({
        'name': 'Base Dataset',

        # Training images and annotations
        'train_images': '/home/sruiz/datasets/labelme/2022-02-25_kalo_jsi_goe_combined/coco',
        'train_info':   '/home/sruiz/datasets/labelme/2022-02-25_kalo_jsi_goe_combined/coco/_train.json',

        # Validation images and annotations.
        'valid_images': '/home/sruiz/datasets/labelme/2022-02-25_kalo_jsi_goe_combined/coco',
        'valid_info':   '/home/sruiz/datasets/labelme/2022-02-25_kalo_jsi_goe_combined/coco/_test.json',

        # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
        'has_gt': True,

        # A list of names for each of you classes.
        'class_names': ('hca_front', 'hca_back', 'hca_side1', 'hca_side2', 'battery', 'pcb', 'pcb_covered', 'internals'),

        # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
        # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
        # If not specified, this just assumes category ids start at 1 and increase sequentially.
        'label_map': None
    })
    
    config_override = {
        'name': 'yolact_base',

        # Dataset stuff
        'dataset': dataset,
        'num_classes': len(dataset.class_names) + 1,

        # Image Size
        'max_size': 1100, #! I changed this, was 550
        
        'save_path': 'data_full/yolact/2022-03-02_kalo/',
        
        # we can override args used in eval.py:        
        'score_threshold': 0.1,
        'top_k': 10
    }
    
    # we can override training args here:
    training_args_override = {
        "batch_size": 2, #! I changed this, was 8
        "save_interval": 100,
        # "resume": 
        "validation_size": 100,
    }
    
    yolact = Yolact(config_override)
    
    ###########################################
    # Training                                #
    ###########################################
    
    print("run training...")
    train(yolact, training_args_override)
    
    ###########################################
    # Inference                               #
    ###########################################
    
    # yolact.eval()
    # yolact.load_weights("./yolact_weights/training_2021-11-05-13꞉09/yolact_base_36_2200.pth")
    
    # frame, classes, scores, boxes, masks = yolact.infer(".data/coco/train_images/00000.jpg")
    # # or: frame, classes, scores, boxes, masks = yolact.infer(yolact, cv2.imread(".data/coco/train_images/00000.jpg"))
    
    # annotated_img = annotate_img(frame, classes, scores, boxes, masks)
    
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("output",annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

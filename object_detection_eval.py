import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
from rich import print
import cv2

from yolact_pkg.utils.augmentations import SSDAugmentation, BaseTransform
from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact
from yolact_pkg.eval import infer, annotate_img, evaluate, parse_args, calc_map_classwise
from yolact_pkg.data.config import MEANS
from yolact_pkg.train import train
from yolact_pkg.data.coco import COCODetection, detection_collate
import torch

from config import load_config

def eval_aps():
    # print("dir()", dir())
    torch.cuda.empty_cache()
    
    
    dataset = Config({
        'name': 'Base Dataset',

        # Training images and annotations
        'train_images': '/home/sruiz/datasets2/reconcycle/2022-05-02_kalo_qundis/coco',
        'train_info':   '/home/sruiz/datasets2/reconcycle/2022-05-02_kalo_qundis/coco/_train.json',

        # Validation images and annotations.
        'valid_images': '/home/sruiz/datasets2/reconcycle/2022-05-02_kalo_qundis/coco',
        'valid_info':   '/home/sruiz/datasets2/reconcycle/2022-05-02_kalo_qundis/coco/_test.json',

        # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
        'has_gt': True,

        # A list of names for each of you classes.
        'class_names': ("hca_front", "hca_back", "hca_side1", "hca_side2", "battery", "pcb", "internals", "pcb_covered", "plastic_clip"),

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
        
        'save_path': 'data_full/yolact/2022-10-17_kalo_qundis/',
        
        # we can override args used in eval.py:
        'score_threshold': 0.1,
        'top_k': 10
    }
    
    # we can override training args here:
    training_args_override = {
        "batch_size": 2, #! I changed this, was 8
        "save_interval": -1, # -1 for saving only at end of the epoch
        # "resume": 
        "validation_size": 100,
    }
    
    yolact = Yolact(config_override)
    
    ###########################################
    # Training                                #
    ###########################################
    
    # print("run training...")
    # train(yolact, training_args_override)
    
    ###########################################
    # Inference                               #
    ###########################################
    
    yolact.eval()
    yolact.load_weights("./data_limited/yolact/2022-05-02_kalo_qundis/yolact_base_274_202125.pth")

    # frame, classes, scores, boxes, masks = yolact.infer("/root/datasets/2022-05-02_kalo_qundis/coco/JPEGImages/1066.jpg")

    # annotated_img = annotate_img(frame, classes, scores, boxes, masks)

    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("output",annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 
    # todo: evaluation
    
    val_dataset = COCODetection(image_path=yolact.cfg.dataset.valid_images,
                                info_file=yolact.cfg.dataset.valid_info,
                                transform=BaseTransform(MEANS))
    
    args = parse_args()
    
    # print("args", args)
    all_maps, ap_data = evaluate(yolact, val_dataset)
    # print("all_maps", all_maps)
    
    # print("ap_data", ap_data)
    print(len(yolact.cfg.dataset.class_names), yolact.cfg.dataset.class_names)

    all_maps2, _ = calc_map_classwise(yolact, ap_data)
    
if __name__ == '__main__':
    eval_aps()
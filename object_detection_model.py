import sys
import os
import cv2
import numpy as np
import json
import argparse
import time
import atexit
from enum import Enum
from rich import print
import commentjson
from types import SimpleNamespace
import torch

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from ultralytics import YOLO


class ModelType(Enum):
    yolact = 1
    yolov8 = 2


class ObjectDetectionModel:
    def __init__(self, config) -> None:
        # model that can load yolov8 or yolact

        self.config = config
        self.model_type = None
        self.yolact = None
        self.yolov8 = None
        self.dataset = None
        
        if config.model.lower() == "yolact":
            self.model_type = ModelType.yolact
            self.yolact, self.dataset = self.load_yolact(config)

        elif config.model.lower() == "yolov8":
            self.model_type = ModelType.yolov8
            self.yolov8, self.dataset = self.load_yolov8(config)



    def load_yolact(self, yolact_config):
        yolact_dataset = None
        
        if os.path.isfile(yolact_config.yolact_dataset_file):
            print("loading", yolact_config.yolact_dataset_file)
            with open(yolact_config.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
        else:
            raise Exception("config.yolact_dataset_file is incorrect: " +  str(yolact_config.yolact_dataset_file))
                
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(yolact_config.yolact_dataset_file), yolact_dataset["model"])

        yolact_dataset = Config(yolact_dataset)
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': yolact_dataset,
            'num_classes': len(yolact_dataset.class_names) + 1,

            # Image Size
            'max_size': 1100,

            # These are in BGR and are for ImageNet
            'MEANS': (103.94, 116.78, 123.68),
            'STD': (57.38, 57.12, 58.40),
            
            # the save path should contain resnet101_reducedfc.pth
            'save_path': './data_limited/yolact/',
            'score_threshold': yolact_config.yolact_score_threshold,
            'top_k': len(yolact_dataset.class_names)
        }
            
        print("model_path", model_path)
        
        yolact = Yolact(config_override)
        yolact.cfg.print()
        yolact.eval()
        yolact.load_weights(model_path)
        
        return yolact, yolact_dataset


    def load_yolov8(self, config):
        # Load finetuned YOLOv8m-seg model
        yolov8 = YOLO(config.yolov8_model_file)
        class_names_dict = yolov8.names

        yolov8_dataset = SimpleNamespace()
        yolov8_dataset.class_names = list(class_names_dict.values())

        return yolov8, yolov8_dataset


    def infer_yolov8(self, colour_img):

        # Run batched inference on a list of images
        results = self.yolov8.predict(colour_img, 
              save=False,
              retina_masks=True,
              imgsz=640,
              verbose=False,
              conf=self.config.yolov8_score_threshold)
        
        if len(results) >= 1:
            result = results[0]
        
            data = result.boxes  # Boxes object for bbox outputs

            scores = data.conf.cpu().numpy()
            classes = data.cls
            classes_int = [int(t.item()) for t in classes]
            boxes = data.xyxy.cpu().numpy() 
            
            masks_results = result.masks
            if masks_results is not None:
                masks = masks_results.data  # Masks object for segmentation masks outputs
                masks = masks.unsqueeze(-1) # make it like yolact
            else:
                masks = torch.empty((0,))

            # print("orig_img", orig_img.shape, type(orig_img))            
            # print("classes", classes.shape) # torch.Size([5])
            # print("scores", scores.shape) # torch.Size([5])
            # print("boxes", boxes.shape) # torch.Size([5, 4])
            # print("masks", masks.shape, type(masks)) # torch.Size([5, 1450, 1450])
            # print("classes", classes)
            # print("classes_int", classes_int)
            # print("scores", scores)

            return colour_img, classes_int, scores, boxes, masks
            
        else:
            return colour_img, np.array([]), np.array([]), np.array([]), torch.empty((0,))

    def infer(self, colour_img):

        if self.model_type == ModelType.yolact:
            frame, classes, scores, boxes, masks = self.yolact.infer(colour_img)

            # all tensors
            # print("frame", frame.shape, type(frame)) # torch.Size([1450, 1450, 3])
            # print("classes", classes.shape, type(classes)) # (4,)
            # print("scores", scores.shape, type(scores)) # (4,)
            # print("boxes", boxes.shape, type(boxes)) # (4, 4)
            # print("masks", masks.shape, type(masks)) # torch.Size([4, 1450, 1450, 1])

            # print("classes", classes) # eg. [ 2  1 10  3]
            # print("scores", scores) # eg. [    0.99976     0.99868     0.99818     0.96942]
            
        elif self.model_type == ModelType.yolov8:
            frame, classes, scores, boxes, masks = self.infer_yolov8(colour_img)

        return frame, classes, scores, boxes, masks

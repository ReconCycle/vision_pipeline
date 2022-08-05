import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
import cv2
from rich import print
from shapely.geometry import Polygon

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from tracker.byte_tracker import BYTETracker

import obb
import graphics
from config import load_config
from helpers import Struct, Detection

import enum


class ObjectDetection:
    def __init__(self, config=None):
        yolact_dataset = None
        
        if config is None:
            config = load_config().obj_detection
        
        if os.path.isfile(config.yolact_dataset_file):
            print("loading", config.yolact_dataset_file)
            with open(config.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
        else:
            raise Exception("config.yolact_dataset_file is incorrect: " +  str(config.yolact_dataset_file))
                
        self.dataset = Config(yolact_dataset)
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': self.dataset,
            'num_classes': len(self.dataset.class_names) + 1,

            # Image Size
            'max_size': 1100,

            # These are in BGR and are for ImageNet
            'MEANS': (103.94, 116.78, 123.68),
            'STD': (57.38, 57.12, 58.40),
            
            # the save path should contain resnet101_reducedfc.pth
            'save_path': './data_limited/yolact/',
            'score_threshold': config.yolact_score_threshold,
            'top_k': len(self.dataset.class_names)
        }
        
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(config.yolact_dataset_file), yolact_dataset["model"])
            
        print("model_path", model_path)
        
        self.yolact = Yolact(config_override)
        self.yolact.cfg.print()
        self.yolact.eval()
        self.yolact.load_weights(model_path)


        # parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
        # parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        # parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
        # parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
        # parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

        self.tracker_args = SimpleNamespace()
        self.tracker_args.track_thresh = 0.1
        self.tracker_args.track_buffer = 10 # num of frames to remember lost tracks
        self.tracker_args.match_thresh = 2.5 # default: 0.9 # higher number is more lenient
        self.tracker_args.min_box_area = 10
        self.tracker_args.mot20 = False
        
        self.tracker = BYTETracker(self.tracker_args)
        self.fps_graphics = -1.
        self.fps_objdet = -1.
        
        # convert class names to enums
        labels = enum.IntEnum('label', self.dataset.class_names, start=0)
        self.labels = labels
        

    def get_prediction(self, img_path, worksurface_detection=None, extra_text=None):
        t_start = time.time()
        
        frame, classes, scores, boxes, masks = self.yolact.infer(img_path)
        fps_nn = 1.0 / (time.time() - t_start)

        detections = []
        for i in np.arange(len(classes)):
                
            detection = Detection()
            detection.id = i
            
            detection.label = self.labels(classes[i]) # self.dataset.class_names[classes[i]]
            
            detection.score = float(scores[i])
            detection.box = boxes[i]
            detection.mask = masks[i]
            
            # compute contour. Required for obb and graph_relations
            mask = masks[i].cpu().numpy().astype("uint8")
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                # get the contour with the largest area. Assume this is the one containing our object
                cnt = max(cnts, key = cv2.contourArea)
                detection.mask_contour = np.squeeze(cnt)

                poly = None
                if len(detection.mask_contour) > 2:
                    poly = Polygon(detection.mask_contour)
                    
                if poly is None or not poly.is_valid:
                    poly = Polygon(detection.obb_corners)

                detection.mask_polygon = poly
            
            detections.append(detection)
                
        tracker_start = time.time()
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        online_targets = self.tracker.update(boxes, scores) #? does the tracker not benefit from the predicted classes?

        for t in online_targets:
            detections[t.input_id].tracking_id = t.track_id
            detections[t.input_id].tracking_box = t.tlbr
            detections[t.input_id].score = float(t.score)

        fps_tracker = 1.0 / (time.time() - tracker_start)
        
        
        obb_start = time.time()
        # calculate the oriented bounding boxes
        for detection in detections:
            corners, center, rot_quat = obb.get_obb_from_contour(detection.mask_contour)
            detection.obb_corners = corners
            detection.obb_center = center
            detection.obb_rot_quat = rot_quat
            if worksurface_detection is not None and corners is not None:
                detection.obb_corners_meters = worksurface_detection.pixels_to_meters(corners)
                detection.obb_center_meters = worksurface_detection.pixels_to_meters(center)
            else:
                detection.obb_corners_meters = None
                detection.obb_center_meters = None
        
        fps_obb = -1
        if time.time() - obb_start > 0:
            fps_obb = 1.0 / (time.time() - obb_start)
                
        graphics_start = time.time()
        if extra_text is not None:
            extra_text + ", "
        else:
            extra_text = ""
        fps_str = extra_text + "objdet: " + str(round(self.fps_objdet, 1)) + ", nn: " + str(round(fps_nn, 1)) + ", tracker: " + str(np.int(round(fps_tracker, 0))) + ", obb: " + str(np.int(round(fps_obb, 0))) + ", graphics: " + str(np.int(round(self.fps_graphics, 0)))
        labelled_img = graphics.get_labelled_img(frame, masks, detections, fps=fps_str, worksurface_detection=worksurface_detection)
        
        self.fps_graphics = 1.0 / (time.time() - graphics_start)
        self.fps_objdet = 1.0 / (time.time() - t_start)
        
        return labelled_img, detections

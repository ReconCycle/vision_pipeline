import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
import cv2
from rich import print

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from tracker.byte_tracker import BYTETracker
from graph_relations import GraphRelations

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
                
        self.dataset = Config(yolact_dataset)
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': self.dataset,
            'num_classes': len(self.dataset.class_names) + 1,

            # Image Size
            'max_size': 1100,
            
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
        self.fps_total = -1.
        
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
            
            detection.score = scores[i]
            detection.box = boxes[i]
            detection.mask = masks[i]
            
            # compute contour. Required for obb and graph_relations
            mask = masks[i].cpu().numpy().astype("uint8")
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                detection.mask_contour = np.squeeze(cnts[0])
            
            detections.append(detection)
                            
            # todo: previously the objects as commented out. This was so it could be dumped to JSON.
            # detection = {}
            # detection["class_name"] = self.dataset.class_names[classes[i]]
            # detection["score"] = float(scores[i])
            # detection["obb_corners"] = worksurface_detection.pixels_to_meters(obb_corners[i]).tolist()
            # detection["obb_center"] = worksurface_detection.pixels_to_meters(obb_centers[i]).tolist()
            # detection["obb_rot_quat"] = obb_rot_quats[i].tolist()
            # detection["tracking_id"] = tracking_ids[i] if tracking_ids[i] is not None else -1
            # detection["tracking_score"] = float(tracking_scores[i]) if tracking_scores[i] is not None else float(-1)
            # detections.append(detection)
                
        tracker_start = time.time()
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        online_targets = self.tracker.update(boxes, scores) #? does the tracker not benefit from the predicted classes?

        for t in online_targets:
            detections[t.input_id].tracking_id = t.track_id
            detections[t.input_id].tracking_box = t.tlbr
            detections[t.input_id].score = t.score

        fps_tracker = 1.0 / (time.time() - tracker_start)
        
        obb_start = time.time()
        # calculate the oriented bounding boxes
        for detection in detections:
            corners, center, rot_quat = obb.get_obb_from_contour(detection.mask_contour)
            detection.obb_corners = corners
            detection.obb_center = center
            detection.obb_rot_quats = rot_quat
        
        fps_obb = 1.0 / (time.time() - obb_start)
                
        graphics_start = time.time()
        if extra_text is not None:
            extra_text + ", "
        else:
            extra_text = ""
        fps_str = extra_text + "fps_total: " + str(np.int(round(self.fps_total, 0))) + ", fps_nn: " + str(np.int(round(fps_nn, 0))) + ", fps_tracker: " + str(np.int(round(fps_tracker, 0))) + ", fps_obb: " + str(np.int(round(fps_obb, 0))) + ", fps_graphics: " + str(np.int(round(self.fps_graphics, 0)))
        labelled_img = graphics.get_labelled_img(frame, masks, detections, fps=fps_str, worksurface_detection=worksurface_detection)
        
        self.fps_graphics = 1.0 / (time.time() - graphics_start)
        self.fps_total = 1.0 / (time.time() - t_start)
        
        graph_relations = GraphRelations(self.labels, detections)
        
        graph_img, action = graph_relations.using_network_x()
        
        joined_img_size = [labelled_img.shape[0], labelled_img.shape[1] + graph_img.shape[1], labelled_img.shape[2]]
        
        joined_img = np.zeros(joined_img_size, dtype=np.uint8)
        joined_img.fill(200)
        joined_img[:labelled_img.shape[0], :labelled_img.shape[1]] = labelled_img
        
        joined_img[:graph_img.shape[0], labelled_img.shape[1]:labelled_img.shape[1]+graph_img.shape[1]] = graph_img
        
        return joined_img, detections

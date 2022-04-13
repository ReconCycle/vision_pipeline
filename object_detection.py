import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
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
        
        print("label", labels)
        print("label.battery", labels.battery, repr(labels.battery), labels.battery==5)
        

    def get_prediction(self, img_path, worksurface_detection=None, extra_text=None):
        t_start = time.time()
        
        frame, classes, scores, boxes, masks = self.yolact.infer(img_path)
        fps_nn = 1.0 / (time.time() - t_start)
        
        tracker_start = time.time()
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        online_targets = self.tracker.update(boxes, scores) #? does the tracker not benefit from the predicted classes?
        
        tracking_ids = np.full(len(classes), None)
        tracking_boxes = np.full(len(classes), None)
        tracking_scores = np.full(len(classes), None)
        
        for t in online_targets:
            tracking_ids[t.input_id] = t.track_id # add the tracking id to the detections
            tracking_boxes[t.input_id] = t.tlbr
            tracking_scores[t.input_id] = t.score

        fps_tracker = 1.0 / (time.time() - tracker_start)
        
        obb_start = time.time()
        # calculate the oriented bounding boxes
        obb_corners = []
        obb_centers = []
        obb_rot_quats = []
        for i in np.arange(len(masks)):
            mask = masks[i].cpu().numpy()
            corners, center, rot_quat = obb.get_obb_from_mask(mask)
            obb_corners.append(corners)
            obb_centers.append(center)
            obb_rot_quats.append(rot_quat)
        fps_obb = 1.0 / (time.time() - obb_start)

        detections = []
        for i in np.arange(len(classes)):
            if obb_corners[i] is not None:
                
                detection = Detection()
                detection.id = i
                
                # print("self.labels(classes[i])", self.labels(classes[i]))
                
                detection.label = self.labels(classes[i]) # self.dataset.class_names[classes[i]]
                
                detection.score = scores[i]
                detection.box = boxes[i]
                detection.mask = masks[i].cpu().numpy()
                
                detection.obb_corners = obb_corners[i] # worksurface_detection.pixels_to_meters(obb_corners[i]).tolist()
                detection.obb_center = obb_centers[i] # worksurface_detection.pixels_to_meters(obb_centers[i]).tolist()
                detection.obb_rot_quat = obb_rot_quats[i] # .tolist()
                
                detection.tracking_id = tracking_ids[i]
                detection.tracking_score = tracking_scores[i]
                detection.tracking_box = tracking_boxes[i]
                
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
                
        graphics_start = time.time()
        if extra_text is not None:
            extra_text + ", "
        else:
            extra_text = ""
        fps_str = extra_text + "fps_total: " + str(np.int(round(self.fps_total, 0))) + ", fps_nn: " + str(np.int(round(fps_nn, 0))) + ", fps_tracker: " + str(np.int(round(fps_tracker, 0))) + ", fps_obb: " + str(np.int(round(fps_obb, 0))) + ", fps_graphics: " + str(np.int(round(self.fps_graphics, 0)))
        labelled_img = graphics.get_labelled_img(frame, self.dataset.class_names, classes, scores, boxes, masks, obb_corners, obb_centers, tracking_ids, tracking_boxes, tracking_scores, fps=fps_str, worksurface_detection=worksurface_detection)
        
        self.fps_graphics = 1.0 / (time.time() - graphics_start)
        self.fps_total = 1.0 / (time.time() - t_start)

        # ! deprecated. Now use detection object
        # graph_relations = GraphRelations(self.dataset.class_names, classes, scores, boxes, masks, obb_corners, obb_centers, tracking_ids, tracking_boxes, tracking_scores)
        
        graph_relations = GraphRelations(self.labels, detections)
        
        graph_img = graph_relations.using_network_x()
        
        joined_img_size = [labelled_img.shape[0], labelled_img.shape[1] + graph_img.shape[1], labelled_img.shape[2]]
        
        joined_img = np.zeros(joined_img_size, dtype=np.uint8)
        joined_img.fill(200)
        joined_img[:labelled_img.shape[0], :labelled_img.shape[1]] = labelled_img
        
        joined_img[:graph_img.shape[0], labelled_img.shape[1]:labelled_img.shape[1]+graph_img.shape[1]] = graph_img
        
        # y_offset = labelled_img.shape[0] - graph_img.shape[0]
        # labelled_img[y_offset:y_offset+graph_img.shape[0], 0:graph_img.shape[1]] = graph_img
        
        return joined_img, detections

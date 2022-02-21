import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
from rich import print

import config_default

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact
from yolact_pkg.eval import infer, annotate_img

from tracker.byte_tracker import BYTETracker

import obb
import graphics


class ObjectDetection:
    def __init__(self):
        yolact_dataset = None
        
        if os.path.isfile(config_default.cfg.yolact_dataset_file):
            print("loading", config_default.cfg.yolact_dataset_file)
            with open(config_default.cfg.yolact_dataset_file, "r") as read_file:
                yolact_dataset = commentjson.load(read_file)
                print("yolact_dataset", yolact_dataset)
                
        self.dataset = Config(yolact_dataset)
        
        config_override = {
            'name': 'yolact_base',

            # Dataset stuff
            'dataset': self.dataset,
            'num_classes': len(self.dataset.class_names) + 1,

            # Image Size
            'max_size': 550,
            
            # the save path should contain resnet101_reducedfc.pth
            'save_path': './data_limited/yolact/',
            'score_threshold': 0.1,
            'top_k': 10
        }
        
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(config_default.cfg.yolact_dataset_file), yolact_dataset["model"])
            
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
        
        self.fps_total = -1

    def get_prediction(self, img_path, worksurface_detection=None):
        t_start = time.time()
        
        frame, classes, scores, boxes, masks = self.yolact.infer(img_path)
        fps_nn = 1.0 / (time.time() - t_start)
        
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        online_targets = self.tracker.update(boxes, scores)
        
        tracking_ids = np.full(len(classes), None)
        tracking_boxes = np.full(len(classes), None)
        tracking_scores = np.full(len(classes), None)
        
        for t in online_targets:
            tracking_ids[t.input_id] = t.track_id # add the tracking id to the detections
            tracking_boxes[t.input_id] = t.tlbr
            tracking_scores[t.input_id] = t.score

        
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
            
        fps_str = "fps_nn: " + str(round(fps_nn, 1)) + ", fps_total: " + str(round(self.fps_total, 1)) 
        labelled_img = graphics.get_labelled_img(frame, self.dataset.class_names, classes, scores, boxes, masks, obb_corners, obb_centers, tracking_ids, tracking_boxes, tracking_scores, fps=fps_str, worksurface_detection=worksurface_detection)

        detections = []
        for i in np.arange(len(classes)):
            if obb_corners[i] is not None:
                detection = {}
                detection["class_name"] = self.dataset.class_names[classes[i]]
                detection["score"] = float(scores[i])
                detection["obb_corners"] = worksurface_detection.pixels_to_meters(obb_corners[i]).tolist()
                detection["obb_center"] = worksurface_detection.pixels_to_meters(obb_centers[i]).tolist()
                detection["obb_rot_quat"] = obb_rot_quats[i].tolist()
                detection["tracking_id"] = tracking_ids[i]
                detection["tracking_score"] = tracking_scores[i]
                detections.append(detection)
                
        self.fps_total = 1.0 / (time.time() - t_start)
        
        # return classes, scores, boxes, masks
        # return frame, classes, scores, boxes, masks, obb_corners, obb_centers, obb_rot_quats
        return labelled_img, detections

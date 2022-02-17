import os
from types import SimpleNamespace
import numpy as np
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
        }
        
        model_path = None
        if "model" in yolact_dataset:
            model_path = os.path.join(os.path.dirname(config_default.cfg.yolact_dataset_file), yolact_dataset["model"])
            
        print("model_path", model_path)
        
        self.yolact = Yolact(config_override)
        self.yolact.eval()
        self.yolact.load_weights(model_path)


        # parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
        # parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        # parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
        # parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
        # parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

        self.tracker_args = SimpleNamespace()
        self.tracker_args.track_thresh = 0.6
        self.tracker_args.track_buffer = 30
        self.tracker_args.match_thresh = 0.9
        self.tracker_args.min_box_area = 100
        self.tracker_args.mot20 = False
        
        self.tracker = BYTETracker(self.tracker_args)

    def get_prediction(self, img_path, worksurface_detection=None, fps=None):
        
        frame, classes, scores, boxes, masks = infer(self.yolact, img_path)
        
        print("boxes", boxes.shape, boxes)
        print("scores", scores.shape, scores)
        
        # dets = np.concatenate((boxes, [scores]), axis=0)
        # print("dets.shape", dets)
        
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        # for help
        # 'dets' (x1, y1, x2, y2, score)
        online_targets = self.tracker.update(boxes, scores)
        
        online_tlwhs = []
        online_tlbrs = [] # top, left, bottom, right
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_tlbrs.append(tlbr)
                online_ids.append(tid)
                online_scores.append(t.score)
        
            print("t.track_id", t.track_id)
            print("t.tlwh", t.tlwh)
        
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
            
            
        labelled_img = graphics.get_labelled_img(frame, self.dataset.class_names, classes, scores, boxes, masks, obb_corners, obb_centers, online_tlbrs, online_ids, online_scores, fps=fps, worksurface_detection=worksurface_detection)

        detections = []
        for i in np.arange(len(classes)):
            if obb_corners[i] is not None:
                detection = {}
                detection["class_name"] = self.dataset.class_names[classes[i]]
                detection["score"] = float(scores[i])
                detection["obb_corners"] = worksurface_detection.pixels_to_meters(obb_corners[i]).tolist()
                detection["obb_center"] = worksurface_detection.pixels_to_meters(obb_centers[i]).tolist()
                detection["obb_rot_quat"] = obb_rot_quats[i].tolist()
                detections.append(detection)
        
        # return classes, scores, boxes, masks
        # return frame, classes, scores, boxes, masks, obb_corners, obb_centers, obb_rot_quats
        return labelled_img, detections

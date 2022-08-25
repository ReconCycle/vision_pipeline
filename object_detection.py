import os
from types import SimpleNamespace
import numpy as np
import time
import commentjson
import cv2
from rich import print
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.validation import explain_validity

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from tracker.byte_tracker import BYTETracker

import obb
import graphics
from helpers import Struct, make_valid_poly
from context_action_framework.types import Detection, Label



class ObjectDetection:
    def __init__(self, yolact, dataset):

        self.yolact = yolact
        self.dataset = dataset

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
        # labels = enum.IntEnum('label', self.dataset.class_names, start=0)
        # self.labels = labels
        

    def get_prediction(self, img_path, worksurface_detection=None, extra_text=None):
        t_start = time.time()
        
        frame, classes, scores, boxes, masks = self.yolact.infer(img_path)
        fps_nn = 1.0 / (time.time() - t_start)

        detections = []
        for i in np.arange(len(classes)):
                
            detection = Detection()
            detection.id = int(i)
            
            detection.label = Label(classes[i]) # self.dataset.class_names[classes[i]]
            
            detection.score = float(scores[i])
            detection.box_px = boxes[i].reshape((-1,2)) # convert tlbr
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
                    poly = make_valid_poly(poly)

                detection.polygon_px = poly
            
            detections.append(detection)
                
        tracker_start = time.time()
        # apply tracker
        # look at: https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py
        online_targets = self.tracker.update(boxes, scores) #? does the tracker not benefit from the predicted classes?

        for t in online_targets:
            detections[t.input_id].tracking_id = int(t.track_id)
            detections[t.input_id].tracking_box = t.tlbr
            detections[t.input_id].score = float(t.score)

        fps_tracker = 1.0 / (time.time() - tracker_start)
        
        
        obb_start = time.time()
        # calculate the oriented bounding boxes
        for detection in detections:
            corners, center, rot_quat = obb.get_obb_from_contour(detection.mask_contour)
            detection.obb_px = corners
            
            # todo: obb_3d_px, obb_3d
            # todo: transform should actually be (x, y, z)
            # (transform (x, y), rotation (x, y, z, w))
            detection.tf_px = [center, rot_quat]
            if worksurface_detection is not None and corners is not None:
                center_meters = worksurface_detection.pixels_to_meters(center)
                detection.tf = [center_meters, rot_quat]
                detection.box = worksurface_detection.pixels_to_meters(detection.box_px)
                detection.obb = worksurface_detection.pixels_to_meters(corners)
                
            else:
                detection.obb = None
                detection.tf = None
        
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

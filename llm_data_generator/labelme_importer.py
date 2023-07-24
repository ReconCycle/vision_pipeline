import json
import warnings
from pathlib import Path
import random
import base64
from io import BytesIO
import os
import sys
import cv2
import obb
import imagesize
# from scipy import ndimage
import natsort
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

# ros package
from context_action_framework.types import Detection, Label

# local imports
from helpers import Struct, make_valid_poly, img_to_camera_coords
from graph_relations import GraphRelations, exists_detection, compute_iou
from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection


class LabelMeImporter():
    def __init__(self, ignore_labels=[]) -> None:
        self.ignore_labels = ignore_labels
        self.worksurface_detection = None
        
        # config
        self.work_surface_ignore_border_width = 100
        self.debug_work_surface_detection = False

        self.object_detection = ObjectDetection(use_ros=False) #! probably we need to add more stuff

    
    def process_labelme_dir(self, labelme_dir, images_dir=None):
        # load in the labelme data
        labelme_dir = Path(labelme_dir)

        if images_dir is None:
            images_dir = labelme_dir
        
        labelme_dir = Path(labelme_dir)
        images_dir = Path(images_dir)
        
        json_paths = list(labelme_dir.glob('*.json'))
        json_paths = natsort.os_sorted(json_paths)
        
        image_paths = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) 
        image_paths = natsort.os_sorted(image_paths)

        tqdm_json_paths = tqdm(json_paths)

        if len(json_paths) == 0:
            print("[red]Folder doesn't contain .json files")

        img_paths = []
        all_detections = []
        all_graph_relations = []

        for idx, json_path in enumerate(tqdm_json_paths):
            tqdm_json_paths.set_description(f"Converting {Path(json_path).stem}")

            # json_path = os.path.join(self.labelme_path, json_name)
            json_data = json.load(open(json_path))
            base_path = os.path.splitext(json_path)[0]

            img_path = None
            img_matches = [_img_path for _img_path in image_paths if base_path in str(_img_path)]

            if len(img_matches) > 0:
                # exists .png or .jpg file
                img_path = img_matches[0]

                if self.worksurface_detection is None:
                    colour_img = cv2.imread(str(img_path))
                    self._process_work_surface_detection(colour_img)

                detections, graph_relations = self._process_labelme_img(json_data, img_path)

                img_paths.append(img_path)
                all_detections.append(detections)
                all_graph_relations.append(graph_relations)
               
            else:
                print(f"[red]No image matched for {base_path}")

            if idx > 20:
                break # ! debug

        return img_paths, all_detections, all_graph_relations

    def _process_work_surface_detection(self, img):
        self.worksurface_detection = WorkSurfaceDetection(img, self.work_surface_ignore_border_width, debug=self.debug_work_surface_detection)

    def _process_labelme_img(self, json_data, img_path):
        detections = []

        # img = Image.open(img_path).convert('RGB') # SLOW
        # img_w, img_h = img.size
        img_w, img_h = imagesize.get(img_path) # fast

        # TODO: we need to get the real world sizes of objects
        # TODO: for basler, we can use work_surface_detection
        # TODO: for realsense, we can use depth

        idx = 0
        for shape in json_data['shapes']:
            # only add items that are in the allowed
            if shape['label'] not in self.ignore_labels:

                if shape['shape_type'] == "polygon":

                    detection = Detection()
                    detection.id = idx
                    detection.tracking_id = idx

                    detection.label = Label[shape['label']]
                    detection.score = float(1.0)
                    
                    # detection.tf_px = # TODO

                    detection.mask_contour = self.points_to_contour(shape['points'])

                    # corners_px, center_px, angle = obb.get_obb_from_contour(detection.mask_contour)
                    # detection.obb_px = corners_px
                    # detection.center_px = center_px
                    # detection.angle_px = angle

                    detection.box_px
                    
                    #! DUPLICATE NOW ALL CODE FROM OBJECT_DETECTION.py

                    
                    
                    # poly = None
                    # if len(detection.mask_contour) > 2:
                    #     poly = Polygon(detection.mask_contour)
                    #     poly = make_valid_poly(poly)

                    # detection.polygon_px = poly
                    
                    # detections.append(detection)

                    idx += 1

        detections, markers, poses, graph_img, graph_relations, fps_obb = self.object_detection.get_detections(detections, worksurface_detectio=self.worksurface_detection)

        return detections, graph_relations    
    

    def points_to_contour(self, points):
        obj_point_list =  points # [(x1,y1),(x2,y2),...]
        obj_point_list = np.array(obj_point_list).astype(int) # convert to int
        # obj_point_list = [tuple(point) for point in obj_point_list] # convert back to list of tuples

        # contour
        return obj_point_list
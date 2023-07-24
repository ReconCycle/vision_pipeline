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
from context_action_framework.types import Detection, Label
from helpers import Struct, make_valid_poly, img_to_camera_coords


class LabelMeImporter():
    def __init__(self, ignore_labels=[]) -> None:
        self.ignore_labels = ignore_labels

    
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

                detections = self._process_labelme_img(json_data, img_path)

                img_paths.append(img_path)
                all_detections.append(detections)
               
            else:
                print(f"[red]No image matched for {base_path}")

        return img_paths, all_detections


    def _process_labelme_img(self, json_data, img_path):
        detections = []

        # img = Image.open(img_path).convert('RGB') # SLOW
        # img_w, img_h = img.size
        img_w, img_h = imagesize.get(img_path) # fast

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

                    corners_px, center_px, angle = obb.get_obb_from_contour(detection.mask_contour)
                    detection.obb_px = corners_px
                    detection.center_px = center_px
                    detection.angle_px = angle
                    
                    poly = None
                    if len(detection.mask_contour) > 2:
                        poly = Polygon(detection.mask_contour)
                        poly = make_valid_poly(poly)

                    detection.polygon_px = poly
                    
                    detections.append(detection)

                    idx += 1

        return detections
    

    def points_to_contour(self, points):
        obj_point_list =  points # [(x1,y1),(x2,y2),...]
        obj_point_list = np.array(obj_point_list).astype(int) # convert to int
        obj_point_list = [tuple(point) for point in obj_point_list] # convert back to list of tuples

        # contour
        return obj_point_list
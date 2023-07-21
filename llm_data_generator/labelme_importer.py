import json
import warnings
from pathlib import Path
import random
import base64
from io import BytesIO
import os
import sys
import cv2
import imagesize
# from scipy import ndimage
import natsort
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from tqdm import tqdm
from context_action_framework.types import Detection


class LabelMeImporter():
    def __init__(self, ignore_labels, foregrounds_dict, labelme_imgs_merge) -> None:
        pass

    
    def process_labelme_dir(self, labelme_dir, images_dir=None):
        # load in the labelme data

        if images_dir is None:
            images_dir = labelme_dir

        json_paths = list(labelme_dir.glob('*.json'))
        json_paths = natsort.os_sorted(json_paths)
        
        image_paths = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) 
        image_paths = natsort.os_sorted(image_paths)

        tqdm_json_paths = tqdm(json_paths)

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

                labelme_img_groups = self._process_labelme_img(json_data, img_path)
               
            else:
                print(f"[red]No image matched for {base_path}")


    def _process_labelme_img(self, json_data, img_path):
        labelme_point_list = []
        labelme_obj_list = []

        # img_h, img_w, _ = cv2.imread(img_path).shape # SLOW
        # img = Image.open(img_path).convert('RGB') # SLOW
        # img_w, img_h = img.size
        img_w, img_h = imagesize.get(img_path) # fast

        for shape in json_data['shapes']:
            # only add items that are in the allowed
            if shape['label'] not in self.ignore_labels:

                if shape['shape_type'] == 'point':
                    point = shape['points'][0]
                    labelme_point_list.append(point)
                elif shape['shape_type'] == "polygon":
                    yolo_obj = self._get_labelme_object(shape, img_h, img_w)
                    labelme_obj_list.append(list(yolo_obj))

        # add the point to the object list
        # ! what does this do?
        # if len(labelme_obj_list) == 1 and len(labelme_point_list) == 1:
        #     labelme_obj_list[0][2] = labelme_point_list[0]

        return img_path, labelme_obj_list
    

    def _get_labelme_object(self, shape, img_h, img_w):
        obj_point_list = shape['points'] # [(x1,y1),(x2,y2),...]
        obj_point_list = np.array(obj_point_list).astype(int) # convert to int
        obj_point_list = [tuple(point) for point in obj_point_list] # convert back to list of tuples

        # return format: label, obj_point_list, point/None
        return shape['label'], obj_point_list, None
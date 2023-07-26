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
from rich import print

# ros package
from context_action_framework.types import Detection, Label, Module, Camera
from sensor_msgs.msg import Image, CameraInfo

# local imports
from helpers import Struct, make_valid_poly, img_to_camera_coords
from graph_relations import GraphRelations, exists_detection, compute_iou
from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection
from types import SimpleNamespace


class LabelMeImporter():
    def __init__(self, ignore_labels=[]) -> None:
        self.ignore_labels = ignore_labels
        self.worksurface_detection = None
        
        # config
        self.work_surface_ignore_border_width = 100
        self.debug_work_surface_detection = False

        self.camera_config = SimpleNamespace()
        self.camera_config.publish_graph_img = False

        self.object_detection = ObjectDetection(camera_config=self.camera_config, use_ros=False) #! probably we need to add more stuff

    
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
        modules = []
        cameras = []

        for idx, json_path in enumerate(tqdm_json_paths):
            tqdm_json_paths.set_description(f"Converting {Path(json_path).stem}")

            # json_path = os.path.join(self.labelme_path, json_name)
            json_data = json.load(open(json_path))
            filename = json_path.stem
            
            print("filename", filename)

            img_path = None
            img_matches = [_img_path for _img_path in image_paths if filename == _img_path.stem.split('_')[0] ]

            if len(img_matches) > 0:
                # exists .png or .jpg file
                img_colour_path = None
                img_depth_path = None
                
                for img_match in img_matches:
                    if "depth" in img_match.stem:
                        img_depth_path = img_match
                        print("[blue]img_depth_path", img_depth_path)
                    else:
                        img_colour_path = img_match

                if self.worksurface_detection is None:
                    colour_img = cv2.imread(str(img_colour_path))
                    self._process_work_surface_detection(colour_img)

                detections, graph_relations, module, camera = self._process_labelme_img(json_data, img_colour_path, img_depth_path)

                img_paths.append(img_path)
                all_detections.append(detections)
                all_graph_relations.append(graph_relations)
                modules.append(module)
                cameras.append(camera)
               
            else:
                print(f"[red]No image matched for {json_path}")

            if idx > 20:
                print("[red] DEBUG: max 20 steps in sequence")
                break # ! debug

        return img_paths, all_detections, all_graph_relations, modules, cameras

    def _process_work_surface_detection(self, img):
        self.worksurface_detection = WorkSurfaceDetection(img, self.work_surface_ignore_border_width, debug=self.debug_work_surface_detection)

    def _process_labelme_img(self, json_data, img_colour_path, img_depth_path=None):
        detections = []

        # img = Image.open(img_path).convert('RGB') # SLOW
        # img_w, img_h = img.size
        img_w, img_h = imagesize.get(img_colour_path) # fast
        
        depth_img = None
        if img_depth_path is not None:
            depth_img = cv2.imread(str(img_depth_path))
            
        # TODO: we might need to multiply depth by 1/1000 like in pipeline.realsense.py
        
        module = None
        if 'module' in json_data:
            module_str = json_data['module']
            if module_str in Module.__members__:
                module = Module[module_str]
        
        camera = None
        if 'camera' in json_data:
            camera_str = json_data['camera']
            if camera_str in Camera.__members__:
                camera = Camera[camera_str]
        
        if camera is not None:
            print(f"[blue]camera: {camera.name}")
        else:
            print("[blue]camera: None")
        
        camera_info = None
        worksurface_detection = None
        if camera is Camera.basler:
            # TODO: really, run this for every module
            worksurface_detection = self.worksurface_detection
        elif camera is Camera.realsense:            
            # TODO: get real camera_info
            camera_info = CameraInfo(
                header= None,
                height= 480,
                width = 640,
                distortion_model = "plumb_bob",
                D = [0.0, 0.0, 0.0, 0.0, 0.0],
                K = [602.1743774414062, 0.0, 325.2048034667969, 0.0, 600.7815551757812, 246.27980041503906, 0.0, 0.0, 1.0],
                R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                P = [602.1743774414062, 0.0, 325.2048034667969, 0.0, 0.0, 600.7815551757812, 246.27980041503906, 0.0, 0.0, 0.0, 1.0, 0.0],
                binning_x = 0,
                binning_y = 0,
                roi = None
            )
            
        
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

                    detection.mask_contour = self.points_to_contour(shape['points'])
                    detection.box_px = self.contour_to_box(detection.mask_contour)

                    mask = np.zeros((img_h, img_w), np.uint8)
                    cv2.drawContours(mask, [detection.mask_contour], -1, (255), -1)
                    detection.mask = mask
                    
                    detections.append(detection)
                    idx += 1

        detections, markers, poses, graph_img, graph_relations, fps_obb = self.object_detection.get_detections(detections, depth_img=depth_img, worksurface_detection=worksurface_detection, camera_info=camera_info)

        return detections, graph_relations, module, camera
    

    def points_to_contour(self, points):
        obj_point_list =  points # [(x1,y1),(x2,y2),...]
        obj_point_list = np.array(obj_point_list).astype(int) # convert to int
        # obj_point_list = [tuple(point) for point in obj_point_list] # convert back to list of tuples

        # contour
        return obj_point_list

    def contour_to_box(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        # (x,y) is the top-left coordinate of the rectangle and (w,h) its width and height
        box = np.array([x, y, x + w, y + h]).reshape((-1,2)) # convert tlbr (top left bottom right)
        return box

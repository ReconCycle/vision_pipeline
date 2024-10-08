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
from types import SimpleNamespace
import pickle
from object_reid import ObjectReId

# ros package
from context_action_framework.types import Detection, Label, Module, Camera, LabelFace
from sensor_msgs.msg import Image, CameraInfo # CameraInfo needed for pickle
import rospy

# local imports
from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection
from gap_detection.gap_detector_clustering import GapDetectorClustering

from config import load_config
from object_detection_model import ObjectDetectionModel


class LabelMeImporter():
    def __init__(self, ignore_labels=[], use_obj_det_model=True) -> None:
        self.ignore_labels = ignore_labels
        self.worksurface_detection = None

        rospy.init_node("asdf") #! we require an init_node
        
        # config
        self.work_surface_ignore_border_width = 50
        self.debug_work_surface_detection = False

        config = load_config(os.path.expanduser("~/vision_pipeline/config.yaml"))

        # overwrite some settings
        config.reid = False
        config.realsense.debug_clustering = False
        config.obj_detection.debug = False

        self.camera_config = SimpleNamespace()
        self.camera_config.publish_graph_img = False

        model = None
        if use_obj_det_model:
            # optional, load the object detection model. We can get the precise label using this.
            
            model = ObjectDetectionModel(config.obj_detection)

        self.object_detection = ObjectDetection(config=config, camera_config=self.camera_config, model=model, use_ros=False) #! probably we need to add more stuff
        
        self.gap_detector = GapDetectorClustering(config) 

    
    def process_labelme_dir(self, labelme_dir, images_dir=None, filter_cropped=True, use_yield=False, reset_worksurface_each_time=True):
        # for each image, run worksurface detection
        if reset_worksurface_each_time:
            print("[blue]resetting worksurface detection")
            self.worksurface_detection = None
        
        # load in the labelme data
        labelme_dir = Path(labelme_dir)

        if images_dir is None:
            images_dir = labelme_dir
        
        labelme_dir = Path(labelme_dir)
        images_dir = Path(images_dir)
        
        json_paths = list(labelme_dir.glob('*.json'))
        json_paths = [path for path in json_paths if "_qa" not in str(path.stem)]
        json_paths = natsort.os_sorted(json_paths)
        
        image_paths = list(images_dir.glob('*.png')) 
        image_paths.extend(list(images_dir.glob('*.jpg')))
        image_paths = [path for path in image_paths if "viz" not in str(path.stem)]
        
        # don't work on the cropped images. Remove them from the image paths
        if filter_cropped:
            image_paths = [path for path in image_paths if "crop" not in str(path.stem)]
        image_paths = natsort.os_sorted(image_paths)
        
        depth_paths = list(images_dir.glob('*_depth.npy'))
        depth_paths = natsort.os_sorted(depth_paths)
        
        camera_info_paths = list(images_dir.glob('*_camera_info.pickle')) 
        camera_info_paths = natsort.os_sorted(camera_info_paths)

        tqdm_json_paths = tqdm(json_paths)

        if len(json_paths) == 0:
            print("[red]Folder doesn't contain .json files")

        img_paths = []
        all_detections = []
        all_graph_relations = []
        modules = []
        cameras = []
        all_batch_crop_imgs = []
        gt_actions = []

        for idx, json_path in enumerate(tqdm_json_paths):
            tqdm_json_paths.set_description(f"Converting {Path(json_path).stem}")

            # json_path = os.path.join(self.labelme_path, json_name)
            json_data = json.load(open(json_path))
            filename = json_path.stem
            
            print("filename", json_path.name)

            img_matches = [_img_path for _img_path in image_paths if filename == _img_path.stem ]
            
            camera_info_matches = [_cinfo for _cinfo in camera_info_paths if filename == _cinfo.stem.split('_')[0]]
            
            depth_matches = [_depth for _depth in depth_paths if filename == _depth.stem.split('_')[0]]

            camera_info = None
            if len(camera_info_matches) == 1:
                camera_info_path = camera_info_matches[0]
                print("found camera info:", camera_info_path)
                
                pickleFile = open(camera_info_path, 'rb')
                camera_info = pickle.load(pickleFile)
                # depth_img = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
            
            depth_img = None
            if len(depth_matches) == 1:
                depth_path = depth_matches[0]
                depth_img = np.load(depth_path)
                
            colour_img = None
            if len(img_matches) > 0:
                for img_match in img_matches:
                    if "depth" not in img_match.stem:
                        colour_img_path = img_match
                        colour_img = cv2.imread(str(colour_img_path))
                        break
                
            
            if colour_img is not None:

                #TODO: only run work surface detection if camera is basler, not realsense
                if self.worksurface_detection is None and camera_info is None:
                    # camera_info is not None only when realsense is used
                    self.worksurface_detection = WorkSurfaceDetection(colour_img, self.work_surface_ignore_border_width, debug=self.debug_work_surface_detection)

                detections, graph_relations, module, camera, batch_crop_imgs, gt_action = self._process_labelme_img(json_data, colour_img, depth_img, camera_info)

                img_paths.append(colour_img_path)
                all_detections.append(detections)
                all_graph_relations.append(graph_relations)
                modules.append(module)
                cameras.append(camera)
                all_batch_crop_imgs.append(batch_crop_imgs)
                gt_actions.append(gt_action)

                if use_yield:
                    yield colour_img_path, colour_img, detections, graph_relations, module, camera, batch_crop_imgs, gt_action

                    if reset_worksurface_each_time:
                        print("[blue]resetting worksurface detection")
                        self.worksurface_detection = None
               
            else:
                print(f"[red]No image matched for {json_path}")

        if not use_yield:
            return img_paths, colour_img, all_detections, all_graph_relations, modules, cameras, all_batch_crop_imgs, gt_actions
        

    def _process_labelme_img(self, json_data, colour_img, depth_img=None, camera_info=None, apply_scale=1.0):
        detections = []

        # img = Image.open(img_path).convert('RGB') # SLOW
        # img_w, img_h = img.size
        # img_w, img_h = imagesize.get(colour_img_path) # fast
        
        
        img_h, img_w = colour_img.shape[:2]    
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

        #! for llm_prompt_generator. Quick way to get gt action.
        gt_action = None
        if 'gt_action' in json_data:
            gt_action = json_data['gt_action']
        
        if camera is not None:
            print(f"[blue]camera: {camera.name}")
        else:
            print("[blue]camera: None")
        
        worksurface_detection = None
        if camera is Camera.basler:
            worksurface_detection = self.worksurface_detection
        elif camera is Camera.realsense:          
            if camera_info is None:
                print("[red] realsense missing camera_info")
            # camera_info = CameraInfo(
            #     header= None,
            #     height= 480,
            #     width = 640,
            #     distortion_model = "plumb_bob",
            #     D = [0.0, 0.0, 0.0, 0.0, 0.0],
            #     K = [602.1743774414062, 0.0, 325.2048034667969, 0.0, 600.7815551757812, 246.27980041503906, 0.0, 0.0, 1.0],
            #     R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            #     P = [602.1743774414062, 0.0, 325.2048034667969, 0.0, 0.0, 600.7815551757812, 246.27980041503906, 0.0, 0.0, 0.0, 1.0, 0.0],
            #     binning_x = 0,
            #     binning_y = 0,
            #     roi = None
            # )
        else:
            print("[red]labelme_importer: camera not set!")
            
        
        idx = 0
        for shape in json_data['shapes']:
            # only add items that are in the allowed
            if shape['label'] not in self.ignore_labels:

                if shape['shape_type'] == "polygon":

                    detection = Detection()
                    detection.id = idx
                    detection.tracking_id = idx

                    label, label_face = ObjectDetection.fix_labels(shape['label'])

                    detection.label = label
                    detection.label_face = label_face
                    detection.score = float(1.0)

                    detection.mask_contour = self.points_to_contour(shape['points'], apply_scale)
                    detection.box_px = self.contour_to_box(detection.mask_contour, apply_scale)

                    mask = np.zeros((img_h, img_w), np.uint8)
                    cv2.drawContours(mask, [detection.mask_contour], -1, (255), -1)
                    detection.mask = mask
                    
                    detections.append(detection)
                    idx += 1

        detections, markers, poses, graph_img, graph_relations, fps_obb, batch_crop_imgs = self.object_detection.get_detections(detections, colour_img, depth_img=depth_img, worksurface_detection=worksurface_detection, camera_info=camera_info)


        def get_first_detection_in_label_list_by_priority(label_list):
            for label_type in label_list:
                for idx, detection in enumerate(detections):
                    if detection.label is label_type:
                        return detection
            return None

        # when no hca or smoke detector crops are found, use the next relevant label
        # TODO: somehow make this nicer using multiple cropped images.
        if len(batch_crop_imgs) == 0:
            # these labels are in order of priority
            detection = get_first_detection_in_label_list_by_priority([Label.pcb, Label.pcb_covered, Label.battery, Label.battery_covered, Label.plastic_clip, Label.internals, Label.hca_empty, Label.hca_empty])

            if detection is not None:
                crop_size = int(detection.edge_px_large*1.4) #crop such that 80% of the image is the device
                sample_crop, _ = ObjectReId.crop_det(colour_img, detection, size=crop_size)
                sample_crop = cv2.resize(sample_crop, (400, 400), interpolation=cv2.INTER_AREA)

                batch_crop_imgs.append(sample_crop) #! here we add it


        
        graph_relations_text = graph_relations.to_text()
        # print("graph:", graph_relations_text)
        # print("list_wc_components", graph_relations.list_wc_components)
        # print("detections", len(detections))
        
        if depth_img is not None:
            gaps, cluster_img, depth_scaled, device_mask, *_ \
                = self.gap_detector.lever_detector(
                    colour_img,
                    depth_img,
                    detections,
                    graph_relations,
                    camera_info
                )
            print("gaps", gaps)
            

        return detections, graph_relations, module, camera, batch_crop_imgs, gt_action
    

    def labelme_to_detections(self, json_data, sample):
        detections = []
        img_h, img_w = sample.shape[:2]    

        idx = 0
        for shape in json_data['shapes']:
            # only add items that are in the allowed
            if shape['label'] not in self.ignore_labels:

                if shape['shape_type'] == "polygon":

                    detection = Detection()
                    detection.id = idx
                    detection.tracking_id = idx

                    detection.label = Label[shape['label']]
                    # print("detection.label", detection.label)
                    detection.score = float(1.0)

                    detection.valid = True

                    detection.mask_contour = self.points_to_contour(shape['points'])
                    detection.box_px = self.contour_to_box(detection.mask_contour)

                    mask = np.zeros((img_h, img_w), np.uint8)
                    cv2.drawContours(mask, [detection.mask_contour], -1, (255), -1)
                    detection.mask = mask
                    
                    detections.append(detection)
                    idx += 1

        detections, markers, poses, graph_img, graph_relations, fps_obb = self.object_detection.get_detections(detections, depth_img=None, worksurface_detection=None, camera_info=None, use_classify=False)

        return detections, graph_relations


    def points_to_contour(self, points, apply_scale=1.0):
        obj_point_list =  points # [(x1,y1),(x2,y2),...]
        obj_point_list = np.array(obj_point_list)
        if apply_scale != 1.0:
            obj_point_list = obj_point_list * apply_scale
        
        obj_point_list = obj_point_list.astype(int) # convert to int
        # obj_point_list = [tuple(point) for point in obj_point_list] # convert back to list of tuples

        # contour
        return obj_point_list

    def contour_to_box(self, contour, apply_scale=1.0):
        x, y, w, h = cv2.boundingRect(contour)

        if apply_scale != 1.0:
            x, y, w, h = x*apply_scale, y*apply_scale, w*apply_scale, h*apply_scale

        # (x,y) is the top-left coordinate of the rectangle and (w,h) its width and height
        box = np.array([x, y, x + w, y + h]).reshape((-1,2)).astype(int) # convert tlbr (top left bottom right)
        return box

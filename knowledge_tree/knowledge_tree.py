

import os
import sys
import numpy as np
import cv2
import time
import torch
import clip
from PIL import Image
from rich import print #! does this break stuff with jupyter?
from tqdm import tqdm
from pathlib import Path
import csv
import json
import click
import natsort
import pickle
from types import SimpleNamespace

from context_action_framework.types import Detection, Label, Module, Camera, LabelFace
from vision_pipeline.llm_data_generator.labelme_importer import LabelMeImporter

from treelib import Node, Tree

class KnowledgeTree:
    def __init__(self) -> None:
        self.tree = None
        self.devices_data = None

        self.device = None # CLIP

        # self.load_knowledge_tree()
        # self.load_CLIP()

    def load_knowledge_tree(self):

        kg_path = Path("~/datasets2/reconcycle/knowledge_graph").expanduser()
        kg_nodes_path = kg_path / "data" / "nodes"
        kg_edges_path = kg_path / "data" / "graph_edges.csv"

        print("kg_nodes_path", kg_nodes_path)


        # load all examples for each specific device into the tree 
        # load all images and text into a devices_data

        self.tree = Tree()
        self.tree.create_node("device", "device")

        with open(kg_edges_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    child, parent = row
                    if parent != "skill" and child != "skill":
                        self.tree.create_node(child, child, parent=parent)

        devices = self.tree.leaves()

        self.devices_data = {}

        for device in devices:
            # print("device_path", device_path.tag)
            device_path = kg_nodes_path / device.tag

            # print("device_path", device_path)
            
            qa_json_paths = list(device_path.glob('*_qa.json'))
            qa_json_paths = natsort.os_sorted(qa_json_paths)

            json_paths = list(device_path.glob('*.json'))
            json_paths = [path for path in json_paths if "_qa" not in str(path.stem)]
            json_paths = natsort.os_sorted(json_paths)

            image_paths = list(device_path.glob('*.png')) 
            image_paths.extend(list(device_path.glob('*.jpg')))

            # TODO: also check by image size
            # image_paths_cropped = [path for path in image_paths if "crop" in str(path.stem)]
            # image_paths_cropped = natsort.os_sorted(image_paths_cropped)

            depth_paths = list(device_path.glob('*_depth.npy'))
            depth_paths = natsort.os_sorted(depth_paths)
            
            camera_info_paths = list(device_path.glob('*_camera_info.pickle')) 
            camera_info_paths = natsort.os_sorted(camera_info_paths)

            # print("qa_json_paths", qa_json_paths)
            img_path_added = []
            for json_path in json_paths:
                tag = device.tag + "_" + str(json_path.stem)

                # add tree node
                self.tree.create_node(tag, tag, parent=device.tag)

                # json_data = json.load(open(json_path))
                filename = json_path.stem

                colour_img_matches = [_img_path for _img_path in image_paths if filename == _img_path.stem ]

                img_crop_matches = [_img_path for _img_path in image_paths if filename + "_crop" == _img_path.stem ]

                qa_json_matches = [_qa_path for _qa_path in qa_json_paths if filename + "_qa" == _qa_path.stem]
                
                camera_info_matches = [_cinfo for _cinfo in camera_info_paths if filename == _cinfo.stem.split('_')[0]]
                
                depth_matches = [_depth for _depth in depth_paths if filename == _depth.stem.split('_')[0]]

                img_depth_viz_matches = [_img_path for _img_path in image_paths if filename + "_depth_viz" == _img_path.stem ]

                camera_info = None
                if len(camera_info_matches) == 1:
                    camera_info_path = camera_info_matches[0]
                    print("found camera info:", camera_info_path)
                    
                    pickleFile = open(camera_info_path, 'rb')
                    camera_info = pickle.load(pickleFile)
                    # depth_img = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
                
                # depth_img = None
                depth_path = None
                if len(depth_matches) == 1:
                    depth_path = depth_matches[0]
                    # depth_img = np.load(depth_path)

                depth_viz_img_path = None
                if len(img_depth_viz_matches) == 1:
                    depth_viz_img_path = img_depth_viz_matches[0]
                elif len(img_depth_viz_matches) > 1:
                    print("[red]why are there multiple depth viz matches?", img_depth_viz_matches)
                
                    
                # colour_img = None
                colour_img_path = None
                if len(colour_img_matches) == 1:
                    colour_img_path = colour_img_matches[0]
                    # colour_img = cv2.imread(str(colour_img_path))
                elif len(colour_img_matches) > 0:
                    print("[red]why are there multiple colour img matches??", colour_img_matches)

                crop_img_path = None
                if len(img_crop_matches) == 1:
                    crop_img_path = img_crop_matches[0]
                elif len(img_crop_matches) > 1:
                    print("[red]why are there multiple crop matches??", img_crop_matches)
                
                qa_json_path = None
                if len(qa_json_matches) == 1:
                    qa_json_path = qa_json_matches[0]
                elif len(qa_json_matches) > 1:
                    print("[red]why are there multiple qa_json matches??", qa_json_matches)
                elif  len(qa_json_matches) == 0:
                    print(f"[red]no QA json for {json_path}")

                if json_path is not None:
                    json_data = json.load(open(json_path))

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

                if qa_json_path is not None:
                    qa_json_data = json.load(open(qa_json_path))

                # add all info to devices_data dict
                img_path_added.extend([colour_img_path, depth_viz_img_path, crop_img_path])

                device_data = SimpleNamespace()
                device_data.json_path = json_path
                device_data.qa_json_path = qa_json_path
                device_data.colour_img_path = colour_img_path
                device_data.depth_path = depth_path
                device_data.depth_viz_img_path = depth_viz_img_path
                device_data.json_data = json_data
                device_data.qa_json_data = qa_json_data
                device_data.camera_info = camera_info
                device_data.camera = camera
                device_data.module = module
                device_data.crop_img_path = crop_img_path
                device_data.crop_img = None
                device_data.crop_img_encoding = None
                self.devices_data[tag] = device_data

            # add remaining images (that don't have qa_json or json_paths, for example crop examples)
            for image_path in image_paths:
                if image_path not in img_path_added:
                    # add tree node
                    tag = device.tag + "_" + str(image_path.stem)

                    self.tree.create_node(tag, tag, parent=device.tag)
                    device_data = SimpleNamespace()
                    device_data.json_path = None
                    device_data.qa_json_path = None
                    device_data.colour_img_path = None
                    device_data.depth_path = None
                    device_data.depth_viz_img_path = None
                    device_data.camera_info = None
                    device_data.crop_img_path = image_path
                    device_data.crop_img = None
                    device_data.crop_img_encoding = None

                    self.devices_data[tag] = device_data

    def print_knowledge_tree(self):
        print(self.tree.show(stdout=False))


    def load_CLIP(self):
        # TODO: use CLIP to find nearest neighbours given test image/text

        # self.crop_imgs = {}

        # load the crop images
        for tag, device_data in self.devices_data.items():
            crop_path = device_data.crop_img_path
            if crop_path is not None:
                crop_img = Image.open(crop_path)

                device_data.crop_img = crop_img

        # print("self.crop_imgs", self.crop_imgs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("self.device", self.device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)


        # pass all crop images through CLIP

        with torch.no_grad():
            for tag, device_data in self.devices_data.items():
                if device_data.crop_img is not None:
                    img_prepped = self.clip_preprocess(device_data.crop_img).unsqueeze(0).to(self.device)
                    # ? this can be sped up by using batches!
                    img_encoding = self.clip_model.encode_image(img_prepped)
                    device_data.crop_img_encoding = torch.squeeze(img_encoding)



    def get_nearest_neighbour(self, query_img, k=3, debug=False):
        
        imgs_encoding = []
        imgs_encoding_keys = []

        # ! we get nearest neighbours only for ones that have qa_json_paths!!
        for tag, device_data in self.devices_data.items():
            if device_data.crop_img_encoding is not None and device_data.qa_json_path is not None:
                imgs_encoding.append(device_data.crop_img_encoding)
                imgs_encoding_keys.append(tag)

        imgs_encoding = torch.stack(imgs_encoding, dim=0)

        print("imgs_encoding.shape", imgs_encoding.shape)

        with torch.no_grad():
            img_prepped = self.clip_preprocess(query_img).unsqueeze(0).to(self.device)
            img_encoding = self.clip_model.encode_image(img_prepped)

            # todo: find dist
            dist = torch.norm(imgs_encoding - img_encoding, dim=1, p=2)
            knn = dist.topk(k, largest=False)

            if debug:
                print("dist.shape", dist.shape)
                print("knn", knn)

            # best_idx = knn.indices[0]
            # best_value = knn.values[0]
            
            # keys = np.array(list(self.crop_imgs.keys()))
            keys = np.array(imgs_encoding_keys)

            topk_keys = keys[knn.indices.cpu()]
            
            if debug:
                print("topk_keys", topk_keys)

            return topk_keys




# TODO: run LLM on top-3 examples

if __name__ == '__main__':
    kt = KnowledgeTree()
    kt.load_knowledge_tree()
    kt.load_CLIP()

    print("kt.devices_data", kt.devices_data)

    # test_crop_img_path = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped/firealarm_back_10/00_template_0068.jpg")
    test_crop_img_path = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted_cropped/hca_back_03/0006.jpg")

    test_crop_img = Image.open(test_crop_img_path)

    kt.get_nearest_neighbour(test_crop_img)


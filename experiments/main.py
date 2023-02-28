import sys
import os
import cv2
import numpy as np
import json
import argparse
import time
import atexit
from rich import print
import commentjson
import asyncio
from PIL import Image
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# do as if we are in the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

from work_surface_detection_opencv import WorkSurfaceDetection
from object_detection import ObjectDetection
from graph_relations import GraphRelations

from object_reid import ObjectReId
from config import load_config

from context_action_framework.types import Camera

from data_loader_even_pairwise import DataLoaderEvenPairwise
from data_loader import DataLoader

class Main():
    def __init__(self) -> None:
        
        img_path = "experiments/datasets/2023-02-20_hca_backs"
        preprocessing_path = "experiments/datasets/2023-02-20_hca_backs_preprocessing"
        seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
        unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]
        
        dl = DataLoaderEvenPairwise(img_path,
                                    preprocessing_path=preprocessing_path,
                                    batch_size=1,
                                    shuffle=True,
                                    seen_classes=seen_classes,
                                    unseen_classes=unseen_classes)
        
        object_reid = ObjectReId()
        
        for i, (sample1, label1, dets1, sample2, label2, dets2) in enumerate(dl.dataloaders["seen_train"]):
            
            # batch size = 1
            item = 0 # first element in batch
            
            print("dets1:", len(dets1[item])) # list of detections
            print("dets2:", len(dets2[item])) # list of detections
                    
            print("labels:", label1[item], label2[item])
            
            # graph relations actually computed in object_detection.py.... but we don't have that result here.
            
            graph1 = GraphRelations(dets1[item])
            graph2 = GraphRelations(dets2[item])
            # form groups, adds group_id property to detections
            graph1.make_groups()
            graph2.make_groups()
            
            img1 = sample1.detach().cpu().numpy()[item]
            img2 = sample2.detach().cpu().numpy()[item]
            
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # print("img1", type(img1), img1.shape)
            
            
            # TODO: plug the two images + detections into object_reid
            # TODO: optimise by moving SIFT calculation to outside of pairwise loop
            result = object_reid.comparison(img1, dets1, graph1, img2, dets2, graph2, visualise=True)
                        
            cv2.waitKey() # visualise
            
            print("result", result)
            print()
            
            if i > 3:
                break #! debug

if __name__ == '__main__':
    main = Main()
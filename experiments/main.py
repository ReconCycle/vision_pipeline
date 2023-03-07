import sys
import os
from datetime import datetime
import cv2
import numpy as np
import json
from rich import print
from PIL import Image
from tqdm import tqdm
import logging

# do as if we are in the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact

# from work_surface_detection_opencv import WorkSurfaceDetection
# from object_detection import ObjectDetection
from graph_relations import GraphRelations

from object_reid_sift import ObjectReIdSift
from object_reid_superglue import ObjectReIdSuperGlue

from config import load_config

# from context_action_framework.types import Camera

from data_loader_even_pairwise import DataLoaderEvenPairwise
import exp_utils as exp_utils

class Main():
    def __init__(self) -> None:
        exp_utils.init_seeds(1, cuda_deterministic=False)
        visualise = True
        cutoff = 0.5
        model = "superglue"
        img_path = "experiments/datasets/2023-02-20_hca_backs"
        preprocessing_path = "experiments/datasets/2023-02-20_hca_backs_preprocessing_opencv"
        results_base_path = "experiments/results/"
        # seen_classes = ["hca_0", "hca_1", "hca_2", "hca_2a", "hca_3", "hca_4", "hca_5", "hca_6"]
        # unseen_classes = ["hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_11a", "hca_12"]
        
        seen_classes = ["hca_0", "hca_1", "hca_2" , "hca_3", "hca_4", "hca_5", "hca_6", "hca_7", "hca_8", "hca_9", "hca_10", "hca_11", "hca_12"]
        unseen_classes = ["hca_2a", "hca_11a"]
        
        results_path = os.path.join(results_base_path, datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
        if os.path.isdir(results_path):
            print("[red]results_path already exists!")
        else:
            os.makedirs(results_path)
        
        logging.basicConfig(filename=os.path.join(results_path, 'eval.log'), level=logging.DEBUG)
        
        logging.info("visualise: " + str(visualise))
        logging.info("cutoff: " + str(cutoff))
        logging.info("model: " + str(model))
        logging.info("img_path: " + str(img_path))
        logging.info("preprocessing_path: " + str(preprocessing_path))
        logging.info("seen_classes: " + str(seen_classes))
        logging.info("unseen_classes: " + str(unseen_classes))
        
        dl = DataLoaderEvenPairwise(img_path,
                                    preprocessing_path=preprocessing_path,
                                    batch_size=1,
                                    shuffle=True,
                                    seen_classes=seen_classes,
                                    unseen_classes=unseen_classes)
        
        if model.lower() == "sift":
            object_reid = ObjectReIdSift()
        elif model.lower() == "superglue":
            object_reid = ObjectReIdSuperGlue()
        
        results = []
        
        for i, (sample1, label1, dets1, sample2, label2, dets2) in tqdm(enumerate(dl.dataloaders["seen_train"])):
            
            # batch size = 1
            item = 0 # first element in batch
            
            # print("dets1:", len(dets1[item])) # list of detections
            # print("dets2:", len(dets2[item])) # list of detections
                    
            # print("labels:", label1[item], label2[item])
            
            ground_truth = label1[item] == label2[item]
            
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
            
            # TODO: log results
            # TODO: optimise by moving SIFT calculation to outside of pairwise loop
            result = object_reid.compare(img1, graph1, img2, graph2, visualise=visualise)
            
            # TODO: should never be None
            if result is None:
                result = 0.0
                
            result_bin = False
            if result > cutoff:
                result_bin = True
            
            accuracy = ground_truth == result_bin
            
            results.append([label1[item], label2[item], ground_truth, result, accuracy])
            
            # if visualise:
                
            
            print("accuracy", i, accuracy)
            
            # if i > 5:
            #     break #! debug
            
            if i > 0 and (i == 5 or i % 100 == 0):
                results_np = np.array(results)
                avg_accuracy = np.sum(results_np[:, -1])/len(results_np)
                print("\navg_accuracy", avg_accuracy, "\n")
                logging.info("avg accuracy, with " + str(i) + " samples: "+ str(avg_accuracy))
                
                # TODO: log this accuracy.

                    
                    # print("path", path[j])
                    # print("filename", filename)
                    # print("dirname", dirname)
                csv_path = os.path.join(results_path, "acc_" + str(i) + ".csv")
                np.savetxt(csv_path, results_np, delimiter=",", fmt=['%i', '%i', '%i', '%.5f', '%.5f'])
                print("saved path:", csv_path)
            
            if i > 10001:
                return

if __name__ == '__main__':
    main = Main()
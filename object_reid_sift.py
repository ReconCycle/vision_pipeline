from __future__ import annotations
import os
import sys
import numpy as np
import time
import cv2
import copy
from rich import print
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, Point
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from types import SimpleNamespace

from action_predictor.graph_relations import GraphRelations, exists_detection, compute_iou
from context_action_framework.types import Action, Detection, Gap, Label

from helpers import scale_img
from object_reid import ObjectReId
from timeit import default_timer as timer

from object_sift import ObjectSift

# detections have the add properties:
# - obb_normed
# - polygon_normed
# these are the rotated and centered at (0, 0).

font_face = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.5
font_thickness = 1

# object reidentification - reidentify object based on features
class ObjectReIdSift(ObjectReId):
    def __init__(self, config, model) -> None:
        super().__init__(config, model)
        
        self.object_sift = ObjectSift()

        jsonpickle_numpy.register_handlers()
    

    # comparison function for experiments
    def compare(self, img1, poly1, img2, poly2, visualise=False):

        keypoints1, descriptors1 = self.object_sift.calculate_sift(img1, poly1)
        keypoints2, descriptors2 = self.object_sift.calculate_sift(img2, poly2)
    
        sift_score = self.calculate_sift_results(img1, keypoints1, descriptors1, img2, keypoints2, descriptors2, poly1, poly2, visualise)

        if visualise:
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        return sift_score


    def process_detection(self, img, detections, graph_relations, visualise=False):
        # ! THIS IS STILL WIP
        
        print("\nobject re-id, processing detections..." + str(len(detections)))
        print("img.shape", img.shape)
        
        # graph_relations.exists provides a list
        detections_hca_back = graph_relations.exists(Label.hca_back)
        print("Num hca_backs: " + str(len(detections_hca_back)))

        if len(detections_hca_back) >= 1:
            # get the first one only for now
            detection_hca_back = detections_hca_back[0]

            img1_cropped, obb1 = self.find_and_crop_det(img, graph_relations)

            print("obb1", obb1)
            print("img1_cropped", img1_cropped.shape)

            # cv2.imwrite("debug_img1_cropped.png", img1_cropped)

            time1 = timer()

            keypoints1, descriptors1 = self.object_sift.calculate_sift(img1_cropped, obb1)

            # TODO: compare with all objects in our library and find which one it is.
            # TODO: precompute SIFT features
            time2 = timer()

            results = []
            for name, device_list in self.reid_dataset.items():
                print("device name", name)
                device = device_list[0]

                obb1_poly = Polygon(obb1)
                obb2_poly = Polygon(device.obb)

                score, angle = self.calculate_sift_results(img1_cropped, keypoints1, descriptors1, 
                                                         device.img, device.sift_keypoints, device.sift_descriptors, 
                                                         obb1_poly, obb2_poly, visualise)

                print("score", score)

                if visualise:
                    cv2.waitKey()
                    cv2.destroyAllWindows()

                results.append([score, angle])

            results = np.array(results)
            idx_max = np.argmax(results[:, 0])
            best_score = results[idx_max][0]
            angle = results[idx_max][1]
            best_name, best_device_list = list(self.reid_dataset.items())[idx_max]

            print(f"[green]best_score {best_score}, angle: {angle}, {best_name}")

            time3 = timer()
            print("time sift:", time2 - time1)
            print("time reid:", time3 - time2)


            # TODO: compute the one with the largest score
            # TODO: maybe I already did this in process_detection_old

            sys.exit() # !DEBUG
    

    def draw_sift_features(self, img, keypoints, obb_poly=None, vis_id=1, visualise=False):
        img_draw = img.copy()
        if obb_poly is not None:
            cv2.drawContours(img_draw, [np.array(obb_poly.exterior.coords).astype(int)], 0, (0, 255, 0), 2)
        # cv2.drawContours(img_cropped_copy, [obb2_arr], 0, (0, 255, 0), 2)
        im_with_keypoints = cv2.drawKeypoints(img_draw,
                                                keypoints,
                                                np.array([]),
                                                (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if visualise:
            cv2.imshow("keypoints_" + str(vis_id), im_with_keypoints)

        return im_with_keypoints
    
    @staticmethod
    def calculate_sift_matches(des1, des2):
        if len(des1) > 0 and len(des2) > 0:
            
            bf = cv2.BFMatcher()
            
            matches = bf.knnMatch(des1,des2,k=2)
            topResults1 = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    topResults1.append([m])

            matches = bf.knnMatch(des2,des1,k=2)
            topResults2 = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    topResults2.append([m])

            topResults = []
            for match1 in topResults1:
                match1QueryIndex = match1[0].queryIdx
                match1TrainIndex = match1[0].trainIdx

                for match2 in topResults2:
                    match2QueryIndex = match2[0].queryIdx
                    match2TrainIndex = match2[0].trainIdx

                    if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                        topResults.append(match1)
            return topResults
        else:
            return []
    
    @staticmethod
    def get_sift_plot(image1, image2, keypoints1, keypoints2, matches):
        matchPlot = cv2.drawMatchesKnn(
            image1,
            keypoints1,
            image2,
            keypoints2,
            matches,
            None,
            [255,255,255],
            flags=2
        )
        return matchPlot
    
    def calculate_sift_results(self, img1, keypoints1, descriptors1, img2, keypoints2, descriptors2, poly1=None, poly2=None, visualise=False):
        
        if descriptors1 is None:
            print("descriptor 1 is none!")
            return 0.0, None
    
        if descriptors2 is None:
            print("descriptor 2 is none!")
            return 0.0, None
        
        matches = self.calculate_sift_matches(descriptors1, descriptors2)
        
        # 3 matches will always score perfectly because of affine transform
        # let's say we want at least 5 matches to work
        if len(matches) <= 5:
            if visualise:
                print("not enough matches for SIFT")
            # todo: return something else than 0.0, more like undefined.
            return 0.0, None
        
        # get matching points
        pts1_matches = np.array([keypoints1[match[0].queryIdx].pt for match in matches])
        pts2_matches = np.array([keypoints2[match[0].trainIdx].pt for match in matches])
        
        mean_error, median_error, max_error, angle = self.calculate_matching_error(pts1_matches, pts2_matches)
        
        min_num_kpts = min(len(keypoints1), len(keypoints2))

        #! score function 1:
        # score = len(matches)/min_num_kpts
        
        # a median error of less than 0.5 is good
        strength = 1.0 # increase strength for harsher score function

        #! score function 2:
        score = 1/(strength*median_error + 1)
        
        # penalty for few matches
        if len(matches) < 10:
            score = np.clip(score - 0.1, 0, 1) #! do we want this?
        
        if len(matches) > 0 and visualise:
            img1_draw = self.draw_sift_features(img1, keypoints1, poly1)
            img2_draw = self.draw_sift_features(img2, keypoints2, poly2)
            plot = self.get_sift_plot(img1_draw, img2_draw, keypoints1, keypoints2, matches)
            
            # cv2.putText(plot, "unknown: " + str(unknown_template.id), [10, 20], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            # cv2.putText(plot, "template: " + str(object_template.id), [400 + 10, 20], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.putText(plot, "sift score: " + str(round(score, 3)), [10, 40], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "num. keypoints 1: " + str(len(keypoints1)), [10, 80], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "num. keypoints 2: " + str(len(keypoints2)), [10, 100], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.putText(plot, "num. matches: " + str(len(matches)), [10, 60], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "mean_error: " + str(np.round(mean_error, 2)), [10, 120], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "median_error: " + str(np.round(median_error, 2)), [10, 140], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "max_error: " + str(np.round(max_error, 2)), [10, 160], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)

            if angle is not None:
                cv2.putText(plot, "angle: " + str(np.round(np.degrees(angle))), [10, 180], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            # cv2.imshow("sift_result_" + str(unknown_template.id) + "_" + str(object_template.id) + "_" + str(int(rotated)), plot)
            cv2.imshow("sift_result", plot)

        return score, angle
    
    def plot_sift_results():
        # todo: implement
        pass
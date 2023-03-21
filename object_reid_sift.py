from __future__ import annotations
import os
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

from graph_relations import GraphRelations, exists_detection, compute_iou

from context_action_framework.types import Action, Detection, Gap, Label

from helpers import scale_img
from object_reid import ObjectReId

# detections have the add properties:
# - obb_normed
# - polygon_normed
# these are the rotated and centered at (0, 0).

@dataclass
class ObjectTemplate:
    id: int = 0
    detections: list[Any] = field(default_factory=list) # convert to detections_ROS for saving
    sift_keypoints: Any = field(default_factory=Any) # serialise for saving
    sift_descriptors: Any = field(default_factory=Any)

@dataclass
class Score:
    rotated_180 = False
    object_template: ObjectTemplate
    tp_matches: list[Any] = field(default_factory=list)
    tp_missing: list[Any] = field(default_factory=list)
    unknown_missing: list[Any] = field(default_factory=list)
    overall_score = 0.0
    iou_score = 0.0
    sift_score = 0.0

font_face = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.5
font_thickness = 1

# object reidentification - reidentify object based on features
class ObjectReIdSift(ObjectReId):
    def __init__(self) -> None:
        
        self.object_templates = []
        self.template_imgs = []
        self.counter = 0
        
        self.OVERALL_SCORE_THRESHOLD = 0.10
        self.IOU_SCORE_THRESHOLD = 0.75
        self.SIFT_THRESHOLD = 0.20
        
        self.folder = "object_templates/"
        
        self.sift = cv2.SIFT_create()

        jsonpickle_numpy.register_handlers()
        
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        
        self.load_templates()
    
    def load_templates(self):
        # load json from file
        if os.path.isdir(self.folder):
            
            list_dir = os.listdir(self.folder)
            img_list = sorted([l for l in list_dir if l.endswith(".png")])
            json_list = sorted([l for l in list_dir if l.endswith(".json")])
            
            print("img_list", img_list)
            for img_filename in img_list:
                img = cv2.imread(os.path.join(self.folder, img_filename))
                self.template_imgs.append(img)
                
            print("json_list", json_list)
            
            print("loading json from file...")
            for json_filename in json_list:
                try:
                    with open(os.path.join(self.folder, json_filename), 'r') as json_file:
                        object_template_serialised = jsonpickle.decode(json_file.read(), keys=True)
                        
                        # for object_template in object_templates_serialised:
                        # fix for cv2.keypoints not being serialised correctly
                        keypoints = []
                        for point in object_template_serialised.sift_keypoints:
                            temp = cv2.KeyPoint(
                                x=point[0][0],
                                y=point[0][1],
                                size=point[1],
                                angle=point[2],
                                response=point[3],
                                octave=point[4],
                                class_id=point[5]
                            )
                            keypoints.append(temp)
                        object_template_serialised.sift_keypoints = keypoints
                        
                        # fix for labels not being serialised properly
                        for detection in object_template_serialised.detections:
                            detection.label = Label(detection.label)
                        
                        self.object_templates.append(object_template_serialised)
                        print("[green]loaded " + json_filename + "[/green]")
                except ValueError as e:
                    print("couldn't read json file properly: ", e)
        else:
            print("[red]" + os.path.join(self.folder, self.filename) + " doesn't exist![/red]")
    
    
    def save_templates(self, new_object_template=None, object_img=None):
        if new_object_template is not None:
            self.object_templates.append(new_object_template)
            self.template_imgs.append(object_img)
            
            filename = "object_template_" + str(len(self.object_templates) - 1)
            if object_img is not None:
                # save object_img
                
                if not os.path.exists(os.path.join(self.folder, filename)):
                    cv2.imwrite(os.path.join(self.folder, filename + ".png"), object_img)
                else:
                    raise ValueError("Image already exists! " + filename)

            serialised_keypoints = []
            for point in new_object_template.sift_keypoints:
                temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
                serialised_keypoints.append(temp)
            
            # fix for labels not being serialised properly
            serialised_detections = copy.deepcopy(new_object_template.detections)
            for detection in serialised_detections:
                detection.label = detection.label.value
            
            object_template_serialised = ObjectTemplate(
                new_object_template.id,
                serialised_detections,
                serialised_keypoints,
                new_object_template.sift_descriptors)
        
            obj_templates_json_str = jsonpickle.encode(object_template_serialised, keys=True, warn=True, indent=2)
            with open(os.path.join(self.folder, filename + ".json"), 'w', encoding='utf-8') as f:
                f.write(obj_templates_json_str)
        
        # convert to json and save
        print("saving object templates!")


    # comparison function for experiments
    def compare(self, img1, poly1, img2, poly2, visualise=False):
        
        # img1_cropped, obb_poly1 = self.find_and_crop_det(img1, graph1)
        # img2_cropped, obb_poly2 = self.find_and_crop_det(img2, graph2)
        
        keypoints1, descriptors1 = self.calculate_sift(img1, poly1, visualise, vis_id=1)
        
        template1 = SimpleNamespace()
        template1.id = 1
        template1.sift_keypoints = keypoints1
        template1.sift_descriptors = descriptors1
        
        keypoints2, descriptors2 = self.calculate_sift(img2, poly2, visualise, vis_id=2)
        
        template2 = SimpleNamespace()
        template2.id = 2
        template2.sift_keypoints = keypoints2
        template2.sift_descriptors = descriptors2
    
        sift_score = self.calculate_sift_results(img1, template1, img2, template2, False, visualise)

        if visualise:
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        return sift_score


    def process_detection(self, img, detections, graph_relations, visualise=False):
        
        print("\nobject re-id, processing detections..." + str(len(detections)))
        print("img.shape", img.shape)
        
        # graph_relations.exists provides a list
        detections_hca_back = graph_relations.exists(Label.hca_back)
        print("Num unknown devices: " + str(len(detections_hca_back)))
        
        # gather all devices into list: unknown_templates
        unknown_imgs = []
        unknown_templates = []
        
        for detection_hca_back in detections_hca_back:
            # todo: we could use the groups instead
            dets = []
            print("detected hca_back")
            dets = graph_relations.get_all_inside(detection_hca_back)
        
            print("inside dets", len(dets))
            unknown_dets = [detection_hca_back] + dets
            
            img_cropped, center_cropped = self.get_det_img(img, detection_hca_back)
            unknown_imgs.append(img_cropped)
            
            # todo: I need the OBB here, to ignore keypoints outside of OBB
            keypoints, descriptors = self.calculate_sift(img_cropped, center_cropped, detection_hca_back, visualise)
            
            # add this property
            for unknown_det in unknown_dets:
                unknown_det.obb_normed = self.rotated_and_centered_obb(unknown_det.obb, detection_hca_back.center, detection_hca_back.tf.rotation, world_coords=True)
                unknown_det.polygon_normed = self.rotated_and_centered_obb(unknown_det.polygon, detection_hca_back.center, detection_hca_back.tf.rotation, world_coords=True)
                
            # unknown_objs.append(unknown_dets)
            unknown_templates.append(ObjectTemplate(len(unknown_templates), unknown_dets, keypoints, descriptors))

        new_id_pairs = []
        best_iou_scores = []
        best_sift_scores = []
        best_overall_scores = []

        for unknown_template_id, unknown_template in enumerate(unknown_templates):
        
            print("\ndoing comparison...")
            # there exist object templates. Let's compare
            scores_list = []
            for object_template_id, object_template in enumerate(self.object_templates):
                
                # ! Move to comparison function: above
                
                # compare
                tp_dets = object_template.detections
                
                # todo: does rotation 180 degrees matter for sift?
                # sift is largely rotation invariant
                uk_img = unknown_imgs[unknown_template_id]
                tp_img = self.template_imgs[object_template_id]
                sift_score = self.calculate_sift_results(uk_img, unknown_template, tp_img, object_template, False, visualise)
                
                # rotate over symmetry
                for rotate_180 in [True, False]:
                    print("rotated:", rotate_180)
                    
                    score = Score(object_template)
                    score.rotated_180 = rotate_180
                    score.sift_score = sift_score
                
                    for tp_det in tp_dets:
                        uk_dets_with_label = exists_detection(unknown_template.detections, tp_det.label)
                        if len(uk_dets_with_label) > 0:
                            # ? maybe we want to iterate over all detections that have the same label
                            uk_det_with_label = uk_dets_with_label[0]
                            print("exists", tp_det.label.name)
                            
                            uk_det_obb_normed = uk_det_with_label.obb_normed
                            if rotate_180:
                                uk_det_obb_normed = self.rotate_180(uk_det_with_label.obb_normed)
                            
                            # todo: compare area
                            # todo: compare height/width ratio
                            # compare IOU
                            iou = compute_iou(tp_det.obb_normed, uk_det_obb_normed)
                            print("iou", iou)

                            score.tp_matches.append([
                                tp_det, uk_det_with_label, iou
                            ])
                            
                        else:
                            print(tp_det.label.name, "doesn't exist in unknown object!")
                            #  detection not found in new device!
                            score.tp_missing.append(tp_det)
                    
                    for unknown_det in unknown_template.detections:
                        if unknown_det not in [tp_match[1] for tp_match in score.tp_matches]:
                            # object missing in detection template
                            print(unknown_det.label.name, "in unknown device, but not in template!")
                            score.unknown_missing.append(unknown_det)
                    
                    ious = [tp_match[2] for tp_match in score.tp_matches]
                    score.iou_score = np.sum(ious) / (len(ious) + len(score.tp_missing) * 0.5 + len(score.unknown_missing) * 0.5)
                    
                    # todo: design better overall score
                    score.overall_score = score.iou_score * score.sift_score
                    
                    scores_list.append(score)
                
            
            iou_scores = np.array([score.iou_score for score in scores_list])
            sift_scores = np.array([score.sift_score for score in scores_list])
            overall_scores = np.array([score.overall_score for score in scores_list])

            # compute iou score
            if len(self.object_templates) > 0:
                # get best score
                best_score_id = np.argmax(iou_scores)
                best_score = scores_list[best_score_id]
                print("best iou score", best_score.iou_score, best_score.object_template.id)
                best_iou_scores.append(best_score)
            else:
                best_iou_scores.append(None)

            # best sift score
            if len(self.object_templates) > 0:
                best_sift_score_id = np.argmax(sift_scores)
                best_sift_score = scores_list[best_sift_score_id]
                print("best sift score", best_sift_score.sift_score, best_sift_score.object_template.id)
                best_sift_scores.append(best_sift_score)
            else:
                best_sift_scores.append(None)
            
            if len(self.object_templates) > 0:
                best_overall_score_id = np.argmax(overall_scores)
                best_overall_score = scores_list[best_overall_score_id]
                print("best overall score", best_overall_score.overall_score, best_overall_score.object_template.id)
                best_overall_scores.append(best_overall_score)
            else:
                best_overall_scores.append(None)
            
            print("overall_scores", overall_scores)
            
            # if the device doesn't match any devices well, add it as a new object template
            # or if there aren't any object templates, add it
            # is_new_device = True
            # if len(self.object_templates) > 0:
            #     #! WE SHOULD NOT DO THIS, but include sift score in overall score!
            #     is_new_device = best_score.overall_score < self.SCORE_THRESHOLD or best_score.sift_score < self.SIFT_THRESHOLD \
            #     or (best_score.overall_score < self.SCORE_THRESHOLD/2 and best_score.sift_score < self.SIFT_THRESHOLD/2)
            
            if len(self.object_templates) == 0 or best_score.overall_score < self.OVERALL_SCORE_THRESHOLD:
                print("[green]ADDED TEMPLATE![/green]")
                new_obj_template_id = len(self.object_templates)

                new_id_pairs.append([new_obj_template_id, unknown_template_id])
                
                self.save_templates(ObjectTemplate(new_obj_template_id, unknown_template.detections, unknown_template.sift_keypoints, unknown_template.sift_descriptors),
                                    unknown_imgs[unknown_template_id])
        
        if visualise:
            self.visualise(img, new_id_pairs, unknown_templates, best_iou_scores, best_sift_scores, best_overall_scores)
        
    
    def visualise(self, img, new_id_pairs, unknown_templates, best_iou_scores, best_sift_scores, best_overall_scores):
        ###################################
        # VISUALISE!
        ###################################
        
        # now rotate the device such that the longest edge is along the y-axis.
        # we rotate all devices on the fly.
        img_info = np.zeros((640, 1400, 3))
        
        # counter for how many times we have run this function
        title_text = str(self.counter) + ", threshold: " + str(self.OVERALL_SCORE_THRESHOLD)
        cv2.putText(img_info, title_text, [10, 10], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
        
        # show object templates
        column_spacing = 200
        row_spacing = 320
        cv2.putText(img_info, "Object Templates:", [10, 30], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
        for idx, object_template in enumerate(self.object_templates):
            tp_dets = object_template.detections
            
            # add new text for new object template
            id_text =  "id: " + str(object_template.id)
            cv2.putText(img_info, id_text, [10 + idx*column_spacing, 50], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
            
            id_pair = next((new_id_pair for new_id_pair in new_id_pairs if new_id_pair[0] == idx), None)
            if id_pair is not None:
                id_text = "NEW! unknown id: " + str(id_pair[1])
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, 70], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
            
            for tp_det in tp_dets:
                obb_px = self.m_to_px(tp_det.obb_normed)
                # shift each object template 200px to right
                obb_px += np.array([100, 180]) + np.array([idx*column_spacing, 0])
                cv2.drawContours(img_info, [obb_px], 0, (0, 255, 0), 2)
                
                # poly_px = self.m_to_px(tp_det.polygon_normed)
                # poly_px += np.array([100, 150]) + np.array([idx*200, 0])
                # cv2.drawContours(img_info, [poly_px], 0, (0, 255, 0), 2)
                
                
                cv2.putText(img_info, tp_det.label.name, obb_px[0], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
        
        # show unknown detections
        cv2.putText(img_info, "Unknown Objects:", [10, row_spacing], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
        for idx, unknown_template in enumerate(unknown_templates):
            if best_iou_scores[idx] is not None:
                id_text = "iou, id: " + str(best_iou_scores[idx].object_template.id) + ", p=" + str(round(best_iou_scores[idx].iou_score, 2))
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 30], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
                
            if best_sift_scores[idx] is not None:
                id_text = "sift, id: " + str(best_sift_scores[idx].object_template.id) + ", p=" + str(round(best_sift_scores[idx].sift_score, 3))
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 50], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
                
            if best_overall_scores[idx] is not None:
                id_text = "overall, id: " + str(best_overall_scores[idx].object_template.id) + ", p=" + str(round(best_sift_scores[idx].overall_score, 3))
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 70], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
            
            id_pair = next((new_id_pair for new_id_pair in new_id_pairs if new_id_pair[1] == idx), None)
            if id_pair is not None:
                id_text = "NEW! template id: " + str(id_pair[0])
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 90], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)

            for unknown_det in unknown_template.detections:
                pass
                obb_px = self.m_to_px(unknown_det.obb_normed)
                obb_px += np.array([100 + idx*column_spacing, row_spacing + 190])
                cv2.drawContours(img_info, [obb_px], 0, (0, 0, 255), 2)
                cv2.putText(img_info, unknown_det.label.name, obb_px[0], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
        
        # try:
        cv2.imshow("img", scale_img(img))
        cv2.imshow("reid", img_info)
        
        # k = cv2.waitKey(0)
        # if k & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        
        # input("Press Enter to continue...")
        # cv2.waitKey(1)
        
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.counter += 1
                            
    def add_template(self):
        pass
        
    
    def calculate_sift(self, img_cropped, obb_poly, visualise=False, vis_id=1):

        keypoints, descriptors = self.sift.detectAndCompute(img_cropped, None)
        keypoints_in_poly = []
        descriptors_in_poly = []

        if keypoints is None:
            print("keypoints is None!")
            return [], []
        
        if descriptors is None:
            print("descriptors is None!")
            return [], []
        
        # only include keypoints that are inside the obb
        for keypoint, descriptor in zip(keypoints, descriptors):
            if obb_poly.contains(Point(*keypoint.pt)):
                keypoints_in_poly.append(keypoint)
                descriptors_in_poly.append(descriptor)
        
        # descriptors is an array, keypoints is a list
        descriptors_in_poly = np.array(descriptors_in_poly)

        if visualise:
            img_cropped_copy = img_cropped.copy()
            cv2.drawContours(img_cropped_copy, [np.array(obb_poly.exterior.coords).astype(int)], 0, (0, 255, 0), 2)
            # cv2.drawContours(img_cropped_copy, [obb2_arr], 0, (0, 255, 0), 2)
            im_with_keypoints = cv2.drawKeypoints(img_cropped_copy,
                                                    keypoints_in_poly,
                                                    np.array([]),
                                                    (0, 0, 255),
                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("keypoints_" + str(vis_id), im_with_keypoints)

        return keypoints_in_poly, descriptors_in_poly
    
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
    
    def calculate_sift_results(self, uk_img, unknown_template, tp_img, object_template, rotated, visualise=False):
        keypoints1 = unknown_template.sift_keypoints
        descriptors1 = unknown_template.sift_descriptors
        keypoints2 = object_template.sift_keypoints
        descriptors2 = object_template.sift_descriptors
        
        if descriptors1 is None:
            print("descriptor 1 is none!")
            return 0.0
    
        if descriptors2 is None:
            print("descriptor 2 is none!")
            return 0.0
        
        matches = self.calculate_sift_matches(descriptors1, descriptors2)
        
        # 3 matches will always score perfectly because of affine transform
        # let's say we want at least 5 matches to work
        if len(matches) <= 5:
            if visualise:
                print("not enough matches for SIFT")
            # todo: return something else than 0.0, more like undefined.
            return 0.0
        
        # get matching points
        pts1_matches = np.array([keypoints1[match[0].queryIdx].pt for match in matches])
        pts2_matches = np.array([keypoints2[match[0].trainIdx].pt for match in matches])
        
        mean_error, median_error, max_error = self.calculate_matching_error(pts1_matches, pts2_matches)
        
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
            plot = self.get_sift_plot(uk_img, tp_img, keypoints1, keypoints2, matches)
            
            cv2.putText(plot, "unknown: " + str(unknown_template.id), [10, 20], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.putText(plot, "template: " + str(object_template.id), [400 + 10, 20], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.putText(plot, "sift score: " + str(round(score, 3)), [10, 40], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "num. keypoints 1: " + str(len(keypoints1)), [10, 80], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "num. keypoints 2: " + str(len(keypoints2)), [10, 100], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.putText(plot, "num. matches: " + str(len(matches)), [10, 60], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "mean_error: " + str(np.round(mean_error, 2)), [10, 120], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "median_error: " + str(np.round(median_error, 2)), [10, 140], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "max_error: " + str(np.round(max_error, 2)), [10, 160], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            # cv2.imshow("sift_result_" + str(unknown_template.id) + "_" + str(object_template.id) + "_" + str(int(rotated)), plot)
            cv2.imshow("sift_result", plot)
        
        # return score
        return score
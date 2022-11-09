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
from shapely.geometry import Polygon
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

from graph_relations import GraphRelations, exists_detection, compute_iou

from context_action_framework.types import Action, Detection, Gap, Label, detections_to_ros, detections_to_py


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
class ObjectReId:
    def __init__(self) -> None:
        
        self.object_templates = []
        self.template_imgs = []
        self.counter = 0
        
        self.OVERALL_SCORE_THRESHOLD = 0.15
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
            dets = []
            print("detected hca_back")
            dets = graph_relations.get_all_inside(detection_hca_back)
        
            print("inside dets", len(dets))
            unknown_dets = [detection_hca_back] + dets
            
            img_cropped, center_cropped = self.get_det_img(img, detection_hca_back)
            unknown_imgs.append(img_cropped)
            
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
                # compare
                tp_dets = object_template.detections
                
                # sift is largely rotation invariant
                uk_img = unknown_imgs[unknown_template_id]
                tp_img = self.template_imgs[object_template_id]
                sift_score = self.calculate_sift_results(uk_img, unknown_template, tp_img, object_template, visualise)
                
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
                    
                    score.overall_score = score.iou_score * score.sift_score
                    
                    scores_list.append(score)
                
            # ! compute overall score
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
                id_text = "iou, id: " + str(best_iou_scores[idx].object_template.id) + " with prob: " + str(round(best_iou_scores[idx].iou_score, 2))
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 30], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
                
            if best_sift_scores[idx] is not None:
                id_text = "sift, id: " + str(best_sift_scores[idx].object_template.id) + " with prob: " + str(round(best_sift_scores[idx].sift_score, 3))
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 50], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
                
            if best_overall_scores[idx] is not None:
                id_text = "overall, id: " + str(best_overall_scores[idx].object_template.id) + " with prob: " + str(round(best_sift_scores[idx].sift_score, 3))
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 70], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
            
            id_pair = next((new_id_pair for new_id_pair in new_id_pairs if new_id_pair[1] == idx), None)
            if id_pair is not None:
                id_text = "NEW! template id: " + str(id_pair[0])
                cv2.putText(img_info, id_text, [10 + idx*column_spacing, row_spacing + 90], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)

            for unknown_det in unknown_template.detections:
                # obb = self.rotated_and_centered_obb(component.obb, detection_hca_back.center, detection_hca_back.tf.rotation)
                obb_px = self.m_to_px(unknown_det.obb_normed)
                obb_px += np.array([100 + idx*column_spacing, row_spacing + 190])
                cv2.drawContours(img_info, [obb_px], 0, (0, 0, 255), 2)
                cv2.putText(img_info, unknown_det.label.name, obb_px[0], font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
        
        # try:
        cv2.imshow("img", self.scale_img(img))
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
    
    @staticmethod
    def scale_img(img, scale_percent=50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def m_to_px(x):
        if isinstance(x, np.ndarray):
            # x is in meters, to show in pixels, scale up...
            x = x * 1500
            x = x.astype(int)
            return x
        elif isinstance(x, Polygon):
            x_arr = np.array(x.exterior.coords)
            x_arr = x_arr * 1500
            x_arr = x_arr.astype(int)
            return x_arr

    @staticmethod
    def ros_quat_to_rad(quat):
        quat_to_np = lambda quat : np.array([quat.x, quat.y, quat.z, quat.w])
        rot_mat = Rotation.from_quat(quat_to_np(quat)).as_euler('xyz')

        # print("rot_mat", rot_mat)

        # rotation is only in one dimenseion
        angle = 0.0
        for angle_in_dim in rot_mat:
            if angle_in_dim != 0.0:
                angle = angle_in_dim
        
        return angle

    @classmethod
    def rotated_and_centered_obb(cls, obb_or_poly, center, quat, new_center=None, world_coords=False):
        if isinstance(obb_or_poly, Polygon):
            points = np.array(obb_or_poly.exterior.coords)
            points = points[:, :2] # ignore the z value
        else:
            # it is already an array
            points = obb_or_poly
        
        # print("points", points.shape)
        # print("center", center.shape)

        # move obb to (0, 0)
        points_centered = points - center

        # sometimes the angle is in the z axis (basler) and for realsense it is different.
        # this is a hack for that
        angle = cls.ros_quat_to_rad(quat)

        # correction to make longer side along y-axis
        angle = ((0.5 * np.pi) - angle) % np.pi

        # for pixel coords we use the same rotation matrix as in getRotationMatrix2D
        # that we use to rotate the image
        # I don't know why OPENCV uses a different rotation matrix in getRotationMatrix2D
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle),  np.cos(angle)]])

        # for world coordinates, use standard rotation matrix
        if world_coords:
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])

        points_rotated = np.dot(points_centered, rot_mat.T)

        # print("points_rotated", points_rotated.shape)
        if new_center is not None:
            points_rotated += new_center
        
        if isinstance(obb_or_poly, Polygon):
            return Polygon(points_rotated)
        else:
            return points_rotated
        
    @staticmethod
    def rotate_180(obb_or_poly):
        if isinstance(obb_or_poly, Polygon):
            points = np.array(obb_or_poly.exterior.coords)
        else:
            # it is already an array
            points = obb_or_poly
            
        angle = np.pi
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle),  np.cos(angle)]])
        
        points_rotated = np.dot(points, rot_mat.T)
        
        if isinstance(obb_or_poly, Polygon):
            return Polygon(points_rotated)
        else:
            return points_rotated
        
    @classmethod
    def get_det_img(cls, img, det):
        center = det.center_px
        center = (int(center[0]), int(center[1]))
        
        height, width = img.shape[:2]
        
        # rotate image around center
        angle_rad = cls.ros_quat_to_rad(det.tf.rotation)
        angle_rad = ((0.5 * np.pi) - angle_rad) % np.pi
        
        # note: getRotationMatrix2D rotation matrix is different from standard rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, np.rad2deg(angle_rad), 1.0)
        img_rot = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        # crop image around center point
        size = 200
        x1 = np.clip(int(det.center_px[0]) - size, 0, width)
        x2 = np.clip(int(det.center_px[0]) + size, 0, width)
        
        y1 = np.clip(int(det.center_px[1]) - size, 0, height)
        y2 = np.clip(int(det.center_px[1]) + size, 0, height)
        
        
        img_cropped = img_rot[y1:y2, x1:x2]
        
        # new center at:
        center_cropped = det.center_px[0] - x1, det.center_px[1] - y1
        
        # debug
        # cv2.circle(img_cropped, (int(center_cropped[0]), int(center_cropped[1])), 6, (0, 0, 255), -1)
        
        return img_cropped, center_cropped
    
    
    def calculate_sift(self, img_cropped, center_cropped, hca_back, visualise=False):
        # todo: ignore keypoints outside hca_back
        keypoints, descriptors = self.sift.detectAndCompute(img_cropped, None)
        
        # print("keypoints", keypoints)

        # unrotated obb:
        # obb = hca_back.obb_px - hca_back.center_px + center_cropped
        # obb_arr = np.array(obb).astype(int)
        # cv2.drawContours(img_cropped, [obb_arr], 0, (0, 255, 255), 2)
        
        # rotated obb:
        obb2 = self.rotated_and_centered_obb(hca_back.obb_px, hca_back.center_px, hca_back.tf.rotation, center_cropped)
        obb2_arr = np.array(obb2).astype(int)
        
        cv2.drawContours(img_cropped, [obb2_arr], 0, (0, 255, 0), 2)
        
        # poly = self.rotated_and_centered_obb(hca_back.polygon_px, hca_back.center_px, hca_back.tf.rotation, center_cropped)
        # poly_arr = np.array(poly.exterior.coords).astype(int)
        # print("poly_arr", poly_arr)
        
        im_with_keypoints = cv2.drawKeypoints(img_cropped,
                                                keypoints,
                                                np.array([]),
                                                (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # todo: only include keypoints that are inside the obb
        if visualise:
            cv2.imshow("keypoints", im_with_keypoints)

        return keypoints, descriptors
    
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
    def calculate_sift_score(num_matches, num_keypoint1, num_keypoint2):
        if num_matches > 0 and num_keypoint1 > 0 and num_keypoint2 > 0:
            return num_matches/min(num_keypoint1, num_keypoint2)
        else:
            return 0
    
    @staticmethod
    def get_sift_plot(image1, image2, keypoint1, keypoint2,matches):
        matchPlot = cv2.drawMatchesKnn(
            image1,
            keypoint1,
            image2,
            keypoint2,
            matches,
            None,
            [255,255,255],
            flags=2
        )
        return matchPlot
    
    def calculate_sift_results(self, uk_img, unknown_template, tp_img, object_template, visualise=False):
        keypoint1 = unknown_template.sift_keypoints
        descriptor1 = unknown_template.sift_descriptors
        keypoint2 = object_template.sift_keypoints
        descriptor2 = object_template.sift_descriptors
        
        if descriptor1 is None:
            print("descriptor 1 is none!")
    
        if descriptor2 is None:
            print("descriptor 2 is none!")
        
        matches = self.calculate_sift_matches(descriptor1, descriptor2)
        score = self.calculate_sift_score(len(matches),len(keypoint1),len(keypoint2))
        
        if len(matches) > 0 and visualise:
            plot = self.get_sift_plot(uk_img, tp_img, keypoint1, keypoint2, matches)
            
            cv2.putText(plot, "unknown: " + str(unknown_template.id), [10, 20], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.putText(plot, "template: " + str(object_template.id), [400 + 10, 20], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.putText(plot, "sift score: " + str(round(score, 3)), [10, 40], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "num. matches: " + str(len(matches)), [10, 60], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "num. keypoints 1: " + str(len(keypoint1)), [10, 80], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            cv2.putText(plot, "num. keypoints 2: " + str(len(keypoint2)), [10, 100], font_face, font_scale, [0, 255, 0], font_thickness, cv2.LINE_AA)
            
            cv2.imshow("sift_result_" + str(unknown_template.id) + "_" + str(object_template.id), plot)
        
        return score
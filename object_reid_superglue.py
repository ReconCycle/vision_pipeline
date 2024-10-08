from __future__ import annotations
import os
import numpy as np
import time
import cv2
from rich import print
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, Point
import matplotlib.cm as cm
from PIL import Image
import math
from scipy.spatial.transform import Rotation

from types import SimpleNamespace

from context_action_framework.graph_relations import GraphRelations, exists_detection, compute_iou

from helpers import scale_img, add_angles
from vision_pipeline.object_reid import ObjectReId
from pathlib import Path
import torch
from superglue_training.models.matching import Matching
from superglue_training.utils.common import make_matching_plot_fast, frame2tensor, VideoStreamer



class ObjectReIdSuperGlue(ObjectReId):
    #! We are using superglue ONLY for rotation estimation
    def __init__(self, model="indoor", match_threshold=0.5) -> None:
        super().__init__()
    
        torch.set_grad_enabled(False)
        
        opt = SimpleNamespace()
        opt.superglue = model
        opt.nms_radius = 4
        opt.sinkhorn_iterations = 20
        opt.match_threshold = match_threshold
        opt.show_keypoints = True
        opt.keypoint_threshold = 0.005
        opt.max_keypoints = -1
        
        weights_mapping = {
                'superpoint': Path(__file__).parent / 'superglue/models/weights/superpoint_v1.pth',
                'indoor': Path(__file__).parent / 'superglue/models/weights/superglue_indoor.pth',
                'outdoor': Path(__file__).parent / 'superglue/models/weights/superglue_outdoor.pth',
                'coco_homo': Path(__file__).parent / 'superglue/models/weights/superglue_cocohomo.pt'
            }

        try:
            curr_weights_path = str(weights_mapping[opt.superglue])
            if not os.path.isfile(curr_weights_path):
                print(f"[red]{curr_weights_path} is not a file!")
        except:
            if os.path.isfile(opt.superglue) and (os.path.splitext(opt.superglue)[-1] in ['.pt', '.pth']):
                curr_weights_path = str(opt.superglue)
            else:
                raise ValueError("Given --superglue path doesn't exist or invalid")

        print("curr_weights_path", curr_weights_path)

        self.opt = opt
        config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights_path': curr_weights_path,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
        # self.timer = AverageTimer()

    def angle_from_homo(self, homo):
        # https://stackoverflow.com/questions/58538984/how-to-get-the-rotation-angle-from-findhomography
        u, _, vh = np.linalg.svd(homo[0:2, 0:2])
        R = u @ vh
        angle = math.atan2(R[1,0], R[0,0]) # angle between [-pi, pi)
        return angle


    def compare_full_img(self, img0, graph0, img1, graph1, visualise=False):
        img0_cropped, obb_poly1 = self.find_and_crop_det(img0, graph0)
        img1_cropped, obb_poly2 = self.find_and_crop_det(img1, graph1)
        
        img0 = cv2.cvtColor(img0_cropped, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img1_cropped, cv2.COLOR_RGB2GRAY)

        self.compare(img0, img1, visualise=visualise)


    def compare(self, img1, img2, gt=None, affine_fit=False, visualise=False, debug=True):
        if visualise:
            print("[blue]starting compare...[/blue]")
        
        img1_tensor = frame2tensor(img1, self.device)
        last_data = self.matching.superpoint({'image': img1_tensor})
        last_data = {k+'0': last_data[k] for k in self.keys}
        last_data['image0'] = img1_tensor
        
        # TODO: ignore matches outside OBB
        
        img2_tensor = frame2tensor(img2, self.device)
        
        pred = self.matching({**last_data, 'image1': img2_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        # confidence = pred['matching_scores0'][0].cpu().numpy()
        #! gives error: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = confidence[valid]

        k_thresh = self.matching.superpoint.config['keypoint_threshold']
        m_thresh = self.matching.superglue.config['match_threshold']
        
        # print("matches[valid].shape", len(matches[valid]), matches[valid].shape)
        # print("kpts0", len(kpts0), kpts0.shape)
        # print("kpts0", len(kpts1), kpts1.shape)

        affine_loss = 100.0
        score_ratio = 0.0
        affine_median_error = 100
        angle_est = None
        est_homo_ransac = None
        angle_from_similarity = None
        angle_from_est_affine_partial = None

        # 3 matches will always score perfectly because of affine transform
        # let's say we want at least 5 matches to work

        #! we set it to 3, because we assume that we have already classified it correctly
        if len(matches[valid]) <= 3:
            if debug:
                print("not enough matches for SuperGlue", len(matches[valid]))
            # todo: return something else than 0.0, more like undefined.
            # return 0.0, None, None
        else:
            # see superglue_training/match_homography.py, for how we do stuff for evaluation
            # sort_index = np.argsort(mconf)[::-1][0:4]
            # est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
            
            if False: #! method disabled because not in use
                est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=300)
                if est_homo_ransac is not None:
                    # get rotation from homography
                    angle_est = self.angle_from_homo(est_homo_ransac)
                    # print("angle before negative:", angle_est, "in degrees:", np.rad2deg(angle_est))
                    angle_est = add_angles(0, angle_est) # make sure angle_est is in [-pi, pi)
                    angle_est = add_angles(0, -angle_est) # take the "inverse". Use add_angles to make sure we are still between -pi, pi.
                    
                    # print("angle_est using findHomography", angle_est, "degrees:", np.rad2deg(angle_est))
                    # angle_gt = angle_from_homo(homo_matrix)
                    # angle_diff = np.abs(difference_angle(angle_est, angle_gt))
                    # print("angle_est", angle_est, "angle_gt", angle_gt)
                    # print("difference", np.round(angle_diff, 4))

                else:
                    print("[red]est_homo_ransac is None")
            
            # print("mkpts0.shape", mkpts0.shape)
            # print("mkpts1.shape", mkpts1.shape)

            R, s, t = estimate_similarity_transformation(mkpts0.T, mkpts1.T)
            # print("R", R)
            angle_from_similarity = np.arctan2(R[1, 0], R[0,0])
            angle_from_similarity = add_angles(0, angle_from_similarity) # make sure angle_est is in [-pi, pi)
            angle_from_similarity = add_angles(0, -angle_from_similarity) # take the "inverse". Use add_angles to make sure we are still between -pi, pi.
            # print("angle estimate_similarity_transformation", angle_from_similarity, "in degrees:", np.rad2deg(angle_from_similarity))
                
            if False: #! method disabled because not in use
                est_affine_partial_2d = cv2.estimateAffinePartial2D(mkpts0, mkpts1)[0] # don't know why it is a list of two arrays
                print("est_affine_partial_2d", est_affine_partial_2d)

                angle_from_est_affine_partial = np.arctan2(est_affine_partial_2d[1, 0], est_affine_partial_2d[0,0])
                angle_from_est_affine_partial = add_angles(0, angle_from_est_affine_partial) # make sure angle_est is in [-pi, pi)
                angle_from_est_affine_partial = add_angles(0, -angle_from_est_affine_partial) # take the "inverse". Use add_angles to make sure we are still between -pi, pi.
                print("angle_from_est_affine_partial:", angle_from_est_affine_partial, "in degrees:", np.rad2deg(angle_from_est_affine_partial))

                # if affine_fit:
                    # mean_error, affine_median_error, max_error, A = ObjectReId.calculate_affine_matching_error(mkpts0, mkpts1)
                    # print("A", A)
                    # if debug:
                    #     print("confidence[valid]", confidence[valid])
                    #     print("matches[valid].shape", len(matches[valid]), matches[valid].shape)
                    #     print("mean_error", mean_error)
                    #     print("median_error", affine_median_error)
                    #     print("max_error", max_error)
                    #     print("affine_loss (lower is better)", affine_loss)
                    #     print("score_ratio (higher is better)", score_ratio)
                    #     print("gt", gt)

        vis_out = None
        if visualise:
            color = cm.jet(confidence[valid])
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                # 'Image Pair: {:06}:{:06}'.format(stem0, stem1),
            ]
            vis_out = make_matching_plot_fast(
                img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=self.opt.show_keypoints, small_text=small_text)
        
        return affine_loss, score_ratio, mconf, affine_median_error, len(matches[valid]), vis_out, angle_est, est_homo_ransac, angle_from_similarity, angle_from_est_affine_partial


def estimate_similarity_transformation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    k, n = source.shape

    # mx = source.mean(axis=1)
    # my = target.mean(axis=1)
    # print("mx", mx.shape)
    # print("my", my.shape)

    mx = np.array([200, 200]) # for image of 400 x 400
    my = np.array([200, 200]) # for image of 400 x 400
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered**2, axis=0))
    sy = np.mean(np.sum(target_centered**2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T
    rank = np.linalg.matrix_rank(Sxy)
    if rank < k:
        raise ValueError("Failed to estimate similarity transformation")

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    s = np.trace(np.diag(D) @ S) / sx
    t = my - s * (R @ mx)

    return R, s, t


if __name__ == '__main__':
    # object_reid_superglue = ObjectReIdSuperGlue(model="indoor")
    object_reid_superglue = ObjectReIdSuperGlue(model="/home/sruiz/projects/reconcycle/superglue_training/output/train/2023-11-18_superglue_model/weights/best.pt")

    vs = VideoStreamer(
        "/home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed/hca_0/", 
        [640, 480], 
        1,
        ['*.png', '*.jpg', '*.jpeg'], 
        1000000)

    img1 = vs.load_image("/home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed/hca_0/0001.jpg")
    img2 = vs.load_image("/home/sruiz/datasets2/reconcycle/2023-02-20_hca_backs_processed/hca_1/0002.jpg")

    print("img1.shape", img1.shape)
    print("img2.shape", img2.shape)

    object_reid_superglue.compare(img1, img2)

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey() # visualise

    


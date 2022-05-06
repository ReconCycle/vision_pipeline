import os
from venv import create
import numpy as np
import time
import cv2
from rich import print
import json
from scipy import stats
import open3d as o3d

from camera_realsense_feed import RealsenseCamera
from gap_detection.gap_detector import GapDetector
from object_detection import ObjectDetection
from helpers import Detection, Action

from helpers import scale_img, get_images, get_images_realsense, EnhancedJSONEncoder
from config import load_config


def imfill(mask):

    im_floodfill = mask.copy()
    cv2.floodFill(im_floodfill, None, (0,0), 255.0)

    im_floodfill = im_floodfill.astype(np.uint8)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return im_floodfill_inv


def mask_from_contours(contour):
    mask = np.zeros((480, 640, 3), np.uint8)
    mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

def create_depth_list(mask):
    depth_list = np.zeros((mask.shape[0]*mask.shape[1], 3))
    count = 0
    for i in np.arange(mask.shape[0]):
        for j in np.arange(mask.shape[1]):
            depth_list[count] = np.array([i, j, mask[i, j]])
            count += 1

    depth_axis_pts = depth_list[:, 2]
    # filtering camara defects, where depth = 0
    depth_list = depth_list[depth_axis_pts != 0]
    
    return depth_list


# Create Box Objects around the detected gaps.
def boxes(gaps):
    boxes = []
    hulls = []
    for gap in gaps:
        center, vertices, img_vertices, simplices, volume, size, num_of_points = gap
        pcd = o3d.geometry.PointCloud()
        bounding_box = o3d.geometry.OrientedBoundingBox()
        points = o3d.utility.Vector3dVector(vertices)
        pcd.points = points
        pcd.paint_uniform_color([0,0,0])
        geom1 = bounding_box.create_from_points(points)
        geom1.color = np.array([1,0,0])
        boxes.append(geom1)
        boxes.append(pcd)

    return boxes


class RealsensePipeline:
    def __init__(self):
        # 1. get image and depth from camera
        # 2. apply yolact to image and get hca_back
        # 3. apply mask to depth image and convert to pointcloud
        # 4. apply gap detection methods to pointcloud
        
        self.config = load_config()
        print("config", self.config)

        self.object_detection = ObjectDetection(self.config.obj_detection)
        self.labels = self.object_detection.labels
    
    def run(self, colour_img, depth_img, depth_colormap=None):
        print("running pipeline realsense frame...")

        # 2. apply yolact to image and get hca_back
        labelled_img, detections = self.object_detection.get_prediction(colour_img, extra_text=None)


        # 3. apply mask to depth image and convert to pointcloud
        mask = None
        # get the first detection that is hca_back
        detection_hca_back = None
        for detection in detections:
            if detection.label == self.labels.hca_back:
                detection_hca_back = detection
                break
        
        if detection_hca_back is not None:
            contour = detection_hca_back.mask_contour
            hull = cv2.convexHull(contour, False)
            mask = mask_from_contours(hull)

            print("mask", mask.shape)

            print("depth_img", depth_img.shape, np.amin(depth_img), np.max(depth_img), stats.mode(depth_img, axis=None).mode)
            depth_img = depth_img * 3 # ! IMPORTANT MULTIPLIER
            depth_masked = cv2.bitwise_and(depth_img, depth_img, mask = mask)

            depth_list = create_depth_list(depth_masked)
            depth_list = ((depth_list - np.amin(depth_list))/np.ptp(depth_list)) # ! required for gaps

            ############### 
            # # working visualisation without gap detection:
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector((depth_list))
            # o3d.visualization.draw_geometries([pcd])
            ###############


            # call the detector for gaps with depth array
            gap_detector = GapDetector()
            # thresholding is turned off, but this is how you would use it. 
            # (and set parameter in gap_detector.py)
            thresholded_depth_list = gap_detector.threshold(depth_list)
            gap_detector.detector_callback(thresholded_depth_list)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector((thresholded_depth_list))

            # create the boxes from detected gaps
            b = boxes(gap_detector.gaps)
            b.append(pcd)
            # visualise
            o3d.visualization.draw_geometries(b)


        # show images
        cv_show = [labelled_img]

        if depth_colormap is not None:
            cv_show.append(depth_colormap)        
        if mask is not None:
            cv_show.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        images = np.hstack(tuple(cv_show))

        cv2.namedWindow('images', cv2.WINDOW_NORMAL)
        cv2.imshow('images', images)

        cv2.waitKey(1)

if __name__ == '__main__':

    pipeline = RealsensePipeline()

    USE_CAMERA = False

    if USE_CAMERA:
        realsense_camera = RealsenseCamera()
        while True:
            try:
                # 1. get image and depth from camera
                colour_img, depth_img, depth_colormap = realsense_camera.get_frame()
                pipeline.run(colour_img, depth_img, depth_colormap)
            except KeyboardInterrupt:
                break
    
    else:
        # load images from folder
        img_path = "/Users/simonblaue/ownCloud/Bachelorarbeit/2022-05-05_kalo_qundis_realsense"
        # img_path = "/Users/sebastian/WorkProjects/datasets/reconcycle/2022-05-05_kalo_qundis_realsense"
        # save_path = "./save_images" # set to None to not save
        # if save_path is not None and not os.path.exists(save_path):
        #     os.makedirs(save_path)
        for colour_img, depth_img, depth_colormap in get_images_realsense(img_path): 
            # 1. get image and depth from camera
            # colour_img, depth_img, depth_colormap = realsense_camera.get_frame()
            pipeline.run(colour_img, depth_img, depth_colormap)

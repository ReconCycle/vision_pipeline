import os
from venv import create
import numpy as np
import time
import cv2
from rich import print
import json
from scipy import stats
open3d_available = True
try:
    import open3d as o3d
except ModuleNotFoundError:
    open3d_available = False
    pass

from camera_realsense_feed import RealsenseCamera
from gap_detection.better_gap_detector import BetterGapDetector
from object_detection import ObjectDetection
from helpers import Detection, Action

from helpers import scale_img, get_images, get_images_realsense, EnhancedJSONEncoder, img_grid
from config import load_config


def imfill(mask):

    im_floodfill = mask.copy()
    cv2.floodFill(im_floodfill, None, (0,0), 255.0)

    im_floodfill = im_floodfill.astype(np.uint8)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return im_floodfill_inv


def mask_from_contour(contour):
    mask = np.zeros((480, 640, 3), np.uint8)
    mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def image_to_depth(mask):
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


# def visualise_3d(geometries):

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible = False)
#     # vis.add_geometry(pcd)
#     # vis.update_geometry(pcd)
#     for geom in b:
#         vis.add_geometry(geom)
#         vis.update_geometry(geom)
#     vis.poll_events()
#     vis.update_renderer()
#     # vis.capture_screen_image(path)
#     o3d_screenshot_mat = vis.capture_screen_float_buffer()
#     # scale and convert to uint8 type
#     o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
#     vis.destroy_window()
#     return o3d_screenshot_mat

# Create Box Objects around the detected gaps.
# def boxes(gaps):
#     boxes = []
#     hulls = []

#     biggest_gap_volume = 0.
#     biggest_gap_index = None

#     # sort by volume
#     gaps.sort(key=lambda x: x[4], reverse=True)

#     # for index, gap in enumerate(gaps):
#     #     center, vertices, img_vertices, simplices, volume, size, num_of_points = gap
#     #     if volume > biggest_gap_volume:
#     #         biggest_gap_volume = volume
#     #         biggest_gap_index = index


#     for index, gap in enumerate(gaps):
#         center, vertices, img_vertices, simplices, volume, size, num_of_points = gap
#         pcd = o3d.geometry.PointCloud()
#         bounding_box = o3d.geometry.OrientedBoundingBox()
#         points = o3d.utility.Vector3dVector(vertices)
#         pcd.points = points
#         pcd.paint_uniform_color([0,0,0])
#         geom1 = bounding_box.create_from_points(points)
#         print("volume", volume)
#         print("geom1" , geom1)
#         print("center", center)
#         if index == 0:
#             geom1.color = np.array([0,0,1])
#         else:
#             geom1.color = np.array([1,0,0])
#         boxes.append(geom1)
#         boxes.append(pcd)

#     return boxes


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
        self.gap_detector = BetterGapDetector()
    
    def process_img(self, colour_img, depth_img, depth_colormap=None, debug=False):
        print("running pipeline realsense frame...")

        # 2. apply yolact to image and get hca_back
        labelled_img, detections = self.object_detection.get_prediction(colour_img, extra_text=None)

        # 3. apply mask to depth image and convert to pointcloud
        lever_actions = None
        mask = None
        cluster_img = None
        pcd = None
        # get the first detection that is hca_back
        detection_hca_back = None
        for detection in detections:
            if detection.label == self.labels.hca_back:
                detection_hca_back = detection
                break
        
        if detection_hca_back is not None and \
                len(detection_hca_back.mask_contour) > self.gap_detector.APPROX_SAMPLE_LENGTH:
            contour = detection_hca_back.mask_contour
            hull = cv2.convexHull(contour, False)
            mask = mask_from_contour(hull)
            hull_center = BetterGapDetector.cnt_center(hull)

            # print("depth_img", depth_img.shape, np.amin(depth_img), np.max(depth_img), stats.mode(depth_img, axis=None).mode)
            depth_img = depth_img * 3  # ! IMPORTANT MULTIPLIER
            depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=mask)

            depth_list = image_to_depth(depth_masked)
            # depth_list = ((depth_list - np.amin(depth_list))/np.ptp(depth_list))  # ! required for gaps

            ############### 
            # # working visualisation without gap detection:
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector((depth_list))
            # o3d.visualization.draw_geometries([pcd])
            ###############

            ############### bbox
            # thresholded_depth_list = gap_detector.threshold(depth_list)
            # gap_detector.detector_callback(depth_list)
            #
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector((depth_list))
            # # create the boxes from detected gaps
            # b = boxes(gap_detector.gaps)
            # b.append(pcd)
            # # visualise
            # o3d.visualization.draw_geometries(b)
            #################
            if len(depth_list) > 0:

                pcd, lever_actions, cluster_img = self.gap_detector.lever_detector(depth_list, depth_masked, hull_center)

                if debug:
                    self.show_img(labelled_img, mask, cluster_img, depth_colormap, pcd)

        json_lever_actions = json.dumps(lever_actions, cls=EnhancedJSONEncoder)

        # if there is no cluster image, return a black image
        if cluster_img is None:
            cluster_img = np.zeros(colour_img.shape, np.uint8)
            cv2.putText(cluster_img, "Clustering failed.", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

        return cluster_img, labelled_img, mask, lever_actions, json_lever_actions

    @staticmethod
    def show_img(labelled_img, mask, cluster_img, depth_colormap, pcd):
        # show images
        cv_show = [labelled_img]

        if depth_colormap is not None:
            cv_show.append(depth_colormap)
        if mask is not None:
            cv_show.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        if cluster_img is not None:
            cv_show.append(cluster_img)

        # images = np.hstack(tuple(cv_show))

        images = img_grid(cv_show)

        cv2.namedWindow('images', cv2.WINDOW_NORMAL)
        cv2.imshow('images', images)

        # if pcd is not None:
        #     o3d.visualization.draw_geometries([pcd])

        cv2.waitKey(1)  # was 1


if __name__ == '__main__':

    pipeline = RealsensePipeline()

    USE_CAMERA = True

    if USE_CAMERA:
        realsense_camera = RealsenseCamera()
        while True:
            try:
                # 1. get image and depth from camera
                colour_img, depth_img, depth_colormap = realsense_camera.get_frame()
                pipeline.process_img(colour_img, depth_img, depth_colormap, debug=True)
            except KeyboardInterrupt:
                break
    
    else:
        # load images from folder
        # img_path = "/Users/simonblaue/ownCloud/Bachelorarbeit/2022-05-05_kalo_qundis_realsense"
        img_path = "/Users/sebastian/WorkProjects/datasets/reconcycle/2022-05-05_kalo_qundis_realsense"
        # save_path = "./save_images" # set to None to not save
        # if save_path is not None and not os.path.exists(save_path):
        #     os.makedirs(save_path)
        for colour_img, depth_img, depth_colormap, colour_img_p in get_images_realsense(img_path):
            if "005" in colour_img_p:  # ! DEBUG
                pipeline.process_img(colour_img, depth_img, depth_colormap, debug=True)

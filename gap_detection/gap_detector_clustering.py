# Numpy and scikit-learn
import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum, threshold_li
from skimage.filters import threshold_yen
import skimage.exposure

import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, Pose, Transform, Vector3, Quaternion

from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import curve_fit
from shapely.geometry import LineString, Point, Polygon
from shapely.validation import make_valid
from shapely.validation import explain_validity

import open3d as o3d

import sklearn.cluster as cluster
import hdbscan
import time
import cv2
from itertools import combinations, product
import math
import random
# Own Modules
from obb import get_obb_from_contour
from helpers import get_colour, get_colour_blue, make_valid_poly, img_to_camera_coords, camera_info_to_o3d_intrinsics
from context_action_framework.types import Gap, Label, Detection


class GapDetectorClustering:
    def __init__(self, config):
        self.config = config

        # threshold max depth distance from camera, in mm
        self.MAX_DEPTH_THRESHOLD = 240 # mm 

        self.MIN_DIST_LEVER_OBJ_CENTER_TO_DEVICE_EDGE = 20

        # min. leverable area. The min. size of the cluster where the lever starts.
        self.MIN_LEVERABLE_AREA = 100
        # min. number of points in cluster (for sanity checks)
        self.MIN_CLUSTER_SIZE = 30

        self.clustering_mode = 3  # 0 to 3, 3 is hdbscan

        # kmeans
        self.KM_number_of_clusters = 3  # 1 to 10

        # birch
        self.B_branching_factor = 50  # 2 to 200
        self.B_threshold = 0.015  # 0.0 to 1.0
        
        # dbscan
        self.DB_eps = 0.01  # 0.0001 to 0.02

        # hdbscan
        self.HDB_min_cluster_size = 150


    def clustering(self, points):
        # points shape (n, 3)
        
        # ----- CLUSTERING THE GAPS -----
        clustering_switch = {
            0: self.kmeans,
            1: self.birch,
            2: self.dbscan,
            3: self.hdbscan
        }
        cluster_algorithm = clustering_switch[self.clustering_mode]
        labels = cluster_algorithm(points)


        labels = np.array(labels) # shape (n,)
        labels_T = np.array([labels]).T # shape (n, 1)
        clustered_points = np.append(points, labels_T, axis=1) # shape (n, 4)

        clusters = []
        for i in set(labels):
            cluster = clustered_points[clustered_points[:, 3] == float(i)]
            cluster = cluster[:, [0, 1, 2]] # shape (n_i, 3)

            # To construct a convex hull a minimum of 4 points is needed
            num_of_points, dim = cluster.shape
            if num_of_points >= 4:
                clusters.append(cluster)

        clusters.sort(key=lambda x: len(x), reverse=True)

        return clusters, num_of_points


    @staticmethod
    def cnt_center(cnt):
        m = cv2.moments(cnt)
        if np.abs(m["m00"]) > 0.01:
            x = int(m["m10"] / m["m00"])
            y = int(m["m01"] / m["m00"])
            return [x, y]
        else:
            return None


    @staticmethod
    def mean_depth(depth_masked, points):
        if len(points) > 0:
            heights = []
            for point in points:
                height = depth_masked[point[0], point[1]]
                if height != 0.0:
                    heights.append(height)

            return np.mean(heights)
        else:
            return None


    @staticmethod
    def image_to_points_list(mask):
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


    @staticmethod
    def mask_from_contour(contour, img_shape):
        mask = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
        mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


    def lever_detector(self, colour_img, depth_img, detections, graph_relations, camera_info, aruco_pose=None, aruco_point=None):

        # print("aruco_pose", aruco_pose)
        # print("aruco_point", aruco_point)
        # print("camera_info", camera_info)
        
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_thickness = 1

        # vars to return
        gaps = None
        img = None
        depth_scaled = None
        device_mask = None
        
        # https://github.com/Vuuuuk/Intel-Realsense-L515-3D-Scanner/blob/master/L515_3D_Scanner.py
        
        # merge colour_img and depth_img
        colour2 = o3d.geometry.Image(cv2.cvtColor(colour_img, cv2.COLOR_RGB2BGR))
        depth2 = o3d.geometry.Image((depth_img*1000).astype(np.uint8)) #! we undo a preprocessing step by *1000, m -> mm to correspond with camera_info?
        
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(colour2, depth2, convert_rgb_to_intensity=False)
        # rgbd_img = np.dstack((colour_img, depth_img))
        # print("rgbd_img.shape", rgbd_img.shape)
        
        # get intrinsics
        intrinsics = camera_info_to_o3d_intrinsics(camera_info)
        
        if self.config.realsense.debug_clustering:
        
            pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsics)

            print(pointcloud)
            print(np.asarray(pointcloud.points))

            o3d.visualization.draw_geometries([pointcloud])

        # get the first detection that is hca_back        
        detections_hca_back = graph_relations.exists(Label.hca_back)
        
        detection_hca_back = None
        if len(detections_hca_back) > 0:
            detection_hca_back = detections_hca_back[0]
        
        # todo: find PCB/PCB covered/internals
        # todo: get the one that is largest
        # todo: get left/right/top/bottom areas between PCB and hca_back
        # todo: find which area contains the biggest gap 
        # todo: lever that one
        
        if detection_hca_back is not None:

            contour = detection_hca_back.mask_contour
            hull = cv2.convexHull(contour, False)
            # get mask of segmentation from contour so that we get only the largest component
            device_mask = self.mask_from_contour(hull, depth_img.shape).astype(np.uint8)
            
            # mask depth image
            depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=device_mask)

            #! we should probably mask with the polygon as well.
            device_poly = detection_hca_back.polygon_px
            
            if device_poly is None:
                return gaps, img, depth_scaled, device_mask
            
            # print("device_poly.area", device_poly.area)
            # print("len(contour)", len(contour))
            # print("len(hull)", len(hull))

            points = self.image_to_points_list(depth_masked)

            if len(points) == 0:
                return gaps, img, depth_scaled, device_mask

        else:
            print("detection_hca_back is None!")
            return gaps, img, depth_scaled, device_mask

        height, width = depth_masked.shape[:2]
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # threshold depth image
        print("num points thresholded", np.count_nonzero(depth_masked > self.MAX_DEPTH_THRESHOLD))
        depth_masked[depth_masked > self.MAX_DEPTH_THRESHOLD] = 0

        depth_masked_np = np.ma.masked_equal(depth_masked, 0.0, copy=False)

        depth_max = np.amax(depth_masked)
        depth_min = np.amin(depth_masked)
        depth_min_nonzero = depth_masked_np.min() # np.min(points)

        # depth_mean = depth_masked_np.mean()
        # print("depth_mean", depth_mean)

        if depth_min == depth_max:
            print("depth_min == depth_max!")
            return gaps, img, depth_masked, device_mask

        if depth_min_nonzero is np.NaN or depth_min_nonzero is None:
            print("depth_min_nonzero is None!")
            return gaps, img, depth_masked, device_mask 

        # print("depth_min_nonzero", depth_min_nonzero)
        # print("depth_max", depth_max)

        # rescale the depth to the range (0, 255) such that the clustering works well
        depth_scaled = skimage.exposure.rescale_intensity(depth_masked, in_range=(depth_min_nonzero, depth_max), out_range=(0,255)).astype(np.uint8) # shape (480, 640)
        depth_scaled_points = self.image_to_points_list(depth_scaled) # shape (n, 3)

        contours = []
        contours_small = []
        clusters = []
        cluster_objs = []
        gaps = []
        gaps_bad = []
        gaps_bad2 = []

        # sanity check
        if len(depth_scaled_points) > self.MIN_CLUSTER_SIZE:
            clusters, num_of_points = self.clustering(depth_scaled_points)

        # get the contour of each cluster
        kernel = np.ones((2, 2), np.uint8)

        for index, cluster in enumerate(clusters):
            if len(cluster) > self.MIN_CLUSTER_SIZE:
                # todo: do we need to do this inverting?
                # todo: what are the funny greenish non-contoured parts of the image?
                # convert cluster to mask and the create contour
                cluster_img = np.zeros((height, width), dtype=np.uint8)
                cluster_colour = np.asarray(get_colour(index), dtype=np.uint8)
                for x, y, z in cluster:
                    img[int(x), int(y)] = cluster_colour
                    cluster_img[int(x), int(y)] = 255

                # apply erosion so that the contour is inside the object
                # cluster_img = cv2.erode(cluster_img, kernel, iterations=1) # ? maybe we can avoid doing this to get a better contour

                cluster_contours, _ = cv2.findContours(cluster_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cluster_contours = list(cluster_contours)
                if len(cluster_contours) > 0:
                    cluster_contours.sort(key=lambda x: len(x), reverse=True)
                    contour = cluster_contours[0]
                    contours.append(contour)
                    contours_small.extend(cluster_contours[1:])

                    cnt_squeeze = contour.squeeze()
                    if len(cnt_squeeze) > 3:
                        try:
                            poly = Polygon(cnt_squeeze)
                            poly = make_valid_poly(poly)
                            
                            # todo: make valid like for the whole device
                        except AssertionError as E:
                            print("[red]error converting contour to polygon![/red]")
                        else:
                            area = cv2.contourArea(contour)
                            center = self.cnt_center(contour)
                            if center is not None:
                                
                                depth = self.mean_depth(depth_masked, cluster[:, :2].astype(int))
                                
                                if depth is not None:
                                    cluster_obj = contour, poly, area, center, depth, index
                                    cluster_objs.append(cluster_obj)



        print("num clusters found:", len(cluster_objs))

        # good_points = []
        lines = []
        points_min_max = []
        counter = 0
        for cluster_obj1, cluster_obj2 in combinations(cluster_objs, 2):
            cnt1, poly1, area1, center1, depth1, cluster_id1 = cluster_obj1
            cnt2, poly2, area2, center2, depth2, cluster_id2 = cluster_obj2

            # the cluster area should be above a minimum otherwise they are too small to insert lever
            if area1 > self.MIN_LEVERABLE_AREA or area2 > self.MIN_LEVERABLE_AREA:

                    # depth is the distance from camera to point
                    # this will tell us which side is the lower side of the gap

                if depth2 > depth1 and area2 > self.MIN_LEVERABLE_AREA:
                    # mean2 is further from camera than mean1. mean2 is the gap
                    # points_low, points_idx_low, cnt_low = points2, points2_idx, cnt2
                    cnt_low = cnt2
                    poly_low = poly2
                    center_low = center2
                    depth_low = depth2
                    cluster_id_low = cluster_id2

                    # points_high, points_idx_high, cnt_high = points1, points1_idx, cnt1
                    # poly_high = poly1
                    cnt_high = cnt1
                    center_high = center1
                    depth_high = depth1
                    cluster_id_high = cluster_id1
                elif depth1 > depth2 and area1 > self.MIN_LEVERABLE_AREA:
                    # mean1 is further from camera than mean2. mean1 is the gap
                    # points_low, points_idx_low, cnt_low = points1, points1_idx, cnt1
                    cnt_low = cnt1
                    poly_low = poly1
                    center_low = center1
                    depth_low = depth1
                    cluster_id_low = cluster_id1

                    # points_high, points_idx_high, cnt_high = points2, points2_idx, cnt2
                    # poly_high = poly2
                    cnt_high = cnt2
                    center_high = center2
                    depth_high = depth2
                    cluster_id_high = cluster_id2
                else:
                    break

                # / 1000, to convert from mm to m, required for img_to_camera_coords(...)

                obb_px, obb_center_px, rot_quat = get_obb_from_contour(cnt_low)

                gap = Gap()
                gap.id = counter

                gap.from_depth = depth_low # / 1000
                gap.to_depth = depth_high # / 1000
                
                # todo: add these properties:
                
                # gap.from_det = 
                # gap.to_det = 

                # todo: convert to meters
                gap.obb = obb_px
                # gap.obb_3d = 

                gap.from_px = np.asarray([center_low[0], center_low[1]])
                gap.to_px = np.asarray([center_high[0], center_high[1]])
                
                from_3d = img_to_camera_coords(gap.from_px, 
                                                        gap.from_depth, 
                                                        camera_info)
                to_3d = img_to_camera_coords(gap.to_px, 
                                                    gap.to_depth, 
                                                    camera_info)
                
                gap.from_tf = Transform(Vector3(*from_3d), Quaternion(0.0, 0.0, 0.1, 0.0))
                gap.to_tf = Transform(Vector3(*to_3d), Quaternion(0.0, 0.0, 0.1, 0.0))
                
                # gap.from_depth = depth_masked[center_low[1], center_low[0]] / 1000
                # gap.to_depth = depth_masked[center_high[1], center_high[0]] / 1000
                
                gap.from_cluster = cluster_id_low
                gap.to_cluster = cluster_id_high
                
                counter += 1
                
                # todo: check if Pose is 0, 0, 0

                p = Pose()
                
                p.position.x = from_3d[0]
                p.position.y = from_3d[1]
                p.position.z = from_3d[2]
                # Make sure the quaternion is valid and normalized
                p.orientation.x = 0.0
                p.orientation.y = 0.0
                p.orientation.z = 0.0
                p.orientation.w = 1.0

                p_stamped = PoseStamped()
                p_stamped.pose = p
                p_stamped.header.stamp = rospy.Time.now()
                p_stamped.header.frame_id = "realsense_link"

                # quaternion should point towards to_camera

                gap.pose_stamped = p_stamped

                # todo: return obb and bb_camera

                # exclude lever actions where the clusters aren't next to each other.
                # do it by checking if the line intersects other clusters
                line_intersects_another_cluster = False
                line_px = LineString([gap.from_px, gap.to_px])
                for cnt, poly, area, center, depth, cluster_id in cluster_objs:
                    if cluster_id != gap.from_cluster and cluster_id != gap.to_cluster:
                        if line_px.intersects(poly):
                            # line intersects another cluster. conclusion: clusters aren't next to each other
                            line_intersects_another_cluster = True
                            break
                
                # lever action is from: center_low -> center_high
                # exclude actions where: center_high is too close to the device edge.
                center_high_pt = Point(center_high[0],center_high[1])
                
                if line_intersects_another_cluster:
                    gaps_bad2.append(gap)
                elif device_poly.exterior.distance(center_high_pt) < self.MIN_DIST_LEVER_OBJ_CENTER_TO_DEVICE_EDGE:
                    # center_high too close to device edge
                    gaps_bad.append(gap)
                else:
                    gaps.append(gap)

        # ? sort the lever actions based on which one is closest to the center of the device
        # gaps.sort(key=lambda gap: np.linalg.norm(gap[0][:2] - obj_center), reverse=False)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        cv2.drawContours(img, contours_small, -1, (0, 0, 255), 1)

        for p_min, p_max in points_min_max:
            cv2.circle(img, p_min, 6, (50, 141, 168), -1)
            cv2.circle(img, p_max, 6, (190, 150, 37), -1)

        for idx, [p1, p2] in enumerate(lines):
            # colour = tuple([int(x) for x in get_colour(idx)])
            colour = [162, 162, 162]
            cv2.line(img, p1, p2, colour, 3)
        
        for idx, gap in enumerate(gaps_bad):
            colour = tuple([int(x) for x in [0, 250, 250]])
            cv2.arrowedLine(img, gap.from_px.astype(int),
                            gap.to_px.astype(int), colour, 3, tipLength=0.3)
            
        for idx, gap in enumerate(gaps_bad2):
            colour = tuple([int(x) for x in [0, 0, 250]])
            cv2.arrowedLine(img, gap.from_px.astype(int),
                            gap.to_px.astype(int), colour, 3, tipLength=0.3)

        for idx, gap in enumerate(gaps):
            colour = tuple([int(x) for x in get_colour_blue(idx)])
            cv2.arrowedLine(img, gap.from_px.astype(int),
                            gap.to_px.astype(int), colour, 3, tipLength=0.3)

        # print avg height of each cluster
        for cluster_obj in cluster_objs:
            _, _, _, center, depth, _ = cluster_obj
            if not np.isnan(depth):
                text = str(round(depth, 2))
            else:
                text = "NaN"
            text_pt = center
            font_scale = 0.4
            color = [255, 255, 255]
            cv2.putText(img, text, text_pt, font_face, font_scale, color, font_thickness, cv2.LINE_AA)
        
        return gaps, img, depth_scaled, device_mask

    # =============== CLUSTER ALGORITHM WRAPPERS ===============
    def kmeans(self, data):
        return cluster.KMeans(n_clusters=self.KM_number_of_clusters).fit_predict(data)

    def birch(self, data):
        params = {'branching_factor': self.B_branching_factor,
                  'threshold': self.B_threshold,
                  'n_clusters': None,
                  'compute_labels': True}

        return cluster.Birch(**params).fit_predict(data)

    def dbscan(self, data):
        return cluster.DBSCAN(eps=self.DB_eps).fit_predict(data)

    def hdbscan(self, data):
        return hdbscan.HDBSCAN(min_cluster_size=self.HDB_min_cluster_size).fit_predict(data)

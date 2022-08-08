# Numpy and scikit-learn
import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum, threshold_li
from skimage.filters import threshold_yen
import skimage.exposure

from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import curve_fit
from shapely.geometry import LineString, Point, Polygon
import sklearn.cluster as cluster
import hdbscan
import time
open3d_available = True
try:
    import open3d as o3d
except ModuleNotFoundError:
    open3d_available = False
    pass
import cv2
from itertools import combinations, product
import math
import random
# Own Modules
from helpers import get_colour, get_colour_blue


class GapDetectorClustering:
    def __init__(self):

        # threshold max depth distance from camera, in mm
        self.MAX_DEPTH_THRESHOLD = 250 # mm 

        self.MIN_DIST_LEVER_OBJ_CENTER_TO_DEVICE_EDGE = 20

        # threshold the cluster sizes. 800 is better, 80 is for debugging
        self.MIN_LEVERABLE_AREA = 200 #? make it the same as min_cluster_size?
        # if the lever_line is too small then levering won't be possible
        self.MIN_LEVERABLE_LENGTH = 5
        # number of points in cluster
        self.MIN_CLUSTER_SIZE = 30
        # distance between clusters, such that there is a possibility to lever between them
        self.MAX_CLUSTER_DISTANCE = 130
        # sample size of points on cnt between clusters to find part of cnt that shares edge with another cluster
        self.APPROX_SAMPLE_LENGTH = 40

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
        # ----- CLUSTERING THE GAPS -----
        clustering_switch = {
            0: self.kmeans,
            1: self.birch,
            2: self.dbscan,
            3: self.hdbscan
        }
        cluster_algorithm = clustering_switch[self.clustering_mode]
        labels = cluster_algorithm(points)

        labels = np.array(labels)
        labels_T = np.array([labels]).T
        clustered_points = np.append(points, labels_T, axis=1)

        clusters = []
        for i in set(labels):
            cluster = clustered_points[clustered_points[:, 3] == float(i)]
            cluster = cluster[:, [0, 1, 2]]

            # To construct a convex hull a minimum of 4 points is needed
            num_of_points, dim = cluster.shape
            if num_of_points >= 4:
                clusters.append(cluster)

        clusters.sort(key=lambda x: len(x), reverse=True)

        # print("num clusters: ", len(clusters), num_of_points)

        return clusters, num_of_points

    def get_close_points_from_2_clusters(self, cnt1, cnt2, cnt1_sample_idx, cnt2_sample_idx):
        points1 = []
        points1_idx = []
        points2 = []
        points2_idx = []
        for point_idx1, point_idx2 in product(cnt1_sample_idx, cnt2_sample_idx):
            point1 = cnt1[point_idx1].squeeze()
            point2 = cnt2[point_idx2].squeeze()
            dist = np.linalg.norm(point1 - point2)
            if dist < self.MAX_CLUSTER_DISTANCE:
                points1.append(point1)
                points1_idx.append(point_idx1)
                points2.append(point2)
                points2_idx.append(point_idx2)
        return points1, points1_idx, points2, points2_idx

    @staticmethod
    def precise_idx_range(cnt, points_idx):
        step_size = 10
        epsilon = 5
        new_points_idx = []
        # around each point, create an epsilon-ball, and add those points to the list
        for point_idx in points_idx:
            min_idx = point_idx - epsilon * step_size
            max_idx = point_idx + epsilon * step_size
            if min_idx < 0:
                min_idx % len(cnt)
                new_points_idx.extend(np.arange(min_idx % len(cnt), stop=len(cnt), step=step_size))

            if max_idx >= len(cnt):
                max_idx % len(cnt)
                new_points_idx.extend(np.arange(0, stop=max_idx % len(cnt), step=step_size))

            new_points_idx.extend(np.arange(np.clip(min_idx, 0, len(cnt)),
                                            stop=np.clip(max_idx, 0, len(cnt)),
                                            step=step_size))

        # remove duplicate points from the list
        new_points_idx = list(set(new_points_idx))

        return new_points_idx

    def generate_pcd(self, clusters):
        pcd = None
        if open3d_available:
            clustered_points = []
            colours = []
            for index, cluster in enumerate(clusters):
                clustered_points.extend(cluster)
                if len(cluster) < self.MIN_CLUSTER_SIZE:
                    cluster_colours = [np.array((180, 180, 180), dtype=np.float64) / 255] * len(cluster)
                else:
                    cluster_colours = [get_colour(index) / 255] * len(cluster)

                colours.extend(cluster_colours)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(clustered_points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colours))
        
        return pcd

    @staticmethod
    def sort_and_remove_duplicates(points_idx, points):
        idxs_and_points = list(zip(points_idx, points))
        idxs_and_points.sort(key=lambda x: x[0], reverse=False)
        idxs_and_points = np.array(idxs_and_points, dtype=object)
        _, indices = np.unique(idxs_and_points[:, 0], return_index=True)
        idxs_and_points = idxs_and_points[indices, :]
        # points_idx_sorted, points_sorted = zip(*idxs_and_points)
        return zip(*idxs_and_points)

    @staticmethod
    def get_regression_line(points):
        # straight line y=f(x)
        def f(x, m, b):
            return m * x + b

        p_opt, p_cov = curve_fit(f, np.array(points)[:, 0], np.array(points)[:, 1])
        m = p_opt[0]  # slope
        b = p_opt[1]  # intercept

        # two points on the line are
        p1 = [0, b]
        if np.abs(m) > 0.1:
            p2 = [-b / m, 0]
        else:
            p2 = [1, m + b]

        return p1, p2

    @staticmethod
    def closest_point_on_line(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        dx, dy = x2 - x1, y2 - y1
        det = dx * dx + dy * dy
        a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
        return x1 + a * dx, y1 + a * dy

    @staticmethod
    def get_closest_point_from_list_to_point(points, p1):
        dists = [np.linalg.norm(point - p1) for point in points]
        return points[np.argmin(dists)]
    
    @staticmethod
    def dist_point_to_list(p1, points):
        dists = [np.linalg.norm(point - p1) for point in points]
        return np.amin(dists)

    @staticmethod
    def get_midpoint(p1, p2):
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        return [(p1_x + p2_x) / 2, (p1_y + p2_y) / 2]

    @staticmethod
    def get_perp(p1, p2, cd_length=50):
        ab = LineString([p1, p2])
        left = ab.parallel_offset(cd_length / 2, 'left')
        right = ab.parallel_offset(cd_length / 2, 'right')
        c = left.boundary[1]
        d = right.boundary[0]  # note the different orientation for right offset
        return c, d

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
    def get_pair_furthest_points(points):
        d = squareform(pdist(points, 'euclidean'))
        n, [I_row, I_col] = np.nanmax(d), np.unravel_index(np.argmax(d), d.shape)
        return points[I_row], points[I_col]

    @staticmethod
    def mean_depth(depth_masked, points):
        if len(points) > 0:
            heights = []
            for point in points:
                height = depth_masked[point[1], point[0]]
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
    def mask_from_contour(contour):
        mask = np.zeros((480, 640, 3), np.uint8)
        mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    def lever_detector(self, depth_img, detections, labels):

        # get the first detection that is hca_back
        detection_hca_back = None
        for detection in detections:
            if detection.label == labels.hca_back:
                detection_hca_back = detection
                break
        
        # if detection_hca_back is not None and len(detection_hca_back.mask_contour) > self.APPROX_SAMPLE_LENGTH
        if detection_hca_back is not None:

            contour = detection_hca_back.mask_contour
            hull = cv2.convexHull(contour, False)            
            # get mask of segmentation from contour so that we get only the largest component
            device_mask = self.mask_from_contour(hull).astype(np.uint8)
            
            # mask depth image
            depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=device_mask)

            device_poly = detection_hca_back.mask_polygon
            if device_poly is None:
                print("device_poly is None!")
                return None, None, None, None, None

            height, width = depth_masked.shape[:2]
            img = np.zeros((height, width, 3), dtype=np.uint8)

            points = self.image_to_points_list(depth_masked)

            if len(points) == 0:
                # if there is no cluster image, return a black image
                cv2.putText(img, "No depth data.", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

                return None, None, img, None, None

        else:
            print("detection_hca_back is None!")
            return None, None, None, None, None


        # threshold depth image
        print("num points thresholded", np.count_nonzero(depth_masked > self.MAX_DEPTH_THRESHOLD))
        depth_masked[depth_masked > self.MAX_DEPTH_THRESHOLD] = 0

        depth_masked_np = np.ma.masked_equal(depth_masked, 0.0, copy=False)

        depth_max = np.amax(depth_masked)
        depth_min = np.amin(depth_masked)
        depth_min_nonzero = depth_masked_np.min() # np.min(points)

        print("depth_min", depth_min)
        print("depth_min_nonzero", depth_min_nonzero)
        print("depth_max", depth_max)

        # rescale the depth to the range (0, 255) such that the clustering works well
        depth_scaled = skimage.exposure.rescale_intensity(depth_masked, in_range=(depth_min_nonzero, depth_max), out_range=(0,255)).astype(np.uint8)
        depth_scaled_points = self.image_to_points_list(depth_scaled)

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_thickness = 1

        lever_actions = []
        lever_actions_bad = []
        clusters = []

        # sanity check
        if len(depth_scaled_points) > self.MIN_CLUSTER_SIZE:
            clusters, num_of_points = self.clustering(depth_scaled_points)
        
        # for debugging
        pcd = self.generate_pcd(clusters)

        # get the contour of each cluster
        kernel = np.ones((2, 2), np.uint8)
        contours = []
        cluster_objs = []
        for index, cluster in enumerate(clusters):
            if len(cluster) > self.MIN_CLUSTER_SIZE:
                # create an inverse image, so that our contour is on the inside of the object
                cluster_img = np.zeros((height, width), dtype=np.uint8)
                for x, y, z in cluster:
                    cluster_colour = np.asarray(get_colour(index), dtype=np.uint8)
                    # print(x, y, z, cluster_colour)
                    img[int(x), int(y)] = cluster_colour
                    cluster_img[int(x), int(y)] = 255

                # apply erosion so that the contour is inside the object
                cluster_img = cv2.erode(cluster_img, kernel, iterations=1)

                # we sample evenly along the contour, and this is better when using CHAIN_APPROX_NONE instead of
                # CHAIN_APPROX_SIMPLE
                # todo: it would be nice to sample this contour evenly over distance (so not every 10th point, but every point 10px away from the last)
                #? we could probably use polygons instead of contours to do this
                cluster_contours, _ = cv2.findContours(cluster_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cluster_contours = list(cluster_contours)
                if len(cluster_contours) > 0:
                    cluster_contours.sort(key=lambda x: len(x), reverse=True)
                    contour = cluster_contours[0]
                    contours.append(contour)

                    poly = Polygon(contour.squeeze())

                    area = cv2.contourArea(contour)
                    center = self.cnt_center(contour)
                    if center is not None:

                        points = contour.squeeze()
                        depth = self.mean_depth(depth_masked, points)
                        if depth is not None:
                            cluster_obj = contour, poly, area, center, depth
                            cluster_objs.append(cluster_obj)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        print("num clusters found:", len(cluster_objs))

        # good_points = []
        lines = []
        points_min_max = []

        for cluster_obj1, cluster_obj2 in combinations(cluster_objs, 2):
            cnt1, poly1, area1, center1, depth1 = cluster_obj1
            cnt2, poly2, area2, center2, depth2 = cluster_obj2
            # contours should be at least the sample length, otherwise they are tiny contours and can be ignored
            if len(cnt1) > self.APPROX_SAMPLE_LENGTH and len(cnt2) > self.APPROX_SAMPLE_LENGTH:

                # the cluster area should be above a minimum otherwise they are too small to insert lever
                if area1 > self.MIN_LEVERABLE_AREA or area2 > self.MIN_LEVERABLE_AREA:

                    # approximately evenly sampled points along the contour
                    # todo: we could already do this earlier for each contour create a "rough" contour that is evenly sampled over distance
                    # cnt1_sample_idx = np.arange(len(cnt1), step=int(len(cnt1) / self.APPROX_SAMPLE_LENGTH))
                    # cnt2_sample_idx = np.arange(len(cnt2), step=int(len(cnt2) / self.APPROX_SAMPLE_LENGTH))

                    # points1, points1_idx, points2, points2_idx = self.get_close_points_from_2_clusters(cnt1, cnt2,
                    #                                                                                    cnt1_sample_idx,
                    #                                                                                    cnt2_sample_idx)

                    # # now get more precise points around this part of the contour
                    # if len(points1) > 0 and len(points2) > 0:
                    #     cnt1_sample_idx_precise = self.precise_idx_range(cnt1, points1_idx)
                    #     cnt2_sample_idx_precise = self.precise_idx_range(cnt2, points2_idx)

                    #     points1_p, points1_idx_p, points2_p, points2_idx_p = self.get_close_points_from_2_clusters(cnt1,
                    #                                             cnt2, cnt1_sample_idx_precise, cnt2_sample_idx_precise)

                    #     points1.extend(points1_p)
                    #     points1_idx.extend(points1_idx_p)
                    #     points2.extend(points2_p)
                    #     points2_idx.extend(points2_idx_p)

                    #     points1_idx, points1 = self.sort_and_remove_duplicates(points1_idx, points1)
                    #     points2_idx, points2 = self.sort_and_remove_duplicates(points2_idx, points2)

                        # ! now uniformly sample these points!

                        # depth is the distance from camera to point
                        # this will tell us which side is the lower side of the gap

                    if depth2 > depth1 and area2 > self.MIN_LEVERABLE_AREA:
                        # mean2 is further from camera than mean1. mean2 is the gap
                        # points_low, points_idx_low, cnt_low = points2, points2_idx, cnt2
                        center_low = center2

                        # points_high, points_idx_high, cnt_high = points1, points1_idx, cnt1
                        center_high = center1
                    elif depth1 > depth2 and area1 > self.MIN_LEVERABLE_AREA:
                        # mean1 is further from camera than mean2. mean1 is the gap
                        # points_low, points_idx_low, cnt_low = points1, points1_idx, cnt1
                        center_low = center1

                        # points_high, points_idx_high, cnt_high = points2, points2_idx, cnt2
                        center_high = center2
                    else:
                        break

                    lever_action = [np.array([*center_low,
                                                    depth_masked[center_low[1], center_low[0]]]),
                                                    np.array([*center_high,
                                                    depth_masked[center_high[1], center_high[0]]])
                                                    ]

                    # lever action is from: center_low -> center_high
                    # exclude actions where: center_high is too close to the device edge.
                    print("center_high", center_high)
                    print("type(device_poly)", type(device_poly))
                    print("device_poly.is_valid", device_poly.is_valid)
                    print("type(device_poly.boundary)", type(device_poly.boundary))

                    center_high_pt = Point(center_high[0],center_high[1])
                    if device_poly.boundary.distance(center_high_pt) < self.MIN_DIST_LEVER_OBJ_CENTER_TO_DEVICE_EDGE:
                        # center_high too close to device edge
                        lever_actions_bad.append(lever_action)
                    else:
                        lever_actions.append(lever_action)


                    # for showing all the points
                    # good_points.extend(points1)
                    # good_points.extend(points2)

        # ? sort the lever actions based on which one is closest to the center of the device
        # lever_actions.sort(key=lambda lever_action: np.linalg.norm(lever_action[0][:2] - obj_center), reverse=False)

        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        # for point in good_points:
        #     cv2.circle(img, tuple(point), 2, (0, 0, 255), -1)

        for p_min, p_max in points_min_max:
            cv2.circle(img, p_min, 6, (50, 141, 168), -1)
            cv2.circle(img, p_max, 6, (190, 150, 37), -1)

        for idx, [p1, p2] in enumerate(lines):
            # colour = tuple([int(x) for x in get_colour(idx)])
            colour = [162, 162, 162]
            cv2.line(img, p1, p2, colour, 3)
        
        for idx, lever_action in enumerate(lever_actions_bad):
            cluster_center, leverpoint = lever_action
            colour = tuple([int(x) for x in [0, 250, 250]])
            cv2.arrowedLine(img, [int(x) for x in cluster_center[:2]],
                            [int(x) for x in leverpoint[:2]], colour, 3, tipLength=0.3)

        for idx, lever_action in enumerate(lever_actions):
            cluster_center, leverpoint = lever_action
            colour = tuple([int(x) for x in get_colour_blue(idx)])
            cv2.arrowedLine(img, [int(x) for x in cluster_center[:2]],
                          [int(x) for x in leverpoint[:2]], colour, 3, tipLength=0.3)

        # print avg height of each cluster
        for cluster_obj in cluster_objs:
            _, _, _, center, depth = cluster_obj
            text = str(np.int(round(depth, 0)))
            text_pt = center
            font_scale = 0.4
            color = [255, 255, 255]
            cv2.putText(img, text, text_pt, font_face, font_scale, color, font_thickness, cv2.LINE_AA)

        return pcd, lever_actions, img, depth_scaled, device_mask

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



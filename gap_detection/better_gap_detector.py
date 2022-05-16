# Numpy and scikit-learn
import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum, threshold_li
from skimage.filters import threshold_yen
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import curve_fit
from shapely.geometry import LineString
import sklearn.cluster as cluster
import hdbscan
import time
import open3d as o3d
import cv2
from itertools import combinations, product
import math
import random

# import config

# Own Modules
import gap_detection.helpers2 as helpers
from helpers import get_colour, get_colour_blue


class BetterGapDetector:
    def __init__(self):

        # threshold the cluster sizes. 800 is better, 80 is for debugging
        self.MIN_LEVERABLE_AREA = 80
        # if the lever_line is too small then levering won't be possible
        self.MIN_LEVERABLE_LENGTH = 20
        # number of points in cluster
        self.MIN_CLUSTER_SIZE = 150
        # distance between clusters, such that there is a possibility to lever between them
        self.MAX_CLUSTER_DISTANCE = 20
        # sample size of points on cnt between clusters to find part of cnt that shares edge with another cluster
        self.APPROX_SAMPLE_LENGTH = 40

        self.depth_axis = 2
        self.clustering_mode = 3  # 0 to 3
        self.create_evaluation = False
        self.automatic_thresholding = -1  # -1 to 3
        self.min_gap_volume = 0.1  # 0.0 to 200.0
        self.max_gap_volume = 20000.0  # 0.0 to 200.0
        self.KM_number_of_clusters = 3  # 1 to 10
        self.B_branching_factor = 50  # 2 to 200
        self.B_threshold = 0.015  # 0.0 to 1.0
        self.DB_eps = 0.01  # 0.0001 to 0.02
        self.HDB_min_cluster_size = 50  # 5 to 150
        self.otsu_bins = 800  # 2 to 1024

    def clustering(self, points):
        print("clustering!")
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
            if (num_of_points >= 4):
                clusters.append(cluster)

        clusters.sort(key=lambda x: len(x), reverse=True)

        print("num clusters: ", len(clusters), num_of_points)

        return (clusters, num_of_points)

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

        print("points x y", np.array(points)[:, 0].shape, np.array(points)[:, 1].shape,
              np.array(points).shape)

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
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        return [x, y]

    @staticmethod
    def get_pair_furthest_points(points):
        d = squareform(pdist(points, 'euclidean'))
        n, [I_row, I_col] = np.nanmax(d), np.unravel_index(np.argmax(d), d.shape)

        print("I_row", I_row, points[I_row])
        print("I_col", I_col, points[I_col])
        return points[I_row], points[I_col]

    @staticmethod
    def mean_depth(depth_masked, points):
        heights = []
        for point in points:
            height = depth_masked[point[1], point[0]]
            if height != 0.0:
                heights.append(height)

        return np.mean(heights)

    def lever_detector(self, points, depth_masked, obj_center):

        lever_actions = []

        clusters, num_of_points = self.clustering(points)

        clustered_points = []
        colours = []
        for index, cluster in enumerate(clusters):
            print("cluster.shape", cluster.shape, np.mean(cluster))

            clustered_points.extend(cluster)
            if len(cluster) < self.MIN_CLUSTER_SIZE:
                cluster_colours = [np.array((180, 180, 180), dtype=np.float64) / 255] * len(cluster)
            else:
                cluster_colours = [get_colour(index) / 255] * len(cluster)

            colours.extend(cluster_colours)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(clustered_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colours))

        height, width = depth_masked.shape[:2]
        kernel = np.ones((2, 2), np.uint8)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        contours = []
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
                cluster_contours, _ = cv2.findContours(cluster_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cluster_contours = list(cluster_contours)
                cluster_contours.sort(key=lambda x: len(x), reverse=True)
                contour = cluster_contours[0]
                contours.append(contour)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        good_points = []
        lines = []
        points_min_max = []
        for cnt1, cnt2 in combinations(contours, 2):
            # contours should be at least the sample length, otherwise they are tiny contours and can be ignored
            if len(cnt1) > self.APPROX_SAMPLE_LENGTH and len(cnt2) > self.APPROX_SAMPLE_LENGTH:

                area1 = cv2.contourArea(cnt1)
                area2 = cv2.contourArea(cnt2)

                if area1 > self.MIN_LEVERABLE_AREA or area2 > self.MIN_LEVERABLE_AREA:

                    # approximately evenly sampled points along the contour
                    cnt1_sample_idx = np.arange(len(cnt1), step=int(len(cnt1) / self.APPROX_SAMPLE_LENGTH))
                    cnt2_sample_idx = np.arange(len(cnt2), step=int(len(cnt2) / self.APPROX_SAMPLE_LENGTH))

                    points1, points1_idx, points2, points2_idx = self.get_close_points_from_2_clusters(cnt1, cnt2,
                                                                                                       cnt1_sample_idx,
                                                                                                       cnt2_sample_idx)

                    # now get more precise points around this part of the contour
                    if len(points1) > 0 and len(points2) > 0:
                        cnt1_sample_idx_precise = self.precise_idx_range(cnt1, points1_idx)
                        cnt2_sample_idx_precise = self.precise_idx_range(cnt2, points2_idx)

                        points1_p, points1_idx_p, points2_p, points2_idx_p = self.get_close_points_from_2_clusters(cnt1,
                                                                cnt2, cnt1_sample_idx_precise, cnt2_sample_idx_precise)

                        points1.extend(points1_p)
                        points1_idx.extend(points1_idx_p)
                        points2.extend(points2_p)
                        points2_idx.extend(points2_idx_p)

                        points1_idx, points1 = self.sort_and_remove_duplicates(points1_idx, points1)
                        points2_idx, points2 = self.sort_and_remove_duplicates(points2_idx, points2)

                        # ! now uniformly sample these points!

                        # compute avg height of points1 and points2
                        # this will tell us which side is the lower side of the gap
                        mean1 = self.mean_depth(depth_masked, points1)
                        mean2 = self.mean_depth(depth_masked, points2)
                        diff = mean1 - mean2

                        if diff < 0 and area2 > self.MIN_LEVERABLE_AREA:
                            points, points_idx, cnt = points2, points2_idx, cnt2
                        elif diff > 0 and area1 > self.MIN_LEVERABLE_AREA:
                            points, points_idx, cnt = points1, points1_idx, cnt1
                        else:
                            break

                        min_point, max_point = self.get_pair_furthest_points(points)

                        # line of best fit
                        # p1, p2 = self.get_regression_line(points)
                        # ! the above isn't working really well.
                        # ! I think the points need to be distributed uniformly for it to work
                        # ! If we use the regression line then, use:
                        # segment_p1 = np.array(self.closest_point_on_line(p1, p2, min_point)).astype(int)
                        # segment_p2 = np.array(self.closest_point_on_line(p1, p2, max_point)).astype(int)
                        segment_p1 = min_point
                        segment_p2 = max_point

                        midpoint = self.get_midpoint(segment_p1, segment_p2)

                        # if np.abs(segment_p1) > 10000 or np.abs(segment_p2) > 10000 or np.abs(midpoint) > 10000:
                        print("segment_p1", segment_p1)
                        print("segment_p2", segment_p2)
                        print("midpoint", midpoint)
                            # break

                        # now get the closest point in points on the cnt to the midpoint
                        midpoint_on_cnt = self.get_closest_point_from_list_to_point(points, midpoint)
                        midpoint_on_cnt = np.array(midpoint_on_cnt).astype(int)
                        cluster_center = self.cnt_center(cnt)
                        lever_line = [midpoint_on_cnt, cluster_center]
                        # if the lever_line is too small then levering won't be possible
                        if np.linalg.norm(midpoint_on_cnt - cluster_center) >= self.MIN_LEVERABLE_LENGTH:

                            lever_actions.append([np.array([*midpoint_on_cnt,
                                                  depth_masked[midpoint_on_cnt[1], midpoint_on_cnt[0]]]),
                                                  np.array([*cluster_center,
                                                  depth_masked[cluster_center[1], cluster_center[0]]])
                                                  ])

                            # p3, p4 = self.get_perp(segment_p1, midpoint)
                            # p3, p4 = np.array(p3).astype(int), np.array(p4).astype(int)
                            # print("p3", p3, "p4", p4)
                            #
                            # # this function requires float32 otherwise it breaks
                            # dist_from_contour = cv2.pointPolygonTest(cnt, np.array(p3).astype(np.float32), True)
                            #
                            # if dist_from_contour >= 0:
                            #     perp_line = [np.array(midpoint).astype(int), p3]
                            # else:
                            #     perp_line = [np.array(midpoint).astype(int), p4]

                            lines.append([segment_p1, segment_p2])
                            # lines.append(lever_line)

                            points_min_max.append([tuple(np.array(min_point).astype(int)),
                                                   tuple(np.array(max_point).astype(int))])

                    # for showing all the points
                    good_points.extend(points1)
                    good_points.extend(points2)

        # sort the lever actions based on which one is closest to the center of the device
        lever_actions.sort(key=lambda lever_action: np.linalg.norm(lever_action[0][:2] - obj_center), reverse=False)

        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        for point in good_points:
            cv2.circle(img, tuple(point), 2, (0, 0, 255), -1)

        for p_min, p_max in points_min_max:
            cv2.circle(img, p_min, 6, (50, 141, 168), -1)
            cv2.circle(img, p_max, 6, (190, 150, 37), -1)

        for idx, [p1, p2] in enumerate(lines):
            # colour = tuple(np.asarray(get_colour(idx)).astype(int))
            # colour = tuple([int(x) for x in get_colour(idx)])
            colour = [162, 162, 162]
            print("colour", colour, p1, p2)
            cv2.line(img, p1, p2, colour, 3)

        for idx, lever_action in enumerate(lever_actions):
            leverpoint, midpoint = lever_action
            colour = tuple([int(x) for x in get_colour_blue(idx)])
            cv2.line(img, [int(x) for x in leverpoint[:2]],
                          [int(x) for x in midpoint[:2]], colour, 3)

        return pcd, lever_actions, img

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



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
from helpers import get_colour


class BetterGapDetector:
    def __init__(self):

        self.MIN_LEVERABLE_AREA = 80  # 800 is better, 80 is for debugging
        self.MIN_CLUSTER_SIZE = 150  # number of points in cluster
        self.MAX_CLUSTER_DISTANCE = 20  # in pixels

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
    def closest_point_on_line(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        dx, dy = x2 - x1, y2 - y1
        det = dx * dx + dy * dy
        a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
        return x1 + a * dx, y1 + a * dy

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
    def get_pair_furthest_points(points):
        D = squareform(pdist(points, 'euclidean'))
        N, [I_row, I_col] = np.nanmax(D), np.unravel_index(np.argmax(D), D.shape)

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

    def get_middle_point(self, points_idx, points, cnt):

        ##############
        # unreliable min, max point finder
        ##############
        # min_idx = np.amin(points_idx)
        # max_idx = np.amax(points_idx)
        #
        # # it could be that the contour starts somewhere in the middle of the segment
        # # eg. contour length 700. Segment exists at 670, 680, 0, 10, 20
        # if (max_idx + 50) % len(cnt) < min_idx:
        #     print("there exists a jump")
        #
        #     prev_point_idx = points_idx[0]
        #     for point_idx in points_idx:
        #         if point_idx > prev_point_idx + 50:
        #             max_idx = prev_point_idx
        #             print("iterating forwards, jump at", prev_point_idx, point_idx)
        #             break
        #
        #         prev_point_idx = point_idx
        #
        #     prev_point_idx = points_idx[-1]
        #     for point_idx in np.flip(points_idx):
        #         if point_idx < prev_point_idx - 50:
        #             print("iterating back, jump at", prev_point_idx, point_idx)
        #             min_idx = prev_point_idx
        #             break
        #
        #         prev_point_idx = point_idx
        #
        # print("min idx:", min_idx)
        # print("max idx:", max_idx)
        # print("len(cnt)", len(cnt))
        # middle_idx_idx = (points_idx.index(min_idx) + int(len(points_idx) / 2)) % len(points_idx)
        # middle_idx = points_idx[middle_idx_idx]
        ################

        min_point, max_point = self.get_pair_furthest_points(points)
        print("min_point", min_point)
        print("max_point", max_point)

        ##################
        # line of best fit
        ##################
        # def f(x, A, B):  # this is your 'straight line' y=f(x)
        #     return A * x + B
        #
        # print("points x y", np.array(points)[:, 0].shape, np.array(points)[:, 1].shape,
        #       np.array(points).shape)
        #
        # popt, pcov = curve_fit(f, np.array(points)[:, 0],
        #                        np.array(points)[:, 1])  # your data x, y to fit
        # m = popt[0]  # slope
        # b = popt[1]  # intercept
        #
        # # two points on the line are
        # p1 = [0, b]
        # if np.abs(m) > 0.1:
        #     p2 = [-b / m, 0]
        # else:
        #     p2 = [1, m + b]
        #########################

        # our line segment is then:
        # print("cnt[min_idx].squeeze()", cnt[min_idx].squeeze())
        # print("cnt[max_idx].squeeze()", cnt[max_idx].squeeze())
        # segment_p1 = np.array(closest_point_on_line(p1, p2, cnt[min_idx].squeeze())).astype(int)
        # segment_p2 = np.array(closest_point_on_line(p1, p2, cnt[max_idx].squeeze())).astype(int)
        # ! the above isn't working really well. I think the points need to be distributed uniformly for it to work

        # ! this is kind of dumb at the moment
        segment_p1 = np.array(self.closest_point_on_line(min_point, max_point, min_point)).astype(int)
        segment_p2 = np.array(self.closest_point_on_line(min_point, max_point, max_point)).astype(int)

        print("segment_p1", segment_p1)
        print("segment_p2", segment_p2)

        midpoint = self.get_midpoint(segment_p1, segment_p2)

        # ! in which direction is the inside of the cluster?
        p3, p4 = self.get_perp(segment_p1, midpoint)
        p3, p4 = np.array(p3).astype(int), np.array(p4).astype(int)
        print("p3", p3, "p4", p4)

        return segment_p1, segment_p2, p3, p4, min_point, max_point

    def lever_detector(self, points, depth_masked):
        # depth_axis_pts = points[:, self.depth_axis]
        # print("depth_axis_pts", depth_axis_pts.shape)
        # points = points[depth_axis_pts != 0]
        # print("new points", points.shape)
        # points = self.threshold(points) #! DISABLED
        print("new2 points", points.shape)
        clusters, num_of_points = self.clustering(points)

        print("number of clusters", len(clusters))

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
                # cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_CLOSE, kernel)
                cluster_img = cv2.erode(cluster_img, kernel, iterations=1)
                # cluster_img = cv2.dilate(cluster_img, kernel, iterations=1)
                # cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_OPEN, kernel)

                # we sample evenly along the contour, and this is better when using CHAIN_APPROX_NONE instead of
                # CHAIN_APPROX_SIMPLE
                cluster_contours, _ = cv2.findContours(cluster_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cluster_contours = list(cluster_contours)
                cluster_contours.sort(key=lambda x: len(x), reverse=True)
                contour = cluster_contours[0]
                contours.append(contour)

                # cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_GRAY2BGR)
                # cv2.drawContours(cluster_img, [contour], -1, (0, 255, 0), 2)
                # cv2.namedWindow('cluster_img', cv2.WINDOW_NORMAL)
                # cv2.imshow('cluster_img', cluster_img)
                # cv2.waitKey(0)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        good_points = []
        lines = []
        points_min_max = []
        medoids = []
        for cnt1, cnt2 in combinations(contours, 2):
            # do these contours share a close edge?
            approx_sample_len = 40
            precise_sample_search_widening = 50

            # contours should be at least the sample length, otherwise they are tiny contours and can be ignored
            if len(cnt1) > approx_sample_len and len(cnt2) > approx_sample_len:

                area1 = cv2.contourArea(cnt1)
                area2 = cv2.contourArea(cnt2)

                if area1 > self.MIN_LEVERABLE_AREA or area2 > self.MIN_LEVERABLE_AREA:

                    # ! we can take this outside of the combinations for loop
                    # approximately evenly sampled points along the contour
                    cnt1_sample_idx = np.arange(len(cnt1), step=int(len(cnt1) / approx_sample_len))
                    cnt2_sample_idx = np.arange(len(cnt2), step=int(len(cnt2) / approx_sample_len))
                    # cnt1_sample = cnt1[cnt1_sample_idx].squeeze()
                    # cnt2_sample = cnt2[cnt2_sample_idx].squeeze()

                    points1, points1_idx, points2, points2_idx = self.get_close_points_from_2_clusters(cnt1, cnt2,
                                                                                                  cnt1_sample_idx,
                                                                                                  cnt2_sample_idx)

                    # now get more precise points around this part of the contour
                    if len(points1) > 0 and len(points2) > 0:
                        cnt1_sample_idx_precise = self.precise_idx_range(cnt1, points1_idx)
                        cnt2_sample_idx_precise = self.precise_idx_range(cnt2, points2_idx)

                        points1_p, points1_idx_p, points2_p, points2_idx_p = self.get_close_points_from_2_clusters(cnt1, cnt2,
                                                                                                              cnt1_sample_idx_precise,
                                                                                                              cnt2_sample_idx_precise)

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

                        print("\n")
                        print("area1", area1, "area2", area2)
                        print("mean diff", diff)
                        print("mean1", mean1, "mean2", mean2)

                        if diff < 0 and area2 > self.MIN_LEVERABLE_AREA:
                            print("lever from cluster 2")
                            points, points_idx, cnt = points2, points2_idx, cnt2
                        elif diff > 0 and area1 > self.MIN_LEVERABLE_AREA:
                            print("lever from cluster 1")
                            points, points_idx, cnt = points1, points1_idx, cnt1
                        else:
                            break

                        segment_p1, segment_p2, p3, p4, min_point, max_point = self.get_middle_point(points_idx, points, cnt)
                        # middle2_idx = self.get_middle_point(points2_idx, points2, cnt2)

                        # middle = cnt1[middle_idx].squeeze()
                        # print("middle", middle)
                        # medoids.append(middle)
                        lines.append([segment_p1, segment_p2])
                        lines.append([p3, p4])

                        # points_min_max.append([tuple(cnt[min_idx].squeeze()), tuple(cnt[max_idx].squeeze())])
                        points_min_max.append([tuple(np.array(min_point).astype(int)), tuple(np.array(max_point).astype(int))])

                        # middle2 = cnt2[middle2_idx].squeeze()
                        # print("middle1", middle1)
                        # print("middle2", middle2)
                        # medoids.extend([middle1, middle2])

                    # for showing all the points
                    good_points.extend(points1)
                    good_points.extend(points2)

        # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # contours, hierarchy = cv2.findContours(img_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))

        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        for point in good_points:
            cv2.circle(img, tuple(point), 2, (0, 0, 255), -1)

        for p_min, p_max in points_min_max:
            cv2.circle(img, p_min, 6, (50, 141, 168), -1)
            cv2.circle(img, p_max, 6, (190, 150, 37), -1)

        for p1, p2 in lines:
            cv2.line(img, p1, p2, (255, 0, 0), 3)



        # for point in medoids:
        #     cv2.circle(img, tuple(point), 6, (255, 0, 0), -1)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)

        return pcd

    # def depth_to_image(depth_list):
    #     height, width = (480, 640)
    #     img = np.zeros((height, width))
    #     for i in np.arange(height):
    #         for j in np.arange(width):
    #             img[i, j] = depth_list[i * j + j]
    #
    #     return img


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




# Numpy and scikit-learn
import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum, threshold_li
from skimage.filters import threshold_yen
from scipy.spatial import ConvexHull
import sklearn.cluster as cluster
import hdbscan
import time
import open3d as o3d
import cv2
from itertools import combinations, product
# import config

# Own Modules
import gap_detection.helpers2 as helpers
from helpers import get_colour

# Events? Dont know what threding is for
import threading

class GapDetector:
    def __init__(self):
        self.detected_gap = threading.Event()  # event variable to signal gap detection
        self.image_received = threading.Event()  # event variable to signal gap detection
         
        self.potential_gaps_pub = []
        self.convex_hull_marker_pub = []
        self.centers_marker_pub = []
        self.volume_text_marker_pub = []

        self.depth_axis = 2
        self.clustering_mode = 3 # 0 to 3
        self.create_evaluation = False
        self.automatic_thresholding = -1 # -1 to 3
        self.min_gap_volume = 0.1 # 0.0 to 200.0
        self.max_gap_volume = 20000.0 # 0.0 to 200.0
        self.KM_number_of_clusters = 3 # 1 to 10
        self.B_branching_factor = 50 # 2 to 200
        self.B_threshold = 0.015 # 0.0 to 1.0
        self.DB_eps = 0.01 # 0.0001 to 0.02
        self.HDB_min_cluster_size = 50 # 5 to 150
        self.otsu_bins = 800 # 2 to 1024

        self.gaps = []  #gap lsit

    def __str__(self):
        if self.gaps:
            return str(np.asanyarray(self.gaps)[:,4]) + "\nNext round!"

        else:
            return str("No Gaps Detected.")

    def threshold(self, points):
        # ----- AUTOMATIC THRESHOLDING TO FIND GAPS -----
       # points = helpers.VTX_to_numpy_array(points)
        depth_axis_pts = points[:,self.depth_axis]
        if self.automatic_thresholding == -1:
            self.surface_height = np.median(points[:, self.depth_axis])
            return points

        if(self.automatic_thresholding == 0):
            try:
                threshold = threshold_minimum(depth_axis_pts)
            except RuntimeError:
               raise Exception('Threshold_minimum was unable to find two maxima in histogram!')
        elif(self.automatic_thresholding == 1):
            threshold = threshold_li(depth_axis_pts)
        elif(self.automatic_thresholding == 2):
            threshold = threshold_yen(depth_axis_pts)
        elif(self.automatic_thresholding == 3):
            threshold = threshold_otsu(depth_axis_pts, self.otsu_bins)
        else:
            raise Exception('Automatic threshold value out of bounds!')

        print("Threshold is: ", threshold)

        device_surface_pts = points[depth_axis_pts <= threshold]
        self.surface_height = np.median(device_surface_pts[:, self.depth_axis])
        print("Device surfaceheight = ", self.surface_height)
        points = points[depth_axis_pts <= threshold]
        

        return points

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
            if(num_of_points >= 4):
                clusters.append(cluster)

        clusters.sort(key=lambda x: len(x), reverse=True)
        
        print("num clusters: ", len(clusters), num_of_points)

        return (clusters, num_of_points)


# =============== MAIN DETECTION LOOP ===============
    def detector_callback(self, points):

        depth_axis_pts = points[:, self.depth_axis]
        print("depth_axis_pts", depth_axis_pts.shape)
        points = points[depth_axis_pts != 0]
        print("new points", points.shape)
        points = self.threshold(points) #! DISABLED
        print("new2 points", points.shape)
        clusters, num_of_points = self.clustering(points)
        # ----- VOLUME CORRECTION -----
        volume_corrected_clusters = []
        num_vol_corrected_pts = 0
        volume_corrected_pts_tuple = ()
        for cluster in clusters:
            hull = ConvexHull(cluster, qhull_options="QJ")

            # Map from vertex to point in cluster
            convex_hull_vertices = []
            for vertex in hull.vertices:
                x, y, z = cluster[vertex]
                convex_hull_vertices.append([x, y, z])

            gap = cluster.tolist()
            for vertex in convex_hull_vertices:
                # For each vertex, add a new point to the gap with the height
                # of the surface and the other axes corresponding to the vertex
                if(self.depth_axis == 0):
                    volume_point = [self.surface_height, vertex[1], vertex[2]]
                elif(self.depth_axis == 1):
                    volume_point = [vertex[0], self.surface_height, vertex[2]]
                elif(self.depth_axis == 2):
                    volume_point = [vertex[0], vertex[1], self.surface_height]

                gap.append(volume_point)
                num_vol_corrected_pts = num_vol_corrected_pts + 1

            volume_corrected_clusters.append(np.array(gap))
            volume_corrected_pts_tuple = volume_corrected_pts_tuple + \
                (num_vol_corrected_pts,)
            num_vol_corrected_pts = 0

        # ---- CALCULATING CONVEX HULLS OF GAPS AND THEIR CENTER -----
        self.convex_hulls_and_info = helpers.calculate_convex_hulls_and_centers(volume_corrected_clusters) 

        # ---- FILTER BASED ON VOLUME -----
        self.gaps = [] 
        for gap_info in self.convex_hulls_and_info:
            gap_volume = gap_info[4]
            if(self.min_gap_volume <= gap_volume <= self.max_gap_volume):
                self.gaps.append(gap_info)

        
        # ----- EVALUATION -----
        if(self.create_evaluation):
            num_of_points = np.subtract(num_of_points, num_vol_corrected_pts)
            helpers.evaluate_detector(num_of_points)
            self.create_evaluation = False

        self.detected_gap.set()  # signal succesful gap detection

    # def get_close_points_from_2_clusters(self, cnt1, cnt2, cnt1_sample_idx, cnt2_sample_idx):
    #     points1 = []
    #     points1_idx = []
    #     points2 = []
    #     points2_idx = []
    #     for point_idx1, point_idx2 in product(cnt1_sample_idx, cnt2_sample_idx):
    #         point1 = cnt1[point_idx1].squeeze()
    #         point2 = cnt2[point_idx2].squeeze()
    #         dist = np.linalg.norm(point1 - point2)
    #         if dist < MAX_CLUSTER_DISTANCE:
    #             points1.append(point1)
    #             points1_idx.append(point_idx1)
    #             points2.append(point2)
    #             points2_idx.append(point_idx2)
    #     return points1, points1_idx, points2, points2_idx
    #
    # @staticmethod
    # def precise_idx_range(cnt, points_idx):
    #     step_size = 10
    #     epsilon = 5
    #     new_points_idx = []
    #     # around each point, create an epsilon-ball, and add those points to the list
    #     for point_idx in points_idx:
    #         min_idx = point_idx - epsilon * step_size
    #         max_idx = point_idx + epsilon * step_size
    #         if min_idx < 0:
    #             min_idx % len(cnt)
    #             new_points_idx.extend(np.arange(min_idx % len(cnt), stop=len(cnt), step=step_size))
    #
    #         if max_idx >= len(cnt):
    #             max_idx % len(cnt)
    #             new_points_idx.extend(np.arange(0, stop=max_idx % len(cnt), step=step_size))
    #
    #         new_points_idx.extend(np.arange(np.clip(min_idx, 0, len(cnt)),
    #                                         stop=np.clip(max_idx, 0, len(cnt)),
    #                                         step=step_size))
    #
    #     # remove duplicate points from the list
    #     new_points_idx = list(set(new_points_idx))
    #
    #     return new_points_idx
    #
    # @staticmethod
    # def sort_and_remove_duplicates(points_idx, points):
    #     idxs_and_points = list(zip(points_idx, points))
    #     idxs_and_points.sort(key=lambda x: x[0], reverse=False)
    #     idxs_and_points = np.array(idxs_and_points, dtype=object)
    #     _, indices = np.unique(idxs_and_points[:, 0], return_index=True)
    #     idxs_and_points = idxs_and_points[indices, :]
    #     # points_idx_sorted, points_sorted = zip(*idxs_and_points)
    #     return zip(*idxs_and_points)
    #
    # @staticmethod
    # def closest_point_on_line(p1, p2, p3):
    #     x1, y1 = p1
    #     x2, y2 = p2
    #     x3, y3 = p3
    #     dx, dy = x2 - x1, y2 - y1
    #     det = dx * dx + dy * dy
    #     a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
    #     return x1 + a * dx, y1 + a * dy
    #
    # @staticmethod
    # def get_midpoint(p1, p2):
    #     p1_x, p1_y = p1
    #     p2_x, p2_y = p2
    #     return [(p1_x + p2_x) / 2, (p1_y + p2_y) / 2]
    #
    # @staticmethod
    # def get_perp(p1, p2, cd_length=50):
    #     ab = LineString([p1, p2])
    #     left = ab.parallel_offset(cd_length / 2, 'left')
    #     right = ab.parallel_offset(cd_length / 2, 'right')
    #     c = left.boundary[1]
    #     d = right.boundary[0]  # note the different orientation for right offset
    #     cd = LineString([c, d])
    #     return c, d
    #
    # def lever_detector(self, points, depth_masked):
    #     MIN_CLUSTER_SIZE = 150  # number of points in cluster
    #     MAX_CLUSTER_DISTANCE = 20  # in pixels
    #
    #     # depth_axis_pts = points[:, self.depth_axis]
    #     # print("depth_axis_pts", depth_axis_pts.shape)
    #     # points = points[depth_axis_pts != 0]
    #     # print("new points", points.shape)
    #     # points = self.threshold(points) #! DISABLED
    #     print("new2 points", points.shape)
    #     clusters, num_of_points = self.clustering(points)
    #
    #     print("number of clusters", len(clusters))
    #
    #     clustered_points = []
    #     colours = []
    #     for index, cluster in enumerate(clusters):
    #         print("cluster.shape", cluster.shape, np.mean(cluster))
    #
    #         clustered_points.extend(cluster)
    #         if len(cluster) < MIN_CLUSTER_SIZE:
    #             cluster_colours = [np.array((180, 180, 180), dtype=np.float64) / 255] * len(cluster)
    #         else:
    #             cluster_colours = [get_colour(index) / 255] * len(cluster)
    #
    #         colours.extend(cluster_colours)
    #
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(np.array(clustered_points))
    #     pcd.colors = o3d.utility.Vector3dVector(np.array(colours))
    #
    #     height, width = (480, 640)
    #     kernel = np.ones((2, 2), np.uint8)
    #     img = np.zeros((height, width, 3), dtype=np.uint8)
    #     contours = []
    #     for index, cluster in enumerate(clusters):
    #         if len(cluster) > MIN_CLUSTER_SIZE:
    #             # create an inverse image, so that our contour is on the inside of the object
    #             cluster_img = np.zeros((height, width), dtype=np.uint8)
    #             for x, y, z in cluster:
    #                 cluster_colour = np.asarray(get_colour(index), dtype=np.uint8)
    #                 # print(x, y, z, cluster_colour)
    #                 img[int(x), int(y)] = cluster_colour
    #                 cluster_img[int(x), int(y)] = 255
    #
    #             # apply erosion so that the contour is inside the object
    #             # cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_CLOSE, kernel)
    #             cluster_img = cv2.erode(cluster_img, kernel, iterations=1)
    #             # cluster_img = cv2.dilate(cluster_img, kernel, iterations=1)
    #             # cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_OPEN, kernel)
    #
    #             # we sample evenly along the contour, and this is better when using CHAIN_APPROX_NONE instead of
    #             # CHAIN_APPROX_SIMPLE
    #             cluster_contours, _ = cv2.findContours(cluster_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #             cluster_contours = list(cluster_contours)
    #             cluster_contours.sort(key=lambda x: len(x), reverse=True)
    #             contour = cluster_contours[0]
    #             contours.append(contour)
    #
    #             # cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_GRAY2BGR)
    #             # cv2.drawContours(cluster_img, [contour], -1, (0, 255, 0), 2)
    #             # cv2.namedWindow('cluster_img', cv2.WINDOW_NORMAL)
    #             # cv2.imshow('cluster_img', cluster_img)
    #             # cv2.waitKey(0)
    #
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #
    #     good_points = []
    #     medoids = []
    #     for cnt1, cnt2 in combinations(contours, 2):
    #         # do these contours share a close edge?
    #         approx_sample_len = 40
    #         precise_sample_search_widening = 50
    #
    #         # contours should be at least the sample length, otherwise they are tiny contours and can be ignored
    #         if len(cnt1) > approx_sample_len and len(cnt2) > approx_sample_len:
    #
    #             area1 = cv2.contourArea(cnt1)
    #             area2 = cv2.contourArea(cnt2)
    #             # ! we can take this outside of the combinations for loop
    #             # approximately evenly sampled points along the contour
    #             cnt1_sample_idx = np.arange(len(cnt1), step=int(len(cnt1)/approx_sample_len))
    #             cnt2_sample_idx = np.arange(len(cnt2), step=int(len(cnt2)/approx_sample_len))
    #             # cnt1_sample = cnt1[cnt1_sample_idx].squeeze()
    #             # cnt2_sample = cnt2[cnt2_sample_idx].squeeze()
    #
    #             points1, points1_idx, points2, points2_idx = get_close_points_from_2_clusters(cnt1, cnt2,
    #                                                                                           cnt1_sample_idx,
    #                                                                                           cnt2_sample_idx)
    #
    #
    #
    #             # now get more precise points around this part of the contour
    #             if len(points1) > 0 and len(points2) > 0:
    #                 cnt1_sample_idx_precise = precise_idx_range(cnt1, points1_idx)
    #                 cnt2_sample_idx_precise = precise_idx_range(cnt2, points2_idx)
    #
    #                 points1_p, points1_idx_p, points2_p, points2_idx_p = get_close_points_from_2_clusters(cnt1, cnt2,
    #                                                                                               cnt1_sample_idx_precise,
    #                                                                                               cnt2_sample_idx_precise)
    #
    #                 points1.extend(points1_p)
    #                 points1_idx.extend(points1_idx_p)
    #                 points2.extend(points2_p)
    #                 points2_idx.extend(points2_idx_p)
    #
    #                 points1_idx, points1 = sort_and_remove_duplicates(points1_idx, points1)
    #                 points2_idx, points2 = sort_and_remove_duplicates(points2_idx, points2)
    #
    #                 # ! uniformly sample these points!
    #
    #                 def get_middle_point(points_idx, points, cnt):
    #
    #                     min_idx = np.amin(points_idx)
    #                     max_idx = np.amax(points_idx)
    #
    #                     # it could be that the contour starts somewhere in the middle of the segment
    #                     # eg. contour length 700. Segment exists at 670, 680, 0, 10, 20
    #                     if (max_idx + 50) % len(cnt1) < min_idx:
    #                         print("there exists a jump")
    #
    #                         prev_point_idx = points_idx[0]
    #                         for point_idx in points_idx:
    #                             if point_idx > prev_point_idx + 50:
    #                                 max_idx = prev_point_idx
    #                                 print("iterating forwards, jump at", prev_point_idx, point_idx)
    #                                 break
    #
    #                             prev_point_idx = point_idx
    #
    #                         prev_point_idx = points_idx[-1]
    #                         for point_idx in np.flip(points_idx):
    #                             if point_idx < prev_point_idx - 50:
    #                                 print("iterating back, jump at", prev_point_idx, point_idx)
    #                                 min_idx = prev_point_idx
    #                                 break
    #
    #                             prev_point_idx = point_idx
    #
    #                     print("min idx:", min_idx)
    #                     print("max idx:", max_idx)
    #                     print("len(cnt)", len(cnt))
    #                     middle_idx_idx = (points_idx.index(min_idx) + int(len(points_idx)/2)) % len(points_idx)
    #                     middle_idx = points_idx[middle_idx_idx]
    #
    #
    #
    #                     from scipy.optimize import curve_fit
    #
    #                     def f(x, A, B):  # this is your 'straight line' y=f(x)
    #                         return A * x + B
    #
    #                     print("points x y", np.array(points)[:, 0].shape, np.array(points)[:, 1].shape, np.array(points).shape)
    #
    #                     popt, pcov = curve_fit(f, np.array(points)[:, 0], np.array(points)[:, 1])  # your data x, y to fit
    #                     m = popt[0]  # slope
    #                     b = popt[1]  # intercept
    #
    #                     # two points on the line are
    #                     p1 = [0, b]
    #                     if np.abs(m) > 0.1:
    #                         p2 = [-b/m, 0]
    #                     else:
    #                         p2 = [1, m+b]
    #
    #                     cv2.circle(img, tuple(cnt[min_idx].squeeze()), 6, (190, 150, 37), -1)
    #                     cv2.circle(img, tuple(cnt[max_idx].squeeze()), 6, (190, 150, 37), -1)
    #
    #
    #                     # our line segment is then:
    #                     print("cnt[min_idx].squeeze()", cnt[min_idx].squeeze())
    #                     print("cnt[max_idx].squeeze()", cnt[max_idx].squeeze())
    #                     # segment_p1 = np.array(closest_point_on_line(p1, p2, cnt[min_idx].squeeze())).astype(int)
    #                     # segment_p2 = np.array(closest_point_on_line(p1, p2, cnt[max_idx].squeeze())).astype(int)
    #                     #! the above isn't working really well. I think the points need to be distributed uniformly for it to work
    #
    #                     segment_p1 = np.array(closest_point_on_line(cnt[min_idx].squeeze(), cnt[max_idx].squeeze(), cnt[min_idx].squeeze())).astype(int)
    #                     segment_p2 = np.array(closest_point_on_line(cnt[min_idx].squeeze(), cnt[max_idx].squeeze(), cnt[max_idx].squeeze())).astype(int)
    #
    #                     print("segment_p1", segment_p1)
    #                     print("segment_p2", segment_p2)
    #
    #
    #
    #                     midpoint = get_midpoint(segment_p1, segment_p2)
    #
    #                     from shapely.geometry import LineString
    #
    #
    #
    #                     p3, p4 = get_perp(segment_p1, midpoint)
    #                     p3, p4 = np.array(p3).astype(int), np.array(p4).astype(int)
    #                     print("p3", p3, "p4", p4)
    #
    #                     cv2.line(img, p3, p4, (255, 0, 0), 3)
    #
    #                     cv2.line(img, segment_p1, segment_p2, (255, 0, 0), 3)
    #
    #                     return middle_idx
    #
    #                 middle1_idx = get_middle_point(points1_idx, points1, cnt1)
    #                 middle2_idx = get_middle_point(points2_idx, points2, cnt2)
    #
    #                 middle1 = cnt1[middle1_idx].squeeze()
    #                 middle2 = cnt2[middle2_idx].squeeze()
    #                 print("middle1", middle1)
    #                 print("middle2", middle2)
    #                 medoids.extend([middle1, middle2])
    #
    #                 # compute avg height of points1 and points2
    #                 # this will tell us which side is the lower side of the gap
    #                 heights1 = []
    #                 heights2 = []
    #                 for point in points1:
    #                     height = depth_masked[point[1], point[0]]
    #                     if height != 0.0:
    #                         heights1.append(height)
    #                 for point in points2:
    #                     height = depth_masked[point[1], point[0]]
    #                     if height != 0.0:
    #                         heights2.append(height)
    #
    #                 mean1 = np.mean(heights1)
    #                 mean2 = np.mean(heights2)
    #                 diff = mean1 - mean2
    #
    #                 print("\n")
    #
    #                 if diff < 0:
    #                     print("lever from cluster 2")
    #                 else:
    #                     print("lever from cluster 1")
    #
    #                 print("area1", area1, "area2", area2)
    #                 print("mean diff", diff)
    #                 print("mean1", mean1, "mean2", mean2)
    #
    #             # for showing all the points
    #             good_points.extend(points1)
    #             good_points.extend(points2)
    #
    #
    #     # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # contours, hierarchy = cv2.findContours(img_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # print(len(contours))
    #
    #     cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    #
    #     for point in good_points:
    #         cv2.circle(img, tuple(point), 2, (0, 0, 255), -1)
    #
    #     for point in medoids:
    #         cv2.circle(img, tuple(point), 6, (255, 0, 0), -1)
    #
    #     cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #     cv2.imshow('img', img)
    #
    #     return pcd

    # def depth_to_image(depth_list):
    #     height, width = (480, 640)
    #     img = np.zeros((height, width))
    #     for i in np.arange(height):
    #         for j in np.arange(width):
    #             img[i, j] = depth_list[i * j + j]
    #
    #     return img


    # =============== GAP DETECTION SERVICE ===============
    def gap_detection_call(self,depth_image):
        current_time = time.localtime()
        self.detected_gap.clear()
        self.detector_callback(depth_image) #pc is the depth frames pointcloud
        self.detected_gap.wait(3.)  # wait for 3 seconds

        if self.gaps is None:
            print('No gaps yet, skipping.')
            return []


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


    
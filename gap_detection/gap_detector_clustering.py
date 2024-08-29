# Numpy and scikit-learn
import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum, threshold_li
from skimage.filters import threshold_yen
import skimage.exposure
from rich import print

import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, Pose, Transform, Vector3, Quaternion

from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import curve_fit
from shapely.geometry import LineString, Point, Polygon
from shapely.validation import make_valid
from shapely.validation import explain_validity
import shapely

import matplotlib
import matplotlib.pyplot as plt

import open3d as o3d

from types import SimpleNamespace
import sklearn.cluster as cluster
import hdbscan
import time
import cv2
from itertools import combinations, product, permutations
import math
import random
# Own Modules
from obb import get_obb_from_contour
from helpers import get_colour, get_colour_blue, make_valid_poly, simplify_poly, img_to_camera_coords, camera_info_to_o3d_intrinsics, robust_minimum, draw_text, COLOURS, COLOURS2
from context_action_framework.types import Gap, Label, LabelFace, Detection


class GapDetectorClustering:
    def __init__(self, config):
        self.config = config

        # threshold maximum depth distance from camera, in meters
        self.MAX_DEPTH_THRESHOLD = 0.5 # meters

        self.MIN_DIST_LEVER_OBJ_CENTER_TO_DEVICE_EDGE = 20 # pixels

        # min. leverable area. The min. size of the cluster where the lever starts.
        # ! This really changes accuracy, but may be necessary for small gaps
        self.MIN_LEVERABLE_AREA = 0.000036 # = 0.006 * 0.006 meters^2, or 6mm * 6mm mm^2
        # self.MIN_LEVERABLE_AREA = 0.0001 # = 0.01 * 0.01 meters^2, or 10mm * 10mm mm^2

        self.MIN_HEIGHT_DIFFERENCE = 0.005 # 5mm

        self.MAX_DISTANCE_BETWEEN_GAP_CLUSTER_AND_LEVER_CLUSTER = 0.01 # 10mm


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

        start_time = time.time()

        labels = cluster_algorithm(points)

        print(f"time clustering: {time.time() - start_time}")

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
    def depth_stats(depth_masked, points):
        if len(points) > 0:
            heights = []
            for point in points:
                height = depth_masked[point[0], point[1]]
                if height != 0.0:
                    heights.append(height)
                
            if len(heights) > 0:
                depth_stats = SimpleNamespace()
                depth_stats.mean = np.mean(heights)
                depth_stats.median = np.median(heights)
                depth_stats.min = np.min(heights)
                depth_stats.max = np.max(heights)
                depth_stats.robust_min = robust_minimum(heights)

                return depth_stats
            
        return None


    @staticmethod
    def image_to_points_list(mask):
        depth_list = np.zeros((mask.shape[0]*mask.shape[1], 3))
        count = 0
        for i in np.arange(mask.shape[0]):
            for j in np.arange(mask.shape[1]):
                # if not np.isnan(mask[i, j]) and mask[i, j] != None:
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
        cluster_img = None
        depth_scaled = None
        device_mask = None
        
        # https://github.com/Vuuuuk/Intel-Realsense-L515-3D-Scanner/blob/master/L515_3D_Scanner.py
        if self.config.obj_detection.debug:
            print(f"median depth: {round(np.median(depth_img), 4)}")
        


        # get the first detection that is hca_back        
        detections_hca_back = graph_relations.exists(Label.hca, LabelFace.back)
        
        detection_hca_back = None
        if len(detections_hca_back) > 0:
            detection_hca_back = detections_hca_back[0]
        
        # todo: find PCB/PCB covered/internals
        # todo: get the one that is largest
        # todo: get left/right/top/bottom areas between PCB and hca_back
        # todo: find which area contains the biggest gap 
        # todo: lever that one
        
        if detection_hca_back is not None and detection_hca_back.polygon_px is not None:

            contour = detection_hca_back.mask_contour
            hull = cv2.convexHull(contour, False)
            # get mask of segmentation from contour so that we get only the largest component
            device_mask = self.mask_from_contour(hull, depth_img.shape).astype(np.uint8)
            
            # mask depth image
            # depth_masked = cv2.bitwise_and(depth_img, depth_img, mask=device_mask)
            depth_masked_ma = np.ma.masked_array(depth_img,255 - device_mask)
            
            # TODO: implement inpainting:
            # https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
            # cv2.inpaint(cluster_img,mask,3,cv.INPAINT_TELEA)


            points = self.image_to_points_list(depth_masked_ma) #! UNUSED.

            if len(points) == 0:
                return gaps, cluster_img, device_mask, depth_masked_ma, None, None, None

        else:
            print("[red]gap detector: detection_hca_back (or polygon_px) is None!")
            return gaps, cluster_img, device_mask, depth_masked_ma, None, None, None



        # threshold depth image
        # TODO: make this more robust by looking at median value.
        if self.config.realsense.debug_clustering:
            print("num points thresholded", np.count_nonzero(depth_masked_ma > self.MAX_DEPTH_THRESHOLD))
        depth_masked_ma[depth_masked_ma > self.MAX_DEPTH_THRESHOLD] = 0
        depth_masked_ma = np.ma.masked_equal(depth_masked_ma, 0.0, copy=False)


        depth_max = np.amax(depth_masked_ma)
        depth_min = np.amin(depth_masked_ma)
        depth_min_nonzero = depth_masked_ma.min() # np.min(points)
        depth_median_nonzero = np.ma.median(depth_masked_ma)

        # TODO: make cut below and above median by around 2cm in each direction. This will remove big outliers.

        if self.config.realsense.debug_clustering:
            print("gap detector, depth_min:", depth_min)
            print("gap detector, depth_min_nonzero:", depth_min_nonzero)
            print("gap detector, depth_median_nonzero:", depth_median_nonzero)
            print("gap detector, depth_max:", depth_max)

        # depth_mean = depth_masked_np.mean()
        # print("depth_mean", depth_mean)

        if depth_min == depth_max:
            if self.config.realsense.debug_clustering:
                print("[red]gap detector: depth_min == depth_max!")
            return gaps, cluster_img, device_mask, depth_masked_ma, None, None, None

        if depth_min_nonzero is np.NaN or depth_min_nonzero is None:
            if self.config.realsense.debug_clustering:
                print("[red]gap detector: depth_min_nonzero is None!")
            return gaps, cluster_img, device_mask, depth_masked_ma, None, None, None
        
        # rescale the depth to the range (0, 255) such that the clustering works well
        depth_scaled = skimage.exposure.rescale_intensity(depth_masked_ma, in_range=(depth_min_nonzero, depth_max), out_range=(0,255)).astype(np.uint8)

        print("depth_scaled", np.min(depth_scaled), np.max(depth_scaled))

        depth_gaussian = cv2.GaussianBlur(depth_scaled, (7, 7), 0) #! VERY USEFUL FOR SMOOTHING.

        #! TESTING:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        depth_opening = cv2.morphologyEx(depth_gaussian, cv2.MORPH_OPEN, kernel)


        depth_scaled_masked = np.ma.masked_array(depth_scaled, 255 - device_mask) #! ? unused?
        depth_scaled_points = self.image_to_points_list(depth_opening) # shape (n, 3) #! Changed from depth_gaussian to depth_opening.

        clusters = []
        cluster_objs = []
        gaps = []
        gaps_bad = []

        # sanity check
        if len(depth_scaled_points) > self.MIN_CLUSTER_SIZE:
            clusters, num_of_points = self.clustering(depth_scaled_points)

        # print("gap detector, num clusters:", len(clusters))

        height, width = depth_masked_ma.shape[:2]
        cluster_img = np.zeros((height, width, 3), dtype=np.uint8)

        # get the contour of each cluster
        for index, cluster in enumerate(clusters):
            # convert cluster to mask and the create contour
            cluster_img_mask = np.zeros((height, width), dtype=np.uint8)
            for x, y, z in cluster:
                cluster_img_mask[int(x), int(y)] = 255

            # use findContours to get the biggest one only
            #! RETR_EXTERNAL is bad because if the cluster has holes, they are removed. (imaging big holes, that would be bad)
            cluster_contours, _ = cv2.findContours(cluster_img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cluster_contours = list(cluster_contours)
            if len(cluster_contours) > 0:
                cluster_contours.sort(key=lambda x: len(x), reverse=True)
                contour_px = cluster_contours[0]
                cnt_px_squeeze = contour_px.squeeze()

                # convert the contour to a polygon
                if len(cnt_px_squeeze) > 3:
                    try:
                        poly_cluster_px = Polygon(cnt_px_squeeze)
                        poly_cluster_px = make_valid_poly(poly_cluster_px)

                    except AssertionError as E:
                        print("[red]gap detector: error converting contour to polygon![/red]")
                    else:
                        center_px = poly_cluster_px.centroid
                        center_px = (int(center_px.x), int(center_px.y))
                        if center_px is not None:
                            depth_stats  = self.depth_stats(depth_masked_ma, cluster[:, :2].astype(int))
                            # make sure depth_stats is not none before continuing
                            if depth_stats is not None:
                                poly_cluster_simple_px = simplify_poly(poly_cluster_px, try_max_num_coords=30)
                                poly_cluster = img_to_camera_coords(poly_cluster_px, depth_img, camera_info)
                                if poly_cluster is None:
                                    # something is wrong with the depth_img possibly. use median instead
                                    poly_cluster = img_to_camera_coords(poly_cluster_px, depth_stats.median, camera_info)

                                area_px = int(poly_cluster_px.area)
                                area = None
                                if poly_cluster is not None:
                                    area = poly_cluster.area

                                cluster_colour = COLOURS2[index % len(COLOURS2)]
                                cluster_colour = (int(cluster_colour[0]), int(cluster_colour[1]), int(cluster_colour[2])) 
                                # cluster_colour = get_colour(index)

                                cluster_obj = SimpleNamespace()
                                cluster_obj.index = index
                                cluster_obj.contour_px = contour_px
                                cluster_obj.poly_px = poly_cluster_px
                                cluster_obj.poly_simple_px = poly_cluster_simple_px
                                cluster_obj.poly = poly_cluster
                                # cluster_obj.poly_proj = poly_cluster_projection

                                cluster_obj.area_px = area_px
                                cluster_obj.area = area
                                cluster_obj.center_px = center_px
                                cluster_obj.depth_stats = depth_stats
                                cluster_obj.colour = cluster_colour
                                cluster_obj.thinness_ratio = None

                                cluster_obj.is_valid = True
                                cluster_obj.invalid_reason = ""
                                cluster_obj.info = ""


                                cluster_objs.append(cluster_obj)

        # sort clusters by size:
        cluster_objs.sort(key=lambda x: x.area_px, reverse=True)

        # TODO: remove cluster that is covered completely by another
        # TODO: really this shouldn't be the case. We should not be removing the inner polygons (e.g. donuts)
        for cluster_obj1, cluster_obj2 in permutations(cluster_objs, 2):
            polygon1 = cluster_obj1.poly_px
            polygon2 = cluster_obj2.poly_px

            if polygon2.contains(polygon1):
                print("[red]polygon 2 is completely inside polygon1")
                cluster_obj2.info += "contains a poly completely. "
                cluster_obj2.is_valid = False
                cluster_obj2.invalid_reason += "contains a poly completely. "


        # filter clusters
        for idx, cluster_obj in enumerate(cluster_objs):
            
            # the center of the cluster should be inside the cluster. This removes some very funny snake like clusters
            is_centroid_inside = cluster_obj.poly_px.contains(cluster_obj.poly_px.centroid)
            if not is_centroid_inside:
                cluster_obj.is_valid = False
                cluster_obj.invalid_reason += "Centroid not inside polygon. "

            # the cluster area should be above a minimum otherwise they are too small to insert lever
            if cluster_obj.area < self.MIN_LEVERABLE_AREA:
                cluster_obj.is_valid = False
                cluster_obj.invalid_reason += f"area too small. "

            # https://math.stackexchange.com/questions/1336265/explanation-of-the-thinness-ratio-formula
            # https://gis.stackexchange.com/questions/316128/identifying-long-and-narrow-polygons-in-with-postgis
            # thinness ratio. Circle has ratio of 1.
            # use the simple polygon because otherwise the perimeter can be very long
            thinness_ratio = (cluster_obj.poly_simple_px.area * 4 * np.pi) / (cluster_obj.poly_simple_px.length**2)
            cluster_obj.thinness_ratio = thinness_ratio
            if thinness_ratio < 0.2:
                cluster_obj.is_valid = False
                cluster_obj.invalid_reason += "too thin. "
            

        counter = 0
        for cluster_obj_low, cluster_obj_high in permutations(cluster_objs, 2):

            info_text = ""
            
            # if either cluster is not valid, don't continue
            if not cluster_obj_low.is_valid or not cluster_obj_high.is_valid:
                continue

            # in this permutation, cluster_obj_low is the lever area, and cluster_obj_high gets levered.
            # the bigger the depth, the lower it is
            if cluster_obj_low.depth_stats.median <= cluster_obj_high.depth_stats.median:
                continue # skip current iteration

            if cluster_obj_low.depth_stats.median < cluster_obj_high.depth_stats.median + self.MIN_HEIGHT_DIFFERENCE:
                msg = f"{cluster_obj_low.depth_stats.median:.3f} < {cluster_obj_high.depth_stats.median:.3f} + {self.MIN_HEIGHT_DIFFERENCE} = {(cluster_obj_high.depth_stats.median + self.MIN_HEIGHT_DIFFERENCE):.3f}"
                print(msg)
                info_text += msg
                continue # skip current iteration

            # lever action is from: center_low -> center_high
            # exclude actions where: center_high is too close to the device edge.
            center_high_pt = Point(cluster_obj_high.center_px[0],cluster_obj_high.center_px[1])
            if detection_hca_back.polygon_px.exterior.distance(center_high_pt) < self.MIN_DIST_LEVER_OBJ_CENTER_TO_DEVICE_EDGE:
                continue # skip current iteration


            
            ################################ create gap object
            gap = Gap()
            gap.id = counter

            gap.from_depth = cluster_obj_low.depth_stats.median
            gap.to_depth = cluster_obj_high.depth_stats.median
            
            # todo: add these properties:
            
            # gap.from_det = 
            # gap.to_det = 

            # todo: convert to meters
            obb_px, obb_center_px, rot_quat = get_obb_from_contour(cluster_obj_low.contour_px)
            gap.obb = obb_px
            # gap.obb_3d = 

            gap.from_px = np.asarray([cluster_obj_low.center_px[0], cluster_obj_low.center_px[1]])
            gap.to_px = np.asarray([cluster_obj_high.center_px[0], cluster_obj_high.center_px[1]])
            
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
            
            gap.from_cluster = cluster_obj_low.index
            gap.to_cluster = cluster_obj_high.index
            gap.info_text = info_text
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

            ################################### END: create gap object


            is_valid_gap = True

            # consider the arrow from the low cluster to the high cluster. 
            # The distance along this line between these clusters must be less than a parameterised distance (eg 10mm).
            # further away than that, and the levering action doesn't make sense
            centroid_from = cluster_obj_low.poly_px.centroid
            centroid_to = cluster_obj_high.poly_px.centroid
            line_meters = LineString([centroid_from, centroid_to])
            intersection_from = line_meters.intersection(cluster_obj_low.poly_px.boundary)
            intersection_to = line_meters.intersection(cluster_obj_high.poly_px.boundary)

            min_dist_between_polys_along_arrow = None
            if not intersection_from.is_empty and not intersection_to.is_empty:
                if intersection_from.geom_type == 'MultiPoint':
                    point_from = max(intersection_from.geoms, key=lambda point: point.distance(centroid_from))
                else:
                    point_from = intersection_from

                if intersection_to.geom_type == 'MultiPoint':
                    point_to = max(intersection_to.geoms, key=lambda point: point.distance(centroid_to))
                else:
                    point_to = intersection_to

                # project from pixels to meters. First convert Point to np.array.
                point_from_arr = np.array([int(point_from.x), int(point_from.y)])
                point_to_arr = np.array([int(point_to.x), int(point_to.y)])
                # we use median instead here, because taking the depth at a point has a chance it is undefined
                point_from_meters = img_to_camera_coords(point_from_arr, cluster_obj_low.depth_stats.median, camera_info)
                point_to_meters = img_to_camera_coords(point_to_arr, cluster_obj_high.depth_stats.median, camera_info)

                # project (x, y, z) points onto the x-y plane.
                point_from_meters = Point(point_from_meters[:2])
                point_to_meters = Point(point_to_meters[:2])

                min_dist_between_polys_along_arrow_meters = point_from_meters.distance(point_to_meters)

                min_dist_between_polys_along_arrow = point_from.distance(point_to)

                info_text = f"min_dist(px) {min_dist_between_polys_along_arrow:.0f}, min_dist(m) {min_dist_between_polys_along_arrow_meters:.3f}"
                gap.info_text += info_text #! add info for debugging

                if min_dist_between_polys_along_arrow_meters > self.MAX_DISTANCE_BETWEEN_GAP_CLUSTER_AND_LEVER_CLUSTER:
                    is_valid_gap = False

            # simpler solution, but not as good. 
            # test if clusters are "next to each other". If there is a small cluster in the way, it should still be okay.
            # min_dist_between_polys = shapely.distance(cluster_obj_low.poly, cluster_obj_high.poly)
            # print("min_dist_between_polys", min_dist_between_polys)
            # if min_dist_between_polys > self.MAX_DISTANCE_BETWEEN_GAP_CLUSTER_AND_LEVER_CLUSTER:
            #     print("too far apart!!", min_dist_between_polys)
            #     is_valid_gap = False

            if is_valid_gap:
                gaps.append(gap)
            else:
                # center_high too close to device edge
                gaps_bad.append(gap)
                

        # ? sort the lever actions based on which one is closest to the center of the device
        # gaps.sort(key=lambda gap: np.linalg.norm(gap[0][:2] - obj_center), reverse=False)
        
        for cluster_obj in cluster_objs:            
            # Convert the Shapely polygon to a NumPy array of integer points
            poly_cluster_px_contour = np.array(cluster_obj.poly_px.exterior.coords, dtype=np.int32)

            # Reshape for OpenCV - contours need to be in shape (number_of_points, 1, 2)
            poly_cluster_px_contour = poly_cluster_px_contour.reshape((-1, 1, 2))

            # colour invalid clusters red
            if cluster_obj.is_valid:
                colour = cluster_obj.colour
            else:
                colour = (0, 0, 255)

            cv2.drawContours(cluster_img, [poly_cluster_px_contour], 0, color=colour, thickness=cv2.FILLED)

            if not cluster_obj.is_valid:
                # draw a white border around the invalid contours
                cv2.drawContours(cluster_img, [poly_cluster_px_contour], 0, color=(255, 255, 255), thickness=1)


        def draw_arrow_with_text(gap, colour, show_text=True):
            cv2.arrowedLine(cluster_img, gap.from_px.astype(int),
                            gap.to_px.astype(int), colour, 3, tipLength=0.3)
            
            if show_text and len(gap.info_text) > 0:
                # write text at midpoint of line
                line_px = LineString([gap.from_px.astype(int), gap.to_px.astype(int)])
                midpoint_line_px = line_px.interpolate(0.5, normalized=True)
                midpoint_line_px = (int(midpoint_line_px.x), int(midpoint_line_px.y))
                draw_text(cluster_img, gap.info_text, pos=midpoint_line_px, font_scale=1, text_color=colour) 

        if self.config.realsense.debug_clustering:
            for idx, gap in enumerate(gaps_bad):
                colour = tuple([int(x) for x in [0, 0, 255]])
                draw_arrow_with_text(gap, colour)

        for idx, gap in enumerate(gaps):
            colour = tuple([int(x) for x in get_colour_blue(idx)])
            draw_arrow_with_text(gap, colour, show_text=self.config.realsense.debug_clustering)

        # print avg height of each cluster
        count_clusters = 0
        for cluster_obj in cluster_objs:
            if cluster_obj.is_valid:
                if not np.isnan(cluster_obj.depth_stats.median):
                    text = f"depth(median: {cluster_obj.depth_stats.median:.3f}, min: {cluster_obj.depth_stats.min:.3f}), area(px: {cluster_obj.area_px}, m: {cluster_obj.area:.5f}) thinness: {cluster_obj.thinness_ratio:.3f}"
                    if len(cluster_obj.info) > 0:
                        text += f" {cluster_obj.info}"
                else:
                    text = "depth median: NaN"
                # draw_text(cluster_img, text, pos=cluster_obj.center_px, font_scale=1, text_color=cluster_obj.colour)
                draw_text(cluster_img, text, pos=(10, 10 + count_clusters*20), font_scale=1, text_color=cluster_obj.colour) # also write at the top left in a list
                count_clusters += 1
            elif len(cluster_obj.invalid_reason) > 0:
                text = f"{cluster_obj.invalid_reason}"
                text += f"depth(median: {cluster_obj.depth_stats.median:.3f}, min: {cluster_obj.depth_stats.min:.3f}), area(px: {cluster_obj.area_px}, m: {cluster_obj.area:.5f}) thinness: {cluster_obj.thinness_ratio:.3f}"
                if len(cluster_obj.info) > 0:
                    text += f"{cluster_obj.info}"

                # ! only for debugging:
                if self.config.realsense.debug_clustering:
                    draw_text(cluster_img, text, pos=cluster_obj.center_px, font_scale=1, text_color=(0, 0, 255))
                    draw_text(cluster_img, text, pos=(10, 10 + count_clusters*20), font_scale=1, text_color=(0, 0, 255))
                    count_clusters += 1

        
        return gaps, cluster_img, device_mask, depth_masked_ma, depth_scaled, depth_scaled_masked, depth_gaussian, depth_opening
    
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

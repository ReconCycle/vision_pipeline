import os
import sys
import numpy as np
from scipy import spatial
import cv2
import math
from rich import print
from probreg import cpd
from probreg import callbacks
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from helpers import scale_img
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import datetime


# https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
class HoughBundler:     
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
      
        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)
    

class WorkSurfaceDetection:
    def __init__(self, img, border_width=0, debug=False):
        
        self.border_width = border_width
        self.debug = debug
        
        self.img_width = None
        self.img_height = None
        self.coord_transform = None
        self.coord_transform_inv = None
        self.circles = None
        self.circles_ignoring = None
        
        self.font_face = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 1.0
        self.font_thickness = 1
        
        # affine transform
        self.pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        self.unpad = lambda x: x[:, :-1]
        
        if isinstance(img, str):
            img = np.array(cv2.imread(img))
        else:
            img = np.array(img)
        
        if img is not None:
            print("running worksurface detection...")
            self.run_detection(img)
        else:
            print("No image found!")

    @staticmethod
    def generate_all_sides(m_bottom_dict, all_m_dict):
        top = lambda x, y: [0.6 - x, 0.6 - y]
        right = lambda x, y: [0.6 - y, x]
        bottom = lambda x, y: [x, y]
        left = lambda x, y: [y, 0.6 - x]
    
        for key in m_bottom_dict:
            for translate, name in [(top, "t"), (right, "r"), (bottom, "b"), (left, "l")]:
                new_key = key + "_" + name
                all_m_dict[new_key] = translate(*m_bottom_dict[key])

    @staticmethod
    def flip_horizontal_m_dict(m_dict):
        copy_dict = m_dict.copy()
        for key, val, in copy_dict.items():
            val[1] = 0.60 - val[1]
        return copy_dict

    def run_detection(self, img):
        
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]

        if self.img_width != 1450 or self.img_height != 1450:
            print("[red]Work surface detection has been tuned for imgs 1450x1450!")
            sys.exit("Work surface detection has been tuned for imgs 1450x1450!")
        
        self.corners_px_dict = {}
        self.bolts_px_dict = {}
        
        self.corners_m_dict = {}
        self.bolts_m_dict = {}

        # this will be generated:        
        # {
        #     'corner_t': (0.6, 0.6),
        #     'corner_r': (0.6, 0),
        #     'corner_b': (0, 0),
        #     'corner_l': (0, 0.6)
        # }
        
        # bolts and corners in real world coords
        self.corner_m_bottom_dict = {
            'corner': [0, 0]
        }
        self.bolts_m_bottom_dict = {
            'bolt0': [0.035, 0.035],
            'bolt1': [0.095, 0.015],
            'bolt2': [0.095, 0.045],
            'bolt3': [0.160, 0.030],
            'bolt4': [0.230, 0.030],
            # 'calibrationmount': [0.03, 0.3], #! only one is on the table. This affects the results
            'bolt5': [0.60 - 0.230, 0.030],
            'bolt6': [0.60 - 0.160, 0.030],
            'bolt7': [0.60 - 0.095, 0.045],
            'bolt8': [0.60 - 0.095, 0.015],
            # 'bolt9': [0.60 - 0.035, 0.035], # the last bolt gets replicated as we generate all the sides
        }
        
        # we rotate in steps of 90 degrees to now get all the bolt hole measurements
        self.generate_all_sides(self.bolts_m_bottom_dict, self.bolts_m_dict)
        self.generate_all_sides(self.corner_m_bottom_dict, self.corners_m_dict)
        
        print("corners_m_dict", self.corners_m_dict)

        # 1. mask everything but the work surface
        # img = self.mask_worksurface(img) #! BROKEN in some cases

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(gray, (5, 5), 0) # remove artefacts in image
        
        # 1st estimate for affine transform using only bolts
        found_bolts = self.find_bolts()
        
        if not found_bolts:
            print("[red]work surface detection: no bolts found![/red]")
            return
        
        self.bolt_point_matching()
        self.compute_affine_transform()
        
        self.estimate_corners_using_transform()
        
        # 2nd estimate for affine transformation using also corners
        #! this usually makes things worse
        # self.improve_corner_estimate_using_corner_detection(img)
        # self.compute_affine_transform()
        
        # for debugging, draw everything
        if self.debug:
            self.draw_corners_and_circles(img, show=False)

            
        
    def mask_worksurface(self, img):
        
        img = img.copy()
        img_height, img_width = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # low_thresh = 0.5*high_thresh
        # print("low_thresh", low_thresh)
        # print("high_thresh", high_thresh)
        
        edges = cv2.Canny(gray, threshold1=50, threshold2=100, apertureSize=3)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        edges_rgb1 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        edges_rgb2 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        edges_rgb3 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
        
        for line_unsqueezed in lines:
            line = line_unsqueezed[0]
            cv2.line(edges_rgb1, (line[0], line[1]), (line[2],line[3]), (0,0,255), 6)
        
        bundler = HoughBundler(min_distance=10,min_angle=5)
        lines_processed = bundler.process_lines(lines)
        lines_processed = lines_processed.reshape(-1, 2, 2)
        
        for line in lines_processed:
            cv2.line(edges_rgb2, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
        
        norm = np.linalg.norm(lines_processed[:, 0] - lines_processed[:, 1], axis=1)
        
        long_lines = []
        for i, line in enumerate(lines_processed):
            if norm[i] > img.shape[0] *0.8: # should be at least 80% of the image long
                long_lines.append(lines_processed[i])
        long_lines = np.array(long_lines)
        
        # for line in long_lines:
            # cv2.line(edges_rgb3, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
            
        # sort into vertical and horizontal lines
        vertical_lines = []
        horizontal_lines = []
        for line in long_lines:
            dx = np.abs(line[0][0] - line[1][0])
            dy = np.abs(line[0][1] - line[1][1])
            if dy != 0 and abs(dx/dy) < np.tan(np.radians(30)):
                # vertical
                vertical_lines.append(line)
            elif dx != 0 and abs(dy/dx) < np.tan(np.radians(30)):
                horizontal_lines.append(line)
        
        def center_point(line):
            return int((line[0][0] + line[1][0])/2), int((line[0][1] + line[1][1])/2)
        
        def sorting_vertical(line):
            # sort by center x coordinate
            return (line[0][0] + line[1][0])/2

        def sorting_horizontal(line):
            # sort by center y coordinate
            return (line[0][1] + line[1][1])/2
        
        # order the vertical_lines and vertical lines
        vertical_lines.sort(key=sorting_vertical)
        horizontal_lines.sort(key=sorting_horizontal, reverse=True) # reverse because of world coords/opencv coords
        
        left_line, right_line, bottom_line, top_line = None, None, None, None

        # determine leftmost/rightmost, ... lines
        max_dist_from_edge = 300
        if len(vertical_lines) > 0:
            if sorting_vertical(vertical_lines[0]) < max_dist_from_edge:
                left_line = vertical_lines[0]
            if sorting_vertical(vertical_lines[-1]) > img_width - max_dist_from_edge:
                right_line = vertical_lines[-1]
        if len(horizontal_lines) > 0:
            # inequalities are opposite to the ones for vertical because of the opencv coordinates
            if sorting_horizontal(horizontal_lines[0]) > img_height - max_dist_from_edge:
                bottom_line = horizontal_lines[0]
            if sorting_horizontal(horizontal_lines[-1]) < max_dist_from_edge:
                top_line = horizontal_lines[-1]
        
        # mask image based on lines
        for line in [left_line, right_line, bottom_line, top_line]:
            if line is not None:
                theta = np.arctan2(line[0][1]-line[1][1], line[0][0]-line[1][0])
                # extend line
                line[0][0] = int(line[0][0] + 1000*np.cos(theta))
                line[0][1] = int(line[0][1] + 1000*np.sin(theta))
                
                line[1][0] = int(line[1][0] - 1000*np.cos(theta))
                line[1][1] = int(line[1][1] - 1000*np.sin(theta))
        

        if top_line is not None:
            top_mask = np.vstack((top_line, np.array([[img_width, 0], [0, 0]])))
            cv2.drawContours(img, [top_mask], -1, (0, 0, 0), -1)
            
        if bottom_line is not None:
            bottom_mask = np.vstack((bottom_line, np.array([[img_width, img_width], [0, img_width]])))
            cv2.drawContours(img, [bottom_mask], -1, (0, 0, 0), -1)
            
        if left_line is not None:
            left_mask = np.vstack((left_line, np.array([[0, img_width], [0, 0]])))
            cv2.drawContours(img, [left_mask], -1, (0, 0, 0), -1)

        if right_line is not None:
            right_mask = np.vstack((right_line, np.array([[img_width, img_width], [img_width, 0]])))
            cv2.drawContours(img, [right_mask], -1, (0, 0, 0), -1)
        
        
        # put labels on image
        for text, line in [("left_line", left_line), ("right_line", right_line), ("bottom_line", bottom_line), ("top_line", top_line)]:
            if line is not None:
                center = center_point(line)
                print(text, center)
                cv2.putText(edges_rgb3, text, center, self.font_face, self.font_scale, (0,0,255), self.font_thickness, cv2.LINE_AA)
                cv2.putText(img, text, center, self.font_face, self.font_scale, (0,0,255), self.font_thickness, cv2.LINE_AA)
    
        for line in [left_line, right_line]:
            if line is not None:
                cv2.line(edges_rgb3, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
                cv2.line(img, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)

        for line in [bottom_line, top_line]:
            if line is not None:
                cv2.line(edges_rgb3, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,255,0), 6)
                cv2.line(img, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,255,0), 6)
        
        if self.debug:
            cv2.imshow("canny", scale_img(edges))
            cv2.imshow("houghlines", scale_img(edges_rgb1))
            cv2.imshow("bundled lines", scale_img(edges_rgb2))
            cv2.imshow("long lines only", scale_img(edges_rgb3))
            cv2.imshow("img", scale_img(img))
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        return img
        
        
    def bolt_point_matching(self):
        source = self.circles[:, :2].astype(np.float32) # x, y pairs
        
        target_keys = list(self.bolts_m_dict.keys())
        target = np.array(list(self.bolts_m_dict.values())).astype(np.float32)
        
        # flip the bolts to go from world coords to opencv coords
        target_flipped = target.copy()
        target_flipped[:, 1] = 0.6 - target_flipped[:, 1]
        
        # scale pixels to the ballpark range of the work surface in meters
        source_scaled = (source / np.max(source))

        # center the pixels to that of the work surface
        source_centroid = source_scaled.mean(axis=0)
        target_centroid = target_flipped.mean(axis=0)
        diff = target_centroid - source_centroid
        source_scaled += diff
        
        if self.debug:
            print("source_centroid", source_centroid)
            print("target_centroid", target_centroid)
            
            print("sourcsource_scalede.shape", source_scaled.shape)
            print("real_world_pts", target_flipped.shape)

            print("source_scaled", source_scaled.shape)
            print("target", target_flipped.shape)

        cbs = []
        if self.debug:
            cbs = [callbacks.Plot2DCallback(source_scaled, target_flipped)]   
        
        # a larger w will cause the point cloud model to approach a uniform 
        res = cpd.registration_cpd(source_scaled, target_flipped, 'affine', w=0.5, maxiter=200, tol=0.00015, callbacks=cbs) # affine vs nonrigid
        
        if self.debug:
            print("res.q", res.q)
            print("sigma2", res.sigma2)
            affine_m = np.array(res.transformation.b).T
            affine_t = np.array([res.transformation.t]).T
            affine = np.hstack((affine_m, affine_t))
            affine = np.vstack((affine, np.array([0, 0, 1])))
            
            print("affine matrix", affine_m)
            print("affine translation", affine_t)
            print("affine", affine)
            plt.gca().set_aspect('equal')
            plt.xlabel("x, meters")
            plt.ylabel("y, meters")
            plt.legend(loc="center left")
            plt.savefig("saves/{date:%Y-%m-%d_%H:%M:%S}_affine.png".format(date=datetime.datetime.now()), bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()
        
        result = np.copy(source_scaled)
        result = res.transformation.transform(result)
        # the same as:
        # result = np.dot(result, res.transformation.b.T) + res.transformation.t
        
        # match the result points with the closest point from the target_flipped
        def closest_node_index(node, nodes):
            index = cdist([node], nodes).argmin()
            return index

        target_flipped_copy = target_flipped.copy()
        matching_idxs = []
        for point in result:
            i = closest_node_index(point, target_flipped_copy)
            matching_dist = np.linalg.norm(target_flipped_copy[i] - point)
            # ignore matchings with too large distances
            if matching_dist < 0.05:
            
                matching_idxs.append(i)
                target_flipped_copy[i] = [2000, 2000] # very large point, so it never gets matched
            else:
                matching_idxs.append(None)
        
        dist = []
        for i in range(len(result)):
            if matching_idxs[i] is not None:
                dist.append(np.linalg.norm(target_flipped[matching_idxs[i]] - result[i]))
        dist = np.array(dist)
        
        if len(dist) == 0:
            print("[red]Something has gone wrong!")

        max_error = dist.max()
        mean_error = dist.mean()
        
        if self.debug:
            print("bolt matching max_error", max_error)
            print("bolt matching mean_error", mean_error)
            
            plt.plot(target_flipped[:,0], target_flipped[:,1],'g^', markersize = 10)
            plt.plot(result[:,0], result[:,1],'bo',  markersize = 7)
            for p in range(len(result)):
                if matching_idxs[p] is not None:
                    plt.plot([target_flipped[matching_idxs[p], 0], result[p,0]], [target_flipped[matching_idxs[p], 1], result[p,1]], 'k')
            plt.gca().set_aspect('equal')
            plt.xlabel("x, meters")
            plt.ylabel("y, meters")
            plt.savefig("saves/{date:%Y-%m-%d_%H:%M:%S}_matching.png".format(date=datetime.datetime.now()), bbox_inches='tight', dpi=300)
            plt.show()


        
        #! Check that the matching didn't rotate all the points around! 
        #! Is the top left point in the top-left quadrant of the image?
        
        # add bolts in pixels to dict        
        for i in np.arange(len(result)):
            if matching_idxs[i] is not None:
                # target_matching[i] corresponds to result[i] 
                key = target_keys[matching_idxs[i]]
                self.bolts_px_dict[key] = source[i]
    
    def find_bolts(self):

        # Finds circles in a grayscale image using the Hough transform
        # minDist: minimum distance between detected circles
        # param1: detects strong edges, as pixels which have gradient value higher than param1
        # param 2: it is the accumulator threshold for the circle centers at the detection stage
        #          The smaller it is, the more false circles may be detected.

        #! the parameters for hough circles depends on the image input size
        circles = cv2.HoughCircles(self.blur, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                    param1=50, param2=20, minRadius=15, maxRadius=20)
        # param1=50, param2=20, minRadius=15, maxRadius=20
        circles_ignoring = []

        if circles is not None:
            circles = np.array([[x, y, r] for (x, y, r) in circles[0]])
            
            # remove circles that are on the edge of the image, these can lead to incorrect results
            # in the case that worksurfaces are next to each other
            circles_inner_region = []
            for (x, y, r) in circles:
                if x < self.border_width \
                    or x > self.img_width - self.border_width \
                    or y < self.border_width \
                    or y > self.img_height - self.border_width:
                    
                    # circle is close to edge of image, so ignore
                    # for visualisation only
                    circles_ignoring.append([x, y, r])
                else:
                    circles_inner_region.append([x, y, r])
            
            circles = np.array(circles_inner_region)
            circles_ignoring = np.array(circles_ignoring)
                
        self.circles = circles
        self.circles_ignoring = circles_ignoring
        
        if circles is None:
            print("[red]No circles found! [/red]")
            return False
        
        return True
            
    
    def improve_corner_estimate_using_corner_detection(self, img):
        
        """in the neighbourhood of the corner estimation, use Harris corner
        detection to find a better approximation for the corner
        """
        img_show = np.zeros(img.shape[:3], np.uint8)
        # iterate over corners
        for key, value in self.corners_px_dict.items():
            x, y = value
            
            # create a circle mask where we apply harris corner detection in
            circle_radius = 100
            corner_mask = np.zeros(img.shape[:2], np.uint8)
            cv2.circle(corner_mask,(int(x), int(y)), circle_radius, 255, thickness=-1)
            img_masked = cv2.bitwise_and(img, img, mask=corner_mask) # for visualisation
            blur_masked = cv2.bitwise_and(self.blur, self.blur, mask=corner_mask)
            
            # get rid of noisier corners from background by using erode + dilate
            kernel = np.ones((5,5), np.uint8)
            blur_masked = cv2.erode(blur_masked, kernel, iterations=1)
            blur_masked = cv2.dilate(blur_masked, kernel, iterations=1)
            # cv2.imshow(key + "_blur+dilate", scale_img(blur_masked))
            
            # if self.debug:
            #     cv2.imshow(key + "blur", scale_img(blur_masked))
            
            # use Harris corner detection
            # blockSize - It is the size of neighbourhood considered for corner detection
            # ksize - Aperture parameter of Sobel derivative used. As size increases, edges will get more blurry
            # k - Harris detector free parameter in the equation. Bigger k, you will get less false corners. Smaller k you will get a lot more corners. Constant in the range: [0.04,0.06]
            dst = cv2.cornerHarris(blur_masked, blockSize=20, ksize=5, k=0.04)
            
            # get better accuracy: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
            dst = cv2.dilate(dst,None)
            ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0) # was 0.01 but changed to 0.001 to see more corners
            dst = np.uint8(dst)
            # find centroids
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            # define the criteria to stop and refine the corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(blur_masked,np.float32(centroids),(5,5),(-1,-1),criteria)
            
            # now find the corner closest to the original estimation
            tree = spatial.KDTree(corners)
            closest_dist, closest_index = tree.query([(x, y)])
            closest_dist = closest_dist[0]
            closest_index = closest_index[0]
            
            # if the precise corner is close enough to the estimate then we assume there were no errors
            # the new position has to be within 20px of the estimate
            if closest_dist < 20:
                # update corner values
                self.corners_px_dict[key] = corners[closest_index]
            else:
                print("[red]Corner detection too far off estimate! [/red]")
                print("[red] Using original estimate from bolt position for: " + key + "[/red]")
            
            # Now draw them
            if self.debug:
                print("img.shape", img.shape)
                print("x, y", x, y)
                print("harris corners shape", corners.shape)
                # print("harris corners", corners)
                
                print("closest_corner: dist, index", closest_dist, closest_index)
                print("closest_corner", corners[closest_index])
                print("")
                
                res = np.hstack((centroids,corners))
                res = np.int0(res)
                
                # Threshold for an optimal value, it may vary depending on the image.
                img_masked[dst>0.01*dst.max()]=[0,0,255]
                
                img_masked[res[:,1],res[:,0]]=[0,0,255] # red centroids
                img_masked[res[:,3],res[:,2]] = [0,255,0] # green corners
            
                cv2.drawMarker(img_masked, tuple(corners[closest_index].astype(int)), color=[0,255,0],
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=1,
                    line_type=8)
                
                img_show += img_masked
                
        
        if self.debug:
            cv2.imshow("corners", scale_img(img_show))
            cv2.waitKey()
            cv2.destroyAllWindows()

    def compute_affine_transform(self):

        # create arrays for affine transform
        points_px = []
        points_m = []
        # append corners
        for key, value in self.corners_px_dict.items():
            if value is not None and key in self.corners_m_dict and self.corners_m_dict[key] is not None:
                points_px.append(value)
                points_m.append(self.corners_m_dict[key])
        
        # append keys
        for key, value in self.bolts_px_dict.items():
            if value is not None and key in self.bolts_m_dict and self.bolts_m_dict[key] is not None:
                points_px.append(value)
                points_m.append(self.bolts_m_dict[key])

        points_px = np.array(points_px)
        points_m = np.array(points_m)

        X = self.pad(points_px)
        Y = self.pad(points_m)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        self.coord_transform = lambda x: self.unpad(np.dot(self.pad(x), A))
        
        A_inv = np.linalg.solve(A.T.dot(A), A.T)
        self.coord_transform_inv = lambda x: self.unpad(np.dot(self.pad(x), A_inv))
        
        # max_error = np.abs(points_m - self.pixels_to_meters(points_px)).max()
        max_error = np.linalg.norm(points_m - self.pixels_to_meters(points_px), axis=1).max()
        mean_error = np.linalg.norm(points_m - self.pixels_to_meters(points_px), axis=1).mean()
        
        if max_error > 0.02:
            print("[red]Max error in work surface position is: " + str(max_error) +"[/red]")
        
        if self.debug:
            # print("points_px", points_px)
            # print("points_m", points_m)
            
            # print("Target:", points_m)
            # print("Result:", self.pixels_to_meters(points_px))
            print("affine max_error", max_error)
            print("affine mean_error", mean_error)

            print("self.meters_to_pixels", self.meters_to_pixels(np.array([0.0, 0.0])))
            print("and back...", self.pixels_to_meters(self.meters_to_pixels(np.array([0.0, 0.0]))))
        
        
    def estimate_corners_using_transform(self):
        for key, value in self.corners_m_dict.items():
            corner_in_meters = self.meters_to_pixels(np.array(value))
            self.corners_px_dict[key] = corner_in_meters
            if corner_in_meters[0] > self.img_width or corner_in_meters[0] < 0 \
                or corner_in_meters[1] > self.img_height or corner_in_meters[1] < 0:
                print("[red]Corner estimate is out of bounds! " + str(corner_in_meters[0]) + ", " + str(corner_in_meters[1]) + "[/red]")
        
        if self.debug:
            print("self.corners_px_dict", self.corners_px_dict)
        
    def draw_corners_and_circles(self, img, show=False):
        # draw stuff on image
        if self.circles is not None:
            # draw all detections in green
            for (x, y, r) in self.circles:
                # Draw the circle in the output image
                cv2.circle(img, (int(x), int(y)), int(r), (255,0,0), 3)
                # Draw a cross in the output image
                cv2.drawMarker(img, (int(x), int(y)), color=[255,0,0],
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                    line_type=8)
                
            for (x, y, r) in self.circles_ignoring:
                # Draw the circle in the output image
                cv2.circle(img, (int(x), int(y)), int(r), (74, 74, 115), 3)
                # Draw a cross in the output image
                cv2.drawMarker(img, (int(x), int(y)), color=[74, 74, 115],
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                    line_type=8)
        
            # for bolts and calibration mounts
            for key in self.bolts_px_dict:
                if self.bolts_px_dict[key] is not None:
                    if len(self.bolts_px_dict[key]) == 3:
                        x, y, r = self.bolts_px_dict[key]
                    else:
                        x, y = self.bolts_px_dict[key]
                        r = None
                    if r is not None:
                        cv2.circle(img, (int(x), int(y)), int(r), (0,255,0), 3)
                    # Draw a cross in the output image
                    cv2.drawMarker(img, (int(x), int(y)), color=[0,255,0],
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2,
                        line_type=8)
                    
                    cv2.putText(img, key, (int(x)-100, int(y)-20), self.font_face, self.font_scale, [0,255,0], self.font_thickness, cv2.LINE_AA)

            img_mask = np.zeros_like(img)
            # draw borders to ignore
            red_color =  (0, 0, 255)
            cv2.rectangle(img_mask, (0, 0), (1450, self.border_width), red_color, -1) # top border
            cv2.rectangle(img_mask, (0, 1450), (1450, 1450 - self.border_width), red_color, -1) # bottom border
            cv2.rectangle(img_mask, (1450, 0), (1450- self.border_width, 1450), red_color, -1) # right border border
            cv2.rectangle(img_mask, (0, 0), (self.border_width, 1450), red_color, -1) # left border

            img = cv2.addWeighted(img, 0.75, img_mask, 0.25, 0)
        
        if self.debug:
            cv2.imwrite("saves/{date:%Y-%m-%d_%H:%M:%S}_bolts.png".format(date=datetime.datetime.now()), scale_img(img))

        if show:
            cv2.imshow("1", scale_img(img))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def pixels_to_meters(self, coords, depth=None):
        if self.coord_transform is None:
            print("[red]coord_transform is None![/red]")
            return coords
        
        if isinstance(coords, Polygon):
            # coords.exterior.coords[:-1] for non repeated list
            polygon_px_coords = np.asarray(list(coords.exterior.coords))
            polygon_coords = self.coord_transform(polygon_px_coords)
            if depth is not None:
                polygon_coords = np.pad(polygon_coords, [(0, 0), (0, 1)], mode='constant', constant_values=depth)
            
            return Polygon(polygon_coords)
            
        elif isinstance(coords, tuple) or len(coords.shape) == 1:
            # single coordinate pair.
            # todo: add depth option
            return self.coord_transform(np.array([coords]))[0]
        else:
            # assume array of coordinate pairs.
            # Each row contains a coordinate (x, y) pair
            # todo: add depth option
            return self.coord_transform(coords)
        
    def meters_to_pixels(self, coords):
        if self.coord_transform is None:
            print("[red]coord_transform is None![/red]")
            return coords
        # todo: deal with depth
        if isinstance(coords, Polygon):
            polygon_coords = np.asarray(list(coords.exterior.coords))
            polygon_px_coords = self.coord_transform_inv(polygon_coords)
            return Polygon(polygon_px_coords)
        elif isinstance(coords, tuple) or len(coords.shape) == 1:
            # single coordinate pair.
            return self.coord_transform_inv(np.array([coords]))[0]
        else:
            # assume array of coordinate pairs.
            # Each row contains a coordinate (x, y) pair
            return self.coord_transform_inv(coords)
        
if __name__ == '__main__':
    from config import load_config

    config = load_config()


    img = cv2.imread(os.path.expanduser("~/datasets2/reconcycle/2022-12-05_work_surface/frame0000.jpg"))
    # img = cv2.imread("data_full/dlc/dlc_work_surface_jsi_05-07-2021/labeled-data/raw_work_surface_jsi_08-07-2021/img000.png")

    print("img.shape", img.shape)
    border_width = config.basler.work_surface_ignore_border_width
    print("border_width", border_width)
    # border_width = 100
    # print("border_width", border_width)

    work_surface_det2 = WorkSurfaceDetection(img, border_width, debug=True)
    
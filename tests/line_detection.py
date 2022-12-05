import sys
import os
sys.path.append(os.path.dirname("/root/vision-pipeline/"))
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from rich import print
from helpers import scale_img

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


if __name__ == '__main__':
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    font_thickness = 1
    
    img = cv2.imread("data_full/2022-12-05_work_surface/frame0000.jpg")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ret, thresh1 = cv2.threshold(gray,127,255, cv2.THRESH_BINARY)
    
    # thresh1 = cv2.bitwise_not(thresh1)
    
    edges = cv2.Canny(gray, threshold1=50, threshold2=200, apertureSize = 3)
    
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    edges_rgb1 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
    edges_rgb2 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
    edges_rgb3 = cv2.cvtColor(edges.copy(), cv2.COLOR_GRAY2BGR)
    # plt.imshow(edges,cmap = 'gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
    
    for line_unsqueezed in lines:
        line = line_unsqueezed[0]
        cv2.line(edges_rgb1, (line[0], line[1]), (line[2],line[3]), (0,0,255), 6)
    
    bundler = HoughBundler(min_distance=10,min_angle=5)
    lines_processed = bundler.process_lines(lines)
    lines_processed = lines_processed.reshape(-1, 2, 2)
    
    # print("lines.shape", lines_processed.shape)
    for line in lines_processed:
        cv2.line(edges_rgb2, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
       
    norm = np.linalg.norm(lines_processed[:, 0] - lines_processed[:, 1], axis=1)
    print("norm", norm)
    
    long_lines = []
    for i, line in enumerate(lines_processed):
        if norm[i] > img.shape[0] *0.8: # should be at least 80% of the image long
            long_lines.append(lines_processed[i])
    long_lines = np.array(long_lines)

    print("long_lines", long_lines.shape, long_lines)
    
    # for line in long_lines:
        # cv2.line(edges_rgb3, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
        
    # todo: get left most/right most... line, and mask everything up to that line.
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
        return (line[1][0] + line[1][1])/2
    
    # order the vertical_lines and vertical lines
    vertical_lines.sort(key=sorting_vertical)
    horizontal_lines.sort(key=sorting_horizontal, reverse=True) # reverse because of world coords
    
    for line in vertical_lines:
        cv2.line(edges_rgb3, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
    
    for line in horizontal_lines:
        cv2.line(edges_rgb3, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,255,0), 6)
    
    left_line, right_line, bottom_line, top_line = None, None, None, None

    if len(vertical_lines) > 0:
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]
    if len(horizontal_lines) > 0:
        bottom_line = horizontal_lines[0]
        top_line = horizontal_lines[-1]

    for text, line in [("left_line", left_line), ("right_line", right_line), ("bottom_line", bottom_line), ("top_line", top_line)]:
        if line is not None:
            center = center_point(line)
            print(center)
            cv2.putText(edges_rgb3, text, center, font_face, font_scale, (0,0,255), font_thickness, cv2.LINE_AA)
    
    
    
    cv2.imshow("canny", scale_img(edges))
    cv2.imshow("houghlines", scale_img(edges_rgb1))
    cv2.imshow("bundled lines", scale_img(edges_rgb2))
    cv2.imshow("long lines only", scale_img(edges_rgb3))
    cv2.imshow("img", scale_img(img))
    cv2.waitKey()
    cv2.destroyAllWindows()
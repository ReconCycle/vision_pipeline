# Calculate oriented bounding boxes for sets of points
# and for binary mask images/label images
# Volker.Hilsenstein@monash.edu
# This is based on the following stackoverflow answer
# by stackoverflow user Quasiomondo (Mario Klingemann)
# https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy

import numpy as np
import skimage.morphology
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import cv2
import os
import time


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

# def quaternion_multiply(quaternion1, quaternion0):
#         x0, y0, z0, w0 = quaternion0
#         x1, y1, z1, w1 = quaternion1

#         # This quat is W X Y Z
#         out_quat = np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
#                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
#                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
#                      x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)

#         # This quat is X Y Z W
#         out = out_quat[[1,2,3,0]]
#         return out

# def better_quaternion(obb_corners):
#     # for detection in detections:
#     corners = np.array(obb_corners)
#     distances = []
#     # Logger.loginfo("{}".format(corners))
    
#     # sanity check
#     if corners.shape != (4, 2):
#         return None

#     first_corner = corners[0]

#     for ic, corner in enumerate(corners):
#         distances.append(np.linalg.norm(corner - first_corner))
#     # Logger.loginfo("Distances: {}".format(distances))
#     distances = np.array(distances)
#     idx_edge = distances.argsort()[-2]
#     # Logger.loginfo("Index of edge: {}".format(idx_edge))
#     second_corner = corners[idx_edge]

#     highest_y = np.argmax([first_corner[1], second_corner[1]])
#     if highest_y == 0:
#         vector_1 = first_corner - second_corner
#     elif highest_y == 1:
#         vector_1 = second_corner - first_corner

#     unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
#     unit_vector_2 = np.array([0, 1])
#     # Logger.loginfo("Vectors: {}, {}".format(vector_1, unit_vector_2))

#     angle = (np.arctan2(unit_vector_1[1], unit_vector_1[0]) -
#                 np.arctan2(unit_vector_2[1], unit_vector_2[0]))


#     # If angle is too negative, add 180 degrees
#     if (angle * 180 / np.pi) < -30:
#         angle = angle + np.pi
#     # Logger.loginfo("Angle: {}".format(angle * 180 / np.pi))

#     # Below code works but z-axis is incorrect, should be rotated by 180 degs
#     angle = -angle

#     obb_rot_quat = np.concatenate((np.sin(angle/2)*np.array([0,0,1]),
#                                 np.array([np.cos(angle/2)])))

#     #rot_quat = np.concatenate((np.sin(angle/2)*np.array([0,0,-1]),
#     #                           np.array([-np.cos(angle/2)])))

#     #Rotate around x-axis by 180 degs
#     obb_rot_quat = quaternion_multiply(np.array([0,1,0,0]), obb_rot_quat)

#     #Rotate around z-axis by 180 degs
#     #obb_rot_quat = quaternion_multiply(np.array([0,0,1,0]), obb_rot_quat)

#     # new_obb_rot_quat = obb_rot_quat.tolist()
#     # detection.obb_rot_quat = np.array([[1,0,0,0]]).tolist()
    
#     return obb_rot_quat
    

def get_obb_using_eig(points, calcconvexhull=True):
    """ given a set of points, calculate the oriented bounding 
    box. 
    
    Parameters:
    points: numpy array of point coordinates with shape (n,2)
            where n is the number of points
    calcconvexhull: boolean, calculate the convex hull of the 
            points before calculating the bounding box. You typically
            want to do that unless you know you are passing in a convex
            point set
    Output:
        tuple of corners, centre
    """

    if points.size == 0 or len(points) < 4:
        return None, None, None

    # start_convexhull = time.time()

    if calcconvexhull:
        try:
            _ch = ConvexHull(points)
        except:
            # something went wrong
            return None, None, None
        points = _ch.points[_ch.vertices]
        
    # fps_convexhull = 1.0 / (time.time() - start_convexhull)
    # print("fps_convexhull", fps_convexhull)
    
    #? could we somehow use max and min points instead?
    # start_maths = time.time()

    #? maybe we can use an approximation for eig instead.
    cov_points = np.cov(points,y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(cov_points)
    tvect = np.transpose(vect)

    #use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    points_rotated = np.dot(points,np.linalg.inv(tvect))
    # get the minimum and maximum x and y 
    mina = np.min(points_rotated,axis=0)
    maxa = np.max(points_rotated,axis=0)
    diff = (maxa - mina)*0.5
    # the centre is just half way between the min and max xy
    center = mina + diff

    # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]]])
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the center back
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)

    # change to ints, and change order
    corners = np.round(corners).astype(int)
    center = np.round(center).astype(int)
    # corners[:, 0], corners[:, 1] = corners[:, 1], corners[:, 0].copy()
    # center[0], center[1] = center[1], center[0].copy()

    # convert 2d rotation matrix to a 3d rotation matrix the rotation is around the z-axis
    # the upper left corner of the 3d rotation matrix is the 2d rotation matrix
    rot_matrix = np.identity(3)
    rot_matrix[:2, :2] = tvect

    rotation_obj = Rotation.from_matrix(rot_matrix)
    rot_quat = rotation_obj.as_quat()
    # degrees = rotation_obj.as_euler('xyz', degrees=True)
    # print("degrees", degrees)
    
    # fps_maths = 1.0 / (time.time() - start_maths)
    # print("fps_maths", fps_maths)

    return corners, center, rot_quat


def get_obb_using_cv(contour, img=None):

    if contour is None or len(contour) < 4:
        return None, None, None
    
    # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(points)
    rect = cv2.minAreaRect(contour) 
    box = np.int0(cv2.boxPoints(rect))
    center = np.int0(rect[0])
    changing_width, changing_height = box_w_h(box)
    changing_rot = rect[2]
    
    if changing_height > changing_width:
        correct_rot = changing_rot
    else:
        correct_rot = changing_rot - 90

    correct_height = changing_height
    correct_width = changing_width
    if changing_height < changing_width:
        correct_height = changing_width
        correct_width = changing_height
    
    # ! rotation is a bit funny, and we should correct for it. see here:
    # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
    
    # clip so that the box is still in the bounds of the img
    if img is not None:
        # invert img.shape because points are x, y
        box = clip_box_to_img_shape(box, img.shape)
        # box = np.clip(box, a_min=np.asarray([0, 0]), a_max=np.asarray(img.shape[:2])[::-1] - 1) 
    
    return box, center, correct_rot

def box_w_h(box):
    return np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])

def clip_box_to_img_shape(box, img_shape):
    box = np.clip(box, a_min=np.asarray([0, 0]), a_max=np.asarray(img_shape[:2])[::-1] - 1) 
    return box


def get_obb_from_contour(contour, img=None):
    """ given a binary mask, calculate the oriented 
    bounding box of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling :func: `get_obb_from_points`

    Parameters:
        mask_im: binary numpy array

    """
    
    # https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
    corners, center, rot = get_obb_using_cv(contour, img)
    # corners, center, rot_quat =  get_obb_using_eig(contour)

    # better_rot_quat = rot_quat
    # if corners is not None:
    #     better_rot_quat = better_quaternion(corners)
    return corners, center, rot

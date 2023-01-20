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
import tf

def obb_px_to_quat(px):
    p1 = px[0]
    p2 = px[1]
    p3 = px[2]
    p4 = px[3]

    #p1 = np.array([px[0],px[1]])
    #p2 = np.array([px[2],px[3]])
    #p3 = np.array([px[4],px[5]])
    #p4 = np.array([px[6],px[7]])
    
    d_edge_1 = np.linalg.norm(p2 - p1)
    
    d_edge_2 = np.linalg.norm(p4 - p1)
    
    if d_edge_1 > d_edge_2:
        long_edge = p1,p2
        short_edge = p1, p4
    else:
        long_edge = p1,p4
        short_edge = p1,p2
    
    # Assert that long_edge[0,0] is BIGGER than long_edge[1,0] (That one X is always bigger
    if long_edge[0][0] < long_edge[1][0]:
        nl = long_edge[1], long_edge[0]
        long_edge = nl
    
    # We get the angle of the LONG edge.
    a = long_edge[0][0] - long_edge[1][0]
    b = long_edge[0][1] - long_edge[1][1]
    
    
    angle = np.arctan2(b,a)
    #ang = 180*angle / np.pi
    #rospy.loginfo("angle:{}".format(ang))
    
    rot = tf.transformations.quaternion_from_euler(np.pi, 0, angle)
    #rot = rot[[3,0,1,2]]
    
    
    #rospy.loginfo("D1:{}".format(d_edge_1))
    #rospy.loginfo("D2:{}".format(d_edge_2))
    return rot

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


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
    
    #  rotation is a bit funny, and we correct for it. see here:
    # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
    if changing_height > changing_width:
        correct_rot = changing_rot
    else:
        correct_rot = changing_rot - 90

    correct_height = changing_height
    correct_width = changing_width
    if changing_height < changing_width:
        correct_height = changing_width
        correct_width = changing_height
    
    # if the width or height of the rectangle is 0, then we return None
    if np.isclose(correct_height, 0.0) or np.isclose(correct_width, 0.0):
        return None, None, None

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

def get_obb_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        # get the contour with the largest area. Assume this is the one containing our object
        cnt = max(cnts, key = cv2.contourArea)
        mask_contour = np.squeeze(cnt)
        
        return get_obb_from_contour(mask_contour)

    return None, None, None

# def get_obb_using_eig(points, calcconvexhull=True):
#     """ given a set of points, calculate the oriented bounding 
#     box. 
    
#     Parameters:
#     points: numpy array of point coordinates with shape (n,2)
#             where n is the number of points
#     calcconvexhull: boolean, calculate the convex hull of the 
#             points before calculating the bounding box. You typically
#             want to do that unless you know you are passing in a convex
#             point set
#     Output:
#         tuple of corners, centre
#     """

#     if points.size == 0 or len(points) < 4:
#         return None, None, None

#     # start_convexhull = time.time()

#     if calcconvexhull:
#         try:
#             _ch = ConvexHull(points)
#         except:
#             # something went wrong
#             return None, None, None
#         points = _ch.points[_ch.vertices]
        
#     # fps_convexhull = 1.0 / (time.time() - start_convexhull)
#     # print("fps_convexhull", fps_convexhull)
    
#     #? could we somehow use max and min points instead?
#     # start_maths = time.time()

#     #? maybe we can use an approximation for eig instead.
#     cov_points = np.cov(points,y = None,rowvar = 0,bias = 1)
#     v, vect = np.linalg.eig(cov_points)
#     tvect = np.transpose(vect)

#     #use the inverse of the eigenvectors as a rotation matrix and
#     # rotate the points so they align with the x and y axes
#     points_rotated = np.dot(points,np.linalg.inv(tvect))
#     # get the minimum and maximum x and y 
#     mina = np.min(points_rotated,axis=0)
#     maxa = np.max(points_rotated,axis=0)
#     diff = (maxa - mina)*0.5
#     # the centre is just half way between the min and max xy
#     center = mina + diff

#     # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
#     corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]]])
#     # use the the eigenvectors as a rotation matrix and
#     # rotate the corners and the center back
#     corners = np.dot(corners,tvect)
#     center = np.dot(center,tvect)

#     # change to ints, and change order
#     corners = np.round(corners).astype(int)
#     center = np.round(center).astype(int)
#     # corners[:, 0], corners[:, 1] = corners[:, 1], corners[:, 0].copy()
#     # center[0], center[1] = center[1], center[0].copy()

#     # convert 2d rotation matrix to a 3d rotation matrix the rotation is around the z-axis
#     # the upper left corner of the 3d rotation matrix is the 2d rotation matrix
#     rot_matrix = np.identity(3)
#     rot_matrix[:2, :2] = tvect

#     rotation_obj = Rotation.from_matrix(rot_matrix)
#     rot_quat = rotation_obj.as_quat()
#     # degrees = rotation_obj.as_euler('xyz', degrees=True)
#     # print("degrees", degrees)
    
#     # fps_maths = 1.0 / (time.time() - start_maths)
#     # print("fps_maths", fps_maths)

#     return corners, center, rot_quat
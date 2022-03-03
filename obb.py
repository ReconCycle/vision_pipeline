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


def get_obb_using_eig(contour, calcconvexhull=True):
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

    points = contour.squeeze()

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


def get_obb_using_cv(contour):
    
    # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
    rect = cv2.minAreaRect(contour)
    box = np.int0(cv2.boxPoints(rect))
    center = np.int0(rect[0])
    rot = rect[2]
    rot_quat = Rotation.from_euler('z', rot, degrees=True).as_quat()
    
    return box, center, rot_quat

def get_obb_from_mask(mask_im):
    """ given a binary mask, calculate the oriented 
    bounding box of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling :func: `get_obb_from_points`

    Parameters:
        mask_im: binary numpy array

    """
    
    # mask = (mask_im[:,:, 0] == 1).astype("uint8")
    mask = mask_im.astype("uint8")
    
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # erosion followed by dilation
    
    # https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1) # maybe applying approximation is good? was: cv2.CHAIN_APPROX_SIMPLE
    
    if len(cnts) > 0:
        return get_obb_using_cv(cnts[0])
        # return get_obb_using_eig(cnts[0])

    return None, None, None
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

global last_quat
last_quat = Rotation([0,0,0,1]).as_quat()

def get_obb_from_points(points, calcconvexhull=True):
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

    if calcconvexhull:
        try:
            _ch = ConvexHull(points)
        except:
            # something went wrong
            return None, None, None
        points = _ch.points[_ch.vertices]

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
    # TODO this can be made nicer
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the center back
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)

    # change to ints, and change order
    corners = np.round(corners).astype(int)
    center = np.round(center).astype(int)
    corners[:, 0], corners[:, 1] = corners[:, 1], corners[:, 0].copy()
    center[0], center[1] = center[1], center[0].copy()

    # convert 2d rotation matrix to a 3d rotation matrix the rotation is around the z-axis
    # the upper left corner of the 3d rotation matrix is the 2d rotation matrix
    rot_matrix = np.identity(3)
    rot_matrix[:2, :2] = tvect

    rotation_obj = Rotation.from_matrix(rot_matrix)
    rot_quat = rotation_obj.as_quat()
    global last_quat
    print(last_quat)
    if np.dot(last_quat,rot_quat)<0:
        print("rotate")
        rot_quat = -rot_quat
    last_quat = rot_quat
    # degrees = rotation_obj.as_euler('xyz', degrees=True)
    # print("degrees", degrees)

    return corners, center, rot_quat

def get_obb_from_labelim(label_im, labels=None):
    """ given a label image, calculate the oriented 
    bounding box of each connected component with 
    label in labels. If labels is None, all labels > 0
    will be analyzed.

    Parameters:
        label_im: numpy array with labelled connected components (integer)

    Output:
        obbs: dictionary of oriented bounding boxes. The dictionary 
        keys correspond to the respective labels
    """
    if labels is None:
        labels = set(np.unique(label_im)) - {0}
    results = {}
    for label in labels:
        results[label] = get_obb_from_mask(label_im == label)
    return results

def get_obb_from_mask(mask_im):
    """ given a binary mask, calculate the oriented 
    bounding box of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling :func: `get_obb_from_points`

    Parameters:
        mask_im: binary numpy array

    """
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)

    # I seem to be getting problems with the above when the mask is really small.
    # Sometimes the mask contains only one point or points on a single line only.
    # The problems lie with the mask.
    # print("boundary_points", boundary_points.shape)
    # int_mask_im = mask_im.astype("uint8")
    # print("mask_im", np.count_nonzero(int_mask_im), int_mask_im.shape)
    # ret, thresh = cv2.threshold(int_mask_im, 0.1, 1, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # try using cv2.CHAIN_APPROX_SIMPLE
    # contours = np.asarray(contours).squeeze()
    # print("contours", contours.shape, type(contours))

    return get_obb_from_points(boundary_points)


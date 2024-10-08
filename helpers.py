import os
from matplotlib.pyplot import step
import regex
import natsort
import argparse
import cv2
import numpy as np
import dataclasses
from rich import print
import json
from torch import Tensor
import pyrealsense2
import re
import open3d as o3d
from pathlib import Path
import pickle
from PIL import Image as PILImage

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import IntEnum
import itertools

from shapely.geometry import LineString, Point, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.validation import explain_validity
import shapely


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_images(input_dir):
    if os.path.isdir(input_dir):
        images = [img for img in os.listdir(input_dir) if img.endswith(".png")]
        images.sort(key=lambda f: int(regex.sub('\D', '', f)))
        images = [os.path.join(input_dir, image) for image in images]
    elif os.path.isfile(input_dir):
        images = [input_dir]
    else:
        images = None
    return images


def get_images_realsense(input_dir):
    images_paths = None
    if os.path.isdir(input_dir):
        images_paths = [img for img in os.listdir(input_dir) if img.endswith(".png") or img.endswith(".npy")]
        images_paths = natsort.os_sorted(images_paths)
        images_paths = [os.path.join(input_dir, images_path) for images_path in images_paths]

        if len(images_paths) % 3 == 0:
            images_paths = np.array(images_paths).reshape((-1, 3))
        else:
            print("Error: number of images in directory not a multiple of 3!")
        
        for colour_img_p, depth_img_p, depth_colormap_p in images_paths:
            print("\nimg path:", colour_img_p)
            colour_img = cv2.imread(colour_img_p)
            depth_img = np.load(depth_img_p)
            depth_colormap = cv2.imread(depth_colormap_p)
            yield [colour_img, depth_img, depth_colormap, colour_img_p]


def camera_info_to_realsense_intrinsics(camera_info):
    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = camera_info.width
    _intrinsics.height = camera_info.height
    _intrinsics.ppx = camera_info.K[2]
    _intrinsics.ppy = camera_info.K[5]
    _intrinsics.fx = camera_info.K[0]
    _intrinsics.fy = camera_info.K[4]
    #_intrinsics.model = camera_info.distortion_model
    _intrinsics.model  = pyrealsense2.distortion.none
    _intrinsics.coeffs = [i for i in camera_info.D]
    
    return _intrinsics

def camera_info_to_o3d_intrinsics(camera_info):
    w, h = camera_info.width, camera_info.height
    fx, fy = camera_info.K[0], camera_info.K[4]
    ppx, ppy = camera_info.K[2], camera_info.K[5]

    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, ppx, ppy)
    return intrinsics

def img_to_camera_coords(x_y, depth, camera_info):
    _intrinsics = camera_info_to_realsense_intrinsics(camera_info)
    
    def pixels_to_meters(x_y):
        if isinstance(depth, np.ndarray):
            # print("depth.shape", depth.shape, "pos:", x_y[1], x_y[0])
            single_depth = depth[int(x_y[1]), int(x_y[0])]
        else:
            single_depth = depth
        if single_depth is None or single_depth == 0.0:
            # print("single_depth", single_depth)
            return None
        
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, x_y, single_depth)
        result = np.asarray(result)

        return result[0], result[1], result[2]
    
    def pixels_to_meters_of_arr(x_y):
        results = []
        for single_x_y in x_y:
            result = pixels_to_meters(single_x_y)
            if result is not None:
                results.append(result)

        return results
    
    if isinstance(x_y, Polygon):
        polygon_coords = np.asarray(list(x_y.exterior.coords))
        
        polygon_coords_m = pixels_to_meters_of_arr(polygon_coords)
        # polygon should have at least 4 coordinates
        if len(polygon_coords_m) < 4:
            print("len(polygon_coords)", len(polygon_coords))
            print("len(polygon_coords_m)", len(polygon_coords_m))
            return None
        return Polygon(polygon_coords_m)

    elif len(x_y.shape) == 2:
        # multiple pairs of x_y
        results_list = pixels_to_meters_of_arr(x_y)
        return np.array(results_list)
    
    else:
        # x, y = x_y
        # single x, y pair
        result = pixels_to_meters(x_y)
        if result is None:
            return None
        
        return np.asarray(result)
            


def scale_img(img):
    img_width = img.shape[1]
    img_height = img.shape[0]
    max_wh = np.max([img_width, img_height])

    scaled_max = 800

    if max_wh > scaled_max:
        scale_factor = scaled_max/max_wh
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        dim = (new_width, new_height)
    
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    else:
        resized = img
    
    return resized


COLOURS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

COLOURS2 = [(0, 255, 0), (255, 0, 255), (0, 127, 255), (255, 127, 0), (127, 191, 127), (185, 2, 62), (75, 1, 221), (20, 87, 110), (172, 109, 251), (228, 253, 9), (0, 255, 255), (94, 139, 5)]

COLOURS_BLUES = ((227, 5, 34),
                 (209, 59, 78),
                 (207, 107, 120),
                 (194, 138, 145))


def get_colour(j):
    colour_idx = (j * 5) % len(COLOURS)
    colour = np.array(COLOURS[colour_idx], dtype=np.float64)
    return colour


def get_colour_blue(j):
    colour_idx = j % len(COLOURS_BLUES)
    colour = np.array(COLOURS_BLUES[colour_idx], dtype=int)
    return colour


def img_grid(imgs, w=2, h=None, margin=0):
    if h is None and isinstance(w, int):
        h = int(np.ceil(len(imgs) / w))
    if w is None and isinstance(h, int):
        w = int(np.ceil(len(imgs) / h))
    n = w * h

    # Define the shape of the image to be replicated (all images should have the same shape)
    img_h, img_w, img_c = imgs[0].shape

    # Define the margins in x and y directions
    m_x = margin
    m_y = margin

    # Size of the full size image
    mat_x = img_w * w + m_x * (w - 1)
    mat_y = img_h * h + m_y * (h - 1)

    # Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
    img_matrix = np.zeros((mat_y, mat_x, img_c), np.uint8)
    img_matrix.fill(255)

    # Prepare an iterable with the right dimensions
    positions = itertools.product(range(h), range(w))

    for (y_i, x_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        img_matrix[y:y + img_h, x:x + img_w, :] = img

    # resized = cv2.resize(img_matrix, (mat_x // 3, mat_y // 3), interpolation=cv2.INTER_AREA)
    # compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    # cv2.imwrite(name, resized, compression_params)
    return img_matrix

def img_row(imgs, spacing=10):
    num_imgs = len(imgs)
    total_width = sum([img.size[0] for img in imgs]) + (num_imgs-1)*spacing
    max_height = max([img.size[1] for img in imgs])
    new_image = PILImage.new('RGB', (total_width, max_height), color=(255, 255, 255))

    past_width = 0
    for idx, img in enumerate(imgs):
        new_image.paste(img, (past_width, 0))
        past_width += img.size[0] + spacing
    
    return new_image

def add_angles(angle1, angle2, degrees=False):
    if degrees:
        angle1 = np.deg2rad(angle1)
        angle2 = np.deg2rad(angle2)
    # angles in range [-pi, pi), then output is again in range [-pi, pi)
    # angles in rad as input and output
    angle_new = (angle1 + angle2) % (2*np.pi)
    angle_new = np.where(angle_new > np.pi, angle_new - 2*np.pi, angle_new)

    if degrees:
        angle_new = np.rad2deg(angle_new)
    return angle_new


def circular_median(angles_list, degrees=False):
    smallest_sum_so_far = np.Inf
    index = None
    for i in np.arange(len(angles_list)):
        # sum this distances from point a[i] to all the other points.
        # the one with the smallest distance is the median
        sum = 0
        for j in np.arange(len(angles_list)):
            if i != j:
                # absolute difference of the angles
                sum += abs(add_angles(angles_list[i], -angles_list[j], degrees=degrees))

        if sum < smallest_sum_so_far:
            smallest_sum_so_far = sum
            index = i

    return index
    

def make_valid_poly(poly):
    if not poly.is_valid:
        # print(explain_validity(poly))
        poly = make_valid(poly)
        
        # we sometimes get a GeometryCollection, where the first item is a MultiPolygon
        idx_largest_poly = None
        len_largest_poly = -1
        if isinstance(poly, MultiPolygon) or isinstance(poly, GeometryCollection):
            for i in np.arange(len(poly.geoms)):
                if poly.geoms[i].area > len_largest_poly:
                    idx_largest_poly = i
                    len_largest_poly = poly.geoms[i].area

            poly = poly.geoms[idx_largest_poly]

        # repeat step because sometimes we have a multipolygon inside a geometry collection
        idx_largest_poly = None
        len_largest_poly = -1
        if isinstance(poly, MultiPolygon) or isinstance(poly, GeometryCollection):
            # print("[red]poly is GeometryCollection")
            for i in np.arange(len(poly.geoms)):                
                if poly.geoms[i].area > len_largest_poly:
                    idx_largest_poly = i
                    len_largest_poly = poly.geoms[i].area

            poly = poly.geoms[idx_largest_poly]

        # return a Polygon or None
        if not isinstance(poly, Polygon):
            print("[red]poly is of type"+ str(type(poly)) + " and not Polygon![/red]")
            if isinstance(poly, GeometryCollection):
                print(list(poly.geoms))
            poly = None
    
    return poly

def simplify_poly(poly, try_max_num_coords=20):
    for _ in np.arange(3):
        if len(poly.exterior.coords) > try_max_num_coords:
            poly = poly.simplify(tolerance=5.) # tolerance in pixels
        else:
            break

    return poly

def rotate_img(img, angle_deg):
    if angle_deg == 0:
        return img
    if angle_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print("[red]Provide rotation multiple of 90 degrees! [/red]")
        return img

def path(*sub_paths):
    """
    Joins paths together, removing extra slashes
    """
    # join path together
    path_str = ""
    for sub_path in sub_paths:
        path_str += "/" + sub_path
        
    # remove duplicate slashes
    path_str = re.sub('/+', '/', path_str)

    return path_str


def load_depth_data_from_filename(file_name_without_ext, images_dir, depth_rescaling_factor=1/1000):

    image_path = images_dir / Path(str(file_name_without_ext) + ".jpg")
    camera_info_path = images_dir / Path(str(file_name_without_ext) + "_camera_info.pickle")
    depth_path = images_dir / Path(str(file_name_without_ext) + "_depth.npy")
    image_viz_path = images_dir / Path(str(file_name_without_ext) + "_depth_viz.jpg")

    colour_img = cv2.imread(str(image_path))
    depth_vis_img = cv2.imread(str(image_viz_path))

    depth_img = np.load(depth_path)
    pickleFile = open(camera_info_path, 'rb')
    camera_info = pickle.load(pickleFile)

    depth_img = depth_img * depth_rescaling_factor

    return colour_img, depth_vis_img, depth_img, camera_info


def robust_minimum(data, trim_percentage=0.1):
    """
    Calculate a robust minimum by trimming a percentage of the smallest values.

    :param data: List or array of numbers.
    :param trim_percentage: Percentage of smallest values to trim (0 to 0.5).
    :return: Robust minimum value.
    """
    if len(data) == 0:
        return None
    
    # Ensure the trim_percentage is between 0 and 0.5
    if trim_percentage < 0 or trim_percentage > 0.5:
        raise ValueError("trim_percentage must be between 0 and 0.5")
    
    # Sort the data
    sorted_data = np.sort(data)
    
    # Calculate the number of values to trim
    n_trim = int(len(sorted_data) * trim_percentage)
    
    if n_trim < len(sorted_data):
        # Trim the smallest values
        trimmed_data = sorted_data[n_trim:]
        
        # Return the minimum of the remaining data
        return np.min(trimmed_data)
    else:
        return None


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    
        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)
    

class Struct(object):
    """
    Holds the configuration for anything you want it to.
    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)



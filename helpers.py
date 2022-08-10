import os
from matplotlib.pyplot import step
import regex
import natsort
import argparse
import cv2
import numpy as np
import dataclasses
from json import JSONEncoder
from torch import Tensor
from shapely.geometry import Polygon
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import IntEnum
import itertools


class Action(IntEnum):
    move = 0
    cut = 1
    lever = 2
    turn_over = 3
    remove_clip = 4


@dataclass
class Detection:
    id: Optional[int] = None
    label: Optional[IntEnum] = None
    
    score: Optional[float] = None
    box: Optional[np.ndarray] = None
    mask: Optional[Tensor] = None
    mask_contour: Optional[np.ndarray] = None
    mask_polygon: Optional[np.ndarray] = None
    
    obb_corners: Optional[np.ndarray] = None
    obb_center: Optional[np.ndarray] = None
    obb_rot_quat: Optional[np.ndarray] = None
    obb_corners_meters: Optional[np.ndarray] = None
    obb_center_meters: Optional[np.ndarray] = None
    
    tracking_id: Optional[int] = None
    tracking_score: Optional[float] = None
    tracking_box: Optional[np.ndarray] = None

@dataclass
class LeverAction:
    from_px = None
    to_px = None

    from_depth = None
    to_depth = None

    # in camera coords
    from_camera = None
    to_camera = None

    obb_px = None
    bb_camera= None

    pose_stamped = None


class EnhancedJSONEncoder(JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        
        # handle IntEnum to give the name instead of the int
        def map_intenum(obj):
            if isinstance(obj, IntEnum):
                # return {'name': obj.name, 'value': obj.value}
                return obj.name
            if isinstance(obj, dict):
                return {k: map_intenum(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [map_intenum(v) for v in obj]
            if dataclasses.is_dataclass(obj):
                dataclass_dict = dataclasses.asdict(obj)
                return {k: map_intenum(v) for k, v in dataclass_dict.items()}
        
            return obj
        
        o = map_intenum(o)
        return super(EnhancedJSONEncoder, self).iterencode(o, _one_shot)
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, Tensor):
            return None
        if isinstance(obj, Polygon):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)    


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


def scale_img(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
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

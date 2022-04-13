import os
import regex
import argparse
import cv2
import numpy as np
from torch import Tensor
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Detection:
    id: Optional[int] = None
    label: Optional[str] = None
    
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

def scale_img(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


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

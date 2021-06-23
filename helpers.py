import os
import regex
import argparse

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
#%%

import sys
import os

# print("sys.path", sys.path)

import matplotlib
import matplotlib.pyplot as plt

import natsort
# import seaborn as sns
import datetime
import json
import warnings
from pathlib import Path
import random
import base64
from io import BytesIO
import cv2
# import vision_pipeline.obb
# import imagesize
# from scipy import ndimage
import natsort
from PIL import Image as PILImage
# from PIL import ImageDraw, ImageFilter
import numpy as np
from tqdm import tqdm
# from shapely.geometry import Polygon
# from rich import print
# from types import SimpleNamespace
import pickle
import imutils

# ros package
from context_action_framework.types import Detection, Label, Module, Camera, detections_to_ros, detections_to_py
from sensor_msgs.msg import Image, CameraInfo # CameraInfo needed for pickle

from context_action_framework.srv import VisionDetection, VisionDetectionResponse, VisionDetectionRequest, ProcessImg, ProcessImgResponse

from context_action_framework.graph_relations import GraphRelations

from cv_bridge import CvBridge
import rospy


rospy.init_node("test_node")

#%%

def call_process_img(img):
    timeout = 3 # 2 second timeout
    rospy.wait_for_service('vision/basler/process_img', timeout)

    imgmsg = CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    try:
        process_img = rospy.ServiceProxy('vision/basler/process_img', ProcessImg)
        response = process_img(imgmsg)
        detections = detections_to_py(response.detections)
        labelled_img = CvBridge().imgmsg_to_cv2(response.labelled_image, desired_encoding='passthrough')

        cropped_img = None
        if response.cropped_image is not None and response.cropped_image.encoding.strip() != "":

            cropped_img = CvBridge().imgmsg_to_cv2(response.cropped_image, desired_encoding='passthrough')
        
        return response.success, detections, labelled_img, cropped_img
        # return response
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

#%%

base_path = Path("~/datasets2/reconcycle/2024-07-10_jsi_devices/2024-07-12_basler_firealarms/").expanduser()

img_paths = base_path.glob("*.jpg")

# alternatively from list:
# img_paths = [
#     base_path / Path("2024-07-12_basler_0002.jpg"),
# ]

img_paths = natsort.os_sorted(img_paths)

# img_path = os.path.expanduser("~/saves/2024-04-22_09:33:47_basler/0001.jpg")
# img_path = os.path.expanduser("~/datasets2/reconcycle/2023-08-01_basler_hca_backs/0001.jpg")
# img_path = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted/firealarm_back_01/00_template_0039.jpg")
# img_path = os.path.expanduser("~/datasets2/reconcycle/2023-12-04_hcas_fire_alarms_sorted/firealarm_back_03.1/0182.jpg")

save_path = Path("~/vision_pipeline/saves/{date:%Y-%m-%d_%H:%M:%S}_process_imgs".format(date=datetime.datetime.now())).expanduser()


# check if file path is empty
if not os.path.exists(save_path):
    print("making folder", save_path)
    os.makedirs(save_path)
else:
    raise ValueError(f"folder already exists! {save_path}")


for img_path in img_paths:
    img = cv2.imread(os.path.expanduser(img_path))
    img = imutils.resize(img, width=1450, height=1450)

    print("img.shape", img.shape, type(img))
    # display(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    success, detections, labelled_img, cropped_img = call_process_img(img)

    plt.imshow(labelled_img)
    plt.show()
    
    
    cv2.imwrite(str(save_path / img_path.name), labelled_img)

    # display(PILImage.fromarray(cv2.cvtColor(labelled_img, cv2.COLOR_BGR2RGB)))
    # display(PILImage.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)))

    print("detections", detections)
# %%

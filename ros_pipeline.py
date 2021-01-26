import sys
import os
import cv2
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
from pipeline_v2 import Pipeline

if __name__ == '__main__':

    # 1. Create ROS camera node and feed images from camera to camera node
    # 2. Create ROS camera labelled node and ROS data node -> feed images and data from pipeline

    camera_publisher = ROSPublisher(node_name="/camera/image_color")
    labelled_img_publisher = ROSPublisher(node_name="/vision_pipeline/image_color")
    pipeline = Pipeline()

    def img_from_camera(img):
        print("image from camera received")
        labelled_img = pipeline.process_img(img)

        camera_publisher.publish_img(img)
        labelled_img_publisher.publish_img(labelled_img)

    camera_feed(callback=img_from_camera)



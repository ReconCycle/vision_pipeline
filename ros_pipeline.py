import sys
import os
import cv2
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
from pipeline_v2 import Pipeline
import json

if __name__ == '__main__':

    # 1. Create ROS camera node and feed images from camera to camera node
    # 2. Create ROS camera labelled node and ROS data node -> feed images and data from pipeline

    camera_publisher = ROSPublisher(topic_name="/camera/image_color", msg_images=True)
    labelled_img_publisher = ROSPublisher(topic_name="/vision_pipeline/image_color", msg_images=True)
    data_publisher = ROSPublisher(topic_name="/vision_pipeline/data", msg_images=False)
    pipeline = Pipeline()

    def img_from_camera(img):
        print("image from camera received")
        labelled_img, detections = pipeline.process_img(img)

        json_detections = json.dumps(detections)

        # print("json_data", json_detections)

        camera_publisher.publish_img(img)
        labelled_img_publisher.publish_img(labelled_img)
        data_publisher.publish_text(json_detections)

    camera_feed(callback=img_from_camera)



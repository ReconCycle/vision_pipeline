import sys
import os
import cv2
import rospy
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
from ros_service import ROSService
from pipeline_v2 import Pipeline
import json
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("publish_continuously", help="Publish labelled images and detections continuously otherwise create a service.", nargs='?', type=bool, default=False)
    args = parser.parse_args()

    # 1. Create ROS camera node and feed images from camera to camera node
    # 2. Create ROS camera labelled node and ROS data node -> feed images and data from pipeline

    rospy.init_node("vision_pipeline")

    camera_publisher = ROSPublisher(topic_name="/camera/image_color", msg_images=True)

    # either create a publisher topic for the labelled images and detections or create a service
    if args.publish_continuously:
        labelled_img_publisher = ROSPublisher(topic_name="/vision_pipeline/image_color", msg_images=True)
        data_publisher = ROSPublisher(topic_name="/vision_pipeline/data", msg_images=False)
        pipeline = Pipeline()
    else:
        ros_service = ROSService(service_name="get_detection", camera_topic="/camera/image_color", also_publish=True)

    #? the following might need to run on a separate thread!

    def img_from_camera_callback(img):
        print("image from camera received")

        camera_publisher.publish_img(img)

        if args.publish_continuously:
            labelled_img, detections = pipeline.process_img(img)
            json_detections = json.dumps(detections)

            labelled_img_publisher.publish_img(labelled_img)
            data_publisher.publish_text(json_detections)

    camera_feed(callback=img_from_camera_callback) # this stops python from quitting since we are looping here

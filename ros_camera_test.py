import sys
import os
import cv2
import rospy
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
from pipeline_v2 import Pipeline

if __name__ == '__main__':

    rospy.init_node("vision_pipeline")
    camera_publisher = ROSPublisher(topic_name="/camera/image_color")

    def img_from_camera(img):
        print("image from camera received")

        if img is not None:
            print("img not none")
        else:
            print("img is none")

        camera_publisher.publish_img(img)

    camera_feed(undistort=True, callback=img_from_camera)



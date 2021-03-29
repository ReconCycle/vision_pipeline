import sys
import os
import cv2
import rospy
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
from pipeline_v2 import Pipeline
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("save", help="Save images to folder..", nargs='?', type=bool, default=False)
    args = parser.parse_args()

    rospy.init_node("vision_pipeline")
    camera_publisher = ROSPublisher(topic_name="/camera/image_color")

    img_count = 0
    def img_from_camera(img):
        global img_count
        print("image from camera received")

        if img is None:
            print("img is none")

        camera_publisher.publish_img(img)

        if args.save:
            save_file_path = os.path.join("./camera_images", str(img_count) + ".png")
            cv2.imwrite(save_file_path, img)
        
        img_count += 1

    camera_feed(undistort=True, callback=img_from_camera)



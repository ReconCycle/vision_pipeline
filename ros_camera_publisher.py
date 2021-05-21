import sys
import os
import cv2
import rospy
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("save", help="Save images to folder..", nargs='?', type=bool, default=False)
    parser.add_argument("camera_topic", help="The name of the camera topic to subscribe to", nargs='?', type=str, default="/camera/image_color")
    parser.add_argument("node_name", help="The name of the node", nargs='?', type=str, default="camera")
    args = parser.parse_args()

    print("\nsave:", args.save)
    print("camera_topic:", args.camera_topic)
    print("node_name:", args.node_name, "\n")

    rospy.init_node(args.node_name)
    camera_publisher = ROSPublisher(topic_name=args.camera_topic)

    save_folder = "./camera_images"
    if args.save and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_count = 0
    def img_from_camera(img):
        global img_count
        print("image from camera received")

        if img is None:
            print("img is none")

        camera_publisher.publish_img(img)

        if args.save:
            save_file_path = os.path.join(save_folder, str(img_count) + ".png")
            cv2.imwrite(save_file_path, img)
        
        img_count += 1

    camera_feed(undistort=True, callback=img_from_camera)



import sys
import os
import shutil
import cv2
import rospy
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
import argparse
import helpers

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_topic", help="The name of the camera topic to subscribe to", nargs='?', type=str, default="/camera/image_color")
    parser.add_argument("--node_name", help="The name of the node", nargs='?', type=str, default="camera")
    parser.add_argument("--save", help="Save images to folder..", nargs='?', type=helpers.str2bool, default=False)
    parser.add_argument("--undistort", help="Use the calibration file to undistort the image", nargs='?', type=helpers.str2bool, default=True)
    parser.add_argument("--fps", help="Set fps of camera", nargs='?', type=float, default=None)
    args = parser.parse_args()

    print("\ncamera_topic:", args.camera_topic)
    print("node_name:", args.node_name)
    print("save:", args.save)
    print("undistort:", args.undistort)
    print("fps:", args.fps, "\n")

    rospy.init_node(args.node_name)
    camera_publisher = ROSPublisher(topic_name=args.camera_topic)

    save_folder = "./camera_images"
    if args.save and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        shutil.rmtree(save_folder)
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
            print("writing image:", save_file_path)
            cv2.imwrite(save_file_path, img)
        
        img_count += 1

    camera_feed(undistort=args.undistort, fps=args.fps, callback=img_from_camera)



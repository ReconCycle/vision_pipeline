import sys
import os
import shutil
import cv2
import time

ros_available = True
try:
    import rospy
    from ros_publisher import ROSPublisher
except ModuleNotFoundError:
    ros_available = False
    pass

import argparse
import helpers

if __name__ == '__main__':
    """Publish images to ROS node from a folder of images. NOT FROM CAMERA.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_topic", help="The name of the camera topic to subscribe to", nargs='?', type=str, default="/camera/image_color")
    parser.add_argument("--node_name", help="The name of the node", nargs='?', type=str, default="camera")
    parser.add_argument("--img_folder", help="Load images from folder..", nargs='?', type=str, default="./folder")
    parser.add_argument("--undistort", help="Use the calibration file to undistort the image", nargs='?', type=helpers.str2bool, default=False)
    parser.add_argument("--fps", help="Set fps of camera", nargs='?', type=float, default=5.0)
    args = parser.parse_args()

    print("\ncamera_topic:", args.camera_topic)
    print("node_name:", args.node_name)
    print("img_folder:", args.img_folder)
    print("undistort:", args.undistort)
    print("fps:", args.fps, "\n")

    if ros_available:
        rospy.init_node(args.node_name)
        camera_publisher = ROSPublisher(topic_name=args.camera_topic)

    img_count = 1
    def publish_img(img):
        global img_count
        print("image from folder received")

        if img is None:
            print("img is none")

        if ros_available:
            camera_publisher.publish_img(img)
        
        img_count += 1
    
    if not os.path.exists(args.img_folder) or not os.path.isdir(args.img_folder):
        print("Image folder does not exist:", args.img_folder)
        sys.exit()
        
    imgs = helpers.get_images(args.img_folder)
    
    starttime = time.time()
    while True:
        for img_p in imgs:
            img = cv2.imread(img_p)
            publish_img(img)
            
            time_to_sleep = (1/args.fps) - ((time.time() - starttime) % (1/args.fps))
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

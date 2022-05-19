import sys
import os
import shutil
import cv2
from rich import print

ros_available = True
try:
    import rospy
    from ros_publisher import ROSPublisher
    from ros_vision_pipeline.msg import ColourDepth
    from cv_bridge import CvBridge
except ModuleNotFoundError:
    ros_available = False
    pass

from camera_feed import camera_feed
from camera_realsense_feed import RealsenseCamera
import argparse
import helpers

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_type", help="Which camera: basler/realsense", nargs='?', type=str, default="basler")
    parser.add_argument("--camera_topic", help="The name of the camera topic to subscribe to", nargs='?', type=str, default="basler")
    parser.add_argument("--node_name", help="The name of the node", nargs='?', type=str, default="basler")
    parser.add_argument("--save", help="Save images to folder..", nargs='?', type=helpers.str2bool, default=False)
    parser.add_argument("--undistort", help="Use the calibration file to undistort the image", nargs='?', type=helpers.str2bool, default=True)
    parser.add_argument("--fps", help="Set fps of camera", nargs='?', type=float, default=None)
    args = parser.parse_args()

    # set the camera_topic to realsense as well, if not set manually
    if args.camera_type == "realsense" and args.camera_topic == "basler":
        args.camera_topic = "realsense"

    if args.camera_type == "realsense" and args.node_name == "basler":
        args.node_name = "realsense"

    print("\ncamera_type:", args.camera_type)
    print("camera_topic:", args.camera_topic)
    print("node_name:", args.node_name)
    print("save:", args.save)
    print("undistort:", args.undistort)
    print("fps:", args.fps, "\n")

    if ros_available:
        rospy.init_node(args.node_name)
        br = CvBridge()
        if args.camera_type == "camera":
            camera_publisher = ROSPublisher(topic_name="/" + args.camera_topic + "/colour")
        elif args.camera_type == "realsense":
            colour_depth_pub = rospy.Publisher("/" + args.camera_topic + "/colour_depth", ColourDepth, queue_size=20)
            colour_realsense_publisher = ROSPublisher(topic_name="/" + args.camera_topic + "/colour")
            depth_realsense_publisher = ROSPublisher(topic_name="/" + args.camera_topic + "/depth")
            depthmap_realsense_publisher = ROSPublisher(topic_name="/" + args.camera_topic + "/depthmap")
    else:
        print("ROS NOT AVAILABLE.")

    save_folder_name = "./camera_images"
    save_folder = save_folder_name
    if args.save:
        folder_counter = 1
        path_exists = os.path.exists(save_folder)
        while(path_exists):
            save_folder = save_folder_name + "_" +  str(folder_counter).zfill(2)
            path_exists = os.path.exists(save_folder)
            folder_counter += 1

        os.makedirs(save_folder)

    img_count = 1
    if args.camera_type == "basler":
        def img_from_camera(img):
            global img_count
            print("image from camera received")

            if img is None:
                print("img is none")

            if ros_available:
                camera_publisher.publish_img(img)

            if args.save:
                save_file_path = os.path.join(save_folder, str(img_count).zfill(3) + ".png")
                print("writing image:", save_file_path)
                cv2.imwrite(save_file_path, img)
            
            img_count += 1

        camera_feed(undistort=args.undistort, fps=args.fps, callback=img_from_camera)

    elif args.camera_type == "realsense":
        realsense_camera = RealsenseCamera()
        while True:
            try:
                # 1. get image and depth from camera
                colour_img, depth_img, depth_colormap = realsense_camera.get_frame()

                if ros_available:
                    colour_realsense_publisher.publish_img(colour_img)
                    depth_realsense_publisher.publish_img(depth_img)
                    depthmap_realsense_publisher.publish_img(depth_colormap)

                    colour_depth = ColourDepth(br.cv2_to_imgmsg(colour_img), br.cv2_to_imgmsg(depth_img))
                    colour_depth_pub.publish(colour_depth)

                if args.save:
                    
                    for img_name, img in zip(["colour", "depth", "depthmap"], [colour_img, depth_img, depth_colormap]):
                        save_file_path = os.path.join(save_folder, str(img_count).zfill(3) + "." + img_name + ".png")
                        
                        cv2.imwrite(save_file_path, img)
                    
                    print("writing image:", save_file_path)
            
                img_count += 1

            except KeyboardInterrupt:
                break

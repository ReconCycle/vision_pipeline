import sys
import os
import cv2
from rich import print
import rospy
from sensor_msgs.msg import Image
from ros_vision_pipeline.msg import ColourDepth
from cv_bridge import CvBridge
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
from ros_service import ROSService
import message_filters
from pipeline_v2 import Pipeline
from pipeline_realsense import RealsensePipeline
from gap_detection.nn_gap_detector import NNGapDetector
import numpy as np
import json
import argparse
import time
import helpers

#! NOTES:
# differs from ros_pipeline_old.py because it uses the official ROS realsense camera publisher.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_type", help="Which camera: basler/realsense", nargs='?', type=str, default="basler")
    parser.add_argument("--publish_continuously", help="Publish continuously otherwise create service.", nargs='?', type=helpers.str2bool, default=True)
    parser.add_argument("--camera_topic", help="The name of the camera topic to subscribe to", nargs='?', type=str, default="basler")
    parser.add_argument("--node_name", help="The name of the node", nargs='?', type=str, default="vision_pipeline_basler")
    args = parser.parse_args()

    # set the camera_topic to realsense as well, if not set manually
    if args.camera_type.startswith("realsense") and args.camera_topic == "basler":
        args.camera_topic = "realsense"

    if args.camera_type.startswith("realsense") and args.node_name == "vision_pipeline_basler":
        args.node_name = "vision_pipeline_realsense"

    print("\ncamera_type:", args.camera_type)
    print("camera_topic:", args.camera_topic)
    print("node_name:", args.node_name)
    print("publish_continuously:", args.publish_continuously, "\n")

    # 1. Create ROS camera node and feed images from camera to camera node
    # 2. Create ROS camera labelled node and ROS data node -> feed images and data from pipeline

    rospy.init_node(args.node_name)
    if args.camera_type == "basler":
        pipeline = Pipeline()
    elif args.camera_type == "realsense":
        pipeline = RealsensePipeline()
    elif args.camera_type == "realsense_nn":
        pipeline = NNGapDetector()
    
    # if there is no camera topic then try and subscribe here and create a publisher for the camera images
    labelled_img_publisher = ROSPublisher(topic_name="/" + args.node_name + "/colour", msg_images=True)
    data_publisher = ROSPublisher(topic_name="/" + args.node_name + "/data", msg_images=False)
    action_publisher = ROSPublisher(topic_name="/" + args.node_name + "/action", msg_images=False)

    clustered_img_publisher = None
    mask_img_publisher = None
    depth_img_publisher = None
    if args.camera_type.startswith("realsense"):
        clustered_img_publisher = ROSPublisher(topic_name="/" + args.node_name + "/cluster", msg_images=True)
        mask_img_publisher = ROSPublisher(topic_name="/" + args.node_name + "/mask", msg_images=True)
        depth_img_publisher = ROSPublisher(topic_name="/" + args.node_name + "/depth", msg_images=True)


    # either create a publisher topic for the labelled images and detections or create a service
    if not args.publish_continuously:
        # ! not working for realsense yet
        ros_service = ROSService(pipeline, service_name="get_detection", camera_topic="/" + args.camera_topic + "/colour")

    colour_img = None
    depth_img = None
    img_id = 0

    img_sub = None
    subscribe_topic = ""
    depth_sub = None

    def img_from_camera_callback(img):
        global colour_img  # access variable from outside callback
        global img_id
        colour_img = CvBridge().imgmsg_to_cv2(img)
        colour_img = np.array(colour_img)
        img_id += 1

    def colour_depth_from_camera_callback(img_ros, depth_ros):
        colour = CvBridge().imgmsg_to_cv2(img_ros)
        colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        depth = CvBridge().imgmsg_to_cv2(depth_ros)

        global colour_img  # access variable from outside callback
        global depth_img
        global img_id
        # current_cam_img = CvBridge().imgmsg_to_cv2(img)
        colour_img = np.array(colour)
        depth_img = np.array(depth)

        img_id += 1

    def subscribe(args):
        global img_sub
        global depth_sub
        global subscribe_topic

        if args.camera_type == "basler":
            subscribe_topic = "/" + args.camera_topic + "/image_rect_color"
            img_sub = rospy.Subscriber(subscribe_topic, Image, img_from_camera_callback)
        elif args.camera_type == "realsense" or args.camera_type == "realsense_nn":
            img_topic = "/" + args.camera_topic + "/color/image_raw"
            depth_topic = "/" + args.camera_topic + "/aligned_depth_to_color/image_raw"

            img_sub = message_filters.Subscriber(img_topic, Image)
            depth_sub = message_filters.Subscriber(depth_topic, Image)

            ts = message_filters.TimeSynchronizer([img_sub, depth_sub], 10)
            ts.registerCallback(colour_depth_from_camera_callback)

    # ? the following might need to run on a separate thread!
    if args.publish_continuously:
        subscribe(args)

        # process the newest image from the camera
        processed_img_id = 0  # don't keep processing the same image
        t_prev = None
        fps = None
        while not rospy.is_shutdown():
            if colour_img is not None and processed_img_id < img_id:
                processed_img_id = img_id
                t_now = time.time()
                if t_prev is not None and t_now - t_prev > 0:
                    fps = "fps_total: " + str(round(1 / (t_now - t_prev), 1)) + ", "

                if args.camera_type == "basler":
                    labelled_img, detections, json_detections, action, json_action = pipeline.process_img(colour_img, fps)
                    print("json_detections", json_detections)

                    labelled_img_publisher.publish_img(labelled_img)
                    data_publisher.publish_text(json_detections)
                    action_publisher.publish_text(json_action)

                    print("json_action", json_action)

                elif args.camera_type == "realsense":
                    cluster_img, labelled_img, mask, lever_actions, json_lever_actions, altered_depth = pipeline.process_img(colour_img, depth_img, fps=fps)
                    if cluster_img is not None and json_lever_actions is not None:

                        clustered_img_publisher.publish_img(cluster_img)
                        labelled_img_publisher.publish_img(labelled_img)
                        mask_img_publisher.publish_img(mask)
                        depth_img_publisher.publish_img(altered_depth)
                        action_publisher.publish_text(json_lever_actions)  # ! this might need to be json
                        print("json_lever_actions", json_lever_actions)
                    else:
                        if cluster_img is None:
                            print("cluster_img is None")
                        if json_lever_actions is None:
                            print("json_lever_actions is None")

                elif args.camera_type == "realsense_nn":
                    labelled_img = pipeline.get_prediction(colour_img, depth_img)
                    labelled_img_publisher.publish_img(labelled_img)
                
                t_prev = t_now
            else:
                print("Waiting to receive image.")

                num_connections = img_sub.get_num_connections()
                if num_connections == 0 and subscribe_topic in list(np.array(rospy.get_published_topics()).flat):
                    print("(Re)subscribing to topic...")
                    if img_sub is not None:
                        img_sub.unregister()
                    if depth_sub is not None:
                        depth_sub.unregister()
                    subscribe(args)

                time.sleep(0.1)

            if img_id == sys.maxsize:
                img_id = 0
                processed_img_id = 0
    else:
        rospy.spin()

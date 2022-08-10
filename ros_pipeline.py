import sys
import os
import cv2
from rich import print
import rospy
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
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
import atexit

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
    lever_pose_publisher = None
    if args.camera_type.startswith("realsense"):
        clustered_img_publisher = ROSPublisher(topic_name="/" + args.node_name + "/cluster", msg_images=True)
        mask_img_publisher = ROSPublisher(topic_name="/" + args.node_name + "/mask", msg_images=True)
        depth_img_publisher = ROSPublisher(topic_name="/" + args.node_name + "/depth", msg_images=True)
        lever_pose_publisher = rospy.Publisher("/" + args.node_name + "/lever", PoseStamped, queue_size=1)


    # either create a publisher topic for the labelled images and detections or create a service
    if not args.publish_continuously:
        # ! not working for realsense yet
        ros_service = ROSService(pipeline, service_name="get_detection", camera_topic="/" + args.camera_topic + "/colour")

    colour_img = None
    depth_img = None
    img_id = 0
    aruco_pose = None
    aruco_point = None
    camera_info = None

    img_topic = ""
    img_sub = None
    depth_sub = None
    aruco_sub = None
    

    def img_from_camera_callback(img):
        global colour_img  # access variable from outside callback
        global img_id
        colour_img = CvBridge().imgmsg_to_cv2(img)
        colour_img = np.array(colour_img)
        img_id += 1

    def colour_depth_aruco_callback(camera_info_ros, img_ros, depth_ros):
        colour = CvBridge().imgmsg_to_cv2(img_ros)
        colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        depth = CvBridge().imgmsg_to_cv2(depth_ros)

        global colour_img  # access variable from outside callback
        global depth_img
        global img_id
        global camera_info
        
        # current_cam_img = CvBridge().imgmsg_to_cv2(img)
        colour_img = np.array(colour)
        depth_img = np.array(depth)

        camera_info = camera_info_ros

        img_id += 1

    def aruco_callback(aruco_pose_ros, aruco_point_ros):
        global aruco_pose
        global aruco_point

        aruco_pose = aruco_pose_ros.pose
        aruco_point = aruco_point_ros

    def subscribe(args):
        global aruco_sub
        global img_sub
        global depth_sub
        global img_topic

        if args.camera_type == "basler":
            img_topic = "/" + args.camera_topic + "/image_rect_color"
            img_sub = rospy.Subscriber(img_topic, Image, img_from_camera_callback)
        elif args.camera_type == "realsense" or args.camera_type == "realsense_nn":
            camear_info_topic = "/" + args.camera_topic + "/color/camera_info"
            img_topic = "/" + args.camera_topic + "/color/image_raw"
            depth_topic = "/" + args.camera_topic + "/aligned_depth_to_color/image_raw"

            # aruco_sub = rospy.Subscriber("/realsense_aruco/pose", PoseStamped, pose_callback)

            camera_info_sub = message_filters.Subscriber(camear_info_topic, CameraInfo)
            img_sub = message_filters.Subscriber(img_topic, Image)
            depth_sub = message_filters.Subscriber(depth_topic, Image)
            aruco_sub = message_filters.Subscriber("/realsense_aruco/pose", PoseStamped)
            aruco_pixel_sub = message_filters.Subscriber("/realsense_aruco/pixel", PointStamped)

            # adding the aruco pose means we cant use the TimeSynchronizer
            # we use the ApproximateTimeSynchronizer but with the slop time very low
            ts = message_filters.ApproximateTimeSynchronizer([camera_info_sub, img_sub, depth_sub], 10, slop=0.01, allow_headerless=False)
            ts.registerCallback(colour_depth_aruco_callback)
            
            # we might not always see the aruco markers, so subscribe to them separately
            ts2 = message_filters.ApproximateTimeSynchronizer([aruco_sub, aruco_pixel_sub], 10, slop=0.01, allow_headerless=False)
            ts2.registerCallback(aruco_callback)

    rospy.wait_for_service("/" + args.camera_topic + "/enable")
    toggle_realsense_service = rospy.ServiceProxy("/" + args.camera_topic + "/enable", SetBool)
    
    def toggle_realsense(new_state):
        try:
            res = toggle_realsense_service(new_state)
            print("toggled realsense:", res)
        except rospy.ServiceException as e:
            print("Service call failed: ", e)

    def exit_handler():
        toggle_realsense(False)
        print("disabled realsense and program exiting...")
    
    atexit.register(exit_handler)

    # ? the following might need to run on a separate thread!
    if args.publish_continuously:
        subscribe(args)

        # enable the camera
        if args.camera_type == "realsense":
            toggle_realsense(True)
            print("enabled realsense camera")

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
                    cluster_img, labelled_img, mask, lever_actions, json_lever_actions, altered_depth \
                        = pipeline.process_img(colour_img, depth_img, camera_info, aruco_pose=aruco_pose, aruco_point=aruco_point, fps=fps)
                    
                    labelled_img_publisher.publish_img(labelled_img)
                    
                    if cluster_img is not None and json_lever_actions is not None:

                        clustered_img_publisher.publish_img(cluster_img)
                        mask_img_publisher.publish_img(mask)
                        depth_img_publisher.publish_img(altered_depth)
                        action_publisher.publish_text(json_lever_actions)  # ! this might need to be json
                        print("json_lever_actions", json_lever_actions)

                        if len(lever_actions) > 0:
                            print("publishing lever")
                            lever_pose_publisher.publish(lever_actions[0].pose_stamped)
                    else:
                        if cluster_img is None:
                            print("cluster_img is None")
                        if json_lever_actions is None:
                            print("json_lever_actions is None")

                elif args.camera_type == "realsense_nn":
                    labelled_img = pipeline.get_prediction(colour_img, depth_img)
                    labelled_img_publisher.publish_img(labelled_img)
                
                #! 
                #!
                #!

                rospy.sleep(0.5)  # ! DEBUG ONLY
                print("SLEEPING! STOP COMPUTER OVERHEATING AND CRASHING?")

                t_prev = t_now
            else:
                print("Waiting to receive image.")

                num_connections = img_sub.get_num_connections()
                if num_connections == 0 and img_topic in list(np.array(rospy.get_published_topics()).flat):
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

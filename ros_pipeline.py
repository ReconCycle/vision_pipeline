import sys
import os
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from camera_feed import camera_feed
from ros_publisher import ROSPublisher
from ros_service import ROSService
from pipeline_v2 import Pipeline
import numpy as np
import json
import argparse
import time
import helpers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--publish_continuously", help="Publish continuously otherwise create service.", nargs='?', type=helpers.str2bool, default=False)
    parser.add_argument("--camera_topic", help="The name of the camera topic to subscribe to", nargs='?', type=str, default="/camera/image_color")
    parser.add_argument("--node_name", help="The name of the node", nargs='?', type=str, default="vision_pipeline")
    args_m = parser.parse_args()

    print("\npublish_continuously:", args_m.publish_continuously)
    print("camera_topic:", args_m.camera_topic)
    print("node_name:", args_m.node_name, "\n")

    # 1. Create ROS camera node and feed images from camera to camera node
    # 2. Create ROS camera labelled node and ROS data node -> feed images and data from pipeline

    rospy.init_node(args_m.node_name)
    pipeline = Pipeline()

    # if there is no camera topic then try and subscribe here and create a publisher for the camera images
    labelled_img_publisher = ROSPublisher(topic_name="/vision_pipeline/image_color", msg_images=True)
    data_publisher = ROSPublisher(topic_name="/vision_pipeline/data", msg_images=False)
    action_publisher = ROSPublisher(topic_name="/vision_pipeline/action", msg_images=False)

    # either create a publisher topic for the labelled images and detections or create a service
    if not args_m.publish_continuously:
        ros_service = ROSService(pipeline, service_name="get_detection", camera_topic=args_m.camera_topic)

    current_cam_img = None
    img_id = 0
    def img_from_camera_callback(img):
        global current_cam_img # access variable from outside callback
        global img_id
        current_cam_img = CvBridge().imgmsg_to_cv2(img)
        current_cam_img = np.array(current_cam_img)
        img_id += 1

    #? the following might need to run on a separate thread!    
    if args_m.publish_continuously:
        rospy.Subscriber(args_m.camera_topic, Image, img_from_camera_callback)
    
        # process the newest image from the camera
        processed_img_id = 0 # don't keep processing the same image
        t_prev = None
        fps = None
        while(True):
            if current_cam_img is not None and processed_img_id < img_id:
                processed_img_id = img_id
                t_now = time.time()
                if t_prev is not None and t_now - t_prev > 0:
                    fps = str(round(1 / (t_now - t_prev), 1)) + " fps (ros)"
                labelled_img, detections, json_detections, action, json_action = pipeline.process_img(current_cam_img, fps)
                print("json_detections", json_detections)
                
                labelled_img_publisher.publish_img(labelled_img)
                data_publisher.publish_text(json_detections)
                action_publisher.publish_text(json_action)
                
                t_prev = t_now
            else:
                print("Waiting to receive image.")
                time.sleep(0.1)

            if img_id == sys.maxsize:
                img_id = 0
                processed_img_id = 0
    else:
        rospy.spin()
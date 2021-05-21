import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from ros_vision_pipeline.srv import Detection, DetectionResponse
from pipeline_v2 import Pipeline
from ros_publisher import ROSPublisher
from cv_bridge import CvBridge
import json


class ROSService:
    def __init__(self, pipeline, service_name="get_detection", camera_topic="/camera/image_color"):
        self.pipeline = pipeline

        rospy.Subscriber(camera_topic, Image, self.camera_img_callback)
        self.camera_img = None

        s = rospy.Service(service_name, Detection, self.service_callback)
        # rospy.spin()
        

    def service_callback(self, req):
        print("service callback")
        if self.camera_img is not None:
            labelled_img, detections = self.pipeline.process_img(CvBridge().imgmsg_to_cv2(self.camera_img))
            json_detections = json.dumps(detections)
            print("type", type(json_detections))

            print("returning labelled image and detections")
            return DetectionResponse(True, CvBridge().cv2_to_imgmsg(labelled_img), json_detections)
        else:
            print("No image from camera!")
            return DetectionResponse(False, None, None)


    def camera_img_callback(self, camera_img):
        print("image from camera")
        self.camera_img = camera_img

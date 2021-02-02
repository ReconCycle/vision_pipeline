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
    def __init__(self, pipeline, service_name="get_detection", camera_topic="/camera/image_color", also_publish=False):
        self.pipeline = pipeline
        self.also_publish = also_publish

        self.br = CvBridge()

        if self.also_publish:
            self.labelled_img_publisher = ROSPublisher(topic_name="/vision_pipeline/image_color", msg_images=True)
            self.data_publisher = ROSPublisher(topic_name="/vision_pipeline/data", msg_images=False)

        rospy.Subscriber(camera_topic, Image, self.camera_img_callback)
        self.camera_img = None

        s = rospy.Service(service_name, Detection, self.service_callback)
        # rospy.spin()
        

    def service_callback(self, req):
        print("service callback")
        if self.camera_img is not None:
            labelled_img, detections = self.pipeline.process_img(self.br.imgmsg_to_cv2(self.camera_img))
            json_detections = json.dumps(detections)
            print("type", type(json_detections))

            if self.also_publish:
                self.labelled_img_publisher.publish_img(labelled_img)
                # self.data_publisher.publish_text(json_detections)

            print("returning labelled image and detections")
            return DetectionResponse(True, self.br.cv2_to_imgmsg(labelled_img), json_detections)
        else:
            print("No image from camera!")
            return DetectionResponse(False, None, None)


    def camera_img_callback(self, camera_img):
        print("image from camera")
        self.camera_img = camera_img

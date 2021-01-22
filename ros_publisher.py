import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from std_msgs.msg import String

class ROSPublisher:
    def __init__(self, node_name):
        rospy.init_node("camera_publisher", anonymous=True)
        
        self.br = CvBridge()
        self.pub = rospy.Publisher(node_name, Image, queue_size=50)

    def publish_img(self, img):
        if not rospy.is_shutdown():
            if img is not None:
                print("publishing image")
                self.pub.publish(self.br.cv2_to_imgmsg(img))

if __name__ == '__main__':
    print("camera publisher node")

    ros_publisher = ROSPublisher(node_name="/camera/image_color")
    img_path = "./example.png"
    img = cv2.imread(img_path)
    loop_rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        
        if img is not None:
            rospy.loginfo("publishing image")
            ros_publisher.publish_img(img)
        else:
            rospy.loginfo("image is none :(")

        loop_rate.sleep()

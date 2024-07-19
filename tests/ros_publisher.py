import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from rich import print

class Nodo(object):
    def __init__(self):
        # Params
        self.image = np.zeros((100, 100, 3))
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Publishers
        self.pub = rospy.Publisher('/imagetimer', Image,queue_size=10)

    #     # Subscribers
    #     rospy.Subscriber("/basler/image_rect_color",Image,self.callback)

    #     rospy.Subscriber("/imagetimer",Image,self.callback2)

    # def callback(self, msg):
    #     rospy.loginfo('Image received...')
    #     self.image = self.br.imgmsg_to_cv2(msg)

    # def callback2(self, msg):
    #     image = self.br.imgmsg_to_cv2(msg)
    #     rospy.loginfo(f'Image received from /imagetimer..., {image.shape}')
        


    def start(self):
        rospy.loginfo("Timing images")
        #rospy.spin()
        while not rospy.is_shutdown():
            rospy.loginfo('publishing image')
            #br = CvBridge()
            if self.image is not None:
                self.pub.publish(self.br.cv2_to_imgmsg(self.image))
            self.loop_rate.sleep()
            
if __name__ == '__main__':
    rospy.init_node("imagetimer111", anonymous=True)
    my_node = Nodo()
    my_node.start()
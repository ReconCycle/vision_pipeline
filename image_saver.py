# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import termios, fcntl, sys, os
from rich import print


# implementation found here:
# https://stackoverflow.com/questions/63605503/listen-for-a-specific-key-using-pynput-keylogger
class KeypressListener():
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()

        self.oldterm = termios.tcgetattr(self.fd)
        newattr = termios.tcgetattr(self.fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, newattr)

        self.oldflags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.oldflags | os.O_NONBLOCK)

        print("\n") # we need this to be in the next line

    def get_keypress(self):
        chars = sys.stdin.readline()
        if len(chars) > 0:
            char = str(chars)[-1]
            return char
        else:
            return None

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.oldterm)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.oldflags)


class Main():
    def __init__(self) -> None:
        
        # keypress listener will break the terminal if we don't close it on exception
        sys.excepthook = self.except_hook

        rospy.init_node('image_listener')
        
        rate = rospy.Rate(50) # polling rate
        
        # Instantiate CvBridge
        self.bridge = CvBridge()

        self.counter = 1
        self.save = False
        
        self.save_path = "experiments/datasets/new_dataset/untitled"
        
        # check if file path is empty
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        if os.listdir(self.save_path):
            print("[red]directory not empty! exiting...[/red]")
            return
        
        # Define your image topic
        image_topic = "/basler/image_rect_color"
        # Set up your subscriber and define its callback
        rospy.Subscriber(image_topic, Image, self.image_callback)
        # Spin until ctrl + c
        # rospy.spin()
        self.keypress_listener = KeypressListener()
        
        # register what to do on shutdown
        rospy.on_shutdown(self.close)
        
        while not rospy.is_shutdown():
            try:
                char = self.parse_keypress()
                if char is not None:
                    self.save = True
                
                rate.sleep()
                
            except KeyboardInterrupt:
                # tidy up
                self.close()
                break

        # tidy up
        self.close()


    def image_callback(self, msg):
        if self.save:
            try:
                # Convert your ROS Image message to OpenCV2
                cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                print(e)
            else:
                # Save your OpenCV2 image as a jpeg
                filename = str(self.counter).zfill(4) + ".jpg"
                file_path = os.path.join(self.save_path, filename)
                print("saving image:", file_path)
                cv2.imwrite(file_path, cv2_img)
                self.counter += 1
                self.save = False

    def except_hook(self, type, value, tb):
        self.close()
    
    def close(self):
        self.keypress_listener.close()
        
    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()
        if char is not None:
            return char
        return None


if __name__ == '__main__':
    main = Main()

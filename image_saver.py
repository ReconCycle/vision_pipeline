import termios, fcntl, sys, os
import cv2
from rich import print
from rich.markup import escape
import numpy as np
import ros_numpy
import pickle
import datetime

# ROS
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from camera_control_msgs.srv import SetSleeping
from std_srvs.srv import SetBool


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

        self.camera_node = "realsense" #! options: realsense/realsensed405/basler
        
        self.save_path = "saves/{date:%Y-%m-%d_%H:%M:%S}_{camera_node}".format(date=datetime.datetime.now(), camera_node=self.camera_node)
 
        # check if file path is empty
        if not os.path.exists(self.save_path):
            print("making folder", self.save_path)
            os.makedirs(self.save_path)
            
        if os.listdir(self.save_path):
            print("[red]directory not empty! exiting...[/red]")
            return
        
        # Set up your subscriber and define its callback
        if self.camera_node == "realsense":
            camera_info_topic = f"/{self.camera_node}/color/camera_info"
            img_topic = f"/{self.camera_node}/color/image_raw"
            depth_topic = f"/{self.camera_node}/aligned_depth_to_color/image_raw"
            camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
            img_sub = message_filters.Subscriber(img_topic, Image)
            depth_sub = message_filters.Subscriber(depth_topic, Image)

            ts = message_filters.ApproximateTimeSynchronizer([img_sub, depth_sub, camera_info_sub], 10, slop=0.05, allow_headerless=False)
            ts.registerCallback(self.img_from_realsense_cb)
        elif self.camera_node == "realsensed405":
            camera_info_topic = f"/{self.camera_node}/color/camera_info"
            img_topic = f"/{self.camera_node}/color/image_raw"
            depth_topic = f"/{self.camera_node}/depth/image_rect_raw"
            camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
            img_sub = message_filters.Subscriber(img_topic, Image)
            depth_sub = message_filters.Subscriber(depth_topic, Image)

            ts = message_filters.ApproximateTimeSynchronizer([img_sub, depth_sub, camera_info_sub], 10, slop=0.05, allow_headerless=False)
            ts.registerCallback(self.img_from_realsense_cb)
        elif self.camera_node == "basler":
            image_topic = f"/{self.camera_node}/image_rect_color"
            rospy.Subscriber(image_topic, Image, self.img_from_basler_cb)
        
        # wake camera
        if self.camera_node == "basler":
            self.enable_camera_invert = True
            self.camera_service = rospy.ServiceProxy(f"/{self.camera_node}/set_sleeping", SetSleeping)
        
        elif self.camera_node == "realsense":
            self.enable_camera_invert = False
            self.camera_service = rospy.ServiceProxy(f"/{self.camera_node}/enable", SetBool)

        self.enable_camera(True)

        # Spin until ctrl + c
        # rospy.spin()
        self.keypress_listener = KeypressListener()
        
        print("\n[green]press any key to save image:")
        
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


    #! duplicate code from pipeline_camera.py:
    def enable_camera(self, state):
        success = False
        msg = self.camera_node + ": "
        try:
            # inverse required for basler camera
            if self.enable_camera_invert:
                res = self.camera_service(not state)
            else:
                res = self.camera_service(state)
            
            success = res.success
            if success:
                if state:
                    msg += "enabled camera"
                else:
                    msg += "disabled camera"
            else:
                if state:
                    msg += "FAILED to enable camera"
                else:
                    msg += "FAILED to disable camera"
            
        except rospy.ServiceException as e:
            if "UVC device is streaming" in str(e) and state:
                msg += "enabled camera (already streaming)"
                success = True
            elif "UVC device is not streaming" in str(e) and not state:
                msg += "disabled camera (already stopped)"
                success = True
            else:
                if state:
                    msg += "FAILED to enable camera"
                else:
                    msg += "FAILED to disable camera"
                
                msg += ", service call failed: " + escape(str(e))
        
        if success: 
            print("[green]" + msg)
            # update internal state
            self.camera_enabled = state
        else:
            print("[red]" + msg)
        
        return success, msg


    def img_from_realsense_cb(self, img_msg, depth_msg, camera_info):
        self.img_saver(img_msg, depth_msg, camera_info)

    def img_from_basler_cb(self, img_msg):
        self.img_saver(img_msg)
    
    
    def img_saver(self, img_msg, depth_msg=None, camera_info=None):
        
        if self.save:
            try:
                # Convert your ROS Image message to OpenCV2
                cv2_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                
                cv2_depth_img = None
                if depth_msg is not None:
                    cv2_depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                    cv2_depth_img = np.array(cv2_depth_img)
                    min = np.min(cv2_depth_img)
                    max = np.max(cv2_depth_img)
                    print(f"min depth: {min}, max depth: {max}")
                
            except CvBridgeError as e:
                print(e)
            else:
                # Save your OpenCV2 image as a jpeg
                filename_colour = str(self.counter).zfill(4) + ".jpg"                
                file_path = os.path.join(self.save_path, filename_colour)
                print("saving image:", file_path)
                cv2.imwrite(file_path, cv2_img)
                
                if cv2_depth_img is not None:
                    filename_depth = str(self.counter).zfill(4) + "_depth.npy"
                    file_path = os.path.join(self.save_path, filename_depth)
                    np.save(file_path, cv2_depth_img)
                    
                    # TODO: improve visualisation image, see earlier work
                    filename_depth_viz = str(self.counter).zfill(4) + "_depth_viz.jpg"
                    depth_viz = cv2_depth_img.copy() * 255 / np.max(cv2_depth_img)
                    file_path = os.path.join(self.save_path, filename_depth_viz)
                    cv2.imwrite(file_path, depth_viz)
                    
                if camera_info is not None:            
                    filename_info = str(self.counter).zfill(4) + "_camera_info.pickle"
                    file_path = os.path.join(self.save_path, filename_info)
                    # camera_info_np = ros_numpy.numpify(camera_info)
                    # np.save(camera_info_np, file_path)/
                    filehandler = open(file_path, "wb")
                    pickle.dump(camera_info,filehandler)
                    filehandler.close()
                
                self.counter += 1
                self.save = False

    def except_hook(self, type, value, tb):
        self.close()
    
    def close(self):
        #! don't disable camera because vision_pipeline might also be running
        # self.enable_camera(False) 
        self.keypress_listener.close()
        
    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()
        if char is not None:
            return char
        return None


if __name__ == '__main__':
    main = Main()

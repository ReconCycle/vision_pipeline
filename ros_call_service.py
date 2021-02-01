import rospy
from ros_vision_pipeline.srv import Detection

if __name__ == '__main__':
    get_detection = rospy.ServiceProxy('get_detection', Detection)
    get_detection()
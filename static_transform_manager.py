import tf2_ros
import tf
import rospy
from geometry_msgs.msg import TransformStamped

# the static broadcast should contain all the transforms we want to have at that moment
# https://answers.ros.org/question/261815/how-can-i-access-all-static-tf2-transforms/

class StaticTransformManager():
    
    def __init__(self) -> None:
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.static_tfs = []
    

    def create_tf(self, child_frame_id, parent_frame=None, transform=None):

        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        
        if parent_frame is not None:
            frame_id = parent_frame
        else:
            frame_id = child_frame_id
            
        static_transformStamped.header.frame_id = frame_id
        static_transformStamped.child_frame_id = child_frame_id

        if transform is None:
            static_transformStamped.transform.translation.x = float(0)
            static_transformStamped.transform.translation.y = float(0)
            static_transformStamped.transform.translation.z = float(0)

            quat = tf.transformations.quaternion_from_euler(float(0),float(0),float(0))
            static_transformStamped.transform.rotation.x = quat[0]
            static_transformStamped.transform.rotation.y = quat[1]
            static_transformStamped.transform.rotation.z = quat[2]
            static_transformStamped.transform.rotation.w = quat[3]
        else:
            static_transformStamped.transform = transform
        
        # remove older duplicates
        self.static_tfs = [tf for tf in self.static_tfs if not (tf.header.frame_id == frame_id and tf.child_frame_id == child_frame_id)]                
        
        # append new transform
        self.static_tfs.append(static_transformStamped)
        
        # broadcast all transforms
        self.broadcast()
        
    def broadcast(self):
        print("broadcasting: number static_tfs:", len(self.static_tfs))
        self.static_broadcaster.sendTransform(self.static_tfs)
#! /usr/bin/env python3

import rospy
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from geometry_msgs.msg import PoseStamped
from rsu_msgs.msg import StampedObjectPoseArray

def load_param(name, value=None):
    """Function used to get parameters from ROS parameter server

    :param name: name of the parameter
    :type name: string
    :param value: default value of the parameter, defaults to None
    :type value: _type_, optional
    :return: value of the parameter
    :rtype: _type_
    """
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


class transform_sensor_objects(object):

    def __init__(self):

        rospy.init_node('transform_sensor_objects')

        self.IN_TOP = load_param('~in_top')
        self.OUT_TOP = load_param('~out_top')
        self.TO_FRAME = load_param('~to_frame')

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        self.out_pub = rospy.Publisher(self.OUT_TOP, StampedObjectPoseArray, queue_size=10)
        self.in_sub = rospy.Subscriber(self.IN_TOP, StampedObjectPoseArray, self.transform_cb)

    def transform_cb(self, msg):

        trans = self.tf_buf.lookup_transform(self.TO_FRAME, msg.header.frame_id, rospy.Time(0))   
        
        for obj in msg.objects:

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = obj.pose.pose
            pose = do_transform_pose(pose, trans)

            obj.pose.pose = pose.pose

        msg.header.frame_id = self.TO_FRAME

        self.out_pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':

    ## Start node ##
    transform_sensor_objects().run()


#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy

from threading import Lock

# SVEA imports
from svea.states import VehicleState
from svea.interfaces import LocalizationInterface, ActuationInterface
from svea.interfaces.rc import RCInterface
from svea.controllers.pure_pursuit import PurePursuitController
from svea.data import RVIZPathHandler
from svea_mocap.mocap import MotionCaptureInterface
from svea_social_navigation.track import Arc, Track

from svea_msgs.msg import lli_ctrl

# ROS imports
import message_filters as mf
from svea_msgs.msg import VehicleState as VehicleStateMsg
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TwistStamped, PointStamped
from sensor_msgs.msg import Joy
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import tf2_ros
import tf2_geometry_msgs
from tf import transformations 


from aug_demo.srv import VerifyState

def state_to_pose(state):
    pose = PoseStamped()
    pose.header = state.header
    pose.pose.position.x = state.x
    pose.pose.position.y = state.y
    qx, qy, qz, qw = quaternion_from_euler(0, 0, state.yaw)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose

def pose_to_state(pose):
    state = VehicleStateMsg()
    state.header = pose.header
    state.x = pose.pose.position.x
    state.y = pose.pose.position.y
    roll, pitch, yaw = euler_from_quaternion([pose.pose.orientation.x,
                                              pose.pose.orientation.y,
                                              pose.pose.orientation.z,
                                              pose.pose.orientation.w])
    state.yaw = yaw
    return state

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

class Avoider(object):

    THRESHOLD = 0.2

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('avoider')

        # Get parameters
        self.NAME = load_param('~name', 'svea')
        self.LOCATION = load_param('~location', 'kip')

        if self.LOCATION == 'sml':
            self._state_pub = rospy.Publisher('state', VehicleStateMsg, latch=True, queue_size=1)
            def state_cb(pose, twist):
                state = VehicleStateMsg()
                state.header = pose.header
                state.child_frame_id = 'svea2'
                state.x = pose.pose.position.x 
                state.y = pose.pose.position.y
                roll, pitch, yaw = euler_from_quaternion([pose.pose.orientation.x,
                                                          pose.pose.orientation.y,
                                                          pose.pose.orientation.z,
                                                          pose.pose.orientation.w])
                state.yaw = yaw
                state.v = twist.twist.linear.x
                self._state_pub.publish(state)
            mf.TimeSynchronizer([
                mf.Subscriber(f'/qualisys/{self.NAME}/pose', PoseStamped),
                mf.Subscriber(f'/qualisys/{self.NAME}/velocity', TwistStamped)
            ], 10).registerCallback(state_cb)

            self._target_pub = rospy.Publisher('target', PointStamped, latch=True, queue_size=1)

        from visualization_msgs.msg import Marker
        marker_pub = rospy.Publisher('pedestrians', Marker, queue_size=10)
        rospy.Subscriber('/sensor/markers', Marker, marker_pub.publish)
        
        rospy.Subscriber('state', VehicleStateMsg, self.state_in_utm_callback, queue_size=1)
        self.state_svea_pub = rospy.Publisher('svea_in_utm', VehicleStateMsg, queue_size=1)
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

        self.localizer = LocalizationInterface().start()
        self.localizer.add_callback(self.state_in_utm_callback)
        self.state = self.localizer.state

        while not self.localizer.is_ready:
            if rospy.is_shutdown():
                raise Exception("Shutdown before initialization was done.")
            rospy.sleep(0.1)
        
        rospy.wait_for_service('ltms/verify_state')
        self.verify_state = rospy.ServiceProxy('ltms/verify_state', VerifyState)

        self.safe = True
        def verify_state_tmr(event):
            if event.last_expected is not None and event.current_real < event.last_expected:
                return # maybe stupiod
            self.safe = self.verify_state(self.state.state_msg).ok
            print(f'{self.safe=}')
        rospy.Timer(rospy.Duration(0.5), verify_state_tmr)
        rospy.sleep(1)

        self.lli_pub = rospy.Publisher('lli/ctrl_request', lli_ctrl, queue_size=1)
        rospy.Subscriber(f'{self.NAME}/remote/ctrl_request', lli_ctrl, self.remote_cb)

        rospy.loginfo('Start node')

    def remote_cb(self, msg):
        if self.safe:
            self.lli_pub.publish(msg)

    def run(self):
        """Run node."""
        rospy.spin()

    def state_in_utm_callback(self, req):
        state = req.state
        state_pose = state_to_pose(state)
        trans = self.buffer.lookup_transform("utm", state.header.frame_id, rospy.Time.now(), rospy.Duration(0.5))
        pose = tf2_geometry_msgs.do_transform_pose(state_pose, trans)
        new_state = pose_to_state(pose)
        new_state.v = state.v

        self.state_svea_pub.publish(new_state)


if __name__ == '__main__':

    ## Start node ##
    Avoider().run()


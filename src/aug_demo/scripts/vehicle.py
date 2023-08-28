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

from aug_demo.srv import VerifyState

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

        self.localizer = LocalizationInterface().start()
        self.state = self.localizer.state

        while not self.localizer.is_ready:
            if rospy.is_shutdown():
                raise Exception("Shutdown before initialization was done.")
            rospy.sleep(0.1)

        # rospy.wait_for_service('ltms/verify_state')
        # self.verify_state = rospy.ServiceProxy('ltms/verify_state', VerifyState)

        self.safe = True
        def verify_state_tmr(event):
            if event.last_expected is not None and event.current_real < event.last_expected:
                return # maybe stupiod
            self.safe = self.verify_state(self.state.state_msg).ok
            print(f'{self.safe=}')
        # rospy.Timer(rospy.Duration(0.5), verify_state_tmr)
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


if __name__ == '__main__':

    ## Start node ##
    Avoider().run()


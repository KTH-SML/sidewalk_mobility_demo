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

class remote_driving(object):

    THRESHOLD = 0.2

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('remote_driving')

        self.rate = rospy.Rate(10)

        # Get parameters
        self.NAME = load_param('~name', 'remote_driving')
        self.VEHICLE_NAME = load_param('~vehicle_name', 'svea')
        self.LOCATION = load_param('~location', 'kip')

        self._path_topic = load_param('~path_topic', 'path')
        self._path_pub = rospy.Publisher(self._path_topic, Path, latch=True, queue_size=1)
        self._path_lock = Lock()

        # Start localization interface based on which localization method is being used
        self.localizer = LocalizationInterface().start()
        self.state = self.localizer.state

        # Start actuation interface 
        self.actuation = ActuationInterface().start()
        self.steering, self.velocity = 0.0, 0.0

        # Subscribe to joy
        rospy.Subscriber('joy', Joy, self.joy_cb, queue_size=1)

        # Instatiate RVIZPathHandler object if publishing to RVIZ
        self.data_handler = RVIZPathHandler()

        rospy.Timer(rospy.Duration(0.1), self.create_path)

        rospy.loginfo('Starting node')

    def joy_cb(self, msg):
        steering = 0.5*msg.axes[0]
        forward = msg.axes[2]
        backward = msg.axes[1]
        steering_input_max = 0.15
        max_velocity = 0.8
        max_steering = 40*np.pi/180
        
        steering = min(steering, steering_input_max)
        steering = max(steering, -steering_input_max)

        forward += 1
        forward /= 2
        forward *= max_velocity

        backward += 1
        backward /= 2
        backward *= -max_velocity

        self.velocity = forward or backward
        self.steering = (steering - (-steering_input_max))*(max_steering*2)/(steering_input_max*2)+(-max_steering)

        rospy.loginfo(f"{self.steering=:.02f}, {self.velocity=:.02f}")

    def create_path(self, event):
        
        if event.last_expected is not None and event.current_real < event.last_expected:
            return # maybe stupiod

        # Using the Track framework to create a sort of frenet path that starts from current state 
        # and stretches 4 second forward (given current velocity).
        # Curvature K = 1/L * tan(delta)
        # Length d = v * 4

        POINT_DENSITY = 50
        BASEWIDTH = 0.32 # [m]
        HEADWAY = 3 # [s]

        # steering, velocity = self.rc_remote.steering, self.rc_remote.velocity
        steering = self.steering
        velocity = self.velocity

        start_point = (self.state.x, self.state.y, self.state.yaw)

        if abs(velocity) < 0.2:
            path_from_track = np.array([start_point[:2]])
        else:
            arc = Arc(HEADWAY*velocity, 1/BASEWIDTH * np.tan(steering))
            track = Track([arc], *start_point, POINT_DENSITY=POINT_DENSITY)
            path_from_track = np.array(track.cartesian).T

        path_array = np.zeros((len(path_from_track), 4))
        path_array[:, 0] = path_from_track[:, 0]
        path_array[:, 1] = path_from_track[:, 1]
        path_array[:, 2] = 0.6 # velocity
        path_array[:, 3] = 0

        path_msg = Path()
        path_msg.header.frame_id = self.state.frame_id
        path_msg.header.stamp = rospy.Time.now()
        path_msg.poses = []

        for i, (x, y) in enumerate(path_array[:, :2]):
            pose = PoseStamped()
            pose.header.frame_id = path_msg.header.frame_id
            pose.header.stamp = path_msg.header.stamp + i*rospy.Duration(HEADWAY/POINT_DENSITY)
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_msg.poses.append(pose)

        # Publish global path on rviz
        self._path_pub.publish(path_msg)
        
    def _visualize_data(self):
        self.data_handler.log_state(self.state)
        self.data_handler.visualize_data()

    def keep_alive(self):
        """
        Keep alive function based on the distance to the goal and current state of node

        :return: True if the node is still running, False otherwise
        :rtype: boolean
        """
        return not rospy.is_shutdown()

    def run(self):
        """Run node."""
        while self.keep_alive():
            self.spin()
            self.rate.sleep()

    def spin(self):
        """Body of main loop."""

        # Get svea state
        if not self.localizer.is_ready: 
            return

        # Send control to actuator interface
        self.actuation.send_control(self.steering, self.velocity)

        # Visualize data on RVIZ
        self._visualize_data()


if __name__ == '__main__':

    ## Start node ##
    remote_driving().run()


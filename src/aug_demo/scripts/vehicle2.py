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
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from sensor_msgs.msg import Joy
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler

from aug_demo.srv import RefinePath

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

def publish_initialpose(state, n=10):
    """Method for publishing initial pose

    :param state: vehicle state
    :type state: VehicleState
    :param n: _description_, defaults to 10
    :type n: int, optional
    """
    p = PoseWithCovarianceStamped()
    p.header.frame_id = 'map'
    p.pose.pose.position.x = state.x
    p.pose.pose.position.y = state.y

    q = quaternion_from_euler(0, 0, state.yaw)
    p.pose.pose.orientation.z = q[2]
    p.pose.pose.orientation.w = q[3]

    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rate = rospy.Rate(10)

    for _ in range(n):
        pub.publish(p)
        rate.sleep()


class Avoider(object):

    THRESHOLD = 0.2

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('avoider')

        # Get parameters
        self.NAME = load_param('~name', 'svea2')
        self.STATE = load_param('~state', [0, 0, 0, 0])
        self.LOCATION = load_param('~location', 'kip')

        self.refine_path = rospy.ServiceProxy('ltms/refine_path', RefinePath)

        self.rate = rospy.Rate(10)

        self._path_topic = load_param('~path_topic', '/path')
        self._path_pub = rospy.Publisher(self._path_topic, Path, latch=True, queue_size=1)
        self._path_lock = Lock()

        # Create Controller
        self.controller = PurePursuitController()
        self.controller.target_velocity = 0.6

        # Start actuation interface 
        self.actuation = ActuationInterface().start()

        # Start localization interface based on which localization method is being used
        self.localizer = (LocalizationInterface().start() if self.LOCATION == 'kip' else
                          MotionCaptureInterface(self.NAME).start())

        # Subscribe to joy
        rospy.Subscriber('joy', Joy, self.joy_cb, queue_size=1)

        # Initialize vehicle state
        publish_initialpose(VehicleState(*self.STATE))
        self.steering = 0.0
        self.velocity = 0.0
        self.state = self.localizer.state

        # Instatiate RVIZPathHandler object if publishing to RVIZ
        self.data_handler = RVIZPathHandler()

        rospy.Timer(rospy.Duration(0.2), self.create_path)

        rospy.sleep(1)
        rospy.loginfo('Start')

    def joy_cb(self, msg):
        steering = 0.5*msg.axes[0] #steering
        forward = msg.axes[2] # forward
        steering_input_max = 0.15
        velocity_input_max = 1.0
        max_steering = 40*np.pi/180
        max_speed = 1.2
        
        steering = min(steering, steering_input_max)
        steering = max(steering, -steering_input_max)
        self.velocity = (forward - (-velocity_input_max))*(max_speed)/(velocity_input_max*2)+0
        self.steering = (steering - (-steering_input_max))*(max_steering*2)/(steering_input_max*2)+(-max_steering)

        rospy.loginfo(f"{self.steering}, {self.velocity}")

    def create_path(self, event):
        
        if event.last_expected is not None and event.current_real < event.last_expected:
            return # maybe stupiod

        # Using the Track framework to create a sort of frenet path that starts from current state 
        # and stretches 4 second forward (given current velocity).
        # Curvature K = 1/L * tan(delta)
        # Length d = v * 4

        POINT_DENSITY = 50
        BASEWIDTH = 0.32 # [m]
        HEADWAY = 2 # [s]

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

        # Create array for MPC reference
        path_array = np.zeros((len(path_from_track), 4))
        path_array[:, 0] = path_from_track[:, 0]
        path_array[:, 1] = path_from_track[:, 1]
        path_array[:, 2] = 0.6 # velocity
        path_array[:, 3] = 0

        path_msg = Path()
        path_msg.header.frame_id = self.state.frame_id
        path_msg.header.stamp = rospy.Time.now()
        path_msg.poses = []

        # Idk why it's flipped...
        for i, (y, x) in enumerate(path_array[:, :2]):
            pose = PoseStamped()
            pose.header.frame_id = path_msg.header.frame_id
            pose.header.stamp = path_msg.header.stamp + i*rospy.Duration(HEADWAY/POINT_DENSITY)
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_msg.poses.append(pose)

        # resp = self.refine_path(path)
        # path_msg = resp.path

        # Publish global path on rviz
        self._path_pub.publish(path_msg)
        
        self.path = path_array
        
    def next_target(self):
        # Get next waypoint index (by computing offset between robot and each point of the path), wrapping it in case of
        # index out of bounds
        x0 = (self.state.x, self.state.y)
        self.distances = np.linalg.norm(self.path[:, 0:2] - x0, axis=1)
        self.waypoint_idx = self.distances.argmin()
        while self.waypoint_idx < len(self.distances):
            if self.THRESHOLD < self.distances[self.waypoint_idx]:
                break
            else:
                self.waypoint_idx += 1
        else:
            self.waypoint_idx -= 1
        self.target = (self.path[self.waypoint_idx, 0], self.path[self.waypoint_idx, 1])
            
    def _visualize_data(self):
        self.data_handler.log_state(self.state)
        self.data_handler.log_ctrl(self.steering, self.velocity, rospy.get_time())
        self.data_handler.update_target(self.target)
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

        self.next_target()
        
        # If final distance, don't send control signal
        # Maybe no work, smol hack
        if self.distances[-1] < self.THRESHOLD:
            return
        print(self.target)
        self.steering, self.velocity = self.controller.compute_control(self.state, self.target)

        # Get optimal velocity (by integrating once the acceleration command and summing it to the current speed) and
        # steering controls
        print(f'Control velocity, steering: ({self.velocity:+.02f}, {self.steering:+.02f})')

        # Send control to actuator interface
        self.actuation.send_control(self.steering, self.velocity)

        # Visualize data on RVIZ
        self._visualize_data()


if __name__ == '__main__':

    ## Start node ##
    Avoider().run()


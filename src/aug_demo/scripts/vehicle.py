#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy
from copy import deepcopy

from threading import Lock

# SVEA imports
from svea.controllers.social_mpc import SMPC
from svea.sensors import Lidar
from svea.models.bicycle_mpc import BicycleModel
from svea.states import VehicleState
from svea.interfaces import LocalizationInterface, ActuationInterface, PlannerInterface
from svea.interfaces.rc import RCInterface
from svea.data import RVIZPathHandler
from svea_mocap.mocap import MotionCaptureInterface
from svea_social_navigation.apf import ArtificialPotentialFieldHelper
from svea_social_navigation.static_unmapped_obstacle_simulator import StaticUnmappedObstacleSimulator
from svea_social_navigation.dynamic_obstacle_simulator import DynamicObstacleSimulator
from svea_social_navigation.sfm_helper_obj_rec import SFMHelper
from svea_social_navigation.track import Track, Arc

# ROS imports
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Path

from sensor_msgs.msg import Joy

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

def lists_to_pose_stampeds(x_list, y_list, yaw_list=None, t_list=None):
    poses = []
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]

        curr_pose = PoseStamped()
        curr_pose.header.frame_id = 'map'
        curr_pose.pose.position.x = x
        curr_pose.pose.position.y = y

        if not yaw_list is None:
            yaw = yaw_list[i]
            quat = quaternion_from_euler(0.0, 0.0, yaw)
            curr_pose.pose.orientation.x = quat[0]
            curr_pose.pose.orientation.y = quat[1]
            curr_pose.pose.orientation.z = quat[2]
            curr_pose.pose.orientation.w = quat[3]

        if not t_list is None:
            t = t_list[i]
            curr_pose.header.stamp = rospy.Time(secs=t)
        else:
            curr_pose.header.stamp = rospy.Time.now()

        poses.append(curr_pose)
    return poses

class SocialAvoidance(object):
    WINDOW_LEN = 10
    DELTA_TIME = 0.1
    DELTA_TIME_REAL = 0.3
    GOAL_THRESH = 0.2
    STRAIGHT_SPEED = 0.7
    TURN_SPEED = 0.5
    MAX_N_STATIC_OBSTACLES = 10
    MAX_N_DYNAMIC_OBSTACLES = 10
    MAX_N_PEDESTRIANS = 10
    MAX_WAIT = 1.0/10.0 # no slower than 10Hz

    def __init__(self):
        """
        Init method for SocialNavigation class
        """
        rospy.init_node('svea_social_navigation')

        # Get parameters
        self.STATE = load_param('~state', [0, 0, 0, 0])
        self.SVEA_NAME = load_param('~name', 'svea2')
        self.IS_PEDSIM = load_param('~is_pedsim', True)
        self.LOCATION = load_param('~location', 'kip')

        self.rate = rospy.Rate(10)

        # Define publisher for MPC predicted path
        self.pred_path_pub = rospy.Publisher("pred_path", Path, queue_size=1, latch=True)

        # Initialize vehicle state
        self.state = VehicleState(*self.STATE)
        self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]
        self.last_state_time = None
        # Publish initial pose
        publish_initialpose(self.state)

        # Instatiate RVIZPathHandler object if publishing to RVIZ
        self.data_handler = RVIZPathHandler()

        # Define planner interface
        self.pi = PlannerInterface(theta_threshold=0.3)
        self.pi.initialize_path_interface()

        # Initialize dynamic obstacles simulator
        self.DYNAMIC_OBS = load_param('~dynamic_obstacles', [])
        self.dynamic_obs_simulator = DynamicObstacleSimulator(self.DYNAMIC_OBS, self.DELTA_TIME)
        self.dynamic_obs_simulator.thread_publish()

        # Initialize static unmapped obstacles simulator
        self.STATIC_UNMAPPED_OBS = load_param('~static_unmapped_obstacles', [])
        self.static_unmapped_obs_simulator = StaticUnmappedObstacleSimulator(self.STATIC_UNMAPPED_OBS)
        self.static_unmapped_obs_simulator.publish_obstacle_msg()

        # Initialize social force model helper
        self.sfm_helper = SFMHelper(is_pedsim=self.IS_PEDSIM)

        # Start lidar
        self.lidar = Lidar().start()
        # Start actuation interface 
        self.actuation = ActuationInterface().start()
        # Start localization interface based on which localization method is being used
        self.localizer = (LocalizationInterface().start() if self.LOCATION == 'kip' else
                          MotionCaptureInterface(self.SVEA_NAME).start())
        
        # Subscribe to joy
        # convert joy data to velocity and steering
        rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=1)
        self.steering = 0.0
        self.velocity = 0.0

        self.path_lock = Lock()
        self.path, self.waypoint_idx = None, 0

        # Planner
        self.path_timer = rospy.Timer(rospy.Duration(0.1), self.plan_path)

        rospy.logwarn("before APF")
        # Create APF object
        self.apf = ArtificialPotentialFieldHelper(svea_name=self.SVEA_NAME)
        #self.apf.wait_for_local_costmap()
        # Create vehicle model object
        self.model = BicycleModel(initial_state=self.x0, dt=self.DELTA_TIME)
        rospy.logwarn("after bicycl emodel ")
        # Define variable bounds
        x_b = np.array([np.inf, np.inf, 1.2, np.inf])
        u_b = np.array([0.5, np.deg2rad(40)])
        # Create MPC controller object
        self.controller = SMPC(
            self.model,
            N=self.WINDOW_LEN,
            Q=[25, 25, 110, 0],
            R=[1, .5],
            S=[0, 0, 125],
            x_lb=-x_b,
            x_ub=x_b,
            u_lb=-u_b,
            u_ub=u_b,
            n_static_obstacles=self.MAX_N_STATIC_OBSTACLES,
            n_dynamic_obstacles=self.MAX_N_DYNAMIC_OBSTACLES,
            n_pedestrians=self.MAX_N_PEDESTRIANS,
            verbose=False
        )

        rospy.logwarn("done initilization")

    def joy_callback(self, msg):
        steering = msg.axes[0] #steering
        forward = msg.axes[2] #forward
        backward = msg.axes[3] #backward
        steering_input_max = 0.15
        velocity_input_max = 1.0 #min -1.0
        max_steering = 40*np.pi/180
        max_speed = 1.2 #0 - 1.2
        
        steering = min(steering, steering_input_max)
        steering = max(steering, -steering_input_max)
        if backward > forward:
            self.velocity = (-backward - (-velocity_input_max))*(max_speed)/(velocity_input_max*2)-1.2
        else:
            self.velocity = (forward - (-velocity_input_max))*(max_speed)/(velocity_input_max*2)+0
        self.steering = (steering - (-steering_input_max))*(max_steering*2)/(steering_input_max*2)+(-max_steering)

        rospy.loginfo(f"{self.steering}, {self.velocity}")

    def wait_for_state_from_localizer(self):
        """Wait for a new state to arrive, or until a maximum time
        has passed since the last state arrived.

        :return: New state when it arrvies, if it arrives before max
                 waiting time, otherwise None
        :rtype: VehicleState, or None
        """
        time = rospy.get_time()
        if self.last_state_time is None:
            timeout = None
        else:
            timeout = self.MAX_WAIT - (time - self.last_state_time)
        if timeout is None or timeout <= 0:
            return deepcopy(self.state)

        self.localizer.ready_event.wait(timeout)
        wait = rospy.get_time() - time
        if wait < self.MAX_WAIT:
            return deepcopy(self.state)
        else:
            return None

    def plan_path(self, event):
        
        if event.last_expected is not None and event.current_real < event.last_expected:
            return # maybe stupiod

        # Using the Track framework to create a sort of frenet path that starts from current state 
        # and stretches 4 second forward (given current velocity).
        # Curvature K = 1/L * tan(delta)
        # Length d = v * 4
        basewidth = 0.32
        # steering, velocity = self.rc_remote.steering, self.rc_remote.velocity
        steering = self.steering
        velocity = self.velocity

        start_point = (self.state.x, self.state.y, self.state.yaw)

        if abs(velocity) < 0.3:
            path_from_track = np.array([start_point[:2]])
        else:

            arc = Arc(2*velocity, 1/basewidth * np.tan(steering))
            track = Track([arc], *start_point, POINT_DENSITY=10)
            path_from_track = np.array(track.cartesian).T

        # Create array for MPC reference
        path = np.zeros((np.shape(path_from_track)[0], 4))
        path[:, 0] = path_from_track[:, 0]
        path[:, 1] = path_from_track[:, 1]
        path[:, 2] = velocity
        path[:, 3] = 0

        # Get next waypoint index (by computing offset between robot and each point of the path), wrapping it in case of
        # index out of bounds
        self.distances = np.linalg.norm(path[:, 0:2] - self.x0[:2], axis=1)

        with self.path_lock:        
            self.path, self.waypoint_idx = path, np.minimum(self.distances.argmin() + 1, np.shape(path)[0] - 1)

        # Re-initialize path interface to visualize on RVIZ socially aware path
        self.pi.set_points_path(self.path[:, 0:2])

        # Publish global path on rviz
        self.pi.publish_path()
            
    def _visualize_data(self, x_pred, y_pred, velocity, steering):
        """Visualize predicted local tracectory"""
        new_pred = lists_to_pose_stampeds(list(x_pred), list(y_pred))
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        path.poses = new_pred
        self.pred_path_pub.publish(path)

        self.data_handler.log_state(self.state)
        self.data_handler.log_ctrl(steering, velocity, rospy.get_time())
        self.data_handler.update_target((self.path[self.waypoint_idx, 0], self.path[self.waypoint_idx, 1]))
        self.data_handler.visualize_data()

    def get_local_agents(self):
        """
        Function to retrieve agents (i.e. static unmapped obstacles, dynamic obstacles, pedestrians) that are inside the
        local costmap bounds

        :return: local static unmapped obstacles, dynamic obstacles, pedestrians
        :rtype: list[tuple[float]]
        """

        # Get static unmapped obstacles position
        static_unmapped_obs_pos = self.static_unmapped_obs_simulator.obs
        # Initialize array of static unmapped obstacles
        local_static_mpc = np.full((2, self.MAX_N_STATIC_OBSTACLES), -100000.0)
        # Get position of obstacles detected in the local costmap
        static_unmapped_local_obs = self.apf.get_local_obstacles(static_unmapped_obs_pos)
        # Insert them into MPC ready structure
        local_static_mpc[:, 0:np.shape(static_unmapped_local_obs)[0]] = static_unmapped_local_obs.T

        # Initialize array of dynamic obstacles
        local_dynamic_mpc = np.full((4, self.MAX_N_DYNAMIC_OBSTACLES), np.array([[-100000.0, -100000.0, 0, 0]]).T)
        # Acquire mutex
        self.dynamic_obs_simulator.mutex.acquire()
        if (len(self.dynamic_obs_simulator.obs)):
            # Get dynamic obstacle position, v, theta
            dynamic_obs_pose = self.dynamic_obs_simulator.obs[:, 0:4]
            # Release mutex
            self.dynamic_obs_simulator.mutex.release()
            # Get position of obstacles detected in the local costmap
            dynamic_local_obs = self.apf.get_local_obstacles(dynamic_obs_pose)
            # Insert them into MPC structure
            local_dynamic_mpc[:, 0:np.shape(dynamic_local_obs)[0]] = dynamic_local_obs.T
        else:
            # Release mutex as soon as possible
            self.dynamic_obs_simulator.mutex.release()
        
        # Initialize empty pedestrian array
        pedestrians = []
        local_pedestrians_mpc = np.full((4, self.MAX_N_PEDESTRIANS), np.array([[-100000.0, -100000.0, 0, 0]]).T)
        if self.IS_PEDSIM:
            # For every pedestrian, insert it into the array (necessary since in sfm pedestrians are stored in a dict)
            for p in self.sfm_helper.ped_pos:
                pedestrians.append(p)
        else:
            # If mocap is being used
            pedestrians.append([self.sfm_helper.pedestrian_localizer.state.x, self.sfm_helper.pedestrian_localizer.state.y, self.sfm_helper.pedestrian_localizer.state.v, self.sfm_helper.pedestrian_localizer.state.yaw])
        # Keep only pedestrians that are in the local costmap
        local_pedestrians = self.apf.get_local_obstacles(pedestrians)
        # Insert them into MPC structure
        local_pedestrians_mpc[:, 0:np.shape(local_pedestrians)[0]] = local_pedestrians.T

        return local_static_mpc, local_dynamic_mpc, local_pedestrians_mpc

    def keep_alive(self):
        """
        Keep alive function based on the distance to the goal and current state of node

        :return: True if the node is still running, False otherwise
        :rtype: boolean
        """
        return not rospy.is_shutdown()

    def run(self):
        """Run node."""
        while self.keep_alive() and self.path is None:
            pass
        while self.keep_alive():
            self.spin()
            self.rate.sleep()

    def spin(self):
        """Body of main loop."""

        # Get svea state
        if not self.localizer.is_ready: 
            return
        
        # Wait for state from localization interface
        self.state = self.localizer.state
        self.x0 = np.array([self.state.x,
                            self.state.y,
                            self.state.v,
                            self.state.yaw])

        # Get local static unmapped obstacles, local dynamic obstacles, local pedestrians
        local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc = self.get_local_agents()

        # If final distance, don't send control signal
        # Maybe no work, smol hack
        if self.distances[-1] < 0.2:
            return

        with self.path_lock:

            # If there are not enough waypoints for concluding the path, then fill in the waypoints array with the desiderd
            # final goal
            if self.waypoint_idx + self.WINDOW_LEN + 1 >= np.shape(self.path)[0]:
                last_iteration_points = self.path[self.waypoint_idx:, :]
                while np.shape(last_iteration_points)[0] < self.WINDOW_LEN + 1:
                    last_iteration_points = np.vstack((last_iteration_points, self.path[-1, :]))
                u, predicted_state = self.controller.get_ctrl(self.x0, last_iteration_points[:, :].T, local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc)
            else:
                u, predicted_state = self.controller.get_ctrl(self.x0, self.path[self.waypoint_idx:self.waypoint_idx + self.WINDOW_LEN + 1, :].T, local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc)
            #u, predicted_state = self.controller.get_ctrl(self.x0, self.path[self.waypoint_idx, :].T, local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc)

        # Get optimal velocity (by integrating once the acceleration command and summing it to the current speed) and
        # steering controls
        velocity = u[0, 0] * self.DELTA_TIME_REAL + self.x0[2]
        steering = u[1, 0]
        print(f'Optimal control (acceleration, velocity, steering): {u[0, 0], velocity, steering}')

        # Send control to actuator interface
        self.actuation.send_control(steering, velocity)

        # Visualize data on RVIZ
        self._visualize_data(predicted_state[0, :], predicted_state[1, :], velocity, steering)


if __name__ == '__main__':

    ## Start node ##
    SocialAvoidance().run()


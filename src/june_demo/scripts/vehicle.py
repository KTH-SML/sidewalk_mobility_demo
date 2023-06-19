#! /usr/bin/env python3
# Plain python imports
import numpy as np
import rospy
from copy import deepcopy

# SVEA imports
from svea.controllers.social_mpc import SMPC
from svea.sensors import Lidar
from svea.models.bicycle_mpc import BicycleModel
from svea.models.bicycle import SimpleBicycleModel
from svea.states import VehicleState
from svea.simulators.sim_SVEA import SimSVEA
from svea.interfaces import LocalizationInterface, ActuationInterface, PlannerInterface
from svea_mocap.mocap import MotionCaptureInterface
from svea.data import RVIZPathHandler
from svea_social_navigation.apf import ArtificialPotentialFieldHelper
from svea_social_navigation.static_unmapped_obstacle_simulator import StaticUnmappedObstacleSimulator
from svea_social_navigation.dynamic_obstacle_simulator import DynamicObstacleSimulator
from svea_social_navigation.sfm_helper import SFMHelper
from svea_social_navigation.track import Track, Arc

# ROS imports
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Path

def load_param(name, value=None):
    """
    Function used to get parameters from ROS parameter server

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
    """
    Method for publishing initial pose

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

class SocialNavigation(object):
    WINDOW_LEN = 10
    DELTA_TIME = 0.1
    DELTA_TIME_REAL = 0.6
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
        self.IS_SIM = load_param('~is_sim', False)
        self.USE_RVIZ = load_param('~use_rviz', True)
        self.REMOTE_RVIZ = load_param('~remote_rviz', False)
        self.IS_MOCAP = load_param('~is_mocap', True)
        self.STATE = load_param('~state', [0, 0, 0, 0])
        self.SVEA_NAME = load_param('~name', 'svea2')
        self.DEBUG = load_param('~debug', False)
        self.IS_PEDSIM = load_param('~is_pedsim', True)
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

        # Create path using track.py
        self.INTERSECTION_1 = [+2.5, -0.6, +np.pi/2]
        self.CIRCUIT = [
            [1.0, 90],
            [3.0],
            [1.0, 90],
            [1.0, 90],
            [3.0],
            [1.0, 90],
        ]

        self.track = Track([
            Arc(*arc) if len(arc) == 1 else
            Arc.from_circle_segment(*arc)
            for arc in self.CIRCUIT
        ], *self.INTERSECTION_1, POINT_DENSITY=100)
        # Make so the path iterates over itself
        self.track.connects_to(self.track)
        self.path_from_track = np.array(self.track.cartesian).T

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

        if self.IS_SIM:
            # Simulator needs a model to simulate
            self.sim_model = SimpleBicycleModel(self.state)
            # Start the simulator immediately, but paused
            self.simulator = SimSVEA(self.sim_model,
                                     vehicle_name=self.SVEA_NAME,
                                     dt=self.DELTA_TIME/10,
                                     run_lidar=True,
                                     start_paused=True).start()
        else:
            # Start lidar
            self.lidar = Lidar().start()
            # Start actuation interface 
            self.actuation = ActuationInterface().start()
            # Start localization interface based on which localization method is being used
            if self.IS_MOCAP:
                self.localizer = MotionCaptureInterface(self.SVEA_NAME).start()
            else:
                self.localizer = LocalizationInterface().start()

        # Start simulator
        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()

        # Create APF object
        self.apf = ArtificialPotentialFieldHelper(svea_name=self.SVEA_NAME)
        self.apf.wait_for_local_costmap()
        # Create vehicle model object
        self.model = BicycleModel(initial_state=self.x0, dt=self.DELTA_TIME)
        # Define variable bounds
        x_b = np.array([np.inf, np.inf, 0.7, np.inf])
        u_b = np.array([0.5, np.deg2rad(40)])
        # Create MPC controller object
        self.controller = SMPC(
            self.model,
            N=self.WINDOW_LEN,
            Q=[30, 30, 1000, 0],
            R=[1, .5],
            S=[0, 0, 150],
            x_lb=-x_b,
            x_ub=x_b,
            u_lb=-u_b,
            u_ub=u_b,
            n_static_obstacles=self.MAX_N_STATIC_OBSTACLES,
            n_dynamic_obstacles=self.MAX_N_DYNAMIC_OBSTACLES,
            n_pedestrians=self.MAX_N_PEDESTRIANS,
            verbose=False
        )

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

    def plan(self):
        # Create array for MPC reference
        self.path = np.zeros((np.shape(self.path_from_track)[0], 4))
        self.path[:, 0] = self.path_from_track[:, 0]
        self.path[:, 1] = self.path_from_track[:, 1]
        self.path[:, 2] = 0.4
        self.path[:, 3] = 0
        # Init visualize path interface
        self.pi.initialize_path_interface()
        # Re-initialize path interface to visualize on RVIZ socially aware path
        self.pi.set_points_path(self.path[:, 0:2])
        print(f'Social navigation path: {self.path[:, 0:2]} size, {np.shape(self.path)[0]}')

        # Publish global path on rviz
        self.pi.publish_path()
            
    def _visualize_data(self, x_pred, y_pred, velocity, steering):
        # Visualize predicted local tracectory
        new_pred = lists_to_pose_stampeds(list(x_pred), list(y_pred))
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        path.poses = new_pred
        self.pred_path_pub.publish(path)

        # Visualize data
        if self.IS_SIM:
            self.data_handler.log_state(self.sim_model.state)
        else:
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
        # Acquire mutex
        self.sfm_helper.mutex.acquire()
        # If pedsim is being used
        if self.IS_PEDSIM:
            # For every pedestrian, insert it into the array (necessary since in sfm pedestrians are stored in a dict)
            for p in self.sfm_helper.pedestrian_states:
                pedestrians.append(self.sfm_helper.pedestrian_states[p])
        else:
            # If mocap is being used
            pedestrians.append([self.sfm_helper.pedestrian_localizer.state.x, self.sfm_helper.pedestrian_localizer.state.y, self.sfm_helper.pedestrian_localizer.state.v, self.sfm_helper.pedestrian_localizer.state.yaw])
        # Release mutex
        self.sfm_helper.mutex.release()
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
        """
        Run method
        """
        # Plan a feasible path
        self.plan()
        # Spin until alive
        while self.keep_alive():
            self.spin()
            # TODO: check that sleep is necessary when running on real svea
            if not self.IS_SIM:
                rospy.sleep(0.05)
        print('--- GOAL REACHED ---')

    def spin(self):
        """
        Main method
        """
        # Get svea state
        if not self.IS_SIM:
            safe = self.localizer.is_ready
            # Wait for state from localization interface
            self.state = self.localizer.state
            self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]
        else:
            self.x0 = [self.sim_model.state.x, self.sim_model.state.y, self.sim_model.state.v, self.sim_model.state.yaw]
        print(f'State: {self.x0}')

        # Get local static unmapped obstacles, local dynamic obstacles, local pedestrians
        local_static_mpc, local_dynamic_mpc, local_pedestrian_mpc = self.get_local_agents()

        # Get next waypoint index (by computing offset between robot and each point of the path), wrapping it in case of
        # index out of bounds
        self.waypoint_idx = np.minimum(np.argmin(np.linalg.norm(self.path[:, 0:2] - np.array([self.x0[0], self.x0[1]]), axis=1)) + 1, np.shape(self.path)[0] - 1)

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
        if self.IS_SIM:
            velocity = u[0, 0] * self.DELTA_TIME + self.x0[2]
        else:
            velocity = u[0, 0] * self.DELTA_TIME_REAL + self.x0[2]
        steering = u[1, 0]
        print(f'Optimal control (acceleration, velocity, steering): {u[0, 0], velocity, steering}')
        
        # Send control to actuator interface
        if not self.IS_SIM and safe:
            self.actuation.send_control(steering, velocity)

        # If model is simulated, then update new state
        if self.IS_SIM:
            self.sim_model.update(steering, velocity, self.DELTA_TIME)
            
        # Visualize data on RVIZ
        self._visualize_data(predicted_state[0, :], predicted_state[1, :], velocity, steering)


if __name__ == '__main__':
    ## Start node ##
    SocialNavigation().run()
#! /usr/bin/env python3
# Plain python imports
import numpy as np
from math import pi
import rospy

from aug_demo.srv import VerifyState, VerifyStateResponse
from rsu_msgs.msg import StampedObjectPoseArray
from svea_msgs.msg import VehicleState as VehicleStateMsg

from geometry_msgs.msg import PoseStamped

import tf2_ros
import tf2_geometry_msgs
from tf import transformations 
from tf.transformations import euler_from_quaternion, quaternion_from_euler

#
# from odp.shapes import *
# from odp.solver import HJSolver 
# from odp.spect import Grid, SVEA
#

import hj_reachability as hj
import hj_reachability.shapes as shp

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

class LTMS(object):

    PADDING = 0.6

    DEMO_AREA_CENTER = (-2.5, 0) # x, y from sensor_map

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('ltms')

        self.LOCATION = load_param('~location', 'kip')

        #
        # grid = Grid('X Y YAW VEL'.split(),
        #             [+2.0, +2.0, +pi, +0.8],
        #             [-2.0, -2.0, -pi, +0.3],
        #             [31, 31, 13, 7],
        #             [False, False, True, False])
        # self.grid = grid._grid 
        #

        max_bounds = np.array([+2.0, +2.0, +pi, +pi/5, +0.8])
        min_bounds = np.array([-2.0, -2.0, -pi, -pi/5, +0.3])
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (31, 31, 13, 7, 7),
                                                               periodic_dims=2)


        horizon = 2
        t_step = 0.2
        small_number = 1e-5
        self.time_frame = np.arange(start=0, stop=horizon + small_number, step=t_step)

        #
        # model_settings = {'uMin': [-pi/5, -0.2],
        #                   'uMax': [+pi/5, +0.2],
        #                   'dMin': [0.0, 0.0, 0.0, 0.0],
        #                   'dMax': [0.0, 0.0, 0.0, 0.0]}
    

        # self.model = SVEA(ctrl_range=[model_settings['uMin'],
        #                               model_settings['uMax']],
        #                   dstb_range=[model_settings['dMin'],
        #                               model_settings['dMax']],
        #                   mode='reach')
        #

        # 
        self.avoid_dynamics = hj.systems.SVEA5D(min_steer=-pi/5, 
                                   max_steer=+pi/5,
                                   min_accel=-0.2,
                                   max_accel=+0.2).with_mode('reach')
        #

        #
        # self.solver = HJSolver(self.grid, self.time_frame, self.model)
        #

        #
        self.solver_settings = hj.SolverSettings.with_accuracy("high")
        #

        self.peds = []
        self.peds_sub = rospy.Subscriber('sensor/objects', StampedObjectPoseArray, self.peds_cb)

        self.verify_state = rospy.Service('ltms/verify_state', VerifyState, self.verify_state_srv)
        
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

        rospy.loginfo('Start')

    def peds_cb(self, msg):
        self.peds = []
        for obj in msg.objects:
            if obj.object.label == 'person':
                self.peds.append((msg.header, obj.pose.pose.position.x, obj.pose.pose.position.y))

    def compute_brs(self, target):
        # target = np.concatenate([target[..., np.newaxis]] * len(self.time_frame), axis=-1)
        # return self.solver(target=target, target_mode='min')
    
        target = shp.make_tube(self.time_frame, target)
        values = hj.solve(self.solver_settings, self.avoid_dynamics, self.grid,
                        self.time_frame, target, None)
        values = np.asarray(values)
        values = np.flip(values, axis=0)

    def verify_state_srv(self, req):
        if self.LOCATION == 'sml':
            map_state = req.state
            delta = req.delta

        # utm_state is either in utm frame or mocap frame, depending on location
        if self.LOCATION == 'kip':
            utm_state = req.state
            state_pose = state_to_pose(utm_state)
            trans = self.buffer.lookup_transform("sensor_map", utm_state.header.frame_id, rospy.Time.now(), rospy.Duration(0.5))
            svea_pose = tf2_geometry_msgs.do_transform_pose(state_pose, trans)
            map_state = pose_to_state(svea_pose)
            map_state.v  = utm_state.v

            map_state.x += self.DEMO_AREA_CENTER[0]
            map_state.y += self.DEMO_AREA_CENTER[1]


            map_state.x += -0.16803447571899888 -  -1.7746380530297756
            map_state.y += +0.7996012858407618 - 1.850046221166849
            
        # return VerifyStateResponse(True)

        peds = []
        for ped_header, x, y in self.peds:
            
            if self.LOCATION == 'kip':
                x += self.DEMO_AREA_CENTER[0]
                y += self.DEMO_AREA_CENTER[1]

            # if not req.state.header.frame_id == ped_header.frame_id:
            #     print(f'Warning! vehicle frame not same as pedestrian frame ({req.state.header.frame_id} != {ped_header.frame_id})')
            #     continue
            # if not self.grid.grid_points[0][0] < x < self.grid.grid_points[0][-1]:
            #     continue
            # if not self.grid.grid_points[1][0] < y < self.grid.grid_points[1][-1]:
            #     continue
            #
            if not self.grid.coordinate_vectors[0][0] < x < self.grid.coordinate_vectors[0][-1]:
                continue
            if not self.grid.coordinate_vectors[1][0] < y < self.grid.coordinate_vectors[1][-1]:
                continue
            #

            #
            # ped = intersection(lower_half_space(self.grid, 0, x + self.PADDING), 
            #                    upper_half_space(self.grid, 0, x - self.PADDING),
            #                    lower_half_space(self.grid, 1, y + self.PADDING), 
            #                    upper_half_space(self.grid, 1, y - self.PADDING))
            # peds.append(ped)
            #

            #
            ped = shp.intersection(shp.lower_half_space(self.grid, 0, x + self.PADDING), 
                               shp.upper_half_space(self.grid, 0, x - self.PADDING),
                               shp.lower_half_space(self.grid, 1, y + self.PADDING), 
                               shp.upper_half_space(self.grid, 1, y - self.PADDING))
            peds.append(ped)
            #
            
            print('ped distance:', np.hypot(x - map_state.x, y - map_state.y),
                  'ped:', (x, y), 
                  'svea:', (map_state.x, map_state.y))

        #
        # target = union(*peds) if peds else np.ones(self.grid.shape)
        #
        target = shp.union(*peds) if peds else np.ones(self.grid.shape)
        #

        result = self.compute_brs(target=target)
        #
        # result = -result.min(axis=(0,3,4,5)) # -shp.project_onto(vf, 1, 2)
        #
        result = -shp.project_onto(result, 1, 2)
        #

        #
        # ix = np.abs(self.grid.grid_points[0] - map_state.x).argmin()
        # iy = np.abs(self.grid.grid_points[1] - map_state.y).argmin()
        # iyaw = np.abs(self.grid.grid_points[2] - map_state.yaw).argmin()
        # ivel = np.abs(self.grid.grid_points[3] - map_state.v).argmin()
        # ok = bool(result[ix, iy, iyaw, ivel] <= 0)
        #
        ix = np.abs(self.grid.coordinate_vectors[0] - map_state.x).argmin()
        iy = np.abs(self.grid.coordinate_vectors[1] - map_state.y).argmin()
        iyaw = np.abs(self.grid.coordinate_vectors[2] - map_state.yaw).argmin()
        id = np.abs(self.grid.coordinate_vectors[3] - delta).argmin()
        ivel = np.abs(self.grid.coordinate_vectors[4] - map_state.v).argmin()
        ok = bool(result[ix, iy, iyaw, id, ivel] <= 0)
        #

        return VerifyStateResponse(ok=ok)

    def keep_alive(self):
        """
        Keep alive function based on the distance to the goal and current state of node

        :return: True if the node is still running, False otherwise
        :rtype: boolean
        """
        return not rospy.is_shutdown()

    def run(self):
        rate = rospy.Rate(0.2)
        target = shp.intersection(shp.lower_half_space(self.grid, 0, 0 + self.PADDING), 
                            shp.upper_half_space(self.grid, 0, 0 - self.PADDING),
                            shp.lower_half_space(self.grid, 1, 0 + self.PADDING), 
                            shp.upper_half_space(self.grid, 1, 0 - self.PADDING))
        while True:
            result = self.compute_brs(target=target)
            result = -shp.project_onto(result, 1, 2)

            rate.Sleep()
        # rospy.spin()


if __name__ == '__main__':

    ## Start node ##
    LTMS().run()


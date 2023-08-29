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

from odp.shapes import *
from odp.solver import HJSolver 
from odp.spect import Grid, SVEA

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

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('ltms')

        self.LOCATION = load_param('~location', 'kip')

        grid = Grid('X Y YAW VEL'.split(),
                    [+2.0, +2.0, +pi, +0.8],
                    [-2.0, -2.0, -pi, +0.3],
                    [31, 31, 13, 7],
                    [False, False, True, False])
        self.grid = grid._grid 

        horizon = 2
        t_step = 0.2
        small_number = 1e-5
        self.time_frame = np.arange(start=0, stop=horizon + small_number, step=t_step)

        model_settings = {'uMin': [-pi/5, -0.2],
                          'uMax': [+pi/5, +0.2],
                          'dMin': [0.0, 0.0, 0.0, 0.0],
                          'dMax': [0.0, 0.0, 0.0, 0.0]}

        self.model = SVEA(ctrl_range=[model_settings['uMin'],
                                      model_settings['uMax']],
                          dstb_range=[model_settings['dMin'],
                                      model_settings['dMax']],
                          mode='reach')

        self.solver = HJSolver(self.grid, self.time_frame, self.model)
        
        self.peds = []
        self.peds_sub = rospy.Subscriber('sensor/objects', StampedObjectPoseArray, self.peds_cb)

        self.verify_state = rospy.Service('ltms/verify_state', VerifyState, self.verify_state_srv)
        
        self.state_pub = rospy.Publisher('state_in_map', VehicleStateMsg, queue_size=1)
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
        target = np.concatenate([target[..., np.newaxis]] * len(self.time_frame), axis=-1)
        return self.solver(target=target, target_mode='min')

    def verify_state_srv(self, req):

        state = req.state
        state_pose = state_to_pose(state)
        trans = self.buffer.lookup_transform("sensor_map", state.header.frame_id, rospy.Time.now(), rospy.Duration(0.5))
        pose = tf2_geometry_msgs.do_transform_pose(state_pose, trans)
        new_state = pose_to_state(pose)
        new_state.v = state.v

        self.state_pub.publish(new_state)


        return VerifyStateResponse(True)

        peds = []
        for ped_header, x, y in self.peds:
            if not req.state.header.frame_id == ped_header.frame_id:
                print(f'Warning! vehicle frame not same as pedestrian frame ({req.state.header.frame_id} != {ped_header.frame_id})')
                continue
            if not self.grid.grid_points[0][0] < x < self.grid.grid_points[0][-1]:
                continue
            if not self.grid.grid_points[1][0] < x < self.grid.grid_points[1][-1]:
                continue
            ped = intersection(lower_half_space(self.grid, 0, x + self.PADDING), 
                               upper_half_space(self.grid, 0, x - self.PADDING),
                               lower_half_space(self.grid, 1, y + self.PADDING), 
                               upper_half_space(self.grid, 1, y - self.PADDING))
            peds.append(ped)

        target = union(*peds) if peds else np.ones(self.grid.shape)

        result = self.compute_brs(target=target)
        result = -result.min(axis=-1)

        ix = np.abs(self.grid.grid_points[0] - req.state.x).argmin()
        iy = np.abs(self.grid.grid_points[1] - req.state.y).argmin()
        iyaw = np.abs(self.grid.grid_points[2] - req.state.yaw).argmin()
        ivel = np.abs(self.grid.grid_points[3] - req.state.yaw).argmin()
        ok = bool(result[ix, iy, iyaw, ivel] <= 0)

        return VerifyStateResponse(ok=ok)

    def keep_alive(self):
        """
        Keep alive function based on the distance to the goal and current state of node

        :return: True if the node is still running, False otherwise
        :rtype: boolean
        """
        return not rospy.is_shutdown()

    def run(self):
        rospy.spin()


if __name__ == '__main__':

    ## Start node ##
    LTMS().run()


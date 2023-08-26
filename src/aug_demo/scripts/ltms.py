#! /usr/bin/env python3
# Plain python imports
import numpy as np
from math import pi
import rospy

from aug_demo.srv import RefinePath, RefinePathResponse
from rsu_msgs.msg import StampedObjectPoseArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from odp.shapes import *
from odp.solver import HJSolver 
from odp.spect import Grid, SVEA 


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

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('ltms')

        # Get parameters
        self.NAME = load_param('~name', 'svea2')
        self.ORIGIN = load_param('~origin', [0, 0, 0])

        grid = Grid('X Y YAW VEL'.split(),
                    [+2.0, +2.0, +pi, +1.0],
                    [-2.0, -2.0, -pi, +0.0],
                    [41, 41, 31, 11],
                    [False, False, True, False])
        self.grid = grid._grid 

        horizon = 2
        t_step = 0.2
        small_number = 1e-5
        self.time_frame = np.arange(start=0, stop=horizon + small_number, step=t_step)

        reach_scenario = {'uMode': 'min', 'dMode': 'max'}
        model_settings = {**reach_scenario, 
                        'uMin': [-pi/5, -0.2],
                        'uMax': [+pi/5, +0.2],
                        'dMin': [0, 0, 0, 0],
                        'dMax': [0, 0, 0, 0]}

        self.model = SVEA(ctrl_range=[model_settings['uMin'],
                                      model_settings['uMax']],
                          dstb_range=[model_settings['dMin'],
                                      model_settings['dMax']],
                          mode='reach')
        
        self.peds_sub = rospy.Subscriber(self.ped_pos_topic, StampedObjectPoseArray, self.peds_cb)

        self.refine_path = rospy.Service('ltms/refine_path', RefinePath, self.refine_path_srv)
        
        rospy.loginfo('Start')

    def peds_cb(self, msg):
        self.peds = []
        for obj in msg.objects:
            if obj.object.label == 'person':
                self.ped_pos.append((obj.pose.pose.position.x, obj.pose.pose.position.y))

    def compute_brs(self, target, constraint):

        target = np.concatenate([target[..., np.newaxis]] * len(self.time_frame), axis=-1)
        constraint = np.concatenate([constraint[..., np.newaxis]] * len(self.time_frame), axis=-1)

        solver = HJSolver(self.grid, self.time_frame, self.model, 
                        target=target, target_mode='min',
                        constraint=constraint, constraint_mode='max')

        return solver()

    def refine_path_srv(self, req):

        target = np.zeros(self.grid.shape)

        here_pose = req.path.poses[0]
        goal_pose = req.path.poses[-1]

        here_point = (here_pose.pose.position.x, here_pose.pose.position.y)
        goal_point = (goal_pose.pose.position.x, goal_pose.pose.position.y)
        
        constraint = np.zeros(self.grid.shape)

        for x, y in self.peds:
            x += self.ORIGIN[0]
            y += self.ORIGIN[1]
            ped = intersect(upper_half_space(self.grid, 0, x + 0.5), lower_half_space(self.grid, 0, x - 0.5),
                            upper_half_space(self.grid, 1, y + 0.5), lower_half_space(self.grid, 1, y - 0.5))
            constraint = union(constraint, ped)

        result = self.compute_brs(target, constraint)

        ## TODO:
        # I should do FRS(poses[0]) intersect BRS(poses[-1])

        point_list = []
        for i in range(result.shape[-1]):
            t = req.path.header.stamp + self.time_frame[i]
            idx = np.unravel_index(result[..., i].argmin(), result.shape)

            if result[idx] > 0:
                break

            x = idx[0] * self.grid.dx[0] - self.ORIGIN[0]
            y = idx[1] * self.grid.dx[1] - self.ORIGIN[1]

            point_list.append((t, x, y))
        
        path = Path()
        path.header = req.path.header
        path.poses = []

        for pose, (t, x, y) in zip(req.path.poses, point_list):
            if t < pose.header.stamp: 
                continue
            pose.header.stamp = t
            pose.pose.position.x = x
            pose.pose.position.y = y
            path.poses.append(pose)

        return RefinePathResponse(path)

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


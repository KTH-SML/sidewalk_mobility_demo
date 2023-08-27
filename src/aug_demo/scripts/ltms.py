#! /usr/bin/env python3
# Plain python imports
import numpy as np
from math import pi
import rospy

from aug_demo.srv import VerifyState, VerifyStateResponse
from rsu_msgs.msg import StampedObjectPoseArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from odp.shapes import *
from odp.solver import HJSolver 
from odp.spect import Grid, SVEA

class LTMS(object):

    PADDING = 0.3

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('ltms')

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

        # return VerifyStateResponse(True)

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

        return VerifyStateResponse(ok=bool(result[ix, iy, iyaw, ivel] <= 0))

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


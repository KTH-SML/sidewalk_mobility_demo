#! /usr/bin/env python3
# Plain python imports
import numpy as np
from math import pi
import rospy

from aug_demo.srv import VerifyState, VerifyStateResponse
from rsu_msgs.msg import StampedObjectPoseArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# from odp.shapes import *
# from odp.solver import HJSolver 
# from odp.spect import Grid, SVEA

class LTMS(object):

    PADDING = 0.3

    def __init__(self):
        """Init method for SocialNavigation class."""

        rospy.init_node('ltms')

        # grid = Grid('X Y YAW VEL'.split(),
        #             [+2.0, +2.0, +pi, +1.0],
        #             [-2.0, -2.0, -pi, +0.3],
        #             [31, 31, 13, 7],
        #             [False, False, True, False])
        # self.grid = grid._grid 

        # horizon = 2
        # t_step = 0.2
        # small_number = 1e-5
        # self.time_frame = np.arange(start=0, stop=horizon + small_number, step=t_step)

        # model_settings = {'uMin': [-pi/5, -0.2],
        #                   'uMax': [+pi/5, +0.2],
        #                   'dMin': [0, 0, 0, 0],
        #                   'dMax': [0, 0, 0, 0]}

        # self.model = SVEA(ctrl_range=[model_settings['uMin'],
        #                               model_settings['uMax']],
        #                   dstb_range=[model_settings['dMin'],
        #                               model_settings['dMax']],
        #                   mode='avoid')

        # self.solver = HJSolver(self.grid, self.time_frame, self.model)
        
        self.peds_sub = rospy.Subscriber(self.ped_pos_topic, StampedObjectPoseArray, self.peds_cb)

        self.verify_state = rospy.Service('ltms/verify_state', VerifyState, self.verify_state_srv)
        
        rospy.loginfo('Start')

    def peds_cb(self, msg):
        self.peds = []
        for obj in msg.objects:
            if obj.object.label == 'person':
                self.ped_pos.append((obj.pose.pose.position.x, obj.pose.pose.position.y))

    def compute_brs(self, target):
        target = np.concatenate([target[..., np.newaxis]] * len(self.time_frame), axis=-1)
        return self.solver(target=target, target_mode='min')

    def verify_state_srv(self, req):

        return VerifyStateResponse(True)

        peds = []
        for x, y in self.peds:
            if not self.grid.grid_points[0][0] < x < self.grid.grid_points[0][-1]:
                continue
            if not self.grid.grid_points[1][0] < x < self.grid.grid_points[1][-1]:
                continue
            ped = intersect(upper_half_space(self.grid, 0, x + self.PADDING), 
                            lower_half_space(self.grid, 0, x - self.PADDING),
                            upper_half_space(self.grid, 1, y + self.PADDING), 
                            lower_half_space(self.grid, 1, y - self.PADDING))
            peds.append(ped)

        result = self.compute_brs(target=union(*peds))
        result = -np.minimum(result, axis=-1)

        ix = (self.grid.grid_points[0] - req.state.x).abs().argmin()
        iy = (self.grid.grid_points[1] - req.state.y).abs().argmin()
        iyaw = (self.grid.grid_points[2] - req.state.yaw).abs().argmin()
        ivel = (self.grid.grid_points[3] - req.state.yaw).abs().argmin()
        
        return VerifyStateResponse(result[ix, iy, iyaw, ivel] < 0)

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


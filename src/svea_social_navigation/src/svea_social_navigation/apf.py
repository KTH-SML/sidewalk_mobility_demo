#! /usr/bin/env python3
# Plain python imports
from typing import Any
import numpy as np
from scipy import spatial
import rospy

# ROS imports
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class ArtificialPotentialFieldHelper(object):
    """
    Class that given a local costmap computes the APF (only related to obstacles) for given robot 
    """
    def __init__(self, svea_name="svea2"):
        """
        Init method for the ArtificialPotentialField class
        """
        self.svea_name = svea_name
        # Get topics names
        self.map_topic = load_param('~local_costmap_topic', '/costmap_node/costmap/costmap')
        self.waypoint_topic = load_param('~waypoint_topic', '/target')

        # Init subscribers
        self.map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self._local_costmap_cb, queue_size=1)
        self.waypoint_sub = rospy.Subscriber(self.waypoint_topic, PointStamped, self._waypoint_cb, queue_size=1)

        #  Waypoint variables
        self._waypoint = None

        # Costmap variables
        self._local_costmap = None
        self._map_width = None
        self._map_height = None
        self._map_x = None
        self._map_y = None
        self._map_resolution = None

    def _waypoint_cb(self, msg):
        """
        Callback method for the waypoint subscriber

        :param msg: waypoint message
        :type msg: PointStamped
        """
        self._waypoint = np.array([msg.point.x, msg.point.y])

    def _local_costmap_cb(self, msg):
        """
        Callback method for the local costmap subscriber

        :param msg: map message
        :type msg: OccupancyGrid
        """
        self._map_width = msg.info.width
        self._map_height = msg.info.height
        self._map_resolution = msg.info.resolution
        self._map_x = msg.info.origin.position.x
        self._map_y = msg.info.origin.position.y
        self._local_costmap = np.reshape(msg.data, (self._map_height, self._map_width))
        
    def get_local_obstacles(self, obs):
        """
        Function to retrieve detected obstacles that are in the local map

        :param obs: list of obstacles
        :type obs: list[tuple[float]]
        :return: obstacles that are inside the local costmap
        :rtype: list[tuple[float]]
        """
        return np.array([])
        map_x_limit = self._map_x + self._map_resolution * self._map_width
        map_y_limit = self._map_y + self._map_resolution * self._map_height
        loc_obs = []
        for p in obs:
            if self._map_x < p[0] < map_x_limit and self._map_y < p[1] < map_y_limit:
                loc_obs.append(p)
        return np.array(loc_obs)
    
    def get_map_dimensions(self):
        """
        Function to retrieve the map's dimensions

        :return: maps dimensions
        :rtype: int, int
        """
        if self._map_height is not None and self._map_width is not None:
            return self._map_height, self._map_width
        
    def wait_for_local_costmap(self):
        while self._local_costmap is None:
            rospy.sleep(1)
        

    
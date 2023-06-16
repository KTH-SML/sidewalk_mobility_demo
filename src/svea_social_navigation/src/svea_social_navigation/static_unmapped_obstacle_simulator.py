#!/usr/bin/env python
import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray

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

class StaticUnmappedObstacleSimulator(object):
    def __init__(self, obs):
        """
        Init method for class StaticUnmappedObstacleSimulator 

        :param obs: list of obstacle positions
        :type obs: list[tuple[float]]
        """
        self.TOPIC_NAME = load_param('~static_unmapped_obstacle_topic', '/static_unmapped_obstacles')
        self.pub = rospy.Publisher(self.TOPIC_NAME, MarkerArray, queue_size=1, latch=True)
        self.obs = np.array(obs)
        self.ns = 'static_unmapped_obstacle_simulator'
        self.r = 0.0
        self.g = 1.0
        self.b = 0.0

    def create_marker_array(self):
        """
        Function to create an array of Markers

        :return: array of markers
        :rtype: list[Marker]
        """
        return [Marker()] * np.shape(self.obs)[0]
    
    def create_marker(self, x, y, id):
        """
        Function to create a single marker

        :param x: x position 
        :type x: float
        :param y: y position
        :type y: float
        :param id: id 
        :type id: integer
        :return: marker
        :rtype: Marker
        """
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        m.ns = self.ns
        m.id = id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.color.a = 1.0 
        m.color.r = self.r
        m.color.g = self.g
        m.color.b = self.b
        return m
    
    def publish_obstacle_msg(self):
        """
        Method to publish the array of markers
        """
        obstacle_msg = MarkerArray()
        obstacle_msg.markers = self.create_marker_array()
        for i in range(np.shape(self.obs)[0]):
            obstacle_msg.markers[i] = self.create_marker(self.obs[i, 0], self.obs[i, 1] ,i)
        self.pub.publish(obstacle_msg)


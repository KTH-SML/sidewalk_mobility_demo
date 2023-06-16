#!/usr/bin/env python
import rospy
import numpy as np
from svea_social_navigation.static_unmapped_obstacle_simulator import StaticUnmappedObstacleSimulator
from visualization_msgs.msg import MarkerArray
from threading import *

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

class DynamicObstacleSimulator(StaticUnmappedObstacleSimulator):
    def __init__(self, obs, dt):
        """
        Init method for class DynamicObstacleSimulator 

        :param dt: delta time
        :type dt: float
        :param obs: list of x, y, velocity, heading, x lower bound, x upper bound, y lower bound, y upper bound
        :type obs: list[tuple[float]]
        """
        super().__init__(obs)
        self.TOPIC_NAME = load_param('~dynamic_obstacle_topic', '/dynamic_obstacles')
        self.pub = rospy.Publisher(self.TOPIC_NAME, MarkerArray, queue_size=1, latch=True)
        self.dt = dt
        self.ns = 'dynamic_obstacle_simulator'
        self.r = 0.0
        self.g = 0.5
        self.b = 1.0
        self.mutex = Lock()
    
    def publish_obstacle_msg(self):
        """
        Method to publish the array of markers
        """
        obstacle_msg = MarkerArray()
        obstacle_msg.markers = self.create_marker_array()
        # Create Markers
        for i in range(np.shape(self.obs)[0]):
            obstacle_msg.markers[i] = self.create_marker(self.obs[i, 0], self.obs[i, 1], i)
        while not rospy.is_shutdown():
            for i in range(np.shape(self.obs)[0]):
                # Compute delta_x and delta_y
                delta_x = self.obs[i, 2] * np.cos(self.obs[i, 3]) * self.dt
                delta_y = self.obs[i, 2] * np.sin(self.obs[i, 3]) * self.dt
                # Get mutex
                self.mutex.acquire()
                # Update new positions of both x and y
                self.obs[i, 0] += delta_x
                self.obs[i, 1] += delta_y
                # Check condition on x and y bounds
                if (self.obs[i, 0] < self.obs[i, 4] or self.obs[i, 0] > self.obs[i, 5]) or (self.obs[i, 1] < self.obs[i, 6] or self.obs[i, 1] > self.obs[i, 7]):
                    if np.isclose(self.obs[i, 3], 0):
                        self.obs[i, 3] = np.pi
                    elif np.isclose(self.obs[i, 3], np.pi):
                        self.obs[i, 3] = 0
                    else:
                        self.obs[i, 3] = -self.obs[i, 3]
                # Release mutex
                self.mutex.release()
                # Create Marker object with new
                obstacle_msg.markers[i].pose.position.x = self.obs[i, 0]
                obstacle_msg.markers[i].pose.position.y = self.obs[i, 1]
            self.pub.publish(obstacle_msg)
            rospy.sleep(self.dt)

    def thread_publish(self):
        """
        Method for creating and starting a thread, so that dynamic obstacles move asynchronously wrt other software components
        """
        t = Thread(target=self.publish_obstacle_msg)
        t.start()


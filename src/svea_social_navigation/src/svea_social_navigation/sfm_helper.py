#!/usr/bin/env python
import rospy
import numpy as np
from threading import Lock

# Import AgentStates message
from rsu_msgs.msg import StampedObjectPoseArray
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class SFMHelper(object):
    def __init__(self, is_pedsim=True):
        """
        Init method for the SFMHelper class
        """
        # Get pedestrian topic
        self.ped_pos_topic = load_param('~pedestrian_position_topic', '/sensor/objects') 
        # Create subscriber
        self.ped_sub = rospy.Subscriber(self.ped_pos_topic, StampedObjectPoseArray, self.pedestrian_cb)
        # Empty array of pedestrain positions
        self.pedestrian_states = {}
        # Pedestrian position array
        self.ped_pos = []
        # Marker array publisher for visualization purposes
        self.pub = rospy.Publisher('/sensor/markers', MarkerArray, queue_size=1, latch=True)
        self.r = 0.0
        self.g = 1.0
        self.b = 0.0
        self.ns = 'sensor'
        # Mutex for mutual exclusion over the access on pedestrian_states
        self.mutex = Lock()

    def pedestrian_cb(self, msg):
        """
        Callback method for the pedestrian state subscriber

        :param msg: message
        :type msg: AgentStates
        """
        ped_detected = False
        # For every agent in the environment
        for obj in msg.objects:
            if obj.object.label == 'person':
                ped_detected = True
                # Transform quaternion into RPY angles
                r, p, y = euler_from_quaternion([obj.pose.pose.orientation.x, obj.pose.pose.orientation.y, obj.pose.pose.orientation.z, obj.pose.pose.orientation.w])
                # Suppose 0 speed for every pedestrian
                v = 0
                # Create state array
                state = [obj.pose.pose.position.x, obj.pose.pose.position.y, v, y]
                # Acquire mutex
                self.mutex.acquire()
                # Updata/insert entry in pedestrian states array 
                self.pedestrian_states.update({0: state})
                # Release mutex
                self.mutex.release()        
                self.ped_pos = [obj.pose.pose.position.x, obj.pose.pose.position.y]
        if not ped_detected:
            self.pedestrian_states.pop(0)
        print(self.pedestrian_states)
        self.publish_obstacle_msg()

    def create_marker_array(self):
        """
        Function to create an array of Markers

        :return: array of markers
        :rtype: list[Marker]
        """
        return [Marker()] * np.shape(self.ped_pos)[0]
    
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
        obstacle_msg.markers[0] = self.create_marker(self.ped_pos[0], self.ped_pos[1], 0)
        self.pub.publish(obstacle_msg)
        
    
if __name__ == '__main__':
    rospy.init_node('test')
    sfm = SFMHelper()
    rospy.spin()

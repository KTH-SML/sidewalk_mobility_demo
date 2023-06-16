#!/usr/bin/env python
import rospy
import numpy as np
from threading import Lock

# Import AgentStates message
from pedsim_msgs.msg import AgentStates
from tf.transformations import euler_from_quaternion

# Import mocap interface for detecting pedestrian position
from svea_mocap.mocap import MotionCaptureInterface

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
        self.ped_pos_topic = load_param('~pedestrian_position_topic', '/pedsim_simulator/simulated_agents')
        self.IS_PEDSIM = is_pedsim
        if self.IS_PEDSIM: 
            # Create subscriber
            self.ped_sub = rospy.Subscriber(self.ped_pos_topic, AgentStates, self.pedestrian_cb)
            # Empty array of pedestrain positions
            self.pedestrian_states = {}
        else:
            self.pedestrian_localizer = MotionCaptureInterface('pedestrian').start()
        # Mutex for mutual exclusion over the access on pedestrian_states
        self.mutex = Lock()

    def pedestrian_cb(self, msg):
        """
        Callback method for the pedestrian state subscriber

        :param msg: message
        :type msg: AgentStates
        """
        # For every agent in the environment
        for agent in msg.agent_states:
            # Transform quaternion into RPY angles
            r, p, y = euler_from_quaternion([agent.pose.orientation.x, agent.pose.orientation.y, agent.pose.orientation.z, agent.pose.orientation.w])
            # Compute speed given linear twist
            v = np.sqrt(agent.twist.linear.x**2 + agent.twist.linear.y**2)
            # Create state array
            state = [agent.pose.position.x, agent.pose.position.y, v, y]
            # Acquire mutex
            self.mutex.acquire()
            # Updata/insert entry in pedestrian states array 
            self.pedestrian_states.update({agent.id: state})
            # Release mutex
            self.mutex.release()
        
    
if __name__ == '__main__':
    rospy.init_node('test')
    sfm = SFMHelper()
    rospy.spin()

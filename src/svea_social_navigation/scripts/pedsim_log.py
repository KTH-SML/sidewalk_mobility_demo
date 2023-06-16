#! /usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from svea_social_navigation.measurement_node import SocialMeasurement
from tf.transformations import euler_from_quaternion

goal_x = 13.0
goal_y = 20.0

measurements = SocialMeasurement(write=True, pedsim=True)

def robot_pose_cb(msg):
    if np.sqrt((msg.pose.pose.position.x - goal_x) ** 2 + (msg.pose.pose.position.y - goal_y) ** 2) < 0.5:
        print('KILLING NODE')
        rospy.signal_shutdown('Goal Reached, stop logging')
    r, p, y = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    speed = np.sqrt(msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2)
    x0 = [msg.pose.pose.position.x, msg.pose.pose.position.y, speed, y]
    measurements.add_robot_pose(x0, rospy.get_time())

def pedestrian_pose_cb(msg):
    for agent in msg.agent_states:
        r, p, y = euler_from_quaternion([agent.pose.orientation.x, agent.pose.orientation.y, agent.pose.orientation.z, agent.pose.orientation.w])
        v = np.sqrt(agent.twist.linear.x**2 + agent.twist.linear.y**2)
        state = [agent.pose.position.x, agent.pose.position.y, v, y]
        measurements.add_pedestrian_pose(agent.id - 1, state, rospy.get_time())

if __name__ == '__main__':
    rospy.init_node('pedsim_log')
    rospy.Subscriber('/pedsim_simulator/robot_position', Odometry, robot_pose_cb)
    rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, pedestrian_pose_cb)
    rospy.spin()
    measurements.close_files()




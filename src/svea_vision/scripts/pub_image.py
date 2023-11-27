#! /usr/bin/env python3

import os
import cv2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class PublishImage:

    def __init__(self):
        try:
            # Initialize node
            rospy.init_node('publish_image', anonymous=True)
            
            # Parameters
            self.image_topic = load_param('~image_topic', 'image')
            self.image_path = load_param('~image_path')
            self.rate = load_param('~rate', 30)

            # CV Bridge
            self.cv_bridge = CvBridge()

            # Publisher
            self.img_pub = rospy.Publisher(self.image_topic, Image, queue_size=1)

        except Exception as e:
            # Log error
            rospy.logerr(e)

        else:
            # Log status
            rospy.loginfo('{} node initialized.'.format(rospy.get_name()))

    def run(self):
        # Load image
        img = cv2.imread(self.image_path)
        img_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding='bgr8')

        # Publish and sleep loop
        rate = rospy.Rate(self.rate)
        try:
            while not rospy.is_shutdown():
                self.img_pub.publish(img_msg)
                rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down {}'.format(rospy.get_name()))


if __name__ == "__main__":
    node = PublishImage()
    node.run()
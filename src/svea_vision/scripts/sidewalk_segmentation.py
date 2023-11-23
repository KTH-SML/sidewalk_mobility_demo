#! /usr/bin/env python3

import os
import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

import cv2
import numpy as np

from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def replace_base(old, new) -> str:
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith('~')
    is_global = new.startswith('/')
    assert not (is_private or is_global)
    ns, _ = split_last(old.split('/'))
    ns += new.split('/')
    return '/'.join(ns)


class SidewalkSegementation:
    
    def __init__(self) -> None:
        try:
            # Initialize node
            rospy.init_node('sidewalk_segmentation')
            
            # Parameters
            self.rgb_topic = load_param('~rgb_topic', 'image')
            self.camera_info_topic = replace_base(self.rgb_topic, 'camera_info')
            
            self.depth_topic = load_param('~depth_topic', 'depth')
            self.pointcloud_topic = load_param('~pointcloud_topic', 'pointcloud')
            
            self.sidewalk_mask_topic = load_param('~sidewalk_mask_topic', 'sidewalk_mask')
            self.sidewalk_ann_topic = load_param('~sidewalk_ann_topic', 'sidewalk_ann')
            
            self.image_width = load_param('~image_width', 640)
            self.image_height = load_param('~image_height', 480)
            
            self.model_name = load_param('~model_name', 'FastSAM-x.pt') # FastSAM-s.pt or FastSAM-x.pt
            self.use_cuda = load_param('~use_cuda', False)
            self.conf = load_param('~conf', 0.4)
            self.iou = load_param('~iou', 0.9)
            
            # Get package path
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('svea_vision')
            
            # Load model
            self.device = 'cuda' if self.use_cuda else 'cpu'
            self.model_path = os.path.join(package_path, 'models', self.model_name)
            self.model = FastSAM(self.model_path)
            if self.use_cuda:
                self.model.to('cuda')
                rospy.loginfo('{}: CUDA enabled'.format(rospy.get_name()))
            else:
                rospy.loginfo('{}: CUDA disabled'.format(rospy.get_name()))
            
            # CV Bridge
            self.cv_bridge = CvBridge()
            
            # Publishers
            self.sidewalk_mask_pub = rospy.Publisher(self.sidewalk_mask_topic, Image, queue_size=10)
            self.sidewalk_ann_pub = rospy.Publisher(self.sidewalk_ann_topic, Image, queue_size=10)
            
            # Subscribers
            rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=10)
            # rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
            # rospy.Subscriber(self.pointcloud_topic, Image, self.pointcloud_callback)
            # rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        
        except Exception as e:
            rospy.logerr(e)

        else:
            rospy.loginfo('{} node initialized with model: {}'.format(rospy.get_name(), self.model_name))
            
    def run(self):
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down {}'.format(rospy.get_name()))

    def rgb_callback(self, msg):
        # Convert ROS image to OpenCV image
        image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # Run inference on the image
        everything_results = self.model(image, device=self.device, imgsz=self.image_width,
                                        conf=self.conf, iou=self.iou, retina_masks=True, verbose=False)
        
        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        
        # Text prompt
        sidewalk_results = prompt_process.text_prompt(text='a sidewalk or footpath or walkway or paved path for humans to walk on')
        
        # Get mask
        sidewalk_mask = sidewalk_results[0].cpu().numpy().masks.data[0].astype('uint8')*255
        
        # Get annotated image
        sidewalk_ann = sidewalk_results[0].plot(masks=True, conf=False, kpt_line=False,
                                                labels=False, boxes=False, probs=False)
        
        # Convert OpenCV image to ROS image
        mask_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_mask, encoding='mono8')
        ann_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_ann, encoding='rgb8')
        
        # Publish mask and annotated image
        self.sidewalk_mask_pub.publish(mask_msg)
        self.sidewalk_ann_pub.publish(ann_msg)
        
    
if __name__ == '__main__':
    node = SidewalkSegementation()
    node.run()
#! /usr/bin/env python3

import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

import os
import time
import cv2
import torch
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
            
            # Topic Parameters
            self.rgb_topic = load_param('~rgb_topic', 'image')
            # self.camera_info_topic = replace_base(self.rgb_topic, 'camera_info')
            
            # self.depth_topic = load_param('~depth_topic', 'depth')
            # self.pointcloud_topic = load_param('~pointcloud_topic', 'pointcloud')
            
            self.sidewalk_mask_topic = load_param('~sidewalk_mask_topic', 'sidewalk_mask')
            self.sidewalk_ann_topic = load_param('~sidewalk_ann_topic', 'sidewalk_ann')
            
            # Model parameters
            self.model_name = load_param('~model_name', 'FastSAM-x.pt') # FastSAM-s.pt or FastSAM-x.pt
            self.use_cuda = load_param('~use_cuda', False)
            self.conf = load_param('~conf', 0.4)
            self.iou = load_param('~iou', 0.9)
            
            # Prompt parameters
            self.prompt_type = load_param('~prompt_type', 'bbox') # bbox or points or text
            self.prompt_bbox = load_param('~bbox_prompt_corners', [0.35, 0.50, 0.65, 0.95]) # [x1, y1, x2, y2] in relative coordinates
            self.prompt_points = load_param('~points_prompt_points', [[0.50, 0.95]]) # [[x1, y1], [x2, y2], ...] in relative coordinates
            self.prompt_text = load_param('~text_prompt_text', 'a sidewalk or footpath or walkway or paved path for humans to walk on')
            
            # Other parameters
            self.publish_ann = load_param('~publish_ann', False)
            self.verbose = load_param('~verbose', False)
            
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
            self.sidewalk_mask_pub = rospy.Publisher(self.sidewalk_mask_topic, Image, queue_size=1)
            if self.publish_ann:
                self.sidewalk_ann_pub = rospy.Publisher(self.sidewalk_ann_topic, Image, queue_size=1)
            
            # Subscribers
            rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1, buff_size=2**24)
            # rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
            # rospy.Subscriber(self.pointcloud_topic, Image, self.pointcloud_callback)
            # rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        
        except Exception as e:
            # Log error
            rospy.logerr(e)

        else:
            # Log status
            rospy.loginfo('{} node initialized with model: {}'.format(rospy.get_name(), self.model_name))
            
    def run(self):
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down {}'.format(rospy.get_name()))

    def rgb_callback(self, img_msg):
        start_time = time.time()
        
        # Convert ROS image to OpenCV image
        image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
        
        # Run inference on the image
        everything_results = self.model(image, device=self.device, imgsz=img_msg.width,
                                        conf=self.conf, iou=self.iou, retina_masks=True, verbose=self.verbose)
        inference_time = time.time()
        
        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        
        # Prompt the results
        if self.prompt_type == 'bbox':
            # Convert bbox from relative to absolute
            bbox = [int(scale*dim) for scale, dim in zip(self.prompt_bbox, 2*[img_msg.width, img_msg.height])]
            sidewalk_results = prompt_process.box_prompt(bbox)
        elif self.prompt_type == 'points':
            # Convert points from relative to absolute
            points=[[int(scale*dim) for scale, dim in zip(point, [img_msg.width, img_msg.height])] for point in self.prompt_points]
            sidewalk_results = prompt_process.point_prompt(points, pointlabel=[1])
        elif self.prompt_type == 'text':
            sidewalk_results = prompt_process.text_prompt(text=self.prompt_text)
        else:
            rospy.logerr("Invalid value for prompt_type parameter")
        
        prompt_time = time.time()
        
        # Postprocess the mask
        sidewalk_mask = sidewalk_results[0].cpu().numpy().masks.data[0].astype('uint8')*255
        contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        sidewalk_mask = np.zeros_like(sidewalk_mask)
        cv2.fillPoly(sidewalk_mask, [max_contour], 255)
        
        # Publish mask
        mask_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_mask, encoding='mono8')
        self.sidewalk_mask_pub.publish(mask_msg)
        
        # Get annotated image and publish 
        if self.publish_ann:
            sidewalk_results[0].masks.data = torch.tensor(np.array([sidewalk_mask]))
            sidewalk_ann = sidewalk_results[0].plot(masks=True, conf=False, kpt_line=False,
                                                    labels=False, boxes=False, probs=False)
            if self.prompt_type=='bbox':
                cv2.rectangle(sidewalk_ann, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)        
            ann_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_ann, encoding='rgb8')
            self.sidewalk_ann_pub.publish(ann_msg)
        
        publish_time = time.time()
        
        # Log times
        if self.verbose:
            rospy.loginfo('{:.3f}s total, {:.3f}s inference, {:.3f}s prompt, {:.3f}s publish'.format(
                publish_time-start_time, inference_time-start_time, prompt_time-inference_time, publish_time-prompt_time))
    
    
if __name__ == '__main__':
    node = SidewalkSegementation()
    node.run()
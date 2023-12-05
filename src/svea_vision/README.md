# svea_vision
This package contains the vision system for SVEA. It contains nodes for object detection, pose estimation, aruco markers and sidewalk segmentation. 

## Installation

Please refer to [svea](https://github.com/KTH-SML/svea) on how to install and run SVEA software.

## ROS Nodes

### object_detect.py
TODO

### object_pose.py
TODO

### aruco_detect.py
TODO

### sidewalk_segmentation.py
This node subscribes to rgb image and point cloud topics and publishes a segmentation mask and a point cloud containing only the sidewalk. The script uses the `FastSAM` model for segmentation and supports three types of prompts for segmentation: bounding box, points, and text, which are configurable through parameters.

_A note on performance of prompt types_: The bounding box prompt is the fastest, followed by the points prompt, and finally the text prompt. The average publishing time per frame for the bounding box prompt is ~0.3 seconds, for the points prompt it is ~0.5 seconds, and for the text prompt it is ~4.0 seconds when running on Zed Box with CUDA enabled and maximum power mode.

#### Subscribed topics
- `~rgb_topic` (sensor_msgs/Image)
- `~pointcloud_topic` (sensor_msgs/PointCloud2)

#### Published topics
- `~sidewalk_mask_topic` (sensor_msgs/Image)
- `~sidewalk_pointcloud_topic` (sensor_msgs/PointCloud2)
- `~sidewalk_ann_topic` (sensor_msgs/Image)

#### Parameters
- `~rgb_topic` (default: 'image'): The topic for RGB images.
- `~pointcloud_topic` (default: 'pointcloud'): The topic for point cloud data.
- `~sidewalk_mask_topic` (default: 'sidewalk_mask'): The topic for the sidewalk mask.
- `~sidewalk_pointcloud_topic` (default: 'sidewalk_pointcloud'): The topic for the sidewalk point cloud.
- `~sidewalk_ann_topic` (default: 'sidewalk_ann'): The topic for sidewalk annotations.
- `~model_name` (default: 'FastSAM-x.pt'): The name of the model to use. Options are 'FastSAM-s.pt' or 'FastSAM-x.pt'.
- `~use_cuda` (default: False): Whether to use CUDA for computation.
- `~conf` (default: 0.4): Confidence threshold for the model.
- `~iou` (default: 0.9): Intersection over Union threshold for the model.
- `~prompt_type` (default: 'bbox'): The type of prompt to use. Options are 'bbox', 'points', or 'text'.
- `~bbox_prompt_corners` (default: [0.35, 0.50, 0.65, 0.95]): The corners of the bounding box prompt in relative coordinates (x1, y1, x2, y2). Only used if prompt_type is 'bbox'.
- `~points_prompt_points` (default: [[0.50, 0.95]]): The points for the points prompt in relative coordinates (x, y). Only used if prompt_type is 'points'.
- `~text_prompt_text` (default: 'a sidewalk or footpath or walkway or paved path for humans to walk on'): The text for the text prompt. Only used if prompt_type is 'text'.
- `~publish_ann` (default: False): Whether to publish annotations.

### static_image_publisher.py
This node publishes a static image or a set of static images from a directory to a topic at a fixed rate. The script is useful for testing other nodes that subscribe to image topics.

#### Published topics
- `~image_topic` (sensor_msgs/Image)

#### Parameters
- `~image_topic` (default: 'image'): The topic for the image.
- `~image_path` (default: None): The path to the image or directory of images. If a directory is specified, the images will be published in a random order. This parameter is required.
- `~rate` (default: 30.0): The rate at which to publish images in Hz.
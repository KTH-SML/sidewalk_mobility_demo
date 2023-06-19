#! /usr/bin/env python3

import numpy as np

import rospy
import message_filters as mf
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel

from rsu_msgs.msg import StampedObjectArray, StampedObjectPoseArray, ObjectPose


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def replace_base(old, new):
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith('~')
    is_global = new.startswith('/')
    assert not (is_private or is_global)
    ns, _ = split_last(old.split('/'))
    ns += new.split('/')
    return '/'.join(ns)


class object_pose:

    def __init__(self):

        ## Initialize node

        rospy.init_node('object_pose')

        ## Parameters

        self.SUB_OBJECTS = load_param('~sub_objects', 'objects')

        self.SUB_DEPTH_IMAGE = load_param('~sub_depth_image', 'depth_image')
        self.SUB_CAMERA_INFO = replace_base(self.SUB_DEPTH_IMAGE, 'camera_info')

        self.PUB_OBJECTPOSES = load_param('~pub_objectposes', 'objectposes')

        ## Camera model

        self.camera_model = PinholeCameraModel()

        ## Publishers

        self.pub_objectposes = rospy.Publisher(self.PUB_OBJECTPOSES, StampedObjectPoseArray, queue_size=10)
        rospy.loginfo(self.PUB_OBJECTPOSES)

        ## Subscribers

        self.ts = mf.TimeSynchronizer([
            mf.Subscriber(self.SUB_OBJECTS, StampedObjectArray),
            mf.Subscriber(self.SUB_DEPTH_IMAGE, Image),
            mf.Subscriber(self.SUB_CAMERA_INFO, CameraInfo),
        ], queue_size=10)
        self.ts.registerCallback(self.callback)

        rospy.loginfo(self.SUB_OBJECTS)
        rospy.loginfo(self.SUB_DEPTH_IMAGE)

    def run(self):
        rospy.spin()

    def callback(self, object_array, image, camera_info):

        ## Load camera info

        self.camera_model.fromCameraInfo(camera_info)

        ## Prepare depth map

        depth_map = np.frombuffer(image.data, dtype=np.float32).reshape(image.height, image.width)
        H, W = depth_map.shape[:2]

        ## Project pixel to 3D coordinate for each object

        objects = []

        for obj in object_array.objects:

            ## Get depth of object
            # 1. create a mask for the region of interest
            # 2. Segment by thresholding (pick out the foreground)
            # 3. Save mean of foreground as distance

            u1 = obj.roi.x_offset
            v1 = obj.roi.y_offset
            u2 = u1 + obj.roi.width
            v2 = v1 + obj.roi.height

            # Rescale to depth_map
            u1 = (u1 * W) // obj.image_width
            u2 = (u2 * W) // obj.image_width
            v1 = (v1 * H) // obj.image_height
            v2 = (v2 * H) // obj.image_height

            roi_mask = np.zeros((H, W), dtype=bool)
            roi_mask[v1:v2, u1:u2] = True

            # Get only usuable depths of the roi
            roi_mask[np.isnan(depth_map)] = False
            roi_mask[np.isinf(depth_map)] = False

            if not roi_mask.sum():
                continue

            # threshold = mean of masked area
            segm_mask = depth_map[roi_mask] < depth_map[roi_mask].mean()

            if not segm_mask.sum():
                continue

            # take mean of the segment as distance
            d = depth_map[roi_mask][segm_mask].mean()

            ## Projection
            # 1. take middle pixel of region of interest
            # 2. get unit vector of projection by `camera_model.projectPixelTo3dRay()`
            # 3. multiply unit vec by distance to get real world coordinates in camera's frame

            u = (u1 + u2) // 2
            v = (v1 + v2) // 2

            ray = self.camera_model.projectPixelTo3dRay((u, v))
            x, y, z = np.array(ray) * d

            objpose = ObjectPose()
            objpose.object = obj
            ## NOTE: Message supports these
            # objpose.pose.covariance = ... # float64
            # objpose.pose.pose.orientation = ... # geometry_msgs/Quaternion
            objpose.pose.pose.position.x = x
            objpose.pose.pose.position.y = y
            objpose.pose.pose.position.z = z

            objects.append(objpose)

        ## Publish

        if objects:

            objectpose_array = StampedObjectPoseArray()
            objectpose_array.header = object_array.header
            objectpose_array.objects = objects

            self.pub_objectposes.publish(objectpose_array)


if __name__ == '__main__':

    ##  Start node  ##

    object_pose().run()


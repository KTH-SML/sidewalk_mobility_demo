#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo

import cv2
import numpy as np

from ultralytics import YOLO
from sort import Sort

from rsu_msgs.msg import Object, StampedObjectArray


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

def iou(box1, box2):

    u11, v11, u21, v21 = box1
    u12, v12, u22, v22 = box2

    u1_max = max(u11, u12)
    v1_max = max(v11, v12)
    u2_min = min(u21, u22)
    v2_min = min(v21, v22)

    # area of intersection
    w = abs(u1_max - u2_min)
    h = abs(v1_max - v2_min)
    area_i = w * h

    # area of box1
    w = abs(u21 - u11)
    h = abs(v21 - v11)
    area_1 = w * h

    # area of box2
    w = abs(u22 - u12)
    h = abs(v22 - v12)
    area_2 = w * h

    # area of union
    area_u = area_1 + area_2 - area_i

    # intersection-over-union
    return area_i / area_u


class object_detect:

    def __init__(self):

        ## Initialize node

        rospy.init_node('object_detect')

        ## Parameters

        self.ENABLE_BBOX_IMAGE = load_param('~enable_bbox_image', False)

        self.SUB_IMAGE = load_param('~sub_image', 'image')
        self.SUB_CAMERA_INFO = replace_base(self.SUB_IMAGE, 'camera_info')

        self.PUB_BBOX_IMAGE = load_param('~pub_bbox_image', 'bbox_image')
        self.PUB_CAMERA_INFO = replace_base(self.PUB_BBOX_IMAGE, 'camera_info')

        self.USE_CUDA = load_param('~use_cuda', False)
        self.MODEL_PATH = load_param('~model_path', 'yolov8n.pt')

        self.MAX_AGE = load_param('~max_age', 30)

        self.PUB_OBJECTS = load_param('~pub_objects', 'objects')

        ## Neural Network

        self.model = YOLO(self.MODEL_PATH)

        if self.USE_CUDA:
            rospy.loginfo('CUDA enabled')
            self.model.to('cuda')
        else:
            rospy.loginfo('CUDA disabled')

        ## SORT multi-object tracker

        self.tracked_objects = []
        self.sort = Sort(max_age=self.MAX_AGE, min_hits=3, iou_threshold=0.3)

        ## Publishers

        self.pub_objects = rospy.Publisher(self.PUB_OBJECTS, StampedObjectArray, queue_size=10)
        rospy.loginfo(self.PUB_OBJECTS)

        if self.ENABLE_BBOX_IMAGE:

            self.pub_bbox_image = rospy.Publisher(self.PUB_BBOX_IMAGE, Image, queue_size=1)
            rospy.loginfo(self.PUB_BBOX_IMAGE)

        ## Subscribers

        rospy.Subscriber(self.SUB_IMAGE, Image, self.callback)
        rospy.loginfo(self.SUB_IMAGE)

        ## Relay (sub->pub) camera info

        if self.ENABLE_BBOX_IMAGE:
            pub = rospy.Publisher(self.PUB_CAMERA_INFO, CameraInfo, queue_size=1)
            rospy.Subscriber(self.SUB_CAMERA_INFO, CameraInfo, pub.publish)

    def run(self):
        rospy.spin()

    def callback(self, image):

        ## Detect objects

        frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        result = self.model.predict(frame)[0]
        result = result.cpu().numpy()
        boxes = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        ## Update tracker

        self.tracked_objects = self.sort.update(boxes)

        ## Create object messages

        objects = []
        for box, c, p  in zip(boxes, cls, conf):
            u1, v1, u2, v2 = box

            # get real pixel coords
            u1, u2 = [min(image.width, max(0, round(u*image.width))) for u in (u1, u2)]
            v1, v2 = [min(image.height, max(0, round(v*image.height))) for v in (v1, v2)]

            # do not continue if box has no size
            if u1 != u2 and v1 != v2:

                # pick best matched tracked object
                trk = [0, 0, 0, 0, 0]
                trk_iou = 0
                trk_ind = -1
                for i, _trk in enumerate(self.tracked_objects):
                    _iou = iou(_trk[:4], box[:4])
                    if trk_iou < _iou:
                        trk = _trk
                        trk_iou = _iou
                        trk_ind = i

                tracker = self.sort.trackers[trk_ind]

                obj = Object()
                obj.id = int(trk[-1])
                obj.label = result.names[c]
                obj.detection_conf = p
                obj.tracking_conf = tracker.kf.likelihood
                obj.roi.x_offset = u1
                obj.roi.y_offset = v1
                obj.roi.width = u2 - u1
                obj.roi.height = v2 - v1
                objects.append(obj)

            # if enabled, modify frame (add bounding boxes)
            if self.ENABLE_BBOX_IMAGE:
                frame = result.plot()

        # Publish objects
        if objects:
            object_array = StampedObjectArray()
            object_array.header = image.header
            object_array.objects = objects
            self.pub_objects.publish(object_array)

        if self.ENABLE_BBOX_IMAGE:
            new_image = Image()
            new_image.header = image.header
            new_image.height = frame.shape[0]
            new_image.width = frame.shape[1]
            new_image.encoding = 'rgb8'
            new_image.step = frame.size // new_image.height
            new_image.data = frame.tostring()

            self.pub_bbox_image.publish(new_image)


if __name__ == '__main__':

    ##  Start node  ##

    object_detect().run()


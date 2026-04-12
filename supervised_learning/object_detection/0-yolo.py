#!/usr/bin/env python3
"""YOLO v3 object detection class"""

import tensorflow.keras as K


class Yolo:
    """Class that uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the YOLO object detector

        Parameters:
        model_path (str): path to the Darknet Keras model
        classes_path (str): path to file containing class names
        class_t (float): box score threshold for filtering
        nms_t (float): IOU threshold for non-max suppression
        anchors (numpy.ndarray): anchor boxes of shape
                                 (outputs, anchor_boxes, 2)
        """

        # Load Keras model
        self.model = K.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Thresholds
        self.class_t = class_t
        self.nms_t = nms_t

        # Anchor boxes
        self.anchors = anchors

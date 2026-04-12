#!/usr/bin/env python3
"""YOLO v3 object detection class"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Class that uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the YOLO model

        Parameters:
        outputs (list): list of numpy arrays from YOLO model
        image_size (numpy.ndarray): [image_height, image_width]

        Returns:
        tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract components
            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            objectness = output[..., 4:5]
            class_probs = output[..., 5:]

            # Grid creation
            cy = np.arange(grid_h)
            cx = np.arange(grid_w)
            cx, cy = np.meshgrid(cx, cy)

            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            grid = np.concatenate((cx, cy), axis=-1)
            grid = np.expand_dims(grid, axis=2)

            # Apply transformations
            b_xy = (self.sigmoid(t_xy) + grid) / [grid_w, grid_h]
            b_wh = (np.exp(t_wh) * self.anchors[i]) / [input_w, input_h]

            # Convert to corners
            x1 = (b_xy[..., 0] - b_wh[..., 0] / 2) * image_w
            y1 = (b_xy[..., 1] - b_wh[..., 1] / 2) * image_h
            x2 = (b_xy[..., 0] + b_wh[..., 0] / 2) * image_w
            y2 = (b_xy[..., 1] + b_wh[..., 1] / 2) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)

            # Process confidences
            box_confidence = self.sigmoid(objectness)
            box_class_prob = self.sigmoid(class_probs)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

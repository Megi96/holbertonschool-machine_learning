#!/usr/bin/env python3
"""Module for YOLO v3 object detection."""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo instance.

        Args:
            model_path (str): Path to a Darknet Keras model.
            classes_path (str): Path to file containing class names.
            class_t (float): Box score threshold for initial filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array of anchor boxes shape
                (outputs, anchor_boxes, 2).
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process outputs from Darknet model for a single image.

        Args:
            outputs (list): List of numpy.ndarrays with Darknet predictions.
                Each output shape:
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes).
            image_size (numpy.ndarray): Image's original size
                [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                boxes: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 4) with
                    boundary boxes (x1, y1, x2, y2) relative to original image.
                box_confidences: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 1).
                box_class_probs: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, classes).
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        # Input dimensions from the model
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract raw values
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Anchor dimensions for this output scale
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Create grid offsets
            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            # Decode box center coordinates (relative to input size)
            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_width
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_height

            # Decode box dimensions (relative to input size)
            bw = (pw * np.exp(t_w)) / input_width
            bh = (ph * np.exp(t_h)) / input_height

            # Convert to (x1, y1, x2, y2) relative to original image
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Box confidence: sigmoid of raw confidence
            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            # Class probabilities: sigmoid of raw class scores
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

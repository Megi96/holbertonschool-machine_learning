#!/usr/bin/env python3
"""Module for YOLO object detection - Task 5: Preprocess Images"""

import numpy as np
import tensorflow.keras as K
import cv2
import os
import glob


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo object detection model.

        Args:
            model_path (str): Path to the Darknet Keras model.
            classes_path (str): Path to the list of class names.
            class_t (float): Box score threshold for initial filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes of shape (outputs, anchor_boxes, 2).
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Apply the sigmoid activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs into bounding boxes and confidences.

        Args:
            outputs (list): List of numpy.ndarrays from the Darknet model.
            image_size (numpy.ndarray): Original image size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                - boxes: list of arrays with bounding boxes per output.
                - box_confidences: list of arrays with box confidences per output.
                - box_class_probs: list of arrays with class probabilities per output.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            box_conf = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])
            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            cx = np.arange(grid_w).reshape(1, grid_w)
            cx = np.repeat(cx, grid_h, axis=0)
            cx = np.repeat(cx[..., np.newaxis], anchor_boxes, axis=2)

            cy = np.arange(grid_h).reshape(grid_h, 1)
            cy = np.repeat(cy, grid_w, axis=1)
            cy = np.repeat(cy[..., np.newaxis], anchor_boxes, axis=2)

            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            anchors_w = self.anchors[i, :, 0]
            anchors_h = self.anchors[i, :, 1]

            bw = (np.exp(tw) * anchors_w) / input_w
            bh = (np.exp(th) * anchors_h) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter bounding boxes by score threshold.

        Args:
            boxes (list): Processed bounding boxes per output.
            box_confidences (list): Box confidences per output.
            box_class_probs (list): Class probabilities per output.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                - filtered_boxes: numpy.ndarray of shape (?, 4).
                - box_classes: numpy.ndarray of class indices, shape (?).
                - box_scores: numpy.ndarray of box scores, shape (?).
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, prob in zip(boxes, box_confidences, box_class_probs):
            scores = conf * prob
            class_idx = np.argmax(scores, axis=-1)
            class_score = np.max(scores, axis=-1)

            mask = class_score >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(class_idx[mask])
            box_scores.append(class_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to remove overlapping boxes.

        Args:
            filtered_boxes (numpy.ndarray): Filtered bounding boxes, shape (?, 4).
            box_classes (numpy.ndarray): Class index per box, shape (?).
            box_scores (numpy.ndarray): Score per box, shape (?).

        Returns:
            tuple: (box_predictions, predicted_box_classes, predicted_box_scores)
                - box_predictions: numpy.ndarray of shape (?, 4).
                - predicted_box_classes: numpy.ndarray of shape (?).
                - predicted_box_scores: numpy.ndarray of shape (?).
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idx = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            sort_idx = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sort_idx]
            cls_scores = cls_scores[sort_idx]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                area0 = ((cls_boxes[0, 2] - cls_boxes[0, 0]) *
                         (cls_boxes[0, 3] - cls_boxes[0, 1]))
                area1 = ((cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                         (cls_boxes[1:, 3] - cls_boxes[1:, 1]))
                union_area = area0 + area1 - inter_area

                iou = inter_area / union_area
                keep = np.where(iou < self.nms_t)[0]
                cls_boxes = cls_boxes[keep + 1]
                cls_scores = cls_scores[keep + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            tuple: (images, image_paths)
                - images: list of images as numpy.ndarrays.
                - image_paths: list of image file paths.
        """
        image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        image_paths += glob.glob(os.path.join(folder_path, '*.jpeg'))
        image_paths += glob.glob(os.path.join(folder_path, '*.png'))

        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images for the Darknet model.

        Resizes each image to the model's input dimensions using inter-cubic
        interpolation and rescales pixel values to the range [0, 1].

        Args:
            images (list): List of images as numpy.ndarrays.

        Returns:
            tuple: (pimages, image_shapes)
                - pimages (numpy.ndarray): Preprocessed images of shape
                  (ni, input_h, input_w, 3).
                - image_shapes (numpy.ndarray): Original image dimensions of
                  shape (ni, 2) as (image_height, image_width).
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])
            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

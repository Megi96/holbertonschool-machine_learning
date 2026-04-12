#!/usr/bin/env python3
"""Module for YOLO v3 object detection."""
import numpy as np
import tensorflow.keras as K
import cv2
import os


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

        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_width
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_height

            bw = (pw * np.exp(t_w)) / input_width
            bh = (ph * np.exp(t_h)) / input_height

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter bounding boxes based on class score threshold.

        Args:
            boxes (list): numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4) containing
                processed boundary boxes for each output.
            box_confidences (list): numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1) containing
                processed box confidences for each output.
            box_class_probs (list): numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes) containing
                processed box class probabilities for each output.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                filtered_boxes: numpy.ndarray of shape (?, 4) with all
                    filtered bounding boxes.
                box_classes: numpy.ndarray of shape (?,) with the class
                    number each box predicts.
                box_scores: numpy.ndarray of shape (?) with the box score
                    for each filtered box.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)

            mask = best_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(best_class[mask])
            box_scores.append(best_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to eliminate redundant overlapping boxes.

        Args:
            filtered_boxes (numpy.ndarray): Shape (?, 4) containing all
                filtered bounding boxes.
            box_classes (numpy.ndarray): Shape (?,) containing the class
                number for each filtered box.
            box_scores (numpy.ndarray): Shape (?,) containing the box
                score for each filtered box.

        Returns:
            tuple: (box_predictions, predicted_box_classes,
                predicted_box_scores)
                box_predictions: numpy.ndarray of shape (?, 4) with
                    predicted bounding boxes ordered by class and score.
                predicted_box_classes: numpy.ndarray of shape (?,) with
                    class numbers ordered by class and score.
                predicted_box_scores: numpy.ndarray of shape (?,) with
                    box scores ordered by class and score.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            kept_boxes = []
            kept_scores = []

            while len(cls_boxes) > 0:
                kept_boxes.append(cls_boxes[0])
                kept_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                area_best = (
                    (cls_boxes[0, 2] - cls_boxes[0, 0]) *
                    (cls_boxes[0, 3] - cls_boxes[0, 1])
                )
                areas_rest = (
                    (cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                    (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                )
                union = area_best + areas_rest - intersection

                iou = intersection / union

                suppress_mask = iou < self.nms_t
                cls_boxes = cls_boxes[1:][suppress_mask]
                cls_scores = cls_scores[1:][suppress_mask]

            kept_boxes = np.array(kept_boxes)
            kept_scores = np.array(kept_scores)

            box_predictions.append(kept_boxes)
            predicted_box_classes.append(
                np.full(len(kept_boxes), cls, dtype=box_classes.dtype)
            )
            predicted_box_scores.append(kept_scores)

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(
            predicted_box_classes, axis=0
        )
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load all images from a given folder.

        Args:
            folder_path (str): Path to the folder holding all images to load.

        Returns:
            tuple: (images, image_paths)
                images: list of images as numpy.ndarrays.
                image_paths: list of paths to the individual images.
        """
        image_paths = [
            os.path.join(folder_path, fname)
            for fname in sorted(os.listdir(folder_path))
        ]

        images = [
            cv2.imread(path)
            for path in image_paths
        ]

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and normalize images for input into the Darknet model.

        Args:
            images (list): List of images as numpy.ndarrays.

        Returns:
            tuple: (pimages, image_shapes)
                pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                    containing all preprocessed images with pixel values
                    in the range [0, 1].
                image_shapes: numpy.ndarray of shape (ni, 2) containing
                    the original height and width of each image as
                    [image_height, image_width].
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

            normalized = resized / 255.0
            pimages.append(normalized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
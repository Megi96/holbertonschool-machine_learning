#!/usr/bin/env python3
"""
YOLO v3 object detection class
"""

import numpy as np
import cv2
import tensorflow.keras as K


class Yolo:
    """
    YOLO v3 object detection class
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_images(self, folder_path):
        """
        Loads images from folder
        """
        images = []
        image_paths = []

        for file in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for YOLO
        """
        image_shapes = []
        processed_images = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            # Save original shape (height, width)
            image_shapes.append([img.shape[0], img.shape[1]])

            # Resize image
            img_resized = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            # Normalize pixel values
            img_resized = img_resized / 255.0

            processed_images.append(img_resized)

        pimages = np.array(processed_images, dtype=np.float32)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

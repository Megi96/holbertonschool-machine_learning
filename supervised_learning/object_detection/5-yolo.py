#!/usr/bin/env python3
import tensorflow.keras as K
import numpy as np
import cv2
import os


class Yolo:
    """YOLO class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO"""

        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_images(self, folder_path):
        """
        Loads all images from a folder

        Returns:
            images: list of images as numpy.ndarrays
            image_paths: list of image paths
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for the YOLO model

        Args:
            images: list of numpy.ndarray images

        Returns:
            pimages: numpy.ndarray of shape
                     (ni, input_h, input_w, 3)
            image_shapes: numpy.ndarray of shape (ni, 2)
                          containing original (h, w)
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            # original shape
            h, w = img.shape[:2]
            image_shapes.append([h, w])

            # resize (IMPORTANT: width, height order)
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            # normalize
            resized = resized / 255.0

            pimages.append(resized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

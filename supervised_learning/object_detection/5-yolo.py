#!/usr/bin/env python3
import numpy as np
import cv2

class Yolo:
    # assume __init__ and other methods already exist

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
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            # Save original shape (height, width)
            h, w = img.shape[:2]
            image_shapes.append([h, w])

            # Resize image
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            # Normalize to [0, 1]
            resized = resized / 255.0

            pimages.append(resized)

        # Convert lists to numpy arrays
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

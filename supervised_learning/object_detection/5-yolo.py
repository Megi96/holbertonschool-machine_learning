#!/usr/bin/env python3

def preprocess_images(self, images):
    """
    Preprocesses images for YOLO model

    Parameters:
    images (list): list of numpy.ndarray images

    Returns:
    tuple: (pimages, image_shapes)
        pimages: numpy.ndarray (ni, input_h, input_w, 3)
        image_shapes: numpy.ndarray (ni, 2)
    """

    image_shapes = []
    processed_images = []

    input_h = self.model.input.shape[1]
    input_w = self.model.input.shape[2]

    for img in images:
        # Save original shape (height, width)
        image_shapes.append([img.shape[0], img.shape[1]])

        # Resize using cubic interpolation
        resized = cv2.resize(img, (input_w, input_h),
                             interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        resized = resized / 255.0

        processed_images.append(resized)

    pimages = np.array(processed_images)
    image_shapes = np.array(image_shapes)

    return pimages, image_shapes

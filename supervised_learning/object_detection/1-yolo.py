#!/usr/bin/env python3

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

    image_h = image_size[0]
    image_w = image_size[1]

    for i, output in enumerate(outputs):
        grid_h = output.shape[0]
        grid_w = output.shape[1]

        # Extract components
        t_xy = output[..., 0:2]
        t_wh = output[..., 2:4]
        objectness = output[..., 4:5]
        class_probs = output[..., 5:]

        # Create grid
        cy = np.arange(grid_h)
        cx = np.arange(grid_w)
        cx, cy = np.meshgrid(cx, cy)

        cx = cx[..., np.newaxis]
        cy = cy[..., np.newaxis]

        grid = np.concatenate((cx, cy), axis=-1)
        grid = grid[np.newaxis, ...]
        grid = np.squeeze(grid)

        # --- YOLO transformations ---

        # Center coordinates
        b_xy = self.sigmoid(t_xy) + grid
        b_xy[..., 0] = b_xy[..., 0] / grid_w
        b_xy[..., 1] = b_xy[..., 1] / grid_h

        # Width and height
        b_wh = np.exp(t_wh) * self.anchors[i]
        b_wh[..., 0] = b_wh[..., 0] * (image_w / input_w)
        b_wh[..., 1] = b_wh[..., 1] * (image_h / input_h)

        # Scale centers to image size
        b_xy[..., 0] = b_xy[..., 0] * image_w
        b_xy[..., 1] = b_xy[..., 1] * image_h

        # Convert to corner coordinates
        x1 = b_xy[..., 0] - (b_wh[..., 0] / 2)
        y1 = b_xy[..., 1] - (b_wh[..., 1] / 2)
        x2 = b_xy[..., 0] + (b_wh[..., 0] / 2)
        y2 = b_xy[..., 1] + (b_wh[..., 1] / 2)

        box = np.stack((x1, y1, x2, y2), axis=-1)

        # Sigmoid for confidence and class probabilities
        box_confidence = self.sigmoid(objectness)
        box_class_prob = self.sigmoid(class_probs)

        boxes.append(box)
        box_confidences.append(box_confidence)
        box_class_probs.append(box_class_prob)

    return boxes, box_confidences, box_class_probs

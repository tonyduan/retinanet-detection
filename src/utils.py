import torch
import numpy as np


def bbox_convert_corners_to_sizes(box):
    """
    Convert (ymin, xmin, ymax, xmax) => (ymin, xmin, height, width) representation.
    """
    ymin, xmin, ymax, xmax = box
    return (ymin, xmin, ymax - ymin, xmax - xmin)


def bbox_convert_sizes_to_corners(box):
    """
    Convert (ymin, xmin, height, width) => (ymin, xmin, ymax, xmax) representation.
    """
    ymin, xmin, height, width = box
    return (ymin, xmin, ymin + height, xmin + width)


def calculate_iou(boxesA, boxesB, lib="torch"):
    """
    Compute IoU between two sets of boxes.

    Parameters
    ----------
    boxesA: M x 4 torch tensor or numpy array
    boxesB: N x 4 torch tensor or numpy array

    Returns
    -------
    ious: M x N tensor or array

    Notes
    -----
    Assumes the representation of boxes are as (xmin, ymin, width, height)
    """
    # extract the corners of the bounding boxes
    A_x1, B_x1 = boxesA[:, 0], boxesB[:, 0]
    A_y1, B_y1 = boxesA[:, 1], boxesB[:, 1]
    A_x2, B_x2 = boxesA[:, 0] + boxesA[:, 2], boxesB[:, 0] + boxesB[:, 2]
    A_y2, B_y2 = boxesA[:, 1] + boxesA[:, 3], boxesB[:, 1] + boxesB[:, 3]

    # intersection coordinates
    if lib == "torch":
        A_x1, A_y1 = A_x1.unsqueeze(1), A_y1.unsqueeze(1)
        A_x2, A_y2 = A_x2.unsqueeze(1), A_y2.unsqueeze(1)
        B_x1, B_y1 = B_x1.unsqueeze(0), B_y1.unsqueeze(0)
        B_x2, B_y2 = B_x2.unsqueeze(0), B_y2.unsqueeze(0)
        x_left = torch.max(A_x1, B_x1)
        y_bottom = torch.max(A_y1, B_y1)
        x_right = torch.min(A_x2, B_x2)
        y_top = torch.min(A_y2, B_y2)
    elif lib == "numpy":
        A_x1, A_y1 = A_x1[:, np.newaxis], A_y1[:, np.newaxis]
        A_x2, A_y2 = A_x2[:, np.newaxis], A_y2[:, np.newaxis]
        B_x1, B_y1 = B_x1[np.newaxis, :], B_y1[np.newaxis, :]
        B_x2, B_y2 = B_x2[np.newaxis, :], B_y2[np.newaxis, :]
        x_left = np.maximum(A_x1, B_x1)
        y_bottom = np.maximum(A_y1, B_y1)
        x_right = np.minimum(A_x2, B_x2)
        y_top = np.minimum(A_y2, B_y2)
    else:
        raise ValueError

    # straightforward area calculation for intersection over union
    intersection_area = (x_right - x_left) * (y_top - y_bottom)
    A_area = (A_x2 - A_x1) * (A_y2 - A_y1)
    B_area = (B_x2 - B_x1) * (B_y2 - B_y1)

    iou = intersection_area / (A_area + B_area - intersection_area)
    iou[x_right < x_left] = 0.0
    iou[y_top < y_bottom] = 0.0
    return iou


def calculate_nms(boxes, scores, threshold=0.5):
    """
    Calculate the non-maximum suppression of a set of boxes.

    Returns
    -------
    sel_boxes: list of selected boxes each a tensor of length 4
    sel_scores: list of selected box scores each a float
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    _, order = scores.sort(0, descending=True)
    sel_boxes, sel_scores = [], []
    
    while len(order) > 0:
        
        i = order[0]
        sel_boxes.append(boxes[i])
        sel_scores.append(scores[i])
        iou = calculate_iou(boxes[order], boxes[i].unsqueeze(0)).squeeze()
        order = order[(iou < threshold) & ~(torch.isnan(iou)) & (order != i)]
    
    return sel_boxes, sel_scores


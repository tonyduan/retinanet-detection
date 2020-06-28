"""
Simple visualization of the anchor targets for an example 224 x 224 image.

Top subfigure: fixed feature map size; draw all anchors at bottom left and top right corners.
Bottom subfigure: fix anchor choice at bottom left conrer; vary feature map size.
"""
import math
import torch
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from src.models import RetinaNet


if __name__ == "__main__":
    
    WIDTH, HEIGHT = 324, 224
    all_fm_anchor_boxes = RetinaNet.get_all_fm_anchor_boxes(HEIGHT, WIDTH)

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 4), nrows=1, ncols=2)
    box = patches.Rectangle((0, 0), WIDTH, HEIGHT, edgecolor="green", facecolor="none", linewidth=1)
    ax1.add_patch(box)
    box = patches.Rectangle((0, 0), WIDTH, HEIGHT, edgecolor="green", facecolor="none", linewidth=1)
    ax2.add_patch(box)

    idx = 1
    for fm_anchor_boxes in all_fm_anchor_boxes:
        print(fm_anchor_boxes.shape)

    for diff_anchor in all_fm_anchor_boxes[idx][:, 0, 0, :]:
        y, x, h, w = diff_anchor
        box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        ax1.add_patch(box)

    fmxmax = all_fm_anchor_boxes[idx].shape[1] - 1
    fmymax = all_fm_anchor_boxes[idx].shape[2] - 1
    for diff_anchor in all_fm_anchor_boxes[idx][:, fmxmax, fmymax, :]:
        y, x, h, w = diff_anchor
        box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        ax1.add_patch(box)

    for fm_anchor_boxes in all_fm_anchor_boxes:
        y, x, h, w = fm_anchor_boxes[4, 0, 0]
        box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        ax2.add_patch(box)

    ax1.set_xlim((-100, WIDTH + 100))
    ax1.set_ylim((-100, HEIGHT + 100))
    ax2.set_xlim((-100, WIDTH + 100))
    ax2.set_ylim((-100, HEIGHT + 100))
    plt.tight_layout()
    plt.show()


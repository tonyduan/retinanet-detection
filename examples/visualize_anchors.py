"""
Simple visualization of the anchor targets.

Top subfigure: fixed feature map size; draw all anchors at bottom left and top right corners.
Bottom subfigure: fix anchor choice at bottom left conrer; vary feature map size.
"""
import math
import torch
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from src.models import RetinaNet


if __name__ == "__main__":
    
    all_fm_anchor_boxes = RetinaNet.get_all_fm_anchor_boxes(224, 224)

    fig, (ax1, ax2) = plt.subplots(figsize=(4, 9), nrows=2, ncols=1)
    box = patches.Rectangle((0, 0), 224, 224, edgecolor="green", facecolor="none", linewidth=1)
    ax1.add_patch(box)
    box = patches.Rectangle((0, 0), 224, 224, edgecolor="green", facecolor="none", linewidth=1)
    ax2.add_patch(box)

    idx = 1
    for fm_anchor_boxes in all_fm_anchor_boxes:
        print(fm_anchor_boxes.shape)

    for diff_anchor in all_fm_anchor_boxes[idx][:, 0, 0, :]:
        x, y, w, h = diff_anchor
        box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        ax1.add_patch(box)

    fmxmax = all_fm_anchor_boxes[idx].shape[1] - 1
    fmymax = all_fm_anchor_boxes[idx].shape[2] - 1
    for diff_anchor in all_fm_anchor_boxes[idx][:, fmxmax, fmymax, :]:
        x, y, w, h = diff_anchor
        box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        ax1.add_patch(box)

    for fm_anchor_boxes in all_fm_anchor_boxes:
        x, y, w, h = fm_anchor_boxes[4, 0, 0]
        box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        ax2.add_patch(box)

    ax1.set_xlim((-100, 324))
    ax1.set_ylim((-100, 324))
    ax2.set_xlim((-100, 324))
    ax2.set_ylim((-100, 324))
    plt.tight_layout()
    plt.show()


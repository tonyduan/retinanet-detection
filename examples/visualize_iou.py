import torch
from matplotlib import pyplot as plt
from matplotlib import patches
from src.utils import *


if __name__ == "__main__":

#    boxes = torch.tensor([[ 87.8899, 136.1943, 341.4030, 328.3182],
#                          [158.8171,  97.9424, 237.4516, 408.0746],
#                          [ 87.5164, 133.3474, 322.7675, 499.1673]])
    boxes = torch.tensor([[ 87.8899, 136.1943, 341.4030, 328.3182],
                          [202.9495, 113.5746, 178.6125, 361.0347],
                          [245.4695, 182.6774, 156.3983, 254.2001]])
    fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)

    for (xmin, ymin, width, height) in boxes:
        box = patches.Rectangle((xmin, ymin), width, height, edgecolor="red", 
                                facecolor="none", linewidth=1)
        ax.add_patch(box)

    print(calculate_iou(boxes, boxes[1:]))

    ax.set_xlim((0, 800))
    ax.set_ylim((0, 800))
    plt.show()

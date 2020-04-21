import torch
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from src.models import RetinaNet


if __name__ == "__main__":
    
    WIDTH, HEIGHT = 448, 448
    image = torch.zeros((3, WIDTH, HEIGHT))

    labels = torch.tensor([0, 1], dtype=torch.long)
    boxes = torch.tensor([[30, 30, 100, 50], [120, 120, 40, 90]], dtype=torch.float32)

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)

    box = patches.Rectangle((0, 0), WIDTH, HEIGHT, edgecolor="green", facecolor="none", linewidth=1)
    ax1.add_patch(box)
    box = patches.Rectangle((0, 0), WIDTH, HEIGHT, edgecolor="green", facecolor="none", linewidth=1)
    ax2.add_patch(box)
    box = patches.Rectangle((0, 0), WIDTH, HEIGHT, edgecolor="green", facecolor="none", linewidth=1)
    ax3.add_patch(box)

    for (x, y, w, h) in boxes:
        box = patches.Rectangle((x, y), w, h, edgecolor="red", facecolor="none", linewidth=1)
        ax1.add_patch(box)

    cls_tgts, reg_tgts = RetinaNet.encode(image, labels, boxes, num_classes=2)

    print("== Classification targets")
    for i in range(RetinaNet.num_feature_maps):
        print(cls_tgts[i].shape)

    print("== Regression targets")
    for i in range(RetinaNet.num_feature_maps):
        print(reg_tgts[i].shape)

    num_pos = sum([(t == 1).sum() for t in cls_tgts])
    num_neg = sum([(t == 0).sum() for t in cls_tgts])
    num_masked = sum([(t == -1).sum() for t in cls_tgts])
    print(f"== Positive anchors: {num_pos}")
    print(f"== Masked anchors: {num_masked}")
    print(f"== Total anchors: {num_pos + num_neg + num_masked}")

    anchor_boxes = RetinaNet.get_all_fm_anchor_boxes(WIDTH, HEIGHT)
    for i in range(RetinaNet.num_feature_maps):
        idxs = (cls_tgts[i] == 1).nonzero()
        for _, anchor_no, fmx, fmy in idxs:
            x, y, w, h = anchor_boxes[i][anchor_no, fmx, fmy]
            box = patches.Rectangle((x, y), w, h, edgecolor="red", facecolor="none", linewidth=1)
            ax2.add_patch(box)

    zeroed_reg_tgts = [torch.zeros_like(t) for t in reg_tgts]
    labels, boxes, scores = RetinaNet.decode(image, cls_tgts, zeroed_reg_tgts, threshold=0.5)
    for (x, y, w, h) in boxes:
        box = patches.Rectangle((x, y), w, h, edgecolor="red", facecolor="none", linewidth=1)
        ax3.add_patch(box)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim((-100, WIDTH + 100))
        ax.set_ylim((-100, HEIGHT + 100))
    ax1.set_title("Annotations")
    ax2.set_title("Encoding: positive anchors")
    ax3.set_title("Decoding: non-maximal suppression")

    plt.tight_layout()
    plt.show()


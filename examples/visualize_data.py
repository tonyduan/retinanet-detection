from argparse import ArgumentParser
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from src.datasets import *


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--idx", default=0, type=int)
    argparser.add_argument("--dataset", default="coco", type=str)
    args = argparser.parse_args()

    dataset = get_dataset(args.dataset, "test")
    id_to_labels = get_id_to_labels(args.dataset)

    x, (labels, boxes) = dataset[args.idx]
    x = unnormalize(x)
    x = x.transpose(0, 2).transpose(0, 1)

    fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=1)
    ax.imshow(x)

    for label_id, (ymin, xmin, ymax, xmax) in zip(labels, boxes):
        box = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                edgecolor="red", facecolor="none", linewidth=1)
        ax.add_patch(box)
        ax.text(xmin, ymin, id_to_labels[label_id], color="white")

    plt.show()


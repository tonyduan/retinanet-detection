import numpy as np
from argparse import ArgumentParser
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import patches 
from torch.utils.data import Subset
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.datasets import *


def micro_avg_auprc(true_labels, true_boxes, pred_labels, pred_boxes, pred_scores, iou_threshold):
    """
    Micro-averaged (across classes) AUPRC at an IoU threshold.

    Notes
    -----
    Regrettably, this is called "average precision" or "AP@T" in the object detection literature.
    """
    n = len(true_labels)
    flattened_labels = []
    flattened_preds = []

    for idx in tqdm(range(n)):
        
        if len(pred_boxes[idx]) == 0:
            flattened_labels.extend([1] * len(true_boxes[idx]))
            flattened_preds.extend([0.01] * len(true_boxes[idx]))
            continue

        ious = calculate_iou(np.array(true_boxes[idx]), np.array(pred_boxes[idx]), lib="numpy")

        pred_labels[idx] = np.array(pred_labels[idx])
        true_labels[idx] = np.array(true_labels[idx])

        # (1) assign each true box to the prediction with which it has largest overlap
        for true_box_idx in range(len(true_labels[idx])):
            
            label_match_mask = pred_labels[idx] == true_labels[idx][true_box_idx]

            # handle the case where no box with the right class was predicted
            if np.sum(label_match_mask) == 0:
                flattened_labels.append(1)
                flattened_preds.append(0.01)

            # pick out the predicted box with highest IoU 
            pred_box_idx = np.argmax(ious[true_box_idx, :] * label_match_mask)
            flattened_labels.append(ious[true_box_idx, pred_box_idx] >= iou_threshold)
            flattened_preds.append(pred_scores[idx][pred_box_idx])

            # mark the proposal box as used
            ious[:, pred_box_idx] = -1
        
        # (2) for all remaining prediction boxes assign a label of zero
        for pred_box_idx in range(len(pred_labels[idx])):
            if ious[0, pred_box_idx] >= 0.0:
                flattened_labels.append(0)
                flattened_preds.append(pred_scores[idx][pred_box_idx])

    flattened_labels = np.array(flattened_labels)
    flattened_preds = np.array(flattened_preds)
#    y, x = calibration_curve(flattened_labels, flattened_preds)
#    plt.scatter(x, y)
#    plt.show()
    return average_precision_score(flattened_labels, flattened_preds, "micro")


def show_image(x, true_labels, true_boxes, pred_labels, pred_boxes, pred_scores):
    fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=1)
    ax.imshow(x)
    id_to_labels = get_id_to_labels("voc")
    for label_id, (xmin, ymin, width, height) in zip(true_labels, true_boxes):
        box = patches.Rectangle((xmin, ymin), width, height, edgecolor="red", 
                                facecolor="none", linewidth=1)
        ax.add_patch(box)
        ax.text(xmin, ymin, id_to_labels[label_id], color="white",
                bbox={"facecolor": "black", "alpha": 0.5})
    for label_id, score, (xmin, ymin, width, height) in zip(pred_labels, pred_scores, pred_boxes):
        if score > 0.5:
            box = patches.Rectangle((xmin, ymin), width, height, edgecolor="green", 
                                    facecolor="none", linewidth=1)
            ax.add_patch(box)
            ax.text(xmin, ymin, id_to_labels[label_id], color="white", 
                    bbox={"facecolor": "black", "alpha": 0.5})
    print([id_to_labels[label_id] for label_id in true_labels])
    print([id_to_labels[label_id] for label_id in pred_labels])
    plt.show()


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--experiment-name", default="voc", type=str)
    argparser.add_argument("--dataset-skip", default=5, type=int)
    argparser.add_argument("--dataset", default="voc", type=str)
    argparser.add_argument("--model", default="RetinaNet", type=str)
    argparser.add_argument("--output-dir", type=str, default="ckpts")
    argparser.add_argument("--save-path", type=str, default=None)
    args = argparser.parse_args()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    true_boxes = np.load(f"{save_path}/true_boxes.npy", allow_pickle=True)
    true_labels = np.load(f"{save_path}/true_labels.npy", allow_pickle=True)
    pred_boxes = np.load(f"{save_path}/pred_boxes.npy", allow_pickle=True)
    pred_labels = np.load(f"{save_path}/pred_labels.npy", allow_pickle=True)
    pred_scores = np.load(f"{save_path}/pred_scores.npy", allow_pickle=True)

    cntr = Counter([x for l in pred_labels for x in l])
    print(cntr)
    
    for iou_threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        ap_at_t = micro_avg_auprc(true_labels, true_boxes, pred_labels, pred_boxes, pred_scores, 
                                  iou_threshold=iou_threshold)
        print(f"AP@{iou_threshold:.2f}: {ap_at_t:.2f}")

    test_dataset = get_dataset("voc", "test")
    dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))

    idx = 20
    x, y = dataset[idx]
    x = unnormalize(x)
    x = x.transpose(0, 2).transpose(0, 1)
    show_image(x, true_labels[idx], true_boxes[idx], pred_labels[idx], pred_boxes[idx], 
               pred_scores[idx])


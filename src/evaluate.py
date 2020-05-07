import numpy as np
import pathlib
from argparse import ArgumentParser
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import patches 
from torch.utils.data import Subset
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
from src.datasets import *


def calibration_curve(labels, preds, n_bins=10, eps=1e-8, raise_on_nan=True):
    """
    Returns calibration curve at the pre-specified number of bins.
    -------
    obs_cdfs: (n_bins,) 
    pred_cdfs: (n_bins,) 
    bin_cnts: (n_bins,) 
    """
    bins = np.linspace(0., 1. + eps, n_bins + 1)
    bin_ids = np.digitize(preds, bins) - 1
    bin_cnts = np.bincount(bin_ids, minlength=n_bins)
    pred_cdfs = np.bincount(bin_ids, weights=preds, minlength=n_bins)
    obs_cdfs = np.bincount(bin_ids, weights=labels, minlength=n_bins)

    if np.any(bin_cnts == 0) and raise_on_nan:
        raise ValueError("Exists a bin with no predictions. Reduce the number of bins.")
    else:
        pred_cdfs = pred_cdfs[bin_cnts > 1] / bin_cnts[bin_cnts > 1]
        obs_cdfs = obs_cdfs[bin_cnts > 1] / bin_cnts[bin_cnts > 1]
        bin_cnts = bin_cnts[bin_cnts > 1]

    return obs_cdfs, pred_cdfs, bin_cnts


def flatten_detections_at_threshold(true_labels, true_boxes, pred_labels, pred_boxes, pred_scores, 
                                    iou_threshold=0.5):
    """
    Flatten a set of ground truth and predicted bouding boxes at a specified IoU threshold.

    Parameters
    ----------
    true_labels: (n, *) list of lists of true classes per image,
    true_boxes: (n, *, 4) list of lists of true box (x, y, w, h) per image,
    pred_labels: (n, *) list of lists of predicted classes per image,
    pred_boxes: (n, *, 4) list of lists of predicted box (x, y, w, h) per image,
    pred_scores: (n, *) list of lists of predicted scores per image.

    Returns
    -------
    flat_labels: (m,) array of binary labels.
    flat_preds: (m,) array of predicted probabilities in [0, 1].
    """
    n = len(true_labels)
    flat_labels = []
    flat_preds = []

    for idx in tqdm(range(n)):
        
        # (1) handle case where no detections were predicted
        if len(pred_boxes[idx]) == 0:
            flat_labels.extend([1] * len(true_boxes[idx]))
            flat_preds.extend([0.01] * len(true_boxes[idx]))
            continue

        ious = calculate_iou(np.array(true_boxes[idx]), np.array(pred_boxes[idx]), lib="numpy")

        pred_labels[idx] = np.array(pred_labels[idx])
        true_labels[idx] = np.array(true_labels[idx])

        # (2) assign each true box to the prediction with which it has largest overlap
        for true_box_idx in range(len(true_labels[idx])):
            
            label_match_mask = pred_labels[idx] == true_labels[idx][true_box_idx]

            # handle the case where no box with the right class was predicted
            if np.sum(label_match_mask) == 0:
                flat_labels.append(1)
                flat_preds.append(0.01)

            # pick out the predicted box with highest IoU 
            pred_box_idx = np.argmax(ious[true_box_idx, :] * label_match_mask)
            flat_labels.append(ious[true_box_idx, pred_box_idx] >= iou_threshold)
            flat_preds.append(pred_scores[idx][pred_box_idx])

            # mark the proposal box as used
            ious[:, pred_box_idx] = -1
        
        # (3) for all remaining predicted boxes assign a label of zero
        for pred_box_idx in range(len(pred_labels[idx])):
            if np.any(ious[:, pred_box_idx] >= 0.0):
                flat_labels.append(0)
                flat_preds.append(pred_scores[idx][pred_box_idx])

    return np.array(flat_labels), np.array(flat_preds)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--experiment-name", default="voc", type=str)
    argparser.add_argument("--dataset", default="voc", type=str)
    argparser.add_argument("--dataset-skip", default=1, type=int)
    argparser.add_argument("--output-dir", type=str, default="ckpts")
    argparser.add_argument("--num-images-saved", type=int, default=500)
    argparser.add_argument("--save-examples", action="store_true")
    args = argparser.parse_args()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    true_boxes = np.load(f"{save_path}/true_boxes.npy", allow_pickle=True)
    true_labels = np.load(f"{save_path}/true_labels.npy", allow_pickle=True)
    pred_boxes = np.load(f"{save_path}/pred_boxes.npy", allow_pickle=True)
    pred_labels = np.load(f"{save_path}/pred_labels.npy", allow_pickle=True)
    pred_scores = np.load(f"{save_path}/pred_scores.npy", allow_pickle=True)

    id_to_labels = get_id_to_labels(args.dataset)

    cntr = Counter([id_to_labels[i] for l in pred_labels for i in l])
    print("Frequency of predicted labels:", cntr)

    fig, (ax1, ax2) = plt.subplots(figsize=(7, 3), nrows=1, ncols=2)
    
    for iou_threshold in (0.5, 0.6, 0.7, 0.8, 0.9):

        # need to flatten labels 
        flat_labels, flat_preds = flatten_detections_at_threshold(true_labels, true_boxes,
                                                                  pred_labels, pred_boxes, 
                                                                  pred_scores, iou_threshold)
        
        # note: this is actually micro-averaged AUPRC across the set of classes
        #       not sure why it's called "average prediction" in the detection literature
        ap_at_t = average_precision_score(flat_labels, flat_preds, "micro")
        print(f"AP@{iou_threshold:.2f}: {ap_at_t:.2f}")

        ax1.clear()
        ax2.clear()

        # plot precision recall curve
        p, r, t = precision_recall_curve(flat_labels, flat_preds)
        ax1.plot(r, p, color="black")
        ax2.set_xlim((0, 1))
        ax2.set_ylim((0, 1))
        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")

        # plot calibration curve
        obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(flat_labels, flat_preds)
        ax2.scatter(pred_cdfs, obs_cdfs, color="black")
        ax2.plot((0, 1), (0, 1), "--", color="grey")
        ax2.set_xlim((0, 1))
        ax2.set_ylim((0, 1))
        ax2.set_xlabel("Expected CDF")
        ax2.set_ylabel("Observed CDF")

        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/{args.experiment_name}/pr_curve_at_{iou_threshold:.2f}.png")

    if not args.save_examples:
        exit()

    test_dataset = get_dataset(args.dataset, "test")
    dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))

    print("== Saving examples...")
    fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=1)

    for i in tqdm(range(args.num_images_saved)):

        x, _ = dataset[i]
        x = unnormalize(x).transpose(0, 2).transpose(0, 1)

        ax.clear()
        ax.imshow(x)

        # plot true boxes in red
        for label_id, (x, y, w, h) in zip(true_labels[i], true_boxes[i]):
            box = patches.Rectangle((x, y), w, h, edgecolor="red", 
                                    facecolor="none", linewidth=1)
            ax.add_patch(box)
            ax.text(x, y, id_to_labels[label_id], color="white",
                    bbox={"facecolor": "black", "alpha": 0.5})

        # plot predicted boxes in green, at a particular score threshold
        for label_id, score, (x, y, w, h) in zip(pred_labels[i], pred_scores[i], pred_boxes[i]):
            if score > 0.5:
                box = patches.Rectangle((x, y), w, h, edgecolor="green", 
                                        facecolor="none", linewidth=1)
                ax.add_patch(box)
                ax.text(x, y, id_to_labels[label_id], color="white", 
                        bbox={"facecolor": "black", "alpha": 0.5})

        folder = pathlib.Path(f"out/{args.experiment_name}")
        folder.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{folder}/{i}.png")


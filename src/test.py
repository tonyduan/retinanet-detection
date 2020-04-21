import numpy as np
import pathlib
import os
import sys
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.models import *
from src.datasets import *
from src.utils import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--dataset-skip", default=5, type=int)
    argparser.add_argument("--experiment-name", default="voc", type=str)
    argparser.add_argument("--dataset", default="voc", type=str)
    argparser.add_argument("--model", default="RetinaNet", type=str)
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    argparser.add_argument("--save-path", type=str, default=None)
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test")
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))

    if not args.save_path:
        save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    else:
        save_path = args.save_path

    model = eval(args.model)(device=args.device, num_classes=get_num_labels(args.dataset))
    saved_dict = torch.load(save_path)
    model.load_state_dict(saved_dict)
    model.eval()

    results = {
        "pred_labels": [],
        "pred_boxes": [],
        "pred_scores": [],
        "true_labels": [],
        "true_boxes": [],
    }

    for i in tqdm(range(len(test_dataset))):
        
        x, (true_labels, true_boxes) = test_dataset[i]
        x = x.to(args.device).transpose(1, 2)

        cls_preds, reg_preds = model.forward(x.unsqueeze(0))
        cls_preds = [p.to("cpu").detach().squeeze() for p in cls_preds]
        reg_preds = [p.to("cpu").detach().squeeze() for p in reg_preds]
        pred_labels, pred_boxes, pred_scores = RetinaNet.decode(x, cls_preds, reg_preds)
        results["pred_labels"].append([int(l) for l in pred_labels])
        results["pred_boxes"].append([b.numpy() for b in pred_boxes])
        results["pred_scores"].append(np.array(pred_scores))
        results["true_labels"].append(true_labels)
        results["true_boxes"].append([bbox_convert_corners_to_sizes(b) for b in true_boxes])

    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)


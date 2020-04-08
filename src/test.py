import numpy as np
import pathlib
import os
import sys
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.models import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--sample-size-pred", default=64, type=int)
    argparser.add_argument("--sample-size-cert", default=100000, type=int)
    argparser.add_argument("--dataset-skip", default=1, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--precision", default="half", type=str)
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    argparser.add_argument("--save-path", type=str, default=None)
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test", precision=args.precision)
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    if not args.save_path:
        save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    else:
        save_path = args.save_path

    model = eval(args.model)(dataset=args.dataset, device=args.device, precision=args.precision)
    saved_dict = torch.load(save_path)
    model.load_state_dict(saved_dict)
    model.eval()

    results = {
        "preds": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "labels": np.zeros(len(test_dataset)),
        "preds_nll": np.zeros(len(test_dataset)),
        "cams": np.zeros((len(test_dataset), 8, 8)),
    }

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        preds = model.forecast(model.forward(x))
        cam = model.class_activation_map(x, torch.argmax(preds.probs, dim=1))
        top_cats = preds.probs.argmax(dim=1)

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        results["preds"][lower:upper, :] = preds.probs.data.cpu().numpy()
        results["labels"][lower:upper] = y.data.cpu().numpy()
        results["cams"][lower:upper] = cam.data.cpu().numpy()
        results["preds_nll"][lower:upper] = -preds.log_prob(y).data.cpu().numpy()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

    train_dataset = get_dataset(args.dataset, "train")
    train_dataset = Subset(train_dataset, list(range(0, len(train_dataset), args.dataset_skip)))
    train_loader = DataLoader(train_dataset, shuffle=False, 
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    acc_meter = meter.AverageValueMeter()

    for x, y in tqdm(train_loader):

        x, y = x.to(args.device), y.to(args.device)
        logits = model.forward(x)
        top_cats = logits.argmax(dim=1)
        acc_meter.add(torch.sum(top_cats == y).cpu().data.numpy(), n=len(x))

    print("Training accuracy: ", acc_meter.value())
    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_path}/acc_train.npy",  acc_meter.value())


import logging
import pathlib
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models import *
from src.datasets import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--lr", default=0.001, type=float)
    argparser.add_argument("--alpha", default=0.25, type=float)
    argparser.add_argument("--gamma", default=2.0, type=float)
    argparser.add_argument("--pi", default=0.01, type=float)
    argparser.add_argument("--batch-size", default=16, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--num-epochs", default=20, type=int)
    argparser.add_argument("--print-every", default=20, type=int)
    argparser.add_argument("--save-every", default=50, type=int)
    argparser.add_argument("--experiment-name", default="coco", type=str)
    argparser.add_argument("--data-parallel", action="store_true")
    argparser.add_argument("--dataset", default="coco", type=str)
    argparser.add_argument('--output-dir', type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = RetinaNet(device=args.device, num_classes=get_num_labels(args.dataset),
                      alpha=args.alpha, gamma=args.gamma, pi=args.pi)
    model = nn.DataParallel(model) if args.data_parallel else model
    model.train()

    dataset = get_dataset(args.dataset, "train")
    train_loader = DataLoader(dataset,
                              shuffle=True,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              collate_fn=lambda x: collate_fn(x, args.dataset))

    x, labels, bbox = next(iter(train_loader))

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-4,
                          nesterov=True)
    annealer = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    loss_meter = meter.AverageValueMeter()
    time_meter = meter.TimeMeter(unit=False)

    train_losses = []

    for epoch in range(args.num_epochs):

        for i, (x, cls_tgts, reg_tgts) in enumerate(train_loader):

            x = x.to(args.device)
            for fm_no in range(len(cls_tgts)):
                cls_tgts[fm_no] = cls_tgts[fm_no].to(args.device)
                reg_tgts[fm_no] = reg_tgts[fm_no].to(args.device)

            loss = model.loss(x, cls_tgts, reg_tgts).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().data.numpy(), n=1)

            if i % args.print_every == 0:
                logger.info(f"Epoch: {epoch}\t"
                            f"Itr: {i} / {len(train_loader)}\t"
                            f"Loss: {loss_meter.value()[0]:.2f}\t"
                            f"Mins: {(time_meter.value() / 60):.2f}\t"
                            f"Experiment: {args.experiment_name}")
                train_losses.append(loss_meter.value()[0])
                loss_meter.reset()

        if (epoch + 1) % args.save_every == 0:
            save_path = f"{args.output_dir}/{args.experiment_name}/{epoch + 1}/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_ckpt.torch")

        annealer.step()

    pathlib.Path(f"{args.output_dir}/{args.experiment_name}").mkdir(parents=True, exist_ok=True)
    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    torch.save(model.state_dict(), save_path)
    args_path = f"{args.output_dir}/{args.experiment_name}/args.pkl"
    pickle.dump(args, open(args_path, "wb"))
    save_path = f"{args.output_dir}/{args.experiment_name}/train_losses.npy"
    np.save(save_path, np.array(train_losses))


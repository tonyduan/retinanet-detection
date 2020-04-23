import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.distributions import Categorical
from src.blocks import BasicBlock, Bottleneck
from src.utils import *


class RetinaNet(nn.Module):
    """
    RetinaNet [Lin et al. ICCV 2017].
    See `blocks.py` for BasicBlock and Bottleneck implementations.

    Consists of several steps:
    1. ResNet V1 backbone (bottom-up pass).
    2. Feature pyramid network (top-down pass).
    3. Classification anchor targets with focal loss.
    4. Regression anchor targets with smooth L1 loss.
    """
    resnet18_layers = [
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 64},  
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 128},
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 256},
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 512},
    ]
    resnet50_layers = [
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 256},
        {"block": Bottleneck, "num_blocks": 4, "num_filters": 512},
        {"block": Bottleneck, "num_blocks": 6, "num_filters": 1024},
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 2048},
    ]

    anchor_areas = [32 * 32, 64 * 64, 128 * 128, 256 * 256, 512 * 512]
    anchor_aspects = [0.5, 1.0, 2.0]
    anchor_scales = [1.0, 2 ** (1/3), 2 ** (2/3)]

    num_feature_maps = len(anchor_areas)
    num_anchors = len(anchor_aspects) * len(anchor_scales)

    def __init__(self, device, num_classes, fpn_dim=256, alpha=0.25, gamma=2.0, pi=0.01,
                 layers_config=resnet50_layers):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.fpn_dim = fpn_dim
        self.alpha = alpha
        self.gamma = gamma
        self.pi = pi
        self.layers_config = layers_config

        # standard initial preamble
        num_filters = 64
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # feature pyramid network
        # (1) first conv of each layer necessitates a downsampling via stride 2 conv
        # (2) don't include 1st layer in feature pyramid so no need for lateral or top down layer
        # (3) two extra convolutional layers for large object detection
        self.bottom_up_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        self.lateral_layers = nn.ModuleList()

        for layer_no, config in enumerate(layers_config):
            blocks = []
            for block_no in range(config["num_blocks"]):
                stride = 2 if layer_no != 0 and block_no == 0 else 1
                blocks.append(config["block"](in_filters=num_filters, 
                                              out_filters=config["num_filters"],
                                              stride=stride))
                num_filters = config["num_filters"]
            self.bottom_up_layers.append(nn.Sequential(*blocks))
            if layer_no == 0:
                continue
            self.lateral_layers.append(nn.Conv2d(num_filters, fpn_dim, kernel_size=1, stride=1))
            self.top_down_layers.append(nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1))

        self.conv6 = nn.Conv2d(num_filters, fpn_dim, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)

        # retina-net heads for classification and localization
        # each consists of four 3x3 convs followed by one 3x3 conv to the appropriate output size
        cls_head, reg_head = [], []
        for _ in range(4):
            cls_head.append(nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1))
            reg_head.append(nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1))
            cls_head.append(nn.ReLU())
            reg_head.append(nn.ReLU())
        cls_head.append(nn.Conv2d(fpn_dim, self.num_anchors*num_classes, kernel_size=3, padding=1))
        reg_head.append(nn.Conv2d(fpn_dim, 4 * self.num_anchors, kernel_size=3, padding=1))
        self.cls_head = nn.Sequential(*cls_head)
        self.reg_head = nn.Sequential(*reg_head)
    
        self.initialize_weights()

    def initialize_weights(self):
        """
        As suggested by [Lin et al. 2017], we initialize:
        1. Final classification layer needs biases = expit(0.01) due to low positive prevalence.
        2. Convolutional layers have weights ~ N(0, 0.1) and biases = 0.
        3. Batch normalization layers have weights = 1 and biases = 0.
        4. ResNet layers are initialized from ImageNet pretraining.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - self.pi) / self.pi))
        if self.layers_config is self.resnet18_layers:
            pretrained_state_dict = torch.load("src/weights/pretrained_resnet18.torch")
        elif self.layers_config is self.resnet50_layers:
            pretrained_state_dict = torch.load("src/weights/pretrained_resnet50.torch")
        else:
            raise ValueError("Trying to select an unsupported layer configuration.")
        self.load_state_dict(pretrained_state_dict, strict=False)
        for m in self.modules():
            m = m.to(self.device)

    def forward(self, x):
        """
        Notes
        -----
        The device of the input x must match device of this model.

        Returns
        -------
        cls_preds: (# fm)-list of preds each of size (num_batch, num_classes, num_anchors, fmw, fmh)
        reg_preds: (# fm)-list of preds each of size (num_batch, 4, num_anchors, fmw, fmh)
        """
        # standard resnet backbone preamble
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        # feature pyramid network, stored in p_layers in order largest feature map -> smallest
        # bottom-up calculates c_layers then top-down calculates p_layers in reverse order
        c_layers, p_layers = [], []
        for bottom_up_layer in self.bottom_up_layers:
            out = bottom_up_layer(out)
            c_layers.append(out)

        out = self.conv6(out)
        p_layers.append(out)
        out = self.conv7(out)
        p_layers.append(out)

        zipitem = zip(self.lateral_layers[::-1], self.top_down_layers[::-1], c_layers[::-1])
        for i, (lateral_layer, top_down_layer, c_layer) in enumerate(zipitem):
            if i == 0:
                out = lateral_layer(c_layer)
            else:
                fmw, fmh = c_layer.shape[-2], c_layer.shape[-1]
                out = F.interpolate(out, size=(fmw, fmh), mode="nearest")
                out = lateral_layer(c_layer) + out
            out = top_down_layer(out)
            p_layers.insert(0, out)
        
        # retina-net heads for classification and regression, reshaped appropriately
        cls_preds = [self.cls_head(p) for p in p_layers]
        reg_preds = [self.reg_head(p) for p in p_layers]
        for i in range(self.num_feature_maps):
            fmw, fmh = cls_preds[i].shape[-2], cls_preds[i].shape[-1]
            cls_preds[i] = cls_preds[i].reshape(-1, self.num_classes, self.num_anchors, fmw, fmh)
            reg_preds[i] = reg_preds[i].reshape(-1, 4, self.num_anchors, fmw, fmh)

        return cls_preds, reg_preds
    
    @classmethod
    def get_all_fm_anchor_widths_heights(cls):
        """
        Returns
        -------
        anchor_heights_widths: (# fm)-list of anchor box sizes each of size (num_anchors, 2)
        """
        anchor_sizes = torch.zeros((cls.num_feature_maps, cls.num_anchors, 2), dtype=torch.float)

        for i, area in enumerate(cls.anchor_areas):
            for j, (aspect_ratio, scale) in enumerate(itertools.product(cls.anchor_aspects, 
                                                                        cls.anchor_scales)):
                height = math.sqrt(area / aspect_ratio) * scale
                width = aspect_ratio * height * scale
                anchor_sizes[i][j][0] = width
                anchor_sizes[i][j][1] = height
        
        return anchor_sizes

    @classmethod
    def get_all_fm_anchor_boxes(cls, width, height):
        """
        Return all anchor boxes as tensors (on CPU), for a given input width and height.

        Returns
        -------
        anchors: (# fm)-length list of anchor boxes each of size (num_anchors, fm width, fm height)
        """
        all_fm_anchor_boxes = []

        for k, anchor_w_h in enumerate(cls.get_all_fm_anchor_widths_heights()):
            
            fmw, fmh = math.ceil(width / 2 ** (3 + k)), math.ceil(height / 2 ** (3 + k))
            grid_width, grid_height = width / fmw, height / fmh
            fm_anchor_boxes = torch.zeros(cls.num_anchors, fmw, fmh, 4)

            for i, j in itertools.product(range(fmw), range(fmh)):
                fm_anchor_boxes[:, i, j, 0] = (i + 0.5) * grid_width - 0.5 * anchor_w_h[:, 0]
                fm_anchor_boxes[:, i, j, 1] = (j + 0.5) * grid_height - 0.5 * anchor_w_h[:, 1]
                fm_anchor_boxes[:, i, j, 2] = anchor_w_h[:, 0]
                fm_anchor_boxes[:, i, j, 3] = anchor_w_h[:, 1]

            all_fm_anchor_boxes.append(fm_anchor_boxes)
            
        return all_fm_anchor_boxes

    @classmethod
    def encode(cls, x, labels, boxes, num_classes):
        """
        Encode an individual image x with corresponding lists of labels and boxes.

        Notes
        -----
        This is meant to be called by the dataloader on CPU. Input should *not* be batched.
        Classification targets are defined as
        -  1 if IoU with the anchor box in [0.5, 1.0] (positive label)
        - -1 if IoU with the anchor box in [0.4, 0.5] (to be ignored in the loss)
        -  0 if IoU with the anchor box in [0.0, 0.4] (negative label)
        Regression targets are defined as in [Ren et al. NeurIPS 2015] for positive classes only.

        Returns
        -------
        all_fm_cls_tgts: (# fm)-list of tgts each of size (num_classes, num_anchors, fmw, fmh)
        all_fm_reg_tgts: (# fm)-list of tgts each of size (4, num_anchors, fmw, fmh)
        """
        C, W, H = x.shape
        all_fm_anchor_boxes = cls.get_all_fm_anchor_boxes(W, H)
        all_fm_cls_tgts = []
        all_fm_reg_tgts = []

        for fm_anchor_boxes in all_fm_anchor_boxes:

            fm_cls_tgts = torch.zeros(fm_anchor_boxes.shape[:-1])
            fm_cls_tgts = fm_cls_tgts.unsqueeze(0).repeat((num_classes, 1, 1, 1))
            fm_reg_tgts = torch.zeros(fm_anchor_boxes.shape[:-1])
            fm_reg_tgts = fm_reg_tgts.unsqueeze(0).repeat((4, 1, 1, 1))

            for label, box in zip(labels, boxes):

                original_shape = fm_anchor_boxes.shape
                iou = calculate_iou(fm_anchor_boxes.reshape(-1, 4), box.unsqueeze(0))
                iou = iou.reshape(original_shape[:-1])

                fm_cls_tgts[label][iou > 0.5] = 1.0
                fm_cls_tgts[label][(iou < 0.5) & (iou > 0.4)] = -1.0
                fm_reg_tgts[0][iou > 0.5] = (box[0] - fm_anchor_boxes[iou > 0.5][:, 0]) / \
                                            fm_anchor_boxes[iou > 0.5][:, 2]
                fm_reg_tgts[1][iou > 0.5] = (box[1] - fm_anchor_boxes[iou > 0.5][:, 1]) / \
                                            fm_anchor_boxes[iou > 0.5][:, 3]
                fm_reg_tgts[2][iou > 0.5] = torch.log(box[2] / fm_anchor_boxes[iou > 0.5][:, 2])
                fm_reg_tgts[3][iou > 0.5] = torch.log(box[3] / fm_anchor_boxes[iou > 0.5][:, 3])

            all_fm_cls_tgts.append(fm_cls_tgts)
            all_fm_reg_tgts.append(fm_reg_tgts)
    
        return all_fm_cls_tgts, all_fm_reg_tgts

    @classmethod
    def decode(cls, x, cls_preds, reg_preds, cls_threshold=0.05, nms_threshold=0.5):
        """
        Decode an individual image x with corresponding lists of classification & regression preds.

        Notes
        -----
        This is meant to be called on CPU. Input should *not* be batched.

        Parameters
        ----------
        cls_threshold: float in [0, 1] to determine minimum probability of detection for results
        nms_threshold: float in [0, 1] to determine IoU threshold for non-maximal suppression

        Returns
        -------
        labels: (# detections)-list of classes
        boxes: (# detections)-list of boxes each of shape (4,)
        scores: (# detections)-list of probabilities in the range [cls_threshold, 1]
        """
        C, W, H = x.shape
        all_fm_anchor_boxes = cls.get_all_fm_anchor_boxes(W, H)

        label_to_all_boxes = defaultdict(list)
        label_to_all_scores = defaultdict(list)
        labels = []
        boxes = []
        scores = []

        for fm_cls_preds, fm_reg_preds, fm_anchor_boxes in zip(cls_preds, reg_preds, 
                                                               all_fm_anchor_boxes):

            fm_cls_preds = torch.sigmoid(fm_cls_preds)
            idxs = (fm_cls_preds > cls_threshold).nonzero()

            for label, anchor_no, i, j in idxs:

                label = int(label)
                anchor_box = fm_anchor_boxes[anchor_no, i, j] 
                offset = fm_reg_preds[:, anchor_no, i, j]
                box = (anchor_box[0] + anchor_box[2] * offset[0],
                       anchor_box[1] + anchor_box[3] * offset[1],
                       anchor_box[2] * math.exp(offset[2]),
                       anchor_box[3] * math.exp(offset[3]))
                label_to_all_boxes[label].append(torch.stack(box))
                label_to_all_scores[label].append(fm_cls_preds[label, anchor_no, i, j])
            
        for label, all_boxes in label_to_all_boxes.items():

            all_boxes = torch.stack(all_boxes)
            all_scores = torch.stack(label_to_all_scores[label])
            sel_boxes, sel_scores = calculate_nms(all_boxes, all_scores, threshold=nms_threshold)
            boxes.extend(sel_boxes)
            scores.extend(sel_scores)
            labels.extend([label] * len(sel_scores))

        assert len(labels) == len(boxes) == len(scores)
        return labels, boxes, scores

    def loss(self, x, cls_tgts, reg_tgts):
        """
        Focal loss.

        Parameters
        ----------
        x: (batch_size, 3, width, height) tensor of images
        cls_tgts: (# fm)-list of tgts each of size (batch_size, num_classes, num_anchors, fmw, fmh)
        reg_tgts: (# fm)-list of tgts each of size (batch_size, 4, num_anchors, fmw, fmh)

        Returns
        -------
        loss: (batch_size)-length tensor of losses
        """
        cls_preds, reg_preds = self.forward(x)
        loss = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        num_positive_anchors = torch.zeros(len(x), device=x.device, dtype=x.dtype)

        for cls_tgt, reg_tgt, cls_pred, reg_pred in zip(cls_tgts, reg_tgts, cls_preds, reg_preds):
            
            # classification: focal loss, ignore cases where cls_tgt == -1
            pos_tgts, neg_tgts = cls_tgt == 1, cls_tgt == 0
            a_t = self.alpha * pos_tgts + (1 - self.alpha) * neg_tgts
            p_t = torch.sigmoid(cls_pred) * pos_tgts + (1 - torch.sigmoid(cls_pred)) * neg_tgts
            log_pt = F.logsigmoid(cls_pred) * pos_tgts + F.logsigmoid(-cls_pred) * neg_tgts
            cls_loss = -a_t * (1 - p_t) ** self.gamma * log_pt

            # regression: include all anchor boxes where *any* class is positive
            reg_mask = (cls_tgt == 1).sum(dim = 1) > 0
            reg_loss = F.smooth_l1_loss(reg_pred, reg_tgt, reduction="none").sum(dim=1) * reg_mask

            cls_loss = cls_loss.sum(dim=(1, 2, 3, 4))
            reg_loss = reg_loss.sum(dim=(1, 2, 3))
            loss += cls_loss + reg_loss
            num_positive_anchors += pos_tgts.sum(dim=(1, 2, 3, 4))

        return loss / num_positive_anchors.clamp(min=1)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.datasets import *
from src.blocks import BasicBlock, Bottleneck, BasicBlockV2, BottleneckV2


class Detector(nn.Module):
    """
    Generic detector module that sets up appropriate normalization and log-lik loss.
    """
    def __init__(self, dataset, device, precision="float"):
        super().__init__()
        self.device = device
        self.precision = precision
        self.norm = NormalizeLayer(get_normalization_shape(dataset),
                                   **get_normalization_stats(dataset))

    def initialize_weights(self):
        raise NotImplementedError

    def set_device_and_precision(self):
        for m in self.modules():
            m = m.to(self.device)
            if self.precision == "half":
                m = m.half()
            if self.precision == "float":
                m = m.float()
            if self.precision == "double":
                m = m.double()

    def forward(self, x):
        raise NotImplementedError

    def forecast(self, theta):
        return Categorical(logits=theta)

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)


class NormalizeLayer(nn.Module):
    """
    Normalizes across the first non-batch axis.

    Examples:
        (64, 3, 32, 32) [CIFAR] => normalizes across channels.
        (64, 8)         [UCI]   => normalizes across features
    """
    def __init__(self, dim, mu, sigma):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu).reshape(dim), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma).reshape(dim), requires_grad=False)

    def forward(self, x):
        return (x - self.mu) /self.sigma


class ResNet(Classifier):
    """
    Classic residual network [He et al. CVPR 2016].
    See `blocks.py` for BasicBlock and Bottleneck implementations.

    Also supports re-ordered "pre-activation" residual network [He et al. ECCV 2016].
    i.e. (conv => batch norm => activation) replaced with (batch norm => activation => conv).
    In this case we need to use BasicBlockV2 or BottleneckV2.

    Consists of:
        1. Initial convolution, batch norm, then max pool (strided if ImageNet).
        2. Sequence of layers, each consisting of a number of blocks with fixed number of filters.
           At the first conv of each layer we downsample by half by setting stride = 2.
        3. Global average pool, then linear layer to logits.

    Configurations for canonical models are described below [Zagoruyko and Komodakis ECCV 2016].
    """
    # == ImageNet models
    resnet18_layers = [
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 64},  # 64 x 56 x 56 output
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 128}, # 128 x 28 x 28 output
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 256}, # 256 x 14 x 14 output
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 512}, # 128 x 7 x 7 output
    ]
    resnet50_layers = [
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 256},
        {"block": Bottleneck, "num_blocks": 4, "num_filters": 512},
        {"block": Bottleneck, "num_blocks": 6, "num_filters": 1024},
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 2048},
    ]
    wrn_50_2_layers = [
        {"block": BottleneckV2, "num_blocks": 3, "num_filters": 256, "squeeze": 2},
        {"block": BottleneckV2, "num_blocks": 4, "num_filters": 512, "squeeze": 2},
        {"block": BottleneckV2, "num_blocks": 6, "num_filters": 1024, "squeeze": 2},
        {"block": BottleneckV2, "num_blocks": 3, "num_filters": 2048, "squeeze": 2},
    ]

    # == CIFAR-10 models
    resnet110_layers = [
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 16}, # 16 x 32 x 32 output
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 32}, # 32 x 16 x 16 output
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 64}, # 64 x 8 x 8 output
    ]
    wrn_40_2_layers = [
        {"block": BasicBlockV2, "num_blocks": 6, "num_filters": 32},  # 32 x 32 x 32 output
        {"block": BasicBlockV2, "num_blocks": 6, "num_filters": 64},  # 64 x 16 x 16 output
        {"block": BasicBlockV2, "num_blocks": 6, "num_filters": 128}, # 128 x 8 x 8 output
    ]
    wrn_28_10_layers = [
        {"block": BasicBlockV2, "num_blocks": 4, "num_filters": 160}, # 160 x 32 x 32 output
        {"block": BasicBlockV2, "num_blocks": 4, "num_filters": 320}, # 320 x 16 x 16 output
        {"block": BasicBlockV2, "num_blocks": 4, "num_filters": 640}, # 640 x 8 x 8 output
    ]

    def __init__(self, dataset, device, precision, pre_activation=True):

        super().__init__(dataset, device, precision)
        self.pre_activation = pre_activation

        if dataset == "imagenet":
            num_filters = 64
            self.conv = nn.Conv2d(3, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            layers_config = self.wrn_50_2_layers
        else:
            num_filters = 16
            self.conv = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
            layers_config = self.wrn_40_2_layers

        self.bn_init = nn.BatchNorm2d(num_filters)
        self.blocks = nn.ModuleList()

        for layer_no, config in enumerate(layers_config):
            for block_no in range(config["num_blocks"]):
                stride = 2 if layer_no != 0 and block_no == 0 else 1
                self.blocks.append(config["block"](in_filters=num_filters, 
                                                   out_filters=config["num_filters"],
                                                   stride=stride,
                                                   **config))
                num_filters = config["num_filters"]

        self.bn_final = nn.BatchNorm2d(num_filters) if pre_activation else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_filters, get_num_labels(dataset))
        self.initialize_weights()
        self.set_device_and_precision()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv(out)
        out = self.bn_init(out)
        out = F.relu(out)
        out = self.maxpool(out)
        for block in self.blocks:
            out = block(out)
        out = self.bn_final(out)
        out = F.relu(out)
        out = self.avgpool(out)
        out = self.linear(self.flatten(out))
        return out

    def class_activation_map(self, x, y):
        out = self.norm(x)
        out = self.conv(out)
        out = self.bn_init(out)
        out = F.relu(out)
        out = self.maxpool(out)
        for block in self.blocks:
            out = block(out)
        out = self.bn_final(out)
        out = F.relu(out)
        cam = self.linear.weight[y].unsqueeze(1) @ out.flatten(start_dim=2, end_dim=-1) 
        cam = cam.squeeze(1) + self.linear.bias[y].unsqueeze(1)
        cam = cam.view(-1, *out.shape[2:]) 
        return cam


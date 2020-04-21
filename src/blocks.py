import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    BasicBlock: two 3x3 convs followed by a residual connection then ReLU.

    [He et al. CVPR 2016]

        BasicBlock(x) = ReLU( x + Conv3x3( ReLU( Conv3x3(x) ) ) )

    Batch norm follows each 3x3 convolution, and residual connection has a projection if
    the dimensionality changes (either due to striding or different in/out filters).

    Note the dimensionality of output equals dimensionality of input divided by stride.
        stride = 1: (32 x 32) => (32 x 32)
        stride = 2: (32 x 32) => (16 x 16)
    """
    def __init__(self, in_filters, out_filters, stride=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)
        if stride == 1 and in_filters == out_filters:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_filters, out_filters, kernel_size=1, 
                                                     stride=stride, bias=False),
                                          nn.BatchNorm2d(out_filters))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck: a 1x1 conv to reduce dimensionality, then 3x3 conv, then 1x1 conv back up.

    [He et al. CVPR 2016]

    Here we squeeze dimensionality by a default factor of 1/4 of the output number of filters.

        Bottleneck(x) = ReLU( x + Conv1x1( ReLU( Conv3x3( ReLu( Conv1x1(x)) ) ) ) )

    Batch norm follows each convolution, and residual connection has a projection if 
    the dimensionality changes (either due to striding or different in/out filters).
    """
    def __init__(self, in_filters, out_filters, stride=1, squeeze=4, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters // squeeze, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters // squeeze)
        self.conv2 = nn.Conv2d(out_filters // squeeze, out_filters // squeeze, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters // squeeze)
        self.conv3 = nn.Conv2d(out_filters // squeeze, out_filters, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_filters)
        if stride == 1 and in_filters == out_filters:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_filters, out_filters, kernel_size=1, 
                                                     stride=stride, bias=False),
                                          nn.BatchNorm2d(out_filters))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out



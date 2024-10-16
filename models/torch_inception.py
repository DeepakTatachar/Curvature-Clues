"""
Author: Deepak Ravikumar Tatachar
Copyright © 2024 Deepak Ravikumar, Nano(Neuro) Electronics Research Lab, Purdue University
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class SmallInception(nn.Module):
    """Simple inception like network for small images (cifar/mnist)."""

    def __init__(self, num_classes=10, with_residual=False, large_inputs=False):
        super(SmallInception, self).__init__()
        self._num_classes = num_classes

        self._large_inputs = large_inputs
        if large_inputs:
            self.conv1 = ConvBNReLU(3, 96, (7, 7), stride=2)
        else:
            self.conv1 = ConvBNReLU(3, 96, (3, 3), padding=1)

        in_ch = 96
        self.stage1 = SmallInceptionStage([(32, 32), (32, 48)], 160, with_residual=with_residual, in_chs=in_ch)

        self.stage2 = SmallInceptionStage(
            [(112, 48), (96, 64), (80, 80), (48, 96)],
            240,
            with_residual=with_residual,
            in_chs=self.stage1.out_ch,
        )

        self.stage3 = SmallInceptionStage(
            [(176, 160), (176, 160)],
            0,
            with_residual=with_residual,
            in_chs=self.stage2.out_ch,
        )

        self._pred = nn.Linear(self.stage3.out_ch, num_classes)

    def compute_repr(self, inputs):
        x = inputs
        if self._large_inputs:
            x = self.conv1(x)
            x = torch.nn.functional.max_pool2d(x, ksize=3, strides=2, padding=1)
        else:
            x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # global pooling
        x = torch.amax(x, axis=(2, 3))
        return x

    def forward(self, inputs):
        logits = self._pred(self.compute_repr(inputs))
        return logits


class MultiBranchBlock(nn.Module):
    """Simple inception-style multi-branch block."""

    def __init__(self, channels_1x1, channels_3x3, in_ch):
        super(MultiBranchBlock, self).__init__()

        self.conv1x1 = ConvBNReLU(in_ch, channels_1x1, 1, padding=0)
        self.conv3x3 = ConvBNReLU(in_ch, channels_3x3, 3, padding=1)
        self.out_ch = channels_1x1 + channels_3x3

    def forward(self, inputs):
        return torch.concat([self.conv1x1(inputs), self.conv3x3(inputs)], axis=1)


class SmallInceptionStage(nn.Module):
    """Stage for SmallInception model."""

    def __init__(self, mb_channels, downsample_channels, with_residual=False, in_chs=0):
        super(SmallInceptionStage, self).__init__()
        self._mb_channels = mb_channels
        self._downsample_channels = downsample_channels
        self._with_residual = with_residual

        self._body = []
        in_ch = in_chs
        for i, (ch1x1, ch3x3) in enumerate(mb_channels):
            mb = MultiBranchBlock(ch1x1, ch3x3, in_ch)
            in_ch = mb.out_ch
            self._body.append((f"{i}", mb))
        if downsample_channels > 0:
            self._downsample = ConvBNReLU(in_ch, downsample_channels, kernel_shape=3, stride=2, padding=1)
            self.out_ch = downsample_channels
        else:
            self._downsample = None
            self.out_ch = in_ch

        self._body = nn.Sequential(OrderedDict(self._body))

    def forward(self, inputs):
        x = inputs
        x = self._body(x)
        if self._with_residual:
            x += inputs

        if self._downsample:
            x = self._downsample(x)
        return x


class ConvBNReLU(nn.Module):
    """Conv -> BatchNorm -> ReLU."""

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_shape,
        stride=1,
        rate=1,
        padding=0,
        w_init=None,
        name=None,
    ):
        super(ConvBNReLU, self).__init__()

        # Replicate tensorflow behavior
        self.asymmetric_pad = padding == 1 and stride == 2
        self.stride = stride

        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.out_ch = output_channels

    def forward(self, inputs):
        if not self.asymmetric_pad:
            x = self.conv(inputs)
        else:
            padded_x = F.pad(inputs, (0, 1, 0, 1), mode="constant", value=0)
            x = F.conv2d(padded_x, self.conv.weight, self.conv.bias, stride=self.stride, padding="valid")
        x = self.bn(x)
        return torch.nn.functional.relu(x)

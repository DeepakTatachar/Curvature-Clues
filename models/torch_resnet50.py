"""
Author: Deepak Ravikumar Tatachar
Copyright © 2024 Deepak Ravikumar, Nano(Neuro) Electronics Research Lab, Purdue University
"""

import torch.nn as nn
import torch
from typing import Mapping, Optional, Sequence, Union
import torch.nn.functional as F

class BottleNeckBlockV1(nn.Module):
    """Bottleneck Block for a ResNet implementation."""

    def __init__(self,
        in_channels: int,     
        channels: int,
        stride: Union[int, Sequence[int]],
        use_projection: bool,
        bn_config: Mapping[str, float]):
        
        super(BottleNeckBlockV1, self).__init__()
        self._channels = channels
        self._stride = stride
        self._use_projection = use_projection
        self._bn_config = bn_config

        batchnorm_args = {}
        batchnorm_args.update(bn_config)

        if self._use_projection:
            self._proj_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                padding=0)

            self._proj_batchnorm = nn.BatchNorm2d(
                channels, 
                **batchnorm_args)

        self._layers = []
        conv_0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=0)

        self._layers.append(
            [
                conv_0,
                nn.BatchNorm2d(channels // 4, **batchnorm_args),
                nn.ReLU()
            ])

        conv_1 = nn.Conv2d(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1)

        self._layers.append(
            [
                conv_1,
                nn.BatchNorm2d(channels // 4, **batchnorm_args),
                nn.ReLU()
            ])

        conv_2 = nn.Conv2d(
            in_channels=channels // 4,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=0)

        batchnorm_2 = nn.BatchNorm2d(channels, **batchnorm_args)
        self.out_channels = channels
        self._layers.append([conv_2, batchnorm_2])
        seq_layers = []
        for layer in self._layers:
            seq_layers.append(nn.Sequential(*layer))
        self._layers = nn.Sequential(*seq_layers)

    def forward(self, inputs):
        if self._use_projection:
            shortcut = self._proj_conv(inputs)
            shortcut = self._proj_batchnorm(shortcut)
        else:
            shortcut = inputs

        net = self._layers(inputs)
        
        return nn.functional.relu(net + shortcut)


class BlockGroup(nn.Module):
    """Higher level block for ResNet implementation."""

    def __init__(
            self,
            in_channels: int,
            channels: int,
            num_blocks: int,
            stride: Union[int, Sequence[int]],
            bn_config: Mapping[str, float]):
        super(BlockGroup, self).__init__()
        self._channels = channels
        self._num_blocks = num_blocks
        self._stride = stride
        self._bn_config = bn_config

        blocks = []
        in_channels = in_channels
        for id_block in range(num_blocks):
            block = BottleNeckBlockV1(
                in_channels=in_channels,
                channels=channels,
                stride=stride if id_block == 0 else 1,
                use_projection=(id_block == 0),
                bn_config=bn_config)

            blocks.append(block)
            in_channels = block.out_channels

        self._blocks = nn.Sequential(*blocks)

        self.out_channels = self._blocks[-1].out_channels

    def forward(self, inputs):
        net = inputs
        for block in self._blocks:
            net = block(net)
        return net


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self,
            blocks_per_group_list: Sequence[int],
            num_classes: int,
            bn_config: Optional[Mapping[str, float]] = None,
            resnet_v2: bool = False,
            channels_per_group_list: Sequence[int] = (256, 512, 1024, 2048),
            name: Optional[str] = None):
        """Constructs a ResNet model.

        Args:
        blocks_per_group_list: A sequence of length 4 that indicates the number of
            blocks created in each group.
        num_classes: The number of classes to classify the inputs into.
        bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
            passed on to the `BatchNorm` layers. By default the `decay_rate` is
            `0.9` and `eps` is `1e-5`.
        resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
            False.
        channels_per_group_list: A sequence of length 4 that indicates the number
            of channels used for each block in each group.
        name: Name of the module.
        """
        super(ResNet, self).__init__()
        if bn_config is None:
            bn_config = {"momentum": 0.1, "eps": 1e-5}
        
        self._bn_config = bn_config
        self._resnet_v2 = resnet_v2

        # Number of blocks in each group for ResNet.
        if len(blocks_per_group_list) != 4:
            raise ValueError(
            "`blocks_per_group_list` must be of length 4 not {}".format(
                len(blocks_per_group_list)))
        self._blocks_per_group_list = blocks_per_group_list

        # Number of channels in each group for ResNet.
        if len(channels_per_group_list) != 4:
            raise ValueError(
            "`channels_per_group_list` must be of length 4 not {}".format(
                len(channels_per_group_list)))
        self._channels_per_group_list = channels_per_group_list

        self._initial_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            bias=False,
            padding=3)

        if not self._resnet_v2:
            self._initial_batchnorm = nn.BatchNorm2d(
                64,
                **bn_config)

        strides = [1, 2, 2, 2]
        in_channels = 64
        self._block_groups = []
        for i in range(4):
            block_group = BlockGroup(
                    in_channels=in_channels,
                    channels=self._channels_per_group_list[i],
                    num_blocks=self._blocks_per_group_list[i],
                    stride=strides[i],
                    bn_config=bn_config)
            
            in_channels = block_group.out_channels
            self._block_groups.append(block_group)

        self._block_groups = nn.Sequential(*self._block_groups)
        
        if self._resnet_v2:
            self._final_batchnorm = nn.BatchNorm2d(
                self._block_groups[-1].out_channels,
                **bn_config)

        self._logits = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, inputs):
        net = inputs
        net = self._initial_conv(net)
        if not self._resnet_v2:
            net = self._initial_batchnorm(net)
            net = nn.functional.relu(net)

        # Tensorflow does asymmetric padding
        net = F.pad(net, (0, 1, 0, 1), mode="constant", value=0)
        net = nn.functional.max_pool2d(
            net, 
            kernel_size=3, 
            stride=2, 
            padding=0)

        for block_group in self._block_groups:
            net = block_group(net)

        if self._resnet_v2:
            net = self._final_batchnorm(net)
            net = nn.functional.relu(net)
        net = torch.mean(net, axis=[2,3])
        return self._logits(net)


class ResNet50(ResNet):
    """ResNet50 module."""

    def __init__(self,
                num_classes: int,
                bn_config: Optional[Mapping[str, float]] = None,
                resnet_v2: bool = False,
                name: Optional[str] = None):
        """Constructs a ResNet model.

        Args:
        num_classes: The number of classes to classify the inputs into.
        bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
            passed on to the `BatchNorm` layers.
        resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
            False.
        name: Name of the module.
        """
        super().__init__(
            [3, 4, 6, 3],
            num_classes=num_classes,
            bn_config=bn_config,
            resnet_v2=resnet_v2,
            name=name)
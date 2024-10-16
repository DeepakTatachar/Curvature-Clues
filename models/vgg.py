"""
Modified from https://github.com/pytorch/vision.git
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class vgg(nn.Module):
    """
    VGG model
    """

    def __init__(self, cfg, batch_norm_conv=True, batch_norm_linear=False, in_channels_conv=3, num_classes=10):
        super(vgg, self).__init__()
        self.num_classes = num_classes
        self.cfg_dict_conv = {
            "05": [
                64,
                "M",
                128,
                "M",
            ],
            "07": [64, "M", 128, "M", 256, "M", 512, "M"],
            "09": [64, "M", 128, "M", 256, "M", 512, "M", 512, 512, "M"],
            "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            "19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
            "05kp": [42, "M", 114, "M"],
            "05kp_": [13, "M", 91, "M"],
            "07kp": [51, "M", 120, "M", 176, "M", 434, "M"],
            "07kp_": [13, "M", 90, "M", 214, "M", 445, "M"],
            "09kp": [50, "M", 106, "M", 189, "M", 396, "M"],
            "09kp_": [13, "M", 85, "M", 199, "M", 449, "M"],
            "11kp": [52, "M", 118, "M", 208, "M", 413, "M"],
            "11kp_": [13, "M", 84, "M", 188, 222, "M", 450, "M"],
            "13kp": [47, "M", 117, "M", 227, "M", 438, "M"],
            "13kp_": [14, 35, "M", 97, 104, "M", 227, 237, "M", 448, "M"],
            "16kp": [49, 54, "M", 112, "M", 236, 242, "M", 454, "M"],
            "16kp_": [14, 37, "M", 97, 101, "M", 221, 232, 239, "M", 422, "M"],
            "19kp": [53, 59, "M", 118, "M", 222, 231, 232, "M", 486, "M"],
            "19kp_": [13, 36, "M", 96, 103, "M", 224, 232, 235, "M", 329, "M"],
            "16pr": [11, 42, "M", 103, 118, "M", 238, 249, "M", 424, "M"],
        }

        self.cfg_dict_linear = {
            "05": [8192, 512, 512, self.num_classes],
            "07": [2048, 512, 512, self.num_classes],
            "09": [512, 512, 512, self.num_classes],
            "13": [512, 512, 512, self.num_classes],
            "11": [512, 512, 512, self.num_classes],
            "16": [512, 512, 512, self.num_classes],
            "19": [512, 512, 512, self.num_classes],
            "05kp": [7296, 512, 512, self.num_classes],
            "05kp_": [5824, 512, 512, self.num_classes],
            "07kp": [1736, 512, 512, self.num_classes],
            "07kp_": [1780, 512, 512, self.num_classes],
            "09kp": [1584, 512, 512, self.num_classes],
            "09kp_": [1796, 512, 512, self.num_classes],
            "11kp": [1652, 512, 512, self.num_classes],
            "11kp_": [1800, 512, 512, self.num_classes],
            "13kp": [1752, 512, 512, self.num_classes],
            "13kp_": [1792, 512, 512, self.num_classes],
            "16kp": [1816, 512, 512, self.num_classes],
            "16kp_": [1688, 512, 512, self.num_classes],
            "19kp": [1944, 512, 512, self.num_classes],
            "19kp_": [1316, 512, 512, self.num_classes],
            "16pr": [1696, 512, 512, self.num_classes],
        }

        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_linear = batch_norm_linear
        self.in_channels_conv = in_channels_conv
        self.features = self.make_layers_conv(cfg)
        self.classifier = self.make_layers_linear(cfg)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def make_layers_conv(self, config):
        layers = []
        in_channels = self.in_channels_conv
        batch_norm = self.batch_norm_conv
        cfg = self.cfg_dict_conv[config]
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def make_layers_linear(self, config):
        layers = []
        cfg = self.cfg_dict_linear[config]
        in_channels = cfg[0]
        cfg = cfg[1:]
        batch_norm = self.batch_norm_linear
        for i in range(len(cfg) - 1):
            v = cfg[i]
            if v == "M":
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                linear = nn.Linear(in_channels, v)
                if batch_norm:
                    layers += [nn.Dropout(), linear, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Dropout(), linear, nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.Linear(in_channels, cfg[i + 1])]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

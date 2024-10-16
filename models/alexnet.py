import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=12, stride=4, padding=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=384)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=384)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(num_features=256)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout5 = nn.Dropout(p=0.5)

        self.linear6 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.batchnorm6 = nn.BatchNorm1d(num_features=4096)
        self.dropout6 = nn.Dropout(p=0.5)

        self.linear7 = nn.Linear(in_features=4096, out_features=4096)
        self.batchnorm7 = nn.BatchNorm1d(num_features=4096)

        self.classifier = nn.Linear(in_features=4096, out_features=self.num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.maxpool2(nn.functional.relu(self.batchnorm2(self.conv2(x))))
        x = self.maxpool3(nn.functional.relu(self.batchnorm3(self.conv3(x))))
        x = nn.functional.relu(self.batchnorm4(self.conv4(x)))
        x = self.maxpool5(nn.functional.relu(self.batchnorm5(self.conv5(x))))

        x = x.flatten(1)
        x = nn.functional.relu(self.batchnorm6(self.linear6(x)))
        x = nn.functional.relu(self.batchnorm7(self.linear7(x)))
        x = self.classifier(x)
        return x

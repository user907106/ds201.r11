import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolution with 5 x 5 kernel + 2 padding: 28 x 28 x 6
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            padding=2
        )

        # Pool with 2 x 2 average kernel + 2 stride: 14 x 14 x 6
        self.pool1 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        # Convolution with 5 x 5 kernel (no pad): 10 x 10 x 16
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5)
        )

        # Pool with 2 x 2 average kernel + 2 stride: 5 x 5 x 16
        self.pool2 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        # Dense: 120 fully connected neurons
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5)
        )

        # Dense: 84 fully connected neurons
        self.fc = nn.Linear(
            in_features=120,
            out_features=84
        )

        # Dense: 10 fully connected neurons
        self.output = nn.Linear(
            in_features=84,
            out_features=num_classes
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        features = self.conv1(images)       # (batch_size,   6, 28, 28)

        features = F.sigmoid(features)      # (batch_size,   6, 28, 28)

        features = self.pool1(features)     # (batch_size,   6, 14, 14)

        features = self.conv2(features)     # (batch_size,  16, 10, 10)

        features = F.sigmoid(features)      # (batch_size,  16, 10, 10)

        features = self.pool2(features)     # (batch_size,  16,  5,  5)

        features = self.conv3(features)     # (batch_size, 120,  1,  1)

        features = F.sigmoid(features)      # (batch_size, 120,  1,  1)

        features = features.view(-1, 120)   # (bacth_size, 120)

        features = self.fc(features)        # (batch_size,  84)

        features = F.sigmoid(features)      # (batch_size,  84)

        outputs = self.output(features)     # (batch_size,  10)

        return outputs

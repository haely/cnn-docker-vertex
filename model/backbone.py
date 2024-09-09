import torch
import torch.nn as nn

class SimpleBackbone(nn.Module):
    def __init__(self, num_channels=3):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        return x


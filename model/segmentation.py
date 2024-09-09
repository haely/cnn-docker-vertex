# simple u-net like structure for esgmentation. Copied from github open. Replace/Extend if needed
import torch
import torch.nn as nn

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Segmentation output


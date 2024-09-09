import torch
import torch.nn as nn

class KeypointDetection(nn.Module):
    def __init__(self, input_dim, num_keypoints):
        super(KeypointDetection, self).__init__()
        self.fc = nn.Linear(input_dim, num_keypoints * 2)  # x, y for each keypoint

    def forward(self, x):
        return self.fc(x)  # Return coordinates for each keypoint


import torch
import torch.nn as nn
from .backbone import SimpleBackbone
from .module import AttentionModule
from .keypoint import KeypointDetection
from .segmentation import SimpleSegmentationModel

class FullModel(nn.Module):
    def __init__(self, num_classes, num_keypoints):
        super(FullModel, self).__init__()
        self.backbone = SimpleBackbone()
        self.attention = AttentionModule(input_dim=128)
        self.keypoint_detector = KeypointDetection(input_dim=128, num_keypoints=num_keypoints)
        self.segmentation_model = SimpleSegmentationModel(num_classes=num_classes)

    def forward(self, x):
        # Pass through backbone
        features = self.backbone(x)

        # Attention module
        features = self.attention(features)

        # Keypoint detection
        keypoints = self.keypoint_detector(features)

        # Segmentation
        segmentation_output = self.segmentation_model(features)

        return segmentation_output, keypoints


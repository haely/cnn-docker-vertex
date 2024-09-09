import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attention_scores = self.softmax(self.attention_weights(x))
        return x * attention_scores


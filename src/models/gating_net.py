import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_CLASSES, DEVICE

class GatingNetwork(nn.Module):
    def __init__(self, input_shape, num_learners):
        """
        Gating Network to select top-k weak learners.
        
        Args:
            input_shape (tuple): (packet_num, num_features)
            num_learners (int): Number of weak learners T in AdaBoost.
        """
        super(GatingNetwork, self).__init__()
        
        # Flatten the input features
        # input_shape = (20, 256) -> 5120
        flatten_dim = input_shape[0] * input_shape[1]
        
        self.net = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_learners)
        )
        
    def forward(self, x):
        # x: (batch, h, w) -> (batch, h*w)
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() == 4: # (batch, 1, h, w)
            x = x.reshape(x.size(0), -1)
            
        return self.net(x)

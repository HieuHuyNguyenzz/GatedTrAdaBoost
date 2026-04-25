import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_CLASSES, DEVICE

class GatingNetwork(nn.Module):
    def __init__(self, input_shape, num_learners, hidden_dim=256, dropout=0.1):
        """
        MLP-based Gating Network to select top-k weak learners.
        """
        super(GatingNetwork, self).__init__()
        
        flatten_dim = input_shape[0] * input_shape[1]
        
        self.net = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_learners)
        )
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() == 4:
            x = x.reshape(x.size(0), -1)
        
        return self.net(x)

class GatingCNN(nn.Module):
    def __init__(self, input_shape, num_learners, dropout=0.1):
        """
        Mini-CNN Gating Network to select top-k weak learners.
        Uses GAP to reduce parameters and prevent overfitting.
        """
        super(GatingCNN, self).__init__()
        
        # Compact CNN architecture
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.attn_conv = nn.Conv2d(16, 1, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(16, num_learners)
        
    def forward(self, x):
        # Add channel dimension: (batch, h, w) -> (batch, 1, h, w)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))   # pool sau cặp conv1-conv2
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))   # pool sau cặp conv3-conv4
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x))))) 
        
        # Attention Pooling: replace GAP with weighted sum
        attn_map = self.attn_conv(F.relu(x))
        B, _, H, W = attn_map.shape
        attn_map = torch.softmax(attn_map.view(B, -1), dim=-1).view(B, 1, H, W)
        x = (x * attn_map).sum(dim=[2, 3])
        
        x = self.dropout(x)
        x = self.fc(x)
        return x



class NoisyTopKGating(nn.Module):
    """Noisy Top-K Gating (GShard style) - adds noise for exploration"""
    def __init__(self, input_shape, num_learners, hidden_dim=256, noise_std=1.0, dropout=0.1):
        super(NoisyTopKGating, self).__init__()
        
        flatten_dim = input_shape[0] * input_shape[1]
        self.noise_std = noise_std
        
        self.w_gate = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_learners)
        )
        
        self.w_noise = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_learners)
        )
    
    def forward(self, x, training=True):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() == 4:
            x = x.reshape(x.size(0), -1)
        
        logits = self.w_gate(x)
        
        if training:
            noise_scale = F.softplus(self.w_noise(x)) + 1e-2
            noise = torch.randn_like(logits) * noise_scale * self.noise_std
            logits = logits + noise
        
        return logits

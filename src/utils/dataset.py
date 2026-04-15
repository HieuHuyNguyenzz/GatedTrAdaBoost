import torch
from torch.utils.data import Dataset
import numpy as np

class ETCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.float()
        self.y = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y.long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

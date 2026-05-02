import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES, NUM_WORKERS, CLIENT_LR, WEIGHT_DECAY
from src.utils.dataset import ETCDataset
import numpy as np
import time

PIN_MEMORY = DEVICE.type == 'cuda'

def train_cnn_baseline(model_class, target_X, target_y, source_X, source_y):
    """
    Trains a single CNN model as a baseline on combined source and target labeled data.
    """
    # Combine target and source data identically to TrAdaBoost
    X_combined = np.concatenate([target_X, source_X], axis=0)
    y_combined = np.concatenate([target_y, source_y], axis=0)
    
    input_shape = X_combined[0].shape
    model = model_class(input_shape=input_shape, num_classes=NUM_CLASSES).to(DEVICE)
    
    dataset = ETCDataset(X_combined, y_combined)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # Dataset for validation during training
    val_dataset = ETCDataset(target_X, target_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    optimizer = optim.Adam(model.parameters(), lr=CLIENT_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Validation Accuracy on Target Labeled Data
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    preds = torch.argmax(output, dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
            acc = 100 * correct / total if total > 0 else 0
            model.train()
            print(f"  Baseline Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}, Target Acc: {acc:.2f}%")
            
    model.eval()
    return model

def predict_cnn_baseline(model, X_test, return_time=False):
    """
    Helper to get predictions from a baseline CNN model with optional timing.
    """
    X_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    if X_tensor.dim() == 3:
        X_tensor = X_tensor.unsqueeze(1)
    
    # Warm up
    n_warmup = min(10, X_tensor.size(0))
    with torch.no_grad():
        for i in range(n_warmup):
            warmup_input = X_tensor[i:i+1]
            _ = model(warmup_input)
            
    # Synchronize before timing
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    # Start timing
    start_time = time.time()
    
    preds = []
    with torch.no_grad():
        for i in range(0, X_tensor.size(0), BATCH_SIZE):
            batch = X_tensor[i : i + BATCH_SIZE]
            out = model(batch)
            preds.append(torch.argmax(out, dim=1).cpu().numpy())
            
    # Synchronize after timing
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    predictions = np.concatenate(preds)
    
    if return_time:
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_sample_ms = total_time_ms / X_test.shape[0]
        return predictions, avg_time_per_sample_ms
        
    return predictions

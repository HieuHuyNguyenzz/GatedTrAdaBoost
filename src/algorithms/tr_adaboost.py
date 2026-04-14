import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from src.config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES, 
    GATING_K, GATING_TAU, GATING_LR, GATING_EPOCHS,
    LAMBDA_KL, LAMBDA_RANK, LAMBDA_SPARSE, LAMBDA_MARGIN
)
from src.models.gating_net import GatingNetwork

class ETCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.float()
        self.y = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y.long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

class MultiClassTrAdaBoostCNN:
    def __init__(self, model_class, n_estimators=10):
        self.model_class = model_class
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        self.gate = None
        
    def fit(self, target_X, target_y, source_X, source_y):
        n_target = len(target_y)
        n_source = len(source_y)
        total_samples = n_target + n_source
        
        X_combined = np.concatenate([target_X, source_X], axis=0)
        y_combined = np.concatenate([target_y, source_y], axis=0)
        combined_dataset = ETCDataset(X_combined, y_combined)
        
        alpha_s = np.log(1 / (1 + np.sqrt(2 * np.log(n_target) / self.n_estimators)))
        beta = np.ones(total_samples) / total_samples
        
        for t in range(self.n_estimators):
            p_t = beta / beta.sum()
            sampler = WeightedRandomSampler(p_t, num_samples=total_samples, replacement=True)
            dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
            
            learner = self.model_class(input_shape=X_combined[0].shape, num_classes=NUM_CLASSES).to(DEVICE)
            optimizer = optim.Adam(learner.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            learner.train()
            for epoch in range(NUM_EPOCHS):
                for data, target in dataloader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = learner(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            learner.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                eval_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
                for data, target in eval_loader:
                    data = data.to(DEVICE)
                    output = learner(data)
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            preds_source = all_preds[n_target:]
            labels_source = all_labels[n_target:]
            beta_source = beta[n_target:]
            
            indicator_source = (preds_source != labels_source).astype(float)
            eps_t = np.sum(beta_source * indicator_source) / np.sum(beta_source)
            eps_t = np.clip(eps_t, 1e-10, (NUM_CLASSES - 1) / NUM_CLASSES - 1e-10)
            
            alpha_t = np.log((1 - eps_t) / eps_t) + np.log(NUM_CLASSES - 1)
            C_t = NUM_CLASSES * (1 - eps_t)
            
            indicator_all = (all_preds != all_labels).astype(float)
            beta[:n_target] *= np.exp(alpha_t * indicator_all[:n_target])
            beta[n_target:] *= C_t * np.exp(alpha_s * indicator_all[n_target:])
            
            print(f"Iteration {t+1}/{self.n_estimators}, Source Error: {eps_t:.4f}")
            
            self.learners.append(learner)
            self.alphas.append(alpha_t)

    def _get_all_predictions(self, X):
        """Helper to get predictions from all learners for a given X."""
        X_tensor = torch.from_numpy(X).float().to(DEVICE)
        if X_tensor.dim() == 3:
            X_tensor = X_tensor.unsqueeze(1)
            
        all_preds = []
        with torch.no_grad():
            for learner in self.learners:
                learner.eval()
                preds = []
                for i in range(0, X_tensor.size(0), BATCH_SIZE):
                    batch = X_tensor[i : i + BATCH_SIZE]
                    out = learner(batch)
                    preds.append(torch.argmax(out, dim=1).cpu().numpy())
                all_preds.append(np.concatenate(preds))
        
        return np.array(all_preds).T # (samples, T)

    def train_gate(self, X_train, y_train):
        """
        Train the gating network based on learner contributions.
        """
        print("Generating contribution labels for Gating Network...")
        # 1. Compute contributions c_{i,t} = alpha_t * I(h_t(x_i) = y_i)
        preds = self._get_all_predictions(X_train) # (N, T)
        alphas = np.array(self.alphas) # (T,)
        
        # Broadcasted indicator: (N, T)
        # y_train: (N,) -> (N, 1)
        contributions = (preds == y_train[:, np.newaxis]) * alphas
        
        # 2. Soft labels q_{i,t} = softmax(c_{i,t} / tau)
        exp_c = np.exp(contributions / GATING_TAU)
        q = exp_c / np.sum(exp_c, axis=1, keepdims=True)
        q_tensor = torch.from_numpy(q).float().to(DEVICE)
        
        X_tensor = torch.from_numpy(X_train).float().to(DEVICE)
        X_dataset = ETCDataset(X_train, y_train)
        dataloader = DataLoader(X_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        
        # 3. Initialize Gate
        self.gate = GatingNetwork(input_shape=X_train[0].shape, num_learners=self.n_estimators).to(DEVICE)
        optimizer = optim.Adam(self.gate.parameters(), lr=GATING_LR)
        
        print("Training Gating Network...")
        self.gate.train()
        for epoch in range(GATING_EPOCHS):
            total_loss = 0
            for i, (data, _) in enumerate(dataloader):
                data = data.to(DEVICE)
                # Get target q for this batch
                batch_q = q_tensor[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                if batch_q.size(0) == 0: continue
                
                optimizer.zero_grad()
                p_logits = self.gate(data)
                p = torch.softmax(p_logits, dim=1)
                
                # KL Divergence Loss
                # KL(q||p) = sum q * log(q/p)
                kl_loss = torch.sum(batch_q * (torch.log(batch_q + 1e-10) - torch.log(p + 1e-10)), dim=1).mean()
                
                # Sparsity Loss (Entropy)
                sparse_loss = torch.sum(p * torch.log(p + 1e-10), dim=1).mean()
                
                loss = LAMBDA_KL * kl_loss + LAMBDA_SPARSE * sparse_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Gate Epoch {epoch+1}/{GATING_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    def predict(self, X_test):
        """Full AdaBoost prediction."""
        X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
        if X_test_tensor.dim() == 3:
            X_test_tensor = X_test_tensor.unsqueeze(1)
            
        vote_matrix = np.zeros((X_test.size(0), NUM_CLASSES))
        
        with torch.no_grad():
            for alpha_t, learner in zip(self.alphas, self.learners):
                learner.eval()
                outputs = []
                for i in range(0, X_test_tensor.size(0), BATCH_SIZE):
                    batch = X_test_tensor[i : i + BATCH_SIZE]
                    out = learner(batch)
                    outputs.append(torch.argmax(out, dim=1).cpu().numpy())
                preds = np.concatenate(outputs)
                for cls in range(NUM_CLASSES):
                    vote_matrix[preds == cls, cls] += alpha_t
                    
        return np.argmax(vote_matrix, axis=1)

    def predict_sparse(self, X_test):
        """
        Gated Sparse Inference prediction.
        """
        if self.gate is None:
            raise ValueError("Gating network not trained. Call train_gate first.")
            
        X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
        if X_test_tensor.dim() == 3:
            X_test_tensor = X_test_tensor.unsqueeze(1)
            
        self.gate.eval()
        with torch.no_grad():
            # 1. Compute gating scores
            g_scores = []
            for i in range(0, X_test_tensor.size(0), BATCH_SIZE):
                batch = X_test_tensor[i : i + BATCH_SIZE]
                out = self.gate(batch)
                g_scores.append(out.cpu().numpy())
            g_scores = np.concatenate(g_scores) # (N, T)
            
            # 2. Select Top-K learners for each sample
            # top_k_idx: (N, K)
            top_k_idx = np.argsort(g_scores, axis=1)[:, -GATING_K:]
            
            # 3. Evaluate only selected learners
            # To do this efficiently, we still evaluate learners one by one 
            # but only add their contribution if they are in the Top-K for that sample.
            vote_matrix = np.zeros((X_test.size(0), NUM_CLASSES))
            
            for t in range(self.n_estimators):
                # Mask for samples where learner t is in top-k
                mask = np.any(top_k_idx == t, axis=1)
                if not np.any(mask):
                    continue
                
                # Only predict for masked samples
                # For simplicity in batching, we can predict for all but only use masked
                # Or we can filter X_test. Filtering is better for speed.
                X_masked = X_test[mask]
                X_masked_tensor = torch.from_numpy(X_masked).float().to(DEVICE)
                if X_masked_tensor.dim() == 3:
                    X_masked_tensor = X_masked_tensor.unsqueeze(1)
                
                self.learners[t].eval()
                preds_masked = []
                with torch.no_grad():
                    for i in range(0, X_masked_tensor.size(0), BATCH_SIZE):
                        batch = X_masked_tensor[i : i + BATCH_SIZE]
                        out = self.learners[t](batch)
                        preds_masked.append(torch.argmax(out, dim=1).cpu().numpy())
                
                preds_masked = np.concatenate(preds_masked)
                
                # Add to vote matrix
                indices = np.where(mask)[0]
                alpha_t = self.alphas[t]
                for idx, pred in zip(indices, preds_masked):
                    vote_matrix[idx, pred] += alpha_t
                    
        return np.argmax(vote_matrix, axis=1)

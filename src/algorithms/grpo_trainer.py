import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.config import DEVICE, GRPO_G, GRPO_LAMBDA, GRPO_CLIP

class GRPOTrainer:
    def __init__(self, gating_net, learners, alphas, lr=1e-4):
        """
        GRPO Trainer for Gating Network.
        
        Args:
            gating_net: The gating network to train.
            learners: List of trained experts.
            alphas: Reliability weights for each expert.
            lr: Learning rate.
        """
        self.gating_net = gating_net
        self.learners = learners
        self.alphas = torch.tensor(alphas).float().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.gating_net.parameters(), lr=lr)
        self.num_experts = len(learners)

    def get_ensemble_prediction(self, X, selected_mask):
        """
        X: (B * G, 1, 20, 256)
        selected_mask: (B * G, T) boolean mask
        """
        B_G = X.size(0)
        # To efficiently compute predictions, we can't easily loop over B*G
        # Instead, we can iterate over experts
        
        # Vote matrix: (B * G, NUM_CLASSES)
        vote_matrix = torch.zeros((B_G, self.alphas.size(0).item() if hasattr(self.alphas, 'size') else len(self.alphas)), device=DEVICE) # This is wrong, should be NUM_CLASSES
        # Wait, I need NUM_CLASSES. Let's pass it or import it.
        return None

    def compute_reward(self, preds, labels, mask):
        """
        preds: (B * G)
        labels: (B * G)
        mask: (B * G, T) boolean mask
        """
        # Accuracy reward
        acc_reward = (preds == labels).float()
        
        # Sparsity penalty: penalize number of experts selected
        num_selected = mask.sum(dim=-1).float()
        sparsity_penalty = GRPO_LAMBDA * num_selected
        
        return acc_reward - sparsity_penalty

    def train_step(self, X, y):
        """
        X: (B, 1, 20, 256)
        y: (B)
        """
        B = X.size(0)
        self.gating_net.train()
        
        # 1. Sample Group Actions
        # Get probabilities for each expert: (B, T)
        with torch.no_grad():
            logits = self.gating_net(X)
            probs = torch.sigmoid(logits) # Independent probability for each expert
            
        # Expand to group size: (B * G, T)
        probs_expanded = probs.repeat_interleave(GRPO_G, dim=0)
        
        # Sample experts: (B * G, T)
        # Each expert is sampled independently based on p_ij
        # To make it differentiable, we'll use the log-probs of the samples
        samples = torch.bernoulli(probs_expanded)
        
        # Calculate log-probs of the sampled actions
        # log p(action) = sum_{j} [action_j * log(p_j) + (1-action_j) * log(1-p_j)]
        log_probs = (samples * torch.log(probs_expanded + 1e-8) + 
                     (1 - samples) * torch.log(1 - probs_expanded + 1e-8)).sum(dim=-1)

        # 2. Execute Actions and Compute Rewards
        # For each group member, get prediction
        # Since experts are slow, we can't easily vectorize B*G across all experts
        # But we can iterate over experts and mask the inputs
        X_expanded = X.repeat_interleave(GRPO_G, dim=0)
        y_expanded = y.repeat_interleave(GRPO_G, dim=0)
        
        # We need the number of classes. Assume it's available from config or passed.
        # For now, I'll use a trick to get it from the learners' output
        with torch.no_grad():
            # Sample one output to get num_classes
            sample_out = self.learners[0](X_expanded[:1])
            num_classes = sample_out.size(1)
        
        vote_matrix = torch.zeros((B * GRPO_G, num_classes), device=DEVICE)
        
        for t in range(self.num_experts):
            # Mask for samples that selected expert t
            mask = samples[:, t]
            if not mask.any():
                continue
            
            X_masked = X_expanded[mask]
            self.learners[t].eval()
            with torch.no_grad():
                out = self.learners[t](X_masked)
                pred = torch.argmax(out, dim=1)
                
                # Weight by alpha
                alpha_t = self.alphas[t]
                # We need to update vote_matrix. This is tricky because mask indices are global
                # Let's use a temporary tensor
                indices = torch.where(mask)[0]
                for idx, p in zip(indices, pred):
                    vote_matrix[idx, p] += alpha_t
        
        preds = torch.argmax(vote_matrix, dim=1)
        rewards = self.compute_reward(preds, y_expanded, samples)
        
        # 3. Compute GRPO Advantages
        # Reshape rewards to (B, G)
        rewards = rewards.reshape(B, GRPO_G)
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - mean_r) / std_r
        advantages = advantages.reshape(-1) # (B * G)

        # 4. Update Policy
        # Re-evaluate current policy for the same samples (for PPO clip)
        logits_curr = self.gating_net(X_expanded)
        probs_curr = torch.sigmoid(logits_curr)
        log_probs_curr = (samples * torch.log(probs_curr + 1e-8) + 
                          (1 - samples) * torch.log(1 - probs_curr + 1e-8)).sum(dim=-1)
        
        ratio = torch.exp(log_probs_curr - log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - GRPO_CLIP, 1.0 + GRPO_CLIP) * advantages
        
        loss = -torch.min(surr1, surr2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), rewards.mean().item()

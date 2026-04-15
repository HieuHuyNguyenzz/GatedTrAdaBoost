import numpy as np
import os
import argparse
from sklearn.metrics import classification_report
from src.config import SAME_DIST_PATH, DIFF_1_DIST_PATH, TEST_DIST_PATH, DEVICE
from src.utils.data_loader import load_feather_data
from src.models.cnn_model import CNNModel
from src.algorithms.original_tr_adaboost import MultiClassTrAdaBoostCNN
from src.algorithms.gated_tr_adaboost import GatedMultiClassTrAdaBoostCNN

def main():
    parser = argparse.ArgumentParser(description="Multi-Class TrAdaBoost-CNN Training and Evaluation")
    parser.add_argument('--mode', type=str, default='test', 
                        choices=['train_full', 'train_gate', 'test_no_gating', 'test_with_gating', 'test'], 
                        help="Execution mode: 'train_full' to train everything, 'train_gate' to only train Gating Network, "
                             "'test_no_gating' for full ensemble evaluation, 'test_with_gating' for sparse evaluation, "
                             "'test' for both.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Execution Mode: {args.mode}")
    
    print("Loading data...")
    # Target domain (Same distribution)
    target_X, target_y = load_feather_data(SAME_DIST_PATH)
    
    # Source domain (Diff distribution)
    source_X, source_y = load_feather_data(DIFF_1_DIST_PATH)
    
    # Test domain
    test_X, test_y = load_feather_data(TEST_DIST_PATH)
    
    if target_X is None or source_X is None or test_X is None:
        print("Error loading datasets. Please check paths in src/config.py")
        return

    input_shape = target_X[0].shape
    print(f"Target shape: {target_X.shape}, Source shape: {source_X.shape}, Test shape: {test_X.shape}")
    
    # Paths for saving models
    ORIG_MODEL_PATH = "model_orig.pth"
    GATED_MODEL_PATH = "model_gated.pth"
    
    # --- Setup Models ---
    model_orig = MultiClassTrAdaBoostCNN(CNNModel, n_estimators=10)
    model_gated = GatedMultiClassTrAdaBoostCNN(CNNModel, n_estimators=10)
    
    # 1. Handle Original Model
    if args.mode == 'train_full':
        print("\nTraining Original Model...")
        model_orig.fit(target_X, target_y, source_X, source_y)
        model_orig.save(ORIG_MODEL_PATH, input_shape)
    elif os.path.exists(ORIG_MODEL_PATH):
        print("Loading pre-trained Original Model...")
        model_orig.load(ORIG_MODEL_PATH)
    else:
        print("Error: No pre-trained Original Model found. Please run with --mode train_full first.")
        return
    
    # 2. Handle Gated Model
    if args.mode == 'train_full':
        print("\nTraining Gated Model (Base Ensemble)...")
        model_gated.fit(target_X, target_y, source_X, source_y)
        print("\nTraining Gating Network for Sparse Inference...")
        model_gated.train_gate(target_X, target_y)
        model_gated.save(GATED_MODEL_PATH, input_shape)
        
    elif args.mode == 'train_gate':
        if os.path.exists(GATED_MODEL_PATH):
            print("Loading existing Gated Model to re-train gate...")
            model_gated.load(GATED_MODEL_PATH)
        elif os.path.exists(ORIG_MODEL_PATH):
            print("Gated model not found. Initializing Gated model from Original ensemble...")
            model_gated.learners = model_orig.learners
            model_gated.alphas = model_orig.alphas
            model_gated.n_estimators = model_orig.n_estimators
        else:
            print("Error: No pre-trained models found. Please run with --mode train_full first.")
            return
            
        print("\nRe-training Gating Network...")
        model_gated.train_gate(target_X, target_y)
        model_gated.save(GATED_MODEL_PATH, input_shape)
    else:
        if os.path.exists(GATED_MODEL_PATH):
            print("Loading pre-trained Gated Model...")
            model_gated.load(GATED_MODEL_PATH)
        else:
            print("Error: No pre-trained Gated Model found. Please run with --mode train_full or train_gate first.")
            return

    # --- Evaluation Phase ---
    
    # Scenario A: WITHOUT GATING (Full Ensemble)
    if args.mode in ['test_no_gating', 'test', 'train_full', 'train_gate']:
        print("\n" + "="*60)
        print(" EVALUATION WITHOUT GATING (Full Ensemble) ")
        print("="*60)
        
        # Original Model
        print("\n[1] Original Multi-class TrAdaBoost-CNN:")
        orig_predictions = model_orig.predict(test_X)
        print(classification_report(test_y, orig_predictions))
        
        # Gated Model (Full mode)
        print("\n[2] Gated Model (Running in Full Mode):")
        gated_full_predictions = model_gated.predict(test_X)
        print(classification_report(test_y, gated_full_predictions))

    # Scenario B: WITH GATING (Sparse Inference)
    if args.mode in ['test_with_gating', 'test', 'train_full', 'train_gate']:
        print("\n" + "="*60)
        print(" EVALUATION WITH GATING (Sparse Inference) ")
        print("="*60)
        
        k_values = [1, 2, 3, 5, 10]
        k_values = [min(k, model_gated.n_estimators) for k in k_values]
        
        for k in k_values:
            gated_sparse_predictions = model_gated.predict_sparse(test_X, k=k)
            print(f"\n--- Gated Sparse AdaBoost (k={k}) ---")
            print(classification_report(test_y, gated_sparse_predictions))

if __name__ == "__main__":
    main()

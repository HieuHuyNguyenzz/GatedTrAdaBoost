import numpy as np
from sklearn.metrics import classification_report
from src.config import SAME_DIST_PATH, DIFF_1_DIST_PATH, TEST_DIST_PATH, DEVICE
from src.utils.data_loader import load_feather_data
from src.models.cnn_model import CNNModel
from src.algorithms.tr_adaboost import MultiClassTrAdaBoostCNN, GatedMultiClassTrAdaBoostCNN

def main():
    print(f"Using device: {DEVICE}")
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

    print(f"Target shape: {target_X.shape}, Source shape: {source_X.shape}, Test shape: {test_X.shape}")
    
    # --- Original Algorithm: MultiClassTrAdaBoostCNN ---
    print("\n" + "="*50)
    print("Algorithm 1: Original Multi-class TrAdaBoost-CNN")
    print("="*50)
    model_orig = MultiClassTrAdaBoostCNN(CNNModel, n_estimators=10)
    print("Training Original Model...")
    model_orig.fit(target_X, target_y, source_X, source_y)
    
    print("\nEvaluating Original Model...")
    orig_predictions = model_orig.predict(test_X)
    print("Original AdaBoost Classification Report:\n")
    print(classification_report(test_y, orig_predictions))
    
    # --- Improved Algorithm: GatedMultiClassTrAdaBoostCNN ---
    print("\n" + "="*50)
    print("Algorithm 2: Improved Gated Multi-class TrAdaBoost-CNN")
    print("="*50)
    model_gated = GatedMultiClassTrAdaBoostCNN(CNNModel, n_estimators=10)
    print("Training Gated Model...")
    model_gated.fit(target_X, target_y, source_X, source_y)
    
    print("\nTraining Gating Network for Sparse Inference...")
    model_gated.train_gate(target_X, target_y)
    
    print("\nEvaluating Gated Model (Full Ensemble)...")
    gated_full_predictions = model_gated.predict(test_X)
    print("Gated Full AdaBoost Classification Report:\n")
    print(classification_report(test_y, gated_full_predictions))
    
    print("\nEvaluating Gated Model (Sparse Ensemble)...")
    gated_sparse_predictions = model_gated.predict_sparse(test_X)
    print("Gated Sparse AdaBoost Classification Report:\n")
    print(classification_report(test_y, gated_sparse_predictions))

if __name__ == "__main__":
    main()

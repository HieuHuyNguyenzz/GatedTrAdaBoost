import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import NUM_FEATURE, PACKET_NUM, SEED

def data_processing(df):
    """
    Processes the input dataframe for traffic classification.
    
    Args:
        df (pd.DataFrame): Input dataframe containing flow data and Label.
        
    Returns:
        tuple: (X, y) where X is the reshaped feature matrix and y is the labels.
    """
    # Extract labels
    y = df['Label'].to_numpy()
    
    # Remove non-feature columns
    # Assuming 'flow_id' and 'Label' are the columns to drop
    X_df = df.drop(['Label', 'flow_id'], axis=1)
    
    # Normalize features to [0, 1]
    X = X_df.to_numpy() / 255.0
    
    # Reshape to (samples, packet_num, num_features)
    X = X.reshape(-1, PACKET_NUM, NUM_FEATURE)
    
    # Handle the labels for flows (take the label of the last packet in the flow)
    # The original code did: y_train = y_train.reshape(-1,20)[:,-1]
    y = y.reshape(-1, PACKET_NUM)[:, -1]
    
    return X.astype(np.float32), y.astype(np.int64)

def load_source_data(path):
    """Loads source domain data (full dataset for training)."""
    try:
        df = pd.read_feather(path)
        return data_processing(df)
    except Exception as e:
        print(f"Error loading source data from {path}: {e}")
        return None, None

def load_target_train_data(path, seed=None):
    """
    Loads target domain training data. Now assumes all training data is labeled.
    """
    try:
        df = pd.read_feather(path)
        X, y = data_processing(df)
        
        print(f"Target Training domain: {len(X)} samples (all labeled)")
        
        return X, y, None, None
    except Exception as e:
        print(f"Error loading target train data from {path}: {e}")
        return None, None, None, None

def load_target_test_data(path):
    """
    Loads target domain test data.
    """
    try:
        df = pd.read_feather(path)
        return data_processing(df)
    except Exception as e:
        print(f"Error loading target test data from {path}: {e}")
        return None, None

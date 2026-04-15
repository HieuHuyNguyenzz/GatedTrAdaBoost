# Multi-Class TrAdaBoost-CNN for Traffic Classification

This repository implements a Transfer Learning-based traffic classification algorithm designed for multi-domain SDN networks. It extends the Multi-class TrAdaBoost approach by integrating a Convolutional Neural Network (CNN) as the weak learner to handle cross-domain classification in encrypted network services.

## Key Features

- **MultiClassTrAdaBoostCNN**: The original implementation of Multi-class TrAdaBoost using CNNs to transfer knowledge from a source domain to a target domain.
- **GatedMultiClassTrAdaBoostCNN**: An improved version featuring a **Gating Network**. This allows for **Sparse Inference**, where only the most relevant weak learners are activated for a given input, reducing computational overhead while maintaining accuracy.

## Project Structure

```text
.
├── main.py                 # Entry point for training and evaluation
├── src/
│   ├── config.py           # Path configurations and hyperparameters
│   ├── algorithms/
│   │   └── tr_adaboost.py  # TrAdaBoost and Gated TrAdaBoost implementations
│   ├── models/
│   │   ├── cnn_model.py    # CNN architecture for weak learners
│   │   └── gating_net.py   # MLP architecture for the Gating Network
│   └── utils/
│       └── data_loader.py  # Utilities for loading feather datasets
└── README.md
```

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- NumPy
- Scikit-learn
- Pandas
- PyArrow (for feather files)
- tqdm

### Running the Project

To run the training and evaluation pipeline:

```bash
python main.py
```

The script will:
1. Load datasets (Source, Target, and Test).
2. Train and evaluate the original `MultiClassTrAdaBoostCNN`.
3. Train and evaluate the `GatedMultiClassTrAdaBoostCNN` (both full and sparse inference).
4. Output classification reports for each model.

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

The project is executed via `main.py` with different modes depending on whether you want to train or evaluate the models.

#### Command Syntax
```bash
python main.py --mode <mode_name>
```

#### Available Modes
| Mode | Description |
| :--- | :--- |
| `train_full` | Trains both the Original and Gated ensembles from scratch and saves them to disk. |
| `train_gate` | Loads existing weak learners and only trains/optimizes the Gating Network. |
| `test_no_gating` | Evaluates the performance of the Full Ensemble (baseline). |
| `test_with_gating` | Evaluates the Sparse Inference performance with different top-k learners. |
| `test` (Default) | Performs both full and sparse evaluations for a comprehensive comparison. |

#### Examples
```bash
# To train everything from scratch
python main.py --mode train_full

# To only update the Gating Network
python main.py --mode train_gate

# To run comprehensive tests
python main.py --mode test
```

The script will automatically handle data loading, model persistence (saving/loading `.pth` files), and generate classification reports for the test dataset.


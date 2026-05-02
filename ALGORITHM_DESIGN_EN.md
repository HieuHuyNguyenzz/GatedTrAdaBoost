# Detailed Algorithm Design Document: Gated TrAdaBoost

This document provides a comprehensive and detailed look at the implementation logic within the codebase, including variables, data structures, and computational workflows.

---

## 1. Configuration and Hyperparameter Management (`src/config.py`)

The system uses a centralized config file to manage all constants.

### 1.1 Data & Model Parameters
| Variable | Value | Meaning |
| :--- | :--- | :--- |
| `NUM_FEATURE` | 256 | Number of features per packet. |
| `PACKET_NUM` | 20 | Number of packets per flow. |
| `NUM_CLASSES` | 3 | Number of traffic classification classes. |
| `NUM_ESTIMATORS`| 20 | Number of weak learners (experts) in the ensemble. |
| `BATCH_SIZE` | 16 | Batch size for training. |
| `NUM_EPOCHS` | 10 | Number of training epochs for each weak learner. |
| `CLIENT_LR` | 1e-4 | Learning rate for weak learners. |
| `WEIGHT_DECAY` | 1e-2 | Regularization coefficient for weak learners. |

### 1.2 Gating Network Parameters
| Variable | Value | Meaning |
| :--- | :--- | :--- |
| `GATING_K` | 14 | Number of experts selected during Sparse Inference. |
| `GATING_TAU` | 1.0 | Softmax temperature. |
| `GATING_LR` | 1e-4 | Learning rate of the Gating Network. |
| `GATING_EPOCHS` | 30 | Number of training epochs for the Gating Network. |
| `GATING_WEIGHT_DECAY` | 1e-2 | Regularization coefficient to prevent overfitting. |
| `GATING_GRAD_CLIP` | 1.0 | Gradient clipping threshold for training stability. |
| `GATING_LAMBDA_LB` | 0.01 | Weight of the Load Balancing Loss. |

### 1.3 Target Domain Data Management
Target domain data is currently managed via separate files:
- **Training Set**: Dataset used for training (Labeled).
- **Test Set**: Independent dataset used for final evaluation.

---

## 2. Data Processing Flow (`src/utils/data_loader.py`)

### 2.1 Preprocessing (`data_processing`)
1. **Normalization**: All feature values are divided by 255.0 to scale them to the $[0, 1]$ range.
2. **Reshaping**: Data is converted from a flat format to a 3D tensor: `(num_samples, 20, 256)`.
3. **Labeling**: The label of the last packet in each flow is taken as the label for the entire flow.

### 2.2 Data Loading Strategy
Instead of splitting from a single file, the system loads data directly from feather files:
1. **`load_target_train_data(path)`**: Loads the target domain training set.
2. **`load_target_test_data(path)`**: Loads the target domain test set.

---

## 3. Model Architecture Details

### 3.1 Weak Learner: CNN Model (`src/models/cnn_model.py`)
Each expert is a CNN with the following structure:
- **Input**: `(Batch, 1, 20, 256)` (Channel dimension = 1 added).
- **Layers**: 
    - 6 Convolutional layers (`Conv2d`) combined with ReLU.
    - 3 Max Pooling layers (`MaxPool2d`) interleaved to reduce spatial dimensions.
- **Flatten**: Converts the final feature map into a flat vector.
- **FC Layers**: 
    - `Linear(flatten_dim, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.1)`.
    - `Linear(256, NUM_CLASSES)` $\rightarrow$ Output logits.

### 3.2 Gating Network: Mini-CNN (`src/models/gating_net.py`)
To optimize expert selection, the Gating Network uses a streamlined CNN architecture instead of a flat MLP:
- **Input**: `(Batch, 1, 20, 256)`.
- **Layers**: 
    - 2 Convolutional layers (`Conv2d`) with 16 filters $\rightarrow$ ReLU.
    - 1 Max Pooling layer (`MaxPool2d`).
- **Flattening**: The Gating Network uses a Flatten layer to convert the final feature map into a flat vector before passing it to the Fully Connected layer. This ensures all spatial information from the Convolutional layers is preserved for the most accurate routing decisions.
- **Final Layer**: `Linear(flatten_dim, num_learners)` $\rightarrow$ Output scores for each expert.

---

## 4. Training Pipeline Details

The training process is divided into stages: establishing a baseline, building the expert ensemble, and training the coordinator (Gating Network).

### Stage 0: Baseline CNN Model Training
*Goal: Create a standard model to serve as a performance benchmark.*
- **Training**: A CNN model is trained directly on the combined source and target dataset.
- **Evaluation**: The results of this model serve as the baseline for accuracy and inference time.

### Stage 1: Base Ensemble Training
*Goal: Create $T$ weak learners capable of classifying traffic well on the target domain.*

**Workflow:**
1. **Initialization**: Assign weight $\beta = 1/N$ to all data samples.
2. **Sequential Training Loop (t = 1 $\rightarrow$ T)**:
   - **Sampling**: Use `WeightedRandomSampler` to sample data based on $\beta$. Samples with higher $\beta$ appear more frequently.
   - **Training**: Train a CNN model on this sampled set with `weight_decay` to prevent overfitting $\rightarrow$ Result: Expert $t$.
   - **Error Evaluation ($\epsilon_t$)**: Run Expert $t$ on the Target domain dataset to calculate the error rate.
   - **Confidence Calculation ($\alpha_t$)**: $\alpha_t = \ln((1-\epsilon_t)/\epsilon_t) + \ln(K-1)$. The fewer errors an expert makes, the higher its $\alpha_t$.
   - **Updating $\beta$ Weights**:
     - **Target sample is incorrect**: $\uparrow$ Increase $\beta$ $\rightarrow$ Expert $t+1$ is forced to learn this sample.
     - **Source sample is incorrect**: $\downarrow$ Decrease $\beta$ $\rightarrow$ Remove knowledge from the source domain that interferes with the target domain.
3. **Result**: A set of $\{ (\text{CNN}_1, \alpha_1), (\text{CNN}_2, \alpha_2), \dots, (\text{CNN}_T, \alpha_T) \}$.

---

### Stage 2: Gating Network Training
*Goal: Learn to select the Top-k best experts for each specific traffic sample.*

**Workflow:**
#### Step 2.1: Supervised Fine-tuning
*Use Oracle labels to refine coordination capabilities.*
- **Oracle Label Generation**: 
  - **Probability-based (Soft Labels)**: Instead of using binary correct/incorrect results, the system calculates each expert's contribution based on the softmax probability of the true class.
  - **Formula**: $\text{Contribution}_{i,t} = P(\text{class}_{true} | \text{Expert}_t, \mathbf{x}_i)$. This allows the Gating Network to learn the confidence level of each expert for each specific data sample.
- **Training**:
  - **Task Loss**: BCE loss between the Gating Net output and the Oracle labels.
  - **Balance Loss**: MSE loss to ensure an even distribution among experts, preventing Expert Collapse.

---

### Stage 3: Inference Pipeline
When a new traffic sample $\mathbf{x}$ arrives:

1. **Gating**: $\mathbf{x} \rightarrow \text{Gating Network (Mini-CNN)} \rightarrow \text{Scores for T experts}$.
2. **Selection**: Select the Top-k experts with the highest scores $\rightarrow$ $\{\text{Exp}_{i_1}, \text{Exp}_{i_2}, \dots, \text{Exp}_{i_k}\}$.
3. **Execution**: Run predictions only for these $k$ experts $\rightarrow$ $\{\hat{y}_{i_1}, \dots, \hat{y}_{i_k}\}$.
4. **Aggregation**: Combine results using weighted voting with $\alpha$:
   $$\text{Result} = \text{argmax} \sum_{j=1}^{k} \alpha_{i_j} \cdot \mathbb{I}(\hat{y}_{i_j} = \text{class})$$

---

## 5. Execution Process in `main.py`

1. **Load Data**: Call `load_source_data`, `load_target_train_data`, and `load_target_test_data`.
2. **Phase 0 (Baseline)**: Train and evaluate the Baseline CNN Model $\rightarrow$ Save `model_baseline.pth`.
3. **Phase 1 (Original)**: Run `model_orig.fit` $\rightarrow$ Save `model_orig.pth`.
4. **Phase 2 (Gating)**:
    - Initialize `model_gated` with the `GatingCNN` architecture.
    - Train the Gating Network using probability-based Oracle labels.
    - Save `model_gated.pth`.
5. **Phase 3 (Evaluation)**: 
    - Compare Accuracy and Inference Time between: Baseline CNN vs Full Ensemble vs Gated Sparse (with `GATING_K`).

---

## 6. Hardware Optimization
- **CUDA**: Use `DEVICE = "cuda"`, `pin_memory=True` in DataLoader.
- **MPS**: Use `DEVICE = "mps"`, call `torch.mps.synchronize()` for accurate time measurement.

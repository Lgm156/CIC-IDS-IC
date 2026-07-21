# MLP Model Parameters - Forest Cover Dataset

This document details the architecture, training hyperparameters, and split configurations for the Multilayer Perceptron (MLP) Neural Network trained on the Forest Cover dataset.

---

## 1. Model Architecture

The MLP classifier is defined in `models/mlp.py` and is instantiated with the following layers:

| Layer | Type | Input Dimension | Output Dimension | Activation / Operations |
| :--- | :--- | :---: | :---: | :--- |
| **Input** | Feature Vector | - | 54 | Standard Scaled numeric features |
| **Hidden 1** | Fully Connected | 54 | 256 | BatchNorm1d, ReLU, Dropout (p=0.3) |
| **Hidden 2** | Fully Connected | 256 | 128 | BatchNorm1d, ReLU, Dropout (p=0.3) |
| **Hidden 3** | Fully Connected | 128 | 64 | BatchNorm1d, ReLU, Dropout (p=0.3) |
| **Output** | Fully Connected | 64 | 7 | Linear Logits (Unnormalized) |

---

## 2. Training Hyperparameters

The network is trained using PyTorch in `forestScripts/train_mlp.py` with the following parameters:

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Optimizer** | `AdamW` | Adam optimizer with decoupled weight decay for better regularization. |
| **Learning Rate** | `1e-3` (`0.001`) | Initial learning rate for gradient updates. |
| **Loss Function** | `CrossEntropyLoss` | Standard softmax cross-entropy loss without class weighting. |
| **Batch Size** | `1024` | Number of samples processed in parallel per gradient step. |
| **Max Epochs** | `20` | Maximum number of training cycles through the dataset. |
| **Early Stopping Patience** | `5` | Stop training if validation loss does not improve for 5 consecutive epochs. |
| **Device** | `CUDA / CPU` | Automatically executes on GPU if CUDA is available, otherwise falls back to CPU. |

---

## 3. Data Split Sizes (Current Run)

| Split | Size (Total Samples) | Samples per Class | Imbalance Ratio (IR) |
| :--- | :---: | :---: | :---: |
| **Train (Baseline)** | 293,000 | Variable (Long-tailed) | **150.00** |
| **Validation** | 3,500 | Exactly 500 (Balanced) | **1.00** |
| **Test** | 3,500 | Exactly 500 (Balanced) | **1.00** |

---

## 4. Key Output Locations

* **Model Checkpoint**: `models/forest/mlp/best_model.pt` (Saves model state dictionary at the epoch with the lowest validation loss).
* **Validation Posteriors**: `models/forest/mlp/validation_posteriors.npy` (Softmax probability outputs on the validation set).
* **Test Posteriors**: `models/forest/mlp/test_posteriors.npy` (Softmax probability outputs on the test set).
* **True Targets**: `models/forest/mlp/validation_targets.npy` and `test_targets.npy` (Corresponding target labels).
* **Training Source Prior**: `results/forest/source_prior.npy` (Class probability distribution of the training set used for Imbalance Calibration).
* **Training Log**: `results/forest/training_history_mlp.csv` (CSV file recording loss and accuracy per epoch).

# Model Overview – Detailed Documentation

## Table of Model Parameters (defaults used in this project)

| Model | Parameter | Default Value | Description |
|-------|-----------|---------------|-------------|
| **Logistic Regression** | `C` | `1.0` | Inverse of regularization strength; larger values mean weaker regularization. |
| | `max_iter` | `1000` | Maximum number of solver iterations. |
| | `solver` | `"lbfgs"` | Optimization algorithm; `lbfgs` works well for multiclass problems. |
| | `random_state` | `42` | Seed for reproducibility. |
| **FT‑Transformer** | `d_token` | `64` | Dimension of the token embedding for each numeric feature. |
| | `n_blocks` | `2` | Number of transformer encoder layers. |
| | `n_heads` | `4` | Number of attention heads per encoder layer. |
| | `d_ffn` | `128` | Hidden size of the feed‑forward network inside each encoder block. |
| | `dropout` | `0.1` | Dropout probability applied after attention and feed‑forward layers. |
| | `lr` | `0.001` | Learning rate for the AdamW optimizer. |
| | `batch_size` | `1024` | Number of samples processed per gradient update. |
| | `epochs` | `20` | Upper limit on training epochs. |
| | `patience` | `5` | Early‑stopping patience; training stops if validation loss does not improve for this many epochs. |
| | `device` | `"cuda" if available else "cpu"` | Device on which the model runs. |
| **LSTM Classifier** | `hidden_dim` | `128` | Size of the hidden state for the LSTM. |
| | `num_layers` | `1` | Number of stacked LSTM layers. |
| | `dropout` | `0.3` (ignored if `num_layers` == 1) | Dropout applied after the LSTM output. |
| | `lr` | `0.001` | Learning rate for the AdamW optimizer. |
| | `batch_size` | `1024` | Mini‑batch size for training. |
| | `epochs` | `20` | Maximum training epochs. |
| | `patience` | `5` | Early‑stopping patience on validation loss. |
| | `device` | `"cuda" if available else "cpu"` | Execution device. |
| **MLP Classifier** | `hidden_layers` | `[256, 128, 64]` | Dimensionality of sequential linear layers. |
| | `dropout` | `0.3` | Dropout probability applied after each hidden layer. |
| | `lr` | `0.001` | Learning rate for the AdamW optimizer. |
| | `batch_size` | `1024` | Number of samples processed per gradient update. |
| | `epochs` | `20` | Upper limit on training epochs. |
| | `patience` | `5` | Early‑stopping patience; training stops if validation loss does not improve for this many epochs. |
| | `device` | `"cuda" if available else "cpu"` | Execution device. |
| **Random Forest** | `n_estimators` | `100` | Number of trees in the forest. |
| | `max_depth` | `None` | Maximum depth of each tree; `None` means nodes are expanded until pure. |
| | `random_state` | `42` | Random seed for reproducibility. |
| **XGBoost** | `n_estimators` | `100` | Number of boosting rounds. |
| | `max_depth` | `6` | Maximum depth of each tree. |
| | `learning_rate` | `0.1` | Step size shrinkage used to prevent over‑fitting. |
| | `random_state` | `42` | Seed for reproducibility. |
| | `eval_metric` | `"mlogloss"` | Metric used for evaluation during training. |

---

## 1. Logistic Regression

Logistic Regression is a linear model that estimates the probability that a given input belongs to a particular class. It works by applying a linear transformation to the input features and then passing the result through a sigmoid (for binary) or softmax (for multiclass) function.

### Key Parameters Explained
- **C**: This controls the strength of L2 regularization. A value of `1.0` means moderate regularization; setting it higher (e.g., `10`) reduces regularization, which may improve fit on training data but risks over‑fitting.
- **max_iter**: The solver iterates up to `1000` times to converge. In practice, the model usually converges much earlier, but the limit protects against infinite loops.
- **solver**: `lbfgs` is a quasi‑Newton method that efficiently handles multiclass problems. Other options exist (`newton-cg`, `sag`, `saga`), but `lbfgs` provides a good balance of speed and stability.
- **random_state**: Using the seed `42` guarantees that the same data shuffling and weight initialisation occur every run, making results reproducible.

### Example Usage
```python
from models.logistic_regression import train, predict, save_model

# Assume X_train, y_train are already prepared numpy arrays
model = train(X_train, y_train, C=0.5, max_iter=500)
# Predict on new data
predictions = predict(model, X_test)
# Persist the model for later use
save_model(model, "models/logistic/model.joblib")
```
In this example we changed `C` to `0.5` to increase regularization, which may be useful if the dataset is noisy.

---

## 2. FT‑Transformer (Feed‑Forward Transformer for Tabular Data)

The FT‑Transformer treats each numeric column as a separate token, embeds it into a vector of size `d_token`, and then processes the sequence of tokens with a stack of transformer encoder layers. A special **CLS** token is prepended; its final representation is used for classification.

### How the Architecture Works
1. **NumericalFeatureTokenizer** creates a token for each feature using learned weight and bias matrices (`weights` and `biases`). The operation is essentially `x * W + b` where `x` is the raw feature value.
2. The tokens are concatenated with a learnable `[CLS]` token, producing a tensor of shape `(batch, num_features + 1, d_token)`.
3. The tensor passes through `n_blocks` transformer encoder layers. Each layer performs multi‑head self‑attention (`n_heads`) followed by a feed‑forward network of size `d_ffn`.
4. After the transformer, the output corresponding to the `[CLS]` token is extracted, layer‑normalised, and fed into a linear head that outputs logits for each class.

### Important Hyper‑parameters
- **d_token (64)**: Larger token dimensions can capture richer interactions but increase memory and computation.
- **n_blocks (2)**: More blocks increase depth of representation; diminishing returns after a few layers for tabular data.
- **n_heads (4)**: Determines how many attention sub‑spaces the model learns. Must divide `d_token` evenly.
- **d_ffn (128)**: Hidden size of the feed‑forward network inside each encoder; typical values are 2–4×`d_token`.
- **dropout (0.1)**: Helps regularise the model, especially when the dataset is small.
- **lr (0.001)**: Learning rate for the AdamW optimizer; a common starting point.
- **batch_size (1024)**: Large batch sizes work well on tabular data because the model sees many examples per update.
- **epochs (20) & patience (5)**: Early stopping halts training if validation loss does not improve for five consecutive epochs, preventing over‑fitting.
- **device**: Automatically selects a GPU (`cuda`) if available; otherwise falls back to CPU.

### Example Training Call
```python
from models.ft_transformer import train, predict_proba, save_model

# X_train/X_val are numpy arrays of shape (samples, num_features)
model = train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    d_token=64,
    n_blocks=2,
    n_heads=4,
    d_ffn=128,
    dropout=0.1,
    lr=0.001,
    batch_size=1024,
    epochs=20,
    patience=5,
    device='cuda'  # or omit to auto‑detect
)
# Obtain class probabilities on test data
test_probs = predict_proba(model, X_test)
# Save the trained checkpoint
save_model(model, "best_model.pt")
```
If you wanted to experiment with deeper attention, you could set `n_blocks=4` and maybe increase `lr` to `0.0005` to keep training stable.

---

## 3. LSTM Classifier

The LSTM model processes the feature vector as a sequence, allowing it to capture temporal‑like relationships even when features are ordered arbitrarily. A single LSTM layer (or multiple stacked layers) encodes the sequence, and the hidden state from the final timestep is fed to a linear classifier.

### Architecture Details
- **Input handling**: If the input tensor has shape `(batch, num_features)`, it is unsqueezed to `(batch, 1, num_features)` so that the LSTM sees a sequence length of one.
- **LSTM layer**: Configured with `input_dim` (number of features) and `hidden_dim` (size of hidden state). The `num_layers` parameter controls stacking.
- **Dropout**: Applied after the LSTM output; ignored when `num_layers == 1` because PyTorch's LSTM only supports dropout between layers.
- **Fully‑connected head**: Maps the final hidden representation to logits for each class.

### Key Hyper‑parameters
- **hidden_dim (128)**: Larger hidden dimensions can model more complex patterns but increase the number of parameters.
- **num_layers (1)**: Adding layers (`2` or `3`) can improve capacity but may require more regularisation.
- **dropout (0.3)**: Helps prevent over‑fitting when multiple layers are used.
- **lr (0.001)**, **batch_size (1024)**, **epochs (20)**, **patience (5)**, **device**: Same semantics as in the FT‑Transformer.

### Example Training Scenario
```python
from models.lstm import train, predict_proba, save_model

# Prepare data matrices
X_train, y_train = ...  # shape (N, num_features)
X_val, y_val = ...

model = train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    hidden_dim=128,
    num_layers=1,
    dropout=0.3,
    lr=0.001,
    batch_size=1024,
    epochs=20,
    patience=5,
    device='cuda'
)
# Predict class probabilities for test set
test_probs = predict_proba(model, X_test)
# Persist the model
save_model(model, "models/lstm/model.pt")
```
If you observe the validation loss plateauing early, you could increase `patience` to `10` or reduce `lr` to `5e‑4`.

---

## 4. MLP Classifier (Multi-Layer Perceptron)

The Multi-Layer Perceptron (MLP) is a feedforward artificial neural network model. It consists of multiple fully-connected linear layers where each hidden layer is followed by a one-dimensional batch normalization layer, a non-linear rectified linear unit (ReLU) activation function, and a dropout layer. This structure allows the model to learn complex non-linear feature combinations from tabular data.

### How the Architecture Works
1. **Hidden Layer 1**: Takes the normalized feature input of size `input_dim` and projects it to a 256-dimensional space. It is then normalized with `BatchNorm1d(256)`, activated via `ReLU`, and regularized with a dropout rate of `0.3`.
2. **Hidden Layer 2**: Projects the 256-dimensional features down to 128 dimensions, normalized with `BatchNorm1d(128)`, activated via `ReLU`, and regularized with a dropout rate of `0.3`.
3. **Hidden Layer 3**: Projects the 128-dimensional features down to 64 dimensions, normalized with `BatchNorm1d(64)`, activated via `ReLU`, and regularized with a dropout rate of `0.3`.
4. **Output Layer**: A final linear layer projecting the 64-dimensional output of the third hidden layer to `num_classes` logits.

### Key Parameters Explained
- **dropout (0.3)**: Applied after each hidden activation layer. It randomly deactivates 30% of neurons during each training step, forcing the network to learn redundant representations and mitigating over-fitting.
- **lr (0.001)**: The learning rate for the AdamW optimizer. It determines the size of parameter steps during gradient updates.
- **batch_size (1024)**: The size of training and validation mini-batches.
- **epochs (20)**: The maximum number of training loops.
- **patience (5)**: The number of epochs the training loop will wait for the validation loss to improve. If validation loss does not decrease for five consecutive epochs, early stopping terminates the process.

### Example Training Call
```python
from models.mlp import MLP
import torch

# Instantiate the network
model = MLP(input_dim=len(features), num_classes=num_classes)

# Example forward pass on batch inputs
batch_X = torch.randn(10, len(features))
logits = model(batch_X)
```

---

## 5. Random Forest

Random Forest builds an ensemble of decision trees, each trained on a bootstrap sample of the data and a random subset of features. The final prediction is obtained by majority voting (classification) or averaging (regression).

### Parameter Insights
- **n_estimators (100)**: Number of trees. More trees usually improve performance but increase training time and memory.
- **max_depth (None)**: When left as `None`, each tree expands until all leaves are pure or contain fewer than `min_samples_split`. Limiting depth (e.g., `max_depth=10`) can reduce over‑fitting.
- **random_state (42)**: Guarantees the same bootstrap samples and feature subsets across runs, enabling reproducibility.

### Example Usage
```python
from models.random_forest import train, predict, save_model

model = train(X_train, y_train, n_estimators=200, max_depth=12)
# Predict class labels for new data
labels = predict(model, X_test)
# Save the trained forest
save_model(model, "models/random_forest/model.joblib")
```
Increasing `n_estimators` to `200` and setting `max_depth=12` often yields a smoother bias‑variance trade‑off on medium‑sized tabular datasets.

---

## 6. XGBoost

XGBoost is a gradient‑boosted decision tree implementation that optimises a differentiable loss function using second‑order gradient information. It is widely used for tabular data because of its speed and accuracy.

### Core Hyper‑parameters
- **n_estimators (100)**: Number of boosting rounds (trees). More rounds increase model capacity but risk over‑fitting; early stopping can be employed.
- **max_depth (6)**: Maximum depth of each tree. Deeper trees capture more complex interactions but also increase over‑fit risk.
- **learning_rate (0.1)**: Shrinks the contribution of each tree; smaller values (`0.01`) require more rounds but often improve generalisation.
- **random_state (42)**: Makes the random subsampling of rows/columns deterministic.
- **eval_metric ("mlogloss")**: Metric used for internal evaluation; `mlogloss` is multinomial log‑loss suitable for multi‑class classification.

### Example Training Call
```python
from models.xgboost_model import train, predict, save_model

model = train(
    X_train, y_train,
    n_estimators=150,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
# Predict class labels
preds = predict(model, X_test)
# Persist the model for later inference
save_model(model, "models/xgboost/model.joblib")
```
In this snippet we increased `n_estimators` and `max_depth` while decreasing `learning_rate` to obtain a more finely‑tuned model.

---

## Shared Pre‑processing Utilities

All models rely on the same preprocessing pipeline stored under the `models/` directory:
- **Feature list** (`features.joblib`): Ordered list of column names used during training. Consistency between train and test is critical.
- **Standard scaler** (`scaler.joblib`): Computes mean and variance on the training set and applies Z‑score normalisation to each numeric feature.
- **Label encoder** (`label_encoder.joblib`): Maps string class names to integer indices required by scikit‑learn and PyTorch classifiers.

Each training script loads these artifacts before fitting a model and saves posterior probability matrices (`*_posteriors.npy`) and the corresponding true target arrays after evaluation.

---

*The documentation above reflects the exact default values and code paths present in the `models/` package of this repository. Adjust any parameter values in the respective training scripts under `scripts/` to explore alternative configurations.*

---

*End of Document*

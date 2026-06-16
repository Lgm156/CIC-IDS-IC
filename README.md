# Imbalance Calibration (IC) on CICIDS2017

This project reproduces the **Imbalance Calibration (IC)** method from the NeurIPS 2020 paper: *"Posterior Re-calibration for Imbalanced Datasets"*.

## Project Overview

We trained an MLP classifier on an artificially long-tailed version of the CICIDS2017 dataset and applied posterior recalibration to improve performance on a balanced test set (label-prior shift).

### Final 7 Classes Used:
- BENIGN
- DDoS
- PortScan
- DoS
- Bot
- BruteForce
- WebAttack

(Infiltration and Heartbleed were dropped due to insufficient samples).

## Key Results

The Imbalance Calibration significantly improved performance on the balanced test set without any retraining.

| Metric | Baseline (Standard Softmax) | IC (Calibrated) |
| :--- | :--- | :--- |
| **Accuracy** | 87.37% | **97.37%** |
| **Balanced Accuracy** | 87.37% | **97.37%** |
| **Macro F1** | 87.62% | **97.36%** |
| **Recall (Bot)** | 49.20% | **99.20%** |
| **Recall (PortScan)** | 81.60% | **97.60%** |

**Best Lambda found:** `1.1462`

## Setup and Reproduction

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data and Analyze
```bash
python scripts/analyze_dataset.py
python scripts/preprocess.py
```

### 3. Create Splits
```bash
python scripts/create_splits.py
```

### 4. Train MLP Model
```bash
export PYTHONPATH=$PYTHONPATH:.
python scripts/train.py
```

### 5. Extract Posteriors
```bash
python scripts/save_posteriors.py
```

### 6. Lambda Search and Evaluation
```bash
python scripts/search_lambda.py
python scripts/evaluate.py
```

## Methodology

1. **Long-Tail Training:** The training set was sampled with an imbalance ratio of **~229**, with BENIGN as the majority and Bot as the minority.
2. **Rebalanced Posterior ($P_r$):**
   $$P_r(y|x) \propto P_d(y|x) \cdot \frac{P_t(y)}{P_s(y)}$$
3. **Imbalance Calibration ($P_f$):**
   $$P_f(y|x) \propto P_d(y|x)^{1-\lambda} \cdot P_r(y|x)^\lambda$$
4. **Lambda Search:** A modified binary search (ternary-like) was used to find the $\lambda$ that maximizes validation accuracy in $O(\log N)$ time.

## Artifacts
- `results/`: Contains metrics, confusion matrices, and the lambda search curve.
- `models/`: Contains the trained MLP, scaler, and label encoder.

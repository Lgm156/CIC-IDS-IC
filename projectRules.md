# Rules

## 1. Follow implementation_plan.md exactly

The implementation plan defines the authoritative workflow for this project.

Do not deviate from it without explicit justification.

---

## 2. Reproduce the IC Method from the Paper

Implement the Imbalance Calibration (IC) method from:

Posterior Re-calibration for Imbalanced Datasets (NeurIPS 2020)

The objective is to reproduce the method faithfully rather than invent alternative calibration strategies.

---

## 3. Use CICIDS2017

Use the following files:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Load all files and merge into a single dataframe.

---

## 4. Use PyTorch

Framework:

```text
PyTorch
```

Use CUDA whenever available.

Target hardware:

```text
RTX 4060 Laptop GPU
8 GB VRAM
```

---

## 5. Use an MLP Classifier

Do not use:

```text
ResNet
CNN
Transformer
TabNet
XGBoost
```

Use a standard multilayer perceptron.

Reason:

CICIDS2017 is tabular data and the paper is architecture agnostic.

---

## 6. Use the Following Classes

```text
BENIGN
DDoS
PortScan
DoS
Bot
BruteForce
WebAttack
Infiltration
```

Mapping:

```text
DoS Hulk
DoS GoldenEye
DoS Slowloris
DoS Slowhttptest
    → DoS

FTP-Patator
SSH-Patator
    → BruteForce

Web Attack – Brute Force
Web Attack – XSS
Web Attack – Sql Injection
    → WebAttack
```

Before training, print final class counts and verify that every class contains enough samples.

---

## 7. Create an Artificially Imbalanced Training Set

The training split must be intentionally long-tailed.

Example target:

```text
BENIGN        100000
DDoS           50000
PortScan       25000
DoS            12000
Bot             6000
BruteForce      3000
WebAttack       1500
Infiltration     750
```

Only the training set should be imbalanced.

Validation and test sets must remain balanced.

---

## 8. Train a Baseline Classifier

Train using:

```text
Cross Entropy Loss
```

No class weighting.

No focal loss.

No LDAM.

No sampling tricks.

The baseline should represent standard supervised learning.

---

## 9. Save Posterior Probabilities

For every validation and test sample save:

```text
Pd(y|x)
```

Softmax probabilities.

These probabilities are required for IC.

---

## 10. Implement Posterior Recalibration

Implement:

```text
Pr(y|x)
=
Pd(y|x)
× Pt(y)
÷ Ps(y)
```

and

```text
Pf(y|x)
=
Pd(y|x)^(1−λ)
×
Pr(y|x)^λ
```

Normalize after every step.

---

## 11. Use Paper-Faithful Lambda Search

Do not use brute-force grid search.

Do not evaluate:

```text
0.0
0.1
0.2
...
1.0
```

Use the modified binary-search strategy described in the paper.

Target complexity:

```text
O(log N)
```

Search λ on the validation set.

---

## 12. Optimize Validation Accuracy

Lambda selection metric:

```text
Validation Accuracy
```

Choose the λ producing the highest validation accuracy.

---

## 13. Evaluate on a Balanced Test Set

Compare:

```text
Baseline
```

vs

```text
IC
```

using identical test data.

---

## 14. Produce All Metrics

Generate:

```text
Accuracy
Precision
Recall
F1
Balanced Accuracy
Per-Class Recall
Confusion Matrix
```

---

## 15. Ensure Reproducibility

Set:

```python
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
```

Use deterministic train/validation/test splits.

---

## 16. Keep Runtime Under One Hour

The full pipeline should complete on an RTX 4060 Laptop GPU in less than one hour.

Simplicity and reproducibility are more important than absolute accuracy.

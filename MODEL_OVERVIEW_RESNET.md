# ResNet-18 Model Overview – Detailed Documentation

This document describes the design decisions, model parameters, and training choices for the **ResNet-18** model applied to the `SkinCancerDataset` (HAM10000) under the Imbalance Calibration (IC) framework.

---

## 1. Summary of Model Parameters

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **Model Architecture** | `ResNet-18` | A deep convolutional neural network with 18 layers. It contains residual connections (skips) that prevent vanishing gradients and make it fast to train. |
| **Pretrained Weights** | `True` (ImageNet) | The model is initialized with weights pre-trained on the ImageNet dataset (1.2 million images). |
| **Image Resolution** | `224 x 224` | Input images are resized from their original dimensions ($600 \times 450$) to $224 \times 224$ pixels. |
| **Learning Rate** | `1e-4` (`0.0001`) | The step size used by the optimizer during backpropagation. |
| **Optimizer** | `AdamW` | Adam with Decoupled Weight Decay. An adaptive learning rate optimizer that prevents overfitting. |
| **Weight Decay** | `1e-2` (`0.01`) | L2 regularization coefficient that penalizes excessively large weights to improve model generalization. |
| **Batch Size** | `64` | The number of images processed before the model updates its internal weights. |
| **Epochs** | `8` | The number of complete passes through the training dataset. |
| **Loss Function** | `CrossEntropyLoss` | Standard cross-entropy loss without any class-weighting (to serve as the uncalibrated baseline). |

---

## 2. Frequently Asked Questions (Q&A)

### Was the model trained from scratch or fine-tuned?
The model was **fine-tuned**. 

* **What is fine-tuning?** Instead of starting with randomly initialized weights (where the model knows absolutely nothing and has to learn from scratch), we load a model that has already been trained on a massive general-image dataset (ImageNet). This pre-trained model has already learned to detect basic shapes, edges, color gradients, and textures.
* **How we did it**: We kept the pre-trained feature extraction layers (the ResNet-18 backbone) and replaced the very last linear layer (the classification head) because the original model was designed to output 1,000 classes, whereas our skin cancer dataset has only 7 classes (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`). During training, we updated both the new classification head and slightly adjusted the weights of the backbone.
* **Why did we do this?** Training a deep vision model from scratch requires millions of medical images to learn high-level representations, which we do not have. Fine-tuning allows us to transfer the "visual knowledge" the model acquired on ImageNet to our dermatoscopic skin lesion task. This yields much higher accuracy, reduces training time, and prevents overfitting.

---

### Why was the number of epochs limited to 8?
We limited training to **8 epochs** for three primary reasons:

1. **Pre-trained Advantage & Fast Convergence**: Since the model starts with pre-trained weights, it already knows how to interpret images. It only needs to learn to map those visual patterns to skin cancer classes. As a result, the training and validation accuracy plateaus very early.
2. **Preventing Overfitting on Rare Classes**: The skin cancer training dataset is extremely imbalanced (long-tailed). For instance, the minority class `df` has only 55 images in training, while the majority class `nv` has over 6,600. If we train for 20 or 30 epochs, the model will start to memorize the specific 55 images of the minority class, leading to severe overfitting and poor generalization on the balanced test set.
3. **Execution Time Limit**: The project rules mandate that the entire pipeline runs in less than one hour on the RTX 4060 Laptop GPU. Training ResNet-18 on ~9,700 images at $224 \times 224$ resolution takes about 1.5 minutes per epoch. Running 8 epochs takes roughly 12 minutes, which is fast and fits comfortably within the time budget.

---

### Why did we resize images to 224x224 pixels?
The original HAM10000 images are high-resolution ($600 \times 450$ pixels). We resized them to $224 \times 224$ for the following reasons:

* **Hardware Limitations (VRAM)**: High-resolution images require a massive amount of GPU memory. Running batches of $600 \times 450$ images would quickly exceed the 8 GB VRAM limit of the RTX 4060 Laptop GPU, resulting in "Out of Memory" (OOM) crashes.
* **Training Speed**: Resizing to $224 \times 224$ reduces the number of input pixels per image by **81%** (from 270,000 down to 50,176 pixels). This speeds up matrix multiplications dramatically and cuts epoch runtimes from 10+ minutes down to 1.5 minutes.
* **ResNet Standard**: $224 \times 224$ is the default input resolution for ResNet architectures. Pre-trained weights are optimized for this size, making transfer learning more effective. It preserves enough dermatoscopic details (color variations, borders, and lesion shapes) for the model to make accurate classifications.

---

### Why did we use a small learning rate of 1e-4?
In standard training from scratch, learning rates of `1e-3` or `1e-2` are common. However, for fine-tuning, we chose a much smaller learning rate of `1e-4` (`0.0001`):

* **Preserving Features**: The ResNet backbone contains highly sensitive pre-trained filters. If the learning rate is too large, the optimizer will make massive weight updates that "destroy" or erase these pre-trained representations (known as *catastrophic forgetting*).
* **Gentle Adjustments**: A learning rate of `1e-4` ensures that the classification head learns quickly, while the pre-trained feature extractor is only gently adjusted to adapt to the new medical domain.

---

### Why did we use AdamW instead of SGD?
* **AdamW** (Adam with weight decay) computes adaptive learning rates for each parameter. It dynamically scales the step size based on how frequently a feature is updated, which leads to much faster and more stable convergence compared to Stochastic Gradient Descent (SGD).
* **Decoupled Regularization**: In vanilla Adam, L2 regularization (weight decay) is mixed with the gradient updates, which can weaken its regularization effect. AdamW decouples weight decay from the gradient updates, resulting in much better generalization and lower validation error for deep neural networks.

---

### How does the Imbalance Calibration (IC) interact with this model?
* **The Bias Problem**: Because our training set is heavily dominated by the `nv` class (Melanocytic nevi), the trained ResNet model learns to output very high probabilities for `nv` and very low probabilities for rare classes like `df`.
* **Post-Training Fix**: Instead of retraining the model, IC intercepts the raw output probabilities (posteriors $P_d(y|x)$) and multiplies them by the target-to-source prior ratio:
  $$\frac{P_t(y)}{P_s(y)}$$
  Since $P_s(\text{df})$ is tiny ($\approx 1\%$) and $P_t(\text{df})$ is balanced ($\approx 14\%$), the scaling factor for `df` is very large ($\approx 12.7$).
* **Shifting Boundaries**: This scaling mathematically shifts the decision boundaries. If the model outputs a low probability for a rare class (e.g., $10\%$ for `df` and $80\%$ for `nv`), multiplying by the prior ratio boosts the calibrated probability of the rare class, shifting the final prediction to the rare class if the visual evidence is supportive.

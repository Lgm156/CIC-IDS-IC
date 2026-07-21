# Methodology: Imbalance Calibration & Lambda Search

This guide explains in simple terms the problem of class imbalance, how the **Imbalance Calibration (IC)** method works, and how we optimized the search for the calibration parameter $\lambda$ using a **Plateau-Robust Binary Search**.

---

## 1. The Core Problem: Class Imbalance

When training a machine learning model on a dataset like **Forest Cover**, some classes are very common while others are extremely rare. For example:
* **Lodgepole Pine**: 150,000 training samples (Majority)
* **Cottonwood Willow**: Only 1,000 training samples (Minority)

Because the model sees the majority class 150 times more often, it becomes heavily biased. It learns to predict "Lodgepole Pine" by default because doing so yields high overall training accuracy. As a result, the model's accuracy on the rare minority classes (called **Recall**) is extremely low.

To evaluate the model fairly, we test it on a **balanced test set** (e.g., exactly 500 samples of each tree type). A biased model will perform poorly here because it will misclassify many minority trees as majority trees.

---

## 2. The Solution: Posterior Re-calibration

Instead of retraining the model (which is expensive and doesn't always work), we can adjust (re-calibrate) the model's predictions after training. 

When a model evaluates a sample, it outputs a probability (called a **posterior**) for each class. Imbalance Calibration adjusts these probabilities by dividing them by how common the class was during training, and multiplying them by how common we expect them to be in the real world.

### The Scaling Formula:
For each class, we calculate a new calibrated probability:

$$\text{Calibrated Probability} \propto \text{Original Probability} \times \left( \frac{\text{Target Prior}}{\text{Source Prior}} \right)^\lambda$$

Where:
* **Source Prior ($P_s$)**: How common a class was in the training set (e.g., 51% for Lodgepole, 0.3% for Cottonwood).
* **Target Prior ($P_t$)**: How common we want the classes to be in evaluation (usually equal/balanced, i.e., $1/7 \approx 14\%$ for each class).
* **Lambda ($\lambda$)**: A scaling factor that controls the **strength** of the calibration:
  * If $\lambda = 0$: No adjustment is made.
  * If $\lambda = 1$: Standard mathematical correction is applied.
  * If $\lambda$ is between $0$ and $3$: We search for the best value to maximize accuracy on our validation set.

---

## 3. Finding the Best Lambda ($\lambda$)

To find the optimal value of $\lambda$, we test different values on a balanced validation set and pick the one that gives the highest accuracy. We compared two search methods:

### Method A: Grid Search
Grid Search is a brute-force approach. It tests every single value in a range with a fixed step size (e.g., `0.00, 0.01, 0.02, 0.03 ... 3.00`).
* **Pros**: Guaranteed to find the absolute best $\lambda$ because it checks everything.
* **Cons**: Slow. If we need to search a large range with high precision, it requires evaluating many points.

### Method B: Binary Search (Divide and Conquer)
Binary Search is a smart, fast approach. Instead of checking every point, it starts in the middle. It checks the local slope (by looking slightly to the left and right) to see if accuracy is increasing or decreasing, then discards half the search range.
* **Pros**: Extremely fast (takes only $\approx 10$ steps instead of 300 steps).
* **Cons**: Assumes the accuracy curve is a smooth hill with a single peak.

---

## 4. The "Plateau" Problem & Our Robust Solution

In real-world datasets, the validation set has a finite number of samples. This means validation accuracy does not change smoothly; it changes in discrete steps (like a staircase) when individual samples flip predictions. This creates:
1. **Flat Plateaus**: Large regions where changing $\lambda$ slightly doesn't change the validation accuracy at all.
2. **Micro-Valleys**: Small local drops due to random noise.

### Why Standard Binary Search Fails:
When the standard binary search lands on a flat plateau or a micro-valley, it computes a zero or incorrect slope. It gets confused, makes a wrong turn, and terminates at a sub-optimal $\lambda$.

### Our Improvement: Plateau-Robust Binary Search
To make the binary search robust, we added a **Plateau Resolution (Look-Ahead Loop)**:
* When the algorithm checks a point and finds that its neighbors have the exact same accuracy (a flat region), it **does not give up**.
* It starts stepping outward dynamically (checking further to the left and right) until it finds a point where the accuracy actually changes.
* This allows the algorithm to determine the true macro-slope of the hill, bypass flat regions and micro-valleys, and guide the binary search directly to the global peak.

This improvement allows the binary search to run with maximum speed while consistently matching the accuracy of the brute-force grid search.

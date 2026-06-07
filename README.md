# MSCS 634 – Lab 2: KNN and Radius Neighbors Classifiers on the Wine Dataset

## Purpose

This lab explores and compares two distance-based classification algorithms—**K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)**—using the Wine Dataset from scikit-learn. The dataset contains 178 samples across three wine classes, each described by 13 chemical properties (e.g., alcohol content, malic acid, flavanoids).

The goals of the lab are to:
- Understand how the choice of **k** (KNN) and **radius** (RNN) affects classification accuracy.
- Visualize accuracy trends across a range of parameter values.
- Compare the two algorithms and reason about when each is preferable.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Lab2_KNN_RNN_Wine.ipynb` | Jupyter Notebook with all code, outputs, plots, and analysis |
| `README.md` | This file — summary of purpose, findings, and decisions |

---

## Key Insights

### KNN Accuracy Trends
- **k = 1** achieves very high accuracy (memorizes training data) but risks overfitting.
- Accuracy remains strong through **k = 5 and k = 11**, representing a solid bias-variance balance.
- Larger k values (15, 21) show a slight decline as distant, less-relevant neighbors begin to influence predictions.
- **Best overall KNN performance** is typically observed around k = 5–11 on this dataset.

### RNN Accuracy Trends
- Features were **standardized** before applying RNN, so radius values (1.0–3.5) are in scaled Euclidean distance units — not the raw feature scale.
- Very small radii may leave test points without any neighbors (handled via `outlier_label='most_frequent'`).
- Accuracy generally increases as the radius grows to include a meaningful neighborhood, then levels off.
- **Best overall RNN performance** typically occurs at a mid-to-large radius in the scaled space.

### KNN vs. RNN — When to Use Each
- **KNN** is simpler to tune (integer k) and works reliably when data density is uniform across classes. It is the preferred choice for this dataset.
- **RNN** naturally adapts to variable data density — regions with denser samples contribute more neighbors, which can be advantageous in real-world datasets where class density is uneven.
- RNN requires careful **feature scaling and radius calibration**; the right radius is heavily data-dependent.

---

## Challenges and Decisions

1. **Feature Scaling:** KNN and RNN are distance-based methods that are highly sensitive to feature scale. The Wine Dataset features span very different numerical ranges (e.g., alcohol ~11–14 vs. proline ~278–1680), so `StandardScaler` was applied before fitting either classifier.

2. **Radius Selection for RNN:** The lab instructions listed radii of 350–600, which are appropriate for the raw, unscaled feature space. After standardization, however, those radii would encompass every training point (trivially inflating accuracy). Radii were therefore adjusted to the **scaled feature space** (1.0–3.5) to produce meaningful, comparable results. This decision is documented in the notebook.

3. **Outlier Handling in RNN:** When a test point falls outside all training radii, `RadiusNeighborsClassifier` raises an error by default. Setting `outlier_label='most_frequent'` assigns the majority class to such points, allowing the evaluation to proceed without crashes.

---

## How to Run

```bash
# Clone the repository

https://github.com/sagar7332/Lab-2-Classification-Using-KNN-and-RNN-Algorithms

# Install dependencies
pip install scikit-learn numpy matplotlib pandas notebook

# Launch Jupyter
jupyter notebook Lab2_KNN_RNN_Wine.ipynb
```

> Python 3.8+ and scikit-learn 1.0+ are recommended.

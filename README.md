# CS 3511 Final Project: Extending AdaBoost for Noisy Data

## Aryav Taneja, Divy Shah, Neil Samant, Shrey Agarwal, Yufei Li

---

## Overview

This project explores extensions to the AdaBoost algorithm to improve robustness under **label noise**.

AdaBoost is highly effective on clean data, but it can overfit to mislabeled points because it repeatedly increases the weight of misclassified examples.

To address this, we implement three modifications:

- **Weight Clipping**: limits extreme weight growth
- **Persistent Error Dampening**: reduces influence of repeatedly misclassified points
- **Soft Boosting**: reduces the aggressiveness of weight updates

We compare these methods across varying noise levels and datasets, and provide an interactive visualization using Streamlit.

---

## Key Idea

Standard AdaBoost update:

$w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i h_t(x_i))$

Our modifications change this update rule to:

- control weight magnitude
- identify noisy points
- smooth learning dynamics

---

## Datasets Used

We evaluate our methods on both synthetic and real datasets from **scikit-learn**:

### Synthetic

- `make_moons`: A synthetic 2D dataset with two interleaving half-moon shapes. Useful for visualizing decision boundaries and seeing how the model separates non-linear data.
- `make_classification`: A synthetic dataset with multiple features and some informative vs noisy features. Useful to test general performance beyond simple visual cases.

### Real-world

- `load_digits`: A real-world dataset of handwritten digits, where we convert it into a binary task: digit 0 vs all others, to test performance on image-like data.
- `load_wine`: A real-world dataset with chemical properties of wines, converted into a binary classification problem (class 0 vs the rest) to test performance on structured tabular data.

---

## Converting to Binary Classification

AdaBoost is implemented here for **binary classification** with labels in {-1, +1}.

For multi-class datasets (Digits, Wine), we convert them into binary problems using a **one-vs-rest approach**:

- **Digits dataset:**
  $y = +1 \text{ if digit = 0, otherwise } -1$

- **Wine dataset:**
  $y = +1 \text{ if class = 0, otherwise } -1$

This allows us to apply AdaBoost consistently across all datasets.

---

## Noise Injection

We simulate noisy data by **randomly flipping labels** in the training set:

- Noise level = probability of flipping a label
- Only applied to training data (test data remains clean)

This enables controlled evaluation of robustness.

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If there is an issue with the requirements file, install manually:

```bash
pip install numpy scikit-learn matplotlib streamlit
```

---

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Using the App

The Streamlit app allows you to:

- Select dataset and noise level
- Choose boosting method (Vanilla, Clipped, Persistent, Soft)
- Adjust parameters dynamically
- Visualize:
  - decision boundaries (for moons)
  - training error curves
  - weight distributions

---

## Accessing the App Online

Users can access the deployed app at [this link](https://group10-boosting-cs3511-spring2026.streamlit.app/).

If it shows that the app is "asleep", then one should be able to click a button to turn it on. After 1-2 minutes, the application should be up and running at the given link. This happens when no one accesses the app for a few hours.

---

## Results Summary

We observe that:

- **Vanilla AdaBoost** performs best on clean data
- **Clipping** improves performance under moderate noise
- **Persistent dampening** helps under high noise by ignoring mislabeled points
- **Soft boosting** provides stable performance across noise levels

No single method dominates, as each addresses a different failure mode.

For more detailed results, please visit [this spreadsheet](https://docs.google.com/spreadsheets/d/17tgOP04jZCkuctsb3PLwqqWd3ie_dv6c1bgs7JSQbFU/edit?usp=sharing). Each tab represents the dataset, and the parameters are also given. The table contains the test accuracy for each combination of noise and boosting.

---

## References

- Freund, Y., & Schapire, R. (1997). _A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting_
- Pedregosa et al. (2011). _Scikit-learn: Machine Learning in Python_

---

## Notes

- All models are implemented from scratch (no sklearn AdaBoost used)
- Designed for interpretability and experimentation
- Focus is on understanding boosting behavior under noise

---

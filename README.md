# ML Algorithms From Scratch

Implementation of three machine learning algorithms from scratch using NumPy only, as part of Machine Learning Assignment 2.

**Student:** Afrazia Umer  
**Roll No:** BSCS23029  
**Seed:** 29

---

## Algorithms Implemented

### 1. k-Means Clustering
- k-Means++ initialization
- Fully vectorised assignment and centroid update
- Convergence via centroid shift tolerance
- Tested on friendly (ARI=1.0) and adversarial (concentric rings) datasets

### 2. Gaussian Naive Bayes
- MLE class priors and per-class Gaussian parameters
- All computation in log space for numerical stability
- Analysed overconfidence on correlated features
- Demonstrated failure on XOR patterns

### 3. Decision Tree C4.5
- Gain Ratio splitting criterion (not ID3)
- Handles continuous features via midpoint thresholds
- Investigated overfitting, greedy failure on XOR, and noise sensitivity

---

## Project Structure

```
a2/
├── implementations.py       # KMeans, GaussianNB, DecisionTreeC45 (from scratch)
├── dataset_generator.py     # All dataset generation functions (SEED=29)
├── run_experiments.py       # All experiments, plots, and results
├── results.json             # Numerical results
└── plots/                   # Generated figures
```

---

## How to Run

```bash
pip install numpy scikit-learn matplotlib
python run_experiments.py
```

All results are fully reproducible with SEED = 29.

---

## Key Results

| Algorithm | Dataset | Metric | Result |
|-----------|---------|--------|--------|
| k-Means | Friendly (4 clusters) | ARI | 1.000 |
| k-Means | Concentric rings | ARI | ~0.000 |
| Gaussian NB | High-dimensional (d=60) | Accuracy | 0.918 |
| Decision Tree C4.5 | Low-noise | Accuracy | 0.930 |
| Decision Tree C4.5 | High-noise (25% flips) | Accuracy | ~0.65 |

---

## Notes

- No external ML libraries used for core algorithms — NumPy only
- scikit-learn used only for `train_test_split`, `accuracy_score`, `adjusted_rand_score`, `confusion_matrix`

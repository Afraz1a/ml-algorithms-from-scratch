"""
dataset_generator.py
--------------------
Generates all synthetic datasets required by ML Assignment 2.

SEED = last 3 digits of roll number (set your own below).
Every dataset satisfies:
    n >= 1000 samples
    d >= 15 features  (>= 5 informative + >= 5 noisy)
At least one dataset has d >= 50 features.
At least one dataset has n >= 5000 samples.
"""

import numpy as np

# ── Set this to the last 3 digits of YOUR roll number ──────────────────────
SEED = 29
# ───────────────────────────────────────────────────────────────────────────

rng_global = np.random.default_rng(SEED)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _add_noise_cols(X: np.ndarray, n_noise: int, rng) -> np.ndarray:
    """Append pure Gaussian noise columns to X."""
    noise = rng.normal(0, 1, (X.shape[0], n_noise))
    return np.hstack([X, noise])


# ══════════════════════════════════════════════════════════════════════════
# Q1 – k-Means datasets
# ══════════════════════════════════════════════════════════════════════════

def make_kmeans_friendly(n: int = 1200, seed: int = SEED) -> tuple:
    """
    4 well-separated spherical Gaussian clusters.
    Assumptions satisfied: spherical, balanced, separable.
    d = 15 (2 informative cluster dims + 13 noise).
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    per_cluster = n // 4
    chunks_X, chunks_y = [], []
    for k, c in enumerate(centres):
        core = rng.normal(c, 0.8, (per_cluster, 2))
        chunks_X.append(core)
        chunks_y.append(np.full(per_cluster, k))
    X_core = np.vstack(chunks_X)
    y = np.concatenate(chunks_y)
    X = _add_noise_cols(X_core, 13, rng)
    idx = rng.permutation(len(y))
    return X[idx], y[idx].astype(int)


def make_kmeans_adversarial(n: int = 1200, seed: int = SEED) -> tuple:
    """
    Two concentric rings – k-Means cannot separate them because the
    Voronoi (convex) partition assumption is violated.
    d = 15 (2 structural + 13 noise).
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    theta_inner = rng.uniform(0, 2 * np.pi, half)
    r_inner = rng.normal(2.0, 0.15, half)
    theta_outer = rng.uniform(0, 2 * np.pi, half)
    r_outer = rng.normal(5.0, 0.15, half)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner),
                                r_inner * np.sin(theta_inner)])
    X_outer = np.column_stack([r_outer * np.cos(theta_outer),
                                r_outer * np.sin(theta_outer)])
    X_core = np.vstack([X_inner, X_outer])
    y = np.array([0] * half + [1] * half)
    X = _add_noise_cols(X_core, 13, rng)
    idx = rng.permutation(len(y))
    return X[idx], y[idx].astype(int)


# ══════════════════════════════════════════════════════════════════════════
# Q2 – Naive Bayes datasets
# ══════════════════════════════════════════════════════════════════════════

def make_correlated_nb(n: int = 2000, seed: int = SEED) -> tuple:
    """
    Binary classification with two highly correlated features.
    Independence assumption of NB is violated -> overconfidence.
    d = 15 (2 correlated informative + 3 independent informative + 10 noise).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # informative base
    X1 = rng.normal(y * 2.0, 1.0, n)
    X2 = X1 + rng.normal(0, 0.1, n)          # nearly identical to X1
    X3 = rng.normal(y * 1.5, 1.2, n)
    X4 = rng.normal(y * 1.0, 1.5, n)
    X5 = rng.normal(y * 0.8, 1.8, n)
    X_core = np.column_stack([X1, X2, X3, X4, X5])
    X = _add_noise_cols(X_core, 10, rng)
    return X, y.astype(int)


def make_nb_friendly(n: int = 1500, seed: int = SEED) -> tuple:
    """
    Many independent features each with weak signal.
    NB aggregates weak evidence and performs surprisingly well.
    d = 20 (15 weak informative + 5 noise).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # 15 weak independent features (small mean shift)
    X_info = rng.normal((y[:, None] * 0.4), 1.5, (n, 15))
    X = _add_noise_cols(X_info, 5, rng)
    return X, y.astype(int)


def make_nb_failure(n: int = 1500, seed: int = SEED) -> tuple:
    """
    XOR pattern: y = sign(X1) XOR sign(X2).
    NB cannot capture the interaction -> near-chance accuracy.
    d = 15 (2 XOR + 13 noise).
    """
    rng = np.random.default_rng(seed)
    X_core = rng.normal(0, 1, (n, 2))
    y = ((X_core[:, 0] > 0) ^ (X_core[:, 1] > 0)).astype(int)
    X = _add_noise_cols(X_core, 13, rng)
    return X, y


# ══════════════════════════════════════════════════════════════════════════
# Q3 – Decision Tree datasets
# ══════════════════════════════════════════════════════════════════════════

def make_low_noise_classification(n: int = 1500, d: int = 15,
                                   seed: int = SEED) -> tuple:
    """
    Clean linearly separable data.
    d = 15 (5 informative + 10 noise).
    Assumptions satisfied: low noise, well-defined boundaries.
    """
    rng = np.random.default_rng(seed)
    n_info = 5
    X_info = rng.normal(0, 1, (n, n_info))
    w = rng.normal(0, 1, n_info)
    score = X_info @ w
    y = (score > 0).astype(int)
    X = _add_noise_cols(X_info, d - n_info, rng)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_high_noise_classification(n: int = 1500, d: int = 20,
                                    seed: int = SEED) -> tuple:
    """
    High label-noise, overlapping classes.
    d = 20 (5 informative + 15 noise).
    Tests overfitting sensitivity.
    """
    rng = np.random.default_rng(seed)
    n_info = 5
    X_info = rng.normal(0, 1, (n, n_info))
    w = rng.normal(0, 1, n_info)
    score = X_info @ w
    y = (score > 0).astype(int)
    # 25 % label noise
    flip = rng.random(n) < 0.25
    y[flip] = 1 - y[flip]
    X = _add_noise_cols(X_info, d - n_info, rng)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_high_dim_classification(n: int = 5000, d: int = 60,
                                  seed: int = SEED) -> tuple:
    """
    HIGH-DIMENSIONAL dataset: d = 60 >= 50 (requirement), n = 5000 >= 5000.
    5 truly informative features + 55 noise features.
    Tests feature selection & overfitting in high dimensions.
    """
    rng = np.random.default_rng(seed)
    n_info = 5
    X_info = rng.normal(0, 1, (n, n_info))
    w = rng.normal(0, 1, n_info)
    score = X_info @ w
    y = (score > 0).astype(int)
    X = _add_noise_cols(X_info, d - n_info, rng)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_label_noise(n: int = 1500, d: int = 15,
                     flip_rate: float = 0.0, seed: int = SEED) -> tuple:
    """
    Clean classification data with a controllable fraction of label flips.
    d = 15 (5 informative + 10 noise).
    """
    rng = np.random.default_rng(seed)
    n_info = 5
    X_info = rng.normal(0, 1, (n, n_info))
    w = rng.normal(0, 1, n_info)
    score = X_info @ w
    y = (score > 0).astype(int)
    if flip_rate > 0:
        flip = rng.random(n) < flip_rate
        y[flip] = 1 - y[flip]
    X = _add_noise_cols(X_info, d - n_info, rng)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_greedy_failure(n: int = 2000, seed: int = SEED) -> tuple:
    """
    XOR pattern + a decoy feature 70 % correlated with y.
    Greedy splitting picks the decoy (high marginal IG).
    The optimal split uses X1 then X2 (XOR) but has 0 marginal IG.
    d = 15 (2 XOR + 1 decoy + 12 noise).
    """
    rng = np.random.default_rng(seed)
    X_core = rng.normal(0, 1, (n, 2))
    y = ((X_core[:, 0] > 0) ^ (X_core[:, 1] > 0)).astype(int)
    # decoy: 70 % correlated with y
    decoy = (rng.random(n) < 0.70) * y + (rng.random(n) < 0.30) * (1 - y)
    decoy = decoy.astype(float) + rng.normal(0, 0.05, n)
    X = np.column_stack([X_core, decoy])
    X = _add_noise_cols(X, 12, rng)
    idx = rng.permutation(n)
    return X[idx], y[idx]

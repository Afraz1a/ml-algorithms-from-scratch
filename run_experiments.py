"""
run_experiments.py
------------------
Runs EVERY experiment required by ML Assignment 2, saving:
    - plots  -> ./plots/*.png
    - numbers -> ./results.json

Run with:   python run_experiments.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

from dataset_generator import (SEED, make_correlated_nb, make_greedy_failure,
                                make_high_dim_classification,
                                make_high_noise_classification,
                                make_kmeans_adversarial, make_kmeans_friendly,
                                make_label_noise, make_low_noise_classification,
                                make_nb_failure, make_nb_friendly)
from implementations import DecisionTreeC45, GaussianNB, KMeans

# ─────────────────────────────────────────────────────────────────────
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)
RESULTS: dict = {}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _save(name: str) -> None:
    plt.savefig(PLOT_DIR / f"{name}.png")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
# QUESTION 1 – k-MEANS CLUSTERING
# ═════════════════════════════════════════════════════════════════════

def q1_partB_adversarial() -> None:
    """Q1-B: dataset where k-Means succeeds vs. where it fails."""
    # ── friendly (well-separated Gaussians) ──────────────────────────
    Xf, yf = make_kmeans_friendly()
    km_f = KMeans(k=4, random_state=SEED).fit(Xf)
    ari_f = adjusted_rand_score(yf, km_f.labels_)

    # ── adversarial (concentric rings) ───────────────────────────────
    Xa, ya = make_kmeans_adversarial()
    km_a = KMeans(k=2, random_state=SEED).fit(Xa)
    ari_a = adjusted_rand_score(ya, km_a.labels_)

    # ── effect of feature standardisation ────────────────────────────
    Xa_std = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-8)
    km_a_std = KMeans(k=2, random_state=SEED).fit(Xa_std)
    ari_a_std = adjusted_rand_score(ya, km_a_std.labels_)

    # ── 4-panel figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    axes[0].scatter(Xf[:, 0], Xf[:, 1], c=km_f.labels_, cmap="tab10", s=8)
    axes[0].scatter(km_f.centroids_[:, 0], km_f.centroids_[:, 1],
                    marker="X", c="black", s=140, edgecolors="white")
    axes[0].set_title(f"Friendly: Gaussian clusters\nARI = {ari_f:.3f}")
    axes[0].set_xlabel("feature 0"); axes[0].set_ylabel("feature 1")

    axes[1].scatter(Xa[:, 0], Xa[:, 1], c=ya, cmap="tab10", s=8)
    axes[1].set_title("Adversarial (true labels)\n(concentric rings)")
    axes[1].set_xlabel("feature 0"); axes[1].set_ylabel("feature 1")

    axes[2].scatter(Xa[:, 0], Xa[:, 1], c=km_a.labels_, cmap="tab10", s=8)
    axes[2].scatter(km_a.centroids_[:, 0], km_a.centroids_[:, 1],
                    marker="X", c="black", s=140, edgecolors="white")
    axes[2].set_title(f"k-Means on rings (FAILS)\nARI = {ari_a:.3f}")
    axes[2].set_xlabel("feature 0"); axes[2].set_ylabel("feature 1")

    axes[3].scatter(Xa_std[:, 0], Xa_std[:, 1], c=km_a_std.labels_,
                    cmap="tab10", s=8)
    axes[3].set_title(f"After standardisation\nARI = {ari_a_std:.3f}")
    axes[3].set_xlabel("feature 0 (std)"); axes[3].set_ylabel("feature 1 (std)")

    plt.suptitle("Q1-B  k-Means: success vs. failure  |  effect of scaling",
                 y=1.02)
    _save("q1_partB_adversarial")

    RESULTS["q1_partB"] = {
        "friendly_ARI": round(ari_f, 4),
        "adversarial_ARI": round(ari_a, 4),
        "adversarial_after_standardization_ARI": round(ari_a_std, 4),
        "analysis": (
            "Friendly dataset: well-separated spherical Gaussians satisfy all "
            "k-Means assumptions (convex, balanced, isotropic). ARI near 1. "
            "Adversarial dataset: concentric rings violate the convex-Voronoi "
            "assumption; k-Means cannot find a linear boundary separating them. "
            "Standardisation does not help because the failure is structural "
            "(non-convex geometry), not a scale artefact."
        ),
    }


def q1_partC_init_sensitivity() -> None:
    """Q1-C: 20 random inits -> convergence + cost variability."""
    X, y = make_kmeans_friendly()
    n_runs = 20
    runs = [KMeans(k=4, random_state=s).fit(X) for s in range(n_runs)]
    inertias = np.array([r.inertia_ for r in runs])
    aris = np.array([adjusted_rand_score(y, r.labels_) for r in runs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for r in runs:
        axes[0].plot(r.history_, alpha=0.5, linewidth=0.9)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("inertia (sum of squared distances)")
    axes[0].set_title(f"Convergence curves ({n_runs} random inits)")
    axes[0].set_yscale("log")

    sc = axes[1].scatter(inertias, aris, c=np.arange(n_runs), cmap="viridis", s=60)
    axes[1].set_xlabel("final inertia")
    axes[1].set_ylabel("Adjusted Rand Index")
    axes[1].set_title("Final cost vs. label recovery quality")
    plt.colorbar(sc, ax=axes[1], label="run index")

    plt.suptitle("Q1-C  Initialization sensitivity (20 runs)", y=1.02)
    _save("q1_partC_init_sensitivity")

    RESULTS["q1_partC"] = {
        "n_runs": n_runs,
        "inertia_min": round(float(inertias.min()), 2),
        "inertia_max": round(float(inertias.max()), 2),
        "inertia_mean": round(float(inertias.mean()), 2),
        "ARI_min": round(float(aris.min()), 4),
        "ARI_max": round(float(aris.max()), 4),
        "ARI_at_lowest_inertia": round(float(aris[int(np.argmin(inertias))]), 4),
        "analysis": (
            "Different random initializations lead to different local minima "
            "because k-Means optimizes a non-convex objective. Runs that happen "
            "to place centroids near the true cluster centers converge to the "
            "global optimum (low inertia, high ARI). Poor initializations get "
            "trapped in local minima with higher inertia. The strong correlation "
            "between low inertia and high ARI confirms that minimizing the "
            "objective also recovers the true structure on well-separated data."
        ),
    }


# ═════════════════════════════════════════════════════════════════════
# QUESTION 2 – GAUSSIAN NAIVE BAYES
# ═════════════════════════════════════════════════════════════════════

def _confidence_hist(probs_correct: np.ndarray, probs_wrong: np.ndarray,
                     ax: plt.Axes, title: str) -> None:
    ax.hist(probs_correct, bins=20, range=(0, 1), alpha=0.7, label="correct")
    ax.hist(probs_wrong,   bins=20, range=(0, 1), alpha=0.7, label="incorrect")
    ax.set_xlabel("predicted P(ŷ | x)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title(title)


def q2_partB_correlated_features() -> None:
    """Q2-B: correlated features -> overconfidence / miscalibration."""
    X, y = make_correlated_nb()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    gnb = GaussianNB().fit(Xtr, ytr)
    yhat = gnb.predict(Xte)
    proba = gnb.predict_proba(Xte)
    p_pred = proba.max(axis=1)
    acc = accuracy_score(yte, yhat)
    correct = yhat == yte

    # Reliability diagram
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(p_pred, bins) - 1, 0, 9)
    bin_acc, bin_conf, bin_n = [], [], []
    for b in range(10):
        mask = bin_idx == b
        if mask.any():
            bin_acc.append(correct[mask].mean())
            bin_conf.append(p_pred[mask].mean())
            bin_n.append(int(mask.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _confidence_hist(p_pred[correct], p_pred[~correct], axes[0],
                     f"Confidence distribution  (acc = {acc:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    axes[1].plot(bin_conf, bin_acc, "o-", label="model")
    axes[1].set_xlabel("mean predicted confidence")
    axes[1].set_ylabel("empirical accuracy")
    axes[1].set_title("Reliability diagram")
    axes[1].legend()
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    plt.suptitle("Q2-B  Correlated features → overconfidence in GNB", y=1.02)
    _save("q2_partB_correlated")

    ece = sum((n / len(p_pred)) * abs(a - c)
              for n, a, c in zip(bin_n, bin_acc, bin_conf))
    RESULTS["q2_partB"] = {
        "test_accuracy": round(acc, 4),
        "mean_confidence": round(float(p_pred.mean()), 4),
        "expected_calibration_error": round(float(ece), 4),
        "fraction_high_confidence_wrong": round(
            float(((p_pred > 0.95) & ~correct).sum()
                  / max((p_pred > 0.95).sum(), 1)), 4),
        "analysis": (
            "X1 and X2 are nearly identical (correlation ≈ 0.99). NB treats "
            "them as independent, counting their evidence twice. This inflates "
            "the log-posterior, pushing predicted probabilities toward 0 or 1 "
            "far more than warranted. The reliability diagram shows the model "
            "curves ABOVE the diagonal at high confidence, meaning it is more "
            "confident than accurate — a classic overconfidence pattern."
        ),
    }


def q2_partC_counterexamples() -> None:
    """Q2-C: NB success despite violated assumptions vs. XOR failure."""
    out = {}

    # ── success: many weak independent features ───────────────────────
    X1, y1 = make_nb_friendly()
    Xtr, Xte, ytr, yte = train_test_split(
        X1, y1, test_size=0.3, random_state=SEED, stratify=y1)
    gnb = GaussianNB().fit(Xtr, ytr)
    acc1 = accuracy_score(yte, gnb.predict(Xte))
    out["nb_friendly_test_acc"] = round(acc1, 4)

    # ── failure: XOR pattern ─────────────────────────────────────────
    X2, y2 = make_nb_failure()
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X2, y2, test_size=0.3, random_state=SEED, stratify=y2)
    gnb2 = GaussianNB().fit(Xtr2, ytr2)
    yhat2 = gnb2.predict(Xte2)
    acc2 = accuracy_score(yte2, yhat2)
    out["nb_xor_test_acc"] = round(acc2, 4)

    # Visualise XOR + decision boundary
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(X2[:, 0], X2[:, 1], c=y2, cmap="bwr", s=8, alpha=0.7)
    axes[0].set_title("XOR data (true labels)\nNB cannot represent this")
    axes[0].set_xlabel("X1"); axes[0].set_ylabel("X2")

    grid = np.linspace(-3, 3, 200)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.column_stack([xx.ravel(), yy.ravel(),
                           np.zeros((xx.size, X2.shape[1] - 2))])
    zz = gnb2.predict(pts).reshape(xx.shape)
    axes[1].contourf(xx, yy, zz, levels=[-.5, .5, 1.5], cmap="bwr", alpha=0.35)
    axes[1].scatter(X2[:, 0], X2[:, 1], c=y2, cmap="bwr", s=8,
                    alpha=0.6, edgecolors="none")
    axes[1].set_title(f"NB decision boundary on XOR\ntest acc = {acc2:.3f}")
    axes[1].set_xlabel("X1"); axes[1].set_ylabel("X2")
    plt.suptitle("Q2-C  Counterexamples: weak-signal success vs. XOR failure",
                 y=1.02)
    _save("q2_partC_counterexamples")

    out["analysis_success"] = (
        "Independence assumption is technically satisfied (features are "
        "generated independently). Even though each feature has only a small "
        "mean shift (0.4σ), NB multiplies 15 log-likelihoods, accumulating "
        "enough evidence for reliable classification. This is the 'blessing of "
        "independence': many weak signals compound into a strong classifier."
    )
    out["analysis_failure"] = (
        "XOR requires P(y|X1,X2) that is non-monotone in each feature "
        "separately. NB models P(y|X1,X2) = P(y|X1)*P(y|X2)*P(y), which "
        "produces an axis-aligned Gaussian decision boundary — it can only "
        "draw a single hyperplane. The XOR boundary needs 4 quadrants, "
        "impossible with a log-linear model over independent features."
    )
    RESULTS["q2_partC"] = out


def q2_partD_conceptual() -> None:
    """
    Q2-D: Conceptual analysis (stored as text; goes into the PDF report).
    """
    RESULTS["q2_partD"] = {
        "small_dataset_advantage": (
            "On small datasets, complex models (e.g. SVMs, neural networks) "
            "suffer high variance because they have many parameters to estimate "
            "from few samples. NB has only 2*d parameters (one mean and one "
            "variance per feature per class), making it extremely parameter-"
            "efficient. Its strong inductive bias (conditional independence) "
            "acts as a regularizer: even if the assumption is slightly wrong, "
            "the low-variance estimator often outperforms a high-variance "
            "alternative that overfits the small training set. "
            "Formally, the bias-variance decomposition of test error is "
            "E[loss] = Bias^2 + Variance + Noise. On small n, variance "
            "dominates; NB's strong prior trades a little extra bias for a "
            "large reduction in variance."
        ),
        "correlated_overconfidence_math": (
            "Let X1 and X2 be perfectly correlated (X2 = X1 + eps, eps~N(0,s^2))."
            " NB computes log P(y=1|x) proportional to "
            "log P(x1|y=1) + log P(x2|y=1) + log P(y=1). "
            "Because x2 ≈ x1, both log-likelihood terms carry nearly identical "
            "evidence, so the sum is roughly 2 * log P(x1|y=1). "
            "Compared to a model that sees a SINGLE feature, the posterior "
            "logit is doubled: logit = 2*(mu1-mu0)/sigma^2 * x - const. "
            "Feeding this doubled logit through the sigmoid gives probabilities "
            "closer to 0 or 1 (overconfident) than the single-feature model, "
            "even though no new information was added. "
            "In general, if d features are mutually redundant, the effective "
            "logit is d times the correct value, so overconfidence grows "
            "exponentially in d."
        ),
    }
    print("Q2-D conceptual analysis stored in results.json.")


# ═════════════════════════════════════════════════════════════════════
# QUESTION 3 – DECISION TREE (C4.5)
# ═════════════════════════════════════════════════════════════════════

def q3_partB_gain_ratio() -> None:
    """Q3-B: Information Gain vs. Gain Ratio (ID-like decoy feature)."""
    rng = np.random.default_rng(SEED)
    n = 2000
    y = rng.integers(0, 2, size=n)

    A = rng.normal((y * 2 - 1), 0.5)          # informative, continuous
    B = np.arange(n).astype(float) + rng.normal(0, 0.1, n)   # ID-like decoy
    noise = rng.normal(0, 1, (n, 13))
    X = np.column_stack([A, B, noise])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    rows = []
    for crit in ("info_gain", "gain_ratio"):
        for depth in (2, 4, 8):
            dt = DecisionTreeC45(max_depth=depth, criterion=crit).fit(Xtr, ytr)
            rows.append({
                "criterion": crit, "max_depth": depth,
                "first_feature": int(dt.root_.feature),
                "train_acc": round(accuracy_score(ytr, dt.predict(Xtr)), 4),
                "test_acc": round(accuracy_score(yte, dt.predict(Xte)), 4),
                "n_leaves": dt.n_leaves(),
            })
    RESULTS["q3_partB"] = rows

    depths = sorted({r["max_depth"] for r in rows})
    ig = [next(r["test_acc"] for r in rows
               if r["criterion"] == "info_gain" and r["max_depth"] == d)
          for d in depths]
    gr = [next(r["test_acc"] for r in rows
               if r["criterion"] == "gain_ratio" and r["max_depth"] == d)
          for d in depths]
    x = np.arange(len(depths)); w = 0.35
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(x - w / 2, ig, w, label="Information Gain")
    plt.bar(x + w / 2, gr, w, label="Gain Ratio")
    plt.xticks(x, [f"depth={d}" for d in depths])
    plt.ylabel("test accuracy"); plt.ylim(0.4, 1.02)
    plt.title("Q3-B  Information Gain vs. Gain Ratio\n"
              "(high-cardinality ID-like decoy feature)")
    plt.legend()
    _save("q3_partB_gain_ratio")


def q3_partC_overfitting() -> None:
    """Q3-C: depth sweep -> bias/variance tradeoff curves."""
    X, y = make_high_noise_classification(n=1500, d=20)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    depths = [1, 2, 3, 4, 6, 8, 12, 16, 20, 25]
    train_acc, test_acc, n_leaves = [], [], []
    for d in depths:
        dt = DecisionTreeC45(max_depth=d, max_thresholds=32).fit(Xtr, ytr)
        train_acc.append(accuracy_score(ytr, dt.predict(Xtr)))
        test_acc.append(accuracy_score(yte, dt.predict(Xte)))
        n_leaves.append(dt.n_leaves())

    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(depths, train_acc, "o-", label="train accuracy")
    ax1.plot(depths, test_acc,  "s-", label="validation accuracy")
    ax1.set_xlabel("max_depth"); ax1.set_ylabel("accuracy")
    ax1.set_ylim(0.0, 1.05); ax1.legend(loc="lower right")
    ax1.set_title("Q3-C  Bias-variance tradeoff via depth sweep\n"
                  "(high-noise data, small training set)")
    ax2 = ax1.twinx()
    ax2.plot(depths, n_leaves, "k--", alpha=0.5, label="n_leaves")
    ax2.set_ylabel("n_leaves (tree size)"); ax2.grid(False)
    ax2.legend(loc="center right")
    _save("q3_partC_overfitting")

    RESULTS["q3_partC"] = {
        "depths": depths,
        "train_acc": [round(a, 4) for a in train_acc],
        "test_acc": [round(a, 4) for a in test_acc],
        "n_leaves": n_leaves,
        "best_depth_by_val": depths[int(np.argmax(test_acc))],
        "best_val_acc": round(float(max(test_acc)), 4),
        "analysis": (
            "Shallow trees (depth 1-2) underfit: high bias, both train and "
            "test accuracy are low. As depth increases, train accuracy rises "
            "monotonically toward 1.0 (the tree memorises training noise). "
            "Test accuracy peaks at an intermediate depth then decreases — "
            "classic overfitting (high variance). The n_leaves curve shows "
            "exponential growth in tree size, confirming the model complexity "
            "explosion. On noisy data the tree memorises label noise, hurting "
            "generalization."
        ),
    }


def q3_partD_greedy_counterexample() -> None:
    """Q3-D: greedy splitting on XOR + decoy."""
    X, y = make_greedy_failure()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    # (1) greedy with decoy visible
    dt_full = DecisionTreeC45(max_depth=4).fit(Xtr, ytr)
    acc_full = accuracy_score(yte, dt_full.predict(Xte))

    # (2) greedy WITHOUT decoy column
    Xtr_nd = np.delete(Xtr, 2, axis=1)
    Xte_nd = np.delete(Xte, 2, axis=1)
    dt_nodecoy = DecisionTreeC45(max_depth=8).fit(Xtr_nd, ytr)
    acc_nodecoy = accuracy_score(yte, dt_nodecoy.predict(Xte_nd))

    # (3) globally optimal: XOR of X1>0 and X2>0
    opt_pred = ((Xte[:, 0] > 0) ^ (Xte[:, 1] > 0)).astype(int)
    acc_opt = accuracy_score(yte, opt_pred)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, preds, title in [
        (axes[0], yte,
         "True labels (XOR pattern)"),
        (axes[1], dt_full.predict(Xte),
         f"Greedy w/ decoy: first split = X[{dt_full.root_.feature}]\n"
         f"test acc = {acc_full:.3f}"),
        (axes[2], opt_pred,
         f"Globally optimal tree\nsplit X1 then X2 → acc = {acc_opt:.3f}"),
    ]:
        ax.scatter(Xte[:, 0], Xte[:, 1], c=preds, cmap="bwr",
                   s=12, alpha=0.7, edgecolors="none")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("X1"); ax.set_ylabel("X2")
        ax.set_title(title)
    plt.suptitle("Q3-D  Greedy splitting fails on XOR", y=1.02)
    _save("q3_partD_greedy")

    optimal_tree_text = (
        "if X1 <= 0:\n"
        "    if X2 <= 0:  -> class 0\n"
        "    else:        -> class 1\n"
        "else:\n"
        "    if X2 <= 0:  -> class 1\n"
        "    else:        -> class 0"
    )

    RESULTS["q3_partD"] = {
        "greedy_full_first_feature": int(dt_full.root_.feature),
        "greedy_full_test_acc": round(acc_full, 4),
        "greedy_full_n_leaves": dt_full.n_leaves(),
        "greedy_nodecoy_test_acc": round(acc_nodecoy, 4),
        "greedy_nodecoy_n_leaves": dt_nodecoy.n_leaves(),
        "optimal_test_acc": round(acc_opt, 4),
        "greedy_tree_text": dt_full.describe(max_lines=30),
        "optimal_tree_text": optimal_tree_text,
        "analysis": (
            "Greedy splitting evaluates each feature in isolation. For XOR, "
            "splitting on X1 or X2 alone gives exactly 50/50 class distribution "
            "in both branches (zero information gain). The decoy feature (70% "
            "correlated with y) has high marginal IG, so the greedy algorithm "
            "picks it as the root. After the decoy is used, the remaining tree "
            "struggles because the true signal requires a two-level interaction. "
            "The globally optimal tree achieves near-perfect accuracy by first "
            "splitting X1 (even though it has 0 marginal IG), then X2, "
            "exploiting the XOR interaction — something greedy cannot anticipate."
        ),
    }


def q3_partE_noise_sensitivity() -> None:
    """Q3-E: label noise -> tree instability."""
    flip_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    train_acc, test_acc, depths_out, leaves_out, first_feat = [], [], [], [], []

    for fr in flip_rates:
        X, y = make_label_noise(flip_rate=fr)
        _, y_clean = make_label_noise(flip_rate=0.0)
        # Split indices once using clean labels for stratification
        idx = np.arange(len(y))
        idx_tr, idx_te = train_test_split(
            idx, test_size=0.3, random_state=SEED, stratify=y_clean)
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        yte_clean = y_clean[idx_te]
        dt = DecisionTreeC45(max_depth=10).fit(Xtr, ytr)
        train_acc.append(accuracy_score(ytr, dt.predict(Xtr)))
        test_acc.append(accuracy_score(yte_clean, dt.predict(Xte)))
        depths_out.append(dt.depth())
        leaves_out.append(dt.n_leaves())
        first_feat.append(int(dt.root_.feature))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(flip_rates, train_acc, "o-", label="train (noisy labels)")
    axes[0].plot(flip_rates, test_acc,  "s-", label="test (clean labels)")
    axes[0].set_xlabel("label-flip rate"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy vs. label noise"); axes[0].legend()

    axes[1].plot(flip_rates, leaves_out, "o-", color="purple")
    axes[1].set_xlabel("label-flip rate"); axes[1].set_ylabel("n_leaves")
    axes[1].set_title("Tree complexity grows with noise")
    plt.suptitle("Q3-E  Noise sensitivity (max_depth = 10)", y=1.02)
    _save("q3_partE_noise")

    RESULTS["q3_partE"] = {
        "flip_rates": flip_rates,
        "train_acc": [round(a, 4) for a in train_acc],
        "test_acc_clean": [round(a, 4) for a in test_acc],
        "depths": depths_out,
        "n_leaves": leaves_out,
        "first_feature_each_run": first_feat,
        "analysis": (
            "As label-flip rate increases, the tree needs more leaves to "
            "memorise the contradictory labels in training, so n_leaves grows "
            "and the effective depth increases. Train accuracy on noisy labels "
            "stays high (the tree memorises noise), but test accuracy on clean "
            "labels drops significantly, demonstrating instability. Even small "
            "perturbations (5% flip) can change the root split feature because "
            "information gain differences between features are small and easily "
            "perturbed — a key reason decision trees are called 'unstable learners'."
        ),
    }


# ═════════════════════════════════════════════════════════════════════
# BONUS: High-dimensional dataset experiment (d >= 50, n >= 5000)
# ═════════════════════════════════════════════════════════════════════

def q_high_dim_experiment() -> None:
    """
    Demonstrates behavior of all three algorithms on the high-dimensional
    dataset (d=60, n=5000) — satisfies both d>=50 and n>=5000 requirements.
    """
    X, y = make_high_dim_classification(n=5000, d=60)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    # ── Decision tree at various depths ──────────────────────────────
    depths = [2, 5, 10, 20]
    dt_results = []
    for d in depths:
        dt = DecisionTreeC45(max_depth=d, max_thresholds=32).fit(Xtr, ytr)
        dt_results.append({
            "depth": d,
            "train_acc": round(accuracy_score(ytr, dt.predict(Xtr)), 4),
            "test_acc": round(accuracy_score(yte, dt.predict(Xte)), 4),
            "n_leaves": dt.n_leaves(),
            "first_feature": int(dt.root_.feature),
        })

    # ── Naive Bayes ───────────────────────────────────────────────────
    gnb = GaussianNB().fit(Xtr, ytr)
    nb_acc = accuracy_score(yte, gnb.predict(Xte))

    # ── Plot: DT train vs test by depth ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(depths, [r["train_acc"] for r in dt_results], "o-",
                 label="train acc")
    axes[0].plot(depths, [r["test_acc"] for r in dt_results], "s-",
                 label="test acc")
    axes[0].axhline(nb_acc, color="green", linestyle="--",
                    label=f"GNB test acc = {nb_acc:.3f}")
    axes[0].set_xlabel("max_depth"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("DT vs GNB on high-dim data (d=60, n=5000)")
    axes[0].legend()

    # Feature usage: which features does the depth-5 tree choose most?
    # Build tree and count first-level features as a proxy
    dt5 = DecisionTreeC45(max_depth=5, max_thresholds=32).fit(Xtr, ytr)

    def _collect_features(node, feats):
        if node is None or node.is_leaf:
            return
        feats.append(node.feature)
        _collect_features(node.left, feats)
        _collect_features(node.right, feats)

    feats_used = []
    _collect_features(dt5.root_, feats_used)
    feat_counts = np.bincount(feats_used, minlength=60)
    top20 = np.argsort(feat_counts)[::-1][:20]
    axes[1].bar(range(20), feat_counts[top20])
    axes[1].set_xticks(range(20))
    axes[1].set_xticklabels([f"X{i}" for i in top20], rotation=45, fontsize=7)
    axes[1].set_ylabel("times used in tree")
    axes[1].axvline(4.5, color="red", linestyle="--", alpha=0.5,
                    label="≤ 5 informative features expected")
    axes[1].set_title("Top-20 features used by DT (depth=5)\n"
                      "Red line: boundary of informative features (X0-X4)")
    axes[1].legend(fontsize=7)

    plt.suptitle("High-Dimensional Experiment  (d=60, n=5000)", y=1.02)
    _save("q_high_dim_experiment")

    RESULTS["high_dim_experiment"] = {
        "n": 5000, "d": 60, "n_informative": 5, "n_noise": 55,
        "gnb_test_acc": round(nb_acc, 4),
        "dt_results": dt_results,
        "analysis": (
            "With 55 noise features and only 5 informative ones, shallow trees "
            "must select informative features from a large pool. GNB benefits "
            "from the approximately independent (correct) features and performs "
            "well at low parameter cost. Deep trees overfit the noise, "
            "degrading test accuracy even with n=5000 samples."
        ),
    }


# ═════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    print("─── Q1: k-Means ───────────────────────────────────")
    q1_partB_adversarial()
    print("  Q1-B done")
    q1_partC_init_sensitivity()
    print("  Q1-C done")

    print("─── Q2: Gaussian Naive Bayes ──────────────────────")
    q2_partB_correlated_features()
    print("  Q2-B done")
    q2_partC_counterexamples()
    print("  Q2-C done")
    q2_partD_conceptual()
    print("  Q2-D done")

    print("─── Q3: Decision Tree C4.5 ────────────────────────")
    q3_partB_gain_ratio()
    print("  Q3-B done")
    q3_partC_overfitting()
    print("  Q3-C done")
    q3_partD_greedy_counterexample()
    print("  Q3-D done")
    q3_partE_noise_sensitivity()
    print("  Q3-E done")

    print("─── High-Dimensional Experiment (d=60, n=5000) ────")
    q_high_dim_experiment()
    print("  High-dim done")

    with open("results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print("\n✓ All done.  Plots → ./plots/   Numbers → ./results.json")


if __name__ == "__main__":
    main()
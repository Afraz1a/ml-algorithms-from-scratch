"""
implementations.py
------------------
From-scratch implementations (NumPy only) of:
    1. KMeans
    2. GaussianNB
    3. DecisionTreeC45  (entropy / information gain / gain ratio)

No scikit-learn classifiers, clustering, or decision-tree APIs are used.
"""

from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# 1.  k-Means Clustering
# ══════════════════════════════════════════════════════════════════════════

class KMeans:
    """
    k-Means clustering from scratch.

    Parameters
    ----------
    k            : number of clusters
    max_iter     : maximum EM iterations
    tol          : centroid-shift tolerance for early stopping
    random_state : seed for reproducibility

    Attributes (set after fit)
    --------------------------
    centroids_ : ndarray (k, d)
    labels_    : ndarray (n,)  – integer cluster assignments
    inertia_   : float         – final sum of squared distances
    history_   : list[float]   – inertia after each iteration
    """

    def __init__(self, k: int = 3, max_iter: int = 300,
                 tol: float = 1e-4, random_state: int = 0) -> None:
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        KMeans++ initialisation: first centroid random, each subsequent
        centroid chosen with probability proportional to squared distance
        from the nearest existing centroid.  Much better than pure random.
        """
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        # 1st centroid: uniformly random
        idx = [int(rng.integers(n))]
        for _ in range(self.k - 1):
            # squared distance from each point to its nearest centroid
            dists = np.array([
                min((np.sum((x - X[c]) ** 2) for c in idx))
                for x in X
            ])
            # sample next centroid proportionally
            probs = dists / dists.sum()
            idx.append(int(rng.choice(n, p=probs)))
        return X[np.array(idx)].copy()

    # ------------------------------------------------------------------
    @staticmethod
    def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Vectorised Euclidean distance -> nearest centroid.
        Shape: X (n,d), centroids (k,d) -> labels (n,)
        Uses broadcasting: (n,1,d) - (1,k,d) -> (n,k,d) -> (n,k)
        """
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]   # (n,k,d)
        sq_dist = (diff ** 2).sum(axis=2)                           # (n,k)
        return sq_dist.argmin(axis=1)                               # (n,)

    # ------------------------------------------------------------------
    @staticmethod
    def _update(X: np.ndarray, labels: np.ndarray,
                k: int, rng: np.random.Generator) -> np.ndarray:
        """
        Vectorised centroid update via boolean masks.
        Empty clusters are reinitialised to a random point using the
        seeded rng (fixes reproducibility bug from np.random.randint).
        """
        centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
            else:
                # Reinitialise empty cluster — use seeded rng, not global random
                centroids[j] = X[rng.integers(X.shape[0])]
        return centroids

    # ------------------------------------------------------------------
    @staticmethod
    def _inertia(X: np.ndarray, labels: np.ndarray,
                 centroids: np.ndarray) -> float:
        diff = X - centroids[labels]
        return float((diff ** 2).sum())

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.random_state)
        centroids = self._init_centroids(X)
        labels = self._assign(X, centroids)
        self.history_: list[float] = []

        for _ in range(self.max_iter):
            new_centroids = self._update(X, labels, self.k, rng)
            new_labels = self._assign(X, new_centroids)
            inertia = self._inertia(X, new_labels, new_centroids)
            self.history_.append(inertia)

            # stopping criterion: centroid shift
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            labels = new_labels
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = self._inertia(X, labels, centroids)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(X, self.centroids_)


# ══════════════════════════════════════════════════════════════════════════
# 2.  Gaussian Naive Bayes
# ══════════════════════════════════════════════════════════════════════════

class GaussianNB:
    """
    Gaussian Naive Bayes from scratch.

    Assumptions
    -----------
    P(y) : class prior, estimated by MLE.
    P(x_j | y) : Gaussian with class-conditional mean & variance.
    Prediction : argmax_y  log P(y) + sum_j log P(x_j | y)

    Numerical stability
    -------------------
    Variances are smoothed by adding a small epsilon to avoid log(0).
    All computations are in log-space.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        self.classes_ = np.unique(y)
        n, d = X.shape

        # Priors
        self.log_prior_ = np.array(
            [np.log((y == c).sum() / n) for c in self.classes_])

        # Per-class mean and variance (MLE)
        self.theta_ = np.zeros((len(self.classes_), d))   # means
        self.sigma_ = np.zeros((len(self.classes_), d))   # variances

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.sigma_[i] = Xc.var(axis=0) + self.var_smoothing

        return self

    # ------------------------------------------------------------------
    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(X | y=c) for every class c.
        Returns shape (n, n_classes).
        Gaussian log-pdf: -0.5*log(2*pi*var) - 0.5*(x-mu)^2/var
        Variances are clipped to var_smoothing to prevent log(0) or div-by-zero.
        """
        n = X.shape[0]
        k = len(self.classes_)
        log_liks = np.zeros((n, k))
        for i in range(k):
            mu = self.theta_[i]          # (d,)
            var = np.maximum(self.sigma_[i], self.var_smoothing)  # safe clip
            log_liks[:, i] = (
                -0.5 * np.sum(np.log(2 * np.pi * var))
                - 0.5 * np.sum(((X - mu) ** 2) / var, axis=1)
            )
        return log_liks

    # ------------------------------------------------------------------
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Log posterior (unnormalised) for each class."""
        return self._log_likelihood(X) + self.log_prior_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalised posterior probabilities."""
        log_post = self.predict_log_proba(X)
        # Numerically stable softmax over classes
        log_post -= log_post.max(axis=1, keepdims=True)
        proba = np.exp(log_post)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_log_proba(X).argmax(axis=1)]


# ══════════════════════════════════════════════════════════════════════════
# 3.  Decision Tree – C4.5
# ══════════════════════════════════════════════════════════════════════════

class _Node:
    """Internal node or leaf of the decision tree."""
    __slots__ = ["feature", "threshold", "left", "right",
                 "is_leaf", "prediction", "n_samples"]

    def __init__(self):
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None
        self.is_leaf: bool = False
        self.prediction: int | None = None
        self.n_samples: int = 0


class DecisionTreeC45:
    """
    Binary Decision Tree using the C4.5 algorithm.

    Split criterion: Gain Ratio  (or Information Gain if criterion='info_gain')
        Entropy(S)      = -sum p_k log2(p_k)
        IG(S, A)        = Entropy(S) - sum |S_v|/|S| * Entropy(S_v)
        SplitInfo(S, A) = -sum |S_v|/|S| * log2(|S_v|/|S|)
        GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

    Parameters
    ----------
    max_depth       : maximum tree depth (None = unlimited)
    min_samples_split: minimum samples to consider splitting
    criterion       : 'gain_ratio' (default) or 'info_gain'
    max_thresholds  : max candidate thresholds per feature (for speed)
    """

    def __init__(self, max_depth: int | None = None,
                 min_samples_split: int = 2,
                 criterion: str = "gain_ratio",
                 max_thresholds: int = 64) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_thresholds = max_thresholds
        self.root_: _Node | None = None

    # ------------------------------------------------------------------
    # Entropy helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / n
        # only terms with p > 0
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    # ------------------------------------------------------------------
    def _best_split(self, X: np.ndarray,
                    y: np.ndarray) -> tuple[int, float, float]:
        """
        Find (feature, threshold, score) that maximises gain ratio (or IG).
        Returns (-1, 0.0, 0.0) if no valid split found.
        """
        n, d = X.shape
        H_parent = self._entropy(y)
        best_score = -np.inf
        best_feat, best_thr = -1, 0.0

        for feat in range(d):
            col = X[:, feat]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            # Sample thresholds for speed
            if len(unique_vals) > self.max_thresholds:
                quantiles = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
                thresholds = np.percentile(col, quantiles)
                thresholds = np.unique(thresholds)
            else:
                # midpoints between consecutive unique values
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                n_l, n_r = left_mask.sum(), right_mask.sum()
                if n_l == 0 or n_r == 0:
                    continue

                # Information Gain
                H_l = self._entropy(y[left_mask])
                H_r = self._entropy(y[right_mask])
                ig = H_parent - (n_l * H_l + n_r * H_r) / n

                if self.criterion == "info_gain":
                    score = ig
                else:
                    # Gain Ratio
                    p_l, p_r = n_l / n, n_r / n
                    split_info = -(p_l * np.log2(p_l) + p_r * np.log2(p_r))
                    if split_info < 1e-10:
                        continue
                    score = ig / split_info

                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr, best_score

    # ------------------------------------------------------------------
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node()
        node.n_samples = len(y)

        # Safe majority-class vote (handles non-contiguous class labels)
        classes, counts = np.unique(y, return_counts=True)
        majority_class = int(classes[np.argmax(counts)])

        # Leaf conditions
        if (len(classes) == 1
                or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            node.is_leaf = True
            node.prediction = majority_class
            return node

        feat, thr, score = self._best_split(X, y)
        if feat == -1 or score <= 0:
            node.is_leaf = True
            node.prediction = majority_class
            return node

        node.feature = feat
        node.threshold = thr
        mask = X[:, feat] <= thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeC45":
        y = y.astype(int)
        self.root_ = self._build(X, y, depth=0)
        return self

    # ------------------------------------------------------------------
    def _predict_one(self, node: _Node, x: np.ndarray) -> int:
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(self.root_, row) for row in X])

    # ------------------------------------------------------------------
    # Tree statistics
    # ------------------------------------------------------------------
    def n_leaves(self) -> int:
        def _count(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root_)

    def depth(self) -> int:
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root_)

    # ------------------------------------------------------------------
    # Text representation (for reports)
    # ------------------------------------------------------------------
    def describe(self, max_lines: int = 50) -> str:
        lines: list[str] = []

        def _walk(node: _Node, indent: int = 0) -> None:
            if len(lines) >= max_lines:
                return
            prefix = "  " * indent
            if node.is_leaf:
                lines.append(f"{prefix}-> class {node.prediction}"
                             f"  (n={node.n_samples})")
            else:
                lines.append(f"{prefix}[X{node.feature} <= "
                             f"{node.threshold:.4f}]  (n={node.n_samples})")
                _walk(node.left, indent + 1)
                _walk(node.right, indent + 1)

        _walk(self.root_)
        if len(lines) >= max_lines:
            lines.append("  ... (truncated)")
        return "\n".join(lines)
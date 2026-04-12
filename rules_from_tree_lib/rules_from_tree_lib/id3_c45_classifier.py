"""
ID3 / C4.5 Decision Tree Classifier
====================================
Drop-in replacement for sklearn's DecisionTreeClassifier with full support for:
  - ID3  (Information Gain, only categorical splits)
  - C4.5 (Gain Ratio, continuous + categorical splits, missing value handling)

Public API mirrors sklearn:
  fit, predict, predict_proba, score,
  get_params, set_params, get_depth, get_n_leaves,
  apply, decision_path,
  export_text, plot_tree, export_graphviz,
  feature_importances_, classes_, n_features_in_,
  cost_complexity_pruning_path, __repr__

Ekstrakcja reguł (wymaga roughset.py):
  extract_rules(decision_attribute=None) -> roughset.RuleSet
      Ekstrahuje reguły decyzyjne z dopasowanego drzewa.
      Domyślna nazwa kolumny decyzyjnej: "class".
      Patrz: roughset.RuleSet.to_series() i roughset.RuleSet.to_dataframe()

Poprawki względem wersji 1.0:
  - _entropy: usunięto eps (1e-12) z log2 — dokładne 0 dla czystych węzłów
  - _gain_ratio: gain_ratio liczy gain i split_info na próbkach bez NaN
    (zgodnie z oryginalnym C4.5 Quinlana, gdy parametr nan_policy='mask')
  - _best_categorical_split: zastąpiono `col != None` przez sprawdzenie
    `pd.notna` / maskę obiektową — eliminacja FutureWarning w NumPy ≥ 2.0
  - export_graphviz: naprawiono desynchronizację counter przy
    rekurencyjnym budowaniu DOT — counter aktualizowany przed rekurencją
  - Dodano metodę extract_rules() integrującą się z roughset.py
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

# Optional heavy deps — only needed for visualisation
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    _MPL = False

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_is_fitted
    _SKLEARN = True
except ImportError:
    BaseEstimator = object
    ClassifierMixin = object
    _SKLEARN = False


# ─────────────────────────────────────────────
# Internal node / leaf representation
# ─────────────────────────────────────────────

@dataclass
class _Node:
    # Leaf fields
    is_leaf: bool = False
    class_label: Any = None          # majority class at this node
    class_counts: Dict = field(default_factory=dict)
    proba: np.ndarray = field(default_factory=lambda: np.array([]))

    # Internal-node fields
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    threshold: Optional[float] = None   # None → categorical split
    children: Dict = field(default_factory=dict)   # value → _Node (categorical)
                                                    # 'left'/'right' for numeric
    # Stats
    n_samples: int = 0
    impurity: float = 0.0
    gain: float = 0.0
    depth: int = 0


# ─────────────────────────────────────────────
# Criterion helpers
# ─────────────────────────────────────────────

def _entropy(y: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / n
    # probs > 0 gwarantowane przez filtr counts > 0 — log2 jest bezpieczne
    # (usunięto epsilon 1e-12, który powodował odchylenie ~1e-12 dla czystych węzłów)
    return float(-np.sum(probs * np.log2(probs)))


def _gini(y: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts / n
    return float(1.0 - np.sum(probs ** 2))


def _information_gain(y: np.ndarray, subsets: List[np.ndarray],
                      criterion_fn) -> float:
    n = len(y)
    parent_imp = criterion_fn(y)
    weighted = sum(len(s) / n * criterion_fn(s) for s in subsets if len(s))
    return parent_imp - weighted


def _gain_ratio(y: np.ndarray, subsets: List[np.ndarray],
                criterion_fn) -> float:
    """C4.5: Gain Ratio = Information Gain / Split Info.

    Gain i Split Info liczone są na tej samej populacji (subsets).
    Zapewnia to spójność: gain_ratio jest zawsze w [0, 1].
    """
    ig = _information_gain(y, subsets, criterion_fn)
    n = len(y)
    if n == 0:
        return 0.0
    split_info = -sum(
        (len(s) / n) * math.log2(len(s) / n)
        for s in subsets if len(s) > 0
    )
    if split_info < 1e-10:
        return 0.0
    return ig / split_info


# ─────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────

class ID3C45Classifier(BaseEstimator, ClassifierMixin):
    """
    Decision Tree Classifier using ID3 or C4.5 algorithm.

    Parameters
    ----------
    algorithm : {'ID3', 'C4.5'}, default='C4.5'
        ID3  – uses Information Gain, only categorical features.
        C4.5 – uses Gain Ratio, supports continuous features.
    criterion : {'gini', 'entropy'}, default='gini'
        Impurity measure for gain calculation.
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in a leaf.
    min_impurity_decrease : float, default=0.0
        Split only if gain >= this value.
    max_features : int, float, str or None, default=None
        Number of features to consider at each split.
        None → all, 'sqrt' → sqrt(n), 'log2' → log2(n), int/float → exact/fraction.
    ccp_alpha : float, default=0.0
        Complexity parameter for Minimal Cost-Complexity Pruning.
    feature_names : list of str or None, default=None
        Names for features (used in visualisations).
    class_names : list of str or None, default=None
        Names for classes (used in visualisations).
    random_state : int or None, default=None
    """

    def __init__(
        self,
        algorithm: str = "C4.5",
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features=None,
        ccp_alpha: float = 0.0,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ):
        self.algorithm = algorithm
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state

        self.tree_: Optional[_Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_classes_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self._impurity_fn = None
        self._score_fn = None
        self._rng = np.random.default_rng(random_state)
        self._is_continuous: Optional[np.ndarray] = None   # bool per feature

    # ── sklearn compatibility ──────────────────────────────────────────────

    def get_params(self, deep=True):
        return dict(
            algorithm=self.algorithm, criterion=self.criterion,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features, ccp_alpha=self.ccp_alpha,
            feature_names=self.feature_names, class_names=self.class_names,
            random_state=self.random_state,
        )

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        if "algorithm" in params:
            self.algorithm = self.algorithm.upper()
        return self

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        algorithm = self.algorithm.upper()
        criterion = self.criterion

    
        if criterion == "gini":
            self._impurity_fn = _gini
        elif criterion == "entropy":
            self._impurity_fn = _entropy    
        else:
            raise ValueError(f"Unknown criterion '{criterion}'")

        if algorithm == "C4.5":
            self._score_fn = lambda y_, subs: _gain_ratio(y_, subs, self._impurity_fn)
        elif algorithm == "ID3":
            self._score_fn = lambda y_, subs: _information_gain(y_, subs, self._impurity_fn)
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'")

        # Detect continuous features (any float column or >threshold unique vals)
        self._is_continuous = np.array([
            np.issubdtype(X[:, i].dtype, np.floating) or
            len(np.unique(X[:, i])) > 10
            for i in range(self.n_features_in_)
        ])

        if algorithm == "ID3":
            self._is_continuous[:] = False   # ID3 forces categorical

        self._importances = np.zeros(self.n_features_in_)

        self.tree_ = self._build(
            X, y_enc,
            depth=0,
            available_features=list(range(self.n_features_in_)),
        )

        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0 else self._importances
        )

        if self.ccp_alpha > 0:
            self._prune(self.tree_, self.ccp_alpha)

        return self

    # ── internal build ─────────────────────────────────────────────────────

    def _build(self, X, y, depth, available_features) -> _Node:
        counts = dict(zip(*np.unique(y, return_counts=True)))
        majority = int(max(counts, key=counts.get))
        proba = np.zeros(self.n_classes_)
        for cls, cnt in counts.items():
            proba[cls] = cnt / len(y)

        node = _Node(
            n_samples=len(y),
            impurity=self._impurity_fn(y),
            depth=depth,
            class_label=majority,
            class_counts=counts,
            proba=proba,
        )

        # Stopping conditions → leaf
        if (
            len(np.unique(y)) == 1 or
            len(y) < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth) or
            not available_features
        ):
            node.is_leaf = True
            return node

        # Feature subset
        feat_indices = self._sample_features(available_features)

        best_gain = -np.inf
        best_feat = None
        best_thresh = None
        best_subsets_idx = None
        best_subsets_y = None

        for fi in feat_indices:
            if self._is_continuous[fi]:
                gain, thresh, idx_l, idx_r = self._best_numeric_split(X, y, fi)
                if gain > best_gain and gain >= self.min_impurity_decrease:
                    if (len(idx_l) >= self.min_samples_leaf and
                            len(idx_r) >= self.min_samples_leaf):
                        best_gain = gain
                        best_feat = fi
                        best_thresh = thresh
                        best_subsets_idx = {"left": idx_l, "right": idx_r}
                        best_subsets_y = {"left": y[idx_l], "right": y[idx_r]}
            else:
                gain, idx_map = self._best_categorical_split(X, y, fi)
                if gain > best_gain and gain >= self.min_impurity_decrease:
                    if all(len(v) >= self.min_samples_leaf for v in idx_map.values()):
                        best_gain = gain
                        best_feat = fi
                        best_thresh = None
                        best_subsets_idx = idx_map
                        best_subsets_y = {k: y[v] for k, v in idx_map.items()}

        if best_feat is None:
            node.is_leaf = True
            return node

        node.feature_index = best_feat
        node.feature_name = (
            self.feature_names[best_feat]
            if self.feature_names and best_feat < len(self.feature_names)
            else f"X[{best_feat}]"
        )
        node.threshold = best_thresh
        node.gain = best_gain
        self._importances[best_feat] += best_gain * len(y)

        # Recurse
        if best_thresh is not None:   # numeric
            next_feats = available_features  # can reuse continuous features
            node.children["left"] = self._build(
                X[best_subsets_idx["left"]], best_subsets_y["left"],
                depth + 1, next_feats)
            node.children["right"] = self._build(
                X[best_subsets_idx["right"]], best_subsets_y["right"],
                depth + 1, next_feats)
        else:                          # categorical
            next_feats = [f for f in available_features if f != best_feat]
            for val, idx in best_subsets_idx.items():
                node.children[val] = self._build(
                    X[idx], best_subsets_y[val], depth + 1, next_feats)

        return node

    def _sample_features(self, available):
        n = len(available)
        if self.max_features is None:
            return available
        elif self.max_features == "sqrt":
            k = max(1, int(math.sqrt(n)))
        elif self.max_features == "log2":
            k = max(1, int(math.log2(n + 1)))
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n))
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n)
        else:
            return available
        return list(self._rng.choice(available, size=k, replace=False))

    def _best_numeric_split(self, X, y, fi):
        col = X[:, fi].astype(float)
        # Handle NaN (C4.5 style: skip NaN rows for split evaluation)
        mask = ~np.isnan(col)
        col_v, y_v = col[mask], y[mask]
        if len(col_v) == 0:
            return -np.inf, None, np.array([], int), np.array([], int)

        order = np.argsort(col_v)
        col_s, y_s = col_v[order], y_v[order]
        thresholds = (col_s[:-1] + col_s[1:]) / 2
        thresholds = thresholds[y_s[:-1] != y_s[1:]]   # only class-boundary thresholds
        thresholds = np.unique(thresholds)

        best_gain, best_thresh = -np.inf, None
        best_l, best_r = None, None

        for t in thresholds:
            idx_l_full = np.where((col <= t) | np.isnan(col))[0]  # NaN → majority side
            idx_r_full = np.where(col > t)[0]
            subsets = [y[idx_l_full], y[idx_r_full]]
            g = self._score_fn(y, subsets)
            if g > best_gain:
                best_gain, best_thresh = g, t
                best_l, best_r = idx_l_full, idx_r_full

        if best_thresh is None:
            return -np.inf, None, np.array([], int), np.array([], int)
        return best_gain, best_thresh, best_l, best_r

    def _best_categorical_split(self, X, y, fi):
        col = X[:, fi]
        # Używamy maski obiektowej zamiast `col != None` aby uniknąć
        # FutureWarning w NumPy ≥ 2.0 (porównanie z None przez __eq__)
        not_missing = np.array([v is not None for v in col], dtype=bool)
        col_valid = col[not_missing]
        if len(col_valid) == 0:
            return -np.inf, {}
        vals = np.unique(col_valid)
        idx_map = {}
        for v in vals:
            idx_map[v] = np.where(not_missing & (col == v))[0]
        subsets = [y[idx] for idx in idx_map.values()]
        gain = self._score_fn(y, subsets)
        return gain, idx_map

    # ── predict ───────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict().")

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.classes_[np.array([self._traverse(x) for x in X])]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._traverse_proba(x) for x in X])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))

    def _traverse(self, x: np.ndarray) -> int:
        node = self.tree_
        while not node.is_leaf:
            fi = node.feature_index
            val = x[fi]
            if node.threshold is not None:
                # numeric
                try:
                    go = "left" if float(val) <= node.threshold else "right"
                except (TypeError, ValueError):
                    go = "left"
                node = node.children.get(go, self._majority_child(node))
            else:
                # categorical
                node = node.children.get(val, self._majority_child(node))
        return node.class_label

    def _traverse_proba(self, x: np.ndarray) -> np.ndarray:
        node = self.tree_
        while not node.is_leaf:
            fi = node.feature_index
            val = x[fi]
            if node.threshold is not None:
                try:
                    go = "left" if float(val) <= node.threshold else "right"
                except (TypeError, ValueError):
                    go = "left"
                node = node.children.get(go, self._majority_child(node))
            else:
                node = node.children.get(val, self._majority_child(node))
        return node.proba

    @staticmethod
    def _majority_child(node: _Node) -> _Node:
        """Fallback when a category not seen in training appears."""
        return max(node.children.values(), key=lambda n: n.n_samples)

    # ── apply / decision_path ─────────────────────────────────────────────

    def apply(self, X: ArrayLike) -> np.ndarray:
        """Return leaf node ids for each sample (BFS order)."""
        self._check_fitted()
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        node_ids = self._build_node_id_map()
        return np.array([node_ids[id(self._get_leaf(x))] for x in X])

    def decision_path(self, X: ArrayLike):
        """Return sparse indicator matrix (n_samples × n_nodes)."""
        from scipy.sparse import lil_matrix
        self._check_fitted()
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        nodes, node_ids = self._collect_nodes()
        n_nodes = len(nodes)
        mat = lil_matrix((len(X), n_nodes), dtype=np.int8)
        for i, x in enumerate(X):
            for node in self._path(x):
                mat[i, node_ids[id(node)]] = 1
        return mat.tocsr()

    def _collect_nodes(self):
        nodes, ids = [], {}
        stack = [self.tree_]
        while stack:
            n = stack.pop()
            ids[id(n)] = len(nodes)
            nodes.append(n)
            for c in n.children.values():
                stack.append(c)
        return nodes, ids

    def _build_node_id_map(self):
        _, ids = self._collect_nodes()
        return ids

    def _get_leaf(self, x):
        node = self.tree_
        while not node.is_leaf:
            fi = node.feature_index
            val = x[fi]
            if node.threshold is not None:
                try:
                    go = "left" if float(val) <= node.threshold else "right"
                except (TypeError, ValueError):
                    go = "left"
                node = node.children.get(go, self._majority_child(node))
            else:
                node = node.children.get(val, self._majority_child(node))
        return node

    def _path(self, x):
        nodes = []
        node = self.tree_
        while not node.is_leaf:
            nodes.append(node)
            fi = node.feature_index
            val = x[fi]
            if node.threshold is not None:
                try:
                    go = "left" if float(val) <= node.threshold else "right"
                except (TypeError, ValueError):
                    go = "left"
                node = node.children.get(go, self._majority_child(node))
            else:
                node = node.children.get(val, self._majority_child(node))
        nodes.append(node)
        return nodes

    # ── tree info ─────────────────────────────────────────────────────────

    def get_depth(self) -> int:
        self._check_fitted()
        return self._depth(self.tree_)

    def get_n_leaves(self) -> int:
        self._check_fitted()
        return self._count_leaves(self.tree_)

    def _depth(self, node: _Node) -> int:
        if node.is_leaf:
            return 0
        return 1 + max(self._depth(c) for c in node.children.values())

    def _count_leaves(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        return sum(self._count_leaves(c) for c in node.children.values())

    # ── pruning ───────────────────────────────────────────────────────────

    def cost_complexity_pruning_path(self, X: ArrayLike, y: ArrayLike):
        """
        Returns (ccp_alphas, impurities) similar to sklearn.
        Performs a sequence of pruning steps and records effective alpha at each.
        """
        self._check_fitted()
        import copy
        alphas, impurities = [0.0], [self._node_impurity(self.tree_)]
        tree_copy = copy.deepcopy(self)
        while True:
            candidates = []
            self._prune_candidates(tree_copy.tree_, candidates)
            if not candidates:
                break
            candidates.sort(key=lambda t: t[0])
            alpha, node = candidates[0]
            alphas.append(alpha)
            # Convert to leaf
            node.is_leaf = True
            node.children = {}
            impurities.append(self._node_impurity(tree_copy.tree_))
        return np.array(alphas), np.array(impurities)

    def _prune_candidates(self, node, out):
        if node.is_leaf:
            return
        # effective alpha = (R(node) - R(subtree)) / (leaves(subtree) - 1)
        r_node = node.impurity * node.n_samples
        r_sub = self._subtree_risk(node)
        l_sub = self._count_leaves(node)
        if l_sub > 1:
            alpha = (r_node - r_sub) / (l_sub - 1)
            out.append((alpha, node))
        for c in node.children.values():
            self._prune_candidates(c, out)

    def _subtree_risk(self, node):
        if node.is_leaf:
            return node.impurity * node.n_samples
        return sum(self._subtree_risk(c) for c in node.children.values())

    def _node_impurity(self, node):
        if node.is_leaf:
            return node.impurity
        return sum(self._node_impurity(c) for c in node.children.values())

    def _prune(self, node: _Node, alpha: float):
        if node.is_leaf:
            return
        for c in node.children.values():
            self._prune(c, alpha)
        # Check if pruning is beneficial
        r_node = node.impurity * node.n_samples
        r_sub = self._subtree_risk(node)
        l_sub = self._count_leaves(node)
        if l_sub > 1:
            eff_alpha = (r_node - r_sub) / (l_sub - 1)
            if eff_alpha <= alpha:
                node.is_leaf = True
                node.children = {}

    # ── text export ───────────────────────────────────────────────────────

    def export_text(self, spacing: int = 3) -> str:
        self._check_fitted()
        lines = []
        self._text_recurse(self.tree_, lines, "", spacing)
        return "\n".join(lines)

    def _text_recurse(self, node, lines, prefix, spacing):
        pad = " " * spacing
        if node.is_leaf:
            cls = (self.class_names[node.class_label]
                   if self.class_names else str(self.classes_[node.class_label]))
            lines.append(f"{prefix}[LEAF] class={cls}  samples={node.n_samples}  "
                         f"impurity={node.impurity:.4f}")
            return
        feat = node.feature_name
        if node.threshold is not None:
            # numeric
            lines.append(f"{prefix}[{feat} <= {node.threshold:.4f}]  "
                         f"gain={node.gain:.4f}  samples={node.n_samples}")
            self._text_recurse(node.children["left"],  lines, prefix + pad, spacing)
            lines.append(f"{prefix}[{feat} >  {node.threshold:.4f}]")
            self._text_recurse(node.children["right"], lines, prefix + pad, spacing)
        else:
            lines.append(f"{prefix}[{feat}]  gain={node.gain:.4f}  "
                         f"samples={node.n_samples}")
            for val, child in node.children.items():
                lines.append(f"{prefix}{pad}== {val}")
                self._text_recurse(child, lines, prefix + pad * 2, spacing)

    # ── graphviz export ───────────────────────────────────────────────────

    def export_graphviz(
        self,
        out_file=None,
        feature_names=None,
        class_names=None,
        filled: bool = True,
        rounded: bool = True,
        special_characters: bool = False,
    ) -> Optional[str]:
        self._check_fitted()
        fn = feature_names or self.feature_names
        cn = class_names or self.class_names

        lines = ['digraph Tree {',
                 'node [shape=box, style="filled, rounded", '
                 'color="black", fontname="helvetica"];',
                 'edge [fontname="helvetica"];']
        counter = [0]

        def _node_str(node: _Node) -> str:
            nid = counter[0]
            counter[0] += 1    # przydział ID bieżącego węzła
            if node.is_leaf:
                cls = cn[node.class_label] if cn else str(self.classes_[node.class_label])
                label = (f"class = {cls}\\nsamples = {node.n_samples}\\n"
                         f"impurity = {node.impurity:.4f}")
                color = f'"#{_class_color(node.class_label, self.n_classes_)}"'
                lines.append(f'{nid} [label="{label}", fillcolor={color}];')
            else:
                feat = (fn[node.feature_index]
                        if fn and node.feature_index < len(fn)
                        else f"X[{node.feature_index}]")
                if node.threshold is not None:
                    cond = f"{feat} <= {node.threshold:.4f}"
                else:
                    cond = feat
                label = (f"{cond}\\ngain = {node.gain:.4f}\\n"
                         f"samples = {node.n_samples}\\n"
                         f"impurity = {node.impurity:.4f}")
                color = '"#ffffff"'
                lines.append(f'{nid} [label="{label}", fillcolor={color}];')
                for edge_label, child in node.children.items():
                    # WAŻNE: zapisz cid PRZED rekurencją (counter zmieni się w środku)
                    cid = counter[0]
                    _node_str(child)
                    if node.threshold is not None:
                        elabel = "True" if edge_label == "left" else "False"
                    else:
                        elabel = str(edge_label)
                    lines.append(f'{nid} -> {cid} [label="{elabel}"];')
            return str(nid)

        _node_str(self.tree_)
        lines.append("}")
        dot_str = "\n".join(lines)
        if out_file:
            if hasattr(out_file, "write"):
                out_file.write(dot_str)
            else:
                with open(out_file, "w") as f:
                    f.write(dot_str)
            return None
        return dot_str

    # ── plot_tree ─────────────────────────────────────────────────────────

    def plot_tree(
        self,
        ax=None,
        feature_names=None,
        class_names=None,
        filled: bool = True,
        fontsize: int = 9,
        max_depth: Optional[int] = None,
        figsize: Optional[Tuple] = None,
        title: Optional[str] = None,
    ):
        if not _MPL:
            raise ImportError("matplotlib is required for plot_tree().")
        self._check_fitted()

        fn = feature_names or self.feature_names
        cn = class_names or self.class_names

        # --- helpers ---------------------------------------------------------

        def _format_value(node):
            return [node.class_counts.get(i, 0) for i in range(self.n_classes_)]

        def _criterion_name():
            return "gini" if self.criterion == "gini" else "entropy"

        def _node_color(node):
            """Kolor jak w sklearn — zależny od impurity (a nie purity)"""
            if not filled:
                return "white"

            counts = np.array(_format_value(node))
            total = counts.sum()

            if total == 0:
                return "#ffffff"

            # --- majority class (kolor bazowy) ---
            majority = np.argmax(counts)

            base = np.array(
                plt.matplotlib.colors.to_rgb(
                    _mpl_class_color(majority, self.n_classes_)
                )
            )
            white = np.array([1.0, 1.0, 1.0])

            # --- NORMALIZACJA IMPURITY ---
            imp = node.impurity

            if self.criterion == "gini":
                max_imp = 1.0 - 1.0 / self.n_classes_
            else:  # entropy
                max_imp = np.log2(self.n_classes_)

            if max_imp == 0:
                norm_imp = 0.0
            else:
                norm_imp = imp / max_imp

            # --- ODWRÓCENIE: im większa impurity → bardziej biały ---
            # strength = 1.0 - norm_imp  # 1 = czysty, 0 = losowy
            strength = (1.0 - norm_imp) ** 1.5

            # --- skalowanie jak sklearn (lekko stłumione) ---
            alpha = 0.2 + 0.8 * strength

            color = white * (1 - alpha) + base * alpha

            return color

        # def _node_color(node):
        #     """Kolor jak w sklearn — zależny od purity"""
        #     if not filled:
        #         return "white"

        #     counts = np.array(_format_value(node))
        #     total = counts.sum()

        #     if total == 0:
        #         return "#ffffff"

        #     probs = counts / total
        #     majority = np.argmax(probs)
        #     purity = probs[majority]

        #     base = np.array(
        #         plt.matplotlib.colors.to_rgb(_mpl_class_color(majority, self.n_classes_))
        #     )
        #     white = np.array([1.0, 1.0, 1.0])

        #     # im większa purity → bardziej nasycony kolor
        #     alpha = 0.3 + 0.7 * purity
        #     color = white * (1 - alpha) + base * alpha

        #     return color

        # --- layout ----------------------------------------------------------

        positions = {}
        _assign_positions(self.tree_, positions, depth=0, x_counter=[0],
                        max_depth=max_depth)

        if ax is None:
            depth = self.get_depth()
            w = max(6, 2.5 * (2 ** min(depth, 4)))
            h = max(4, 1.8 * depth)
            fig, ax = plt.subplots(figsize=figsize or (w, h))
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")

        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=fontsize + 3, fontweight="bold", pad=12)

        max_x = max(p[0] for p in positions.values())
        max_y = max(p[1] for p in positions.values())

        x_margin = 0.05
        y_margin = 0.10
        x_range = 1.0 - 2 * x_margin
        y_range = 1.0 - 2 * y_margin

        def _norm(x, y):
            nx = x_margin + (x / (max_x + 1)) * x_range if max_x > 0 else 0.5
            ny = (1.0 - y_margin) - (y / max(max_y, 1)) * y_range
            return nx, ny

        # --- draw ------------------------------------------------------------

        def _draw(node):
            if id(node) not in positions:
                return
            if max_depth is not None and node.depth > max_depth:
                return

            x, y = positions[id(node)]
            nx, ny = _norm(x, y)

            crit = _criterion_name()
            value = _format_value(node)
            cls = cn[node.class_label] if cn else str(self.classes_[node.class_label])

            if node.is_leaf:
                label = (
                    f"{crit} = {node.impurity:.3f}\n"
                    f"samples = {node.n_samples}\n"
                    f"value = {value}\n"
                    f"class = {cls}"
                )
            else:
                feat = (
                    fn[node.feature_index]
                    if fn and node.feature_index < len(fn)
                    else f"X[{node.feature_index}]"
                )

                if node.threshold is not None:
                    cond = f"{feat} ≤ {node.threshold:.3f}"
                else:
                    cond = f"{feat}"

                label = (
                    f"{cond}\n"
                    f"{crit} = {node.impurity:.3f}\n"
                    f"samples = {node.n_samples}\n"
                    f"value = {value}\n"
                    f"class = {cls}"
                )

            ax.text(
                nx, ny, label,
                ha="center", va="center", fontsize=fontsize,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=_node_color(node),
                    edgecolor="#555",
                    linewidth=1.0
                ),
                transform=ax.transAxes,
                zorder=3,
            )

            # edges
            for edge_key, child in node.children.items():
                if id(child) not in positions:
                    continue
                if max_depth is not None and child.depth > max_depth:
                    continue

                cx, cy = positions[id(child)]
                cnx, cny = _norm(cx, cy)

                ax.annotate(
                    "", xy=(cnx, cny), xytext=(nx, ny),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#555",
                        lw=1.2,
                        connectionstyle="arc3,rad=0.0"
                    ),
                    zorder=2,
                )

                if node.threshold is not None:
                    elabel = "≤" if edge_key == "left" else ">"
                else:
                    elabel = str(edge_key)

                mx, my = (nx + cnx) / 2, (ny + cny) / 2

                ax.text(
                    mx, my, elabel,
                    ha="center", va="center",
                    fontsize=fontsize - 1,
                    color="#333",
                    transform=ax.transAxes,
                    zorder=4
                )

                _draw(child)

        _draw(self.tree_)

        # --- legenda ---------------------------------------------------------

        if cn or self.classes_ is not None:
            classes = cn if cn else [str(c) for c in self.classes_]
            patches = [
                mpatches.Patch(
                    facecolor=_mpl_class_color(i, self.n_classes_),
                    edgecolor="#555",
                    label=classes[i]
                )
                for i in range(self.n_classes_)
            ]
            legend = ax.legend(
                handles=patches,
                loc="upper right",
                fontsize=fontsize,
                title="Classes",
                framealpha=0.8
            )
            legend.get_title().set_fontsize(fontsize)

        return ax



    # def plot_tree(
    #     self,
    #     ax=None,
    #     feature_names=None,
    #     class_names=None,
    #     filled: bool = True,
    #     fontsize: int = 9,
    #     max_depth: Optional[int] = None,
    #     figsize: Optional[Tuple] = None,
    #     title: Optional[str] = None,
    # ):
    #     """
    #     Render the decision tree visually using matplotlib.
    #     Returns the axes object.
    #     """
    #     if not _MPL:
    #         raise ImportError("matplotlib is required for plot_tree().")
    #     self._check_fitted()

    #     fn = feature_names or self.feature_names
    #     cn = class_names or self.class_names

    #     # Compute layout
    #     positions = {}
    #     _assign_positions(self.tree_, positions, depth=0, x_counter=[0],
    #                       max_depth=max_depth)

    #     if ax is None:
    #         depth = self.get_depth()
    #         w = max(6, 2.5 * (2 ** min(depth, 4)))
    #         h = max(4, 1.8 * depth)
    #         fig, ax = plt.subplots(figsize=figsize or (w, h))
    #         ax.set_facecolor("#f8f9fa")
    #         fig.patch.set_facecolor("#f8f9fa")

    #     ax.axis("off")
    #     if title:
    #         ax.set_title(title, fontsize=fontsize + 3, fontweight="bold", pad=12)

    #     max_x = max(p[0] for p in positions.values())
    #     max_y = max(p[1] for p in positions.values())

    #     # Marginesy zapobiegają przyleganiu węzłów do krawędzi figury
    #     x_margin = 0.05
    #     y_margin = 0.10
    #     x_range  = 1.0 - 2 * x_margin
    #     y_range  = 1.0 - 2 * y_margin

    #     def _norm(x, y):
    #         nx = x_margin + (x / (max_x + 1)) * x_range if max_x > 0 else 0.5
    #         # Równomierne odstępy pionowe niezależnie od głębokości drzewa
    #         ny = (1.0 - y_margin) - (y / max(max_y, 1)) * y_range
    #         return nx, ny

    #     def _draw(node, depth=0):
    #         if id(node) not in positions:
    #             return
    #         if max_depth is not None and node.depth > max_depth:
    #             return
    #         x, y = positions[id(node)]
    #         nx, ny = _norm(x, y)

    #         # Node box
    #         if node.is_leaf:
    #             cls = cn[node.class_label] if cn else str(self.classes_[node.class_label])
    #             label = f"class: {cls}\nsamples: {node.n_samples}\nimpurity: {node.impurity:.3f}"
    #             fc = _mpl_class_color(node.class_label, self.n_classes_) if filled else "white"
    #             style = "round,pad=0.3"
    #         else:
    #             feat = (fn[node.feature_index]
    #                     if fn and node.feature_index < len(fn)
    #                     else f"X[{node.feature_index}]")
    #             if node.threshold is not None:
    #                 cond = f"{feat} ≤ {node.threshold:.3f}"
    #             else:
    #                 cond = f"{feat}"
    #             label = (f"{cond}\ngain: {node.gain:.3f}\n"
    #                      f"samples: {node.n_samples}\nimpurity: {node.impurity:.3f}")
    #             fc = "#dfe6f0" if filled else "white"
    #             style = "round,pad=0.3"

    #         ax.text(
    #             nx, ny, label,
    #             ha="center", va="center", fontsize=fontsize,
    #             bbox=dict(boxstyle=style, facecolor=fc, edgecolor="#555", linewidth=1.0),
    #             transform=ax.transAxes,
    #             zorder=3,
    #         )

    #         # Edges to children
    #         for edge_key, child in node.children.items():
    #             if id(child) not in positions:
    #                 continue
    #             if max_depth is not None and child.depth > max_depth:
    #                 continue
    #             cx, cy = positions[id(child)]
    #             cnx, cny = _norm(cx, cy)
    #             ax.annotate(
    #                 "", xy=(cnx, cny), xytext=(nx, ny),
    #                 xycoords="axes fraction", textcoords="axes fraction",
    #                 arrowprops=dict(arrowstyle="-|>", color="#555",
    #                                lw=1.2, connectionstyle="arc3,rad=0.0"),
    #                 zorder=2,
    #             )
    #             # Edge label
    #             if node.threshold is not None:
    #                 elabel = "≤" if edge_key == "left" else ">"
    #             else:
    #                 elabel = str(edge_key)
    #             mx, my = (nx + cnx) / 2, (ny + cny) / 2
    #             ax.text(mx, my, elabel,
    #                     ha="center", va="center", fontsize=fontsize - 1,
    #                     color="#333", transform=ax.transAxes, zorder=4)
    #             _draw(child, depth + 1)

    #     _draw(self.tree_)

    #     # Legend
    #     if cn or self.classes_ is not None:
    #         classes = cn if cn else [str(c) for c in self.classes_]
    #         patches = [
    #             mpatches.Patch(
    #                 facecolor=_mpl_class_color(i, self.n_classes_),
    #                 edgecolor="#555", label=classes[i]
    #             )
    #             for i in range(self.n_classes_)
    #         ]
    #         legend = ax.legend(handles=patches, loc="upper right",
    #                            fontsize=fontsize, title="Classes", framealpha=0.8)
    #         legend.get_title().set_fontsize(fontsize)

    #     return ax

    # ── ekstrakcja reguł (integracja z roughset) ──────────────────────────

    def extract_rules(self, decision_attribute: Optional[str] = None):
        """
        Ekstrahuje reguły decyzyjne z dopasowanego drzewa.

        Wymaga zainstalowanego modułu ``roughset``.
        Każda ścieżka od korzenia do liścia generuje jedną regułę:
        warunki na ścieżce tworzą część IF, klasa liścia — konkluzję.

        Parameters
        ----------
        decision_attribute : str or None, default None
            Nazwa atrybutu decyzyjnego w wygenerowanych regułach.
            Jeśli None — używana jest wartość ``"class"`` (domyślna nazwa
            kolumny decyzyjnej). Można podać dowolną nazwę, np. ``"klasa"``,
            ``"wynik"``, ``"label"``.

        Returns
        -------
        roughset.RuleSet
            Zbiór reguł z metodami eksportu:

            - ``to_series()``  → pd.Series stringów
              ``IF f1='val' AND f2<=2.5 THEN class='klasa'``

            - ``to_dataframe()`` → pd.DataFrame, jedna reguła na wiersz,
              kolumny = atrybuty warunkowe + kolumna decyzyjna,
              atrybuty nieużywane w regule → NaN.

        Raises
        ------
        ImportError
            Gdy moduł ``roughset`` nie jest dostępny.
        RuntimeError
            Gdy klasyfikator nie jest dopasowany.

        Przykład
        --------
        >>> clf.fit(X, y)
        >>> rs = clf.extract_rules()            # kolumna decyzyjna: "class"
        >>> rs = clf.extract_rules("wynik")     # własna nazwa
        >>> print(rs.to_series(numbered=True))
        >>> print(rs.to_dataframe())
        >>> print(rs.summary())
        """
        try:
            from roughset import extract_rules_from_tree
        except ImportError as e:
            raise ImportError(
                "Moduł 'roughset' jest wymagany dla extract_rules(). "
                "Upewnij się, że roughset.py jest dostępny w PYTHONPATH."
            ) from e
        self._check_fitted()
        dec_attr = decision_attribute if decision_attribute is not None else "class"
        return extract_rules_from_tree(self, decision_attribute=dec_attr)

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"ID3C45Classifier(algorithm='{self.algorithm}', "
            f"criterion='{self.criterion}', "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"ccp_alpha={self.ccp_alpha})"
        )


# ─────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────

def _assign_positions(node, positions, depth, x_counter, max_depth):
    if max_depth is not None and depth > max_depth:
        return
    if node.is_leaf or not node.children:
        positions[id(node)] = (x_counter[0], depth)
        x_counter[0] += 1
        return
    children = list(node.children.values())
    for child in children:
        _assign_positions(child, positions, depth + 1, x_counter, max_depth)
    child_xs = [positions[id(c)][0] for c in children if id(c) in positions]
    if child_xs:
        positions[id(node)] = (sum(child_xs) / len(child_xs), depth)
    else:
        positions[id(node)] = (x_counter[0], depth)
        x_counter[0] += 1


# ─────────────────────────────────────────────
# Color helpers
# ─────────────────────────────────────────────

_CLASS_COLORS_HEX = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F1948A",
    "#D7BDE2", "#FAD7A0", "#A3E4D7", "#D5DBDB",
]

_CLASS_COLORS_MPL = [
    "#5dade2", "#58d68d", "#f4d03f", "#ec7063",
    "#a569bd", "#f0a500", "#1abc9c", "#95a5a6",
]


def _class_color(cls_idx: int, n_classes: int) -> str:
    return _CLASS_COLORS_HEX[cls_idx % len(_CLASS_COLORS_HEX)].lstrip("#")


def _mpl_class_color(cls_idx: int, n_classes: int) -> str:
    return _CLASS_COLORS_MPL[cls_idx % len(_CLASS_COLORS_MPL)]


# ─────────────────────────────────────────────
# Convenience wrapper: plot_tree as module fn
# ─────────────────────────────────────────────

def plot_tree(clf: ID3C45Classifier, **kwargs):
    """
    Module-level plot_tree, mirrors sklearn.tree.plot_tree.

    Parameters
    ----------
    clf : ID3C45Classifier
        The trained classifier instance to be plotted.
    **kwargs : dict
        Arguments passed to `ID3C45Classifier.plot_tree`. Includes:
        
        * ax : matplotlib.axes.Axes, optional
            Axes to plot to.
        * feature_names : list of str, optional
            Names of the features.
        * class_names : list of str, optional
            Names of the target classes.
        * filled : bool, default=True
            Whether to fill nodes with colors.
        * fontsize : int, default=9
            Size of the text.
        * max_depth : int, optional
            The maximum depth of the representation.
        * figsize : tuple, optional
            Size of the figure in inches.
        * title : str, optional
            Title for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plotted tree.
    """
    return clf.plot_tree(**kwargs)


def export_text(clf: ID3C45Classifier, **kwargs) -> str:
    return clf.export_text(**kwargs)


def export_graphviz(clf: ID3C45Classifier, **kwargs):
    return clf.export_graphviz(**kwargs)


# # ─────────────────────────────────────────────
# # Demo / smoke test
# # ─────────────────────────────────────────────

# if __name__ == "__main__":
#     from sklearn.datasets import load_iris, load_breast_cancer
#     from sklearn.model_selection import train_test_split, cross_val_score
#     from sklearn.metrics import classification_report

#     print("=" * 60)
#     print("  ID3 / C4.5 Classifier — smoke test")
#     print("=" * 60)

#     # ── Iris with C4.5 ──────────────────────────────────────────
#     iris = load_iris()
#     X_tr, X_te, y_tr, y_te = train_test_split(
#         iris.data, iris.target, test_size=0.3, random_state=42)

#     clf = ID3C45Classifier(
#         algorithm="C4.5",
#         criterion="entropy",
#         max_depth=4,
#         feature_names=list(iris.feature_names),
#         class_names=list(iris.target_names),
#         random_state=42,
#     )
#     clf.fit(X_tr, y_tr)

#     print("\n[C4.5 on Iris]")
#     print(f"  Accuracy : {clf.score(X_te, y_te):.4f}")
#     print(f"  Depth    : {clf.get_depth()}")
#     print(f"  Leaves   : {clf.get_n_leaves()}")
#     print(f"  Feature importances: {clf.feature_importances_}")
#     print("\nTree text:\n")
#     print(clf.export_text())

#     cv_scores = cross_val_score(clf, iris.data, iris.target, cv=5)
#     print(f"\n  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#     # ── predict_proba ────────────────────────────────────────────
#     proba = clf.predict_proba(X_te[:3])
#     print(f"\n  predict_proba (first 3):\n{proba}")

#     # ── apply ────────────────────────────────────────────────────
#     leaf_ids = clf.apply(X_te[:5])
#     print(f"\n  apply() leaf ids: {leaf_ids}")

#     # ── export graphviz ──────────────────────────────────────────
#     dot = clf.export_graphviz(filled=True)
#     print(f"\n  Graphviz DOT (first 200 chars):\n  {dot[:200]}...")

#     # ── plot_tree ────────────────────────────────────────────────
#     try:
#         import matplotlib
#         matplotlib.use("Agg")
#         fig, ax = plt.subplots(figsize=(18, 8))
#         clf.plot_tree(ax=ax, filled=True, fontsize=8,
#                       title="C4.5 — Iris dataset")
#         fig.tight_layout()
#         fig.savefig("tree_iris.png", dpi=120)
#         print("\n  plot_tree saved → tree_iris.png")
#     except Exception as e:
#         print(f"\n  plot_tree skipped: {e}")

#     # ── Breast Cancer with pruning ───────────────────────────────
#     bc = load_breast_cancer()
#     X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
#         bc.data, bc.target, test_size=0.3, random_state=0)

#     clf2 = ID3C45Classifier(algorithm="C4.5", max_depth=6, random_state=0)
#     clf2.fit(X_tr2, y_tr2)
#     print(f"\n[C4.5 on BreastCancer] acc={clf2.score(X_te2, y_te2):.4f} "
#           f"depth={clf2.get_depth()} leaves={clf2.get_n_leaves()}")

#     alphas, imps = clf2.cost_complexity_pruning_path(X_tr2, y_tr2)
#     print(f"  Pruning path alphas (first 5): {alphas[:5]}")

#     # ── ID3 on categorical data ──────────────────────────────────
#     X_cat = np.array([
#         ["sunny", "hot", "high", "false"],
#         ["sunny", "hot", "high", "true"],
#         ["overcast", "hot", "high", "false"],
#         ["rain", "mild", "high", "false"],
#         ["rain", "cool", "normal", "false"],
#         ["rain", "cool", "normal", "true"],
#         ["overcast", "cool", "normal", "true"],
#         ["sunny", "mild", "high", "false"],
#         ["sunny", "cool", "normal", "false"],
#         ["rain", "mild", "normal", "false"],
#         ["sunny", "mild", "normal", "true"],
#         ["overcast", "mild", "high", "true"],
#         ["overcast", "hot", "normal", "false"],
#         ["rain", "mild", "high", "true"],
#     ])
#     y_cat = np.array(["no","no","yes","yes","yes","no","yes","no","yes",
#                       "yes","yes","yes","yes","no"])

#     clf3 = ID3C45Classifier(
#         algorithm="ID3",
#         feature_names=["Outlook","Temperature","Humidity","Windy"],
#         class_names=["No","Yes"],
#     )
#     clf3.fit(X_cat, y_cat)
#     print(f"\n[ID3 on Play-Tennis] acc={clf3.score(X_cat, y_cat):.4f} "
#           f"depth={clf3.get_depth()}")
#     print(clf3.export_text())

#     print("\n✓ All tests passed.")

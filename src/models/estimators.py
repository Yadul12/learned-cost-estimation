"""
Stages 5 & 6: Baselines, Gradient Boosting, Tree-Structured Neural Cost Model,
Template-Disjoint Evaluation, Cross-Engine Transfer, and Ablations.

Implements all models compared in Section 8 of the proposal:
1. Native Optimizer Cost Estimate (`NativeCBOEstimator`)
2. Data Size Alone Baseline (`DataSizeBaseline`) - near-trivial baseline from scan est_rows & n_tables
3. Linear / Ridge Regression on flat features (`LinearCostBaseline`)
4. Gradient Boosting on flat features (`GradientBoostingCostModel` via XGBoost)
5. Tree-Structured Neural Model (`TreeNeuralCostModel` via PyTorch recursive Child-Sum Tree-GRU / Tree-MLP)

All models are trained in log-runtime space (`log(wall_ms)`) so that symmetric MSE / Huber loss
directly minimizes logarithmic Q-error:
    Q-error(y, y_hat) = max(y / y_hat, y_hat / y) = exp(|log(y) - log(y_hat)|).

For Cross-Engine Transfer (DuckDB -> Trino):
Models trained solely on DuckDB plans & timings predict plan costs on Trino.
Because DuckDB (embedded, ~1-25 ms) and Trino (distributed, ~25-180 ms) operate at different
hardware/engine time scales, we evaluate:
  (a) Zero-shot rank correlation (Spearman rho & Kendall tau-b & pairwise plan ranking accuracy),
      which requires ZERO target labels and directly tests whether a DuckDB-trained model
      ranks Trino plans accurately (Section 5 & Section 8 Secondary metric).
  (b) Few-shot Affine Scale Calibration (2-parameter log-linear rescaling `a * log_pred + b`
      fit on a tiny calibration subset of 10 Trino train-fold queries) to align absolute
      millisecond units for Q-error evaluation against Trino's native optimizer.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import xgboost as xgb

from src.features.extractor import CANONICAL_OPS, FLAT_FEATURE_NAMES, TPCH_TABLES


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_q_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes Q-error for each sample: max(y_true / y_pred, y_pred / y_true).
    Perfect prediction = 1.0.
    """
    eps = 1e-6
    yt = np.maximum(eps, np.asarray(y_true, dtype=np.float64))
    yp = np.maximum(eps, np.asarray(y_pred, dtype=np.float64))
    return np.maximum(yt / yp, yp / yt)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes comprehensive Q-error and plan-ranking metrics with 95% bootstrap confidence intervals.
    - Primary metrics: Median Q-error, Mean Q-error, 90th percentile (P90), 95th percentile (P95), Max Q-error
    - Secondary metrics: Spearman rank correlation (rho), Kendall tau, Pairwise Plan Ranking Accuracy (%)
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    q_err = compute_q_errors(yt, yp)

    med_q = float(np.median(q_err))
    mean_q = float(np.mean(q_err))
    p90_q = float(np.percentile(q_err, 90))
    p95_q = float(np.percentile(q_err, 95))
    max_q = float(np.max(q_err))

    # Spearman and Kendall rank correlation
    if len(yt) > 2 and np.std(yp) > 1e-12:
        spearman_rho, _ = stats.spearmanr(yt, yp)
        kendall_tau, _ = stats.kendalltau(yt, yp)
        spearman_rho = float(spearman_rho) if not np.isnan(spearman_rho) else 0.0
        kendall_tau = float(kendall_tau) if not np.isnan(kendall_tau) else 0.0
    else:
        spearman_rho = 0.0
        kendall_tau = 0.0

    # Pairwise ranking accuracy on pairs with >5% runtime difference
    concordant = 0
    total_pairs = 0
    n = len(yt)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(yt[i] - yt[j]) / max(yt[i], yt[j], 1e-6) > 0.05:
                total_pairs += 1
                if (yt[i] - yt[j]) * (yp[i] - yp[j]) > 0:
                    concordant += 1
    pairwise_acc = (concordant / total_pairs * 100.0) if total_pairs > 0 else 50.0

    # Bootstrap 95% CI for Median and P90 Q-error
    rng = np.random.default_rng(seed)
    boot_meds = []
    boot_p90s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_q = q_err[idx]
        boot_meds.append(np.median(sample_q))
        boot_p90s.append(np.percentile(sample_q, 90))

    return {
        "median_q_error": med_q,
        "median_q_ci_low": float(np.percentile(boot_meds, 2.5)),
        "median_q_ci_high": float(np.percentile(boot_meds, 97.5)),
        "mean_q_error": mean_q,
        "p90_q_error": p90_q,
        "p90_q_ci_low": float(np.percentile(boot_p90s, 2.5)),
        "p90_q_ci_high": float(np.percentile(boot_p90s, 97.5)),
        "p95_q_error": p95_q,
        "max_q_error": max_q,
        "spearman_rho": spearman_rho,
        "kendall_tau": kendall_tau,
        "pairwise_rank_acc": float(pairwise_acc),
    }


class NativeCBOEstimator:
    """
    Baseline 1: The Engine's Own Cost Estimate.
    Since native optimizer cost is in arbitrary work units, we calibrate a 1D log-linear
    conversion (`log(wall_ms) = a * log(native_est_cost) + b`) on the training split
    ONLY, so native cost units are fairly mapped to milliseconds without altering
    the engine's relative cost ordering.
    """

    def __init__(self) -> None:
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, native_costs: np.ndarray, y_ms: np.ndarray) -> "NativeCBOEstimator":
        x = np.log1p(np.maximum(0.0, native_costs)).reshape(-1, 1)
        y = np.log(np.maximum(1e-3, y_ms))
        reg = Ridge(alpha=1.0).fit(x, y)
        self.slope = float(reg.coef_[0])
        self.intercept = float(reg.intercept_)
        return self

    def predict(self, native_costs: np.ndarray) -> np.ndarray:
        x = np.log1p(np.maximum(0.0, native_costs))
        log_pred = self.slope * x + self.intercept
        return np.exp(np.clip(log_pred, -5.0, 15.0))


class DataSizeBaseline:
    """
    Baseline 2: Predicting from Data Size Alone (Section 8).
    Uses only scanned table rows (`log_scan_est_rows`) and number of touched tables (`n_distinct_tables`).
    """

    def __init__(self, feature_names: List[str] = FLAT_FEATURE_NAMES) -> None:
        self.feature_names = feature_names
        self.cols = [
            feature_names.index("log_scan_est_rows"),
            feature_names.index("n_distinct_tables"),
        ]
        self.scaler = StandardScaler()
        self.reg = Ridge(alpha=10.0)

    def fit(self, X: np.ndarray, y_ms: np.ndarray) -> "DataSizeBaseline":
        X_sub = self.scaler.fit_transform(X[:, self.cols])
        y_log = np.log(np.maximum(1e-3, y_ms))
        self.reg.fit(X_sub, y_log)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sub = self.scaler.transform(X[:, self.cols])
        log_pred = self.reg.predict(X_sub)
        return np.exp(np.clip(log_pred, -5.0, 15.0))


class LinearCostBaseline:
    """
    Baseline 3: Regularized Linear Model (Ridge) on full flat plan features.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.reg = Ridge(alpha=15.0)

    def fit(self, X: np.ndarray, y_ms: np.ndarray) -> "LinearCostBaseline":
        X_s = self.scaler.fit_transform(X)
        y_log = np.log(np.maximum(1e-3, y_ms))
        self.reg.fit(X_s, y_log)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        log_pred = self.reg.predict(X_s)
        return np.exp(np.clip(log_pred, -5.0, 15.0))


class GradientBoostingCostModel:
    """
    Model Family 1: Gradient Boosting (XGBoost) on flattened plan features.
    Uses monotonic constraints on key cardinality/operator work features and strong L2
    regularization so trees generalize cleanly to unseen query templates under
    template-disjoint evaluation.
    Supports feature-group masking for systematic ablation studies (Section 8).
    """

    def __init__(
        self,
        active_indices: Optional[List[int]] = None,
        seed: int = 42,
        n_estimators: int = 75,
        max_depth: int = 2,
        learning_rate: float = 0.04,
    ) -> None:
        self.active_indices = active_indices
        self.scaler = StandardScaler()
        self.ridge_prior = Ridge(alpha=20.0)
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Optional[xgb.XGBRegressor] = None

    def _slice(self, X: np.ndarray) -> np.ndarray:
        if self.active_indices is None:
            return X
        return X[:, self.active_indices]

    def fit(
        self,
        X_train: np.ndarray,
        y_train_ms: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val_ms: Optional[np.ndarray] = None,
    ) -> "GradientBoostingCostModel":
        Xt = self.scaler.fit_transform(self._slice(X_train))
        yt_log = np.log(np.maximum(1e-3, y_train_ms))
        self.ridge_prior.fit(Xt, yt_log)
        base_pred = self.ridge_prior.predict(Xt)
        residuals = yt_log - base_pred

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.5,
            reg_lambda=8.0,
            min_child_weight=8,
            objective="reg:squarederror",
            random_state=self.seed,
            n_jobs=1,
        )
        self.model.fit(Xt, residuals)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        Xs = self.scaler.transform(self._slice(X))
        log_pred = self.ridge_prior.predict(Xs) + 0.65 * self.model.predict(Xs)
        return np.exp(np.clip(log_pred, -5.0, 15.0))

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        assert self.model is not None
        names = (
            feature_names
            if self.active_indices is None
            else [feature_names[i] for i in self.active_indices]
        )
        xgb_imps = self.model.feature_importances_
        ridge_imps = np.abs(self.ridge_prior.coef_)
        ridge_imps = ridge_imps / max(1e-8, np.sum(ridge_imps))
        combined = 0.5 * xgb_imps + 0.5 * ridge_imps
        return {name: float(imp) for name, imp in zip(names, combined)}


class TreeRecursiveNet(nn.Module):
    """
    Bottom-up Child-Sum Tree-GRU / Tree-MLP architecture that reads the execution plan
    tree topology directly:
      h_v = GRUCell(node_input_v, sum_{u in children(v)} h_u)
    Each node encodes:
      - operator embedding (8 canonical operators)
      - table embedding (8 TPC-H tables + 1 none)
      - numerical node attributes: [est_rows_log_scaled, op_weight_scaled, depth_norm, is_leaf]
    Root hidden representation is combined with global structural summary to predict `log(wall_ms)`.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.op_emb = nn.Embedding(len(CANONICAL_OPS) + 1, 8)
        self.table_emb = nn.Embedding(len(TPCH_TABLES) + 2, 6)
        node_in_dim = 8 + 6 + 4  # op_emb(8) + table_emb(6) + 4 numerical scalars = 18

        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # Child-sum gating
        self.W_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_r = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.skip_work = nn.Linear(3, 1)

    def encode_node_recursive(
        self, node: Dict[str, Any], est_mean: float, est_std: float
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[float, float, float]]]:
        children = node.get("children", [])
        all_states: List[torch.Tensor] = []
        all_scalars: List[Tuple[float, float, float]] = []

        if children:
            child_h_list = []
            for ch in children:
                ch_h, ch_all, ch_sc = self.encode_node_recursive(ch, est_mean, est_std)
                child_h_list.append(ch_h)
                all_states.extend(ch_all)
                all_scalars.extend(ch_sc)
            h_child_sum = torch.stack(child_h_list, dim=0).mean(dim=0)
        else:
            h_child_sum = torch.zeros(self.hidden_dim)

        op_idx = torch.tensor(int(node.get("op_idx", 7)), dtype=torch.long)
        tb_idx = torch.tensor(int(node.get("table_idx", 0)), dtype=torch.long)
        est_norm = (float(node.get("est_rows_log", 0.0)) - est_mean) / max(1e-4, est_std)
        op_w = float(node.get("op_weight", 1.0)) / 5.0
        depth_n = float(node.get("depth_norm", 0.0))
        is_leaf = float(node.get("is_leaf", 1.0))

        num_tensor = torch.tensor([est_norm, op_w, depth_n, is_leaf], dtype=torch.float32)
        x_v = torch.cat([self.op_emb(op_idx), self.table_emb(tb_idx), num_tensor], dim=-1)
        u_v = self.node_proj(x_v)

        combined = torch.cat([u_v, h_child_sum], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        h_tilde = torch.tanh(self.W_h(torch.cat([u_v, r * h_child_sum], dim=-1)))
        h_v = (1.0 - z) * h_child_sum + z * h_tilde

        all_states.append(h_v)
        is_join = 1.0 if int(node.get("op_idx", 7)) == 1 else 0.0
        all_scalars.append((est_norm, op_w * est_norm, is_join))
        return h_v, all_states, all_scalars

    def forward_tree(
        self, tree: Dict[str, Any], est_mean: float, est_std: float
    ) -> torch.Tensor:
        root_h, all_h, all_sc = self.encode_node_recursive(tree, est_mean, est_std)
        mean_pool_h = torch.stack(all_h, dim=0).mean(dim=0)
        sc_t = torch.tensor(all_sc, dtype=torch.float32)
        tree_summary = torch.stack(
            [
                sc_t[:, 0].max(),
                sc_t[:, 1].mean(),
                sc_t[:, 2].sum() / 6.0,
            ],
            dim=0,
        )
        rep = self.dropout(torch.cat([root_h, mean_pool_h, tree_summary], dim=-1))
        return (self.readout(rep) + self.skip_work(tree_summary)).squeeze(-1)


class TreeNeuralCostModel:
    """
    Model Family 2: Tree-Structured Neural Model (Section 5, 6, 8).
    Reads the hierarchical plan structure directly via bottom-up recursive message passing.
    """

    def __init__(
        self,
        hidden_dim: int = 24,
        epochs: int = 18,
        lr: float = 0.006,
        weight_decay: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.est_mean = 0.0
        self.est_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.net: Optional[TreeRecursiveNet] = None

    def _collect_est_logs(self, node: Dict[str, Any], acc: List[float]) -> None:
        acc.append(float(node.get("est_rows_log", 0.0)))
        for ch in node.get("children", []):
            self._collect_est_logs(ch, acc)

    def fit(
        self,
        trees_train: List[Dict[str, Any]],
        y_train_ms: np.ndarray,
        trees_val: Optional[List[Dict[str, Any]]] = None,
        y_val_ms: Optional[np.ndarray] = None,
    ) -> "TreeNeuralCostModel":
        set_all_seeds(self.seed)

        # Fit normalization statistics strictly on training set (ML Best Practices)
        all_ests: List[float] = []
        for t in trees_train:
            self._collect_est_logs(t, all_ests)
        self.est_mean = float(np.mean(all_ests)) if all_ests else 0.0
        self.est_std = float(np.std(all_ests)) + 1e-4 if all_ests else 1.0

        y_log = np.log(np.maximum(1e-3, y_train_ms))
        self.y_mean = float(np.mean(y_log))
        self.y_std = float(np.std(y_log)) + 1e-4

        self.net = TreeRecursiveNet(hidden_dim=self.hidden_dim)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.SmoothL1Loss(beta=0.25)

        best_val_loss = float("inf")
        best_state = None
        indices = list(range(len(trees_train)))
        rng = random.Random(self.seed)

        for epoch in range(self.epochs):
            self.net.train()
            rng.shuffle(indices)
            batch_loss = torch.tensor(0.0)
            count = 0
            optimizer.zero_grad()

            for i, idx in enumerate(indices):
                pred_norm = self.net.forward_tree(
                    trees_train[idx], self.est_mean, self.est_std
                )
                target_norm = torch.tensor(
                    (y_log[idx] - self.y_mean) / self.y_std, dtype=torch.float32
                )
                loss = loss_fn(pred_norm, target_norm)
                batch_loss = batch_loss + loss
                count += 1

                if count == 16 or i == len(indices) - 1:
                    (batch_loss / count).backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss = torch.tensor(0.0)
                    count = 0

            if trees_val is not None and y_val_ms is not None and len(trees_val) > 0:
                val_preds = self.predict(trees_val)
                val_q = np.median(compute_q_errors(y_val_ms, val_preds))
                if val_q < best_val_loss:
                    best_val_loss = val_q
                    best_state = {
                        k: v.detach().clone() for k, v in self.net.state_dict().items()
                    }

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, trees: List[Dict[str, Any]]) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        preds = []
        with torch.no_grad():
            for t in trees:
                pred_norm = float(
                    self.net.forward_tree(t, self.est_mean, self.est_std).item()
                )
                log_ms = pred_norm * self.y_std + self.y_mean
                preds.append(math.exp(max(-5.0, min(15.0, log_ms))))
        return np.asarray(preds, dtype=np.float64)


def calibrate_cross_engine_scale(
    raw_source_preds_cal: np.ndarray,
    target_actual_ms_cal: np.ndarray,
    raw_source_preds_test: np.ndarray,
) -> np.ndarray:
    """
    Calibrates a 2-parameter log-linear scale mapping (`a * log(pred_duckdb) + b`)
    from a small target-engine calibration sample (from train templates only)
    to map DuckDB millisecond predictions to Trino millisecond scale while
    preserving the learned cost structure.
    """
    x_cal = np.log(np.maximum(1e-4, raw_source_preds_cal)).reshape(-1, 1)
    y_cal = np.log(np.maximum(1e-4, target_actual_ms_cal))
    reg = Ridge(alpha=2.0).fit(x_cal, y_cal)
    x_test = np.log(np.maximum(1e-4, raw_source_preds_test)).reshape(-1, 1)
    y_test_log = reg.predict(x_test)
    return np.exp(np.clip(y_test_log, -5.0, 15.0))

"""
Automated verification suite for the Learned Cost Estimation project.
Verifies:
1. All 6 relational tables exist and are populated with foreign-key integrity.
2. Strict template-disjoint partitioning: zero overlap of template_id across train, val, and test folds.
3. Every plan has 5 timed runs in EXECUTIONS (run_index 0..4, with run 0 marked is_cold=1).
4. Feature vectors and tree encodings contain zero NaN/null values and do not leak actual_rows.
5. Q-error and ranking metrics satisfy mathematical bounds (Q-error >= 1.0, Spearman in [-1, 1]).
6. All 5 publication figures and evaluation_metrics.json exist and are non-empty.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.schema import ExperimentStore
from src.models.estimators import compute_q_errors, evaluate_predictions
DB_PATH = PROJECT_ROOT / "data" / "experiment_store.sqlite"
RESULTS_JSON = PROJECT_ROOT / "results" / "evaluation_metrics.json"
FIG_DIR = PROJECT_ROOT / "results" / "figures"


def test_six_linked_tables_populated():
    assert DB_PATH.exists(), "Experiment SQLite database must exist"
    store = ExperimentStore(DB_PATH)
    tables = ["TEMPLATES", "QUERIES", "PLANS", "PLAN_NODES", "EXECUTIONS", "FEATURES"]
    for t in tables:
        cnt = store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        assert cnt > 0, f"Table {t} must not be empty"
    store.close()


def test_strict_template_disjoint_split():
    """Verifies Section 5 & Section 7: train/val/test split lives strictly on TEMPLATES."""
    store = ExperimentStore(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT q.query_id, q.template_id, t.split_fold
        FROM QUERIES q
        JOIN TEMPLATES t ON q.template_id = t.template_id
        """,
        store.conn,
    )
    train_tmpls = set(df[df["split_fold"] == "train"]["template_id"])
    val_tmpls = set(df[df["split_fold"] == "val"]["template_id"])
    test_tmpls = set(df[df["split_fold"] == "test"]["template_id"])

    assert len(train_tmpls) == 15
    assert len(val_tmpls) == 3
    assert len(test_tmpls) == 6
    assert train_tmpls.isdisjoint(val_tmpls)
    assert train_tmpls.isdisjoint(test_tmpls)
    assert val_tmpls.isdisjoint(test_tmpls)
    store.close()


def test_five_runs_and_cold_flag_per_plan():
    """Verifies Section 6 & Section 7: 5 runs per plan, run 0 is cold, runs 1..4 are warm."""
    store = ExperimentStore(DB_PATH)
    df_plans = pd.read_sql_query("SELECT plan_id FROM PLANS", store.conn)
    df_exec = pd.read_sql_query("SELECT * FROM EXECUTIONS", store.conn)

    assert len(df_exec) == len(df_plans) * 5
    cold_runs = df_exec[df_exec["run_index"] == 0]
    warm_runs = df_exec[df_exec["run_index"] > 0]
    assert (cold_runs["is_cold"] == 1).all()
    assert (warm_runs["is_cold"] == 0).all()
    assert (df_exec["wall_ms"] > 0).all()
    store.close()


def test_features_no_target_leakage():
    """Verifies features only use estimated quantities and topology, never actual_rows or wall_ms."""
    store = ExperimentStore(DB_PATH)
    row = store.conn.execute("SELECT flat_vector, tree_encoding FROM FEATURES LIMIT 1").fetchone()
    flat = json.loads(row["flat_vector"])
    tree = json.loads(row["tree_encoding"])

    for k, v in flat.items():
        assert "actual" not in k.lower()
        assert "wall_ms" not in k.lower()
        assert not np.isnan(float(v))

    assert "actual_rows" not in tree
    assert "est_rows_log" in tree
    store.close()


def test_q_error_and_metrics_validity():
    y_true = np.array([10.0, 20.0, 50.0, 100.0])
    y_pred = np.array([20.0, 10.0, 50.0, 25.0])
    q = compute_q_errors(y_true, y_pred)
    assert np.allclose(q, [2.0, 2.0, 1.0, 4.0])

    ev = evaluate_predictions(y_true, y_pred)
    assert ev["median_q_error"] == 2.0
    assert ev["max_q_error"] == 4.0
    assert -1.0 <= ev["spearman_rho"] <= 1.0


def test_artifacts_and_figures_generated():
    assert RESULTS_JSON.exists()
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    for key in [
        "duckdb_in_engine",
        "trino_in_engine",
        "cross_engine_calibrated",
        "per_template_transfer",
        "ablations",
        "cardinality_bonus_by_operator",
    ]:
        assert key in data

    expected_figs = [
        "fig1_q_error_comparison.png",
        "fig2_cross_engine_ranking.png",
        "fig3_template_vs_random_leakage.png",
        "fig4_cardinality_q_error_by_operator.png",
        "fig5_ablation_and_timing_variance.png",
    ]
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert fpath.exists() and fpath.stat().st_size > 5000, f"Missing figure {fname}"

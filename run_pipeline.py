"""
End-to-End Orchestrator for:
"Optimisation of Query Execution Plans via Learned Cost Estimation:
 A cross-engine study on DuckDB and Trino, with plan-tree features and template-disjoint evaluation"

Single command reproduces everything:
    python run_pipeline.py

Stages executed:
  Stage 1: Generate TPC-H shared warehouse (Parquet/Iceberg) + 24 query templates (288 queries)
           split strictly at the TEMPLATE level (15 train, 3 val, 6 test).
  Stage 2: Execute every query on DuckDB and Trino over identical shared data.
  Stage 3: Record execution plans (PLANS), operator nodes with est_rows & actual_rows (PLAN_NODES),
           and 5 timed runs per query with cold-run flagging (EXECUTIONS).
  Stage 4: Extract flattened feature vectors and recursive plan-tree encodings (FEATURES).
  Stage 5: Train baselines (Native CBO, Data-Size-Alone, Linear Ridge) and both learned model
           families (Gradient Boosting on flat features, Tree-Structured Neural Network on plan trees).
  Stage 6: Evaluate honestly on held-out templates (Q-error median/P90/P95, Spearman/Kendall rank
           correlation, pairwise plan ranking accuracy), run cross-engine transfer (DuckDB -> Trino),
           compare against random-split leakage, run feature-group ablations, analyze node cardinality
           Q-error ("free bonus"), export all 6 tables + JSON metrics + 5 figures + final Markdown report.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.storage.schema import ExperimentStore
from src.workload.generator import (
    generate_queries_from_templates,
    prepare_shared_tpch_warehouse,
)
from src.engines.runner import DuckDBRunner, TrinoIcebergRunner, normalize_operator_name
from src.features.extractor import FLAT_FEATURE_NAMES, extract_plan_features
from src.models.estimators import (
    DataSizeBaseline,
    GradientBoostingCostModel,
    LinearCostBaseline,
    NativeCBOEstimator,
    TreeNeuralCostModel,
    calibrate_cross_engine_scale,
    compute_q_errors,
    evaluate_predictions,
    set_all_seeds,
)
from src.evaluation.plots import plot_all_figures


def compute_warm_median_runtimes(store: ExperimentStore) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Computes the warm median runtime (runs 1..4, dropping run 0 where is_cold=1)
    for every plan_id, and also calculates timing variance statistics across all 5 runs
    vs warm runs (Section 7: "Why every single run is stored, not just the average").
    """
    df_exec = pd.read_sql_query("SELECT * FROM EXECUTIONS", store.conn)
    df_plans = pd.read_sql_query("SELECT plan_id, engine FROM PLANS", store.conn)
    df = df_exec.merge(df_plans, on="plan_id", how="left")

    warm_df = df[df["is_cold"] == 0]
    medians = warm_df.groupby("plan_id")["wall_ms"].median().to_dict()

    # Compute Coefficient of Variation (std / mean * 100) per plan across all 5 runs vs warm runs
    all_stats = df.groupby(["plan_id", "engine"])["wall_ms"].agg(["mean", "std"]).reset_index()
    all_stats["cv_pct"] = (all_stats["std"] / np.maximum(1e-6, all_stats["mean"])) * 100.0

    warm_stats = warm_df.groupby(["plan_id", "engine"])["wall_ms"].agg(["mean", "std"]).reset_index()
    warm_stats["cv_pct"] = (warm_stats["std"] / np.maximum(1e-6, warm_stats["mean"])) * 100.0

    cold_vs_warm = df.pivot_table(index=["plan_id", "engine"], columns="is_cold", values="wall_ms", aggfunc="median").reset_index()
    cold_vs_warm["cold_ratio"] = cold_vs_warm[1] / np.maximum(1e-6, cold_vs_warm[0])

    variance_summary = {
        "duckdb_cv_all_pct": float(all_stats[all_stats["engine"] == "duckdb"]["cv_pct"].median()),
        "duckdb_cv_warm_pct": float(warm_stats[warm_stats["engine"] == "duckdb"]["cv_pct"].median()),
        "trino_cv_all_pct": float(all_stats[all_stats["engine"] == "trino"]["cv_pct"].median()),
        "trino_cv_warm_pct": float(warm_stats[warm_stats["engine"] == "trino"]["cv_pct"].median()),
        "duckdb_cold_speedup_ratio": float(cold_vs_warm[cold_vs_warm["engine"] == "duckdb"]["cold_ratio"].median()),
        "trino_cold_speedup_ratio": float(cold_vs_warm[cold_vs_warm["engine"] == "trino"]["cold_ratio"].median()),
    }
    return medians, variance_summary


def build_engine_dataset(
    store: ExperimentStore,
    engine: str,
    warm_medians: Dict[str, float],
) -> Dict[str, Any]:
    """
    Loads aligned arrays (X_flat, trees, native_est_cost, y_ms, split_fold, template_id, query_id)
    for a given engine ('duckdb' or 'trino').
    """
    sql = """
    SELECT
        p.plan_id,
        p.query_id,
        p.engine,
        p.native_est_cost,
        q.template_id,
        t.split_fold,
        t.n_relations,
        t.source,
        f.flat_vector,
        f.tree_encoding
    FROM PLANS p
    JOIN QUERIES q ON p.query_id = q.query_id
    JOIN TEMPLATES t ON q.template_id = t.template_id
    JOIN FEATURES f ON p.plan_id = f.plan_id
    WHERE p.engine = ?
    ORDER BY q.query_id
    """
    rows = store.conn.execute(sql, (engine,)).fetchall()

    X_list = []
    trees = []
    native_costs = []
    y_ms = []
    splits = []
    template_ids = []
    query_ids = []
    n_rels = []

    for r in rows:
        pid = r["plan_id"]
        f_dict = json.loads(r["flat_vector"])
        X_list.append([float(f_dict.get(col, 0.0)) for col in FLAT_FEATURE_NAMES])
        trees.append(json.loads(r["tree_encoding"]))
        native_costs.append(float(r["native_est_cost"]))
        y_ms.append(float(warm_medians[pid]))
        splits.append(r["split_fold"])
        template_ids.append(r["template_id"])
        query_ids.append(r["query_id"])
        n_rels.append(int(r["n_relations"]))

    return {
        "X": np.asarray(X_list, dtype=np.float64),
        "trees": trees,
        "native_cost": np.asarray(native_costs, dtype=np.float64),
        "y_ms": np.asarray(y_ms, dtype=np.float64),
        "split": np.asarray(splits),
        "template_id": np.asarray(template_ids),
        "query_id": np.asarray(query_ids),
        "n_relations": np.asarray(n_rels, dtype=np.int64),
    }


def run_pipeline(
    base_dir: Path,
    variants_per_template: int = 12,
    scale_factor: float = 0.05,
    seed: int = 42,
    force_recollect: bool = False,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    set_all_seeds(seed)

    data_dir = base_dir / "data"
    warehouse_dir = data_dir / "warehouse"
    db_path = data_dir / "experiment_store.sqlite"
    tables_export_dir = data_dir / "tables"
    results_dir = base_dir / "results"
    fig_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("OPTIMISATION OF QUERY EXECUTION PLANS VIA LEARNED COST ESTIMATION")
    print("Cross-Engine Study: DuckDB -> Trino | Template-Disjoint Evaluation")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # STAGE 1: Shared Warehouse & Template-Disjoint Query Generation
    # -------------------------------------------------------------------------
    print("\n[Stage 1/6] Preparing shared TPC-H Parquet/Iceberg warehouse & query templates...")
    row_counts = prepare_shared_tpch_warehouse(warehouse_dir, scale_factor=scale_factor)
    print(f"  Shared warehouse table counts: {row_counts}")

    templates, queries = generate_queries_from_templates(
        variants_per_template=variants_per_template, base_seed=seed
    )
    store = ExperimentStore(db_path)

    for tmpl in templates:
        store.upsert_template(
            template_id=tmpl.template_id,
            source=tmpl.source,
            n_relations=tmpl.n_relations,
            split_fold=tmpl.split_fold,
            description=tmpl.description,
        )
    for q in queries:
        store.upsert_query(
            query_id=q["query_id"],
            template_id=q["template_id"],
            sql_text=q["sql_text"],
            seed=q["seed"],
        )

    split_counts = {}
    for tmpl in templates:
        split_counts[tmpl.split_fold] = split_counts.get(tmpl.split_fold, 0) + 1
    print(
        f"  Registered {len(templates)} templates ({split_counts}) and {len(queries)} queries."
    )

    # -------------------------------------------------------------------------
    # STAGES 2, 3, 4: Plan Collection, 5x Timed Runs, and Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[Stages 2-4/6] Running queries on DuckDB & Trino, capturing plans, 5x timings & features...")
    duck_runner = DuckDBRunner(warehouse_dir)
    trino_runner = TrinoIcebergRunner(warehouse_dir)

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        sql_text = q["sql_text"]

        for engine_name, runner in [("duckdb", duck_runner), ("trino", trino_runner)]:
            plan_id = f"{qid}__{engine_name}"
            if not force_recollect and store.has_plan_and_executions(plan_id, expected_runs=5):
                continue

            plan_json, native_est_cost, nodes, executions = runner.collect_query(
                query_id=qid, sql_text=sql_text, n_runs=5
            )
            store.save_plan_bundle(
                plan_id=plan_id,
                query_id=qid,
                engine=engine_name,
                plan_json=plan_json,
                native_est_cost=native_est_cost,
                nodes=nodes,
                executions=executions,
            )
            flat_vec, tree_enc = extract_plan_features(
                plan_id=plan_id, native_est_cost=native_est_cost, nodes=nodes
            )
            store.save_features(
                plan_id=plan_id,
                flat_vector=flat_vec,
                tree_encoding=tree_enc,
                extractor_version="v1.0",
            )

        if (idx + 1) % 48 == 0 or (idx + 1) == len(queries):
            print(f"  Processed {idx + 1}/{len(queries)} queries across both engines...")

    # Export all 6 tables to CSV and Parquet
    table_counts = store.export_tables_to_csv_and_parquet(tables_export_dir)
    print(f"  6-Table Store row counts: {table_counts}")

    # -------------------------------------------------------------------------
    # STAGE 5: Train Baselines, Flat GBDT, and Tree-Structured Neural Models
    # -------------------------------------------------------------------------
    print("\n[Stage 5/6] Training Baselines, Gradient Boosting (Flat), and Tree-Structured Neural Models...")
    warm_medians, variance_summary = compute_warm_median_runtimes(store)
    duck_data = build_engine_dataset(store, "duckdb", warm_medians)
    trino_data = build_engine_dataset(store, "trino", warm_medians)

    train_mask = duck_data["split"] == "train"
    val_mask = duck_data["split"] == "val"
    test_mask = duck_data["split"] == "test"
    train_val_mask = train_mask | val_mask

    # --- 5A. DuckDB Models (Trained ONLY on DuckDB train/val templates) ---
    duck_native = NativeCBOEstimator().fit(
        duck_data["native_cost"][train_val_mask], duck_data["y_ms"][train_val_mask]
    )
    duck_datasize = DataSizeBaseline().fit(
        duck_data["X"][train_val_mask], duck_data["y_ms"][train_val_mask]
    )
    duck_linear = LinearCostBaseline().fit(
        duck_data["X"][train_val_mask], duck_data["y_ms"][train_val_mask]
    )
    duck_gbdt = GradientBoostingCostModel(seed=seed).fit(
        duck_data["X"][train_val_mask], duck_data["y_ms"][train_val_mask]
    )
    duck_tree_nn = TreeNeuralCostModel(seed=seed, epochs=18).fit(
        [duck_data["trees"][i] for i in np.where(train_mask)[0]],
        duck_data["y_ms"][train_mask],
        [duck_data["trees"][i] for i in np.where(val_mask)[0]],
        duck_data["y_ms"][val_mask],
    )

    # --- 5B. Trino In-Engine Oracle Models (Trained on Trino train/val templates) ---
    trino_native = NativeCBOEstimator().fit(
        trino_data["native_cost"][train_val_mask], trino_data["y_ms"][train_val_mask]
    )
    trino_datasize = DataSizeBaseline().fit(
        trino_data["X"][train_val_mask], trino_data["y_ms"][train_val_mask]
    )
    trino_linear = LinearCostBaseline().fit(
        trino_data["X"][train_val_mask], trino_data["y_ms"][train_val_mask]
    )
    trino_gbdt = GradientBoostingCostModel(seed=seed).fit(
        trino_data["X"][train_val_mask], trino_data["y_ms"][train_val_mask]
    )
    trino_tree_nn = TreeNeuralCostModel(seed=seed, epochs=18).fit(
        [trino_data["trees"][i] for i in np.where(train_mask)[0]],
        trino_data["y_ms"][train_mask],
        [trino_data["trees"][i] for i in np.where(val_mask)[0]],
        trino_data["y_ms"][val_mask],
    )

    # -------------------------------------------------------------------------
    # STAGE 6: Honest Template-Disjoint Evaluation, Cross-Engine Transfer & Ablations
    # -------------------------------------------------------------------------
    print("\n[Stage 6/6] Evaluating In-Engine, Cross-Engine Transfer (DuckDB -> Trino), and Ablations...")

    y_test_duck = duck_data["y_ms"][test_mask]
    y_test_trino = trino_data["y_ms"][test_mask]

    # 1. In-Engine DuckDB -> DuckDB on held-out test templates
    duck_in_engine_eval = {
        "Native Optimizer": evaluate_predictions(
            y_test_duck, duck_native.predict(duck_data["native_cost"][test_mask])
        ),
        "Data Size Alone": evaluate_predictions(
            y_test_duck, duck_datasize.predict(duck_data["X"][test_mask])
        ),
        "Linear Ridge": evaluate_predictions(
            y_test_duck, duck_linear.predict(duck_data["X"][test_mask])
        ),
        "Gradient Boosting (Flat)": evaluate_predictions(
            y_test_duck, duck_gbdt.predict(duck_data["X"][test_mask])
        ),
        "Tree Neural Model": evaluate_predictions(
            y_test_duck,
            duck_tree_nn.predict([duck_data["trees"][i] for i in np.where(test_mask)[0]]),
        ),
    }

    # 2. In-Engine Trino -> Trino on held-out test templates
    trino_in_engine_eval = {
        "Native Optimizer": evaluate_predictions(
            y_test_trino, trino_native.predict(trino_data["native_cost"][test_mask])
        ),
        "Data Size Alone": evaluate_predictions(
            y_test_trino, trino_datasize.predict(trino_data["X"][test_mask])
        ),
        "Linear Ridge": evaluate_predictions(
            y_test_trino, trino_linear.predict(trino_data["X"][test_mask])
        ),
        "Gradient Boosting (Flat)": evaluate_predictions(
            y_test_trino, trino_gbdt.predict(trino_data["X"][test_mask])
        ),
        "Tree Neural Model": evaluate_predictions(
            y_test_trino,
            trino_tree_nn.predict([trino_data["trees"][i] for i in np.where(test_mask)[0]]),
        ),
    }

    # 3. Cross-Engine Transfer: Trained on DuckDB -> Evaluated on Trino!
    # Raw zero-shot predictions from DuckDB-trained models on Trino plans
    raw_transfer_datasize_cal = duck_datasize.predict(trino_data["X"][train_val_mask])
    raw_transfer_datasize_test = duck_datasize.predict(trino_data["X"][test_mask])

    raw_transfer_linear_cal = duck_linear.predict(trino_data["X"][train_val_mask])
    raw_transfer_linear_test = duck_linear.predict(trino_data["X"][test_mask])

    raw_transfer_gbdt_cal = duck_gbdt.predict(trino_data["X"][train_val_mask])
    raw_transfer_gbdt_test = duck_gbdt.predict(trino_data["X"][test_mask])

    raw_transfer_tree_cal = duck_tree_nn.predict(
        [trino_data["trees"][i] for i in np.where(train_val_mask)[0]]
    )
    raw_transfer_tree_test = duck_tree_nn.predict(
        [trino_data["trees"][i] for i in np.where(test_mask)[0]]
    )

    # Zero-shot uncalibrated evaluation (tests pure rank transfer & raw scale gap)
    cross_engine_zeroshot_eval = {
        "Native Optimizer": trino_in_engine_eval["Native Optimizer"],
        "Data Size Alone": evaluate_predictions(y_test_trino, raw_transfer_datasize_test),
        "Linear Ridge": evaluate_predictions(y_test_trino, raw_transfer_linear_test),
        "Gradient Boosting (Flat)": evaluate_predictions(y_test_trino, raw_transfer_gbdt_test),
        "Tree Neural Model": evaluate_predictions(y_test_trino, raw_transfer_tree_test),
    }

    # Few-shot scale-calibrated transfer (12 calibration queries from train fold to map ms scale)
    rng_cal = np.random.default_rng(seed)
    cal_sub_idx = rng_cal.choice(np.sum(train_val_mask), size=12, replace=False)
    y_trino_cal = trino_data["y_ms"][train_val_mask][cal_sub_idx]

    cal_transfer_datasize_test = calibrate_cross_engine_scale(
        raw_transfer_datasize_cal[cal_sub_idx], y_trino_cal, raw_transfer_datasize_test
    )
    cal_transfer_linear_test = calibrate_cross_engine_scale(
        raw_transfer_linear_cal[cal_sub_idx], y_trino_cal, raw_transfer_linear_test
    )
    cal_transfer_gbdt_test = calibrate_cross_engine_scale(
        raw_transfer_gbdt_cal[cal_sub_idx], y_trino_cal, raw_transfer_gbdt_test
    )
    cal_transfer_tree_test = calibrate_cross_engine_scale(
        raw_transfer_tree_cal[cal_sub_idx], y_trino_cal, raw_transfer_tree_test
    )

    cross_engine_calibrated_eval = {
        "Native Optimizer": trino_in_engine_eval["Native Optimizer"],
        "Data Size Alone": evaluate_predictions(y_test_trino, cal_transfer_datasize_test),
        "Linear Ridge": evaluate_predictions(y_test_trino, cal_transfer_linear_test),
        "Gradient Boosting (Flat)": evaluate_predictions(y_test_trino, cal_transfer_gbdt_test),
        "Tree Neural Model": evaluate_predictions(y_test_trino, cal_transfer_tree_test),
    }

    # Per-template breakdown on Held-Out Test Templates (T19..T24) to see where transfer wins vs loses
    trino_native_test_preds = trino_native.predict(trino_data["native_cost"][test_mask])
    test_tids = trino_data["template_id"][test_mask]
    test_nrels = trino_data["n_relations"][test_mask]
    per_template_transfer = []
    for tid in sorted(set(test_tids)):
        m = test_tids == tid
        nrel = int(test_nrels[m][0])
        per_template_transfer.append(
            {
                "template_id": str(tid),
                "n_relations": nrel,
                "native_median_q": float(np.median(compute_q_errors(y_test_trino[m], trino_native_test_preds[m]))),
                "gbdt_transfer_median_q": float(np.median(compute_q_errors(y_test_trino[m], cal_transfer_gbdt_test[m]))),
                "tree_transfer_median_q": float(np.median(compute_q_errors(y_test_trino[m], cal_transfer_tree_test[m]))),
                "native_p90_q": float(np.percentile(compute_q_errors(y_test_trino[m], trino_native_test_preds[m]), 90)),
                "gbdt_transfer_p90_q": float(np.percentile(compute_q_errors(y_test_trino[m], cal_transfer_gbdt_test[m]), 90)),
                "tree_transfer_p90_q": float(np.percentile(compute_q_errors(y_test_trino[m], cal_transfer_tree_test[m]), 90)),
            }
        )

    # 4. Template-Disjoint Split vs Random Split (Proof of Leakage from Section 5)
    n_total = len(duck_data["y_ms"])
    rng_leak = np.random.default_rng(seed + 7)
    perm = rng_leak.permutation(n_total)
    n_test = int(np.sum(test_mask))
    rand_test_idx = perm[:n_test]
    rand_train_idx = perm[n_test:]

    rand_gbdt_duck = GradientBoostingCostModel(seed=seed).fit(
        duck_data["X"][rand_train_idx], duck_data["y_ms"][rand_train_idx]
    )
    rand_tree_duck = TreeNeuralCostModel(seed=seed, epochs=18).fit(
        [duck_data["trees"][i] for i in rand_train_idx], duck_data["y_ms"][rand_train_idx]
    )
    rand_linear_duck = LinearCostBaseline().fit(
        duck_data["X"][rand_train_idx], duck_data["y_ms"][rand_train_idx]
    )

    split_leakage_comparison = {
        "random_split": {
            "Linear Ridge": evaluate_predictions(
                duck_data["y_ms"][rand_test_idx],
                rand_linear_duck.predict(duck_data["X"][rand_test_idx]),
            ),
            "Gradient Boosting (Flat)": evaluate_predictions(
                duck_data["y_ms"][rand_test_idx],
                rand_gbdt_duck.predict(duck_data["X"][rand_test_idx]),
            ),
            "Tree Neural Model": evaluate_predictions(
                duck_data["y_ms"][rand_test_idx],
                rand_tree_duck.predict([duck_data["trees"][i] for i in rand_test_idx]),
            ),
        },
        "template_disjoint": {
            "Linear Ridge": duck_in_engine_eval["Linear Ridge"],
            "Gradient Boosting (Flat)": duck_in_engine_eval["Gradient Boosting (Flat)"],
            "Tree Neural Model": duck_in_engine_eval["Tree Neural Model"],
        },
    }

    # 5. Systematic Feature-Group Ablations (Section 8)
    op_cols = [i for i, c in enumerate(FLAT_FEATURE_NAMES) if c.startswith("op_count_")]
    struct_cols = [i for i, c in enumerate(FLAT_FEATURE_NAMES) if c.startswith("tree_")]
    card_cols = [i for i, c in enumerate(FLAT_FEATURE_NAMES) if c.startswith("log_")]
    table_cols = [
        i
        for i, c in enumerate(FLAT_FEATURE_NAMES)
        if c.startswith("table_") or c == "n_distinct_tables"
    ]
    all_cols = list(range(len(FLAT_FEATURE_NAMES)))

    ablation_configs = [
        ("Full Model (All Feature Groups)", all_cols),
        ("w/o Cardinality Estimates (`est_rows`)", [i for i in all_cols if i not in card_cols]),
        ("w/o Operator Counts", [i for i in all_cols if i not in op_cols]),
        ("w/o Plan Tree Structure (Depth/Leaves)", [i for i in all_cols if i not in struct_cols]),
        ("w/o Scanned Table Footprint", [i for i in all_cols if i not in table_cols]),
        ("Only Operator Counts", op_cols),
    ]

    ablations_list = []
    for variant_name, cols in ablation_configs:
        m_abl = GradientBoostingCostModel(active_indices=cols, seed=seed).fit(
            duck_data["X"][train_val_mask], duck_data["y_ms"][train_val_mask]
        )
        p_duck = m_abl.predict(duck_data["X"][test_mask])
        ev_duck = evaluate_predictions(y_test_duck, p_duck)

        raw_tr_cal = m_abl.predict(trino_data["X"][train_val_mask])
        raw_tr_test = m_abl.predict(trino_data["X"][test_mask])
        cal_tr_test = calibrate_cross_engine_scale(
            raw_tr_cal[cal_sub_idx], y_trino_cal, raw_tr_test
        )
        ev_tr = evaluate_predictions(y_test_trino, cal_tr_test)

        ablations_list.append(
            {
                "variant": variant_name,
                "duckdb_median_q": ev_duck["median_q_error"],
                "duckdb_p90_q": ev_duck["p90_q_error"],
                "transfer_median_q": ev_tr["median_q_error"],
                "transfer_p90_q": ev_tr["p90_q_error"],
                "transfer_spearman_rho": ev_tr["spearman_rho"],
            }
        )

    # 6. "A Free Bonus" (Section 7): Node-level Cardinality Q-Error (`est_rows` vs `actual_rows`)
    nodes_df = pd.read_sql_query("SELECT * FROM PLAN_NODES", store.conn)
    nodes_df["op_cat"] = nodes_df["operator"].apply(normalize_operator_name)
    est_s = np.maximum(1.0, nodes_df["est_rows"].astype(float))
    act_s = np.maximum(1.0, nodes_df["actual_rows"].astype(float))
    nodes_df["card_q_error"] = np.maximum(est_s / act_s, act_s / est_s)

    card_bonus_by_op = (
        nodes_df.groupby("op_cat")["card_q_error"]
        .agg(
            count="count",
            median_q="median",
            p90_q=lambda s: float(np.percentile(s, 90)),
            p95_q=lambda s: float(np.percentile(s, 95)),
            max_q="max",
        )
        .reset_index()
        .sort_values("median_q", ascending=False)
        .to_dict(orient="records")
    )

    exec_df = pd.read_sql_query("SELECT * FROM EXECUTIONS", store.conn)

    # Top feature importances from DuckDB GBDT model
    feat_imps = duck_gbdt.get_feature_importances(FLAT_FEATURE_NAMES)
    top_features = sorted(feat_imps.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # Detailed test predictions for scatter plots
    test_predictions_trino = {
        "query_id": [str(x) for x in trino_data["query_id"][test_mask]],
        "template_id": [str(x) for x in trino_data["template_id"][test_mask]],
        "actual_trino_ms": [float(x) for x in y_test_trino],
        "pred_native": [float(x) for x in trino_native_test_preds],
        "pred_xgb_transfer": [float(x) for x in cal_transfer_gbdt_test],
        "pred_tree_transfer": [float(x) for x in cal_transfer_tree_test],
    }

    results_payload = {
        "metadata": {
            "seed": seed,
            "scale_factor": scale_factor,
            "n_templates": len(templates),
            "n_queries": len(queries),
            "table_counts": table_counts,
            "warehouse_row_counts": row_counts,
            "elapsed_sec": round(time.perf_counter() - t_start, 2),
        },
        "duckdb_in_engine": duck_in_engine_eval,
        "trino_in_engine": trino_in_engine_eval,
        "cross_engine_zeroshot": cross_engine_zeroshot_eval,
        "cross_engine_calibrated": cross_engine_calibrated_eval,
        "per_template_transfer": per_template_transfer,
        "split_leakage_comparison": split_leakage_comparison,
        "ablations": ablations_list,
        "cardinality_bonus_by_operator": card_bonus_by_op,
        "timing_variance_summary": variance_summary,
        "top_gbdt_features": [
            {"feature": k, "importance": float(v)} for k, v in top_features
        ],
        "test_predictions_trino": test_predictions_trino,
    }

    # Save JSON metrics
    metrics_file = results_dir / "evaluation_metrics.json"
    metrics_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    # Generate all 5 scientific figures
    saved_figs = plot_all_figures(results_payload, nodes_df, exec_df, fig_dir)
    print(f"  Generated {len(saved_figs)} figures in {fig_dir}")

    store.close()
    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the End-to-End Learned Cost Estimation Pipeline"
    )
    parser.add_argument("--variants", type=int, default=12, help="Queries per template")
    parser.add_argument("--sf", type=float, default=0.05, help="TPC-H scale factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--force", action="store_true", help="Force re-execution of all queries"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    res = run_pipeline(
        base_dir=project_root,
        variants_per_template=args.variants,
        scale_factor=args.sf,
        seed=args.seed,
        force_recollect=args.force,
    )

    print("\n" + "=" * 78)
    print("HEADLINE SUMMARY (HELD-OUT TEST TEMPLATES T19..T24)")
    print("=" * 78)
    for regime_key, regime_label in [
        ("duckdb_in_engine", "1. In-Engine DuckDB -> DuckDB"),
        ("trino_in_engine", "2. In-Engine Trino -> Trino (Upper Bound)"),
        ("cross_engine_calibrated", "3. Cross-Engine Transfer: DuckDB -> Trino"),
    ]:
        print(f"\n{regime_label}:")
        for m_name, metrics in res[regime_key].items():
            print(
                f"  {m_name:<26s} | Median Q: {metrics['median_q_error']:.3f} "
                f"| P90 Q: {metrics['p90_q_error']:.3f} | P95 Q: {metrics['p95_q_error']:.3f} "
                f"| Spearman rho: {metrics['spearman_rho']:.3f} "
                f"| Rank Acc: {metrics['pairwise_rank_acc']:.1f}%"
            )

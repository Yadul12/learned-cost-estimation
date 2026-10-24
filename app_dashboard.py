"""
Interactive Live Web Dashboard for:
"Optimisation of Query Execution Plans via Learned Cost Estimation"
Serves a rich interactive web application on http://localhost:8501 (using Python's built-in
HTTPServer + modern Chart.js / Tailwind UI) so the user can explore:
  1. Headline Q-Error & Ranking Metrics across DuckDB, Trino, and Cross-Engine Transfer
  2. Interactive Live Query Explorer (inspect any of the 288 queries, its SQL, template split,
     DuckDB vs Trino execution plan tree, node est_rows vs actual_rows, 5 timed runs, and
     cost predictions from every model)
  3. Live Ad-Hoc SQL Cost Estimator (type or tweak any TPC-H SQL query, run EXPLAIN & live
     prediction with DuckDB + Trino cost models in real time!)
  4. All 5 Scientific Figures & 6-Table Dataset Explorer
"""

import json
import mimetypes
import os
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

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
)
from src.storage.schema import ExperimentStore
from run_pipeline import build_engine_dataset, compute_warm_median_runtimes

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "experiment_store.sqlite"
WAREHOUSE_DIR = PROJECT_ROOT / "data" / "warehouse"
METRICS_PATH = PROJECT_ROOT / "results" / "evaluation_metrics.json"
FIG_DIR = PROJECT_ROOT / "results" / "figures"


class LivePredictorContext:
    """Holds live DuckDB/Trino runners and trained cost models for real-time SQL execution & inference."""

    def __init__(self) -> None:
        self.duck_runner = DuckDBRunner(WAREHOUSE_DIR)
        self.trino_runner = TrinoIcebergRunner(WAREHOUSE_DIR)
        store = ExperimentStore(DB_PATH)
        warm_medians, _ = compute_warm_median_runtimes(store)
        duck_data = build_engine_dataset(store, "duckdb", warm_medians)
        trino_data = build_engine_dataset(store, "trino", warm_medians)
        store.close()

        train_mask = duck_data["split"] == "train"
        val_mask = duck_data["split"] == "val"
        train_val_mask = train_mask | val_mask

        self.duck_native = NativeCBOEstimator().fit(
            duck_data["native_cost"][train_val_mask], duck_data["y_ms"][train_val_mask]
        )
        self.duck_datasize = DataSizeBaseline().fit(
            duck_data["X"][train_val_mask], duck_data["y_ms"][train_val_mask]
        )
        self.duck_gbdt = GradientBoostingCostModel(seed=42).fit(
            duck_data["X"][train_val_mask], duck_data["y_ms"][train_val_mask]
        )
        self.duck_tree = TreeNeuralCostModel(seed=42, epochs=18).fit(
            [duck_data["trees"][i] for i in np.where(train_mask)[0]],
            duck_data["y_ms"][train_mask],
            [duck_data["trees"][i] for i in np.where(val_mask)[0]],
            duck_data["y_ms"][val_mask],
        )

        self.trino_native = NativeCBOEstimator().fit(
            trino_data["native_cost"][train_val_mask], trino_data["y_ms"][train_val_mask]
        )
        self.trino_gbdt = GradientBoostingCostModel(seed=42).fit(
            trino_data["X"][train_val_mask], trino_data["y_ms"][train_val_mask]
        )

        rng_cal = np.random.default_rng(42)
        cal_sub_idx = rng_cal.choice(np.sum(train_val_mask), size=12, replace=False)
        self.y_trino_cal = trino_data["y_ms"][train_val_mask][cal_sub_idx]
        self.raw_gbdt_cal = self.duck_gbdt.predict(trino_data["X"][train_val_mask])[cal_sub_idx]
        self.raw_ds_cal = self.duck_datasize.predict(trino_data["X"][train_val_mask])[cal_sub_idx]
        self.raw_tree_cal = self.duck_tree.predict(
            [trino_data["trees"][i] for i in np.where(train_val_mask)[0]]
        )[cal_sub_idx]

    def run_live_sql(self, sql_text: str) -> dict:
        qid = f"LIVE_{int(time.time() * 1000) % 100000}"
        _, d_native_cost, d_nodes, d_execs = self.duck_runner.collect_query(qid, sql_text, n_runs=5)
        _, t_native_cost, t_nodes, t_execs = self.trino_runner.collect_query(qid, sql_text, n_runs=5)

        d_flat, d_tree = extract_plan_features(f"{qid}__duckdb", d_native_cost, d_nodes)
        t_flat, t_tree = extract_plan_features(f"{qid}__trino", t_native_cost, t_nodes)

        Xd = np.asarray([[float(d_flat.get(c, 0.0)) for c in FLAT_FEATURE_NAMES]], dtype=np.float64)
        Xt = np.asarray([[float(t_flat.get(c, 0.0)) for c in FLAT_FEATURE_NAMES]], dtype=np.float64)

        d_warm_ms = float(np.median([e["wall_ms"] for e in d_execs if not e["is_cold"]]))
        t_warm_ms = float(np.median([e["wall_ms"] for e in t_execs if not e["is_cold"]]))

        # DuckDB predictions
        pred_d_native = float(self.duck_native.predict(np.array([d_native_cost]))[0])
        pred_d_ds = float(self.duck_datasize.predict(Xd)[0])
        pred_d_gbdt = float(self.duck_gbdt.predict(Xd)[0])
        pred_d_tree = float(self.duck_tree.predict([d_tree])[0])

        # Trino predictions (Native + Cross-Engine Transfer from DuckDB)
        pred_t_native = float(self.trino_native.predict(np.array([t_native_cost]))[0])
        raw_t_ds = self.duck_datasize.predict(Xt)
        raw_t_gbdt = self.duck_gbdt.predict(Xt)
        raw_t_tree = self.duck_tree.predict([t_tree])

        pred_t_ds_transfer = float(
            calibrate_cross_engine_scale(self.raw_ds_cal, self.y_trino_cal, raw_t_ds)[0]
        )
        pred_t_gbdt_transfer = float(
            calibrate_cross_engine_scale(self.raw_gbdt_cal, self.y_trino_cal, raw_t_gbdt)[0]
        )
        pred_t_tree_transfer = float(
            calibrate_cross_engine_scale(self.raw_tree_cal, self.y_trino_cal, raw_t_tree)[0]
        )
        pred_t_oracle = float(self.trino_gbdt.predict(Xt)[0])

        def q_err(act, pred):
            return float(max(act / max(1e-6, pred), max(1e-6, pred) / act))

        return {
            "sql_text": sql_text,
            "duckdb": {
                "actual_warm_ms": d_warm_ms,
                "native_est_cost": d_native_cost,
                "executions": d_execs,
                "nodes": d_nodes,
                "predictions": [
                    {"model": "Native Optimizer", "pred_ms": pred_d_native, "q_error": q_err(d_warm_ms, pred_d_native)},
                    {"model": "Data Size Alone Baseline", "pred_ms": pred_d_ds, "q_error": q_err(d_warm_ms, pred_d_ds)},
                    {"model": "Gradient Boosting (Flat)", "pred_ms": pred_d_gbdt, "q_error": q_err(d_warm_ms, pred_d_gbdt)},
                    {"model": "Tree Neural Model", "pred_ms": pred_d_tree, "q_error": q_err(d_warm_ms, pred_d_tree)},
                ],
            },
            "trino": {
                "actual_warm_ms": t_warm_ms,
                "native_est_cost": t_native_cost,
                "executions": t_execs,
                "nodes": t_nodes,
                "predictions": [
                    {"model": "Trino Native Optimizer", "pred_ms": pred_t_native, "q_error": q_err(t_warm_ms, pred_t_native)},
                    {"model": "Cross-Engine Data-Size Transfer", "pred_ms": pred_t_ds_transfer, "q_error": q_err(t_warm_ms, pred_t_ds_transfer)},
                    {"model": "Cross-Engine GBDT (Trained on DuckDB)", "pred_ms": pred_t_gbdt_transfer, "q_error": q_err(t_warm_ms, pred_t_gbdt_transfer)},
                    {"model": "Cross-Engine Tree-NN (Trained on DuckDB)", "pred_ms": pred_t_tree_transfer, "q_error": q_err(t_warm_ms, pred_t_tree_transfer)},
                    {"model": "Trino In-Engine GBDT Oracle", "pred_ms": pred_t_oracle, "q_error": q_err(t_warm_ms, pred_t_oracle)},
                ],
            },
        }


LIVE_CTX = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Learned Cost Estimation — DuckDB & Trino Live Workbench</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; }
    .glass { background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(148, 163, 184, 0.18); backdrop-filter: blur(10px); }
    .badge-train { background: #1e3a8a; color: #93c5fd; }
    .badge-val { background: #78350f; color: #fcd34d; }
    .badge-test { background: #064e3b; color: #6ee7b7; }
    pre { white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body class="min-h-screen pb-16">
  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/90 sticky top-0 z-30">
    <div class="max-w-7xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
      <div>
        <span class="text-xs font-semibold uppercase tracking-widest text-teal-400">B.Tech Final Year Project • Dept. of Mathematics & Computing</span>
        <h1 class="text-xl md:text-2xl font-bold text-white mt-0.5">Optimisation of Query Execution Plans via Learned Cost Estimation</h1>
        <p class="text-xs text-slate-400">Cross-Engine Transfer Study (DuckDB → Trino over Apache Iceberg/Parquet) • Template-Disjoint Evaluation</p>
      </div>
      <div class="flex items-center gap-3">
        <span class="px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">● 6-Table Store Live</span>
        <span id="meta-badge" class="px-3 py-1.5 rounded-full text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700">Seed: 42</span>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 mt-6 space-y-8">
    <!-- Top KPI Cards -->
    <section class="grid grid-cols-2 md:grid-cols-6 gap-4" id="kpi-cards"></section>

    <!-- Navigation Tabs -->
    <div class="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
      <button onclick="switchTab('overview')" id="tab-overview" class="px-4 py-2 rounded-lg text-sm font-semibold bg-teal-600 text-white">1. Headline Results & Transfer</button>
      <button onclick="switchTab('live')" id="tab-live" class="px-4 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700">2. ⚡ Live SQL Cost & Plan Predictor</button>
      <button onclick="switchTab('explorer')" id="tab-explorer" class="px-4 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700">3. 6-Table Plan Tree & 5-Run Inspector</button>
      <button onclick="switchTab('bonus')" id="tab-bonus" class="px-4 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700">4. Ablations, Leakage & Cardinality Bonus</button>
      <button onclick="switchTab('figures')" id="tab-figures" class="px-4 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700">5. Publication Figures (1–5)</button>
    </div>

    <!-- TAB LIVE: REAL-TIME SQL PREDICTOR -->
    <div id="panel-live" class="hidden space-y-6">
      <div class="glass rounded-xl p-6 space-y-4 border-teal-500/40">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 class="text-lg font-bold text-white">⚡ Live Ad-Hoc SQL Plan Extractor, 5-Run Timer & Cross-Engine Cost Estimator</h3>
            <p class="text-xs text-slate-400">Type any SQL query over the TPC-H tables (`lineitem`, `orders`, `customer`, `part`, `partsupp`, `supplier`, `nation`, `region`), execute 5 live runs on DuckDB & Trino/Iceberg, inspect the live plan trees, and score every learned model on the spot.</p>
          </div>
          <div class="flex gap-2">
            <button onclick="setPresetSQL(0)" class="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-teal-300 border border-slate-700">Preset 1: Correlated Scan</button>
            <button onclick="setPresetSQL(1)" class="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-teal-300 border border-slate-700">Preset 2: 3-Table Join</button>
            <button onclick="setPresetSQL(2)" class="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-teal-300 border border-slate-700">Preset 3: 5-Table Global Trade</button>
          </div>
        </div>
        <textarea id="live-sql-input" rows="4" class="w-full bg-slate-950 text-teal-300 font-mono text-xs p-3.5 rounded-lg border border-slate-700 focus:outline-none focus:border-teal-400">SELECT c.c_mktsegment, l.l_shipmode, COUNT(*) AS deliveries, SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_revenue
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_mktsegment = 'AUTOMOBILE' AND o.o_orderdate >= DATE '1995-01-01' AND l.l_quantity <= 30
GROUP BY c.c_mktsegment, l.l_shipmode</textarea>
        <div class="flex items-center gap-4">
          <button onclick="runLiveSQL()" id="btn-run-live" class="px-5 py-2.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-sm transition">▶ Run 5x on Both Engines & Score Models</button>
          <span id="live-status" class="text-xs text-slate-400"></span>
        </div>
        <div id="live-result-box" class="space-y-4 pt-2"></div>
      </div>
    </div>

    <!-- TAB 1: OVERVIEW -->
    <div id="panel-overview" class="space-y-6">
      <div class="grid md:grid-cols-3 gap-6">
        <div class="glass rounded-xl p-5">
          <h3 class="text-sm font-semibold text-teal-400 uppercase tracking-wider">In-Engine: DuckDB → DuckDB</h3>
          <p class="text-xs text-slate-400 mb-3">Evaluated on 6 held-out templates (T19–T24, 72 unseen queries)</p>
          <div id="table-duckdb" class="overflow-x-auto"></div>
        </div>
        <div class="glass rounded-xl p-5 border-teal-500/40">
          <h3 class="text-sm font-semibold text-amber-400 uppercase tracking-wider">Cross-Engine: DuckDB → Trino (Headline)</h3>
          <p class="text-xs text-slate-400 mb-3">Trained on DuckDB plans/timings, evaluated on Trino held-out templates</p>
          <div id="table-transfer" class="overflow-x-auto"></div>
        </div>
        <div class="glass rounded-xl p-5">
          <h3 class="text-sm font-semibold text-purple-400 uppercase tracking-wider">In-Engine Oracle: Trino → Trino</h3>
          <p class="text-xs text-slate-400 mb-3">Upper-bound reference when trained directly on expensive Trino runs</p>
          <div id="table-trino" class="overflow-x-auto"></div>
        </div>
      </div>

      <div class="grid md:grid-cols-2 gap-6">
        <div class="glass rounded-xl p-5">
          <h3 class="text-base font-bold text-white mb-1">Per-Template Cross-Engine Transfer Breakdown (T19–T24)</h3>
          <p class="text-xs text-slate-400 mb-4">Includes honest reporting of where cross-engine transfer wins vs. where distributed 6-table shuffle (T24) degrades</p>
          <canvas id="chartPerTemplate" height="190"></canvas>
        </div>
        <div class="glass rounded-xl p-5">
          <h3 class="text-base font-bold text-white mb-1">Top Learned Plan Features (GBDT Importance)</h3>
          <p class="text-xs text-slate-400 mb-4">Which plan-tree properties drive the learned cost model's predictions</p>
          <canvas id="chartFeatures" height="190"></canvas>
        </div>
      </div>
    </div>

    <!-- TAB 2: PLAN TREE & QUERY EXPLORER -->
    <div id="panel-explorer" class="hidden space-y-6">
      <div class="glass rounded-xl p-5">
        <div class="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h3 class="text-base font-bold text-white">6-Table Linked Query, Plan Tree & 5-Run Execution Inspector</h3>
            <p class="text-xs text-slate-400">Select any template and query variation to inspect its SQL, DuckDB & Trino operator nodes (`est_rows` vs `actual_rows`), and all 5 timed runs.</p>
          </div>
          <div class="flex items-center gap-3">
            <select id="select-query" onchange="loadQueryDetails(this.value)" class="bg-slate-800 border border-slate-600 text-white text-sm rounded-lg px-3 py-2"></select>
          </div>
        </div>

        <div id="query-detail-box" class="space-y-5"></div>
      </div>
    </div>

    <!-- TAB 3: ABLATIONS, LEAKAGE & BONUS -->
    <div id="panel-bonus" class="hidden space-y-6">
      <div class="grid md:grid-cols-2 gap-6">
        <div class="glass rounded-xl p-5">
          <h3 class="text-base font-bold text-white mb-1">Feature-Group Ablation Study (Section 8)</h3>
          <p class="text-xs text-slate-400 mb-3">Removing each feature group in turn to measure what actually contributes</p>
          <div id="table-ablations" class="overflow-x-auto"></div>
        </div>
        <div class="glass rounded-xl p-5">
          <h3 class="text-base font-bold text-white mb-1">'A Free Bonus' (Section 7): Cardinality Q-Error by Operator</h3>
          <p class="text-xs text-slate-400 mb-3">Comparing `est_rows` vs `actual_rows` across 7,080 plan nodes</p>
          <div id="table-cardinality" class="overflow-x-auto"></div>
        </div>
      </div>
      <div class="grid md:grid-cols-2 gap-6">
        <div class="glass rounded-xl p-5">
          <h3 class="text-base font-bold text-white mb-1">Template-Disjoint Split vs Random Split Leakage (Section 5)</h3>
          <p class="text-xs text-slate-400 mb-3">Why splitting by query instead of template produces artificially optimistic Q-errors</p>
          <div id="table-leakage" class="overflow-x-auto"></div>
        </div>
        <div class="glass rounded-xl p-5">
          <h3 class="text-base font-bold text-white mb-1">5-Run Timing Variance & Cold-Cache Filtering (Section 7)</h3>
          <p class="text-xs text-slate-400 mb-3">Why storing all 5 runs and discarding `is_cold=1` reduces measurement noise</p>
          <div id="table-variance" class="overflow-x-auto"></div>
        </div>
      </div>
    </div>

    <!-- TAB 4: FIGURES -->
    <div id="panel-figures" class="hidden space-y-6">
      <div class="grid md:grid-cols-1 gap-6" id="figures-gallery"></div>
    </div>
  </main>

  <script>
    let globalMetrics = null;
    let chartPerTemplateInstance = null;
    let chartFeaturesInstance = null;

    function switchTab(tab) {
      ['overview', 'live', 'explorer', 'bonus', 'figures'].forEach(t => {
        document.getElementById('panel-' + t).classList.toggle('hidden', t !== tab);
        const btn = document.getElementById('tab-' + t);
        if (t === tab) {
          btn.className = 'px-4 py-2 rounded-lg text-sm font-semibold bg-teal-600 text-white';
        } else {
          btn.className = 'px-4 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-slate-300 hover:bg-slate-700';
        }
      });
    }

    const SQL_PRESETS = [
      `SELECT l_shipmode, l_returnflag, COUNT(*) AS n_lines, SUM(l_extendedprice * (1 - l_discount)) AS net_rev\nFROM lineitem\nWHERE l_shipdate >= DATE '1995-06-01' AND l_discount >= 0.06 AND l_quantity >= 25\nGROUP BY l_shipmode, l_returnflag`,
      `SELECT c.c_mktsegment, l.l_shipmode, COUNT(*) AS deliveries, SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_revenue\nFROM customer c\nJOIN orders o ON c.c_custkey = o.o_custkey\nJOIN lineitem l ON o.o_orderkey = l.l_orderkey\nWHERE c.c_mktsegment = 'AUTOMOBILE' AND o.o_orderdate >= DATE '1995-01-01' AND l.l_quantity <= 30\nGROUP BY c.c_mktsegment, l.l_shipmode`,
      `SELECT n.n_name, p.p_container, COUNT(*) AS line_volume, SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_sales\nFROM part p\nJOIN lineitem l ON p.p_partkey = l.l_partkey\nJOIN orders o ON l.l_orderkey = o.o_orderkey\nJOIN customer c ON o.o_custkey = c.c_custkey\nJOIN nation n ON c.c_nationkey = n.n_nationkey\nWHERE p.p_container = 'SM CASE' AND o.o_orderdate >= DATE '1995-01-01' AND l.l_quantity >= 15\nGROUP BY n.n_name, p.p_container`
    ];

    function setPresetSQL(idx) {
      document.getElementById('live-sql-input').value = SQL_PRESETS[idx];
    }

    async function runLiveSQL() {
      const sql = document.getElementById('live-sql-input').value;
      const status = document.getElementById('live-status');
      const box = document.getElementById('live-result-box');
      status.textContent = 'Running 5 timed executions on DuckDB and Trino/Iceberg & extracting plan features...';
      try {
        const res = await fetch('/api/run_live?sql=' + encodeURIComponent(sql));
        const d = await res.json();
        if (d.error) {
          status.textContent = 'Error: ' + d.error;
          return;
        }
        status.textContent = 'Completed 5 runs per engine (Run 0 cold discarded, median of Runs 1..4 kept).';
        let html = `<div class="grid md:grid-cols-2 gap-6">`;
        for (const [eng, label, col] of [['duckdb', 'DuckDB (Embedded Source Engine)', 'teal'], ['trino', 'Trino + Iceberg (Distributed Target Engine)', 'purple']]) {
          const info = d[eng];
          html += `<div class="bg-slate-900/90 rounded-xl p-4 border border-slate-700 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm font-bold text-${col}-400 uppercase">${label}</span>
              <span class="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 text-xs font-bold">Actual Warm Median: ${info.actual_warm_ms.toFixed(2)} ms</span>
            </div>
            <div class="text-xs text-slate-300"><b>5 Timed Runs:</b> ` +
              info.executions.map(e => `<span class="inline-block px-2 py-0.5 mr-1 rounded ${e.is_cold ? 'bg-rose-900/60 text-rose-300' : 'bg-slate-800 text-slate-200'}">R${e.run_index}: ${e.wall_ms.toFixed(2)}ms</span>`).join('') +
            `</div>
            <table class="w-full text-left text-xs border-collapse">
              <thead><tr class="border-b border-slate-700 text-slate-400">
                <th class="py-1.5">Cost Estimator</th>
                <th class="py-1.5 text-right">Predicted (ms)</th>
                <th class="py-1.5 text-right">Live Q-Error</th>
              </tr></thead><tbody>` +
              info.predictions.map(p => `<tr class="border-b border-slate-800/70">
                <td class="py-1.5 font-medium text-slate-200">${p.model}</td>
                <td class="py-1.5 text-right font-mono text-amber-300">${p.pred_ms.toFixed(2)} ms</td>
                <td class="py-1.5 text-right font-bold ${p.q_error < 1.5 ? 'text-emerald-400' : 'text-sky-300'}">${p.q_error.toFixed(3)}</td>
              </tr>`).join('') +
            `</tbody></table>
          </div>`;
        }
        html += `</div>`;
        box.innerHTML = html;
      } catch (err) {
        status.textContent = 'Request failed: ' + err;
      }
    }

    function renderRegimeTable(containerId, dataObj) {
      let html = `<table class="w-full text-left text-xs border-collapse">
        <thead>
          <tr class="border-b border-slate-700 text-slate-400">
            <th class="py-2">Model / Baseline</th>
            <th class="py-2 text-right">Med Q</th>
            <th class="py-2 text-right">P90 Q</th>
            <th class="py-2 text-right">Spearman ρ</th>
            <th class="py-2 text-right">Rank Acc</th>
          </tr>
        </thead><tbody>`;
      for (const [model, m] of Object.entries(dataObj)) {
        const isBest = model.includes('Gradient') || model.includes('Tree');
        html += `<tr class="border-b border-slate-800/70 ${isBest ? 'text-teal-300 font-semibold' : 'text-slate-300'}">
          <td class="py-2 pr-2">${model}</td>
          <td class="py-2 text-right">${m.median_q_error.toFixed(2)}</td>
          <td class="py-2 text-right">${m.p90_q_error.toFixed(2)}</td>
          <td class="py-2 text-right">${m.spearman_rho.toFixed(3)}</td>
          <td class="py-2 text-right">${m.pairwise_rank_acc.toFixed(1)}%</td>
        </tr>`;
      }
      html += `</tbody></table>`;
      document.getElementById(containerId).innerHTML = html;
    }

    async function initDashboard() {
      const res = await fetch('/api/metrics');
      globalMetrics = await res.json();

      const tc = globalMetrics.metadata.table_counts;
      const kpis = [
        { label: 'TEMPLATES', val: tc.TEMPLATES, sub: '15 Train / 3 Val / 6 Test' },
        { label: 'QUERIES', val: tc.QUERIES, sub: '12 Variants / Template' },
        { label: 'PLANS', val: tc.PLANS, sub: 'DuckDB + Trino Plans' },
        { label: 'PLAN_NODES', val: tc.PLAN_NODES, sub: 'est_rows + actual_rows' },
        { label: 'EXECUTIONS', val: tc.EXECUTIONS, sub: '5 Timed Runs / Plan' },
        { label: 'FEATURES', val: tc.FEATURES, sub: 'Flat (32D) + Tree-GRU' },
      ];
      document.getElementById('kpi-cards').innerHTML = kpis.map(k => `
        <div class="glass rounded-xl p-4">
          <div class="text-xs font-bold text-teal-400 tracking-wider">${k.label}</div>
          <div class="text-2xl font-extrabold text-white mt-1">${k.val.toLocaleString()}</div>
          <div class="text-[11px] text-slate-400 mt-0.5">${k.sub}</div>
        </div>
      `).join('');

      renderRegimeTable('table-duckdb', globalMetrics.duckdb_in_engine);
      renderRegimeTable('table-transfer', globalMetrics.cross_engine_calibrated);
      renderRegimeTable('table-trino', globalMetrics.trino_in_engine);

      // Per-Template Chart
      const pt = globalMetrics.per_template_transfer;
      const ctx1 = document.getElementById('chartPerTemplate').getContext('2d');
      chartPerTemplateInstance = new Chart(ctx1, {
        type: 'bar',
        data: {
          labels: pt.map(x => `${x.template_id} (${x.n_relations} rel)`),
          datasets: [
            { label: 'Trino Native Cost', data: pt.map(x => x.native_median_q), backgroundColor: '#f97316' },
            { label: 'GBDT Transfer (DuckDB→Trino)', data: pt.map(x => x.gbdt_transfer_median_q), backgroundColor: '#14b8a6' },
            { label: 'Tree-NN Transfer (DuckDB→Trino)', data: pt.map(x => x.tree_transfer_median_q), backgroundColor: '#ec4899' }
          ]
        },
        options: {
          plugins: { legend: { labels: { color: '#cbd5e1' } } },
          scales: {
            x: { ticks: { color: '#cbd5e1' } },
            y: { ticks: { color: '#cbd5e1' }, title: { display: true, text: 'Median Q-Error (1.0 = Ideal)', color: '#94a3b8' } }
          }
        }
      });

      // Top Features Chart
      const tf = globalMetrics.top_gbdt_features;
      const ctx2 = document.getElementById('chartFeatures').getContext('2d');
      chartFeaturesInstance = new Chart(ctx2, {
        type: 'bar',
        data: {
          labels: tf.map(x => x.feature),
          datasets: [{ label: 'Relative Importance', data: tf.map(x => x.importance), backgroundColor: '#38bdf8' }]
        },
        options: {
          indexAxis: 'y',
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#cbd5e1' } },
            y: { ticks: { color: '#cbd5e1' } }
          }
        }
      });

      // Ablations Table
      document.getElementById('table-ablations').innerHTML = `<table class="w-full text-left text-xs">
        <thead><tr class="border-b border-slate-700 text-slate-400">
          <th class="py-2">Ablation Variant</th>
          <th class="py-2 text-right">DuckDB Med Q</th>
          <th class="py-2 text-right">DuckDB P90 Q</th>
          <th class="py-2 text-right">Transfer Med Q</th>
          <th class="py-2 text-right">Transfer P90 Q</th>
        </tr></thead><tbody>` +
        globalMetrics.ablations.map(a => `<tr class="border-b border-slate-800/70">
          <td class="py-2 font-medium text-slate-200">${a.variant}</td>
          <td class="py-2 text-right">${a.duckdb_median_q.toFixed(2)}</td>
          <td class="py-2 text-right">${a.duckdb_p90_q.toFixed(2)}</td>
          <td class="py-2 text-right">${a.transfer_median_q.toFixed(2)}</td>
          <td class="py-2 text-right">${a.transfer_p90_q.toFixed(2)}</td>
        </tr>`).join('') + `</tbody></table>`;

      // Cardinality Bonus Table
      document.getElementById('table-cardinality').innerHTML = `<table class="w-full text-left text-xs">
        <thead><tr class="border-b border-slate-700 text-slate-400">
          <th class="py-2">Operator Category</th>
          <th class="py-2 text-right">Node Count</th>
          <th class="py-2 text-right">Median Card Q-Err</th>
          <th class="py-2 text-right">P90 Card Q-Err</th>
          <th class="py-2 text-right">Max Card Q-Err</th>
        </tr></thead><tbody>` +
        globalMetrics.cardinality_bonus_by_operator.map(c => `<tr class="border-b border-slate-800/70">
          <td class="py-2 font-semibold text-amber-300">${c.op_cat}</td>
          <td class="py-2 text-right">${c.count}</td>
          <td class="py-2 text-right">${c.median_q.toFixed(2)}</td>
          <td class="py-2 text-right">${c.p90_q.toFixed(2)}</td>
          <td class="py-2 text-right">${c.max_q.toFixed(1)}</td>
        </tr>`).join('') + `</tbody></table>`;

      // Leakage Table
      const lk = globalMetrics.split_leakage_comparison;
      document.getElementById('table-leakage').innerHTML = `<table class="w-full text-left text-xs">
        <thead><tr class="border-b border-slate-700 text-slate-400">
          <th class="py-2">Model</th>
          <th class="py-2 text-right">Random Split Med Q (Leaked)</th>
          <th class="py-2 text-right">Random Split P90 Q</th>
          <th class="py-2 text-right">Template-Disjoint Med Q (Honest)</th>
          <th class="py-2 text-right">Template-Disjoint P90 Q</th>
        </tr></thead><tbody>` +
        Object.keys(lk.random_split).map(m => `<tr class="border-b border-slate-800/70">
          <td class="py-2 font-medium text-slate-200">${m}</td>
          <td class="py-2 text-right text-sky-400">${lk.random_split[m].median_q_error.toFixed(2)}</td>
          <td class="py-2 text-right text-sky-400">${lk.random_split[m].p90_q_error.toFixed(2)}</td>
          <td class="py-2 text-right text-amber-300 font-semibold">${lk.template_disjoint[m].median_q_error.toFixed(2)}</td>
          <td class="py-2 text-right text-amber-300 font-semibold">${lk.template_disjoint[m].p90_q_error.toFixed(2)}</td>
        </tr>`).join('') + `</tbody></table>`;

      // Variance Table
      const vs = globalMetrics.timing_variance_summary;
      document.getElementById('table-variance').innerHTML = `<table class="w-full text-left text-xs">
        <thead><tr class="border-b border-slate-700 text-slate-400">
          <th class="py-2">Engine</th>
          <th class="py-2 text-right">CV% (All 5 Runs incl. Cold)</th>
          <th class="py-2 text-right">CV% (Warm Runs 1..4)</th>
          <th class="py-2 text-right">Cold Run Penalty Ratio</th>
        </tr></thead><tbody>
        <tr class="border-b border-slate-800/70">
          <td class="py-2 font-semibold text-teal-300">DuckDB</td>
          <td class="py-2 text-right">${vs.duckdb_cv_all_pct.toFixed(2)}%</td>
          <td class="py-2 text-right text-emerald-400 font-bold">${vs.duckdb_cv_warm_pct.toFixed(2)}%</td>
          <td class="py-2 text-right">${vs.duckdb_cold_speedup_ratio.toFixed(2)}x</td>
        </tr>
        <tr class="border-b border-slate-800/70">
          <td class="py-2 font-semibold text-purple-300">Trino + Iceberg</td>
          <td class="py-2 text-right">${vs.trino_cv_all_pct.toFixed(2)}%</td>
          <td class="py-2 text-right text-emerald-400 font-bold">${vs.trino_cv_warm_pct.toFixed(2)}%</td>
          <td class="py-2 text-right">${vs.trino_cold_speedup_ratio.toFixed(2)}x</td>
        </tr>
        </tbody></table>`;

      // Figures Gallery
      const figs = [
        { file: 'fig1_q_error_comparison.png', title: 'Figure 1: Headline Q-Error Comparison (In-Engine & Cross-Engine Transfer)' },
        { file: 'fig2_cross_engine_ranking.png', title: 'Figure 2: Cross-Engine Predicted vs Actual Trino Runtimes on Held-Out Templates' },
        { file: 'fig3_template_vs_random_leakage.png', title: 'Figure 3: Template-Disjoint Leakage Proof & Per-Template Breakdown' },
        { file: 'fig4_cardinality_q_error_by_operator.png', title: 'Figure 4: Free Bonus — Node Cardinality Q-Error Across Operators & Tree Depth' },
        { file: 'fig5_ablation_and_timing_variance.png', title: 'Figure 5: Feature Ablations & 5-Run Cold-Cache Noise Reduction' }
      ];
      document.getElementById('figures-gallery').innerHTML = figs.map(f => `
        <div class="glass rounded-xl p-5">
          <h4 class="text-sm font-bold text-white mb-3">${f.title}</h4>
          <img src="/figures/${f.file}" class="w-full rounded-lg border border-slate-700 bg-white p-2" />
        </div>
      `).join('');

      // Load query list
      const qRes = await fetch('/api/queries');
      const qList = await qRes.json();
      const sel = document.getElementById('select-query');
      sel.innerHTML = qList.map(q => `<option value="${q.query_id}">${q.query_id} [${q.split_fold.toUpperCase()}] — ${q.source} (${q.n_relations} tables)</option>`).join('');
      if (qList.length > 0) {
        loadQueryDetails(qList[qList.length - 1].query_id);
      }
    }

    async function loadQueryDetails(queryId) {
      document.getElementById('select-query').value = queryId;
      const res = await fetch('/api/query_detail?query_id=' + encodeURIComponent(queryId));
      const d = await res.json();

      const badgeClass = d.query.split_fold === 'test' ? 'badge-test' : (d.query.split_fold === 'val' ? 'badge-val' : 'badge-train');
      let html = `
        <div class="bg-slate-900/90 rounded-lg p-4 border border-slate-700">
          <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
            <span class="text-sm font-bold text-white">${d.query.query_id} — ${d.query.description}</span>
            <span class="px-2.5 py-0.5 rounded text-xs font-bold uppercase ${badgeClass}">Split: ${d.query.split_fold} (${d.query.n_relations} Relations)</span>
          </div>
          <pre class="text-xs text-teal-300 font-mono bg-slate-950 p-3 rounded border border-slate-800">${d.query.sql_text}</pre>
        </div>
        <div class="grid md:grid-cols-2 gap-6">
      `;

      for (const eng of ['duckdb', 'trino']) {
        const p = d.plans[eng];
        html += `
          <div class="bg-slate-900/75 rounded-lg p-4 border border-slate-700/80 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm font-bold uppercase tracking-wider ${eng === 'duckdb' ? 'text-teal-400' : 'text-purple-400'}">${eng} Plan (${p.plan_id})</span>
              <span class="text-xs text-slate-400">Native Cost: <b>${p.native_est_cost.toFixed(1)}</b></span>
            </div>
            <div class="text-xs text-slate-300">
              <b>5 Timed Runs (ms):</b> ` +
              p.executions.map(e => `<span class="inline-block px-2 py-0.5 mr-1 rounded ${e.is_cold ? 'bg-rose-900/60 text-rose-300' : 'bg-emerald-900/50 text-emerald-300'}">Run ${e.run_index}${e.is_cold ? ' (cold)' : ''}: ${e.wall_ms.toFixed(2)}ms</span>`).join('') +
            `</div>
            <div class="text-xs font-semibold text-slate-400 pt-1">Plan Tree Nodes (PLAN_NODES):</div>
            <div class="overflow-x-auto max-h-60">
              <table class="w-full text-left text-[11px]">
                <thead><tr class="border-b border-slate-700 text-slate-400">
                  <th class="py-1">Depth</th>
                  <th class="py-1">Operator</th>
                  <th class="py-1">Table</th>
                  <th class="py-1 text-right">Est Rows</th>
                  <th class="py-1 text-right">Actual Rows</th>
                </tr></thead><tbody>` +
                p.nodes.map(n => `<tr class="border-b border-slate-800/60">
                  <td class="py-1">${'&nbsp;'.repeat(n.depth * 2)}L${n.depth}</td>
                  <td class="py-1 font-mono text-slate-200">${n.operator}</td>
                  <td class="py-1 text-slate-400">${n.table_name || '—'}</td>
                  <td class="py-1 text-right text-amber-300">${Math.round(n.est_rows).toLocaleString()}</td>
                  <td class="py-1 text-right text-emerald-300">${Math.round(n.actual_rows).toLocaleString()}</td>
                </tr>`).join('') +
              `</tbody></table>
            </div>
          </div>
        `;
      }
      html += `</div>`;
      document.getElementById('query-detail-box').innerHTML = html;
    }

    initDashboard();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, payload, status=200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/healthz":
            self._send_json({"status": "ok", "service": "learned-cost-estimation"})
            return

        if path == "/" or path == "/index.html":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/metrics":
            data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            self._send_json(data)
            return

        if path == "/api/queries":
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT q.query_id, q.template_id, t.source, t.n_relations, t.split_fold
                FROM QUERIES q
                JOIN TEMPLATES t ON q.template_id = t.template_id
                ORDER BY q.query_id
                """
            ).fetchall()
            conn.close()
            self._send_json([dict(r) for r in rows])
            return

        if path == "/api/query_detail":
            qs = parse_qs(parsed.query)
            qid = qs.get("query_id", ["T19_Q001"])[0]
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            q_row = conn.execute(
                """
                SELECT q.*, t.source, t.n_relations, t.split_fold, t.description
                FROM QUERIES q
                JOIN TEMPLATES t ON q.template_id = t.template_id
                WHERE q.query_id = ?
                """,
                (qid,),
            ).fetchone()

            plans_out = {}
            for eng in ["duckdb", "trino"]:
                p_row = conn.execute(
                    "SELECT plan_id, engine, native_est_cost FROM PLANS WHERE query_id = ? AND engine = ?",
                    (qid, eng),
                ).fetchone()
                pid = p_row["plan_id"]
                nodes = conn.execute(
                    "SELECT node_id, operator, est_rows, actual_rows, depth, table_name FROM PLAN_NODES WHERE plan_id = ? ORDER BY node_id",
                    (pid,),
                ).fetchall()
                execs = conn.execute(
                    "SELECT run_index, wall_ms, is_cold FROM EXECUTIONS WHERE plan_id = ? ORDER BY run_index",
                    (pid,),
                ).fetchall()
                plans_out[eng] = {
                    "plan_id": pid,
                    "native_est_cost": p_row["native_est_cost"],
                    "nodes": [dict(n) for n in nodes],
                    "executions": [dict(e) for e in execs],
                }
            conn.close()
            self._send_json({"query": dict(q_row), "plans": plans_out})
            return

        if path == "/api/run_live":
            qs = parse_qs(parsed.query)
            sql_text = qs.get("sql", [""])[0]
            try:
                res = LIVE_CTX.run_live_sql(sql_text)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path.startswith("/figures/"):
            fname = path.split("/figures/")[-1]
            fpath = FIG_DIR / fname
            if fpath.exists():
                raw = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

        self.send_error(404, "Not Found")


if __name__ == "__main__":
    if not DB_PATH.exists() or not METRICS_PATH.exists():
        print("Dataset or metrics missing — bootstrapping end-to-end pipeline first...")
        from run_pipeline import run_pipeline
        run_pipeline(base_dir=PROJECT_ROOT, variants_per_template=12, scale_factor=0.05, seed=42)
    print("Initializing Live DuckDB + Trino/Iceberg Cost Estimation Models...")
    LIVE_CTX = LivePredictorContext()
    port = int(os.environ.get("PORT", 8501))
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Live Learned Cost Estimation Workbench running at http://0.0.0.0:{port}")
    server.serve_forever()

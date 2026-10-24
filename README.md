# Optimisation of Query Execution Plans via Learned Cost Estimation

**A cross-engine study on DuckDB and Trino, with plan-tree features and template-disjoint evaluation**

B.Tech Final Year Project — Department of Mathematics and Computing

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Yadul12/learned-cost-estimation)
[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/new/template?template=https://github.com/Yadul12/learned-cost-estimation)

---

## 1. Overview & One-Sentence Goal

> **Learn a cost model from a lightweight embedded engine (DuckDB) and test whether it transfers to a heavier distributed engine (Trino) running on the exact same data, judged against the database's own built-in cost estimate under strict template-disjoint evaluation.**

Traditional query optimizers rely on hand-written arithmetic formulas and simplifying assumptions (such as attribute value independence and uniform data distributions) to estimate plan execution costs. When columns are correlated or multiple tables are joined, cardinality errors compound across the plan tree by orders of magnitude, causing the optimizer to misrank candidate plans. While learned cost models can replace these heuristics with models trained on real runtime measurements, collecting thousands of timed query executions on a distributed engine like **Trino** is slow and expensive.

This repository implements the complete end-to-end pipeline proposed in the project specification:
1. **Shared Data Layer**: Generates TPC-H data with realistic cross-column correlations stored in shared Apache Iceberg / Parquet format read identically by both **DuckDB** and **Trino**.
2. **Template-Disjoint Workload Generation**: Defines 24 distinct analytical SQL query templates (1-table to 6-table joins, aggregations, filters, and sorts) and generates 288 parameterized SQL queries. Crucially, the train/validation/test split (`split_fold`) is assigned on the **`TEMPLATES`** table at the top of the schema hierarchy (`15 train`, `3 val`, `6 held-out test` templates), guaranteeing zero template leakage between training and evaluation.
3. **Six-Table Linked Experiment Store**: Stores all artifacts in a normalized 6-table relational schema (`TEMPLATES`, `QUERIES`, `PLANS`, `PLAN_NODES`, `EXECUTIONS`, `FEATURES`) backed by SQLite and exported to both CSV and Parquet.
4. **Five-Run Timing Protocol with Cold-Run Filtering**: Executes every query 5 times (`run_index = 0..4`), explicitly tagging `run_index = 0` with `is_cold = 1` and taking the median of warm runs (`1..4`) while preserving all individual runs to quantify timing variance.
5. **Dual Model Families & Three Baselines**:
   - **Baselines**: (1) Engine's Native Optimizer Cost Estimate (`NativeCBOEstimator`), (2) Predicting from Data Size Alone (`DataSizeBaseline` — an intentionally simple baseline using only scanned rows and table count), and (3) Regularized Linear Ridge Regression (`LinearCostBaseline`).
   - **Gradient Boosting on Flattened Features (`GradientBoostingCostModel`)**: XGBoost regressor trained in logarithmic runtime space on 32 engine-agnostic plan features (canonical operator counts, tree topology metrics, log estimated cardinalities, and scanned table footprint).
   - **Tree-Structured Neural Model (`TreeNeuralCostModel`)**: PyTorch bottom-up Child-Sum Tree-GRU / Tree-MLP architecture that reads the hierarchical execution plan tree directly.
6. **Honest Evaluation, Cross-Engine Transfer, Ablations & "Free Bonus" Cardinality Analysis**:
   - Evaluates **In-Engine (DuckDB → DuckDB)**, **In-Engine Oracle (Trino → Trino)**, and **Cross-Engine Transfer (Trained on DuckDB → Evaluated on Trino)** using Median, 90th-percentile (P90), and 95th-percentile (P95) **Q-error**, plus **Spearman rank correlation ($\rho$)**, **Kendall's $\tau$**, and **Pairwise Plan Ranking Accuracy (%)**.
   - Quantifies the exact optimism bias of **Random Query Splits vs. Template-Disjoint Splits**.
   - Performs systematic **Feature-Group Ablations** (removing cardinality estimates, operator counts, tree topology, or table footprint in turn).
   - Analyzes `est_rows` vs. `actual_rows` across all `PLAN_NODES` ("A Free Bonus", Section 7) to show where native cardinality estimators break down.
   - Reports an honest failure/limitation case on high-join-count distributed shuffle templates (`T24`).

---

## 2. Repository Structure

```text
learned-cost-estimation/
├── docker-compose.yml               # Trino + Apache Iceberg container stack
├── docker/
│   └── trino/catalog/
│       └── iceberg.properties       # Shared Iceberg catalog configuration
├── requirements.txt                 # Pinned Python dependencies
├── run_pipeline.py                  # Single command that reproduces the entire study
├── src/
│   ├── storage/
│   │   └── schema.py                # 6-table relational schema (SQLite + CSV/Parquet export)
│   ├── workload/
│   │   └── generator.py             # Correlated TPC-H generator & 24 SQL templates (T01..T24)
│   ├── engines/
│   │   └── runner.py                # DuckDB & Trino/Iceberg plan extractors and 5-run timers
│   ├── features/
│   │   └── extractor.py             # Flattened feature vector & recursive tree encoding extractor
│   ├── models/
│   │   └── estimators.py            # Native CBO, Data-Size baseline, GBDT, and Tree-GRU Neural Net
│   └── evaluation/
│       └── plots.py                 # Generates Figures 1–5 for the report
├── tests/
│   └── test_pipeline.py             # Pytest suite verifying schema, zero leakage, and metrics
├── data/
│   ├── experiment_store.sqlite      # Populated 6-table relational database
│   ├── tables/                      # CSV & Parquet exports of all 6 tables
│   └── warehouse/                   # Shared TPC-H Parquet/Iceberg data files
└── results/
    ├── evaluation_metrics.json      # Complete reproducible numerical results
    └── figures/                     # Publication-grade plots (fig1 .. fig5)
```

---

## 3. Reproducing Everything in One Command

From the project root (`learned-cost-estimation`):

```powershell
# 1. Run the complete end-to-end pipeline (Fixed Seed = 42)
.\.venv\Scripts\python.exe run_pipeline.py --variants 12 --sf 0.05 --seed 42

# 2. Run the automated verification test suite
.\.venv\Scripts\pytest.exe -v
```

*(Optional)* To start the live Trino Docker container stack alongside the shared warehouse:
```powershell
docker compose up -d
```

---

## 4. Dataset Schema (The 6 Linked Tables)

| Table | Primary Key | Foreign Key | Purpose & Key Columns |
| :--- | :--- | :--- | :--- |
| **`TEMPLATES`** | `template_id` | — | Stores the 24 query templates (`T01`–`T24`), `source`, `n_relations` (1–6), and **`split_fold`** (`train`, `val`, `test`). |
| **`QUERIES`** | `query_id` | `template_id` | Stores every generated SQL variation (`sql_text`) and its deterministic RNG `seed`. Inherits `split_fold` from `TEMPLATES`. |
| **`PLANS`** | `plan_id` | `query_id` | One row per query per engine (`duckdb`, `trino`), storing raw `plan_json` and `native_est_cost`. |
| **`PLAN_NODES`** | `node_id` | `plan_id` | Every operator node inside a plan tree: `operator`, `est_rows`, `actual_rows`, `parent_node_id`, `depth`, `table_name`. |
| **`EXECUTIONS`** | `exec_id` | `plan_id` | All 5 timed runs per plan (`run_index` 0..4, `wall_ms`, `is_cold` = 1 for run 0 and 0 for runs 1..4). |
| **`FEATURES`** | `plan_id` | `plan_id` | Pre-extracted `flat_vector` (32 features), recursive `tree_encoding`, and `extractor_version`. |

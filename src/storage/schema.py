"""
Database schema and storage layer for Learned Cost Estimation.
Implements the 6 linked tables described in Section 7 (Figure 2) of the proposal:
1. TEMPLATES (template_id PK, source, n_relations, split_fold)
2. QUERIES (query_id PK, template_id FK, sql_text, seed)
3. PLANS (plan_id PK, query_id FK, engine, plan_json, native_est_cost)
4. PLAN_NODES (node_id PK, plan_id FK, operator, est_rows, actual_rows, parent_node_id, depth, table_name, extra_json)
5. EXECUTIONS (exec_id PK, plan_id FK, wall_ms, run_index, is_cold)
6. FEATURES (plan_id PK, flat_vector, tree_encoding, extractor_version)
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS TEMPLATES (
    template_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    n_relations INTEGER NOT NULL,
    split_fold TEXT NOT NULL CHECK (split_fold IN ('train', 'val', 'test')),
    description TEXT
);

CREATE TABLE IF NOT EXISTS QUERIES (
    query_id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    sql_text TEXT NOT NULL,
    seed INTEGER NOT NULL,
    FOREIGN KEY (template_id) REFERENCES TEMPLATES(template_id)
);

CREATE TABLE IF NOT EXISTS PLANS (
    plan_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    engine TEXT NOT NULL CHECK (engine IN ('duckdb', 'trino')),
    plan_json TEXT NOT NULL,
    native_est_cost REAL NOT NULL,
    FOREIGN KEY (query_id) REFERENCES QUERIES(query_id)
);

CREATE TABLE IF NOT EXISTS PLAN_NODES (
    node_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    operator TEXT NOT NULL,
    est_rows REAL NOT NULL,
    actual_rows REAL NOT NULL,
    parent_node_id TEXT,
    depth INTEGER NOT NULL DEFAULT 0,
    table_name TEXT,
    extra_json TEXT,
    FOREIGN KEY (plan_id) REFERENCES PLANS(plan_id)
);

CREATE TABLE IF NOT EXISTS EXECUTIONS (
    exec_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    wall_ms REAL NOT NULL,
    run_index INTEGER NOT NULL,
    is_cold INTEGER NOT NULL CHECK (is_cold IN (0, 1)),
    FOREIGN KEY (plan_id) REFERENCES PLANS(plan_id)
);

CREATE TABLE IF NOT EXISTS FEATURES (
    plan_id TEXT PRIMARY KEY,
    flat_vector TEXT NOT NULL,
    tree_encoding TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES PLANS(plan_id)
);

CREATE INDEX IF NOT EXISTS idx_queries_template ON QUERIES(template_id);
CREATE INDEX IF NOT EXISTS idx_plans_query_engine ON PLANS(query_id, engine);
CREATE INDEX IF NOT EXISTS idx_nodes_plan ON PLAN_NODES(plan_id);
CREATE INDEX IF NOT EXISTS idx_exec_plan ON EXECUTIONS(plan_id);
"""


class ExperimentStore:
    """Manages SQLite persistence and Parquet exports for the 6-table experiment store."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)

    def upsert_template(
        self,
        template_id: str,
        source: str,
        n_relations: int,
        split_fold: str,
        description: str = "",
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO TEMPLATES (template_id, source, n_relations, split_fold, description)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(template_id) DO UPDATE SET
                    source=excluded.source,
                    n_relations=excluded.n_relations,
                    split_fold=excluded.split_fold,
                    description=excluded.description
                """,
                (template_id, source, n_relations, split_fold, description),
            )

    def upsert_query(
        self, query_id: str, template_id: str, sql_text: str, seed: int
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO QUERIES (query_id, template_id, sql_text, seed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_id) DO UPDATE SET
                    template_id=excluded.template_id,
                    sql_text=excluded.sql_text,
                    seed=excluded.seed
                """,
                (query_id, template_id, sql_text, seed),
            )

    def has_plan_and_executions(self, plan_id: str, expected_runs: int = 5) -> bool:
        cur = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM EXECUTIONS WHERE plan_id = ?", (plan_id,)
        )
        row = cur.fetchone()
        return bool(row and row["cnt"] >= expected_runs)

    def save_plan_bundle(
        self,
        plan_id: str,
        query_id: str,
        engine: str,
        plan_json: str,
        native_est_cost: float,
        nodes: List[Dict[str, Any]],
        executions: List[Dict[str, Any]],
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO PLANS (plan_id, query_id, engine, plan_json, native_est_cost)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    plan_json=excluded.plan_json,
                    native_est_cost=excluded.native_est_cost
                """,
                (plan_id, query_id, engine, plan_json, float(native_est_cost)),
            )
            self.conn.execute("DELETE FROM PLAN_NODES WHERE plan_id = ?", (plan_id,))
            self.conn.executemany(
                """
                INSERT INTO PLAN_NODES (
                    node_id, plan_id, operator, est_rows, actual_rows,
                    parent_node_id, depth, table_name, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        n["node_id"],
                        plan_id,
                        n["operator"],
                        float(n["est_rows"]),
                        float(n["actual_rows"]),
                        n.get("parent_node_id"),
                        int(n.get("depth", 0)),
                        n.get("table_name", ""),
                        json.dumps(n.get("extra", {})),
                    )
                    for n in nodes
                ],
            )
            self.conn.execute("DELETE FROM EXECUTIONS WHERE plan_id = ?", (plan_id,))
            self.conn.executemany(
                """
                INSERT INTO EXECUTIONS (exec_id, plan_id, wall_ms, run_index, is_cold)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        e["exec_id"],
                        plan_id,
                        float(e["wall_ms"]),
                        int(e["run_index"]),
                        1 if e["is_cold"] else 0,
                    )
                    for e in executions
                ],
            )

    def save_features(
        self,
        plan_id: str,
        flat_vector: Dict[str, float],
        tree_encoding: Dict[str, Any],
        extractor_version: str = "v1.0",
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO FEATURES (plan_id, flat_vector, tree_encoding, extractor_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    flat_vector=excluded.flat_vector,
                    tree_encoding=excluded.tree_encoding,
                    extractor_version=excluded.extractor_version
                """,
                (
                    plan_id,
                    json.dumps(flat_vector),
                    json.dumps(tree_encoding),
                    extractor_version,
                ),
            )

    def export_tables_to_csv_and_parquet(self, out_dir: Path) -> Dict[str, int]:
        import pandas as pd

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tables = [
            "TEMPLATES",
            "QUERIES",
            "PLANS",
            "PLAN_NODES",
            "EXECUTIONS",
            "FEATURES",
        ]
        counts: Dict[str, int] = {}
        for t in tables:
            df = pd.read_sql_query(f"SELECT * FROM {t}", self.conn)
            df.to_csv(out_dir / f"{t.lower()}.csv", index=False)
            df.to_parquet(out_dir / f"{t.lower()}.parquet", index=False)
            counts[t] = len(df)
        return counts

    def close(self) -> None:
        self.conn.close()

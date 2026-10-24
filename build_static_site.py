"""
Exports a static version of the interactive workbench into `docs/` so GitHub Pages
serves a permanent 24/7 public interactive site at:
https://yadul12.github.io/learned-cost-estimation/
while Render/Railway hosts the live Python + DuckDB execution backend.
"""

import json
import shutil
import sqlite3
from pathlib import Path
from app_dashboard import HTML_PAGE

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "experiment_store.sqlite"
METRICS_PATH = PROJECT_ROOT / "results" / "evaluation_metrics.json"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"


def build_static_site() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    # Copy all 5 figures
    for f in FIG_DIR.glob("*.png"):
        shutil.copy2(f, DOCS_DIR / "figures" / f.name)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    q_rows = conn.execute(
        """
        SELECT q.query_id, q.template_id, q.sql_text, q.seed, t.source, t.n_relations, t.split_fold, t.description
        FROM QUERIES q
        JOIN TEMPLATES t ON q.template_id = t.template_id
        ORDER BY q.query_id
        """
    ).fetchall()
    queries_list = [dict(r) for r in q_rows]

    details_map = {}
    for q in queries_list:
        qid = q["query_id"]
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
        details_map[qid] = {"query": q, "plans": plans_out}
    conn.close()

    (DOCS_DIR / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (DOCS_DIR / "queries.json").write_text(json.dumps(queries_list), encoding="utf-8")
    (DOCS_DIR / "query_details.json").write_text(json.dumps(details_map), encoding="utf-8")

    # Patch fetch calls in HTML_PAGE so it works seamlessly both on static GitHub Pages and live server
    static_html = HTML_PAGE.replace(
        "const res = await fetch('/api/metrics');",
        "const res = await fetch('./metrics.json');",
    ).replace(
        "const qRes = await fetch('/api/queries');",
        "const qRes = await fetch('./queries.json');",
    ).replace(
        "src=\"/figures/",
        "src=\"./figures/",
    ).replace(
        "const res = await fetch('/api/query_detail?query_id=' + encodeURIComponent(queryId));\n      const d = await res.json();",
        "if (!window._allDetails) { const r = await fetch('./query_details.json'); window._allDetails = await r.json(); }\n      const d = window._allDetails[queryId];",
    )

    (DOCS_DIR / "index.html").write_text(static_html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built static site in {DOCS_DIR}")


if __name__ == "__main__":
    build_static_site()

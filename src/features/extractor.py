"""
Stage 4: Feature Extraction (Plan Tree Turned Into Numbers).
Converts each query execution plan into two structured representations stored in FEATURES:
1. flat_vector: Engine-agnostic flattened plan features for Gradient Boosting (XGBoost)
   and tabular baselines:
   - Operator counts by canonical category (SCAN, JOIN, AGGREGATE, SORT, FILTER, PROJECTION, EXCHANGE)
   - Plan tree structural metrics (max_depth, num_nodes, num_leaf_scans, num_joins, branching_factor, join_density)
   - Cardinality features (log_sum_est_rows, log_max_est_rows, log_scan_est_rows, log_join_est_rows,
     log_agg_est_rows, log_root_est_rows, log_native_est_cost, weighted_op_work_log)
   - Table access indicators (accesses_lineitem, accesses_orders, accesses_customer, accesses_part,
     accesses_partsupp, accesses_supplier, accesses_nation, accesses_region)
2. tree_encoding: Recursive tree structure with per-node feature vectors for the
   Tree-Structured Neural Network (Tree-MLP / Tree-GRU) that reads plan structure directly.

Strictly uses ONLY pre-execution plan properties (est_rows, operator types, tree topology,
table names) as model inputs — never actual_rows or timings — to prevent target leakage.
"""

import math
from typing import Any, Dict, List, Tuple
from src.engines.runner import normalize_operator_name, OPERATOR_UNIT_WEIGHTS

CANONICAL_OPS = [
    "SCAN",
    "JOIN",
    "AGGREGATE",
    "SORT",
    "FILTER",
    "PROJECTION",
    "EXCHANGE",
    "OTHER",
]

TPCH_TABLES = [
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
]

FLAT_FEATURE_NAMES = [
    # Group 1: Operator counts
    "op_count_scan",
    "op_count_join",
    "op_count_aggregate",
    "op_count_sort",
    "op_count_filter",
    "op_count_projection",
    "op_count_exchange",
    "op_count_other",
    # Group 2: Structural / Topology features
    "tree_num_nodes",
    "tree_max_depth",
    "tree_avg_depth",
    "tree_num_leaves",
    "tree_join_depth_max",
    "tree_join_ratio",
    # Group 3: Estimated Cardinality & Cost features (log1p scale)
    "log_sum_est_rows",
    "log_max_est_rows",
    "log_mean_est_rows",
    "log_scan_est_rows",
    "log_join_est_rows",
    "log_agg_est_rows",
    "log_root_est_rows",
    "log_native_est_cost",
    "log_join_amplification",
    # Group 4: Scanned Table Footprint features
    "table_lineitem",
    "table_orders",
    "table_customer",
    "table_part",
    "table_partsupp",
    "table_supplier",
    "table_nation",
    "table_region",
    "n_distinct_tables",
]


def extract_plan_features(
    plan_id: str,
    native_est_cost: float,
    nodes: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Computes both flat_vector and tree_encoding for a single plan.
    """
    if not nodes:
        empty_flat = {k: 0.0 for k in FLAT_FEATURE_NAMES}
        empty_tree = {
            "op_idx": 7,
            "est_rows_log": 0.0,
            "depth_norm": 0.0,
            "is_leaf": 1.0,
            "table_idx": 0,
            "children": [],
        }
        return empty_flat, empty_tree

    op_counts = {op: 0.0 for op in CANONICAL_OPS}
    depths = []
    est_all = []
    scan_est = 0.0
    join_est = 0.0
    agg_est = 0.0
    join_depths = []
    tables_touched = set()

    parent_ids = {n.get("parent_node_id") for n in nodes if n.get("parent_node_id")}
    num_leaves = 0

    for n in nodes:
        op_cat = normalize_operator_name(n["operator"])
        op_counts[op_cat] = op_counts.get(op_cat, 0.0) + 1.0
        d = float(n.get("depth", 0))
        depths.append(d)
        est = max(0.0, float(n.get("est_rows", 0.0)))
        est_all.append(est)

        if op_cat == "SCAN":
            scan_est += est
        elif op_cat == "JOIN":
            join_est += est
            join_depths.append(d)
        elif op_cat == "AGGREGATE":
            agg_est += est

        tname = str(n.get("table_name", "")).lower().strip()
        if tname in TPCH_TABLES:
            tables_touched.add(tname)

        if n["node_id"] not in parent_ids:
            num_leaves += 1

    num_nodes = float(len(nodes))
    max_depth = max(depths) if depths else 0.0
    avg_depth = sum(depths) / num_nodes if num_nodes > 0 else 0.0
    sum_est = sum(est_all)
    max_est = max(est_all) if est_all else 0.0
    mean_est = sum_est / num_nodes if num_nodes > 0 else 0.0
    root_est = float(nodes[0].get("est_rows", 1.0))
    join_amp = (join_est + 1.0) / (scan_est + 1.0)

    flat_vector: Dict[str, float] = {
        "op_count_scan": op_counts["SCAN"],
        "op_count_join": op_counts["JOIN"],
        "op_count_aggregate": op_counts["AGGREGATE"],
        "op_count_sort": op_counts["SORT"],
        "op_count_filter": op_counts["FILTER"],
        "op_count_projection": op_counts["PROJECTION"],
        "op_count_exchange": op_counts["EXCHANGE"],
        "op_count_other": op_counts["OTHER"],
        "tree_num_nodes": num_nodes,
        "tree_max_depth": max_depth,
        "tree_avg_depth": avg_depth,
        "tree_num_leaves": float(num_leaves),
        "tree_join_depth_max": max(join_depths) if join_depths else 0.0,
        "tree_join_ratio": op_counts["JOIN"] / max(1.0, num_nodes),
        "log_sum_est_rows": math.log1p(max(0.0, sum_est)),
        "log_max_est_rows": math.log1p(max(0.0, max_est)),
        "log_mean_est_rows": math.log1p(max(0.0, mean_est)),
        "log_scan_est_rows": math.log1p(max(0.0, scan_est)),
        "log_join_est_rows": math.log1p(max(0.0, join_est)),
        "log_agg_est_rows": math.log1p(max(0.0, agg_est)),
        "log_root_est_rows": math.log1p(max(0.0, root_est)),
        "log_native_est_cost": math.log1p(max(0.0, float(native_est_cost))),
        "log_join_amplification": math.log1p(max(0.0, join_amp)),
        "table_lineitem": 1.0 if "lineitem" in tables_touched else 0.0,
        "table_orders": 1.0 if "orders" in tables_touched else 0.0,
        "table_customer": 1.0 if "customer" in tables_touched else 0.0,
        "table_part": 1.0 if "part" in tables_touched else 0.0,
        "table_partsupp": 1.0 if "partsupp" in tables_touched else 0.0,
        "table_supplier": 1.0 if "supplier" in tables_touched else 0.0,
        "table_nation": 1.0 if "nation" in tables_touched else 0.0,
        "table_region": 1.0 if "region" in tables_touched else 0.0,
        "n_distinct_tables": float(len(tables_touched)),
    }

    # Build recursive tree encoding for the Tree-Structured Neural Model
    children_map: Dict[str, List[Dict[str, Any]]] = {}
    roots: List[Dict[str, Any]] = []
    for n in nodes:
        pid = n.get("parent_node_id")
        if pid is None:
            roots.append(n)
        else:
            children_map.setdefault(pid, []).append(n)

    def encode_node(node: Dict[str, Any]) -> Dict[str, Any]:
        op_cat = normalize_operator_name(node["operator"])
        op_idx = CANONICAL_OPS.index(op_cat) if op_cat in CANONICAL_OPS else 7
        est = max(0.0, float(node.get("est_rows", 0.0)))
        d = float(node.get("depth", 0))
        tname = str(node.get("table_name", "")).lower().strip()
        t_idx = (TPCH_TABLES.index(tname) + 1) if tname in TPCH_TABLES else 0
        ch_nodes = children_map.get(node["node_id"], [])
        return {
            "node_id": node["node_id"],
            "op_cat": op_cat,
            "op_idx": op_idx,
            "est_rows_log": math.log1p(est),
            "op_weight": OPERATOR_UNIT_WEIGHTS.get(op_cat, 1.0),
            "depth_norm": d / max(1.0, max_depth),
            "is_leaf": 1.0 if len(ch_nodes) == 0 else 0.0,
            "table_idx": t_idx,
            "children": [encode_node(c) for c in ch_nodes],
        }

    if len(roots) == 1:
        tree_encoding = encode_node(roots[0])
    else:
        tree_encoding = {
            "node_id": f"{plan_id}__VIRTUAL_ROOT",
            "op_cat": "PROJECTION",
            "op_idx": CANONICAL_OPS.index("PROJECTION"),
            "est_rows_log": math.log1p(root_est),
            "op_weight": 0.5,
            "depth_norm": 0.0,
            "is_leaf": 0.0,
            "table_idx": 0,
            "children": [encode_node(r) for r in roots],
        }

    return flat_vector, tree_encoding

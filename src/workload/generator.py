"""
Stage 1: Query Generation & Correlated Benchmark Data Setup.
Generates TPC-H schema data stored in shared Apache Iceberg / Parquet warehouse format,
with realistic column correlations (so independence assumptions in native cost estimators
are genuinely stressed, as described in Section 4.2 of the proposal).

Also defines 24 distinct analytical SQL query templates (1-table to 6-table joins,
aggregations, filters, subqueries) partitioned strictly at the TEMPLATE level into
train (15 templates), val (3 templates), and test (6 templates).
"""

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import duckdb


@dataclass
class QueryTemplate:
    template_id: str
    source: str
    n_relations: int
    split_fold: str
    description: str
    sql_pattern: str
    param_space: Dict[str, List]


def get_templates() -> List[QueryTemplate]:
    """
    Returns 24 distinct analytical SQL templates across TPC-H relations:
    lineitem, orders, customer, part, supplier, partsupp, nation, region.
    Split is fixed at the template level:
    - 15 train templates (T01..T15)
    - 3 val templates (T16..T18)
    - 6 test templates (T19..T24)
    """
    shipmodes = ["'MAIL'", "'SHIP'", "'AIR'", "'RAIL'", "'TRUCK'", "'FOB'", "'REG AIR'"]
    segments = ["'AUTOMOBILE'", "'BUILDING'", "'FURNITURE'", "'MACHINERY'", "'HOUSEHOLD'"]
    priorities = ["'1-URGENT'", "'2-HIGH'", "'3-MEDIUM'", "'4-NOT SPECIFIED'", "'5-LOW'"]
    returnflags = ["'R'", "'A'", "'N'"]
    containers = ["'SM CASE'", "'SM BOX'", "'SM PACK'", "'LG CASE'", "'LG BOX'", "'MED BAG'", "'JUMBO PKG'"]
    brands = [f"'Brand#{i}{j}'" for i in range(1, 6) for j in range(1, 6)]
    regions = ["'AMERICA'", "'ASIA'", "'EUROPE'", "'AFRICA'", "'MIDDLE EAST'"]
    dates_start = [
        "DATE '1993-01-01'",
        "DATE '1994-01-01'",
        "DATE '1995-01-01'",
        "DATE '1996-01-01'",
        "DATE '1997-01-01'",
        "DATE '1994-06-01'",
        "DATE '1995-06-01'",
        "DATE '1996-06-01'",
    ]
    quantities = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    discounts = ["0.02", "0.04", "0.05", "0.06", "0.07", "0.08"]
    acctbals = [-500, 0, 500, 1500, 3000, 5000, 7000]
    prices = [10000, 25000, 50000, 100000, 150000, 200000]

    templates = [
        # --- TRAIN TEMPLATES (T01 .. T15) ---
        QueryTemplate(
            template_id="T01",
            source="TPC-H-Q1-Var",
            n_relations=1,
            split_fold="train",
            description="Single-table pricing summary report on lineitem with date & quantity filter",
            sql_pattern=(
                "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS sum_qty, "
                "SUM(l_extendedprice) AS sum_base_price, COUNT(*) AS count_order "
                "FROM lineitem "
                "WHERE l_shipdate >= {date_start} AND l_quantity <= {qty} "
                "GROUP BY l_returnflag, l_linestatus "
                "ORDER BY l_returnflag, l_linestatus"
            ),
            param_space={"date_start": dates_start, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T02",
            source="TPC-H-Q6-Var",
            n_relations=1,
            split_fold="train",
            description="Forecasting revenue change single-table scan with correlated discount/quantity",
            sql_pattern=(
                "SELECT SUM(l_extendedprice * l_discount) AS revenue, COUNT(*) AS cnt "
                "FROM lineitem "
                "WHERE l_shipdate >= {date_start} "
                "AND l_discount <= {disc} AND l_quantity < {qty}"
            ),
            param_space={"date_start": dates_start, "disc": discounts, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T03",
            source="TPC-H-Orders-Agg",
            n_relations=1,
            split_fold="train",
            description="Orders priority & status distribution filtered by totalprice and orderdate",
            sql_pattern=(
                "SELECT o_orderpriority, o_orderstatus, COUNT(*) AS order_cnt, "
                "AVG(o_totalprice) AS avg_price "
                "FROM orders "
                "WHERE o_orderdate >= {date_start} AND o_totalprice > {price} "
                "GROUP BY o_orderpriority, o_orderstatus"
            ),
            param_space={"date_start": dates_start, "price": prices},
        ),
        QueryTemplate(
            template_id="T04",
            source="TPC-H-Q12-Var",
            n_relations=2,
            split_fold="train",
            description="2-table join (orders, lineitem) shipping modes and order priority",
            sql_pattern=(
                "SELECT l.l_shipmode, o.o_orderpriority, COUNT(*) AS line_cnt, "
                "SUM(l.l_extendedprice) AS total_ext "
                "FROM orders o "
                "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
                "WHERE l.l_shipmode = {shipmode} "
                "AND l.l_receiptdate >= {date_start} "
                "AND o.o_totalprice >= {price} "
                "GROUP BY l.l_shipmode, o.o_orderpriority"
            ),
            param_space={"shipmode": shipmodes, "date_start": dates_start, "price": prices},
        ),
        QueryTemplate(
            template_id="T05",
            source="TPC-H-Q14-Var",
            n_relations=2,
            split_fold="train",
            description="2-table join (lineitem, part) promotion effect by brand and quantity",
            sql_pattern=(
                "SELECT p.p_brand, p.p_container, SUM(l.l_extendedprice * (1 - l.l_discount)) AS promo_rev, "
                "COUNT(*) AS matched_lines "
                "FROM lineitem l "
                "JOIN part p ON l.l_partkey = p.p_partkey "
                "WHERE l.l_shipdate >= {date_start} "
                "AND p.p_brand = {brand} AND l.l_quantity <= {qty} "
                "GROUP BY p.p_brand, p.p_container"
            ),
            param_space={"date_start": dates_start, "brand": brands[:12], "qty": quantities},
        ),
        QueryTemplate(
            template_id="T06",
            source="TPC-H-CustOrders",
            n_relations=2,
            split_fold="train",
            description="2-table join (customer, orders) customer segment spending analysis",
            sql_pattern=(
                "SELECT c.c_mktsegment, o.o_orderpriority, COUNT(*) AS num_orders, "
                "SUM(o.o_totalprice) AS sum_totalprice "
                "FROM customer c "
                "JOIN orders o ON c.c_custkey = o.o_custkey "
                "WHERE c.c_mktsegment = {segment} "
                "AND c.c_acctbal > {acctbal} "
                "AND o.o_orderdate >= {date_start} "
                "GROUP BY c.c_mktsegment, o.o_orderpriority"
            ),
            param_space={"segment": segments, "acctbal": acctbals, "date_start": dates_start},
        ),
        QueryTemplate(
            template_id="T07",
            source="TPC-H-Q3-Var",
            n_relations=3,
            split_fold="train",
            description="3-table join (customer, orders, lineitem) shipping priority query",
            sql_pattern=(
                "SELECT l.l_orderkey, o.o_orderdate, o.o_shippriority, "
                "SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
                "FROM customer c "
                "JOIN orders o ON c.c_custkey = o.o_custkey "
                "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
                "WHERE c.c_mktsegment = {segment} "
                "AND o.o_orderdate < {date_start} "
                "AND l.l_shipdate > {date_start} "
                "AND l.l_quantity >= {qty} "
                "GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority "
                "ORDER BY revenue DESC LIMIT 20"
            ),
            param_space={"segment": segments, "date_start": dates_start, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T08",
            source="TPC-H-PartSuppLine",
            n_relations=3,
            split_fold="train",
            description="3-table join (part, partsupp, lineitem) supply cost vs extended price",
            sql_pattern=(
                "SELECT p.p_brand, SUM(ps.ps_supplycost * l.l_quantity) AS total_supply_cost, "
                "SUM(l.l_extendedprice) AS total_price, COUNT(*) AS cnt "
                "FROM part p "
                "JOIN partsupp ps ON p.p_partkey = ps.ps_partkey "
                "JOIN lineitem l ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey "
                "WHERE p.p_container = {container} "
                "AND l.l_quantity <= {qty} "
                "AND l.l_shipdate >= {date_start} "
                "GROUP BY p.p_brand"
            ),
            param_space={"container": containers, "qty": quantities, "date_start": dates_start},
        ),
        QueryTemplate(
            template_id="T09",
            source="TPC-H-SuppNationLine",
            n_relations=3,
            split_fold="train",
            description="3-table join (supplier, nation, lineitem) supplier nation volume",
            sql_pattern=(
                "SELECT n.n_name, l.l_returnflag, COUNT(*) AS items_shipped, "
                "SUM(l.l_extendedprice) AS gross_rev "
                "FROM supplier s "
                "JOIN nation n ON s.s_nationkey = n.n_nationkey "
                "JOIN lineitem l ON s.s_suppkey = l.l_suppkey "
                "WHERE s.s_acctbal >= {acctbal} "
                "AND l.l_returnflag = {retflag} "
                "AND l.l_shipdate >= {date_start} "
                "GROUP BY n.n_name, l.l_returnflag"
            ),
            param_space={"acctbal": acctbals, "retflag": returnflags, "date_start": dates_start},
        ),
        QueryTemplate(
            template_id="T10",
            source="TPC-H-Q10-Var",
            n_relations=4,
            split_fold="train",
            description="4-table join (customer, orders, lineitem, nation) returned item reporting",
            sql_pattern=(
                "SELECT n.n_name, c.c_mktsegment, "
                "SUM(l.l_extendedprice * (1 - l.l_discount)) AS lost_revenue, COUNT(*) AS ret_cnt "
                "FROM customer c "
                "JOIN orders o ON c.c_custkey = o.o_custkey "
                "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
                "JOIN nation n ON c.c_nationkey = n.n_nationkey "
                "WHERE o.o_orderdate >= {date_start} "
                "AND l.l_returnflag = {retflag} "
                "AND c.c_acctbal >= {acctbal} "
                "GROUP BY n.n_name, c.c_mktsegment"
            ),
            param_space={"date_start": dates_start, "retflag": returnflags, "acctbal": acctbals},
        ),
        QueryTemplate(
            template_id="T11",
            source="TPC-H-Q16-Style",
            n_relations=4,
            split_fold="train",
            description="4-table join (part, partsupp, supplier, nation) supplier-part availability",
            sql_pattern=(
                "SELECT n.n_name, p.p_brand, COUNT(DISTINCT ps.ps_suppkey) AS supplier_cnt, "
                "SUM(ps.ps_availqty) AS total_avail "
                "FROM part p "
                "JOIN partsupp ps ON p.p_partkey = ps.ps_partkey "
                "JOIN supplier s ON ps.ps_suppkey = s.s_suppkey "
                "JOIN nation n ON s.s_nationkey = n.n_nationkey "
                "WHERE p.p_container = {container} "
                "AND s.s_acctbal > {acctbal} "
                "GROUP BY n.n_name, p.p_brand"
            ),
            param_space={"container": containers, "acctbal": acctbals, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T12",
            source="TPC-H-CustRegOrders",
            n_relations=4,
            split_fold="train",
            description="4-table join (region, nation, customer, orders) regional order volume",
            sql_pattern=(
                "SELECT r.r_name, n.n_name, o.o_orderpriority, COUNT(*) AS ord_cnt, "
                "SUM(o.o_totalprice) AS ord_sum "
                "FROM region r "
                "JOIN nation n ON r.r_regionkey = n.n_regionkey "
                "JOIN customer c ON n.n_nationkey = c.c_nationkey "
                "JOIN orders o ON c.c_custkey = o.o_custkey "
                "WHERE r.r_name = {region} "
                "AND o.o_orderdate >= {date_start} "
                "AND o.o_totalprice >= {price} "
                "GROUP BY r.r_name, n.n_name, o.o_orderpriority"
            ),
            param_space={"region": regions, "date_start": dates_start, "price": prices},
        ),
        QueryTemplate(
            template_id="T13",
            source="TPC-H-Q5-Var",
            n_relations=6,
            split_fold="train",
            description="6-table join (customer, orders, lineitem, supplier, nation, region) local supplier volume",
            sql_pattern=(
                "SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue, "
                "COUNT(*) AS tx_count "
                "FROM customer c "
                "JOIN orders o ON c.c_custkey = o.o_custkey "
                "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
                "JOIN supplier s ON l.l_suppkey = s.s_suppkey AND c.c_nationkey = s.s_nationkey "
                "JOIN nation n ON s.s_nationkey = n.n_nationkey "
                "JOIN region r ON n.n_regionkey = r.r_regionkey "
                "WHERE r.r_name = {region} "
                "AND o.o_orderdate >= {date_start} "
                "AND l.l_quantity <= {qty} "
                "GROUP BY n.n_name "
                "ORDER BY revenue DESC"
            ),
            param_space={"region": regions, "date_start": dates_start, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T14",
            source="TPC-H-Q9-Subset",
            n_relations=5,
            split_fold="train",
            description="5-table join (part, supplier, lineitem, partsupp, orders) product type profit",
            sql_pattern=(
                "SELECT p.p_brand, o.o_orderstatus, "
                "SUM(l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity) AS profit "
                "FROM part p "
                "JOIN lineitem l ON p.p_partkey = l.l_partkey "
                "JOIN supplier s ON s.s_suppkey = l.l_suppkey "
                "JOIN partsupp ps ON ps.ps_suppkey = l.l_suppkey AND ps.ps_partkey = l.l_partkey "
                "JOIN orders o ON o.o_orderkey = l.l_orderkey "
                "WHERE p.p_container = {container} "
                "AND o.o_orderdate >= {date_start} "
                "AND l.l_quantity <= {qty} "
                "GROUP BY p.p_brand, o.o_orderstatus"
            ),
            param_space={"container": containers, "date_start": dates_start, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T15",
            source="TPC-H-Q2-Core",
            n_relations=5,
            split_fold="train",
            description="5-table join (part, partsupp, supplier, nation, region) minimum cost supplier lookup",
            sql_pattern=(
                "SELECT n.n_name, p.p_type, MIN(ps.ps_supplycost) AS min_cost, "
                "SUM(ps.ps_availqty) AS sum_avail "
                "FROM part p "
                "JOIN partsupp ps ON p.p_partkey = ps.ps_partkey "
                "JOIN supplier s ON s.s_suppkey = ps.ps_suppkey "
                "JOIN nation n ON s.s_nationkey = n.n_nationkey "
                "JOIN region r ON n.n_regionkey = r.r_regionkey "
                "WHERE r.r_name = {region} "
                "AND p.p_container = {container} "
                "AND s.s_acctbal >= {acctbal} "
                "GROUP BY n.n_name, p.p_type"
            ),
            param_space={"region": regions, "container": containers, "acctbal": acctbals},
        ),

        # --- VALIDATION TEMPLATES (T16 .. T18) ---
        QueryTemplate(
            template_id="T16",
            source="TPC-H-Val-2Join",
            n_relations=2,
            split_fold="val",
            description="Validation 2-table join (partsupp, supplier) inventory valuation",
            sql_pattern=(
                "SELECT s.s_nationkey, COUNT(*) AS num_parts, "
                "SUM(ps.ps_supplycost * ps.ps_availqty) AS inv_value "
                "FROM partsupp ps "
                "JOIN supplier s ON ps.ps_suppkey = s.s_suppkey "
                "WHERE s.s_acctbal >= {acctbal} "
                "AND ps.ps_availqty <= {price} "
                "GROUP BY s.s_nationkey"
            ),
            param_space={"acctbal": acctbals, "price": [500, 1500, 3000, 5000, 8000, 9999]},
        ),
        QueryTemplate(
            template_id="T17",
            source="TPC-H-Val-3Join",
            n_relations=3,
            split_fold="val",
            description="Validation 3-table join (orders, lineitem, part) priority brand fulfillment",
            sql_pattern=(
                "SELECT o.o_orderpriority, p.p_brand, COUNT(*) AS item_cnt, "
                "AVG(l.l_extendedprice) AS avg_ext "
                "FROM orders o "
                "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
                "JOIN part p ON l.l_partkey = p.p_partkey "
                "WHERE o.o_orderpriority = {priority} "
                "AND l.l_shipdate >= {date_start} "
                "AND l.l_quantity <= {qty} "
                "GROUP BY o.o_orderpriority, p.p_brand"
            ),
            param_space={"priority": priorities, "date_start": dates_start, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T18",
            source="TPC-H-Val-5Join",
            n_relations=5,
            split_fold="val",
            description="Validation 5-table join (customer, nation, region, orders, lineitem) regional shipping delay",
            sql_pattern=(
                "SELECT r.r_name, l.l_shipmode, COUNT(*) AS late_cnt, "
                "SUM(l.l_extendedprice) AS total_val "
                "FROM region r "
                "JOIN nation n ON r.r_regionkey = n.n_regionkey "
                "JOIN customer c ON n.n_nationkey = c.c_nationkey "
                "JOIN orders o ON c.c_custkey = o.o_custkey "
                "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
                "WHERE r.r_name = {region} "
                "AND l.l_commitdate < l.l_receiptdate "
                "AND o.o_orderdate >= {date_start} "
                "AND l.l_quantity >= {qty} "
                "GROUP BY r.r_name, l.l_shipmode"
            ),
            param_space={"region": regions, "date_start": dates_start, "qty": quantities},
        ),

        # --- HELD-OUT TEST TEMPLATES (T19 .. T24) ---
        QueryTemplate(
            template_id="T19",
            source="TPC-H-Test-1Scan",
            n_relations=1,
            split_fold="test",
            description="Held-out 1-table scan (lineitem) shipmode & returnflag multi-aggregate",
            sql_pattern=(
                "SELECT l_shipmode, l_returnflag, COUNT(*) AS n_lines, "
                "AVG(l_quantity) AS avg_qty, SUM(l_extendedprice * l_tax) AS total_tax "
                "FROM lineitem "
                "WHERE l_receiptdate >= {date_start} "
                "AND l_discount <= {disc} "
                "AND l_quantity >= {qty} "
                "GROUP BY l_shipmode, l_returnflag"
            ),
            param_space={"date_start": dates_start, "disc": discounts, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T20",
            source="TPC-H-Test-2Join-Correlated",
            n_relations=2,
            split_fold="test",
            description="Held-out 2-table join (orders, lineitem) with correlated totalprice & discount filter",
            sql_pattern=(
                "SELECT o.o_orderstatus, l.l_linestatus, COUNT(*) AS cnt, "
                "SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_amount "
                "FROM orders o "
                "JOIN lineitem l ON o.o_orderkey = l.l_orderkey "
                "WHERE o.o_totalprice >= {price} "
                "AND l.l_discount >= {disc} "
                "AND o.o_orderdate >= {date_start} "
                "GROUP BY o.o_orderstatus, l.l_linestatus"
            ),
            param_space={"price": prices, "disc": discounts, "date_start": dates_start},
        ),
        QueryTemplate(
            template_id="T21",
            source="TPC-H-Test-3Join-CustPart",
            n_relations=3,
            split_fold="test",
            description="Held-out 3-table join (customer, orders, nation) high-balance customer nations",
            sql_pattern=(
                "SELECT n.n_name, o.o_orderstatus, COUNT(*) AS order_count, "
                "SUM(o.o_totalprice) AS revenue_sum "
                "FROM customer c "
                "JOIN nation n ON c.c_nationkey = n.n_nationkey "
                "JOIN orders o ON c.c_custkey = o.o_custkey "
                "WHERE c.c_mktsegment = {segment} "
                "AND c.c_acctbal >= {acctbal} "
                "AND o.o_orderdate >= {date_start} "
                "GROUP BY n.n_name, o.o_orderstatus"
            ),
            param_space={"segment": segments, "acctbal": acctbals, "date_start": dates_start},
        ),
        QueryTemplate(
            template_id="T22",
            source="TPC-H-Test-4Join-SupplyChain",
            n_relations=4,
            split_fold="test",
            description="Held-out 4-table join (supplier, lineitem, orders, customer) cross-segment fulfillment",
            sql_pattern=(
                "SELECT c.c_mktsegment, l.l_shipmode, COUNT(*) AS deliveries, "
                "SUM(l.l_extendedprice) AS billed "
                "FROM supplier s "
                "JOIN lineitem l ON s.s_suppkey = l.l_suppkey "
                "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                "JOIN customer c ON o.o_custkey = c.c_custkey "
                "WHERE c.c_mktsegment = {segment} "
                "AND l.l_shipdate >= {date_start} "
                "AND l.l_quantity <= {qty} "
                "GROUP BY c.c_mktsegment, l.l_shipmode"
            ),
            param_space={"segment": segments, "date_start": dates_start, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T23",
            source="TPC-H-Test-5Join-GlobalTrade",
            n_relations=5,
            split_fold="test",
            description="Held-out 5-table join (part, lineitem, orders, customer, nation) part demand by nation",
            sql_pattern=(
                "SELECT n.n_name, p.p_container, COUNT(*) AS line_volume, "
                "SUM(l.l_extendedprice * (1 - l.l_discount)) AS net_sales "
                "FROM part p "
                "JOIN lineitem l ON p.p_partkey = l.l_partkey "
                "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                "JOIN customer c ON o.o_custkey = c.c_custkey "
                "JOIN nation n ON c.c_nationkey = n.n_nationkey "
                "WHERE p.p_container = {container} "
                "AND o.o_orderdate >= {date_start} "
                "AND l.l_quantity >= {qty} "
                "GROUP BY n.n_name, p.p_container"
            ),
            param_space={"container": containers, "date_start": dates_start, "qty": quantities},
        ),
        QueryTemplate(
            template_id="T24",
            source="TPC-H-Test-6Join-FullStar",
            n_relations=6,
            split_fold="test",
            description="Held-out 6-table join (part, partsupp, supplier, lineitem, orders, nation) complex multi-join",
            sql_pattern=(
                "SELECT n.n_name, o.o_orderpriority, COUNT(*) AS total_items, "
                "SUM(l.l_extendedprice - ps.ps_supplycost * l.l_quantity) AS margin "
                "FROM part p "
                "JOIN partsupp ps ON p.p_partkey = ps.ps_partkey "
                "JOIN supplier s ON ps.ps_suppkey = s.s_suppkey "
                "JOIN lineitem l ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey "
                "JOIN orders o ON l.l_orderkey = o.o_orderkey "
                "JOIN nation n ON s.s_nationkey = n.n_nationkey "
                "WHERE p.p_container = {container} "
                "AND o.o_orderdate >= {date_start} "
                "AND l.l_quantity <= {qty} "
                "GROUP BY n.n_name, o.o_orderpriority"
            ),
            param_space={"container": containers, "date_start": dates_start, "qty": quantities},
        ),
    ]
    return templates


def generate_queries_from_templates(
    variants_per_template: int = 12, base_seed: int = 42
) -> Tuple[List[QueryTemplate], List[Dict]]:
    """
    Produces deterministic SQL query variations for each template.
    """
    templates = get_templates()
    queries = []
    for t_idx, tmpl in enumerate(templates):
        rng = random.Random(base_seed + t_idx * 1009)
        keys = sorted(tmpl.param_space.keys())
        seen_sql = set()
        attempts = 0
        v_idx = 0
        while v_idx < variants_per_template and attempts < variants_per_template * 20:
            attempts += 1
            chosen = {k: rng.choice(tmpl.param_space[k]) for k in keys}
            sql = tmpl.sql_pattern.format(**chosen)
            if sql in seen_sql and attempts < variants_per_template * 5:
                continue
            seen_sql.add(sql)
            q_seed = base_seed + t_idx * 1000 + v_idx
            query_id = f"{tmpl.template_id}_Q{v_idx+1:03d}"
            queries.append(
                {
                    "query_id": query_id,
                    "template_id": tmpl.template_id,
                    "sql_text": sql,
                    "seed": q_seed,
                }
            )
            v_idx += 1
    return templates, queries


def prepare_shared_tpch_warehouse(
    warehouse_dir: Path, scale_factor: float = 0.05
) -> Dict[str, int]:
    """
    Generates TPC-H tables in DuckDB, injects realistic column correlations
    (e.g., high-quantity lineitems have higher discount and specific shipmodes;
    high-balance customers place higher-value orders) so native independence
    assumptions are stressed, and exports the exact same data to shared Parquet/Iceberg
    files read by both DuckDB and Trino.
    """
    warehouse_dir = Path(warehouse_dir)
    parquet_dir = warehouse_dir / "tpch_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    db_file = warehouse_dir / "duckdb_tpch.duckdb"

    con = duckdb.connect(str(db_file))
    con.execute("INSTALL tpch; LOAD tpch;")

    existing = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "lineitem" not in existing:
        con.execute(f"CALL dbgen(sf={scale_factor});")
        # Inject realistic data skew & cross-column correlation (Section 4.2 of proposal)
        # 1. Correlate l_discount and l_shipmode with l_quantity and l_returnflag
        con.execute(
            """
            UPDATE lineitem
            SET l_discount = CASE
                WHEN l_quantity >= 30 THEN 0.08
                WHEN l_quantity >= 20 THEN 0.06
                ELSE 0.02
            END,
            l_returnflag = CASE
                WHEN l_quantity >= 28 AND l_shipmode IN ('MAIL', 'SHIP') THEN 'R'
                ELSE l_returnflag
            END
            """
        )
        # 2. Correlate o_totalprice and o_orderpriority with customer segment
        con.execute(
            """
            UPDATE orders
            SET o_orderpriority = CASE
                WHEN o_totalprice > 120000 THEN '1-URGENT'
                WHEN o_totalprice > 60000 THEN '2-HIGH'
                ELSE o_orderpriority
            END
            """
        )
        con.execute("ANALYZE;")

    tables = [
        "region",
        "nation",
        "supplier",
        "customer",
        "part",
        "partsupp",
        "orders",
        "lineitem",
    ]
    row_counts = {}
    for t in tables:
        out_file = parquet_dir / f"{t}.parquet"
        if not out_file.exists():
            con.execute(f"COPY {t} TO '{out_file.as_posix()}' (FORMAT PARQUET);")
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        row_counts[t] = int(cnt)

    con.close()
    return row_counts

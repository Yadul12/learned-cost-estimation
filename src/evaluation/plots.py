"""
Stage 6: Visualization & Scientific Figures Generator.
Generates all publication-grade figures for the project report and evaluation suite:
1. fig1_q_error_comparison.png: In-engine (DuckDB->DuckDB, Trino->Trino) and Cross-Engine
   (DuckDB->Trino) Median, P90, and P95 Q-error across all models and baselines.
2. fig2_cross_engine_ranking.png: Scatter & rank alignment of DuckDB-trained models
   predicting Trino execution times on held-out templates (T19..T24).
3. fig3_template_vs_random_leakage.png: Empirical proof of Section 5 ("Honest evaluation"):
   comparing Random Query Split (leaked templates) vs Template-Disjoint Split.
4. fig4_cardinality_q_error_by_operator.png: The "Free Bonus" analysis (Section 7):
   comparing `est_rows` vs `actual_rows` across operators and join depths.
5. fig5_ablation_and_timing_variance.png: Ablation study (removing feature groups) +
   Cold-run vs Warm-run timing variance (Section 7 & Section 8).
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
    }
)


def plot_all_figures(
    results: Dict[str, Any],
    nodes_df: pd.DataFrame,
    exec_df: pd.DataFrame,
    fig_dir: Path,
) -> List[Path]:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    saved: List[Path] = []
    saved.append(_plot_fig1_headline_q_error(results, fig_dir))
    saved.append(_plot_fig2_cross_engine_scatter(results, fig_dir))
    saved.append(_plot_fig3_leakage_and_per_template(results, fig_dir))
    saved.append(_plot_fig4_cardinality_bonus(nodes_df, fig_dir))
    saved.append(_plot_fig5_ablations_and_variance(results, exec_df, fig_dir))
    return saved


def _plot_fig1_headline_q_error(results: Dict[str, Any], fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    regimes = [
        ("duckdb_in_engine", "In-Engine: DuckDB → DuckDB\n(Held-Out Test Templates)"),
        ("trino_in_engine", "In-Engine: Trino → Trino\n(Upper-Bound Oracle Reference)"),
        ("cross_engine_calibrated", "Cross-Engine: DuckDB → Trino\n(Headline Transfer Goal)"),
    ]

    palette = {
        "Native Optimizer": "#d95f02",
        "Data Size Alone": "#7570b3",
        "Linear Ridge": "#66a61e",
        "Gradient Boosting (Flat)": "#1b9e77",
        "Tree Neural Model": "#e7298a",
    }

    for ax, (key, title) in zip(axes, regimes):
        reg_data = results[key]
        models = list(reg_data.keys())
        meds = [reg_data[m]["median_q_error"] for m in models]
        p90s = [reg_data[m]["p90_q_error"] for m in models]

        x = np.arange(len(models))
        width = 0.36
        b1 = ax.bar(x - width / 2, meds, width, label="Median Q-Error", color="#2b5c8f")
        b2 = ax.bar(x + width / 2, p90s, width, label="90th Pct (Tail) Q-Error", color="#e07a5f")

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Perfect (Q=1.0)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=28, ha="right", fontsize=8.8)
        ax.set_ylabel("Q-Error (lower is better, 1.0 = ideal)")
        for bar in b1:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        for bar in b2:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_ylim(0, max(p90s) * 1.22)
        if key == "duckdb_in_engine":
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Figure 1: Headline Q-Error Evaluation on Held-Out Templates (T19–T24)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    out = fig_dir / "fig1_q_error_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_fig2_cross_engine_scatter(results: Dict[str, Any], fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    test_details = pd.DataFrame(results["test_predictions_trino"])

    panels = [
        ("pred_native", "Trino Native Cost Estimate", "#d95f02"),
        ("pred_xgb_transfer", "Cross-Engine GBDT (DuckDB → Trino)", "#1b9e77"),
        ("pred_tree_transfer", "Cross-Engine Tree-NN (DuckDB → Trino)", "#e7298a"),
    ]

    for ax, (col, label, color) in zip(axes, panels):
        y_true = test_details["actual_trino_ms"].values
        y_pred = test_details[col].values
        ax.scatter(
            y_true,
            y_pred,
            c=color,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.4,
            s=46,
        )
        lims = [
            min(y_true.min(), y_pred.min()) * 0.85,
            max(y_true.max(), y_pred.max()) * 1.15,
        ]
        ax.plot(lims, lims, "k--", linewidth=1.1, label="Ideal 1:1 (Q=1.0)")
        ax.plot(lims, [l * 2.0 for l in lims], ":", color="gray", linewidth=0.9, label="Q-Error = 2.0 band")
        ax.plot(lims, [l * 0.5 for l in lims], ":", color="gray", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Trino Runtime (ms, log scale)")
        ax.set_ylabel("Predicted Trino Runtime (ms, log scale)")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Figure 2: Cross-Engine Transfer Predictions vs Actual Measured Trino Runtimes (Held-Out Templates)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    out = fig_dir / "fig2_cross_engine_ranking.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_fig3_leakage_and_per_template(results: Dict[str, Any], fig_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    # Left: Random Split (Leaked) vs Template-Disjoint Split (Honest)
    leak_comp = results["split_leakage_comparison"]
    models = list(leak_comp["random_split"].keys())
    rand_p90 = [leak_comp["random_split"][m]["p90_q_error"] for m in models]
    tmpl_p90 = [leak_comp["template_disjoint"][m]["p90_q_error"] for m in models]

    x = np.arange(len(models))
    w = 0.36
    ax1.bar(x - w / 2, rand_p90, w, label="Random Split (Template Leakage!)", color="#8da0cb")
    ax1.bar(x + w / 2, tmpl_p90, w, label="Template-Disjoint Split (Honest)", color="#fc8d62")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=22, ha="right", fontsize=8.8)
    ax1.set_ylabel("90th Percentile Q-Error")
    ax1.set_title("Why Template-Disjoint Evaluation Matters (Section 5)")
    ax1.legend(fontsize=8.5)

    # Right: Per-Template Cross-Engine Q-Error showing where transfer succeeds vs fails
    per_t = pd.DataFrame(results["per_template_transfer"])
    x2 = np.arange(len(per_t))
    w2 = 0.26
    ax2.bar(
        x2 - w2,
        per_t["native_median_q"],
        w2,
        label="Trino Native Estimate",
        color="#d95f02",
    )
    ax2.bar(
        x2,
        per_t["gbdt_transfer_median_q"],
        w2,
        label="GBDT Transfer (DuckDB→Trino)",
        color="#1b9e77",
    )
    ax2.bar(
        x2 + w2,
        per_t["tree_transfer_median_q"],
        w2,
        label="Tree-NN Transfer (DuckDB→Trino)",
        color="#e7298a",
    )
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [f"{r['template_id']}\n({r['n_relations']} rels)" for _, r in per_t.iterrows()],
        fontsize=8.5,
    )
    ax2.set_ylabel("Median Q-Error on Trino")
    ax2.set_title("Per-Template Cross-Engine Transfer Breakdown (T19–T24)")
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Figure 3: Template-Disjoint Leakage Proof & Per-Template Transfer Analysis (Including Failure Case)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    out = fig_dir / "fig3_template_vs_random_leakage.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_fig4_cardinality_bonus(nodes_df: pd.DataFrame, fig_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.5))

    df = nodes_df.copy()
    df["est_rows_safe"] = np.maximum(1.0, df["est_rows"].astype(float))
    df["actual_rows_safe"] = np.maximum(1.0, df["actual_rows"].astype(float))
    df["card_q_error"] = np.maximum(
        df["est_rows_safe"] / df["actual_rows_safe"],
        df["actual_rows_safe"] / df["est_rows_safe"],
    )

    op_order = (
        df.groupby("op_cat")["card_q_error"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(
        data=df,
        x="op_cat",
        y="card_q_error",
        hue="op_cat",
        legend=False,
        order=op_order,
        ax=ax1,
        palette="Set2",
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Canonical Plan Operator Type")
    ax1.set_ylabel("Cardinality Q-Error (log scale)")
    ax1.set_title("Node Cardinality Q-Error (`est_rows` vs `actual_rows`) by Operator")
    ax1.tick_params(axis="x", rotation=25)

    depth_summary = (
        df.groupby("depth")["card_q_error"]
        .agg(median="median", p90=lambda s: np.percentile(s, 90))
        .reset_index()
    )
    ax2.plot(
        depth_summary["depth"],
        depth_summary["median"],
        marker="o",
        linewidth=2.0,
        label="Median Node Cardinality Q-Error",
        color="#1b9e77",
    )
    ax2.plot(
        depth_summary["depth"],
        depth_summary["p90"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        label="90th Percentile Cardinality Q-Error",
        color="#d95f02",
    )
    ax2.set_yscale("log")
    ax2.set_xlabel("Plan Tree Depth (0 = Root, Higher = Deeper Subtrees)")
    ax2.set_ylabel("Cardinality Q-Error (log scale)")
    ax2.set_title("How Cardinality Estimation Errors Compound Across Plan Tree Depth")
    ax2.legend()

    fig.suptitle(
        "Figure 4: 'A Free Bonus' (Section 7) — Where Native Cardinality Estimators Go Wrong",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    out = fig_dir / "fig4_cardinality_q_error_by_operator.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_fig5_ablations_and_variance(
    results: Dict[str, Any], exec_df: pd.DataFrame, fig_dir: Path
) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.5))

    # Left: Feature Group Ablations
    abl = pd.DataFrame(results["ablations"])
    y_pos = np.arange(len(abl))
    ax1.barh(
        y_pos - 0.18,
        abl["duckdb_p90_q"],
        height=0.35,
        label="In-Engine DuckDB P90 Q-Error",
        color="#2b5c8f",
    )
    ax1.barh(
        y_pos + 0.18,
        abl["transfer_p90_q"],
        height=0.35,
        label="Cross-Engine Trino P90 Q-Error",
        color="#e07a5f",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(abl["variant"])
    ax1.invert_yaxis()
    ax1.set_xlabel("90th Percentile Q-Error (lower is better)")
    ax1.set_title("Feature Ablation Study (Section 8: What Breaks When Removed)")
    ax1.legend(fontsize=8.5)

    # Right: Cold Run (run 0) vs Warm Runs (runs 1..4) Coefficient of Variation
    var_stats = results["timing_variance_summary"]
    categories = ["DuckDB\nAll 5 Runs (incl. Cold)", "DuckDB\nWarm Runs (1..4)", "Trino\nAll 5 Runs (incl. Cold)", "Trino\nWarm Runs (1..4)"]
    cvs = [
        var_stats["duckdb_cv_all_pct"],
        var_stats["duckdb_cv_warm_pct"],
        var_stats["trino_cv_all_pct"],
        var_stats["trino_cv_warm_pct"],
    ]
    colors = ["#a6bddb", "#2b8cbe", "#fdbb84", "#e34a33"]
    bars = ax2.bar(categories, cvs, color=colors, width=0.52)
    for b in bars:
        h = b.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_ylabel("Median Coefficient of Variation (CV %)")
    ax2.set_title("Why Recording All 5 Runs & Dropping Cold Run Matters (Section 7)")
    ax2.set_ylim(0, max(cvs) * 1.25)

    fig.suptitle(
        "Figure 5: Component Ablations & Timing Noise Reduction From Cold-Run Filtering",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    out = fig_dir / "fig5_ablation_and_timing_variance.png"
    fig.savefig(out)
    plt.close(fig)
    return out

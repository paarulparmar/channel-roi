"""
Visualization layer.

Generates the full set of publication-quality figures the project brief
requires and writes them as PNGs to the outputs directory. A single consistent
visual theme is applied so the deck reads as one professional analysis.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .modeling import TrainedModel

sns.set_theme(style="whitegrid", context="talk")
PALETTE = "crest"
_ACCENT = "#2A6F7A"
_ACCENT2 = "#C05746"


def _save(fig, name: str) -> str:
    path = config.OUTPUTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# EDA figures
# --------------------------------------------------------------------------- #
def plot_cac_by_channel(econ: pd.DataFrame) -> str:
    d = econ.sort_values("avg_cac", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=d["avg_cac"], y=d.index, hue=d.index, palette=PALETTE,
                legend=False, ax=ax)
    for i, v in enumerate(d["avg_cac"]):
        ax.text(v + 0.4, i, f"${v:,.1f}", va="center", fontsize=13)
    ax.set_title("Customer Acquisition Cost (CAC) by Channel", weight="bold")
    ax.set_xlabel("Average CAC (USD)")
    ax.set_ylabel("")
    return _save(fig, "01_cac_by_channel.png")


def plot_revenue_by_channel(econ: pd.DataFrame) -> str:
    d = econ.sort_values("avg_revenue", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=d["avg_revenue"], y=d.index, hue=d.index, palette=PALETTE,
                legend=False, ax=ax)
    for i, v in enumerate(d["avg_revenue"]):
        ax.text(v + 2, i, f"${v:,.0f}", va="center", fontsize=13)
    ax.set_title("Average Lifetime Revenue by Channel", weight="bold")
    ax.set_xlabel("Average Revenue (USD)")
    ax.set_ylabel("")
    return _save(fig, "02_revenue_by_channel.png")


def plot_roi_by_channel(econ: pd.DataFrame) -> str:
    d = econ.sort_values("roi", ascending=False)
    colors = [_ACCENT if v >= 0 else _ACCENT2 for v in d["roi"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(d.index, d["roi"], color=colors)
    ax.invert_yaxis()
    for i, v in enumerate(d["roi"]):
        ax.text(v + 0.15, i, f"{v:.1f}x", va="center", fontsize=13)
    ax.set_title("Return on Investment (ROI) by Channel", weight="bold")
    ax.set_xlabel("ROI  =  (Revenue - Spend) / Spend")
    ax.set_ylabel("")
    return _save(fig, "03_roi_by_channel.png")


def plot_high_value_rate(econ: pd.DataFrame) -> str:
    d = econ.sort_values("high_value_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=d["high_value_rate"] * 100, y=d.index, hue=d.index,
                palette=PALETTE, legend=False, ax=ax)
    for i, v in enumerate(d["high_value_rate"]):
        ax.text(v * 100 + 0.4, i, f"{v:.1%}", va="center", fontsize=13)
    ax.set_title("Actual High-Value Customer Rate by Channel", weight="bold")
    ax.set_xlabel("High-Value Rate (%)")
    ax.set_ylabel("")
    return _save(fig, "04_high_value_rate_by_channel.png")


# --------------------------------------------------------------------------- #
# Model figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(model: TrainedModel) -> str:
    cm = model.confusion
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="crest", cbar=False,
        xticklabels=["Low-Value (0)", "High-Value (1)"],
        yticklabels=["Low-Value (0)", "High-Value (1)"], ax=ax,
        annot_kws={"size": 18},
    )
    ax.set_title(f"Confusion Matrix -- {model.name}", weight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return _save(fig, "05_confusion_matrix.png")


def plot_roc_curves(results: dict[str, TrainedModel], best_name: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 8))
    for name, r in results.items():
        fpr, tpr = r.roc_curve
        lw = 3.0 if name == best_name else 1.8
        label = f"{name} (AUC={r.metrics['roc_auc']:.3f})"
        ax.plot(fpr, tpr, lw=lw, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance")
    ax.set_title("ROC Curves -- Model Comparison", weight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=13)
    return _save(fig, "06_roc_curves.png")


def plot_model_comparison(metrics_tbl: pd.DataFrame) -> str:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    d = metrics_tbl[cols]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    d.plot(kind="bar", ax=ax, colormap="crest", width=0.8)
    ax.set_title("Model Performance Comparison (Test Set)", weight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", ncol=5, fontsize=11)
    plt.xticks(rotation=0)
    return _save(fig, "07_model_comparison.png")


def plot_feature_importance(imp: pd.DataFrame, model_name: str) -> str:
    d = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(x=d["importance"], y=d["feature"], hue=d["feature"],
                palette=PALETTE, legend=False, ax=ax)
    ax.set_title(f"Feature Importance -- {model_name}", weight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return _save(fig, "08_feature_importance.png")


def plot_shap_summary(model: TrainedModel, X_sample: pd.DataFrame) -> str | None:
    """Best-effort SHAP summary for tree models. Returns None if unavailable."""
    try:
        import shap
    except Exception:
        return None
    clf = model.pipeline.named_steps["clf"]
    prep = model.pipeline.named_steps["prep"]
    if not hasattr(clf, "feature_importances_"):
        return None
    try:
        from .preprocessing import get_feature_names
        Xt = prep.transform(X_sample)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        names = get_feature_names(prep)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        sv = np.asarray(sv)
        # Newer SHAP returns (n, features, n_classes) for classifiers; select
        # the positive class. Older versions return a list [neg, pos].
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        elif isinstance(sv, list):  # pragma: no cover
            sv = sv[1]
        plt.figure(figsize=(11, 7))
        shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=12)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Summary -- {model.name}", weight="bold", y=1.02)
        return _save(fig, "09_shap_summary.png")
    except Exception:
        plt.close("all")
        return None


# --------------------------------------------------------------------------- #
# Business figures
# --------------------------------------------------------------------------- #
def plot_predicted_hv_rate(scorecard: pd.DataFrame) -> str:
    d = scorecard.sort_values("predicted_hv_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["actual_hv_rate"] * 100, width=0.4,
           label="Actual", color=_ACCENT)
    ax.bar(x + 0.2, d["predicted_hv_rate"] * 100, width=0.4,
           label="Predicted", color=_ACCENT2)
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=25, ha="right")
    ax.set_title("Predicted vs Actual High-Value Rate by Channel", weight="bold")
    ax.set_ylabel("High-Value Rate (%)")
    ax.legend()
    return _save(fig, "10_predicted_hv_rate_by_channel.png")


def plot_budget_reallocation(budget_df: pd.DataFrame) -> str:
    d = budget_df.sort_values("optimized_budget", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["current_budget"], width=0.4,
           label="Before (current mix)", color=_ACCENT)
    ax.bar(x + 0.2, d["optimized_budget"], width=0.4,
           label="After (optimized)", color=_ACCENT2)
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=25, ha="right")
    ax.set_title("Marketing Budget Allocation: Before vs After Optimization",
                 weight="bold")
    ax.set_ylabel("Budget (USD)")
    ax.legend()
    return _save(fig, "11_budget_before_after.png")


def plot_roi_simulation(sim: pd.DataFrame) -> str:
    scenarios = ["Before (current mix)", "After (optimized)"]
    rev = [sim.loc[s, "expected_revenue"] for s in scenarios]
    roi = [sim.loc[s, "roi"] for s in scenarios]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    a1.bar(scenarios, rev, color=[_ACCENT, _ACCENT2])
    a1.set_title("Expected Revenue", weight="bold")
    a1.set_ylabel("USD")
    for i, v in enumerate(rev):
        a1.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=12)
    a2.bar(scenarios, roi, color=[_ACCENT, _ACCENT2])
    a2.set_title("ROI", weight="bold")
    a2.set_ylabel("ROI (x)")
    for i, v in enumerate(roi):
        a2.text(i, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=12)
    for a in (a1, a2):
        a.set_xticklabels(scenarios, rotation=12)
    fig.suptitle("Budget Optimization Impact Simulation", weight="bold", y=1.02)
    return _save(fig, "12_roi_simulation.png")


def generate_all(
    econ, results, best_name, imp, scorecard, budget_df, sim, X_sample
) -> list[str]:
    paths = [
        plot_cac_by_channel(econ),
        plot_revenue_by_channel(econ),
        plot_roi_by_channel(econ),
        plot_high_value_rate(econ),
        plot_confusion_matrix(results[best_name]),
        plot_roc_curves(results, best_name),
        plot_model_comparison(
            pd.DataFrame({n: r.metrics for n, r in results.items()}).T
        ),
        plot_feature_importance(imp, best_name),
        plot_predicted_hv_rate(scorecard),
        plot_budget_reallocation(budget_df),
        plot_roi_simulation(sim),
    ]
    shap_path = plot_shap_summary(results[best_name], X_sample)
    if shap_path:
        paths.append(shap_path)
    return paths

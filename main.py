"""
End-to-End Marketing Channel ROI Prediction & Budget Optimization
=================================================================

Single entry point that runs the complete pipeline:

    1. Generate (or load) the synthetic customer dataset
    2. Business-focused EDA on channel unit economics
    3. Feature engineering + train/compare Logistic Regression, Random Forest, XGBoost
    4. Evaluate, select the best model, extract feature importance / SHAP
    5. Score every customer and build the channel decision scorecard
    6. Optimize the marketing budget and simulate before/after ROI
    7. Render all visualizations and persist every artifact

Run:  python main.py
"""
from __future__ import annotations

import warnings

import pandas as pd

from src import config
from src.data_generation import load_or_generate
from src.real_data import load_or_build as load_or_build_olist
from src import eda
from src.preprocessing import engineer_features, get_feature_frame
from src import modeling
from src import business_layer as biz
from src import visualization as viz

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


def load_dataset(force_regen: bool = False) -> pd.DataFrame:
    """Load the dataset from the configured source (real Olist or synthetic)."""
    if config.DATA_SOURCE == "olist":
        return load_or_build_olist(force=force_regen)
    return load_or_generate(force=force_regen)


def banner(title: str) -> None:
    print("\n" + "#" * 72)
    print(f"# {title}")
    print("#" * 72)


def main(force_regen: bool = False) -> None:
    # ---------------------------------------------------------------- 1. DATA
    banner("STEP 1 | DATASET")
    df = load_dataset(force_regen=force_regen)
    src_label = ("REAL (Olist Marketing Funnel + E-Commerce)"
                 if config.DATA_SOURCE == "olist" else "SYNTHETIC")
    print(f"Source: {src_label}")
    print(f"Loaded {len(df):,} customers -> {config.RAW_DATA_PATH}")

    # ---------------------------------------------------------------- 2. EDA
    banner("STEP 2 | EXPLORATORY DATA ANALYSIS")
    econ = eda.print_eda_report(df)

    # -------------------------------------------------- 3+4. FEATURES & MODELS
    banner("STEP 3 | FEATURE ENGINEERING & MODEL TRAINING")
    df_feat = engineer_features(df)
    results, best_name, split = modeling.train_and_compare(df_feat)
    X_train, X_test, y_train, y_test = split

    metrics_tbl = modeling.metrics_table(results)
    print("\nModel comparison (sorted by test ROC-AUC):")
    print(metrics_tbl.to_string())
    print(f"\n>>> Best model selected: {best_name} "
          f"(ROC-AUC = {results[best_name].metrics['roc_auc']:.4f})")

    modeling.save_best_model(results, best_name)
    modeling.save_metrics(metrics_tbl)
    imp = modeling.feature_importance(results[best_name])
    print("\nTop feature importances:")
    print(imp.head(8).to_string(index=False))

    # ---------------------------------------------------- 5. SCORING + SCORECARD
    banner("STEP 4 | CUSTOMER SCORING & CHANNEL SCORECARD")
    scored = biz.score_customers(df_feat, results[best_name])
    scored.to_csv(config.SCORED_DATA_PATH, index=False)
    scorecard = biz.channel_scorecard(scored)
    scorecard.to_csv(config.SCORECARD_PATH)
    show_cols = ["customers", "avg_cac", "roi", "predicted_hv_rate",
                 "efficiency_score", "recommendation"]
    print(scorecard[show_cols].to_string())

    # ---------------------------------------------------- 6. BUDGET OPTIMIZATION
    banner("STEP 5 | BUDGET OPTIMIZATION & ROI SIMULATION")
    budget_df = biz.optimize_budget(scorecard)
    budget_df.to_csv(config.BUDGET_PATH)
    print("Budget reallocation (USD):")
    print(budget_df[["current_budget", "optimized_budget", "budget_delta"]]
          .to_string())

    sim = biz.simulate_before_after(scorecard, budget_df)
    print("\nBefore vs After simulation:")
    print(sim.to_string())
    lift = sim.loc["Lift %", "roi"]
    rev_lift = sim.loc["Lift %", "expected_revenue"]
    print(f"\n>>> Optimization lifts expected revenue by {rev_lift:.1f}% "
          f"and ROI by {lift:.1f}%.")

    # ----------------------------------------------------------- 7. VISUALIZE
    banner("STEP 6 | VISUALIZATIONS")
    X_sample = get_feature_frame(df_feat).sample(
        min(600, len(df_feat)), random_state=config.RANDOM_STATE
    )
    paths = viz.generate_all(
        econ, results, best_name, imp, scorecard, budget_df, sim, X_sample
    )
    for p in paths:
        print(f"  saved  {p}")

    banner("PIPELINE COMPLETE")
    print(f"Dataset     : {config.RAW_DATA_PATH}")
    print(f"Best model  : {config.BEST_MODEL_PATH}")
    print(f"Scorecard   : {config.SCORECARD_PATH}")
    print(f"Budget plan : {config.BUDGET_PATH}")
    print(f"Figures     : {config.OUTPUTS_DIR} ({len(paths)} PNGs)")


if __name__ == "__main__":
    main()

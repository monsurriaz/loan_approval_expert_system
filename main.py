# =============================================================================
# main.py
# Entry point - run the full loan approval expert system pipeline.
#
# Usage:
#   python main.py
#
# Stage 1 outputs (core):
#   results/metrics.json
#   results/predictions.csv
#   results/confusion_matrix.png
#
# Stage 2 outputs (after --stage2 flag or once core works):
#   results/feature_importance.png
#   results/component_comparison.png
#   results/bias_analysis.json
# =============================================================================

import sys
import pickle
import numpy as np

# -- Project modules -------------------------------------------------------
from src.data_handler       import load_data, basic_eda
from src.preprocessing      import run_preprocessing
from src.knowledge_base     import build_knowledge_base
from src.bayesian_reasoning import train_bayesian
from src.ml_model           import train_model, predict, save_model, get_feature_importance
from src.expert_system      import ExpertSystem
from src.evaluation         import (
    compute_metrics, save_metrics,
    save_predictions, plot_confusion_matrix,
    # Stage 2
    plot_feature_importance, plot_component_comparison, run_bias_analysis
)


def main(stage2: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  AI-Based Expert System for Loan Approval Decision Support")
    print("=" * 60)

    # -- Step 1: Load data -------------------------------------------------
    print("\n[STEP 1] Loading data...")
    df = load_data()
    basic_eda(df)

    # -- Step 2: Preprocess ------------------------------------------------
    print("[STEP 2] Preprocessing...")
    X_train, X_test, y_train, y_test, processed_df = run_preprocessing(df)

    # Keep a copy of the processed training DataFrame for the Knowledge Base
    from src.config import TARGET_COL
    train_df_with_target = X_train.copy()
    train_df_with_target[TARGET_COL] = y_train.values

    # -- Step 3: Knowledge Base --------------------------------------------
    print("[STEP 3] Building knowledge base...")
    knowledge = build_knowledge_base(train_df_with_target)

    # -- Step 4: Bayesian Reasoning ----------------------------------------
    print("[STEP 4] Training Bayesian reasoner...")
    priors, likelihoods, bin_edges = train_bayesian(X_train, y_train)

    # -- Step 5: ML Model --------------------------------------------------
    print("[STEP 5] Training ML model (RandomForest)...")
    model = train_model(X_train, y_train)
    save_model(model)

    # Save system metadata for demo.py and inference
    import os
    from src.config import RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)
    feature_names = list(X_train.columns)
    system_meta = {
        "knowledge": knowledge,
        "priors": priors,
        "likelihoods": likelihoods,
        "bin_edges": bin_edges,
        "feature_names": feature_names
    }
    meta_path = os.path.join(RESULTS_DIR, "system_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(system_meta, f)
    print(f"[main] System metadata saved -> {meta_path}")

    # -- Step 6: Integrated Expert System (batch predict on test set) ------
    print("[STEP 6] Running integrated expert system on test set...")
    es = ExpertSystem(knowledge, priors, likelihoods, bin_edges, model, feature_names)
    results = es.predict_batch(X_test)

    # Convert integrated decisions to binary predictions for metric computation
    y_pred_integrated = np.array([
        1 if r["decision"] == "Approve" else 0
        for r in results
    ])

    # Also get pure ML predictions for comparison
    y_pred_ml, _ = predict(model, X_test)

    # -- Step 7: Stage 1 Evaluation ----------------------------------------
    print("[STEP 7] Evaluating (Stage 1)...")
    metrics = compute_metrics(y_test, y_pred_integrated)
    save_metrics(metrics)
    save_predictions(X_test, y_test, results)
    plot_confusion_matrix(y_test, y_pred_integrated)

    # -- Step 8: Stage 2 (optional) ----------------------------------------
    if stage2:
        print("[STEP 8] Running Stage 2 analysis...")
        importance_df = get_feature_importance(model, feature_names)
        plot_feature_importance(importance_df)
        plot_component_comparison(y_test, results)
        run_bias_analysis(X_test, y_test, results)
    else:
        print("[STEP 8] Skipping Stage 2. Run with --stage2 flag to generate extra plots.")

    # -- Done ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print("=" * 60 + "\n")

    # Print a sample prediction for verification
    sample = X_test.iloc[0].to_dict()
    sample_result = es.predict(sample)
    print("-- Sample prediction (first test applicant) -----------")
    print(f"  {sample_result['explanation']}")
    print("----------------------------------------------------------\n")


if __name__ == "__main__":
    run_stage2 = "--stage2" in sys.argv
    main(stage2=run_stage2)

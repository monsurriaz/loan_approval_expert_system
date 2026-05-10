#!/usr/bin/env python3
"""Quick test of notebook cells"""
import sys
sys.path.insert(0, '.')

print("Testing Notebook Cells...")
print("\n" + "="*60)

# Test Cell 2: Load Data and EDA
print("Cell 2: Load Data and EDA")
try:
    from src.config import TARGET_COL
    from src.data_handler import load_data

    df = load_data()
    print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[OK] Target distribution:\n{df[TARGET_COL].value_counts().to_string()}")
    print(f"[OK] Approval rate: {(df[TARGET_COL] == 'Y').mean():.1%}")
except Exception as e:
    print(f"[FAIL] Cell 2 failed: {e}")

# Test Cell 3: Preprocessing
print("\n" + "="*60)
print("Cell 3: Preprocessing")
try:
    from src.data_handler import load_data
    from src.preprocessing import run_preprocessing

    df = load_data()
    X_train, X_test, y_train, y_test, _ = run_preprocessing(df)
    print(f"[OK] Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"[OK] Features: {len(X_train.columns)}")
except Exception as e:
    print(f"[FAIL] Cell 3 failed: {e}")

# Test Cell 4: Knowledge Base
print("\n" + "="*60)
print("Cell 4: Knowledge Base")
try:
    from src.data_handler import load_data
    from src.preprocessing import run_preprocessing
    from src.knowledge_base import build_knowledge_base
    from src.config import TARGET_COL

    df = load_data()
    X_train, X_test, y_train, y_test, _ = run_preprocessing(df)
    train_df_with_target = X_train.copy()
    train_df_with_target[TARGET_COL] = y_train.values

    knowledge = build_knowledge_base(train_df_with_target)
    print(f"[OK] Knowledge base built with {len(knowledge['rules'])} rules")
    print(f"[OK] Income threshold: {knowledge['income_threshold']:.0f}")
except Exception as e:
    print(f"[FAIL] Cell 4 failed: {e}")

# Test Cell 5: Bayesian Reasoning
print("\n" + "="*60)
print("Cell 5: Bayesian Reasoning")
try:
    from src.data_handler import load_data
    from src.preprocessing import run_preprocessing
    from src.bayesian_reasoning import train_bayesian

    df = load_data()
    X_train, X_test, y_train, y_test, _ = run_preprocessing(df)
    priors, likelihoods, bin_edges = train_bayesian(X_train, y_train)
    print(f"[OK] Priors: P(Approved)={priors['approved']:.4f}")
    print(f"[OK] Likelihoods computed for {len(likelihoods)} features")
except Exception as e:
    print(f"[FAIL] Cell 5 failed: {e}")

# Test Cell 6: ML Model
print("\n" + "="*60)
print("Cell 6: ML Model")
try:
    from src.data_handler import load_data
    from src.preprocessing import run_preprocessing
    from src.ml_model import train_model, predict
    from sklearn.metrics import accuracy_score

    df = load_data()
    X_train, X_test, y_train, y_test, _ = run_preprocessing(df)
    model = train_model(X_train, y_train)
    y_pred, _ = predict(model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[OK] ML model trained")
    print(f"[OK] Test accuracy: {accuracy:.4f}")
except Exception as e:
    print(f"[FAIL] Cell 6 failed: {e}")

# Test Cell 7: Integrated System
print("\n" + "="*60)
print("Cell 7: Integrated Expert System")
try:
    from src.data_handler import load_data
    from src.preprocessing import run_preprocessing
    from src.knowledge_base import build_knowledge_base
    from src.bayesian_reasoning import train_bayesian
    from src.ml_model import train_model
    from src.expert_system import ExpertSystem
    from src.config import TARGET_COL

    df = load_data()
    X_train, X_test, y_train, y_test, _ = run_preprocessing(df)
    train_df_with_target = X_train.copy()
    train_df_with_target[TARGET_COL] = y_train.values

    knowledge = build_knowledge_base(train_df_with_target)
    priors, likelihoods, bin_edges = train_bayesian(X_train, y_train)
    model = train_model(X_train, y_train)
    feature_names = list(X_train.columns)

    es = ExpertSystem(knowledge, priors, likelihoods, bin_edges, model, feature_names)

    # Test on first sample
    sample = X_test.iloc[0].to_dict()
    result = es.predict(sample)
    print(f"[OK] Expert system prediction: {result['decision']}")
    print(f"[OK] Confidence: {result['confidence']:.4f}")
except Exception as e:
    print(f"[FAIL] Cell 7 failed: {e}")

# Test Cell 8: Visualizations
print("\n" + "="*60)
print("Cell 8: Visualizations")
try:
    import os
    from src.config import RESULTS_DIR

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    cc_path = os.path.join(RESULTS_DIR, "component_comparison.png")

    if os.path.exists(cm_path) and os.path.exists(cc_path):
        print(f"[OK] Confusion matrix found")
        print(f"[OK] Component comparison found")
    else:
        print(f"[WARN] Image files not found (expected after --stage2 run)")
except Exception as e:
    print(f"[FAIL] Cell 8 failed: {e}")

print("\n" + "="*60)
print("Notebook Cell Tests Complete!")
print("="*60)

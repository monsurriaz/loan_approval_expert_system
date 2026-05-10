#!/usr/bin/env python3
"""Final comprehensive test of the fixed system"""
import pickle
import os
import sys
sys.path.insert(0, '.')

from src.config import MODEL_PATH, RESULTS_DIR
from src.expert_system import ExpertSystem

print("\n" + "="*70)
print("  Loan Approval Expert System - Final Test")
print("="*70)

# Load system
model_path = MODEL_PATH
meta_path = os.path.join(RESULTS_DIR, "system_meta.pkl")

if not os.path.exists(model_path) or not os.path.exists(meta_path):
    print("  ERROR: Trained components not found.")
    sys.exit(1)

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

es = ExpertSystem(
    knowledge=meta["knowledge"],
    priors=meta["priors"],
    likelihoods=meta["likelihoods"],
    bin_edges=meta["bin_edges"],
    model=model,
    feature_names=meta["feature_names"]
)

print("\n  System loaded successfully.\n")

# Test applicant
applicant_raw = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000.0,
    "CoapplicantIncome": 2000.0,
    "LoanAmount": 120.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1,
    "Property_Area": "Urban",
}

# Encode
encodings = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

applicant_enc = {}
for k, v in applicant_raw.items():
    if k in encodings:
        applicant_enc[k] = encodings[k].get(str(v), v)
    else:
        applicant_enc[k] = v

# Predict
result = es.predict(applicant_enc)

# Display
decision = result["decision"]
marker = {
    "Approve": "[APPROVED]",
    "Manual Review": "[REVIEW]",
    "Reject": "[REJECTED]"
}.get(decision, "")

print("="*70)
print(f"  DECISION: {marker}  {decision.upper()}")
print(f"  Confidence Score: {result['confidence']:.2%}")
print("="*70)
print(f"  Knowledge Base Score : {result['kb_score']:.4f}")
print(f"  Bayesian Probability : {result['bayes_prob']:.4f}")
print(f"  ML Model Probability : {result['ml_prob']:.4f}")
print(f"  Final Approval Score : {result['final_approval_score']:.4f}")
print(f"  Rules Fired          : {', '.join(result['fired_rules']) if result['fired_rules'] else 'None'}")
print("\n  Candidate Rankings:")
for c in result["all_candidates"]:
    print(f"    {c['label']:<15} score={c['score']:.4f}")
print("="*70)

print(f"\n  {result['explanation']}\n")

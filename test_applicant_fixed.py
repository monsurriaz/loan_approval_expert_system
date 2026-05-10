#!/usr/bin/env python3
"""Test the fixed system with the provided applicant"""
import pickle
import os
import sys
sys.path.insert(0, '.')

from src.config import MODEL_PATH, RESULTS_DIR
from src.expert_system import ExpertSystem
from demo import encode_applicant

# Load system
model_path = MODEL_PATH
meta_path = os.path.join(RESULTS_DIR, "system_meta.pkl")

if not os.path.exists(model_path) or not os.path.exists(meta_path):
    print("ERROR: Trained components not found.")
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

# Test applicant (per user specification)
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
applicant_enc = encode_applicant(applicant_raw)

# Predict
result = es.predict(applicant_enc)

# Display results
print("\n" + "="*70)
print("TEST APPLICANT - FIXED SYSTEM VALIDATION")
print("="*70)

print("\nInput Data:")
for k, v in applicant_raw.items():
    print(f"  {k:<20}: {v}")

print("\nEncoded Features:")
for k, v in applicant_enc.items():
    print(f"  {k:<20}: {v}")

print("\n" + "-"*70)
print("EXPERT SYSTEM DECISION")
print("-"*70)

print(f"\nFinal Approval Score: {result['final_approval_score']:.4f}")
print(f"Decision             : {result['decision']}")
print(f"Confidence Score     : {result['confidence']:.4f}")

print(f"\nComponent Scores:")
print(f"  Knowledge Base Score : {result['kb_score']:.4f}")
print(f"  Bayesian Probability : {result['bayes_prob']:.4f}")
print(f"  ML Model Probability : {result['ml_prob']:.4f}")

print(f"\nRules Fired: {result['fired_rules']}")

print(f"\nCandidate Rankings (by score):")
for i, candidate in enumerate(result['all_candidates'], 1):
    print(f"  {i}. {candidate['label']:<15} score={candidate['score']:.4f}")

print(f"\nFull Explanation:")
print(f"  {result['explanation']}")

print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

# Verify the fixes
checks_passed = 0
checks_total = 4

# Check 1: Candidate scores are different
unique_scores = set(c['score'] for c in result['all_candidates'])
if len(unique_scores) == 3:
    print("[PASS] Candidate scores are different")
    checks_passed += 1
else:
    print("[FAIL] Candidate scores are identical!")

# Check 2: Confidence matches winning candidate
winning_score = result['all_candidates'][0]['score']
if abs(result['confidence'] - winning_score) < 0.0001:
    print("[PASS] Confidence matches winning candidate score")
    checks_passed += 1
else:
    print(f"[FAIL] Confidence ({result['confidence']:.4f}) != winning score ({winning_score:.4f})")

# Check 3: final_approval_score is present
if 'final_approval_score' in result:
    print("[PASS] final_approval_score is present in result")
    checks_passed += 1
else:
    print("[FAIL] final_approval_score missing from result")

# Check 4: Explanation format includes all required info
required_strings = ['Decision:', 'confidence=', 'Approval score=', 'KB=', 'Bayes=', 'ML=', 'Rules:']
if all(s in result['explanation'] for s in required_strings):
    print("[PASS] Explanation format is correct")
    checks_passed += 1
else:
    print("[FAIL] Explanation format is incorrect")

print(f"\nPassed: {checks_passed}/{checks_total}")
print("="*70 + "\n")

#!/usr/bin/env python3
# =============================================================================
# test_demo.py
# Test the demo with a specific applicant without interactive input
# =============================================================================

import pickle
import os
from src.config import MODEL_PATH, RESULTS_DIR
from src.expert_system import ExpertSystem
from demo import encode_applicant, print_result


def test_applicant():
    """Test the expert system with a specific applicant."""

    # Load system
    model_path = MODEL_PATH
    meta_path = os.path.join(RESULTS_DIR, "system_meta.pkl")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("ERROR: Trained components not found. Please run `python main.py` first.")
        return

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

    # Test applicant data (raw)
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
    print("\n" + "=" * 60)
    print("  TEST APPLICANT EVALUATION")
    print("=" * 60)
    print("\n-- Input (Raw) -------------------------------------------")
    for k, v in applicant_raw.items():
        print(f"  {k:<20}: {v}")

    print("\n-- Input (Encoded) ----------------------------------------")
    for k, v in applicant_enc.items():
        print(f"  {k:<20}: {v}")

    print("\n-- Expert System Decision ---------------------------------")
    print_result(result)


if __name__ == "__main__":
    test_applicant()

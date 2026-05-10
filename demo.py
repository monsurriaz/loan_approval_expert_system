# =============================================================================
# demo.py
# Interactive console demo for evaluation / viva day.
# Loads the trained system and accepts applicant data as input.
#
# Usage:
#   python demo.py
# =============================================================================

import pickle
import json
import os
from src.config import MODEL_PATH, RESULTS_DIR
from src.expert_system import ExpertSystem


def load_system() -> ExpertSystem:
    """
    Load all trained components from disk.
    Must run main.py at least once before running demo.py.
    """
    model_path   = MODEL_PATH
    meta_path    = os.path.join(RESULTS_DIR, "system_meta.pkl")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Trained components not found. Please run `python main.py` first."
        )

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
    return es


def get_input(prompt: str, valid: list | None = None, cast=str):
    """Helper to get validated console input."""
    while True:
        val = input(f"  {prompt}: ").strip()
        try:
            val = cast(val)
            if valid and val not in valid:
                print(f"    ✗ Enter one of: {valid}")
                continue
            return val
        except ValueError:
            print(f"    ✗ Invalid input. Expected {cast.__name__}.")


def collect_applicant() -> dict:
    """
    Collect applicant data interactively from the console.
    Returns a dict with raw (pre-encoded) values.
    """
    print("\n-- Enter Applicant Details --------------------------------")
    print("Note: Income values are treated as monthly income in the dataset's")
    print("original unit. LoanAmount is in thousands, so 120 means 120,000.\n")
    applicant_raw = {
        "Gender":          get_input("Gender (Male/Female)",    ["Male", "Female"]),
        "Married":         get_input("Married (Yes/No)",        ["Yes", "No"]),
        "Dependents":      get_input("Dependents (0/1/2/3+)",   ["0","1","2","3+"]),
        "Education":       get_input("Education (Graduate/Not Graduate)", ["Graduate", "Not Graduate"]),
        "Self_Employed":   get_input("Self Employed (Yes/No)",  ["Yes", "No"]),
        "ApplicantIncome": get_input("Applicant Monthly Income (number, dataset unit)", cast=float),
        "CoapplicantIncome": get_input("Coapplicant Monthly Income (0 if none, dataset unit)", cast=float),
        "LoanAmount":      get_input("Loan Amount in thousands (example: 120 means 120,000)", cast=float),
        "Loan_Amount_Term":get_input("Loan Term in months (e.g. 360)", cast=float),
        "Credit_History":  get_input("Credit History (1=Good, 0=Bad)", [1, 0], cast=int),
        "Property_Area":   get_input("Property Area (Urban/Semiurban/Rural)",
                                     ["Urban", "Semiurban", "Rural"]),
    }
    return applicant_raw


def encode_applicant(applicant_raw: dict) -> dict:
    """
    Simple encoding to match training encoding.
    Mirrors the logic in preprocessing.py for demo use.
    """
    encodings = {
        "Gender":       {"Male": 1, "Female": 0},
        "Married":      {"Yes": 1, "No": 0},
        "Dependents":   {"0": 0, "1": 1, "2": 2, "3+": 3},
        "Education":    {"Graduate": 0, "Not Graduate": 1},
        "Self_Employed":{"Yes": 1, "No": 0},
        "Property_Area":{"Rural": 0, "Semiurban": 1, "Urban": 2},
    }

    encoded = {}
    for k, v in applicant_raw.items():
        if k in encodings:
            encoded[k] = encodings[k].get(str(v), v)
        else:
            encoded[k] = v

    return encoded


def print_result(result: dict) -> None:
    """Pretty-print the expert system decision."""
    decision = result["decision"]
    colour = {
        "Approve":       "[APPROVED]",
        "Manual Review": "[REVIEW]",
        "Reject":        "[REJECTED]"
    }.get(decision, "")

    print("\n" + "=" * 60)
    print(f"  DECISION: {colour}  {decision.upper()}")
    print(f"  Confidence Score: {result['confidence']:.2%}")
    print("=" * 60)
    print(f"  Knowledge Base Score : {result['kb_score']:.4f}")
    print(f"  Bayesian Probability : {result['bayes_prob']:.4f}")
    print(f"  ML Model Probability : {result['ml_prob']:.4f}")
    print(f"  Rules Fired          : {result['fired_rules'] or 'None'}")
    print("\n  Candidate Rankings:")
    for c in result["all_candidates"]:
        print(f"    {c['label']:<15} score={c['score']:.4f}")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  Loan Approval Expert System - Interactive Demo")
    print("=" * 60)

    try:
        es = load_system()
        print("  System loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    while True:
        applicant_raw = collect_applicant()
        applicant_enc = encode_applicant(applicant_raw)
        result        = es.predict(applicant_enc)
        print_result(result)

        again = input("  Evaluate another applicant? (yes/no): ").strip().lower()
        if again != "yes":
            print("\n  Exiting demo. Thank you!\n")
            break


if __name__ == "__main__":
    main()

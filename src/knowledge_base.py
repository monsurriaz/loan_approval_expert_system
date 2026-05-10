# =============================================================================
# knowledge_base.py
# Component 2: Knowledge Representation — data-driven expert rules
# Thresholds are extracted from training data, not hardcoded
# =============================================================================

import pandas as pd
from src.config import TARGET_COL


def build_knowledge_base(train_df: pd.DataFrame) -> dict:
    """
    Extract thresholds from training data and build an expert rule dictionary.
    Thresholds are computed from the approved applicants' statistics so rules
    are data-driven rather than manually hardcoded.

    Args:
        train_df: Processed training DataFrame (numeric encoded).

    Returns:
        Dictionary containing thresholds and rule metadata.
    """
    approved = train_df[train_df[TARGET_COL] == 1]

    knowledge = {
        "income_threshold":      approved["ApplicantIncome"].median(),
        "loan_amount_threshold":  approved["LoanAmount"].median(),
        "term_standard":          360.0,   # most common loan term in months
        "rules": [
            {
                "id": "R1",
                "name": "Good Credit History",
                "description": "Applicant has positive credit history",
                "weight": 0.40,
            },
            {
                "id": "R2",
                "name": "Income Above Threshold",
                "description": "Applicant income exceeds median approved income",
                "weight": 0.25,
            },
            {
                "id": "R3",
                "name": "Reasonable Loan Amount",
                "description": "Loan amount is at or below median approved loan",
                "weight": 0.20,
            },
            {
                "id": "R4",
                "name": "Employed",
                "description": "Applicant is not self-employed (lower income volatility)",
                "weight": 0.10,
            },
            {
                "id": "R5",
                "name": "Standard Loan Term",
                "description": "Loan term is the standard 360 months",
                "weight": 0.05,
            },
        ]
    }

    print(f"[knowledge_base] Income threshold  : {knowledge['income_threshold']:.0f}")
    print(f"[knowledge_base] LoanAmount threshold: {knowledge['loan_amount_threshold']:.0f}")
    return knowledge


def apply_rules(applicant: dict, knowledge: dict) -> tuple[float, list]:
    """
    Evaluate expert rules against a single applicant and return a
    weighted confidence score between 0.0 and 1.0.

    Args:
        applicant : Dict of feature_name → encoded numeric value.
        knowledge : Knowledge base dict from build_knowledge_base().

    Returns:
        (score, fired_rules)
        score       : float in [0, 1] representing rule-based approval confidence.
        fired_rules : list of rule IDs that matched (for explanation).
    """
    score       = 0.0
    fired_rules = []

    for rule in knowledge["rules"]:
        rid = rule["id"]

        if rid == "R1":
            # Credit_History encoded: 1 = good, 0 = bad
            if applicant.get("Credit_History", 0) == 1:
                score += rule["weight"]
                fired_rules.append(rid)

        elif rid == "R2":
            total_income = (
                applicant.get("ApplicantIncome", 0)
                + applicant.get("CoapplicantIncome", 0)
            )
            if total_income >= knowledge["income_threshold"]:
                score += rule["weight"]
                fired_rules.append(rid)

        elif rid == "R3":
            if applicant.get("LoanAmount", float("inf")) <= knowledge["loan_amount_threshold"]:
                score += rule["weight"]
                fired_rules.append(rid)

        elif rid == "R4":
            # Self_Employed encoded; 0 typically means "No" after pd.Categorical
            if applicant.get("Self_Employed", 1) == 0:
                score += rule["weight"]
                fired_rules.append(rid)

        elif rid == "R5":
            if applicant.get("Loan_Amount_Term", 0) == knowledge["term_standard"]:
                score += rule["weight"]
                fired_rules.append(rid)

    return round(score, 4), fired_rules

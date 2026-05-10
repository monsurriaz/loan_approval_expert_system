# =============================================================================
# search.py
# Component 4: Search Algorithm - Greedy Best-First Search
# Ranks three candidate decisions (Approve / Manual Review / Reject)
# using a weighted heuristic of KB + Bayesian + ML scores
# =============================================================================

from src.config import (
    WEIGHT_KB, WEIGHT_BAYES, WEIGHT_ML,
    APPROVE_THRESHOLD, MANUAL_REVIEW_THRESHOLD
)


# -- Candidate decisions ---------------------------------------------------

CANDIDATES = ["Approve", "Manual Review", "Reject"]


# -- State representation --------------------------------------------------

class DecisionState:
    """
    Represents a candidate decision node in the search space.

    Attributes:
        label      : One of "Approve", "Manual Review", "Reject".
        score      : Heuristic score in [0, 1]. Higher = better for approval.
        kb_score   : Contribution from knowledge base rules.
        bayes_prob : Contribution from Bayesian reasoning.
        ml_prob    : Contribution from ML model.
    """
    def __init__(
        self,
        label: str,
        kb_score: float,
        bayes_prob: float,
        ml_prob: float
    ):
        self.label      = label
        self.kb_score   = kb_score
        self.bayes_prob = bayes_prob
        self.ml_prob    = ml_prob
        self.score      = self._compute_heuristic()

    def _compute_heuristic(self) -> float:
        """
        Weighted combination of all three component scores.
        This is the heuristic function h(state) for Best-First Search.
        """
        raw = (
            WEIGHT_KB    * self.kb_score
            + WEIGHT_BAYES * self.bayes_prob
            + WEIGHT_ML    * self.ml_prob
        )
        return round(raw, 4)

    def __repr__(self):
        return (
            f"DecisionState(label='{self.label}', score={self.score:.4f}, "
            f"kb={self.kb_score:.2f}, bayes={self.bayes_prob:.2f}, ml={self.ml_prob:.2f})"
        )


# -- Heuristic mapping -----------------------------------------------------

def _label_for_score(combined_score: float) -> str:
    """
    Map a numeric combined score to one of the three decision labels.

    Args:
        combined_score: Weighted heuristic score in [0, 1].

    Returns:
        Decision label string.
    """
    if combined_score >= APPROVE_THRESHOLD:
        return "Approve"
    elif combined_score >= MANUAL_REVIEW_THRESHOLD:
        return "Manual Review"
    else:
        return "Reject"


# -- Core search -----------------------------------------------------------

def score_candidates(
    kb_score: float,
    bayes_prob: float,
    ml_prob: float
) -> list[DecisionState]:
    """
    Create a DecisionState for each candidate label and score them.

    Each candidate uses the same component scores but represents a different
    decision label. The search algorithm will select the best-matching label.

    Args:
        kb_score  : Float in [0, 1] from knowledge_base.apply_rules().
        bayes_prob: Float in [0, 1] from bayesian_reasoning.bayesian_score().
        ml_prob   : Float in [0, 1] from ml_model.predict() probability output.

    Returns:
        List of DecisionState objects, one per candidate label.
    """
    states = [
        DecisionState(label, kb_score, bayes_prob, ml_prob)
        for label in CANDIDATES
    ]
    return states


def greedy_best_first_search(
    kb_score: float,
    bayes_prob: float,
    ml_prob: float
) -> dict:
    """
    Greedy Best-First Search over candidate decisions.

    Algorithm:
    1. Generate all candidate DecisionState nodes.
    2. Compute heuristic score h(state) for each (weighted combination).
    3. Select the candidate whose label matches the heuristic score range.
    4. Return the best decision with full explanation.

    Args:
        kb_score  : Rule-based score from knowledge base.
        bayes_prob: Approval probability from Bayesian reasoner.
        ml_prob   : Approval probability from ML model.

    Returns:
        Dict with keys: decision, confidence, kb_score, bayes_prob, ml_prob,
        all_candidates (ranked list for transparency).
    """
    states = score_candidates(kb_score, bayes_prob, ml_prob)

    # Combined weighted score
    combined = (
        WEIGHT_KB    * kb_score
        + WEIGHT_BAYES * bayes_prob
        + WEIGHT_ML    * ml_prob
    )
    combined = round(combined, 4)

    # Greedy selection: pick label matching score range (best-first)
    best_label = _label_for_score(combined)
    best_state = next(s for s in states if s.label == best_label)

    # Rank all candidates by descending score for report transparency
    ranked = sorted(states, key=lambda s: s.score, reverse=True)

    return {
        "decision":       best_label,
        "confidence":     combined,
        "kb_score":       kb_score,
        "bayes_prob":     bayes_prob,
        "ml_prob":        ml_prob,
        "all_candidates": [
            {"label": s.label, "score": s.score} for s in ranked
        ]
    }

# =============================================================================
# expert_system.py
# Integration Layer - combines KB + Bayesian + ML via Best-First Search
# ExpertSystem.predict() returns decision + confidence + full explanation
# =============================================================================

import pandas as pd
from src.knowledge_base    import apply_rules
from src.bayesian_reasoning import bayesian_score, bin_numerical_columns
from src.ml_model           import load_model, predict
from src.search             import greedy_best_first_search
from src.config             import MODEL_PATH


class ExpertSystem:
    """
    AI-Based Expert System for Loan Approval Decision Support.

    Architecture:
        Applicant data
             |
             |---> Knowledge Base (rules)      ---> kb_score   -|
             |---> Bayesian Reasoner            ---> bayes_prob  |---> Best-First Search ---> Decision
             +---> ML Model (RandomForest)      ---> ml_prob    -|

    Usage:
        es = ExpertSystem(knowledge, priors, likelihoods, bin_edges, model)
        result = es.predict(applicant_dict)
    """

    def __init__(
        self,
        knowledge:    dict,
        priors:       dict,
        likelihoods:  dict,
        bin_edges:    dict,
        model,
        feature_names: list
    ):
        """
        Initialise the expert system with all trained components.

        Args:
            knowledge    : Knowledge base dict from build_knowledge_base().
            priors       : Bayesian prior probabilities.
            likelihoods  : Bayesian conditional likelihoods.
            bin_edges    : Bin edges from training (to bin test applicants).
            model        : Fitted RandomForestClassifier.
            feature_names: Ordered list of feature names used during training.
        """
        self.knowledge     = knowledge
        self.priors        = priors
        self.likelihoods   = likelihoods
        self.bin_edges     = bin_edges
        self.model         = model
        self.feature_names = feature_names

    def predict(self, applicant: dict) -> dict:
        """
        Generate a loan approval decision for a single applicant.

        Steps:
        1. Apply expert rules -> kb_score
        2. Bin numeric features, then compute Bayesian probability -> bayes_prob
        3. Run ML model -> ml_prob
        4. Greedy Best-First Search combines all three -> final decision

        Args:
            applicant: Dict mapping feature_name -> encoded numeric value.
                       Must include all features in self.feature_names.

        Returns:
            Dict with:
                decision       : "Approve" | "Manual Review" | "Reject"
                confidence     : Combined weighted score in [0, 1]
                kb_score       : Knowledge base contribution
                bayes_prob     : Bayesian probability contribution
                ml_prob        : ML model probability contribution
                fired_rules    : List of KB rule IDs that matched
                all_candidates : Ranked list of all decision candidates
                explanation    : Human-readable string
        """
        # -- Step 1: Knowledge Base ----------------------------------------
        kb_score, fired_rules = apply_rules(applicant, self.knowledge)

        # -- Step 2: Bayesian Reasoning ------------------------------------
        applicant_df = pd.DataFrame([applicant])
        binned_df, _ = bin_numerical_columns(applicant_df, bin_edges=self.bin_edges)
        applicant_binned = binned_df.iloc[0].to_dict()
        bayes_prob = bayesian_score(applicant_binned, self.priors, self.likelihoods)

        # -- Step 3: ML Model -----------------------------------------------
        applicant_ml_df = pd.DataFrame([applicant])[self.feature_names]
        _, ml_probs = predict(self.model, applicant_ml_df)
        ml_prob = float(ml_probs[0])

        # -- Step 4: Best-First Search (integration) -----------------------
        search_result = greedy_best_first_search(kb_score, bayes_prob, ml_prob)

        # -- Step 5: Build explanation string ----------------------------
        rule_text = ", ".join(fired_rules) if fired_rules else "none"
        explanation = (
            f"Decision: {search_result['decision']} "
            f"(confidence={search_result['confidence']:.2f}) | "
            f"KB rules fired: [{rule_text}] (score={kb_score:.2f}) | "
            f"Bayesian P(Approved)={bayes_prob:.2f} | "
            f"ML P(Approved)={ml_prob:.2f}"
        )

        return {
            "decision":        search_result["decision"],
            "confidence":      search_result["confidence"],
            "kb_score":        kb_score,
            "bayes_prob":      bayes_prob,
            "ml_prob":         ml_prob,
            "fired_rules":     fired_rules,
            "all_candidates":  search_result["all_candidates"],
            "explanation":     explanation,
        }

    def predict_batch(self, X: pd.DataFrame) -> list[dict]:
        """
        Run predict() on every row of a DataFrame.
        Used by evaluation.py to generate predictions.csv.

        Args:
            X: Feature DataFrame (encoded, same columns as training).

        Returns:
            List of result dicts, one per row.
        """
        results = []
        for _, row in X.iterrows():
            applicant = row.to_dict()
            results.append(self.predict(applicant))
        return results

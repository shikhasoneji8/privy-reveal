# interpret_service.py
"""
Local explanations using InterpretML (ExplainableBoostingClassifier)
for your privacy-policy sentence classifier.

Usage (from your Gradio app, for example):

    from interpret_service import explain_with_ebm

    result = explain_with_ebm(sentence_text, top_k=10)
    # result is a dict:
    # {
    #   "pred_label": str,
    #   "confidence": float,
    #   "top_features": List[Tuple[str, float]],
    #   "class_probs": List[Tuple[str, float]],
    # }

This uses a surrogate model (EBM) trained on the same CSV you already
use for the explainer pipeline in core_logic.py.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from interpret.glassbox import ExplainableBoostingClassifier

# We re-use your existing CSV loader
from core_logic import _load_training_csv

# -------------------------------------------------------------------
# Module-level cache so we only train the EBM once
# -------------------------------------------------------------------
_VEC: CountVectorizer = None
_EBM: ExplainableBoostingClassifier = None
_CLASS_NAMES: List[str] = None


def _ensure_ebm_model(max_features: int = 5000) -> None:
    """
    Train the EBM + vectorizer once and cache them.
    """
    global _VEC, _EBM, _CLASS_NAMES
    if _EBM is not None and _VEC is not None:
        return

    # Load your training CSV via core_logic
    df, text_col, label_col = _load_training_csv()
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    # Bag-of-words over 1–2 grams to keep things manageable
    vec = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features
    )
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()

    # Explainable Boosting Model (GAM-style, inherently interpretable)
    ebm = ExplainableBoostingClassifier(
        feature_names=feature_names,
        interactions=0,             # just main effects to keep it simple
        outer_bags=4,
        inner_bags=0,
        max_leaves=3,
        max_bins=64,
        learning_rate=0.05,
        random_state=0
    )
    ebm.fit(X, labels)

    _VEC = vec
    _EBM = ebm
    _CLASS_NAMES = list(ebm.classes_)

    print(
        f"[INTERPRET] EBM trained on {len(texts)} examples, "
        f"{len(feature_names)} features, {len(_CLASS_NAMES)} classes."
    )


def explain_with_ebm(text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Explain a single sentence using the cached EBM model.

    Returns a dict with:
        pred_label: str
        confidence: float
        top_features: List[(word, contribution)]
        class_probs: List[(label, probability)]
    """
    _ensure_ebm_model()

    if not text or not text.strip():
        return {
            "pred_label": None,
            "confidence": 0.0,
            "top_features": [],
            "class_probs": []
        }

    # Vectorize the sentence
    X = _VEC.transform([text])

    # Class probabilities from the EBM
    proba = _EBM.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = _EBM.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    # Local explanation
    # ebm_local.data() returns a dict with scores per feature
    local_expl = _EBM.explain_local(X)
    data = local_expl.data()

    # For a single instance: scores[0] is a list of contributions
    contribs: List[float] = data["scores"][0]
    feat_names: List[str] = data["feature_names"]

    # Pair (feature, contribution) and sort by |contribution|
    pairs = [
        (feat_names[i], float(contribs[i]))
        for i in range(len(feat_names))
        if contribs[i] != 0.0  # drop true zeros
    ]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)

    # Take top_k for display
    top_features = pairs[:top_k]

    # Pack probabilities as (label, prob) pairs
    class_probs: List[Tuple[str, float]] = [
        (str(label), float(p)) for label, p in zip(_EBM.classes_, proba)
    ]

    return {
        "pred_label": str(pred_label),
        "confidence": confidence,
        "top_features": top_features,
        "class_probs": class_probs,
    }


# Optional: tiny CLI test
if __name__ == "__main__":
    _ensure_ebm_model()
    sample = "Zoom does not allow children under the age of 16 to sign up for a Zoom account."
    res = explain_with_ebm(sample, top_k=8)
    print("Predicted:", res["pred_label"], "conf:", res["confidence"])
    print("Top features:")
    for w, c in res["top_features"]:
        print(f"  {w:20s} {c:+.3f}")
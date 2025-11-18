# lime_service_direct.py
from lime.lime_text import LimeTextExplainer
import numpy as np
import re

from core_logic import get_explainer_pipeline, get_class_names

_EXPLAINER = None
_PIPE = None

def _ensure():
    global _EXPLAINER, _PIPE
    if _PIPE is None:
        _PIPE = get_explainer_pipeline()
    if _EXPLAINER is None:
        classes = get_class_names() or None
        _EXPLAINER = LimeTextExplainer(class_names=classes)

def _clean_token(tok: str) -> str:
    """Collapse whitespace, keep visible punctuation, strip control chars."""
    if tok is None:
        return ""
    t = re.sub(r"\s+", " ", str(tok)).strip()
    # If you prefer to *show* punctuation instead of dropping it, comment next line.
    return t

def explain_text(text, top_labels=1, num_features=12, num_samples=800, include_native=False):
    _ensure()

    exp = _EXPLAINER.explain_instance(
        text,
        _PIPE.predict_proba,
        top_labels=top_labels,
        num_features=num_features,
        num_samples=num_samples
    )
    label_idx = int(exp.top_labels[0])

    # token list from LIME
    toks_attr = exp.domain_mapper.indexed_string.as_list
    toks = toks_attr() if callable(toks_attr) else list(toks_attr)

    raw_map = exp.as_map()[label_idx]  # [(token_index, weight), ...]
    token_weights = []
    for i, w in raw_map:
        tok = toks[i] if 0 <= i < len(toks) else ""
        tok = _clean_token(tok)
        if tok:  # drop empties
            token_weights.append((tok, float(w)))

    # Sort by |weight| so strongest factors appear first
    token_weights.sort(key=lambda tw: abs(tw[1]), reverse=True)

    # Class probabilities
    proba = _PIPE.predict_proba([text])[0]
    classes = list(_PIPE.classes_)
    pred_idx = int(np.argmax(proba))
    pred_label = classes[pred_idx]
    confidence = float(proba[pred_idx])

    # Full class list sorted by prob (for the “other classes” section)
    classes_sorted = sorted(
        [(cls, float(p)) for cls, p in zip(classes, proba)],
        key=lambda x: -x[1]
    )

    out = {
        "pred_label": pred_label,
        "confidence": confidence,
        "token_weights": token_weights,
        "classes_sorted": classes_sorted,      # NEW: full list, high→low
        "pred_idx": pred_idx,
    }

    if include_native:
        out["lime_native_html"] = exp.as_html(labels=[label_idx])

    return out
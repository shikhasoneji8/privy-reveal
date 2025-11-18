from typing import List, Tuple
import numpy as np
import shap
import torch
import models  # your loaded tokenizer/model/id2label/device
import config

# cache
_EXPLAINER = None

def _predict_proba_texts(texts: List[str]):
    # -> np.array [batch, n_classes]
    enc = models.tokenizer(
        texts, truncation=True, max_length=config.MAX_LEN, padding=True, return_tensors="pt"
    )
    enc = {k: v.to(models.device) for k, v in enc.items()}
    with torch.no_grad():
        out = models.model(**enc)
    logits = out.logits.detach().cpu().numpy()
    # softmax
    x = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(x, dtype=np.float64)
    return e / e.sum(axis=1, keepdims=True)

def _ensure():
    global _EXPLAINER
    if _EXPLAINER is not None:
        return
    masker = shap.maskers.Text(tokenizer=lambda s: s.split())
    # IMPORTANT: pass a callable that accepts List[str] and returns proba
    _EXPLAINER = shap.Explainer(
        _predict_proba_texts,
        masker=masker,
        output_names=list(models.id2label.values())  # or models.id2label (if list)
    )

def explain_text(text: str, num_features: int = 10, num_samples: int = 200):
    _ensure()
    proba = _predict_proba_texts([text])[0]
    pred_idx = int(np.argmax(proba))
    pred_label = list(models.id2label.values())[pred_idx]
    confidence = float(proba[pred_idx])

    exp = _EXPLAINER([text], nsamples=num_samples)

    # ---- robust extraction (same as earlier)
    tokens = exp.data[0]
    if not isinstance(tokens, (list, tuple, np.ndarray)):
        tokens = [str(tokens)]
    else:
        tokens = list(tokens)

    vals = exp.values
    if isinstance(vals, np.ndarray):
        token_weights_arr = vals[0, :, pred_idx]
    elif isinstance(vals, list):
        if len(vals) == len(models.id2label):
            class_block = np.array(vals[pred_idx])
            token_weights_arr = class_block[0] if class_block.ndim == 2 else class_block
        else:
            sample_block = np.array(vals[0])
            token_weights_arr = sample_block[:, pred_idx] if sample_block.ndim == 2 else sample_block
    elif isinstance(vals, dict):
        label = pred_label
        class_block = np.array(vals.get(label, vals.get(str(label), vals.get(pred_idx))))
        if class_block is None:
            raise RuntimeError(f"Class {label} not found in SHAP dict.")
        token_weights_arr = class_block[0] if class_block.ndim == 2 else class_block
    else:
        raise RuntimeError(f"Unsupported SHAP values type: {type(vals)}")

    pairs = []
    for tok, w in zip(tokens, token_weights_arr):
        tok = (tok or "").strip()
        if tok and abs(float(w)) > 1e-6:
            pairs.append((tok, float(w)))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return str(pred_label), pairs[:num_features], confidence
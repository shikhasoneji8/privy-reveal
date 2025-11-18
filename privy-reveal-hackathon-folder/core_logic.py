# core_logic.py
import re
import numpy as np
import torch
from typing import List, Tuple
import tempfile
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd, os, sys
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression


# Import from our custom modules
import models
import config
import data_loader

# --- CRITICAL FIX: Use a dictionary for safer module-level caching ---
_CACHE = {'pipeline': None} 
# --------------------------------------------------------------------

# Try these names in order for text and label
TEXT_CANDIDATES  = ["Text", "Data", "Sentence", "text", "content", "snippet", "Description"]
LABEL_CANDIDATES = ["Case", "Predicted_Category", "label", "Label"]
# Columns used internally for explainer model loading
_TEXT_COLS = ["Data","Text","Sentence","text","content","snippet"]
_LABEL_COLS = ["Case","Predicted_Category","label","Label"]


def _resolve_cols(df: pd.DataFrame) -> Tuple[str, str]:
    text_col = next((c for c in TEXT_CANDIDATES  if c in df.columns), None)
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if not text_col or not label_col:
        raise ValueError(f"[TRAIN] Missing columns. Have={list(df.columns)} "
                         f"Need text in {TEXT_CANDIDATES} and label in {LABEL_CANDIDATES}")
    return text_col, label_col

def _load_training_csv() -> Tuple[pd.DataFrame, str, str]:
    # ENV override else fallbacks
    csv_path = os.getenv("PRIV_TRAIN_CSV", None)
    if csv_path is None:
        # common local names you used
        for candidate in ["test_data_clean.csv", "test_data_clean.csv"]:
            if os.path.exists(candidate):
                csv_path = candidate
                break
    if csv_path is None:
        raise FileNotFoundError(
            "[TRAIN] No training CSV found. Set PRIV_TRAIN_CSV env var or place "
            "'test_data_clean.csv' in the app folder."
        )

    df = pd.read_csv(csv_path)
    text_col, label_col = _resolve_cols(df)
    # Basic cleaning
    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "Text", label_col: "Label"})
    # Ensure strings
    df["Text"] = df["Text"].astype(str)
    df["Label"] = df["Label"].astype(str)
    print(f"[TRAIN] Loaded {len(df)} rows from {csv_path} (text='Text', label='Label')")
    return df, "Text", "Label"

# =========================
# Utility Functions
# =========================
def sent_split(text: str) -> List[str]:
    """Simple sentence splitter."""
    if not text: return []
    parts = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
    out = [q.strip() for p in parts for q in re.split(r'[\n;]+', p) if q.strip()]
    return out or [text.strip()]

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x, dtype=np.float64)
    return e / e.sum(axis=-1, keepdims=True)

# =========================
# Core Classification Logic
# =========================
def classify_sentences(sentences: List[str], threshold: float = config.THRESHOLD) -> pd.DataFrame:
    """Classifies a list of sentences and returns a structured DataFrame."""
    if not sentences:
        return pd.DataFrame()

    enc = models.tokenizer(
        sentences, truncation=True, max_length=config.MAX_LEN, padding=True, return_tensors="pt"
    )
    enc = {k: v.to(models.device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = models.model(**enc)
        
    logits = out.logits.detach().cpu().numpy()
    probs = softmax_np(logits)

    rows = []
    for s, pr in zip(sentences, probs):
        top_idx = np.argmax(pr)
        conf = float(pr[top_idx])

        if conf >= threshold:
            case = models.id2label[top_idx]
            rating = data_loader.RATING_MAP.get(case, "neutral").lower()
            case_score = data_loader.WEIGHT_MAP.get(case, 0.0)
            cat, sub, fine = data_loader.CAT_MAP.get(case, ("Other", "—", "—"))
            
            top3_indices = np.argsort(-pr)[:config.TOP_K]
            top3_str = ", ".join([f"{models.id2label[i]} ({pr[i]:.2f})" for i in top3_indices])

            rows.append({
                "text": s,
                "top1_label": case,
                "rating": rating,
                "Category": cat,
                "SubCategory": sub,
                "FineGrained": fine,
                "case_score": case_score,
                "confidence": round(conf, 4),
                "top3": top3_str,
                "icon_uri": data_loader.ICON_URIS.get(rating, ""),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "id", range(1, len(df) + 1))
    return df

# =========================
# Grading and Summarization
# =========================
def determine_counts(pred_df: pd.DataFrame) -> dict:
    counts = {"good": 0.0, "bad": 0.0, "blocker": 0.0}
    if pred_df is None or pred_df.empty: return counts
    for rating in pred_df["rating"]:
        if rating in config.VALID_RATINGS:
            counts[rating] += 1.0
    return counts

def determine_balance(counts: dict) -> float:
    return counts.get("good", 0.0) - counts.get("bad", 0.0) - (counts.get("blocker", 0.0) * 3.0)

def calculate_grade(counts: dict) -> str:
    if sum(counts.values()) == 0: return "N/A"
    balance = determine_balance(counts)
    if balance <= -10 or counts.get("blocker", 0.0) > counts.get("good", 0.0): return "E"
    if counts.get("blocker", 0.0) >= 3 or counts.get("bad", 0.0) > counts.get("good", 0.0): return "D"
    if balance < 5: return "C"
    if counts.get("bad", 0.0) > 0: return "B"
    return "A"

def summarize_grade(pred_df: pd.DataFrame) -> str:
    counts = determine_counts(pred_df)
    grade = calculate_grade(counts)
    return f"## Overall grade: **{grade}**"

# =========================
# Post-Processing and Helpers
# =========================
def save_df_csv(df: pd.DataFrame, filename: str) -> str:
    """Saves a DataFrame to a temporary file for download."""
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, filename)
    if df is None or df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return path
    
    cols_to_save = [c for c in [
        "id", "text", "top1_label", "rating", "Category", "SubCategory", 
        "FineGrained", "confidence", "case_score", "top3"
    ] if c in df.columns]
    df.to_csv(path, index=False, columns=cols_to_save)
    return path

def _first_number(s) -> int:
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else 0

def merge_nearby_cases(df: pd.DataFrame, gap: int = 1) -> pd.DataFrame:
    """Merges adjacent DataFrame rows with the same classification."""
    if df is None or df.empty or 'id' not in df.columns:
        return df.copy()

    work = df.copy()
    work["__id_int"] = work["id"].apply(_first_number)
    work = work.sort_values("__id_int").reset_index(drop=True)

    merged, cluster = [], []
    
    group_cols = ["top1_label", "Category", "SubCategory", "FineGrained"]

    def flush():
        if not cluster: return
        
        # Use first row as base
        base = cluster[0].copy()
        
        # Combine IDs
        start_id, end_id = cluster[0]["id"], cluster[-1]["id"]
        base["id"] = str(start_id) if start_id == end_id else f"{start_id}–{end_id}"
        
        # Merge text and average confidence
        texts = [str(r["text"]).strip() for r in cluster if str(r["text"]).strip()]
        confs = [float(r.get("confidence", 0.0)) for r in cluster]
        
        base["merged_text"] = " ".join(texts)
        base["confidence"] = round(float(np.mean(confs)), 4)
        
        merged.append(base)

    for _, row in work.iterrows():
        r = row.to_dict()
        if not cluster:
            cluster.append(r)
            continue
        
        prev = cluster[-1]
        is_close = (r["__id_int"] - prev["__id_int"]) <= gap
        is_same_group = all(str(r[col]) == str(prev[col]) for col in group_cols)

        if is_close and is_same_group:
            cluster.append(r)
        else:
            flush()
            cluster = [r]
    flush()

    if not merged:
        return pd.DataFrame()

    out = pd.DataFrame(merged)
    # Re-sort by the original first ID number
    out = out.sort_values(by="id", key=lambda s: s.map(_first_number)).reset_index(drop=True)
    return out


# In core_logic.py, add this entire block to the end of the file.

import re

# =========================
# Lightweight Retrieval QA
# =========================
_STOP = {
    "the","a","an","of","to","and","or","in","on","for","with","by","as","at","from",
    "that","this","these","those","it","its","is","are","was","were","be","been","being",
    "we","you","they","he","she","their","our","your","i","me","my","mine","ours","yours",
    "will","shall","can","could","should","would","may","might","do","does","did","not","no"
}

def _tok(s: str):
    """Simple tokenizer that removes stopwords and punctuation."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOP]
    return toks

def _score_overlap(q_tokens, s_tokens):
    """Scores sentence based on keyword overlap with the question."""
    if not q_tokens or not s_tokens:
        return 0.0
    qs = set(q_tokens)
    ss = set(s_tokens)
    intersection = len(qs & ss)
    # Simple score: intersection divided by the number of question tokens
    return intersection / len(qs) if len(qs) > 0 else 0.0

def retrieve_qa_answer(question: str, sentences, top_k: int = 5) -> str:
    """Finds and formats the most relevant sentences from the text for a given question."""
    question = (question or "").strip()
    if not question:
        return "Type a question about the pasted policy."
    if not sentences:
        return "No policy text available. Paste text and click Annotate first."

    q_tokens = _tok(question)
    scored_sentences = []
    for s in sentences:
        s_tokens = _tok(s)
        score = _score_overlap(q_tokens, s_tokens)
        if score > 0:
            scored_sentences.append((score, s))

    if not scored_sentences:
        return "I couldn’t find anything relevant in the current text."

    # Sort by score descending and take the top_k
    scored_sentences.sort(key=lambda x: -x[0])
    top_passages = [s for _, s in scored_sentences[:top_k]]

    def _highlight(text: str):
        """Highlights matching keywords in the text."""
        highlighted_text = text
        for token in sorted(set(q_tokens), key=len, reverse=True):
            if len(token) < 3: continue
            # Use regex for case-insensitive, whole-word matching
            highlighted_text = re.sub(rf"(?i)\b({re.escape(token)})\b", r"<mark>\1</mark>", highlighted_text)
        return highlighted_text

    bullets = "\n".join(f"- {_highlight(p)}" for p in top_passages)
    return f"**Most relevant passages**:\n\n{bullets}"



# =========================
# Explainer Pipeline support 
# =========================

# core_logic.py — replace the body of get_explainer_pipeline() with this version
def get_explainer_pipeline():
    """
    Return an object with:
      - predict_proba(List[str]) -> np.ndarray [N, num_classes]
      - classes_ -> np.ndarray of class names
    This wraps the HuggingFace model directly. No CSV training.
    """
    class HFProbaWrapper:
        def __init__(self):
            # get class names from your loaded model/id2label
            self._classes = np.array([models.id2label[i] for i in range(len(models.id2label))])

        @property
        def classes_(self):
            return self._classes

        def predict_proba(self, texts):
            enc = models.tokenizer(
                texts, truncation=True, max_length=config.MAX_LEN,
                padding=True, return_tensors="pt"
            )
            enc = {k: v.to(models.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = models.model(**enc).logits.detach().cpu().numpy()
            # use your existing softmax
            return softmax_np(logits)

    return HFProbaWrapper()


def get_class_names() -> List[str]:
    """Helper to retrieve class names from the trained explainer pipeline."""
    pipeline = get_explainer_pipeline()
    if pipeline and hasattr(pipeline, 'classes_'):
        return pipeline.classes_.tolist()
    return []

def get_pipeline_status():
    """Small helper to debug from a REPL."""
    p = get_explainer_pipeline()
    if p is None:
        return "Pipeline is None"
    ok = hasattr(p, "predict_proba")
    return f"Pipeline: {type(p).__name__}, predict_proba={ok}"
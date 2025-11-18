
import os
import re
import json
import numpy as np
import pandas as pd
import torch
import gradio as gr
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import base64, os
import html
# ---- SBERT + QA ----
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from sentence_transformers import SentenceTransformer, CrossEncoder
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # fast, solid
# OPTIONAL reranker (better answers if you can afford it)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def build_passage_index(passages: list):
    if not passages:
        return passages, None
    embs = sbert.encode(passages, normalize_embeddings=True)  # [N, d]
    return passages, embs

qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

# NLI for yes/no
nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_mod = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli = pipeline("text-classification", model=nli_mod, tokenizer=nli_tok, return_all_scores=True)

# optional small generator for short answers
gen = pipeline("text2text-generation", model="google/flan-t5-base")  # lightweight
# =========================
# Configuration
# =========================
# Point this to your local folder that already contains:
# config.json, model.safetensors, tokenizer.json, tokenizer_config.json,
# vocab.json (and merges.txt for BPE), special_tokens_map.json, training_args.bin (optional)
MODEL_DIR = "./privbert_model"   # change if your folder name is different
MAX_LEN = 512
TOP_K = 3
BATCH_SIZE = 8  # increase for faster CSV inference if you have enough memory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================
# Device selection
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))

# =========================
# Load tokenizer and model
# =========================
# Since you have all tokenizer files locally, load from MODEL_DIR
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

# Load the fine-tuned classifier (reads config.json + model.safetensors from MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,  # safe default across CPU/MPS
)
model.to(device)
model.eval()

# Determine label mapping to show human-friendly names when available
def _load_id2label_from_config(cfg) -> Dict[int, str]:
    # Hugging Face stores id2label as a dict with string keys in many cases
    # e.g., {"0": "Data Retention", "1": "Third-Party Sharing", ...}
    if hasattr(cfg, "id2label") and cfg.id2label:
        try:
            return {int(k): v for k, v in cfg.id2label.items()}
        except Exception:
            # already int-keyed or irregular; do a safe conversion
            out = {}
            for k, v in cfg.id2label.items():
                try:
                    out[int(k)] = v
                except Exception:
                    pass
            if out:
                return out
    # fallback: numeric ids only
    return {i: str(i) for i in range(cfg.num_labels)}

id2label = _load_id2label_from_config(model.config)

# =========================
# Helpers
# =========================
def sent_split(text: str) -> List[str]:
    """Simple sentence splitter. Replace with spaCy/NLTK if needed."""
    if not text:
        return []
    # split on punctuation + space
    parts = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
    out = []
    for p in parts:
        # also split on newlines/semicolons to avoid super long segments
        for q in re.split(r'[\n;]+', p):
            if q.strip():
                out.append(q.strip())
    return out or [text.strip()]

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x, dtype=np.float64)
    return e / e.sum(axis=-1, keepdims=True)


# === Case ratings (your CSV) ===



# --- Files ---
CASE_RATINGS_CSV = "./case_ratings.csv"       # has columns: case, class, score (as today)
EMNLP_CAT_SUB_CSV = "./emnlp_cat_sub.csv"     # has columns: Case, Category, Sub-Category, Fine-grained category

def load_case_and_categories(case_csv: str, cat_csv: str):
    """
    Returns:
        rating_map: {case -> rating 'good'|'bad'|'neutral'|'blocker'}
        weight_map: {case -> float score}
        cat_map:    {case -> (Category, SubCategory, FineGrained)}
    """
    # --- ratings/scores ---
    try:
        cf = pd.read_csv(case_csv)
    except Exception as e:
        print(f"WARNING: failed to read {case_csv}: {e}")
        cf = pd.DataFrame(columns=["case","class","score"])

    if "case" not in cf.columns:
        cf["case"] = []
    cf["case"] = cf["case"].astype(str)

    if "class" in cf.columns:
        cf["class"] = cf["class"].astype(str).str.strip().str.lower()
    else:
        cf["class"] = "neutral"

    if "score" in cf.columns:
        cf["score"] = pd.to_numeric(cf["score"], errors="coerce").fillna(0.0)
    else:
        cf["score"] = 0.0

    rating_map = {r["case"]: r["class"] for _, r in cf.iterrows()}
    weight_map = {r["case"]: float(r["score"]) for _, r in cf.iterrows()}

    # --- hierarchical categories ---
    try:
        hf = pd.read_csv(cat_csv)
    except Exception as e:
        print(f"WARNING: failed to read {cat_csv}: {e}")
        hf = pd.DataFrame(columns=["Case","Category","Sub-Category","Fine-grained category"])

    # normalize column names
    col_case = "Case"
    col_cat = "Category"
    col_sub = "Sub-Category"
    col_fine = "Fine-grained category"
    for need in [col_case, col_cat, col_sub, col_fine]:
        if need not in hf.columns:
            hf[need] = ""

    # clean strings
    for c in [col_case, col_cat, col_sub, col_fine]:
        hf[c] = hf[c].astype(str).fillna("").str.strip()

    cat_map = {
        row[col_case]: (row[col_cat] or "Other",
                        row[col_sub] or "—",
                        row[col_fine] or "—")
        for _, row in hf.iterrows()
    }

    return rating_map, weight_map, cat_map

RATING_MAP, WEIGHT_MAP, CAT_MAP = load_case_and_categories(CASE_RATINGS_CSV, EMNLP_CAT_SUB_CSV)

# helper for filters (we’ll filter by top-level Category)
ALL_CATEGORIES = sorted({ (v[0] or "Other") for v in CAT_MAP.values() })
VALID_RATINGS = {"good","bad","blocker"}  # unchanged


def batch_predict(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Return probs [N, C] for a list of texts using batched inference."""
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()  # [B, C]
            probs = softmax_np(logits)
            all_probs.append(probs)
    return np.vstack(all_probs) if all_probs else np.empty((0, model.config.num_labels))

def format_topk_row(probs_row: np.ndarray, k: int = TOP_K) -> List[Tuple[str, float]]:
    idxs = np.argsort(-probs_row)[:k]
    return [(id2label[int(i)], float(probs_row[i])) for i in idxs]

def classify_sentences(sentences, threshold=0.75):
    enc = tokenizer(
        sentences,
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits.detach().cpu().numpy()
    probs = softmax_np(logits)

    rows = []
    for s, pr in zip(sentences, probs):
        idxs = np.argsort(-pr)[:TOP_K]
        top1 = int(idxs[0])
        conf = float(pr[top1])
        if conf < threshold:
            continue

        case = id2label[top1]
        rating = RATING_MAP.get(case, "neutral").lower()
        case_score = WEIGHT_MAP.get(case, 0.0)

        # NEW: look up hierarchy; default to "Other/—/—" if the case is missing
        cat, sub, fine = CAT_MAP.get(case, ("Other", "—", "—"))

        rows.append({
            "text": s,
            "top1_label": case,
            "rating": rating,
            "Category": cat,
            "SubCategory": sub,
            "FineGrained": fine,
            "case_score": case_score,
            "confidence": round(conf, 4),
            "top3": ", ".join([f"{id2label[int(i)]} ({pr[i]:.2f})" for i in idxs]),
            "icon_uri": rating_icon_uri(rating),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "id", range(1, len(df)+1))
    return df


def determine_counts(pred_df: pd.DataFrame) -> dict:
    """
    Counts based on rating (ignores 'neutral' and unknown).
    If you want *weighted* counts, replace +1 with case_score.
    """
    counts = {"good": 0.0, "bad": 0.0, "blocker": 0.0}
    if pred_df is None or pred_df.empty:
        return counts
    for _, row in pred_df.iterrows():
        rating = str(row.get("rating", "neutral")).lower()
        if rating in VALID_RATINGS:
            counts[rating] += 1.0   # or WEIGHT_MAP.get(row["top1_label"], 1.0) for weighted counts
    return counts

def determine_balance(counts: dict) -> float:
    num_bad = counts.get("bad", 0.0)
    num_blocker = counts.get("blocker", 0.0)
    num_good = counts.get("good", 0.0)
    return num_good - num_bad - (num_blocker * 3.0)

def calculate_grade(counts: dict) -> str:
    total = counts.get("good", 0.0) + counts.get("bad", 0.0) + counts.get("blocker", 0.0)
    if total == 0:
        return "N/A"
    balance = determine_balance(counts)
    if balance <= -10 or counts.get("blocker", 0.0) > counts.get("good", 0.0):
        return "E"
    elif counts.get("blocker", 0.0) >= 3 or counts.get("bad", 0.0) > counts.get("good", 0.0):
        return "D"
    elif balance < 5:
        return "C"
    elif counts.get("bad", 0.0) > 0:
        return "B"
    else:
        return "A"


def weighted_summary(pred_df: pd.DataFrame) -> dict:
    # sum of weights by rating (ignores neutral)
    sums = {"good": 0.0, "bad": 0.0, "blocker": 0.0}
    if pred_df is None or pred_df.empty:
        return sums
    for _, row in pred_df.iterrows():
        rating = str(row.get("rating", "neutral")).lower()
        w = float(row.get("case_score", 0.0))
        if rating in VALID_RATINGS:
            sums[rating] += w
    return sums


def summarize_grade(pred_df: pd.DataFrame, show_details: bool = False) -> str:
    counts = determine_counts(pred_df)
    grade = calculate_grade(counts)
    if not show_details:
        return f"## Overall grade: **{grade}**"
    # (optional details you can turn on later)
    balance = determine_balance(counts)
    w = weighted_summary(pred_df)
    lines = [
        f"## Overall grade: **{grade}**",
        "",
        "#### Counts (unweighted)",
        f"- good: **{counts['good']}**",
        f"- bad: **{counts['bad']}**",
        f"- blocker: **{counts['blocker']}**",
        f"- balance = good - bad - 3×blocker = **{balance:.1f}**",
    ]
    if any(v > 0 for v in w.values()):
        lines += [
            "",
            "#### Weighted totals (sum of case score)",
            f"- good: **{w['good']:.1f}**",
            f"- bad: **{w['bad']:.1f}**",
            f"- blocker: **{w['blocker']:.1f}**",
        ]
    return "\n".join(lines)

# =========================
# Gradio callbacks
# =========================
def df_to_html(df: pd.DataFrame, font_px: int = 15) -> str:
    if df is None or df.empty:
        return "<div class='section'>No results above the threshold.</div>"

    # Ensure expected columns exist
    for c in ["id","icon_uri","rating","top1_label",
              "Category","SubCategory","FineGrained",
              "confidence","merged_text"]:
        if c not in df.columns:
            df[c] = ""
    
    rows = []
    for _, r in df.iterrows():
        rating = str(r["rating"]).lower()
        display_text = str(r.get("merged_text") or r.get("text") or "")
        if r["icon_uri"]:
            icon = (
                f"<img src='{r['icon_uri']}' alt='{rating}' "
                "style='display:block;margin:0 auto;width:28px;height:28px;object-fit:contain'/>"
            )
        else:
            emoji = {"good":"👍","bad":"👎","blocker":"❌","neutral":"⚪"}.get(rating, "⚪")
            icon = f"<span style='font-size:22px;display:block;text-align:center'>{emoji}</span>"

        row_cls = f"rate-{rating}"

        rows.append(
            f"<tr class='{row_cls}'>"
            f"<td class='num'>{int(r['id']) if str(r['id']).isdigit() else r['id']}</td>"
            # f"<td class='col-text'>{html.escape(str(r['text']))}</td>"
            f"<td class='col-text'>{html.escape(display_text)}</td>"
            f"<td class='col-case'>{html.escape(str(r['top1_label']))}</td>"
            f"<td class='col-rating'>{html.escape(rating)}</td>"
            f"<td class='col-icon'>{icon}</td>"
            f"<td class='col-cat'>{html.escape(str(r['Category']))}</td>"
            f"<td class='col-sub'>{html.escape(str(r['SubCategory']))}</td>"
            f"<td class='col-fine'>{html.escape(str(r['FineGrained']))}</td>"
            f"<td class='col-conf'>{float(r['confidence']):.2f}</td>"
            "</tr>"
        )

    # ⬇️ Build the table AFTER the loop, then return once.
    table = (
        f"<style>.predtbl{{font-size:{int(font_px)}px}}</style>"
        "<table class='predtbl' role='table' aria-label='Predictions table'>"
        "<thead><tr>"
        "<th scope='col' class='num'>#</th>"
        "<th scope='col' class='col-text'>text</th>"
        "<th scope='col' class='col-case'>case</th>"
        "<th scope='col' class='col-rating'>rating</th>"
        "<th scope='col' class='col-icon'></th>"
        "<th scope='col' class='col-cat'>Category</th>"
        "<th scope='col' class='col-sub'>Sub-Category</th>"
        "<th scope='col' class='col-fine'>Fine-grained</th>"
        "<th scope='col' class='col-conf'>confidence</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    return table

def classify_free_text(big_text: str, thr: float):
    empty_df = pd.DataFrame(columns=["id","text","top1_label","rating","case_score","confidence","top3","icon_uri"])
    if not big_text or not big_text.strip():
        return df_to_html(empty_df), "## Overall grade: **N/A**"
    df = classify_sentences(sent_split(big_text), threshold=float(thr))
    return df_to_html(df), summarize_grade(df, show_details=False)

def classify_csv(csv_file, text_column: str):
    empty_df = pd.DataFrame(columns=["id","text","top1_label","rating","case_score","confidence","top3","icon_uri"])
    if csv_file is None:
        return df_to_html(empty_df), "## Overall grade: **N/A**"
    try:
        df_in = pd.read_csv(csv_file.name)
    except Exception as e:
        err = pd.DataFrame({"error":[f"Failed to read CSV: {e}"]})
        return df_to_html(err), "## Overall grade: **N/A**"
    if text_column not in df_in.columns:
        err = pd.DataFrame({"error":[f"Column '{text_column}' not found. Available: {list(df_in.columns)}"]})
        return df_to_html(err), "## Overall grade: **N/A**"
    texts = df_in[text_column].astype(str).fillna("").tolist()
    df = classify_sentences(texts, threshold=0.75)
    return df_to_html(df), summarize_grade(df, show_details=False)


import base64, os, html

ICON_FILES = {
    "good":     "static/icons/good.png",
    "bad":      "static/icons/bad.png",
    "blocker":  "static/icons/blocker.png",
    "neutral":  "static/icons/neutral.png",
}

def _to_data_uri(path: str) -> str:
    # Resolve to absolute path so it works regardless of current working dir
    abs_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    with open(abs_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

def load_icon_uris(icon_files: dict) -> dict:
    uris = {}
    for k, p in icon_files.items():
        uris[k] = _to_data_uri(p) if os.path.exists(p) else ""
    return uris

for k, p in ICON_FILES.items():
    print("ICON PATH", k, os.path.exists(os.path.join(BASE_DIR, p)), os.path.join(BASE_DIR, p))

ICON_URIS = load_icon_uris(ICON_FILES)
print("ICON_URIS loaded:", {k: bool(v) for k, v in ICON_URIS.items()})

def rating_icon_uri(rating: str) -> str:
    return ICON_URIS.get(rating.lower(), "")



def save_df_csv(df: pd.DataFrame, filename: str) -> str:
    """Save DF to /tmp and return a path Gradio can offer for download."""
    import tempfile, os
    if df is None or df.empty:
        # create an empty file so the widget still shows something to click
        tmp = tempfile.gettempdir()
        path = os.path.join(tmp, filename)
        pd.DataFrame(columns=["id","rating","case","confidence","text"]).to_csv(path, index=False)
        return path
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, filename)
    # Save only the visible columns (you can add case_score/top3 if you want)
    cols = [c for c in [
        "id","text","top1_label","rating",
        "Category","SubCategory","FineGrained",
        "confidence","case_score","top3"
    ] if c in df.columns]
    df.to_csv(path, index=False, columns=cols)
    return path

THRESHOLD = 0.75

def build_help_html() -> str:
    return """
    <div class="help-tip" tabindex="0" aria-label="How the score is calculated">
      <span class="icon" aria-hidden="true">?</span>
      <div class="bubble" role="tooltip">
        <h4>How the score is calculated</h4>
        <ol style="padding-left:18px;margin:6px 0;">
          <li><b>Sentence classification:</b> each sentence is labeled with a ToS;DR case using the model.</li>
          <li><b>Mapping to ratings:</b> cases map to <i>good</i>, <i>bad</i>, or <i>blocker</i> (we ignore <i>neutral</i> for grading).</li>
          <li><b>Confidence threshold:</b> only sentences with confidence ≥ the slider are counted.</li>
          <li><b>Grade formula:</b> balance = <code>good − bad − 3×blocker</code> → letter grade.</li>
          <li><b>Filters:</b> Category filter also changes which sentences (and the grade) you see.</li>
        </ol>
      </div>
    </div>
    """

# --- helpers: lock threshold is passed in as `thr` ---
def _classify_free_text_raw(big_text, thr, show_scores_col):
    empty = pd.DataFrame(columns=["id","text","top1_label","rating","case_score","confidence","top3","icon_uri"])
    if not big_text or not big_text.strip():
        return empty, "## Overall grade: **N/A**"

    thr = float(thr)
    df  = classify_sentences(sent_split(big_text), threshold=thr)

    # grade must be computed on the full raw df (needs 'rating')
    grade_md = summarize_grade(df, show_details=False)

    # hide columns only for display, after we have the grade
    # if not show_scores_col and not df.empty:
    #     for col in ("rating","case_score"):
    #         if col in df.columns:
    #             df = df.drop(columns=[col])

    return df, grade_md


def _classify_csv_raw(fileobj, text_col, thr, show_scores_col):
    empty = pd.DataFrame(columns=["id","text","top1_label","rating","case_score","confidence","top3","icon_uri"])
    if not fileobj:
        return empty, "## Overall grade: **N/A**"
    try:
        raw = pd.read_csv(fileobj.name)
    except Exception as e:
        return pd.DataFrame({"text":[f"Failed to read CSV: {e}"]}), "## Overall grade: **N/A**"
    if text_col not in raw.columns:
        return pd.DataFrame({"text":[f"Column '{text_col}' not found. Available: {list(raw.columns)}"]}), "## Overall grade: **N/A**"

    thr = float(thr)
    df  = classify_sentences(raw[text_col].astype(str).fillna("").tolist(), threshold=thr)

    grade_md = summarize_grade(df, show_details=False)

    if not show_scores_col and not df.empty:
        for col in ("rating","case_score"):
            if col in df.columns:
                df = df.drop(columns=[col])

    return df, grade_md


def _norm_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

def _span_to_string(ids):
    ids = sorted(set(ids))
    # turn [2,3,6] → "2–3, 6"
    spans, start = [], ids[0]
    prev = start
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        spans.append((start, prev))
        start = prev = i
    spans.append((start, prev))
    parts = [f"{a}–{b}" if a != b else f"{a}" for a, b in spans]
    return ", ".join(parts)


def _first_number(s) -> int:
    """
    Return the first integer found in a string like '2', '2–3', '2-3, 8', '2, 8'.
    If nothing is found, return 0.
    """
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else 0

def _as_int_any(s) -> int:
    # used when turning 'id' into a numeric for adjacency checks
    return _first_number(s)

def merge_nearby_cases(df: pd.DataFrame, gap: int = 1, join_sep: str = " ") -> pd.DataFrame:
    """
    1) First pass: merge adjacent rows (id distance <= gap) that share
       top1_label + Category + SubCategory + FineGrained.
    2) Second pass: within each (case, Category, SubCategory, FineGrained),
       drop exact duplicate short texts (e.g., 'Learn more.').
    """
    if df is None or df.empty:
        return df.copy()

    # Ensure columns exist
    for c in ["id","text","top1_label","rating","Category","SubCategory",
              "FineGrained","confidence","case_score","icon_uri"]:
        if c not in df.columns:
            df[c] = ""

    # --- helpers
    def _first_number(s) -> int:
        m = re.search(r"\d+", str(s))
        return int(m.group()) if m else 0

    # ----- First pass: merge nearby identical case/category -----
    work = df.copy()
    work["__id_int"] = work["id"].map(_first_number)
    work = work.sort_values("__id_int").reset_index(drop=True)

    merged, cluster = [], []

    def flush():
        if not cluster:
            return
        start = cluster[0]["id"]; end = cluster[-1]["id"]
        id_str = str(start) if start == end else f"{start}–{end}"

        # combine texts in order; drop empties
        texts  = [str(r["text"]).strip() for r in cluster if str(r["text"]).strip()]
        confs  = [float(r.get("confidence", 0.0)) for r in cluster]

        base   = cluster[0].copy()
        base["id"] = id_str
        base["merged_text"] = " ".join(texts) if texts else str(base.get("text", ""))
        base["text"] = base["merged_text"]          # <- use merged text going forward
        base["confidence"] = round(float(np.mean(confs)), 2)

        merged.append(base)

    for _, row in work.iterrows():
        r = row.to_dict()
        if not cluster:
            cluster = [r]; continue
        prev = cluster[-1]
        same_case = str(r["top1_label"]) == str(prev["top1_label"])
        same_cat  = (str(r["Category"])    == str(prev["Category"]) and
                     str(r["SubCategory"]) == str(prev["SubCategory"]) and
                     str(r["FineGrained"]) == str(prev["FineGrained"]))
        close     = (r["__id_int"] - prev["__id_int"]) <= gap
        if same_case and same_cat and close:
            cluster.append(r)
        else:
            flush(); cluster = [r]
    flush()

    out = pd.DataFrame(merged)
    if out.empty:
        return out

    # ----- Second pass: drop exact duplicate short texts per group -----
    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip().lower()

    out["__norm"]  = out["text"].str.lower().str.replace(r"\s+"," ", regex=True).str.strip()
    out["__len"]   = out["text"].str.len()
    # keep the longest in each group
    out = (out.sort_values(["top1_label","Category","SubCategory","FineGrained","__len"],
                        ascending=[True,True,True,True,False])
            .drop_duplicates(subset=["top1_label","Category","SubCategory","FineGrained","__norm"],
                            keep="first")
            .drop(columns=["__norm","__len","__id_int"], errors="ignore"))

    return out.sort_values(by="id", key=lambda s: s.map(_first_number)).reset_index(drop=True)

# --------------------------
# Lightweight retrieval QA
# --------------------------
_STOP = {
    "the","a","an","of","to","and","or","in","on","for","with","by","as","at","from",
    "that","this","these","those","it","its","is","are","was","were","be","been","being",
    "we","you","they","he","she","their","our","your","i","me","my","mine","ours","yours",
    "will","shall","can","could","should","would","may","might","do","does","did","not","no"
}

def _tok(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOP]
    return toks

def _score_overlap(q_tokens, s_tokens):
    if not q_tokens or not s_tokens:
        return 0.0
    qs = set(q_tokens); ss = set(s_tokens)
    inter = len(qs & ss)
    # scale by question size, slight length normalization
    return inter / (len(qs) ** 0.75 * (1 + 0.25 * max(len(ss) - 12, 0)))

def retrieve_qa_answer(question: str, sentences, top_k: int = 3) -> str:
    question = (question or "").strip()
    if not question:
        return "Type a question about the pasted policy."
    if not sentences:
        return "No policy text available. Paste text and click Annotate first."

    qtok = _tok(question)
    scored = []
    for s in sentences:
        stok = _tok(s)
        score = _score_overlap(qtok, stok)
        if score > 0:
            scored.append((score, s))

    if not scored:
        return "I couldn’t find anything relevant in the current text."

    # pick top-k and format
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:top_k]]

    # simple highlight: wrap matching terms
    def _highlight(text: str):
        out = text
        for t in sorted(set(qtok), key=len, reverse=True):
            if len(t) < 3: 
                continue
            out = re.sub(rf"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", out)
        return out

    bullets = "\n".join(f"- {_highlight(s)}" for s in top)
    return f"**Most relevant passages**:\n\n{bullets}"



with gr.Blocks(
    title="ClearTerms: Privacy Policy Annotator",
    analytics_enabled=False,
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="gray"),
    css="""
:root{
  /* light mint background + strong dark text */
  --bg:#f3fbf4;          /* page background (very light green) */
  --panel:#ffffff;       /* cards/panels */
  --muted:#bcd7c3;       /* borders/lines */
  --text:#0b2e13;        /* near-black green for ALL text */

  --accent:#2e7d32;      /* action green */
  --accent2:#43a047;     /* lighter action green */

  --bad:#e65100;         /* orange */
  --blocker:#c62828;     /* red */
  --neutral:#616161;     /* gray */
  --good:#2e7d32;        /* green */
}

/* Base */
.gradio-container{
  background:var(--bg);
  color:var(--text);
  font-size:15px;
}

/* Ensure ALL prose uses dark text (Gradio sometimes lowers contrast) */
.gradio-container, .gradio-container *:not(svg):not(path){
  color:var(--text);
}

/* Cards */
.section{
  background:var(--panel);
  border:1px solid var(--muted);
  border-radius:12px;
  padding:16px;
  box-shadow:0 1px 2px rgba(0,0,0,.04);
}

/* Headings */
h1.title{
  margin:0;
  font-weight:800;
  font-size:28px;
  line-height:1.2;
  color:var(--accent);
}
.sub{color:#1b5e20}

/* Buttons */
button, .btn > button{
  background-color:var(--accent) !important;
  color:#fff !important;
  border:none !important;
  border-radius:8px !important;
}
button:hover, .btn > button:hover{
  background-color:var(--accent2) !important;
}

/* Small “Download CSV” button */
#dl_btn > button{
  background:#f7faf7 !important;
  color:var(--text) !important;
  border:1px solid var(--muted) !important;
  font-size:13px !important;
  padding:4px 10px !important;
  width:auto !important;
  height:32px !important;
  border-radius:6px !important;
}

/* Table: make rows WHITE with dark text for max contrast */
.predtbl{border-collapse:collapse;width:100%;font-size:15px;background:#fff}
.predtbl th,.predtbl td{
  border:1px solid var(--muted);
  padding:10px;
  vertical-align:top;
  color:var(--text);
}
.predtbl th{
  background:#e3f3e6;   /* soft mint header */
  font-weight:700;
}
.predtbl td{background:#ffffff;}
.predtbl tr:hover td{background:#f0f7f1;}
.predtbl td.num{width:54px;text-align:right}
.predtbl td.ico{width:56px;text-align:center;vertical-align:middle}
.predtbl td.conf{width:120px;text-align:right}
.predtbl td.Topic{width:200px}
.predtbl a{color:var(--accent); text-decoration:underline}

/* Inputs/Textareas */
textarea, input, select{
  color:var(--text) !important;
  background:#ffffff !important;
  border:1px solid var(--muted) !important;
}

-----/* Make the grade Markdown clearly visible */
#results_header { 
  align-items: center; 
  justify-content: flex-start; 
  gap: 4px; 
  margin: 6px 0 8px; 
}

#results_header{
  position: relative;
  z-index: 5;                 /* sits above the table */
  overflow: visible !important;
}

#grade_live .prose :where(h1,h2,h3){
  color: var(--text) !important;   /* dark text */
  opacity: 1 !important;           /* some themes lower heading opacity */
  margin: 0 !important;
}

#grade_live strong{
  color: var(--text) !important;
}

#grade_live{
  background:#f7faf7; 
  border:1px solid var(--muted); 
  border-radius:8px; 
  padding:8px 12px;
}

/* Visible grade badge */
#grade_live .grade-badge{
  display:flex; align-items:center; gap:10px;
  background:#f7faf7; border:1px solid var(--muted);
  border-radius:10px; padding:8px 12px;
  font-size:16px;
}
#grade_live .grade{
  display:inline-block; min-width:2.2em; text-align:center;
  padding:2px 10px; border-radius:999px; color:#fff;
}
#grade_live .grade-A{ background:#2e7d32; }
#grade_live .grade-B{ background:#4caf50; }
#grade_live .grade-C{ background:#ff9800; }
#grade_live .grade-D{ background:#f57c00; }
#grade_live .grade-E, 
#grade_live .grade-F{ background:#c62828; }
#grade_live .grade-N\/A{ background:#9e9e9e; }

#grade_help { display:flex; align-items:center; margin-left:6px; }

----

/* LEFT RAIL — light look to match the rest of the app */
#left_rail .section,
#left_rail .gr-group,
#left_rail .gr-box,
#left_rail .gr-panel {
  background: var(--panel) !important;

  border: 1px solid var(--muted) !important;
}

/* LEFT RAIL — light look */
#left_rail .section,
#left_rail .gr-group,
#left_rail .gr-box,
#left_rail .gr-panel {
  background:#ffffff !important;
  color:var(--text) !important;
  border:1px solid var(--muted) !important;
}


#left_rail label,
#left_rail .prose,
#left_rail .prose * { color:var(--text) !important; }

#left_rail input,
#left_rail select,
#left_rail textarea {
  background:#ffffff !important;
  color:var(--text) !important;
  border:1px solid var(--muted) !important;
}

#left_rail .token, 
#left_rail .tag{
  background:#f4fbf4 !important;
  color:var(--text) !important;
  border:1px solid var(--muted) !important;
}


/* Make the "Controls" heading readable on the dark left rail */
#left_rail h3,
#left_rail .section h3,
#left_rail .prose :where(h1,h2,h3,h4,h5,h6) {
  color: #f4fff4 !important;        /* light mint/near-white */
  text-shadow: 0 1px 0 rgba(0,0,0,.25);  /* slight lift on dark bg (optional) */
}

/* --- Help tooltip --- */
.help-tip{
  position: relative;
  display: inline-flex;
  align-items: center;
  cursor: help;
  outline: none;
}
.help-tip .icon{
  width: 18px; height: 18px;
  border-radius: 50%;
  font-size: 12px; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--accent); color: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,.15);
}
/* Ensure the help-tip sits on top of nearby elements */
#help_tip{ position: relative; z-index: 6; }  /* the <gr.HTML> that holds the tip */
.help-tip{ position: relative; z-index: 10 !important; }

.help-tip .bubble{
  position: absolute; z-index: 60;
  left: 0; top: 28px;   /* shows below the icon */
  min-width: 320px; max-width: 520px;
  background: #fff; color: var(--text);
  border: 1px solid var(--muted); border-radius: 10px;
  padding: 10px 12px;
  box-shadow: 0 8px 28px rgba(0,0,0,.12);
  display: none;
white-space: normal;        /* allow wrapping */
  overflow: visible !important;
}
.help-tip .bubble h4{
  margin: 0 0 6px 0; font-size: 14px;
}
.help-tip .bubble p, .help-tip .bubble li{
  font-size: 13px; line-height: 1.35; margin: 4px 0;
}
.help-tip .bubble kbd{
  border:1px solid var(--muted); border-bottom-width:2px;
  background:#f6f8f6; padding:1px 6px; border-radius:6px; font-size:12px;
}
/* Force readable, dark text inside the tooltip regardless of theme */
.help-tip .bubble,
.help-tip .bubble * {
  color: var(--text) !important;   /* dark green you set in :root */
  opacity: 1 !important;           /* some themes reduce heading opacity */
  text-shadow: none !important;
  background: #fff !important;     /* ensure white panel */
}

/* Extra clarity for headings and lists inside bubble */
.help-tip .bubble h4 { color: var(--text) !important; }
.help-tip .bubble li::marker { color: var(--text) !important; }

/* show on hover and keyboard focus */
.help-tip:hover .bubble,
.help-tip:focus-within .bubble { display: block; }

/* --- Table column sizing --- */
.predtbl { border-collapse: collapse; width: 100%; background: #fff; }
.predtbl th, .predtbl td { border: 1px solid var(--muted); padding: 10px; vertical-align: top; color: var(--text); }
.predtbl th { background: #e3f3e6; font-weight: 700; }

.predtbl .num       { width: 48px; text-align: right; }
.predtbl .col-icon  { width: 56px; text-align: center; vertical-align: middle; }
.predtbl .col-conf  { width: 90px;  text-align: right; }
.predtbl .col-topic { width: 180px; }
.predtbl .col-rating{ width: 110px; text-transform: lowercase; font-weight: 600; }

/* Let text/case flex naturally */
.predtbl .col-text  { width: 42%; }
.predtbl .col-case  { width: 26%; }

/* Row hover */
.predtbl tr:hover td { background: #f0f7f1; }

/* --- Accessible row tints by rating (subtle) --- */
.predtbl tr.rate-good    td { background: #f4fbf4; }     /* green-ish tint */
.predtbl tr.rate-bad     td { background: #fff7ef; }     /* orange-ish tint */
.predtbl tr.rate-blocker td { background: #fff3f3; }     /* red-ish tint */
.predtbl tr.rate-neutral td { background: #f8f9fb; }     /* gray-ish tint */

/* Put the table lower in the stack so it doesn't cover the bubble */
#pred_table_html{
  position: relative;
  z-index: 1;                 /* lower than the header/tip */
}
/* put this in your existing CSS block */
.predtbl th.col-icon, .predtbl td.col-icon{
  width:48px; min-width:48px; text-align:center;
}
.predtbl td.col-icon img{
  width:28px; height:28px; object-fit:contain; display:block; margin:0 auto;
}
/* widths for hierarchy columns */
.predtbl th.col-cat, .predtbl td.col-cat { width: 160px; }
.predtbl th.col-sub, .predtbl td.col-sub { width: 180px; }
.predtbl th.col-fine, .predtbl td.col-fine { width: 200px; }

/* Make the predictions HTML container expand/scroll instead of clipping */
#pred_table_html {
  max-height: 70vh;            /* or remove this line if you want full auto height */
  overflow-y: auto !important; /* allow scrolling when long */
  overflow-x: hidden;
  padding: 0;                  /* optional */
}

/* Ensure normal table flow */
#pred_table_html .predtbl tbody { 
  display: table-row-group !important;
}
/* ----- Fix dropdown (multi-select) palette: readable, light menu ----- */
#left_rail [role="listbox"]{
  background:#ffffff !important;
  color:var(--text) !important;
  border:1px solid var(--muted) !important;
  box-shadow:0 8px 28px rgba(0,0,0,.12) !important;
}

#left_rail [role="option"]{
  background:#ffffff !important;
  color:var(--text) !important;
}
#left_rail [role="option"]:hover{
  background:#f0f7f1 !important;
  color:var(--text) !important;
}
#left_rail [role="option"][aria-selected="true"]{
  background:#e3f3e6 !important;
  color:var(--text) !important;
  font-weight:600;
}

/* tokens/chips inside the multiselect input */
#left_rail .token, 
#left_rail .tag{
  background:#f4fbf4 !important;
  color:var(--text) !important;
  border:1px solid var(--muted) !important;
}

/* the text you type into the multiselect input */
#left_rail input[type="text"]{
  color:var(--text) !important;
}

/* make sure the menu appears above panels */
#left_rail [role="listbox"]{ z-index: 1000 !important; }

/* Keep the tooltip behind the table so it doesn't cover rows */
.help-tip { z-index: 1; }
#pred_table_html { position: relative; z-index: 2; }
"""
) as demo:
    # Header
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""<div class="section" role="banner" aria-label="Header">
                <h1 class="title">PrivacyLens: Understanding What You Really Agreed To</h1>
                <div class="sub">Paste a policy section or upload a CSV. Get sentence-level ToS;DR cases, icons, and an overall grade.</div>
            </div>""")

    with gr.Row():
    # LEFT rail
        with gr.Column(scale=1, min_width=340, elem_id="left_rail"):
            with gr.Group(elem_classes=["section"]):
                gr.HTML("<h3 style='margin:0'>Controls</h3>")
                threshold = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Confidence threshold",
                                    info="Only show sentences ≥ this score")
                font_size = gr.Slider(12, 20, value=15, step=1, label="Table font size (px)")
                show_scores = gr.Checkbox(value=True, label="Show rating label & case score")
                apply_styles = gr.Button("Apply style", variant="secondary")
                # in the Controls group
                merge_similar = gr.Checkbox(value=True, label="Merge nearby/duplicate rows")

            with gr.Group(elem_classes=["section"]):
                with gr.Group(elem_id="Topic_wrap"):
                    # MUST be empty here; we'll set choices after annotate
                    Topic_filter = gr.Dropdown(
                        choices=[],          # <- keep empty
                        multiselect=True,
                        value=None,
                        label="Filter by Category",
                        info="Top-level Categories found in the current results."
                    )
                with gr.Row():
                    btn_Topics_all   = gr.Button("Select all", variant="secondary", elem_classes="btn-xs")
                    btn_Topics_clear = gr.Button("Clear",      variant="secondary", elem_classes="btn-xs")

    # RIGHT column stays the same…

            # tiny helpers for the dropdown
            btn_Topics_all.click(lambda choices: choices, inputs=Topic_filter, outputs=Topic_filter)
            btn_Topics_clear.click(lambda: [], None, Topic_filter)

    # RIGHT: paste/annotate tabs + results
        with gr.Column(scale=3):
            with gr.Tab("Paste Text / Section", id="tab_text"):
                text_in = gr.Textbox(
                    lines=12, label="Policy text or section",
                    placeholder="Paste policy text here…", elem_id="text_input"
                )
                with gr.Row():
                    btn_annotate_text = gr.Button("Annotate", variant="primary", elem_classes="btn")
                    btn_clear_text    = gr.Button("Clear", variant="secondary")
                # a compact header row: grade on left, small download button on right
                with gr.Row(elem_id="results_header"):
                    out_grade = gr.Markdown(elem_id="grade_live", value="")
                    help_tip  = gr.HTML("", elem_id="help_tip")  # ⬅ placeholder to fill later
                    dl_text   = gr.DownloadButton("Download CSV", variant="secondary",
                                                elem_id="dl_btn", visible=False)

                # the table goes below the header row
                out_html  = gr.HTML(label="Predictions (with icons)", elem_id="pred_table_html")
                # ---- Q&A panel (hidden by default) ----
                qa_toggle = gr.Checkbox(value=False, label="Enable Q&A panel")
                qa_mode   = gr.Radio(choices=["After table", "Replace table"], value="After table",
                                    label="Q&A display mode")
                with gr.Group(visible=False) as qa_panel:
                    qa_q     = gr.Textbox(label="Ask a question about this policy", placeholder="e.g., How long do they retain my data?")
                    qa_topk  = gr.Slider(1, 5, value=3, step=1, label="# of passages to show")
                    qa_btn   = gr.Button("Ask", variant="primary")
                    qa_md    = gr.Markdown(value="")
                
                

            with gr.Tab("Upload CSV", id="tab_csv"):
                csv_in   = gr.File(label="Upload CSV with a text column")
                col_name = gr.Textbox(label="Text column name", value="sentence")
                with gr.Row():
                    btn_annotate_csv = gr.Button("Annotate CSV", variant="primary", elem_classes="btn")
                    btn_clear_csv    = gr.Button("Clear", variant="secondary")
                out_html2  = gr.HTML(label="Predictions (with icons)")
                out_grade2 = gr.Markdown(value="")
                dl_csv     = gr.File(label="Download annotated CSV", interactive=False)

            # Legend + tips
            gr.HTML("""<div class="section" aria-label="Legend and tips">
              <div class="legend" aria-label="Icon legend">
                <span class="tag"><span class="swatch good"></span> good</span>
                <span class="tag"><span class="swatch bad"></span> bad</span>
                <span class="tag"><span class="swatch blocker"></span> blocker</span>
                <span class="tag"><span class="swatch neutral"></span> neutral</span>
              </div>
              <div style="margin-top:10px;color:#cdd6e1">
                Tips: Press <kbd>⌘/Ctrl</kbd>+<kbd>Enter</kbd> to annotate.
              </div>
            </div>""")

    def _badge(grade: str) -> str:
        return f"""<div class="grade-badge">
            <span>Overall grade:</span>
            <strong class="grade grade-{grade}">{grade}</strong>
        </div>"""
    state_sentences = gr.State([])
    state_passages  = gr.State([])  
    state_pass_embs = gr.State(None) 
    # wire up callbacks
    def _annotate_text(big_text, thr, fsize, show_scores_col, selected_Topics, do_merge):

        raw_sentences_all = sent_split(big_text or "")

        # 1) classify once
        raw_df, _ = _classify_free_text_raw(big_text, thr, show_scores_col)
        print(f"[thr={thr:.2f}] raw={len(raw_df)}")

        # 2) categories present / active
        cats_present = sorted(set(raw_df.get("Category", pd.Series(dtype=str)).astype(str))) if not raw_df.empty else []
        active_cats  = selected_Topics if selected_Topics else cats_present
        filtered_raw = raw_df[raw_df["Category"].isin(active_cats)] if active_cats else raw_df.iloc[0:0]

        # 3) grade on filtered_raw
        grade_md = summarize_grade(filtered_raw, show_details=False)

        # 4) ALWAYS initialize table_df; then try to merge
        table_df = filtered_raw.copy()
        if do_merge:
            try:
                table_df = merge_nearby_cases(filtered_raw, gap=1)
            except Exception as e:
                print("merge_nearby_cases failed:", e)
                table_df = filtered_raw.copy()

        # 5) render table + CSV
        html_table = df_to_html(table_df, font_px=int(fsize))
        csv_path   = save_df_csv(table_df, "predictions_text.csv")
        tip_html   = build_help_html() if not table_df.empty else ""

        # 6) build QA sentences from the **visible** rows (merged_text if present)
        if not table_df.empty:
            if "merged_text" in table_df.columns:
                qa_sentences = table_df["merged_text"].fillna(table_df.get("text","")).astype(str).tolist()
            else:
                qa_sentences = table_df.get("text", pd.Series([], dtype=str)).astype(str).tolist()
        else:
            qa_sentences = raw_sentences_all  # graceful fallback
        passages = []
        if not table_df.empty:
            # prefer merged_text if present; fallback to text
            txt_col = "merged_text" if "merged_text" in table_df.columns else "text"
            passages = [str(x).strip() for x in table_df[txt_col].tolist() if str(x).strip()]

        # cache embeddings for speed (None if no passages)
        embs = sbert.encode(passages, normalize_embeddings=True) if passages else None
        return (
            html_table, grade_md, csv_path,
            gr.update(choices=cats_present, value=active_cats),
            gr.update(visible=bool(not table_df.empty), value=csv_path),
            tip_html,
            qa_sentences,
            passages, 
            embs  
        )


    def _annotate_csv(fileobj, text_col, thr, fsize, show_scores_col, selected_Topics):
        raw_df, _ = _classify_csv_raw(fileobj, text_col, thr, show_scores_col)

        cats_present = sorted(set(raw_df.get("Category", pd.Series(dtype=str)).astype(str))) if not raw_df.empty else []
        active_cats  = selected_Topics if selected_Topics else cats_present
        if active_cats:
            filtered_raw = raw_df[raw_df["Category"].isin(active_cats)]
        else:
            filtered_raw = raw_df.iloc[0:0]

        grade_md  = summarize_grade(filtered_raw, show_details=False)
        merged_df = merge_nearby_cases(filtered_raw, gap=1)

        html_table = df_to_html(merged_df, font_px=int(fsize))
        csv_path   = save_df_csv(merged_df, "predictions_csv.csv")
        tip_html   = build_help_html() if not merged_df.empty else ""

        return (
            html_table,
            grade_md,
            csv_path,
            gr.update(choices=cats_present, value=active_cats),
            gr.update(visible=bool(not merged_df.empty), value=csv_path),
            tip_html
        )

    # add state_passages, state_pass_embs to outputs
    btn_annotate_text.click(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip,
                state_sentences, state_passages, state_pass_embs]
    )


    btn_annotate_csv.click(
    fn=_annotate_csv,
    inputs=[csv_in, col_name, threshold, font_size, show_scores, Topic_filter],
    outputs=[out_html2, out_grade2, dl_csv, Topic_filter, dl_csv, help_tip]
    )
    threshold.input(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip, state_sentences]
    )

    Topic_filter.change(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip, state_sentences]
    )

    merge_similar.change(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip, state_sentences]
    )

    btn_Topics_all.click(
        lambda choices: choices, inputs=Topic_filter, outputs=Topic_filter
    ).then(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip, state_sentences]
    )

    btn_Topics_clear.click(
        lambda: [], None, Topic_filter
    ).then(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip, state_sentences]
    )

    show_scores.change(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip, state_sentences]
    )

    font_size.release(
        fn=_annotate_text,
        inputs=[text_in, threshold, font_size, show_scores, Topic_filter, merge_similar],
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip, state_sentences]
    )
    def _sync_qa_visibility(enable, mode):
    # show the QA panel only when enabled
        show_panel = bool(enable)
        # hide table only when enabled AND mode == Replace table
        hide_table = bool(enable and mode == "Replace table")
        return (
            gr.update(visible=not hide_table),  # out_html
            gr.update(visible=show_panel)       # qa_panel
        )

    qa_toggle.change(
        fn=_sync_qa_visibility,
        inputs=[qa_toggle, qa_mode],
        outputs=[out_html, qa_panel]
    )
    qa_mode.change(
        fn=_sync_qa_visibility,
        inputs=[qa_toggle, qa_mode],
        outputs=[out_html, qa_panel]
    )

    import numpy as np

    def yes_no_from_evidence(question: str, evidence: str, pos_claim: str, neg_claim: str, thr=0.55):
        # Example: pos_claim="The service sells personal data."
        #          neg_claim="The service does NOT sell personal data."
        # premise = evidence; hypothesis = claim
        def score(hypo):
            out = nli(f"{evidence} </s> {hypo}")[0]  # returns list of label/score
            lab2s = {d["label"].lower(): d["score"] for d in out}
            return lab2s.get("entailment", 0.0), lab2s.get("contradiction", 0.0)

        ent_pos, con_pos = score(pos_claim)
        ent_neg, con_neg = score(neg_claim)

        if ent_pos >= thr and con_pos < 0.45:
            return "Yes"
        if ent_neg >= thr and con_neg < 0.45:
            return "No"
        return "Unclear"

    def _qa_answer(
    question: str,
    passages: list,
    embs,
    topk: int,
    sim_threshold: float = 0.45,     # tune 0.35–0.55
    use_reranker: bool = True,
    ):
        question = (question or "").strip()
        if not question:
            return "Type a question about the pasted policy."
        if not passages:
            return "No policy text available. Paste text and click Annotate first."

        # SBERT embeddings
        q_emb = sbert.encode([question], normalize_embeddings=True)[0]
        if embs is None:
            embs = sbert.encode(passages, normalize_embeddings=True)

        sims = np.dot(embs, q_emb)  # [N]
        # grab a pool (top 10) then threshold
        pool_idx = np.argsort(-sims)[:10]
        pool = [(i, float(sims[i])) for i in pool_idx if sims[i] >= sim_threshold]

        if not pool:
            return f"I couldn’t find anything above similarity {sim_threshold:.2f}."

        # optional cross-encoder rerank for precision
        if use_reranker and len(pool) > 1:
            pairs = [(question, passages[i]) for i, _ in pool]
            rerank_scores = reranker.predict(pairs)  # higher is better
            pool = sorted([(i, s, r) for (i, s), r in zip(pool, rerank_scores)], key=lambda x: -x[2])

        # keep top-k
        k = max(1, int(topk))
        keep = pool[:k]
        ctx_chunks = [passages[i] for i, *_ in keep]
        best_idx = keep[0][0]
        evidence = passages[best_idx]

        # 1) Try extractive QA to anchor the answer
        context = "\n\n".join(ctx_chunks)
        span = qa_pipe(question=question, context=context)
        span_text = (span.get("answer") or "").strip()
        span_conf = float(span.get("score", 0.0))

        # 2) Build a short, *generated* answer for readability
        #    (condition on question + the single best evidence)
        prompt = f"Question: {question}\nEvidence: {evidence}\nAnswer briefly and concretely:"
        gen_txt = gen(prompt, max_new_tokens=64, num_beams=4, do_sample=False)[0]["generated_text"].strip()

        # If the question looks yes/no, compute a crisp verdict
        yn = None
        q_l = question.lower()
        if any(q_l.startswith(w) for w in ["is ", "are ", "do ", "does ", "can ", "will ", "should "]) \
        or q_l.endswith("?"):
            # specialize for “sold data”
            yn = yes_no_from_evidence(
                question,
                evidence,
                pos_claim="The service sells personal data.",
                neg_claim="The service does not sell personal data.",
                thr=0.55
            )

        # Final assembly logic:
        # - prefer generated sentence for readability,
        # - if yes/no available, prepend “Yes/No/Unclear –”
        # - only keep span if it’s meaningful and not a header-like fragment
        def looks_like_header(t): return (len(t.split()) <= 6 and (t.endswith(":") or t.istitle()))

        pieces = []
        if yn:
            pieces.append(f"**Answer:** {yn} — {gen_txt if gen_txt else span_text}")
        else:
            pieces.append(f"**Answer:** {gen_txt if gen_txt else span_text}")

        if not pieces[-1] or looks_like_header(pieces[-1].replace('**Answer:**', '').strip()):
            # fall back to evidence sentence if generation was empty/garbled
            pieces[-1] = f"**Answer:** {span_text if span_text else evidence}"

        # concise provenance (folded)
        pieces.append(f"<details><summary>Why?</summary><blockquote>{evidence}</blockquote></details>")

        return "\n\n".join(pieces)

    qa_btn.click(
        fn=_qa_answer,
        inputs=[qa_q, state_passages, state_pass_embs, qa_topk],
        outputs=[qa_md]
    )
        # Clear actions
    def _clear_text():
        return "", "", None, gr.update(choices=[], value=[]), gr.update(visible=False, value=None), "", [], [], None
    btn_clear_text.click(fn=_clear_text, inputs=None,
        outputs=[out_html, out_grade, dl_text, Topic_filter, dl_text, help_tip,
                state_sentences, state_passages, state_pass_embs])

    def _clear_csv():
        return None, "", "", None, ""  # file, table, grade, download, tip
    btn_clear_csv.click(fn=_clear_csv, inputs=None, outputs=[csv_in, out_html2, out_grade2, dl_csv, help_tip])

    # Live style change (font size)
    def _apply_style(px):
        # return a tiny style tag to enlarge font in table only
        return f"<style>.predtbl{{font-size:{int(px)}px}}</style>"
    apply_styles.click(fn=_apply_style, inputs=[font_size], outputs=[out_html])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)

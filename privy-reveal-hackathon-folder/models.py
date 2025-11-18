# models.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Dict
import config

# =========================
# Device Selection
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# =========================
# Load All Models
# =========================

# --- Main Classification Model ---
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    config.MODEL_DIR,
    torch_dtype=torch.float32,
)
model.to(device)
model.eval()

# --- Sentence Transformers & Reranker ---
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Pipeline Models ---
qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
gen = pipeline("text2text-generation", model="google/flan-t5-base", device=device)

# --- NLI Model for Yes/No ---
nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_mod = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_mod.to(device)
nli = pipeline("text-classification", model=nli_mod, tokenizer=nli_tok, return_all_scores=True, device=device)

# =========================
# Model-related Helpers
# =========================
def _load_id2label_from_config(cfg) -> Dict[int, str]:
    """Determine label mapping from model config."""
    if hasattr(cfg, "id2label") and cfg.id2label:
        try:
            return {int(k): v for k, v in cfg.id2label.items()}
        except Exception:
            # Handle potential non-integer keys gracefully
            out = {int(k): v for k, v in cfg.id2label.items() if k.isdigit()}
            return out if out else {i: str(i) for i in range(cfg.num_labels)}
    return {i: str(i) for i in range(cfg.num_labels)}

# Load the label mapping for the main classifier
id2label = _load_id2label_from_config(model.config)
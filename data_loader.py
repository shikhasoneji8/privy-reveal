# data_loader.py
import pandas as pd
import os
import base64
import config

# Global maps to be populated on load
TOSDR_URL_MAP = {}
RATING_MAP = {}
WEIGHT_MAP = {}
CAT_MAP = {}
ALL_CATEGORIES = []
ICON_URIS = {}

def load_case_and_categories(case_csv: str, cat_csv: str):
    """
    Loads data from CSVs and populates the global mapping dictionaries.
    """
    global TOSDR_URL_MAP, RATING_MAP, WEIGHT_MAP, CAT_MAP, ALL_CATEGORIES
    
    # --- Load Case Ratings and Scores ---
    try:
        cf = pd.read_csv(case_csv)
    except Exception as e:
        print(f"WARNING: failed to read {case_csv}: {e}")
        cf = pd.DataFrame(columns=["case", "class", "score", "Title", "URL"])

    for col in ["case", "class", "score", "Title", "URL"]:
        if col not in cf.columns:
            cf[col] = "" if col != "score" else 0.0

    cf["case"] = cf["case"].astype(str).str.strip()
    cf["class"] = cf["class"].astype(str).str.strip().str.lower()
    cf["score"] = pd.to_numeric(cf["score"], errors="coerce").fillna(0.0)
    cf["Title"] = cf["Title"].astype(str).str.strip()
    cf["URL"] = cf["URL"].astype(str).str.strip()

    RATING_MAP = dict(zip(cf["case"], cf["class"]))
    WEIGHT_MAP = dict(zip(cf["case"], cf["score"]))
    TOSDR_URL_MAP = {r["Title"]: r["URL"] for _, r in cf.iterrows() if r["Title"] and r["URL"]}
    print(f"Loaded {len(TOSDR_URL_MAP)} case-URL mappings.")

    # --- Load Hierarchical Categories ---
    try:
        hf = pd.read_csv(cat_csv)
    except Exception as e:
        print(f"WARNING: failed to read {cat_csv}: {e}")
        hf = pd.DataFrame(columns=["Case", "Category", "Sub-Category", "Fine-grained category"])

    for col in ["Case", "Category", "Sub-Category", "Fine-grained category"]:
        if col not in hf.columns:
            hf[col] = ""
        hf[col] = hf[col].astype(str).fillna("").str.strip()

    temp_cat_map = {
        row["Case"]: (row["Category"] or "Other", row["Sub-Category"] or "—", row["Fine-grained category"] or "—")
        for _, row in hf.iterrows()
    }
    
    # Normalize category names
    for case, (cat, sub, fine) in temp_cat_map.items():
        normalized_cat = cat.replace('\xa0', ' ').strip()
        if normalized_cat.lower() == "first party collection/use":
            normalized_cat = "First Party Collection/Use"
        if normalized_cat.lower().startswith("user access,"):
            normalized_cat = "User Access, Edit and Deletion"
        CAT_MAP[case] = (normalized_cat, sub, fine)

    ALL_CATEGORIES = sorted({v[0] for v in CAT_MAP.values() if v[0]}) or ["Other"]


def _to_data_uri(path: str) -> str:
    """Converts an image file to a base64 data URI."""
    abs_path = path if os.path.isabs(path) else os.path.join(config.BASE_DIR, path)
    if not os.path.exists(abs_path):
        return ""
    with open(abs_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

def load_icon_uris(icon_files: dict):
    """Loads all icon files into a dictionary of data URIs."""
    global ICON_URIS
    ICON_URIS = {k: _to_data_uri(p) for k, p in icon_files.items()}
    print("ICON_URIS loaded:", {k: bool(v) for k, v in ICON_URIS.items()})

# --- Initialize data on import ---
load_case_and_categories(config.CASE_RATINGS_CSV, config.EMNLP_CAT_SUB_CSV)
load_icon_uris(config.ICON_FILES)
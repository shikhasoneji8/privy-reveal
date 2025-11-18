# config.py
import os

# =========================
# File and Directory Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "privbert_model")
CASE_RATINGS_CSV = os.path.join(BASE_DIR, "case_ratings.csv")
EMNLP_CAT_SUB_CSV = os.path.join(BASE_DIR, "emnlp_cat_sub.csv")

# =========================
# Model & Prediction Configuration
# =========================
MAX_LEN = 512
TOP_K = 3
BATCH_SIZE = 8
THRESHOLD = 0.75 # Default confidence threshold

# =========================
# Application Data & Mappings
# =========================
VALID_RATINGS = {"good", "bad", "blocker"}

# Icon file paths relative to the base directory
ICON_FILES = {
    "good":     "static/icons/good.png",
    "bad":      "static/icons/bad.png",
    "blocker":  "static/icons/blocker.png",
    "neutral":  "static/icons/neutral.png",
}

# Color mapping for UI categories
CATEGORY_COLOR_MAP = {
    "First Party Collection/Use": "cat-collection",
    "Third Party Sharing/Collection": "cat-sharing",
    "Data Security": "cat-security",
    "User Access, Edit and Deletion": "cat-access",
    "Policy Change": "cat-policy",
    "International and Specific Audiences": "cat-compliance",
    "Data Retention": "cat-retention",
    "Service Description": "cat-description",
    "Governing Law": "cat-law",
    "Dispute Resolution": "cat-dispute",
    "Limitation of Liability": "cat-liability",
    "Indemnification": "cat-indemnity",
    "Disclaimer": "cat-disclaimer",
    "Do Not Track": "cat-dnt",
    "User Choice": "cat-choice",
    "User Responsibility": "cat-user-resp",
    "Registration & Account Security": "cat-reg-sec",
    "Misc": "cat-misc",
    "Other": "cat-other",
}
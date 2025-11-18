# app.py
import gradio as gr
# Save original function
import gradio_client.utils as gc_utils
import gradio as gr
import gradio_client.utils as gc_utils
import elevenlabs
from typing import Optional
import os
# elevenlabs.set_api_key("sk_a768c54d34bd63fe13ef794b09aed3a603297dc9659a0f44")

from elevenlabs import ElevenLabs

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVEN_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not set in environment.")

eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# Try to patch the internal recursive function that chokes on "schema=True"
try:
    _orig_inner = gc_utils._json_schema_to_python_type
except AttributeError:
    _orig_inner = None

def _patched_inner(schema, defs=None):
    # If schema is literally True/False, treat it as "Any" instead of crashing
    if isinstance(schema, bool):
        return "Any"
    # Otherwise, delegate to the original implementation
    if _orig_inner is not None:
        return _orig_inner(schema, defs)
    # Fallback (shouldn't normally be hit)
    return gc_utils.json_schema_to_python_type(schema)

if _orig_inner is not None:
    gc_utils._json_schema_to_python_type = _patched_inner

import pandas as pd
import sys # Keep sys import for dynamic loading
import os
# from deepgram import Deepgram
import os
import json
import textwrap
import os
import textwrap
# import boto3
# from botocore.exceptions import ClientError
import requests



import random # Used for simulating Galileo metrics

# Import from our new, organized modules
import data_loader
import core_logic
import presentation
import html 
import os
# Import default threshold
from config import THRESHOLD 
from core_logic import retrieve_qa_answer 

from lime_service_direct import explain_text as explain_with_lime
from interpret_service import explain_with_ebm


# BEDROCK_REGION = "us-west-2"  # or the region they gave you for the hackathon
# CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"  # from AWS docs / hackathon docs
# client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# BEDROCK_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

# if not BEDROCK_TOKEN:
#     raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK not set. Run: export AWS_BEARER_TOKEN_BEDROCK=your_token_here")


# This uses your AWS credentials (from env, profile, or hackathon setup)


js_head = """
<script>
function setDarkMode(is_dark) {
    const container = document.querySelector('.gradio-container');
    if (container) {
        if (is_dark) {
            container.classList.add('dark');
            localStorage.setItem('dark_mode', 'true');
        } else {
            container.classList.remove('dark');
            localStorage.setItem('dark_mode', 'false');
        }
    }
}

function setupToggleListener() {
    const toggle = document.querySelector('#dark-mode-toggle input');
    
    // If the toggle exists and we haven't already attached the listener
    if (toggle && !toggle.dataset.listenerAttached) {
        // Stop the interval check
        clearInterval(window.themeInterval);

        // Retrieve the saved theme preference and apply it
        const is_dark = localStorage.getItem('dark_mode') === 'true';
        toggle.checked = is_dark;
        setDarkMode(is_dark);

        // Add the main event listener to toggle the theme
        toggle.addEventListener('change', (e) => setDarkMode(e.target.checked));
        
        // Mark the element so we know the listener is attached
        toggle.dataset.listenerAttached = 'true';
    }
}

// Start an interval to run the setup function every 100 milliseconds
window.themeInterval = setInterval(setupToggleListener, 100);
</script>
"""

LIME_CSS_PATCH = """
<style>
  :root { color-scheme: light; }
  body{
    margin:0; padding:16px 20px; background:#ffffff; color:#111;
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  h1,h2,h3 { color:#0b2e13; margin:0 0 10px; font-weight:700; }

  /* Ensure all text/bars render dark, overriding LIME’s light styles */
  .lime, .lime * { color:#111 !important; fill:#111 !important; stroke:none !important; opacity:1 !important; }

  /* Prevent clipping and crowding inside the SVG */
  svg { overflow: visible !important; }

  /* LIME prints the full sentence as a giant SVG text over the plots; hide it to stop overlap */
  svg text:first-of-type { display: none !important; }
  svg text[font-size="24.0"], svg text[font-size="26.0"] { display: none !important; }

  /* Make axis/labels readable but compact */
  svg text { font-size: 13px !important; dominant-baseline: middle; }

  /* Keep the figure blocks from collapsing on each other */
  figure { display:block; margin: 0; }
</style>
"""


CSS = """

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
  
  /* --- NEW: Category Colors --- */
  --cat-collection-main: #3F51B5;  /* Indigo */
  --cat-collection-bg: #E8EAF6;
  --cat-sharing-main: #FF9800;     /* Orange */
  --cat-sharing-bg: #FFF3E0;
  --cat-security-main: #009688;    /* Teal */
  --cat-security-bg: #E0F2F1;
  --cat-access-main: #4CAF50;      /* Green */
  --cat-access-bg: #E8F5E9;
  --cat-policy-main: #F44336;      /* Red */
  --cat-policy-bg: #FFEBEE;
  --cat-compliance-main: #9C27B0;  /* Purple */
  --cat-compliance-bg: #F3E5F5;
  --cat-retention-main: #607D8B;   /* Blue Gray */
  --cat-retention-bg: #ECEFF1;
  --cat-description-main: #2196F3; /* Blue */
  --cat-description-bg: #E3F2FD;
  --cat-other-main: #9E9E9E;       /* Gray */
  --cat-other-bg: #F5F5F5;
  --cat-law-main: #673AB7;       
  --cat-dispute-main: #FFC107;   
  --cat-liability-main: #03A9F4; 
  --cat-indemnity-main: #CDDC39; 
  --cat-disclaimer-main: #9E9E9E;
  --cat-dnt-main: #795548;      
  --cat-choice-main: #E91E63;    
  --cat-user-resp-main: #00BCD4; 
  --cat-reg-sec-main: #7CB342;   
  --cat-misc-main: #FF5722;      
}

/* Base */
.gradio-container{
  background:var(--bg);
  color: black;
  font-size:15px;
}


/* *** ACCESSIBILITY FIX: Force near-black text for maximum contrast on light backgrounds *** */
.cards-wrap, .card {
  color: #111111 !important;
}
.cards-wrap *, .card * {
  color: #111111 !important; 
}
.card .badge {
    color: #ffffff !important; /* Keep badge text white */
}
/* --- END ACCESSIBILITY FIX --- */

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

/* ========================================================================= */
/* --- FINAL FIX: Grade Header Layout --- */
/* ========================================================================= */

#results_header {
  display: flex !important;
  align-items: center !important; /* Vertically center all items */
  column-gap: 2px !important;    /* << REDUCED SPACE for a much tighter layout */
  margin: 0 !important;
  padding: 0 !important;
}

/* Force all direct children (component wrappers) to have no space */
#results_header > .gradio-html,
#results_header > .gradio-markdown {
  margin: 0 !important;
  padding: 0 !important;
  min-width: fit-content !important; /* Stop wrappers from taking extra space */
  flex-shrink: 0 !important; /* Prevent items from shrinking */
}

/* Style the grade text: "Overall grade: C" */
#grade_live .prose h2 {
  margin: 0 !important;
  padding: 0 !important;
  line-height: 1.2 !important;
  color: #111111 !important; /* Dark text for "Overall grade:" */
  font-size: 20px !important;
  font-weight: 600 !important;
}
#grade_desc .prose p {
    font-size: 14px !important;
    color: #4a5568 !important; /* A muted, dark gray color */
    margin: 0 !important;
    padding: 0 4px !important; /* Add a little horizontal padding */
    line-height: 1.3 !important;
}

#grade_desc {
    flex-grow: 1;
    flex-shrink: 1;
}

/* Make the grade letter (A, B, C...) dark */
#grade_live .prose h2 strong {
  color: #111111 !important;
}

/* Push the Download button to the far right */
#results_header #dl_btn {
  margin-left: auto !important;
}

/* ========================================================================= */
/* --- END FINAL FIX --- */
/* ========================================================================= */


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


/* Fix checkbox label text color on the dark left rail */
#left_rail .gr-check-label {
    /* Use white text to contrast with the dark panel */
    color: #ffffff !important; 
    opacity: 1 !important;
}
/* Ensure the text inside the labels (like 'Table font size') is also white */
#left_rail label,
#left_rail .gradio-html * {
    color: var(--text) !important; /* Set to dark text to match light background */
    opacity: 1 !important;
}


#left_rail .token, 
#left_rail .tag{
  background:#f4fbf4 !important;
  color:var(--text) !important;
  border:1px solid var(--muted) !important;
}


/* Make "Controls" heading readable on dark backgrounds */
#left_rail h3,
#left_rail .section h3,
#left_rail .prose :where(h1,h2,h3,h4,h5,h6) {
  color: #f0f0f0 !important;     
  text-shadow: none;  
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

/* ========================================================================= */
/* --- Stacking Order Fix --- */
/* This section ensures the help bubble appears ON TOP of the results. */
/* ========================================================================= */
/* Ensure the results container (table or cards) is on a base layer */
#pred_table_html {
  position: relative;
  z-index: 1;
}
/* Ensure the help tip container is layered above the results,
   allowing its popup bubble to be visible. */
#help_tip {
  position: relative; /* Establishes a stacking context */
  z-index: 50;        /* Must be higher than #pred_table_html */
}
/* ========================================================================= */

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
#left_rail [role="listbox"] {

}
/* When the dropdown component receives focus (user clicks to open it), restore the list height */
#left_rail [role="listbox"]:focus-within {

}

/* Ensure the main input area for the dropdown remains visible */
#left_rail .gr-dropdown-container {
}

/* Hide the individual option containers to complete the illusion */
#left_rail .gradio-dropdown-option {
    display: flex !important; /* Must remain flex when open */
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

-----
/* ===== Card view (horizontal) ===== */
.cards-wrap{
  display:flex; gap:14px; align-items:stretch;
  overflow-x:auto; overflow-y:hidden;
  padding:10px 6px; scroll-snap-type:x mandatory;
  background:#fff; border:1px solid var(--muted); border-radius:10px;
}
.card{
  flex: 0 0 360px; /* card width */
  display:flex; flex-direction:column; justify-content:flex-start;
  scroll-snap-align:start;
  border:1px solid var(--muted); border-radius:12px;
  background:#fff; padding:12px;
  box-shadow:0 1px 2px rgba(0,0,0,.05);
}

/* rating tint on the left edge */
.card.rate-good{  border-left:6px solid var(--good); }
.card.rate-bad{   border-left:6px solid var(--bad); }
.card.rate-blocker{ border-left:6px solid var(--blocker); }
.card.rate-neutral{ border-left:6px solid var(--neutral); }

/* --- ACCESSIBILITY FIX: Lighten background tints for better contrast with dark text --- */
.card.rate-bad     { background: #fffaf5 !important; } 
.card.rate-blocker { background: #fffafa !important; }
.card.rate-good    { background: #fafffb !important; }
.card.rate-neutral { background: #fcfcfc !important; } 
/* --- NEW: Per-Card Background Color based on Category (Subtle) --- */
/* Example: .bg-cat-collection applies to the entire card */
.bg-cat-collection { background-color: var(--cat-collection-bg) !important; }
.bg-cat-sharing    { background-color: var(--cat-sharing-bg) !important; }
.bg-cat-security   { background-color: var(--cat-security-bg) !important; }
.bg-cat-access     { background-color: var(--cat-access-bg) !important; }
.bg-cat-policy     { background-color: var(--cat-policy-bg) !important; }
.bg-cat-compliance { background-color: var(--cat-compliance-bg) !important; }
.bg-cat-retention  { background-color: var(--cat-retention-bg) !important; }
.bg-cat-description{ background-color: var(--cat-description-bg) !important; }
.bg-cat-other      { background-color: var(--cat-other-bg) !important; }

/* --- CARD CONTENT STYLES --- */
.card .meta{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.card .rating-icon{ width:24px; height:24px; object-fit:contain; }
.card .badge{
  display:inline-block; padding:2px 8px; border-radius:999px;
  font-size:12px; font-weight:700; color:#fff !important; line-height:1.4;
}
.badge.good{ background:var(--good); }
.badge.bad{ background:var(--bad); }
.badge.blocker{ background:var(--blocker); }
.badge.neutral{ background:var(--neutral); }

/* --- CHIPS (Category Tags) --- */
/* --- CHIPS (Category Tags) --- */
.card .chips{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 0 0; }
.card .chip{
  /* Reduced padding to make them smaller swatches */
  font-size:12px; padding: 4px 10px; border-radius:4px;
  font-weight: 500;
  border:1px solid rgba(0,0,0,0.1); 
  background:#f7faf7; /* Base color for non-category chips (e.g. confidence) */
  color: #111111 !important;
  /* Fixed width/height for category chips */
  min-width: 14px;
  min-height: 14px;
  text-align: center;
}
.card .chip.conf-chip {
    background: #e3f3e6;
}

/* --- Dynamic Chip Coloring (Fix: Apply MAIN Color and White Text) --- */
/* The main category chip is the primary color swatch */
.chip.cat-main, 
.chip.cat-sub, 
.chip.cat-fine { /* Apply fixed styling to ALL hierarchy chips */
    color: #ffffff !important; /* Ensure the text (if present) is white */
    font-weight: 700;
    border: none;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    padding: 4px 10px; /* Reset padding for category chips */
}

/* 1. Apply MAIN Colors for all three levels of hierarchy */
.chip.chip-cat-collection, .chip-sub-cat-collection, .chip-fine-cat-collection { 
    background-color: var(--cat-collection-main) !important;
}
.chip.chip-cat-sharing, .chip-sub-cat-sharing, .chip-fine-cat-sharing { 
    background-color: var(--cat-sharing-main) !important; 
}
.chip.chip-cat-security, .chip-sub-cat-security, .chip-fine-cat-security { 
    background-color: var(--cat-security-main) !important; 
}
.chip.chip-cat-access, .chip-sub-cat-access, .chip-fine-cat-access { 
    background-color: var(--cat-access-main) !important; 
}
.chip.chip-cat-policy, .chip-sub-cat-policy, .chip-fine-cat-policy { 
    background-color: var(--cat-policy-main) !important; 
}
.chip.chip-cat-compliance, .chip-sub-cat-compliance, .chip-fine-cat-compliance { 
    background-color: var(--cat-compliance-main) !important; 
}
.chip.chip-cat-retention, .chip-sub-cat-retention, .chip-fine-cat-retention { 
    background-color: var(--cat-retention-main) !important; 
}
.chip.chip-cat-description, .chip-sub-cat-description, .chip-fine-cat-description { 
    background-color: var(--cat-description-main) !important; 
}
.chip.chip-cat-other, .chip-sub-cat-other, .chip-fine-cat-other { 
    background-color: var(--cat-other-main) !important; 
}
/* --- End Dynamic Chip Coloring Fix --- */
/* --- End Dynamic Chip Coloring --- */

.card .field{ font-size:13px; color: #111111 !important; margin:6px 0; }
.card .label{ font-weight:700; margin-right:6px; }
.card .txt{ font-size:14px; line-height:1.45; color: #111111 !important; white-space:pre-wrap; }
.cards-wrap:focus, .card:focus{ outline: 2px solid var(--accent); outline-offset: 2px; }

/* responsive: narrower cards on small screens */
@media (max-width: 640px){
  .card{ flex-basis: 85vw; }
}

/* --- Legend Box Styling --- */
/* --- Legend Box Styling --- */
.legend-box {
    margin-top: 20px;
}
.legend-section {
    margin-bottom: 15px;
}
.legend-title {
    font-size: 16px;
    font-weight: 700;
    /* *** FIX: Ensure title text is dark *** */
    color: #111111 !important; 
    margin-bottom: 8px;
}
.legend-items {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 20px;
}
.legend-item {
    display: flex;
    align-items: center;
    font-size: 14px;
    /* *** FIX: Ensure item text is dark *** */
    color: #111111 !important;
}
.legend-swatch {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    margin-right: 6px;
    display: inline-block;
}
/* Ensure swatch uses the main color for the legend */
.legend-swatch.cat-main.chip-cat-collection { background-color: var(--cat-collection-main); }
.legend-swatch.cat-main.chip-cat-sharing    { background-color: var(--cat-sharing-main); }
.legend-swatch.cat-main.chip-cat-security   { background-color: var(--cat-security-main); }
.legend-swatch.cat-main.chip-cat-access     { background-color: var(--cat-access-main); }
.legend-swatch.cat-main.chip-cat-policy     { background-color: var(--cat-policy-main); }
.legend-swatch.cat-main.chip-cat-compliance { background-color: var(--cat-compliance-main); }
.legend-swatch.cat-main.chip-cat-retention  { background-color: var(--cat-retention-main); }
.legend-swatch.cat-main.chip-cat-description{ background-color: var(--cat-description-main); }
.legend-swatch.cat-main.chip-cat-other      { background-color: var(--cat-other-main); }

/* Rating Swatch in Legend */
.legend-swatch.badge.good   { background: var(--good); }
.legend-swatch.badge.bad    { background: var(--bad); }
.legend-swatch.badge.blocker{ background: var(--blocker); }
.legend-swatch.badge.neutral{ background: var(--neutral); }
/* --- End Legend Box Styling --- */
----

/* Style for the link inside the case field */
.card .field-case a {
    color: var(--accent); /* Uses your main green color */
    font-weight: 600;
    text-decoration: underline;
    text-decoration-thickness: 1px;
    cursor: pointer;
}
.card .field-case a:hover {
    text-decoration: none;
    color: var(--accent2);
}
#left_rail [role="radio"][aria-checked="false"] {
  color: #111111 !important; /* Forces dark text when unselected/greyed out */
}
/* Remove default radio/checkbox indicator from Multiselect Dropdown */
#left_rail .gradio-dropdown-option .option-mark {
    display: none !important;
}

/* Freeze Table Header Row */
.predtbl thead {
    position: sticky;
    top: 0; /* Freeze it at the top of its container */
    z-index: 10; /* Ensure it stays above the scrolling body content */
}

/* Ensure the header cells have a solid background color (already set, but important for sticky) */
.predtbl th {
  background: #e3f3e6;   /* soft mint header */
  font-weight: 700;
}

/* Ensure the container of the table has defined height and scroll */
#pred_table_html {
  max-height: 70vh;
  overflow-y: auto !important; /* This is key for sticky positioning to work */
  overflow-x: hidden;
  padding: 0;
}
/* Custom fix for the two checkboxes */
.light-label-fix label {
    color: #ffffff !important; /* Forces the label text to white */
    opacity: 1 !important;
}
.light-label-fix .prose * {
    color: #ffffff !important;
    opacity: 1 !important;
}
/* FIX: Stop Gradio from intercepting external links with this class */
.external-link {
    pointer-events: auto !important;
}
/* Ensure the link color remains consistent */
.card .field-case a.external-link {
    color: var(--accent);
    /* You may need to add text-decoration: underline !important; here if the link doesn't look clickable */
}

/* --- Grade key next to overall grade --- */
.grade-key{
  display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  padding:6px 8px; border:1px solid var(--muted); border-radius:8px;
  background:#f7faf7;
}
.gk-item{ display:flex; align-items:center; gap:6px; }
.gk-text{ font-size:12px; color:#111111 !important; }
#results_header .grade-key .grade{ min-width:1.8em; padding:1px 8px; }

.grade-key.single {
  display:flex;
  gap:8px;
  align-items:center;
  padding:6px 10px;
  border-radius:8px;
  border:1px solid var(--muted);
  background:#f7faf7;
}

.grade-key.single .gk-text {
  font-size:13px;
  font-weight: 500;
}
.grade-key.single .gk-item{ display:flex; align-items:center; gap:8px; }
.grade-key.single .gk-text{ font-size:13px; color:#111111 !important; }
/* --- Grade chip palette + spacing --- */
:root{
  --chip-A-bg:#2e7d32;  /* green */
  --chip-B-bg:#1e88e5;  /* blue  */
  --chip-C-bg:#ffe082;  /* light amber (mixed) */
  --chip-D-bg:#c62828;  /* red   */
  --chip-E-bg:#6d4c41;  /* brown */
  --chip-light-text:#ffffff;
  --chip-dark-text:#111111;
}

/* Base chip layout */
.grade-key.single{
  display:flex; gap:8px; align-items:center;
  padding:6px 10px; border-radius:8px;
  border:1px solid var(--muted);
}

.grade-chip-A { background-color: #00704A; } /* Excellent */
.grade-chip-B { background-color: #1E88E5; } /* Good */
.grade-chip-C { background-color: #E6A700; } /* Mixed */
.grade-chip-D { background-color: #D9534F; } /* Risky */
.grade-chip-E { background-color: #6C757D; } /* Poor */

/* This is the KEY rule that makes the chip text ("Risky") use the correct color set above */
.grade-key.single .gk-item,
.grade-key.single .gk-text {
  color: inherit !important;
}

/* Optional shadow for darker chips */
.grade-chip-A, .grade-chip-B, .grade-chip-D, .grade-chip-E {
  box-shadow: 0 1px 2px rgba(0,0,0,.15);
}

/* --- Q&A Highlight Color --- */
#qa_md .prose mark {
    background-color: #dcedc8 !important; /* A soft, light green */
    color: #1b5e20 !important;            /* Dark green text for contrast */
    padding: 2px 4px;
    border-radius: 4px;
}

.dark #qa_md .prose mark {
    background-color: #388e3c !important; /* A medium green for dark mode */
    color: #ffffff !important;            /* White text */
}

.raw-sentence {
  background: #ffffff;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px;
  line-height: 1.6;
  color: #111;
  font-size: 15px;
  white-space: pre-wrap;
}

#expl_out .section{
  background:#fff;
  border:1px solid var(--muted);
  border-radius:12px;
  padding:16px;
  box-shadow:0 1px 2px rgba(0,0,0,.04);
  margin-top: 12px;
}
#expl_out .raw-sentence {
  background:#ffffff;
  border:1px solid #ddd;
  border-radius:8px;
  padding:10px;
  line-height:1.6;
  color:#111;
  font-size:15px;
  white-space:pre-wrap;
}

/* Fix light gray headings in the explainer */
#explainer_panel, #explainer_panel * {
  color: #1b1b1b !important;
  fill: #1b1b1b !important;
  stroke: none !important;
  opacity: 1 !important;
}

/* Make "Predicted" header bold and darker */
#explainer_panel .section strong {
  color: #0d2f12 !important;
  font-weight: 700;
}

/* Fix LIME SVG overlap */
.lime svg text {
  font-size: 14px !important;
  dominant-baseline: middle;
}

/* Prevent overlapping charts and words */
.lime figure {
  display: block;
  width: 100%;
  overflow: visible !important;
}
#explain_block h3 { color:#111 !important; }
.section .raw-sentence { color:#111 !important; }
.section .pred-label, .section .pred-label * { color:#111 !important; }
#explainer_panel, #explainer_panel * { color:#0b2e13 !important; opacity:1 !important; }
"""

# In app.py, replace your old _run_annotation function with this one.

def _run_annotation(text_input, threshold):
    if not text_input or not text_input.strip():
        return (pd.DataFrame(), gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), "*Annotate text to ask questions.*")
    sentences = core_logic.sent_split(text_input)
    raw_df = core_logic.classify_sentences(sentences, threshold=float(threshold))
    cats_present = sorted(raw_df['Category'].unique()) if not raw_df.empty else []
    fine_present = sorted(raw_df['FineGrained'].unique()) if not raw_df.empty else []
    qa_placeholder = "Ask a question about the annotated policy." if not raw_df.empty else "*Annotate text to ask questions.*"
    return (raw_df, gr.update(choices=cats_present, value=[]), gr.update(choices=fine_present, value=[]), qa_placeholder)

def extract_grade_letter(grade_md: str) -> str:
    """
    Your grade markdown looks like: '## Overall grade: **C**'
    Pull out the letter between the asterisks.
    """
    import re
    m = re.search(r"\*\*([A-E]|N/A)\*\*", grade_md or "")
    return m.group(1) if m else ""

# In app.py, replace the existing function with this one.

def _apply_filters_and_render(raw_df, selected_topics, selected_fine, view_mode, fsize, enable_expl=False):
    import numpy as np
    if raw_df is None or raw_df.empty:
        empty_html = presentation.df_to_html(pd.DataFrame(), view_mode=view_mode)
        return (
            empty_html,                           # out_html
            "## Overall grade: **N/A**",          # out_grade
            None,                                 # dl_btn value
            gr.update(visible=False),             # dl_btn props
            "",                                   # help_tip
            gr.update(value="", visible=False),   # grade_chip
            gr.update(value="", visible=False),   # grade_description
            gr.update(choices=[], value=None),    # expl_id choices
            gr.update(visible=False),             # expl_btn
            gr.update(value="", visible=False),   # expl_out
            {}                                    # state_view_map  <-- IMPORTANT
        )

    # --- Apply filters (PRESERVE original index) ---
    filtered_df = raw_df
    if selected_topics:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_topics)]
    if selected_fine:
        filtered_df = filtered_df[filtered_df["FineGrained"].isin(selected_fine)]

    # If nothing left
    if filtered_df.empty:
        empty_html = presentation.df_to_html(pd.DataFrame(), view_mode=view_mode)
        return (
            empty_html,
            "## Overall grade: **N/A**",
            None,
            gr.update(visible=False),
            "",
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            {}   # empty mapping
        )

    # --- Grade ---
    grade_md = core_logic.summarize_grade(filtered_df)
    grade_letter = extract_grade_letter(grade_md)
    chip_html, chip_vis = "", False
    grade_desc_text, grade_desc_vis = "", False
    if grade_letter in {"A","B","C","D","E"}:
        chip_html = build_grade_chip_html(grade_letter)
        chip_vis = True
        grade_desc_text = GRADE_DESCRIPTIONS.get(grade_letter, "")
        grade_desc_vis = bool(grade_desc_text)

    # --- Build display dataframe with stable mapping ---
    # 1) carry original row index
    tmp = filtered_df.copy()
    tmp = tmp.assign(raw_idx=tmp.index)

    # 2) reset for display and add display id without colliding with any existing 'id'
    display_df = tmp.reset_index(drop=True)
    disp_id_col = "disp_id" if "id" in display_df.columns else "id"
    display_df.insert(0, disp_id_col, range(1, len(display_df) + 1))

    # 3) OPTIONAL: merge nearby cases on a copy that keeps mapping columns
    # If merge_nearby_cases changes row counts, you’ll lose mapping.
    # So either (a) don’t merge before building mapping, or (b) ensure it preserves 1:1 rows.
    # Here we assume it preserves row alignment; if not, move it AFTER we drop raw_idx and skip mapping through it.
    display_df = core_logic.merge_nearby_cases(display_df)

    # 4) Build mapping (disp_id -> raw_idx)
    view_map = dict(zip(display_df[disp_id_col].tolist(), display_df["raw_idx"].tolist()))

    # 5) Hide mapping column(s) from UI/CSV
    safe_display = display_df.drop(columns=["raw_idx"], errors="ignore")

    # --- Render outputs ---
    output_html = presentation.df_to_html(safe_display, font_px=int(fsize), view_mode=view_mode)
    csv_path = core_logic.save_df_csv(safe_display, "predictions.csv")
    help_html = presentation.build_help_html() if not safe_display.empty else ""

    # Explanations UI: populate choices with currently visible IDs
    id_choices = safe_display[disp_id_col].tolist()
    # Gradio dropdown likes strings; convert but keep ints in view_map keys too
    id_choices_str = [str(x) for x in id_choices]
    expl_btn_vis = bool(id_choices) and bool(enable_expl)

    # Pick a default (string) value
    default_choice = id_choices_str[0] if id_choices_str else None

    return (
        output_html,                               # out_html
        grade_md,                                  # out_grade
        csv_path,                                  # dl_btn value
        gr.update(visible=not safe_display.empty, value=csv_path),
        help_html,                                 # help_tip
        gr.update(value=chip_html, visible=chip_vis),
        gr.update(value=grade_desc_text, visible=grade_desc_vis),
        gr.update(choices=id_choices_str, value=default_choice),  # expl_id (strings)
        gr.update(visible=expl_btn_vis),           # expl_btn
        gr.update(value="", visible=expl_btn_vis), # expl_out (cleared)
        view_map                                   # <-- write mapping to state
    )


def _toggle_expl_visibility(enable, current_choices):
    return (
        gr.update(visible=bool(enable) and bool(current_choices)),  # expl_id
        gr.update(visible=bool(enable) and bool(current_choices)),  # expl_btn
        gr.update(value="", visible=False),                         # expl_out
    )


TEXT_COL_CANDIDATES = [
    "Text", "Sentence", "Policy Text from Your Original Pasted Content",
    "PolicyText", "Data", "text", "content", "snippet"
]

def _pick_text_from_row(row):
    for col in TEXT_COL_CANDIDATES:
        if col in row.index:
            val = str(row[col])
            if val and val.strip():
                return val
    return ""


from deepgram import DeepgramClient


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError(
        "DEEPGRAM_API_KEY environment variable not set. "
        "Run: export DEEPGRAM_API_KEY='your_real_key_here'"
    )

dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

from deepgram import DeepgramClient

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

def transcribe_audio(audio_path: str) -> str:
    if not audio_path:
        return ""

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = dg_client.listen.v1.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        smart_format=True
    )

    return response.results.channels[0].alternatives[0].transcript



def _explain_selected(raw_df, selected_id, selected_topics, selected_fine, expl_method):
    import html as ihtml
    import importlib.util, sys
    import numpy as np
    import gradio as gr
    

    if raw_df is None or raw_df.empty or not selected_id:
        return gr.update(value="<div class='section'>No sentence selected.</div>", visible=True)

    try:
        disp_id = int(selected_id)
    except Exception:
        return gr.update(value="<div class='section'>Invalid selection.</div>", visible=True)

    filtered = raw_df.copy()
    if selected_topics:
        filtered = filtered[filtered["Category"].isin(selected_topics)]
    if selected_fine:
        filtered = filtered[filtered["FineGrained"].isin(selected_fine)]
    if filtered.empty:
        return gr.update(value="<div class='section'>No rows under current filters.</div>", visible=True)

    filtered = filtered.reset_index(drop=True)
    disp_col = "disp_id" if "id" in filtered.columns else "id"
    filtered.insert(0, disp_col, range(1, len(filtered)+1))

    row = filtered[filtered[disp_col] == disp_id]
    if row.empty:
        return gr.update(value="<div class='section'>Sentence not found (check filters).</div>", visible=True)

    TEXT_COLS = ["Text","Sentence","Policy Text from Your Original Pasted Content",
                 "PolicyText","Data","text","content","snippet"]
    r = row.iloc[0]
    text = ""
    for c in TEXT_COLS:
        if c in row.columns and isinstance(r[c], str) and r[c].strip():
            text = r[c]; break
    if not text:
        return gr.update(value="<div class='section'>No text available for this row.</div>", visible=True)

    # --- Load the LIME module you just created ---
    try:
    # dynamically import the **direct LIME** service
        module_name = 'lime_service_direct'
        if module_name not in sys.modules:
            import importlib.util
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                raise ImportError(f"Service module '{module_name}' not found. Ensure lime_service_direct.py exists.")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules[module_name]

        result = module.explain_text(text, include_native=False)
        pred_label     = result["pred_label"]
        confidence     = result["confidence"]
        token_weights  = result["token_weights"]
        classes_sorted = result["classes_sorted"]

        # Build a sentence block (stays dark/large and full-width)
        sentence_block = (
            "<div class='section' style='margin-top:8px;'>"
            "<div style='font-weight:700; margin-bottom:6px; color:#0b2e13;'>Sentence</div>"
            f"<div class='raw-sentence' style='font-size:20px; line-height:1.45; color:#0b2e13;'>{html.escape(text)}</div>"
            "</div>"
        )

        # Use our new, overlap-free LIME panel
        lime_panel = presentation.render_lime_panel(
          text=text,
          pred_label=pred_label,
          confidence=confidence,
          token_weights=token_weights,   # now cleaned = no blank labels
          classes_sorted=classes_sorted, # for top 5 + others
          prob_title="Prediction probabilities",
          others_top_k=4,
          others_min_prob=0.001
      )


        html_out = sentence_block + lime_panel
        return gr.update(value=html_out, visible=True)

    except Exception as e:
        return gr.update(value=f"<div class='section'>Explanation error: {e}</div>", visible=True)


# def ask_claude_bearer(policy_text: str, question: str) -> str:
#     """
#     Call Claude 3 Haiku on Amazon Bedrock using the boto3 Converse API.
#     Answers ONLY using the given policy text.
#     """
#     if not policy_text or not policy_text.strip():
#         return "Claude error: No policy text provided."
#     if not question or not question.strip():
#         return "Claude error: No question provided."

#     user_message = textwrap.dedent(f"""
#     You are a privacy policy explainer for end users.

#     Below is a privacy policy. Answer the question ONLY using that policy text.
#     If the answer is not clearly stated, say you cannot find a clear answer in this policy.

#     --- POLICY START ---
#     {policy_text}
#     --- POLICY END ---

#     Question: {question}
#     """).strip()

#     conversation = [
#         {
#             "role": "user",
#             "content": [{"text": user_message}],
#         }
#     ]

#     try:
#         response = client.converse(
#             modelId=CLAUDE_MODEL_ID,
#             messages=conversation,
#             inferenceConfig={
#                 "maxTokens": 512,
#                 "temperature": 0.4,
#                 "topP": 0.9,
#             },
#         )
#     except ClientError as e:
#         msg = e.response.get("Error", {}).get("Message") or str(e)
#         return f"Claude error (ClientError): {msg}"
#     except Exception as e:
#         return f"Claude error: {e}"

#     # Parse successful response
#     try:
#         output_message = response["output"]["message"]
#         content_list = output_message.get("content", [])
#         if not content_list:
#             return f"Claude error: Empty content in response: {response}"

#         answer_text = content_list[0].get("text", "").strip()
#         if not answer_text:
#             return f"Claude error: No text in response content: {response}"

#         return answer_text
#     except Exception as e:
#         return f"Claude parse error: {e}; raw response: {response}"




def handle_voice_qa(policy_text, audio_path):
    """Handle a voice question: transcribe with Deepgram, then answer from the policy."""
    if not policy_text or not policy_text.strip():
        return "Please paste and annotate some text before asking a question."
    if audio_path is None:
        return "Please record a question."

    # 1) Transcribe
    question = transcribe_audio(audio_path)
    if not question or not question.strip():
        return "I couldn't hear a question clearly. Please try again."

    # 2) Run your existing QA flow
    sentences = core_logic.sent_split(policy_text)
    answer = retrieve_qa_answer(question, sentences)

    # 3) Return nicely formatted markdown
    return f"**You asked (transcribed):** {question}\n\n**Answer:**\n{answer}"


# def handle_qa_request(question, source_text):
#     if not source_text or not source_text.strip():
#         return "Please paste the policy text before asking a question."
#     if not question or not question.strip():
#         return "Please enter a question."

#     answer = ask_claude_bearer(source_text, question)

#     return f"**Question:** {question}\n\n**Answer:**\n{answer}"

def handle_qa_request(question, source_text):
    """Handles a click on the 'Ask' button for the Q&A panel."""
    if not source_text or not source_text.strip():
        return "Please paste and annotate some text before asking a question."
    if not question or not question.strip():
        return "Please enter a question."
    
    sentences = core_logic.sent_split(source_text)
    answer = retrieve_qa_answer(question, sentences)
    return answer


from typing import Optional

def synthesize_answer_with_elevenlabs(answer_text: str) -> Optional[str]:
    """
    Turn the QA answer into speech using ElevenLabs (new SDK v1)
    and return the path to an MP3 file.
    """
    if not answer_text or not answer_text.strip():
        return None

    response = eleven_client.text_to_speech.convert(
        voice_id="YOUR_VOICE_ID_HERE",     # e.g., “21m00Tcm4TlvDq8ikWAM”
        model_id="eleven_multilingual_v2",
        optimize_streaming_latency="0",    # lowest latency
        output_format="mp3_44100_128",
        text=answer_text,
    )

    # Join streaming chunks
    audio_bytes = b"".join(chunk for chunk in response)

    out_path = "answer_tts.mp3"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    return out_path

def _clear_all():
    return (
        "",                                          # text_in
        pd.DataFrame(),                              # state_raw_df
        "",                                          # out_html
        "## Overall grade: **N/A**",                 # out_grade
        None,                                        # dl_btn (value)
        gr.update(visible=False, value=None),        # dl_btn (props)
        "",                                          # help_tip
        gr.update(choices=data_loader.ALL_CATEGORIES, value=[]), # Topic_filter
        gr.update(choices=[], value=[]),             # Fine_filter
        gr.update(value="", visible=False), 
        gr.update(value="", visible=False),          # grade_chip
        "*Annotate text to ask questions.*",          # qa_md
    )

def handle_file_upload(file_obj, text_column, threshold, view_mode, fsize):
    if file_obj is None:
        return "Please upload a file.", "## Grade: N/A", None, gr.update(visible=False)

    try:
        filename = file_obj.name
        sentences = []
        if filename.lower().endswith('.txt'):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            sentences = core_logic.sent_split(content)
        elif filename.lower().endswith('.csv'):
            df_in = pd.read_csv(filename)
            if text_column not in df_in.columns:
                error_html = f"<div class='section'>Error: Column '<b>{text_column}</b>' not found in CSV. Available columns: {list(df_in.columns)}</div>"
                return error_html, "## Grade: Error", None, gr.update(visible=False)
            sentences = df_in[text_column].astype(str).fillna("").tolist()
        else:
            return "<div class='section'>Error: Unsupported file type. Please upload a .txt or .csv file.</div>", "## Grade: Error", None, gr.update(visible=False)

        if not sentences:
            return "<div class='section'>No text found in the uploaded file.</div>", "## Grade: N/A", None, gr.update(visible=False)

        # Process the sentences
        results_df = core_logic.classify_sentences(sentences, threshold=float(threshold))
        
        # Generate outputs
        output_html = presentation.df_to_html(results_df, font_px=int(fsize), view_mode=view_mode)
        grade_md = core_logic.summarize_grade(results_df)
        csv_path = core_logic.save_df_csv(results_df, f"annotated_{os.path.basename(filename)}.csv")

        return output_html, grade_md, csv_path, gr.update(visible=not results_df.empty, value=csv_path)

    except Exception as e:
        return f"<div class='section'>An error occurred: {e}</div>", "## Grade: Error", None, gr.update(visible=False)

def build_grade_key_html() -> str:
    return """
    <div class="grade-key" role="note" aria-label="Grade legend">
      <span class="gk-item" title="Excellent: no risky clauses found or only minor positives.">
        <span class="grade grade-A">A</span><span class="gk-text">Excellent</span>
      </span>
      <span class="gk-item" title="Good: a few minor concerns; mostly positive.">
        <span class="grade grade-B">B</span><span class="gk-text">Good</span>
      </span>
      <span class="gk-item" title="Mixed: notable concerns along with positives.">
        <span class="grade grade-C">C</span><span class="gk-text">Mixed</span>
      </span>
      <span class="gk-item" title="Risky: many concerning clauses; proceed with caution.">
        <span class="grade grade-D">D</span><span class="gk-text">Risky</span>
      </span>
      <span class="gk-item" title="Poor: severe or blocker issues dominate.">
        <span class="grade grade-E">E</span><span class="gk-text">Poor</span>
      </span>
    </div>
    """


# Dictionary to hold the explanations for each grade
GRADE_DESCRIPTIONS = {
    "A": "Excellent: no risky clauses found or only minor positives.",
    "B": "Good: a few minor concerns; mostly positive.",
    "C": "Mixed: notable concerns along with positives.",
    "D": "Risky: many concerning clauses; proceed with caution.",
    "E": "Poor: severe or blocker issues dominate."
}


def build_grade_chip_html(grade_letter: str) -> str:
    labels = {
        "A": "Excellent",
        "B": "Good",
        "C": "Mixed",
        "D": "Risky",
        "E": "Poor",
    }
    colors = {
        "A": "#00704A",  # green
        "B": "#1E88E5",  # blue
        "C": "#E6A700",  # amber
        "D": "#D9534F",  # red
        "E": "#6C757D",  # gray
    }
    color = colors.get(grade_letter, "#444")
    label = labels.get(grade_letter, "")
    if not label:
        return ""
    # add a per-grade class for coloring
    chip_cls = f"grade-chip-{grade_letter}"
    return f"""
    <div style="
        display:inline-block;
        background-color:{color};
        color:#fff;
        font-weight:600;
        font-size:15px;
        padding:6px 14px;
        border-radius:20px;
        box-shadow:0 2px 4px rgba(0,0,0,0.15);
    ">
      {label}
    </div>
    """

def simulate_galileo_evaluation(raw_df):
    """
    Simulates sending the prediction DataFrame to Galileo for evaluation.

    Returns a markdown summary of evaluation + policy-risk metrics.
    """
    if raw_df is None or raw_df.empty:
        return "*No annotations yet. Click **Annotate** first, then re-run this check.*"

    total_sentences = len(raw_df)

    # --- Existing metric: avg confidence ---
    avg_conf = raw_df["Confidence"].mean() if "Confidence" in raw_df.columns else 0.85

    # --- NEW: Risk profile metrics (derived from your model outputs) ---
    rating_col = None
    for cand in ["Rating", "rating", "RiskRating", "risk"]:
        if cand in raw_df.columns:
            rating_col = cand
            break

    risky_rate = None
    blocker_rate = None
    if rating_col is not None:
        ratings = raw_df[rating_col].astype(str).str.lower()
        risky_mask = ratings.isin(["bad", "blocker"])
        blocker_mask = ratings.eq("blocker")

        risky_rate = risky_mask.mean()
        blocker_rate = blocker_mask.mean()

    # --- NEW: Category coverage ---
    cat_col = None
    for cand in ["Category", "category", "BroadCategory"]:
        if cand in raw_df.columns:
            cat_col = cand
            break

    unique_cats = raw_df[cat_col].nunique() if cat_col is not None else None

    # --- NEW: Average sentence length (in words) ---
    text_col = None
    for cand in ["Text", "Sentence", "PolicyText", "Policy Text from Your Original Pasted Content",
                 "text", "content", "snippet", "Data"]:
        if cand in raw_df.columns:
            text_col = cand
            break

    avg_len_words = None
    if text_col is not None:
        avg_len_words = (
            raw_df[text_col].astype(str)
            .str.split()
            .apply(len)
            .mean()
        )

    # --- Simulated Galileo-style metrics (still randomised) ---
    sim_accuracy = random.uniform(0.88, 0.95)
    sim_pii_rate = random.uniform(0.00, 0.005)
    sim_halluc = random.uniform(0.01, 0.05)

    # ------------------------------------------------------------------ #
    # Build markdown output with TWO tables: Reliability + Risk profile
    # ------------------------------------------------------------------ #
    md = "### 🛡️ Galileo Evaluation Summary\n"

    # 1) Reliability table
    md += "\n#### Reliability Metrics\n"
    md += "| Metric | Value | Meaning |\n"
    md += "| :--- | :---: | :--- |\n"
    md += f"| **Accuracy (Sim.)** | **{sim_accuracy:.2f}** | Model's reliability on a hidden test set (simulated). |\n"
    md += f"| PII Guardrail Rate | {sim_pii_rate:.4f} | Approx. rate at which sensitive data might be leaked (simulated). |\n"
    md += f"| Hallucination Score | {sim_halluc:.2f} | Risk of generating false / unsupported classifications (simulated). |\n"
    md += f"| Avg Confidence | {avg_conf:.3f} | Average certainty of the model's classifications. |\n"
    md += f"| Total Samples | {total_sentences} | Number of sentences evaluated. |\n"

    # 2) Policy risk profile (derived from *your* DF, not simulated)
    md += "\n#### Policy Risk Profile (from current batch)\n"
    md += "| Metric | Value | Meaning |\n"
    md += "| :--- | :---: | :--- |\n"

    if risky_rate is not None:
        md += (
            f"| Risky Sentences (%) | {100*risky_rate:.1f}% | "
            "Portion of sentences labelled as **bad** or **blocker**. |\n"
        )
    if blocker_rate is not None:
        md += (
            f"| Blocker Sentences (%) | {100*blocker_rate:.1f}% | "
            "Sentences marked as blocker-level risk. |\n"
        )
    if unique_cats is not None:
        md += (
            f"| Category Coverage | {unique_cats} | "
            "Number of unique categories flagged in this policy. |\n"
        )
    if avg_len_words is not None:
        md += (
            f"| Avg Sentence Length | {avg_len_words:.1f} words | "
            "Average length of analysed sentences. |\n"
        )

    return md
with gr.Blocks(title="PrivyReveal", theme=gr.themes.Soft(primary_hue="green"), css=CSS) as demo:
    # State to hold the full, unfiltered results of the last annotation
    state_raw_df = gr.State(pd.DataFrame())
    state_view_map = gr.State({})
    # --- UI Layout ---
    gr.HTML("""
        <div class="section" role="banner">
            <h1 class="title">PrivyReveal: Bringing Hidden Privacy Risks Into the Light</h1>
            <div class="sub">Paste a policy, get sentence-level analysis and an overall grade.</div>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=340, elem_id="left_rail"):
            with gr.Group(elem_classes=["section"]):
                # dark_mode_toggle = gr.Checkbox(label="🌙 Dark Mode", value=False, elem_id="dark-mode-toggle")
                view_mode = gr.Radio(["Table", "Cards"], value="Cards", label="Results View Mode")
                font_size = gr.Slider(12, 20, value=15, step=1, label="Table Font Size (px)")
                # merge_similar = gr.Checkbox(True, label="Merge nearby/duplicate rows")

            with gr.Group(elem_classes=["section"]):
                gr.HTML("<h3>Controls</h3>")
                threshold = gr.Slider(0.0, 1.0, value=THRESHOLD, step=0.01, label="Confidence threshold")
            
            with gr.Group(elem_classes=["section"]):
                Topic_filter = gr.Dropdown(
                    choices=data_loader.ALL_CATEGORIES, multiselect=True, value=[], label="Filter by Category"
                )
                btn_Topics_clear = gr.Button("Clear", variant="secondary")
            
            with gr.Group(elem_classes=["section"]):
                Fine_filter = gr.Dropdown(
                    choices=[], multiselect=True, value=[], label="Filter by Fine-Grained Category"
                )
                btn_Fine_clear = gr.Button("Clear", variant="secondary")
            
            with gr.Group(elem_classes=["section"]):
              gr.HTML("<h3>Ask a Question</h3>")
              qa_q = gr.Textbox(
                  label="Ask about the current policy",
                  placeholder="e.g., How is my data shared?",
                  info="Requires text to be annotated first."
              )
              qa_btn = gr.Button("Ask (Text)", variant="primary")

              # 🎙 NEW: Voice input
              qa_audio = gr.Audio(
                  sources=["microphone"],
                  type="filepath",
                  label="Or ask with your voice"
              )
              qa_voice_btn = gr.Button("🎙 Ask with Voice", variant="secondary")

              qa_md = gr.Markdown(
                  value="*Annotate text to ask questions.*",
                  elem_id="qa_md"
              )
              qa_audio_out = gr.Audio(
                label="Spoken answer (ElevenLabs)",
                type="filepath",
                interactive=False
            )
            with gr.Group(elem_classes=["section"]):
              gr.HTML("<h3>Why this label? (Advanced)</h3>")
              
              # SHAP ONLY - Radio button removed, using a static label/hidden field
              expl_method = gr.Radio(
                  ["SHAP (Fast Approx)"], 
                  value="SHAP (Fast Approx)", 
                  label="Explanation Method",
                  visible=False # Hide the radio button since there's only one choice
              )
              gr.Markdown("**Explanation Method: SHAP (Fast Approx)**")

              show_expl = gr.Checkbox(label="Enable Explanations (off by default)", value=False)
              expl_id = gr.Dropdown(choices=[], value=None, label="Sentence ID to explain", visible=False)
              expl_btn = gr.Button("Explain", variant="secondary", visible=False)
              expl_out = gr.HTML(value="", visible=False, elem_id="explainer_panel")

        with gr.Column(scale=3):
            with gr.Tab("Paste Text"):
                text_in = gr.Textbox(lines=12, label="Policy text", placeholder="Paste policy text here…")
                
                with gr.Row():
                    btn_annotate = gr.Button("Annotate", variant="primary")
                    btn_clear = gr.Button("Clear", variant="secondary")
                
                with gr.Row(elem_id="results_header"):
                  out_grade = gr.Markdown("## Overall grade: **N/A**", elem_id="grade_live")
                  grade_chip = gr.Markdown(value="", visible=False, elem_id="grade_chip")
                  grade_description = gr.Markdown(value="", visible=False, elem_id="grade_desc")
                  help_tip = gr.HTML(value="", elem_id="help_tip")
                  dl_btn = gr.DownloadButton("Download CSV", variant="secondary", visible=False, elem_id="dl_btn")
                
                out_html = gr.HTML(label="Predictions", elem_id="pred_table_html")
                expl_out = gr.HTML(value="", visible=False, elem_id="expl_out")

            with gr.Tab("Upload File"):
                gr.Markdown("Upload a `.txt` or `.csv` file for annotation. For CSV files, specify the column containing the text to analyze.")
                file_in = gr.File(label="Upload .txt or .csv File", file_types=['.txt', '.csv'])
                csv_col_name = gr.Textbox(label="Text Column Name (for CSV only)", value="text", info="The name of the column in your CSV that contains the sentences.")
                btn_annotate_file = gr.Button("Annotate File", variant="primary")
                
                # Outputs for the file tab
                out_grade_file = gr.Markdown("## Grade: N/A")
                dl_btn_file = gr.DownloadButton("Download Annotated CSV", variant="secondary", visible=False)
                out_html_file = gr.HTML(label="File Predictions")
            with gr.Tab("Galileo Reliability"):
                gr.Markdown("## Reliability and Safety Evaluation")
                gr.Markdown(
                    "This section simulates integration with **Galileo**, an AI observability platform, "
                    "to show the reliability and safety of the Transformer model *behind* the classifications."
                )
                galileo_btn = gr.Button("Run Galileo Evaluation (Simulated)", variant="primary")
                galileo_out = gr.Markdown(value="*Click the button to run the simulated evaluation on the last annotated text.*")
    # --- Event Handlers ---
    
    # inputs/outputs lists
    render_inputs  = [state_raw_df, Topic_filter, Fine_filter, view_mode, font_size, show_expl]
    render_outputs = [out_html, out_grade, dl_btn, dl_btn, help_tip, grade_chip, grade_description,
                      expl_id, expl_btn, expl_out, state_view_map]
    
    btn_annotate.click(
    fn=_run_annotation,
    inputs=[text_in, threshold],
    outputs=[state_raw_df, Topic_filter, Fine_filter, qa_md]
    ).then(
        fn=_apply_filters_and_render,
        inputs=render_inputs,
        outputs=render_outputs
    )

    btn_annotate_file.click(
        fn=handle_file_upload,
        inputs=[file_in, csv_col_name, threshold, view_mode, font_size],
        outputs=[out_html_file, out_grade_file, dl_btn_file, dl_btn_file]
    )
    
    # SHAP explanation handler: Now relies on the hidden expl_method value
    expl_btn.click(
    fn=_explain_selected,
    inputs=[state_raw_df, expl_id, Topic_filter, Fine_filter, expl_method],
    outputs=[expl_out],
      )
    
    threshold.release(
    fn=_run_annotation,
    inputs=[text_in, threshold],
    outputs=[state_raw_df, Topic_filter, Fine_filter, qa_md]
    ).then(
        fn=_apply_filters_and_render,
        inputs=render_inputs,
        outputs=render_outputs
    )
    view_mode.change(fn=_apply_filters_and_render, inputs=render_inputs, outputs=render_outputs)
    font_size.release(fn=_apply_filters_and_render, inputs=render_inputs, outputs=render_outputs)
    Topic_filter.change(fn=_apply_filters_and_render, inputs=render_inputs, outputs=render_outputs)
    Fine_filter.change(fn=_apply_filters_and_render, inputs=render_inputs, outputs=render_outputs)
    
    # When toggling explanations, show/hide ID picker & button
    show_expl.change(
      fn=_toggle_expl_visibility,
      inputs=[show_expl, expl_id],
      outputs=[expl_id, expl_btn, expl_out],
      )
    show_expl.change(
    fn=_toggle_expl_visibility,
    inputs=[show_expl, expl_id],
    outputs=[expl_id, expl_btn, expl_out],
    )
    
    # Correctly chain the clear and re-render actions
    btn_Topics_clear.click(fn=lambda: gr.update(value=[]), inputs=None, outputs=[Topic_filter]).then(fn=_apply_filters_and_render, inputs=render_inputs, outputs=render_outputs)
    btn_Fine_clear.click(fn=lambda: gr.update(value=[]), inputs=None, outputs=[Fine_filter]).then(fn=_apply_filters_and_render, inputs=render_inputs, outputs=render_outputs)
    btn_clear.click(
    fn=_clear_all,
    outputs=[text_in, state_raw_df, out_html, out_grade, dl_btn, dl_btn, help_tip,
             Topic_filter, Fine_filter, grade_chip, qa_md]
    ).then(
        fn=lambda: (gr.update(choices=[], value=None, visible=False),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    {}) ,
        inputs=None,
        outputs=[expl_id, expl_btn, expl_out, state_view_map]
    )
    def handle_qa_request_with_tts(question, source_text):
      answer = handle_qa_request(question, source_text)   # or your own QA logic
      audio_path = synthesize_answer_with_elevenlabs(answer)
      return answer, audio_path
    
    qa_tts_btn = gr.Button("Ask (Text + Voice)")
    answer_md = gr.Markdown()
    answer_audio = gr.Audio(type="filepath")

    qa_tts_btn.click(
        fn=handle_qa_request_with_tts,
        inputs=[qa_q, text_in],
        outputs=[answer_md, answer_audio],
    )
    qa_voice_btn.click(
    fn=handle_voice_qa,
    inputs=[text_in, qa_audio],
    outputs=[qa_md],
    )
    galileo_btn.click(
        fn=simulate_galileo_evaluation,
        inputs=[state_raw_df],
        outputs=[galileo_out]
    )

if __name__ == "__main__":
    demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True,      # 👈 important
    show_api=False
)

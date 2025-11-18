# presentation.py
import pandas as pd
import html
import config
import data_loader
from typing import Dict, List, Tuple

def get_category_color(category_name: str) -> str:
    """Returns the CSS class name for a given category."""
    return config.CATEGORY_COLOR_MAP.get(category_name, "cat-other")

def df_to_cards_html(df: pd.DataFrame) -> str:
    """Render DataFrame rows as a horizontal scrollable list of cards."""
    if df is None or df.empty:
        return "<div class='section'>No results found matching your criteria.</div>"

    cards = []
    for _, r in df.iterrows():
        rating = str(r.get("rating", "neutral")).lower()
        display_text = str(r.get("merged_text") or r.get("text") or "")
        case_text = str(r.get('top1_label', ''))
        category_name = str(r.get('Category', 'Other'))
        cat_class_name = get_category_color(category_name)
        
        # Build category chips
        chips = f"<div class='chips chips-{cat_class_name}'>"
        chips += f"<span class='chip cat-main chip-{cat_class_name}' title='Category: {html.escape(category_name)}'>{html.escape(category_name)}</span>"
        for sub_level in ["SubCategory", "FineGrained"]:
            sub_name = str(r.get(sub_level, ''))
            if sub_name and sub_name != "—":
                chips += f"<span class='chip cat-sub chip-sub-{cat_class_name}' title='{sub_level}: {html.escape(sub_name)}'>{html.escape(sub_name)}</span>"
        chips += "</div>"
        
        icon_uri = r.get('icon_uri', '')
        icon_html = f"<img class='rating-icon' src='{icon_uri}' alt='{rating}'>" if icon_uri else ""
        
        case_url = data_loader.TOSDR_URL_MAP.get(case_text, "#")
        case_link_html = f"<a href='{html.escape(case_url)}' target='_blank' rel='noopener noreferrer' class='case-link external-link'>{html.escape(case_text)}</a>"

        card_html = f"""
        <div class='card rate-{rating} bg-cat-{cat_class_name}' tabindex='0'>
            <div class='meta'>
                <span class='badge {rating}'>{rating.upper()}</span>
                {icon_html}
                <span class='chip conf-chip'>Confidence: {float(r.get('confidence', 0)):.2f}</span>
            </div>
            <div class='field field-case'><span class='label'>Case:</span> {case_link_html}</div>
            {chips}
            <div class='field field-text'><span class='label'>Policy Text from Your Original Pasted Content:</span></div>
            <div class='txt'>{html.escape(display_text)}</div>
        </div>
        """
        cards.append(card_html)

    return "<div class='cards-wrap'>" + "".join(cards) + "</div>"

def tokens_to_html(text: str, token_weights, pos_color="#1e88e5", neg_color="#e65100"):
    """
    token_weights: List[(token, weight)] from LIME, positive supports label, negative opposes.
    We render a simple legend + token list. (Lightweight, no JS.)
    """
    # De-duplicate per token and keep the max |weight|
    agg = {}
    for tok, w in token_weights:
        if tok.strip():
            if tok not in agg or abs(w) > abs(agg[tok]):
                agg[tok] = w
    items = sorted(agg.items(), key=lambda x: -abs(x[1]))[:20]

    def bar(w):
        width = min(100, int(abs(w) * 100))
        color = pos_color if w >= 0 else neg_color
        return f'<span style="display:inline-block;height:8px;width:{width}px;background:{color};border-radius:4px;margin-left:8px;"></span>'

    lis = "".join(
        f'<li><code>{tok}</code> <small>({w:+.3f})</small>{bar(w)}</li>'
        for tok, w in items
    )
    legend = f"""
    <div class="section" aria-label="Explanation">
      <strong>Why this label?</strong>
      <div style="margin-top:6px;font-size:13px;">
        <span style="display:inline-block;width:14px;height:14px;background:{pos_color};border-radius:3px;margin-right:6px;"></span>
        pushes toward this label
        &nbsp;&nbsp;
        <span style="display:inline-block;width:14px;height:14px;background:{neg_color};border-radius:3px;margin-right:6px;"></span>
        pushes against
      </div>
      <ol style="margin-top:10px;">{lis}</ol>
      <p style="font-size:12px;color:#555;margin-top:10px;">Note: LIME shows a local approximation of the model’s decision for this sentence.</p>
    </div>
    """
    return legend

def df_to_table_html(df: pd.DataFrame, font_px: int = 15) -> str:
    """Render DataFrame as a standard HTML table."""
    if df is None or df.empty:
        return "<div class='section'>No results found matching your criteria.</div>"
    
    rows = []
    for _, r in df.iterrows():
        rating = str(r.get("rating", "")).lower()
        icon_uri = r.get("icon_uri", "")
        icon = f"<img src='{icon_uri}' alt='{rating}' style='display:block;margin:0 auto;width:28px;height:28px;object-fit:contain'/>" if icon_uri else "⚪"
        
        rows.append(f"""
        <tr class='rate-{rating}'>
            <td class='num'>{r.get('id', '')}</td>
            <td class='col-text'>{html.escape(str(r.get("merged_text") or r.get("text") or ""))}</td>
            <td class='col-case'>{html.escape(str(r.get('top1_label', '')))}</td>
            <td class='col-rating'>{html.escape(rating)}</td>
            <td class='col-icon'>{icon}</td>
            <td class='col-cat'>{html.escape(str(r.get('Category', '')))}</td>
            <td class='col-sub'>{html.escape(str(r.get('SubCategory', '')))}</td>
            <td class='col-fine'>{html.escape(str(r.get('FineGrained', '')))}</td>
            <td class='col-conf'>{float(r.get('confidence', 0)):.2f}</td>
        </tr>
        """)

    table_html = f"""
    <style>.predtbl{{font-size:{int(font_px)}px}}</style>
    <table class='predtbl' role='table'>
        <thead><tr>
            <th scope='col' class='num'>#</th><th scope='col' class='col-text'>Text</th>
            <th scope='col' class='col-case'>Case</th><th scope='col' class='col-rating'>Rating</th>
            <th scope='col' class='col-icon'></th><th scope='col' class='col-cat'>Category</th>
            <th scope='col' class='col-sub'>Sub-Category</th><th scope='col' class='col-fine'>Fine-grained</th>
            <th scope='col' class='col-conf'>Confidence</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """
    return table_html

def df_to_html(df: pd.DataFrame, font_px: int = 15, view_mode: str = "Table") -> str:
    """Main function to select rendering mode."""
    if view_mode == "Cards":
        return df_to_cards_html(df)
    return df_to_table_html(df, font_px=font_px)

def build_help_html() -> str:
    """Generates the HTML for the help tooltip."""
    return """
    <div class="help-tip" tabindex="0" aria-label="How the score is calculated">
      <span class="icon" aria-hidden="true">?</span>
      <div class="bubble" role="tooltip">
        <h4>How the score is calculated</h4>
        <ol style="padding-left:18px;margin:6px 0;">
          <li><b>Sentence classification:</b> each sentence is labeled with a ToS;DR case.</li>
          <li><b>Mapping to ratings:</b> cases map to <i>good</i>, <i>bad</i>, or <i>blocker</i>.</li>
          <li><b>Confidence threshold:</b> only sentences with confidence ≥ the slider are counted.</li>
          <li><b>Grade formula:</b> balance = <code>good − bad − 3×blocker</code> → letter grade.</li>
          <li><b>Filters:</b> Use filters to refine the results and the grade.</li>
        </ol>
      </div>
    </div>
    """

# presentation.py
from html import escape as _esc

def _bar_html(value, max_abs=0.2):
    v = max(-max_abs, min(max_abs, value))
    pct = abs(v) / max_abs * 100.0
    side = "pos" if v >= 0 else "neg"
    return f"""
    <div class="tw-row">
      <div class="tw-axis"></div>
      <div class="tw-bar tw-{side}" style="width:{pct:.1f}%"></div>
      <div class="tw-val">{value:+.2f}</div>
    </div>
    """

def render_lime_panel(
    *, 
    text,
    pred_label,
    confidence,
    token_weights,
    classes_sorted,
    prob_title="Prediction probabilities",
    others_top_k=4,           # show only next 4
    others_min_prob=0.001     # and only if prob >= 0.001
):
    # ---------- TOP CHIPS ----------
    chips_html = f"""
    <div class="chips">
      <span class="chip chip-case">Case: {_esc(pred_label)}</span>
      <span class="chip chip-conf">Confidence {_esc(f'{confidence:.2f}')}</span>
    </div>
    """

    # ---------- LEFT: TOP-5 PROBS ----------
    probs_rows = []
    for cls, p in classes_sorted[:5]:
        label = cls if len(cls) <= 22 else (cls[:22] + "…")
        probs_rows.append(f"""
          <div class="prob-row">
            <div class="prob-label">{_esc(label)}</div>
            <div class="prob-bar"><span style="width:{p*100:.0f}%"></span></div>
            <div class="prob-val">{p:.2f}</div>
          </div>
        """)
    probs_html = "\n".join(probs_rows)

    # ---------- MIDDLE: CONTRIBUTING WORDS (± bars) ----------
    top10 = token_weights[:10]
    maxw = max((abs(w) for _, w in top10), default=0.2) or 0.2

    tw_rows = []
    for tok, w in top10:
        tw_rows.append(f"""
          <div class="tok-row">
            <div class="tok-word">{_esc(tok)}</div>
            {_bar_html(w, max_abs=maxw)}
          </div>
        """)
    tw_html = "\n".join(tw_rows)

    # ---------- RIGHT: HIGHLIGHT TEXT ----------
    pos = {tok for tok, w in token_weights if w > 0}
    neg = {tok for tok, w in token_weights if w < 0}
    def _hl(word):
        if word in pos: return f"<mark class='hl-pos'>{_esc(word)}</mark>"
        if word in neg: return f"<mark class='hl-neg'>{_esc(word)}</mark>"
        return _esc(word)
    highlighted = " ".join(_hl(w) for w in text.split())

    # ---------- BOTTOM: OTHER CLASSES (NEXT 4, THRESHOLDED) ----------
    others_all = classes_sorted[1:]  # skip top-1
    others_kept = [(c, p) for c, p in others_all if p >= others_min_prob][:others_top_k]
    if others_kept:
        others_rows = [
            f"<div class='other-row'><span class='other-label'>{_esc(c)}</span>"
            f"<span class='other-dot'></span><span class='other-val'>{p:.3f}</span></div>"
            for c, p in others_kept
        ]
        others_html = "\n".join(others_rows)
    else:
        others_html = "<div class='muted'>No other classes above threshold.</div>"

    # ---------- STYLE + LAYOUT ----------
    return f"""
    <style>
      .chips {{ display:flex; gap:10px; margin:0 0 10px 0; align-items:center; }}
      .chip {{
        display:inline-flex; align-items:center; gap:8px;
        padding:6px 10px; border-radius:999px; font-weight:600; font-size:12px;
        border:1px solid #dbe7dc; background:#f3faf3; color:#23532a;
      }}
      .chip-conf {{ background:#eef6ff; border-color:#d6e8ff; color:#0b3d91; }}

      .prob-row, .tok-row, .other-row {{ display:flex; align-items:center; gap:10px; margin:6px 0; }}
      .prob-label {{ width:180px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
      .prob-bar {{ flex:1; height:10px; background:#edf5ef; border-radius:6px; position:relative; }}
      .prob-bar > span {{ position:absolute; left:0; top:0; bottom:0; background:#2e7d32; border-radius:6px; }}
      .prob-val {{ width:44px; text-align:right; }}

      .tok-word {{ width:120px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
      .tw-row {{ flex:1; display:flex; align-items:center; gap:8px; }}
      .tw-axis {{ width:1px; height:12px; background:#999; }}
      .tw-bar {{ height:10px; }}
      .tw-pos {{ background:#2e7d32; border-radius:0 6px 6px 0; }}
      .tw-neg {{ background:#d9534f; border-radius:6px 0 0 6px; margin-left:auto; }}
      .tw-val {{ width:46px; text-align:right; color:#444; }}

      .hl-pos {{ background:#dcedc8; color:#0b2e13; padding:2px 4px; border-radius:4px; }}
      .hl-neg {{ background:#ffebee; color:#b71c1c; padding:2px 4px; border-radius:4px; }}

      .other-label {{ flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
      .other-dot {{ width:6px; height:6px; border-radius:50%; background:#cbd5ce; display:inline-block; margin:0 8px; }}
      .other-val {{ width:64px; text-align:right; color:#555; }}
      .muted {{ color:#6b7280; }}

      .section .sub {{ font-weight:700; margin-bottom:6px; }}
    </style>

    {chips_html}

    <div class="section">
      <div style="display:grid; grid-template-columns: 1.1fr 1fr 1.2fr; gap:20px;">
        <div>
          <div class="sub">{_esc(prob_title)}</div>
          {probs_html}
        </div>
        <div>
          <div class="sub">Top contributing words</div>
          {tw_html}
        </div>
        <div>
          <div class="sub">Text with highlighted words</div>
          <div style="background:#f7fbf7; border:1px solid #e6efe7; border-radius:10px; padding:10px; line-height:1.55;">
            {highlighted}
          </div>
        </div>
      </div>

      <div style="margin-top:14px; padding-top:14px; border-top:1px solid #e6efe7;">
        <div class="sub">Other classes (low probability)</div>
        {others_html}
      </div>
    </div>
    """
"""
streamlit_app_v2.py — PDF Translator v2
Cải tiến so với v1:
  • Load pdf_analysis.json → biết gap, neighbor, role của từng dòng
  • enrich_groups(): tính relationships trực tiếp khi dịch (không cần tra JSON)
  • 3 strategies: shrink_font_first / expand_down_then_shrink / keep_original
  • TOC_ENTRY: chỉ dịch phần tiêu đề, giữ nguyên dấu ... và số trang
  • Không expand qua drawing/image
  • Redact padding từ config thực tế thay vì cố định
"""

import os, re, time, json, tempfile, threading
from pathlib import Path

import streamlit as st
import fitz
from google import genai
from google.genai import types as gtypes
from datetime import datetime

# ── Import toàn bộ từ v1 (không đụng code cũ) ────────────────────────────────
from streamlit_app import (
    check_password,
    find_font,
    _get_bold_font_path,
    extract_line_groups,
    translate_page,
    parse_page_range,
    call_gemini_live,
    _parse_json,
    int_to_rgb,
    API_KEY, APP_PASSWORD, MODEL_NAME,
    PRICE_INPUT, PRICE_OUTPUT, USD_TO_VND,
    DELAY_SEC, MAX_RETRIES, LANGUAGES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONFIG — đọc pdf_analysis.json
# ═══════════════════════════════════════════════════════════════════════════════

def load_layout_config():
    """
    Tìm pdf_analysis.json trong cùng folder với script.
    Trả về translation_config dict, hoặc None nếu không có.
    """
    candidates = [
        Path(__file__).parent / "pdf_analysis.json",
        Path("pdf_analysis.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                cfg = data.get("translation_config", {})
                if cfg:
                    return cfg
            except Exception:
                pass
    return None

# Load 1 lần khi app khởi động
LAYOUT_CFG = load_layout_config()

# Fallback nếu không có file JSON → dùng giá trị cứng (behavior v1)
DEFAULT_CFG = {
    "redact_padding":        {"top": 2, "bottom": 3.0, "left": 3, "right": 4},
    "safe_expand_text_pt":   5.0,
    "safe_expand_drawing_pt": 0,
    "expand_ratio_of_gap":   0.80,
    "min_fontsize_pt":       6.0,
    "shrink_step_pt":        0.5,
    "strategy_by_role": {
        "HEADING_1":       "shrink_font_first",
        "HEADING_2":       "shrink_font_first",
        "HEADING_3":       "shrink_font_first",
        "BODY_TEXT":       "expand_down_then_shrink",
        "LABEL_SMALL":     "expand_down_then_shrink",
        "TABLE_CELL":      "shrink_font_first",
        "TECHNICAL_VALUE": "keep_original_if_possible",
        "HEADER":          "shrink_font_first",
        "FOOTER":          "shrink_font_first",
        "TOC_ENTRY":       "translate_title_only",
    },
    "expansion_ratio": {"CJK": 1.9, "LATIN": 1.1, "OTHER": 1.15},
}

CFG = LAYOUT_CFG if LAYOUT_CFG else DEFAULT_CFG


# ═══════════════════════════════════════════════════════════════════════════════
# ENRICH GROUPS — tính relationships cho từng line ngay trong translator
# Không cần tra JSON — tính lại từ chính page đang xử lý
# ═══════════════════════════════════════════════════════════════════════════════

RE_TECHNICAL = re.compile(
    r'^[\d\s\.\,\-\+\/\(\)°×xX]*'
    r'(mm|cm|m|kg|kN|kNm|N|kW|W|rpm|m\/s|km\/h|Hz|V|A|°C|°|%|MPa|kPa|bar|pcs|nos?\.?)'
    r'[\d\s\.\,\-\+\/\(\)°×xX]*$', re.IGNORECASE
)
RE_CJK   = re.compile(r'[\u3000-\u9fff\uac00-\ud7af]')
RE_TOC   = re.compile(r'\.{4,}')

def _detect_role(text, fontsize, is_bold, bbox, pw, ph, ca):
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    rel_y = y0 / ph

    if RE_TOC.search(text):
        return "TOC_ENTRY"
    if rel_y < 0.07:
        return "HEADER" if w < pw * 0.40 else ("HEADING_1" if is_bold and fontsize >= 12 else "BODY_TEXT")
    if rel_y > 0.93:
        return "FOOTER"
    if is_bold:
        if fontsize >= 14: return "HEADING_1"
        if fontsize >= 12: return "HEADING_2"
        if fontsize >= 10: return "HEADING_3"
    if RE_TECHNICAL.match(text):
        return "TECHNICAL_VALUE"
    if len(text) <= 35 and fontsize <= 9:
        return "LABEL_SMALL"
    ca_w = max(ca[2] - ca[0], 1)
    if len(text) <= 50 and (x0 - ca[0]) / ca_w > 0.25 and w < ca_w * 0.35:
        return "TABLE_CELL"
    return "BODY_TEXT"

def _lang_hint(text):
    if RE_CJK.search(text): return "CJK"
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return "LATIN" if ascii_count / max(len(text), 1) > 0.60 else "OTHER"

def _h_overlap(a, b, tol=5):
    return a[0] < b[2] + tol and a[2] > b[0] - tol

def _get_page_meta(page):
    """Lấy content_area, images, drawings của page."""
    pw, ph = page.rect.width, page.rect.height

    # Images
    images = []
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            images.append([r.x0, r.y0, r.x1, r.y1])

    # Drawings (gộp)
    raw = page.get_drawings()
    drw_rects = []
    for path in raw:
        r = path.get("rect")
        if r and (r.width >= 2 or r.height >= 2):
            drw_rects.append(fitz.Rect(r))
    drawings = []
    used = [False] * len(drw_rects)
    for i, ri in enumerate(drw_rects):
        if used[i]: continue
        g = fitz.Rect(ri); used[i] = True
        for j, rj in enumerate(drw_rects):
            if used[j]: continue
            if fitz.Rect(g.x0-5, g.y0-5, g.x1+5, g.y1+5).intersects(rj):
                g |= rj; used[j] = True
        drawings.append([g.x0, g.y0, g.x1, g.y1])

    return pw, ph, images, drawings

def enrich_groups(groups, page):
    """
    Bổ sung vào mỗi group:
      - text_role, lang_hint
      - gap_to_next_pt, neighbor_below
      - available_expand_down
    Dùng để _insert_line_v2 chọn đúng strategy.
    """
    if not groups:
        return groups

    pw, ph, images, drawings = _get_page_meta(page)

    # Content area từ groups
    all_x0 = [g["bbox"][0] for g in groups]
    all_x1 = [g["bbox"][2] for g in groups]
    all_y0 = [g["bbox"][1] for g in groups]
    all_y1 = [g["bbox"][3] for g in groups]
    ca = (min(all_x0), min(all_y0), max(all_x1), max(all_y1))

    for i, g in enumerate(groups):
        x0, y0, x1, y1 = g["bbox"]
        text     = g["text"].strip()
        fontsize = g["size"]
        is_bold  = g.get("bold", False)

        # Role & lang
        g["text_role"] = _detect_role(text, fontsize, is_bold, g["bbox"], pw, ph, ca)
        g["lang_hint"] = _lang_hint(text)

        # Right margin
        g["right_margin_pt"] = max(round(ca[2] - x1, 2), 0)

        # Neighbor below: tìm phần tử gần nhất bên dưới
        best_gap  = ph - y1
        best_type = "page_end"

        for j, other in enumerate(groups):
            if i == j: continue
            oy0 = other["bbox"][1]
            if oy0 <= y1 + 1: continue
            if not _h_overlap(g["bbox"], other["bbox"]): continue
            gap = oy0 - y1
            if gap < best_gap:
                best_gap, best_type = gap, "text"

        for img in images:
            iy0 = img[1]
            if iy0 <= y1 + 1: continue
            if not _h_overlap(g["bbox"], img): continue
            gap = iy0 - y1
            if gap < best_gap:
                best_gap, best_type = gap, "image"

        for drw in drawings:
            dy0 = drw[1]
            if dy0 <= y1 + 1: continue
            if not _h_overlap(g["bbox"], drw): continue
            gap = dy0 - y1
            if gap < best_gap:
                best_gap, best_type = gap, "drawing"

        g["gap_to_next_pt"]   = round(max(best_gap, 0), 2)
        g["neighbor_below"]   = best_type

        # available_expand_down = 0 nếu drawing/image bên dưới
        if best_type in ("drawing", "image"):
            g["available_expand_down"] = 0.0
        else:
            g["available_expand_down"] = round(best_gap * CFG["expand_ratio_of_gap"], 2)

    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# TOC HANDLER — tách và ghép lại dòng mục lục
# ═══════════════════════════════════════════════════════════════════════════════

def split_toc(text):
    """
    Tách dòng TOC thành (title, dots, page_num).
    VD: "1.1 Safety .............. 5" → ("1.1 Safety", "...............", "5")
    """
    m = re.search(r'(\.{4,})\s*(\d+\s*)?$', text)
    if not m:
        return text, "", ""
    dots     = m.group(1)
    page_num = (m.group(2) or "").strip()
    title    = text[:m.start()].strip()
    return title, dots, page_num

def rebuild_toc(translated_title, dots, page_num):
    """Ghép lại sau khi dịch title."""
    if not dots:
        return translated_title
    parts = [translated_title, " ", dots]
    if page_num:
        parts += [" ", page_num]
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# INSERT LINE V2 — ghi text dịch vào page theo đúng strategy
# ═══════════════════════════════════════════════════════════════════════════════

def _insert_line_v2(page, font_path, font_bold_path, group, translated_text):
    """
    Ghi text đã dịch vào page với strategy phù hợp từng role.

    Strategy A — shrink_font_first (HEADING, TABLE_CELL, HEADER, FOOTER):
      Shrink font từng 0.5pt cho đến khi vừa bbox gốc.

    Strategy B — expand_down_then_shrink (BODY_TEXT, LABEL_SMALL):
      Mở rộng y1 xuống dưới = available_expand_down trước.
      Nếu vẫn không vừa → shrink font.

    Strategy C — keep_original_if_possible (TECHNICAL_VALUE):
      Nếu bản dịch ngắn hơn → giữ nguyên.
      Nếu dài hơn → shrink nhẹ 1-2 bước thôi.
    """
    x0, y0, x1, y1 = group["bbox"]
    role      = group.get("text_role", "BODY_TEXT")
    avail_dwn = group.get("available_expand_down", 0.0)
    orig_size = group["size"]
    bold      = group.get("bold", False)
    color     = group.get("rgb", (0, 0, 0))

    pw = page.rect.width
    ph = page.rect.height
    min_fs    = CFG["min_fontsize_pt"]
    step      = CFG["shrink_step_pt"]
    strategy  = CFG["strategy_by_role"].get(role, "expand_down_then_shrink")

    # Setup font
    if font_path:
        fontfile = font_bold_path if bold and font_bold_path else font_path
        fontname = "FBold" if bold else "FReg"
        try:
            page.insert_font(fontname=fontname, fontfile=fontfile)
        except Exception:
            pass
    else:
        fontname = "hebo" if bold else "helv"

    # x1 không vượt quá page width - margin
    x1_safe = min(x1, pw - 20)

    # ── Strategy A: shrink_font_first ─────────────────────────────────────────
    if strategy == "shrink_font_first":
        size = orig_size
        while size >= min_fs:
            rc = page.insert_textbox(
                fitz.Rect(x0, y0, x1_safe, y1),
                translated_text,
                fontsize=size, fontname=fontname,
                color=color, align=0,
            )
            if rc >= 0:
                return
            size -= step
        # Vẫn không vừa → truncate
        _insert_truncated(page, fontname, x0, y0, x1_safe, y1,
                          translated_text, min_fs, color)
        return

    # ── Strategy B: expand_down_then_shrink ───────────────────────────────────
    if strategy == "expand_down_then_shrink":
        # Thử với bbox gốc trước
        rc = page.insert_textbox(
            fitz.Rect(x0, y0, x1_safe, y1),
            translated_text,
            fontsize=orig_size, fontname=fontname,
            color=color, align=0,
        )
        if rc >= 0:
            return

        # Expand y1 xuống = available_expand_down (đã = 0 nếu drawing bên dưới)
        if avail_dwn > 0:
            y1_exp = min(y1 + avail_dwn, ph - 10)
            rc = page.insert_textbox(
                fitz.Rect(x0, y0, x1_safe, y1_exp),
                translated_text,
                fontsize=orig_size, fontname=fontname,
                color=color, align=0,
            )
            if rc >= 0:
                return
            # Shrink font trong expanded bbox
            size = orig_size - step
            while size >= min_fs:
                rc = page.insert_textbox(
                    fitz.Rect(x0, y0, x1_safe, y1_exp),
                    translated_text,
                    fontsize=size, fontname=fontname,
                    color=color, align=0,
                )
                if rc >= 0:
                    return
                size -= step
        else:
            # Không expand được → shrink font trong bbox gốc
            size = orig_size - step
            while size >= min_fs:
                rc = page.insert_textbox(
                    fitz.Rect(x0, y0, x1_safe, y1),
                    translated_text,
                    fontsize=size, fontname=fontname,
                    color=color, align=0,
                )
                if rc >= 0:
                    return
                size -= step

        _insert_truncated(page, fontname, x0, y0, x1_safe, y1,
                          translated_text, min_fs, color)
        return

    # ── Strategy C: keep_original_if_possible ─────────────────────────────────
    if strategy == "keep_original_if_possible":
        # Thử với font gốc
        rc = page.insert_textbox(
            fitz.Rect(x0, y0, x1_safe, y1),
            translated_text,
            fontsize=orig_size, fontname=fontname,
            color=color, align=0,
        )
        if rc >= 0:
            return
        # Chỉ shrink 2 bước thôi
        for i in range(1, 3):
            size = orig_size - step * i
            if size < min_fs:
                break
            rc = page.insert_textbox(
                fitz.Rect(x0, y0, x1_safe, y1),
                translated_text,
                fontsize=size, fontname=fontname,
                color=color, align=0,
            )
            if rc >= 0:
                return
        # Fallback: giữ nguyên text gốc
        page.insert_textbox(
            fitz.Rect(x0, y0, x1_safe, y1),
            group["text"],
            fontsize=orig_size, fontname=fontname,
            color=color, align=0,
        )
        return

    # ── Fallback ──────────────────────────────────────────────────────────────
    size = orig_size
    while size >= min_fs:
        rc = page.insert_textbox(
            fitz.Rect(x0, y0, x1_safe, y1),
            translated_text,
            fontsize=size, fontname=fontname,
            color=color, align=0,
        )
        if rc >= 0:
            return
        size -= step

def _insert_truncated(page, fontname, x0, y0, x1, y1, text, fontsize, color):
    """Truncate text với dấu … nếu không thể fit dù đã shrink hết."""
    short = text
    while len(short) > 3:
        short = short[:-4] + "…"
        rc = page.insert_textbox(
            fitz.Rect(x0, y0, x1, y1), short,
            fontsize=fontsize, fontname=fontname,
            color=color, align=0,
        )
        if rc >= 0:
            return
    # Cuối cùng: bỏ qua
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE TRANSLATED PDF V2
# ═══════════════════════════════════════════════════════════════════════════════

def write_translated_pdf_v2(src, dst, all_groups, all_trans, font_path):
    """
    Ghi PDF đã dịch với layout-aware insertion.
    Cải tiến so với v1:
      - Redact padding từ CFG thực tế (không hardcode)
      - Expand redact xuống = available_expand_down của từng line
      - Không expand qua drawing/image
      - Chọn strategy theo text_role
      - TOC_ENTRY: chỉ dịch title, giữ dấu ...
    """
    doc            = fitz.open(src)
    font_bold_path = _get_bold_font_path(font_path)
    pad            = CFG["redact_padding"]

    for pi, groups in all_groups.items():
        trans = all_trans.get(pi, [])
        if not groups or not trans:
            continue

        page = doc[pi]

        # Enrich groups với relationship data
        enrich_groups(groups, page)

        # ── Bước 1: Redact toàn bộ text gốc ─────────────────────────────────
        for g in groups:
            r    = fitz.Rect(g["bbox"])
            avail = g.get("available_expand_down", 0.0)

            # Expand redact xuống đúng khoảng cần thiết
            # Nếu neighbor là drawing/image → không expand xuống (avail=0)
            expand_bottom = min(avail, CFG["safe_expand_text_pt"])

            redact_rect = fitz.Rect(
                r.x0 - pad["left"],
                r.y0 - pad["top"],
                r.x1 + pad["right"],
                r.y1 + pad["bottom"] + expand_bottom,
            )
            page.add_redact_annot(redact_rect.intersect(page.rect))

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # ── Bước 2: Insert text đã dịch ──────────────────────────────────────
        for g, t in zip(groups, trans):
            role = g.get("text_role", "BODY_TEXT")

            # Xử lý text dịch
            if not t or not t.strip():
                final_text = g["text"]  # fallback về text gốc
            elif role == "TOC_ENTRY":
                # Chỉ dịch phần title, giữ dấu ... và số trang
                orig_title, dots, page_num = split_toc(g["text"])
                trans_title, _, _          = split_toc(t)
                # Dùng trans_title nếu khác rỗng, không thì dùng orig
                title_to_use = trans_title.strip() if trans_title.strip() else orig_title
                final_text = rebuild_toc(title_to_use, dots, page_num)
            elif role == "TECHNICAL_VALUE":
                # Giữ nguyên nếu bản dịch dài hơn nhiều (có thể AI hallucinate)
                exp = CFG["expansion_ratio"].get(g.get("lang_hint", "LATIN"), 1.1)
                if len(t) > len(g["text"]) * exp * 1.3:
                    final_text = g["text"]
                else:
                    final_text = t.strip()
            else:
                final_text = t.strip()

            try:
                _insert_line_v2(page, font_path, font_bold_path, g, final_text)
            except Exception:
                # Fallback: insert text gốc
                try:
                    _insert_line_v2(page, font_path, font_bold_path, g, g["text"])
                except Exception:
                    pass

    doc.save(dst, garbage=4, deflate=True)
    doc.close()


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI — giữ nguyên y chang v1, chỉ thay write function
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dịch PDF v2 — Vi Nguyen",
    page_icon="⬡",
    layout="centered",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .block-container { max-width: 780px; padding-top: 2rem; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li, span { color: #e2e8f0; }

    div[data-testid="stFileUploader"] { border: 1.5px dashed #4a5080 !important; border-radius: 10px !important; padding: 8px !important; background: #1a1d27 !important; }
    div[data-testid="stFileUploader"] label { color: #e2e8f0 !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #242838 !important; border-color: #4a5080 !important; }
    [data-testid="stFileUploaderDropzone"] > div { background-color: #242838 !important; }
    [data-testid="stFileUploaderDropzone"] span { color: #e2e8f0 !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #2d3149 !important; color: #e2e8f0 !important; border-color: #4a5080 !important; }
    [data-testid="stFileUploaderDropzone"] p { color: #94a3b8 !important; }

    label, .stSelectbox label, .stTextInput label,
    div[data-testid="stWidgetLabel"] p { color: #e2e8f0 !important; font-weight: 500; }
    .stSelectbox div[data-baseweb="select"] > div { background: #1a1d27 !important; color: #e2e8f0 !important; border-color: #4a5080 !important; }
    .stTextInput input { background: #1a1d27 !important; color: #e2e8f0 !important; border-color: #4a5080 !important; }
    input::placeholder { color: #64748b !important; }

    .stat-box { background: #1a1d27; border-radius: 10px; padding: 14px 18px; text-align: center; border: 1px solid #4a5080; }
    .stat-val { font-size: 1.4rem; font-weight: bold; color: #e2e8f0; }
    .stat-lbl { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }

    .timer-box { background: #1a1d27; border-radius: 10px; padding: 12px 18px; text-align: center; border: 1px solid #6c63ff; margin-bottom: 10px; }
    .timer-val { font-size: 2rem; font-weight: bold; color: #a78bfa; font-family: 'Courier New', monospace; }
    .timer-lbl { font-size: 0.78rem; color: #94a3b8; margin-top: 2px; }
    .timer-status { font-size: 0.85rem; color: #818cf8; margin-top: 6px; }

    .log-box { background: #1a1d27; border-radius: 8px; padding: 14px 18px; font-family: 'Courier New', monospace; font-size: 0.83rem; max-height: 340px; overflow-y: auto; border: 1px solid #4a5080; white-space: pre-wrap; color: #c4cde0; line-height: 1.6; }

    .stButton > button { background-color: #6c63ff !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 10px 28px !important; font-weight: bold !important; font-size: 1rem !important; width: 100% !important; }
    .stButton > button:hover { background-color: #a78bfa !important; }
    .stButton > button:disabled { background-color: #2d3149 !important; color: #64748b !important; }

    .stDownloadButton > button { background-color: #059669 !important; color: white !important; border-radius: 8px !important; font-weight: bold !important; font-size: 1rem !important; width: 100% !important; }
    .stDownloadButton > button:hover { background-color: #10b981 !important; }

    div[data-testid="stProgressBar"] > div { background-color: #1a1d27 !important; }
    div[data-testid="stProgressBar"] > div > div { background-color: #6c63ff !important; }

    hr { border-color: #2d3149 !important; }
    div[data-testid="stAlert"] { background: #1a1d27 !important; border-color: #4a5080 !important; color: #e2e8f0 !important; }

    .login-box { background: #1a1d27; border: 1px solid #4a5080; border-radius: 14px; padding: 2.5rem 2rem; max-width: 380px; margin: 4rem auto 0 auto; text-align: center; }
    .login-title { font-size: 1.6rem; font-weight: bold; color: #e2e8f0; margin-bottom: 0.3rem; }
    .login-sub   { font-size: 0.85rem; color: #64748b; margin-bottom: 1.8rem; }

    .v2-badge { display:inline-block; background:#4c1d95; color:#c4b5fd; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:bold; margin-left:8px; }
    .cfg-box { background:#1a1d27; border:1px solid #4a5080; border-radius:8px; padding:10px 14px; font-size:0.8rem; color:#94a3b8; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# Password gate
if not check_password():
    st.stop()

# Header
st.markdown(
    "## ⬡ Dịch PDF sang PDF "
    "<span class='v2-badge'>v2</span>",
    unsafe_allow_html=True,
)
st.markdown(
    "<span style='color:#64748b;font-size:0.9rem'>Powered by Gemini Flash — Vi Nguyen</span>",
    unsafe_allow_html=True,
)

# Config status
if LAYOUT_CFG:
    st.markdown(
        f"<div class='cfg-box'>✅ Layout config loaded — "
        f"safe_expand: <b>{CFG['safe_expand_text_pt']}pt</b> · "
        f"min_font: <b>{CFG['min_fontsize_pt']}pt</b> · "
        f"strategies: {len(CFG['strategy_by_role'])} roles</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='cfg-box'>⚠️ Không tìm thấy pdf_analysis.json — dùng config mặc định</div>",
        unsafe_allow_html=True,
    )

col_title, col_logout = st.columns([5, 1])
with col_logout:
    if st.button("🚪 Logout"):
        st.session_state.pop("authenticated", None)
        st.rerun()

st.divider()

uploaded = st.file_uploader("📄 Chọn file PDF cần dịch", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    lang = st.selectbox("🌐 Ngôn ngữ đích", LANGUAGES)
with col2:
    pages_s = st.text_input(
        "📑 Trang cụ thể (tuỳ chọn)",
        placeholder="Vd: 1-5,8  •  Để trống = tất cả",
    )

st.divider()

run_btn = st.button("▶  Bắt đầu dịch", disabled=(uploaded is None))

# ── Chạy dịch ────────────────────────────────────────────────────────────────
if run_btn and uploaded:
    for k in ["pdf_bytes", "out_name", "summary"]:
        st.session_state.pop(k, None)

    st.markdown("### 📊 Tiến độ")
    timer_ph = st.empty()

    col_pg, col_ln, col_usd, col_vnd = st.columns(4)
    ph_pages = col_pg.empty()
    ph_lines = col_ln.empty()
    ph_usd   = col_usd.empty()
    ph_vnd   = col_vnd.empty()

    def render_stats(pages_done, total_pg, total_lines, tok_in, tok_out):
        usd = ((tok_in / 1e6) * PRICE_INPUT + (tok_out / 1e6) * PRICE_OUTPUT) * 10
        vnd = usd * USD_TO_VND
        ph_pages.markdown(f"<div class='stat-box'><div class='stat-val'>{pages_done}/{total_pg}</div><div class='stat-lbl'>Trang</div></div>", unsafe_allow_html=True)
        ph_lines.markdown(f"<div class='stat-box'><div class='stat-val'>{total_lines:,}</div><div class='stat-lbl'>Dòng text</div></div>", unsafe_allow_html=True)
        ph_usd.markdown(f"<div class='stat-box'><div class='stat-val'>${usd:.4f}</div><div class='stat-lbl'>USD</div></div>", unsafe_allow_html=True)
        ph_vnd.markdown(f"<div class='stat-box'><div class='stat-val'>{vnd:,.0f}₫</div><div class='stat-lbl'>VND</div></div>", unsafe_allow_html=True)

    render_stats(0, 0, 0, 0, 0)
    progress = st.progress(0, text="Đang chuẩn bị...")

    st.markdown("### 📋 Nhật ký hoạt động")
    log_ph    = st.empty()
    log_lines = []

    def add_log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        log_ph.markdown(
            f"<div class='log-box'>{'<br>'.join(log_lines[-40:])}</div>",
            unsafe_allow_html=True,
        )

    src_path = dst_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
            tmp_in.write(uploaded.read())
            src_path = tmp_in.name
        tmp_out  = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        dst_path = tmp_out.name
        tmp_out.close()

        add_log(f"📄 Đã nhận file: {uploaded.name}")

        probe    = fitz.open(src_path)
        total_pg = len(probe)
        probe.close()

        targets = (parse_page_range(pages_s, total_pg) if pages_s.strip()
                   else list(range(total_pg)))
        add_log(f"📄 {total_pg} trang tổng, sẽ dịch {len(targets)} trang")
        if LAYOUT_CFG:
            add_log(f"⚙️  v2 layout-aware: expand={CFG['safe_expand_text_pt']}pt · min_font={CFG['min_fontsize_pt']}pt")

        progress.progress(5, text="Trích xuất text...")
        add_log("🔍 Trích xuất text từ PDF...")
        all_groups, _ = extract_line_groups(src_path, targets)
        total_lines   = sum(len(v) for v in all_groups.values())
        add_log(f"✅ {total_lines} dòng text")
        render_stats(0, total_pg, total_lines, 0, 0)

        client = genai.Client(api_key=API_KEY)
        add_log(f"🤖 Kết nối {MODEL_NAME} thành công")

        all_trans = {}
        tok_in = tok_out = 0
        t0 = time.time()

        for idx, pi in enumerate(targets):
            groups     = all_groups.get(pi, [])
            page_start = time.time()
            pct        = int(10 + (idx / len(targets)) * 80)

            progress.progress(pct, text=f"Dịch trang {pi + 1}/{total_pg}...")
            add_log(f"📄 Trang {pi + 1}/{total_pg}: {len(groups)} dòng — gửi Gemini API...")

            if not groups:
                all_trans[pi] = []
                timer_ph.markdown(
                    f"""<div class='timer-box'>
                        <div class='timer-val'>⏱ {time.time()-t0:.0f}s</div>
                        <div class='timer-lbl'>Tổng thời gian đã chạy</div>
                        <div class='timer-status'>⏭ Trang {pi+1} trống, bỏ qua</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                continue

            try:
                trans, in_t, out_t = translate_page(
                    client, groups, lang, pi,
                    timer_ph, t0, page_start, log_lines, log_ph,
                )
                tok_in  += in_t
                tok_out += out_t
                add_log(f"   ✅ {len(trans)} dòng ({in_t:,} in / {out_t:,} out tok) — {time.time()-page_start:.1f}s")
            except Exception as e:
                add_log(f"   ❌ Lỗi trang {pi + 1}: {e}")
                trans = [g["text"] for g in groups]

            all_trans[pi] = trans
            render_stats(idx + 1, total_pg, total_lines, tok_in, tok_out)
            time.sleep(DELAY_SEC)

        progress.progress(92, text="Tạo PDF...")
        add_log("💾 Đang tạo PDF (v2 layout-aware)...")
        timer_ph.markdown(
            f"""<div class='timer-box'>
                <div class='timer-val'>⏱ {time.time()-t0:.0f}s</div>
                <div class='timer-lbl'>Tổng thời gian đã chạy</div>
                <div class='timer-status'>💾 Đang ghi file PDF...</div>
            </div>""",
            unsafe_allow_html=True,
        )

        font_path = find_font()
        add_log(f"🔤 Font: {os.path.basename(font_path) if font_path else 'built-in'}")

        # ← Đây là thay đổi duy nhất so với v1: dùng write_translated_pdf_v2
        write_translated_pdf_v2(src_path, dst_path, all_groups, all_trans, font_path)

        elapsed = time.time() - t0
        usd = ((tok_in / 1e6) * PRICE_INPUT + (tok_out / 1e6) * PRICE_OUTPUT) * 10
        vnd = usd * USD_TO_VND
        render_stats(len(targets), total_pg, total_lines, tok_in, tok_out)
        progress.progress(100, text="✅ Hoàn thành!")

        timer_ph.markdown(
            f"""<div class='timer-box' style='border-color:#059669'>
                <div class='timer-val' style='color:#10b981'>✅ {elapsed:.1f}s</div>
                <div class='timer-lbl'>Tổng thời gian</div>
                <div class='timer-status' style='color:#10b981'>Dịch xong {len(targets)} trang!</div>
            </div>""",
            unsafe_allow_html=True,
        )

        add_log("─" * 44)
        add_log(f"🎉 Xong {len(targets)} trang trong {elapsed:.1f}s")
        add_log(f"💰 Token: {tok_in:,} in + {tok_out:,} out")
        add_log(f"💵 Chi phí: ${usd:.4f} USD ≈ {vnd:,.0f} VND")

        with open(dst_path, "rb") as f:
            st.session_state["pdf_bytes"] = f.read()
        st.session_state["out_name"] = uploaded.name.replace(".pdf", f"_v2_{lang[:2]}.pdf")
        st.session_state["summary"]  = (
            f"✅ Dịch xong {len(targets)} trang trong {elapsed:.1f}s  |  "
            f"${usd:.4f} USD ≈ {vnd:,.0f} VND"
        )

    except Exception as e:
        add_log(f"❌ Lỗi: {e}")
        st.error(f"❌ Có lỗi xảy ra: {e}")
        timer_ph.markdown(
            f"""<div class='timer-box' style='border-color:#ef4444'>
                <div class='timer-val' style='color:#ef4444'>❌ Lỗi</div>
                <div class='timer-lbl'>Đã dừng</div>
                <div class='timer-status' style='color:#ef4444'>{str(e)[:80]}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    finally:
        for p in [src_path, dst_path]:
            try:
                if p: os.unlink(p)
            except Exception:
                pass

# ── Download ──────────────────────────────────────────────────────────────────
if "pdf_bytes" in st.session_state:
    st.divider()
    st.success(st.session_state["summary"])
    st.download_button(
        label="⬇️  Tải PDF đã dịch (v2)",
        data=st.session_state["pdf_bytes"],
        file_name=st.session_state["out_name"],
        mime="application/pdf",
        use_container_width=True,
    )
elif not uploaded:
    st.info("👆 Vui lòng upload file PDF để bắt đầu")

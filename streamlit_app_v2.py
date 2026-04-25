"""
streamlit_app_v2.py — PDF Translator v2 (FIXED)

Fix so với v2 gốc:
  • enrich_groups(): thêm check "same-row neighbor" cho bảng multi-column
    - Nếu 1 span có neighbor ở cùng y-band nhưng x-column khác → đây là TABLE ROW
    - y1_insert_max bị giới hạn bởi y1 của row đó (không expand vượt hàng)
    - Strategy tự động chuyển sang shrink_font_first thay vì expand_down_then_shrink
  • Không còn tình trạng block "Schindler/GFQE/CH-6030" expand xuống và đẩy
    "ngoại lệ:" vào vùng của cột khác
"""

import os, re, time, json, tempfile, threading
from pathlib import Path
from datetime import datetime

import streamlit as st
import fitz
from google import genai
from google.genai import types as gtypes

from streamlit_app import (
    check_password, find_font, _get_bold_font_path,
    extract_line_groups, translate_page, parse_page_range,
    call_gemini_live, _parse_json, int_to_rgb,
    API_KEY, APP_PASSWORD, MODEL_NAME,
    PRICE_INPUT, PRICE_OUTPUT, USD_TO_VND,
    DELAY_SEC, MAX_RETRIES, LANGUAGES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cfg():
    for name in ["pdf_analysis_slim.json", "pdf_analysis.json"]:
        for base in [Path(__file__).parent, Path(".")]:
            p = base / name
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    cfg = d.get("translation_config", {})
                    if cfg:
                        return cfg, str(p)
                except Exception:
                    pass
    return None, None

_raw_cfg, _cfg_path = _load_cfg()

CFG = _raw_cfg or {
    "redact_padding":   {"top": 2, "bottom": 3, "left": 3, "right": 4},
    "min_fontsize_pt":  6.5,
    "shrink_step_pt":   0.5,
    "strategy_by_role": {
        "HEADING_1": "expand_down_then_shrink",
        "HEADING_2": "expand_down_then_shrink",
        "HEADING_3": "expand_down_then_shrink",
        "BODY_TEXT": "expand_down_then_shrink",
        "LABEL_SMALL": "expand_down_then_shrink",
        "TABLE_CELL": "shrink_font_first",
        "TECHNICAL_VALUE": "keep_original_if_possible",
        "HEADER": "shrink_font_first",
        "FOOTER": "shrink_font_first",
        "TOC_ENTRY": "translate_title_only",
    },
    "expansion_ratio": {"CJK": 1.9, "LATIN": 1.1, "OTHER": 1.15},
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

RE_TECH = re.compile(
    r'^[\d\s\.\,\-\+\/\(\)°×xX]*'
    r'(mm|cm|m|kg|kN|kNm|N|kW|W|rpm|m\/s|km\/h|Hz|V|A|°C|°|%|MPa|kPa|bar|pcs|nos?\.?)'
    r'[\d\s\.\,\-\+\/\(\)°×xX]*$', re.IGNORECASE
)
RE_TOC = re.compile(r'\.{4,}')
RE_CJK = re.compile(r'[\u3000-\u9fff\uac00-\ud7af]')

def _classify(text, size, bold, bbox, pw, ph):
    x0, y0, x1, y1 = bbox
    rel_y = y0 / ph
    if RE_TOC.search(text):                                 return "TOC_ENTRY"
    if rel_y < 0.07 and (x1 - x0) < pw * 0.40:            return "HEADER"
    if rel_y > 0.93:                                        return "FOOTER"
    if bold:
        if size >= 14: return "HEADING_1"
        if size >= 12: return "HEADING_2"
        if size >= 10: return "HEADING_3"
    if RE_TECH.match(text):                                 return "TECHNICAL_VALUE"
    if len(text) <= 35 and size <= 9:                       return "LABEL_SMALL"
    return "BODY_TEXT"

def _lang(text):
    if RE_CJK.search(text): return "CJK"
    return "LATIN" if sum(1 for c in text if ord(c) < 128) / max(len(text), 1) > 0.6 else "OTHER"

def _h_overlap(a, b, tol=3):
    return a[0] < b[2] + tol and a[2] > b[0] - tol


# ═══════════════════════════════════════════════════════════════════════════════
# ENRICH GROUPS — FIXED
# ═══════════════════════════════════════════════════════════════════════════════

def _get_obstacles(page):
    obs = []
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            obs.append([r.x0, r.y0, r.x1, r.y1, "image"])
    for path in page.get_drawings():
        r = path.get("rect")
        if r and r.width >= 5 and r.height >= 5:
            obs.append([r.x0, r.y0, r.x1, r.y1, "drawing"])
    return obs


def _has_same_row_neighbor(block, all_groups, y_tol=4):
    """
    Kiểm tra xem block có span nào CÙNG ROW (y gần nhau) nhưng KHÁC CỘT (x cách xa) không.
    Nếu có → đây là bảng multi-column → không được expand y1 vượt qua row đó.

    Trả về y1_row_limit nếu phát hiện table row, None nếu không.
    """
    block_x0 = min(g["bbox"][0] for g in block)
    block_x1 = max(g["bbox"][2] for g in block)
    block_y0 = min(g["bbox"][1] for g in block)
    block_y1 = max(g["bbox"][3] for g in block)

    for g in all_groups:
        if any(g is b for b in block):
            continue
        gx0, gy0, gx1, gy1 = g["bbox"]

        # Cùng row: y-band overlap với block
        y_same_row = gy0 < block_y1 + y_tol and gy1 > block_y0 - y_tol

        # Khác cột: KHÔNG overlap ngang (kể cả tol lớn)
        x_different_col = not _h_overlap(
            [block_x0, block_y0, block_x1, block_y1],
            [gx0, gy0, gx1, gy1],
            tol=15
        )

        if y_same_row and x_different_col:
            # Phát hiện multi-column table row
            # Giới hạn y1 = max(block_y1, y1 của tất cả neighbors cùng row)
            return block_y1
    return None


def enrich_groups(groups, page):
    """
    Tính y1_insert_max cho mỗi group.

    FIX: Thêm kiểm tra "same-row neighbor" để phát hiện bảng multi-column.
    Nếu block nằm trong bảng (có neighbor cùng row, khác cột):
      - y1_insert_max = y1 của block (không expand xuống)
      - strategy → shrink_font_first
    """
    if not groups:
        return groups

    pw, ph = page.rect.width, page.rect.height
    obstacles = _get_obstacles(page)

    # Bước 1: Classify
    for g in groups:
        g["text_role"] = _classify(g["text"].strip(), g["size"], g.get("bold", False), g["bbox"], pw, ph)
        g["lang_hint"] = _lang(g["text"])

    # Sắp xếp theo y (top to bottom)
    groups.sort(key=lambda g: (round(g["bbox"][1] / 3) * 3, g["bbox"][0]))

    # Bước 2: Nhóm thành paragraph blocks
    blocks = []
    cur_block = [groups[0]]

    for i in range(1, len(groups)):
        prev = groups[i - 1]
        curr = groups[i]
        py1 = prev["bbox"][3]
        cy0 = curr["bbox"][1]
        gap = cy0 - py1
        line_h = max(prev["bbox"][3] - prev["bbox"][1], 1)

        same_block = (
            gap < line_h * 0.7 and
            _h_overlap(prev["bbox"], curr["bbox"], tol=20)
        )
        if same_block:
            cur_block.append(curr)
        else:
            blocks.append(cur_block)
            cur_block = [curr]
    blocks.append(cur_block)

    # Bước 3: Tính y1_insert_max cho mỗi block
    for block in blocks:
        block_x0 = min(g["bbox"][0] for g in block)
        block_x1 = max(g["bbox"][2] for g in block)
        block_y1 = max(g["bbox"][3] for g in block)
        block_bbox = [block_x0, block[0]["bbox"][1], block_x1, block_y1]

        # ── FIX: Kiểm tra same-row neighbor (multi-column table) ──
        row_limit = _has_same_row_neighbor(block, groups, y_tol=4)
        is_table_row = row_limit is not None

        if is_table_row:
            # Bảng: không expand, giữ nguyên y1 của block
            block_y1_max = block_y1
            nb_type = "table_row"
        else:
            # Không phải bảng: tìm neighbor bên dưới như cũ
            best_y  = ph - 10
            nb_type = "page_end"

            for g in groups:
                if any(g is b for b in block):
                    continue
                gy0 = g["bbox"][1]
                if gy0 <= block_y1 + 1:
                    continue
                if not _h_overlap(block_bbox, g["bbox"], tol=10):
                    continue
                candidate = gy0 - 2
                if candidate < best_y:
                    best_y  = candidate
                    nb_type = "text"

            for obs in obstacles:
                oy0 = obs[1]
                if oy0 <= block_y1 + 1:
                    continue
                if not _h_overlap(block_bbox, obs, tol=10):
                    continue
                if oy0 - 2 < best_y:
                    best_y  = block_y1
                    nb_type = obs[4]

            block_y1_max = max(best_y, block_y1)

        # Bước 4: Gán cho tất cả dòng trong block
        for g in block:
            g["y1_insert_max"] = block_y1_max
            g["neighbor_below"] = nb_type
            g["is_table_row"] = is_table_row   # flag để dùng ở write_translated_pdf

    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# TOC
# ═══════════════════════════════════════════════════════════════════════════════

def _split_toc(text):
    m = re.search(r'(\.{4,})\s*(\d+\s*)?$', text)
    if not m:
        return text, "", ""
    return text[:m.start()].strip(), m.group(1), (m.group(2) or "").strip()

def _rebuild_toc(title, dots, num):
    if not dots:
        return title
    return f"{title} {dots}" + (f" {num}" if num else "")


# ═══════════════════════════════════════════════════════════════════════════════
# INSERT LINE V2 — FIXED
# ═══════════════════════════════════════════════════════════════════════════════

def _insert_line_v2(page, font_path, font_bold_path, group, text):
    x0, y0, x1, y1 = group["bbox"]
    role      = group.get("text_role", "BODY_TEXT")
    bold      = group.get("bold", False)
    color     = group.get("rgb", (0, 0, 0))
    orig_size = group["size"]
    y1_max    = group.get("y1_insert_max", y1)
    is_table  = group.get("is_table_row", False)

    pw     = page.rect.width
    min_fs = CFG["min_fontsize_pt"]
    step   = CFG["shrink_step_pt"]

    # FIX: Nếu là table row → luôn dùng shrink_font_first
    if is_table:
        strategy = "shrink_font_first"
    else:
        strategy = CFG["strategy_by_role"].get(role, "expand_down_then_shrink")

    if font_path:
        fontfile = font_bold_path if bold and font_bold_path else font_path
        fontname = "FBold" if bold else "FReg"
        try:
            page.insert_font(fontname=fontname, fontfile=fontfile)
        except Exception:
            pass
    else:
        fontname = "hebo" if bold else "helv"

    x1_safe = min(x1, pw - 15)

    def try_box(ya, yb, sz):
        return page.insert_textbox(
            fitz.Rect(x0, ya, x1_safe, yb), text,
            fontsize=sz, fontname=fontname, color=color, align=0,
        )

    # Strategy A: shrink font trong bbox gốc (table cell / header / footer)
    if strategy == "shrink_font_first":
        sz = orig_size
        while sz >= min_fs:
            if try_box(y0, y1, sz) >= 0:
                return
            sz -= step
        _truncate(page, fontname, x0, y0, x1_safe, y1, text, min_fs, color)
        return

    # Strategy B: expand xuống đến y1_max, rồi shrink nếu cần
    if strategy == "expand_down_then_shrink":
        if try_box(y0, y1, orig_size) >= 0:
            return
        if y1_max > y1 + 1:
            if try_box(y0, y1_max, orig_size) >= 0:
                return
            sz = orig_size - step
            while sz >= min_fs:
                if try_box(y0, y1_max, sz) >= 0:
                    return
                sz -= step
            _truncate(page, fontname, x0, y0, x1_safe, y1_max, text, min_fs, color)
        else:
            sz = orig_size - step
            while sz >= min_fs:
                if try_box(y0, y1, sz) >= 0:
                    return
                sz -= step
            _truncate(page, fontname, x0, y0, x1_safe, y1, text, min_fs, color)
        return

    # Strategy C: giữ nguyên nếu có thể
    if strategy == "keep_original_if_possible":
        if try_box(y0, y1, orig_size) >= 0:
            return
        for i in range(1, 4):
            sz = orig_size - step * i
            if sz < min_fs:
                break
            if try_box(y0, y1, sz) >= 0:
                return
        page.insert_textbox(
            fitz.Rect(x0, y0, x1_safe, y1), group["text"],
            fontsize=orig_size, fontname=fontname, color=color, align=0,
        )
        return

    # Fallback
    sz = orig_size
    while sz >= min_fs:
        if try_box(y0, y1, sz) >= 0:
            return
        sz -= step


def _truncate(page, fontname, x0, y0, x1, y1, text, fontsize, color):
    s = text
    while len(s) > 3:
        s = s[:-4] + "…"
        if page.insert_textbox(
            fitz.Rect(x0, y0, x1, y1), s,
            fontsize=fontsize, fontname=fontname, color=color, align=0,
        ) >= 0:
            return


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE TRANSLATED PDF V2 — FIXED
# ═══════════════════════════════════════════════════════════════════════════════

def write_translated_pdf_v2(src, dst, all_groups, all_trans, font_path):
    doc            = fitz.open(src)
    font_bold_path = _get_bold_font_path(font_path)
    pad            = CFG["redact_padding"]

    for pi, groups in all_groups.items():
        trans = all_trans.get(pi, [])
        if not groups or not trans:
            continue

        page = doc[pi]

        # Enrich: tính text_role + y1_insert_max + is_table_row
        enrich_groups(groups, page)

        # Redact
        for g in groups:
            r        = fitz.Rect(g["bbox"])
            y1_max   = g.get("y1_insert_max", r.y1)
            is_table = g.get("is_table_row", False)
            strategy = "shrink_font_first" if is_table else \
                       CFG["strategy_by_role"].get(g.get("text_role", "BODY_TEXT"), "expand_down_then_shrink")

            # FIX: Table cell → redact chỉ bbox gốc (không mở rộng xuống)
            if strategy == "expand_down_then_shrink":
                redact_y1 = y1_max
            else:
                redact_y1 = r.y1 + pad["bottom"]

            page.add_redact_annot(fitz.Rect(
                r.x0 - pad["left"],
                r.y0 - pad["top"],
                r.x1 + pad["right"],
                redact_y1,
            ).intersect(page.rect))

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # Insert text đã dịch
        for g, t in zip(groups, trans):
            role = g.get("text_role", "BODY_TEXT")

            if not t or not t.strip():
                final = g["text"]
            elif role == "TOC_ENTRY":
                ot, dots, num = _split_toc(g["text"])
                tt, _, _      = _split_toc(t)
                final = _rebuild_toc(tt.strip() or ot, dots, num)
            elif role == "TECHNICAL_VALUE":
                ratio = CFG["expansion_ratio"].get(g.get("lang_hint", "LATIN"), 1.1)
                final = g["text"] if len(t) > len(g["text"]) * ratio * 1.4 else t.strip()
            else:
                final = t.strip()

            try:
                _insert_line_v2(page, font_path, font_bold_path, g, final)
            except Exception:
                try:
                    _insert_line_v2(page, font_path, font_bold_path, g, g["text"])
                except Exception:
                    pass

    doc.save(dst, garbage=4, deflate=True)
    doc.close()


# ═══════════════════════════════════════════════════════════════════════════════
# UI (giữ nguyên từ v2 gốc)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Dịch PDF v2 — Vi Nguyen", page_icon="⬡", layout="centered")

st.markdown("""<style>
    .stApp{background:#0f1117;color:#e2e8f0}.block-container{max-width:780px;padding-top:2rem}
    h1,h2,h3{color:#e2e8f0!important}p,li,span{color:#e2e8f0}
    div[data-testid="stFileUploader"]{border:1.5px dashed #4a5080!important;border-radius:10px!important;padding:8px!important;background:#1a1d27!important}
    div[data-testid="stFileUploader"] label{color:#e2e8f0!important}
    [data-testid="stFileUploaderDropzone"]{background-color:#242838!important;border-color:#4a5080!important}
    [data-testid="stFileUploaderDropzone"]>div{background-color:#242838!important}
    [data-testid="stFileUploaderDropzone"] span{color:#e2e8f0!important}
    [data-testid="stFileUploaderDropzone"] button{background-color:#2d3149!important;color:#e2e8f0!important;border-color:#4a5080!important}
    [data-testid="stFileUploaderDropzone"] p{color:#94a3b8!important}
    label,.stSelectbox label,.stTextInput label,div[data-testid="stWidgetLabel"] p{color:#e2e8f0!important;font-weight:500}
    .stSelectbox div[data-baseweb="select"]>div{background:#1a1d27!important;color:#e2e8f0!important;border-color:#4a5080!important}
    .stTextInput input{background:#1a1d27!important;color:#e2e8f0!important;border-color:#4a5080!important}
    input::placeholder{color:#64748b!important}
    .stat-box{background:#1a1d27;border-radius:10px;padding:14px 18px;text-align:center;border:1px solid #4a5080}
    .stat-val{font-size:1.4rem;font-weight:bold;color:#e2e8f0}
    .stat-lbl{font-size:0.78rem;color:#94a3b8;margin-top:4px}
    .timer-box{background:#1a1d27;border-radius:10px;padding:12px 18px;text-align:center;border:1px solid #6c63ff;margin-bottom:10px}
    .timer-val{font-size:2rem;font-weight:bold;color:#a78bfa;font-family:'Courier New',monospace}
    .timer-lbl{font-size:0.78rem;color:#94a3b8;margin-top:2px}
    .timer-status{font-size:0.85rem;color:#818cf8;margin-top:6px}
    .log-box{background:#1a1d27;border-radius:8px;padding:14px 18px;font-family:'Courier New',monospace;font-size:0.83rem;max-height:340px;overflow-y:auto;border:1px solid #4a5080;white-space:pre-wrap;color:#c4cde0;line-height:1.6}
    .stButton>button{background-color:#6c63ff!important;color:white!important;border:none!important;border-radius:8px!important;padding:10px 28px!important;font-weight:bold!important;font-size:1rem!important;width:100%!important}
    .stButton>button:hover{background-color:#a78bfa!important}
    .stButton>button:disabled{background-color:#2d3149!important;color:#64748b!important}
    .stDownloadButton>button{background-color:#059669!important;color:white!important;border-radius:8px!important;font-weight:bold!important;font-size:1rem!important;width:100%!important}
    .stDownloadButton>button:hover{background-color:#10b981!important}
    div[data-testid="stProgressBar"]>div{background-color:#1a1d27!important}
    div[data-testid="stProgressBar"]>div>div{background-color:#6c63ff!important}
    hr{border-color:#2d3149!important}
    div[data-testid="stAlert"]{background:#1a1d27!important;border-color:#4a5080!important;color:#e2e8f0!important}
    .login-box{background:#1a1d27;border:1px solid #4a5080;border-radius:14px;padding:2.5rem 2rem;max-width:380px;margin:4rem auto 0 auto;text-align:center}
    .login-title{font-size:1.6rem;font-weight:bold;color:#e2e8f0;margin-bottom:0.3rem}
    .login-sub{font-size:0.85rem;color:#64748b;margin-bottom:1.8rem}
    .v2-badge{display:inline-block;background:#4c1d95;color:#c4b5fd;padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:bold;margin-left:8px}
    .cfg-box{background:#1a1d27;border:1px solid #4a5080;border-radius:8px;padding:10px 14px;font-size:0.8rem;color:#94a3b8;margin-bottom:12px}
</style>""", unsafe_allow_html=True)

if not check_password():
    st.stop()

st.markdown("## ⬡ Dịch PDF sang PDF <span class='v2-badge'>v2 fixed</span>", unsafe_allow_html=True)
st.markdown("<span style='color:#64748b;font-size:0.9rem'>Powered by Gemini Flash — Vi Nguyen</span>", unsafe_allow_html=True)

if _cfg_path:
    st.markdown(f"<div class='cfg-box'>✅ Config: <b>{Path(_cfg_path).name}</b> — min_font: <b>{CFG['min_fontsize_pt']}pt</b></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='cfg-box'>⚠️ Dùng config mặc định (chưa có pdf_analysis_slim.json)</div>", unsafe_allow_html=True)

col_t, col_l = st.columns([5, 1])
with col_l:
    if st.button("🚪 Logout"):
        st.session_state.pop("authenticated", None)
        st.rerun()

st.divider()
uploaded = st.file_uploader("📄 Chọn file PDF cần dịch", type=["pdf"])
col1, col2 = st.columns(2)
with col1: lang    = st.selectbox("🌐 Ngôn ngữ đích", LANGUAGES)
with col2: pages_s = st.text_input("📑 Trang cụ thể (tuỳ chọn)", placeholder="Vd: 1-5,8  •  Để trống = tất cả")
st.divider()
run_btn = st.button("▶  Bắt đầu dịch", disabled=(uploaded is None))

if run_btn and uploaded:
    for k in ["pdf_bytes", "out_name", "summary"]:
        st.session_state.pop(k, None)

    st.markdown("### 📊 Tiến độ")
    timer_ph = st.empty()
    col_pg, col_ln, col_usd, col_vnd = st.columns(4)
    ph_pages = col_pg.empty(); ph_lines = col_ln.empty()
    ph_usd   = col_usd.empty(); ph_vnd  = col_vnd.empty()

    def render_stats(pd, tp, tl, ti, to):
        u = ((ti/1e6)*PRICE_INPUT + (to/1e6)*PRICE_OUTPUT) * 10
        v = u * USD_TO_VND
        ph_pages.markdown(f"<div class='stat-box'><div class='stat-val'>{pd}/{tp}</div><div class='stat-lbl'>Trang</div></div>", unsafe_allow_html=True)
        ph_lines.markdown(f"<div class='stat-box'><div class='stat-val'>{tl:,}</div><div class='stat-lbl'>Dòng</div></div>", unsafe_allow_html=True)
        ph_usd.markdown(f"<div class='stat-box'><div class='stat-val'>${u:.4f}</div><div class='stat-lbl'>USD</div></div>", unsafe_allow_html=True)
        ph_vnd.markdown(f"<div class='stat-box'><div class='stat-val'>{v:,.0f}₫</div><div class='stat-lbl'>VND</div></div>", unsafe_allow_html=True)

    render_stats(0,0,0,0,0)
    progress  = st.progress(0, text="Đang chuẩn bị...")
    st.markdown("### 📋 Nhật ký")
    log_ph    = st.empty()
    log_lines = []

    def add_log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        log_ph.markdown(f"<div class='log-box'>{'<br>'.join(log_lines[-40:])}</div>", unsafe_allow_html=True)

    src_path = dst_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read()); src_path = tmp.name
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        dst_path = tmp_out.name; tmp_out.close()

        add_log(f"📄 {uploaded.name}")
        probe    = fitz.open(src_path)
        total_pg = len(probe); probe.close()
        targets  = parse_page_range(pages_s, total_pg) if pages_s.strip() else list(range(total_pg))
        add_log(f"📄 {total_pg} trang tổng, dịch {len(targets)} trang")

        progress.progress(5, text="Trích xuất text...")
        all_groups, _ = extract_line_groups(src_path, targets)
        total_lines   = sum(len(v) for v in all_groups.values())
        add_log(f"✅ {total_lines} dòng text"); render_stats(0, total_pg, total_lines, 0, 0)

        client = genai.Client(api_key=API_KEY)
        add_log(f"🤖 {MODEL_NAME}")
        all_trans = {}; tok_in = tok_out = 0; t0 = time.time()

        for idx, pi in enumerate(targets):
            groups     = all_groups.get(pi, [])
            page_start = time.time()
            progress.progress(int(10 + (idx/len(targets))*80), text=f"Dịch trang {pi+1}/{total_pg}...")
            add_log(f"📄 Trang {pi+1}/{total_pg}: {len(groups)} dòng...")
            if not groups:
                all_trans[pi] = []; continue
            try:
                trans, in_t, out_t = translate_page(client, groups, lang, pi, timer_ph, t0, page_start, log_lines, log_ph)
                tok_in += in_t; tok_out += out_t
                add_log(f"   ✅ {len(trans)} dòng — {time.time()-page_start:.1f}s")
            except Exception as e:
                add_log(f"   ❌ {e}"); trans = [g["text"] for g in groups]
            all_trans[pi] = trans
            render_stats(idx+1, total_pg, total_lines, tok_in, tok_out)
            time.sleep(DELAY_SEC)

        progress.progress(92, text="Tạo PDF...")
        add_log("💾 Đang tạo PDF (v2-fixed: table-aware)...")
        font_path = find_font()
        add_log(f"🔤 Font: {os.path.basename(font_path) if font_path else 'built-in'}")
        write_translated_pdf_v2(src_path, dst_path, all_groups, all_trans, font_path)

        elapsed = time.time() - t0
        usd = ((tok_in/1e6)*PRICE_INPUT + (tok_out/1e6)*PRICE_OUTPUT) * 10
        vnd = usd * USD_TO_VND
        render_stats(len(targets), total_pg, total_lines, tok_in, tok_out)
        progress.progress(100, text="✅ Hoàn thành!")
        timer_ph.markdown(
            f"<div class='timer-box' style='border-color:#059669'>"
            f"<div class='timer-val' style='color:#10b981'>✅ {elapsed:.1f}s</div>"
            f"<div class='timer-lbl'>Tổng thời gian</div>"
            f"<div class='timer-status' style='color:#10b981'>Xong {len(targets)} trang!</div></div>",
            unsafe_allow_html=True,
        )
        add_log(f"🎉 {len(targets)} trang / {elapsed:.1f}s / ${usd:.4f} ≈ {vnd:,.0f}₫")
        with open(dst_path, "rb") as f:
            st.session_state["pdf_bytes"] = f.read()
        st.session_state["out_name"] = uploaded.name.replace(".pdf", f"_v2fixed_{lang[:2]}.pdf")
        st.session_state["summary"]  = f"✅ {len(targets)} trang — {elapsed:.1f}s — ${usd:.4f} ≈ {vnd:,.0f}₫"

    except Exception as e:
        add_log(f"❌ {e}"); st.error(f"❌ {e}")
    finally:
        for p in [src_path, dst_path]:
            try:
                if p: os.unlink(p)
            except Exception:
                pass

if "pdf_bytes" in st.session_state:
    st.divider()
    st.success(st.session_state["summary"])
    st.download_button("⬇️  Tải PDF đã dịch (v2-fixed)", data=st.session_state["pdf_bytes"],
        file_name=st.session_state["out_name"], mime="application/pdf", use_container_width=True)
elif not uploaded:
    st.info("👆 Vui lòng upload file PDF để bắt đầu")

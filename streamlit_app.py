"""
PDF Translator — Gemini Flash 3  [Streamlit Web App]
• Upload PDF → Dịch → Download PDF đã dịch
• Dịch theo DÒNG: liền mạch, không khoảng cách nhân tạo
• Thống kê token + chi phí USD/VND (×10)
"""

import os, time, json, tempfile
import streamlit as st
import fitz
from google import genai
from google.genai import types as gtypes
from datetime import datetime

# ── Hằng số ───────────────────────────────────────────────────────────────────
API_KEY      = st.secrets.get("GEMINI_API_KEY", "")
MODEL_NAME   = "gemini-3-flash-preview"
PRICE_INPUT  = 0.10   # USD per 1M input tokens
PRICE_OUTPUT = 0.40   # USD per 1M output tokens
USD_TO_VND   = 25400
DELAY_SEC    = 0.3
TIMEOUT_WARN = 4
TICKER_INT   = 0.4

UNICODE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/unifont/unifont.otf",
    "C:/Windows/Fonts/Arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
LANGUAGES = ["Tiếng Việt", "Tiếng Anh", "Tiếng Nhật", "Tiếng Trung", "Tiếng Pháp", "Tiếng Đức"]


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND (giữ nguyên logic, không đổi)
# ═══════════════════════════════════════════════════════════════════════════════

def find_font():
    for p in UNICODE_FONTS:
        if os.path.isfile(p): return p
    return None

def int_to_rgb(c):
    return ((c >> 16) & 0xFF) / 255, ((c >> 8) & 0xFF) / 255, (c & 0xFF) / 255

def extract_line_groups(pdf_path, page_nums=None):
    result = {}
    doc = fitz.open(pdf_path)
    total = len(doc)
    targets = page_nums if page_nums else list(range(total))
    for pi in targets:
        if not (0 <= pi < total): continue
        page = doc[pi]
        data = page.get_text("dict",
                  flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)
        groups = []
        for block in data["blocks"]:
            if block.get("type") != 0: continue
            for line in block["lines"]:
                spans = [sp for sp in line["spans"] if sp["text"].strip()]
                if not spans: continue
                merged = "".join(sp["text"] for sp in spans).strip()
                if not merged: continue
                x0 = min(sp["bbox"][0] for sp in spans)
                y0 = min(sp["bbox"][1] for sp in spans)
                x1 = max(sp["bbox"][2] for sp in spans)
                y1 = max(sp["bbox"][3] for sp in spans)
                groups.append({
                    "bbox": (x0, y0, x1, y1),
                    "text": merged,
                    "size": max(sp["size"] for sp in spans),
                    "rgb":  int_to_rgb(spans[0]["color"]),
                    "bold": any(bool(sp["flags"] & (1 << 4)) for sp in spans),
                })
        result[pi] = groups
    doc.close()
    return result, total

def _get_bold_font_path(font_path):
    if not font_path: return None
    for regular, bold in [
        ("Arial.ttf",      "Arialbd.ttf"),
        ("calibri.ttf",    "calibrib.ttf"),
        ("times.ttf",      "timesbd.ttf"),
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ]:
        candidate = font_path.replace(regular, bold)
        if os.path.isfile(candidate): return candidate
    return None

def _insert_line(page, font_path, font_bold_path, bbox, text, fontsize, color, bold):
    x0, y0, x1, y1 = bbox
    pw = page.rect.width
    ph = page.rect.height
    x1_use = pw - 30
    line_h  = max(y1 - y0, fontsize * 1.5)
    y1_use  = min(y0 + line_h * 3, ph - 10)
    if font_path:
        fontfile = font_bold_path if bold and font_bold_path else font_path
        fontname = "FBold" if bold else "FReg"
        try: page.insert_font(fontname=fontname, fontfile=fontfile)
        except: pass
    else:
        fontname = "hebo" if bold else "helv"
    size = max(fontsize, 5.0)
    while size >= 4.0:
        rc = page.insert_textbox(
            fitz.Rect(x0, y0, x1_use, y1_use),
            text, fontsize=size, fontname=fontname, color=color, align=0,
        )
        if rc >= 0: break
        size -= 0.5

def write_translated_pdf(src, dst, all_groups, all_trans, font_path):
    doc = fitz.open(src)
    font_bold_path = _get_bold_font_path(font_path)
    for pi, groups in all_groups.items():
        trans = all_trans.get(pi, [])
        if not groups or not trans: continue
        page = doc[pi]
        for g in groups:
            r = fitz.Rect(g["bbox"])
            expanded = fitz.Rect(r.x0 - 10, r.y0 - 5, r.x1 + 10, r.y1 + 5)
            page.add_redact_annot(expanded.intersect(page.rect))
        page.apply_redactions(images=0)
        for g, t in zip(groups, trans):
            text = t.strip() if t and t.strip() else g["text"]
            try:
                _insert_line(page, font_path, font_bold_path,
                             g["bbox"], text, g["size"], g["rgb"], g["bold"])
            except Exception:
                pass
    doc.save(dst, garbage=4, deflate=True)
    doc.close()

def _parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    raw = raw.strip()
    try: return json.loads(raw)
    except: pass
    for end_token in ["},\n", "}, \n", "},"]:
        last = raw.rfind(end_token)
        if last > 0:
            try: return json.loads(raw[:last + 1] + "]")
            except: pass
    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e > s:
        try: return json.loads(raw[s:e + 1])
        except: pass
    return None

def call_gemini(client, contents, max_tokens=65536, temperature=0.1):
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=gtypes.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    meta  = getattr(resp, "usage_metadata", None)
    in_t  = getattr(meta, "prompt_token_count",     0) or 0
    out_t = getattr(meta, "candidates_token_count", 0) or 0
    return resp.text.strip(), in_t, out_t

def translate_page(client, groups, target_lang, page_idx):
    numbered = "\n".join(f"[{i}] {g['text']}" for i, g in enumerate(groups))
    prompt = (
        f"Dịch sang {target_lang}. Giữ nguyên số thứ tự [0]...[{len(groups) - 1}].\n"
        f"Trả về JSON object, ĐẦY ĐỦ từ \"0\" đến \"{len(groups) - 1}\", không bỏ sót:\n"
        f"{{\"0\": \"bản dịch\", \"1\": \"bản dịch\", ...}}\n"
        f"Chỉ JSON, không giải thích.\n\n"
        f"{numbered}"
    )
    raw, in_t, out_t = call_gemini(client, prompt)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict):
        return [str(parsed.get(str(i), groups[i]["text"])) for i in range(len(groups))], in_t, out_t
    if isinstance(parsed, list) and len(parsed) == len(groups):
        return [str(x) for x in parsed], in_t, out_t
    return [g["text"] for g in groups], in_t, out_t

def parse_page_range(s, total):
    pages = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a) - 1, int(b)))
        elif part.isdigit():
            pages.append(int(part) - 1)
    return sorted(p for p in set(pages) if 0 <= p < total)


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dịch PDF — Vi Nguyen",
    page_icon="⬡",
    layout="centered",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .block-container { max-width: 780px; padding-top: 2rem; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li, span { color: #e2e8f0; }

    div[data-testid="stFileUploader"] { border: 1.5px dashed #4a5080; border-radius: 10px; padding: 8px; background: #1a1d27; }
    div[data-testid="stFileUploader"] * { color: #e2e8f0 !important; }
    div[data-testid="stFileUploader"] small { color: #94a3b8 !important; }

    label, .stSelectbox label, .stTextInput label,
    div[data-testid="stWidgetLabel"] p { color: #e2e8f0 !important; font-weight: 500; }

    .stSelectbox div[data-baseweb="select"] > div { background: #1a1d27 !important; color: #e2e8f0 !important; border-color: #4a5080 !important; }
    .stTextInput input { background: #1a1d27 !important; color: #e2e8f0 !important; border-color: #4a5080 !important; }
    input::placeholder { color: #64748b !important; }

    .stat-box { background: #1a1d27; border-radius: 10px; padding: 14px 18px; text-align: center; border: 1px solid #4a5080; }
    .stat-val { font-size: 1.4rem; font-weight: bold; color: #e2e8f0; }
    .stat-lbl { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }

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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## ⬡ Dịch PDF sang PDF")
st.markdown("<span style='color:#64748b;font-size:0.9rem'>Powered by Gemini Flash — Vi Nguyen</span>", unsafe_allow_html=True)
st.divider()

# ── Upload + options ──────────────────────────────────────────────────────────
uploaded = st.file_uploader("📄 Chọn file PDF cần dịch", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    lang = st.selectbox("🌐 Ngôn ngữ đích", LANGUAGES)
with col2:
    pages_s = st.text_input("📑 Trang cụ thể (tuỳ chọn)", placeholder="Vd: 1-5,8  •  Để trống = tất cả")

st.divider()

# ── Nút dịch ─────────────────────────────────────────────────────────────────
run_btn = st.button("▶  Bắt đầu dịch", disabled=(uploaded is None))

# ── Khu vực kết quả ──────────────────────────────────────────────────────────
if run_btn and uploaded:

    # Stats placeholders
    st.markdown("### 📊 Tiến độ")
    col_pg, col_ln, col_usd, col_vnd = st.columns(4)
    ph_pages = col_pg.empty()
    ph_lines = col_ln.empty()
    ph_usd   = col_usd.empty()
    ph_vnd   = col_vnd.empty()

    def render_stats(pages_done, total_pg, total_lines, tok_in, tok_out):
        usd = ((tok_in / 1e6) * PRICE_INPUT + (tok_out / 1e6) * PRICE_OUTPUT) * 10
        vnd = usd * USD_TO_VND
        ph_pages.markdown(f"""<div class='stat-box'><div class='stat-val'>{pages_done}/{total_pg}</div><div class='stat-lbl'>Trang</div></div>""", unsafe_allow_html=True)
        ph_lines.markdown(f"""<div class='stat-box'><div class='stat-val'>{total_lines:,}</div><div class='stat-lbl'>Dòng text</div></div>""", unsafe_allow_html=True)
        ph_usd.markdown(f"""<div class='stat-box'><div class='stat-val'>${usd:.4f}</div><div class='stat-lbl'>USD</div></div>""", unsafe_allow_html=True)
        ph_vnd.markdown(f"""<div class='stat-box'><div class='stat-val'>{vnd:,.0f}₫</div><div class='stat-lbl'>VND</div></div>""", unsafe_allow_html=True)

    render_stats(0, 0, 0, 0, 0)

    # Progress bar
    progress = st.progress(0, text="Đang chuẩn bị...")

    # Log
    st.markdown("### 📋 Nhật ký hoạt động")
    log_ph  = st.empty()
    log_lines = []

    def add_log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        log_ph.markdown(
            f"<div class='log-box'>{'<br>'.join(log_lines[-40:])}</div>",
            unsafe_allow_html=True
        )

    # ── Chạy dịch ────────────────────────────────────────────────────────────
    try:
        # 1. Lưu upload vào temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
            tmp_in.write(uploaded.read())
            src_path = tmp_in.name

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        dst_path = tmp_out.name
        tmp_out.close()

        add_log(f"📄 Đã nhận file: {uploaded.name}")

        # 2. Đọc tổng số trang
        probe = fitz.open(src_path)
        total_pg = len(probe)
        probe.close()

        targets = (parse_page_range(pages_s, total_pg) if pages_s.strip()
                   else list(range(total_pg)))
        add_log(f"📄 {total_pg} trang tổng, sẽ dịch {len(targets)} trang")

        # 3. Extract line groups
        progress.progress(5, text="Trích xuất text...")
        add_log("🔍 Trích xuất text từ PDF...")
        all_groups, _ = extract_line_groups(src_path, targets)
        total_lines = sum(len(v) for v in all_groups.values())
        add_log(f"✅ {total_lines} dòng text")
        render_stats(0, total_pg, total_lines, 0, 0)

        # 4. Kết nối Gemini
        client = genai.Client(api_key=API_KEY)
        add_log(f"🤖 Kết nối {MODEL_NAME} thành công")

        # 5. Dịch từng trang
        all_trans = {}
        tok_in = tok_out = 0
        t0 = time.time()

        for idx, pi in enumerate(targets):
            groups = all_groups.get(pi, [])
            add_log(f"📄 Trang {pi + 1}/{total_pg}: {len(groups)} dòng")

            pct = int(10 + (idx / len(targets)) * 80)
            progress.progress(pct, text=f"Dịch trang {pi + 1}/{total_pg}...")

            if not groups:
                all_trans[pi] = []
                continue

            try:
                trans, in_t, out_t = translate_page(client, groups, lang, pi)
                tok_in  += in_t
                tok_out += out_t
                add_log(f"   ✅ {len(trans)} dòng ({in_t:,} in / {out_t:,} out tok)")
            except Exception as e:
                add_log(f"   ❌ Lỗi dịch trang {pi + 1}: {e}")
                trans = [g["text"] for g in groups]

            all_trans[pi] = trans
            render_stats(idx + 1, total_pg, total_lines, tok_in, tok_out)
            time.sleep(DELAY_SEC)

        # 6. Ghi PDF
        progress.progress(92, text="Tạo PDF...")
        add_log("💾 Đang tạo PDF...")
        font_path = find_font()
        write_translated_pdf(src_path, dst_path, all_groups, all_trans, font_path)

        # 7. Tổng kết
        elapsed = time.time() - t0
        usd = ((tok_in / 1e6) * PRICE_INPUT + (tok_out / 1e6) * PRICE_OUTPUT) * 10
        vnd = usd * USD_TO_VND
        render_stats(len(targets), total_pg, total_lines, tok_in, tok_out)
        progress.progress(100, text="✅ Hoàn thành!")

        add_log("─" * 44)
        add_log(f"🎉 Xong {len(targets)} trang trong {elapsed:.1f}s")
        add_log(f"💰 Token: {tok_in:,} in + {tok_out:,} out")
        add_log(f"💵 Chi phí: ${usd:.4f} USD ≈ {vnd:,.0f} VND")

        # 8. Download button
        st.divider()
        st.success(f"✅ Dịch xong {len(targets)} trang trong {elapsed:.1f}s  |  ${usd:.4f} USD ≈ {vnd:,.0f} VND")

        with open(dst_path, "rb") as f:
            pdf_bytes = f.read()

        out_name = uploaded.name.replace(".pdf", f"_translated_{lang[:2]}.pdf")
        st.download_button(
            label="⬇️  Tải PDF đã dịch",
            data=pdf_bytes,
            file_name=out_name,
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as e:
        add_log(f"❌ Lỗi: {e}")
        st.error(f"❌ Có lỗi xảy ra: {e}")

    finally:
        # Dọn temp files
        for p in [src_path, dst_path]:
            try: os.unlink(p)
            except: pass

elif not uploaded:
    st.info("👆 Vui lòng upload file PDF để bắt đầu")

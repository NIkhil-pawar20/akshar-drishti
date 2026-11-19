# Akshar Drishti web application (backend)
# ---------------------------------------------------------------
# Run inside the Colab notebook AFTER cells 3, 4 (+ the fp16 line), S1, S2, A2, A3 and A4:
#     %run -i /content/akshar_web.py
# It serves akshar_web.html as a normal website and exposes a small JSON API that reuses
# process_document(), archive_search() and the SQLite archive from the notebook.
# The site gets its own public link (Gradio share tunnel), so it opens in any browser tab.

import os, re, json, uuid, threading, html as _html

import gradio as gr
import torch
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from starlette.routing import Route

_needed = ["parser", "process_document", "archive_search", "con"]
_missing = [n for n in _needed if n not in globals()]
if _missing:
    raise RuntimeError("Run the notebook cells 3, 4, S1, S2, A2, A3 and A4 first. Missing: " + ", ".join(_missing))

WEB_DIR = "/content"
WEB_HTML_PATH = f"{WEB_DIR}/akshar_web.html"
OUT_ROOT = "/content/app_output"
UP_DIR = "/content/web_uploads"
MAX_MB = 25
os.makedirs(UP_DIR, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)
GPU_LOCK = threading.Lock()


# >>> MASK: improved image masking (replaces the notebook's sensitive_boxes) ---------------
# Old method: line = block height / number of text lines; x = character position / line length
# across the whole block. It failed on (1) Marathi vowel signs and halant, which count as
# characters but take no width, (2) short lines inside wide blocks, (3) table HTML whose
# formatting newlines made the line count wrong.
# New method: count only real table rows; find the real text rows and table cell borders in
# the page image; estimate x by visual width (marks weigh 0) inside the line's own ink span;
# cover to the end of the line when the number is the last thing on it; snap to table cells.
import unicodedata as _ud
import numpy as _np
from PIL import Image as _PILImage


_CELL = "␟"   # cell separator kept inside table lines


def _mask_lines(text):
    t = (text or "").replace("▁", " ")
    if "<" in t and ("<tr" in t or "<td" in t):
        t = re.sub(r"\s+", " ", t)                 # HTML formatting newlines are not lines
        t = re.sub(r"<br\s*/?>", " ", t)
        t = re.sub(r"</t[dh]>", " " + _CELL + " ", t)   # remember where each cell ends
        t = re.sub(r"</tr>", "\n", t)
        t = re.sub(r"<[^>]+>", "", t)
    t = _html.unescape(t).replace("$", "")
    return [ln for ln in t.split("\n") if ln.strip(" " + _CELL)]


def _vis(s):
    """Rough visual width: combining marks (matras below/above, halant, anusvara) take ~0 width."""
    w = 0.0
    for ch in s:
        cat = _ud.category(ch)
        w += 0.0 if cat in ("Mn", "Me", "Cf") else (0.6 if ch.isspace() else 1.0)
    return w


def _runs(flags, min_gap=1, min_len=1):
    runs, start, last, gap = [], None, None, 0
    for i, v in enumerate(flags):
        if v:
            if start is None:
                start = i
            last, gap = i, 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                runs.append((start, last)); start, gap = None, 0
    if start is not None:
        runs.append((start, last))
    return [r for r in runs if r[1] - r[0] + 1 >= min_len]


def _page_image_path(page):
    """IndicOCR stores only the file name; find the actual file on disk."""
    p = page.get("image") or ""
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    base = os.path.basename(p)
    cands = [p]
    if base.endswith("_input.png"):
        name = base[: -len("_input.png")]
        cands.append(os.path.join(OUT_ROOT, name, base))
    cands += [os.path.join(UP_DIR, base), os.path.join("/content", base)]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def sensitive_boxes(page):
    """Image boxes to paint black over Aadhaar / PAN numbers (improved estimate)."""
    try:
        gray = _np.asarray(_PILImage.open(_page_image_path(page)).convert("L"))
    except Exception:
        gray = None
    boxes = []
    for b in page.get("blocks", []):
        lines = _mask_lines(b.get("text"))
        hits = [(i, s, e) for i, ln in enumerate(lines) for _, s, e in find_spans(ln, SENSITIVE)]
        if not hits:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in b["bbox_xyxy"]]
        ink = None
        if gray is not None:
            H, W = gray.shape
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 > 5 and y1 - y0 > 5:
                ink = gray[y0:y1, x0:x1] < 160
        bw, bh = x1 - x0, y1 - y0
        rows, vlines = [], []
        if ink is not None:
            # table column borders: columns inked over most of the block height
            vl = _runs(ink.sum(0) >= 0.8 * bh, 1, 1) if bh >= 60 else []
            ink = ink.copy()
            for a, c in vl:
                ink[:, max(0, a - 2):c + 3] = False
            rowsum = ink.sum(1)
            rule = rowsum > 0.85 * bw                      # table border rows
            rows = _runs((rowsum > max(2, 0.004 * bw)) & ~rule, min_gap=3, min_len=6)
            if len(rows) >= 2:
                vlines = [(a + c) // 2 for a, c in vl]
        total = sum(_vis(l) for l in lines) or 1.0
        for i, s, e in hits:
            line = lines[i]
            # 1. vertical band: the matching ink row
            if rows and len(rows) == len(lines):
                ry0, ry1 = rows[i]
            elif rows:
                before = sum(_vis(l) for l in lines[:i]) + _vis(line[:s])
                ry0, ry1 = rows[min(len(rows) - 1, int(len(rows) * before / total))]
            else:
                lh = bh / max(len(lines), 1)
                ry0, ry1 = int(i * lh), int((i + 1) * lh)
            # 2. horizontal: inside this row's own ink span, ignoring table borders
            L, R = 0, bw
            if ink is not None:
                band = ink[max(0, ry0):ry1 + 1].copy()
                for v in vlines:
                    band[:, max(0, v - 3):v + 4] = False
                cols = _np.where(band.sum(0) > 0)[0]
                if cols.size:
                    L, R = int(cols[0]), int(cols[-1])
            n = _vis(line) or 1.0
            fs, fe = _vis(line[:s]) / n, _vis(line[:e]) / n
            span = max(R - L, 1)
            sx0 = L + fs * span - 0.08 * span
            tail = line[e:].strip(" .,;:)|/-")
            sx1 = R + 6 if not tail else L + fe * span + 0.08 * span
            # 3. tables: cover the exact cell the number is in
            n_cells = line.count(_CELL)
            if len(vlines) >= 2 and n_cells == len(vlines) - 1:
                k = line[:s].count(_CELL)
                sx0, sx1 = vlines[k] + 4, vlines[k + 1] - 3
            elif vlines:
                edges = [0] + vlines + [bw]
                cx = L + (fs + fe) / 2 * span
                cells = [(a, c) for a, c in zip(edges, edges[1:]) if c - a > 8 and
                         (a <= cx <= c or (min(c, sx1) - max(a, sx0)) >= 0.6 * (c - a))]
                if cells:
                    sx0, sx1 = min(a for a, _ in cells) + 3, max(c for _, c in cells) - 3
            boxes.append([x0 + max(0, sx0), y0 + max(0, ry0 - 4),
                          x0 + min(bw, sx1), y0 + min(bh, ry1 + 4)])
    return boxes


globals()["sensitive_boxes"] = sensitive_boxes   # process_document now uses this version
# <<< MASK ------------------------------------------------------------------------------------


# >>> FIELDS: better key-field extraction (replaces the notebook's extract_fields) ------------
# 1. duplicates: values are compared after normalising ("411001" == "411 001", "+91 98765 43210"
#    == "9876543210", "15/06/2024" == "15.06.2024").
# 2. new formats: dates with month names (Marathi, Hindi, English) and landlines with STD code.
# 3. context: the label written just before each value ("आधार क्रमांक / Aadhaar No.", "DOB").
# Same return shape as before ([[field, value], ...]) so process_document is unchanged; the
# contexts are kept in LAST_FIELD_CONTEXTS (same order) and attached after processing.
_MR_M = ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून", "जुलै", "ऑगस्ट", "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"]
_HI_M = [["जनवरी"], ["फरवरी", "फ़रवरी"], ["मार्च"], ["अप्रैल"], ["मई"], ["जून"], ["जुलाई"], ["अगस्त"],
         ["सितंबर", "सितम्बर"], ["अक्टूबर", "अक्तूबर"], ["नवंबर", "नवम्बर"], ["दिसंबर", "दिसम्बर"]]
_EN_M = [r"jan(?:uary)?", r"feb(?:ruary)?", r"mar(?:ch)?", r"apr(?:il)?", r"may", r"june?", r"july?",
         r"aug(?:ust)?", r"sep(?:t(?:ember)?)?", r"oct(?:ober)?", r"nov(?:ember)?", r"dec(?:ember)?"]
_MONTH_ALT = "|".join(sorted(set(_MR_M + [w for ws in _HI_M for w in ws]), key=len, reverse=True) + _EN_M)
_NOT_LETTER = r"(?<![A-Za-zऀ-ॿ])"
_EXTRA_PATTERNS = [
    ("Date", r"(?<!\d)\d{1,2}(?:st|nd|rd|th)?\s*[-.,/]?\s*" + _NOT_LETTER + r"(?:" + _MONTH_ALT + r")\.?\s*[-.,/]?\s*\d{4}(?!\d)", re.I),
    ("Date", _NOT_LETTER + r"(?:" + _MONTH_ALT + r")\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}(?!\d)", re.I),
    ("Phone", r"(?<![\d+])\(?0\d{2,4}\)?[\s-]?\d{2,4}[\s-]?\d{3,4}(?!\d)", 0),   # landline, 11 digits
]
LAST_FIELD_CONTEXTS = []


def _month_no(t):
    low = t.lower()
    for i, w in enumerate(_MR_M):
        if w in t:
            return i + 1
    for i, ws in enumerate(_HI_M):
        if any(w in t for w in ws):
            return i + 1
    for i, p in enumerate(_EN_M):
        if re.search(r"(?<![a-z])" + p + r"(?![a-z])", low):
            return i + 1
    return 0


def _field_key(kind, v):
    t = v.translate(DEV_DIGITS)
    d = re.sub(r"\D", "", t)
    if kind in ("Aadhaar", "PIN code"):
        return d
    if kind == "Phone":
        if len(d) == 12 and d.startswith("91"):
            d = d[2:]
        if len(d) == 11 and d.startswith("0") and d[1] in "6789":
            d = d[1:]
        return d
    if kind == "PAN":
        return re.sub(r"\s", "", t).upper()
    if kind == "Email":
        return t.strip().lower()
    if kind == "Survey / Gat no.":
        return re.sub(r"[^\d/]", "", t)
    if kind == "Amount":
        m = re.search(r"\d[\d,]*(?:\.\d+)?", t)
        return m.group(0).replace(",", "") if m else t
    if kind == "Date":
        nums = [int(x) for x in re.findall(r"\d+", t)]
        mo = _month_no(t)
        if mo and len(nums) >= 2:
            day, yr = nums[0], nums[-1]
        elif len(nums) >= 3:
            day, mo, yr = nums[0], nums[1], nums[2]
        else:
            return " ".join(t.split())
        if yr < 100:
            yr += 2000
        return (day, mo, yr)
    return " ".join(t.split()).lower()


def _field_context(t, s, floor=0):
    """Label written just before the value: same line (after any earlier value), else the end of the previous line."""
    ls = t.rfind("\n", 0, s) + 1
    start = max(ls, floor)
    pre = re.split(r"[।;|]", t[start:s])[-1].strip(" \t:–—-=,(")
    if not pre.strip() and ls > 0 and start == ls:
        prev_start = t.rfind("\n", 0, ls - 1) + 1
        pre = t[prev_start:ls - 1].rstrip(" \t:–—-=,(")
    ctx = " ".join(pre.split()[-6:]).strip(" :")
    if len(ctx) > 60:
        ctx = "…" + ctx[-59:]
    try:
        ctx = mask_text(ctx)          # never leak another ID number through a label
    except Exception:
        pass
    return ctx


def extract_fields(text):
    global LAST_FIELD_CONTEXTS
    t = (text or "").translate(DEV_DIGITS)
    order = {k: i for i, k in enumerate(FIELD_PATTERNS)}
    found = []
    for kind, (pat, flags) in FIELD_PATTERNS.items():
        found += [(order[kind], m.start(), m.end(), kind) for m in re.finditer(pat, t, flags)]
    for kind, pat, flags in _EXTRA_PATTERNS:
        for m in re.finditer(pat, t, flags):
            if kind == "Phone" and len(re.sub(r"\D", "", m.group(0))) != 11:
                continue
            found.append((order.get(kind, 99), m.start(), m.end(), kind))
    rows, ctxs, seen, spans = [], [], set(), []
    for _, s, e, kind in sorted(found):
        if any(s < e2 and s2 < e and k2 == kind for s2, e2, k2 in spans):   # same field found twice
            continue
        spans.append((s, e, kind))
        v = " ".join(t[s:e].split())
        key = (kind, _field_key(kind, v))
        if key in seen:
            continue
        seen.add(key)
        rows.append([kind, v])
        floor = max([e2 for _, s2, e2, _k in found if e2 <= s] + [0])   # end of the previous value
        ctxs.append(_field_context(t, s, floor))
    LAST_FIELD_CONTEXTS = ctxs
    return rows


def _attach_contexts(name):
    """Add the context labels to the saved JSON and the Excel 'Fields' sheet. Never fails the request."""
    try:
        out = os.path.join(OUT_ROOT, name)
        jp = os.path.join(out, f"{name}.json")
        page = json.load(open(jp, encoding="utf-8"))
        F, C = page.get("fields", []), list(LAST_FIELD_CONTEXTS)
        if len(F) != len(C):
            return
        for f, c in zip(F, C):
            f["context"] = c
        json.dump(page, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        xp = os.path.join(out, f"{name}.xlsx")
        if os.path.isfile(xp):
            from openpyxl import load_workbook
            wb = load_workbook(xp)
            if "Fields" in wb.sheetnames:
                ws = wb["Fields"]
                ws.delete_rows(1, ws.max_row)
                ws.append(["Field", "Value", "Context"])
                for f in F:
                    ws.append([f.get("field"), f.get("value"), f.get("context", "")])
                wb.save(xp)
    except Exception as e:
        print("context step skipped:", e)


globals()["extract_fields"] = extract_fields   # process_document now uses this version
# <<< FIELDS -----------------------------------------------------------------------------------


def _verdict_text(sigs, stamps):
    return (f"Signature: {'✔ found (' + str(len(sigs)) + ')' if sigs else '✘ not found'}\n"
            f"Stamp/seal: {'✔ found (' + str(len(stamps)) + ')' if stamps else '✘ not found'}\n"
            f"Status: {'COMPLETE - signed and stamped' if sigs and stamps else 'INCOMPLETE - check manually'}")


def _doc_payload(name, timing=None):
    """Everything the web page needs about one processed document, read from its output folder."""
    out = os.path.join(OUT_ROOT, name)
    jp = os.path.join(out, f"{name}.json")
    if not os.path.isfile(jp):
        raise HTTPException(status_code=404, detail="This document's results are not on the server any more.")
    page = json.load(open(jp, encoding="utf-8"))

    def url(p, dl=False):
        rel = os.path.relpath(p, out).replace(os.sep, "/")
        return f"/files/{name}/{rel}" + ("?dl=1" if dl else "")

    marks = page.get("marks", [])
    sigs = [m for m in marks if m.get("type") == "signature"]
    stamps = [m for m in marks if m.get("type") == "stamp"]
    blocks = page.get("blocks", [])
    types = {}
    for b in blocks:
        types[b.get("type", "Text")] = types.get(b.get("type", "Text"), 0) + 1

    downloads = []
    for fn, label, kind in [
        (f"{name}_report.html", "Printable report", "HTML"),
        (f"{name}.xlsx", "Fields, marks and tables", "Excel"),
        (f"{name}.json", "Full page record", "JSON"),
        (f"{name}.md", "Text in reading order", "Markdown"),
        (f"{name}_annotated.png", "Annotated page", "PNG"),
    ]:
        p = os.path.join(out, fn)
        if os.path.isfile(p):
            downloads.append({"label": label, "kind": kind, "name": fn, "url": url(p, dl=True),
                              "size_kb": round(os.path.getsize(p) / 1024, 1)})

    ann = os.path.join(out, f"{name}_annotated.png")
    return {
        "id": name,
        "verdict": _verdict_text(sigs, stamps),
        "signed": bool(sigs), "stamped": bool(stamps),
        "n_signatures": len(sigs), "n_stamps": len(stamps),
        "complete": bool(sigs and stamps),
        "annotated_url": url(ann) if os.path.isfile(ann) else None,
        "markdown": page.get("markdown", ""),
        "fields": page.get("fields", []),
        "marks": [{"type": m.get("type"), "method": m.get("method"), "conf": m.get("conf"),
                   "bbox": m.get("bbox_xyxy"),
                   "crop_url": url(m["crop"]) if m.get("crop") and os.path.isfile(m["crop"]) else None}
                  for m in marks],
        "n_blocks": len(blocks),
        "block_types": types,
        "n_tagged": sum(1 for b in blocks if b.get("inside_mark")),
        "downloads": downloads,
        "timing": timing,
    }


# ---------------- routes ----------------
def page_index():
    try:
        return HTMLResponse(open(WEB_HTML_PATH, encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse("<h1>akshar_web.html is missing</h1><p>Upload it to /content and reload.</p>", status_code=500)


def api_health():
    return {"ok": True, "gpu": torch.cuda.is_available(), "gpu_busy": GPU_LOCK.locked()}


def api_process(file: UploadFile = File(...), mask: str = Form("true")):
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="The server has no GPU right now. In Colab choose Runtime > Change runtime type > T4 GPU, run the setup cells again, then reload this page.")
    fname = file.filename or "page.png"
    stem, ext = os.path.splitext(os.path.basename(fname))
    ext = ext.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        raise HTTPException(status_code=400, detail="Please upload a page image (PNG or JPG).")
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"The image is larger than {MAX_MB} MB.")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:40] or "page"
    name = f"{safe}_{uuid.uuid4().hex[:6]}"
    path = os.path.join(UP_DIR, name + ext)
    with open(path, "wb") as f:
        f.write(data)
    do_mask = str(mask).lower() not in ("false", "0", "no", "off")
    with GPU_LOCK:  # one page at a time on the single GPU
        globals()["LAST_FIELD_CONTEXTS"] = []
        try:
            process_document(path, mask=do_mask)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
        _attach_contexts(name)
        timing = {k: round(v, 2) for k, v in globals().get("LAST_TIMING", {}).items()}
    timing["total"] = round(sum(timing.values()), 2) if timing else None
    payload = _doc_payload(name, timing)
    payload["masked"] = do_mask
    return JSONResponse(payload)


def api_doc(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise HTTPException(status_code=400, detail="Bad document id.")
    return JSONResponse(_doc_payload(name))


def api_search(q: str = ""):
    df = archive_search(q or "")
    rows = []
    for r in df.to_dict("records"):
        rows.append({"id": int(r["ID"]), "name": r["Document"], "added": r["Added"],
                     "signed": r["Signed"] == "✔", "stamped": r["Stamped"] == "✔",
                     "snippet": r["Match"],
                     "has_results": os.path.isfile(os.path.join(OUT_ROOT, r["Document"], f"{r['Document']}.json"))})
    return JSONResponse({"query": q, "count": len(rows), "rows": rows[:200]})


def api_stats():
    n, s, st, both = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(signed),0), COALESCE(SUM(stamped),0), "
        "COALESCE(SUM(CASE WHEN signed=1 AND stamped=1 THEN 1 ELSE 0 END),0) FROM docs").fetchone()
    return {"documents": n, "signed": s, "stamped": st, "complete": both, "incomplete": n - both}


def file_route(doc: str, fpath: str, dl: int = 0):
    base = os.path.realpath(OUT_ROOT)
    p = os.path.realpath(os.path.join(OUT_ROOT, doc, fpath))
    if not p.startswith(base + os.sep) or not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(p, filename=os.path.basename(p) if dl else None)


def _attach(app):
    """Add the site's routes to the running server and put them in front of Gradio's own routes."""
    before = len(app.router.routes)
    app.add_api_route("/", page_index, methods=["GET"], response_class=HTMLResponse, include_in_schema=False)
    app.add_api_route("/api/health", api_health, methods=["GET"])
    app.add_api_route("/api/process", api_process, methods=["POST"])
    app.add_api_route("/api/doc/{name}", api_doc, methods=["GET"])
    app.add_api_route("/api/search", api_search, methods=["GET"])
    app.add_api_route("/api/stats", api_stats, methods=["GET"])
    app.add_api_route("/files/{doc}/{fpath:path}", file_route, methods=["GET"])
    new = app.router.routes[before:]
    del app.router.routes[before:]
    app.router.routes[0:0] = new


# ---------------- launch ----------------
for _old in ("demo", "web_shell"):
    try:
        globals()[_old].close()
    except Exception:
        pass

with gr.Blocks(title="Akshar Drishti") as web_shell:
    gr.Markdown("Akshar Drishti server")

web_shell.launch(share=True, server_port=7860, prevent_thread_lock=True, quiet=True, inline=False)
_attach(web_shell.app)
WEB_URL = getattr(web_shell, "share_url", None)

if not WEB_URL:
    # No public tunnel: serve through Colab's proxy instead (opens in this browser only).
    web_shell.close()
    from google.colab.output import eval_js
    WEB_URL = eval_js("google.colab.kernel.proxyPort(7860)")
    web_shell.launch(share=False, server_port=7860, root_path=WEB_URL.rstrip("/"), prevent_thread_lock=True, quiet=True, inline=False)
    _attach(web_shell.app)

print("=" * 70)
print("Akshar Drishti web app is running. Open this link in Chrome:")
print("   ", WEB_URL)
print("=" * 70)

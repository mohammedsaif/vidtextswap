"""VidTextSwap web demo (Gradio) - for Hugging Face Spaces or local use.

Run locally:   pip install vidtextswap gradio   &&   python app.py
"""
import os
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import cv2
import gradio as gr

from vidtextswap import Rule, __version__
from vidtextswap.api import is_media, process_files, scan_files
from vidtextswap.media import VIDEO_EXT

# Free Hugging Face hardware is a small CPU, so keep demo jobs short.
MAX_FILES = int(os.environ.get("VTS_MAX_FILES", 6))
MAX_VIDEO_SECONDS = int(os.environ.get("VTS_MAX_VIDEO_SECONDS", 60))
MAX_RULES = 10
KEEP_MINUTES = 60
WORK = Path(tempfile.gettempdir()) / "vidtextswap_runs"
HERE = Path(__file__).parent
EXAMPLES = [str(HERE / "examples" / n) for n in ("demo_post.jpg", "demo_reel.mp4")]
REPO = "https://github.com/mohammedsaif/vidtextswap"


def _warm_up():
    """Load the OCR engine and AI repair model in the background, so the first
    user doesn't wait for downloads."""
    try:
        from vidtextswap.api import _finder
        _finder([], 0.85)
        from vidtextswap import inpaint
        inpaint.get_lama(log=lambda *_: None)
    except Exception as e:  # the app still works; the first job loads them
        print("warm-up:", e)


threading.Thread(target=_warm_up, daemon=True).start()


def _cleanup():
    WORK.mkdir(parents=True, exist_ok=True)
    for d in WORK.iterdir():
        if time.time() - d.stat().st_mtime > KEEP_MINUTES * 60:
            shutil.rmtree(d, ignore_errors=True)


def _paths(files):
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]
    paths = [p for p in paths if is_media(p)]
    if not paths:
        raise gr.Error("Upload at least one image (.jpg, .png, .webp) or video (.mp4, .mov, .webm).")
    if len(paths) > MAX_FILES:
        raise gr.Error(f"This free demo takes up to {MAX_FILES} files at a time.")
    for p in paths:
        if Path(p).suffix.lower() in VIDEO_EXT:
            cap = cv2.VideoCapture(p)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            secs = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
            cap.release()
            if secs > MAX_VIDEO_SECONDS:
                raise gr.Error(f"{Path(p).name} is {secs:.0f}s long. The free demo takes videos up to "
                               f"{MAX_VIDEO_SECONDS}s. For longer videos, install the tool (see GitHub).")
    return paths


def scan(files, progress=gr.Progress()):
    paths = _paths(files)
    progress(0, desc="Reading the text in your files...")
    items, texts = scan_files(paths)
    if not texts:
        return "No text found on screen.", [["", ""]]
    md = ["**Found on screen:**"]
    suggestions = []
    for cat, found in items.items():
        md.append(f"- **{cat}:** " + ", ".join(f"`{t}`" for t in list(found)[:6]))
        suggestions += [[t, ""] for t in list(found)[:3]]
    if len(md) == 1:
        md.append("- " + ", ".join(f"`{t}`" for t in list(texts)[:8]))
    md.append("\nThe table on the right now holds what was found. Type the new text in "
              "**Replace with**, and clear any row you don't want to change.")
    return "\n".join(md), (suggestions[:MAX_RULES] or [["", ""]])


def _rules(rows):
    rules, problems = [], []
    for row in (rows.values.tolist() if hasattr(rows, "values") else rows) or []:
        find, replace = (str(c).strip() if c is not None else "" for c in (list(row) + ["", ""])[:2])
        if not find or find.lower() == "nan":
            continue
        if not replace or replace.lower() == "nan":
            problems.append(f"'{find}' has no replacement")
            continue
        try:
            rules.append(Rule(find, replace))
        except ValueError as e:
            problems.append(str(e))
    if problems and not rules:
        raise gr.Error("; ".join(problems))
    if not rules:
        raise gr.Error("Fill in at least one row: the old text in Find, the new text in Replace with.")
    return rules[:MAX_RULES], problems


def run(files, rows, repair, reflow, progress=gr.Progress()):
    paths = _paths(files)
    rules, problems = _rules(rows)
    _cleanup()
    out = Path(tempfile.mkdtemp(dir=WORK))

    def report(i, n, name):
        progress((i, n), desc=f"Working on {name}" if name else "Done", unit="files")

    results = process_files(paths, rules, out / "updated", inpaint="ai" if repair == "AI (best quality)"
                            else "fast" if repair == "Fast" else "auto", reflow=reflow, progress=report)

    gallery, lines = [], []
    for r in results:
        name = Path(r.file).name
        if r.status == "updated":
            gallery.append((r.preview, f"{name}: before | after"))
            lines.append(f"- ✅ **{name}**: {r.matches} change{'s' if r.matches != 1 else ''}")
            lines += [f"    - {d}" for d in r.details]
        elif r.status == "no match":
            lines.append(f"- ⚪ **{name}**: the text wasn't found")
        else:
            lines.append(f"- ⚠️ **{name}**: {r.status}")
    lines += [f"- ⚠️ skipped rule: {p}" for p in problems]

    done = [r for r in results if r.status == "updated"]
    if not done:
        return gallery, "\n".join(lines) or "Nothing found.", gr.DownloadButton(visible=False)
    zip_path = out / "vidtextswap_updated.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r in done:
            z.write(r.output, Path(r.output).name)
    return (gallery, "\n".join(lines),
            gr.DownloadButton(f"Download {len(done)} updated file{'s' if len(done) != 1 else ''} (.zip)",
                              value=str(zip_path), visible=True))


def load_examples():
    return EXAMPLES, [["050 123 4567", "050 765 4321"], ["Al Quoz", "JLT"], ["31 Dec", "15 Jan"]]


INTRO = f"""
# VidTextSwap
**Change text inside your videos and images, such as a phone number, branch name, website or offer date,
and keep the original style.** Upload, say what to change, and download the updated files.

It copies the size, colour, outline and boldness, and uses AI to rebuild the background behind the old text.
Videos keep their sound, and the new text shows only for the seconds the old text did.
[Open source on GitHub]({REPO}) · v{__version__}
"""

HOW = f"""
**How to use**
1. Upload up to {MAX_FILES} images or videos (videos up to {MAX_VIDEO_SECONDS}s), or click **Try the sample files**.
2. Click **Find text** to see the phone numbers, websites and dates on screen. Or just type in the table.
3. Put the new text next to each old text, then click **Replace**.

*Privacy: files are processed on this server and deleted within an hour. Don't upload anything confidential.
For private or long videos, [install the free tool]({REPO}) and run it on your own computer.*
"""

with gr.Blocks(title="VidTextSwap - replace text in videos", fill_width=False) as demo:
    gr.Markdown(INTRO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            files = gr.File(label="Your images / videos", file_count="multiple",
                            file_types=["image", "video"], height=180)
            with gr.Row():
                sample_btn = gr.Button("Try the sample files", variant="secondary", size="sm")
                scan_btn = gr.Button("1. Find text", variant="secondary", size="sm")
            found = gr.Markdown(HOW)
        with gr.Column(scale=1):
            rules = gr.Dataframe(headers=["Find (old text)", "Replace with"], value=[["", ""], ["", ""]],
                                 datatype=["str", "str"], type="array", interactive=True,
                                 row_count=2, row_limits=(1, MAX_RULES), column_count=2,
                                 label="What to change (one per row)")
            with gr.Accordion("Options", open=False):
                repair = gr.Radio(["Auto", "AI (best quality)", "Fast"], value="Auto",
                                  label="Background repair",
                                  info="Auto uses AI on images and still shots, and fast repair while the camera moves.")
                reflow = gr.Checkbox(value=True, label="Move the rest of the line if the new text is shorter or longer")
            go = gr.Button("2. Replace", variant="primary", size="lg")
    gallery = gr.Gallery(label="Before | after", columns=2, height="auto", object_fit="contain")
    with gr.Row():
        summary = gr.Markdown()
    download = gr.DownloadButton("Download", visible=False, variant="primary")

    sample_btn.click(load_examples, outputs=[files, rules])
    scan_btn.click(scan, inputs=files, outputs=[found, rules])
    go.click(run, inputs=[files, rules, repair, reflow], outputs=[gallery, summary, download],
             concurrency_limit=1)   # one job at a time on free hardware

if __name__ == "__main__":
    demo.queue(max_size=20).launch(theme=gr.themes.Soft(), allowed_paths=[str(HERE / "examples")])

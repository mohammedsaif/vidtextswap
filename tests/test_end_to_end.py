"""Real OCR on generated images and video: the output must read the NEW text."""
import csv
import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from vidtextswap.cli import main
from vidtextswap.fonts import candidate_fonts, load_font
from vidtextswap.media import find_ffmpeg

BOLD = candidate_fonts()[0]


@pytest.fixture(scope="session")
def ocr():
    from vidtextswap.finder import default_ocr
    engine = default_ocr()

    def read(path_or_img):
        img = cv2.imread(str(path_or_img)) if not isinstance(path_or_img, np.ndarray) else path_or_img
        result, _ = engine(img)
        return " ".join(t for _, t, _ in result or []).replace(" ", "").lower()
    return read


def textured(w, h, colour, seed=0):
    noise = np.random.default_rng(seed).normal(0, 1, (h, w, 3)) * 16
    img = Image.fromarray(np.clip(np.array(colour) + noise, 0, 255).astype("uint8"))
    return img.filter(ImageFilter.GaussianBlur(2))


def make_poster(path):
    img = textured(1080, 1080, (70, 80, 95))
    d = ImageDraw.Draw(img)
    f = load_font(BOLD, 64)
    for y, t in ((300, "Now open in Al Quoz"), (420, "Call 050 123 4567 today")):
        w = d.textlength(t, font=f)
        d.text(((1080 - w) / 2, y), t, font=f, fill="white", stroke_width=3, stroke_fill=(0, 0, 0))
    img.save(path, quality=95)


def test_image_end_to_end(tmp_path, ocr):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_poster(src / "poster.jpg")
    (src / "caption.json").write_text('{"caption": "Call 050 123 4567"}', encoding="utf-8")

    assert main(["-i", str(src), "-o", str(out), "--find", "050 123 4567", "--replace", "050 765 4321",
                 "--find", "Al Quoz", "--replace", "JLT", "--inpaint", "fast"]) == 0

    text = ocr(out / "poster.jpg")
    assert "0507654321" in text and "jlt" in text
    assert "0501234567" not in text and "quoz" not in text
    assert "nowopenin" in text and "today" in text            # neighbours untouched
    assert (out / "_previews" / "poster.jpg").exists()
    rows = list(csv.DictReader(open(out / "report.csv", encoding="utf-8-sig")))
    assert rows[0]["status"] == "updated" and rows[0]["matches"] == "2"


def test_dry_run_writes_no_media(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_poster(src / "poster.jpg")
    main(["-i", str(src), "-o", str(out), "--find", "Al Quoz", "--replace", "JLT", "--dry-run", "--inpaint", "fast"])
    assert not (out / "poster.jpg").exists()
    rows = list(csv.DictReader(open(out / "report.csv", encoding="utf-8-sig")))
    assert rows[0]["status"] == "found (dry run)"


def test_no_match_leaves_nothing(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_poster(src / "poster.jpg")
    main(["-i", str(src), "-o", str(out), "--find", "Downtown", "--replace", "Marina", "--inpaint", "fast"])
    assert not (out / "poster.jpg").exists()


def test_video_end_to_end(tmp_path, ocr):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    # 3 s clip at 25 fps, moving texture, number on screen from 1 s, with audio.
    # Frames are drawn in Python so the test doesn't need ffmpeg's drawtext filter.
    tex = textured(1400, 1400, (150, 160, 150), seed=3)
    font = load_font(BOLD, 60)
    proc = subprocess.Popen([find_ffmpeg(), "-loglevel", "error", "-y",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "720x1280", "-r", "25", "-i", "-",
                             "-f", "lavfi", "-i", "sine=f=300:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-c:a", "aac", "-shortest", str(src / "reel.mp4")], stdin=subprocess.PIPE)
    for i in range(75):
        t = i / 25
        fr = tex.crop((int(t * 40), int(t * 40), int(t * 40) + 720, int(t * 40) + 1280))
        if t >= 1:
            d = ImageDraw.Draw(fr)
            w = d.textlength("Call 0501234567", font=font)
            d.text(((720 - w) / 2, 500), "Call 0501234567", font=font, fill="white",
                   stroke_width=4, stroke_fill="black")
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0

    main(["-i", str(src), "-o", str(out), "--find", "050 123 4567", "--replace", "050 765 4321",
          "--inpaint", "fast"])

    def frame(path, t):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        cap.release()
        assert ok
        return f

    assert "0507654321" in ocr(frame(out / "reel.mp4", 2.0))
    assert "0501234567" not in ocr(frame(out / "reel.mp4", 2.0))
    before = frame(src / "reel.mp4", 0.5).astype(int)
    after = frame(out / "reel.mp4", 0.5).astype(int)
    assert np.abs(before - after).mean() < 3            # untouched before the number appears
    probe = subprocess.run([find_ffmpeg(), "-i", str(out / "reel.mp4")], capture_output=True, text=True).stderr
    assert "Audio" in probe                             # sound kept


def test_instruct_end_to_end_with_fake_claude(tmp_path, ocr, monkeypatch):
    """--instruct: scan text, 'Claude' proposes rules, the poster is edited."""
    from types import SimpleNamespace

    from vidtextswap import assistant
    seen = {}

    def create(**kw):
        seen["prompt"] = kw["messages"][0]["content"]
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={
            "rules": [{"find": "050 123 4567", "replace": "050 765 4321", "kind": "phone"},
                      {"find": "Al Quoz", "replace": "JLT", "kind": "text"}]})])
    monkeypatch.setattr(assistant, "get_client",
                        lambda *a, **k: assistant.ClaudeBackend(SimpleNamespace(messages=SimpleNamespace(create=create)), "m"))

    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_poster(src / "poster.jpg")
    assert main(["-i", str(src), "-o", str(out), "--instruct", "new number 050 765 4321, we moved to JLT",
                 "--yes", "--inpaint", "fast"]) == 0
    assert "Al Quoz" in seen["prompt"]                       # Claude was shown the on-screen text
    text = ocr(out / "poster.jpg")
    assert "0507654321" in text and "jlt" in text
    assert (out / "rules_used.csv").read_text(encoding="utf-8-sig").splitlines()[1].startswith("050 123 4567")


def test_scan_lists_items(tmp_path, capsys):
    src = tmp_path / "in"
    src.mkdir()
    make_poster(src / "poster.jpg")
    assert main(["-i", str(src), "-o", str(tmp_path / "out"), "--scan"]) == 0
    printed = capsys.readouterr().out
    assert "Phone numbers:" in printed and "050 123 4567" in printed
    assert (tmp_path / "out" / "scan.txt").exists()

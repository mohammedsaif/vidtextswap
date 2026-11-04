"""Images and videos: find matches, plan edits, write results."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import style
from .fonts import candidate_fonts

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found. Run:  pip install imageio-ffmpeg") from e


def read_image(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # non-ASCII safe
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def write_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".webp":
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(path, quality=95)
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
        cv2.imencode(ext, img, params)[1].tofile(str(path))


def save_preview(pairs, path, max_h=900):
    """Before | after, one row per edit (up to 4 rows)."""
    rows = []
    for before, after in pairs[:4]:
        row = np.hstack([before, np.full((before.shape[0], 12, 3), 255, np.uint8), after])
        rows.append(row)
    w = min(r.shape[1] for r in rows)
    rows = [cv2.resize(r, (w, int(r.shape[0] * w / r.shape[1]))) for r in rows]
    sheet = np.vstack([np.vstack([r, np.full((12, w, 3), 255, np.uint8)]) for r in rows])[:-12]
    s = min(1.0, max_h * min(len(rows), 2) / sheet.shape[0])
    if s < 1:
        sheet = cv2.resize(sheet, (int(sheet.shape[1] * s), int(sheet.shape[0] * s)))
    write_image(sheet, path.with_suffix(".jpg"))


def _fonts_for(rule, user_font):
    return candidate_fonts(rule.font or user_font)


def _describe(hit_or_seg, plan, times=None):
    rule = hit_or_seg["rule"]
    t = f" @ {times[0]:.1f}-{times[1]:.1f}s" if times else ""
    return f"{rule.label()}{t} [{plan['desc']}]"


# ----------------------------------------------------------------------------- images

def plan_image(path, finder, opts):
    img = read_image(path)
    hits = finder.find(img)
    if not hits:
        return None
    after = img.copy()
    descs = []
    for h in hits:
        new = h["rule"].replacement_for(h["old_text"], opts.keep_case)
        p = style.analyse(img, h, new, _fonts_for(h["rule"], opts.font), opts.reflow)
        p["inpaint"] = "fast" if opts.inpaint == "fast" else "ai"
        style.apply(after, p)
        descs.append(_describe(h, p))
    return {"kind": "image", "count": len(hits), "descs": descs, "after": after,
            "preview": [(img, after)]}


def write_image_plan(plan, out_path):
    write_image(plan["after"], out_path)


# ----------------------------------------------------------------------------- videos

def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _scan(path, finder, step):
    """Sample frames; group detections of the same rule in the same place."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = max(1, int(round(step * fps)))
    segments, active, idx = [], [], 0
    while cap.grab():
        if idx % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t, still = idx / fps, []
            for h in finder.find(frame):
                for sg in active:
                    if sg["rule"] is h["rule"] and _iou(sg["box"], h["box"]) > 0.3:
                        b = sg["box"]
                        sg["box"] = (min(b[0], h["box"][0]), min(b[1], h["box"][1]),
                                     max(b[2], h["box"][2]), max(b[3], h["box"][3]))
                        sg["t1"], sg["n"] = t, sg["n"] + 1
                        if sg["n"] == 2:   # 2nd sighting: past any fade-in -> reference frame
                            sg.update(frame=frame.copy(), hit=h)
                        still.append(sg)
                        break
                else:
                    sg = {"rule": h["rule"], "box": h["box"], "hit": h, "frame": frame.copy(),
                          "t0": t, "t1": t, "n": 1}
                    segments.append(sg)
                    still.append(sg)
            active = still
        idx += 1
    cap.release()
    return segments, fps


def _refine(path, segments, finder, fps, step):
    """Check every frame around each segment's edges for frame-accurate timing."""
    for s in segments:
        s["start"], s["end"] = s["t0"], s["t1"]
    windows = [(s, side, a, b) for s in segments
               for side, a, b in (("start", s["t0"] - step, s["t0"]), ("end", s["t1"], s["t1"] + step))]
    cap, idx = cv2.VideoCapture(str(path)), 0
    while windows and cap.grab():
        t = idx / fps
        live = [w for w in windows if w[2] < t < w[3]]
        if live:
            ok, frame = cap.retrieve()
            for s, side, _, _ in live:
                x0, y0, x1, y1 = s["box"]
                p = int((y1 - y0) * 0.6)
                crop = frame[max(0, int(y0) - p):int(y1) + p, max(0, int(x0) - p):int(x1) + p]
                if crop.size and any(h["rule"] is s["rule"] for h in finder.find(crop)):
                    s[side] = min(s[side], t) if side == "start" else max(s[side], t)
        idx += 1
    cap.release()
    for s in segments:
        s["start"], s["end"] = max(0.0, s["start"] - 0.5 / fps), s["end"] + 1.0 / fps


def plan_video(path, finder, opts):
    segments, fps = _scan(path, finder, opts.step)
    if not segments:
        return None
    _refine(path, segments, finder, fps, opts.step)
    descs, preview = [], []
    for sg in segments:
        h = sg["hit"]   # measure on the clean reference frame's own box (not the tracking union)
        new = sg["rule"].replacement_for(h["old_text"], opts.keep_case)
        sg["plan"] = style.analyse(sg["frame"], h, new, _fonts_for(sg["rule"], opts.font), opts.reflow)
        sg["plan"]["inpaint"] = opts.inpaint
        descs.append(_describe(sg, sg["plan"], (sg["start"], sg["end"])))
        preview.append((sg["frame"], style.apply(sg["frame"].copy(), dict(sg["plan"]))))
    return {"kind": "video", "count": len(segments), "descs": descs, "segments": segments,
            "fps": fps, "source": path, "preview": preview}


def write_video_plan(plan, out_path, ffmpeg):
    cap = cv2.VideoCapture(str(plan["source"]))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("could not read video")
    H, W = frame.shape[:2]
    fps = plan["fps"]
    out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-",
           "-i", str(plan["source"]), "-map", "0:v", "-map", "1:a?",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           # mp4/mov audio is copied untouched; other containers' audio is converted to AAC
           "-c:a", "copy" if Path(plan["source"]).suffix.lower() in (".mp4", ".mov", ".m4v") else "aac",
           "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    idx = 0
    try:
        while ok:
            t = idx / fps
            for sg in plan["segments"]:
                if sg["start"] <= t <= sg["end"]:
                    style.apply(frame, sg["plan"])
            proc.stdin.write(frame.tobytes())
            ok, frame = cap.read()
            idx += 1
        proc.stdin.close()
    except BrokenPipeError:
        pass
    cap.release()
    err = proc.stderr.read().decode(errors="ignore")
    if proc.wait() != 0:
        raise RuntimeError(err.strip()[-400:])
    return out_path

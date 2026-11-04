"""Python API - use VidTextSwap from your own code (or a web app) instead of the CLI.

    from vidtextswap import Rule
    from vidtextswap.api import process_files, scan_files

    results = process_files(["post.jpg", "reel.mp4"], [Rule("050 123 4567", "050 765 4321")], "out")
    for r in results:
        print(r.file, r.status, r.output)
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from .media import (IMAGE_EXT, VIDEO_EXT, find_ffmpeg, plan_image, plan_video, save_preview,
                    write_image_plan, write_video_plan)

_FINDER = None


def _finder(rules, fuzz):
    """One OCR engine per process (loading it takes a few seconds)."""
    global _FINDER
    from .finder import Finder
    if _FINDER is None:
        _FINDER = Finder(rules, fuzz=fuzz)
    else:
        _FINDER.set_rules(rules)
        _FINDER.fuzz = fuzz
    return _FINDER


@dataclass
class Result:
    file: str                    # input path
    kind: str                    # "image" or "video"
    status: str                  # "updated", "no match" or "error: ..."
    matches: int = 0
    details: list = field(default_factory=list)   # one line per edit: rule, timing, detected style
    output: str | None = None    # updated file
    preview: str | None = None   # before | after image


def is_media(path):
    return Path(path).suffix.lower() in VIDEO_EXT | IMAGE_EXT


def scan_files(paths, progress=None):
    """What text is on screen. Returns (items, texts): items groups phone numbers,
    websites, emails, prices, dates and offers {category: {text: n_files}};
    texts maps every distinct line of text to the set of files it appears in."""
    from .assistant import collect_texts, find_items
    files = [Path(p) for p in paths if is_media(p)]
    texts = collect_texts(files, _finder([], 0.85), log=progress or (lambda *_: None))
    return find_items(texts), texts


def process_files(paths, rules, out_dir, *, inpaint="auto", reflow=True, keep_case=True,
                  step=0.5, fuzz=0.85, font=None, progress=None):
    """Replace text in every image/video in `paths` according to `rules`.
    Only files with a match are written to `out_dir` (videos as .mp4), each with a
    before/after preview in `out_dir/_previews`. `progress(i, n, name)` is called
    before each file."""
    if not rules:
        raise ValueError("No replacement rules given")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = SimpleNamespace(step=step, font=font, keep_case=keep_case, reflow=reflow, inpaint=inpaint)
    finder = _finder(rules, fuzz)
    files = [Path(p) for p in paths if is_media(p)]
    ffmpeg = find_ffmpeg() if any(p.suffix.lower() in VIDEO_EXT for p in files) else None
    results, used = [], set()
    for i, path in enumerate(files):
        if progress:
            progress(i, len(files), path.name)
        kind = "video" if path.suffix.lower() in VIDEO_EXT else "image"
        name = path.stem if path.stem not in used else f"{path.stem}_{i}"   # same name uploaded twice
        used.add(name)
        try:
            plan = (plan_video if kind == "video" else plan_image)(path, finder, opts)
            if not plan:
                results.append(Result(str(path), kind, "no match"))
                continue
            preview = out_dir / "_previews" / f"{name}.jpg"
            save_preview(plan["preview"], preview)
            if kind == "video":
                output = write_video_plan(plan, out_dir / f"{name}.mp4", ffmpeg)
            else:
                output = out_dir / f"{name}{path.suffix.lower()}"
                write_image_plan(plan, output)
            results.append(Result(str(path), kind, "updated", plan["count"], plan["descs"],
                                  str(output), str(preview)))
        except Exception as e:   # keep going with the other files
            results.append(Result(str(path), kind, f"error: {e}"))
    if progress:
        progress(len(files), len(files), "")
    return results

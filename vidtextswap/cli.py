"""Command line / interactive front end."""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import __version__
from .media import (IMAGE_EXT, VIDEO_EXT, find_ffmpeg, plan_image, plan_video, save_preview,
                    write_image_plan, write_video_plan)
from .rules import Rule, load_rules_csv, normalise, only_digits

MARKER = ".vidtextswap_output"


def log(msg=""):
    print(msg, flush=True)


def ask(prompt, default=None):
    v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip().strip('"').strip("'")
    return v.replace("\\ ", " ") or default   # macOS drag-and-drop escapes spaces


def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def build_parser():
    ap = argparse.ArgumentParser(
        prog="vidtextswap",
        description="Find text in videos and images and replace it, keeping the original style.",
        epilog='Example: vidtextswap -i media -o updated --find "050 123 4567" --replace "050 765 4321"')
    ap.add_argument("-i", "--input", help="Folder (searched recursively) or a single video/image")
    ap.add_argument("-o", "--output", help="Folder for updated copies (default: <input>_updated)")
    ap.add_argument("--find", action="append", default=[], help="Text to find (repeatable, pairs with --replace)")
    ap.add_argument("--replace", action="append", default=[], help="Replacement for the matching --find")
    ap.add_argument("--rules", help="CSV file with columns find,replace[,type][,font]")
    ap.add_argument("--review", action="store_true", help="Show a preview and ask before saving each file")
    ap.add_argument("--dry-run", action="store_true", help="Only report where the text appears")
    ap.add_argument("--step", type=float, default=0.5, help="Seconds between scanned video frames (default 0.5)")
    ap.add_argument("--fuzz", type=float, default=0.85,
                    help="Match tolerance for OCR mistakes, 0.5-1.0 (1.0 = exact only; default 0.85)")
    ap.add_argument("--font", help="Font file (.ttf/.otf) for the new text (default: auto bold/regular)")
    ap.add_argument("--exact-case", action="store_true",
                    help="Draw replacements exactly as typed (default: follow ALL CAPS originals)")
    ap.add_argument("--no-reflow", action="store_true",
                    help="Don't move neighbouring words; squeeze/stretch the new text into the old space")
    ai = ap.add_argument_group("AI features")
    ai.add_argument("--instruct", metavar="TEXT",
                    help='Describe the change in plain words, e.g. "change the phone number to 050 765 4321 '
                         'and the offer date to 15 Jan". Uses the AI chosen with --llm.')
    ai.add_argument("--yes", action="store_true", help="Accept the rules proposed by --instruct without asking")
    ai.add_argument("--llm", default="auto", choices=["auto", "claude", "gemini", "groq", "ollama", "openai"],
                    help="AI for --instruct. Free: ollama (on your computer), gemini / groq (free tiers). "
                         "auto = first one set up (default)")
    ai.add_argument("--llm-model", default=None, help="Model name (defaults: claude-haiku-5-5, gemini-2.5-flash, "
                                                       "llama-3.3-70b-versatile, qwen2.5:7b)")
    ai.add_argument("--llm-url", default=None, help="Server address for --llm openai/ollama "
                                                     "(e.g. http://localhost:1234/v1 for LM Studio)")
    ai.add_argument("--scan", action="store_true",
                    help="Only list the text found on screen (phone numbers, websites, dates, prices...) and stop")
    ai.add_argument("--inpaint", choices=["auto", "ai", "fast"], default="auto",
                    help="Background repair: ai = LaMa model (best on busy backgrounds), fast = OpenCV, "
                         "auto = AI for images and still video shots, fast while the camera moves (default)")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def collect_rules(args):
    rules = []
    if args.rules:
        rules += load_rules_csv(args.rules)
    if len(args.find) != len(args.replace):
        sys.exit("Each --find needs a matching --replace.")
    rules += [Rule(f, r) for f, r in zip(args.find, args.replace)]
    return rules


def ask_rules():
    rules = []
    log("\nWhat should be replaced? (leave 'Find' empty when you are done)")
    while True:
        f = ask(f"  Find #{len(rules) + 1}")
        if not f:
            break
        r = ask("  Replace with")
        if r is None:
            log("  (replacement can't be empty - skipped)")
            continue
        try:
            rules.append(Rule(f, r))
        except ValueError as e:
            log(f"  {e}")
    return rules


def save_rules(rules, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["find", "replace", "type", "font"])
        for r in rules:
            w.writerow([r.find, r.replace, r.kind, r.font or ""])


def rules_from_instruction(instruction, files, finder, args, interactive):
    from .assistant import collect_texts, get_client, propose_rules
    client = get_client(args.llm, args.llm_model, args.llm_url)   # fail fast if not set up
    log("\nReading the text in your files...")
    texts = collect_texts(files, finder, log=log)
    log(f"Found {len(texts)} distinct line(s) of text. Asking {client.label} to work out the changes...")
    try:
        rules, notes, rejected = propose_rules(instruction, texts, client=client)
    except RuntimeError:
        raise
    except Exception as e:           # network / auth / model errors from the API
        raise RuntimeError(f"{client.label} request failed: {e}") from None
    log("\nProposed changes:")
    for r in rules:
        log(f"   {r.label()}   ({r.kind})")
    if not rules:
        log("   (none)")
    for find, why in rejected:
        log(f"   dropped '{find}': {why}")
    if notes:
        log(f"\nNote from {client.label}: {notes}")
    if rules and not args.yes:
        if not (ask("\nApply these changes? (y/n)", "y") or "y").lower().startswith("y"):
            sys.exit("Cancelled - nothing was changed.")
    return rules


def run_scan(files, finder, out_dir):
    from .assistant import collect_texts, find_items
    log("\nReading the text in your files...")
    texts = collect_texts(files, finder, log=log)
    items = find_items(texts)
    log("\n================ TEXT FOUND ================")
    if not items:
        log("No phone numbers, websites, emails, prices, dates or offers found.")
    for cat, found in items.items():
        log(f"\n{cat}:")
        for item, n in found.items():
            log(f"   {item}   (in {n} file{'s' if n != 1 else ''})")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "scan.txt"
    with open(report, "w", encoding="utf-8") as f:
        for text, fs in sorted(texts.items(), key=lambda kv: -len(kv[1])):
            f.write(f"{text}\t{len(fs)} file(s)\t" + "; ".join(sorted(p.name for p in fs)[:10]) + "\n")
    log(f"\nAll {len(texts)} lines of text are listed in: {report}")
    return 0


def scan_text_files(root, rules, skip_dirs):
    """Caption/description files (e.g. an Instagram export) that mention the old text."""
    hits = []
    files = [root] if root.is_file() else root.rglob("*")
    for p in files:
        if p.suffix.lower() not in (".html", ".json", ".txt", ".srt", ".vtt") or not p.is_file():
            continue
        if any(d in p.parents for d in skip_dirs):
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        digits, norm = only_digits(raw), normalise(raw)
        for r in rules:
            if any(t in (digits if r.kind == "phone" else norm) for t in r.targets):
                hits.append((p, r))
    return hits


def main(argv=None):
    args = build_parser().parse_args(argv)
    interactive = args.input is None
    if interactive:
        log(f"\n=== VidTextSwap {__version__} ===")
        log("Replaces text in your videos and images, keeping the original style.\n")
        args.input = ask("Folder (or file) with your videos/images")
        if not args.input:
            sys.exit("No input given.")
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        sys.exit(f"Not found: {src}")
    base = src if src.is_dir() else src.parent
    default_out = str(base.parent / f"{base.name}_updated") if src.is_dir() else str(base / "updated")
    if not args.output:
        args.output = ask("Folder for updated copies", default_out) if interactive else default_out
    out_dir = Path(args.output).expanduser().resolve()
    if out_dir == base:
        sys.exit("Output folder must be different from the input folder.")
    if not 0.5 <= args.fuzz <= 1.0:
        sys.exit("--fuzz must be between 0.5 and 1.0")
    rules = collect_rules(args)
    instruction = args.instruct
    if interactive and not rules and not instruction and not args.scan:
        log("\nDescribe the change in plain words, e.g. 'change the phone number to 050 765 4321'.")
        log("(Uses AI: free with Ollama, Gemini or Groq - see README. Press Enter to type the exact")
        log(" old and new text instead.)")
        instruction = ask("Change")
    if interactive and not args.review and not args.scan:
        args.review = (ask("Preview and approve each file before saving? (y/n)", "n") or "n").lower().startswith("y")

    skip_dirs = {m.parent for m in base.rglob(MARKER)} | {out_dir} if src.is_dir() else {out_dir}
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.is_file()
                       and p.suffix.lower() in VIDEO_EXT | IMAGE_EXT
                       and not any(d in p.parents for d in skip_dirs))
    else:
        files = [src]

    log("\nLoading text recognition...")
    from .finder import Finder   # heavy import, after argument errors
    finder = Finder(rules, fuzz=args.fuzz)
    if args.scan:
        return run_scan(files, finder, out_dir)
    if not rules and instruction:
        try:
            rules = rules_from_instruction(instruction, files, finder, args, interactive)
        except RuntimeError as e:
            log(f"\n{e}")
            if not interactive:
                return 1
            rules = ask_rules()
    elif not rules and interactive:
        rules = ask_rules()
    if not rules:
        sys.exit("Nothing to replace. Use --find/--replace, --rules or --instruct.")
    finder.set_rules(rules)

    opts = SimpleNamespace(step=args.step, font=args.font, keep_case=not args.exact_case,
                           reflow=not args.no_reflow, inpaint=args.inpaint)
    ffmpeg = None if args.dry_run else find_ffmpeg()
    log(f"\nFound {len(files)} media file(s). Rules:")
    for r in rules:
        log(f"   {r.label()}   ({r.kind})")
    log("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MARKER).write_text("Created by vidtextswap - skipped when scanning.\n", encoding="utf-8")
    save_rules(rules, out_dir / "rules_used.csv")   # re-run later with --rules
    preview_dir = out_dir / "_previews"
    rows, updated, apply_all, t0 = [], 0, False, time.time()

    for n, path in enumerate(files, 1):
        rel = path.relative_to(base)
        kind = "video" if path.suffix.lower() in VIDEO_EXT else "image"
        out_path = out_dir / (rel.with_suffix(".mp4") if kind == "video" else rel)
        log(f"[{n}/{len(files)}] {rel}")
        try:
            plan = (plan_video if kind == "video" else plan_image)(path, finder, opts)
            if not plan:
                log("      no match")
                rows.append([str(rel), kind, "no match", 0, "", ""])
                continue
            for d in plan["descs"]:
                log(f"      found {d}")
            if args.dry_run:
                rows.append([str(rel), kind, "found (dry run)", plan["count"], " | ".join(plan["descs"]), ""])
                continue
            preview = preview_dir / rel.with_suffix(".jpg")
            save_preview(plan["preview"], preview)
            if args.review and not apply_all:
                open_file(preview.with_suffix(".jpg"))
                a = (ask(f"      Preview: {preview.with_suffix('.jpg')}\n      Save this file? "
                         "[y]es / [n]o / [a]ll remaining / [q]uit", "y") or "y").lower()[:1]
                if a == "q":
                    rows.append([str(rel), kind, "stopped by user", plan["count"], "", ""])
                    break
                if a == "n":
                    log("      skipped")
                    rows.append([str(rel), kind, "skipped in review", plan["count"], " | ".join(plan["descs"]), ""])
                    continue
                apply_all = a == "a"
            if kind == "video":
                out_path = write_video_plan(plan, out_path, ffmpeg)
            else:
                write_image_plan(plan, out_path)
            updated += 1
            log("      saved")
            rows.append([str(rel), kind, "updated", plan["count"], " | ".join(plan["descs"]),
                         str(out_path.relative_to(out_dir))])
        except Exception as e:  # keep going with the other files
            log(f"      ERROR: {e}")
            rows.append([str(rel), kind, f"error: {e}", 0, "", ""])

    report = out_dir / "report.csv"
    with open(report, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "type", "status", "matches", "details", "updated_file"])
        w.writerows(rows)

    log("\n================ DONE ================")
    log(f"Files scanned : {len(files)}   ({(time.time() - t0) / 60:.1f} min)")
    log(f"{'Files matched' if args.dry_run else 'Files updated'} : "
        f"{sum(1 for r in rows if r[2].startswith('found')) if args.dry_run else updated}")
    log(f"Output folder : {out_dir}")
    log(f"Report        : {report}")
    if updated:
        log(f"Previews      : {preview_dir}  (before | after)")
    text_hits = scan_text_files(src, rules, skip_dirs)
    if text_hits:
        log("\nThe old text also appears in caption/text files - update those by hand:")
        for p, r in text_hits:
            log(f"   {p.relative_to(base)}   {r.label()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Plain-language instructions -> replacement rules, plus a quick scan of what
text is on screen.

1. scan: OCR every image and a frame every ~1 s of every video, collect the
   distinct lines of text.
2. instruct: send the user's request and those lines to an AI model (Claude,
   or free: Ollama on your computer, Gemini or Groq free tiers), which returns
   find/replace rules in a strict JSON shape. Every rule is then checked
   against the text actually seen, so the model can't invent text to replace.
"""
import os
import re
from collections import defaultdict

import cv2

from .media import IMAGE_EXT, read_image
from .rules import Rule, normalise, only_digits

DEFAULT_MODEL = "claude-haiku-5-5"


# ----------------------------------------------------------------------------- scan

def collect_texts(files, finder, step=1.0, max_frames=60, log=print):
    """{line text: set(files)} for every distinct line of on-screen text."""
    seen = defaultdict(set)
    for n, path in enumerate(files, 1):
        log(f"  reading [{n}/{len(files)}] {path.name}")
        frames = [read_image(path)] if path.suffix.lower() in IMAGE_EXT else _sample(path, step, max_frames)
        for frame in frames:
            for line in finder._lines(finder.ocr_boxes(frame)):
                text = re.sub(r"\s+", " ", " ".join(b["text"] for b in line)).strip()
                if len(normalise(text)) >= 2:
                    seen[text].add(path)
    return seen


def _sample(path, step, max_frames):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every = max(1, int(round(step * fps)), (total // max_frames) + 1 if total else 1)
    idx = 0
    while cap.grab():
        if idx % every == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield frame
        idx += 1
    cap.release()


_PATTERNS = {
    "Phone numbers": r"(?<![\w])\+?[\d][\d\s\-().]{6,}\d(?![\w])",
    "Websites": r"(?i)\b(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|ae|in|net|org|co|io|sa|qa|om|kw|bh|me|info|biz|shop|store)\b(?:/\S*)?",
    "Emails": r"(?i)\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "Prices": r"(?i)(?:AED|Dhs?\.?|SAR|QAR|OMR|KWD|BHD|USD|INR|Rs\.?|₹|\$|€|£)\s?\d[\d,.]*|\d[\d,.]*\s?(?:AED|Dhs|SAR|QAR|/-)",
    "Dates": r"(?i)\b\d{1,2}(?:st|nd|rd|th)?\s?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:\s?\d{2,4})?\b"
             r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s\d{1,2}(?:st|nd|rd|th)?\b"
             r"|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    "Offers": r"(?i)\b\d{1,2}\s?%\s?(?:off)?",
}


def find_items(texts):
    """Things people usually need to update, grouped: {category: {item: n_files}}."""
    out = {}
    for name, pat in _PATTERNS.items():
        found = defaultdict(set)
        for text, files in texts.items():
            for m in re.finditer(pat, text):
                item = m.group().strip(" .-")
                if name == "Phone numbers" and len(only_digits(item)) < 7:
                    continue
                found[item] |= files
        if found:
            out[name] = {k: len(v) for k, v in sorted(found.items(), key=lambda kv: -len(kv[1]))}
    return out


# ----------------------------------------------------------------------------- instruct

SYSTEM = """You turn a user's plain-language request into exact find/replace rules for \
a tool that edits text shown on screen in their videos and images.

You get the request and every distinct line of on-screen text that OCR found.

Rules for your answer:
- `find` must be copied from the on-screen lines (a substring of one line; OCR may have \
small errors - copy what is shown). Use only the part that changes, e.g. the number, \
not the whole sentence, unless more context is needed to avoid changing other text.
- `replace` is the new text. If the user gave an exact format for the new value, use it \
exactly. Otherwise follow the format of the old text.
- A phone number written several ways (0501234567, 050 123 4567, +971 50 123 4567) \
needs only ONE rule with kind "phone"; the tool matches all formats.
- Do not invent text that is not on screen. If part of the request matches nothing on \
screen, or is ambiguous (e.g. two different dates and the user said "the date"), make \
no rule for it and explain in `notes`.
- Never add changes the user didn't ask for."""

TOOL = {
    "name": "set_replacement_rules",
    "description": "Return the find/replace rules that carry out the user's request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "find": {"type": "string", "description": "Old text, as it appears on screen"},
                        "replace": {"type": "string", "description": "New text"},
                        "kind": {"type": "string", "enum": ["phone", "text"]},
                        "reason": {"type": "string", "description": "Which part of the request this does"},
                    },
                    "required": ["find", "replace", "kind"],
                },
            },
            "notes": {"type": "string", "description": "Anything you could not do, or questions for the user"},
        },
        "required": ["rules"],
    },
}


# ----------------------------------------------------------------------------- AI providers

JSON_SHAPE = ('Reply with ONLY a JSON object, no other text, in this shape:\n'
              '{"rules": [{"find": "old text", "replace": "new text", "kind": "phone" or "text", '
              '"reason": "why"}], "notes": "anything you could not do"}')

PROVIDERS = {
    # name: how to call it, API key variable, default model, where to get a key
    "claude": {"api": "anthropic", "key": "ANTHROPIC_API_KEY", "model": DEFAULT_MODEL,
               "signup": "https://console.anthropic.com", "label": "Claude"},
    "gemini": {"api": "openai", "key": "GEMINI_API_KEY", "model": "gemini-2.5-flash",
               "url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "signup": "https://aistudio.google.com/apikey", "label": "Google Gemini (free tier)"},
    "groq": {"api": "openai", "key": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile",
             "url": "https://api.groq.com/openai/v1",
             "signup": "https://console.groq.com/keys", "label": "Groq (free tier)"},
    "ollama": {"api": "ollama", "key": None, "model": "qwen2.5:7b",
               "url": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
               "signup": "https://ollama.com/download", "label": "Ollama (free, on your computer)"},
    "openai": {"api": "openai", "key": "LLM_API_KEY", "model": None, "url": None,
               "signup": "", "label": "OpenAI-compatible server"},
}

FREE_HELP = """No AI provider is set up. Free options:
  1. Ollama - runs on your computer, no account, no limits, text never leaves your PC:
       install from https://ollama.com/download, then run:  ollama pull qwen2.5:7b
       and use  --llm ollama
  2. Google Gemini free tier - get a key at https://aistudio.google.com/apikey
       set GEMINI_API_KEY=...        and use  --llm gemini
  3. Groq free tier - get a key at https://console.groq.com/keys
       set GROQ_API_KEY=...          and use  --llm groq
  Paid: Claude - set ANTHROPIC_API_KEY=...  and use  --llm claude
  (Mac/Linux: use  export NAME=value  instead of  set NAME=value)"""


def _http_json(url, payload, headers=None, timeout=180):
    import json
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:400]
        raise _HTTPError(e.code, body) from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach {url}: {e.reason}") from None


class _HTTPError(RuntimeError):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}: {body}")
        self.code, self.body = code, body


def parse_json(text):
    """JSON object from a model reply, tolerating ```json fences or chatter around it."""
    import json
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    try:
        return json.loads(t)
    except ValueError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise RuntimeError(f"The model did not reply with JSON: {t[:200]}") from None


class ClaudeBackend:
    def __init__(self, client, model):
        self.client, self.model, self.label = client, model, "Claude"

    def ask(self, system, user):
        msg = self.client.messages.create(
            model=self.model, max_tokens=2000, system=system, tools=[TOOL],
            tool_choice={"type": "tool", "name": TOOL["name"]},
            messages=[{"role": "user", "content": user}])
        block = next((b for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
        if block is None:
            raise RuntimeError("The model did not return any rules.")
        return block.input or {}


class OpenAICompatBackend:
    """Gemini, Groq, OpenRouter, LM Studio, vLLM... anything with /chat/completions."""

    def __init__(self, url, key, model, label):
        self.url, self.key, self.model, self.label = url.rstrip("/"), key, model, label

    def ask(self, system, user):
        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": system + "\n\n" + JSON_SHAPE},
                                {"role": "user", "content": user}],
                   "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        try:
            data = _http_json(self.url + "/chat/completions", payload, headers)
        except _HTTPError as e:
            if e.code != 400 or "response_format" not in e.body:
                raise RuntimeError(f"{self.label}: {e}") from None
            payload.pop("response_format")          # server doesn't support JSON mode
            data = _http_json(self.url + "/chat/completions", payload, headers)
        return parse_json(data["choices"][0]["message"]["content"])


class OllamaBackend:
    """Local models through Ollama's native API, with the answer forced to our JSON schema."""

    def __init__(self, url, model):
        self.url, self.model, self.label = url.rstrip("/"), model, f"Ollama ({model})"

    def ask(self, system, user):
        payload = {"model": self.model, "stream": False, "options": {"temperature": 0},
                   "format": TOOL["input_schema"],
                   "messages": [{"role": "system", "content": system + "\n\n" + JSON_SHAPE},
                                {"role": "user", "content": user}]}
        try:
            data = _http_json(self.url + "/api/chat", payload, timeout=600)
        except _HTTPError as e:
            if e.code == 404 or "not found" in e.body:
                raise RuntimeError(f"Ollama doesn't have the model '{self.model}' yet. Run:  "
                                   f"ollama pull {self.model}") from None
            raise RuntimeError(f"Ollama: {e}") from None
        return parse_json(data["message"]["content"])


def _ollama_running(url):
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def get_client(provider="auto", model=None, url=None):
    """The AI backend to use. 'auto' picks the first that is set up:
    Claude key, Gemini key, Groq key, then a running Ollama."""
    if provider == "auto":
        for name in ("claude", "gemini", "groq"):
            if os.environ.get(PROVIDERS[name]["key"]):
                provider = name
                break
        else:
            if _ollama_running(url or PROVIDERS["ollama"]["url"]):
                provider = "ollama"
            else:
                raise RuntimeError(FREE_HELP)
    if provider not in PROVIDERS:
        raise RuntimeError(f"Unknown AI provider '{provider}'. Choose: {', '.join(PROVIDERS)}")
    p = PROVIDERS[provider]
    model = model or p["model"]
    if p["api"] == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Claude needs the Anthropic SDK:  pip install anthropic") from None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set your Anthropic API key first (get one at https://console.anthropic.com):\n"
                               "   Windows:  set ANTHROPIC_API_KEY=sk-ant-...\n"
                               "   Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...\n\n" + FREE_HELP)
        return ClaudeBackend(anthropic.Anthropic(), model)
    if p["api"] == "ollama":
        url = url or p["url"]
        if not _ollama_running(url):
            raise RuntimeError(f"Ollama isn't running at {url}. Install it from https://ollama.com/download, "
                               f"start it, then run:  ollama pull {model}")
        return OllamaBackend(url, model)
    url = url or p["url"]
    key = os.environ.get(p["key"]) or (os.environ.get("OPENAI_API_KEY") if provider == "openai" else None)
    if not url:
        raise RuntimeError("--llm openai needs --llm-url, e.g. http://localhost:1234/v1 for LM Studio")
    if not model:
        raise RuntimeError("--llm openai needs --llm-model")
    if p["key"] and not key and provider != "openai":
        raise RuntimeError(f"Set {p['key']} first (free key: {p['signup']})\n"
                           f"   Windows:  set {p['key']}=...\n   Mac/Linux: export {p['key']}=...")
    return OpenAICompatBackend(url, key, model, p["label"])


def _seen(rule, texts):
    for text in texts:
        if rule.kind == "phone":
            if any(t in only_digits(text) for t in rule.targets):
                return True
        elif rule.targets[0] in normalise(text):
            return True
    return False


def propose_rules(instruction, texts, model=DEFAULT_MODEL, client=None, max_chars=24000):
    """Ask the model for rules. Returns (rules, notes, rejected[(find, why)]).
    `client` is a backend from get_client() (or an anthropic.Anthropic-like client)."""
    client = client or get_client()
    if not hasattr(client, "ask"):           # raw Anthropic SDK client
        client = ClaudeBackend(client, model)
    lines, size = [], 0
    for text, files in sorted(texts.items(), key=lambda kv: -len(kv[1])):
        entry = f"- {text}   (in {len(files)} file{'s' if len(files) != 1 else ''})"
        size += len(entry)
        if size > max_chars:
            lines.append("- ... (more lines omitted)")
            break
        lines.append(entry)
    user = (f"Request: {instruction}\n\nOn-screen text found in the user's files:\n"
            + ("\n".join(lines) if lines else "(no text found)"))
    data = client.ask(SYSTEM, user)
    if not isinstance(data, dict):
        raise RuntimeError("The model's answer was not in the expected format.")

    rules, rejected = [], []
    for r in data.get("rules") or []:
        if not isinstance(r, dict):
            continue
        r = {k: (v if isinstance(v, str) else "") for k, v in r.items()}
        if r.get("kind") not in ("phone", "text"):
            r["kind"] = "auto"
        try:
            rule = Rule(r.get("find", ""), r.get("replace", ""), r.get("kind") or "auto")
        except ValueError as e:
            rejected.append((r.get("find", ""), str(e)))
            continue
        if not r.get("replace", "").strip():
            rejected.append((rule.find, "empty replacement"))
        elif not _seen(rule, texts):
            rejected.append((rule.find, "not found in the on-screen text"))
        elif any(x.kind == rule.kind and x.targets == rule.targets for x in rules):
            continue   # duplicate
        else:
            rules.append(rule)
    return rules, (data.get("notes") or "").strip(), rejected

"""Free AI providers: real HTTP calls to local stand-in servers that speak the
Ollama and OpenAI-compatible (Gemini, Groq, LM Studio...) APIs."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from vidtextswap import assistant
from vidtextswap.assistant import OllamaBackend, OpenAICompatBackend, get_client, parse_json, propose_rules

TEXTS = {"Call us: 050 123 4567": {Path("a.jpg")}, "Now open in Al Quoz": {Path("a.jpg")}}
ANSWER = {"rules": [{"find": "050 123 4567", "replace": "050 765 4321", "kind": "phone"},
                    {"find": "Al Quoz", "replace": "JLT", "kind": "text"}], "notes": ""}


@pytest.fixture
def server():
    """Tiny HTTP server; `server.reply(path, status, body)` sets responses, `.requests` records calls."""
    state = {"replies": {}, "requests": []}

    class H(BaseHTTPRequestHandler):
        def _send(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state["requests"].append((self.path, dict(self.headers), json.loads(body) if body else None))
            replies = state["replies"].get(self.path) or [(404, {"error": "not found"})]
            status, payload = replies.pop(0) if len(replies) > 1 else replies[0]
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _send

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    class S:
        url = f"http://127.0.0.1:{httpd.server_port}"
        requests = state["requests"]

        @staticmethod
        def reply(path, *responses):
            state["replies"][path] = list(responses)
    yield S
    httpd.shutdown()


def test_ollama_backend_uses_schema_and_parses_answer(server):
    server.reply("/api/chat", (200, {"message": {"content": json.dumps(ANSWER)}}))
    rules, _, rejected = propose_rules("new number 050 765 4321, now in JLT", TEXTS,
                                       client=OllamaBackend(server.url, "qwen2.5:7b"))
    assert [(r.find, r.replace) for r in rules] == [("050 123 4567", "050 765 4321"), ("Al Quoz", "JLT")]
    path, _, body = server.requests[-1]
    assert body["model"] == "qwen2.5:7b" and body["stream"] is False
    assert body["format"]["properties"]["rules"]["type"] == "array"      # answer forced to our schema


def test_ollama_missing_model_message(server):
    server.reply("/api/chat", (404, {"error": "model 'qwen2.5:7b' not found"}))
    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:7b"):
        OllamaBackend(server.url, "qwen2.5:7b").ask("s", "u")


def test_openai_compatible_backend_gemini_groq_style(server):
    server.reply("/v1/chat/completions",
                 (200, {"choices": [{"message": {"content": "```json\n" + json.dumps(ANSWER) + "\n```"}}]}))
    b = OpenAICompatBackend(server.url + "/v1", "free-key", "gemini-2.5-flash", "Gemini")
    rules, _, _ = propose_rules("x", TEXTS, client=b)
    assert len(rules) == 2
    path, headers, body = server.requests[-1]
    assert headers["Authorization"] == "Bearer free-key"
    assert body["response_format"] == {"type": "json_object"}
    assert "ONLY a JSON object" in body["messages"][0]["content"]


def test_openai_compatible_retries_without_json_mode(server):
    server.reply("/v1/chat/completions",
                 (400, {"error": "response_format is not supported"}),
                 (200, {"choices": [{"message": {"content": "Here you go: " + json.dumps(ANSWER)}}]}))
    data = OpenAICompatBackend(server.url + "/v1", None, "local-model", "LM Studio").ask("s", "u")
    assert len(data["rules"]) == 2
    assert "response_format" not in server.requests[-1][2]


def test_auto_picks_first_configured(monkeypatch, server):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert "Groq" in get_client().label
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert "Gemini" in get_client().label                       # Gemini before Groq
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GROQ_API_KEY")
    server.reply("/api/tags", (200, {"models": []}))
    assert "Ollama" in get_client(url=server.url).label         # a running Ollama is found


def test_auto_with_nothing_set_up_explains_free_options(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="Free options"):
        get_client(url="http://127.0.0.1:9")                    # nothing listening


def test_parse_json_tolerates_chatter():
    assert parse_json('Sure!\n{"rules": [], "notes": "x"}\nHope that helps')["notes"] == "x"
    with pytest.raises(RuntimeError):
        parse_json("no json here")


def test_bad_rule_shapes_are_ignored():
    class B:
        label = "x"

        def ask(self, s, u):
            return {"rules": ["junk", {"find": "Al Quoz", "replace": "JLT", "kind": "weird"}, {"find": 5}]}
    rules, _, _ = propose_rules("x", TEXTS, client=B())
    assert [(r.find, r.kind) for r in rules] == [("Al Quoz", "text")]

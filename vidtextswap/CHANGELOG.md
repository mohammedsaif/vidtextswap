# Changelog

## 0.2.0
- **AI background repair** with LaMa (`--inpaint auto|ai|fast`). It runs once per still
  video shot and the result is reused, so videos stay fast.
- **Plain-language instructions** with Claude (`--instruct "..."`). Proposed rules are
  checked against the on-screen text and shown for approval. They are saved to `rules_used.csv`.
- **Free AI options** for `--instruct` (`--llm`): Ollama (local, offline, unlimited), the Google
  Gemini and Groq free tiers, any OpenAI-compatible server (LM Studio, OpenRouter...), or Claude.
  `auto` picks the first one set up. No extra packages are needed for the free options.
- **`--scan`** lists the phone numbers, websites, emails, prices, dates and offers on screen.
- Text detection on busy backgrounds (tiles, photos) by text colour.
- Background sampling ignores neighbouring text lines, which gives more accurate size and outline.
- Font weight is chosen by letter thickness, which is robust to compressed video.
- Switched to `rapidocr` 3.x, which works on Python 3.13+. `rapidocr_onnxruntime` is still used if installed.

## 0.1.0
- First release: find/replace any text in videos and images in the original style,
  CSV rules, review mode, line reflow, previews and report.

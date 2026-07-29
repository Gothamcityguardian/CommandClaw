# CommandClaw

An offline prompt construction tool. Before you send a query to a top-tier AI model, CommandClaw interviews you — mapping what you know and don't know — then constructs a precise, confident prompt on your behalf.

Based on the *known knowns / known unknowns / unknown knowns / unknown unknowns* framework.

Runs fully offline using local GGUF models. Optional web access for gap-filling.

---

## How it works

```
  You                CommandClaw (local LLM)
  ─────────────────────────────────────────────
  Phase 1: Goal     ← What are you trying to achieve?
  Phase 2: Knowledge← What do you know? What are your gaps?
  Phase 3: Scope    ← What goes in the prompt now vs later?
  ─────────────────────────────────────────────
  Confidence Map    → Visual 4-quadrant knowledge summary
  Gap Fill (opt.)   → Web search to raise weak-spot confidence
  Prompt Output     → Finished prompt, ready to paste
```

The tool never sends your data anywhere. Everything runs on your machine.

---

## Requirements

- Python 3.10 or newer
- 4 GB RAM minimum (8+ GB recommended)
- NVIDIA GPU (CUDA), Apple Silicon (Metal), AMD GPU (ROCm), or CPU-only

---

## One-click setup

```bash
git clone <repo-url> CommandClaw
cd CommandClaw
bash setup.sh
```

`setup.sh` will:
1. Create a Python virtual environment (`.venv/`)
2. Install all dependencies
3. Detect your GPU and install `llama-cpp-python` with the right backend
4. Download the appropriate model for your hardware
5. Create `commandclaw.sh` launcher

---

## Launching

```bash
bash commandclaw.sh
```

---

## Model tiers

| Tier | Hardware | Model | Size |
|------|----------|-------|------|
| 1 | CPU only / < 6 GB VRAM | Qwen3-4B Q4_K_M | ~2.5 GB |
| 2 | 6–8 GB VRAM **(default)** | Qwen3-8B Q4_K_M | ~5 GB |
| 3 | 16+ GB VRAM | Qwen3-8B + Qwen3-14B | ~5 + 8 GB |

To force a specific tier:

```bash
source .venv/bin/activate
python download_models.py --tier 2
```

Models are downloaded from HuggingFace and stored in `~/models/`.

---

## Windows

Use WSL2 (Windows Subsystem for Linux) with Ubuntu 22.04+:

```powershell
wsl --install -d Ubuntu-22.04
```

Then follow the Linux setup instructions inside the WSL terminal.

---

## Configuration

Edit `config.py` to customise:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODELS_DIR` | `~/models` | Where GGUF models are stored |
| `USE_SINGLE_MODEL` | `True` | Use one model for all phases (saves memory) |
| `INTERVIEW_MODEL_PRIORITY` | `["8B","9B","7B","4B","14B","27B"]` | Model size preference for interview phases |
| `SYNTHESIS_MODEL_PRIORITY` | `["14B","27B","8B","9B","7B","4B"]` | Model size preference for prompt synthesis |
| `N_CTX` | `8192` | Context window |
| `INTERVIEW_TEMPERATURE` | `0.7` | Temperature for interview turns |
| `SYNTHESIS_TEMPERATURE` | `0.2` | Temperature for prompt synthesis (lower = more focused) |

---

## Session output

Each session saves two files to `sessions/`:

| File | Contents |
|------|----------|
| `session_TIMESTAMP.json` | Full session data (goal, knowledge map, scope, web findings) |
| `prompt_TIMESTAMP.txt` | The final constructed prompt |

---

## Project structure

```
CommandClaw/
├── main.py                    # Entry point
├── commandclaw.sh             # Launcher (created by setup.sh)
├── setup.sh                   # One-click setup
├── download_models.py         # Model downloader (auto or --tier N)
├── config.py                  # Paths, model settings, GPU detection
├── requirements.txt
├── sessions/                  # Saved session JSON + prompt text files
├── llm/
│   └── runner.py              # llama-cpp-python wrapper (stream + extract)
├── interview/
│   ├── session.py             # Interview orchestrator (3 phases)
│   ├── knowledge_mapper.py    # 4-quadrant data structures
│   └── web_searcher.py        # DuckDuckGo search (no API key needed)
├── prompt_builder/
│   ├── confidence_map.py      # Terminal confidence map renderer
│   └── constructor.py         # Final prompt synthesis
└── gui/
    └── terminal_ui.py         # Rich terminal UI
```

---

## Commands during interview

| Input | Action |
|-------|--------|
| Any text | Continue the conversation |
| `/next` | Advance to the next phase immediately |
| `Ctrl+C` | Exit |


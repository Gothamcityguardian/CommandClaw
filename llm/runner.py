"""
LLM runner — Ollama primary backend, llama-cpp-python fallback.
Uses only stdlib for Ollama (urllib) so no extra deps are needed on first run.
"""

from __future__ import annotations
import json
import re
import urllib.request
import urllib.error
import subprocess
from pathlib import Path
from typing import Generator, Optional

import config

OLLAMA_HOST = "http://localhost:11434"

# Prefer models good at reasoning / instruction following
OLLAMA_MODEL_PRIORITY = [
    "qwen3.5:27b", "qwen3.5:latest", "qwen3.5",
    "gemma3:12b", "gemma3n", "gemma4",
    "nemotron", "ministral",
    "gemma3:4b",
]


# ── model discovery ──────────────────────────────────────────────────────────

def _ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as r:
            return [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:
        try:
            out = subprocess.check_output(["ollama", "list"], text=True, timeout=5)
            return [line.split()[0] for line in out.splitlines()[1:] if line.split()]
        except Exception:
            return []


def _score_ollama(name: str) -> int:
    for i, pat in enumerate(OLLAMA_MODEL_PRIORITY):
        if name.startswith(pat.split(":")[0]) or name == pat:
            return len(OLLAMA_MODEL_PRIORITY) - i + 100
    return 0


def find_model() -> Optional[dict]:
    """Return best available model as {name, backend} dict."""
    ollama = _ollama_models()
    if ollama:
        best = max(ollama, key=_score_ollama)
        return {"name": best, "backend": "ollama"}

    # Fallback: GGUF files in ~/models (recursive search)
    if config.MODELS_DIR.exists():
        ggufs = [p for p in config.MODELS_DIR.rglob("*.gguf")
                 if "mmproj" not in p.name.lower()
                 and "minicpm" not in p.name.lower()
                 and p.stat().st_size > 1_000_000]
        if ggufs:
            for token in config.INTERVIEW_MODEL_PRIORITY:
                for p in ggufs:
                    if token.upper() in p.name.upper():
                        return {"name": p.name, "path": str(p), "backend": "gguf"}
            return {"name": ggufs[0].name, "path": str(ggufs[0]), "backend": "gguf"}

    return None


# ── runner ───────────────────────────────────────────────────────────────────

class LLMRunner:
    def __init__(self, model_info: dict):
        self._info = model_info
        self._gguf = None
        if model_info["backend"] == "gguf":
            self._load_gguf(model_info["path"])

    def _load_gguf(self, path: str):
        from llama_cpp import Llama
        self._gguf = Llama(
            model_path=path,
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=config.N_CTX,
            verbose=False,
        )

    # ── streaming chat ────────────────────────────────────────────────────────

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = config.INTERVIEW_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS_INTERVIEW,
    ) -> Generator[str, None, None]:
        if self._info["backend"] == "ollama":
            yield from self._ollama_stream(messages, temperature, max_tokens)
        else:
            yield from self._gguf_stream(messages, temperature, max_tokens)

    def _ollama_stream(self, messages, temperature, max_tokens):
        payload = json.dumps({
            "model": self._info["name"],
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": config.N_CTX,
            },
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass

    def _gguf_stream(self, messages, temperature, max_tokens):
        for chunk in self._gguf.create_chat_completion(
            messages=messages, stream=True,
            temperature=temperature, max_tokens=max_tokens,
        ):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]

    # ── single-shot (extraction / synthesis) ─────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = config.SYNTHESIS_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS_SYNTHESIS,
    ) -> str:
        if self._info["backend"] == "ollama":
            return self._ollama_chat(messages, temperature, max_tokens)
        resp = self._gguf.create_chat_completion(
            messages=messages, stream=False,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]

    def _ollama_chat(self, messages, temperature, max_tokens) -> str:
        payload = json.dumps({
            "model": self._info["name"],
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": config.N_CTX,
            },
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]

    def extract_json(self, messages: list[dict]) -> Optional[dict]:
        raw = self.chat(messages, temperature=0.1, max_tokens=2048)
        # Strip <think>...</think> blocks (Qwen3 reasoning tokens)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def unload(self):
        self._gguf = None

    @property
    def model_name(self) -> str:
        return self._info["name"]

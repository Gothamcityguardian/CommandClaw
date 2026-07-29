"""
Thin wrapper around llama-cpp-python.
Supports streaming chat and single-shot extraction calls.
"""

from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Generator, Optional

import config


class LLMRunner:
    def __init__(self, model_path: Path, n_gpu_layers: int = config.N_GPU_LAYERS,
                 n_ctx: int = config.N_CTX):
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = config.INTERVIEW_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS_INTERVIEW,
    ) -> Generator[str, None, None]:
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]

    def chat(
        self,
        messages: list[dict],
        temperature: float = config.SYNTHESIS_TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS_SYNTHESIS,
    ) -> str:
        resp = self._llm.create_chat_completion(
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]

    def extract_json(self, messages: list[dict]) -> Optional[dict]:
        raw = self.chat(messages, temperature=0.1, max_tokens=2048)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def unload(self):
        del self._llm
        self._llm = None


def find_model(priority: list[str]) -> Optional[Path]:
    """Return the first GGUF in MODELS_DIR matching a priority substring."""
    if not config.MODELS_DIR.exists():
        return None
    candidates = list(config.MODELS_DIR.glob("*.gguf"))
    for token in priority:
        for p in candidates:
            if token.upper() in p.name.upper():
                return p
    return candidates[0] if candidates else None

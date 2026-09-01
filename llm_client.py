"""Central Ollama client with caching, retries, usage logging, and session stats.

Colab fix: _client is created lazily on first use via _get_client() instead of
at module import time. This ensures OLLAMA_HOST env changes made AFTER import
(e.g. in a Colab cell) are always picked up, and that a new client is created
after a connection failure rather than reusing a stale socket.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any

from dotenv import load_dotenv
import ollama

load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
# Raised from 5 → 8: Ollama's cold-start inside a Colab VM can take 15–20 s,
# so we need more retry budget before giving up.
MAX_RETRIES   = int(os.getenv("OLLAMA_MAX_RETRIES", "8"))

# Lazy client: created on first call, reset after connection errors.
_client: "ollama.Client | None" = None
_client_host: str = ""          # tracks which host the current client was built for
_cache: dict[str, str] = {}
_lock = threading.Lock()


@dataclass
class UsageStats:
    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0


_stats = UsageStats()


def _get_client() -> "ollama.Client":
    """Return (or lazily create) an Ollama client for the current OLLAMA_HOST.

    Re-creates the client whenever OLLAMA_HOST changes so that env-var overrides
    made inside a Colab cell (os.environ['OLLAMA_HOST'] = '...') take effect
    without requiring a Python kernel restart.
    """
    global _client, _client_host
    host = os.environ.get("OLLAMA_HOST", OLLAMA_HOST)
    if _client is None or _client_host != host:
        _client = ollama.Client(host=host)
        _client_host = host
        print(f"[ollama] client (re)created -> {host}")
    return _client


def _reset_client() -> None:
    """Force the next call_llm() to create a fresh client (call after connection errors)."""
    global _client
    _client = None


def _cache_key(prompt: str, system: str, model: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "system": system,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_usage_stats() -> dict[str, int]:
    """Return a copy of current in-process usage statistics."""
    with _lock:
        return asdict(_stats)


def reset_usage_stats() -> None:
    """Reset counters without clearing cached responses."""
    global _stats
    with _lock:
        _stats = UsageStats()


def clear_response_cache() -> None:
    with _lock:
        _cache.clear()


def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str | None = None,
) -> str:
    """Call a local Ollama model through one guarded, cached entry point.

    Ollama is local and has no Groq-style RPM quota, but retries still protect the
    app from transient daemon/model-loading failures (especially on Colab where
    the Ollama process can take 15-20 s to fully start). All model calls in this
    repository must go through this function.
    """
    selected_model = model or DEFAULT_MODEL
    key = _cache_key(prompt, system, selected_model, temperature, max_tokens)

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _stats.cache_hits += 1
            print(f"[ollama] cache hit model={selected_model}")
            return cached

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = _get_client()
            response: Any = client.chat(
                model=selected_model,
                messages=messages,
                stream=False,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            text = response["message"]["content"].strip()
            prompt_tokens    = int(response.get("prompt_eval_count") or 0)
            completion_tokens = int(response.get("eval_count")       or 0)
            duration          = int(response.get("total_duration")   or 0)

            with _lock:
                _cache[key] = text
                _stats.requests          += 1
                _stats.prompt_tokens     += prompt_tokens
                _stats.completion_tokens += completion_tokens
                _stats.total_duration_ns += duration
                cumulative = asdict(_stats)

            print(
                "[ollama] "
                f"model={selected_model} prompt_tokens={prompt_tokens} "
                f"completion_tokens={completion_tokens} cumulative={cumulative}"
            )
            return text

        except Exception as exc:
            last_error = exc
            # Reset the client so the next attempt gets a fresh socket.
            _reset_client()
            if attempt >= MAX_RETRIES:
                break
            # Exponential back-off (capped at 30 s) — gives Ollama time to
            # finish loading the model on first call inside a Colab VM.
            delay = min(30.0, 2.0 * (2 ** attempt)) + random.uniform(0.0, 1.0)
            print(f"[ollama] attempt {attempt+1}/{MAX_RETRIES} failed ({exc}); "
                  f"retrying in {delay:.1f}s …")
            time.sleep(delay)

    raise RuntimeError(
        "Ollama call failed repeatedly. Confirm that Ollama is running, the model "
        f"'{selected_model}' is pulled, and OLLAMA_HOST is correct.\n"
        f"Last error: {last_error}"
    )

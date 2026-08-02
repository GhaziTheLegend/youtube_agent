"""
Provider-agnostic LLM wrapper.

Why this file exists: every module that needs an LLM (research, scripting,
metadata) should call `generate()` and never touch the Gemini or Anthropic
SDK directly. That means switching providers, or using different providers
for different tasks later, is a one-file change instead of a find-and-replace
across the whole codebase.

Provider selection:
- Set LLM_PROVIDER=gemini or LLM_PROVIDER=claude in .env to force one.
- If unset, we auto-pick: Gemini if GEMINI_API_KEY is set, else Claude.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-flash-lite-latest"
CLAUDE_MODEL = "claude-sonnet-4-6"


def _resolve_provider() -> str:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced:
        return forced
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    raise RuntimeError(
        "No LLM provider configured. Set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env"
    )


def _generate_gemini(prompt: str, system: str | None, json_mode: bool) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json" if json_mode else None,
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=config
    )
    return response.text


def _generate_claude(prompt: str, system: str | None, json_mode: bool) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Claude has no strict JSON mode - we lean on prompt instructions instead
    # (see research.py, which explicitly asks for JSON and strips code fences).
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate(prompt: str, system: str | None = None, json_mode: bool = False) -> str:
    """
    Send a prompt to whichever LLM provider is configured, return raw text.

    json_mode: when True, hints the provider to return valid JSON (Gemini
    enforces this natively; for Claude we just rely on prompt wording, so
    callers should still defensively parse the result - see research.py).
    """
    provider = _resolve_provider()
    if provider == "gemini":
        return _generate_gemini(prompt, system, json_mode)
    elif provider == "claude":
        return _generate_claude(prompt, system, json_mode)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

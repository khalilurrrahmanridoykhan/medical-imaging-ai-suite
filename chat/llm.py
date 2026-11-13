"""Thin wrapper around a local Ollama LLM for natural-language chat.

Requires Ollama running locally (`brew services start ollama`, or
`ollama serve`) and the configured MODEL pulled (`ollama pull
qwen2.5-coder:7b`, or swap in any other locally pulled model). This only
ever talks to localhost; no API key, no cost, no data leaves the machine.
"""

import ollama

MODEL = "qwen2.5-coder:7b"


def chat(messages: list) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]"""
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]

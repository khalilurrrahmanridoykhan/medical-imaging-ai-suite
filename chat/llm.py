"""Thin wrapper around local Ollama language and vision models.

Requires Ollama running locally (`brew services start ollama`, or
`ollama serve`) and the configured models pulled. This only ever talks to
localhost; no API key, no cost, no data leaves the machine.
"""

import os

import ollama

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")


def chat(messages: list) -> str:
    """Send conversation messages, routing image histories through a vision model."""
    has_images = any(message.get("images") for message in messages)
    model = VISION_MODEL if has_images else MODEL
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]

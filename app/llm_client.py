# app/llm_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]


class OllamaClient:
    """
    Minimal Ollama chat client.
    Expects Ollama at http://localhost:11434
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        stream: bool = False,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                # Ollama uses num_predict for max tokens
                "num_predict": max_tokens,
            },
        }

        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # Ollama returns: {"message": {"role": "assistant", "content": "..."} , ...}
        text = ""
        if isinstance(data, dict):
            msg = data.get("message") or {}
            text = msg.get("content", "") if isinstance(msg, dict) else ""

        return LLMResponse(text=text, raw=data)

    def tags(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

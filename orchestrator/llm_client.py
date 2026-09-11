"""
Тонкий клиент к локальному Ollama.

REST-запрос к http://localhost:11434.
Модель по умолчанию — llama3.2, замените на свою (см. `ollama list`).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.2,
            json_mode: bool = False,
    ) -> str:
        """
        Синхронный вызов /api/chat, non-streaming. Возвращает текст ответа.

        json_mode=True включает встроенный в Ollama принудительный JSON-режим
        Это плюс по надежности, чем просить об этом только в тексте промпта.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Не удалось достучаться до Ollama по адресу {self.host}. "
                f"Проверьте, что `ollama serve` запущен и модель {self.model!r} загружена "
                f"(`ollama pull {self.model}`). Исходная ошибка: {e}"
            ) from e
        return body["message"]["content"]


class FakeLLMClient:
    """Мок-клиент для тестирования оркестратора без запущенного Ollama."""

    def __init__(self, canned_response: str = ""):
        self.canned_response = canned_response
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.canned_response

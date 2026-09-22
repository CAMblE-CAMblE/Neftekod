"""
Тонкий клиент к локальному Ollama.

REST-запрос к http://localhost:11434.
Модель по умолчанию — llama3.2, замените на свою (см. `ollama list`).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


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
        Синхронный вызов Ollama, non-streaming. Возвращает текст ответа.

        Используем /api/generate: в Ollama 0.34.x на Windows /api/chat
        может возвращать 502 для локального llama3.2, хотя generate работает.
        """
        try:
            return self._generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                json_mode=json_mode,
            )
        except (urllib.error.URLError, OSError) as e:
            try:
                return self._cli_fallback(system_prompt, user_prompt, json_mode)
            except (OSError, subprocess.SubprocessError) as cli_error:
                raise RuntimeError(
                    f"Не удалось достучаться до Ollama по адресу {self.host}. "
                    f"Проверьте, что `ollama serve` запущен и модель {self.model!r} загружена "
                    f"(`ollama pull {self.model}`). Исходная ошибка: {e}. "
                    f"CLI fallback тоже не сработал: {cli_error}"
                ) from e

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        prompt = (
            "Инструкция:\n"
            f"{system_prompt}\n\n"
            "Данные и задача:\n"
            f"{user_prompt}"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["response"]

    def _cli_fallback(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        """Вызывает `ollama run`, если HTTP API локального Ollama возвращает 502."""

        prompt = (
            "Инструкция:\n"
            f"{system_prompt}\n\n"
            "Данные и задача:\n"
            f"{user_prompt}"
        )
        cmd = [str(self._ollama_executable()), "run", "--nowordwrap"]
        if json_mode:
            cmd.extend(["--format", "json"])
        cmd.extend([self.model, prompt])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"ollama exited with code {result.returncode}")
        return self._strip_ansi(result.stdout).strip()

    @staticmethod
    def _ollama_executable() -> Path:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.exists():
                return candidate
        return Path("ollama")

    @staticmethod
    def _strip_ansi(value: str) -> str:
        return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", value)


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

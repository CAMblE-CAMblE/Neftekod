"""Загрузка пользовательских путей и артефактов модели."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "model"


def read_json(path: str | Path) -> Any:
    """Читает JSON-файл в кодировке UTF-8."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path) -> None:
    """Сохраняет JSON-файл в кодировке UTF-8."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    """Загружает YAML-конфиг или возвращает пустой словарь."""

    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str | Path | None, *, base_dir: str | Path) -> Path | None:
    """Разрешает относительный путь относительно директории конфига."""

    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def model_config_section(model_config: dict[str, Any], name: str) -> dict[str, Any]:
    """Возвращает секцию сохраненного config.json с понятной ошибкой."""

    section = model_config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"В config.json нет секции {name!r}")
    return section

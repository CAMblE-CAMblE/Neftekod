"""Адаптер ответа агента качества к контракту оркестратора."""

from __future__ import annotations

import math
from dataclasses import asdict


def to_orchestrator_quality(row: dict) -> dict:
    """Преобразует строку инференса в JSON качества для оркестратора.

    Вход: словарь из `predict_frame`. Выход: JSON-совместимый словарь с
    `predicted_quality`. Существенное условие: вероятность нарушения и
    уверенность не выдумываются; для старого контракта возвращаются `null` и
    пояснение в assumptions.
    """

    assumptions = list(row.get("assumptions") or [])
    assumptions.append(
        "Совместимость: текущий contracts.py требует spec_violation_risk и confidence, но агент их не оценивает; требуется разрешить null или заменить на status."
    )
    return {
        "predicted_quality": {"sulfur_mg_kg": float(row["predicted_sulfur_mg_kg"])},
        "spec_violation_risk": None,
        "confidence": None,
        "assumptions": assumptions,
    }


def to_quality_assessment(row: dict) -> object:
    """Возвращает объект `QualityAssessment`, если модуль оркестратора доступен.

    Вход: строка инференса. Выход: dataclass оркестратора или dict. Существенное
    условие: при строгом старом контракте с float-полями используется dict,
    чтобы не подставлять произвольные числа.
    """

    data = to_orchestrator_quality(row)
    try:
        from orchestrator.contracts import QualityAssessment
    except ImportError:
        return data
    try:
        return QualityAssessment(**data)
    except TypeError:
        return data

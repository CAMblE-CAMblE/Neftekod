"""
Жёсткие ограничения.

Проверки не доверяются LLM. LLM в оркестраторе используется
только для формулировки объяснения и выбора между уже допустимыми вариантами.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import AgentInputs, OptimizationScenario

MAX_SULFUR_MG_KG = 10.0
MAX_LIMS_AGE_MINUTES = 24 * 60      # TODO: уточнить по регламенту — сейчас допущение "болванки"
STALE_DATA_CONFIDENCE_PENALTY = 0.3


@dataclass
class ConstraintCheckResult:
    scenario_id: str
    passed: bool
    violations: list[str]


def check_scenario(scenario: OptimizationScenario, inputs: AgentInputs) -> ConstraintCheckResult:
    violations: list[str] = []

    sulfur = scenario.expected_quality.get("sulfur_mg_kg")
    if sulfur is not None and sulfur > MAX_SULFUR_MG_KG:
        violations.append(
            f"сера {sulfur:.2f} мг/кг превышает предел {MAX_SULFUR_MG_KG} мг/кг"
        )

    blend_fracs = {
        k: v for k, v in scenario.parameter_changes.items() if k.startswith("blend_")
    }
    if blend_fracs:
        total = sum(blend_fracs.values())
        if abs(total - 1.0) > 1e-3:
            violations.append(f"доли компонентов блендинга дают {total:.3f}, а не 1.0")

    for tag, new_value in scenario.parameter_changes.items():
        bounds = inputs.reliability.optimization_constraints.get(tag)
        if bounds is not None:
            lo, hi = bounds
            if not (lo <= new_value <= hi):
                violations.append(f"{tag}={new_value} вне допустимого диапазона [{lo}, {hi}]")

    if not scenario.within_hard_limits:
        violations.append("агент оптимизации пометил сценарий как выходящий за жёсткие пределы")

    if not inputs.reliability.regime_allowed:
        violations.append("текущий/предлагаемый режим отмечен агентом надёжности как недопустимый")

    return ConstraintCheckResult(
        scenario_id=scenario.scenario_id,
        passed=len(violations) == 0,
        violations=violations,
    )


def filter_admissible_scenarios(
    inputs: AgentInputs,
) -> tuple[list[OptimizationScenario], list[ConstraintCheckResult]]:
    """Возвращает (допустимые сценарии, все результаты проверок)."""
    results = [check_scenario(s, inputs) for s in inputs.optimization.scenarios]
    admissible = [
        s for s, r in zip(inputs.optimization.scenarios, results) if r.passed
    ]
    return admissible, results


def data_is_too_stale(inputs: AgentInputs) -> bool:
    age = inputs.state.lims_age_minutes
    return age is not None and age > MAX_LIMS_AGE_MINUTES


def classify_violation(violation_text: str) -> str:
    """Грубая классификация текста нарушения в категорию для читаемой фразы отказа."""
    t = violation_text.lower()
    if "сера" in t or "quality" in t:
        return "нарушают ограничение по качеству"
    if "диапазон" in t or "вне допустимого" in t:
        return "выходят за заданный модельный диапазон"
    if "блендинг" in t or "доли компонентов" in t:
        return "нарушают требование по долям блендинга"
    if "режим" in t or "надёжности" in t:
        return "отмечены агентом надёжности как недопустимые"
    return "не проходят жёсткие проверки"


def build_refusal_reason(
    *,
    too_stale: bool,
    lims_age_minutes: float | None,
    all_checks: list[ConstraintCheckResult],
) -> str:
    """
    Собирает читаемую причину отказа по образцу из ТЗ (раздел 5):
    «последнее лабораторное значение устарело, а доступные варианты либо
    нарушают ограничение по качеству, либо выходят за заданный модельный
    диапазон».
    """
    parts: list[str] = []

    if too_stale:
        age_str = f" ({lims_age_minutes:.0f} мин назад)" if lims_age_minutes is not None else ""
        parts.append(f"последнее лабораторное значение устарело{age_str}")

    violations = [v for r in all_checks for v in r.violations]
    if violations:
        categories = list(dict.fromkeys(classify_violation(v) for v in violations))  # уникальные, с сохранением порядка
        if len(categories) == 1:
            parts.append(f"доступные варианты {categories[0]}")
        else:
            parts.append("доступные варианты либо " + ", либо ".join(categories))
    elif not all_checks:
        parts.append("ни один вариант не был предложен агентом оптимизации")

    if not parts:
        return "не выполнены минимальные жёсткие проверки (см. ТЗ, раздел 4)"

    return ", а ".join(parts)

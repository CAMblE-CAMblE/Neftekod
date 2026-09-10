"""
Контракты данных между агентами и оркестратором.

Это "болванка": реальные агенты качества/надёжности/оптимизации будут
классическими ML-моделями и появятся позже. Их выходы уже сейчас
описаны как dataclass по таблице из ТЗ (раздел 3), чтобы:
  1) можно было писать и тестировать оркестратор на начальном этапе;
  2) когда появятся реальные агенты, им нужно будет просто вернуть объект
     нужного типа — оркестратор менять не придётся.

Если реальный ответ агента в итоге будет отличаться (другие поля,
другие имена) — нужно поменять dataclass здесь, а не логику оркестратора.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DataFreshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"


@dataclass
class ProcessState:
    """Снимок состояния процесса на момент цикла принятия решения."""
    timestamp: datetime
    tags: dict[str, float]                     # тег -> текущее значение (КИП)
    lims_age_minutes: float | None = None      # возраст последнего ЛИМС
    pak_age_minutes: float | None = None       # возраст последнего ПАК
    data_freshness: DataFreshness = DataFreshness.FRESH
    notes: str = ""


@dataclass
class QualityAssessment:
    """Выход агента качества."""
    predicted_quality: dict[str, float]        # показатель -> прогноз (напр. {"sulfur_mg_kg": 8.2})
    spec_violation_risk: float                 # 0..1
    confidence: float                          # 0..1, уверенность/качество данных
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ReliabilityAssessment:
    """Выход агента надёжности."""
    risk_index: float                           # 0..1 или произвольная шкала — договориться при интеграции
    risk_class: str                             # напр. "low" / "medium" / "high"
    risk_factors: list[str] = field(default_factory=list)
    regime_allowed: bool = True                 # допустимость текущего/предлагаемого режима
    optimization_constraints: dict[str, tuple[float, float]] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)


@dataclass
class OptimizationScenario:
    """
    Один из возможных сценариев изменения режима,
    предложенный агентом оптимизации.
    """
    scenario_id: str
    parameter_changes: dict[str, float]         # тег -> новое значение
    expected_quality: dict[str, float] = field(default_factory=dict)
    expected_throughput_delta: float | None = None
    expected_energy_cost_proxy: float | None = None
    expected_risk_index: float | None = None
    within_hard_limits: bool = True             # агент оптимизации уже отбросил explicit-нарушения
    notes: str = ""


@dataclass
class OptimizationOutput:
    """Выход агента оптимизации."""
    scenarios: list[OptimizationScenario]
    assumptions: list[str] = field(default_factory=list)


@dataclass
class AgentInputs:
    """Все, что оркестратор получает от 3х агентов за один цикл."""
    state: ProcessState
    quality: QualityAssessment
    reliability: ReliabilityAssessment
    optimization: OptimizationOutput


@dataclass
class Recommendation:
    """
    Финальный ответ оператору — по структуре из ТЗ, раздел 5:
    время/состояние, проблема/риск, действие, эффект, проверка ограничений,
    уверенность, объяснение.
    """
    timestamp: datetime
    state_summary: str
    problem_or_risk: str
    action: str | None                          # None, если рекомендаций нет
    expected_effect: str
    constraints_checked: list[str]
    confidence: str
    explanation: str
    is_actionable: bool                          # False -> «надёжной рекомендации нет»
    chosen_scenario_id: str | None = None

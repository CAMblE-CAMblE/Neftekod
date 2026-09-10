"""
Агенты будут независимыми процессами и будут отдавать результат в JSON.
Оркестратор ничего не знает про то, как агент получил свои цифры —
только про согласованный контракт полей.

Если реальный формат в итоге будет отличаться — нужно менять только этот файл
(parse_agent_inputs), контракты и остальной оркестратор не потребуют правок,
пока набор полей в contracts.py покрывает то, что реально приходит.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .contracts import (
    AgentInputs,
    DataFreshness,
    OptimizationOutput,
    OptimizationScenario,
    ProcessState,
    QualityAssessment,
    Recommendation,
    ReliabilityAssessment,
)


def parse_agent_inputs(data: dict) -> AgentInputs:
    """
    Ожидаемая структура:
    {
      "state": {...},
      "quality": {...},
      "reliability": {...},
      "optimization": {"scenarios": [...], "assumptions": [...]}
    }
    """
    s = data["state"]
    state = ProcessState(
        timestamp=datetime.fromisoformat(s["timestamp"]),
        tags=s.get("tags", {}),
        lims_age_minutes=s.get("lims_age_minutes"),
        pak_age_minutes=s.get("pak_age_minutes"),
        data_freshness=DataFreshness(s.get("data_freshness", "fresh")),
        notes=s.get("notes", ""),
    )

    q = data["quality"]
    quality = QualityAssessment(
        predicted_quality=q.get("predicted_quality", {}),
        spec_violation_risk=q["spec_violation_risk"],
        confidence=q["confidence"],
        assumptions=q.get("assumptions", []),
    )

    r = data["reliability"]
    reliability = ReliabilityAssessment(
        risk_index=r["risk_index"],
        risk_class=r["risk_class"],
        risk_factors=r.get("risk_factors", []),
        regime_allowed=r.get("regime_allowed", True),
        optimization_constraints={
            tag: (bounds[0], bounds[1]) for tag, bounds in
            r.get("optimization_constraints", {}).items()
        },
        assumptions=r.get("assumptions", []),
    )

    o = data["optimization"]
    scenarios = [
        OptimizationScenario(
            scenario_id=sc["scenario_id"],
            parameter_changes=sc.get("parameter_changes", {}),
            expected_quality=sc.get("expected_quality", {}),
            expected_throughput_delta=sc.get("expected_throughput_delta"),
            expected_energy_cost_proxy=sc.get("expected_energy_cost_proxy"),
            expected_risk_index=sc.get("expected_risk_index"),
            within_hard_limits=sc.get("within_hard_limits", True),
            notes=sc.get("notes", ""),
        )
        for sc in o.get("scenarios", [])
    ]
    optimization = OptimizationOutput(scenarios=scenarios,
                                      assumptions=o.get("assumptions", []))

    return AgentInputs(state=state, quality=quality, reliability=reliability,
                       optimization=optimization)


def load_agent_inputs(path: str | Path) -> AgentInputs:
    with open(path, encoding="utf-8") as f:
        return parse_agent_inputs(json.load(f))


def load_and_combine(
        *,
        state_path: str | Path,
        quality_path: str | Path,
        reliability_path: str | Path,
        optimization_path: str | Path,
) -> AgentInputs:
    """
    Для случая, когда агенты присылают JSON по отдельности, а не одним
    пакетом. Читает 4 файла и собирает их в структуру, ожидаемую parse_agent_inputs.
    """

    def _load(path: str | Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    combined = {
        "state": _load(state_path),
        "quality": _load(quality_path),
        "reliability": _load(reliability_path),
        "optimization": _load(optimization_path),
    }
    return parse_agent_inputs(combined)


def recommendation_to_json(rec: Recommendation, *, indent: int = 2) -> str:
    return json.dumps(asdict(rec), ensure_ascii=False, indent=indent, default=str)

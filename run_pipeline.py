from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OPTIMIZER_DIR = ROOT / "optimizer"
sys.path.insert(0, str(OPTIMIZER_DIR))
sys.path.insert(0, str(ROOT))

from orchestrator.contracts import (  # noqa: E402
    AgentInputs,
    DataFreshness,
    OptimizationOutput,
    OptimizationScenario,
    ProcessState,
    QualityAssessment,
    ReliabilityAssessment,
    Recommendation,
)
from orchestrator.llm_client import OllamaClient  # noqa: E402
from orchestrator.orchestrator import Orchestrator  # noqa: E402

from scenario_generator import generate_scenarios  # noqa: E402
from quality_evaluator import evaluate_scenarios  # noqa: E402
from energy_proxy import calculate_scenarios_energy  # noqa: E402
from pareto_optimizer import calculate_pareto_front, select_scenario  # noqa: E402

BASE_STATE_PATH = OPTIMIZER_DIR / "examples" / "base_state.csv"
BOUNDS_PATH = OPTIMIZER_DIR / "examples" / "controllable_bounds.csv"
QUALITY_MODEL_DIR = ROOT / "quality_agent_package" / "model"

SCENARIOS_PATH = OPTIMIZER_DIR / "logs" / "scenarios.csv"
QUALITY_RESULTS_PATH = OPTIMIZER_DIR / "logs" / "quality_results.csv"
ENERGY_RESULTS_PATH = OPTIMIZER_DIR / "logs" / "energy_results.csv"

_RELIABILITY_FIELDS = {
    "risk_index",
    "risk_class",
    "risk_factors",
    "regime_allowed",
    "optimization_constraints",
    "assumptions",
}


def build_reliability_assessment(raw: dict) -> ReliabilityAssessment:
    """Переводит сырой dict от ReliabilityAgent.assess(...) в контракт."""
    clean = {k: v for k, v in raw.items() if k in _RELIABILITY_FIELDS}
    constraints = clean.get("optimization_constraints") or {}
    clean["optimization_constraints"] = {k: tuple(v) for k, v in constraints.items()}
    return ReliabilityAssessment(**clean)


def build_ahtung_refusal(timestamp: datetime, reliability: ReliabilityAssessment) -> Recommendation:
    """Отказ — режим недопустим, оптимизатор не запускается"""
    reasons = "; ".join(reliability.risk_factors) or "режим отмечен как недопустимый"
    reason_text = f"Режим недопустим — {reasons}"
    return Recommendation(
        timestamp=timestamp,
        state_summary=f"Данные на {timestamp.isoformat()}",
        problem_or_risk=reason_text,
        action=None,
        expected_effect="—",
        constraints_checked=[reason_text],
        confidence="низкая",
        explanation=reason_text + ". Оптимизатор не запускался.",
        is_actionable=False,
        chosen_scenario_id=None,
    )


def run_pipeline(
    timestamp: datetime,
    reliability_config_path: str | Path | None = None,
    llm_model: str = "llama3.2",
) -> Recommendation:

    # 1. Агент надёжности — один раз, гейт
    print("[1/6] Агент надёжности: проверка режима...", flush=True)
    from reliability.reliability_agent import ReliabilityAgent

    agent = ReliabilityAgent(reliability_config_path)
    reliability_raw = agent.assess(timestamp)
    reliability = build_reliability_assessment(reliability_raw)

    if not reliability.regime_allowed:
        print("[1/6] Режим НЕ допустим — оптимизатор не запускается.")
        return build_ahtung_refusal(timestamp, reliability)
    print(f"[1/6] Режим допустим (risk_class={reliability.risk_class}). Продолжаем.")

    # 2. Сетка сценариев (пишет optimizer/logs/scenarios.csv)
    print("[2/6] Строим сетку сценариев (может занять время при больших N)...", flush=True)
    scenarios = generate_scenarios(
        base_state_path=BASE_STATE_PATH,
        bounds_path=BOUNDS_PATH,
        output_path=SCENARIOS_PATH,
    )
    print(f"[2/6] Готово: {len(scenarios)} сценариев -> {SCENARIOS_PATH}")

    # 3. Качество (реальный CatBoost, пишет quality_results.csv)
    print("[3/6] Агент качества (CatBoost): прогноз серы по всем сценариям...", flush=True)
    quality_results = evaluate_scenarios(
        model_dir=QUALITY_MODEL_DIR,
        base_state_path=BASE_STATE_PATH,
        scenarios_path=SCENARIOS_PATH,
        bounds_path=BOUNDS_PATH,
        output_path=QUALITY_RESULTS_PATH,
    )
    n_feasible = int(quality_results["quality_feasible"].sum())
    print(f"[3/6] Готово: {n_feasible} из {len(quality_results)} проходят ограничение по сере")

    # 4. Энергия (пишет energy_results.csv)
    print("[4/6] Считаем энергетический proxy по сценариям...", flush=True)
    energy_results = calculate_scenarios_energy(
        base_state_path=BASE_STATE_PATH,
        scenarios_path=SCENARIOS_PATH,
        bounds_path=BOUNDS_PATH,
        output_path=ENERGY_RESULTS_PATH,
    )
    print(f"[4/6] Готово: {len(energy_results)} сценариев -> {ENERGY_RESULTS_PATH}")

    # 5. Pareto-отбор
    print("[5/6] Строим Pareto front и выбираем сценарий...", flush=True)
    merged = quality_results.merge(energy_results, on="candidate_id", how="inner", suffixes=("", "_energy"))
    pareto_all = calculate_pareto_front(merged)
    selected = select_scenario(pareto_all).iloc[0]
    n_pareto = int(pareto_all["pareto_optimal"].sum())
    print(
        f"[5/6] Готово: {n_pareto} на Pareto front, выбран candidate_id={selected['candidate_id']} "
        f"(сера={selected['predicted_sulfur_mg_kg']:.2f} мг/кг, энергия={selected['energy_proxy_eu_h']:.1f} EU/ч)"
    )

    # 6. LLM объясняет выбранный сценарий
    print("[6/6] LLM формулирует объяснение для оператора (может занять до нескольких минут)...", flush=True)
    bounds = pd.read_csv(BOUNDS_PATH)
    params = bounds["parameter"].tolist()
    base_state_df = pd.read_csv(BASE_STATE_PATH)
    base_row = base_state_df.iloc[0].to_dict()

    # Прогноз для текущего режима без изменений
    # Нужен для LLM, чтобы сравнивать "было vs стало"
    from quality_agent.inference import load_bundle, predict_base_state

    bundle = load_bundle(QUALITY_MODEL_DIR)
    baseline_prediction = predict_base_state(bundle, base_state_df)
    baseline_sulfur = float(baseline_prediction.iloc[0]["predicted_sulfur_mg_kg"])

    process_state = ProcessState(
        timestamp=timestamp,
        # LLM получает только регулируемые параметры сценария, а не все
        # 100+ тегов телеметрии — иначе промпт становится огромным
        tags={p: base_row[p] for p in params},
        data_freshness=DataFreshness.FRESH,
    )
    quality = QualityAssessment(
        predicted_quality={"sulfur_mg_kg": float(selected["predicted_sulfur_mg_kg"])},
        spec_violation_risk=0.0,
        confidence=0.0,
        assumptions=[
            f"Текущий режим без изменений (baseline): сера = {baseline_sulfur:.2f} мг/кг",
            "spec_violation_risk/confidence: not_estimated агентом качества",
        ],
    )
    chosen_scenario = OptimizationScenario(
        scenario_id=str(selected["candidate_id"]),
        parameter_changes={p: float(selected[p]) for p in params},
        expected_quality={"sulfur_mg_kg": float(selected["predicted_sulfur_mg_kg"])},
        expected_energy_cost_proxy=float(selected["energy_proxy_eu_h"]),
        within_hard_limits=True,
        notes="выбран Pareto-оптимизатором",
    )
    inputs = AgentInputs(
        state=process_state,
        quality=quality,
        reliability=reliability,
        optimization=OptimizationOutput(scenarios=[chosen_scenario]),
    )

    orchestrator = Orchestrator(llm=OllamaClient(model=llm_model))
    recommendation = orchestrator.run_cycle(inputs)

    # spec_violation_risk/confidence агента качества сейчас not_estimated
    from dataclasses import replace

    return replace(
        recommendation,
        confidence="не оценена (агент качества: spec_violation_risk/confidence — not_estimated)",
    )


if __name__ == "__main__":
    import json
    from dataclasses import asdict

    ts = datetime(2025, 7, 16, 6, 30, 0)
    recommendation = run_pipeline(ts)
    print("\nГОТОВО\n")
    print(json.dumps(asdict(recommendation), ensure_ascii=False, indent=2, default=str))
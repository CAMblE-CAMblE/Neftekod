from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
OPTIMIZER_DIR = ROOT / "optimizer"
QUALITY_AGENT_PACKAGE = ROOT / "quality_agent_package"
sys.path.insert(0, str(OPTIMIZER_DIR))
sys.path.insert(0, str(QUALITY_AGENT_PACKAGE))
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

from scenario_generator import N_STEPS, STEP_FRACTION, generate_scenarios  # noqa: E402
from quality_evaluator import evaluate_scenarios  # noqa: E402
from energy_proxy import calculate_scenarios_energy  # noqa: E402
from pareto_optimizer import calculate_pareto_front, select_scenario  # noqa: E402

BASE_STATE_PATH = OPTIMIZER_DIR / "examples" / "base_state.csv"
BOUNDS_PATH = OPTIMIZER_DIR / "examples" / "controllable_bounds.csv"
QUALITY_MODEL_DIR = ROOT / "quality_agent_package" / "model"
QUALITY_CONFIG_PATH = ROOT / "quality" / "configs" / "quality_agent.yaml"

SCENARIOS_PATH = OPTIMIZER_DIR / "logs" / "scenarios.csv"
QUALITY_RESULTS_PATH = OPTIMIZER_DIR / "logs" / "quality_results.csv"
ENERGY_RESULTS_PATH = OPTIMIZER_DIR / "logs" / "energy_results.csv"

DATASET_ENV_VAR = "NEFTEKOD_QUALITY_DATASET"
TIMESTAMP_COLUMNS = ("state_time", "date", "timestamp", "time")

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


def load_quality_config(config_path: str | Path = QUALITY_CONFIG_PATH) -> dict[str, Any]:
    """Загружает конфиг агента качества для UI и CLI-обвязки."""

    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_dataset_path(
    dataset_path: str | Path | None = None,
    config_path: str | Path = QUALITY_CONFIG_PATH,
) -> Path:
    """Возвращает путь к подготовленному датасету без привязки к машине пользователя."""

    if dataset_path is None:
        dataset_path = os.getenv(DATASET_ENV_VAR)
    if dataset_path is None:
        config = load_quality_config(config_path)
        dataset_path = config.get("data", {}).get("prepared_path")
    if not dataset_path:
        raise ValueError(
            f"Не задан путь к датасету: укажите data.prepared_path или {DATASET_ENV_VAR}."
        )

    path = Path(dataset_path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_state_dataset(
    dataset_path: str | Path | None = None,
    config_path: str | Path = QUALITY_CONFIG_PATH,
) -> pd.DataFrame:
    """Читает подготовленный датасет состояний и нормализует колонку state_time."""

    path = resolve_dataset_path(dataset_path, config_path)
    if not path.exists():
        raise FileNotFoundError(f"Датасет состояний не найден: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Неподдерживаемый формат датасета: {path.suffix}")

    time_col = _find_timestamp_column(frame)
    if time_col != "state_time":
        frame = frame.copy()
        frame["state_time"] = frame[time_col]
    frame["state_time"] = pd.to_datetime(frame["state_time"])
    return frame.sort_values("state_time").reset_index(drop=True)


def available_state_times(frame: pd.DataFrame) -> list[pd.Timestamp]:
    """Возвращает доступные временные метки из подготовленного датасета."""

    if "state_time" not in frame.columns:
        raise ValueError("В датасете нет колонки state_time")
    return sorted(pd.to_datetime(frame["state_time"]).dropna().drop_duplicates().to_list())


def get_state_by_timestamp(frame: pd.DataFrame, timestamp: datetime | pd.Timestamp | str) -> pd.Series:
    """Возвращает полную строку состояния для точной временной метки."""

    if "state_time" not in frame.columns:
        raise ValueError("В датасете нет колонки state_time")
    selected_time = pd.Timestamp(timestamp)
    mask = pd.to_datetime(frame["state_time"]) == selected_time
    if not mask.any():
        raise ValueError(f"В датасете нет состояния для времени {selected_time}")
    return frame.loc[mask].iloc[0].copy()


def load_quality_bundle(model_dir: str | Path = QUALITY_MODEL_DIR):
    """Загружает рабочую модель качества для повторного инференса."""

    from quality_agent.inference import load_bundle

    return load_bundle(model_dir)


def predict_current_quality(
    base_state: pd.DataFrame | pd.Series | dict[str, Any],
    model_dir: str | Path = QUALITY_MODEL_DIR,
    bundle=None,
) -> pd.DataFrame:
    """Считает прогноз текущей серы существующим инференсом агента качества."""

    from quality_agent.inference import load_bundle, predict_base_state

    active_bundle = bundle or load_bundle(model_dir)
    return predict_base_state(active_bundle, _base_state_frame(base_state))


def predict_current_sulfur(
    base_state: pd.DataFrame | pd.Series | dict[str, Any],
    model_dir: str | Path = QUALITY_MODEL_DIR,
    bundle=None,
) -> float:
    """Возвращает прогноз серы для выбранного текущего режима."""

    prediction = predict_current_quality(base_state, model_dir=model_dir, bundle=bundle)
    return float(prediction.iloc[0]["predicted_sulfur_mg_kg"])


def estimate_scenario_count(
    base_state: pd.DataFrame | pd.Series | dict[str, Any],
    bounds_path: str | Path = BOUNDS_PATH,
) -> int:
    """Оценивает размер сетки сценариев теми же правилами, что и оптимизатор."""

    base_row = _base_state_frame(base_state).iloc[0].to_dict()
    bounds = pd.read_csv(bounds_path)
    count = 1
    for _, row in bounds.iterrows():
        parameter = row["parameter"]
        if parameter not in base_row:
            raise ValueError(f"Parameter '{parameter}' not found in base_state.csv")

        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        current = float(base_row[parameter])
        if not lower <= current <= upper:
            raise ValueError(f"{parameter}: current value {current} is outside [{lower}, {upper}]")

        step = (upper - lower) * STEP_FRACTION
        if step == 0:
            value_count = 1
        else:
            values = [
                max(lower, min(upper, current + i * step))
                for i in range(-N_STEPS, N_STEPS + 1)
            ]
            value_count = len(set(values))
        count *= value_count
    return count


def estimate_pipeline_runtime_minutes(
    base_state: pd.DataFrame | pd.Series | dict[str, Any],
    bounds_path: str | Path = BOUNDS_PATH,
) -> int:
    """Возвращает грубую оценку времени расчета по размеру сетки сценариев."""

    scenario_count = estimate_scenario_count(base_state, bounds_path)
    return max(1, int((2 + scenario_count / 25_000) + 0.999))


def format_recommendation_text(recommendation: Recommendation) -> str:
    """Формирует полный операторский текст из ответа оркестратора."""

    action = recommendation.action or "надежной рекомендации нет"
    parts = [
        f"Состояние: {recommendation.state_summary}",
        f"Риск: {recommendation.problem_or_risk}",
        f"Действие: {action}",
        f"Ожидаемый эффект: {recommendation.expected_effect}",
        f"Уверенность: {recommendation.confidence}",
        f"Объяснение: {recommendation.explanation}",
    ]
    if recommendation.chosen_scenario_id:
        parts.append(f"Сценарий: {recommendation.chosen_scenario_id}")
    return "\n\n".join(part for part in parts if part.strip())


def _find_timestamp_column(frame: pd.DataFrame) -> str:
    for column in TIMESTAMP_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(
        "В датасете нет временной колонки. Ожидается одна из: "
        + ", ".join(TIMESTAMP_COLUMNS)
    )


def _base_state_frame(base_state: pd.DataFrame | pd.Series | dict[str, Any]) -> pd.DataFrame:
    if isinstance(base_state, pd.DataFrame):
        frame = base_state.copy()
    elif isinstance(base_state, pd.Series):
        frame = pd.DataFrame([base_state.to_dict()])
    else:
        frame = pd.DataFrame([dict(base_state)])
    if len(frame) != 1:
        raise ValueError("base_state должен содержать ровно одну строку")
    if "state_time" not in frame.columns:
        raise ValueError("В выбранном состоянии нет колонки state_time")
    return frame


@contextmanager
def _base_state_path(
    base_state: pd.DataFrame | pd.Series | dict[str, Any] | None,
    base_state_path: str | Path | None,
):
    if base_state is not None and base_state_path is not None:
        raise ValueError("Передайте только один источник базового состояния: base_state или base_state_path")
    if base_state is None:
        yield Path(base_state_path) if base_state_path is not None else BASE_STATE_PATH
        return

    with tempfile.TemporaryDirectory(prefix="neftekod_base_state_") as tmp_dir:
        path = Path(tmp_dir) / "base_state.csv"
        _base_state_frame(base_state).to_csv(path, index=False)
        yield path


def run_pipeline(
    timestamp: datetime,
    reliability_config_path: str | Path | None = None,
    llm_model: str = "llama3.2",
    base_state: pd.DataFrame | pd.Series | dict[str, Any] | None = None,
    base_state_path: str | Path | None = None,
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

    SCENARIOS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _base_state_path(base_state, base_state_path) as active_base_state_path:

        # 2. Сетка сценариев (пишет optimizer/logs/scenarios.csv)
        print("[2/6] Строим сетку сценариев (может занять время при больших N)...", flush=True)
        scenarios = generate_scenarios(
            base_state_path=active_base_state_path,
            bounds_path=BOUNDS_PATH,
            output_path=SCENARIOS_PATH,
        )
        print(f"[2/6] Готово: {len(scenarios)} сценариев -> {SCENARIOS_PATH}")

        # 3. Качество (реальный CatBoost, пишет quality_results.csv)
        print("[3/6] Агент качества (CatBoost): прогноз серы по всем сценариям...", flush=True)
        quality_results = evaluate_scenarios(
            model_dir=QUALITY_MODEL_DIR,
            base_state_path=active_base_state_path,
            scenarios_path=SCENARIOS_PATH,
            bounds_path=BOUNDS_PATH,
            output_path=QUALITY_RESULTS_PATH,
        )
        n_feasible = int(quality_results["quality_feasible"].sum())
        print(f"[3/6] Готово: {n_feasible} из {len(quality_results)} проходят ограничение по сере")

        # 4. Энергия (пишет energy_results.csv)
        print("[4/6] Считаем энергетический proxy по сценариям...", flush=True)
        energy_results = calculate_scenarios_energy(
            base_state_path=active_base_state_path,
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
        base_state_df = pd.read_csv(active_base_state_path)
        base_row = base_state_df.iloc[0].to_dict()

        # Прогноз для текущего режима без изменений
        # Нужен для LLM, чтобы сравнивать "было vs стало"
        baseline_sulfur = predict_current_sulfur(base_state_df)

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

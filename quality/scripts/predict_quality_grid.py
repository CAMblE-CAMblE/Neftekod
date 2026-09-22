"""Диагностическая сетка сценариев для агента качества."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from quality_agent.data import read_table  # noqa: E402
from quality_agent.inference import load_bundle, predict_diagnostic_grid  # noqa: E402


def main() -> None:
    """Разбирает аргументы и сохраняет диагностический CSV сетки.

    Вход: каталог модели, небольшой файл состояний и диапазоны T6/F9/P13.
    Выход: CSV с кандидатами. Существенное условие: сценарный расчет
    помечается как диагностика реакции регрессии, а не как рекомендация.
    """

    parser = argparse.ArgumentParser(description="Диагностическая сетка T6/F9/P13 для агента качества")
    parser.add_argument("--model-dir", default="artifacts/quality_agent/first_quality_current")
    parser.add_argument("--input", default="examples/quality_agent/first_quality_current/sample_states.csv")
    parser.add_argument("--output", default="examples/quality_agent/first_quality_current/grid_predictions.csv")
    parser.add_argument("--state-time", default=None, help="Время исходного состояния, например 2026-02-01 12:00:00")
    parser.add_argument("--state-index", type=int, default=0, help="Индекс строки, если --state-time не задан")
    parser.add_argument("--input-sulfur", type=float, required=True, help="Сценарная входящая сера, мг/кг")
    parser.add_argument("--t6-values", required=True, help="Список T6 через запятую")
    parser.add_argument("--f9-values", required=True, help="Список F9 через запятую")
    parser.add_argument("--p13-values", required=True, help="Список P13 через запятую")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bundle = load_bundle(args.model_dir)
    states = read_table(args.input)
    base = select_state(states, args.state_time, args.state_index)
    result = predict_diagnostic_grid(
        bundle,
        base,
        scenario_input_sulfur_mg_kg=args.input_sulfur,
        t6_values=parse_float_list(args.t6_values),
        f9_values=parse_float_list(args.f9_values),
        p13_values=parse_float_list(args.p13_values),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    logging.info("Диагностическая сетка сохранена: %s", output)


def parse_float_list(value: str) -> list[float]:
    """Преобразует строку `1,2,3` в список чисел."""

    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("Список значений не должен быть пустым")
    return result


def select_state(states: pd.DataFrame, state_time: str | None, state_index: int) -> pd.Series:
    """Выбирает исходное состояние по времени или индексу."""

    if state_time:
        target = pd.Timestamp(state_time)
        matched = states[pd.to_datetime(states["state_time"]) == target]
        if matched.empty:
            raise ValueError(f"Не найдено состояние state_time={state_time}")
        return matched.iloc[0]
    if state_index < 0 or state_index >= len(states):
        raise ValueError(f"state-index вне диапазона 0..{len(states) - 1}")
    return states.iloc[state_index]


if __name__ == "__main__":
    main()

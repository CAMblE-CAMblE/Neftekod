"""Streamlit-интерфейс демонстрационного симулятора гидроочистки."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulator.scenario import (  # noqa: E402
    evaluate_baseline,
    load_demo_states,
    load_simulator_config,
    run_recommendation,
    run_time_simulation,
)
from simulator.state import CandidateAction, ControlSettings, HistoricalState, Recommendation, SimulationResult  # noqa: E402
from simulator.visualization import sulfur_trajectory_chart  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "simulator.yaml"
STATES_PATH = ROOT / "data" / "simulator" / "demo_states.yaml"


def main() -> None:
    """Запускает одностраничный интерфейс симулятора."""

    st.set_page_config(page_title="Система расчета регулирования работы ABT", layout="wide")
    st.title("Система расчета регулирования работы ABT")
    st.caption("Данные, модель")

    base_config = load_simulator_config(CONFIG_PATH)
    states = load_demo_states(STATES_PATH)
    _ensure_session_state(states[0].id, states[0].sulfur_in)

    with st.sidebar:
        st.header("Сценарий")
        selected_id = st.selectbox(
            "Исходный режим",
            options=[state.id for state in states],
            format_func=lambda value: _state_by_id(states, value).name,
            key="selected_state_id",
        )
        selected_state = _state_by_id(states, selected_id)
        if st.session_state.active_state_id != selected_id:
            st.session_state.active_state_id = selected_id
            st.session_state.input_sulfur = float(selected_state.sulfur_in)
            st.session_state.baseline_quality = None
            st.session_state.recommendation = None
            st.session_state.candidates = []
            st.session_state.simulation_result = None
        input_sulfur = st.slider(
            "Входящая сера, мг/кг",
            min_value=300.0,
            max_value=900.0,
            value=float(st.session_state.get("input_sulfur", selected_state.sulfur_in)),
            step=10.0,
            key="input_sulfur",
        )
        sulfur_limit = st.number_input(
            "Заданный предел серы, мг/кг",
            min_value=1.0,
            max_value=30.0,
            value=float(base_config.sulfur_limit_mg_kg),
            step=0.5,
            key="sulfur_limit",
        )
        config = replace(base_config, sulfur_limit_mg_kg=float(sulfur_limit))
        _invalidate_when_inputs_changed(selected_state.id, input_sulfur, sulfur_limit)

        if st.button("Оценить изменение сырья", use_container_width=True):
            st.session_state.baseline_quality = evaluate_baseline(selected_state, input_sulfur, config)
            st.session_state.simulation_result = None

        if st.button("Подобрать режим", use_container_width=True):
            st.session_state.baseline_quality = evaluate_baseline(selected_state, input_sulfur, config)
            recommendation, candidates = run_recommendation(selected_state, input_sulfur, config)
            st.session_state.recommendation = recommendation
            st.session_state.candidates = candidates
            st.session_state.simulation_result = None

        if st.button("Применить рекомендацию", use_container_width=True):
            recommendation = st.session_state.get("recommendation")
            if recommendation and recommendation.candidate:
                st.session_state.simulation_result = run_time_simulation(
                    selected_state,
                    input_sulfur,
                    recommendation.candidate.controls,
                    config,
                )
            else:
                st.warning("Сначала подберите допустимый режим.")

        if st.button("Сбросить сценарий", use_container_width=True):
            _reset_session(selected_state.id, selected_state.sulfur_in)
            st.rerun()

    _render_state_summary(selected_state)
    _render_metrics(selected_state, config.sulfur_limit_mg_kg)
    _render_controls_table(selected_state, st.session_state.get("recommendation"))
    _render_chart(st.session_state.get("simulation_result"), config.sulfur_limit_mg_kg)
    _render_decision_block(st.session_state.get("recommendation"), st.session_state.get("simulation_result"))


def _ensure_session_state(default_state_id: str, default_sulfur: float) -> None:
    """Инициализирует ключи состояния Streamlit."""

    defaults = {
        "selected_state_id": default_state_id,
        "active_state_id": default_state_id,
        "input_sulfur": float(default_sulfur),
        "last_signature": None,
        "baseline_quality": None,
        "recommendation": None,
        "candidates": [],
        "simulation_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _invalidate_when_inputs_changed(state_id: str, input_sulfur: float, sulfur_limit: float) -> None:
    """Сбрасывает устаревшие расчеты после изменения режима или сырья."""

    signature = (state_id, float(input_sulfur), float(sulfur_limit))
    if st.session_state.get("last_signature") is None:
        st.session_state.last_signature = signature
        return
    if st.session_state.last_signature != signature:
        st.session_state.baseline_quality = None
        st.session_state.recommendation = None
        st.session_state.candidates = []
        st.session_state.simulation_result = None
        st.session_state.last_signature = signature


def _reset_session(state_id: str, input_sulfur: float) -> None:
    """Возвращает сценарий к начальному состоянию."""

    st.session_state.input_sulfur = float(input_sulfur)
    st.session_state.active_state_id = state_id
    st.session_state.last_signature = (state_id, float(input_sulfur), st.session_state.get("sulfur_limit"))
    st.session_state.baseline_quality = None
    st.session_state.recommendation = None
    st.session_state.candidates = []
    st.session_state.simulation_result = None


def _state_by_id(states: list[HistoricalState], state_id: str) -> HistoricalState:
    """Находит исходное состояние по идентификатору."""

    for state in states:
        if state.id == state_id:
            return state
    raise ValueError(f"Неизвестное состояние: {state_id}")


def _render_state_summary(state: HistoricalState) -> None:
    """Показывает исходные данные выбранного режима."""

    with st.expander("Исходное состояние", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Время", state.timestamp)
        cols[1].metric("Входящая сера", f"{state.sulfur_in:.0f} мг/кг")
        cols[2].metric("T6, температура ГСС на входе", f"{state.controls.t6:.1f}")
        cols[3].metric("F9, расход сырья на установку, массовый", f"{state.controls.f9:.1f}")
        cols[4].metric("P13, давление на входе", f"{state.controls.p13:.2f}")


def _render_metrics(state: HistoricalState, sulfur_limit: float) -> None:
    """Выводит основные показатели сценария."""

    baseline_quality = st.session_state.get("baseline_quality")
    recommendation: Recommendation | None = st.session_state.get("recommendation")
    recommended_q21 = recommendation.candidate.quality.predicted_q21 if recommendation and recommendation.candidate else None

    cols = st.columns(4)
    cols[0].metric("Исходная выходная сера", f"{state.q21:.2f} мг/кг")
    cols[1].metric(
        "Новое сырье, текущий режим",
        _format_q21(baseline_quality.predicted_q21 if baseline_quality else None),
    )
    cols[2].metric("После рекомендации", _format_q21(recommended_q21))
    cols[3].metric("Заданный предел", f"{sulfur_limit:.2f} мг/кг")


def _render_controls_table(state: HistoricalState, recommendation: Recommendation | None) -> None:
    """Показывает сравнение исходного режима и рекомендации."""

    recommended = recommendation.candidate.controls if recommendation and recommendation.candidate else None
    rows = []
    for label, original, new_value in [
        ("T6", state.controls.t6, recommended.t6 if recommended else None),
        ("F9", state.controls.f9, recommended.f9 if recommended else None),
        ("P13", state.controls.p13, recommended.p13 if recommended else None),
    ]:
        rows.append(
            {
                "Параметр": label,
                "Исходный режим": original,
                "Рекомендация": new_value,
                "Изменение": None if new_value is None else new_value - original,
            }
        )
    st.subheader("Сравнение режима")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_chart(result: SimulationResult | None, sulfur_limit: float) -> None:
    """Показывает график временной симуляции или подсказку."""

    st.subheader("Временная симуляция")
    if result is None:
        st.info("Нажмите «Применить рекомендацию», чтобы построить траектории на 3 часа.")
        return
    st.plotly_chart(sulfur_trajectory_chart(result, sulfur_limit), use_container_width=True)


def _render_decision_block(recommendation: Recommendation | None, result: SimulationResult | None) -> None:
    """Показывает результат решения и проверки агентов."""

    st.subheader("Результат решения")
    if recommendation is None:
        st.info("Решение еще не сформировано.")
        return

    candidate: CandidateAction | None = recommendation.candidate
    if candidate is None:
        st.error(recommendation.status)
        st.write(recommendation.explanation)
        return

    st.success(recommendation.status)
    st.write(recommendation.explanation)
    st.write(f"Проверка качества: Q21 = {candidate.quality.predicted_q21:.2f} мг/кг.")
    st.write("Проверка оборудования: " + " ".join(candidate.reliability.reasons))
    if result:
        col1, col2 = st.columns(2)
        col1.metric("Время выше предела: исходный режим", f"{result.baseline_time_above_limit_minutes} мин")
        col2.metric("Время выше предела: рекомендация", f"{result.recommended_time_above_limit_minutes} мин")


def _format_q21(value: float | None) -> str:
    """Форматирует Q21 для метрик интерфейса."""

    if value is None:
        return "не рассчитано"
    return f"{value:.2f} мг/кг"


if __name__ == "__main__":
    main()

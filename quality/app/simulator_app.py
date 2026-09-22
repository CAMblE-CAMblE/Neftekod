"""Streamlit-интерфейс запуска готового пайплайна оптимизации."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import (  # noqa: E402
    DATASET_ENV_VAR,
    available_state_times,
    estimate_pipeline_runtime_minutes,
    estimate_scenario_count,
    format_recommendation_text,
    get_state_by_timestamp,
    load_quality_bundle,
    load_state_dataset,
    predict_current_quality,
    resolve_dataset_path,
    run_pipeline,
)

LOG_PATH = PROJECT_ROOT / "optimizer" / "logs" / "ui_errors.log"
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Запускает простой экран выбора состояния и оптимизации."""

    st.set_page_config(page_title="Оптимизация гидроочистки", layout="centered")
    _inject_styles()
    _configure_logging()
    _ensure_session_state()

    st.title("Оптимизация работы ABT и Гидроочистки")

    dataset_path = _resolve_dataset_path_for_ui()
    if dataset_path is None:
        return

    try:
        dataset = _load_dataset_cached(str(dataset_path), dataset_path.stat().st_mtime)
        bundle = _load_bundle_cached()
        times = available_state_times(dataset)
    except Exception as exc:
        LOGGER.exception("Не удалось подготовить данные UI")
        st.error(f"Не удалось загрузить данные: {exc}")
        return

    if not times:
        st.error("В датасете нет доступных временных меток.")
        return

    st.subheader("Выбор состояния системы")
    selected_timestamp = _render_time_selector(times, disabled=st.session_state.optimization_running)
    _clear_result_when_timestamp_changed(selected_timestamp)

    st.caption(f"Выбранная временная метка: {selected_timestamp:%Y-%m-%d %H:%M:%S}")

    state = None
    prediction = None
    estimated_minutes = None
    try:
        state = get_state_by_timestamp(dataset, selected_timestamp)
        prediction = predict_current_quality(state, bundle=bundle)
        sulfur = float(prediction.iloc[0]["predicted_sulfur_mg_kg"])
        st.metric("Прогноз серы для текущего состояния, мг/кг", f"{sulfur:.2f}")
        scenario_count = estimate_scenario_count(state)
        estimated_minutes = estimate_pipeline_runtime_minutes(state)
        scenario_count_text = f"{scenario_count:,}".replace(",", " ")
        st.caption(
            f"Оценка расчета: около {_format_minutes(estimated_minutes)} "
            f"для {scenario_count_text} сценариев."
        )
        warnings = prediction.iloc[0].get("missing_or_stale_inputs") or []
        if warnings:
            st.warning("Прогноз рассчитан с предупреждениями: " + "; ".join(warnings))
    except Exception as exc:
        LOGGER.exception("Не удалось рассчитать текущий прогноз серы")
        st.error(f"Выбранное состояние невозможно обработать: {exc}")

    can_run = state is not None and prediction is not None and not st.session_state.optimization_running
    if st.button("Оптимизировать", type="primary", disabled=not can_run, use_container_width=True):
        _run_optimization(selected_timestamp, state, estimated_minutes)

    st.subheader("Рекомендация")
    if st.session_state.recommendation_text:
        st.text_area(
            "Итоговый текст LLM",
            value=st.session_state.recommendation_text,
            height=280,
            label_visibility="collapsed",
        )
    elif st.session_state.error_message:
        st.error(st.session_state.error_message)
    else:
        st.info("Рекомендация появится после завершения расчета.")


def _configure_logging() -> None:
    """Настраивает файл ошибок UI один раз за процесс."""

    if LOGGER.handlers:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _inject_styles() -> None:
    """Добавляет минимальные стили для основного действия."""

    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #0ea5e9;
            border-color: #0284c7;
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #0284c7;
            border-color: #0369a1;
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:focus {
            box-shadow: 0 0 0 0.2rem rgba(14, 165, 233, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_session_state() -> None:
    """Инициализирует состояние экрана Streamlit."""

    defaults = {
        "selected_timestamp": None,
        "recommendation_text": None,
        "error_message": None,
        "optimization_running": False,
        "optimization_timestamp": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _resolve_dataset_path_for_ui() -> Path | None:
    """Показывает понятную ошибку, если путь к датасету не настроен."""

    try:
        return resolve_dataset_path()
    except Exception as exc:
        LOGGER.exception("Не удалось определить путь к датасету")
        st.error(f"Не удалось определить путь к датасету: {exc}")
        st.info(
            "Укажите путь в quality/configs/quality_agent.yaml -> data.prepared_path "
            f"или переменной окружения {DATASET_ENV_VAR}."
        )
        return None


@st.cache_data(show_spinner=False)
def _load_dataset_cached(path: str, mtime: float) -> pd.DataFrame:
    """Кеширует подготовленный датасет между действиями UI."""

    return load_state_dataset(path)


@st.cache_resource(show_spinner=False)
def _load_bundle_cached():
    """Кеширует CatBoost bundle между действиями UI."""

    return load_quality_bundle()


def _render_time_selector(times: list[pd.Timestamp], disabled: bool) -> pd.Timestamp:
    """Рисует выбор даты и времени из существующих меток датасета."""

    dates = sorted({item.date() for item in times})
    selected_date = st.selectbox(
        "Дата",
        options=dates,
        format_func=lambda value: value.strftime("%Y-%m-%d"),
        disabled=disabled,
    )

    day_times = [item for item in times if item.date() == selected_date]
    selected_time = st.selectbox(
        "Время",
        options=day_times,
        format_func=lambda value: value.strftime("%H:%M:%S"),
        disabled=disabled,
    )
    return pd.Timestamp(selected_time)


def _clear_result_when_timestamp_changed(selected_timestamp: pd.Timestamp) -> None:
    """Очищает старую рекомендацию при смене временной метки."""

    if st.session_state.selected_timestamp == selected_timestamp:
        return
    st.session_state.selected_timestamp = selected_timestamp
    st.session_state.recommendation_text = None
    st.session_state.error_message = None
    st.session_state.optimization_timestamp = None


def _run_optimization(
    selected_timestamp: pd.Timestamp,
    state: pd.Series,
    estimated_minutes: int | None,
) -> None:
    """Запускает пайплайн для выбранной строки датасета."""

    st.session_state.optimization_running = True
    st.session_state.optimization_timestamp = selected_timestamp
    st.session_state.recommendation_text = None
    st.session_state.error_message = None

    try:
        spinner_text = "Выполняется расчет..."
        if estimated_minutes is not None:
            spinner_text = f"Выполняется расчет... расчет займет около {_format_minutes(estimated_minutes)}."
        with st.spinner(spinner_text):
            recommendation = run_pipeline(
                selected_timestamp.to_pydatetime(),
                base_state=state.copy(),
            )
        st.session_state.recommendation_text = format_recommendation_text(recommendation)
    except Exception as exc:
        LOGGER.exception("Ошибка запуска пайплайна из UI")
        st.session_state.error_message = f"Расчет не выполнен: {exc}. Подробности записаны в {LOG_PATH}."
    finally:
        st.session_state.optimization_running = False


def _format_minutes(minutes: int) -> str:
    """Форматирует оценку времени расчета для интерфейса."""

    if 11 <= minutes % 100 <= 14:
        suffix = "минут"
    elif minutes % 10 == 1:
        suffix = "минуту"
    elif 2 <= minutes % 10 <= 4:
        suffix = "минуты"
    else:
        suffix = "минут"
    return f"{minutes} {suffix}"


if __name__ == "__main__":
    main()

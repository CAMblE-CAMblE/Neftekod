"""Инференс агента качества и проверка сценарных вариантов режима."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import QualityAgentConfig
from .data import build_base_frame, forecast_horizon
from .features import prepare_features
from .model import load_model_bundle


CONTROL_PARAMETER_INFO = {
    "hdt_T6": {
        "name": "температура на входе реактора",
        "unit": "°C",
        "note": "управляющий параметр сценарной диагностики",
    },
    "hdt_F9": {
        "name": "массовый расход сырья",
        "unit": "т/ч",
        "note": "управляющий параметр сценарной диагностики; единицу нужно подтвердить по промышленному справочнику тегов",
    },
    "hdt_P13": {
        "name": "давление на входе реактора",
        "unit": "МПа",
        "note": "управляющий параметр сценарной диагностики",
    },
    "avt_T6": {
        "name": "Температура низа К-1",
        "unit": "°C",
        "note": "АВТ-6; справочник `теги АВТ_24-2000.xlsx`, лист `АВТ`.",
    },
    "avt_F7": {
        "name": "Расход обессоленной нефти, 3-й ход в Т-4/2",
        "unit": "т/ч",
        "note": "АВТ-6; участвует в AVT6:240-350:T50.",
    },
    "avt_F9": {
        "name": "Расход обессоленной нефти, 2-й ход в Т-10/2",
        "unit": "т/ч",
        "note": "АВТ-6; хранится отдельно от hdt_F9.",
    },
}


@dataclass
class QualityModelBundle:
    """Сохраненный комплект модели для инференса.

    Вход: модель CatBoost, конфигурация, признаки и метаданные. Выход:
    объект, достаточный для пакетного прогноза. Существенное условие:
    порядок признаков берется из артефакта обучения.
    """

    model: object
    config: QualityAgentConfig
    feature_names: list[str]
    metadata: dict[str, object]


def load_bundle(run_dir: str | Path) -> QualityModelBundle:
    """Загружает комплект модели качества.

    Вход: каталог запуска. Выход: `QualityModelBundle`. Существенное условие:
    отсутствие любого обязательного файла считается ошибкой инференса.
    """

    model, config, feature_names, metadata = load_model_bundle(run_dir)
    return QualityModelBundle(model=model, config=config, feature_names=feature_names, metadata=metadata)


def predict_frame(frame: pd.DataFrame, bundle: QualityModelBundle) -> pd.DataFrame:
    """Выполняет пакетный прогноз по подготовленным состояниям.

    Вход: канонический датафрейм со `state_time` и сигналами. Выход: таблица
    ответа агента с прогнозом, сравнением с пределом и диагностикой входов.
    Существенное условие: сравнение с пределом выполняется после регрессии.
    """

    frame = frame.copy()
    if "prediction_time" not in frame.columns:
        if bundle.config.training.mode == "forecast":
            frame["prediction_time"] = pd.to_datetime(frame["state_time"]) + forecast_horizon(bundle.config)
        else:
            frame["prediction_time"] = pd.to_datetime(frame["state_time"])
    X, _, diagnostics = prepare_features(frame, bundle.config, feature_names=bundle.feature_names)
    prediction = bundle.model.predict(X)
    out = pd.DataFrame(
        {
            "state_time": pd.to_datetime(frame["state_time"]),
            "prediction_time": pd.to_datetime(
                frame.get("prediction_time", frame["state_time"])
            ),
            "predicted_sulfur_mg_kg": prediction,
        }
    )
    limit = bundle.config.inference.sulfur_limit_mg_kg
    out["sulfur_limit_mg_kg"] = limit
    out["limit_exceeded"] = out["predicted_sulfur_mg_kg"] > limit
    missing_or_stale = []
    for idx in range(len(frame)):
        row_issues = list(diagnostics["missing_features"])
        if "input_sulfur_mg_kg" in frame.columns and pd.isna(frame.iloc[idx].get("input_sulfur_mg_kg")):
            row_issues.append("input_sulfur_mg_kg:missing_or_too_old")
        if "input_sulfur_is_stale" in frame.columns and bool(frame.iloc[idx].get("input_sulfur_is_stale")):
            row_issues.append("input_sulfur:stale")
        feature_na = X.iloc[idx].isna()
        row_issues.extend([f"{name}:nan" for name in X.columns[feature_na] if name not in row_issues])
        missing_or_stale.append(sorted(set(row_issues)))
    out["missing_or_stale_inputs"] = missing_or_stale
    out["model_version"] = str(bundle.metadata.get("run_id") or bundle.metadata.get("git_revision") or "unknown")
    assumptions = [
        f"Предел {limit} мг/кг применяется как настройка сценария для выхода гидроочистки.",
        "Интервалы прогноза и вероятность нарушения не оценены.",
    ]
    if bundle.config.lims.max_input_age_is_experimental:
        assumptions.append("Предельный возраст входного ЛИМС является экспериментальной настройкой.")
    out["assumptions"] = [assumptions] * len(out)
    out["interval_status"] = "not_estimated"
    out["violation_probability_status"] = "not_estimated"
    return out


def predict_diagnostic_grid(
    bundle: QualityModelBundle,
    base_state: pd.Series | dict[str, object],
    *,
    scenario_input_sulfur_mg_kg: float,
    t6_values: list[float],
    f9_values: list[float],
    p13_values: list[float],
) -> pd.DataFrame:
    """Строит исследовательскую сетку T6/F9/P13 и прогнозирует серу.

    Вход: один исходный снимок установки, сценарное значение входящей серы и
    списки значений T6, F9, P13. Выход: таблица кандидатов с прогнозом и
    диагностическим статусом. Существенное условие: функция не меняет запрет
    `allow_scenario_assessment`; результат является диагностикой реакции
    регрессии, а не производственной рекомендацией.
    """

    base = dict(base_state)
    if "state_time" not in base:
        raise ValueError("В исходном состоянии должна быть колонка state_time")
    candidates: list[dict[str, object]] = []
    candidate_id = 0
    for t6 in t6_values:
        for f9 in f9_values:
            for p13 in p13_values:
                candidate = dict(base)
                candidate["hdt_T6"] = float(t6)
                candidate["hdt_F9"] = float(f9)
                candidate["hdt_P13"] = float(p13)
                candidate["input_sulfur_mg_kg"] = float(scenario_input_sulfur_mg_kg)
                candidate["candidate_id"] = candidate_id
                candidates.append(candidate)
                candidate_id += 1
    frame = pd.DataFrame(candidates)
    prediction = predict_frame(frame, bundle)
    result = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "source_state_time": pd.to_datetime(frame["state_time"]),
            "input_sulfur_mg_kg": frame["input_sulfur_mg_kg"],
            "T6": frame["hdt_T6"],
            "F9": frame["hdt_F9"],
            "P13": frame["hdt_P13"],
            "predicted_sulfur_mg_kg": prediction["predicted_sulfur_mg_kg"],
        }
    )
    reasons = prediction["missing_or_stale_inputs"].map(lambda items: "; ".join(items) if items else "")
    result["status"] = reasons.map(lambda value: "ok" if value == "" else "calculated_with_warnings")
    result["status_reason"] = reasons
    result["assessment_scope"] = "diagnostic_regression_response_not_recommendation"
    return result


def predict_from_sources(
    telemetry_df: pd.DataFrame,
    bundle: QualityModelBundle,
    *,
    avt_df: pd.DataFrame | None = None,
    pak_df: pd.DataFrame | None = None,
    lims_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Готовит признаки и прогнозирует по исходным источникам.

    Вход: телеметрия и необязательные ПАК/ЛИМС. Выход: таблица ответа агента.
    Существенное условие: используется тот же `build_base_frame`, что при
    обучении.
    """

    frame = build_base_frame(telemetry_df, pak_df, lims_df, bundle.config, avt_df=avt_df)
    return predict_frame(frame, bundle)


def assess_scenario(
    base_state: dict[str, float | str],
    scenario_input_sulfur_mg_kg: float,
    control_changes: dict[str, float],
    horizon_hours: float,
    bundle: QualityModelBundle,
) -> dict[str, object]:
    """Проверяет интерфейс сценарной оценки режима.

    Вход: исходное состояние, сценарная входящая сера, изменения T6/F9/P13 и
    горизонт. Выход: статус и, если разрешено, прогноз. Существенное условие:
    использование модели для действий должно быть включено явно после проверки
    пригодности на истории.
    """

    aliases = {"T6": "hdt_T6", "F9": "hdt_F9", "P13": "hdt_P13"}
    control_changes = {aliases.get(name, name): value for name, value in control_changes.items()}
    allowed_controls = {"hdt_T6", "hdt_F9", "hdt_P13"}
    unknown = sorted(set(control_changes) - allowed_controls)
    if unknown:
        return {"status": "invalid_arguments", "message": f"Недопустимые управляющие параметры: {unknown}"}
    if horizon_hours < 0 or horizon_hours > bundle.config.training.max_forecast_horizon_hours:
        return {
            "status": "invalid_arguments",
            "message": f"Горизонт должен быть в диапазоне 0..{bundle.config.training.max_forecast_horizon_hours} ч.",
        }
    if bundle.config.training.mode == "forecast" and horizon_hours != bundle.config.training.forecast_horizon_hours:
        return {
            "status": "invalid_arguments",
            "message": "Горизонт сценария должен совпадать с горизонтом обученной forecast-модели.",
        }
    if not bundle.config.inference.allow_scenario_assessment:
        return {
            "status": "scenario_model_not_validated",
            "message": "Оценка действий отключена: пригодность модели для сценарных изменений еще не проверена.",
            "required_validation": "Нужна отдельная историческая проверка достоверности оценки предложенных действий.",
        }
    state = dict(base_state)
    state.update(control_changes)
    state["input_sulfur_mg_kg"] = scenario_input_sulfur_mg_kg
    if "state_time" not in state:
        return {"status": "invalid_arguments", "message": "В base_state должен быть state_time"}
    frame = pd.DataFrame([state])
    frame["state_time"] = pd.to_datetime(frame["state_time"])
    frame["prediction_time"] = frame["state_time"] + pd.Timedelta(hours=horizon_hours)
    result = predict_frame(frame, bundle).iloc[0].to_dict()
    result["status"] = "ok"
    result["unchanged_signal_rule"] = "Все неуправляющие сигналы берутся из исходного состояния; зависимые расчетные признаки пересчитываются функцией подготовки."
    return result

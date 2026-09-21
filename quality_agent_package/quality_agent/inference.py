"""Инференс модели качества и оценка произвольных кандидатов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor

from .config import DEFAULT_MODEL_DIR, read_json
from .data import forbidden_columns, read_table, restore_datetime_columns
from .features import prepare_features


@dataclass
class QualityModelBundle:
    """Загруженная модель и все артефакты, нужные для повторного инференса."""

    model: CatBoostRegressor
    model_config: dict[str, Any]
    feature_names: list[str]
    metadata: dict[str, Any]
    model_dir: Path


def load_bundle(model_dir: str | Path = DEFAULT_MODEL_DIR) -> QualityModelBundle:
    """Загружает CatBoost, config.json, feature_names.json и metadata.json один раз."""

    model_dir = Path(model_dir)
    model = CatBoostRegressor()
    model.load_model(model_dir / "model.cbm")
    return QualityModelBundle(
        model=model,
        model_config=read_json(model_dir / "config.json"),
        feature_names=read_json(model_dir / "feature_names.json"),
        metadata=read_json(model_dir / "metadata.json"),
        model_dir=model_dir,
    )


def predict_base_state(bundle: QualityModelBundle, base_state: pd.DataFrame | pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Строит прогноз для одного или нескольких готовых базовых состояний."""

    if isinstance(base_state, pd.DataFrame):
        frame = base_state.copy()
    elif isinstance(base_state, pd.Series):
        frame = pd.DataFrame([base_state.to_dict()])
    else:
        frame = pd.DataFrame([dict(base_state)])
    return predict_frame(frame, bundle)


def predict_frame(frame: pd.DataFrame, bundle: QualityModelBundle) -> pd.DataFrame:
    """Выполняет пакетный прогноз по подготовленным состояниям."""

    frame = restore_datetime_columns(frame)
    if "state_time" not in frame.columns:
        raise ValueError("В состоянии должна быть колонка state_time")
    out_frame = frame.copy()
    if "prediction_time" not in out_frame.columns:
        out_frame["prediction_time"] = pd.to_datetime(out_frame["state_time"])
    X, diagnostics = prepare_features(out_frame, bundle.model_config, bundle.feature_names)
    prediction = bundle.model.predict(X)
    limit = float(bundle.model_config.get("inference", {}).get("sulfur_limit_mg_kg", 10.0))
    result = pd.DataFrame(
        {
            "state_time": pd.to_datetime(out_frame["state_time"]),
            "prediction_time": pd.to_datetime(out_frame["prediction_time"]),
            "predicted_sulfur_mg_kg": prediction,
            "sulfur_limit_mg_kg": limit,
        }
    )
    result["limit_exceeded"] = result["predicted_sulfur_mg_kg"] > limit
    warnings = []
    for idx in range(len(out_frame)):
        row_warnings = list(diagnostics["missing_features"])
        if "input_sulfur_mg_kg" in out_frame.columns and pd.isna(out_frame.iloc[idx].get("input_sulfur_mg_kg")):
            row_warnings.append("input_sulfur_mg_kg:missing_or_too_old")
        if "input_sulfur_is_stale" in out_frame.columns and bool(out_frame.iloc[idx].get("input_sulfur_is_stale")):
            row_warnings.append("input_sulfur:stale")
        feature_na = X.iloc[idx].isna()
        row_warnings.extend([f"{name}:nan" for name in X.columns[feature_na]])
        warnings.append(sorted(set(row_warnings)))
    result["missing_or_stale_inputs"] = warnings
    result["model_version"] = str(bundle.metadata.get("run_id") or bundle.metadata.get("git_revision") or "unknown")
    result["interval_status"] = "not_estimated"
    result["violation_probability_status"] = "not_estimated"
    return result


def evaluate_candidates(
    bundle: QualityModelBundle,
    base_state: pd.DataFrame | pd.Series | dict[str, Any],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Оценивает кандидатов с абсолютными переопределениями исходных параметров.

    candidates должен содержать candidate_id и произвольный набор известных исходных
    колонок модели или входов формул. Технологические диапазоны здесь не проверяются:
    их задает внешний оптимизатор.
    """

    if "candidate_id" not in candidates.columns:
        raise ValueError("В candidates должна быть колонка candidate_id")
    if candidates["candidate_id"].isna().any():
        raise ValueError("candidate_id не должен содержать пустые значения")
    if candidates["candidate_id"].duplicated().any():
        duplicates = candidates.loc[candidates["candidate_id"].duplicated(), "candidate_id"].tolist()
        raise ValueError(f"candidate_id должен быть уникальным; дубли: {duplicates}")
    override_cols = [name for name in candidates.columns if name != "candidate_id"]
    if not override_cols:
        raise ValueError("В candidates нет колонок переопределения")
    if candidates[override_cols].isna().any().any():
        bad = candidates[override_cols].columns[candidates[override_cols].isna().any()].tolist()
        raise ValueError(f"Пустые значения в колонках переопределения запрещены: {bad}")
    _validate_override_columns(bundle, override_cols)
    base = _base_state_to_dict(base_state)
    rows = []
    for _, candidate in candidates.iterrows():
        row = dict(base)
        for name in override_cols:
            row[name] = candidate[name]
        row["candidate_id"] = candidate["candidate_id"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    prediction = predict_frame(frame.drop(columns=["candidate_id"]), bundle)
    result = pd.DataFrame({"candidate_id": frame["candidate_id"].to_list()})
    for name in override_cols:
        result[name] = frame[name].to_list()
    result["source_state_time"] = pd.to_datetime(frame["state_time"]).to_list()
    result["predicted_sulfur_mg_kg"] = prediction["predicted_sulfur_mg_kg"].to_list()
    result["sulfur_limit_mg_kg"] = prediction["sulfur_limit_mg_kg"].to_list()
    result["limit_exceeded"] = prediction["limit_exceeded"].to_list()
    reasons = prediction["missing_or_stale_inputs"].map(lambda items: "; ".join(items) if items else "")
    result["status"] = reasons.map(lambda value: "ok" if value == "" else "calculated_with_warnings").to_list()
    result["status_reason"] = reasons.to_list()
    result["assessment_scope"] = "diagnostic_regression_response_not_recommendation"
    result["interval_status"] = "not_estimated"
    result["violation_probability_status"] = "not_estimated"
    return result


def _base_state_to_dict(base_state: pd.DataFrame | pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(base_state, pd.DataFrame):
        if len(base_state) != 1:
            raise ValueError("base_state DataFrame должен содержать ровно одну строку")
        return base_state.iloc[0].to_dict()
    if isinstance(base_state, pd.Series):
        return base_state.to_dict()
    return dict(base_state)


def _validate_override_columns(bundle: QualityModelBundle, columns: list[str]) -> None:
    features_cfg = bundle.model_config.get("features", {})
    raw_inputs = set(features_cfg.get("raw_features", []))
    lab_inputs = set(features_cfg.get("lab_features", []))
    formula_inputs = {"pipeline_d15", "pipeline_95pct_t"}
    allowed = raw_inputs | lab_inputs | formula_inputs
    forbidden = forbidden_columns(bundle.model_config, bundle.feature_names)
    service = {
        "state_time",
        "prediction_time",
        "candidate_id",
        "avt_source_time",
        "avt_match_age_minutes",
        "avt_match_status",
        "input_sulfur_sample_time",
        "input_sulfur_available_at",
        "input_sulfur_available_at_assumed",
        "input_sulfur_is_stale",
        "pipeline_d15_sample_time",
        "pipeline_d15_available_at",
        "pipeline_d15_available_at_assumed",
        "pipeline_d15_age_hours",
        "pipeline_d15_is_stale",
        "pipeline_95pct_t_sample_time",
        "pipeline_95pct_t_available_at",
        "pipeline_95pct_t_available_at_assumed",
        "pipeline_95pct_t_age_hours",
        "pipeline_95pct_t_is_stale",
    }
    blocked = sorted(set(columns) & (forbidden | service))
    if blocked:
        raise ValueError(f"Эти колонки нельзя менять напрямую: {blocked}")
    unknown = sorted(set(columns) - allowed)
    if unknown:
        raise ValueError(f"Неизвестные или неразрешенные имена колонок: {unknown}")


def read_base_state(path: str | Path) -> pd.DataFrame:
    """Читает base_state.csv для CLI и внешнего API."""

    return read_table(path)

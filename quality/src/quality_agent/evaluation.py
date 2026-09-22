"""Метрики качества прогноза агента качества."""

from __future__ import annotations

import math

import pandas as pd

from .data import to_datetime_ns


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    """Считает базовые метрики регрессии.

    Вход: фактические и прогнозные значения. Выход: MAE, RMSE, средняя
    знаковая ошибка и число наблюдений. Существенное условие: пары с NaN
    исключаются только из расчета метрик, но не из сохраненных прогнозов.
    """

    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if data.empty:
        return {"mae": math.nan, "rmse": math.nan, "mean_signed_error": math.nan, "n": 0}
    errors = data["y_pred"] - data["y_true"]
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float((errors.pow(2).mean()) ** 0.5),
        "mean_signed_error": float(errors.mean()),
        "n": int(len(data)),
    }


def evaluate_against_lims(
    predictions: pd.DataFrame,
    output_lims: pd.DataFrame,
    *,
    tolerance_minutes: float,
) -> dict[str, float | int]:
    """Оценивает прогноз относительно выходного ЛИМС.

    Вход: прогнозы с `prediction_time` и лабораторные события с `sample_time`.
    Выход: метрики по совпавшим пробам. Существенное условие: лабораторный
    результат привязывается ко времени отбора пробы, а не публикации.
    """

    if predictions.empty or output_lims.empty:
        return {"mae": math.nan, "rmse": math.nan, "mean_signed_error": math.nan, "n": 0}
    predictions = predictions.copy()
    output_lims = output_lims.copy()
    predictions["prediction_time"] = to_datetime_ns(predictions["prediction_time"]).to_numpy()
    output_lims["sample_time"] = to_datetime_ns(output_lims["sample_time"]).to_numpy()
    merged = pd.merge_asof(
        output_lims.sort_values("sample_time"),
        predictions.sort_values("prediction_time"),
        left_on="sample_time",
        right_on="prediction_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    return regression_metrics(merged["output_lims_sulfur_mg_kg"], merged["prediction_mg_kg"])

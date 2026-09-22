"""Расчет признаков модели в сохраненном порядке."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import model_config_section
from .formulas_avt6 import Avt6Tags, compute_all as compute_avt_all
from .formulas_hdt import HdtTags, compute_all as compute_hdt_all


def safe_float(value: object) -> float:
    """Преобразует значение в float, сохраняя пропуски как NaN."""

    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def add_calculated_features(frame: pd.DataFrame, model_config: dict[str, Any]) -> pd.DataFrame:
    """Пересчитывает ВАК обеих установок по текущим исходным входам."""

    out = frame.copy()
    calculated = model_config_section(model_config, "features").get("calculated_features", [])
    hdt_fields = HdtTags.__dataclass_fields__.keys()
    avt_fields = Avt6Tags.__dataclass_fields__.keys()
    values: dict[str, list[float]] = {name: [] for name in calculated}
    for _, row in out.iterrows():
        hdt_tags = HdtTags(**{field: safe_float(row.get(f"hdt_{field}", row.get(field))) for field in hdt_fields})
        t95 = safe_float(row.get("pipeline_95pct_t"))
        d15 = safe_float(row.get("pipeline_d15"))
        all_values: dict[str, float] = {}
        all_values.update(
            compute_hdt_all(
                hdt_tags,
                None if math.isnan(t95) else t95,
                None if math.isnan(d15) else d15,
            )
        )
        avt_tags = Avt6Tags(**{field: safe_float(row.get(f"avt_{field}")) for field in avt_fields})
        all_values.update(compute_avt_all(avt_tags))
        for name in calculated:
            values[name].append(all_values.get(name, float("nan")))
    for name, column in values.items():
        out[name] = pd.to_numeric(pd.Series(column, index=out.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def prepare_features(
    frame: pd.DataFrame,
    model_config: dict[str, Any],
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Собирает X в порядке feature_names.json и возвращает диагностику."""

    enriched = add_calculated_features(frame, model_config)
    features_cfg = model_config_section(model_config, "features")
    leakage = set(features_cfg.get("leakage_features", [])) | {"target_sulfur_mg_kg"}
    calculated = set(features_cfg.get("calculated_features", []))
    bad = [
        name
        for name in feature_names
        if name in leakage
        or name.startswith("target_")
        or name.endswith("_target")
        or name.endswith("_rejected")
        or name.endswith("_rejection_reason")
    ]
    if bad:
        raise ValueError(f"Запрещенные целевые признаки попали в X: {bad}")
    missing = [name for name in feature_names if name not in enriched.columns]
    for name in missing:
        enriched[name] = pd.NA
    X = enriched[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    diagnostics = {
        "missing_features": missing,
        "calculated_features": sorted(calculated & set(feature_names)),
    }
    return X, diagnostics

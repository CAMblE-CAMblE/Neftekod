"""Подготовка признаков для обучения и инференса агента качества."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quality_formulas import AvtGodtTags, compute_all
from avt6_formulas import Avt6Tags, compute_all as compute_avt6_all

from .config import QualityAgentConfig


def allowed_calculated_features(config: QualityAgentConfig) -> list[str]:
    """Возвращает разрешенные расчетные признаки.

    Вход: конфигурация с источниками ВАК. Выход: список признаков, не зависящих
    от запрещенных колонок. Существенное условие: целевые и сравнительные
    колонки не могут попасть в матрицу признаков через ВАК.
    """

    forbidden = set(config.features.forbidden_base_features) | set(config.features.leakage_features)
    result = []
    for name in config.features.calculated_features:
        sources = set(config.features.calculated_feature_sources.get(name, []))
        if sources.isdisjoint(forbidden):
            result.append(name)
    return result


def ordered_feature_names(config: QualityAgentConfig) -> list[str]:
    """Формирует единый порядок признаков.

    Вход: конфигурация. Выход: список колонок для CatBoost и baseline.
    Существенное условие: базовые запрещенные признаки удаляются даже если они
    ошибочно попали в YAML.
    """

    forbidden = set(config.features.forbidden_base_features) | set(config.features.leakage_features)
    names = [name for name in config.features.raw_features if name not in forbidden]
    names.extend(config.features.lab_features)
    names.extend(allowed_calculated_features(config))
    seen: set[str] = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _safe_float(value: object) -> float:
    """Преобразует значение в float с NaN при невозможности.

    Вход: произвольное значение из строки датафрейма. Выход: число с плавающей
    точкой. Существенное условие: пропуски не превращаются в нули.
    """

    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def add_calculated_features(frame: pd.DataFrame, config: QualityAgentConfig) -> pd.DataFrame:
    """Добавляет разрешенные расчетные ВАК-признаки.

    Вход: канонический датафрейм состояний. Выход: копия с колонками ВАК и
    словарем происхождения в `attrs`. Существенное условие: используются
    существующие функции `quality_formulas`, а признаки с запрещенными
    источниками не добавляются в список обучения.
    """

    out = frame.copy()
    needed = allowed_calculated_features(config)
    if not needed:
        return out
    hdt_required = set(AvtGodtTags.__dataclass_fields__)
    avt_required = set(Avt6Tags.__dataclass_fields__)
    values: dict[str, list[float]] = {name: [] for name in needed}
    for _, row in out.iterrows():
        hdt_tags = AvtGodtTags(**{field: _safe_float(row.get(f"hdt_{field}", row.get(field))) for field in hdt_required})
        t95_pipeline = _safe_float(row.get("pipeline_95pct_t"))
        d15_pipeline = _safe_float(row.get("pipeline_d15"))
        all_values: dict[str, float] = {}
        hdt_values = compute_all(
            hdt_tags,
            lims_95pct_t_pipeline=t95_pipeline if not math.isnan(t95_pipeline) else None,
            lims_d15_pipeline=d15_pipeline if not math.isnan(d15_pipeline) else None,
        )
        all_values.update({f"hdt_{name}": value for name, value in hdt_values.items()})
        avt_tags = Avt6Tags(**{field: _safe_float(row.get(f"avt_{field}")) for field in avt_required})
        all_values.update(compute_avt6_all(avt_tags))
        for name in needed:
            values[name].append(all_values.get(name, float("nan")))
    for name, column in values.items():
        out[name] = pd.to_numeric(pd.Series(column, index=out.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    out.attrs["calculated_feature_origin"] = {
        name: config.features.calculated_feature_sources.get(name, []) for name in needed
    }
    return out


def prepare_features(
    frame: pd.DataFrame,
    config: QualityAgentConfig,
    *,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Готовит матрицу признаков в едином порядке.

    Вход: канонический датафрейм и опционально сохраненный список признаков.
    Выход: X, порядок признаков и сведения об отсутствующих колонках.
    Существенное условие: эта же функция используется в обучении и инференсе.
    """

    enriched = add_calculated_features(frame, config)
    names = feature_names or ordered_feature_names(config)
    forbidden = set(config.features.forbidden_base_features)
    leakage_set = set(config.features.leakage_features) | {config.data.target_col, config.data.target_source_col}
    leakage = [
        name
        for name in names
        if name in forbidden
        or name in leakage_set
        or name.startswith("target_")
        or name.endswith("_target")
        or name.endswith("_rejected")
        or name.endswith("_rejection_reason")
    ]
    if leakage:
        raise ValueError(f"Запрещенные или целевые признаки попали в матрицу: {leakage}")
    missing = [name for name in names if name not in enriched.columns]
    for name in missing:
        enriched[name] = pd.NA
    X = enriched[names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    diagnostics = {
        "missing_features": missing,
        "forbidden_features": sorted(forbidden & set(frame.columns)),
        "calculated_features": allowed_calculated_features(config),
    }
    return X, names, diagnostics


def train_feature_exclusions(X_train: pd.DataFrame) -> dict[str, list[str]]:
    """Определяет признаки, которые нельзя фиксировать в X по обучающей выборке.

    Вход: матрица train после всех расчетных преобразований. Выход: списки
    полностью пустых и постоянных признаков. Существенное условие: проверка
    выполняется только на train, чтобы validation/test не влияли на состав X.
    """

    all_missing = [name for name in X_train.columns if X_train[name].isna().all()]
    constant = []
    for name in X_train.columns:
        if name in all_missing:
            continue
        if X_train[name].dropna().nunique() <= 1:
            constant.append(name)
    return {"all_missing": all_missing, "constant": constant}

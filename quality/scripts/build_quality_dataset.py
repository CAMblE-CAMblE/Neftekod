"""CLI-сборка реального датасета агента качества без обучения модели."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from quality_agent.config import config_to_dict, load_config  # noqa: E402
from quality_agent.data import build_base_frame, extract_output_lims, load_sources, write_table  # noqa: E402
from quality_agent.features import add_calculated_features, ordered_feature_names  # noqa: E402


def main() -> None:
    """Разбирает аргументы и собирает датасет качества."""

    parser = argparse.ArgumentParser(description="Сборка датасета агента качества 24-2000")
    parser.add_argument("--config", default="configs/quality_agent.yaml")
    parser.add_argument("--telemetry-path", default=None)
    parser.add_argument("--avt-telemetry-path", default=None)
    parser.add_argument("--lims-path", default=None)
    parser.add_argument("--pak-path", default=None)
    parser.add_argument("--output", default="data/processed/quality_dataset.parquet")
    parser.add_argument("--output-lims", default="data/processed/quality_output_lims.parquet")
    parser.add_argument("--report", default="data/processed/quality_dataset_report.json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.telemetry_path:
        config.data.telemetry_path = args.telemetry_path
    if args.avt_telemetry_path:
        config.data.avt_telemetry_path = args.avt_telemetry_path
    if args.lims_path:
        config.data.lims_path = args.lims_path
    if args.pak_path:
        config.data.pak_path = args.pak_path
    config.data.prepared_path = args.output
    config.data.output_lims_path = args.output_lims

    _check_required_paths(config.data.telemetry_path, config.data.lims_path, config.data.avt_telemetry_path)
    telemetry, avt, pak, lims = load_sources(config)
    frame = build_base_frame(telemetry, pak, lims, config, avt_df=avt)
    output_lims = extract_output_lims(lims, config)
    write_table(frame, args.output)
    write_table(output_lims, args.output_lims)

    report = build_report(frame, output_lims, telemetry, config)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Датасет сохранен: {args.output}")
    print(f"Выходной ЛИМС сохранен: {args.output_lims}")
    print(f"Отчет сохранен: {args.report}")


def _check_required_paths(telemetry_path: str | None, lims_path: str | None, avt_path: str | None = None) -> None:
    """Проверяет наличие обязательных источников и печатает точные ожидания."""

    missing = [str(path) for path in [telemetry_path, lims_path, avt_path] if not path or not Path(path).exists()]
    if missing:
        expected = "\n".join(
            [
                "Ожидаемые обязательные пути:",
                f"- telemetry: {telemetry_path}",
                f"- avt_telemetry: {avt_path}",
                f"- lims: {lims_path}",
            ]
        )
        raise SystemExit(f"Не найдены исходники: {missing}\n{expected}")


def build_report(
    frame: pd.DataFrame,
    output_lims: pd.DataFrame,
    raw_telemetry: pd.DataFrame,
    config: Any,
) -> dict[str, Any]:
    """Формирует отчет сборки датасета.

    Вход: подготовленный датасет, выходной ЛИМС, исходная телеметрия и конфиг.
    Выход: JSON-совместимый словарь с диагностикой периода, пропусков,
    дубликатов, целей, ЛИМС и ВАК.
    """

    enriched = add_calculated_features(frame, config)
    features = ordered_feature_names(config)
    duplicates = _duplicate_report(frame)
    return {
        "sources": {
            "telemetry_path": config.data.telemetry_path,
            "avt_telemetry_path": config.data.avt_telemetry_path,
            "lims_path": config.data.lims_path,
            "pak_path": config.data.pak_path,
            "target": "hdt_Q21 из телеметрии гидроочистки",
            "target_unit": "ppm, численно эквивалентно мг/кг",
        },
        "period": {
            "state_time_min": _safe_str(frame["state_time"].min()),
            "state_time_max": _safe_str(frame["state_time"].max()),
        },
        "rows": int(len(frame)),
        "raw_telemetry_columns": list(raw_telemetry.columns),
        "time_checks": {
            "is_sorted": bool(frame["state_time"].is_monotonic_increasing),
            "duplicates": duplicates,
            "interval_minutes": _interval_report(frame["state_time"]),
            "avt_join": frame.attrs.get("avt_join_report", {}),
        },
        "data_quality": {
            "missing_by_column": {name: int(value) for name, value in frame.isna().sum().items() if int(value) > 0},
            "infinite_by_column": _inf_report(frame),
            "duplicate_policy": "Конфликтующие дубликаты не удаляются автоматически; они отражены в отчете и требуют отдельного решения.",
            "sulfur_above_10_policy": "Q21 выше 10 мг/кг сохраняется как возможное реальное событие.",
            "known_cell_replacements": _known_cell_replacements(frame),
        },
        "target": {
            "column": config.data.target_col,
            "available_rows": int(frame[config.data.target_col].notna().sum()),
            "missing_rows": int(frame[config.data.target_col].isna().sum()),
            "raw_column": "target_sulfur_raw_mg_kg",
            "rejection_flag_column": "target_sulfur_rejected",
            "rejection_reason_column": "target_sulfur_rejection_reason",
            "confirmed_rejections": _target_rejections(frame),
            "source_counts": frame.get(config.data.target_source_col, pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        },
        "lims": {
            "input_sulfur_samples_attached": int(frame["input_sulfur_sample_time"].notna().sum()),
            "input_sulfur_values_after_delay_and_age_limit": int(frame["input_sulfur_mg_kg"].notna().sum()),
            "input_sulfur_stale_rows_excluded": int(frame["input_sulfur_is_stale"].sum()),
            "max_input_age_hours": config.lims.max_input_age_hours,
            "max_input_age_is_experimental": config.lims.max_input_age_is_experimental,
            "output_lims_events": int(len(output_lims)),
        },
        "vak": {
            name: {
                "filled_rows": int(enriched[name].notna().sum()) if name in enriched.columns else 0,
                "source_columns": config.features.calculated_feature_sources.get(name, []),
            }
            for name in config.features.calculated_features
        },
        "features": {
            "used_features": features,
            "excluded_columns": sorted(set(config.features.forbidden_base_features) | set(config.features.leakage_features)),
        },
        "preparation_settings": config_to_dict(config),
    }


def _duplicate_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Считает дубликаты времени и конфликтующие дубликаты."""

    duplicated = frame[frame["state_time"].duplicated(keep=False)]
    if duplicated.empty:
        return {"duplicate_rows": 0, "duplicate_timestamps": 0, "conflicting_timestamps": 0}
    conflicts = 0
    for _, group in duplicated.groupby("state_time"):
        comparable = group.drop(columns=["state_time"], errors="ignore")
        if len(comparable.drop_duplicates()) > 1:
            conflicts += 1
    return {
        "duplicate_rows": int(len(duplicated)),
        "duplicate_timestamps": int(duplicated["state_time"].nunique()),
        "conflicting_timestamps": int(conflicts),
    }


def _interval_report(times: pd.Series) -> dict[str, Any]:
    """Возвращает сводку интервалов между состояниями."""

    minutes = times.sort_values().diff().dropna() / pd.Timedelta(minutes=1)
    if minutes.empty:
        return {"min": None, "median": None, "max": None}
    return {"min": float(minutes.min()), "median": float(minutes.median()), "max": float(minutes.max())}


def _inf_report(frame: pd.DataFrame) -> dict[str, int]:
    """Считает бесконечные значения по числовым колонкам."""

    result: dict[str, int] = {}
    numeric = frame.select_dtypes(include="number")
    for name in numeric.columns:
        count = int(numeric[name].map(lambda value: math.isinf(float(value)) if pd.notna(value) else False).sum())
        if count:
            result[name] = count
    return result


def _known_cell_replacements(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Возвращает подтвержденные точечные замены на NaN с причиной."""

    replacements: list[dict[str, Any]] = []
    f26_col = "hdt_F26" if "hdt_F26" in frame.columns else "F26"
    if f26_col in frame.columns:
        replacements.append(
            {
                "derived_column": "hdt_T90",
                "source_condition": f"{f26_col} == 0",
                "affected_rows": int((pd.to_numeric(frame[f26_col], errors="coerce") == 0).sum()),
                "reason": "Формула T90 делит на F26; для нулевого F26 возвращается NaN.",
            }
        )
    return replacements


def _target_rejections(frame: pd.DataFrame) -> dict[str, Any]:
    """Возвращает сводку подтвержденных отклонений целевых измерений."""

    flag_col = "target_sulfur_rejected"
    reason_col = "target_sulfur_rejection_reason"
    raw_col = "target_sulfur_raw_mg_kg"
    if flag_col not in frame.columns:
        return {"rows": 0, "by_reason": {}}
    mask = frame[flag_col].fillna(False).astype(bool)
    return {
        "rows": int(mask.sum()),
        "raw_307_rows": int(pd.to_numeric(frame.get(raw_col, pd.Series(dtype=float)), errors="coerce").eq(307.0).sum()),
        "target_307_remaining_rows": int(pd.to_numeric(frame.get("target_sulfur_mg_kg", pd.Series(dtype=float)), errors="coerce").eq(307.0).sum()),
        "by_reason": frame.loc[mask, reason_col].value_counts(dropna=False).to_dict() if reason_col in frame.columns else {},
        "policy": "Q21 == 307 ppm подтвержден экспертами как ошибочное измерение; целевое значение заменяется на NaN, строка и остальные измерения сохраняются.",
    }


def _safe_str(value: object) -> str | None:
    """Преобразует значение в строку или None."""

    if pd.isna(value):
        return None
    return str(value)


if __name__ == "__main__":
    main()

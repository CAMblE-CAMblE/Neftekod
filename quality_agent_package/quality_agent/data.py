"""Подготовка базового состояния из исходных файлов или готового parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import model_config_section, resolve_path
from .parsers import parse_lims, parse_pak


TIME_COLUMNS = {
    "state_time",
    "prediction_time",
    "target_time",
    "sample_time",
    "available_at",
    "avt_source_time",
    "input_sulfur_sample_time",
    "input_sulfur_available_at",
    "pipeline_d15_sample_time",
    "pipeline_d15_available_at",
    "pipeline_95pct_t_sample_time",
    "pipeline_95pct_t_available_at",
}


def to_datetime_ns(values: object, *, fmt: str | None = None) -> pd.Series:
    """Приводит значения времени к pandas datetime64[ns]."""

    return pd.Series(pd.to_datetime(values, errors="coerce", format=fmt)).astype("datetime64[ns]")


def read_table(path: str | Path) -> pd.DataFrame:
    """Читает CSV или parquet и восстанавливает временные колонки."""

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return restore_datetime_columns(pd.read_parquet(path))
    if path.suffix.lower() == ".csv":
        return restore_datetime_columns(pd.read_csv(path))
    raise ValueError(f"Неподдерживаемый формат таблицы: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Сохраняет таблицу CSV или parquet."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Неподдерживаемый формат результата: {path}")


def restore_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Распознает временные колонки после чтения CSV/parquet."""

    out = df.copy()
    for name in out.columns:
        if name in TIME_COLUMNS or name.endswith("_time") or name.endswith("_at"):
            out[name] = to_datetime_ns(out[name]).to_numpy()
    return out


def normalize_telemetry(df: pd.DataFrame, model_config: dict[str, Any], *, prefix: str | None) -> pd.DataFrame:
    """Нормализует телеметрию к колонке state_time и нужному префиксу."""

    data_cfg = model_config_section(model_config, "data")
    out = df.copy()
    unnamed = [name for name in out.columns if str(name).startswith("Unnamed") or str(name) == ""]
    if unnamed:
        out = out.drop(columns=unnamed)
    if "state_time" not in out.columns:
        timestamp_col = data_cfg.get("timestamp_col", "date")
        if timestamp_col not in out.columns:
            raise ValueError(f"В телеметрии нет колонки времени {timestamp_col!r}")
        out = out.rename(columns={timestamp_col: "state_time"})
    out["state_time"] = to_datetime_ns(out["state_time"], fmt=data_cfg.get("timestamp_format")).to_numpy()
    if out["state_time"].isna().any():
        raise ValueError("В телеметрии есть строки с нераспознанным временем")
    if prefix:
        rename = {
            name: f"{prefix}_{name}"
            for name in out.columns
            if name != "state_time" and not str(name).startswith(f"{prefix}_")
        }
        out = out.rename(columns=rename)
    return out.sort_values("state_time").reset_index(drop=True)


def attach_avt_telemetry(frame: pd.DataFrame, avt_df: pd.DataFrame | None, model_config: dict[str, Any]) -> pd.DataFrame:
    """Присоединяет АВТ backward-asof без будущих измерений."""

    out = frame.sort_values("state_time").copy()
    data_cfg = model_config_section(model_config, "data")
    if avt_df is None:
        out["avt_source_time"] = pd.NaT
        out["avt_match_age_minutes"] = pd.NA
        out["avt_match_status"] = "missing_source"
        out.attrs["avt_join_report"] = {"mode": "missing_source", "matched_rows": 0, "unmatched_rows": int(len(out))}
        return out
    avt = normalize_telemetry(avt_df, model_config, prefix="avt").rename(columns={"state_time": "avt_source_time"})
    exact = len(out) == len(avt) and out["state_time"].reset_index(drop=True).equals(avt["avt_source_time"].reset_index(drop=True))
    if exact:
        merged = out.merge(avt, left_on="state_time", right_on="avt_source_time", how="left")
        mode = "exact"
    else:
        merged = pd.merge_asof(
            out.sort_values("state_time"),
            avt.sort_values("avt_source_time"),
            left_on="state_time",
            right_on="avt_source_time",
            direction="backward",
            tolerance=pd.Timedelta(minutes=float(data_cfg.get("avt_asof_tolerance_minutes", 10.0))),
        )
        mode = "backward_asof"
    merged["avt_match_age_minutes"] = (merged["state_time"] - merged["avt_source_time"]) / pd.Timedelta(minutes=1)
    matched = merged["avt_source_time"].notna()
    merged["avt_match_status"] = matched.map({True: mode, False: "unmatched"})
    merged.attrs["avt_join_report"] = {
        "mode": mode,
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "future_measurements_used": False,
    }
    return merged.sort_values("state_time").reset_index(drop=True)


def canonicalize_lims_parameter(
    lims_df: pd.DataFrame | None,
    *,
    sampling_point: str,
    parameter: str,
    model_config: dict[str, Any],
    value_col: str,
) -> pd.DataFrame:
    """Выделяет один лабораторный показатель и время его доступности."""

    if lims_df is None:
        return pd.DataFrame(columns=["sample_time", "available_at", value_col, "available_at_assumed"])
    lab = lims_df[(lims_df["sampling_point"] == sampling_point) & (lims_df["parameter"] == parameter)].copy()
    if lab.empty:
        return pd.DataFrame(columns=["sample_time", "available_at", value_col, "available_at_assumed"])
    lab["sample_time"] = to_datetime_ns(lab["timestamp"]).to_numpy()
    values = pd.to_numeric(lab["value"], errors="coerce")
    if parameter == "Mass.Sulfur":
        units = lab["unit"].astype(str).str.lower()
        values = values.where(~units.str.contains("%"), values * 10000.0)
    lab[value_col] = values
    if "available_at" in lab.columns:
        lab["available_at"] = to_datetime_ns(lab["available_at"]).to_numpy()
        lab["available_at_assumed"] = lab["available_at"].isna()
    else:
        lab["available_at"] = pd.NaT
        lab["available_at_assumed"] = True
    delay = float(model_config_section(model_config, "lims").get("assumed_publication_delay_hours", 4.0))
    lab["available_at"] = lab["available_at"].fillna(lab["sample_time"] + pd.Timedelta(hours=delay))
    return lab[["sample_time", "available_at", value_col, "available_at_assumed"]].dropna(subset=["sample_time", "available_at"]).sort_values("available_at").reset_index(drop=True)


def attach_latest_lims_feature(
    frame: pd.DataFrame,
    lab: pd.DataFrame,
    *,
    value_col: str,
    output_prefix: str,
    model_config: dict[str, Any],
) -> pd.DataFrame:
    """Присоединяет последний опубликованный лабораторный анализ."""

    out = frame.sort_values("state_time").copy()
    if lab.empty:
        out[f"{output_prefix}_sample_time"] = pd.NaT
        out[f"{output_prefix}_available_at"] = pd.NaT
        out[f"{output_prefix}_available_at_assumed"] = True
        out[value_col] = pd.NA
        out[f"{output_prefix}_age_hours"] = pd.NA
        out[f"{output_prefix}_is_stale"] = False
        return out
    merged = pd.merge_asof(out, lab, left_on="state_time", right_on="available_at", direction="backward")
    merged = merged.rename(
        columns={
            "sample_time": f"{output_prefix}_sample_time",
            "available_at": f"{output_prefix}_available_at",
            "available_at_assumed": f"{output_prefix}_available_at_assumed",
        }
    )
    age_col = f"{output_prefix}_age_hours"
    merged[age_col] = (merged["state_time"] - merged[f"{output_prefix}_sample_time"]) / pd.Timedelta(hours=1)
    max_age = float(model_config_section(model_config, "lims").get("max_input_age_hours", 240.0))
    too_old = merged[age_col] > max_age
    merged[f"{output_prefix}_is_stale"] = too_old.fillna(False)
    merged.loc[too_old, value_col] = pd.NA
    return merged


def build_state_table_from_sources(paths_config: dict[str, Any], model_config: dict[str, Any]) -> pd.DataFrame:
    """Собирает таблицу состояний из телеметрии HDT, телеметрии АВТ и ЛИМС."""

    config_dir = Path(paths_config.get("_config_dir", "."))
    source_cfg = paths_config.get("sources", paths_config)
    hdt_path = resolve_path(source_cfg.get("telemetry_path"), base_dir=config_dir)
    avt_path = resolve_path(source_cfg.get("avt_telemetry_path"), base_dir=config_dir)
    lims_path = resolve_path(source_cfg.get("lims_path"), base_dir=config_dir)
    pak_path = resolve_path(source_cfg.get("pak_path"), base_dir=config_dir)
    if hdt_path is None or avt_path is None or lims_path is None:
        raise ValueError("Для сборки из сырья нужны telemetry_path, avt_telemetry_path и lims_path")
    hdt = read_table(hdt_path)
    avt = read_table(avt_path)
    lims = parse_lims(str(lims_path))
    frame = normalize_telemetry(hdt, model_config, prefix="hdt")
    frame = attach_avt_telemetry(frame, avt, model_config)
    data_cfg = model_config_section(model_config, "data")
    input_lims = canonicalize_lims_parameter(
        lims,
        sampling_point=data_cfg["input_lims_sampling_point"],
        parameter=data_cfg["input_lims_parameter"],
        model_config=model_config,
        value_col="input_sulfur_mg_kg",
    )
    frame = attach_latest_lims_feature(frame, input_lims, value_col="input_sulfur_mg_kg", output_prefix="input_sulfur", model_config=model_config)
    for parameter, value_col, prefix in [
        ("D15", "pipeline_d15", "pipeline_d15"),
        ("95%.T", "pipeline_95pct_t", "pipeline_95pct_t"),
    ]:
        lab = canonicalize_lims_parameter(
            lims,
            sampling_point=data_cfg["input_lims_sampling_point"],
            parameter=parameter,
            model_config=model_config,
            value_col=value_col,
        )
        frame = attach_latest_lims_feature(frame, lab, value_col=value_col, output_prefix=prefix, model_config=model_config)
    if pak_path is not None:
        pak = parse_pak(str(pak_path))
        frame.attrs["pak_rows_loaded"] = int(len(pak))
    else:
        frame.attrs["pak_rows_loaded"] = 0
    return frame.sort_values("state_time").reset_index(drop=True)


def select_base_state(
    table: pd.DataFrame,
    *,
    state_time: str,
    allow_previous: bool,
    model_config: dict[str, Any],
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Выбирает одну строку состояния и формирует отчет сборки."""

    frame = restore_datetime_columns(table).sort_values("state_time").reset_index(drop=True)
    requested = pd.Timestamp(state_time)
    exact = frame[frame["state_time"] == requested]
    if not exact.empty:
        selected = exact.iloc[[0]].copy()
        selection_mode = "exact"
    elif allow_previous:
        previous = frame[frame["state_time"] <= requested]
        if previous.empty:
            raise ValueError(f"Нет строки состояния в {requested} и нет более ранних строк")
        selected = previous.iloc[[-1]].copy()
        selection_mode = "previous"
    else:
        nearest_hint = ""
        if not frame.empty:
            earlier = frame.loc[frame["state_time"] <= requested, "state_time"].max()
            later = frame.loc[frame["state_time"] >= requested, "state_time"].min()
            nearest_hint = f"; ближайшие времена: earlier={earlier}, later={later}"
        raise ValueError(f"Нет строки состояния ровно в {requested}. Для выбора более ранней строки задайте --allow-previous{nearest_hint}")
    selected = filter_base_state_columns(selected, model_config=model_config, feature_names=feature_names)
    missing_inputs = [name for name in required_runtime_inputs(model_config, feature_names) if name not in selected.columns or selected.iloc[0].get(name) is pd.NA or pd.isna(selected.iloc[0].get(name))]
    lab_ages = {
        name: (None if name not in selected.columns or pd.isna(selected.iloc[0].get(name)) else float(selected.iloc[0].get(name)))
        for name in ["input_sulfur_age_hours", "pipeline_d15_age_hours", "pipeline_95pct_t_age_hours"]
    }
    report = {
        "requested_time": str(requested),
        "selected_time": str(pd.Timestamp(selected.iloc[0]["state_time"])),
        "selection_mode": selection_mode,
        "allow_previous": bool(allow_previous),
        "missing_inputs": missing_inputs,
        "lab_age_hours": lab_ages,
    }
    return selected, report


def filter_base_state_columns(frame: pd.DataFrame, *, model_config: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    """Оставляет в base_state только исходные входы и служебные поля без целей и прогнозов."""

    service = [
        "state_time",
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
    ]
    keep = ["state_time"]
    for name in model_config_section(model_config, "features").get("raw_features", []):
        if name in frame.columns and name not in keep:
            keep.append(name)
    for name in ["input_sulfur_mg_kg", "input_sulfur_age_hours", "pipeline_d15", "pipeline_95pct_t", *service]:
        if name in frame.columns and name not in keep:
            keep.append(name)
    forbidden = forbidden_columns(model_config, feature_names)
    keep = [name for name in keep if name in frame.columns and name not in forbidden]
    return frame[keep].copy()


def required_runtime_inputs(model_config: dict[str, Any], feature_names: list[str]) -> list[str]:
    """Возвращает исходные входы, необходимые для сборки X и формул."""

    features_cfg = model_config_section(model_config, "features")
    required = []
    for name in features_cfg.get("raw_features", []):
        if name in feature_names:
            required.append(name)
    required.extend(features_cfg.get("lab_features", []))
    required.extend(["pipeline_d15", "pipeline_95pct_t"])
    return list(dict.fromkeys(required))


def forbidden_columns(model_config: dict[str, Any], feature_names: list[str]) -> set[str]:
    """Возвращает целевые, расчетные и служебные колонки, которые нельзя переопределять."""

    features_cfg = model_config_section(model_config, "features")
    forbidden = set(features_cfg.get("leakage_features", []))
    forbidden.update(features_cfg.get("forbidden_base_features", []))
    forbidden.update(features_cfg.get("calculated_features", []))
    forbidden.update(name for name in feature_names if name in features_cfg.get("calculated_features", []))
    forbidden.update(
        {
            "target_sulfur_mg_kg",
            "target_sulfur_raw_mg_kg",
            "target_sulfur_rejected",
            "target_sulfur_rejection_reason",
            "target_sulfur_above_10_mg_kg",
            "prediction_time",
            "predicted_sulfur_mg_kg",
            "sulfur_limit_mg_kg",
            "limit_exceeded",
        }
    )
    return forbidden

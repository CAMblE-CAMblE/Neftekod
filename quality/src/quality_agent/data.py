"""Загрузка, канонизация и временное сопоставление данных агента качества."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lims_parser import parse_lims
from pak_parser import parse_pak

from .config import QualityAgentConfig


def to_datetime_ns(values: object, *, fmt: str | None = None, dayfirst: bool = False) -> pd.Series:
    """Приводит время к единому dtype datetime64[ns].

    Вход: колонка или серия со временем. Выход: pandas Series с наносекундным
    dtype. Существенное условие: единый dtype нужен для `merge_asof` в pandas 3.
    """

    return pd.Series(pd.to_datetime(values, errors="coerce", format=fmt, dayfirst=dayfirst)).astype("datetime64[ns]")


def restore_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Восстанавливает datetime-колонки после чтения CSV/Parquet.

    Вход: подготовленная таблица. Выход: копия с распознанными временными
    колонками. Существенное условие: невременные исходные значения не меняются.
    """

    out = df.copy()
    for name in out.columns:
        if name in {"state_time", "prediction_time", "target_time", "sample_time", "available_at"}:
            out[name] = to_datetime_ns(out[name]).to_numpy()
        elif name.endswith("_time") or name.endswith("_at"):
            out[name] = to_datetime_ns(out[name]).to_numpy()
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    """Читает таблицу CSV или Parquet.

    Вход: путь к файлу. Выход: `DataFrame`. Существенное условие: при чтении
    CSV явно восстанавливаются datetime-колонки подготовленной схемы.
    """

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return restore_datetime_columns(pd.read_parquet(path))
    if path.suffix.lower() == ".csv":
        return restore_datetime_columns(pd.read_csv(path))
    raise ValueError(f"Неподдерживаемый формат таблицы: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Сохраняет таблицу CSV или Parquet.

    Вход: датафрейм и путь. Выход: файл на диске. Существенное условие:
    Parquet является основным форматом, но CSV оставлен для локальной проверки.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Неподдерживаемый формат сохранения: {path}")


def load_sources(config: QualityAgentConfig) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Загружает исходные источники проекта.

    Вход: конфигурация с путями telemetry, ПАК и ЛИМС. Выход: три датафрейма,
    где ПАК и ЛИМС могут быть None. Существенное условие: для Excel ПАК/ЛИМС
    используются существующие парсеры команды.
    """

    telemetry = read_table(config.data.telemetry_path)
    avt = read_table(config.data.avt_telemetry_path) if config.data.avt_telemetry_path else None
    pak = parse_pak(config.data.pak_path) if config.data.pak_path else None
    lims = parse_lims(config.data.lims_path) if config.data.lims_path else None
    return telemetry, avt, pak, lims


def normalize_telemetry(df: pd.DataFrame, config: QualityAgentConfig, *, prefix: str | None = None) -> pd.DataFrame:
    """Приводит телеметрию 24-2000 к каноническому времени состояния.

    Вход: таблица `242000_tags.csv` или подготовленная таблица с аналогичными
    колонками. Выход: датафрейм с `state_time`. Существенное условие: теги не
    переименовываются, а дата CSV разбирается по явному формату из конфига.
    """

    out = df.copy()
    unnamed = [name for name in out.columns if str(name).startswith("Unnamed") or str(name) == ""]
    if unnamed:
        out = out.drop(columns=unnamed)
    time_col = config.data.timestamp_col
    if "state_time" not in out.columns:
        if time_col not in out.columns:
            raise ValueError(f"В телеметрии нет колонки времени '{time_col}'")
        out = out.rename(columns={time_col: "state_time"})
    out["state_time"] = to_datetime_ns(out["state_time"], fmt=config.data.timestamp_format).to_numpy()
    if out["state_time"].isna().any():
        raise ValueError("В телеметрии есть строки с нераспознанным временем")
    if prefix:
        rename = {name: f"{prefix}_{name}" for name in out.columns if name != "state_time" and not str(name).startswith(f"{prefix}_")}
        out = out.rename(columns=rename)
    return out.sort_values("state_time").reset_index(drop=True)


def attach_avt_telemetry(frame: pd.DataFrame, avt_df: pd.DataFrame | None, config: QualityAgentConfig) -> pd.DataFrame:
    """Присоединяет АВТ к основной сетке гидроочистки без будущих измерений.

    Вход: строки гидроочистки с `state_time` и сырая телеметрия АВТ. Выход:
    таблица с колонками `avt_*`, `avt_source_time`, `avt_match_age_minutes`
    и диагностикой соединения в `attrs["avt_join_report"]`.
    """

    out = frame.sort_values("state_time").copy()
    if avt_df is None:
        out.attrs["avt_join_report"] = {
            "mode": "missing_source",
            "matched_rows": 0,
            "unmatched_rows": int(len(out)),
            "unmatched_share": 1.0 if len(out) else 0.0,
            "future_measurements_used": False,
        }
        return out

    avt = normalize_telemetry(avt_df, config, prefix="avt")
    if avt["state_time"].duplicated().any():
        duplicates = int(avt["state_time"].duplicated().sum())
        raise ValueError(f"АВТ содержит дубли временных меток: {duplicates}")
    avt = avt.rename(columns={"state_time": "avt_source_time"})
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
            tolerance=pd.Timedelta(minutes=config.data.avt_asof_tolerance_minutes),
        )
        mode = "backward_asof"
    merged["avt_match_age_minutes"] = (merged["state_time"] - merged["avt_source_time"]) / pd.Timedelta(minutes=1)
    matched = merged["avt_source_time"].notna()
    merged["avt_match_status"] = matched.map({True: mode, False: "unmatched"})
    merged.attrs["avt_join_report"] = {
        "mode": mode,
        "tolerance_minutes": config.data.avt_asof_tolerance_minutes,
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "unmatched_share": float((~matched).mean()) if len(merged) else 0.0,
        "max_age_minutes": None if not matched.any() else float(merged.loc[matched, "avt_match_age_minutes"].max()),
        "future_measurements_used": False,
    }
    return merged.sort_values("state_time").reset_index(drop=True)


def canonicalize_pak(pak_df: pd.DataFrame | None, config: QualityAgentConfig) -> pd.DataFrame:
    """Выделяет сравнительную серу ПАК в мг/кг.

    Вход: длинная таблица ПАК из `parse_pak` или None. Выход: `pak_time` и
    `pak_sulfur_mg_kg`. Существенное условие: ПАК не является целью первого
    обучения, а используется только для сравнения.
    """

    if pak_df is None:
        return pd.DataFrame(columns=["pak_time", "pak_sulfur_mg_kg"])
    pak = pak_df[pak_df["tag"] == config.data.pak_target_tag].copy()
    pak["pak_time"] = to_datetime_ns(pak["timestamp"]).to_numpy()
    pak["pak_sulfur_mg_kg"] = pd.to_numeric(pak["value"], errors="coerce")
    return (
        pak[["pak_time", "pak_sulfur_mg_kg"]]
        .dropna(subset=["pak_time"])
        .sort_values("pak_time")
        .reset_index(drop=True)
    )


def canonicalize_lims_parameter(
    lims_df: pd.DataFrame | None,
    *,
    sampling_point: str,
    parameter: str,
    config: QualityAgentConfig,
    value_col: str,
) -> pd.DataFrame:
    """Выделяет один лабораторный показатель и время его доступности.

    Вход: длинная таблица ЛИМС, точка отбора и параметр. Выход: `sample_time`,
    `available_at`, значение и признак допущенной публикации. Существенное
    условие: `% масс.` для Mass.Sulfur переводится в мг/кг умножением на 10000.
    """

    if lims_df is None:
        return pd.DataFrame(columns=["sample_time", "available_at", value_col, "available_at_assumed"])
    lab = lims_df[
        (lims_df["sampling_point"] == sampling_point) & (lims_df["parameter"] == parameter)
    ].copy()
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
        lab["available_at_assumed"] = True
        lab["available_at"] = pd.NaT
    assumed = lab["sample_time"] + pd.Timedelta(hours=config.lims.assumed_publication_delay_hours)
    lab["available_at"] = lab["available_at"].fillna(assumed)
    return (
        lab[["sample_time", "available_at", value_col, "available_at_assumed"]]
        .dropna(subset=["sample_time", "available_at"])
        .sort_values("available_at")
        .reset_index(drop=True)
    )


def attach_latest_lims_feature(
    frame: pd.DataFrame,
    lab: pd.DataFrame,
    *,
    value_col: str,
    output_prefix: str,
    config: QualityAgentConfig,
) -> pd.DataFrame:
    """Присоединяет последний доступный лабораторный анализ.

    Вход: состояния установки и лабораторные события. Выход: состояния с
    последним известным значением, временем отбора, временем доступности,
    возрастом пробы и флагом устаревания. Существенное условие: используется
    только backward join по `available_at`.
    """

    out = frame.sort_values("state_time").copy()
    out["state_time"] = to_datetime_ns(out["state_time"]).to_numpy()
    if lab.empty:
        out[f"{output_prefix}_sample_time"] = pd.NaT
        out[f"{output_prefix}_available_at"] = pd.NaT
        out[f"{output_prefix}_available_at_assumed"] = True
        out[value_col] = pd.NA
        out[f"{output_prefix}_age_hours"] = pd.NA
        out[f"{output_prefix}_is_stale"] = False
        return out
    lab = lab.copy()
    lab["available_at"] = to_datetime_ns(lab["available_at"]).to_numpy()
    merged = pd.merge_asof(
        out,
        lab,
        left_on="state_time",
        right_on="available_at",
        direction="backward",
    )
    merged = merged.rename(
        columns={
            "sample_time": f"{output_prefix}_sample_time",
            "available_at": f"{output_prefix}_available_at",
            "available_at_assumed": f"{output_prefix}_available_at_assumed",
        }
    )
    age_col = f"{output_prefix}_age_hours"
    merged[age_col] = (merged["state_time"] - merged[f"{output_prefix}_sample_time"]) / pd.Timedelta(hours=1)
    too_old = merged[age_col] > config.lims.max_input_age_hours
    merged[f"{output_prefix}_is_stale"] = too_old.fillna(False)
    merged.loc[too_old, value_col] = pd.NA
    return merged


def forecast_horizon(config: QualityAgentConfig) -> pd.Timedelta:
    """Проверяет и возвращает горизонт forecast."""

    horizon = float(config.training.forecast_horizon_hours)
    if horizon < 0 or horizon > config.training.max_forecast_horizon_hours:
        raise ValueError(
            f"Горизонт forecast должен быть в диапазоне 0..{config.training.max_forecast_horizon_hours} ч"
        )
    return pd.Timedelta(hours=horizon)


def attach_q21_target(frame: pd.DataFrame, config: QualityAgentConfig) -> pd.DataFrame:
    """Формирует каноническую цель `target_sulfur_mg_kg` из Q21.

    Вход: нормализованная телеметрия 24-2000. Выход: датафрейм с
    `prediction_time`, `target_time`, целью и происхождением. Существенное
    условие: current берет Q21 из той же строки без заполнения соседями.
    """

    q21_col = "hdt_Q21" if "hdt_Q21" in frame.columns else "Q21"
    if q21_col not in frame.columns:
        raise ValueError("Для первого обучения нужна колонка hdt_Q21 в телеметрии гидроочистки")
    out = frame.copy()
    target_col = config.data.target_col
    if config.training.mode == "current":
        out["prediction_time"] = out["state_time"]
        out["target_time"] = out["state_time"]
        out[target_col] = pd.to_numeric(out[q21_col], errors="coerce")
    elif config.training.mode == "forecast":
        out["prediction_time"] = out["state_time"] + forecast_horizon(config)
        target = out[["state_time", q21_col]].rename(
            columns={"state_time": "target_time", q21_col: target_col}
        )
        target[target_col] = pd.to_numeric(target[target_col], errors="coerce")
        out = pd.merge_asof(
            out.sort_values("prediction_time"),
            target.sort_values("target_time"),
            left_on="prediction_time",
            right_on="target_time",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=config.training.target_match_tolerance_minutes),
        ).sort_values("state_time")
    else:
        raise ValueError("training.mode должен быть 'current' или 'forecast'")
    out[config.data.target_source_col] = f"telemetry:{q21_col}:ppm"
    out["target_unit"] = "мг/кг"
    out = apply_confirmed_q21_rejections(out, config)
    out["target_sulfur_above_10_mg_kg"] = out[target_col] > 10.0
    return out


def apply_confirmed_q21_rejections(frame: pd.DataFrame, config: QualityAgentConfig) -> pd.DataFrame:
    """Отклоняет подтвержденные ошибочные целевые измерения Q21.

    Вход: датафрейм после формирования `target_sulfur_mg_kg`. Выход: копия,
    где исходное целевое значение сохранено в `target_sulfur_raw_mg_kg`, а
    подтвержденный экспертами код `Q21 == 307 ppm` заменен на NaN только в
    целевой колонке. Существенное условие: строка и остальные измерения не
    удаляются, превышения 10 мг/кг без подтверждения ошибки сохраняются.
    """

    target_col = config.data.target_col
    out = frame.copy()
    raw_col = "target_sulfur_raw_mg_kg"
    flag_col = "target_sulfur_rejected"
    reason_col = "target_sulfur_rejection_reason"
    out[raw_col] = pd.to_numeric(out[target_col], errors="coerce")
    confirmed = out[raw_col].eq(307.0)
    out[flag_col] = confirmed
    out[reason_col] = ""
    out.loc[confirmed, reason_col] = "Подтвержденная экспертами ошибка: Q21=307 ppm"
    out.loc[confirmed, target_col] = pd.NA
    return out


def attach_pak_comparison(frame: pd.DataFrame, pak: pd.DataFrame, config: QualityAgentConfig) -> pd.DataFrame:
    """Привязывает ПАК как сравнительный источник около `prediction_time`."""

    if pak.empty:
        return frame
    out = frame.copy()
    out["prediction_time"] = to_datetime_ns(out["prediction_time"]).to_numpy()
    pak = pak.copy()
    pak["pak_time"] = to_datetime_ns(pak["pak_time"]).to_numpy()
    return pd.merge_asof(
        out.sort_values("prediction_time"),
        pak.sort_values("pak_time"),
        left_on="prediction_time",
        right_on="pak_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=config.training.target_match_tolerance_minutes),
    ).sort_values("state_time")


def build_base_frame(
    telemetry_df: pd.DataFrame,
    pak_df: pd.DataFrame | None,
    lims_df: pd.DataFrame | None,
    config: QualityAgentConfig,
    *,
    avt_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Собирает канонический подготовленный датафрейм.

    Вход: телеметрия гидроочистки, необязательные АВТ, ПАК и ЛИМС. Выход:
    датафрейм с временем состояния, целью hdt_Q21, входящей серой ЛИМС,
    сигналами обеих установок и ВАК-входами.
    """

    frame = attach_q21_target(normalize_telemetry(telemetry_df, config, prefix="hdt"), config)
    frame = attach_avt_telemetry(frame, avt_df, config)
    avt_join_report = frame.attrs.get("avt_join_report", {})
    input_lims = canonicalize_lims_parameter(
        lims_df,
        sampling_point=config.data.input_lims_sampling_point,
        parameter=config.data.input_lims_parameter,
        config=config,
        value_col="input_sulfur_mg_kg",
    )
    frame = attach_latest_lims_feature(
        frame,
        input_lims,
        value_col="input_sulfur_mg_kg",
        output_prefix="input_sulfur",
        config=config,
    )
    pipeline_d15 = canonicalize_lims_parameter(
        lims_df,
        sampling_point=config.data.input_lims_sampling_point,
        parameter="D15",
        config=config,
        value_col="pipeline_d15",
    )
    frame = attach_latest_lims_feature(
        frame,
        pipeline_d15,
        value_col="pipeline_d15",
        output_prefix="pipeline_d15",
        config=config,
    )
    pipeline_95pct_t = canonicalize_lims_parameter(
        lims_df,
        sampling_point=config.data.input_lims_sampling_point,
        parameter="95%.T",
        config=config,
        value_col="pipeline_95pct_t",
    )
    frame = attach_latest_lims_feature(
        frame,
        pipeline_95pct_t,
        value_col="pipeline_95pct_t",
        output_prefix="pipeline_95pct_t",
        config=config,
    )
    pak = canonicalize_pak(pak_df, config)
    if not pak.empty:
        frame = attach_pak_comparison(frame, pak, config)
    frame = frame.sort_values("state_time").reset_index(drop=True)
    frame.attrs["avt_join_report"] = avt_join_report
    return frame


def extract_output_lims(lims_df: pd.DataFrame | None, config: QualityAgentConfig) -> pd.DataFrame:
    """Возвращает лабораторную выходную серу как отдельные события.

    Вход: длинная таблица ЛИМС. Выход: `sample_time`, `available_at`,
    `output_lims_sulfur_mg_kg`. Существенное условие: таблица используется для
    проверки и не размножается по строкам телеметрии как цель.
    """

    return canonicalize_lims_parameter(
        lims_df,
        sampling_point=config.data.output_lims_sampling_point,
        parameter=config.data.output_lims_parameter,
        config=config,
        value_col="output_lims_sulfur_mg_kg",
    )


def dataset_checksum(paths: list[str | None]) -> str:
    """Считает контрольную сумму входных файлов.

    Вход: список путей, где None пропускается. Выход: SHA-256 по содержимому
    существующих файлов. Существенное условие: отсутствующие необязательные
    источники не считаются ошибкой.
    """

    import hashlib

    digest = hashlib.sha256()
    for value in paths:
        if not value:
            continue
        path = Path(value)
        if not path.exists():
            continue
        digest.update(str(path).encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

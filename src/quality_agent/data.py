"""Загрузка, канонизация и временное сопоставление данных агента качества."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lims_parser import parse_lims
from pak_parser import parse_pak

from .config import QualityAgentConfig


def to_datetime_ns(values: object) -> pd.Series:
    """Приводит время к единому dtype datetime64[ns].

    Вход: колонка или серия со временем. Выход: pandas Series с наносекундным
    dtype. Существенное условие: единый dtype нужен для `merge_asof` в pandas 3.
    """

    return pd.Series(pd.to_datetime(values, errors="coerce")).astype("datetime64[ns]")


def read_table(path: str | Path) -> pd.DataFrame:
    """Читает таблицу CSV или Parquet.

    Вход: путь к файлу. Выход: `DataFrame`. Существенное условие: расширение
    определяет формат, остальные форматы считаются ошибкой конфигурации.
    """

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
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


def load_sources(config: QualityAgentConfig) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Загружает исходные источники проекта.

    Вход: конфигурация с путями telemetry, ПАК и ЛИМС. Выход: три датафрейма,
    где ПАК и ЛИМС могут быть None. Существенное условие: для Excel ПАК/ЛИМС
    используются существующие парсеры команды.
    """

    telemetry = read_table(config.data.telemetry_path)
    pak = None
    lims = None
    if config.data.pak_path:
        pak = parse_pak(config.data.pak_path)
    if config.data.lims_path:
        lims = parse_lims(config.data.lims_path)
    return telemetry, pak, lims


def normalize_telemetry(df: pd.DataFrame, config: QualityAgentConfig) -> pd.DataFrame:
    """Приводит телеметрию к каноническому времени состояния.

    Вход: таблица `242000_tags.csv` или подготовленная таблица с аналогичными
    колонками. Выход: датафрейм с `state_time`. Существенное условие: пустая
    индексная колонка из CSV удаляется, а теги не переименовываются.
    """

    out = df.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])
    if "" in out.columns:
        out = out.drop(columns=[""])
    time_col = config.data.timestamp_col
    if "state_time" not in out.columns:
        if time_col not in out.columns:
            raise ValueError(f"В телеметрии нет колонки времени '{time_col}'")
        out = out.rename(columns={time_col: "state_time"})
    out["state_time"] = to_datetime_ns(out["state_time"]).to_numpy()
    if out["state_time"].isna().any():
        raise ValueError("В телеметрии есть строки с нераспознанным временем")
    return out.sort_values("state_time").reset_index(drop=True)


def canonicalize_pak(pak_df: pd.DataFrame | None, config: QualityAgentConfig) -> pd.DataFrame:
    """Выделяет целевую серу ПАК в мг/кг.

    Вход: длинная таблица ПАК из `parse_pak` или None. Выход: колонки
    `target_time`, `pak_sulfur_mg_kg`. Существенное условие: ppm для серы ПАК
    принимается эквивалентным мг/кг по README проекта.
    """

    if pak_df is None:
        return pd.DataFrame(columns=["target_time", "pak_sulfur_mg_kg"])
    target = pak_df[pak_df["tag"] == config.data.pak_target_tag].copy()
    target["target_time"] = to_datetime_ns(target["timestamp"]).to_numpy()
    target["pak_sulfur_mg_kg"] = pd.to_numeric(target["value"], errors="coerce")
    return (
        target[["target_time", "pak_sulfur_mg_kg"]]
        .dropna(subset=["target_time"])
        .sort_values("target_time")
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
    последним известным значением, временем отбора, временем доступности и
    возрастом пробы. Существенное условие: используется только backward join
    по `available_at`, будущие анализы не интерполируются.
    """

    out = frame.sort_values("state_time").copy()
    out["state_time"] = to_datetime_ns(out["state_time"]).to_numpy()
    if lab.empty:
        out[f"{output_prefix}_sample_time"] = pd.NaT
        out[f"{output_prefix}_available_at"] = pd.NaT
        out[f"{output_prefix}_available_at_assumed"] = True
        out[value_col] = pd.NA
        out[f"{output_prefix}_age_hours"] = pd.NA
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


def build_base_frame(
    telemetry_df: pd.DataFrame,
    pak_df: pd.DataFrame | None,
    lims_df: pd.DataFrame | None,
    config: QualityAgentConfig,
) -> pd.DataFrame:
    """Собирает канонический подготовленный датафрейм.

    Вход: телеметрия, ПАК и ЛИМС. Выход: датафрейм с временем состояния,
    технологическими сигналами, входящей серой ЛИМС и целевой ПАК. Существенное
    условие: выходной ЛИМС не размножается по строкам телеметрии как цель.
    """

    frame = normalize_telemetry(telemetry_df, config)
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
    target = canonicalize_pak(pak_df, config)
    if not target.empty:
        frame = attach_pak_target(frame, target, config)
        frame["target_pak_sulfur_above_10_mg_kg"] = frame["target_pak_sulfur_mg_kg"] > 10.0
    return frame.sort_values("state_time").reset_index(drop=True)


def attach_pak_target(frame: pd.DataFrame, target: pd.DataFrame, config: QualityAgentConfig) -> pd.DataFrame:
    """Привязывает целевую серу ПАК для текущего режима или горизонта.

    Вход: состояния и ряд ПАК. Выход: датафрейм с `prediction_time` и
    `target_pak_sulfur_mg_kg`. Существенное условие: для режима forecast цель
    ищется около `state_time + horizon` с заданной погрешностью.
    """

    out = frame.copy()
    horizon = pd.Timedelta(hours=config.training.forecast_horizon_hours)
    if config.training.mode == "current":
        out["prediction_time"] = out["state_time"]
    elif config.training.mode == "forecast":
        out["prediction_time"] = out["state_time"] + horizon
    else:
        raise ValueError("training.mode должен быть 'current' или 'forecast'")
    tolerance = pd.Timedelta(minutes=config.training.target_match_tolerance_minutes)
    out["prediction_time"] = to_datetime_ns(out["prediction_time"]).to_numpy()
    target = target.copy()
    target["target_time"] = to_datetime_ns(target["target_time"]).to_numpy()
    merged = pd.merge_asof(
        out.sort_values("prediction_time"),
        target.sort_values("target_time"),
        left_on="prediction_time",
        right_on="target_time",
        direction="nearest",
        tolerance=tolerance,
    )
    return merged.rename(columns={"pak_sulfur_mg_kg": "target_pak_sulfur_mg_kg"}).sort_values("state_time")


def extract_output_lims(lims_df: pd.DataFrame | None, config: QualityAgentConfig) -> pd.DataFrame:
    """Возвращает лабораторную выходную серу как отдельные события.

    Вход: длинная таблица ЛИМС. Выход: `sample_time`, `available_at`,
    `output_lims_sulfur_mg_kg`. Существенное условие: время пробы хранится
    отдельно от времени публикации и используется для проверки качества.
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

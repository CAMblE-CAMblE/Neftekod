"""
Расчет показателей качества гидроочищенного дизтоплива (24-2000:ГОДТ)
по формулам ВАК из Теги_хакатон.xlsx, лист «ВАК» — в исправленной версии,
полученной от организаторов хакатона (переписка по уточнению ТЗ).

ВАЖНО: формулы ниже отличаются от того, что буквально записано в ячейках
файла Теги_хакатон.xlsx! Организаторы подтвердили ошибки в исходных
формулах и прислали исправленные версии — именно они реализованы здесь.

T95 и D15 дополнительно требуют значений с точки отбора ЛИМС, названной в
формулах "Pipeline" (LIMS:24-2000.Pipeline.95%.T и LIMS:24-2000.Pipeline.D15).
Организаторы подтвердили: "Pipeline" — это точка отбора "Гидроочистка:1"
(сырье на входе в гидроочистку, продукт "ФРАКЦ_ДИЗ") в файле ЛИМС, не
отдельный источник. Функция attach_pipeline_lims ниже делает join тегов
с этой точкой по ближайшему времени, с учетом задержки публикации ЛИМС
до 4 часов.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UPLOADS = Path(__file__).resolve().parent / "data"

@dataclass
class AvtGodtTags:
    """
    Один снимок тегов КИП, нужных для формул ВАК (24-2000:ГОДТ).
    Имена полей совпадают с именами колонок в 242000_tags.csv.
    """
    F1: float
    F2: float
    F9: float
    F14: float
    F15: float
    F22: float
    F25: float
    F26: float
    P8: float
    P13: float
    P24: float
    T5: float
    T6: float
    T11: float
    T12: float
    T16: float
    T23: float
    W4: float
    W7: float

    @classmethod
    def from_row(cls, row: dict) -> "AvtGodtTags":
        """row — словарь или pandas.Series (строка из 242000_tags.csv)."""
        fields = cls.__dataclass_fields__.keys()
        return cls(**{f: float(row[f]) for f in fields})


def compute_t90(tags: AvtGodtTags) -> float:
    """
    24-2000:GODT:T90 — температура выкипания 90% фракции, °С.

    Возвращает NaN, если F26 == 0 — в датасете встречаются нулевые/битые
    показания этого тега (вероятно, пропуск датчика), а формула делит
    на F26, что иначе дает ZeroDivisionError на таких строках.
    """
    if tags.F26 == 0:
        return float("nan")
    return (
            162.998
            + 0.12945 * tags.T12
            + 59.57 * (tags.F15 / 2000)
            + 0.00036 * tags.W7
            + 0.26366 * tags.T23
            - 424.72638 * tags.F1 / tags.F26
    )


def compute_t50(tags: AvtGodtTags) -> float:
    """24-2000:GODT:T50 — температура выкипания 50% фракции, °С."""
    return 44.625 + 10.0224 * tags.P13 + 0.06981 * tags.F9 + 0.471 * tags.T6


def compute_cloud_point(tags: AvtGodtTags) -> float:
    """24-2000:GODT:CloudPoint — температура помутнения, °С."""
    return (
        0.0002 * tags.F22
        + 0.0021 * tags.W7
        + 0.00008 * tags.F25
        - 0.30656 * tags.F1
        + 0.12018 * tags.T6
        + 0.01916 * tags.F9
        - 48.254
        - 0.05249 * tags.T16
        + 0.00011
    )


def compute_cfpp(tags: AvtGodtTags) -> float:
    """24-2000:GODT:CFPP — предельная температура фильтруемости, °С."""
    return (
        0.22088 * tags.T23
        - 102.375
        - 47.75834 * tags.P8
        + 0.03862 * tags.F9
        + 43.60207 * tags.W7
        + 43.81849 * tags.P24
    )


def compute_i250(tags: AvtGodtTags) -> float:
    """24-2000:GODT:I250 — доля выкипания до 250°С, % об."""
    return (
        84.585
        - 0.21172 * tags.T5
        + 0.12137 * tags.T11
        - 0.00014 * tags.F25
        + 0.56248 * tags.F14
        - 0.16317 * tags.T23
        + 0.20272 * tags.T16
    )


def compute_ibp(tags: AvtGodtTags) -> float:
    """24-2000:GODT:IBP — начало кипения, °С."""
    return (
        137.762
        - 0.0653 * tags.F26
        + 0.00011 * tags.F22
        + 5.78137 * tags.P13
        - 34.58028 * tags.P24
        - 0.00993 * tags.F14
        - 0.99962 * tags.W4
        + 0.32232 * tags.T23
        - 0.09406 * tags.T16
    )


def compute_t95(tags: AvtGodtTags, lims_95pct_t_pipeline: float) -> float:
    """24-2000:GODT:T95 — температура выкипания 95% фракции, °С."""
    return (
            0.03814 * tags.F9
            - 9.201
            - 0.00002 * tags.F2
            + 0.50 * tags.T6
            + 0.48321 * lims_95pct_t_pipeline
    )


def compute_d15(tags: AvtGodtTags, lims_d15_pipeline: float) -> float:
    """24-2000:GODT:D15 — плотность при 15°С, кг/м3."""
    return (
            667.881 + 0.15417 * lims_d15_pipeline
            + 0.00005 * tags.F22
            + 0.10774 * tags.T11
    )


def attach_pipeline_lims(
        tags_df: pd.DataFrame,
        lims_df: pd.DataFrame,
        parameter: str,
        timestamp_col: str = "date",
        publication_delay_hours: float = 4.0,
) -> pd.Series:
    """
    Для каждой строки tags_df подбирает ближайшее ПО ВРЕМЕНИ значение
    указанного параметра с точки отбора "Гидроочистка:1" (= "Pipeline"
    в формулах ВАК), учитывая задержку публикации анализа в ЛИМС (до 4
    часов — подтверждено организаторами).

    tags_df — датафрейм тегов (242000_tags.csv) с колонкой времени
    (timestamp_col, по умолчанию "date").
    lims_df — результат lims_parser.parse_lims(...) (длинный формат:
    timestamp, sampling_point, product, parameter, unit, value).
    parameter — "95%.T" или "D15".

    Возвращает pd.Series той же длины и в том же порядке, что tags_df —
    последнее ИЗВЕСТНОЕ на момент строки значение параметра (NaN, если на
    этот момент ещt не было ни одного измерения).
    """
    lims_slice = (
        lims_df[
            (lims_df["sampling_point"] == "Гидроочистка:1")
            & (lims_df["parameter"] == parameter)
            ]
        .loc[:, ["timestamp", "value"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    # Значение реально доступно оператору не раньше timestamp + задержка публикации
    lims_slice["available_at"] = lims_slice["timestamp"] + pd.Timedelta(
        hours=publication_delay_hours
    )

    tags_sorted = tags_df[[timestamp_col]].copy()
    tags_sorted[timestamp_col] = pd.to_datetime(tags_sorted[timestamp_col])
    tags_sorted = tags_sorted.sort_values(timestamp_col)

    merged = pd.merge_asof(
        tags_sorted,
        lims_slice[["available_at", "value"]],
        left_on=timestamp_col,
        right_on="available_at",
        direction="backward",
    )
    merged.index = tags_sorted.index
    # merge_asof требует сортировки по времени — возвращаем в исходном порядке tags_df
    return merged["value"].reindex(tags_df.index)


def compute_all(
    tags: AvtGodtTags,
    lims_95pct_t_pipeline: float | None = None,
    lims_d15_pipeline: float | None = None,
) -> dict[str, float]:
    """
    Считает все доступные показатели качества по одному снимку тегов.
    T95/D15 считаются, только если переданы соответствующие значения из
    ЛИМС (точка "Гидроочистка:1").
    """
    result = {
        "T90": compute_t90(tags),
        "T50": compute_t50(tags),
        "I250": compute_i250(tags),
        "IBP": compute_ibp(tags),
        "CloudPoint": compute_cloud_point(tags),
        "CFPP": compute_cfpp(tags),
    }
    if lims_95pct_t_pipeline is not None:
        result["T95"] = compute_t95(tags, lims_95pct_t_pipeline)
    if lims_d15_pipeline is not None:
        result["D15"] = compute_d15(tags, lims_d15_pipeline)
    return result


if __name__ == "__main__":
    df = pd.read_csv(f"{UPLOADS}/242000_tags.csv")
    print(df.shape)
    print(df.head(5))

    tags = AvtGodtTags.from_row(df.iloc[0])
    print(compute_all(tags))

    quality_df = pd.DataFrame(
        [compute_all(AvtGodtTags.from_row(row)) for _, row in df.iterrows()]
    )
    print(quality_df.describe())

    nan_t90_count = quality_df["T90"].isna().sum()
    total = len(quality_df)
    print(
        f"T90 = NaN: {nan_t90_count} из {total} строк ({nan_t90_count / total:.2%}) — из-за F26 = 0")

    # Физически разумный диапазон для T90 — примерно [100, 500] °С
    implausible_t90 = quality_df[(quality_df["T90"] < 100) | (quality_df["T90"] > 500)]
    print(
        f"T90 вне разумного диапазона [100, 500]°С: {len(implausible_t90)} из {total} строк "
        f"({len(implausible_t90) / total:.2%})"
    )

    from lims_parser import parse_lims

    lims_df = parse_lims(f"{UPLOADS}/ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    t95_values = attach_pipeline_lims(df, lims_df, parameter="95%.T")
    d15_values = attach_pipeline_lims(df, lims_df, parameter="D15")
    full_quality_df = pd.DataFrame(
        [
            compute_all(AvtGodtTags.from_row(row), t95, d15)
            for row, t95, d15 in zip(df.to_dict("records"), t95_values, d15_values)
        ]
    )
    print(full_quality_df.describe())

    # Физически разумные диапазоны для T95 (°С) и D15 (кг/м3)
    implausible_t95 = full_quality_df[
        (full_quality_df["T95"] < 100) | (full_quality_df["T95"] > 500)
    ]
    print(
        f"T95 вне разумного диапазона [100, 500]°С: {len(implausible_t95)} из {total} строк "
        f"({len(implausible_t95) / total:.2%})"
    )
    implausible_d15 = full_quality_df[
        (full_quality_df["D15"] < 700) | (full_quality_df["D15"] > 1000)
    ]
    print(
        f"D15 вне разумного диапазона [700, 1000] кг/м3: {len(implausible_d15)} из {total} строк "
        f"({len(implausible_d15) / total:.2%})"
    )

    combined_df = pd.concat([df.reset_index(drop=True), full_quality_df], axis=1)
    combined_df.to_csv(f"{UPLOADS}/tags_with_quality.csv", index=False)

    first_valid_idx = combined_df["T95"].first_valid_index()
    print(combined_df.loc[first_valid_idx, ["date", "T95", "D15"]])

    first_valid_d15_idx = combined_df["D15"].first_valid_index()
    print(combined_df.loc[first_valid_d15_idx, ["date", "T95", "D15"]])


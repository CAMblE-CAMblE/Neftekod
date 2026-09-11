"""
Расчет показателей качества гидроочищенного дизтоплива (24-2000:ГОДТ)
по формулам ВАК из Теги_хакатон.xlsx, лист «ВАК» — в исправленной версии,
полученной от организаторов хакатона (переписка по уточнению ТЗ).

ВАЖНО: формулы ниже отличаются от того, что буквально записано в ячейках
файла Теги_хакатон.xlsx! Организаторы подтвердили ошибки в исходных
формулах и прислали исправленные версии — именно они реализованы здесь.

T95 дополнительно требует LIMS:24-2000.Pipeline.95%.T — нужно понять, что требуется
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
    T6: float
    T12: float
    T16: float
    T23: float
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


def compute_all(tags: AvtGodtTags) -> dict[str, float]:
    """
    Считает все доступные показатели качества по одному снимку тегов.
    T95 не считается — нужно думать
    """
    return {
        "T90": compute_t90(tags),
        "T50": compute_t50(tags),
        "CloudPoint": compute_cloud_point(tags),
        "CFPP": compute_cfpp(tags),
    }


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


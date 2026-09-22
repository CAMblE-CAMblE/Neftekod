"""
Параметры, для которых объявленная в файле единица измерения (строка 3)
ошибочна хотя бы в одном блоке. Подтверждено с организаторами хакатона:
"В ЛИМСах действительно возникла ошибка в строке единиц измерения, смотрите
на показатели качества в названиях." Ошибка обнаружена именно
в блоке "Установка 'АВТ'. Точка отбора '1'" —
юниты у 50%.T, EBP.T, D15, I350 там сдвинуты.

Вместо единицы из файла используем каноническую единицу по имени параметра
(большинством голосов по всем вхождениям параметра в файле). Это устойчиво
к другим возможным будущим сдвигам такого же рода, не только к уже найденным.
"""

from __future__ import annotations

import collections

import pandas as pd

from pathlib import Path

UPLOADS = Path(__file__).resolve().parent / "data"

_MANUAL_UNIT_OVERRIDE: dict[str, str] = {
    # I350 в данных дает ничью 1:1 между "°С" и "% об." — разрешаем вручную
    # по аналогии с I250 (везде "% об.", это доля выкипания до T, а не температура).
    "I350": "% об.",
}


def _canonical_units(param_row: pd.Series, unit_row: pd.Series) -> dict[str, str]:
    """
    Строит соответствие параметр -> единица измерения большинством
    голосов по всем встреченным в файле вхождениям этого параметра.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p, u in zip(param_row, unit_row):
        if pd.isna(p):
            continue
        p = str(p).strip()
        if pd.isna(u):
            continue
        votes[p][str(u).strip()] += 1

    canonical: dict[str, str] = {}
    for p, counter in votes.items():
        canonical[p] = counter.most_common(1)[0][0]
    canonical.update(_MANUAL_UNIT_OVERRIDE)
    return canonical


def parse_lims(path: str, sheet_name: str = 0) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # Берем последнее непустое значение и протягиваем его
    # вправо по всем NaN, пока не встретим следующее непустое
    group_header = raw.iloc[0].ffill()
    param_row = raw.iloc[1]      # короткое имя показателя
    unit_row = raw.iloc[2]       # единица измерения
    data = raw.iloc[4:].reset_index(drop=True)  # с 4-й строки идут дата/значение

    canonical_units = _canonical_units(param_row, unit_row)

    records = []
    n_cols = raw.shape[1]

    col = 0
    while col < n_cols - 1:
        group = group_header.iloc[col]
        param = param_row.iloc[col]
        raw_unit = unit_row.iloc[col]
        if pd.isna(group) or pd.isna(param):
            col += 1
            continue

        param_clean = str(param).strip()
        # Единица берется из канонического соответствия
        unit = canonical_units.get(param_clean)
        if unit is None:
            unit = None if pd.isna(raw_unit) else str(raw_unit).strip()

        # Разбираем "Установка 'АВТ'. Точка отбора '1'. Продукт 'Дизельное топливо'"
        sampling_point, product = _split_group(str(group))

        dates = pd.to_datetime(data.iloc[:, col], errors="coerce")
        values = pd.to_numeric(data.iloc[:, col + 1], errors="coerce")
        mask = dates.notna() & values.notna()
        if mask.any():
            block = pd.DataFrame(
                {
                    "timestamp": dates[mask].values,
                    "sampling_point": sampling_point,
                    "product": product,
                    "parameter": param_clean,
                    "unit": unit,
                    "value": values[mask].values,
                }
            )
            records.append(block)
        col += 2

    if not records:
        return pd.DataFrame(
            columns=["timestamp", "sampling_point", "product", "parameter", "unit", "value"]
        )

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _split_group(group: str) -> tuple[str, str]:
    """
    'Установка \'АВТ\'. Точка отбора \'1\'. Продукт \'Дизельное топливо\'' ->
    ("АВТ:1", "Дизельное топливо")
    """
    import re

    unit_m = re.search(r"Установка '([^']*)'", group)
    point_m = re.search(r"Точка отбора '([^']*)'", group)
    product_m = re.search(r"Продукт '([^']*)'", group)

    unit_name = unit_m.group(1) if unit_m else "?"
    point_name = point_m.group(1) if point_m else "?"
    product_name = product_m.group(1) if product_m else "?"
    return f"{unit_name}:{point_name}", product_name


if __name__ == "__main__":
    df = parse_lims(f"{UPLOADS}/ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    print(df.shape)
    print(df.head(15))
    print(df["parameter"].value_counts().head(20))
    print(df["sampling_point"].value_counts())
    print(df.groupby("parameter")["unit"].unique())

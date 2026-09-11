"""
Парсер файла ПАК (поточный анализатор качества) —
"Выгрузка_ПАК_01_01_2023_-_н_в_.xlsx".

Ряды разной длины (например, Mg.Sulfur и D15 покрывают разные периоды и
имеют разное число точек) — это нормально, обрабатывается через маску
непустых пар (дата, значение) отдельно по каждому тегу.

Единицы измерения по подтверждению организаторов:
- "ppm" для 24-2000:Mg.Sulfur эквивалентен мг/кг (пересчет не нужен) —
  это тот же физический показатель, что и Mg.Sulfur в ЛИМС, только
  непрерывное поточное измерение вместо периодической лабораторной пробы.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UPLOADS = Path(__file__).resolve().parent / "data"


def parse_pak(path: str, sheet_name: str = 0) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    tag_row = raw.iloc[0]
    unit_row = raw.iloc[1]
    data = raw.iloc[2:].reset_index(drop=True)

    records = []
    n_cols = raw.shape[1]

    col = 0
    while col < n_cols - 1:
        tag = tag_row.iloc[col]
        unit = unit_row.iloc[col]
        if pd.isna(tag):
            col += 1
            continue

        dates = pd.to_datetime(data.iloc[:, col], errors="coerce")
        values = pd.to_numeric(data.iloc[:, col + 1], errors="coerce")
        mask = dates.notna() & values.notna()
        if mask.any():
            block = pd.DataFrame(
                {
                    "timestamp": dates[mask].values,
                    "tag": str(tag).strip(),
                    "unit": None if pd.isna(unit) else str(unit).strip(),
                    "value": values[mask].values,
                }
            )
            records.append(block)
        col += 2

    if not records:
        return pd.DataFrame(columns=["timestamp", "tag", "unit", "value"])

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["tag", "timestamp"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = parse_pak(f"{UPLOADS}/Выгрузка ПАК 01.01.2023 - н.в_.xlsx")
    print(df.shape)
    print(df.head(10))
    print(df["tag"].value_counts())
    print(df.groupby("tag")["timestamp"].agg(["min", "max"]))
    print(df.groupby("tag")["unit"].unique())

"""Парсеры исходных Excel-таблиц ЛИМС и ПАК."""

from __future__ import annotations

import collections
import re

import pandas as pd


_MANUAL_UNIT_OVERRIDE = {"I350": "% об."}


def parse_lims(path: str, sheet_name: str | int = 0) -> pd.DataFrame:
    """Читает ЛИМС Excel в длинный формат timestamp/sampling_point/parameter/value."""

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    group_header = raw.iloc[0].ffill()
    param_row = raw.iloc[1]
    unit_row = raw.iloc[2]
    data = raw.iloc[4:].reset_index(drop=True)
    canonical_units = _canonical_units(param_row, unit_row)
    records = []
    col = 0
    while col < raw.shape[1] - 1:
        group = group_header.iloc[col]
        param = param_row.iloc[col]
        raw_unit = unit_row.iloc[col]
        if pd.isna(group) or pd.isna(param):
            col += 1
            continue
        param_clean = str(param).strip()
        sampling_point, product = _split_group(str(group))
        unit = canonical_units.get(param_clean)
        if unit is None:
            unit = None if pd.isna(raw_unit) else str(raw_unit).strip()
        dates = pd.to_datetime(data.iloc[:, col], errors="coerce")
        values = pd.to_numeric(data.iloc[:, col + 1], errors="coerce")
        mask = dates.notna() & values.notna()
        if mask.any():
            records.append(
                pd.DataFrame(
                    {
                        "timestamp": dates[mask].values,
                        "sampling_point": sampling_point,
                        "product": product,
                        "parameter": param_clean,
                        "unit": unit,
                        "value": values[mask].values,
                    }
                )
            )
        col += 2
    if not records:
        return pd.DataFrame(columns=["timestamp", "sampling_point", "product", "parameter", "unit", "value"])
    return pd.concat(records, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def parse_pak(path: str, sheet_name: str | int = 0) -> pd.DataFrame:
    """Читает Excel ПАК в длинный формат timestamp/tag/unit/value."""

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    tag_row = raw.iloc[0]
    unit_row = raw.iloc[1]
    data = raw.iloc[2:].reset_index(drop=True)
    records = []
    col = 0
    while col < raw.shape[1] - 1:
        tag = tag_row.iloc[col]
        unit = unit_row.iloc[col]
        if pd.isna(tag):
            col += 1
            continue
        dates = pd.to_datetime(data.iloc[:, col], errors="coerce")
        values = pd.to_numeric(data.iloc[:, col + 1], errors="coerce")
        mask = dates.notna() & values.notna()
        if mask.any():
            records.append(
                pd.DataFrame(
                    {
                        "timestamp": dates[mask].values,
                        "tag": str(tag).strip(),
                        "unit": None if pd.isna(unit) else str(unit).strip(),
                        "value": values[mask].values,
                    }
                )
            )
        col += 2
    if not records:
        return pd.DataFrame(columns=["timestamp", "tag", "unit", "value"])
    return pd.concat(records, ignore_index=True).sort_values(["tag", "timestamp"]).reset_index(drop=True)


def _canonical_units(param_row: pd.Series, unit_row: pd.Series) -> dict[str, str]:
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for param, unit in zip(param_row, unit_row):
        if pd.isna(param) or pd.isna(unit):
            continue
        votes[str(param).strip()][str(unit).strip()] += 1
    result = {param: counter.most_common(1)[0][0] for param, counter in votes.items()}
    result.update(_MANUAL_UNIT_OVERRIDE)
    return result


def _split_group(group: str) -> tuple[str, str]:
    unit_m = re.search(r"Установка '([^']*)'", group)
    point_m = re.search(r"Точка отбора '([^']*)'", group)
    product_m = re.search(r"Продукт '([^']*)'", group)
    unit_name = unit_m.group(1) if unit_m else "?"
    point_name = point_m.group(1) if point_m else "?"
    product_name = product_m.group(1) if product_m else "?"
    return f"{unit_name}:{point_name}", product_name

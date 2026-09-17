"""Синтетический пример для проверки каркаса без производственных данных."""

from __future__ import annotations

import math

import pandas as pd


def make_synthetic_sources(n: int = 96) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Создает малый маркированный набор телеметрии, ПАК и ЛИМС.

    Вход: число 10-минутных состояний. Выход: три датафрейма в форматах
    существующих парсеров. Существенное условие: данные искусственные и не
    отражают реальную установку.
    """

    times = pd.date_range("2026-01-01", periods=n, freq="10min")
    telemetry = pd.DataFrame({"date": times})
    columns = [
        "F1",
        "F2",
        "P3",
        "W4",
        "T5",
        "T6",
        "W7",
        "P8",
        "F9",
        "W10",
        "T11",
        "T12",
        "P13",
        "F14",
        "F15",
        "T16",
        "F17",
        "T18",
        "F19",
        "Q20",
        "Q21",
        "F22",
        "T23",
        "P24",
        "F25",
        "F26",
    ]
    for idx, name in enumerate(columns):
        telemetry[name] = 10 + idx + pd.Series(range(n)).map(lambda x: math.sin(x / 8 + idx) * 0.5)
    telemetry["P8"] = 340 + pd.Series(range(n)).map(lambda x: math.sin(x / 12) * 4)
    telemetry["T11"] = 120 + pd.Series(range(n)).map(lambda x: math.cos(x / 10) * 3)
    telemetry["F19"] = 4.0 + pd.Series(range(n)).map(lambda x: math.sin(x / 9) * 0.2)
    sulfur = 6.0 + (telemetry["T11"] - 120) * 0.05 - (telemetry["P8"] - 340) * 0.02 + pd.Series(range(n)) * 0.005
    pak = pd.DataFrame(
        {
            "timestamp": times,
            "tag": "24-2000:Mg.Sulfur",
            "unit": "ppm",
            "value": sulfur,
        }
    )
    lab_times = pd.date_range("2026-01-01 00:20", periods=8, freq="2h")
    lims_in = pd.DataFrame(
        {
            "timestamp": lab_times,
            "sampling_point": "Гидроочистка:1",
            "product": "ФРАКЦ_ДИЗ",
            "parameter": "Mass.Sulfur",
            "unit": "% масс.",
            "value": [0.055 + i * 0.001 for i in range(len(lab_times))],
        }
    )
    lims_out = pd.DataFrame(
        {
            "timestamp": lab_times + pd.Timedelta(minutes=20),
            "sampling_point": "Гидроочистка:2",
            "product": "Дизельное топливо",
            "parameter": "Mg.Sulfur",
            "unit": "мг/кг",
            "value": [6.1 + i * 0.03 for i in range(len(lab_times))],
        }
    )
    return telemetry, pak, pd.concat([lims_in, lims_out], ignore_index=True)

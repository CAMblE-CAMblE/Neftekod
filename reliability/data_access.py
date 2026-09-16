"""
Доступ к телеметрии для агента надёжности.

Агенту нужна ИСТОРИЯ (тренды, дрейф, деградация), а не только снимок.
Поэтому читаем ряд телеметрии и отдаём окно строго до момента цикла
(t <= now) — без утечки будущего.

Крупный CSV (24-2000 ~91 МБ, АВТ ~241 МБ) читается один раз и кэшируется
в parquet, дальше — быстрые срезы в памяти.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _drop_service_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

def load_telemetry(csv_path: str | Path, cache_dir: str | Path = ".cache") -> pd.DataFrame:
    """Загрузить ряд телеметрии, `date` -> DatetimeIndex"""
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (csv_path.stem + ".parquet")

    if cache.exists() and cache.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_parquet(cache)

    df = pd.read_csv(csv_path)
    df = _drop_service_cols(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # числовые
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(cache)
    return df

class TelemetrySource:
    """Обёртка: держит ряд в памяти, отдаёт окна до заданного момента"""

    def __init__(self, csv_path: str | Path, cache_dir: str | Path = ".cache") -> None:
        self.df = load_telemetry(csv_path, cache_dir)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.df.index  # type: ignore[return-value]

    def snapshot(self, ts) -> pd.Series:
        """Последняя строка на момент ts (без утечки будущего)"""
        ts = pd.Timestamp(ts)
        sub = self.df.loc[:ts]
        if sub.empty:
            raise ValueError(f"нет данных на {ts}")
        return sub.iloc[-1]

    def window(self, ts, periods: int) -> pd.DataFrame:
        """Последние `periods` точек по времени, заканчивая на ts (t <= ts)"""
        ts = pd.Timestamp(ts)
        return self.df.loc[:ts].tail(periods)

    def history(self, ts) -> pd.DataFrame:
        """Вся история до ts включительно"""
        return self.df.loc[: pd.Timestamp(ts)]

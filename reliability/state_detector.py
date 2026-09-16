"""
Блок A - детектор состояния установки.

steady_normal / shutdown / startup / transient.

Зачем: эксперт подтвердил — установки стоят 3–4% времени (T реактора -> ~0,
расход -> ~0). Во время останова/пуска рекомендации давать нельзя.
Проверено по данным 24-2000: near-zero температуры реактора ~2.9% точек,
расход ~4% — чёткая бимодальность, останов отделяется порогом.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class StateResult:
    state: str # steady_normal | shutdown | startup | transient
    regime_allowed: bool
    factors: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)

def _first_present(row: pd.Series, tags: list[str]) -> tuple[str | None, float]:
    for t in tags:
        if t in row and pd.notna(row[t]):
            return t, float(row[t])
    return None, float("nan")

def _robust_working_median(series: pd.Series) -> float:
    """Медиана рабочих значений"""
    s = series.dropna()
    if s.empty:
        return float("nan")
    working = s[s > s.quantile(0.5) * 0.3]  # грубо отсечь останов перед оценкой полки
    return float(working.median()) if not working.empty else float(s.median())

def detect_state(window: pd.DataFrame, row: pd.Series, cfg: dict) -> StateResult:
    rt_tags = cfg["risk_tags"]["reactor_temp"]
    feed_tags = cfg["risk_tags"]["feed_flow"]
    sd = cfg["state_detector"]

    rt_tag, rt_val = _first_present(row, rt_tags)
    feed_tag, feed_val = _first_present(row, feed_tags)

    rt_med = _robust_working_median(window[rt_tag]) if rt_tag else float("nan")
    feed_med = _robust_working_median(window[feed_tag]) if feed_tag else float("nan")

    factors: list[str] = []
    details = {
        "reactor_temp": rt_val,
        "reactor_temp_working_median": rt_med,
        "feed": feed_val,
        "feed_working_median": feed_med 
    }

    temp_down = np.isfinite(rt_med) and rt_val < sd["shutdown_temp_frac"] * rt_med
    feed_down = np.isfinite(feed_med) and feed_val < sd["shutdown_feed_frac"] * feed_med

    # останов
    if temp_down or feed_down:
        # различаем останов и пуск по знаку тренда температуры
        trend = _trend(window[rt_tag], sd["drift_window"]) if rt_tag else 0.0
        details["reactor_temp_trend"] = trend
        if trend > 0:
            factors.append(f"пуск установки: {rt_tag}={rt_val:.1f} растёт, ещё не вышел на режим")
            return StateResult("startup", False, factors, details)
        factors.append(
            f"останов установки: {rt_tag}={rt_val:.1f} (норма ~{rt_med:.0f}), расход {feed_tag}={feed_val:.1f}"
        )
        return StateResult("shutdown", False, factors, details)

    # переходный режим (быстрый дрейф ключевых величин)
    z = _drift_z(window[rt_tag], sd["drift_window"]) if rt_tag else 0.0
    details["reactor_temp_drift_z"] = z
    if abs(z) > sd["transient_z"]:
        factors.append(f"переходный режим: {rt_tag} быстро меняется (z={z:.1f}), прогнозы ненадёжны")
        return StateResult("transient", False, factors, details)

    return StateResult("steady_normal", True, ["режим установившийся"], details)

def _trend(series: pd.Series, window: int) -> float:
    """Знак/величина среднего приращения за окно"""
    s = series.dropna().tail(window)
    if len(s) < 2:
        return 0.0
    return float(s.iloc[-1] - s.iloc[0])

def _drift_z(series: pd.Series, window: int) -> float:
    """Скорость дрейфа за окно, нормированная на робастную σ приращений всей истории"""
    s = series.dropna()
    if len(s) < window + 1:
        return 0.0
    diffs = s.diff().dropna()
    mad = (diffs - diffs.median()).abs().median()
    sigma = 1.4826 * mad if mad > 0 else diffs.std()
    if not sigma or not np.isfinite(sigma):
        return 0.0
    recent_rate = (s.iloc[-1] - s.iloc[-window]) / window
    return float(recent_rate / sigma)
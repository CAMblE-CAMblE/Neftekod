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

def _smoothed_last(series: pd.Series, k: int = 3) -> float:
    """Текущее значение как медиана последних k точек гасит одиночные всплески/NaN"""
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.tail(k).median())


def detect_state(window: pd.DataFrame, row: pd.Series, cfg: dict, history: pd.DataFrame | None = None) -> StateResult:
    """
    window — последние N точек до ts
    row — снимок на ts
    history — вся история до ts; по ней считаем ГЛОБАЛЬНУЮ норму (устойчиво в длинном останове)
    """
    rt_tags = cfg["risk_tags"]["reactor_temp"]
    feed_tags = cfg["risk_tags"]["feed_flow"]
    sd = cfg["state_detector"]
    ref = history if history is not None else window     # глобальная норма, а не окно
    k = int(sd.get("smooth_k", 3))
    n = int(sd.get("steady_min_run", 18))

    rt_tag, _ = _first_present(row, rt_tags)
    feed_tag, _ = _first_present(row, feed_tags)

    # НОРМА — по глобальной истории (в длинном простое окно бы схлопнулось)
    rt_med = _robust_working_median(ref[rt_tag]) if rt_tag else float("nan")
    feed_med = _robust_working_median(ref[feed_tag]) if feed_tag else float("nan")

    # ТЕКУЩЕЕ — сглаженное (медиана последних k точек)
    rt_val = _smoothed_last(window[rt_tag], k) if rt_tag else float("nan")
    feed_val = _smoothed_last(window[feed_tag], k) if feed_tag else float("nan")

    temp_down = np.isfinite(rt_med) and np.isfinite(rt_val) and rt_val < sd["shutdown_temp_frac"] * rt_med
    feed_down = np.isfinite(feed_med) and np.isfinite(feed_val) and feed_val < sd["shutdown_feed_frac"] * feed_med
    down_now = temp_down or feed_down

    # ГИСТЕРЕЗИС: доля точек за последние n, где режим «внизу» (устойчивость простоя)
    recent = window.tail(n)
    down_mask = pd.Series(False, index=recent.index)
    if rt_tag and np.isfinite(rt_med):
        down_mask |= recent[rt_tag] < sd["shutdown_temp_frac"] * rt_med
    if feed_tag and np.isfinite(feed_med):
        down_mask |= recent[feed_tag] < sd["shutdown_feed_frac"] * feed_med
    down_frac = float(down_mask.mean()) if len(down_mask) else 0.0

    factors: list[str] = []
    details = {
        "reactor_temp": rt_val,
        "reactor_temp_working_median": rt_med,
        "feed": feed_val,
        "feed_working_median": feed_med,
        "down_frac": down_frac
    }

    # устойчивый простой (внизу и сейчас, и большую часть последних n точек)
    if down_now and down_frac >= 0.5:
        trend = _trend(window[rt_tag], sd["drift_window"]) if rt_tag else 0.0
        details["reactor_temp_trend"] = trend
        if trend > 0:
            factors.append(f"пуск установки: {rt_tag}={rt_val:.1f} растёт к норме ~{rt_med:.0f}")
            return StateResult("startup", False, factors, details)
        factors.append(
            f"останов установки: {rt_tag}={rt_val:.1f} (норма ~{rt_med:.0f}), расход {feed_tag}={feed_val:.1f}"
        )
        return StateResult("shutdown", False, factors, details)

    # краткий провал (сейчас внизу, но это не устойчивый простой) — вход/выход из останова
    if down_now:
        factors.append(f"переходный режим: краткий провал {rt_tag}/{feed_tag}, ещё не установившийся")
        return StateResult("transient", False, factors, details)

    # быстрый дрейф ключевой величины
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
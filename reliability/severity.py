"""
Блок B - индекс тяжести режима и риск оборудования/катализатора.

risk_index (0..1) = взвешенная свертка прокси-факторов

Факторы:
  temp_approach - близость температуры реактора к верхнему рабочему перцентилю
  catalyst_trend - деградация катализатора: медленный рост температуры реактора
                    по кампании (чтобы держать серу, WABT приходится повышать)
  reactor_dp - перепад давления на реакторе (загрязнение/коксование слоя)
  gas_feed_ratio - соотношение газ поддува / сырьё (низкое = риск коксования)
  drift - скорость дрейфа режима (переходность)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class SeverityResult:
    risk_index: float
    risk_class: str
    factors: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)

def _first_tag(cols, tags: list[str]) -> str | None:
    return next((t for t in tags if t in cols), None)

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def _approach_score(val: float, lo: float, hi: float) -> float:
    """0 у медианы диапазона, 1 на верхней границе и выше"""
    if not np.isfinite(val) or hi <= lo:
        return 0.0
    return _clip01((val - lo) / (hi - lo))

def _catalyst_deactivation(history: pd.DataFrame, rt_tag: str | None, sulf_tag: str | None, cfg: dict):
    """
    Прокси деградации катализатора: скорость роста WABT, нужной чтобы держать серу у спеки.

    Идея: если катализатор стареет, для той же степени обессеривания приходится
    повышать температуру. Берём точки кампании, где сера у целевого уровня (±band),
    и считаем наклон температуры во времени. Возвращает (score 0..1, °C/мес, n точек).
    """
    sev = cfg["severity"]
    cat = sev["catalyst"]
    if rt_tag is None or sulf_tag is None or sulf_tag not in history.columns:
        return 0.0, 0.0, 0
    win = history[[rt_tag, sulf_tag]].dropna().tail(sev["catalyst_trend_window"])
    if len(win) < 1000:
        return 0.0, 0.0, 0
    s = win[sulf_tag]
    # выкинуть выбросы серы (Q21=307) и останов (низкая температура)
    ok = (s > 0) & (s < cat["sulfur_outlier_ppm"]) & (win[rt_tag] > win[rt_tag].median() * 0.5)
    lo = cat["target_sulfur_ppm"] - cat["sulfur_band_ppm"]
    hi = cat["target_sulfur_ppm"] + cat["sulfur_band_ppm"]
    band = win[ok & (s >= lo) & (s <= hi)]
    if len(band) < 200:
        return 0.0, 0.0, len(band)
    t_days = (band.index - band.index[0]).total_seconds().to_numpy() / 86400.0
    slope_per_day = float(np.polyfit(t_days, band[rt_tag].to_numpy(), 1)[0])
    per_month = slope_per_day * 30.0
    return _clip01(per_month / cat["ref_deg_per_month"]), per_month, len(band)


def compute_severity(history: pd.DataFrame, row: pd.Series, cfg: dict, drift_z: float = 0.0) -> SeverityResult:
    rt_tags = cfg["risk_tags"]["reactor_temp"]
    dp_tags = cfg["risk_tags"]["reactor_dp"]
    gas_tags = cfg["risk_tags"]["makeup_gas"]
    feed_tags = cfg["risk_tags"]["feed_flow"]
    sev = cfg["severity"]
    w = sev["weights"]

    rt = _first_tag(history.columns, rt_tags)
    dp = _first_tag(history.columns, dp_tags)
    gas = _first_tag(history.columns, gas_tags)
    feed = _first_tag(history.columns, feed_tags)

    win = history.tail(sev["window_steady"])
    factors: list[str] = []
    contrib: dict[str, float] = {}

    # temp_approach
    f_temp = 0.0
    if rt:
        lo = win[rt].quantile(0.5)
        hi = win[rt].quantile(sev["approach_hi_pctl"] / 100)
        f_temp = _approach_score(row.get(rt, np.nan), lo, hi)
        if f_temp > 0.7:
            factors.append(f"температура реактора {rt}={row[rt]:.1f} у верхнего рабочего предела")
    contrib["temp_approach"] = f_temp

    # catalyst_trend - деградация катализатора: рост WABT при удержании серы у спеки
    sulf_tag = _first_tag(history.columns, cfg["risk_tags"].get("sulfur_out", []))
    f_cat, cat_per_month, cat_n = _catalyst_deactivation(history, rt, sulf_tag, cfg)
    if f_cat > 0.4 and cat_n >= 200:
        factors.append(
            f"деградация катализатора: чтобы держать серу ~{sev['catalyst']['target_sulfur_ppm']:.0f} ppm, "
            f"WABT растёт ~{cat_per_month:.1f}°C/мес"
        )
    contrib["catalyst_trend"] = f_cat

    # reactor_dp - перепад давления к верхнему перцентилю
    f_dp = 0.0
    if dp:
        lo = win[dp].quantile(0.5)
        hi = win[dp].quantile(sev["approach_hi_pctl"] / 100)
        f_dp = _approach_score(row.get(dp, np.nan), lo, hi)
        if f_dp > 0.7:
            factors.append(f"перепад давления {dp}={row[dp]:.2f} высок -> загрязнение слоя")
    contrib["reactor_dp"] = f_dp

    # gas_feed_ratio - низкое соотношение газ/сырьё -> риск коксования
    f_gas = 0.0
    if gas and feed and row.get(feed):
        ratio = row.get(gas, np.nan) / row[feed] if row[feed] else np.nan
        hist_ratio = (win[gas] / win[feed].replace(0, np.nan)).dropna()
        if len(hist_ratio) > 100 and np.isfinite(ratio):
            lo = hist_ratio.quantile(sev["approach_lo_pctl"] / 100)
            med = hist_ratio.quantile(0.5)
            # чем нниже отношение тем выше риск
            f_gas = _clip01((med - ratio) / (med - lo)) if med > lo else 0.0
            if f_gas > 0.7:
                factors.append("низкое соотношение газ/сырьё -> риск коксования катализатора")
    contrib["gas_feed_ratio"] = f_gas

    # drift
    f_drift = _clip01(abs(drift_z) / (2 * cfg["state_detector"]["transient_z"]))
    contrib["drift"] = f_drift

    # свертка
    total_w = sum(w.values())
    risk_index = sum(w[k] * contrib.get(k, 0.0) for k in w) / total_w
    risk_index = round(_clip01(risk_index), 3)

    ct = sev["class_thresholds"]
    risk_class = "low" if risk_index < ct["low"] else ("medium" if risk_index < ct["medium"] else "high")

    if not factors:
        factors.append("режим в рабочих полосах, нет критичных факторов риска")

    return SeverityResult(risk_index, risk_class, factors, {k: round(v, 3) for k, v in contrib.items()})

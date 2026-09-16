"""
Блок C - коридоры для оптимизатора (optimization_constraints).

На каждый управляемый тег отдаём [lo, hi]. База — робастные перцентили
ТОЛЬКО по steady-периодам (иначе останов/пуск растянет коридор в ноль).
Правило ТЗ: исторический min/max - не паспортный предел, помечаем как допущение.
При высоком risk_index коридор сужаем к медиане.
"""

from __future__ import annotations
import pandas as pd

def _steady_mask(history: pd.DataFrame, reactor_tags: list[str], feed_tags: list[str]) -> pd.Series:
    """Грубая маска рабочих периодов: температура реактора и расход выше доли медианы"""
    mask = pd.Series(True, index=history.index)
    for tags in (reactor_tags, feed_tags):
        tag = next((t for t in tags if t in history.columns), None)
        if tag is None:
            continue
        s = history[tag]
        working_med = s[s > s.median() * 0.3].median()
        mask &= s > 0.5 * working_med
    return mask

def compute_constraints(history: pd.DataFrame, cfg: dict, risk_index: float) -> tuple[dict[str, tuple[float, float]], list[str]]:
    cc = cfg["constraints"]
    control_tags = cc["control_tags_u242"] + cc["control_tags_avt"]
    reactor_tags = cfg["risk_tags"]["reactor_temp"]
    feed_tags = cfg["risk_tags"]["feed_flow"]

    steady = history[_steady_mask(history, reactor_tags, feed_tags)]

    constraints: dict[str, tuple[float, float]] = {}
    assumptions: list[str] = [
        "технологические режимные границы: коридоры выведены из истории рабочих (steady) "
        "периодов, а не из паспортных пределов оборудования"
    ]
    high_risk = risk_index >= cfg["severity"]["class_thresholds"]["medium"]

    for tag in control_tags:
        if tag not in steady.columns:
            continue
        s = steady[tag].dropna()
        if len(s) < 100:
            continue
        lo = float(s.quantile(cc["lo_pctl"] / 100))
        hi = float(s.quantile(cc["hi_pctl"] / 100))
        if high_risk:
            med = float(s.median())
            k = cc["tighten_on_high_risk"]
            lo, hi = med - (med - lo) * (1 - k), med + (hi - med) * (1 - k)
        constraints[tag] = (round(lo, 3), round(hi, 3))

    if high_risk:
        assumptions.append("режим тяжелый, коридоры сужены к медиане")
    return constraints, assumptions

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

def _percentile_corridor(steady: pd.DataFrame, tag: str, cc: dict, high_risk: bool):
    if tag not in steady.columns:
        return None
    s = steady[tag].dropna()
    if len(s) < 100:
        return None
    lo = float(s.quantile(cc["lo_pctl"] / 100))
    hi = float(s.quantile(cc["hi_pctl"] / 100))
    if high_risk:
        med = float(s.median())
        k = cc["tighten_on_high_risk"]
        lo, hi = med - (med - lo) * (1 - k), med + (hi - med) * (1 - k)
    return round(lo, 3), round(hi, 3)

def compute_constraints(history: pd.DataFrame, cfg: dict, risk_index: float, avt_history: pd.DataFrame | None = None) -> tuple[dict[str, tuple[float, float]], list[str]]:
    cc = cfg["constraints"]
    reactor_tags = cfg["risk_tags"]["reactor_temp"]
    feed_tags = cfg["risk_tags"]["feed_flow"]

    steady = history[_steady_mask(history, reactor_tags, feed_tags)]

    constraints: dict[str, tuple[float, float]] = {}
    assumptions: list[str] = [
        "технологические режимные границы: коридоры выведены из истории рабочих (steady) "
        "периодов, а не из паспортных пределов оборудования"
    ]
    high_risk = risk_index >= cfg["severity"]["class_thresholds"]["medium"]

    # управляемые теги 24-2000
    for tag in cc["control_tags_u242"]:
        corr = _percentile_corridor(steady, tag, cc, high_risk)
        if corr is not None:
            constraints[tag] = corr

    # управляемые теги АВТ - только если явно включено и переданы данные АВТ
    # ВАЖНО: имена тегов пересекаются между установками (T6, F25 есть и там, и там),
    # поэтому ключи неймспейсим как "AVT:<tag>", иначе коллизия смысла
    avt_tags = cc.get("control_tags_avt", [])
    if avt_tags:
        if cc.get("include_avt") and avt_history is not None:
            steady_avt = avt_history  # для АВТ отдельная steady-маска — TODO (backlog)
            for tag in avt_tags:
                corr = _percentile_corridor(steady_avt, tag, cc, high_risk)
                if corr is not None:
                    constraints[f"AVT:{tag}"] = corr
        else:
            assumptions.append(
                "коридоры АВТ не рассчитаны: нужна телеметрия АВТ + отдельная steady-маска "
            )

    if high_risk:
        assumptions.append("режим тяжёлый, коридоры сужены к медиане")
    return constraints, assumptions

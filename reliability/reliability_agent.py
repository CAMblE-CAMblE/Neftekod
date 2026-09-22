"""
Агент надёжности — главный класс.

assess(timestamp) -> dict в формате контракта `reliability`
(парсится оркестратором в json_io.parse_agent_inputs -> ReliabilityAssessment).

Пайплайн одного цикла:
  1. взять снимок + окно телеметрии до ts
  2. Блок A: состояние установки -> regime_allowed
  3. Блок B: индекс тяжести режима / риск катализатора
  4. Блок C: коридоры для оптимизатора
  5. собрать JSON контракта
"""

from __future__ import annotations
from pathlib import Path
import yaml
from .constraints import compute_constraints
from .data_access import TelemetrySource
from .severity import compute_severity
from .state_detector import detect_state

_HERE = Path(__file__).parent

class ReliabilityAgent:
    def __init__(self, config_path: str | Path | None = None) -> None:
        cfg_path = Path(config_path) if config_path else _HERE / "config.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        d = self.cfg["data"]

        root = _HERE.parent
        self.u242 = TelemetrySource(root / d["u242_csv"], root / d["cache_dir"])
        # АВТ подключаем только если нужны его теги в коридорах
        self._avt_path = root / d["avt_csv"]
        self._avt_cache = root / d["cache_dir"]
        self._avt: TelemetrySource | None = None

    @property
    def avt(self) -> TelemetrySource:
        if self._avt is None:
            self._avt = TelemetrySource(self._avt_path, self._avt_cache)
        return self._avt

    def assess(self, timestamp) -> dict:
        cfg = self.cfg
        row = self.u242.snapshot(timestamp)
        window = self.u242.window(timestamp, cfg["severity"]["window_steady"])
        history = self.u242.history(timestamp)

        # Блок A: состояние (норма — по глобальной истории)
        state = detect_state(window, row, cfg, history=history)

        # если установка стоит/пускается/переходный, то риск максимальный, рекомендаций нет
        if not state.regime_allowed:
            return {
                "risk_index": 1.0,
                "risk_class": "high",
                "risk_factors": state.factors,
                "regime_allowed": False,
                "optimization_constraints": {},
                "assumptions": [
                    f"состояние установки: {state.state}",
                    "во время останова/пуска/перехода рекомендации не выдаются"
                ],
                "_state": state.state, # это для отладки
                "_details": state.details
            }

        # Блок B: тяжесть режима
        drift_z = float(state.details.get("reactor_temp_drift_z", 0.0))
        sev = compute_severity(history, row, cfg, drift_z=drift_z)

        # Блок C: коридоры
        constraints, cons_assumptions = compute_constraints(history, cfg, sev.risk_index)

        return {
            "risk_index": sev.risk_index,
            "risk_class": sev.risk_class,
            "risk_factors": sev.factors,
            "regime_allowed": True,
            "optimization_constraints": {k: [v[0], v[1]] for k, v in constraints.items()},
            "assumptions": cons_assumptions,
            "_state": state.state,
            "_contributions": sev.contributions
        }

    # Оценка конкретного кандидата управляющих
    _SIM_CONTROLS = ("T6", "F9", "P13")

    def _corridor_base(self, timestamp) -> dict:
        """Коридоры + режим на момент timestamp"""
        key = str(timestamp)
        cached = getattr(self, "_cand_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        base = self.assess(timestamp)
        info = {
            "regime_allowed": base["regime_allowed"],
            "risk_index": base["risk_index"],
            "risk_class": base["risk_class"],
            "state": base.get("_state"),
            "corridors": base.get("optimization_constraints", {}) or {}
        }
        self._cand_cache = (key, info)
        return info

    @staticmethod
    def _normalize_controls(controls=None, **kw) -> dict:
        """Принимает dict {'T6':..}, объект с .t6/.f9/.p13 (напр. ControlSettings
        симулятора — без импорта его класса), или kwargs t6=/f9=/p13=."""
        src: dict = {}
        if controls is not None:
            if isinstance(controls, dict):
                src.update({str(k).upper(): v for k, v in controls.items()})
            else:
                for attr in ("t6", "f9", "p13", "f15", "f25"):
                    if hasattr(controls, attr):
                        src[attr.upper()] = getattr(controls, attr)
        src.update({str(k).upper(): v for k, v in kw.items()})
        return {k: float(v) for k, v in src.items() if v is not None}

    def assess_candidate(self, timestamp, controls=None, **kw) -> dict:
        """Допустим ли кандидат управляющих в момент timestamp.

        Возвращает контракт симулятора {is_allowed, reasons} + доп. поля
        (risk_index, risk_class, state, checked, corridors) — их конвертер может
        игнорировать. Правило: во время останова/пуска/перехода кандидаты не
        оцениваются; иначе проверяем каждый управляющий против его рабочего коридора"""
        cand = self._normalize_controls(controls, **kw)
        info = self._corridor_base(timestamp)

        if not info["regime_allowed"]:
            return {
                "is_allowed": False,
                "reasons": [f"режим не допускает рекомендаций (состояние: {info['state']}) — "
                            "во время останова/пуска/перехода кандидаты не оцениваются"],
                "risk_index": info["risk_index"], 
                "risk_class": info["risk_class"],
                "state": info["state"], 
                "checked": {}, 
                "corridors": info["corridors"]
            }

        corridors = info["corridors"]
        reasons: list[str] = []
        checked: dict[str, bool] = {}
        # проверяем управляющие симулятора (если переданы иные, то тоже проверим при наличии коридора)
        names = [n for n in self._SIM_CONTROLS if n in cand] or list(cand.keys())
        for name in names:
            v = cand[name]
            if name in corridors:
                lo, hi = corridors[name]
                ok = lo <= v <= hi
                checked[name] = ok
                if not ok:
                    reasons.append(f"{name}={v:.2f} вне рабочего коридора {lo:.2f}..{hi:.2f}")
            else:
                reasons.append(f"{name}: нет коридора (не проверено)")

        is_allowed = all(checked.values()) and len(checked) > 0
        if is_allowed and not any("вне рабочего коридора" in r for r in reasons):
            reasons = ["все управляющие в допустимых рабочих коридорах"] + reasons
        return {
            "is_allowed": is_allowed, 
            "reasons": reasons,
            "risk_index": info["risk_index"], 
            "risk_class": info["risk_class"],
            "state": info["state"], 
            "checked": checked, 
            "corridors": corridors
        }

    def assess_candidates(self, timestamp, candidates) -> list[dict]:
        """Пакетная оценка сетки кандидатов в одной точке"""
        return [self.assess_candidate(timestamp, c) for c in candidates]


def assess(timestamp, config_path: str | Path | None = None) -> dict:
    return ReliabilityAgent(config_path).assess(timestamp)

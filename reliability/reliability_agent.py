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

        # Блок A: состояние
        state = detect_state(window, row, cfg)

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

def assess(timestamp, config_path: str | Path | None = None) -> dict:
    return ReliabilityAgent(config_path).assess(timestamp)

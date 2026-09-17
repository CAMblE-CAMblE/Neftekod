"""Демонстрационная проверка надежности по допустимым диапазонам."""

from __future__ import annotations

from dataclasses import dataclass

from simulator.state import ControlSettings, ReliabilityAssessment


@dataclass(frozen=True)
class ControlRange:
    """Диапазон и шаг одного управляющего параметра."""

    min: float
    max: float
    step: float


@dataclass(frozen=True)
class ControlRanges:
    """Допустимые диапазоны T6, F9 и P13."""

    t6: ControlRange
    f9: ControlRange
    p13: ControlRange


@dataclass(frozen=True)
class DemoReliability:
    """Проверяет режим по простым границам из конфигурации.

    Вход: управляющие параметры. Выход: разрешение режима и причины отказа.
    Вероятности отказа и произвольные индексы риска не рассчитываются.
    """

    ranges: ControlRanges

    def assess(self, controls: ControlSettings) -> ReliabilityAssessment:
        """Возвращает результат проверки режима по T6, F9 и P13."""

        reasons: list[str] = []
        self._check("T6", controls.t6, self.ranges.t6, reasons)
        self._check("F9", controls.f9, self.ranges.f9, reasons)
        self._check("P13", controls.p13, self.ranges.p13, reasons)
        if reasons:
            return ReliabilityAssessment(is_allowed=False, reasons=reasons)
        return ReliabilityAssessment(is_allowed=True, reasons=["Режим находится в демонстрационных допустимых диапазонах."])

    @staticmethod
    def _check(name: str, value: float, control_range: ControlRange, reasons: list[str]) -> None:
        """Добавляет причину отказа, если значение вне диапазона."""

        if value < control_range.min or value > control_range.max:
            reasons.append(f"{name}={value:.2f} вне диапазона {control_range.min:.2f}..{control_range.max:.2f}.")

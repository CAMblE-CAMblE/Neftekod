"""Простой демонстрационный перебор режимов вокруг текущей точки."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from simulator.adapters.reliability import ControlRanges, DemoReliability
from simulator.models.base import QualityModel
from simulator.state import CandidateAction, ControlSettings, HistoricalState


@dataclass(frozen=True)
class DemoOptimizer:
    """Генерирует и оценивает сетку кандидатов T6/F9/P13.

    Вход: модель качества, проверка надежности, диапазоны и дискретные смещения.
    Выход: список оцененных кандидатов. Финальный выбор делает оркестратор.
    """

    quality_model: QualityModel
    reliability: DemoReliability
    ranges: ControlRanges
    search_steps: list[int]

    def build_candidates(
        self,
        base_state: HistoricalState,
        input_sulfur: float,
        sulfur_limit: float,
    ) -> list[CandidateAction]:
        """Строит оцененные варианты режима вокруг исходных T6/F9/P13."""

        candidates: list[CandidateAction] = []
        base = base_state.controls
        for t6_offset, f9_offset, p13_offset in product(self.search_steps, repeat=3):
            controls = ControlSettings(
                t6=round(base.t6 + t6_offset * self.ranges.t6.step, 3),
                f9=round(base.f9 + f9_offset * self.ranges.f9.step, 3),
                p13=round(base.p13 + p13_offset * self.ranges.p13.step, 3),
            )
            quality = self.quality_model.assess(base_state, input_sulfur, controls, sulfur_limit)
            reliability = self.reliability.assess(controls)
            deviation = self._deviation_score(base, controls)
            candidates.append(
                CandidateAction(
                    id=f"cand_{len(candidates) + 1:03d}",
                    controls=controls,
                    quality=quality,
                    reliability=reliability,
                    deviation_score=deviation,
                )
            )
        return candidates

    def _deviation_score(self, base: ControlSettings, candidate: ControlSettings) -> float:
        """Считает нормированное отклонение кандидата от исходного режима."""

        return (
            abs(candidate.t6 - base.t6) / self.ranges.t6.step
            + abs(candidate.f9 - base.f9) / self.ranges.f9.step
            + abs(candidate.p13 - base.p13) / self.ranges.p13.step
        )

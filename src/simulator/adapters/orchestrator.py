"""Детерминированный оркестратор демонстрационного симулятора."""

from __future__ import annotations

from dataclasses import dataclass

from simulator.state import CandidateAction, ControlSettings, Recommendation


@dataclass(frozen=True)
class DemoOrchestrator:
    """Выбирает лучший допустимый вариант без LLM.

    Вход: оцененные кандидаты. Выход: рекомендация или отказ. Правило выбора:
    сначала соблюдение серы и разрешение надежности, затем минимальное
    отклонение от исходного режима, затем меньшее значение Q21.
    """

    def recommend(self, candidates: list[CandidateAction], base_controls: ControlSettings) -> Recommendation:
        """Возвращает рекомендацию по списку оцененных кандидатов."""

        allowed = [
            item
            for item in candidates
            if item.quality.is_within_limit and item.reliability.is_allowed
        ]
        if not allowed:
            return Recommendation(
                is_actionable=False,
                status="решение отсутствует",
                explanation=(
                    "Не найден режим, который одновременно проходит ограничение по сере "
                    "и демонстрационную проверку оборудования."
                ),
            )
        best = sorted(allowed, key=lambda item: (item.deviation_score, item.quality.predicted_q21))[0]
        if best.controls == base_controls:
            explanation = "Текущий режим уже проходит ограничение; изменение режима не требуется."
            status = "допустим"
        else:
            explanation = (
                "Выбран ближайший к исходному допустимый режим, который снижает Q21 "
                "ниже заданного ограничения и проходит проверку оборудования."
            )
            status = "допустим"
        return Recommendation(is_actionable=True, status=status, explanation=explanation, candidate=best)

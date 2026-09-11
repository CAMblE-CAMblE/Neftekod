from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from .contracts import AgentInputs, Recommendation
from .hard_constraints import build_refusal_reason, data_is_too_stale, filter_admissible_scenarios
from .json_io import parse_agent_inputs
from .llm_client import FakeLLMClient, OllamaClient
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class Orchestrator:
    def __init__(self, llm: OllamaClient | FakeLLMClient):
        self.llm = llm

    def run_cycle_from_json(self, data: dict) -> Recommendation:
        """Точка входа, когда агенты присылают JSON.

        Не используется в demo.py сейчас (там load_agent_inputs вызывается заранее),
        но пригодится, если нужно будет вызывать оркестратор прямо из места,
        где данные приходят как raw JSON (например, HTTP-эндпоинт).
        """
        return self.run_cycle(parse_agent_inputs(data))

    def run_cycle(self, inputs: AgentInputs) -> Recommendation:
        # Жёсткие проверки — независимо от LLM. Смотрим ОБЕ причины разом
        # (устаревшие данные и отсутствие допустимых сценариев), чтобы отказ
        # мог сочетать обе, как в примере из ТЗ (раздел 5).
        admissible, all_checks = filter_admissible_scenarios(inputs)
        too_stale = data_is_too_stale(inputs)

        if too_stale or not admissible:
            reason = build_refusal_reason(
                too_stale=too_stale,
                lims_age_minutes=inputs.state.lims_age_minutes,
                all_checks=all_checks,
            )
            return self._refusal(inputs, reason=reason)

        # LLM выбирает и объясняет — только среди уже допустимых вариантов.
        user_prompt = USER_PROMPT_TEMPLATE.format(
            state_json=json.dumps(asdict(inputs.state), ensure_ascii=False, default=_json_default),
            quality_json=json.dumps(asdict(inputs.quality), ensure_ascii=False),
            reliability_json=json.dumps(asdict(inputs.reliability), ensure_ascii=False),
            scenarios_json=json.dumps([asdict(s) for s in admissible], ensure_ascii=False),
        )
        parsed = self._parse_llm_response_with_retry(SYSTEM_PROMPT, user_prompt)

        chosen_id = parsed.get("chosen_scenario_id")
        # Страховка: даже если LLM выбрал id вне списка допустимых — не доверяем.
        valid_ids = {s.scenario_id for s in admissible}
        if chosen_id not in valid_ids:
            chosen_id = admissible[0].scenario_id

        constraints_checked = [
            f"{r.scenario_id}: {'OK' if r.passed else '; '.join(r.violations)}"
            for r in all_checks
        ]

        return Recommendation(
            timestamp=inputs.state.timestamp,
            state_summary=parsed.get("state_summary", ""),
            problem_or_risk=parsed.get("problem_or_risk", ""),
            action=parsed.get("action"),
            expected_effect=parsed.get("expected_effect", ""),
            constraints_checked=constraints_checked,
            confidence=parsed.get("confidence", ""),
            explanation=parsed.get("explanation", ""),
            is_actionable=True,
            chosen_scenario_id=chosen_id,
        )

    def _refusal(self, inputs: AgentInputs, reason: str) -> Recommendation:
        return Recommendation(
            timestamp=inputs.state.timestamp,
            state_summary=f"Данные на {inputs.state.timestamp.isoformat()}",
            problem_or_risk=reason,
            action=None,
            expected_effect="—",
            constraints_checked=[reason],
            confidence="низкая",
            explanation=(
                "Надёжной рекомендации нет: " + reason + ". "
                "Умение корректно отказаться — часть корректной работы системы (см. ТЗ, раздел 5)."
            ),
            is_actionable=False,
            chosen_scenario_id=None,
        )

    @staticmethod
    def _try_parse_json(raw: str) -> dict | None:
        """Пробует распарсить чистый JSON, а если модель добавила текст
        вокруг — вырезает { ... }. Возвращает None, если не получилось."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start: end + 1])
                except json.JSONDecodeError:
                    pass
            return None

    def _parse_llm_response_with_retry(
            self, system_prompt: str, user_prompt: str, max_retries: int = 2
    ) -> dict:
        """
        Пытается получить валидный JSON от LLM. При неудаче переспрашивает,
        явно указав модели, что предыдущий ответ был невалидным —
        и только если все попытки провалились, возвращает заглушку с
        пометкой "не удалось разобрать ответ LLM".
        """
        raw = self.llm.chat(system_prompt, user_prompt, json_mode=True)

        for attempt in range(max_retries):
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                return parsed

            if attempt < max_retries - 1:
                retry_prompt = (
                    user_prompt
                    + f"\n\nТвой предыдущий ответ не был валидным JSON: {raw[:200]!r}. "
                    "Ответь СТРОГО в формате JSON без пояснений до или после."
                )
                raw = self.llm.chat(system_prompt, retry_prompt, json_mode=True)

        return {
            "state_summary": "",
            "problem_or_risk": "не удалось разобрать ответ LLM",
            "action": None,
            "expected_effect": "",
            "confidence": "низкая",
            "explanation": f"LLM вернул невалидный JSON после {max_retries} попыток: {raw[:300]!r}",
            "chosen_scenario_id": None,
        }

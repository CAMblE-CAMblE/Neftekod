"""
Демонстрация полного цикла оркестратора на 3х сценариях из ТЗ (раздел 6):
  1. штатный период,
  2. период с риском ухудшения качества,
  3. устаревшие данные + варианты, не проходящие жёсткие проверки.

Входные данные — JSON-файлы в sample_data/, имитирующие то, что реально
пришлют агенты качества/надёжности/оптимизации.

По умолчанию использует локальный Ollama (llama3.2). Если Ollama
не запущен, укажите --fake, чтобы прогнать логику на заглушке LLM и убедиться,
что жесткие проверки и склейка данных работают независимо от модели.

Запуск:
    python demo.py
    python demo.py --fake
    python demo.py --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestrator.json_io import load_agent_inputs, recommendation_to_json
from orchestrator.llm_client import FakeLLMClient, OllamaClient
from orchestrator.orchestrator import Orchestrator

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

FAKE_JSON_RESPONSE = json.dumps(
    {
        "state_summary": "Режим близок к границе по сере, температура АВТ повышена.",
        "problem_or_risk": "Прогнозная сера приближается к пределу 10 мг/кг.",
        "action": "Снизить AVT_T101 до 352.0",
        "expected_effect": "Сера снижается до ~8.1 мг/кг, риск оборудования падает незначительно.",
        "confidence": "средняя — прогноз качества основан на прокси-модели",
        "explanation": "Сценарий lower_temp — единственный допустимый после жёстких проверок и даёт запас по сере.",
        "chosen_scenario_id": "lower_temp",
    },
    ensure_ascii=False,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true",
                        help="использовать FakeLLMClient вместо Ollama")
    parser.add_argument("--model", default="llama3.2", help="имя модели в Ollama")
    args = parser.parse_args()

    llm = FakeLLMClient(
        canned_response=FAKE_JSON_RESPONSE) if args.fake else OllamaClient(
        model=args.model)
    orchestrator = Orchestrator(llm=llm)

    for path in sorted(SAMPLE_DATA_DIR.glob("*.json")):
        print(f"\n{'=' * 60}\n{path.name}\n{'=' * 60}")
        inputs = load_agent_inputs(path)
        try:
            rec = orchestrator.run_cycle(inputs)
        except RuntimeError as e:
            print(f"[Ошибка LLM] {e}")
            continue
        print(recommendation_to_json(rec))


if __name__ == "__main__":
    main()

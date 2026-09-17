"""CLI-запуск инференса агента качества."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from quality_agent.data import read_table
from quality_agent.inference import load_bundle, predict_frame


def main() -> None:
    """Разбирает аргументы и выполняет пакетный прогноз.

    Вход: каталог модели, входной подготовленный файл и путь результата.
    Выход: CSV/Parquet с ответом агента. Существенное условие: вход должен
    соответствовать канонической схеме, описанной в документации.
    """

    parser = argparse.ArgumentParser(description="Инференс агента качества 24-2000")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help="Подготовленный parquet/csv со state_time и сигналами")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle = load_bundle(args.model_dir)
    frame = read_table(args.input)
    result = predict_frame(frame, bundle)
    if args.output.lower().endswith(".parquet"):
        result.to_parquet(args.output, index=False)
    else:
        result.to_csv(args.output, index=False)
    logging.info("Прогноз сохранен: %s", args.output)


if __name__ == "__main__":
    main()

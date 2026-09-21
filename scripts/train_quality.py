"""CLI-запуск обучения агента качества."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from quality_agent.config import load_config
from quality_agent.data import build_base_frame, extract_output_lims, write_table
from quality_agent.model import train_from_sources, train_model
from quality_agent.synthetic import make_synthetic_sources


def main() -> None:
    """Разбирает аргументы и запускает обучение.

    Вход: `--config`, `--run-id` и `--synthetic`. Выход: каталог артефактов.
    Существенное условие: по умолчанию производственное обучение не запускается
    без явно указанных путей в конфигурации.
    """

    parser = argparse.ArgumentParser(description="Обучение агента качества 24-2000")
    parser.add_argument("--config", default="configs/quality_agent.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Запустить малый синтетический пример")
    parser.add_argument("--save-prepared", default=None, help="Сохранить подготовленный датасет в parquet/csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.synthetic:
        config.training.iterations = min(config.training.iterations, 30)
        telemetry, pak, lims = make_synthetic_sources()
        frame = build_base_frame(telemetry, pak, lims, config)
        output_lims = extract_output_lims(lims, config)
        if args.save_prepared:
            write_table(frame, args.save_prepared)
        run_dir = train_model(frame, output_lims, config, run_id=args.run_id or "synthetic")
    else:
        if not config.data.prepared_path and not (config.data.telemetry_path and config.data.lims_path):
            raise SystemExit("Для реального обучения задайте data.prepared_path или data.telemetry_path + data.lims_path.")
        if args.save_prepared and not config.data.prepared_path:
            from quality_agent.data import load_sources

            telemetry, avt, pak, lims = load_sources(config)
            frame = build_base_frame(telemetry, pak, lims, config, avt_df=avt)
            write_table(frame, args.save_prepared)
            if config.data.output_lims_path:
                write_table(extract_output_lims(lims, config), config.data.output_lims_path)
        run_dir = train_from_sources(config, run_id=args.run_id)
    logging.info("Артефакты сохранены: %s", run_dir)


if __name__ == "__main__":
    main()

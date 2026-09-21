"""Командная строка автономного агента качества."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_MODEL_DIR, load_yaml, write_json
from .data import build_state_table_from_sources, read_table, select_base_state, write_table
from .inference import evaluate_candidates, load_bundle, predict_base_state


def main() -> None:
    """Точка входа CLI."""

    parser = argparse.ArgumentParser(description="Автономный инференс агента качества")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Каталог с model.cbm и JSON-артефактами")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-base-state", help="Собрать base_state.csv")
    build.add_argument("--paths-config", required=True, help="YAML с путями к исходным файлам")
    build.add_argument("--state-time", required=True, help="Искомое время состояния, например 2025-07-16 06:30:00")
    build.add_argument("--allow-previous", action="store_true", help="Разрешить выбор ближайшего более раннего времени")
    build.add_argument("--output", required=True, help="Путь результата base_state.csv")
    build.add_argument("--report", required=True, help="Путь отчета base_state_report.json")

    predict = subparsers.add_parser("predict", help="Прогноз для готового base_state.csv")
    predict.add_argument("--base-state", required=True, help="Путь к base_state.csv")
    predict.add_argument("--output", required=True, help="Путь результата prediction.csv")

    candidates = subparsers.add_parser("evaluate-candidates", help="Оценить candidates.csv относительно base_state.csv")
    candidates.add_argument("--base-state", required=True, help="Путь к base_state.csv")
    candidates.add_argument("--candidates", required=True, help="Путь к candidates.csv")
    candidates.add_argument("--output", required=True, help="Путь результата candidate_predictions.csv")

    args = parser.parse_args()
    bundle = load_bundle(args.model_dir)
    if args.command == "build-base-state":
        paths_config = load_yaml(args.paths_config)
        paths_config["_config_dir"] = str(Path(args.paths_config).resolve().parent)
        source_cfg = paths_config.get("sources", paths_config)
        prepared_path = source_cfg.get("prepared_path")
        if prepared_path:
            from .config import resolve_path

            table_path = resolve_path(prepared_path, base_dir=paths_config["_config_dir"])
            table = read_table(table_path)
            source_mode = "prepared"
        else:
            table = build_state_table_from_sources(paths_config, bundle.model_config)
            source_mode = "raw_sources"
        state, report = select_base_state(
            table,
            state_time=args.state_time,
            allow_previous=args.allow_previous,
            model_config=bundle.model_config,
            feature_names=bundle.feature_names,
        )
        report["source_mode"] = source_mode
        write_table(state, args.output)
        write_json(report, args.report)
        print(f"base_state сохранен: {args.output}")
        print(f"отчет сохранен: {args.report}")
    elif args.command == "predict":
        base_state = read_table(args.base_state)
        prediction = predict_base_state(bundle, base_state)
        write_table(prediction, args.output)
        print(f"прогноз сохранен: {args.output}")
    elif args.command == "evaluate-candidates":
        base_state = read_table(args.base_state)
        candidate_frame = pd.read_csv(args.candidates)
        result = evaluate_candidates(bundle, base_state, candidate_frame)
        write_table(result, args.output)
        print(f"оценка кандидатов сохранена: {args.output}")


if __name__ == "__main__":
    main()

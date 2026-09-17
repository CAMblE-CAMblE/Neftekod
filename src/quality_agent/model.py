"""Обучение, сохранение и загрузка моделей агента качества."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - проверяется сообщением при обучении.
    CatBoostRegressor = None

from .config import QualityAgentConfig, config_to_dict
from .data import dataset_checksum, extract_output_lims
from .evaluation import evaluate_against_lims, regression_metrics
from .features import prepare_features


@dataclass
class MedianBaseline:
    """Медианный базовый прогноз.

    Вход: целевая переменная обучающей выборки. Выход: константный прогноз.
    Существенное условие: baseline нужен только для сравнения с CatBoost и не
    заменяет основную модель.
    """

    value: float

    def predict(self, X: pd.DataFrame) -> list[float]:
        """Возвращает константный прогноз.

        Вход: матрица признаков любой ширины. Выход: список длины X.
        """

        return [self.value] * len(X)


def chronological_split(frame: pd.DataFrame, config: QualityAgentConfig) -> dict[str, pd.DataFrame]:
    """Делит данные на train, validation и test по времени.

    Вход: подготовленный датафрейм с целевой переменной. Выход: словарь трех
    выборок. Существенное условие: в режиме forecast обучающие строки с целями
    за пределами своего периода исключаются на границах.
    """

    data = frame.dropna(subset=["target_pak_sulfur_mg_kg"]).sort_values("state_time").reset_index(drop=True)
    if data.empty:
        raise ValueError("Нет строк с целевой переменной target_pak_sulfur_mg_kg")
    n = len(data)
    train_end = max(1, int(n * config.training.train_fraction))
    val_end = max(train_end + 1, int(n * (config.training.train_fraction + config.training.validation_fraction)))
    val_end = min(val_end, n)
    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    if config.training.mode == "forecast":
        train_limit = train["state_time"].max()
        val_limit = validation["state_time"].max() if not validation.empty else train_limit
        train = train[train["prediction_time"] <= train_limit]
        validation = validation[validation["prediction_time"] <= val_limit]
    return {"train": train, "validation": validation, "test": test}


def _git_revision() -> str | None:
    """Возвращает текущий Git SHA при наличии репозитория.

    Вход: нет. Выход: строка SHA или None. Существенное условие: отсутствие Git
    не прерывает обучение.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _dependency_versions() -> dict[str, str]:
    """Собирает версии ключевых зависимостей.

    Вход: нет. Выход: словарь версий Python, pandas и CatBoost.
    """

    try:
        import catboost
        catboost_version = catboost.__version__
    except ImportError:
        catboost_version = "not_installed"

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "catboost": catboost_version,
    }


def train_model(
    frame: pd.DataFrame,
    output_lims: pd.DataFrame | None,
    config: QualityAgentConfig,
    *,
    run_id: str | None = None,
    input_paths: list[str | None] | None = None,
) -> Path:
    """Обучает CatBoost и сохраняет комплект артефактов.

    Вход: подготовленный датафрейм, отдельные события выходного ЛИМС и конфиг.
    Выход: путь к каталогу запуска. Существенное условие: все обучаемые
    преобразования и порядок признаков фиксируются по train.
    """

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost не установлен. Установите зависимости из requirements.txt.")
    run_dir = Path(config.artifacts.root_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(frame, config)
    X_train, feature_names, train_diag = prepare_features(splits["train"], config)
    y_train = splits["train"]["target_pak_sulfur_mg_kg"]
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=config.training.iterations,
        learning_rate=config.training.learning_rate,
        depth=config.training.depth,
        random_seed=config.training.random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    eval_set = None
    if not splits["validation"].empty:
        X_val, _, _ = prepare_features(splits["validation"], config, feature_names=feature_names)
        eval_set = (X_val, splits["validation"]["target_pak_sulfur_mg_kg"])
    model.fit(X_train, y_train, eval_set=eval_set)
    baseline = MedianBaseline(float(y_train.median()))

    metrics: dict[str, object] = {"catboost": {}, "baseline": {}, "diagnostics": train_diag}
    predictions = []
    for name, split in splits.items():
        if split.empty:
            metrics["catboost"][name] = regression_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
            metrics["baseline"][name] = regression_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
            continue
        X, _, _ = prepare_features(split, config, feature_names=feature_names)
        pred = pd.Series(model.predict(X), index=split.index)
        base_pred = pd.Series(baseline.predict(X), index=split.index)
        y = split["target_pak_sulfur_mg_kg"]
        metrics["catboost"][name] = regression_metrics(y, pred)
        metrics["baseline"][name] = regression_metrics(y, base_pred)
        block = split[["state_time", "prediction_time", "target_pak_sulfur_mg_kg"]].copy()
        block["split"] = name
        block["prediction_mg_kg"] = pred.values
        block["baseline_prediction_mg_kg"] = base_pred
        predictions.append(block)

    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if output_lims is not None and not output_lims.empty and not predictions_df.empty:
        test_predictions = predictions_df[predictions_df["split"] == "test"]
        metrics["output_lims_test"] = evaluate_against_lims(
            test_predictions,
            output_lims,
            tolerance_minutes=config.training.output_lims_match_tolerance_minutes,
        )
    else:
        metrics["output_lims_test"] = {"mae": None, "rmse": None, "mean_signed_error": None, "n": 0}

    model.save_model(run_dir / "model.cbm")
    (run_dir / "baseline.json").write_text(json.dumps({"median": baseline.value}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config_to_dict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "feature_names.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    predictions_df.to_csv(run_dir / "validation_predictions.csv", index=False)
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    periods = {
        name: {
            "state_time_min": None if split.empty else str(split["state_time"].min()),
            "state_time_max": None if split.empty else str(split["state_time"].max()),
            "n": int(len(split)),
        }
        for name, split in splits.items()
    }
    metadata = {
        "run_id": run_id,
        "seed": config.training.random_seed,
        "periods": periods,
        "dependency_versions": _dependency_versions(),
        "git_revision": _git_revision(),
        "dataset_checksum": dataset_checksum(input_paths or []),
        "feature_units": config.features.units,
        "feature_preparation": {
            "lims_delay_hours": config.lims.assumed_publication_delay_hours,
            "max_input_age_hours": config.lims.max_input_age_hours,
            "max_input_age_is_experimental": config.lims.max_input_age_is_experimental,
            "calculated_feature_sources": config.features.calculated_feature_sources,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return run_dir


def load_model_bundle(run_dir: str | Path) -> tuple[CatBoostRegressor, QualityAgentConfig, list[str], dict[str, object]]:
    """Загружает сохраненный комплект модели.

    Вход: путь к каталогу запуска. Выход: модель, конфигурация, порядок
    признаков и метаданные. Существенное условие: инференс использует именно
    сохраненный порядок признаков.
    """

    from .config import _merge_dataclass

    run_dir = Path(run_dir)
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost не установлен. Установите зависимости из requirements.txt.")
    model = CatBoostRegressor()
    model.load_model(run_dir / "model.cbm")
    config_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = _merge_dataclass(QualityAgentConfig(), config_data)
    feature_names = json.loads((run_dir / "feature_names.json").read_text(encoding="utf-8"))
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return model, config, feature_names, metadata


def train_from_sources(config: QualityAgentConfig, *, run_id: str | None = None) -> Path:
    """Запускает обучение от путей в конфигурации.

    Вход: конфиг с путями. Выход: путь к артефактам. Существенное условие:
    если задан `prepared_path`, используется уже подготовленный датасет.
    """

    from .data import build_base_frame, load_sources, read_table

    if config.data.prepared_path:
        frame = read_table(config.data.prepared_path)
        output_lims = pd.DataFrame()
    else:
        telemetry, pak, lims = load_sources(config)
        frame = build_base_frame(telemetry, pak, lims, config)
        output_lims = extract_output_lims(lims, config)
    return train_model(
        frame,
        output_lims,
        config,
        run_id=run_id,
        input_paths=[config.data.telemetry_path, config.data.pak_path, config.data.lims_path, config.data.prepared_path],
    )

"""Конфигурация агента качества и загрузка настроек из YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Пути и соответствие колонок.

    Вход: пути к исходным или подготовленным таблицам и имена колонок в них.
    Выход: объект с каноническими настройками загрузки. Существенное условие:
    автоматическое переименование тегов по буквам не выполняется, все признаки
    должны быть перечислены явно.
    """

    telemetry_path: str = "data/242000_tags.csv"
    pak_path: str | None = None
    lims_path: str | None = None
    prepared_path: str | None = None
    timestamp_col: str = "date"
    pak_target_tag: str = "24-2000:Mg.Sulfur"
    input_lims_sampling_point: str = "Гидроочистка:1"
    input_lims_parameter: str = "Mass.Sulfur"
    output_lims_sampling_point: str = "Гидроочистка:2"
    output_lims_parameter: str = "Mg.Sulfur"


@dataclass
class LimsConfig:
    """Правила доступности лабораторных анализов.

    Вход: задержка публикации и предельный возраст анализа. Выход: параметры
    для обратного временного сопоставления. Существенное условие: если
    фактическое время публикации неизвестно, используется допущение
    `sample_time + assumed_publication_delay_hours`.
    """

    assumed_publication_delay_hours: float = 4.0
    max_input_age_hours: float = 240.0
    max_input_age_is_experimental: bool = True


@dataclass
class FeatureConfig:
    """Список признаков и правила исключений.

    Вход: явный список признаков и происхождение расчетных показателей. Выход:
    проверенный порядок колонок для обучения и инференса. Существенное условие:
    W7, T6, P13 и расчетные ВАК, зависящие от них, исключаются до обучения.
    """

    raw_features: list[str] = field(
        default_factory=lambda: [
            "F1",
            "F2",
            "P3",
            "W4",
            "T5",
            "P8",
            "F9",
            "W10",
            "T11",
            "T12",
            "F14",
            "F15",
            "T16",
            "F17",
            "T18",
            "F19",
            "Q20",
            "Q21",
            "F22",
            "T23",
            "P24",
            "F25",
            "F26",
        ]
    )
    lab_features: list[str] = field(
        default_factory=lambda: ["input_sulfur_mg_kg", "input_sulfur_age_hours"]
    )
    calculated_features: list[str] = field(default_factory=lambda: ["I250", "D15"])
    forbidden_base_features: list[str] = field(default_factory=lambda: ["W7", "T6", "P13"])
    calculated_feature_sources: dict[str, list[str]] = field(
        default_factory=lambda: {
            "T90": ["F1", "F15", "F26", "T12", "T23", "W7"],
            "T50": ["F9", "P13", "T6"],
            "I250": ["F14", "F25", "T5", "T11", "T16", "T23"],
            "IBP": ["F14", "F22", "F26", "P13", "P24", "T16", "T23", "W4"],
            "CloudPoint": ["F1", "F9", "F22", "F25", "T6", "T16", "W7"],
            "CFPP": ["F9", "P8", "P24", "T23", "W7"],
            "T95": ["F2", "F9", "T6", "LIMS:Гидроочистка:1:95%.T"],
            "D15": ["F22", "T11", "LIMS:Гидроочистка:1:D15"],
        }
    )
    units: dict[str, str] = field(
        default_factory=lambda: {
            "input_sulfur_mg_kg": "мг/кг",
            "input_sulfur_age_hours": "ч",
            "I250": "% об.",
            "D15": "кг/м3",
        }
    )


@dataclass
class TrainingConfig:
    """Параметры обучения и временной проверки.

    Вход: режим задачи, горизонт, доли хронологического разбиения и seed.
    Выход: настройки для построения целевой переменной и CatBoost. Существенное
    условие: test используется только для итоговой оценки.
    """

    mode: str = "current"
    forecast_horizon_hours: float = 4.0
    target_match_tolerance_minutes: float = 20.0
    output_lims_match_tolerance_minutes: float = 60.0
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    random_seed: int = 42
    iterations: int = 200
    learning_rate: float = 0.05
    depth: int = 6


@dataclass
class InferenceConfig:
    """Настройки ответа агента качества.

    Вход: предел серы и флаги сценарной оценки. Выход: параметры сравнения
    прогноза с пределом. Существенное условие: предел не встроен в регрессию,
    он применяется отдельной проверкой сценария.
    """

    sulfur_limit_mg_kg: float = 10.0
    limit_applies_to_hdt_output: bool = True
    allow_scenario_assessment: bool = False


@dataclass
class ArtifactConfig:
    """Настройки сохранения результатов запуска.

    Вход: корневой каталог артефактов. Выход: путь конкретного run_id.
    Существенное условие: каждый запуск сохраняется в отдельную папку.
    """

    root_dir: str = "artifacts/quality_agent"


@dataclass
class QualityAgentConfig:
    """Полная конфигурация агента качества.

    Вход: секции data, lims, features, training, inference и artifacts.
    Выход: единый объект настроек для подготовки данных, обучения и инференса.
    """

    data: DataConfig = field(default_factory=DataConfig)
    lims: LimsConfig = field(default_factory=LimsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    """Обновляет dataclass словарем.

    Вход: экземпляр dataclass и словарь из YAML. Выход: новый экземпляр того же
    типа. Существенное условие: неизвестные ключи считаются ошибкой, чтобы
    опечатки в конфигурации не проходили молча.
    """

    known = {item.name for item in fields(instance)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Неизвестные ключи конфигурации {type(instance).__name__}: {sorted(unknown)}")
    kwargs = {}
    for item in fields(instance):
        current = getattr(instance, item.name)
        value = values.get(item.name, current)
        if is_dataclass(current) and isinstance(value, dict):
            value = _merge_dataclass(current, value)
        kwargs[item.name] = value
    return type(instance)(**kwargs)


def load_config(path: str | Path | None = None) -> QualityAgentConfig:
    """Загружает конфигурацию агента.

    Вход: путь к YAML или None. Выход: `QualityAgentConfig` с дефолтами и
    переопределениями из файла. Существенное условие: при None возвращаются
    встроенные безопасные значения.
    """

    config = QualityAgentConfig()
    if path is None:
        return config
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _merge_dataclass(config, data)


def config_to_dict(config: QualityAgentConfig) -> dict[str, Any]:
    """Преобразует конфигурацию в обычный словарь.

    Вход: объект конфигурации. Выход: JSON/YAML-совместимый словарь для
    сохранения в артефакты запуска.
    """

    return asdict(config)

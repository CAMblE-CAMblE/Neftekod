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
    avt_telemetry_path: str | None = None
    pak_path: str | None = None
    lims_path: str | None = None
    prepared_path: str | None = None
    output_lims_path: str | None = "data/processed/quality_output_lims.parquet"
    timestamp_col: str = "date"
    timestamp_format: str | None = "%Y-%m-%d %H:%M:%S"
    pak_target_tag: str = "24-2000:Mg.Sulfur"
    target_col: str = "target_sulfur_mg_kg"
    target_source_col: str = "target_source"
    input_lims_sampling_point: str = "Гидроочистка:1"
    input_lims_parameter: str = "Mass.Sulfur"
    output_lims_sampling_point: str = "Гидроочистка:2"
    output_lims_parameter: str = "Mg.Sulfur"
    avt_asof_tolerance_minutes: float = 10.0


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
    целевые и сравнительные колонки исключаются даже при ошибке в YAML.
    """

    raw_features: list[str] = field(
        default_factory=lambda: [
            "hdt_F1",
            "hdt_F2",
            "hdt_P3",
            "hdt_W4",
            "hdt_T5",
            "hdt_T6",
            "hdt_W7",
            "hdt_P8",
            "hdt_F9",
            "hdt_W10",
            "hdt_T11",
            "hdt_T12",
            "hdt_P13",
            "hdt_F14",
            "hdt_F15",
            "hdt_T16",
            "hdt_F17",
            "hdt_T18",
            "hdt_F19",
            "hdt_F22",
            "hdt_T23",
            "hdt_P24",
            "hdt_F25",
            "hdt_F26",
            "avt_T1",
            "avt_P2",
            "avt_F3",
            "avt_P4",
            "avt_F5",
            "avt_T6",
            "avt_F7",
            "avt_F8",
            "avt_F9",
            "avt_D10",
            "avt_T11",
            "avt_F12",
            "avt_T13",
            "avt_F14",
            "avt_T15",
            "avt_F16",
            "avt_T17",
            "avt_T18",
            "avt_F19",
            "avt_T20",
            "avt_P21",
            "avt_P22",
            "avt_P23",
            "avt_T24",
            "avt_F25",
            "avt_F26",
            "avt_F27",
            "avt_F28",
            "avt_F29",
            "avt_F30",
            "avt_F31",
            "avt_F32",
            "avt_T33",
            "avt_F34",
            "avt_F35",
            "avt_F36",
            "avt_T37",
            "avt_T38",
            "avt_T39",
            "avt_T40",
            "avt_F41",
            "avt_T42",
            "avt_L43",
            "avt_P44",
            "avt_F45",
            "avt_F46",
            "avt_T47",
            "avt_T48",
            "avt_T49",
            "avt_P50",
            "avt_P51",
            "avt_P52",
            "avt_F53",
            "avt_F54",
            "avt_T55",
            "avt_F56",
            "avt_F57",
            "avt_T58",
            "avt_F59",
            "avt_F60",
            "avt_T61",
            "avt_F62",
            "avt_F63",
            "avt_F64",
            "avt_F65",
            "avt_T66",
            "avt_P67",
            "avt_F68",
            "avt_F69",
            "avt_W70",
            "avt_T71",
        ]
    )
    lab_features: list[str] = field(
        default_factory=lambda: ["input_sulfur_mg_kg", "input_sulfur_age_hours"]
    )
    calculated_features: list[str] = field(
        default_factory=lambda: [
            "hdt_T90",
            "hdt_T50",
            "hdt_I250",
            "hdt_IBP",
            "hdt_CloudPoint",
            "hdt_CFPP",
            "hdt_T95",
            "hdt_D15",
            "AVT6:240-350:D15",
            "AVT6:240-350:T50",
            "AVT6:240-350:EBP",
            "AVT6:240-350:CFPP",
            "AVT6:350:T50",
            "AVT6:350:I350",
            "AVT6:350:D15",
            "AVT6:350:CFPP",
            "AVT6:350-500:ViscosityK",
        ]
    )
    forbidden_base_features: list[str] = field(default_factory=lambda: ["hdt_Q20", "hdt_Q21"])
    leakage_features: list[str] = field(
        default_factory=lambda: [
            "target_sulfur_mg_kg",
            "target_sulfur_raw_mg_kg",
            "target_sulfur_rejected",
            "target_sulfur_rejection_reason",
            "target_pak_sulfur_mg_kg",
            "pak_sulfur_mg_kg",
            "output_lims_sulfur_mg_kg",
            "hdt_Q20",
            "hdt_Q21",
            "Mg.Sulfur",
            "24-2000:Mg.Sulfur",
        ]
    )
    calculated_feature_sources: dict[str, list[str]] = field(
        default_factory=lambda: {
            "hdt_T90": ["hdt_F1", "hdt_F15", "hdt_F26", "hdt_T12", "hdt_T23", "hdt_W7"],
            "hdt_T50": ["hdt_F9", "hdt_P13", "hdt_T6"],
            "hdt_I250": ["hdt_F14", "hdt_F25", "hdt_T5", "hdt_T11", "hdt_T16", "hdt_T23"],
            "hdt_IBP": ["hdt_F14", "hdt_F22", "hdt_F26", "hdt_P13", "hdt_P24", "hdt_T16", "hdt_T23", "hdt_W4"],
            "hdt_CloudPoint": ["hdt_F1", "hdt_F9", "hdt_F22", "hdt_F25", "hdt_T6", "hdt_T16", "hdt_W7"],
            "hdt_CFPP": ["hdt_F9", "hdt_P8", "hdt_P24", "hdt_T23", "hdt_W7"],
            "hdt_T95": ["hdt_F2", "hdt_F9", "hdt_T6", "LIMS:Гидроочистка:1:95%.T"],
            "hdt_D15": ["hdt_F22", "hdt_T11", "LIMS:Гидроочистка:1:D15"],
            "AVT6:240-350:D15": ["avt_F65", "avt_F32", "avt_F30", "avt_T66", "avt_T33"],
            "AVT6:240-350:T50": ["avt_F7", "avt_F30", "avt_F34", "avt_F45", "avt_F59", "avt_F63"],
            "AVT6:240-350:EBP": ["avt_F30", "avt_T33", "avt_F36", "avt_T37", "avt_T40", "avt_T58"],
            "AVT6:240-350:CFPP": ["avt_T33", "avt_P67", "avt_P4", "avt_F65", "avt_F32", "avt_F30"],
            "AVT6:350:T50": ["avt_T42", "avt_T48", "avt_F31", "avt_F57", "avt_T66", "avt_T33"],
            "AVT6:350:I350": ["avt_L43", "avt_T6", "avt_T18", "avt_F64", "avt_T15", "avt_T11"],
            "AVT6:350:D15": ["avt_T42", "avt_T48", "avt_F31", "avt_F57"],
            "AVT6:350:CFPP": ["avt_T48", "avt_T40", "avt_F31", "avt_F57"],
            "AVT6:350-500:ViscosityK": ["avt_T6", "avt_T13", "avt_T18", "avt_T20", "avt_L43", "avt_T48", "avt_P50", "avt_F53", "avt_P51", "avt_F59", "avt_T61"],
        }
    )
    units: dict[str, str] = field(
        default_factory=lambda: {
            "input_sulfur_mg_kg": "мг/кг",
            "input_sulfur_age_hours": "ч",
            "I250": "% об.",
            "T95": "°С",
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
    forecast_horizon_hours: float = 3.0
    max_forecast_horizon_hours: float = 3.0
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

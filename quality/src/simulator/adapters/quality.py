"""Точка замены демонстрационной модели на реальный агент качества."""

from __future__ import annotations

from simulator.models.demo_quality import DemoQualityModel


def build_demo_quality_model(config: dict[str, float]) -> DemoQualityModel:
    """Создает демонстрационную модель качества из словаря конфигурации."""

    return DemoQualityModel(
        sulfur_in_coefficient=float(config["sulfur_in_coefficient"]),
        t6_coefficient=float(config["t6_coefficient"]),
        f9_coefficient=float(config["f9_coefficient"]),
        p13_coefficient=float(config["p13_coefficient"]),
        min_q21=float(config.get("min_q21", 0.1)),
    )

"""Каркас агента качества для прогноза серы на выходе гидроочистки 24-2000."""

from .config import QualityAgentConfig, load_config
from .inference import load_bundle, predict_frame
from .model import train_model

__all__ = ["QualityAgentConfig", "load_config", "load_bundle", "predict_frame", "train_model"]

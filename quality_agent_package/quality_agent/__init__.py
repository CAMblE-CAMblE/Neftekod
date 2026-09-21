"""Автономный пакет инференса агента качества."""

from .inference import (
    QualityModelBundle,
    evaluate_candidates,
    load_bundle,
    predict_base_state,
    predict_frame,
)

__all__ = [
    "QualityModelBundle",
    "evaluate_candidates",
    "load_bundle",
    "predict_base_state",
    "predict_frame",
]

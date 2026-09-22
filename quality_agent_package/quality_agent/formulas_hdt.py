"""Формулы ВАК гидроочистки 24-2000 для инференса."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HdtTags:
    """Один срез исходных тегов гидроочистки, нужных для формул ВАК."""

    F1: float
    F2: float
    F9: float
    F14: float
    F15: float
    F22: float
    F25: float
    F26: float
    P8: float
    P13: float
    P24: float
    T5: float
    T6: float
    T11: float
    T12: float
    T16: float
    T23: float
    W4: float
    W7: float


def compute_all(tags: HdtTags, lims_95pct_t_pipeline: float | None, lims_d15_pipeline: float | None) -> dict[str, float]:
    """Считает все ВАК гидроочистки, доступные для модели."""

    result = {
        "hdt_T90": float("nan") if tags.F26 == 0 else (
            162.998
            + 0.12945 * tags.T12
            + 59.57 * (tags.F15 / 2000)
            + 0.00036 * tags.W7
            + 0.26366 * tags.T23
            - 424.72638 * tags.F1 / tags.F26
        ),
        "hdt_T50": 44.625 + 10.0224 * tags.P13 + 0.06981 * tags.F9 + 0.471 * tags.T6,
        "hdt_I250": (
            84.585
            - 0.21172 * tags.T5
            + 0.12137 * tags.T11
            - 0.00014 * tags.F25
            + 0.56248 * tags.F14
            - 0.16317 * tags.T23
            + 0.20272 * tags.T16
        ),
        "hdt_IBP": (
            137.762
            - 0.0653 * tags.F26
            + 0.00011 * tags.F22
            + 5.78137 * tags.P13
            - 34.58028 * tags.P24
            - 0.00993 * tags.F14
            - 0.99962 * tags.W4
            + 0.32232 * tags.T23
            - 0.09406 * tags.T16
        ),
        "hdt_CloudPoint": (
            0.0002 * tags.F22
            + 0.0021 * tags.W7
            + 0.00008 * tags.F25
            - 0.30656 * tags.F1
            + 0.12018 * tags.T6
            + 0.01916 * tags.F9
            - 48.254
            - 0.05249 * tags.T16
            + 0.00011
        ),
        "hdt_CFPP": (
            0.22088 * tags.T23
            - 102.375
            - 47.75834 * tags.P8
            + 0.03862 * tags.F9
            + 43.60207 * tags.W7
            + 43.81849 * tags.P24
        ),
    }
    if lims_95pct_t_pipeline is not None:
        result["hdt_T95"] = (
            0.03814 * tags.F9
            - 9.201
            - 0.00002 * tags.F2
            + 0.50 * tags.T6
            + 0.48321 * lims_95pct_t_pipeline
        )
    if lims_d15_pipeline is not None:
        result["hdt_D15"] = (
            667.881
            + 0.15417 * lims_d15_pipeline
            + 0.00005 * tags.F22
            + 0.10774 * tags.T11
        )
    return result

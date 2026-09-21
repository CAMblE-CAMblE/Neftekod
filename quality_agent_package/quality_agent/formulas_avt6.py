"""Формулы ВАК АВТ-6 для инференса."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Avt6Tags:
    """Один срез исходных тегов АВТ-6, нужных для формул ВАК."""

    P4: float
    F7: float
    T6: float
    T11: float
    T13: float
    T15: float
    T18: float
    T20: float
    F30: float
    F31: float
    F32: float
    T33: float
    F34: float
    F36: float
    T37: float
    T40: float
    T42: float
    L43: float
    F45: float
    T48: float
    P50: float
    P51: float
    F53: float
    F57: float
    T58: float
    F59: float
    T61: float
    F63: float
    F64: float
    F65: float
    T66: float
    P67: float


def compute_all(tags: Avt6Tags) -> dict[str, float]:
    """Считает все ВАК АВТ-6, доступные для модели."""

    f32_f30 = tags.F32 + tags.F30
    return {
        "AVT6:240-350:D15": float("nan") if f32_f30 == 0 else (
            791.22872 - 5.30294 * (tags.F65 / f32_f30) + 0.52755 * tags.T66 - 0.15629 * tags.T33
        ),
        "AVT6:240-350:T50": (
            283.177
            - 0.01685 * tags.F7
            + 0.06248 * tags.F30
            + 0.22048 * tags.F34
            - 0.25816 * tags.F45
            - 0.12159 * tags.F59
            + 0.01221 * tags.F63
        ),
        "AVT6:240-350:EBP": (
            813.883
            + 2.66463 * tags.F30
            - 0.20239 * tags.T33
            - 3.65888 * tags.F36
            - 14.08235 * tags.T37
            - 1.32603 * tags.T40
            + 14.60206 * tags.T58
        ),
        "AVT6:240-350:CFPP": float("nan") if f32_f30 == 0 else (
            31.40363
            - 0.06784 * tags.T33
            + 17.411 * tags.P67
            - 8.11544 * tags.P4
            - 0.47309 * (tags.F65 / f32_f30)
        ),
        "AVT6:350:T50": (
            493.6798
            + 1.281193 * tags.T42
            - 0.955342 * tags.T48
            - 0.018454 * tags.F31
            + 0.265904 * tags.F57
            - 0.082047 * tags.T66
            - 0.545083 * tags.T33
        ),
        "AVT6:350:I350": (
            39.562
            - 1.62865 * tags.L43
            + 0.76664 * tags.T6
            - 0.22361 * tags.T18
            + 0.00031 * tags.F64 * (tags.T15 - tags.T11)
        ),
        "AVT6:350:D15": float("nan") if tags.F57 == 0 else (
            983.092 + 0.27467 * tags.T42 - 0.49014 * tags.T48 - 0.32983 * tags.F31 / tags.F57
        ),
        "AVT6:350:CFPP": float("nan") if tags.F57 == 0 else (
            19.27111 - 0.10582 * tags.T48 + 0.13836 * tags.T40 - 0.42304 * (tags.F31 / tags.F57)
        ),
        "AVT6:350-500:ViscosityK": (
            5.831
            + 0.00976 * tags.T6
            + 0.01188 * tags.T13
            + 0.00224 * tags.T18
            + 0.01905 * tags.T20
            + 0.00794 * tags.L43
            - 0.02496 * tags.T48
            - 0.00008 * tags.P50
            - 0.00882 * tags.F53
            - 0.00394 * tags.P51
            - 0.00255 * tags.F59
            + 0.01153 * tags.T61
        ),
    }

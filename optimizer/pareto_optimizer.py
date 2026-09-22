from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Пути
ROOT = Path(__file__).resolve().parent.parent

QUALITY_RESULTS_PATH = ROOT / "optimizer" / "logs" / "quality_results.csv"
ENERGY_RESULTS_PATH = ROOT / "optimizer" / "logs" / "energy_results.csv"

OUTPUT_DIR = ROOT / "optimizer" / "logs"

PARETO_RESULTS_PATH = OUTPUT_DIR / "pareto_results.csv"
SELECTED_SCENARIO_PATH = OUTPUT_DIR / "selected_scenario.csv"
PARETO_PLOT_PATH = OUTPUT_DIR / "pareto_front.png"


# Логика метода Парето
def is_dominated(row, candidates):
    """
    Проверяет, существует ли кандидат, который:
    - не хуже по обоим критериям;
    - строго лучше хотя бы по одному.

    Оба критерия минимизируются:
    - predicted_sulfur_mg_kg
    - energy_proxy_eu_h
    """

    sulfur = row["predicted_sulfur_mg_kg"]
    energy = row["energy_proxy_eu_h"]

    for _, other in candidates.iterrows():
        other_sulfur = other["predicted_sulfur_mg_kg"]
        other_energy = other["energy_proxy_eu_h"]

        no_worse = (
            other_sulfur <= sulfur
            and other_energy <= energy
        )

        strictly_better = (
            other_sulfur < sulfur
            or other_energy < energy
        )

        if no_worse and strictly_better:
            return True

    return False


def calculate_pareto_front(df):
    """
    Добавляет колонку pareto_optimal.

    Pareto front рассчитывается только среди
    quality-feasible кандидатов.
    """

    df = df.copy()

    df["pareto_optimal"] = False

    feasible_mask = df["quality_feasible"] == True
    feasible = df.loc[feasible_mask].copy()

    for idx, row in feasible.iterrows():
        candidates = feasible.drop(index=idx)

        if not is_dominated(row, candidates):
            df.loc[idx, "pareto_optimal"] = True

    return df


# Выбор сценария
def select_scenario(df):
    """
    Выбирает один сценарий с Pareto-фронта.

    Используется расстояние до идеальной точки:
    - минимальная сера;
    - минимальная энергия.

    Оба критерия нормализуются в диапазон [0, 1].
    """

    pareto = df[df["pareto_optimal"]].copy()

    if pareto.empty:
        raise RuntimeError(
            "Pareto front is empty. No feasible scenario available."
        )

    # Если только один кандидат
    if len(pareto) == 1:
        selected = pareto.iloc[[0]].copy()
        selected["distance_to_ideal"] = 0.0
        return selected

    sulfur_min = pareto["predicted_sulfur_mg_kg"].min()
    sulfur_max = pareto["predicted_sulfur_mg_kg"].max()

    energy_min = pareto["energy_proxy_eu_h"].min()
    energy_max = pareto["energy_proxy_eu_h"].max()

    # Нормализация серы
    if sulfur_max == sulfur_min:
        pareto["sulfur_normalized"] = 0.0
    else:
        pareto["sulfur_normalized"] = (
            pareto["predicted_sulfur_mg_kg"] - sulfur_min
        ) / (sulfur_max - sulfur_min)

    # Нормализация энергии
    if energy_max == energy_min:
        pareto["energy_normalized"] = 0.0
    else:
        pareto["energy_normalized"] = (
            pareto["energy_proxy_eu_h"] - energy_min
        ) / (energy_max - energy_min)

    # Расстояние до идеальной точки (0, 0)
    pareto["distance_to_ideal"] = (
        pareto["sulfur_normalized"] ** 2
        + pareto["energy_normalized"] ** 2
    ) ** 0.5

    selected = pareto.loc[
        [pareto["distance_to_ideal"].idxmin()]
    ].copy()

    return selected


# Визуализация*
def plot_pareto_front(df):
    """
    На графике отображается ТОЛЬКО Pareto front.

    Остальные кандидаты не рисуются.
    """

    pareto = df[df["pareto_optimal"]].copy()

    if pareto.empty:
        raise RuntimeError(
            "Cannot plot Pareto front: front is empty."
        )

    pareto = pareto.sort_values("energy_proxy_eu_h")

    plt.figure(figsize=(10, 6))

    # Только Pareto-точки
    plt.scatter(
        pareto["energy_proxy_eu_h"],
        pareto["predicted_sulfur_mg_kg"],
        s=70,
        label="Pareto front",
    )

    # Соединяем точки Pareto-фронта
    if len(pareto) > 1:
        plt.plot(
            pareto["energy_proxy_eu_h"],
            pareto["predicted_sulfur_mg_kg"],
            linestyle="--",
        )

    # Жёсткий лимит по сере
    plt.axhline(
        y=10,
        linestyle=":",
        label="Sulfur limit (10 mg/kg)",
    )

    plt.xlabel("Energy proxy, EU/h")
    plt.ylabel("Predicted sulfur, mg/kg")
    plt.title("Pareto front")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PARETO_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# Main
def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Результаты

    quality = pd.read_csv(
        QUALITY_RESULTS_PATH
    )

    energy = pd.read_csv(
        ENERGY_RESULTS_PATH
    )

    if "candidate_id" not in quality.columns:
        raise ValueError(
            "quality_results.csv must contain candidate_id"
        )

    if "candidate_id" not in energy.columns:
        raise ValueError(
            "energy_results.csv must contain candidate_id"
        )

    # Merge
    result = quality.merge(
        energy,
        on="candidate_id",
        how="inner",
        suffixes=("", "_energy"),
    )

    if result.empty:
        raise RuntimeError(
            "No candidates after merging quality and energy results."
        )

    # Quality
    if "quality_feasible" not in result.columns:
        if "limit_exceeded" not in result.columns:
            raise ValueError(
                "Quality results must contain "
                "quality_feasible or limit_exceeded."
            )

        result["quality_feasible"] = (
            ~result["limit_exceeded"].astype(bool)
        )

    # Только допустимые по качеству сценарии
    feasible = result[
        result["quality_feasible"] == True
    ].copy()

    if feasible.empty:
        raise RuntimeError(
            "No quality-feasible scenarios available. "
            "Pareto optimization cannot be performed."
        )

    # Check required columns
    required_columns = [
        "candidate_id",
        "predicted_sulfur_mg_kg",
        "energy_proxy_eu_h",
    ]

    missing = [
        column
        for column in required_columns
        if column not in feasible.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Pareto front
    result = calculate_pareto_front(
        result
    )

    pareto = result[
        result["pareto_optimal"] == True
    ].copy()

    if pareto.empty:
        raise RuntimeError(
            "Pareto front is empty."
        )

    # Один сценарий по расстоянию
    selected = select_scenario(
        result
    )

    # Save results
    result.to_csv(
        PARETO_RESULTS_PATH,
        index=False,
    )

    selected.to_csv(
        SELECTED_SCENARIO_PATH,
        index=False,
    )

    # Plot
    plot_pareto_front(
        result
    )

    # Console output
    print("PARETO OPTIMIZER")
    print(
        f"Total candidates:      {len(result)}"
    )
    print(
        f"Quality-feasible:      {len(feasible)}"
    )
    print(
        f"Pareto-optimal:        {len(pareto)}"
    )
    print()
    print("Selected scenario:")
    print(
        selected[
            [
                "candidate_id",
                "predicted_sulfur_mg_kg",
                "energy_proxy_eu_h",
                "distance_to_ideal",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Pareto results:   {PARETO_RESULTS_PATH}")
    print(f"Selected scenario:{SELECTED_SCENARIO_PATH}")
    print(f"Pareto plot:      {PARETO_PLOT_PATH}")

if __name__ == "__main__":
    main()